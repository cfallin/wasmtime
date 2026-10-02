//! Wasmtime debugger functionality.
//!
//! This crate builds on top of the core Wasmtime crate's
//! guest-debugger APIs to present an environment where a debugger
//! runs as a "co-running process" and sees the debuggee as a a
//! provider of a stream of events, on which actions can be taken
//! between each event.
//!
//! In the future, this crate will also provide a WIT-level API and
//! world in which to run debugger components.

use std::{
    any::Any,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
};
use wasmtime::{
    AsContext, AsContextMut, BreakpointState, DebugEvent, DebugHandler, Engine, ExnRef,
    FrameHandle, Frozen, Func, Instance, Result, Store, StoreContext, StoreContextMut, Trap,
    WatchpointHit,
};

mod host;

/// A paused debuggee's store, as given to [`Debuggee::with_store`].
///
/// A live debuggee's store may be changed, through
/// [`DebuggeeStore::store_mut`]. A replay's store is read-only: it can only be
/// inspected, through [`AsContext`], and only its debugging state, such as its
/// breakpoints, changed, through [`DebuggeeStore::edit_breakpoints`].
pub struct DebuggeeStore<'a, T: 'static>(Access<'a, T>);

enum Access<'a, T: 'static> {
    Live(StoreContextMut<'a, T>),
    #[cfg(feature = "rr")]
    Replay(&'a mut (dyn PausedReplay<T> + Send)),
}

/// What a debugger may use of a stopped replay.
#[cfg(feature = "rr")]
trait PausedReplay<T: 'static> {
    fn store(&self) -> StoreContext<'_, T>;
    fn edit_breakpoints(&mut self) -> Option<BreakpointState<'_>>;
    fn exit_frames(&mut self) -> Vec<FrameHandle>;
    fn debug_function(&mut self, instance: Instance, index: u32) -> Option<Func>;
}

#[cfg(feature = "rr")]
impl<T: Send + 'static> PausedReplay<T> for wasmtime::rr::Replayer<'_, T> {
    fn store(&self) -> StoreContext<'_, T> {
        wasmtime::rr::Replayer::store(self)
    }

    fn edit_breakpoints(&mut self) -> Option<BreakpointState<'_>> {
        wasmtime::rr::Replayer::edit_breakpoints(self)
    }

    fn exit_frames(&mut self) -> Vec<FrameHandle> {
        self.debug_exit_frames()
    }

    fn debug_function(&mut self, instance: Instance, index: u32) -> Option<Func> {
        wasmtime::rr::Replayer::debug_function(self, instance, index)
    }
}

impl<'a, T: 'static> DebuggeeStore<'a, T> {
    /// Mutable access to the store, which only a live debuggee gives: `None`
    /// for a replay.
    pub fn store_mut(&mut self) -> Option<StoreContextMut<'_, T>> {
        match &mut self.0 {
            Access::Live(store) => Some(store.as_context_mut()),
            #[cfg(feature = "rr")]
            Access::Replay(_) => None,
        }
    }

    /// The frames of the paused execution; see
    /// [`StoreContextMut::debug_exit_frames`].
    pub fn debug_exit_frames(&mut self) -> Vec<FrameHandle> {
        match &mut self.0 {
            Access::Live(store) => store.debug_exit_frames().collect(),
            #[cfg(feature = "rr")]
            Access::Replay(replay) => replay.exit_frames(),
        }
    }

    /// A function in `instance`'s function index space; see
    /// [`Instance::debug_function`].
    pub fn debug_function(&mut self, instance: Instance, index: u32) -> Option<Func> {
        match &mut self.0 {
            Access::Live(store) => instance.debug_function(store, index),
            #[cfg(feature = "rr")]
            Access::Replay(replay) => replay.debug_function(instance, index),
        }
    }

    fn reborrow(&mut self) -> DebuggeeStore<'_, T> {
        DebuggeeStore(match &mut self.0 {
            Access::Live(store) => Access::Live(store.as_context_mut()),
            #[cfg(feature = "rr")]
            Access::Replay(replay) => Access::Replay(&mut **replay),
        })
    }

    /// Borrow the debuggee's [`BreakpointState`], to edit its breakpoints,
    /// single-stepping, and watchpoints; see [`Store::edit_breakpoints`].
    pub fn edit_breakpoints(&mut self) -> Option<BreakpointState<'_>> {
        match &mut self.0 {
            Access::Live(store) => store.as_context_mut().edit_breakpoints(),
            #[cfg(feature = "rr")]
            Access::Replay(replay) => replay.edit_breakpoints(),
        }
    }
}

impl<T: 'static> AsContext for DebuggeeStore<'_, T> {
    type Data = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        match &self.0 {
            Access::Live(store) => store.as_context(),
            #[cfg(feature = "rr")]
            Access::Replay(replay) => replay.store(),
        }
    }
}

pub use host::{DebuggerComponent, add_debuggee, add_to_linker, wit};

/// A `Debuggee` wraps up state associated with debugging the code
/// running in a single `Store`.
///
/// It acts as a Future combinator, wrapping an inner async body that
/// performs some actions on a store. Those actions are subject to the
/// debugger, and debugger events will be raised as appropriate. From
/// the "outside" of this combinator, it is always in one of two
/// states: running or paused. When paused, it acts as a
/// `StoreContextMut` and can allow examining the paused execution's
/// state. One runs until the next event suspends execution by
/// invoking `Debuggee::run`.
pub struct Debuggee<T: Send + 'static> {
    /// A handle to the Engine that the debuggee store lives within.
    engine: Engine,
    /// State: either a task handle or the store when passed out of
    /// the complete task.
    state: DebuggeeState,
    /// The store, once complete.
    store: Option<Store<T>>,
    in_tx: mpsc::Sender<Command<T>>,
    out_rx: mpsc::Receiver<Response<T>>,
    handle: Option<JoinHandle<Result<()>>>,
    /// Flag shared with the inner handler: set to `true` by
    /// `interrupt()` so the next epoch yield is surfaced as an
    /// `Interrupted` event rather than eaten by the handler. Epoch
    /// yields serve two purposes, namely ensuring regular yields to
    /// the event loop and enacting an explicit interrupt, and this
    /// flag distinguishes those cases.
    interrupt_pending: Arc<AtomicBool>,
}

/// State machine from the perspective of the outer logic.
///
/// The intermediate states here, and the separation of these states
/// from the `JoinHandle` above, are what allow us to implement a
/// cancel-safe version of `Debuggee::run` below.
///
/// The state diagram for the outer logic is:
///
/// ```plain
///              (start)
///                 v
///                 |
/// .--->---------. v
/// |     .----<  Paused  <-----------------------------------------------.
/// |     |         v                                                     |
/// |     |         | (async fn run() starts, sends Command::Continue)    |
/// |     |         |                                                     |
/// |     |         v                                                     ^
/// |     |      Running                                                  |
/// |     |       v v (async fn run() receives Response::Paused, returns) |
/// |     |       | |_____________________________________________________|
/// |     |       |
/// |     |       | (async fn run() receives Response::Finished, returns)
/// |     |       v
/// |     |     Complete
/// |     |
/// ^     | (async fn with_store() starts, sends Command::Query)
/// |     v
/// |   Queried
/// |     |
/// |     | (async fn with_store() receives Response::QueryResponse, returns)
/// `---<-'
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DebuggeeState {
    /// Inner body has just been started.
    Initial,
    /// Inner body is running in an async task and not in a debugger
    /// callback. Outer logic is waiting for a `Response::Paused` or
    /// `Response::Complete`.
    Running,
    /// Inner body is running in an async task and at a debugger
    /// callback (or in the initial trampoline waiting for the first
    /// `Continue`). `Response::Paused` has been received. Outer
    /// logic has not sent any commands.
    Paused,
    /// We have sent a command to the inner body and are waiting for a
    /// response.
    Queried,
    /// Inner body is complete (has sent `Response::Finished` and we
    /// have received it). We may or may not have joined yet; if so,
    /// the `Option<JoinHandle<...>>` will be `None`.
    Complete,
}

/// Message from "outside" to the debug hook.
///
/// The `Query` catch-all with a boxed closure is a little janky, but
/// is the way that we provide access
/// from outside to the Store (which is owned by `inner` above)
/// only during pauses. Note that the future cannot take full
/// ownership or a mutable borrow of the Store, because it cannot
/// hold this across async yield points.
///
/// Instead, the debugger body sends boxed closures which take the
/// Store as a parameter (lifetime-limited not to escape that
/// closure) out to this crate's implementation that runs inside of
/// debugger-instrumentation callbacks (which have access to the
/// Store during their duration). We send return values
/// back. Return values are boxed Any values.
///
/// If we wanted to make this a little more principled, we could
/// come up with a Command/Response pair of enums for all possible
/// closures and make everything more statically typed and less
/// Box'd, but that would severely restrict the flexibility of the
/// abstraction here and essentially require writing a full proxy
/// of the debugger API.
///
/// Furthermore, we expect to rip this out eventually when we move
/// the debugger over to an async implementation based on
/// `run_concurrent` and `Accessor`s (see #11896). Building things
/// this way now will actually allow a less painful transition at
/// that time, because we will have a bunch of closures accessing
/// the store already and we can run those "with an accessor"
/// instead.
enum Command<T: 'static> {
    Continue,
    Query(Box<dyn FnOnce(DebuggeeStore<'_, T>) -> Box<dyn Any + Send> + Send>),
}

enum Response<T: 'static> {
    Paused(DebugRunResult),
    QueryResponse(Box<dyn Any + Send>),
    Finished(Store<T>),
}

struct HandlerInner<T: Send + 'static> {
    in_rx: Mutex<mpsc::Receiver<Command<T>>>,
    out_tx: mpsc::Sender<Response<T>>,
    interrupt_pending: Arc<AtomicBool>,
}

struct Handler<T: Send + 'static>(Arc<HandlerInner<T>>);

impl<T: Send + 'static> std::clone::Clone for Handler<T> {
    fn clone(&self) -> Self {
        Handler(self.0.clone())
    }
}

impl<T: Send + 'static> DebugHandler for Handler<T> {
    type Data = T;
    async fn handle(&self, store: StoreContextMut<'_, T>, event: DebugEvent<'_>) {
        let result = match event {
            DebugEvent::HostcallError(_) => DebugRunResult::HostcallError,
            DebugEvent::Exception(exn) => DebugRunResult::Exception(
                exn.to_frozen(&store)
                    .expect("the event's exception is rooted"),
            ),
            DebugEvent::Trap(trap) => DebugRunResult::Trap(trap),
            DebugEvent::Breakpoint => DebugRunResult::Breakpoint,
            DebugEvent::Watchpoint(hit) => DebugRunResult::Watchpoint(hit),
            DebugEvent::EpochYield => {
                // Only pause on epoch yields that were requested via
                // interrupt(). Other epoch ticks simply yield to the
                // event loop (functionality already implemented in
                // core Wasmtime; no need to do that yield here in the
                // debug handler).
                if !self.0.interrupt_pending.swap(false, Ordering::SeqCst) {
                    return;
                }
                DebugRunResult::EpochYield
            }
        };
        self.pause(DebuggeeStore(Access::Live(store)), result).await;
    }
}

impl<T: Send + 'static> Handler<T> {
    /// Reports `result` to the outer `Debuggee` and serves its queries until
    /// it continues.
    async fn pause(&self, mut store: DebuggeeStore<'_, T>, result: DebugRunResult) {
        let mut in_rx = self.0.in_rx.lock().await;
        if self.0.out_tx.send(Response::Paused(result)).await.is_err() {
            // Outer Debuggee has been dropped: just continue
            // executing.
            return;
        }

        while let Some(cmd) = in_rx.recv().await {
            match cmd {
                Command::Query(closure) => {
                    let result = closure(store.reborrow());
                    if self
                        .0
                        .out_tx
                        .send(Response::QueryResponse(result))
                        .await
                        .is_err()
                    {
                        // Outer Debuggee has been dropped: just
                        // continue executing.
                        return;
                    }
                }
                Command::Continue => {
                    break;
                }
            }
        }
    }
}

impl<T: Send + 'static> Debuggee<T> {
    /// Create a new Debugger that attaches to the given Store and
    /// runs the given inner body.
    ///
    /// The debugger is always in one of two states: running or
    /// paused.
    ///
    /// When paused, the holder of this object can invoke
    /// `Debuggee::run` to enter the running state. The inner body
    /// will run until paused by a debug event. While running, the
    /// future returned by either of these methods owns the `Debuggee`
    /// and hence no other methods can be invoked.
    ///
    /// When paused, the holder of this object can access the `Store`
    /// indirectly by providing a closure
    pub fn new<F>(store: Store<T>, inner: F) -> Debuggee<T>
    where
        F: for<'a> FnOnce(
                &'a mut Store<T>,
            ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>
            + Send
            + 'static,
    {
        Self::spawn(store, move |mut store, handler| async move {
            // Emulate a breakpoint at startup.
            log::trace!("inner debuggee task: first breakpoint");
            handler
                .handle(store.as_context_mut(), DebugEvent::Breakpoint)
                .await;
            log::trace!("inner debuggee task: first breakpoint resumed");

            // Now invoke the actual inner body.
            store.set_debug_handler(handler);
            log::trace!("inner debuggee task: running `inner`");
            let result = inner(&mut store).await;
            log::trace!("inner debuggee task: done with `inner`");
            (store, result)
        })
    }

    /// Create a new Debugger that replays `trace` in `store`, which must be
    /// empty and use an engine configured for replay with guest debugging.
    /// `bytecode` holds the modules or components that the trace instantiates
    /// (see [`wasmtime::rr::Replayer::add_bytecode`]).
    ///
    /// The replay is debugged like a live execution: it starts paused,
    /// stops at breakpoints, single steps, and watchpoints, and while paused
    /// the store shows the replay's state. Modules in the trace are compiled
    /// and registered before the initial pause, so breakpoints can be set in
    /// them. The replay cannot be changed: [`Debuggee::with_store`] gives
    /// only read-only access to its store, apart from its debugging state
    /// (see [`DebuggeeStore`]). `setup` can configure the replayer before it
    /// starts, for example to observe embedder events such as WASI output.
    ///
    /// With an engine configured for epoch interruption, an interrupt request
    /// (see [`Debuggee::interrupt_pending`]) pauses the replay, as an epoch
    /// yield would, once the engine's epoch next advances.
    #[cfg(feature = "rr")]
    pub fn new_replay(
        store: Store<T>,
        trace: wasmtime::rr::Trace,
        bytecode: Vec<Vec<u8>>,
        setup: impl FnOnce(&mut wasmtime::rr::Replayer<'_, T>) + Send + 'static,
    ) -> Debuggee<T> {
        Self::spawn(store, move |mut store, handler| async move {
            let result = async {
                let mut replayer = store.replayer(&trace)?;
                for wasm in &bytecode {
                    replayer.add_bytecode(wasm)?;
                }
                replayer.preload_modules()?;
                replayer.set_interrupt_flag(handler.0.interrupt_pending.clone());
                setup(&mut replayer);
                handler
                    .pause(
                        DebuggeeStore(Access::Replay(&mut replayer)),
                        DebugRunResult::Breakpoint,
                    )
                    .await;
                loop {
                    use wasmtime::rr::ReplayStop;
                    let result = match replayer.run().await? {
                        ReplayStop::Finished => break,
                        ReplayStop::Breakpoint => DebugRunResult::Breakpoint,
                        ReplayStop::Interrupted => DebugRunResult::EpochYield,
                        ReplayStop::Watchpoint(hit) => DebugRunResult::Watchpoint(hit),
                        _ => continue,
                    };
                    handler
                        .pause(DebuggeeStore(Access::Replay(&mut replayer)), result)
                        .await;
                }
                Ok(())
            }
            .await;
            (store, result)
        })
    }

    /// Spawns the task that runs a debuggee's body, `body`, which receives
    /// the store and the handler for its debug events and returns the store.
    fn spawn<F, Fut>(store: Store<T>, body: F) -> Debuggee<T>
    where
        F: FnOnce(Store<T>, Handler<T>) -> Fut + Send + 'static,
        Fut: Future<Output = (Store<T>, Result<()>)> + Send,
    {
        let engine = store.engine().clone();
        let (in_tx, in_rx) = mpsc::channel(1);
        let (out_tx, out_rx) = mpsc::channel(1);
        let interrupt_pending = Arc::new(AtomicBool::new(false));

        let handle = tokio::spawn({
            let interrupt_pending = interrupt_pending.clone();
            async move {
                let out_tx_clone = out_tx.clone();
                let handler = Handler(Arc::new(HandlerInner {
                    in_rx: Mutex::new(in_rx),
                    out_tx,
                    interrupt_pending,
                }));
                let (store, result) = body(store, handler).await;
                let _ = out_tx_clone.send(Response::Finished(store)).await;
                result
            }
        });

        Debuggee {
            engine,
            state: DebuggeeState::Initial,
            store: None,
            in_tx,
            out_rx,
            interrupt_pending,
            handle: Some(handle),
        }
    }

    /// Is the inner body done running?
    pub fn is_complete(&self) -> bool {
        match self.state {
            DebuggeeState::Complete => true,
            _ => false,
        }
    }

    /// Get the Engine associated with the debuggee.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get the interrupt-pending flag. Setting this to `true` causes
    /// the next epoch yield to surface as an `Interrupted` event.
    pub fn interrupt_pending(&self) -> &Arc<AtomicBool> {
        &self.interrupt_pending
    }

    async fn wait_for_initial(&mut self) -> Result<()> {
        if let DebuggeeState::Initial = &self.state {
            // Need to receive and discard first `Paused`.
            let response = self
                .out_rx
                .recv()
                .await
                .ok_or_else(|| wasmtime::format_err!("Premature close of debugger channel"))?;
            assert!(matches!(response, Response::Paused(_)));
            self.state = DebuggeeState::Paused;
        }
        Ok(())
    }

    /// Run the inner body until the next debug event.
    ///
    /// This method is cancel-safe, and no events will be lost.
    pub async fn run(&mut self) -> Result<DebugRunResult> {
        log::trace!("running: state is {:?}", self.state);

        self.wait_for_initial().await?;

        match self.state {
            DebuggeeState::Initial => unreachable!(),
            DebuggeeState::Paused => {
                log::trace!("sending Continue");
                self.in_tx
                    .send(Command::Continue)
                    .await
                    .map_err(|_| wasmtime::format_err!("Failed to send over debug channel"))?;
                log::trace!("sent Continue");

                // If that `send` was canceled, the command was not
                // sent, so it's fine to remain in `Paused`. If it
                // succeeded and we reached here, transition to
                // `Running` so we don't re-send.
                self.state = DebuggeeState::Running;
            }
            DebuggeeState::Running => {
                // Previous `run()` must have been canceled; no action
                // to take here.
            }
            DebuggeeState::Queried => {
                // We expect to receive a `QueryResponse`; drop it if
                // the query was canceled, then transition back to
                // `Paused`.
                log::trace!("in Queried; receiving");
                let response =
                    self.out_rx.recv().await.ok_or_else(|| {
                        wasmtime::format_err!("Premature close of debugger channel")
                    })?;
                log::trace!("in Queried; received, dropping");
                assert!(matches!(response, Response::QueryResponse(_)));
                self.state = DebuggeeState::Paused;

                // Now send a `Continue`, as above.
                log::trace!("in Paused; sending Continue");
                self.in_tx
                    .send(Command::Continue)
                    .await
                    .map_err(|_| wasmtime::format_err!("Failed to send over debug channel"))?;
                self.state = DebuggeeState::Running;
            }
            DebuggeeState::Complete => {
                panic!("Cannot `run()` an already-complete Debuggee");
            }
        }

        // At this point, the inner task is in Running state. We
        // expect to receive a message when it next pauses or
        // completes. If this `recv()` is canceled, no message is
        // lost, and the state above accurately reflects what must be
        // done on the next `run()`.
        log::trace!("waiting for response");
        let response = self
            .out_rx
            .recv()
            .await
            .ok_or_else(|| wasmtime::format_err!("Premature close of debugger channel"))?;

        match response {
            Response::Finished(store) => {
                log::trace!("got Finished");
                self.state = DebuggeeState::Complete;
                self.store = Some(store);
                Ok(DebugRunResult::Finished)
            }
            Response::Paused(result) => {
                log::trace!("got Paused");
                self.state = DebuggeeState::Paused;
                Ok(result)
            }
            Response::QueryResponse(_) => {
                panic!("Invalid debug response");
            }
        }
    }

    /// Run the debugger body until completion, with no further events.
    pub async fn finish(&mut self) -> Result<()> {
        if self.is_complete() {
            return Ok(());
        }
        loop {
            match self.run().await? {
                DebugRunResult::Finished => break,
                e => {
                    log::trace!("finish: event {e:?}");
                }
            }
        }
        if let Some(handle) = self.handle.take() {
            handle.await??;
        }
        assert!(self.is_complete());
        Ok(())
    }

    /// Perform some action on the contained `Store` while not running.
    ///
    /// This may only be invoked before the inner body finishes and
    /// when it is paused; that is, when the `Debuggee` is initially
    /// created and after any call to `run()` returns a result other
    /// than `DebugRunResult::Finished`. If an earlier `run()`
    /// invocation was canceled, it must be re-invoked and return
    /// successfully before a query is made.
    ///
    /// The store of a replay is read-only (see [`DebuggeeStore`]).
    ///
    /// This is cancel-safe; if canceled, the result of the query will
    /// be dropped.
    pub async fn with_store<
        F: FnOnce(DebuggeeStore<'_, T>) -> R + Send + 'static,
        R: Send + 'static,
    >(
        &mut self,
        f: F,
    ) -> Result<R> {
        if let Some(store) = self.store.as_mut() {
            return Ok(f(DebuggeeStore(Access::Live(store.as_context_mut()))));
        }

        self.wait_for_initial().await?;

        match self.state {
            DebuggeeState::Initial => unreachable!(),
            DebuggeeState::Queried => {
                // Earlier query canceled; drop its response first.
                let response =
                    self.out_rx.recv().await.ok_or_else(|| {
                        wasmtime::format_err!("Premature close of debugger channel")
                    })?;
                assert!(matches!(response, Response::QueryResponse(_)));
                self.state = DebuggeeState::Paused;
            }
            DebuggeeState::Running => {
                // Results from a canceled `run()`; `run()` must
                // complete before this can be invoked.
                panic!("Cannot query in Running state");
            }
            DebuggeeState::Complete => {
                panic!("Cannot query when complete");
            }
            DebuggeeState::Paused => {
                // OK -- this is the state we want.
            }
        }

        log::trace!("sending query in with_store");
        self.in_tx
            .send(Command::Query(Box::new(|store| Box::new(f(store)))))
            .await
            .map_err(|_| wasmtime::format_err!("Premature close of debugger channel"))?;
        self.state = DebuggeeState::Queried;

        let response = self
            .out_rx
            .recv()
            .await
            .ok_or_else(|| wasmtime::format_err!("Premature close of debugger channel"))?;
        let Response::QueryResponse(resp) = response else {
            wasmtime::bail!("Incorrect response from debugger task");
        };
        self.state = DebuggeeState::Paused;

        Ok(*resp.downcast::<R>().expect("type mismatch"))
    }
}

/// The result of one call to `Debuggee::run()`.
///
/// This is similar to `DebugEvent` but without the lifetime, so it
/// can be sent across async tasks, and incorporates the possibility
/// of completion (`Finished`) as well.
#[derive(Debug)]
pub enum DebugRunResult {
    /// Execution of the inner body finished.
    Finished,
    /// An error was raised by a hostcall.
    HostcallError,
    /// Wasm execution was interrupted by an epoch change.
    EpochYield,
    /// An exception is thrown by Wasm. The current state is at the
    /// throw-point. The exception can be read until execution resumes.
    Exception(Frozen<ExnRef>),
    /// A Wasm trap occurred.
    Trap(Trap),
    /// A breakpoint was reached.
    Breakpoint,
    /// Wasm is about to write watched memory; the write happens when
    /// execution continues.
    Watchpoint(WatchpointHit),
}

#[cfg(test)]
mod test {
    use super::*;
    use wasmtime::*;

    #[cfg(feature = "rr")]
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(miri, ignore)]
    async fn replay_debugging() -> wasmtime::Result<()> {
        let _ = env_logger::try_init();
        let wat = r#"
            (module
              (import "" "host" (func $host (result i32)))
              (memory (export "memory") 1)
              (func (export "main") (result i32)
                (i32.store (i32.const 0) (call $host))
                (i32.add (i32.load (i32.const 0)) (i32.const 1))))
        "#;
        let mut config = Config::new();
        config.rr(RRConfig::Recording);
        let engine = Engine::new(&config)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let host = Func::wrap(&mut store, || 41);
        let module = Module::new(&engine, wat)?;
        let instance = Instance::new(&mut store, &module, &[host.into()])?;
        let main = instance.get_typed_func::<(), i32>(&mut store, "main")?;
        assert_eq!(main.call(&mut store, ())?, 42);
        let trace = store.finish_recording()?;

        let mut config = Config::new();
        config.rr(RRConfig::Replaying).guest_debug(true);
        let engine = Engine::new(&config)?;
        let mut debuggee =
            Debuggee::new_replay(Store::new(&engine, ()), trace, vec![wat.into()], |_| {});
        // The trace's module is available at the initial pause.
        let modules = debuggee
            .with_store(|mut store| {
                store.edit_breakpoints().unwrap().single_step(true).unwrap();
                store.as_context().debug_all_modules().len()
            })
            .await?;
        assert_eq!(modules, 1);
        let mut steps = 0;
        loop {
            match debuggee.run().await? {
                DebugRunResult::Breakpoint => {
                    steps += 1;
                    let frames = debuggee
                        .with_store(|mut store| store.debug_exit_frames().len())
                        .await?;
                    assert!(frames >= 1);
                }
                DebugRunResult::Finished => break,
                e => panic!("unexpected event {e:?}"),
            }
        }
        assert!(steps > 3, "{steps} steps");
        debuggee.finish().await?;
        Ok(())
    }

    #[cfg(feature = "rr")]
    #[tokio::test(flavor = "multi_thread")]
    #[cfg_attr(miri, ignore)]
    async fn replay_interrupt() -> wasmtime::Result<()> {
        let wat = r#"
            (module
              (func (export "main") (result i32)
                (local $i i32)
                loop
                  (local.set $i (i32.add (local.get $i) (i32.const 1)))
                  (br_if 0 (i32.lt_u (local.get $i) (i32.const 1000000)))
                end
                local.get $i))
        "#;
        let mut config = Config::new();
        config.rr(RRConfig::Recording);
        let engine = Engine::new(&config)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let module = Module::new(&engine, wat)?;
        let instance = Instance::new(&mut store, &module, &[])?;
        let main = instance.get_typed_func::<(), i32>(&mut store, "main")?;
        main.call(&mut store, ())?;
        let trace = store.finish_recording()?;

        let mut config = Config::new();
        config
            .rr(RRConfig::Replaying)
            .guest_debug(true)
            .epoch_interruption(true);
        let engine = Engine::new(&config)?;
        let mut debuggee =
            Debuggee::new_replay(Store::new(&engine, ()), trace, vec![wat.into()], |_| {});
        // Wait for the initial pause, then interrupt, as the debugger API's
        // `interrupt` does.
        debuggee.with_store(|_| ()).await?;
        debuggee.interrupt_pending().store(true, Ordering::SeqCst);
        engine.increment_epoch();
        assert!(matches!(debuggee.run().await?, DebugRunResult::EpochYield));
        let frames = debuggee
            .with_store(|mut store| store.debug_exit_frames().len())
            .await?;
        assert_eq!(frames, 1);
        assert!(matches!(debuggee.run().await?, DebugRunResult::Finished));
        debuggee.finish().await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn exception_events_are_frozen() -> wasmtime::Result<()> {
        let mut config = Config::new();
        config.guest_debug(true).wasm_exceptions(true);
        let engine = Engine::new(&config)?;
        let module = Module::new(
            &engine,
            r#"
                (module
                  (tag $t (param i32 i64))
                  (func (export "main")
                    i32.const 7
                    i64.const 8
                    throw $t))
            "#,
        )?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new_async(&mut store, &module, &[]).await?;
        let main = instance.get_func(&mut store, "main").unwrap();
        let mut debuggee = Debuggee::new(store, move |store| {
            Box::pin(async move {
                assert!(main.call_async(&mut *store, &[], &mut []).await.is_err());
                Ok(())
            })
        });

        let DebugRunResult::Exception(exn) = debuggee.run().await? else {
            panic!("expected an exception event");
        };
        debuggee
            .with_store(move |store| {
                let fields = exn.fields(&store).unwrap();
                assert_eq!(fields[0].unwrap_i32(), 7);
                assert_eq!(fields[1].unwrap_i64(), 8);
                assert_eq!(exn.tag(&store).unwrap().ty(&store).ty().params().len(), 2);
            })
            .await?;
        // The exception can only be read while execution is paused.
        while !matches!(debuggee.run().await?, DebugRunResult::Finished) {}
        assert!(
            !debuggee
                .with_store(move |store| exn.is_valid(&store))
                .await?
        );
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn basic_debugger() -> wasmtime::Result<()> {
        let _ = env_logger::try_init();

        let mut config = Config::new();
        config.guest_debug(true);
        let engine = Engine::new(&config)?;
        let module = Module::new(
            &engine,
            r#"
                (module
                  (func (export "main") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add))
            "#,
        )?;

        let mut store = Store::new(&engine, ());
        let instance = Instance::new_async(&mut store, &module, &[]).await?;
        let main = instance.get_func(&mut store, "main").unwrap();

        let mut debuggee = Debuggee::new(store, move |store| {
            Box::pin(async move {
                let mut results = [Val::I32(0)];
                store.edit_breakpoints().unwrap().single_step(true).unwrap();
                main.call_async(&mut *store, &[Val::I32(1), Val::I32(2)], &mut results[..])
                    .await?;
                assert_eq!(results[0].unwrap_i32(), 3);
                main.call_async(&mut *store, &[Val::I32(3), Val::I32(4)], &mut results[..])
                    .await?;
                assert_eq!(results[0].unwrap_i32(), 7);
                Ok(())
            })
        });

        let event = debuggee.run().await?;
        assert!(matches!(event, DebugRunResult::Breakpoint));
        // At (before executing) first `local.get`.
        debuggee
            .with_store(|mut store| {
                let frame = store.debug_exit_frames().remove(0);
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .0
                        .as_u32(),
                    0
                );
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .1
                        .raw(),
                    36
                );
                assert_eq!(frame.num_locals(&mut store).unwrap(), 2);
                assert_eq!(frame.num_stacks(&mut store).unwrap(), 0);
                assert_eq!(frame.local(&store, 0).unwrap().unwrap_i32(), 1);
                assert_eq!(frame.local(&store, 1).unwrap().unwrap_i32(), 2);
                let frame = frame.parent(&mut store).unwrap();
                assert!(frame.is_none());
            })
            .await?;

        let event = debuggee.run().await?;
        // At second `local.get`.
        assert!(matches!(event, DebugRunResult::Breakpoint));
        debuggee
            .with_store(|mut store| {
                let frame = store.debug_exit_frames().remove(0);
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .0
                        .as_u32(),
                    0
                );
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .1
                        .raw(),
                    38
                );
                assert_eq!(frame.num_locals(&mut store).unwrap(), 2);
                assert_eq!(frame.num_stacks(&mut store).unwrap(), 1);
                assert_eq!(frame.local(&store, 0).unwrap().unwrap_i32(), 1);
                assert_eq!(frame.local(&store, 1).unwrap().unwrap_i32(), 2);
                assert_eq!(frame.stack(&store, 0).unwrap().unwrap_i32(), 1);
                let frame = frame.parent(&mut store).unwrap();
                assert!(frame.is_none());
            })
            .await?;

        let event = debuggee.run().await?;
        // At `i32.add`.
        assert!(matches!(event, DebugRunResult::Breakpoint));
        debuggee
            .with_store(|mut store| {
                let frame = store.debug_exit_frames().remove(0);
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .0
                        .as_u32(),
                    0
                );
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .1
                        .raw(),
                    40
                );
                assert_eq!(frame.num_locals(&mut store).unwrap(), 2);
                assert_eq!(frame.num_stacks(&mut store).unwrap(), 2);
                assert_eq!(frame.local(&store, 0).unwrap().unwrap_i32(), 1);
                assert_eq!(frame.local(&store, 1).unwrap().unwrap_i32(), 2);
                assert_eq!(frame.stack(&store, 0).unwrap().unwrap_i32(), 1);
                assert_eq!(frame.stack(&store, 1).unwrap().unwrap_i32(), 2);
                let frame = frame.parent(&mut store).unwrap();
                assert!(frame.is_none());
            })
            .await?;

        let event = debuggee.run().await?;
        // At return point.
        assert!(matches!(event, DebugRunResult::Breakpoint));
        debuggee
            .with_store(|mut store| {
                let frame = store.debug_exit_frames().remove(0);
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .0
                        .as_u32(),
                    0
                );
                assert_eq!(
                    frame
                        .wasm_function_index_and_pc(&mut store)
                        .unwrap()
                        .unwrap()
                        .1
                        .raw(),
                    41
                );
                assert_eq!(frame.num_locals(&mut store).unwrap(), 2);
                assert_eq!(frame.num_stacks(&mut store).unwrap(), 1);
                assert_eq!(frame.local(&store, 0).unwrap().unwrap_i32(), 1);
                assert_eq!(frame.local(&store, 1).unwrap().unwrap_i32(), 2);
                assert_eq!(frame.stack(&store, 0).unwrap().unwrap_i32(), 3);
                let frame = frame.parent(&mut store).unwrap();
                assert!(frame.is_none());
            })
            .await?;

        // Now disable breakpoints before continuing. Second call should proceed with no more events.
        debuggee
            .with_store(|mut store| {
                store
                    .edit_breakpoints()
                    .unwrap()
                    .single_step(false)
                    .unwrap();
            })
            .await?;

        let event = debuggee.run().await?;
        assert!(matches!(event, DebugRunResult::Finished));

        assert!(debuggee.is_complete());

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn watchpoints() -> wasmtime::Result<()> {
        let _ = env_logger::try_init();

        let mut config = Config::new();
        config.guest_debug(true);
        let engine = Engine::new(&config)?;
        let module = Module::new(
            &engine,
            r#"
                (module
                  (memory (export "memory") 1)
                  (func (export "main")
                    (i32.store8 (i32.const 7) (i32.const 1))
                    (i32.store offset=6 (i32.const 0) (i32.const 0x11223344))
                    (i32.store8 (i32.const 12) (i32.const 2))
                    (memory.fill (i32.const 0) (i32.const 0xaa) (i32.const 16))))
            "#,
        )?;

        let mut store = Store::new(&engine, ());
        let instance = Instance::new_async(&mut store, &module, &[]).await?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();
        let main = instance.get_typed_func::<(), ()>(&mut store, "main")?;

        // Watch bytes [8, 12): the first store is just below the
        // range, the second overlaps it partially, the third is just
        // above it, and the fill covers all of it.
        store
            .edit_breakpoints()
            .unwrap()
            .watch(memory, 8..12, true)?;

        let mut debuggee = Debuggee::new(store, move |store| {
            Box::pin(async move {
                main.call_async(&mut *store, ()).await?;
                Ok(())
            })
        });

        let event = debuggee.run().await?;
        let DebugRunResult::Watchpoint(hit) = event else {
            panic!("expected a watchpoint, got {event:?}");
        };
        assert_eq!(
            hit,
            WatchpointHit {
                memory,
                address: 6,
                len: 4,
                value: Some(0x11223344),
            }
        );
        // The event is raised before the write happens.
        debuggee
            .with_store(move |store| {
                assert_eq!(&memory.data(&store)[6..10], &[0, 1, 0, 0]);
            })
            .await?;

        let event = debuggee.run().await?;
        let DebugRunResult::Watchpoint(hit) = event else {
            panic!("expected a watchpoint, got {event:?}");
        };
        assert_eq!(
            hit,
            WatchpointHit {
                memory,
                address: 0,
                len: 16,
                value: None,
            }
        );
        debuggee
            .with_store(move |mut store| {
                // The previous write has now happened, but not the
                // fill.
                assert_eq!(&memory.data(&store)[6..10], &0x11223344_u32.to_le_bytes());
                assert_eq!(memory.data(&store)[12], 2);
                // Stop watching; the rest of execution runs freely.
                store
                    .edit_breakpoints()
                    .unwrap()
                    .watch(memory, 0..16, false)
                    .unwrap();
            })
            .await?;

        let event = debuggee.run().await?;
        assert!(matches!(event, DebugRunResult::Finished));
        debuggee
            .with_store(move |store| {
                assert_eq!(&memory.data(&store)[..16], &[0xaa; 16]);
            })
            .await?;

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn early_finish() -> Result<()> {
        let _ = env_logger::try_init();

        let mut config = Config::new();
        config.guest_debug(true);
        let engine = Engine::new(&config)?;
        let module = Module::new(
            &engine,
            r#"
                (module
                  (func (export "main") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add))
            "#,
        )?;

        let mut store = Store::new(&engine, ());
        let instance = Instance::new_async(&mut store, &module, &[]).await?;
        let main = instance.get_func(&mut store, "main").unwrap();

        let mut debuggee = Debuggee::new(store, move |store| {
            Box::pin(async move {
                let mut results = [Val::I32(0)];
                store.edit_breakpoints().unwrap().single_step(true).unwrap();
                main.call_async(&mut *store, &[Val::I32(1), Val::I32(2)], &mut results[..])
                    .await?;
                assert_eq!(results[0].unwrap_i32(), 3);
                Ok(())
            })
        });

        debuggee.finish().await?;
        assert!(debuggee.is_complete());

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn drop_debuggee_and_store() -> Result<()> {
        let _ = env_logger::try_init();

        let mut config = Config::new();
        config.guest_debug(true);
        let engine = Engine::new(&config)?;
        let module = Module::new(
            &engine,
            r#"
                (module
                  (func (export "main") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add))
            "#,
        )?;

        let mut store = Store::new(&engine, ());
        let instance = Instance::new_async(&mut store, &module, &[]).await?;
        let main = instance.get_func(&mut store, "main").unwrap();

        let mut debuggee = Debuggee::new(store, move |store| {
            Box::pin(async move {
                let mut results = [Val::I32(0)];
                store.edit_breakpoints().unwrap().single_step(true).unwrap();
                main.call_async(&mut *store, &[Val::I32(1), Val::I32(2)], &mut results[..])
                    .await?;
                assert_eq!(results[0].unwrap_i32(), 3);
                Ok(())
            })
        });

        // Step once, then drop everything at the end of this
        // function. Wasmtime's fiber cleanup should safely happen
        // without attempting to raise debug async handler calls with
        // missing async context.
        let _ = debuggee.run().await?;

        Ok(())
    }
}
