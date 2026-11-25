//! The replay of a trace.
//!
//! A single loop consumes the trace and owns all guest activations: host calls
//! suspend an activation and yield to this loop, so no guest code ever runs
//! beneath a host function. Activations may complete in any order.

use super::*;
use crate::{AsContext, Extern, Instance, Module, StoreContext, StoreContextMut, Val};
use alloc::collections::BTreeMap;
use core::mem::MaybeUninit;
use core::sync::atomic::AtomicBool;
use core::task::Poll;
use objects::{RecordedGlobal, RecordedModule};

mod activation;
mod checkpoint;
#[cfg(feature = "debug")]
mod count;
#[cfg(feature = "gc")]
mod host_gc;
mod image;
mod init;
#[cfg(feature = "debug")]
mod reverse;
mod stop;
mod track;

use activation::{Activation, HostCall, HostValues, Serial, Trampolines, Yield};
pub use checkpoint::Checkpoint;
#[cfg(feature = "debug")]
pub(super) use count::EventCounter;
#[cfg(feature = "gc")]
pub(crate) use image::PagedImage;
#[cfg(feature = "debug")]
pub use reverse::Position;
pub(super) use stop::{Epochs, Stop};
pub(super) use track::Histories;

/// Replays a trace, one stop at a time, without calling the original host
/// functions.
///
/// Replay reconstructs everything in the trace in a store that the replayer
/// owns, which can be inspected with [`Replayer::store`]. The store has no host
/// data, as there are no host functions to use it. Replay checks everything
/// that the guest does that the host can observe: the functions that it calls,
/// their arguments, and its results and traps. Any divergence from the trace is
/// an error, while a guest trap that was recorded is not.
pub struct Replayer {
    store: Store<()>,
    trampolines: Trampolines,
    bytecode: Bytecode,
    trace: Arc<Trace>,
    /// The offset of the next record in the trace.
    offset: usize,
    /// The next module to be instantiated. Those from here on in `Objects` were
    /// preloaded.
    next_module: RecordedModule,
    activations: TryVec<Activation>,
    /// The activation that stopped at a point that isn't in the trace, which
    /// resumes before anything else.
    paused: Option<RecordedCall>,
    /// The write that the paused activation, which stopped at a watchpoint,
    /// will perform.
    #[cfg(feature = "debug")]
    pending_write: Option<(Memory, Range<usize>)>,
    /// What guest code did when it last ran, until the trace's next record is
    /// checked against it.
    observed: Option<Observed>,
    /// The instances, including those that have not been created at the
    /// current point of a replay that was rewound.
    instances: TryVec<Instance>,
    /// How many objects have been created at the current point.
    defined: Defined,
    next_serial: Serial,
    checkpoints: checkpoint::Checkpoints,
    /// The startup function of a newly created instance, which must run before
    /// anything else.
    pending_startup: Option<RecordedFunc>,
    /// Observers of embedder events, by tag.
    observers: Vec<(u32, Observer)>,
    stop_at_events: bool,
    /// Why `run` should return, once the current step completes.
    stop: Option<ReplayStop>,
    /// Whether `run` should yield to the async executor.
    yield_now: bool,
    /// Whether the count of debug events reached its target.
    at_target: bool,
    /// Whether the paused activation is at the check of a debug event.
    at_check: bool,
    /// Whether to leave single-stepping enabled after the interrupt that is
    /// being settled, if any.
    #[cfg(feature = "debug")]
    settling: Option<bool>,
    #[cfg(feature = "debug")]
    reverse: Option<reverse::Reverse>,
    /// Storage for encoding values to compare with the trace.
    scratch: TryVec<u8>,
    finished: bool,
    /// Whether a step failed, or unwound, so that replay cannot continue.
    failed: bool,
}

/// Why [`Replayer::run`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayStop {
    /// The entire trace was replayed.
    Finished,
    /// Guest code reached a breakpoint or single-stepped, as set with
    /// [`Replayer::edit_breakpoints`]. The stopped frames are available from
    /// [`Replayer::debug_exit_frames`].
    Breakpoint,
    /// Guest code is about to write to memory that is being watched. The write
    /// happens when replay continues.
    #[cfg(feature = "debug")]
    Watchpoint(crate::WatchpointHit),
    /// Replay was interrupted through [`Replayer::set_interrupt_flag`].
    Interrupted,
    /// Reverse execution reached the beginning of the replay.
    Beginning,
    /// An embedder event with this tag was replayed, after its observers ran.
    /// Only reported when enabled with [`Replayer::stop_at_events`].
    Event(u32),
}

/// How many of the objects in the store have been created at a point in the
/// replay. A replay that is rewound keeps the rest.
#[derive(Clone, Copy, Default)]
struct Defined {
    memories: usize,
    tables: usize,
    globals: usize,
    instances: usize,
}

enum Observed {
    /// An activation called a host function.
    Host {
        activation: RecordedCall,
        func: RecordedFunc,
        values: HostValues,
    },
    /// An activation finished.
    Complete {
        activation: RecordedCall,
        result: Result<()>,
    },
}

type Observer = Box<dyn FnMut(&[u8]) -> Result<()> + Send>;

/// The bytecode that the embedder provided for the trace's modules.
#[derive(Default)]
struct Bytecode(BTreeMap<Checksum, Vec<u8>>);

impl Bytecode {
    fn get(&self, checksum: &Checksum) -> Result<&[u8]> {
        self.0.get(checksum).map(|b| &b[..]).ok_or_else(|| {
            format_err!(
                "the trace instantiates a module whose bytecode was not provided \
                 (SHA-256 {})",
                checksum_hex(checksum)
            )
        })
    }
}

impl Replayer {
    /// Starts replaying `trace` in a new store. The trace's modules must be
    /// provided with [`Replayer::add_bytecode`].
    ///
    /// # Errors
    ///
    /// Returns an error if `engine` is not configured with
    /// [`crate::RrConfig::Replaying`], if its configuration or the host is
    /// incompatible with replay, or on allocation failure.
    pub fn new(engine: &Engine, trace: impl Into<Arc<Trace>>) -> Result<Self> {
        ensure!(engine.is_replaying(), "replay requires RrConfig::Replaying");
        let trampolines = Trampolines::new(engine)?;
        let mut store = Store::try_new(engine, ())?;
        store.as_context_mut().0.rr.session = Some(try_new::<Box<_>>(Session {
            objects: Objects::default(),
            handles: Handles::default(),
            #[cfg(feature = "gc")]
            reads: Default::default(),
            mode: Mode::Replaying(Replaying::default()),
            failure: None,
        })?);
        #[cfg(target_has_atomic = "64")]
        if engine.tunables().epoch_interruption {
            store.set_epoch_deadline(1);
        }
        Ok(Replayer {
            store,
            trampolines,
            bytecode: Bytecode::default(),
            trace: trace.into(),
            // `Trace` checks the magic.
            offset: codec::MAGIC.len(),
            next_module: RecordedModule::from_u32(0),
            activations: TryVec::new(),
            paused: None,
            #[cfg(feature = "debug")]
            pending_write: None,
            observed: None,
            instances: TryVec::new(),
            defined: Defined::default(),
            next_serial: 0,
            checkpoints: Default::default(),
            pending_startup: None,
            observers: Vec::new(),
            stop_at_events: false,
            stop: None,
            yield_now: false,
            at_target: false,
            at_check: false,
            #[cfg(feature = "debug")]
            settling: None,
            #[cfg(feature = "debug")]
            reverse: None,
            scratch: TryVec::new(),
            finished: false,
            failed: false,
        })
    }

    /// Provides the bytecode of modules that the trace instantiates, which
    /// traces identify only by checksum (see [`Trace::module_checksums`]).
    ///
    /// `wasm` is a core module or a component, in the binary or (with the `wat`
    /// feature) text format; a component provides the modules that
    /// [`core_modules`] finds in it. Modules that the trace does not instantiate
    /// are ignored. Replay fails if it reaches the first instantiation of a
    /// module whose bytecode has not been provided.
    pub fn add_bytecode(&mut self, wasm: &[u8]) -> Result<()> {
        self.bytecode
            .0
            .extend(core_modules(self.store.engine(), wasm)?);
        Ok(())
    }

    /// Calls `observer` with each recorded event of type `E` as replay reaches
    /// it, including again after replay is rewound. Observers cannot affect the replay.
    pub fn on_event<E: TraceEvent>(&mut self, mut observer: impl FnMut(E) + Send + 'static) {
        self.observers.push((
            E::TAG,
            Box::new(move |bytes| {
                observer(postcard::from_bytes(bytes)?);
                Ok(())
            }),
        ));
    }

    /// Sets whether [`Replayer::run`] stops after each embedder event.
    pub fn stop_at_events(&mut self, enabled: bool) {
        self.stop_at_events = enabled;
    }

    /// Makes replay stop with [`ReplayStop::Interrupted`] once `flag` is set,
    /// and clear it.
    ///
    /// This requires an engine that enables
    /// [`Config::epoch_interruption`](crate::Config::epoch_interruption). Guest
    /// code checks the flag when the engine's epoch advances, so call
    /// [`Engine::increment_epoch`] after setting it.
    pub fn set_interrupt_flag(&mut self, flag: Arc<AtomicBool>) {
        self.replaying().epochs.interrupt = Some(flag);
    }

    /// Sets how many epochs apart guest code yields to the async executor. See
    /// [`Replayer::run`].
    #[cfg(target_has_atomic = "64")]
    pub fn epoch_yield_interval(&mut self, interval: Option<u64>) {
        let epoch = self.store.engine().current_epoch();
        self.replaying().epochs.yields = interval.map(|delta| (delta, epoch.saturating_add(delta)));
    }

    /// Sets how much fuel guest code consumes before yielding to the async
    /// executor, as for [`Store::fuel_async_yield_interval`]. See
    /// [`Replayer::run`].
    #[cfg(feature = "async")]
    pub fn fuel_yield_interval(&mut self, interval: Option<u64>) -> Result<()> {
        self.store.fuel_async_yield_interval(interval)
    }

    /// Compiles every module in the trace now, rather than when replay reaches
    /// it, and returns them in the order of the trace. Their bytecode must
    /// already have been provided.
    ///
    /// With guest debugging enabled, they are also registered for debugging, as
    /// with [`Store::debug_register_module`], so that a debugger can set
    /// breakpoints in them before replay starts.
    pub fn preload_modules(&mut self) -> Result<Vec<Module>> {
        let trace = self.trace.clone();
        let mut reader = Reader::at(&trace.bytes, codec::MAGIC.len());
        while reader.peek_tag().is_some() {
            let (tag, mut body) = reader.record()?;
            if tag == codec::MODULE {
                let id = RecordedModule::from_u32(body.u32()?);
                self.compile(id, &body.checksum()?)?;
            }
        }
        let modules = &self.store.as_context().0.rr_objects().modules;
        let modules = modules.values().cloned().collect::<Vec<_>>();
        #[cfg(feature = "debug")]
        if self.store.engine().tunables().debug_guest {
            for module in &modules {
                self.store.debug_register_module(module)?;
            }
        }
        Ok(modules)
    }

    /// Replays until the next stop.
    ///
    /// This yields to the async executor periodically, between records of the
    /// trace. Guest code that runs for longer also yields if the engine enables
    /// epoch interruption or fuel, and an interval is set with
    /// [`Replayer::epoch_yield_interval`] or [`Replayer::fuel_yield_interval`].
    /// Epochs and fuel otherwise do not affect replay: guest code never runs out
    /// of fuel or traps at an epoch deadline, as the recording did not.
    ///
    /// Dropping the future cancels replay, which can then be resumed by calling
    /// this again. After an error, replay cannot be resumed.
    pub async fn run(&mut self) -> Result<ReplayStop> {
        loop {
            #[cfg(feature = "debug")]
            self.prepare_to_run_forward();
            let stop = self.run_to_stop().await?;
            #[cfg(feature = "debug")]
            let stop = match stop {
                Some(stop) => self.settle_interrupt(stop)?,
                // A snapshot is due.
                None => {
                    self.take_snapshot()?;
                    None
                }
            };
            let Some(stop) = stop else {
                continue;
            };
            #[cfg(feature = "debug")]
            if stop != ReplayStop::Finished {
                self.take_snapshot()?;
            }
            return Ok(stop);
        }
    }

    /// Interrupts stop at epoch checks, where frames cannot be inspected, so
    /// with guest debugging this single-steps to the next instruction. Returns
    /// `None` if replay should continue.
    #[cfg(feature = "debug")]
    fn settle_interrupt(&mut self, stop: ReplayStop) -> Result<Option<ReplayStop>> {
        if stop == ReplayStop::Interrupted && self.store.engine().tunables().debug_guest {
            if self.settling.is_none() {
                self.settling = Some(self.store().is_single_step());
                self.edit_breakpoints(|b| b.single_step(true))?;
            }
            return Ok(None);
        }
        let Some(single_step) = self.settling.take() else {
            return Ok(Some(stop));
        };
        self.edit_breakpoints(|b| b.single_step(single_step))?;
        Ok(Some(match stop {
            ReplayStop::Breakpoint => ReplayStop::Interrupted,
            stop => stop,
        }))
    }

    /// Replays until the next stop, or to the target for the count of debug
    /// events, for which this returns `None`.
    async fn run_to_stop(&mut self) -> Result<Option<ReplayStop>> {
        // Invalidate handles to state that Wasm may mutate, as ordinary calls
        // into Wasm do when they return to the host.
        self.store.as_context_mut().0.on_wasm_to_host_transition();
        core::future::poll_fn(|cx| {
            for _ in 0..256 {
                if self.finished && self.paused.is_none() {
                    return Poll::Ready(Ok(Some(ReplayStop::Finished)));
                }
                let offset = self.offset;
                if let Err(e) = self.step() {
                    let e = e.context(format!("replaying the trace record at byte {offset}"));
                    return Poll::Ready(Err(e));
                }
                if let Some(stop) = self.stop.take() {
                    return Poll::Ready(Ok(Some(stop)));
                }
                if core::mem::take(&mut self.at_target) {
                    return Poll::Ready(Ok(None));
                }
                if core::mem::take(&mut self.yield_now) {
                    break;
                }
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await
    }

    /// The instances that have been created so far, in order. Components replay
    /// as the core instances that they instantiate.
    pub fn instances(&self) -> &[Instance] {
        &self.instances[..self.defined.instances]
    }

    /// The store being replayed into, for inspection.
    ///
    /// Only the replay may change the store's state, so this gives only shared
    /// access to it, but see [`Replayer::edit_breakpoints`]. GC objects are read as [`Unrooted`](crate::Unrooted)
    /// references, which are valid until replay continues.
    pub fn store(&self) -> StoreContext<'_, ()> {
        self.store.as_context()
    }

    /// The most recent exception that guest code threw and that reached the
    /// host.
    ///
    /// It remains available after the recorded host took it from the store (with
    /// [`Store::take_pending_exception`]), until guest code throws another
    /// exception.
    #[cfg(feature = "gc")]
    pub fn pending_exception(&self) -> Option<crate::Unrooted<crate::ExnRef>> {
        self.store.as_context().0.rr_pending_exception()
    }

    /// Edits the store's breakpoints, single-stepping, and watchpoints, as
    /// [`Store::edit_breakpoints`] does. These are not part of the replay's
    /// state.
    #[cfg(feature = "debug")]
    pub fn edit_breakpoints<R>(
        &mut self,
        f: impl FnOnce(&mut crate::BreakpointState<'_>) -> Result<R>,
    ) -> Result<R> {
        self.store.edit_breakpoints(f)
    }

    /// The exit frames of the activation that stopped, as for
    /// [`Store::debug_exit_frames`], and then of those suspended in host calls,
    /// most recently started first. For nested calls, where a host function
    /// calls back into the guest, this is the order of the calls.
    ///
    /// This is empty unless [`Replayer::run`] stopped within guest code. The
    /// frames can be inspected with [`Replayer::store`] until replay continues.
    #[cfg(feature = "debug")]
    pub fn debug_exit_frames(&mut self) -> Vec<crate::FrameHandle> {
        let Some(paused) = self.paused else {
            return Vec::new();
        };
        let (paused, parked) = self
            .activations
            .iter()
            .rev()
            .filter(|a| a.call == paused || a.host.is_some())
            .partition::<Vec<_>, _>(|a| a.call == paused);
        // SAFETY: suspended activations' stacks are intact until replay
        // continues, which invalidates the frames.
        unsafe {
            let activations = paused
                .into_iter()
                .chain(parked)
                .map(|a| a.exit_state().activation());
            let store = self.store.as_context_mut().0;
            store.debug_exit_frames_of(activations).collect()
        }
    }

    /// Looks up a function in `instance`'s function index space, as
    /// [`Instance::debug_function`] does. See [`Replayer::get_export`].
    #[cfg(feature = "debug")]
    pub fn debug_function(&mut self, instance: Instance, function_index: u32) -> Option<Func> {
        instance.debug_function(&mut self.store, function_index)
    }

    /// Looks up `instance`'s export `name`, as [`Instance::get_export`] does.
    ///
    /// Functions are initialized lazily, which [`Replayer::store`]'s shared
    /// access does not allow. A replay's functions can be inspected, but not
    /// called.
    pub fn get_export(&mut self, instance: Instance, name: &str) -> Option<Extern> {
        instance.get_export(&mut self.store, name)
    }

    fn replaying(&mut self) -> &mut Replaying {
        self.store.as_context_mut().0.rr_session().mode.replaying()
    }

    /// Checks that no guest code is running: every activation is parked at a
    /// host call that has been matched with the trace.
    fn require_idle(&self) -> Result<()> {
        ensure!(
            self.observed.is_none() && self.activations.iter().all(|a| a.host.is_some()),
            "the guest diverged from the trace: it should not have reached a host boundary"
        );
        Ok(())
    }

    /// Replays one record.
    fn step(&mut self) -> Result<()> {
        ensure!(!self.failed, "replay cannot continue after it failed");
        self.failed = true;
        if let Some(paused) = self.paused.take() {
            self.at_check = false;
            #[cfg(feature = "debug")]
            {
                self.pending_write = None;
            }
            let index = self.activations.iter().position(|a| a.call == paused);
            self.resume_running(index.unwrap(), None)?;
            self.failed = false;
            return Ok(());
        }
        if !self.replay_creations()? {
            self.step_record()?;
        }
        self.failed = false;
        Ok(())
    }

    fn step_record(&mut self) -> Result<()> {
        let trace = self.trace.clone();
        let mut reader = Reader::at(&trace.bytes, self.offset);
        let (tag, mut body) = reader.record()?;
        self.offset = reader.position();
        self.replay_record(tag, &mut body)?;
        if tag == codec::END {
            reader.end()?;
        }
        Ok(())
    }

    fn replay_record(&mut self, tag: u8, body: &mut Reader<'_>) -> Result<()> {
        ensure!(
            self.pending_startup.is_none()
                || matches!(tag, codec::ENTER_WASM | codec::WRITE | codec::GLOBAL_WRITE),
            "instance startup missing from trace"
        );
        match tag {
            codec::ENTER_WASM => self.enter_wasm(body),
            codec::ENTER_HOST => self.enter_host(body),
            codec::LEAVE_HOST => self.leave_host(body),
            codec::LEAVE_WASM => self.leave_wasm(body),
            // This precedes the record of a guest's call or return to the host,
            // which replay has reached.
            codec::RELEASE => {
                let handle = body.u32()?;
                body.end()?;
                self.store.as_context_mut().0.rr_release_handle(handle)
            }
            codec::GROWTH_FAILED => bail!("growth failure outside of guest execution in trace"),
            _ => {
                self.require_idle()?;
                match tag {
                    codec::HOST
                    | codec::TYPE
                    | codec::GLOBAL
                    | codec::MEMORY
                    | codec::TABLE
                    | codec::TAG => {
                        let store = self.store.as_context_mut().0;
                        let scope = store.enter_gc_lifo_scope();
                        let result = init::replay_object(store, &self.trampolines, tag, body);
                        store.exit_gc_lifo_scope(scope);
                        result?;
                        self.note_defined();
                        Ok(())
                    }
                    codec::MODULE => self.module(body),
                    codec::INSTANCE => {
                        // Allocating the instance may collect garbage.
                        self.publish_parked(None)?;
                        let store = self.store.as_context_mut().0;
                        let (instance, startup) = init::replay_instance(store, body)?;
                        self.instances.push(instance)?;
                        self.pending_startup = startup;
                        self.note_defined();
                        Ok(())
                    }
                    codec::WRITE => self.write(body),
                    codec::GLOBAL_WRITE => self.global_write(body),
                    codec::RESIZE => self.resize(body),
                    codec::EVENT => self.event(body),
                    #[cfg(feature = "gc")]
                    codec::ALLOC => self.allocate(body),
                    #[cfg(feature = "gc")]
                    codec::READ => self.read_ref(body),
                    #[cfg(feature = "gc")]
                    codec::OBJECT_WRITE => self.object_write(body),
                    codec::END => {
                        // Activations may remain parked at host calls if the
                        // recording ended while they were unfinished.
                        body.end()?;
                        self.finished = true;
                        Ok(())
                    }
                    _ => bail!("unknown trace record {tag}"),
                }
            }
        }
    }

    /// Notes the objects that were just created.
    fn note_defined(&mut self) {
        let objects = self.store.as_context().0.rr_objects();
        self.defined = Defined {
            memories: objects.memories.len(),
            tables: objects.tables.len(),
            globals: objects.globals.len(),
            instances: self.instances.len(),
        };
    }

    /// Replays a module's first instantiation, which the instance's record
    /// follows.
    fn module(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let id = RecordedModule::from_u32(body.u32()?);
        let checksum = body.checksum()?;
        body.end()?;
        self.compile(id, &checksum)?;
        ensure!(id == self.next_module, "invalid module ID in trace");
        self.next_module = RecordedModule::from_u32(id.as_u32() + 1);
        Ok(())
    }

    /// Compiles the trace's module with the given ID, if it has not been.
    fn compile(&mut self, id: RecordedModule, checksum: &Checksum) -> Result<()> {
        let modules = &self.store.as_context().0.rr_objects().modules;
        if modules.is_valid(id) {
            return Ok(());
        }
        ensure!(id == modules.next_key(), "invalid module ID in trace");
        let bytecode = self.bytecode.get(checksum)?;
        let module = init::compile_module(self.store.engine(), bytecode)?;
        let store = self.store.as_context_mut().0;
        store.rr_session().objects.modules.push(module)?;
        Ok(())
    }

    /// Starts a recorded call into Wasm.
    fn enter_wasm(&mut self, body: &mut Reader<'_>) -> Result<()> {
        self.require_idle()?;
        let func = RecordedFunc::from_u32(body.u32()?);
        let call = RecordedCall::from_u32(body.u32()?);
        ensure!(
            self.activations.iter().all(|a| a.call != call),
            "duplicate call ID in trace"
        );
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let info = store.rr_objects().funcs.get(func);
        let info = info
            .filter(|f| !f.host)
            .ok_or_else(|| format_err!("invalid function ID in trace"))?;
        match self.pending_startup.take() {
            Some(startup) => ensure!(func == startup, "instance startup missing from trace"),
            None => ensure!(!info.startup, "repeated instance startup in trace"),
        }
        let mut values = TryVec::new();
        values.resize_with(info.params.len().max(info.results.len()), || {
            ValRaw::v128(0)
        })?;
        let ty = info.func.load_ty(store);
        let params = ty.params().collect::<Vec<_>>();
        let mut slots = TryVec::new();
        slots.resize_with(params.len(), MaybeUninit::uninit)?;
        body.values(&params, &mut slots, store)?;
        body.end()?;
        for (slot, value) in slots.iter().zip(values.iter_mut()) {
            // SAFETY: `values` initialized this.
            *value = unsafe { slot.assume_init() };
        }
        self.activations.reserve(1)?;
        let serial = self.next_serial;
        self.next_serial += 1;
        let activation = Activation::new(store, &self.trampolines, serial, func, call, values)?;
        self.activations.push(activation)?;
        self.resume(self.activations.len() - 1, None)
    }

    /// Checks a guest's call to a host function.
    fn enter_host(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let func = RecordedFunc::from_u32(body.u32()?);
        let call = RecordedCall::from_u32(body.u32()?);
        let Some(Observed::Host {
            activation,
            func: observed,
            values,
        }) = self.observed.take()
        else {
            bail!("the guest diverged from the trace: it should have called the host");
        };
        ensure!(
            func == observed,
            "the guest diverged from the trace: it should have called host function {}, \
             but called {}",
            func.as_u32(),
            observed.as_u32(),
        );
        let hosts = self.activations.iter().filter_map(|a| a.host.as_ref());
        ensure!(
            hosts.clone().all(|h| h.call != call),
            "duplicate call ID in trace"
        );
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        self.scratch.clear();
        // SAFETY: the activation is parked at the call, for which it
        // initialized the parameter slots with this store's values.
        unsafe {
            store.rr_boundary_refs(func, values.as_ptr(), false, Direction::ToHost)?;
            let session = store.rr_session();
            let params = &session.objects.funcs[func].params;
            codec::values(&mut self.scratch, params, values.as_ptr(), session.refs())?
        };
        ensure!(
            body.rest() == &self.scratch[..],
            "the guest diverged from the trace: it called host function {} with \
             different arguments",
            func.as_u32(),
        );
        let activation = self.activations.iter_mut().find(|a| a.call == activation);
        activation.unwrap().host = Some(HostCall { call, func, values });
        Ok(())
    }

    /// Returns from a host call with the recorded results.
    fn leave_host(&mut self, body: &mut Reader<'_>) -> Result<()> {
        self.require_idle()?;
        let call = RecordedCall::from_u32(body.u32()?);
        let (index, mut host) = self
            .activations
            .iter()
            .enumerate()
            .find_map(|(i, a)| Some((i, a.host.filter(|h| h.call == call)?)))
            .ok_or_else(|| format_err!("invalid call ID in trace"))?;
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let ty = store.rr_objects().funcs[host.func].func.load_ty(store);
        let results = ty.results().collect::<Vec<_>>();
        // SAFETY: the activation is parked at this call.
        let values = unsafe { host.values.as_mut() };
        let outcome = body.outcome(&results, values, store)?;
        self.resume(index, outcome.err())
    }

    /// Checks how a call into Wasm returned.
    fn leave_wasm(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let Some(Observed::Complete { activation, result }) = self.observed.take() else {
            bail!("the guest diverged from the trace: it should have returned");
        };
        let index = self
            .activations
            .iter()
            .position(|a| a.call == activation)
            .unwrap();
        let activation = &self.activations[index];
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let mut exception = 0;
        match &result {
            // SAFETY: the activation holds its results, which are this store's
            // values.
            Ok(()) => unsafe {
                let results = activation.results();
                store.rr_boundary_refs(activation.func, results, true, Direction::ToHost)?;
            },
            // The recorded host may not have seen the exception itself.
            Err(e) if e.is::<crate::ThrownException>() && !body.has_null_exception() => {
                exception = store.rr_pending_exception_raw();
                store.rr_create_handle(exception)?;
            }
            Err(_) => {}
        }
        let session = store.rr_session();
        self.scratch.clear();
        // SAFETY: as above.
        unsafe {
            codec::outcome(
                &mut self.scratch,
                codec::LEAVE_WASM,
                activation.call,
                &result,
                &session.objects.funcs[activation.func].results,
                activation.results(),
                session.refs(),
                exception,
            )?;
        }
        let (_, observed) = Reader::new(&self.scratch).record()?;
        ensure!(
            body.bytes() == observed.bytes(),
            "the guest diverged from the trace: it returned different results{}",
            match &result {
                Ok(()) => String::new(),
                Err(e) => format!(" ({e:#})"),
            }
        );
        // Keep these in the order in which they started.
        let activation = self.activations.drain(index..=index).next().unwrap();
        self.retire(activation)
    }

    /// Replays a write to a memory by the host.
    fn write(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let id = RecordedMemory::from_u32(body.u32()?);
        let offset = usize::try_from(body.u64()?)?;
        let bytes = body.rest();
        let store = self.store.as_context_mut().0;
        let memory = store.rr_replayed_memory(id)?;
        let range = offset
            .checked_add(bytes.len())
            .filter(|end| *end <= memory.internal_data_size(store))
            .map(|end| offset..end)
            .ok_or_else(|| format_err!("out-of-bounds memory write in trace"))?;
        store.rr_memory_written(memory, range.clone())?;
        memory.data_mut_untracked(store)[range].copy_from_slice(bytes);
        Ok(())
    }

    /// Replays the host's write to a component instance flag.
    fn global_write(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let id = RecordedGlobal::from_u32(body.u32()?);
        let value = Val::I32(body.u32()?.cast_signed());
        body.end()?;
        let store = self.store.as_context_mut().0;
        let global = store.rr_objects().globals.get(id).copied();
        let global = global.ok_or_else(|| format_err!("invalid global ID in trace"))?;
        ensure!(
            global._ty(store).mutability().is_var(),
            "write to an immutable global in trace"
        );
        global._set(store, value)
    }

    /// Replays a memory's growth by the host.
    fn resize(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let id = RecordedMemory::from_u32(body.u32()?);
        let new_size = body.u64()?;
        body.end()?;
        let store = self.store.as_context_mut().0;
        let memory = store.rr_replayed_memory(id)?;
        let old_size = u64::try_from(memory.internal_data_size(store))?;
        let page_size = memory.wasmtime_ty(store).page_size();
        let delta = new_size
            .checked_sub(old_size)
            .filter(|delta| delta % page_size == 0)
            .ok_or_else(|| format_err!("invalid memory size in trace"))?;
        memory.grow(StoreContextMut(store), delta / page_size)?;
        Ok(())
    }

    /// Replays an embedder event.
    fn event(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let tag = body.u32()?;
        let payload = body.rest();
        #[cfg(feature = "debug")]
        if self.store.as_context_mut().0.rr_moving_backward() {
            return Ok(());
        }
        for (_, observer) in self.observers.iter_mut().filter(|(t, _)| *t == tag) {
            observer(payload).context("failed to observe a trace event")?;
        }
        if self.stop_at_events {
            self.stop = Some(ReplayStop::Event(tag));
        }
        Ok(())
    }

    /// Runs the activation at `index`, which is new or parked at a host call,
    /// until its next yield; see [`Activation::resume`].
    fn resume(&mut self, index: usize, pending: Option<Error>) -> Result<()> {
        let mut growth_failures = self.read_growth_failures()?;
        growth_failures.reverse();
        self.replaying().growth_failures = growth_failures;
        self.resume_running(index, pending)
    }

    /// Like `resume`, but the activation may also be paused.
    fn resume_running(&mut self, index: usize, pending: Option<Error>) -> Result<()> {
        self.publish_parked(Some(index))?;
        let store = self.store.as_context_mut().0;
        let activation = &mut self.activations[index];
        let yielded = activation.resume(store, pending)?;
        // Failures from code that ran on the activation and could not return
        // them.
        if let Some(e) = store.rr_session().failure.take() {
            return Err(e);
        }
        if let Yield::Stopped = yielded {
            self.paused = Some(activation.call);
            match self.replaying().stop.take() {
                Some(Stop::Yield) => self.yield_now = true,
                #[cfg(feature = "debug")]
                Some(Stop::Breakpoint) => {
                    self.at_check = true;
                    self.stop = Some(ReplayStop::Breakpoint);
                }
                #[cfg(feature = "debug")]
                Some(Stop::EventTarget) => {
                    self.at_check = true;
                    self.at_target = true;
                }
                #[cfg(feature = "debug")]
                Some(Stop::Watchpoint(hit)) => {
                    let start = usize::try_from(hit.address).unwrap_or(usize::MAX);
                    let len = usize::try_from(hit.len).unwrap_or(usize::MAX);
                    self.pending_write = Some((hit.memory, start..start.saturating_add(len)));
                    self.stop = Some(ReplayStop::Watchpoint(hit));
                }
                Some(Stop::Interrupt) => self.stop = Some(ReplayStop::Interrupted),
                None => bail!("replay activation stopped for no reason"),
            }
            return Ok(());
        }
        let activation = activation.call;
        ensure!(
            self.replaying().growth_failures.is_empty(),
            "the guest diverged from the trace: its memory or table growth should have failed"
        );
        self.observed = Some(match yielded {
            Yield::Host(func, values) => Observed::Host {
                activation,
                func,
                values,
            },
            Yield::Complete(result) => Observed::Complete { activation, result },
            Yield::Stopped => unreachable!(),
        });
        Ok(())
    }

    /// Publishes the suspended activations, other than the `running` one, whose
    /// frames the garbage collector must trace.
    fn publish_parked(&mut self, running: Option<usize>) -> Result<()> {
        let store = self.store.as_context_mut().0;
        let parked = &mut store.rr_session().mode.replaying().parked;
        parked.clear();
        for (i, activation) in self.activations.iter().enumerate() {
            if Some(i) != running && activation.host.is_some() {
                parked.push(activation.exit_state())?;
            }
        }
        Ok(())
    }

    /// Reads the records of growth that fails while guest code runs, which
    /// follow the record that resumes it.
    fn read_growth_failures(&mut self) -> Result<TryVec<GrowthFailure>> {
        let mut reader = Reader::at(&self.trace.bytes, self.offset);
        let mut failures = TryVec::new();
        while reader.peek_tag() == Some(codec::GROWTH_FAILED) {
            let (_, mut body) = reader.record()?;
            failures.push(GrowthFailure::decode(&mut body)?)?;
        }
        self.offset = reader.position();
        Ok(failures)
    }
}

impl Drop for Replayer {
    fn drop(&mut self) {
        while let Some(activation) = self.activations.pop() {
            activation.dispose(self.store.as_context_mut().0);
        }
        for activation in self.checkpoints.take_retired() {
            activation.dispose(self.store.as_context_mut().0);
        }
    }
}

impl StoreOpaque {
    /// Traces the GC roots in the frames of suspended activations, which are on
    /// other fibers than the running one.
    #[cfg(feature = "gc")]
    pub(crate) fn rr_trace_parked_roots(&self, roots: &mut crate::vm::GcRootsList) {
        let Some(Mode::Replaying(replaying)) = self.rr.session.as_deref().map(|s| &s.mode) else {
            return;
        };
        for exit in &replaying.parked {
            // SAFETY: the stacks of suspended activations are intact, and their
            // exit state describes their Wasm frames.
            let _ = unsafe {
                let fp =
                    crate::vm::VMStoreContext::wasm_exit_fp_from_trampoline_fp(exit.trampoline_fp);
                wasmtime_unwinder::visit_frames::<()>(
                    self.unwinder(),
                    exit.pc,
                    fp,
                    exit.entry_fp,
                    |frame| {
                        StoreOpaque::trace_wasm_stack_frame(self.modules(), roots, frame);
                        core::ops::ControlFlow::Continue(())
                    },
                )
            };
        }
    }

    fn rr_replayed_memory(&self, id: RecordedMemory) -> Result<Memory> {
        self.rr_objects()
            .memories
            .get(id)
            .copied()
            .ok_or_else(|| format_err!("invalid memory ID in trace"))
    }
}
