//! The only driver of replay activations. Host boundaries yield control here;
//! no host callback recursively runs another activation.
//!
//! Each guest activation runs on a raw fiber whose only frames, whenever it is
//! suspended, are generated code and the fiber library's audited switch and
//! start routines:
//!
//! ```text
//! fiber start (asm) -> ReplayStart -> array-to-Wasm -> guest ...
//!     -> Wasm-to-array -> ReplayHostCall -> fiber switch (asm)
//! ```
//!
//! The activation's generated code communicates with this driver only through
//! its `VMReplayControl`. A guest return or trap is reported by a final yield
//! from `ReplayStart`, after which the fiber is destroyed without resuming it.
//! Runtime state that an ordinary call keeps on the host stack, such as the
//! `CallThreadState` and the store's entry/exit registers, is installed by
//! this driver around each resumption instead. Value buffers belong to the
//! driver; suspended trampolines only borrow them.

use super::*;
use crate::StoreContextMut;
use crate::runtime::func::EntryStoreContext;
use crate::runtime::vm::mpk::{self, ProtectionMask};
use crate::runtime::vm::{
    SendSyncPtr, VMArrayCallHostFuncContext, VMArrayCallNative, VMCommonStackInformation,
    VMOpaqueContext, VMReplayControl, VmPtr,
};
use core::mem::MaybeUninit;
use core::task::Poll;
use wasmtime_environ::{
    FuncKey, VM_REPLAY_DEBUG, VM_REPLAY_HOST_CALL, VM_REPLAY_RETURNED, VM_REPLAY_TRAPPED,
};

mod checkpoint;
pub use checkpoint::Checkpoint;
use wasmtime_fiber::RawFiber;

/// The generated code that replay activations run. It is compiled into an
/// otherwise empty module, which keeps it alive.
pub(super) struct Trampolines {
    module: crate::Module,
    start: SendSyncPtr<u8>,
    host_call: SendSyncPtr<u8>,
}

impl Trampolines {
    fn new(engine: &crate::Engine) -> Result<Self> {
        ensure!(
            RawFiber::is_supported(),
            "replay is not supported on this platform"
        );
        #[cfg(any(feature = "cranelift", feature = "winch"))]
        {
            let module = crate::Module::new(engine, b"\0asm\x01\0\0\0")?;
            let code = module.compiled_module();
            let get = |key| {
                code.replay_trampoline(key)
                    .map(SendSyncPtr::new)
                    .ok_or_else(|| format_err!("engine is not configured for replay"))
            };
            Ok(Trampolines {
                start: get(FuncKey::ReplayStart)?,
                host_call: get(FuncKey::ReplayHostCall)?,
                module,
            })
        }
        #[cfg(not(any(feature = "cranelift", feature = "winch")))]
        {
            let _ = engine;
            bail!("replay requires a compiler")
        }
    }

    /// Creates a replay stub for a recorded host function. Its array-call
    /// entry is the generated host-call trampoline.
    pub(super) fn host_stub(&self, store: &mut StoreOpaque, ty: crate::FuncType) -> Result<Func> {
        // SAFETY: the host-call trampoline has the array calling convention
        // for every signature, and the stub keeps the module owning it alive.
        unsafe {
            Func::rr_replay_stub(
                store,
                ty,
                core::mem::transmute::<*mut u8, VMArrayCallNative>(self.host_call.as_ptr()),
                try_new::<Box<_>>(self.module.clone())?,
            )
        }
    }
}

/// A guest activation and everything its suspended fiber refers to. All of
/// it has a stable address until the activation is disposed.
struct Activation {
    // Identifies this activation across checkpoints.
    serial: u64,
    fiber: Option<RawFiber>,
    // An owned allocation (from `Box`). The fiber's generated code writes to
    // it while running, so the driver only accesses it through raw pointers
    // and never holds a reference across a resume.
    control: SendSyncPtr<VMReplayControl>,
    // The activation's `VMStoreContext` state while it is not running, and
    // the store's while it is.
    context: core::mem::ManuallyDrop<EntryStoreContext>,
    // The stack-switching information `context.stack_chain` points to.
    _stack: Box<VMCommonStackInformation>,
    mpk: Option<ProtectionMask>,
    func: usize,
    call: usize,
    host: Option<HostCall>,
    // Never resize this buffer while the fiber can access it.
    values: Vec<ValRaw>,
}

// SAFETY: the raw pointers in an activation only refer to its own boxed
// storage, its fiber stack, and code and contexts rooted in the store that
// the driver exclusively borrows. No thread-local state refers to the fiber
// while it is suspended, and only the driver resumes it.
unsafe impl Send for Activation {}

impl Activation {
    fn dispose(mut self, store: &mut StoreOpaque) {
        // Nothing on a raw fiber's stack needs to be run or unwound.
        if let Some(fiber) = self.fiber.take() {
            store.deallocate_fiber_stack(fiber.into_stack());
        }
        // SAFETY: the fiber that referred to the control block is gone.
        drop(unsafe { Box::from_raw(self.control.as_ptr()) });
    }
}

/// A borrowed array-call buffer on a suspended activation's stack. Only the
/// driver accesses it, and only before resuming or disposing that activation.
#[derive(Clone, Copy)]
struct HostCall {
    func: usize,
    call: usize,
    values: SendSyncPtr<[MaybeUninit<ValRaw>]>,
}

enum Observed {
    Host(usize, HostCall),
    Complete(usize, Result<()>),
}

struct Driver<'a, T: 'static> {
    store: &'a mut StoreInner<T>,
    trampolines: Trampolines,
    reader: Reader<'a>,
    activations: Vec<Activation>,
    observed: Option<Observed>,
    scratch: Vec<u8>,
    instances: Vec<crate::Instance>,
    // A newly allocated instance must run its startup before any other code
    // can observe it, including when the trace is malformed.
    pending_startup: Option<usize>,
    // Observers of embedder-defined events, by tag.
    observers: Vec<(u32, Box<dyn FnMut(&[u8]) -> Result<()> + Send>)>,
    stop_at_events: bool,
    finished: bool,
    // The activation stopped at a debug event, which resumes before any
    // further trace events are processed.
    paused: Option<u64>,
    // The write that the paused activation, stopped at a watchpoint, performs
    // when resumed. It has passed its shadow check, so checkpoints that would
    // mark its pages clean must leave them dirty.
    pending_write: Option<(Memory, Range<usize>)>,
    // Why the current `run` should return, once the current step completes.
    stop: Option<ReplayStop>,
    next_serial: u64,
    checkpoints: checkpoint::Checkpoints,
}

/// Replays a trace in a store, one stop at a time.
///
/// Created by [`Store::replayer`](crate::Store::replayer). Dropping a replayer
/// frees all of its suspended activations; the store then remains available
/// for inspection, but its functions cannot be called.
pub struct Replayer<'a, T: 'static> {
    driver: Driver<'a, T>,
}

/// Why [`Replayer::run`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayStop {
    /// The entire trace was replayed.
    Finished,
    /// Guest code reached a breakpoint or single-stepped, as configured
    /// through the store's breakpoint API. The stopped frames are available
    /// from `Replayer::debug_exit_frames`.
    Breakpoint,
    /// An embedder event with this tag was replayed, after its observers ran.
    /// Only reported when enabled with [`Replayer::stop_at_events`].
    Event(u32),
    /// Guest code is about to write bytes watched with
    /// [`Memory::debug_watch`](crate::Memory::debug_watch); the write happens
    /// when replay continues. The stopped frames are available as for
    /// breakpoints.
    #[cfg(feature = "debug")]
    Watchpoint(crate::WatchpointHit),
}

impl<'a, T: Send + 'static> Replayer<'a, T> {
    pub(super) fn new(store: &'a mut StoreInner<T>, trace: &'a Trace) -> Result<Self> {
        store.rr_validate()?;
        ensure!(
            store.engine().is_replaying(),
            "replay requires RRConfig::Replaying"
        );
        // `Trace` construction checked the version and framing.
        let mut reader = Reader::new(&trace.bytes);
        reader.take(codec::MAGIC.len())?;
        let trampolines = Trampolines::new(store.engine())?;
        store.rr.session = Some(try_new::<Box<_>>(Session {
            objects: Objects::default(),
            mode: Mode::Replaying {
                growth_failures: Vec::new(),
                #[cfg(feature = "debug")]
                watchpoint: None,
                histories: Default::default(),
                page_size: 4096,
                parked: Vec::new(),
                embedder_access: false,
            },
            pending: Vec::new(),
            failure: None,
            sink: None,
        })?);
        Ok(Replayer {
            driver: Driver {
                store,
                trampolines,
                reader,
                activations: Vec::new(),
                observed: None,
                scratch: Vec::new(),
                instances: Vec::new(),
                pending_startup: None,
                observers: Vec::new(),
                stop_at_events: false,
                finished: false,
                paused: None,
                pending_write: None,
                stop: None,
                next_serial: 0,
                checkpoints: Default::default(),
            },
        })
    }

    /// Calls `observer` with each recorded event of type `E` as replay reaches
    /// it, including again after rewinding. Observers cannot affect replay.
    pub fn on_event<E: TraceEvent>(&mut self, mut observer: impl FnMut(E) + Send + 'static) {
        self.driver.observers.push((
            E::TAG,
            Box::new(move |bytes| {
                observer(postcard::from_bytes(bytes)?);
                Ok(())
            }),
        ));
    }

    /// Replays until the next stop.
    ///
    /// This yields to the async executor periodically, so a long replay can
    /// be cancelled by dropping the future.
    pub async fn run(&mut self) -> Result<ReplayStop> {
        let driver = &mut self.driver;
        driver.set_embedder_access(false);
        core::future::poll_fn(|cx| {
            // Give the executor a chance to cancel long traces between events.
            for _ in 0..256 {
                if driver.finished && driver.paused.is_none() {
                    return Poll::Ready(Ok(ReplayStop::Finished));
                }
                if let Err(e) = driver.step() {
                    return Poll::Ready(Err(e.context(format!(
                        "replaying trace at byte {}",
                        driver.reader.position()
                    ))));
                }
                if let Some(stop) = driver.stop.take() {
                    return Poll::Ready(Ok(stop));
                }
            }
            cx.waker().wake_by_ref();
            Poll::Pending
        })
        .await
    }

    /// The frames of the activation stopped at a debug event, as for
    /// [`Store::debug_exit_frames`](crate::Store::debug_exit_frames): one
    /// exit frame for it, followed by one for each activation parked at a host
    /// call, most recently started first. For nested calls (a host function
    /// calling back into the guest) this is their logical call order. This is
    /// empty unless [`Replayer::run`] returned [`ReplayStop::Breakpoint`].
    ///
    /// The frames can be inspected through [`Replayer::store`] until replay
    /// continues or is restored.
    #[cfg(feature = "debug")]
    pub fn debug_exit_frames(&mut self) -> Vec<crate::FrameHandle> {
        let driver = &mut self.driver;
        let Some(paused) = driver.paused else {
            return Vec::new();
        };
        let store: &mut StoreOpaque = driver.store;
        // The paused activation is not parked at a host call; order it first.
        let mut order = driver
            .activations
            .iter_mut()
            .rev()
            .filter(|a| a.serial == paused || a.host.is_some())
            .collect::<Vec<_>>();
        order.sort_by_key(|a| a.serial != paused);
        let mut frames = Vec::new();
        for activation in order {
            activation.context.rr_swap();
            crate::runtime::vm::with_parked_activation(store, &mut activation.context, |store| {
                frames.extend(store.debug_exit_frames());
            });
            activation.context.rr_swap();
        }
        frames
    }

    /// Sets the granularity, in bytes, at which checkpoints track writes to
    /// guest memory: a checkpoint copies each such block written since the
    /// previous one. The default is 4096. Must be set before the first
    /// checkpoint.
    pub fn set_checkpoint_page_size(&mut self, size: usize) -> Result<()> {
        ensure!(size > 0, "checkpoint page size must be nonzero");
        let Mode::Replaying {
            histories,
            page_size,
            ..
        } = &mut self.driver.store.rr.session.as_mut().unwrap().mode
        else {
            unreachable!()
        };
        ensure!(
            histories.is_empty(),
            "checkpoint page size set after a checkpoint"
        );
        *page_size = size;
        Ok(())
    }

    /// Also stop [`Replayer::run`] after each embedder event is replayed.
    pub fn stop_at_events(&mut self, enabled: bool) {
        self.driver.stop_at_events = enabled;
    }

    /// Core instances constructed so far, in their recorded construction
    /// order. Components replay as their constituent core instances.
    pub fn instances(&self) -> &[crate::Instance] {
        &self.driver.instances
    }

    /// The store being replayed into, for inspection.
    ///
    /// Only the replay may change the store's state. Through this context,
    /// operations that would (calling functions, writing or growing memory,
    /// creating objects, allocating GC objects or collecting garbage) fail;
    /// infallible ones, such as [`Memory::data_mut`](crate::Memory::data_mut),
    /// instead make the replay fail when it continues. Debugger configuration,
    /// such as breakpoints and watchpoints, may be changed.
    pub fn store(&mut self) -> StoreContextMut<'_, T> {
        self.driver.set_embedder_access(true);
        StoreContextMut(self.driver.store)
    }

    pub(super) fn into_replay(mut self) -> Replay {
        Replay {
            instances: core::mem::take(&mut self.driver.instances),
        }
    }
}

impl<T: 'static> Driver<'_, T> {
    /// Checks that no guest code is running: every activation is parked at a
    /// matched host call.
    fn require_idle(&self, unmatched: &str, running: &str) -> Result<()> {
        ensure!(self.observed.is_none(), "{unmatched}");
        ensure!(
            self.activations.iter().all(|a| a.host.is_some()),
            "{running}"
        );
        Ok(())
    }

    fn step(&mut self) -> Result<()> {
        if let Some(serial) = self.paused.take() {
            self.pending_write = None;
            let index = self.index_of(serial);
            return self.resume(index, None, true);
        }
        let (tag, mut body) = self.reader.record()?;
        if self.pending_startup.is_some() {
            ensure!(
                matches!(tag, codec::ENTER_WASM | codec::GLOBAL_WRITE | codec::WRITE),
                "instance startup missing from trace"
            );
        }
        match tag {
            codec::END => {
                // Activations may remain suspended in host calls if the
                // recording ended while they were unfinished.
                ensure!(
                    self.observed.is_none() && self.activations.iter().all(|a| a.host.is_some()),
                    "trace ended while guest code was running"
                );
                body.end()?;
                self.reader.end()?;
                self.finished = true;
            }
            codec::EVENT => {
                self.require_idle(
                    "event before matching guest execution",
                    "event while guest is running",
                )?;
                let tag = body.u32()?;
                let payload = body.rest();
                for (_, observer) in self.observers.iter_mut().filter(|(t, _)| *t == tag) {
                    observer(payload).context("failed to observe a trace event")?;
                }
                if self.stop_at_events {
                    self.stop = Some(ReplayStop::Event(tag));
                }
            }
            codec::HOST
            | codec::MODULE
            | codec::INSTANCE
            | codec::GLOBAL
            | codec::GLOBAL_WRITE
            | codec::MEMORY
            | codec::TABLE => {
                self.require_idle(
                    "initialization before matching guest execution",
                    "initialization while guest is running",
                )?;
                if let Some((instance, startup)) =
                    super::init::replay_event(self.store, &self.trampolines, tag, &mut body)?
                {
                    self.instances.push(instance);
                    self.pending_startup = startup;
                }
            }
            codec::ENTER_WASM => {
                self.require_idle(
                    "guest execution diverged before EnterWasm",
                    "nested guest entry without a host boundary",
                )?;
                let id = usize::try_from(body.u32()?)?;
                let call = usize::try_from(body.u32()?)?;
                ensure!(
                    !self.activations.iter().any(|a| a.call == call),
                    "duplicate activation id"
                );
                let session = self.store.rr.session.as_ref().unwrap();
                let bound = session
                    .objects
                    .funcs
                    .get(id)
                    .ok_or_else(|| format_err!("unknown function id"))?;
                ensure!(!bound.host, "EnterWasm names a host function");
                match self.pending_startup.take() {
                    Some(startup) => ensure!(id == startup, "expected instance startup"),
                    None => ensure!(!bound.startup, "unexpected repeated instance startup"),
                }
                let mut values = value_slots(bound)?;
                body.values(&bound.params, &mut values, |id| {
                    session.objects.decode_ref(self.store, id)
                })?;
                body.end()?;
                let func = bound.func.vm_func_ref(self.store);
                self.activations
                    .try_reserve(1)
                    .map_err(|_| OutOfMemory::new(core::mem::size_of::<Activation>()))?;
                let activation = self.activate(func, id, call, values)?;
                self.activations.push(activation);
                self.resume(self.activations.len() - 1, None, false)?;
            }
            codec::ENTER_HOST => {
                let Some(Observed::Host(activation, mut call)) = self.observed.take() else {
                    bail!("expected guest to reach a host boundary");
                };
                let id = usize::try_from(body.u32()?)?;
                call.call = usize::try_from(body.u32()?)?;
                ensure!(
                    !self
                        .activations
                        .iter()
                        .filter_map(|a| a.host.as_ref())
                        .any(|h| h.call == call.call),
                    "duplicate host call id"
                );
                ensure!(
                    call.func == id,
                    "host call target diverged: expected {id}, observed {}",
                    call.func
                );
                let bound = &self.store.rr.session.as_ref().unwrap().objects.funcs[id];
                self.scratch.clear();
                codec::reserve(&mut self.scratch, codec::values_len(&bound.params))?;
                // SAFETY: the activation is suspended and initialized the
                // parameter slots. The trampoline checked the buffer size.
                unsafe {
                    codec::values(
                        &mut self.scratch,
                        &bound.params,
                        call.values.as_ptr().cast(),
                        |ptr| {
                            self.store
                                .rr
                                .session
                                .as_ref()
                                .unwrap()
                                .objects
                                .encode_ref(ptr)
                        },
                    )?
                };
                ensure!(
                    body.take(self.scratch.len())? == self.scratch,
                    "host call arguments diverged"
                );
                body.end()?;
                self.activations
                    .iter_mut()
                    .find(|a| a.call == activation)
                    .unwrap()
                    .host = Some(call);
            }
            codec::LEAVE_HOST => {
                ensure!(self.observed.is_none(), "unexpected LeaveHost");
                let id = usize::try_from(body.u32()?)?;
                let index = self
                    .activations
                    .iter()
                    .position(|a| a.host.as_ref().is_some_and(|h| h.call == id))
                    .ok_or_else(|| format_err!("LeaveHost without a pending host call"))?;
                let mut call = self.activations[index].host.take().unwrap();
                let session = self.store.rr.session.as_ref().unwrap();
                let bound = &session.objects.funcs[call.func];
                // SAFETY: the activation is parked at this host call. Its
                // array-call storage is live and exclusively borrowed here.
                let values = unsafe { call.values.as_mut() };
                let outcome = body.outcome(&bound.results, values, |id| {
                    session.objects.decode_ref(self.store, id)
                })?;
                self.resume(index, outcome.err(), false)?;
            }
            codec::LEAVE_WASM => {
                let Some(Observed::Complete(actual_call, result)) = self.observed.take() else {
                    bail!("expected guest to return or trap");
                };
                let call = usize::try_from(body.u32()?)?;
                ensure!(call == actual_call, "guest return activation diverged");
                let index = self
                    .activations
                    .iter()
                    .position(|a| a.call == call)
                    .unwrap();
                let activation = &self.activations[index];
                let bound = &self.store.rr.session.as_ref().unwrap().objects.funcs[activation.func];
                self.scratch.clear();
                // SAFETY: successful guest calls initialize all result slots.
                unsafe {
                    codec::outcome(
                        &mut self.scratch,
                        codec::LEAVE_WASM,
                        call,
                        &result,
                        &bound.results,
                        activation.values.as_ptr(),
                        |ptr| {
                            self.store
                                .rr
                                .session
                                .as_ref()
                                .unwrap()
                                .objects
                                .encode_ref(ptr)
                        },
                    )?;
                }
                ensure!(
                    body.bytes() == &self.scratch[5..],
                    "guest return values or trap diverged"
                );
                let activation = self.activations.remove(index);
                self.retire(activation);
            }
            codec::WRITE => {
                self.require_idle(
                    "memory update before matching guest execution",
                    "memory update while guest is running",
                )?;
                let id = usize::try_from(body.u32()?)?;
                let offset = usize::try_from(body.u64()?)?;
                let memory = *self
                    .store
                    .rr
                    .session
                    .as_ref()
                    .unwrap()
                    .objects
                    .memories
                    .get(id)
                    .ok_or_else(|| format_err!("unknown memory id"))?;
                let bytes = body.rest();
                let end = offset
                    .checked_add(bytes.len())
                    .ok_or_else(|| format_err!("memory range overflow"))?;
                ensure!(
                    end <= memory.internal_data_size(self.store),
                    "trace memory write out of bounds"
                );
                self.store.rr_dirty(memory, offset..end)?;
                memory
                    .rr_data_mut(self.store)
                    .get_mut(offset..end)
                    .ok_or_else(|| format_err!("trace memory write out of bounds"))?
                    .copy_from_slice(bytes);
            }
            codec::RESIZE => {
                self.require_idle(
                    "memory growth before matching guest execution",
                    "memory growth while guest is running",
                )?;
                let id = usize::try_from(body.u32()?)?;
                let old = body.u64()?;
                let new = body.u64()?;
                body.end()?;
                let memory = *self
                    .store
                    .rr
                    .session
                    .as_ref()
                    .unwrap()
                    .objects
                    .memories
                    .get(id)
                    .ok_or_else(|| format_err!("unknown memory id"))?;
                let page_size = memory.wasmtime_ty(self.store).page_size();
                ensure!(
                    old == u64::try_from(memory.internal_data_size(self.store))?
                        && new >= old
                        && (new - old) % page_size == 0,
                    "invalid replay memory growth"
                );
                memory.grow(StoreContextMut(self.store), (new - old) / page_size)?;
            }
            _ => bail!("unknown trace event {tag}"),
        }
        Ok(())
    }

    /// Creates the raw fiber and state for a new activation of `func`. Its
    /// arguments are in `values`, which also receives its results.
    fn activate(
        &mut self,
        func: NonNull<VMFuncRef>,
        id: usize,
        call: usize,
        mut values: Vec<ValRaw>,
    ) -> Result<Activation> {
        let store: &mut StoreOpaque = self.store;
        let mut stack_info = try_new::<Box<_>>(VMCommonStackInformation::running_default())?;
        let control = try_new::<Box<_>>(VMReplayControl {
            switch: VmPtr::from(NonNull::new(RawFiber::switch_routine() as *mut u8).unwrap()),
            switch_arg: VmPtr::from(NonNull::<u8>::dangling()),
            entry: VmPtr::from(func),
            entry_caller: VmPtr::from(VMOpaqueContext::from_vmcontext(store.default_caller())),
            entry_values: VmPtr::from(NonNull::from(values.as_mut_slice()).cast()),
            entry_values_len: values.len(),
            reason: 0,
            host_succeeded: 0,
            host_callee: None,
            host_values: None,
            host_values_len: 0,
        })?;
        let control = SendSyncPtr::new(NonNull::from(Box::leak(control)));
        let free_control = || {
            // SAFETY: no fiber refers to the control block.
            drop(unsafe { Box::from_raw(control.as_ptr()) })
        };
        let stack = match store.allocate_fiber_stack() {
            Ok(stack) => stack,
            Err(e) => {
                free_control();
                return Err(e);
            }
        };
        // SAFETY: the start trampoline follows the raw fiber entry contract,
        // and `control` outlives the fiber (see `Activation::dispose`). The
        // control block is not shared until the fiber runs.
        let fiber = unsafe {
            if let Some(top) = stack.top() {
                (*control.as_ptr()).switch_arg = VmPtr::from(NonNull::new(top).unwrap());
            }
            RawFiber::new(
                stack,
                core::mem::transmute::<*mut u8, wasmtime_fiber::RawFiberEntry>(
                    self.trampolines.start.as_ptr(),
                ),
                control.as_ptr().cast(),
            )
        };
        let fiber = match fiber {
            Ok(fiber) => fiber,
            Err((e, stack)) => {
                store.deallocate_fiber_stack(stack);
                free_control();
                return Err(e);
            }
        };
        debug_assert_eq!(
            // SAFETY: as above.
            unsafe { (*control.as_ptr()).switch_arg.as_ptr() },
            fiber.switch_arg()
        );

        // Like an ordinary async call, Wasm may use `max_wasm_stack` bytes of
        // the fiber's stack and host libcalls the remainder.
        let range = fiber.stack().range().unwrap();
        let stack_limit = range
            .end
            .saturating_sub(store.engine().config().max_wasm_stack)
            .max(range.start);
        let context =
            EntryStoreContext::rr_initial(store, stack_limit, NonNull::from(&mut *stack_info));
        self.next_serial += 1;
        Ok(Activation {
            serial: self.next_serial,
            fiber: Some(fiber),
            control,
            context,
            _stack: stack_info,
            mpk: store.has_pkey().then(ProtectionMask::all),
            func: id,
            call,
            host: None,
            values,
        })
    }

    fn set_embedder_access(&mut self, access: bool) {
        let Mode::Replaying {
            embedder_access, ..
        } = &mut self.store.rr.session.as_mut().unwrap().mode
        else {
            unreachable!()
        };
        *embedder_access = access;
    }

    fn growth_failures(&mut self) -> &mut Vec<[u8; codec::GROWTH_FAILED_LEN]> {
        let Mode::Replaying {
            growth_failures, ..
        } = &mut self.store.rr.session.as_mut().unwrap().mode
        else {
            unreachable!()
        };
        growth_failures
    }

    /// Frees a completed activation, unless a checkpoint can restore it.
    fn retire(&mut self, activation: Activation) {
        if let Some(activation) = self.checkpoints.retire(activation) {
            activation.dispose(self.store);
        }
    }

    fn index_of(&self, serial: u64) -> usize {
        self.activations
            .iter()
            .position(|a| a.serial == serial)
            .unwrap()
    }

    /// Runs the activation at `index` until its next yield and records what it
    /// yielded for. A `pending` error resumes a parked host call as failed.
    /// `paused` resumes an activation stopped at a debug event.
    fn resume(&mut self, index: usize, pending: Option<Error>, paused: bool) -> Result<()> {
        // Guest growth failures recorded while the activation ran follow the
        // event that resumes it; queue them for the growth libcalls.
        if !paused {
            let mut failures = Vec::new();
            while self.reader.peek_tag() == Some(codec::GROWTH_FAILED) {
                let (_, mut body) = self.reader.record()?;
                let record = body.take(codec::GROWTH_FAILED_LEN)?;
                body.end()?;
                failures.try_reserve(1)?;
                failures.push(record.try_into().unwrap());
            }
            *self.growth_failures() = failures;
        }

        // A collection while this activation runs must find the GC roots of
        // the others, which are parked on their own fibers.
        let Mode::Replaying { parked, .. } = &mut self.store.rr.session.as_mut().unwrap().mode
        else {
            unreachable!()
        };
        parked.clear();
        parked.try_reserve(self.activations.len())?;
        parked.extend(
            self.activations
                .iter()
                .enumerate()
                .filter(|(i, a)| *i != index && (a.host.is_some() || Some(a.serial) == self.paused))
                .map(|(_, a)| {
                    let cx = &a.context;
                    (
                        cx.last_wasm_exit_pc,
                        cx.last_wasm_exit_trampoline_fp,
                        cx.last_wasm_entry_fp,
                    )
                }),
        );

        let store: &mut StoreOpaque = self.store;
        let activation = &mut self.activations[index];
        let fiber = activation.fiber.as_mut().unwrap();
        let control = activation.control.as_ptr();
        // SAFETY: the activation is not running.
        unsafe {
            (*control).reason = 0;
            (*control).host_succeeded = u32::from(pending.is_none());
        }

        // Install this activation's runtime state. Everything installed here
        // is restored before returning, so nothing refers to the fiber while
        // it is suspended.
        let cx = store.vm_store_context_mut();
        cx.replay_control = Some(VmPtr::from(activation.control.as_non_null()));
        let guard_range = core::mem::replace(
            &mut cx.async_guard_range,
            fiber
                .stack()
                .guard_range()
                .unwrap_or(core::ptr::null_mut()..core::ptr::null_mut()),
        );
        let mpk = activation.mpk.map(|mask| {
            let current = mpk::current_mask();
            mpk::allow(mask);
            current
        });
        activation.context.rr_swap();
        let context = &mut *activation.context;
        let result = crate::runtime::vm::catch_replay_traps(store, context, pending, || {
            // SAFETY: the fiber's control block and buffers are live, and the
            // driver installed its runtime state above. Only parked
            // activations are resumed, never terminal ones.
            let resumed = unsafe { fiber.resume() };
            debug_assert!(resumed.is_ok());
            // SAFETY: the activation has yielded.
            unsafe { (*control).reason != VM_REPLAY_TRAPPED }
        });
        activation.context.rr_swap();
        if let Some(mask) = mpk {
            activation.mpk = Some(mpk::current_mask());
            mpk::allow(mask);
        }
        let cx = store.vm_store_context_mut();
        cx.async_guard_range = guard_range;
        cx.replay_control = None;
        #[cfg(feature = "std")]
        crate::runtime::vm::AsyncWasmCallState::assert_current_state_not_in_range(
            fiber.stack().range().unwrap(),
        );
        verify_suspended_stack(fiber)?;
        // Failures from code that ran on the activation and could not return
        // them, such as checkpoint tracking of host-side writes.
        if let Some(e) = store.rr.session.as_mut().unwrap().failure.take() {
            return Err(e);
        }
        let serial = activation.serial;
        let call = activation.call;

        // SAFETY: the activation has yielded, and this reference does not
        // outlive this function.
        let control = unsafe { &*control };
        if control.reason == VM_REPLAY_DEBUG {
            self.paused = Some(serial);
            #[cfg(feature = "debug")]
            let Mode::Replaying { watchpoint, .. } =
                &mut self.store.rr.session.as_mut().unwrap().mode
            else {
                unreachable!()
            };
            #[cfg(feature = "debug")]
            if let Some(hit) = watchpoint.take() {
                let start = usize::try_from(hit.address).unwrap_or(usize::MAX);
                let end = start.saturating_add(usize::try_from(hit.len).unwrap_or(usize::MAX));
                self.pending_write = Some((hit.memory, start..end));
                self.stop = Some(ReplayStop::Watchpoint(hit));
                return Ok(());
            }
            self.stop = Some(ReplayStop::Breakpoint);
            return Ok(());
        }
        ensure!(
            self.growth_failures().is_empty(),
            "replay diverged: a recorded guest growth failure did not occur"
        );
        let store: &mut StoreOpaque = self.store;
        match control.reason {
            VM_REPLAY_HOST_CALL => {
                let callee = control.host_callee.unwrap().as_non_null();
                let values = control.host_values.unwrap().as_non_null();
                let values_len = control.host_values_len;
                // SAFETY: replay host calls are only reachable through stubs
                // whose context is a `VMArrayCallHostFuncContext`.
                let func = unsafe {
                    NonNull::from(
                        &VMArrayCallHostFuncContext::from_opaque(callee)
                            .as_ref()
                            .func_ref,
                    )
                };
                let objects = &store.rr.session.as_ref().unwrap().objects;
                let id = objects.find_func(func)?;
                let bound = &objects.funcs[id];
                ensure!(bound.host, "replay boundary is not a host function");
                ensure!(
                    values_len >= bound.params.len().max(bound.results.len()),
                    "invalid replay value storage"
                );
                let values = SendSyncPtr::from(NonNull::slice_from_raw_parts(
                    values.cast::<MaybeUninit<ValRaw>>(),
                    values_len,
                ));
                self.observed = Some(Observed::Host(
                    call,
                    HostCall {
                        func: id,
                        call: 0,
                        values,
                    },
                ));
            }
            VM_REPLAY_RETURNED | VM_REPLAY_TRAPPED => {
                let index = self.index_of(serial);
                self.activations[index].fiber.as_mut().unwrap().finish()?;
                self.observed = Some(Observed::Complete(call, result));
            }
            reason => bail!("replay activation yielded with unknown reason {reason}"),
        }
        Ok(())
    }
}

impl<T: 'static> Drop for Driver<'_, T> {
    fn drop(&mut self) {
        // Free activations in reverse order while the store and session still
        // exist. Nothing runs on their fibers, and no original host
        // implementation runs during cancellation.
        while let Some(activation) = self.activations.pop() {
            activation.dispose(self.store);
        }
        for activation in self.checkpoints.take_retired() {
            activation.dispose(self.store);
        }
        self.store.rr.session = None;
        // Replayed host functions can only run under the driver.
        self.store.rr.replayed = true;
    }
}

fn value_slots(bound: &RecordedFunc) -> Result<Vec<ValRaw>> {
    let len = bound.params.len().max(bound.results.len());
    let mut values = Vec::new();
    values
        .try_reserve(len)
        .map_err(|_| OutOfMemory::new(len.saturating_mul(core::mem::size_of::<ValRaw>())))?;
    values.resize(len, ValRaw::v128(0));
    Ok(values)
}

/// Checks that a suspended activation's stack contains only generated code,
/// beneath the switch routine and above the fiber's start routine. This walks
/// the frame-pointer chain, which all generated code maintains.
#[cfg(has_host_compiler_backend)]
fn verify_suspended_stack(fiber: &RawFiber) -> Result<()> {
    use crate::runtime::vm::Unwind;
    let Some((mut pc, mut fp)) = fiber.suspended_frame() else {
        return Ok(());
    };
    let range = fiber.stack().range().unwrap();
    let unwind = &crate::runtime::vm::UnwindHost;
    loop {
        ensure!(
            crate::module::lookup_code(pc).is_some(),
            "replay activation suspended with a host frame at {pc:#x}"
        );
        ensure!(
            range.contains(&fp),
            "replay activation has an invalid frame chain"
        );
        // SAFETY: `fp` is a frame pointer of generated code on this stack.
        let (older_pc, older_fp) = unsafe {
            (
                unwind.get_next_older_pc_from_fp(fp),
                *(fp as *const usize).byte_add(unwind.next_older_fp_from_fp_offset()),
            )
        };
        // The start trampoline's caller is the fiber's start routine, whose
        // frame pointer is the initial one rather than a frame on the stack.
        if !range.contains(&older_fp) || older_fp <= fp {
            return Ok(());
        }
        (pc, fp) = (older_pc, older_fp);
    }
}

// Replay requires native code, so there is nothing to verify.
#[cfg(not(has_host_compiler_backend))]
fn verify_suspended_stack(_: &RawFiber) -> Result<()> {
    Ok(())
}
