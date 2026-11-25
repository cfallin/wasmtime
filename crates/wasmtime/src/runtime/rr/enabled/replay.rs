//! The replay of a trace.
//!
//! A single loop consumes the trace and owns all guest activations: host calls
//! suspend an activation and yield to this loop, so no guest code ever runs
//! beneath a host function. Activations may complete in any order.

use super::*;
use crate::{AsContext, Extern, Instance, StoreContext, StoreContextMut, Val};
use alloc::collections::BTreeMap;
use core::task::Poll;
use objects::RecordedGlobal;

mod activation;
mod init;

use activation::{Activation, HostCall, HostValues, Trampolines, Yield};

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
    activations: TryVec<Activation>,
    /// What guest code did when it last ran, until the trace's next record is
    /// checked against it.
    observed: Option<Observed>,
    instances: TryVec<Instance>,
    /// The startup function of a newly created instance, which must run before
    /// anything else.
    pending_startup: Option<RecordedFunc>,
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
        let tunables = engine.tunables();
        ensure!(
            !tunables.debug_guest,
            "replay does not support guest debugging"
        );
        // Fuel and epochs would stop or suspend replay where the recording did
        // not. Recordings made with them replay without them.
        ensure!(
            !tunables.consume_fuel && !tunables.epoch_interruption,
            "replay does not support fuel or epoch interruption"
        );
        let trampolines = Trampolines::new(engine)?;
        let mut store = Store::try_new(engine, ())?;
        store.as_context_mut().0.rr.session = Some(try_new::<Box<_>>(Session {
            objects: Objects::default(),
            mode: Mode::Replaying(Replaying::default()),
            failure: None,
        })?);
        Ok(Replayer {
            store,
            trampolines,
            bytecode: Bytecode::default(),
            trace: trace.into(),
            // `Trace` checks the magic.
            offset: codec::MAGIC.len(),
            activations: TryVec::new(),
            observed: None,
            instances: TryVec::new(),
            pending_startup: None,
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

    /// Replays until the next stop.
    ///
    /// This yields to the async executor periodically, so a long replay can
    /// be cancelled by dropping the future, and then resumed by calling this
    /// again. After an error, replay cannot be resumed.
    pub async fn run(&mut self) -> Result<ReplayStop> {
        // Invalidate handles to state that Wasm may mutate, as ordinary calls
        // into Wasm do when they return to the host.
        self.store.as_context_mut().0.on_wasm_to_host_transition();
        core::future::poll_fn(|cx| {
            for _ in 0..256 {
                if self.finished {
                    return Poll::Ready(Ok(ReplayStop::Finished));
                }
                let offset = self.offset;
                if let Err(e) = self.step() {
                    let e = e.context(format!("replaying the trace record at byte {offset}"));
                    return Poll::Ready(Err(e));
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
        &self.instances
    }

    /// The store being replayed into, for inspection.
    ///
    /// Only the replay may change the store's state, so this gives only shared
    /// access to it. GC objects are read as [`Unrooted`](crate::Unrooted)
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

    /// Looks up `instance`'s export `name`, as [`Instance::get_export`] does.
    ///
    /// Functions are initialized lazily, which [`Replayer::store`]'s shared
    /// access does not allow. A replay's functions can be inspected, but not
    /// called.
    pub fn get_export(&mut self, instance: Instance, name: &str) -> Option<Extern> {
        instance.get_export(&mut self.store, name)
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
        let trace = self.trace.clone();
        let mut reader = Reader::at(&trace.bytes, self.offset);
        let (tag, mut body) = reader.record()?;
        self.offset = reader.position();
        self.replay_record(tag, &mut body)?;
        if tag == codec::END {
            reader.end()?;
        }
        self.failed = false;
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
            codec::GROWTH_FAILED => bail!("growth failure outside of guest execution in trace"),
            _ => {
                self.require_idle()?;
                match tag {
                    codec::HOST | codec::MODULE | codec::GLOBAL | codec::MEMORY | codec::TABLE => {
                        let store = self.store.as_context_mut().0;
                        init::replay_object(store, &self.trampolines, &self.bytecode, tag, body)
                    }
                    codec::INSTANCE => {
                        // Allocating the instance may collect garbage.
                        self.publish_parked(None)?;
                        let store = self.store.as_context_mut().0;
                        let (instance, startup) = init::replay_instance(store, body)?;
                        self.instances.push(instance)?;
                        self.pending_startup = startup;
                        Ok(())
                    }
                    codec::WRITE => self.write(body),
                    codec::GLOBAL_WRITE => self.global_write(body),
                    codec::RESIZE => self.resize(body),
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

    /// Starts a recorded call into Wasm.
    fn enter_wasm(&mut self, body: &mut Reader<'_>) -> Result<()> {
        self.require_idle()?;
        let func = RecordedFunc::from_u32(body.u32()?);
        let call = RecordedCall::from_u32(body.u32()?);
        ensure!(
            self.activations.iter().all(|a| a.call != call),
            "duplicate call ID in trace"
        );
        let store = self.store.as_context_mut().0;
        let objects = store.rr_objects();
        let info = objects
            .funcs
            .get(func)
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
        for (kind, slot) in info.params.iter().zip(values.iter_mut()) {
            *slot = body.value(*kind, store, objects)?;
        }
        body.end()?;
        self.activations.reserve(1)?;
        let activation = Activation::new(store, &self.trampolines, func, call, values)?;
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
        let objects = self.store.as_context().0.rr_objects();
        self.scratch.clear();
        // SAFETY: the activation is parked at the call, for which it
        // initialized the parameter slots with this store's values.
        unsafe {
            codec::values(
                &mut self.scratch,
                &objects.funcs[func].params,
                values.as_ptr(),
                |ptr| objects.encode_ref(ptr),
            )?
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
        let store = self.store.as_context().0;
        let objects = store.rr_objects();
        // SAFETY: the activation is parked at this call.
        let values = unsafe { host.values.as_mut() };
        let outcome = body.outcome(&objects.funcs[host.func].results, values, store, objects)?;
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
        let objects = self.store.as_context().0.rr_objects();
        self.scratch.clear();
        // SAFETY: if the call succeeded then the activation holds its results,
        // which are this store's values.
        unsafe {
            codec::outcome(
                &mut self.scratch,
                codec::LEAVE_WASM,
                activation.call,
                &result,
                &objects.funcs[activation.func].results,
                activation.results(),
                |ptr| objects.encode_ref(ptr),
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
        let activation = self.activations.swap_remove(index);
        activation.dispose(self.store.as_context_mut().0);
        Ok(())
    }

    /// Replays a write to a memory by the host.
    fn write(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let id = RecordedMemory::from_u32(body.u32()?);
        let offset = usize::try_from(body.u64()?)?;
        let bytes = body.rest();
        let store = self.store.as_context_mut().0;
        let memory = store.rr_replayed_memory(id)?;
        offset
            .checked_add(bytes.len())
            .and_then(|end| memory.data_mut_untracked(store).get_mut(offset..end))
            .ok_or_else(|| format_err!("out-of-bounds memory write in trace"))?
            .copy_from_slice(bytes);
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

    /// Runs the activation at `index` until its next yield; see
    /// [`Activation::resume`].
    fn resume(&mut self, index: usize, pending: Option<Error>) -> Result<()> {
        let mut growth_failures = self.read_growth_failures()?;
        growth_failures.reverse();
        let store = self.store.as_context_mut().0;
        store.rr_session().mode.replaying().growth_failures = growth_failures;
        self.publish_parked(Some(index))?;
        let store = self.store.as_context_mut().0;
        let activation = &mut self.activations[index];
        let yielded = activation.resume(store, pending)?;
        // Failures from code that ran on the activation and could not return
        // them.
        if let Some(e) = store.rr_session().failure.take() {
            return Err(e);
        }
        ensure!(
            store
                .rr_session()
                .mode
                .replaying()
                .growth_failures
                .is_empty(),
            "the guest diverged from the trace: its memory or table growth should have failed"
        );
        let activation = activation.call;
        self.observed = Some(match yielded {
            Yield::Host(func, values) => Observed::Host {
                activation,
                func,
                values,
            },
            Yield::Complete(result) => Observed::Complete { activation, result },
        });
        Ok(())
    }

    /// Publishes the suspended activations, other than the `running` one, whose
    /// frames the garbage collector must trace.
    fn publish_parked(&mut self, running: Option<usize>) -> Result<()> {
        let parked = &mut self
            .store
            .as_context_mut()
            .0
            .rr_session()
            .mode
            .replaying()
            .parked;
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
