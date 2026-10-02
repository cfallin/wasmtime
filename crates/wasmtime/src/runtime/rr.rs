//! Core WebAssembly record/replay for a whole store.
//!
//! Recording and replay begin with empty stores. Objects receive numeric
//! identities automatically; all construction is part of the trace.
//! Replay uses only the modules in the trace and reconstructs core instances,
//! including those inside components. It executes guest code on independent
//! fibers and never invokes the original host functions or component builtins.
//!
//! Core function boundaries support numbers, vectors, and nullable abstract
//! function references. GC and typed function references are unsupported.
//! Shared memory, host table/global mutation, resource limiters, call hooks,
//! custom signal handlers, Wasm stack switching, epochs, and fuel are
//! unsupported, as is recording with guest debugging. Host writes through the
//! memory APIs, including slices from [`Memory::data_mut`], are recorded;
//! writes through raw pointers such as [`Memory::data_ptr`] are not. Replay
//! requires a compiler and a native (non-Pulley) target, and is unsupported on
//! Windows, under Miri, and with AddressSanitizer.
//!
//! A [`Replayer`] replays step by step: it can stop at embedder events
//! ([`record_event`]) and, with guest debugging enabled on the replaying
//! engine, at breakpoints and single steps, and it can take and restore
//! [`Checkpoint`]s, which together support reversible debugging.
//!
//! Traces are private to this Wasmtime version. They contain host-supplied data
//! and can be large; applications should impose their own storage limits.

use crate::prelude::*;
use crate::runtime::vm::VMFuncRef;
use crate::store::{StoreInner, StoreOpaque};
use crate::{AsContextMut, Func, Memory, Store, ValRaw};
use core::ops::Range;
use core::ptr::NonNull;

mod codec;
mod overlay;
pub(crate) mod replay;
use codec::{Kind, Reader};
pub use replay::{Checkpoint, ReplayStop, Replayer};

/// Core instances constructed while replaying initialization.
///
/// Components replay as their constituent core instances. Their exports can
/// be inspected here without rebuilding component host state.
#[derive(Debug)]
pub struct Replay {
    instances: Vec<crate::Instance>,
}

impl Replay {
    /// Returns instances in their recorded construction order.
    pub fn instances(&self) -> &[crate::Instance] {
        &self.instances
    }
}

/// An embedder-defined event that can be recorded in a trace with
/// [`record_event`], and observed during replay with [`Replayer::on_event`].
///
/// Events carry information about the recorded execution, such as its
/// output, that the embedder wants to see again when replaying. They cannot
/// affect the replay itself.
pub trait TraceEvent: serde::Serialize + serde::de::DeserializeOwned + 'static {
    /// Identifies this type of event in traces. It must be unique among an
    /// embedding's event types and stable between the processes that record
    /// and replay a trace.
    const TAG: u32;
}

/// Records `event` at the current point of a recording. This does nothing if
/// the store is not recording.
pub fn record_event<E: TraceEvent>(mut store: impl AsContextMut, event: &E) -> Result<()> {
    let store = store.as_context_mut().0;
    if !store.rr.recording() {
        return Ok(());
    }
    let result = (|| {
        let Mode::Recording { bytes, .. } = &mut store.rr_session().mode else {
            unreachable!()
        };
        let start = bytes.len();
        codec::record(bytes, codec::EVENT, 4)?;
        bytes.extend_from_slice(&E::TAG.to_le_bytes());
        // Serialize in place; the length is patched afterwards.
        *bytes = postcard::to_extend(event, core::mem::take(bytes))?;
        let len = u32::try_from(bytes.len() - start - 5)?;
        bytes[start + 1..start + 5].copy_from_slice(&len.to_le_bytes());
        Ok(())
    })();
    store.rr_poison_on_err(result)
}

/// A complete execution trace, including object construction and startup.
pub struct Trace {
    bytes: Vec<u8>,
}

impl Trace {
    /// Returns the serialized trace.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Takes ownership of a serialized trace, checking its framing and version.
    /// Record contents, object construction, and execution order are checked
    /// during replay.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let mut reader = Reader::new(&bytes);
        ensure!(
            reader.take(codec::MAGIC.len())? == codec::MAGIC,
            "unsupported record/replay trace version"
        );
        // Record contents are checked during replay.
        loop {
            let (tag, body) = reader.record()?;
            if tag == codec::END {
                body.end()?;
                reader.end()?;
                break;
            }
        }
        Ok(Self { bytes })
    }
}

#[cfg(feature = "component-model")]
pub(crate) mod component;
mod init;
mod objects;
use objects::{Objects, RecordedFunc};

fn func_key(func: NonNull<VMFuncRef>) -> (usize, usize) {
    // SAFETY: all callers pass function references rooted in the same store.
    unsafe {
        let func = func.as_ref();
        // Host references may have store-local copies with a filled-in
        // wasm_call trampoline while the original host context does not.
        (func.vmctx.addr().get(), func.array_call.addr().get())
    }
}

#[derive(Default)]
pub(crate) struct State {
    session: Option<Box<Session>>,
    // Set once a replay ends: replayed host functions only run under the
    // replay driver, so the store's functions may no longer be called.
    replayed: bool,
    passthrough: TryHashSet<(usize, usize)>,
}

struct Session {
    objects: Objects,
    mode: Mode,
    pending: Vec<(usize, Range<usize>)>,
    failure: Option<Error>,
}

enum Mode {
    Recording {
        bytes: Vec<u8>,
        // `(call, function)` IDs of calls that have not yet returned.
        outstanding: Vec<(usize, usize)>,
        next_call: u32,
    },
    Replaying {
        // Recorded guest growth failures that the running activation has yet
        // to reproduce, in order.
        growth_failures: Vec<[u8; codec::GROWTH_FAILED_LEN]>,
        // The watchpoint at which the running activation requested a stop.
        #[cfg(feature = "debug")]
        watchpoint: Option<crate::WatchpointHit>,
        // The checkpointed contents of each memory that has been
        // checkpointed, by `rr_key`.
        histories: alloc::collections::BTreeMap<usize, overlay::History>,
        // The granularity at which checkpoints track memory writes.
        page_size: usize,
    },
}

/// An object that guest code can grow.
pub(crate) enum Growable {
    Memory(Memory),
    Table(crate::Table),
}

impl State {
    pub(crate) fn active(&self) -> bool {
        self.session.is_some()
    }

    pub(crate) fn recording(&self) -> bool {
        matches!(
            self.session.as_deref().map(|s| &s.mode),
            Some(Mode::Recording { .. })
        )
    }

    /// Records that an active session cannot represent `operation`, which
    /// still takes place. The recording can then only be discarded.
    pub(crate) fn poison(&mut self, operation: &'static str) {
        if let Some(session) = &mut self.session {
            session.fail(format_err!("record/replay does not support {operation}"));
        }
    }

    /// Like `poison`, but also fails the operation.
    pub(crate) fn reject(&mut self, operation: &'static str) -> Result<()> {
        if self.active() {
            self.poison(operation);
            bail!("record/replay does not support {operation}");
        }
        Ok(())
    }
}

impl Session {
    /// Keeps the first failure of a session; later ones are its consequences.
    fn fail(&mut self, error: Error) {
        self.failure.get_or_insert(error);
    }

    /// Appends a complete record when recording.
    fn append(&mut self, tag: u8, body: &[u8]) -> Result<()> {
        let Mode::Recording { bytes, .. } = &mut self.mode else {
            return Ok(());
        };
        codec::record(bytes, tag, body.len())?;
        bytes.extend_from_slice(body);
        Ok(())
    }
}

impl<T: 'static> Store<T> {
    /// Starts a recording in an empty store, including object construction and startup.
    /// The engine must use [`crate::RRConfig::Recording`]. See [`crate::rr`]
    /// for the restrictions on the recorded execution. Store data `T` may be
    /// initialized, but no functions, memories, globals, tables, GC objects, or
    /// core/component instances may have been created in this store.
    pub fn start_recording(&mut self) -> Result<()> {
        let store = self.as_context_mut().0;
        store.rr_validate()?;
        ensure!(
            store.engine().is_recording(),
            "recording requires RRConfig::Recording"
        );
        // Guest debugging is supported on replay instead: a debug handler
        // could run arbitrary host code that the trace does not record.
        ensure!(
            !store.engine().tunables().debug_guest,
            "record/replay does not support recording with guest debugging"
        );
        let objects = Objects::default();
        let mut bytes = Vec::new();
        codec::reserve(&mut bytes, codec::MAGIC.len())?;
        bytes.extend_from_slice(codec::MAGIC);
        store.rr.session = Some(try_new::<Box<_>>(Session {
            objects,
            mode: Mode::Recording {
                bytes,
                outstanding: Vec::new(),
                next_call: 0,
            },
            pending: Vec::new(),
            failure: None,
        })?);
        Ok(())
    }

    /// Finishes a recording, flushing pending memory writes. Returns an error
    /// for a poisoned recording; such a trace cannot be replayed. A failed
    /// recording is discarded and recording is disabled on the store.
    ///
    /// A recording may end while guest calls are unfinished, for example
    /// with component-model tasks suspended in host calls or after a host
    /// panic; replay then ends with them suspended.
    pub fn finish_recording(&mut self) -> Result<Trace> {
        let store = self.as_context_mut().0;
        ensure!(store.rr.recording(), "store is not recording");
        let flushed = store.rr_flush();
        let session = store.rr.session.take().unwrap();
        if let Some(e) = session.failure {
            return Err(e);
        }
        flushed?;
        let Mode::Recording { mut bytes, .. } = session.mode else {
            unreachable!()
        };
        codec::record(&mut bytes, codec::END, 0)?;
        Ok(Trace { bytes })
    }

    /// Replays a trace on independent fibers, without calling the original
    /// host functions. The store must be empty, as for [`Store::start_recording`].
    /// Modules and initialization come entirely from the trace. The engine must
    /// use [`crate::RRConfig::Replaying`].
    ///
    /// This verifies recorded guest outcomes, including traps. Therefore a
    /// correctly reproduced guest trap is a successful replay. Divergence or
    /// an invalid trace returns an error. Dropping this future frees all
    /// suspended activations before releasing the store. Failed or cancelled
    /// replay does not roll back initialization; retry with a fresh store.
    ///
    /// Afterwards the store's state can be inspected, for example through
    /// [`Replay::instances`], but its functions can no longer be called.
    pub async fn replay(&mut self, trace: &Trace) -> Result<Replay>
    where
        T: Send,
    {
        let mut replayer = self.replayer(trace)?;
        while replayer.run().await? != ReplayStop::Finished {}
        Ok(replayer.into_replay())
    }

    /// Starts replaying a trace, as for [`Store::replay`], with control over
    /// how it proceeds.
    pub fn replayer<'a>(&'a mut self, trace: &'a Trace) -> Result<Replayer<'a, T>>
    where
        T: Send,
    {
        Replayer::new(self.as_context_mut().0, trace)
    }
}

impl StoreOpaque {
    fn rr_session(&mut self) -> &mut Session {
        self.rr.session.as_deref_mut().unwrap()
    }

    fn rr_growth_record(
        &mut self,
        object: &Growable,
        delta: u64,
    ) -> Result<[u8; codec::GROWTH_FAILED_LEN]> {
        let (kind, id, size) = match object {
            Growable::Memory(memory) => {
                let key = memory.rr_key(self);
                let id = self.rr_session().objects.memories_by_key.get(&key).copied();
                (0, id, memory.internal_data_size(self))
            }
            Growable::Table(table) => {
                let id = self
                    .rr_session()
                    .objects
                    .tables_by_key
                    .get(&table.rr_key())
                    .copied();
                (1, id, usize::try_from(table.size_(self))?)
            }
        };
        let id = id.ok_or_else(|| format_err!("guest grew an unregistered object"))?;
        let mut record = [0; codec::GROWTH_FAILED_LEN];
        record[0] = kind;
        record[1..5].copy_from_slice(&u32::try_from(id)?.to_le_bytes());
        record[5..13].copy_from_slice(&u64::try_from(size)?.to_le_bytes());
        record[13..].copy_from_slice(&delta.to_le_bytes());
        Ok(record)
    }

    /// Records that `range` of `memory` is about to be written during replay,
    /// for checkpoints.
    pub(crate) fn rr_dirty(&mut self, memory: Memory, range: Range<usize>) -> Result<()> {
        if range.is_empty() {
            return Ok(());
        }
        self.rr_with_history(memory, |history, memory| history.write(memory, range))
            .map(|_| ())
    }

    /// Runs `f` on the checkpoint history of `memory`, if it has one. The
    /// history is detached from the session meanwhile.
    fn rr_with_history<R>(
        &mut self,
        memory: Memory,
        f: impl FnOnce(&mut overlay::History, &mut overlay::StoreMemory<'_>) -> Result<R>,
    ) -> Result<Option<R>> {
        let key = memory.rr_key(self);
        let Some(Mode::Replaying { histories, .. }) =
            self.rr.session.as_deref_mut().map(|s| &mut s.mode)
        else {
            return Ok(None);
        };
        let Some(mut history) = histories.remove(&key) else {
            return Ok(None);
        };
        let result = f(
            &mut history,
            &mut overlay::StoreMemory {
                store: self,
                memory,
            },
        );
        let Mode::Replaying { histories, .. } = &mut self.rr_session().mode else {
            unreachable!()
        };
        histories.insert(key, history);
        result.map(Some)
    }

    /// Requests that the running replay activation, if any, stop at a
    /// watchpoint, reporting `hit`.
    #[cfg(feature = "debug")]
    pub(crate) fn rr_debug_stop_at_watchpoint(&mut self, hit: crate::WatchpointHit) -> bool {
        if !self.rr_debug_stop() {
            return false;
        }
        let Mode::Replaying { watchpoint, .. } = &mut self.rr_session().mode else {
            unreachable!()
        };
        *watchpoint = Some(hit);
        true
    }

    /// Requests that the running replay activation, if any, stop for a debug
    /// event. Its breakpoint trampoline yields to the driver once the libcall
    /// requesting this returns.
    pub(crate) fn rr_debug_stop(&mut self) -> bool {
        match self.vm_store_context().replay_control {
            Some(control) => {
                // SAFETY: the running activation's control block is live, and
                // the driver holds no reference to it while it runs.
                unsafe {
                    (*control.as_ptr()).reason = wasmtime_environ::VM_REPLAY_DEBUG;
                }
                true
            }
            None => false,
        }
    }

    /// Whether replay must fail this guest growth because it failed when it
    /// was recorded. Called before attempting the growth.
    pub(crate) fn rr_replay_growth_fails(&mut self, object: &Growable, delta: u64) -> Result<bool> {
        if !self.rr.active() || self.rr.recording() {
            return Ok(false);
        }
        let record = self.rr_growth_record(object, delta)?;
        let Mode::Replaying {
            growth_failures, ..
        } = &mut self.rr_session().mode
        else {
            unreachable!()
        };
        if growth_failures.first() == Some(&record) {
            growth_failures.remove(0);
            return Ok(true);
        }
        Ok(false)
    }

    /// Records a failed guest growth, or reports a divergence if it failed
    /// during replay without having failed when recorded.
    pub(crate) fn rr_growth_failed(&mut self, object: &Growable, delta: u64) -> Result<()> {
        if !self.rr.active() {
            return Ok(());
        }
        ensure!(
            self.rr.recording(),
            "replay diverged: guest growth failed that succeeded when recorded"
        );
        let result = self
            .rr_growth_record(object, delta)
            .and_then(|record| self.rr_session().append(codec::GROWTH_FAILED, &record));
        self.rr_poison_on_err(result)
    }

    /// Poisons an active session if recording failed, so that the trace can
    /// never be finalized with a partial record or a missing effect.
    fn rr_poison_on_err<R>(&mut self, result: Result<R>) -> Result<R> {
        if let (Err(e), Some(session)) = (&result, &mut self.rr.session) {
            session.fail(format_err!("failed to record execution: {e}"));
        }
        result
    }

    /// The ID of a memory written by the host, poisoning the recording if
    /// the memory was never registered.
    fn rr_memory_id(&mut self, memory: Memory) -> Option<usize> {
        let key = memory.rr_key(self);
        let session = self.rr_session();
        let id = session.objects.memories_by_key.get(&key).copied();
        if id.is_none() {
            session.fail(format_err!("host modified an unregistered memory"));
        }
        id
    }

    #[cfg(feature = "component-model")]
    /// Tracks a host write to `written` (clamped to the memory's size) of a
    /// component's memory.
    pub(crate) fn rr_track_memory_definition(
        &mut self,
        definition: NonNull<crate::vm::VMMemoryDefinition>,
        written: Range<usize>,
    ) {
        if !self.rr.active() {
            return;
        }
        let objects = &self.rr_session().objects;
        let memory = objects
            .memories_by_key
            .get(&(definition.as_ptr() as usize))
            .map(|id| objects.memories[*id]);
        match memory {
            Some(memory) => {
                let len = memory.internal_data_size(self);
                self.rr_track_memory(memory, written.start.min(len)..written.end.min(len));
            }
            None => self
                .rr_session()
                .fail(format_err!("unregistered component memory")),
        }
    }

    #[cfg(feature = "component-model")]
    pub(crate) fn rr_track_raw_memory(&mut self, ptr: *mut u8, len: usize) {
        if !self.rr.active() || len == 0 {
            return;
        }
        let address = ptr as usize;
        let memory = self
            .all_memories()
            .filter_map(|m| m.unshared())
            .find_map(|m| {
                let data = m.rr_data(self);
                let offset = address.checked_sub(data.as_ptr() as usize)?;
                (offset.checked_add(len)? <= data.len()).then_some((m, offset))
            });
        match memory {
            Some((memory, offset)) => self.rr_track_memory(memory, offset..offset + len),
            None => self
                .rr_session()
                .fail(format_err!("component write outside registered memories")),
        }
    }

    pub(crate) fn rr_memory_grown(&mut self, memory: Memory, old_size: usize) -> Result<()> {
        if !self.rr.recording() {
            return Ok(());
        }
        let Some(id) = self.rr_memory_id(memory) else {
            bail!("host grew an unregistered memory");
        };
        let new_size = memory.internal_data_size(self);
        let result = (|| {
            let mut body = [0; 20];
            body[..4].copy_from_slice(&u32::try_from(id)?.to_le_bytes());
            body[4..12].copy_from_slice(&u64::try_from(old_size)?.to_le_bytes());
            body[12..].copy_from_slice(&u64::try_from(new_size)?.to_le_bytes());
            self.rr_session().append(codec::RESIZE, &body)
        })();
        self.rr_poison_on_err(result)
    }

    fn rr_check(&self) -> Result<()> {
        if let Some(session) = &self.rr.session {
            ensure!(
                session.failure.is_none(),
                "recording has failed; discard it with finish_recording"
            );
        }
        Ok(())
    }

    /// Commit host writes while the store is exclusively borrowed and before
    /// any guest execution. Pending ranges also cover forgotten guards.
    pub(crate) fn rr_flush(&mut self) -> Result<()> {
        self.rr_flush_boundary(true)
    }

    fn rr_flush_boundary(&mut self, flags: bool) -> Result<()> {
        self.rr_check()?;
        if !self.rr.recording() {
            return Ok(());
        }
        let result = self.rr_flush_writes(flags);
        self.rr_poison_on_err(result)
    }

    fn rr_flush_writes(&mut self, flags: bool) -> Result<()> {
        if flags {
            self.rr_flush_flags()?;
        }
        // Detach the pending ranges and the trace while reading memories; this
        // never runs guest or embedder code.
        let session = self.rr_session();
        let mut pending = core::mem::take(&mut session.pending);
        let Mode::Recording { bytes, .. } = &mut session.mode else {
            unreachable!()
        };
        let mut bytes = core::mem::take(bytes);
        // Emit each written byte once, in a deterministic order.
        pending.sort_unstable_by_key(|(id, range)| (*id, range.start));
        pending.dedup_by(|(id, next), (prev_id, prev)| {
            let overlaps = id == prev_id && next.start <= prev.end;
            if overlaps {
                prev.end = prev.end.max(next.end);
            }
            overlaps
        });
        let result = (|| {
            for (id, range) in &pending {
                let memory = self.rr.session.as_ref().unwrap().objects.memories[*id];
                let data = memory
                    .rr_data(self)
                    .get(range.clone())
                    .ok_or_else(|| format_err!("tracked memory range is no longer valid"))?;
                let len = 12usize
                    .checked_add(data.len())
                    .ok_or_else(|| format_err!("memory record too large"))?;
                codec::record(&mut bytes, codec::WRITE, len)?;
                bytes.extend_from_slice(&u32::try_from(*id)?.to_le_bytes());
                bytes.extend_from_slice(&u64::try_from(range.start)?.to_le_bytes());
                bytes.extend_from_slice(data);
            }
            Ok(())
        })();
        pending.clear();
        let session = self.rr_session();
        session.pending = pending;
        let Mode::Recording { bytes: trace, .. } = &mut session.mode else {
            unreachable!()
        };
        *trace = bytes;
        result
    }

    pub(crate) fn rr_track_memory(&mut self, memory: Memory, range: Range<usize>) {
        if !self.rr.active() || range.is_empty() {
            return;
        }
        if !self.rr.recording() {
            // Replay reproduces the write, but checkpoints must see it.
            if let Err(e) = self.rr_dirty(memory, range) {
                self.rr_session().fail(e);
            }
            return;
        }
        let Some(id) = self.rr_memory_id(memory) else {
            return;
        };
        let session = self.rr_session();
        // Writes are usually sequential, so coalesce with the previous range
        // here, in constant time; the flush merges any other overlaps.
        if let Some((other, old)) = session.pending.last_mut()
            && *other == id
            && range.start <= old.end
            && old.start <= range.end
        {
            old.start = old.start.min(range.start);
            old.end = old.end.max(range.end);
            return;
        }
        if session.pending.try_reserve(1).is_err() {
            session.fail(OutOfMemory::new(core::mem::size_of::<(usize, Range<usize>)>()).into());
        } else {
            session.pending.push((id, range));
        }
    }

    // SAFETY: the caller supplies initialized slots of this function's signature.
    unsafe fn rr_boundary_refs(
        &mut self,
        func: usize,
        raw: *const ValRaw,
        results: bool,
    ) -> Result<()> {
        let bound = &self.rr_session().objects.funcs[func];
        let count = if results {
            bound.results.len()
        } else {
            bound.params.len()
        };
        for i in 0..count {
            let bound = &self.rr_session().objects.funcs[func];
            let kind = if results {
                bound.results[i]
            } else {
                bound.params[i]
            };
            if kind == Kind::FuncRef {
                // SAFETY: the slot is initialized and has funcref type.
                if let Some(ptr) = NonNull::new(unsafe { (*raw.add(i)).get_funcref() }) {
                    self.rr_register_func_reference(ptr.cast())?;
                }
            }
        }
        Ok(())
    }

    /// The host-call boundary of `HostFunc` trampolines. The arguments must
    /// be initialized values of the callee's signature.
    pub(crate) unsafe fn rr_enter_host(
        &mut self,
        caller: crate::store::InstanceId,
        callee: NonNull<crate::vm::VMOpaqueContext>,
        args: *const ValRaw,
    ) -> Result<Option<usize>> {
        // Calls from the host or a dummy instance are not guest boundaries.
        if !self.rr.active() || !self.rr_guest_caller(caller) {
            return Ok(None);
        }
        // SAFETY: `HostFunc` trampolines are called with their own context.
        let func = unsafe {
            NonNull::from(
                &crate::vm::VMArrayCallHostFuncContext::from_opaque(callee)
                    .as_ref()
                    .func_ref,
            )
        };
        // SAFETY: inherited from this function's contract.
        unsafe { self.rr_enter(func, args, true) }
    }

    /// The raw slots must contain initialized values of the callee's signature.
    pub(crate) unsafe fn rr_enter(
        &mut self,
        func: NonNull<VMFuncRef>,
        raw: *const ValRaw,
        host: bool,
    ) -> Result<Option<usize>> {
        if !self.rr.active() {
            ensure!(
                !self.rr.replayed,
                "a replayed store's functions can only be inspected, not called"
            );
            return Ok(None);
        }
        // The replay driver enters Wasm and handles host calls itself.
        ensure!(
            self.rr.recording(),
            "replay calls must be driven by Store::replay"
        );
        // SAFETY: inherited from this function's contract.
        let result = unsafe { self.rr_record_enter(func, raw, host) };
        self.rr_poison_on_err(result)
    }

    unsafe fn rr_record_enter(
        &mut self,
        func: NonNull<VMFuncRef>,
        raw: *const ValRaw,
        host: bool,
    ) -> Result<Option<usize>> {
        self.rr_flush_boundary(!host)?;
        // Calling a host Func from host code is not a Wasm boundary. Its
        // effects (writes and any guest callbacks) are recorded normally.
        // SAFETY: the caller supplies a live, store-rooted function reference.
        let magic = unsafe { func.as_ref().vmctx.as_non_null().as_ref().magic };
        let is_host = magic == wasmtime_environ::VM_ARRAY_CALL_HOST_FUNC_MAGIC;
        #[cfg(feature = "component-model")]
        let is_host = is_host || magic == wasmtime_environ::component::VMCOMPONENT_MAGIC;
        if !host && is_host {
            return Ok(None);
        }
        if host && self.rr.passthrough.contains(&func_key(func)) {
            return Ok(None);
        }
        let id = self.rr_session().objects.find_func(func)?;
        // SAFETY: inherited from this function's typed-slot contract.
        unsafe {
            self.rr_boundary_refs(id, raw, false)?;
        }
        let session = self.rr_session();
        let bound = &session.objects.funcs[id];
        let Mode::Recording {
            bytes,
            outstanding,
            next_call,
        } = &mut session.mode
        else {
            unreachable!()
        };
        let call = *next_call as usize;
        *next_call = next_call
            .checked_add(1)
            .ok_or_else(|| format_err!("too many recorded calls"))?;
        outstanding
            .try_reserve(1)
            .map_err(|_| OutOfMemory::new(core::mem::size_of::<(usize, usize)>()))?;
        let tag = if host {
            codec::ENTER_HOST
        } else {
            codec::ENTER_WASM
        };
        codec::record(bytes, tag, 8 + codec::values_len(&bound.params))?;
        bytes.extend_from_slice(&u32::try_from(id)?.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(call)?.to_le_bytes());
        // SAFETY: inherited from this function's contract.
        unsafe {
            codec::values(bytes, &bound.params, raw, |ptr| {
                session.objects.encode_ref(ptr)
            })?
        };
        outstanding.push((call, id));
        Ok(Some(call))
    }

    /// Result slots are initialized exactly when result is successful.
    pub(crate) unsafe fn rr_leave(
        &mut self,
        call: Option<usize>,
        raw: *const ValRaw,
        result: &Result<()>,
        host: bool,
    ) -> Result<()> {
        let Some(call) = call else {
            return Ok(());
        };
        // SAFETY: inherited from this function's contract.
        let recorded = unsafe { self.rr_record_leave(call, raw, result, host) };
        self.rr_poison_on_err(recorded)
    }

    unsafe fn rr_record_leave(
        &mut self,
        call: usize,
        raw: *const ValRaw,
        result: &Result<()>,
        host: bool,
    ) -> Result<()> {
        // The recording may have ended while this call was unfinished.
        if !self.rr.recording() {
            return Ok(());
        }
        self.rr_flush_boundary(host)?;
        let Mode::Recording { outstanding, .. } = &self.rr_session().mode else {
            unreachable!()
        };
        let position = outstanding
            .iter()
            .position(|(c, _)| *c == call)
            .ok_or_else(|| format_err!("return without recorded call"))?;
        let func = outstanding[position].1;
        if result.is_ok() {
            // SAFETY: successful calls initialize their result slots.
            unsafe {
                self.rr_boundary_refs(func, raw, true)?;
            }
        }
        let session = self.rr_session();
        let Mode::Recording {
            bytes, outstanding, ..
        } = &mut session.mode
        else {
            unreachable!()
        };
        let tag = if host {
            codec::LEAVE_HOST
        } else {
            codec::LEAVE_WASM
        };
        let results = &session.objects.funcs[func].results;
        // SAFETY: inherited from this function's contract.
        unsafe {
            codec::outcome(bytes, tag, call, result, results, raw, |ptr| {
                session.objects.encode_ref(ptr)
            })?;
        }
        outstanding.swap_remove(position);
        Ok(())
    }
}
