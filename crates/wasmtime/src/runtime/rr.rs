//! Core WebAssembly record/replay for a whole store.
//!
//! Recording and replay begin with empty stores. Objects receive numeric
//! identities automatically; all construction is part of the trace.
//! Traces identify core modules only by the [`Checksum`]s of their bytecode,
//! so that they stay small: replay compiles each module from bytecode that
//! the embedder provides ([`Replayer::add_bytecode`]), and reconstructs core
//! instances, including those inside components. It executes guest code on
//! independent fibers and never invokes the original host functions or
//! component builtins.
//!
//! Core function boundaries support numbers, vectors, and nullable abstract
//! function references. Modules that use GC or exceptions are unsupported.
//! Shared memory, host table/global mutation, call hooks,
//! custom signal handlers, Wasm stack switching, epochs, and fuel are
//! unsupported, as is guest debugging. Host writes through the
//! memory APIs, including slices from [`Memory::data_mut`], are recorded;
//! writes through raw pointers such as [`Memory::data_ptr`] are not. Replay
//! requires a compiler and a native (non-Pulley) target, and is unsupported
//! under Miri, with AddressSanitizer, and with hardware-enforced shadow stacks.
//!
//! A [`Replayer`] replays step by step: it can stop at embedder events
//! ([`record_event`]), and it can take and restore [`Checkpoint`]s.
//!
//! Traces are private to this Wasmtime version. They contain host-supplied data
//! and can be large; applications should impose their own storage limits.

use crate::prelude::*;
use crate::runtime::vm::VMFuncRef;
use crate::store::{StoreInner, StoreOpaque};
use crate::{AsContextMut, Func, Memory, Store, ValRaw};
use alloc::sync::Arc;
use core::ops::Range;
use core::ptr::NonNull;

mod codec;
mod overlay;
pub(crate) mod replay;
use codec::{Kind, Reader};
pub use replay::{Checkpoint, ReplayStop, Replayer};

/// The SHA-256 checksum of a core module's bytecode, by which traces
/// identify the modules they instantiate.
pub type Checksum = [u8; 32];

/// Returns the core modules that `engine` compiles from `wasm`, with their
/// checksums. `wasm` is a core module or a component, in the binary or (with
/// the `wat` feature) text format. A core module is returned as is. A
/// component's modules are those nested in it, at any depth, and the adapter
/// modules that Wasmtime generates for it, which depend on the engine's
/// configuration.
///
/// A trace refers to the modules it instantiates by these checksums, so this
/// can be used to find or package the bytecode that replaying it requires.
pub fn core_modules(engine: &crate::Engine, wasm: &[u8]) -> Result<Vec<(Checksum, Vec<u8>)>> {
    #[cfg(feature = "wat")]
    let parsed = wat::parse_bytes(wasm).map_err(|e| format_err!("{e}"))?;
    #[cfg(feature = "wat")]
    let wasm = &*parsed;
    let checksum = |wasm: &[u8]| *wasmtime_environ::WasmChecksum::from_binary(wasm, true);
    if !wasmparser::Parser::is_component(wasm) {
        return Ok(vec![(checksum(wasm), wasm.to_vec())]);
    }
    #[cfg(all(
        feature = "component-model",
        any(feature = "cranelift", feature = "winch")
    ))]
    {
        use wasmtime_environ::ScopeVec;
        use wasmtime_environ::component::{ComponentTypesBuilder, Translator};
        // Translate the component as compilation would, which generates its
        // adapter modules, without compiling it.
        let scope = ScopeVec::new();
        let mut validator = wasmparser::Validator::new_with_features(engine.features());
        let mut types = ComponentTypesBuilder::new(&validator);
        let (_, translations) =
            Translator::new(engine.tunables(), &mut validator, &mut types, &scope)
                .translate(wasm)?;
        Ok(translations
            .values()
            .map(|t| (checksum(t.wasm), t.wasm.to_vec()))
            .collect())
    }
    #[cfg(not(all(
        feature = "component-model",
        any(feature = "cranelift", feature = "winch")
    )))]
    {
        let _ = engine;
        bail!("finding the core modules of a component requires component-model and a compiler")
    }
}

/// Formats a checksum as hexadecimal, as in file names and messages.
pub fn checksum_hex(checksum: &Checksum) -> String {
    checksum.iter().map(|b| format!("{b:02x}")).collect()
}

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
    /// and replay a trace. Tags with the high bit set
    /// ([`RESERVED_TAGS`]) are reserved for Wasmtime's own crates, such as
    /// `wasmtime-wasi`.
    const TAG: u32;
}

/// The event tags reserved for Wasmtime's own crates; see [`TraceEvent::TAG`].
pub const RESERVED_TAGS: core::ops::RangeInclusive<u32> = 0x8000_0000..=u32::MAX;

/// Records `event` at the current point of a recording. This does nothing if
/// the store is not recording.
pub fn record_event<E: TraceEvent>(mut store: impl AsContextMut, event: &E) -> Result<()> {
    let store = store.as_context_mut().0;
    if !store.rr.recording() {
        return Ok(());
    }
    let Mode::Recording { bytes, .. } = &mut store.rr_session().mode else {
        unreachable!()
    };
    let result = encode_event(bytes, event);
    store.rr_poison_on_err(result)
}

fn encode_event<E: TraceEvent>(bytes: &mut Vec<u8>, event: &E) -> Result<()> {
    let start = bytes.len();
    codec::record(bytes, codec::EVENT, 4)?;
    bytes.extend_from_slice(&E::TAG.to_le_bytes());
    // Serialize in place; the length is patched afterwards.
    *bytes = postcard::to_extend(event, core::mem::take(bytes))?;
    let len = u32::try_from(bytes.len() - start - 5)?;
    bytes[start + 1..start + 5].copy_from_slice(&len.to_le_bytes());
    Ok(())
}

/// Creates an event channel: a sink through which code without access to a
/// store, such as a host stream's implementation or a background task, records
/// events, and a receiver that delivers them into a store's recording once
/// attached with [`Store::rr_attach_events`].
///
/// The two ends work like those of a channel. The sink can be handed to event
/// producers (for example, built into a WASI context) before the store starts
/// recording; events recorded before the receiver is attached are buffered.
/// Once attached, events enter the trace at the store's next boundary: the next
/// guest entry, host return, or the end of the recording. When the receiver is
/// dropped, for example when its recording finishes, recording through the
/// sink does nothing.
pub fn event_channel() -> (EventSink, EventReceiver) {
    let buffer = Arc::new(crate::sync::RwLock::new(SinkBuffer::default()));
    (
        EventSink {
            buffer: buffer.clone(),
        },
        EventReceiver { buffer },
    )
}

/// The producing end of an [`event_channel`]. Sinks can be cloned and sent to
/// other threads.
#[derive(Clone)]
pub struct EventSink {
    buffer: Arc<crate::sync::RwLock<SinkBuffer>>,
}

/// The receiving end of an [`event_channel`], to attach to a recording store
/// with [`Store::rr_attach_events`].
pub struct EventReceiver {
    buffer: Arc<crate::sync::RwLock<SinkBuffer>>,
}

impl Drop for EventReceiver {
    fn drop(&mut self) {
        let mut buffer = self.buffer.write();
        buffer.closed = true;
        buffer.bytes = Vec::new();
    }
}

impl EventReceiver {
    /// Takes the events recorded since the last call.
    fn take(&self) -> Result<Vec<u8>> {
        let mut buffer = self.buffer.write();
        if let Some(e) = buffer.failure.take() {
            return Err(e);
        }
        Ok(core::mem::take(&mut buffer.bytes))
    }
}

#[derive(Default)]
struct SinkBuffer {
    // Encoded event records, in the order they were recorded.
    bytes: Vec<u8>,
    // Set when the receiver is dropped.
    closed: bool,
    // The first encoding failure, which fails the recording.
    failure: Option<Error>,
}

impl EventSink {
    /// Records `event`, as for [`record_event`].
    pub fn record<E: TraceEvent>(&self, event: &E) {
        let mut buffer = self.buffer.write();
        if buffer.closed || buffer.failure.is_some() {
            return;
        }
        let start = buffer.bytes.len();
        if let Err(e) = encode_event(&mut buffer.bytes, event) {
            buffer.bytes.truncate(start);
            buffer.failure = Some(e);
        }
    }
}

/// A complete execution trace, including object construction and startup.
///
/// Traces refer to the modules they instantiate only by [`Checksum`]; see
/// [`Trace::module_checksums`].
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

    /// Returns the checksums of the core modules that the trace instantiates,
    /// in the order in which they are first instantiated, without duplicates.
    /// Replaying the trace requires their bytecode.
    pub fn module_checksums(&self) -> Result<Vec<Checksum>> {
        let mut reader = Reader::new(&self.bytes);
        reader.take(codec::MAGIC.len())?;
        let mut checksums = Vec::new();
        loop {
            let (tag, mut body) = reader.record()?;
            match tag {
                codec::END => return Ok(checksums),
                codec::MODULE => {
                    body.u32()?;
                    let checksum = body.checksum()?;
                    if !checksums.contains(&checksum) {
                        checksums.push(checksum);
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(feature = "component-model")]
pub(crate) mod component;
mod init;
mod objects;
use objects::{Objects, RecordedFunc};

/// The call ID with which `rr_enter` reports a call that is not recorded.
pub(crate) const NOT_RECORDED: u32 = u32::MAX;

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
    // The receivers of the event channels attached to the session.
    receivers: Vec<EventReceiver>,
}

enum Mode {
    Recording {
        bytes: Vec<u8>,
        // `(call, function)` IDs of calls that have not yet returned.
        outstanding: Vec<(usize, usize)>,
        next_call: u32,
    },
    Replaying {
        // The bytecode the embedder provided for the trace's modules.
        bytecode: alloc::collections::BTreeMap<Checksum, Vec<u8>>,
        // Recorded guest growth failures that the running activation has yet
        // to reproduce, in order.
        growth_failures: Vec<[u8; codec::GROWTH_FAILED_LEN]>,
        // The checkpointed contents of each memory that has been
        // checkpointed, by `rr_key`.
        histories: alloc::collections::BTreeMap<overlay::TrackedKey, overlay::History>,
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

    /// Fails the active session with `error`, from code that cannot return
    /// it. Recording reports it when finishing; replay when the running
    /// activation next yields.
    pub(crate) fn fail(&mut self, error: Error) {
        if let Some(session) = &mut self.session {
            session.fail(error);
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
    /// The bytecode provided for the module with `checksum`, during replay.
    fn bytecode(&self, checksum: &Checksum) -> Result<&[u8]> {
        let Mode::Replaying { bytecode, .. } = &self.mode else {
            unreachable!()
        };
        bytecode.get(checksum).map(|b| &b[..]).ok_or_else(|| {
            format_err!(
                "the trace instantiates a module whose bytecode was not provided \
                 (SHA-256 {})",
                checksum_hex(checksum)
            )
        })
    }

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
        // A debug handler could run arbitrary host code that the trace does
        // not record.
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
            receivers: Vec::new(),
        })?);
        store.vm_store_context_mut().recording = 1;
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
        store.vm_store_context_mut().recording = 0;
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
    /// Initialization comes entirely from the trace, and modules from `wasm`,
    /// the bytecode of the modules or components the trace instantiates (see
    /// [`Replayer::add_bytecode`]). The engine must use
    /// [`crate::RRConfig::Replaying`].
    ///
    /// This verifies recorded guest outcomes, including traps. Therefore a
    /// correctly reproduced guest trap is a successful replay. Divergence or
    /// an invalid trace returns an error. Dropping this future frees all
    /// suspended activations before releasing the store. Failed or cancelled
    /// replay does not roll back initialization; retry with a fresh store.
    ///
    /// Afterwards the store's state can be inspected, for example through
    /// [`Replay::instances`], but its functions can no longer be called.
    pub async fn replay(&mut self, trace: &Trace, wasm: &[&[u8]]) -> Result<Replay>
    where
        T: Send,
    {
        let mut replayer = self.replayer(trace)?;
        for wasm in wasm {
            replayer.add_bytecode(wasm)?;
        }
        while replayer.run().await? != ReplayStop::Finished {}
        Ok(replayer.into_replay())
    }

    /// Attaches the receiving end of an [`event_channel`] to this store's
    /// recording, so that events recorded through its sinks enter the trace.
    /// Fails if the store is not recording.
    pub fn rr_attach_events(&mut self, receiver: EventReceiver) -> Result<()> {
        let store = self.as_context_mut().0;
        ensure!(store.rr.recording(), "store is not recording");
        let receivers = &mut store.rr_session().receivers;
        receivers.try_reserve(1)?;
        receivers.push(receiver);
        Ok(())
    }

    /// Returns a sink attached to this store's recording, as for an
    /// [`event_channel`] whose receiver is attached right away, or `None` if
    /// the store is not recording.
    pub fn rr_event_sink(&mut self) -> Option<EventSink> {
        let (sink, receiver) = event_channel();
        self.rr_attach_events(receiver).ok()?;
        Some(sink)
    }

    /// Starts replaying a trace, as for [`Store::replay`], with control over
    /// how it proceeds. The replayer must be given the bytecode of the
    /// trace's modules with [`Replayer::add_bytecode`].
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
        self.rr_with_history(overlay::Tracked::Memory(memory), |history, memory| {
            history.write(memory, range)
        })
        .map(|_| ())
    }

    /// Records that `len` slots of `table` starting at `index` are about to be
    /// written during replay, for checkpoints. Slots beyond the table's size
    /// are being added by growth.
    pub(crate) fn rr_table_dirty(
        &mut self,
        table: crate::Table,
        index: u64,
        len: u64,
    ) -> Result<()> {
        if len == 0 || !self.rr.active() || self.rr.recording() {
            return Ok(());
        }
        let size = table.rr_slot_size(self);
        let start = usize::try_from(index).unwrap_or(usize::MAX);
        let end = start.saturating_add(usize::try_from(len).unwrap_or(usize::MAX));
        let range = start.saturating_mul(size)..end.saturating_mul(size);
        self.rr_with_history(overlay::Tracked::Table(table), |history, table| {
            history.write(table, range)
        })
        .map(|_| ())
    }

    /// Runs `f` on the checkpoint history of `object`, if it has one. The
    /// history is detached from the session meanwhile.
    fn rr_with_history<R>(
        &mut self,
        object: overlay::Tracked,
        f: impl FnOnce(&mut overlay::History, &mut dyn overlay::TrackedMemory) -> Result<R>,
    ) -> Result<Option<R>> {
        let key = object.key(self);
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
            &mut overlay::StoreObject {
                store: self,
                object,
            },
        );
        let Mode::Replaying { histories, .. } = &mut self.rr_session().mode else {
            unreachable!()
        };
        histories.insert(key, history);
        result.map(Some)
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
        // Events recorded through sinks since the last boundary.
        let sunk = session
            .receivers
            .iter()
            .map(|r| r.take())
            .collect::<Vec<_>>();
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
            for events in sunk {
                let events = events?;
                codec::reserve(&mut bytes, events.len())?;
                bytes.extend_from_slice(&events);
            }
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
                self.rr.fail(e);
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

    /// Fails calls into a store whose functions only the replay driver may
    /// run.
    pub(crate) fn rr_check_call(&self) -> Result<()> {
        if !self.rr.active() {
            ensure!(
                !self.rr.replayed,
                "a replayed store's functions can only be inspected, not called"
            );
            return Ok(());
        }
        // The replay driver enters Wasm and handles host calls itself.
        ensure!(
            self.rr.recording(),
            "replay calls must be driven by Store::replay"
        );
        Ok(())
    }

    /// The `rr_enter` builtin, which the trampolines of record/replay engines
    /// call while the store records: at calls by `caller` to the host if
    /// `host`, and otherwise at host-to-Wasm calls. Returns the call's ID, or
    /// [`NOT_RECORDED`].
    ///
    /// The raw slots must contain initialized values of the callee's signature.
    pub(crate) unsafe fn rr_enter(
        &mut self,
        caller: crate::store::InstanceId,
        func: NonNull<VMFuncRef>,
        raw: *const ValRaw,
        host: bool,
    ) -> Result<u32> {
        // The host calls component functions with the store's dummy instance
        // as their caller. Such calls are not guest boundaries.
        if !self.rr.recording() || (host && !self.rr_guest_caller(caller)) {
            return Ok(NOT_RECORDED);
        }
        // SAFETY: inherited from this function's contract.
        let result = unsafe { self.rr_record_enter(func, raw, host) };
        self.rr_poison_on_err(result)
    }

    unsafe fn rr_record_enter(
        &mut self,
        func: NonNull<VMFuncRef>,
        raw: *const ValRaw,
        host: bool,
    ) -> Result<u32> {
        self.rr_flush_boundary(!host)?;
        // Resource destructors are wrapped as host functions only to record
        // their calls as host-to-Wasm calls.
        if host && self.rr.passthrough.contains(&func_key(func)) {
            return Ok(NOT_RECORDED);
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
        let call = *next_call;
        *next_call = next_call
            .checked_add(1)
            .filter(|c| *c != NOT_RECORDED)
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
        bytes.extend_from_slice(&call.to_le_bytes());
        // SAFETY: inherited from this function's contract.
        unsafe {
            codec::values(bytes, &bound.params, raw, |ptr| {
                session.objects.encode_ref(ptr)
            })?
        };
        outstanding.push((call as usize, id));
        Ok(call)
    }

    /// The `rr_leave` builtin, which records the return of `call` from
    /// [`StoreOpaque::rr_enter`]. Result slots are initialized if `succeeded`;
    /// otherwise the call fails with the error that is being raised.
    pub(crate) unsafe fn rr_leave(
        &mut self,
        call: u32,
        raw: *const ValRaw,
        host: bool,
        succeeded: bool,
    ) -> Result<()> {
        if call == NOT_RECORDED {
            return Ok(());
        }
        let result = if succeeded {
            Ok(())
        } else {
            match crate::runtime::vm::pending_unwind_error() {
                Some(error) => Err(error),
                // A host panic is not recorded; it propagates through the
                // recording.
                None => return Ok(()),
            }
        };
        // SAFETY: inherited from this function's contract.
        let recorded = unsafe { self.rr_record_leave(call as usize, raw, &result, host) };
        let recorded = self.rr_poison_on_err(recorded);
        // A failing call keeps its own error; the poisoned recording reports
        // this one.
        if succeeded { recorded } else { Ok(()) }
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
