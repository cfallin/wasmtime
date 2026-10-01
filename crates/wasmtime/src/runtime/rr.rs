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
//! epochs, and fuel are unsupported. Raw-pointer writes are not tracked: use
//! safe memory APIs or [`Memory::data_mut_tracked`]. Replaying construction
//! requires a compiler. Guest allocation failures, guest debug hooks, and
//! snapshot/restore remain future work.
//!
//! Traces are private to this Wasmtime version. They contain host-supplied data
//! and can be large; applications should impose their own storage limits.

use crate::prelude::*;
use crate::runtime::vm::VMFuncRef;
use crate::store::{StoreInner, StoreOpaque};
use crate::{AsContextMut, Func, Memory, Store, ValRaw};
use core::ops::{Deref, DerefMut, Range};
use core::ptr::NonNull;

mod codec;
pub(crate) mod replay;
use codec::{Kind, Reader};

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
    /// Object construction and execution order are checked during replay.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let mut reader = Reader::new(&bytes);
        ensure!(
            reader.take(codec::MAGIC.len())? == codec::MAGIC,
            "unsupported record/replay trace version"
        );
        loop {
            let (tag, body) = reader.record()?;
            match tag {
                codec::END => {
                    body.end()?;
                    reader.end()?;
                    break;
                }
                codec::ENTER_WASM
                | codec::LEAVE_WASM
                | codec::ENTER_HOST
                | codec::LEAVE_HOST
                | codec::WRITE
                | codec::RESIZE
                | codec::HOST
                | codec::MODULE
                | codec::INSTANCE
                | codec::GLOBAL
                | codec::GLOBAL_WRITE
                | codec::MEMORY
                | codec::TABLE => {}
                _ => bail!("unknown record/replay event {tag}"),
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
    unavailable: Option<&'static str>,
    passthrough: TryHashMap<(usize, usize), ()>,
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
        outstanding: Vec<(bool, usize, usize)>,
        next_call: u32,
    },
    Replaying {
        trampolines: replay::Trampolines,
    },
}

impl State {
    pub(crate) fn validate_available(&self) -> Result<()> {
        if let Some(reason) = self.unavailable {
            bail!("record/replay does not support {reason}");
        }
        Ok(())
    }
    pub(crate) fn active(&self) -> bool {
        self.session.is_some()
    }

    pub(crate) fn recording(&self) -> bool {
        matches!(
            self.session.as_deref().map(|s| &s.mode),
            Some(Mode::Recording { .. })
        )
    }

    pub(crate) fn reject(&mut self, operation: &'static str) -> Result<()> {
        if let Some(session) = &mut self.session {
            session.failure = Some(format_err!("record/replay does not support {operation}"));
            bail!("record/replay does not support {operation}");
        }
        Ok(())
    }

    pub(crate) fn mark_unavailable(&mut self, reason: &'static str) -> Result<()> {
        self.unavailable = Some(reason);
        self.reject(reason)
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
    /// for an incomplete or poisoned recording; such a trace cannot be replayed.
    /// A failed recording is discarded and recording is disabled on the store.
    pub fn finish_recording(&mut self) -> Result<Trace> {
        let store = self.as_context_mut().0;
        ensure!(store.rr.recording(), "store is not recording");
        let flushed = store.rr_flush();
        let session = store.rr.session.take().unwrap();
        if let Some(e) = session.failure {
            return Err(e);
        }
        flushed?;
        let Mode::Recording {
            mut bytes,
            outstanding,
            ..
        } = session.mode
        else {
            unreachable!()
        };
        ensure!(
            outstanding.is_empty(),
            "recording ended with unfinished calls (possibly a host panic)"
        );
        codec::record(&mut bytes, codec::END, 0)?;
        Ok(Trace { bytes })
    }

    /// Replays a trace on independent async fibers, without calling the original
    /// host functions. The store must be empty, as for [`Store::start_recording`].
    /// Modules and initialization come entirely from the trace. The engine must
    /// use [`crate::RRConfig::Replaying`].
    ///
    /// This verifies recorded guest outcomes, including traps. Therefore a
    /// correctly reproduced guest trap is a successful replay. Divergence or
    /// an invalid trace returns an error. Dropping this future disposes all
    /// suspended activations before releasing the store. Failed or cancelled
    /// replay does not roll back initialization; retry with a fresh store.
    pub async fn replay(&mut self, trace: &Trace) -> Result<Replay>
    where
        T: Send,
    {
        replay::run(self.as_context_mut().0, trace).await
    }
}

impl StoreOpaque {
    #[cfg(feature = "component-model")]
    pub(crate) fn rr_track_memory_definition(
        &mut self,
        definition: NonNull<crate::vm::VMMemoryDefinition>,
    ) {
        if !self.rr.recording() {
            return;
        }
        let objects = &self.rr.session.as_ref().unwrap().objects;
        let memory = objects
            .memories_by_key
            .get(&(definition.as_ptr() as usize))
            .map(|id| objects.memories[*id]);
        if let Some(memory) = memory {
            let len = memory.internal_data_size(self);
            self.rr_track_memory(memory, 0..len);
        } else {
            self.rr.session.as_mut().unwrap().failure =
                Some(format_err!("unregistered component memory"));
        }
    }

    #[cfg(feature = "component-model")]
    pub(crate) fn rr_track_raw_memory(&mut self, ptr: *mut u8, len: usize) {
        if !self.rr.recording() || len == 0 {
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
        if let Some((memory, offset)) = memory {
            self.rr_track_memory(memory, offset..offset + len);
        } else {
            self.rr.session.as_mut().unwrap().failure =
                Some(format_err!("component write outside registered memories"));
        }
    }

    pub(crate) fn rr_memory_grown(&mut self, memory: Memory, old_size: usize) -> Result<()> {
        if !self.rr.recording() {
            return Ok(());
        }
        let key = memory.rr_key(self);
        let id = self
            .rr
            .session
            .as_ref()
            .unwrap()
            .objects
            .memories_by_key
            .get(&key)
            .copied();
        let new_size = memory.internal_data_size(self);
        let session = self.rr.session.as_mut().unwrap();
        let Some(id) = id else {
            session.failure = Some(format_err!("host grew an unregistered memory"));
            bail!("host grew an unregistered memory");
        };
        let Mode::Recording { bytes, .. } = &mut session.mode else {
            unreachable!()
        };
        if let Err(e) = codec::record(bytes, codec::RESIZE, 20) {
            session.failure = Some(format_err!("failed to record memory growth"));
            return Err(e);
        }
        bytes.extend_from_slice(&u32::try_from(id)?.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(old_size)?.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(new_size)?.to_le_bytes());
        Ok(())
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
        let Some(session) = &mut self.rr.session else {
            return Ok(());
        };
        if !matches!(session.mode, Mode::Recording { .. }) {
            return Ok(());
        }
        let pending = core::mem::take(&mut session.pending);
        let result = (|| {
            if flags {
                self.rr_flush_flags()?;
            }
            for (id, range) in &pending {
                let memory = self.rr.session.as_ref().unwrap().objects.memories[*id];
                // Borrow the memory separately from the recorder. The store is
                // exclusive, memory cannot grow, and appending only allocates
                // trace storage; it never calls guest or embedder code.
                let data = memory.rr_data(self);
                let data = data
                    .get(range.clone())
                    .ok_or_else(|| format_err!("tracked memory range is no longer valid"))?;
                let ptr = data.as_ptr();
                let len = data.len();
                let Mode::Recording { bytes, .. } = &mut self.rr.session.as_mut().unwrap().mode
                else {
                    unreachable!()
                };
                codec::record(
                    bytes,
                    codec::WRITE,
                    12usize
                        .checked_add(len)
                        .ok_or_else(|| format_err!("memory record too large"))?,
                )?;
                bytes.extend_from_slice(&u32::try_from(*id)?.to_le_bytes());
                bytes.extend_from_slice(&u64::try_from(range.start)?.to_le_bytes());
                // SAFETY: see the disjoint borrow explanation above.
                bytes.extend_from_slice(unsafe { core::slice::from_raw_parts(ptr, len) });
            }
            Ok(())
        })();
        let session = self.rr.session.as_mut().unwrap();
        let mut pending = pending;
        pending.clear();
        session.pending = pending;
        if result.is_err() {
            session.failure = Some(format_err!("failed to record memory writes"));
        }
        result
    }

    pub(crate) fn rr_track_memory(&mut self, memory: Memory, range: Range<usize>) {
        if !self.rr.recording() || range.is_empty() {
            return;
        }
        let key = memory.rr_key(self);
        let id = self
            .rr
            .session
            .as_ref()
            .unwrap()
            .objects
            .memories_by_key
            .get(&key)
            .copied();
        let session = self.rr.session.as_mut().unwrap();
        let Some(id) = id else {
            session.failure = Some(format_err!("host wrote an unregistered memory"));
            return;
        };
        // Union overlapping ranges only within one uninterrupted host segment.
        for (other, old) in &mut session.pending {
            if *other == id && range.start <= old.end && old.start <= range.end {
                old.start = old.start.min(range.start);
                old.end = old.end.max(range.end);
                return;
            }
        }
        if session.pending.try_reserve(1).is_err() {
            session.failure =
                Some(OutOfMemory::new(core::mem::size_of::<(usize, Range<usize>)>()).into());
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
        let bound = &self.rr.session.as_ref().unwrap().objects.funcs[func];
        let count = if results {
            bound.results.len()
        } else {
            bound.params.len()
        };
        for i in 0..count {
            let bound = &self.rr.session.as_ref().unwrap().objects.funcs[func];
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

    /// The raw slots must contain initialized values of the callee's signature.
    pub(crate) unsafe fn rr_enter(
        &mut self,
        func: NonNull<VMFuncRef>,
        raw: *const ValRaw,
        host: bool,
    ) -> Result<Option<usize>> {
        if !self.rr.active() {
            return Ok(None);
        }
        // The replay driver enters Wasm and handles host calls itself.
        ensure!(
            self.rr.recording(),
            "replay calls must be driven by Store::replay"
        );
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
        if host && self.rr.passthrough.get(&func_key(func)).is_some() {
            return Ok(None);
        }
        let id = match self.rr.session.as_ref().unwrap().objects.find_func(func) {
            Ok(id) => id,
            Err(e) => {
                self.rr.session.as_mut().unwrap().failure = Some(format_err!(
                    "unregistered function crossed the recording boundary"
                ));
                return Err(e);
            }
        };
        // SAFETY: inherited from this function's typed-slot contract.
        unsafe {
            self.rr_boundary_refs(id, raw, false)?;
        }
        let session = self.rr.session.as_mut().unwrap();
        let bound = &session.objects.funcs[id];
        ensure!(
            bound.host == host,
            "direct host calls are not guest activations"
        );
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
            .map_err(|_| OutOfMemory::new(core::mem::size_of::<(bool, usize, usize)>()))?;
        codec::record(
            bytes,
            if host {
                codec::ENTER_HOST
            } else {
                codec::ENTER_WASM
            },
            8 + codec::values_len(&bound.params),
        )?;
        bytes.extend_from_slice(&u32::try_from(id)?.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(call)?.to_le_bytes());
        // SAFETY: inherited from this function's contract.
        unsafe {
            codec::values(bytes, &bound.params, raw, |ptr| {
                session.objects.encode_ref(ptr)
            })?
        };
        outstanding.push((host, call, id));
        Ok(Some(call))
    }

    /// Result slots are initialized exactly when result is successful.
    pub(crate) unsafe fn rr_leave(
        &mut self,
        id: Option<usize>,
        raw: *const ValRaw,
        result: &Result<()>,
        host: bool,
    ) -> Result<()> {
        let Some(id) = id else {
            return Ok(());
        };
        self.rr_flush_boundary(host)?;
        if result.is_ok() {
            let Mode::Recording { outstanding, .. } = &self.rr.session.as_ref().unwrap().mode
            else {
                unreachable!()
            };
            let func = outstanding
                .iter()
                .find(|(h, call, _)| *h == host && *call == id)
                .ok_or_else(|| format_err!("return without recorded call"))?
                .2;
            // SAFETY: successful calls initialize their result slots.
            unsafe {
                self.rr_boundary_refs(func, raw, true)?;
            }
        }
        let session = self.rr.session.as_mut().unwrap();
        let Mode::Recording {
            bytes, outstanding, ..
        } = &mut session.mode
        else {
            unreachable!()
        };
        let position = outstanding
            .iter()
            .position(|(h, call, _)| *h == host && *call == id)
            .ok_or_else(|| format_err!("return without recorded call"))?;
        let func = outstanding[position].2;
        // SAFETY: inherited from this function's contract.
        unsafe {
            codec::outcome(
                bytes,
                if host {
                    codec::LEAVE_HOST
                } else {
                    codec::LEAVE_WASM
                },
                id,
                result,
                &session.objects.funcs[func].results,
                raw,
                |ptr| session.objects.encode_ref(ptr),
            )?;
        }
        outstanding.swap_remove(position);
        Ok(())
    }
}

/// A tracked mutable view of a range of guest memory.
///
/// Dropping this guard commits pending writes. Forgetting it is also supported:
/// the store commits pending writes before guest execution or finalization.
pub struct MemoryMut<'a> {
    store: &'a mut StoreOpaque,
    memory: Memory,
    range: Range<usize>,
}

impl Deref for MemoryMut<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.memory.rr_data(self.store)[self.range.clone()]
    }
}

impl DerefMut for MemoryMut<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.store.rr_track_memory(self.memory, self.range.clone());
        &mut self.memory.rr_data_mut(self.store)[self.range.clone()]
    }
}

impl Drop for MemoryMut<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.store.rr_flush() {
            if let Some(session) = &mut self.store.rr.session {
                session.failure = Some(e);
            }
        }
    }
}

impl Memory {
    /// Borrows a byte range, recording its final contents when the returned
    /// guard is dropped. This also works when recording is disabled.
    pub fn data_mut_tracked<'a, T: 'static>(
        &self,
        store: impl Into<crate::StoreContextMut<'a, T>>,
        range: Range<usize>,
    ) -> Result<MemoryMut<'a>> {
        let store = store.into().0;
        ensure!(
            self.rr_data(store).get(range.clone()).is_some(),
            "memory range out of bounds"
        );
        Ok(MemoryMut {
            store,
            memory: *self,
            range,
        })
    }
}
