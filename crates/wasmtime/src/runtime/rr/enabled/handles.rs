//! References to GC objects in traces.
//!
//! A trace refers to GC objects by *handle*. A handle is created when a
//! reference to an object that has none passes from the guest to the host, or
//! when the host creates an `externref`, which are both recorded, so replay
//! creates the same handles at the same points. The host can then pass the
//! object to the guest.
//!
//! Handles are not identities, so nothing needs to be known of objects that do
//! not cross the boundary, or after the host is done with them. A handle keeps
//! its object alive, in recordings and replays alike, until a recording notices
//! that the host has no roots for it, and records that it is released. If the
//! object later crosses the boundary again then it has a new handle.

use super::*;
use crate::runtime::vm::VMGcRef;
use crate::{HeapTopType, Ref, ValType};
use alloc::collections::BTreeMap;
#[cfg(feature = "gc")]
use wasmtime_environ::VMGcKind;

/// The number of handles to create before first looking for those to release.
const FIRST_PRUNE: usize = 1024;

#[derive(Default)]
pub(super) struct Handles {
    refs: BTreeMap<u32, VMGcRef>,
    next: u32,
    /// The handles of objects by their raw references, unless the garbage
    /// collector may have moved them.
    by_raw: Option<BTreeMap<u32, u32>>,
    /// The number of handles at which to look for those to release.
    prune_at: usize,
}

/// A null reference, an `i31ref`, or a handle, in a trace.
const NULL: u64 = 0;
const I31: u64 = 1 << 32;
const HANDLE: u64 = 2 << 32;

impl Handles {
    fn index(&mut self) -> &mut BTreeMap<u32, u32> {
        let refs = &self.refs;
        self.by_raw
            .get_or_insert_with(|| refs.iter().map(|(h, r)| (r.as_raw_u32(), *h)).collect())
    }

    /// Encodes a raw reference, whose object must have a handle.
    pub fn encode(&self, raw: u32) -> Result<u64> {
        let Some(gc_ref) = VMGcRef::from_raw_u32(raw) else {
            return Ok(NULL);
        };
        if let Some(i31) = gc_ref.as_i31() {
            return Ok(I31 | u64::from(i31.get_u32()));
        }
        let handle = match &self.by_raw {
            Some(by_raw) => by_raw.get(&raw).copied(),
            None => self
                .refs
                .iter()
                .find_map(|(h, r)| (r.as_raw_u32() == raw).then_some(*h)),
        };
        let handle = handle.ok_or_else(|| {
            format_err!(
                "record/replay does not support passing the guest GC objects that the host \
                 allocated, or that it read from the guest's state"
            )
        })?;
        Ok(HANDLE | u64::from(handle))
    }

    /// Decodes a reference, without exposing it to Wasm.
    pub fn decode(&self, value: u64) -> Result<Option<VMGcRef>> {
        let low = u32::try_from(value & u64::from(u32::MAX))?;
        match value - u64::from(low) {
            NULL if low == 0 => Ok(None),
            I31 => {
                let i31 = crate::vm::I31::new_u32(low);
                let i31 = i31.ok_or_else(|| format_err!("invalid i31ref in trace"))?;
                Ok(Some(VMGcRef::from_i31(i31)))
            }
            HANDLE => {
                let gc_ref = self.refs.get(&low);
                let gc_ref = gc_ref.ok_or_else(|| format_err!("invalid GC handle in trace"))?;
                Ok(Some(gc_ref.unchecked_copy()))
            }
            _ => bail!("invalid GC reference in trace"),
        }
    }

    /// The raw references, without barriers, for checkpoints.
    pub fn save(&self) -> HandlesImage {
        HandlesImage {
            refs: self
                .refs
                .iter()
                .map(|(h, r)| (*h, r.as_raw_u32()))
                .collect(),
            next: self.next,
        }
    }

    pub fn restore(&mut self, image: &HandlesImage) {
        let refs = image.refs.iter();
        self.refs = refs
            .filter_map(|(h, raw)| Some((*h, VMGcRef::from_raw_u32(*raw)?)))
            .collect();
        self.next = image.next;
        self.by_raw = None;
    }
}

/// The handles at a checkpoint.
pub(super) struct HandlesImage {
    refs: Vec<(u32, u32)>,
    next: u32,
}

impl StoreOpaque {
    /// Runs `f` with the session's handles detached from the store.
    fn rr_with_handles<R>(&mut self, f: impl FnOnce(&mut Self, &mut Handles) -> R) -> R {
        let mut handles = core::mem::take(&mut self.rr_session().handles);
        let result = f(self, &mut handles);
        self.rr_session().handles = handles;
        result
    }

    /// Creates a handle for the object of a raw reference, if it needs one.
    pub(super) fn rr_create_handle(&mut self, raw: u32) -> Result<()> {
        let Some(gc_ref) = VMGcRef::from_raw_u32(raw).filter(|r| !r.is_i31()) else {
            return Ok(());
        };
        self.rr_with_handles(|store, handles| {
            if handles.index().contains_key(&raw) {
                return Ok(());
            }
            let handle = handles.next;
            handles.next = handle
                .checked_add(1)
                .ok_or_else(|| format_err!("too many GC handles"))?;
            let gc_ref = store.require_gc_store_mut()?.clone_gc_ref(&gc_ref);
            handles.refs.insert(handle, gc_ref);
            handles.index().insert(raw, handle);
            Ok(())
        })
    }

    /// Releases a handle during replay.
    pub(super) fn rr_release_handle(&mut self, handle: u32) -> Result<()> {
        self.rr_with_handles(|store, handles| {
            let gc_ref = handles.refs.remove(&handle);
            let gc_ref = gc_ref.ok_or_else(|| format_err!("invalid GC handle in trace"))?;
            handles.by_raw = None;
            store.require_gc_store_mut()?.drop_gc_ref(gc_ref);
            Ok(())
        })
    }

    /// Releases all handles, as a recording ends.
    pub(super) fn rr_release_all_handles(&mut self) {
        self.rr_with_handles(|store, handles| {
            for (_, gc_ref) in core::mem::take(&mut handles.refs) {
                if let Some(gc_store) = store.optional_gc_store_mut() {
                    gc_store.drop_gc_ref(gc_ref);
                }
            }
        });
    }

    /// Exposes decoded references to Wasm. They must be among `slots`, whose
    /// types are `kinds`.
    pub(super) fn rr_expose_refs(
        &mut self,
        kinds: &[Kind],
        slots: &mut [core::mem::MaybeUninit<ValRaw>],
    ) -> Result<()> {
        for (kind, slot) in kinds.iter().zip(slots) {
            if let Kind::GcRef { .. } = kind {
                // SAFETY: the slot was just initialized, and all GC references
                // have the same representation.
                let raw = unsafe { slot.assume_init_ref().get_anyref() };
                if let Some(gc_ref) = VMGcRef::from_raw_u32(raw).filter(|r| !r.is_i31()) {
                    let gc_store = self.require_gc_store_mut()?;
                    let gc_ref = gc_store.clone_gc_ref(&gc_ref);
                    let _ = gc_store.expose_gc_ref_to_wasm(gc_ref)?;
                }
            }
        }
        Ok(())
    }

    /// Records the release of handles whose objects the host has no roots for,
    /// once in a while.
    ///
    /// This must be called as the guest passes control to the host, before
    /// references pass to it, as unrooted references are then invalid.
    pub(super) fn rr_prune_handles(&mut self) -> Result<()> {
        let handles = &mut self.rr_session().handles;
        if handles.refs.len() < handles.prune_at.max(FIRST_PRUNE) {
            return Ok(());
        }
        let roots = self.rr_raw_host_roots();
        let handles = &self.rr_session().handles.refs;
        let unused = handles
            .iter()
            .filter(|(_, r)| !roots.contains(&r.as_raw_u32()))
            .map(|(h, _)| *h)
            .collect::<Vec<_>>();
        for handle in unused {
            self.rr_release_handle(handle)?;
            let session = self.rr_session();
            session.append(codec::RELEASE, &handle.to_le_bytes())?;
        }
        let handles = &mut self.rr_session().handles;
        handles.prune_at = 2 * handles.refs.len();
        Ok(())
    }

    /// Records the creation of an `externref` by the host.
    #[cfg(feature = "gc")]
    pub(crate) fn rr_created_externref(&mut self, gc_ref: &VMGcRef) -> Result<()> {
        if !self.rr.recording() {
            return Ok(());
        }
        let raw = gc_ref.as_raw_u32();
        let result = (|| {
            self.rr_flush()?;
            self.rr_create_handle(raw)?;
            self.rr_session().append(codec::EXTERNREF, &[])
        })();
        self.rr_poison_on_err(result)
    }

    /// Checks that a decoded reference, if it is one, is of type `ty`.
    pub(super) fn rr_check_ref_type(&self, ty: &ValType, raw: &ValRaw) -> Result<()> {
        let ValType::Ref(ty) = ty else {
            return Ok(());
        };
        let value = match ty.heap_type().top() {
            HeapTopType::Func => {
                let func = NonNull::new(raw.get_funcref().cast());
                // SAFETY: the reference was decoded for this store.
                Ref::Func(func.map(|f| unsafe { Func::from_vm_func_ref(self.id(), f) }))
            }
            #[cfg(feature = "gc")]
            top => {
                let gc_ref = VMGcRef::from_raw_u32(raw.get_anyref());
                let gc_ref = gc_ref.as_ref();
                // The checks below assume that the object is in the type's
                // hierarchy. Structs, arrays, and `externref`s can be
                // converted between `anyref` and `externref`.
                if let Some(object) = gc_ref.filter(|r| !r.is_i31()) {
                    let exn = self.require_gc_store()?.kind(object)? == VMGcKind::ExnRef;
                    ensure!(
                        exn == matches!(top, HeapTopType::Exn),
                        "value of the wrong type in trace"
                    );
                } else {
                    ensure!(
                        gc_ref.is_none() || !matches!(top, HeapTopType::Exn),
                        "value of the wrong type in trace"
                    );
                }
                match top {
                    HeapTopType::Any => gc_ref
                        .map(|r| crate::Unrooted::<crate::AnyRef>::new(self, r))
                        .into(),
                    // There are no concrete `externref` types.
                    HeapTopType::Extern => gc_ref
                        .map(|r| crate::Unrooted::<crate::ExternRef>::new(self, r))
                        .into(),
                    HeapTopType::Exn => gc_ref
                        .map(|r| crate::Unrooted::<crate::ExnRef>::new(self, r))
                        .into(),
                    HeapTopType::Func | HeapTopType::Cont => bail!(codec::UNSUPPORTED),
                }
            }
            #[cfg(not(feature = "gc"))]
            _ => bail!(codec::UNSUPPORTED),
        };
        value
            .ensure_matches_ty(self, ty)
            .context("value of the wrong type in trace")
    }

    /// Throws the exception `value`, as a host function does during replay.
    #[cfg(feature = "gc")]
    pub(super) fn rr_throw(&mut self, value: u64) -> Result<Error> {
        let Some(gc_ref) = self.rr_handles().decode(value)? else {
            return Ok(crate::ThrownException.into());
        };
        let raw = ValRaw::exnref(gc_ref.as_raw_u32());
        self.rr_check_ref_type(&ValType::EXNREF, &raw)?;
        Ok(self.set_pending_exception(&gc_ref))
    }

    #[cfg(not(feature = "gc"))]
    pub(super) fn rr_throw(&mut self, _value: u64) -> Result<Error> {
        bail!("invalid outcome in trace")
    }

    /// Traces the objects of handles as garbage collection roots.
    #[cfg(feature = "gc")]
    pub(crate) fn rr_trace_handles(&mut self, roots: &mut crate::vm::GcRootsList) {
        let Some(session) = &mut self.rr.session else {
            return;
        };
        // The collector may move the objects.
        session.handles.by_raw = None;
        for gc_ref in session.handles.refs.values_mut() {
            // SAFETY: the reference stays in place while it is a root.
            unsafe { roots.add_vmgcref_root(gc_ref.into(), "record/replay handle") }
        }
    }
}
