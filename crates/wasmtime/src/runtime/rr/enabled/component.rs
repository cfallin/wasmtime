//! Component resource destructors, and the memory that the canonical ABI writes.
//!
//! Resource destructors are core Wasm functions that `resource.drop`
//! trampolines, in generated code, call directly within their own recorded host
//! call. A recording store replaces each with a host function that calls it
//! like the host does, which records a host-to-Wasm call. The call of the host
//! function itself is not recorded, but if it fails then the trampoline's is
//! recorded as failing.

use super::*;
use crate::runtime::vm::{UncaughtException, VMMemoryDefinition};
use crate::{AsContext, StoreContextMut};

/// Wraps the resource destructor `dtor` if the store is recording.
pub(crate) fn wrap_destructor<T: 'static>(
    store: &mut StoreContextMut<'_, T>,
    dtor: NonNull<VMFuncRef>,
) -> Result<NonNull<VMFuncRef>> {
    if !store.0.rr.recording() {
        return Ok(dtor);
    }
    // Copy the reference, as `dtor` may point to the slot in the component that
    // is about to refer to the wrapper.
    // SAFETY: the caller passes a function reference from a live component,
    // whose code and context are rooted by the instance that created it.
    let dtor = unsafe { dtor.as_ref().clone() };
    let dtor = store.0.func_refs_and_modules().0.rr_copy(dtor)?;
    // SAFETY: the copy is owned by this store.
    let dtor = unsafe { Func::from_vm_func_ref(store.0.id(), dtor) };
    let ty = dtor.load_ty(store.0);
    // SAFETY: this forwards its arguments and result storage to a function of
    // the same type.
    let wrapper = unsafe {
        Func::new_unchecked(&mut *store, ty, move |mut caller, values| {
            let dtor = dtor.vm_func_ref(caller.as_context().0);
            Func::call_unchecked_raw(
                &mut caller.as_context_mut(),
                dtor,
                NonNull::new_unchecked(NonNull::from(values).as_ptr() as *mut [ValRaw]),
                UncaughtException::Trap,
            )
        })
    };
    // Trampolines call the wrapper through its `wasm_call`.
    let (refs, modules) = store.0.func_refs_and_modules();
    refs.fill(modules);
    let raw = wrapper.vm_func_ref(store.0);
    // SAFETY: the store owns the wrapper.
    let key = unsafe { raw.as_ref().hash_key() };
    let recording = store.0.rr_session().mode.recording();
    recording.destructors.insert(key)?;
    Ok(raw)
}

impl StoreOpaque {
    /// Tracks a host write to `written` (clamped to the memory's size) of a
    /// component's memory.
    pub(crate) fn rr_track_memory_definition(
        &mut self,
        definition: NonNull<VMMemoryDefinition>,
        written: Range<usize>,
    ) {
        if !self.rr.recording() {
            return;
        }
        let memory = self.rr_objects().find_memory_definition(definition);
        match memory {
            Some(memory) => {
                let len = memory.internal_data_size(self);
                self.rr_track_memory(memory, written.start.min(len)..written.end.min(len));
            }
            None => self
                .rr_session()
                .fail(format_err!("host modified an unregistered memory")),
        }
    }

    /// Tracks a host write to the `len` bytes at `ptr`, which are in a
    /// component's memory.
    pub(crate) fn rr_track_raw_memory(&mut self, ptr: *mut u8, len: usize) {
        if !self.rr.recording() || len == 0 {
            return;
        }
        let memory = self.rr_objects().memories.values().find_map(|memory| {
            let data = memory.data_in_store(self);
            let offset = ptr.addr().checked_sub(data.as_ptr().addr())?;
            (offset.checked_add(len)? <= data.len()).then_some((*memory, offset))
        });
        match memory {
            Some((memory, offset)) => self.rr_track_memory(memory, offset..offset + len),
            None => self
                .rr_session()
                .fail(format_err!("host modified an unregistered memory")),
        }
    }
}
