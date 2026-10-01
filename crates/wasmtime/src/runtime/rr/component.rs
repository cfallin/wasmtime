//! Component core-callable trampolines use the same boundary as host imports.
//! The wrapper encloses generated checks and resource operations, including
//! callbacks from generated resource destructors back into core Wasm.

use super::*;
use crate::{AsContext, StoreContextMut};

pub(crate) fn wrap<T: 'static>(
    store: &mut StoreContextMut<'_, T>,
    original: NonNull<VMFuncRef>,
) -> Result<VMFuncRef> {
    // Keep the original reference separate: the component's slot is replaced
    // with the wrapper after this function returns.
    // SAFETY: the caller passes a function reference from a live component.
    let original = unsafe { original.as_ref().clone() };
    let raw = store.0.func_refs_and_modules().0.rr_copy(original)?;
    // SAFETY: the original reference is now rooted in this store. Its code
    // and context remain rooted by the component/instance that created it.
    let original = unsafe { Func::from_vm_func_ref(store.0.id(), raw) };
    let ty = original.load_ty(store.0);
    // SAFETY: this rooted function's context remains live in the store.
    let callback = unsafe { raw.as_ref().vmctx.as_non_null().as_ref().magic }
        == wasmtime_environ::VMCONTEXT_MAGIC;
    // SAFETY: forward the exact signature and storage to the original callee.
    let wrapper = unsafe {
        Func::new_unchecked(&mut *store, ty, move |mut caller, values| {
            let raw = NonNull::slice_from_raw_parts(
                NonNull::new(values.as_mut_ptr().cast()).unwrap(),
                values.len(),
            );
            let func_ref = original.vm_func_ref(caller.as_context().0);
            Func::call_unchecked_raw(&mut caller.as_context_mut(), func_ref, raw)
        })
    };
    let (refs, modules) = store.0.func_refs_and_modules();
    refs.fill(modules);
    let raw = wrapper.vm_func_ref(store.0);
    if callback {
        store.0.rr.passthrough.insert(func_key(raw), ())?;
    }
    // SAFETY: wrapper belongs to this store; all referenced data stays rooted.
    let result = unsafe { raw.as_ref().clone() };
    // Builtins lifted directly to component exports may have no Wasm caller
    // and hence no wasm_call trampoline. All modules that can import this
    // reference were registered before wrapping, so imported ones are filled.
    Ok(result)
}
