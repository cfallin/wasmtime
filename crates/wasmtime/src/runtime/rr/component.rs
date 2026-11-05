//! Resource destructors are core Wasm functions that generated `resource.drop`
//! trampolines call directly, within the recorded host call of the
//! trampoline. Recording wraps each destructor in a host function that enters
//! it through `Func::call_unchecked_raw`, which records the call as a
//! host-to-Wasm call; the wrapper's own call is not recorded.

use super::*;
use crate::{AsContext, StoreContextMut};

pub(crate) fn wrap_destructor<T: 'static>(
    store: &mut StoreContextMut<'_, T>,
    original: NonNull<VMFuncRef>,
) -> Result<VMFuncRef> {
    // Keep the original reference separate: the component's slot is replaced
    // with the wrapper after this function returns.
    // SAFETY: the caller passes a function reference from a live component.
    let original = unsafe { original.as_ref().clone() };
    let raw = store.0.func_refs_and_modules().0.rr_copy(original)?;
    // SAFETY: the original reference is now rooted in this store. Its code
    // and context remain rooted by the instance that created it.
    let original = unsafe { Func::from_vm_func_ref(store.0.id(), raw) };
    let ty = original.load_ty(store.0);
    // SAFETY: forward the exact signature and storage to the original callee.
    let wrapper = unsafe {
        Func::new_unchecked(&mut *store, ty, move |mut caller, values| {
            let raw = NonNull::slice_from_raw_parts(
                NonNull::new(values.as_mut_ptr().cast()).unwrap(),
                values.len(),
            );
            let func_ref = original.vm_func_ref(caller.as_context().0);
            Func::call_unchecked_raw(
                &mut caller.as_context_mut(),
                func_ref,
                raw,
                crate::runtime::vm::UncaughtException::Trap,
            )
        })
    };
    // The component's `resource.drop` trampolines call the wrapper through
    // its wasm_call trampoline.
    let (refs, modules) = store.0.func_refs_and_modules();
    refs.fill(modules);
    let raw = wrapper.vm_func_ref(store.0);
    store.0.rr.passthrough.insert(func_key(raw))?;
    // SAFETY: wrapper belongs to this store; all referenced data stays rooted.
    Ok(unsafe { raw.as_ref().clone() })
}
