//! Guest activations: calls from the host into Wasm that replay runs.
//!
//! Each activation runs on a raw fiber whose only frames, whenever it is
//! suspended, are generated code and the fiber library's switch and start
//! routines:
//!
//! ```text
//! fiber start (asm) -> ReplayStart -> array-to-Wasm -> guest ...
//!     -> Wasm-to-array -> ReplayHostCall -> fiber switch (asm)
//! ```
//!
//! Its generated code communicates with the replayer only through its
//! `VMReplayControl`. A guest return or trap is reported by a final yield from
//! `ReplayStart`, after which the fiber is freed without resuming it.
//!
//! Runtime state that an ordinary call keeps on the host stack, such as the
//! `CallThreadState` and the store's entry/exit registers, is installed around
//! each resumption instead.

use super::super::*;
use crate::runtime::func::EntryStoreContext;
use crate::runtime::vm::mpk::{self, ProtectionMask};
use crate::runtime::vm::{
    SendSyncPtr, VMArrayCallHostFuncContext, VMArrayCallNative, VMCommonStackInformation,
    VMOpaqueContext, VMReplayControl, VmPtr,
};
use core::mem::MaybeUninit;
use wasmtime_environ::{FuncKey, RrYield};
use wasmtime_fiber::{RawFiber, RawFiberEntry};

/// The generated code that activations run.
pub(super) struct Trampolines {
    /// An otherwise empty module that contains the trampolines.
    module: crate::Module,
    start: SendSyncPtr<u8>,
    host_call: SendSyncPtr<u8>,
}

impl Trampolines {
    pub fn new(engine: &Engine) -> Result<Self> {
        ensure!(
            RawFiber::is_supported(),
            "replay is not supported on this platform"
        );
        let module = super::init::compile_module(engine, b"\0asm\x01\0\0\0")?;
        let get = |key| {
            module
                .compiled_module()
                .replay_trampoline(key)
                .map(SendSyncPtr::new)
                .ok_or_else(|| format_err!("engine is not configured for replay"))
        };
        Ok(Trampolines {
            start: get(FuncKey::ReplayStart)?,
            host_call: get(FuncKey::ReplayHostCall)?,
            module,
        })
    }

    /// Creates the function that stands in for a recorded host function.
    pub fn host_stub(&self, store: &mut StoreOpaque, ty: crate::FuncType) -> Result<Func> {
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

/// Why an activation yielded.
pub(super) enum Yield {
    /// It called a host function.
    Host(RecordedFunc, HostValues),
    /// It finished.
    Complete(Result<()>),
}

/// The argument and result storage of a host call, on the stack of an
/// activation that is parked at it.
#[derive(Clone, Copy)]
pub(super) struct HostValues(SendSyncPtr<[MaybeUninit<ValRaw>]>);

impl HostValues {
    pub fn as_ptr(&self) -> *const ValRaw {
        self.0.as_ptr().cast()
    }

    /// # Safety
    ///
    /// The activation must be parked at this host call, and nothing else may
    /// refer to the storage while the result is in use.
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<ValRaw>] {
        // SAFETY: inherited from this function's contract.
        unsafe { self.0.as_mut() }
    }
}

/// A guest activation and everything its suspended fiber refers to, which has a
/// stable address until [`Activation::dispose`].
pub(super) struct Activation {
    fiber: RawFiber,
    /// An owned allocation (from `Box`). The fiber's generated code writes to
    /// it while running, so it is only accessed through raw pointers, and no
    /// reference to it is live across a resumption.
    control: SendSyncPtr<VMReplayControl>,
    /// The activation's `VMStoreContext` state while it is not running, and
    /// the store's while it is.
    context: core::mem::ManuallyDrop<EntryStoreContext>,
    /// The stack-switching information `context.stack_chain` points to.
    _stack: Box<VMCommonStackInformation>,
    mpk: Option<ProtectionMask>,
    /// The entry function.
    pub func: RecordedFunc,
    /// The recorded call of the entry function.
    pub call: RecordedCall,
    /// The recorded host call that this is parked at, if any.
    pub host: Option<HostCall>,
    /// The arguments, and then results, of the entry function. The fiber
    /// refers to this, so it is never resized.
    values: TryVec<ValRaw>,
}

/// A host call that has been matched with the trace.
#[derive(Clone, Copy)]
pub(super) struct HostCall {
    pub call: RecordedCall,
    pub func: RecordedFunc,
    pub values: HostValues,
}

// SAFETY: the raw pointers in an activation only refer to its own boxed
// storage, its fiber stack, and code and contexts rooted in the store that
// the replayer owns. No thread-local state refers to the fiber while it is
// suspended, and only the replayer resumes it.
unsafe impl Send for Activation {}

impl Activation {
    /// Creates an activation that will call `func` with the arguments at the
    /// start of `values`, which must have room for its arguments and for its
    /// results.
    pub fn new(
        store: &mut StoreOpaque,
        trampolines: &Trampolines,
        func: RecordedFunc,
        call: RecordedCall,
        mut values: TryVec<ValRaw>,
    ) -> Result<Self> {
        let entry = store.rr_objects().funcs[func].func.vm_func_ref(store);
        let mut stack_info = try_new::<Box<_>>(VMCommonStackInformation::running_default())?;
        let control = try_new::<Box<_>>(VMReplayControl {
            switch: VmPtr::from(NonNull::new(RawFiber::switch_routine() as *mut u8).unwrap()),
            switch_arg: VmPtr::from(NonNull::<u8>::dangling()),
            entry: VmPtr::from(entry),
            entry_caller: VmPtr::from(VMOpaqueContext::from_vmcontext(store.default_caller())),
            entry_values: VmPtr::from(NonNull::from(&mut values[..]).cast()),
            entry_values_len: values.len(),
            reason: 0,
            host_succeeded: 0,
            host_callee: None,
            host_values: None,
            host_values_len: 0,
        })?;
        let control = SendSyncPtr::new(NonNull::from(Box::leak(control)));
        let fiber = match Self::new_fiber(store, trampolines, control) {
            Ok(fiber) => fiber,
            Err(e) => {
                // SAFETY: no fiber refers to the control block.
                drop(unsafe { Box::from_raw(control.as_ptr()) });
                return Err(e);
            }
        };
        // SAFETY: the control block is not shared until the fiber runs.
        unsafe {
            (*control.as_ptr()).switch_arg = VmPtr::from(NonNull::new(fiber.switch_arg()).unwrap());
        }

        // Like an ordinary async call, Wasm may use `max_wasm_stack` bytes of
        // the fiber's stack and host libcalls the remainder.
        let range = fiber.stack().range().unwrap();
        let stack_limit = range
            .end
            .saturating_sub(store.engine().config().max_wasm_stack)
            .max(range.start);
        let context =
            EntryStoreContext::rr_initial(store, stack_limit, NonNull::from(&mut *stack_info));
        Ok(Activation {
            fiber,
            control,
            context,
            _stack: stack_info,
            mpk: store.has_pkey().then(ProtectionMask::all),
            func,
            call,
            host: None,
            values,
        })
    }

    fn new_fiber(
        store: &mut StoreOpaque,
        trampolines: &Trampolines,
        control: SendSyncPtr<VMReplayControl>,
    ) -> Result<RawFiber> {
        let stack = store.allocate_fiber_stack()?;
        // SAFETY: the start trampoline follows the raw fiber entry contract,
        // and `control` outlives the fiber (see `Activation::dispose`).
        let fiber = unsafe {
            RawFiber::new(
                stack,
                core::mem::transmute::<*mut u8, RawFiberEntry>(trampolines.start.as_ptr()),
                control.as_ptr().cast(),
            )
        };
        fiber.map_err(|(e, stack)| {
            store.deallocate_fiber_stack(stack);
            e
        })
    }

    /// Frees an activation. Nothing on its fiber runs or is unwound.
    pub fn dispose(self, store: &mut StoreOpaque) {
        store.deallocate_fiber_stack(self.fiber.into_stack());
        // SAFETY: the fiber that referred to the control block is gone.
        drop(unsafe { Box::from_raw(self.control.as_ptr()) });
    }

    /// The results of the entry function, once the activation has finished
    /// successfully.
    pub fn results(&self) -> *const ValRaw {
        self.values.as_ptr()
    }

    /// Runs this activation until its next yield. A `pending` error makes the
    /// host call that it is parked at fail; its results must otherwise have
    /// been written.
    pub fn resume(&mut self, store: &mut StoreOpaque, pending: Option<Error>) -> Result<Yield> {
        self.host = None;
        let control = self.control.as_ptr();
        // SAFETY: the activation is not running.
        unsafe {
            (*control).reason = 0;
            (*control).host_succeeded = u32::from(pending.is_none());
        }

        // Install this activation's runtime state. Everything installed here
        // is restored before returning, so nothing refers to the fiber while
        // it is suspended.
        let guard_range = self
            .fiber
            .stack()
            .guard_range()
            .unwrap_or(core::ptr::null_mut()..core::ptr::null_mut());
        let cx = store.vm_store_context_mut();
        cx.replay_control = Some(VmPtr::from(self.control.as_non_null()));
        let guard_range = core::mem::replace(&mut cx.async_guard_range, guard_range);
        let mpk = self.mpk.map(|mask| {
            let current = mpk::current_mask();
            mpk::allow(mask);
            current
        });
        self.context.rr_swap();
        let fiber = &mut self.fiber;
        let result =
            crate::runtime::vm::catch_replay_traps(store, &mut self.context, pending, || {
                // SAFETY: the fiber's control block and buffers are live, and its
                // runtime state is installed. Finished activations are not resumed.
                let resumed = unsafe { fiber.resume() };
                debug_assert!(resumed.is_ok());
                // SAFETY: the activation has yielded.
                unsafe { (*control).reason != RrYield::Trapped as u32 }
            });
        self.context.rr_swap();
        if let Some(mask) = mpk {
            self.mpk = Some(mpk::current_mask());
            mpk::allow(mask);
        }
        let cx = store.vm_store_context_mut();
        cx.async_guard_range = guard_range;
        cx.replay_control = None;
        #[cfg(feature = "std")]
        crate::runtime::vm::AsyncWasmCallState::assert_current_state_not_in_range(
            self.fiber.stack().range().unwrap(),
        );
        if cfg!(debug_assertions) {
            assert_only_generated_code(&self.fiber);
        }

        // SAFETY: the activation has yielded, and this reference does not
        // outlive this function.
        let control = unsafe { &*control };
        match RrYield::from_u32(control.reason) {
            Some(RrYield::HostCall) => {
                let callee = control.host_callee.unwrap().as_non_null();
                let objects = store.rr_objects();
                // SAFETY: replay host calls are only reachable through stubs
                // whose context is a `VMArrayCallHostFuncContext`.
                let id = objects.find_func_ref(unsafe {
                    &VMArrayCallHostFuncContext::from_opaque(callee)
                        .as_ref()
                        .func_ref
                })?;
                let func = &objects.funcs[id];
                ensure!(
                    func.host
                        && control.host_values_len >= func.params.len().max(func.results.len()),
                    "invalid replayed host call"
                );
                let values = NonNull::slice_from_raw_parts(
                    control.host_values.unwrap().as_non_null().cast(),
                    control.host_values_len,
                );
                Ok(Yield::Host(id, HostValues(SendSyncPtr::from(values))))
            }
            Some(RrYield::Returned | RrYield::Trapped) => {
                self.fiber.finish()?;
                Ok(Yield::Complete(result))
            }
            None => bail!(
                "replay activation yielded with unknown reason {}",
                control.reason
            ),
        }
    }
}

/// Asserts that a suspended activation's stack contains only generated code,
/// beneath the switch routine and above the fiber's start routine, by walking
/// its frame-pointer chain. Code is looked up in the global code registry,
/// which only exists with native signals.
#[cfg(all(has_host_compiler_backend, has_native_signals))]
fn assert_only_generated_code(fiber: &RawFiber) {
    use crate::runtime::vm::{Unwind, UnwindHost};
    let Some((mut pc, mut fp)) = fiber.suspended_frame() else {
        return;
    };
    let range = fiber.stack().range().unwrap();
    loop {
        assert!(
            crate::module::lookup_code(pc).is_some(),
            "replay activation suspended with a host frame at {pc:#x}"
        );
        assert!(
            range.contains(&fp),
            "replay activation has an invalid frame chain"
        );
        // SAFETY: `fp` is a frame pointer of generated code on this stack.
        let (older_pc, older_fp) = unsafe {
            (
                UnwindHost.get_next_older_pc_from_fp(fp),
                *(fp as *const usize).byte_add(UnwindHost.next_older_fp_from_fp_offset()),
            )
        };
        // The start trampoline's caller is the fiber's start routine, whose
        // frame pointer is the initial one rather than a frame on the stack.
        if !range.contains(&older_fp) || older_fp <= fp {
            return;
        }
        (pc, fp) = (older_pc, older_fp);
    }
}

#[cfg(not(all(has_host_compiler_backend, has_native_signals)))]
fn assert_only_generated_code(_: &RawFiber) {}
