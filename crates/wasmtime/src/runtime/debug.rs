//! Debugging API.

use crate::{
    AnyRef, AsContext, AsContextMut, ExnRef, ExternRef, Func, Instance, Module, OwnedRooted,
    StoreContext, StoreContextMut, Val,
    runtime::store::AsStoreOpaque,
    store::{AutoAssertNoGc, StoreInner, StoreOpaque},
    vm::{
        Activation, ActivationsBacktrace, AsyncWasmCallState, CallThreadState, FrameOrHost,
        VMContext,
    },
};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::{ffi::c_void, pin::Pin, ptr::NonNull, task::Poll};
#[cfg(feature = "gc")]
use wasmtime_environ::FrameTable;
use wasmtime_environ::{
    DefinedFuncIndex, FrameInstPos, FrameStackShape, FrameStateSlot, FrameStateSlotOffset,
    FrameTableDescriptorIndex, FrameValType, FuncKey, Trap,
};
use wasmtime_unwinder::Frame;

impl<'a, T> StoreContextMut<'a, T> {
    /// Provide an object that captures Wasm stack state, including
    /// Wasm VM-level values (locals and operand stack).
    ///
    /// This object views all activations for the current store that
    /// are on the stack. An activation is a contiguous sequence of
    /// Wasm frames (called functions) that were called from host code
    /// and called back out to host code. If there are activations
    /// from multiple stores on the stack, for example if Wasm code in
    /// one store calls out to host code which invokes another Wasm
    /// function in another store, then the other stores are "opaque"
    /// to our view here in the same way that host code is.
    ///
    /// Returns `None` if debug instrumentation is not enabled for
    /// the engine containing this store.
    pub fn debug_frames(self) -> Option<DebugFrameCursor<'a, T>> {
        if !self.engine().tunables().debug_guest {
            return None;
        }

        let store_opaque = self.0.as_store_opaque();
        let (entry_fp, exit_fp, exit_pc) = unsafe {
            (
                *store_opaque.vm_store_context().last_wasm_entry_fp.get(),
                store_opaque.vm_store_context().last_wasm_exit_fp(),
                *store_opaque.vm_store_context().last_wasm_exit_pc.get(),
            )
        };

        // SAFETY: This takes a mutable borrow of `self` (the
        // `StoreOpaque`), which owns all active stacks in the
        // store. We do not provide any API that could mutate the
        // frames that we are walking on the `DebugFrameCursor`.
        let iter = unsafe {
            ActivationsBacktrace::new(
                self,
                vec![Activation {
                    entry_fp,
                    exit_fp,
                    exit_pc,
                }],
            )
        };
        let mut view = DebugFrameCursor {
            iter,
            is_trapping_frame: false,
            frames: vec![],
            current: None,
        };
        view.move_to_parent(); // Load the first frame.
        Some(view)
    }

    /// Start a "debug session".
    ///
    /// Runs an async body on a store within a debugger context,
    /// providing intermediate debug-step results when execution
    /// yields for any debug-related event.
    ///
    /// Returns `None` if debug instrumentation is not enabled for the
    /// engine containing this store.
    pub fn with_debugger<F: Future<Output = anyhow::Result<()>> + 'a>(
        self,
        body: impl FnOnce(StoreWithDebugSession<'a, T>) -> F,
    ) -> Option<DebugSession<'a, T>> {
        assert!(self.engine().config().async_support);
        if !self.engine().tunables().debug_guest {
            return None;
        }

        let raw_store = NonNull::from_mut(self.0);
        let store = StoreWithDebugSession::new(self);
        let future = body(store);
        Some(crate::DebugSession::new(raw_store, future))
    }
}

/// A view of an active stack frame, with the ability to move up the
/// stack.
///
/// See the documentation on `Store::stack_value` for more information
/// about which frames this view will show.
pub struct DebugFrameCursor<'a, T: 'static> {
    /// Iterator over frames.
    ///
    /// This iterator owns the store while the view exists (accessible
    /// as `iter.store`).
    iter: ActivationsBacktrace<'a, T>,

    /// Is the next frame to be visited by the iterator a trapping
    /// frame?
    ///
    /// This alters how we interpret `pc`: for a trap, we look at the
    /// instruction that *starts* at `pc`, while for all frames
    /// further up the stack (i.e., at a callsite), we look at the
    /// instruction that *ends* at `pc`.
    is_trapping_frame: bool,

    /// Virtual frame queue: decoded from `iter`, not yet
    /// yielded. Innermost frame on top (last).
    ///
    /// This is only non-empty when there is more than one virtual
    /// frame in a physical frame (i.e., for inlining); thus, its size
    /// is bounded by our inlining depth.
    frames: Vec<VirtualFrame>,

    /// Currently focused virtual frame.
    current: Option<FrameData>,
}

/// Result of [`DebugFrameCursor::move_to_parent`]: indicates whether
/// the cursor moved to a new Wasm activation (i.e., across host
/// frames).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugParentFrame {
    /// The `move_to_parent()` invocation moved to a new frame in the
    /// same Wasm activation.
    SameActivation,
    /// The `move_to_parent()` invocation moved to a new frame in the
    /// next older Wasm activation, skipping over one or more host
    /// frames between Wasm exit and Wasm entry.
    NewActivation,
    /// The `move_to_parent()` invocation reached the end of all Wasm
    /// frames.
    Done,
}

impl<'a, T: 'static> DebugFrameCursor<'a, T> {
    /// Move up to the next frame in the activation.
    pub fn move_to_parent(&mut self) -> DebugParentFrame {
        // If there are no virtual frames to yield, take and decode
        // the next physical frame.
        //
        // Note that `if` rather than `while` here, and the assert
        // that we get some virtual frames back, enforce the invariant
        // that each physical frame decodes to at least one virtual
        // frame (i.e., there are no physical frames for interstitial
        // functions or other things that we completely ignore). If
        // this ever changes, we can remove the assert and convert
        // this to a loop that polls until it finds virtual frames.
        let mut result = DebugParentFrame::SameActivation;
        self.current = None;
        if self.frames.is_empty() {
            let next_frame = loop {
                match self.iter.next() {
                    Some(FrameOrHost::Frame(frame)) => {
                        break frame;
                    }
                    Some(FrameOrHost::Host) => {
                        result = DebugParentFrame::NewActivation;
                        continue;
                    }
                    None => {
                        return DebugParentFrame::Done;
                    }
                }
            };
            self.frames = VirtualFrame::decode(
                self.iter.store.0.as_store_opaque(),
                next_frame,
                self.is_trapping_frame,
            );

            debug_assert!(!self.frames.is_empty());
            self.is_trapping_frame = false;
        };

        // Take a frame and focus it as the current one.
        self.current = self.frames.pop().map(|vf| FrameData::compute(vf));
        result
    }

    /// Has the iterator reached the end of all Wasm frames?
    pub fn done(&self) -> bool {
        self.current.is_none()
    }

    fn frame_data(&self) -> &FrameData {
        self.current.as_ref().expect("No current frame")
    }

    fn raw_instance(&self) -> &crate::vm::Instance {
        // Read out the vmctx slot.

        // SAFETY: vmctx is always at offset 0 in the slot.
        // (See crates/cranelift/src/func_environ.rs in `update_stack_slot_vmctx()`.)
        let vmctx: *mut VMContext = unsafe { *(self.frame_data().slot_addr as *mut _) };
        let vmctx = NonNull::new(vmctx).expect("null vmctx in debug state slot");
        // SAFETY: the stored vmctx value is a valid instance in this
        // store; we only visit frames from this store in the
        // backtrace.
        let instance = unsafe { crate::vm::Instance::from_vmctx(vmctx) };
        // SAFETY: the instance pointer read above is valid.
        unsafe { instance.as_ref() }
    }

    /// Get the instance associated with the current frame.
    pub fn instance(&mut self) -> Instance {
        let instance = self.raw_instance();
        Instance::from_wasmtime(instance.id(), self.iter.store.0.as_store_opaque())
    }

    /// Get the module associated with the current frame, if any
    /// (i.e., not a container instance for a host-created entity).
    pub fn module(&self) -> Option<&Module> {
        let instance = self.raw_instance();
        instance.runtime_module()
    }

    /// Get the raw function index associated with the current frame, and the
    /// PC as an offset within its code section, if it is a Wasm
    /// function directly from the given `Module` (rather than a
    /// trampoline).
    pub fn wasm_function_index_and_pc(&self) -> Option<(DefinedFuncIndex, u32)> {
        let data = self.frame_data();
        let FuncKey::DefinedWasmFunction(module, func) = data.func_key else {
            return None;
        };
        debug_assert_eq!(
            module,
            self.module()
                .expect("module should be defined if this is a defined function")
                .env_module()
                .module_index
        );
        Some((func, data.wasm_pc))
    }

    /// Get the number of locals in this frame.
    pub fn num_locals(&self) -> u32 {
        u32::try_from(self.frame_data().locals.len()).unwrap()
    }

    /// Get the depth of the operand stack in this frame.
    pub fn num_stacks(&self) -> u32 {
        u32::try_from(self.frame_data().stack.len()).unwrap()
    }

    /// Get the type and value of the given local in this frame.
    ///
    /// # Panics
    ///
    /// Panics if the index is out-of-range (greater than
    /// `num_locals()`).
    pub fn local(&mut self, index: u32) -> Val {
        let data = self.frame_data();
        let (offset, ty) = data.locals[usize::try_from(index).unwrap()];
        let slot_addr = data.slot_addr;
        // SAFETY: compiler produced metadata to describe this local
        // slot and stored a value of the correct type into it.
        unsafe { read_value(&mut self.iter.store.0, slot_addr, offset, ty) }
    }

    /// Get the type and value of the given operand-stack value in
    /// this frame.
    ///
    /// Index 0 corresponds to the bottom-of-stack, and higher indices
    /// from there are more recently pushed values.  In other words,
    /// index order reads the Wasm virtual machine's abstract stack
    /// state left-to-right.
    pub fn stack(&mut self, index: u32) -> Val {
        let data = self.frame_data();
        let (offset, ty) = data.stack[usize::try_from(index).unwrap()];
        let slot_addr = data.slot_addr;
        // SAFETY: compiler produced metadata to describe this
        // operand-stack slot and stored a value of the correct type
        // into it.
        unsafe { read_value(&mut self.iter.store.0, slot_addr, offset, ty) }
    }
}

/// Internal data pre-computed for one stack frame.
///
/// This combines physical frame info (pc, fp) with the module this PC
/// maps to (yielding a frame table) and one frame as produced by the
/// progpoint lookup (Wasm PC, frame descriptor index, stack shape).
struct VirtualFrame {
    /// The frame pointer.
    fp: *const u8,
    /// The resolved module handle for the physical PC.
    ///
    /// The module for each inlined frame within the physical frame is
    /// resolved from the vmctx reachable for each such frame; this
    /// module isused only for looking up the frame table.
    module: Module,
    /// The Wasm PC for this frame.
    wasm_pc: u32,
    /// The frame descriptor for this frame.
    frame_descriptor: FrameTableDescriptorIndex,
    /// The stack shape for this frame.
    stack_shape: FrameStackShape,
}

impl VirtualFrame {
    /// Return virtual frames corresponding to a physical frame, from
    /// outermost to innermost.
    fn decode(store: &mut StoreOpaque, frame: Frame, is_trapping_frame: bool) -> Vec<VirtualFrame> {
        let module = store
            .modules()
            .lookup_module_by_pc(frame.pc())
            .expect("Wasm frame PC does not correspond to a module");
        let base = module.code_object().code_memory().text().as_ptr() as usize;
        let pc = frame.pc().wrapping_sub(base);
        let table = module.frame_table().unwrap();
        let pc = u32::try_from(pc).expect("PC offset too large");
        let pos = if is_trapping_frame {
            FrameInstPos::Pre
        } else {
            FrameInstPos::Post
        };
        let program_points = table.find_program_point(pc, pos).expect("There must be a program point record in every frame when debug instrumentation is enabled");

        program_points
            .map(|(wasm_pc, frame_descriptor, stack_shape)| VirtualFrame {
                fp: core::ptr::with_exposed_provenance(frame.fp()),
                module: module.clone(),
                wasm_pc,
                frame_descriptor,
                stack_shape,
            })
            .collect()
    }
}

/// Data computed when we visit a given frame.
struct FrameData {
    slot_addr: *const u8,
    func_key: FuncKey,
    wasm_pc: u32,
    /// Shape of locals in this frame.
    ///
    /// We need to store this locally because `FrameView` cannot
    /// borrow the store: it needs a mut borrow, and an iterator
    /// cannot yield the same mut borrow multiple times because it
    /// cannot control the lifetime of the values it yields (the
    /// signature of `next()` does not bound the return value to the
    /// `&mut self` arg).
    locals: Vec<(FrameStateSlotOffset, FrameValType)>,
    /// Shape of the stack slots at this program point in this frame.
    ///
    /// In addition to the borrowing-related reason above, we also
    /// materialize this because we want to provide O(1) access to the
    /// stack by depth, and the frame slot descriptor stores info in a
    /// linked-list (actually DAG, with dedup'ing) way.
    stack: Vec<(FrameStateSlotOffset, FrameValType)>,
}

impl FrameData {
    fn compute(frame: VirtualFrame) -> Self {
        let frame_table = frame.module.frame_table().unwrap();
        // Parse the frame descriptor.
        let (data, slot_to_fp_offset) = frame_table
            .frame_descriptor(frame.frame_descriptor)
            .unwrap();
        let frame_state_slot = FrameStateSlot::parse(data).unwrap();
        let slot_addr = frame
            .fp
            .wrapping_sub(usize::try_from(slot_to_fp_offset).unwrap());

        // Materialize the stack shape so we have O(1) access to its
        // elements, and so we don't need to keep the borrow to the
        // module alive.
        let mut stack = frame_state_slot
            .stack(frame.stack_shape)
            .collect::<Vec<_>>();
        stack.reverse(); // Put top-of-stack last.

        // Materialize the local offsets/types so we don't need to
        // keep the borrow to the module alive.
        let locals = frame_state_slot.locals().collect::<Vec<_>>();

        FrameData {
            slot_addr,
            func_key: frame_state_slot.func_key(),
            wasm_pc: frame.wasm_pc,
            stack,
            locals,
        }
    }
}

/// Read the value at the given offset.
///
/// # Safety
///
/// The `offset` and `ty` must correspond to a valid value written
/// to the frame by generated code of the correct type. This will
/// be the case if this information comes from the frame tables
/// (as long as the frontend that generates the tables and
/// instrumentation is correct, and as long as the tables are
/// preserved through serialization).
unsafe fn read_value(
    store: &mut StoreOpaque,
    slot_base: *const u8,
    offset: FrameStateSlotOffset,
    ty: FrameValType,
) -> Val {
    let address = unsafe { slot_base.offset(isize::try_from(offset.offset()).unwrap()) };

    // SAFETY: each case reads a value from memory that should be
    // valid according to our safety condition.
    match ty {
        FrameValType::I32 => {
            let value = unsafe { *(address as *const i32) };
            Val::I32(value)
        }
        FrameValType::I64 => {
            let value = unsafe { *(address as *const i64) };
            Val::I64(value)
        }
        FrameValType::F32 => {
            let value = unsafe { *(address as *const u32) };
            Val::F32(value)
        }
        FrameValType::F64 => {
            let value = unsafe { *(address as *const u64) };
            Val::F64(value)
        }
        FrameValType::V128 => {
            let value = unsafe { *(address as *const u128) };
            Val::V128(value.into())
        }
        FrameValType::AnyRef => {
            let mut nogc = AutoAssertNoGc::new(store);
            let value = unsafe { *(address as *const u32) };
            let value = AnyRef::_from_raw(&mut nogc, value);
            Val::AnyRef(value)
        }
        FrameValType::ExnRef => {
            let mut nogc = AutoAssertNoGc::new(store);
            let value = unsafe { *(address as *const u32) };
            let value = ExnRef::_from_raw(&mut nogc, value);
            Val::ExnRef(value)
        }
        FrameValType::ExternRef => {
            let mut nogc = AutoAssertNoGc::new(store);
            let value = unsafe { *(address as *const u32) };
            let value = ExternRef::_from_raw(&mut nogc, value);
            Val::ExternRef(value)
        }
        FrameValType::FuncRef => {
            let value = unsafe { *(address as *const *mut c_void) };
            let value = unsafe { Func::_from_raw(store, value) };
            Val::FuncRef(value)
        }
        FrameValType::ContRef => {
            unimplemented!("contref values are not implemented in the host API yet")
        }
    }
}

/// Compute raw pointers to all GC refs in the given frame.
// Note: ideally this would be an impl Iterator, but this is quite
// awkward because of the locally computed data (FrameStateSlot::parse
// structured result) within the closure borrowed by a nested closure.
#[cfg(feature = "gc")]
pub(crate) fn gc_refs_in_frame<'a>(ft: FrameTable<'a>, pc: u32, fp: *mut usize) -> Vec<*mut u32> {
    let fp = fp.cast::<u8>();
    let mut ret = vec![];
    if let Some(frames) = ft.find_program_point(pc, FrameInstPos::Post) {
        for (_wasm_pc, frame_desc, stack_shape) in frames {
            let (frame_desc_data, slot_to_fp_offset) = ft.frame_descriptor(frame_desc).unwrap();
            let frame_base = unsafe { fp.offset(-isize::try_from(slot_to_fp_offset).unwrap()) };
            let frame_desc = FrameStateSlot::parse(frame_desc_data).unwrap();
            for (offset, ty) in frame_desc.stack_and_locals(stack_shape) {
                match ty {
                    FrameValType::AnyRef | FrameValType::ExnRef | FrameValType::ExternRef => {
                        let slot = unsafe {
                            frame_base
                                .offset(isize::try_from(offset.offset()).unwrap())
                                .cast::<u32>()
                        };
                        ret.push(slot);
                    }
                    FrameValType::ContRef | FrameValType::FuncRef => {}
                    FrameValType::I32
                    | FrameValType::I64
                    | FrameValType::F32
                    | FrameValType::F64
                    | FrameValType::V128 => {}
                }
            }
        }
    }
    ret
}

impl<'a, T: 'static> AsContext for DebugFrameCursor<'a, T> {
    type Data = T;
    fn as_context(&self) -> StoreContext<'_, Self::Data> {
        StoreContext(self.iter.store.0)
    }
}
impl<'a, T: 'static> AsContextMut for DebugFrameCursor<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, Self::Data> {
        StoreContextMut(self.iter.store.0)
    }
}

/// A "debug session": a future that is the result of
/// `Store::with_debugger(...)`.
///
/// A debug session owns the store while active, and runs an inner
/// user-provided closure that performs some action (e.g., invoke some
/// Wasm function) while setting up a debug context on the store.
///
/// This debug context allows the debug session to return an
/// intermediate [`DebugStepResult`] each time some debug action
/// occurs. These actions could include completing the inner closure,
/// hitting a breakpoint or watchpoint, or hitting a trap.
pub struct DebugSession<'a, T: 'static> {
    /// Raw pointer to store. We own this only when control is yielded
    /// back to us from the future, and we use it only to introspect
    /// the stack (and eventually to update breakpoints, etc).
    /// Ownership of the store itself tmus passes back and forth
    /// dynamically.
    ///
    /// As long as we know that the future is not running, we can
    /// safely access this (i.e., we own the store). We never let
    /// `future` escape, so we maintain this invariant solely within
    /// this file.
    ///
    /// Note that we need the `Store<T>`, and not just the
    /// `StoreOpaque`, because we need to provide an `AsContextMut`
    /// for GC accessors and other methods when the guest is paused
    /// for debugging.
    raw_store: NonNull<StoreInner<T>>,
    state: DebugSessionState<'a>,
}

type BodyFuture<'a> = Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;

enum DebugSessionState<'a> {
    Running(BodyFuture<'a>),
    TrappedFromHost {
        future: BodyFuture<'a>,
        activations: Vec<Activation>,
    },
    Finished,
}

/// State kept on `StoreOpaque` and used/updated by trap handling to
/// know when debugging is active and to pass state back to us.
#[derive(Default)]
pub(crate) struct DebuggingStoreState {
    /// Is debugging active on this store?
    pub(crate) active: bool,
    /// State yielded back when the inner future returns from its
    /// poll.
    pub(crate) debug_yield: Option<DebugYield>,
    /// The raw CallThreadState linked list for each fiber, from
    /// youngest to oldest, in the invocation chain when the debug
    /// event yields.
    ///
    /// These pointers are taken from the chain of
    /// `AsyncWasmCallState`s that are saved when we suspend each
    /// fiber in turn.
    ///
    /// SAFETY: These are on stacks owned by fibers that are in turn
    /// owned by futures, the outermost of which is kept alive by the
    /// debugging session. The list is taken and immediately cleared
    /// here when the outermost future returns pending into the
    /// debugging session implementation below; the copy that is taken
    /// is alive only as long as that future is held alive and not
    /// polled again.
    activations: Vec<Activation>,
}

impl DebuggingStoreState {
    /// Collect activations be observed during a debug yield.
    ///
    /// We collect these and preprocess into FP/PC data during fiber
    /// suspends; this is needed because there is no store-accessible
    /// linkage of `CallThreadState` otherwise (they are unlinked from
    /// the TLS chain and saved in `AsyncWasmCallState` structs in
    /// fibers, which are each owned by a stack frame in a parent
    /// fiber). The only way to get at them is to observe them as we
    /// unlink on suspend. Furthermore, the O(1) state-saving
    /// mechanism, by which the data for the *youngest* activation on
    /// a stack is saved in the previous-values for the *oldest* (see
    /// `AsyncWasmCallState::push()`'s comments in traphandlers.rs),
    /// means that unless we retain the grouping of
    /// activations-per-fiber (a list-of-lists), we need to reorder
    /// appropriately in an eager way as we unlink.
    pub(crate) fn push_activations(&mut self, state: &AsyncWasmCallState) {
        if self.debug_yield.is_some() {
            // The linked list on `state.state` is in
            // oldest-to-youngest order; so we traverse and push in
            // that order but then reverse, except that the oldest
            // remains the oldest (see comment above). This is
            // equivalent to rotating-by-one then reversing.
            //
            // In other words, if we have
            //
            // (oldest)
            //   A      ->   B   ->        C   ->     D
            //
            // with `old_state`s
            //
            // (D's data)   (A's data) (B's data) (C's data)
            //
            // Then we want
            //
            // D            C              B            A
            //
            // to get youngest-to-oldest ordering.
            let start = self.activations.len();
            let mut state: *mut CallThreadState = state.state;
            while !state.is_null() {
                let (entry_fp, exit_fp, exit_pc) = unsafe {
                    (
                        (*state).old_last_wasm_entry_fp(),
                        (*state).old_last_wasm_exit_fp(),
                        (*state).old_last_wasm_exit_pc(),
                    )
                };
                self.activations.push(Activation {
                    entry_fp,
                    exit_fp,
                    exit_pc,
                });
                state = unsafe { (*state).prev() };
            }
            let end = self.activations.len();
            if end > start {
                self.activations[(start + 1)..end].reverse();
            }
        }
    }
}

/// State set on a store to communicate a debug event on yield.
pub(crate) enum DebugYield {
    /// A trap raised by a hostcall.
    TrapHost(Trap),
    /// An error raised by a hostcall.
    TrapHostError,
    /// An exception raised by a hostcall and not caught by Wasm code.
    TrapHostException,
}

/// A "debug step result": one event that happens when running actions
/// in a `debug_call` invocation on a store.
#[derive(Debug)]
pub enum DebugStepResult {
    /// The call completed and returned a value.
    Complete,
    /// The call completed and return an error.
    Error(anyhow::Error),
    /// An `anyhow::Error` was raised by a hostcall. This event occurs
    /// before the eventual Error return event. The error is not
    /// available at this step because it must be propagated to the
    /// actual return.
    HostcallError,
    /// The call completed and threw an uncaught exception.
    UncaughtException(OwnedRooted<ExnRef>),
    /// A Wasm trap occurred.
    Trap(Trap),
}

impl StoreOpaque {
    pub(crate) fn debug_yield(&mut self, yield_: DebugYield) {
        let old = self.debugging_state.debug_yield.replace(yield_);
        assert!(old.is_none());
        self.with_blocking(|_store, ctx| {
            ctx.suspend(crate::fiber::StoreFiberYield::KeepStore)
                .unwrap();
        });
    }
}

impl<'a, T: 'static> DebugSession<'a, T> {
    pub(crate) fn new(
        raw_store: NonNull<StoreInner<T>>,
        future: impl Future<Output = anyhow::Result<()>> + 'a,
    ) -> Self {
        Self {
            raw_store,
            state: DebugSessionState::Running(Box::pin(future)),
        }
    }

    /// Get an owning borrow of the associated store.
    ///
    /// # Safety
    ///
    /// Requires that `future` is not running and has yielded from
    /// within Wasm code at a point that logically passes ownership of
    /// the store back (i.e., a trap or breakpoint).
    unsafe fn store_opaque_mut(&mut self) -> &mut StoreOpaque {
        let store = unsafe { self.raw_store.as_mut() };
        store.as_store_opaque()
    }

    /// Run until the next debug step result: completion, a trap, or
    /// some kind of pause/break.
    pub fn next(&mut self) -> impl Future<Output = Option<DebugStepResult>> + '_ {
        struct Fut<'b, 'c, T: 'static>(&'b mut DebugSession<'c, T>);

        impl<'b, 'c, T: 'static> Future for Fut<'b, 'c, T> {
            type Output = Option<DebugStepResult>;
            fn poll(
                mut self: Pin<&mut Self>,
                cx: &mut core::task::Context<'_>,
            ) -> Poll<Self::Output> {
                match core::mem::replace(&mut self.0.state, DebugSessionState::Finished) {
                    DebugSessionState::Finished => Poll::Ready(None),
                    // N.B.: we poll the Wasm function execution
                    // future if it's still normally running *or* if
                    // it trapped; in the latter case, when we resume,
                    // it performs the ordinary exception-throw to
                    // unwind the fiber, because we can't drop the
                    // fiber without completing it. In essence, a trap
                    // "pauses" with a debug-step result so we can
                    // examine the stack, then finishes the teardown.
                    DebugSessionState::TrappedFromHost { mut future, .. }
                    | DebugSessionState::Running(mut future) => {
                        match future.as_mut().poll(cx) {
                            Poll::Ready(value) => {
                                self.0.state = DebugSessionState::Finished;
                                match value {
                                    Ok(()) => Poll::Ready(Some(DebugStepResult::Complete)),
                                    Err(e) => Poll::Ready(Some(DebugStepResult::Error(e))),
                                }
                            }
                            Poll::Pending => {
                                // If there was an explicit debug yield,
                                // translate that to a step result. Otherwise,
                                // yield nothing.
                                let store = unsafe { self.0.store_opaque_mut() };

                                match store.debugging_state.debug_yield.take() {
                                    Some(y @ DebugYield::TrapHost(_))
                                    | Some(y @ DebugYield::TrapHostError)
                                    | Some(y @ DebugYield::TrapHostException) => {
                                        let result = match y {
                                            DebugYield::TrapHost(trapcode) => {
                                                DebugStepResult::Trap(trapcode)
                                            }
                                            DebugYield::TrapHostError => {
                                                DebugStepResult::HostcallError
                                            }
                                            DebugYield::TrapHostException => {
                                                let exn =
                                                    store.take_pending_exception_owned_rooted();
                                                DebugStepResult::UncaughtException(exn.expect("exception should have been stored on the Store"))
                                            }
                                        };
                                        let activations =
                                            core::mem::take(&mut store.debugging_state.activations);
                                        self.0.state = DebugSessionState::TrappedFromHost {
                                            future,
                                            activations,
                                        };
                                        Poll::Ready(Some(result))
                                    }
                                    None => {
                                        self.0.state = DebugSessionState::Running(future);
                                        Poll::Pending
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Fut(self)
    }

    /// Provide a view of the current execution state.
    ///
    /// Returns `None` if execution has already completed.
    pub fn debug_frames(&mut self) -> Option<DebugFrameCursor<'_, T>> {
        let activations = match &self.state {
            DebugSessionState::Running(..) | DebugSessionState::Finished => return None,
            DebugSessionState::TrappedFromHost { activations, .. } => activations.clone(),
        };

        let store = self.as_context_mut();
        let iter = unsafe { ActivationsBacktrace::new(store, activations) };
        let mut view = DebugFrameCursor {
            iter,
            is_trapping_frame: false,
            frames: vec![],
            current: None,
        };
        view.move_to_parent(); // Load the first frame.
        Some(view)
    }
}

impl<'a, T> AsContext for DebugSession<'a, T> {
    type Data = T;
    fn as_context(&self) -> StoreContext<'_, T> {
        let store_inner = unsafe { self.raw_store.as_ref() };
        StoreContext(store_inner)
    }
}
impl<'a, T> AsContextMut for DebugSession<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        let store_inner = unsafe { self.raw_store.as_mut() };
        StoreContextMut(store_inner)
    }
}

impl<'a, T> Drop for DebugSession<'a, T> {
    fn drop(&mut self) {
        // SAFETY: if we are dropping the session, then the inner
        // future is not running, because we do not let it escape and
        // a poll on `next()`'s future (which takes a mut-borrow on
        // `DebugSession`) is the only way to run it. Hence, we can
        // claim ownership of the store.
        let store = unsafe { self.store_opaque_mut() };
        assert!(store.debugging_state.debug_yield.is_none());
    }
}

/// A `Store` with an active debugging session.
///
/// This type wraps a `Store` and is exposed to the body given to
/// [`Store::with_debugger`]. It provides a context using
/// [`AsContext`]/[`AsContextMut`] and ensures that nested store
/// contexts are not created.
pub struct StoreWithDebugSession<'a, T: 'static> {
    store: StoreContextMut<'a, T>,
}

impl<'a, T: 'static> StoreWithDebugSession<'a, T> {
    fn new(mut store: StoreContextMut<'a, T>) -> StoreWithDebugSession<'a, T> {
        let was_active =
            core::mem::replace(&mut store.as_context_mut().0.debugging_state.active, true);
        assert!(
            !was_active,
            "Nested debugging sessions on one store are not supported"
        );
        Self { store }
    }
}

impl<'a, T: 'static> Drop for StoreWithDebugSession<'a, T> {
    fn drop(&mut self) {
        let was_active = core::mem::replace(
            &mut self.store.as_context_mut().0.debugging_state.active,
            false,
        );
        assert!(
            was_active,
            "Debugging session was not active when dropping StoreWithDebugSession"
        );
    }
}

impl<'a, T: 'static> AsContext for StoreWithDebugSession<'a, T> {
    type Data = T;
    fn as_context(&self) -> StoreContext<'_, T> {
        self.store.as_context()
    }
}
impl<'a, T: 'static> AsContextMut for StoreWithDebugSession<'a, T> {
    fn as_context_mut(&mut self) -> StoreContextMut<'_, T> {
        self.store.as_context_mut()
    }
}
