//! ABI definitions.

use super::{Reg, SpillSlot};
use crate::binemit::StackMap;
use crate::ir::{Signature, StackSlot, Function};
use crate::isa::CallConv;
use crate::machinst::*;
use crate::result::CodegenResult;
use crate::settings;
use smallvec::SmallVec;

/// A small vector of instructions (with some reasonable size);
/// appropriate for a small fixed sequence implementing one operation.
pub type SmallInstVec<I> = SmallVec<[I; 4]>;

/// Trait implemented by an object that tracks ABI-related state (e.g., stack
/// layout) and can generate code while emitting the *body* of a function.
pub trait ABICallee: Sized {
    /// The instruction type for the ISA associated with this ABI.
    type I: VCodeInst;

    /// Create a new ABI instance.
    fn new(f: &Function, flags: settings::Flags) -> CodegenResult<Self>;

    /// Access the (possibly legalized) signature.
    fn signature(&self) -> &Signature;

    /// Get the settings controlling this function's compilation.
    fn flags(&self) -> &settings::Flags;

    /// Get the calling convention implemented by this ABI object.
    fn call_conv(&self) -> CallConv;

    /// Number of arguments.
    fn num_args(&self) -> usize;

    /// Number of return values.
    fn num_retvals(&self) -> usize;

    /// Number of stack slots (not spill slots).
    fn num_stackslots(&self) -> usize;

    /// The offsets of all stack slots (not spill slots) for debuginfo purposes.
    fn stackslot_offsets(&self) -> &PrimaryMap<StackSlot, u32>;

    /// Generate an args instruction sequence: the given vregs will be
    /// assigned values from function arguments, either via
    /// constraining their locations with a constraint pseudo-inst or
    /// via explicit copying. This is meant to be the first
    /// instruction in the function body after the prologue (which is
    /// generated later).
    ///
    /// This takes a `&mut self` so that it can record various
    /// parameters that need to be used later, e.g. StructReturn
    /// pointers.
    ///
    /// This also takes the lowering context so that it can allocate
    /// any temps that are necessary.
    fn gen_arg_seq<C: LowerCtx<I = Self::I>>(
        &self,
        ctx: &mut C,
        args: Vec<ValueRegs<VReg>>,
    ) -> SmallInstVec<Self::I>;

    /// Generate a return-value instruction sequence: the given vregs
    /// will be used as return values and copied or constrained to end
    /// up in the proper locations. This sequence should not, however,
    /// include the actual return instruction.
    ///
    /// This also handles ABI-inserted return values, such as
    /// StructReturn pointers that must come from the args.
    ///
    /// This also takes the lowering context so that it can allocate
    /// any temps that are necessary.
    fn gen_ret_seq<C: LowerCtx<I = Self::I>>(
        &self,
        ctx: &mut C,
        retvals: Vec<ValueRegs<VReg>>,
    ) -> SmallInstVec<Self::I>;

    /// Get the address of a stackslot.
    fn gen_stackslot_addr(&self, slot: StackSlot, offset: u32, into: Reg) -> Self::I;

    // -----------------------------------------------------------------
    // Every function above this line may only be called pre-regalloc.
    // Every function below this line may only be called post-regalloc.
    // `spillslots()` must be called before any other post-regalloc
    // function.
    // ----------------------------------------------------------------

    /// Update with the number of spillslots, post-regalloc.
    fn set_num_spillslots(&mut self, slots: usize);

    /// Update with the clobbered registers, post-regalloc.
    fn set_clobbered(&mut self, clobbered: Vec<PReg>);

    /// Generate a stack map, given a list of spillslots and the emission state
    /// at a given program point (prior to emission fo the safepointing
    /// instruction).
    fn spillslots_to_stack_map(
        &self,
        slots: &[SpillSlot],
        state: &<Self::I as MachInstEmit>::State,
    ) -> StackMap;

    /// Generate a prologue, post-regalloc. This should include any stack
    /// frame or other setup necessary to load the arguments, and
    /// should save the clobbers that were provided to
    /// `set_clobbered()`.  `self` is mutable so that we can store
    /// information in it which will be useful when creating the
    /// epilogue.
    fn gen_prologue(&mut self) -> SmallInstVec<Self::I>;

    /// Generate an epilogue, post-regalloc.
    fn gen_epilogue(&self) -> SmallInstVec<Self::I>;

    /// Returns the full frame size for the given function, after prologue
    /// emission has run. This comprises the spill slots and stack-storage slots
    /// (but not storage for clobbered callee-save registers, arguments pushed
    /// at callsites within this function, or other ephemeral pushes).  This is
    /// used for ABI variants where the client generates prologue/epilogue code,
    /// as in Baldrdash (SpiderMonkey integration).
    fn frame_size(&self) -> u32;

    /// Returns the size of arguments expected on the stack.
    fn stack_args_size(&self) -> u32;

    /// Get the spill-slot size.
    fn get_spillslot_size(&self, rc: RegClass, ty: Type) -> u32;

    /// Generate a spill. The type, if known, is given; this can be used to
    /// generate a store instruction optimized for the particular type rather
    /// than the RegClass (e.g., only F64 that resides in a V128 register). If
    /// no type is given, the implementation should spill the whole register.
    ///
    /// This returns the instruction (rather than emitting to the
    /// context) because it is used post-regalloc to implement edits,
    /// rather than during lowering itself.
    fn gen_spill(&self, to_slot: SpillSlot, from_reg: PReg) -> SmallInstVec<Self::I>;

    /// Generate a reload (fill). As for spills, the type may be given to allow
    /// a more optimized load instruction to be generated.
    ///
    /// This returns the instruction (rather than emitting to the
    /// context) because it is used post-regalloc to implement edits,
    /// rather than during lowering itself.
    fn gen_reload(&self, to_reg: PReg, from_slot: SpillSlot) -> SmallInstVec<Self::I>;

    /// Generate a stack-to-stack move. Some architectures may be able
    /// to do this without requiring a scratch register; on others, a
    /// scratch register should be reserved as needed (and not
    /// provided to the allocator).
    fn gen_stack_move(
        &self,
        to_slot: SpillSlot,
        from_slot: SpillSlot,
        ty: Type,
    ) -> SmallInstVec<Self::I>;
}

/// Trait implemented by an object that tracks ABI-related state and can
/// generate code while emitting a *call* to a function.
///
/// An instance of this trait returns information for a *particular*
/// callsite. It will usually be computed from the called function's
/// signature.
///
/// Unlike `ABICallee` above, methods on this trait are not invoked directly
/// by the machine-independent code. Rather, the machine-specific lowering
/// code will typically create an `ABICaller` when creating machine instructions
/// for an IR call instruction inside `lower()`, directly emit the arg and
/// and retval copies, and attach the register use/def info to the call.
///
/// This trait is thus provided for convenience to the backends.
pub trait ABICaller {
    /// The instruction type for the ISA associated with this ABI.
    type I: VCodeInst;

    /// Get the number of arguments expected.
    fn num_args(&self) -> usize;

    /// Access the (possibly legalized) signature.
    fn signature(&self) -> &Signature;

    /// Specify argument input vregs for one argument.
    fn add_argument(&mut self, regs: ValueRegs<VReg>);

    /// Specify return-value output vregs for one return val.
    fn add_retval(&mut self, regs: ValueRegs<VReg>);

    /// Emit the call itself.
    fn emit_call<C: LowerCtx<I = Self::I>>(&mut self, ctx: &mut C);
}
