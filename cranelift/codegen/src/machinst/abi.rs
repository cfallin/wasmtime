//! ABI definitions.

use super::{Reg, SpillSlot, Writable};
use crate::binemit::StackMap;
use crate::ir::{Signature, StackSlot};
use crate::isa::CallConv;
use crate::machinst::*;
use crate::settings;

/// Trait implemented by an object that tracks ABI-related state (e.g., stack
/// layout) and can generate code while emitting the *body* of a function.
pub trait ABICallee {
    /// The instruction type for the ISA associated with this ABI.
    type I: VCodeInst;

    /// Creeate a new callee-side ABI object.
    fn create() -> Self;

    /// Initialize, allocating any temps that will be necessary.
    fn init<C: LowerCtx<I = Self::I>>(&mut self, ctx: &mut C) -> Self;

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

    /// Generate an args pseudo-instruction. This instruction is meant
    /// to come first in the function body and capture the values of
    /// function arguments, which it returns in VRegs. Once regalloc
    /// is complete, it will be passed to `emit_prologue` below to
    /// generate a true prologue.
    ///
    /// This takes a `&mut self` so that it can record various
    /// parameters that need to be used later, e.g. StructReturn
    /// pointers.
    fn gen_args(&mut self, args: &[ValueRegs<Writable<Reg>>]) -> Self::I;

    /// Generate a return pseudo-instruction. The given values will be
    /// used as return values. Once regalloc is complete, this
    /// pseudo-instruction will be passed to `emit_epilogue` below to
    /// generate a true epilogue.
    ///
    /// This also handles ABI-inserted return values, such as
    /// StructReturn pointers that must come from the args.
    fn gen_ret(&self, retvals: &[ValueRegs<Reg>], is_fallthrough: bool) -> Self::I;

    /// Get the address of a stackslot.
    fn gen_stackslot_addr(&self, slot: StackSlot, offset: u32, into: Writable<Reg>) -> Self::I;

    // -----------------------------------------------------------------
    // Every function above this line may only be called pre-regalloc.
    // Every function below this line may only be called post-regalloc.
    // `spillslots()` must be called before any other post-regalloc
    // function.
    // ----------------------------------------------------------------

    /// Update with the number of spillslots, post-regalloc.
    fn set_num_spillslots(&mut self, slots: usize);

    /// Update with the clobbered registers, post-regalloc.
    fn set_clobbered(&mut self, clobbered: Vec<Writable<PReg>>);

    /// Generate a stack map, given a list of spillslots and the emission state
    /// at a given program point (prior to emission fo the safepointing
    /// instruction).
    fn spillslots_to_stack_map(
        &self,
        slots: &[SpillSlot],
        state: &<Self::I as MachInstEmit>::State,
    ) -> StackMap;

    /// Generate a prologue, post-regalloc, given the args
    /// pseudo-instruction as input. This should include any stack
    /// frame or other setup necessary to load the arguments, and
    /// should save the clobbers that were provided to
    /// `set_clobbered()`.  `self` is mutable so that we can store
    /// information in it which will be useful when creating the
    /// epilogue.
    fn gen_prologue(&mut self, arginst: Self::I) -> Vec<Self::I>;

    /// Generate an epilogue, post-regalloc. This is provided the
    /// return pseudo-instruction as input.
    fn gen_epilogue(&self, retinst: Self::I) -> Vec<Self::I>;

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
    fn gen_spill(&self, to_slot: SpillSlot, from_reg: Reg, ty: Option<Type>) -> Self::I;

    /// Generate a reload (fill). As for spills, the type may be given to allow
    /// a more optimized load instruction to be generated.
    ///
    /// This returns the instruction (rather than emitting to the
    /// context) because it is used post-regalloc to implement edits,
    /// rather than during lowering itself.
    fn gen_reload(&self, to_reg: Writable<Reg>, from_slot: SpillSlot, ty: Option<Type>) -> Self::I;

    /// Generate a stack-to-stack move. Some architectures may be able
    /// to do this without requiring a scratch register; on others, a
    /// scratch register should be reserved as needed (and not
    /// provided to the allocator).
    fn gen_stack_move(&self, to_slot: SpillSlot, from_slot: SpillSlot, ty: Type) -> Self::I;
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
    fn add_retval(&mut self, regs: ValueRegs<Writable<VReg>>);

    /// Emit the call itself.
    fn emit_call<C: LowerCtx<I = Self::I>>(&mut self, ctx: &mut C);

    /// Return a static list of clobbers for a call with this calling
    /// convention and possibly return-value configuration.
    fn clobbers(&self) -> &'static [PReg];
}
