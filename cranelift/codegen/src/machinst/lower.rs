//! This module implements lowering (instruction selection) from Cranelift IR
//! to machine instructions with virtual registers. This is *almost* the final
//! machine code, except for register allocation.

// TODO: separate the IR-query core of `LowerCtx` from the lowering logic built
// on top of it, e.g. the side-effect/coloring analysis and the scan support.

use super::{Reg, Writable};
use crate::data_value::DataValue;
use crate::entity::SecondaryMap;
use crate::fx::{FxHashMap, FxHashSet};
use crate::inst_predicates::{has_lowering_side_effect, is_constant_64bit};
use crate::ir::instructions::BranchInfo;
use crate::ir::{
    ArgumentPurpose, Block, Constant, ConstantData, ExternalName, Function, GlobalValueData, Inst,
    InstructionData, MemFlags, Opcode, Signature, SourceLoc, Type, Value, ValueDef,
    ValueLabelAssignments, ValueLabelStart,
};
use crate::machinst::{
    writable_value_regs, ABICallee, BlockIndex, BlockLoweringOrder, LoweredBlock, MachLabel, VCode,
    VCodeBuilder, VCodeConstant, VCodeConstantData, VCodeConstants, VCodeInst, VReg, ValueRegs,
};
use crate::CodegenResult;
use alloc::vec::Vec;
use core::convert::TryInto;
use log::debug;
use regalloc2::Operand;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;

/// An "instruction color" partitions CLIF instructions by side-effecting ops.
/// All instructions with the same "color" are guaranteed not to be separated by
/// any side-effecting op (for this purpose, loads are also considered
/// side-effecting, to avoid subtle questions w.r.t. the memory model), and
/// furthermore, it is guaranteed that for any two instructions A and B such
/// that color(A) == color(B), either A dominates B and B postdominates A, or
/// vice-versa. (For now, in practice, only ops in the same basic block can ever
/// have the same color, trivially providing the second condition.) Intuitively,
/// this means that the ops of the same color must always execute "together", as
/// part of one atomic contiguous section of the dynamic execution trace, and
/// they can be freely permuted (modulo true dataflow dependencies) without
/// affecting program behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct InstColor(u32);
impl InstColor {
    fn new(n: u32) -> InstColor {
        InstColor(n)
    }

    /// Get an arbitrary index representing this color. The index is unique
    /// *within a single function compilation*, but indices may be reused across
    /// functions.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A context that machine-specific lowering code can use to emit lowered
/// instructions. This is the view of the machine-independent per-function
/// lowering context that is seen by the machine backend.
pub trait LowerCtx {
    /// The instruction type for which this lowering framework is instantiated.
    type I: VCodeInst;

    // Function-level queries:

    /// Returns the vreg containing the VmContext parameter, if there is one.
    fn get_vm_context(&self) -> Option<Reg>;

    // General instruction queries:

    /// Get the instdata for a given IR instruction.
    fn data(&self, ir_inst: Inst) -> &InstructionData;
    /// Get the controlling type for a polymorphic IR instruction.
    fn ty(&self, ir_inst: Inst) -> Type;
    /// Get the target for a call instruction, as an `ExternalName`. Returns a tuple
    /// providing this name and the "relocation distance", i.e., whether the backend
    /// can assume the target will be "nearby" (within some small offset) or an
    /// arbitrary address. (This comes from the `colocated` bit in the CLIF.)
    fn call_target<'b>(&'b self, ir_inst: Inst) -> Option<(&'b ExternalName, RelocDistance)>;
    /// Get the signature for a call or call-indirect instruction.
    fn call_sig<'b>(&'b self, ir_inst: Inst) -> Option<&'b Signature>;
    /// Get the symbol name, relocation distance estimate, and offset for a
    /// symbol_value instruction.
    fn symbol_value<'b>(&'b self, ir_inst: Inst) -> Option<(&'b ExternalName, RelocDistance, i64)>;
    /// Returns the memory flags of a given memory access.
    fn memflags(&self, ir_inst: Inst) -> Option<MemFlags>;
    /// Get the source location for a given instruction.
    fn srcloc(&self, ir_inst: Inst) -> SourceLoc;

    // Instruction input/output queries:

    /// Get the number of inputs to the given IR instruction.
    fn num_inputs(&self, ir_inst: Inst) -> usize;
    /// Get the number of outputs to the given IR instruction.
    fn num_outputs(&self, ir_inst: Inst) -> usize;
    /// Get the type for an instruction's input.
    fn input_ty(&self, ir_inst: Inst, idx: usize) -> Type;
    /// Get the type for an instruction's output.
    fn output_ty(&self, ir_inst: Inst, idx: usize) -> Type;
    /// Get the value of a constant instruction (`iconst`, etc.) as a 64-bit
    /// value, if possible.
    fn get_constant(&self, ir_inst: Inst) -> Option<u64>;
    /// Get the input as one of two options other than a direct register:
    ///
    /// - An instruction, given that it is effect-free or able to sink its
    ///   effect to the current instruction being lowered, and given it has only
    ///   one output, and if effect-ful, given that this is the only use;
    /// - A constant, if the value is a constant.
    ///
    /// The instruction input may be available in either of these forms.  It may
    /// be available in neither form, if the conditions are not met; if so, use
    /// `input_regs()` instead to get it in a register.
    ///
    /// If the backend merges the effect of a side-effecting instruction, it
    /// must call `sink_inst()`. When this is called, it indicates that the
    /// effect has been sunk to the current scan location. The sunk
    /// instruction's result(s) must have *no* uses remaining, because it will
    /// not be codegen'd (it has been integrated into the current instruction).
    fn get_input_as_source_or_const(&self, ir_inst: Inst, idx: usize) -> NonRegInput;
    /// Get the `idx`th input as register(s) to be used at the
    /// beginning of the instruction. (N.B.: this means that these
    /// input register(s) may be reused by outputs that are defined at
    /// the end of the instruction. If the input is read after the
    /// lowering starts to write outputs, use `input_at_end_regs()`
    /// instead.)
    fn input_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Reg>;
    /// Get the `idx`th input as register(s) to be used at the end of the instruction.
    fn input_at_end_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Reg>;
    /// Get the reg(s) for the `idx`th output of an instruction, to be
    /// defined at the end of the instruction. (N.B.: this means that
    /// these register(s) may overlap with inputs used at the
    /// beginning of the instruction. If the output is written before
    /// all inputs are read, use `output_at_start_regs()` instead.)
    fn output_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Writable<Reg>>;
    /// Get the reg(s) for the `idx`th output of an instruction, to be
    /// defined at the beginning of the instruction.
    fn output_at_start_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Writable<Reg>>;
    /// Get the reg(s) for the `idx_in`th input and `idx_out`th output
    /// of an instruction, specifying that the output *must* reuse the
    /// given input's registers. The input and output must have the
    /// same type.
    fn reused_input_output_regs(
        &mut self,
        ir_inst: Inst,
        input_idx: usize,
        output_idx: usize,
    ) -> (Reg, Writable<Reg>);

    // Codegen primitives: allocate temps, emit instructions, set result registers,
    // ask for an input to be gen'd into a register.

    /// Get a new temp. Because VCode is SSA, this can only be written once.
    fn alloc_tmp(&mut self, ty: Type) -> ValueRegs<Writable<Reg>>;
    /// Emit a machine instruction.
    fn emit(&mut self, mach_inst: Self::I);
    /// Emit a machine instruction that is a safepoint.
    fn emit_safepoint(&mut self, mach_inst: Self::I);
    /// Indicate that the side-effect of an instruction has been sunk to the
    /// current scan location. This should only be done with the instruction's
    /// original results are not used (i.e., `put_input_in_regs` is not invoked
    /// for the input produced by the sunk instruction), otherwise the
    /// side-effect will occur twice.
    fn sink_inst(&mut self, ir_inst: Inst);
    /// Retrieve constant data given a handle.
    fn get_constant_data(&self, constant_handle: Constant) -> &ConstantData;
    /// Indicate that a constant should be emitted.
    fn use_constant(&mut self, constant: VCodeConstantData) -> VCodeConstant;
    /// Retrieve the value immediate from an instruction. This will perform necessary lookups on the
    /// `DataFlowGraph` to retrieve even large immediates.
    fn get_immediate(&self, ir_inst: Inst) -> Option<DataValue>;
}

/// A representation of all of the ways in which a value is available, aside
/// from as a direct register.
///
/// - An instruction, if it would be allowed to occur at the current location
///   instead (see [LowerCtx::get_input_as_source_or_const()] for more
///   details).
///
/// - A constant, if the value is known to be a constant.
#[derive(Clone, Copy, Debug)]
pub struct NonRegInput {
    /// An instruction produces this value (as the given output), and its
    /// computation (and side-effect if applicable) could occur at the
    /// current instruction's location instead.
    ///
    /// If this instruction's operation is merged into the current instruction,
    /// the backend must call [LowerCtx::sink_inst()].
    pub inst: Option<(Inst, usize)>,
    /// The value is a known constant.
    pub constant: Option<u64>,
}

/// A machine backend.
pub trait LowerBackend {
    /// The machine instruction type.
    type MInst: VCodeInst;

    /// Lower a single instruction.
    ///
    /// For a branch, this function should not generate the actual branch
    /// instruction. However, it must force any values it needs for the branch
    /// edge (block-param actuals) into registers, because the actual branch
    /// generation (`lower_branch_group()`) happens *after* any possible merged
    /// out-edge.
    fn lower<C: LowerCtx<I = Self::MInst>>(&self, ctx: &mut C, inst: Inst) -> CodegenResult<()>;

    /// Lower a block-terminating group of branches (which together can be seen
    /// as one N-way branch), given a vcode MachLabel for each target.
    fn lower_branch_group<C: LowerCtx<I = Self::MInst>>(
        &self,
        ctx: &mut C,
        insts: &[Inst],
        targets: &[MachLabel],
    ) -> CodegenResult<()>;
}

/// A pending instruction to insert and auxiliary information about it: its source location and
/// whether it is a safepoint.
struct InstTuple<I: VCodeInst> {
    loc: SourceLoc,
    is_safepoint: bool,
    inst: I,
}

/// Machine-independent lowering driver / machine-instruction container. Maintains a correspondence
/// from original Inst to MachInsts.
pub struct Lower<'func, I: VCodeInst, A: ABICallee<I = I>> {
    /// The function to lower.
    f: &'func Function,

    /// Lowered machine instructions.
    vcode: VCodeBuilder<I, A>,

    /// Mapping from `Value` (SSA value in IR) to virtual register.
    value_regs: SecondaryMap<Value, ValueRegs<VReg>>,

    /// Instruction colors at block exits. From this map, we can recover all
    /// instruction colors by scanning backward from the block end and
    /// decrementing on any color-changing (side-effecting) instruction.
    block_end_colors: SecondaryMap<Block, InstColor>,

    /// Instruction colors at side-effecting ops. This is the *entry* color,
    /// i.e., the version of global state that exists before an instruction
    /// executes.  For each side-effecting instruction, the *exit* color is its
    /// entry color plus one.
    side_effect_inst_entry_colors: FxHashMap<Inst, InstColor>,

    /// Current color as we scan during lowering. While we are lowering an
    /// instruction, this is equal to the color *at entry to* the instruction.
    cur_scan_entry_color: Option<InstColor>,

    /// Current instruction as we scan during lowering.
    cur_inst: Option<Inst>,

    /// Instruction constant values, if known.
    inst_constants: FxHashMap<Inst, u64>,

    /// Use-counts per SSA value, as counted in the input IR.
    value_uses: SecondaryMap<Value, u32>,

    /// Actual uses of each SSA value so far, incremented while lowering.
    value_lowered_uses: SecondaryMap<Value, u32>,

    /// Effectful instructions that have been sunk; they are not codegen'd at
    /// their original locations.
    inst_sunk: FxHashSet<Inst>,

    /// Next virtual register number to allocate.
    next_vreg: u32,

    /// Insts in reverse block order, before final copy to vcode.
    block_insts: Vec<InstTuple<I>>,

    /// Ranges in `block_insts` constituting BBs.
    block_ranges: Vec<(usize, usize)>,

    /// Instructions collected for the BB in progress, in reverse order, with
    /// source-locs attached.
    bb_insts: Vec<InstTuple<I>>,

    /// Instructions collected for the CLIF inst in progress, in forward order.
    ir_insts: Vec<InstTuple<I>>,

    /// The vreg containing the special VmContext parameter, if it is present in the current
    /// function's signature.
    vm_context: Option<Reg>,
}

/// Notion of "relocation distance". This gives an estimate of how far away a symbol will be from a
/// reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocDistance {
    /// Target of relocation is "nearby". The threshold for this is fuzzy but should be interpreted
    /// as approximately "within the compiled output of one module"; e.g., within AArch64's +/-
    /// 128MB offset. If unsure, use `Far` instead.
    Near,
    /// Target of relocation could be anywhere in the address space.
    Far,
}

fn alloc_vregs<I: VCodeInst>(
    ty: Type,
    next_vreg: &mut u32,
    vcode: &mut VCodeBuilder<I>,
) -> CodegenResult<ValueRegs<Reg>> {
    let v = *next_vreg;
    let (regclasses, tys) = I::rc_for_type(ty)?;
    *next_vreg += regclasses.len() as u32;
    let regs = match regclasses {
        &[rc0] => ValueRegs::one(VReg::new(v, rc0)),
        &[rc0, rc1] => ValueRegs::two(VReg::new(v, rc0), VReg::new(v + 1, rc1)),
        #[cfg(feature = "arm32")]
        &[rc0, rc1, rc2, rc3] => ValueRegs::four(
            VReg::new(v, rc0),
            VReg::new(v + 1, rc1),
            VReg::new(v + 2, rc2),
            VReg::new(v + 3, rc3),
        ),
        _ => panic!("Value must reside in 1, 2 or 4 registers"),
    };
    for (&reg_ty, &reg) in tys.iter().zip(regs.regs().iter()) {
        vcode.set_vreg_type(reg.to_virtual_reg(), reg_ty);
    }
    Ok(regs)
}

enum ReturnFallthrough {
    Yes,
    No,
}

impl<'func, I: VCodeInst, A: ABICallee<I = I>> Lower<'func, I, A> {
    /// Prepare a new lowering context for the given IR function.
    pub fn new(
        f: &'func Function,
        emit_info: I::Info,
        block_order: BlockLoweringOrder,
    ) -> CodegenResult<Lower<'func, I, A>> {
        let constants = VCodeConstants::with_capacity(f.dfg.constants.len());
        let abi = A::create();
        let mut vcode = VCodeBuilder::new(abi, emit_info, block_order, constants);

        let mut next_vreg: u32 = 0;

        let mut value_regs = SecondaryMap::with_default(ValueRegs::invalid());

        // Assign a vreg to each block param and each inst result.
        for bb in f.layout.blocks() {
            for &param in f.dfg.block_params(bb) {
                let ty = f.dfg.value_type(param);
                if value_regs[param].is_invalid() {
                    let regs = alloc_vregs(ty, &mut next_vreg, &mut vcode)?;
                    value_regs[param] = regs;
                    debug!("bb {} param {}: regs {:?}", bb, param, regs);
                }
            }
            for inst in f.layout.block_insts(bb) {
                for &result in f.dfg.inst_results(inst) {
                    let ty = f.dfg.value_type(result);
                    if value_regs[result].is_invalid() {
                        let regs = alloc_vregs(ty, &mut next_vreg, &mut vcode)?;
                        value_regs[result] = regs;
                        debug!(
                            "bb {} inst {} ({:?}): result regs {:?}",
                            bb, inst, f.dfg[inst], regs,
                        );
                    }
                }
            }
        }

        let vm_context = vcode
            .abi()
            .signature()
            .special_param_index(ArgumentPurpose::VMContext)
            .map(|vm_context_index| {
                let entry_block = f.layout.entry_block().unwrap();
                let param = f.dfg.block_params(entry_block)[vm_context_index];
                value_regs[param].only_reg().unwrap()
            });

        // Compute instruction colors, find constant instructions, and find instructions with
        // side-effects, in one combined pass.
        let mut cur_color = 0;
        let mut block_end_colors = SecondaryMap::with_default(InstColor::new(0));
        let mut side_effect_inst_entry_colors = FxHashMap::default();
        let mut inst_constants = FxHashMap::default();
        let mut value_uses = SecondaryMap::with_default(0);
        for bb in f.layout.blocks() {
            cur_color += 1;
            for inst in f.layout.block_insts(bb) {
                let side_effect = has_lowering_side_effect(f, inst);

                debug!("bb {} inst {} has color {}", bb, inst, cur_color);
                if side_effect {
                    side_effect_inst_entry_colors.insert(inst, InstColor::new(cur_color));
                    debug!(" -> side-effecting; incrementing color for next inst");
                    cur_color += 1;
                }

                // Determine if this is a constant; if so, add to the table.
                if let Some(c) = is_constant_64bit(f, inst) {
                    debug!(" -> constant: {}", c);
                    inst_constants.insert(inst, c);
                }

                // Count uses of all arguments.
                for arg in f.dfg.inst_args(inst) {
                    let arg = f.dfg.resolve_aliases(*arg);
                    value_uses[arg] += 1;
                }
            }

            block_end_colors[bb] = InstColor::new(cur_color);
        }

        Ok(Lower {
            f,
            vcode,
            value_regs,
            block_end_colors,
            side_effect_inst_entry_colors,
            inst_constants,
            next_vreg,
            value_uses,
            value_lowered_uses: SecondaryMap::default(),
            inst_sunk: FxHashSet::default(),
            cur_scan_entry_color: None,
            cur_inst: None,
            block_insts: vec![],
            block_ranges: vec![],
            bb_insts: vec![],
            ir_insts: vec![],
            vm_context,
        })
    }

    fn gen_prologue(&mut self) {
        if let Some(entry_bb) = self.f.layout.entry_block() {
            debug!(
                "gen_arg_setup: entry BB {} args are:\n{:?}",
                entry_bb,
                self.f.dfg.block_params(entry_bb)
            );
            // Collect arg vregs.
            let arg_regs: Vec<Writable<VReg>> = self
                .f
                .dfg
                .block_params(entry_bb)
                .iter()
                .map(|param| writable_value_regs(self.value_regs[*param]))
                .collect();
            // Generate whatever sequence is necessary to
            for inst in self.vcode.abi().gen_arg_seq(arg_regs) {
                self.emit(inst);
            }
        }
    }

    fn gen_epilogue(&mut self, inst: Inst) {
        // Hack: to keep `vmctx` alive, if it exists, we emit a value label here
        // for it if debug info is requested. This ensures that it exists either
        // in a register or spillslot throughout the entire function body, and
        // allows for a better debugging experience.
        if let Some(vmctx_val) = self.f.special_param(ArgumentPurpose::VMContext) {
            self.emit_value_label_marks_for_value(vmctx_val);
        }

        // Collect return values.
        let mut retvals: SmallVec<[ValueRegs<Reg>; 4]> = smallvec![];
        for val in self.f.dfg.inst_args(inst) {
            let regs = self.put_value_in_regs(val);
            retvals.push(regs);
        }

        // Call ABI impl to emit a return-value sequence; this copies
        // values and/or adds constraint pseudoinsts as needed.
        for inst in self.vcode.abi().gen_ret_seq(retvals) {
            self.emit(inst);
        }

        // Add a return instruction; this is converted to a true
        // epilogue post-regalloc.
        let ret = I::gen_ret();
        self.emit(ret);
    }

    /// Has this instruction been sunk to a use-site (i.e., away from its
    /// original location)?
    fn is_inst_sunk(&self, inst: Inst) -> bool {
        self.inst_sunk.contains(&inst)
    }

    // Is any result of this instruction needed?
    fn is_any_inst_result_needed(&self, inst: Inst) -> bool {
        self.f
            .dfg
            .inst_results(inst)
            .iter()
            .any(|&result| self.value_lowered_uses[result] > 0)
    }

    fn lower_clif_block<B: LowerBackend<MInst = I>>(
        &mut self,
        backend: &B,
        block: Block,
    ) -> CodegenResult<()> {
        self.cur_scan_entry_color = Some(self.block_end_colors[block]);
        // Lowering loop:
        // - For each non-branch instruction, in reverse order:
        //   - If side-effecting (load, store, branch/call/return, possible trap), or if
        //     used outside of this block, or if demanded by another inst, then lower.
        //
        // That's it! Lowering of side-effecting ops will force all *needed*
        // (live) non-side-effecting ops to be lowered at the right places, via
        // the `use_input_reg()` callback on the `LowerCtx` (that's us). That's
        // because `use_input_reg()` sets the eager/demand bit for any insts
        // whose result registers are used.
        //
        // We build up the BB in reverse instruction order in `bb_insts`.
        // Because the machine backend calls `ctx.emit()` in forward order, we
        // collect per-IR-inst lowered instructions in `ir_insts`, then reverse
        // these and append to `bb_insts` as we go backward through the block.
        // `bb_insts` are then reversed again and appended to the VCode at the
        // end of the BB (in the toplevel driver `lower()`).
        for inst in self.f.layout.block_insts(block).rev() {
            let data = &self.f.dfg[inst];
            let has_side_effect = has_lowering_side_effect(self.f, inst);
            // If  inst has been sunk to another location, skip it.
            if self.is_inst_sunk(inst) {
                continue;
            }
            // Are any outputs used at least once?
            let value_needed = self.is_any_inst_result_needed(inst);
            debug!(
                "lower_clif_block: block {} inst {} ({:?}) is_branch {} side_effect {} value_needed {}",
                block,
                inst,
                data,
                data.opcode().is_branch(),
                has_side_effect,
                value_needed,
            );

            // Update scan state to color prior to this inst (as we are scanning
            // backward).
            self.cur_inst = Some(inst);
            if has_side_effect {
                let entry_color = *self
                    .side_effect_inst_entry_colors
                    .get(&inst)
                    .expect("every side-effecting inst should have a color-map entry");
                self.cur_scan_entry_color = Some(entry_color);
            }

            // Skip lowering branches; these are handled separately
            // (see `lower_clif_branches()` below).
            if self.f.dfg[inst].opcode().is_branch() {
                continue;
            }

            // Normal instruction: codegen if the instruction is side-effecting
            // or any of its outputs its used.
            if has_side_effect || value_needed {
                debug!("lowering: inst {}: {:?}", inst, self.f.dfg[inst]);
                backend.lower(self, inst)?;
                // Emit value-label markers if needed, to later recover debug
                // mappings.
                self.emit_value_label_markers_for_inst(inst);
            }
            if data.opcode().is_return() {
                // Return: handle specially, using ABI-appropriate sequence.
                let _is_fallthrough = if data.opcode() == Opcode::Return {
                    false
                } else {
                    panic!("FallthroughReturn not yet implemented with new-regalloc integration");
                };
                self.gen_epilogue(inst);
            }

            let loc = self.srcloc(inst);
            self.finish_ir_inst(loc);
        }
        self.cur_scan_entry_color = None;
        Ok(())
    }

    fn get_value_labels<'a>(&'a self, val: Value, depth: usize) -> Option<&'a [ValueLabelStart]> {
        if let Some(ref values_labels) = self.f.dfg.values_labels {
            debug!(
                "get_value_labels: val {} -> {} -> {:?}",
                val,
                self.f.dfg.resolve_aliases(val),
                values_labels.get(&self.f.dfg.resolve_aliases(val))
            );
            let val = self.f.dfg.resolve_aliases(val);
            match values_labels.get(&val) {
                Some(&ValueLabelAssignments::Starts(ref list)) => Some(&list[..]),
                Some(&ValueLabelAssignments::Alias { value, .. }) if depth < 10 => {
                    self.get_value_labels(value, depth + 1)
                }
                _ => None,
            }
        } else {
            None
        }
    }

    fn emit_value_label_marks_for_value(&mut self, val: Value) {
        let mut markers: SmallVec<[I; 4]> = smallvec![];
        let regs = self.value_regs[val];
        if regs.len() > 1 {
            return;
        }
        let reg = regs.only_reg().unwrap();

        if let Some(label_starts) = self.get_value_labels(val, 0) {
            let labels = label_starts
                .iter()
                .map(|&ValueLabelStart { label, .. }| label)
                .collect::<FxHashSet<_>>();
            for label in labels {
                debug!(
                    "value labeling: defines val {:?} -> reg {:?} -> label {:?}",
                    val, reg, label,
                );
                markers.push(I::gen_value_label_marker(label, reg));
            }
        }
        for marker in markers {
            self.emit(marker);
        }
    }

    fn emit_value_label_markers_for_inst(&mut self, inst: Inst) {
        if self.f.dfg.values_labels.is_none() {
            return;
        }

        debug!(
            "value labeling: srcloc {}: inst {}",
            self.srcloc(inst),
            inst
        );
        for &val in self.f.dfg.inst_results(inst) {
            self.emit_value_label_marks_for_value(val);
        }
    }

    fn emit_value_label_markers_for_block_args(&mut self, block: Block) {
        if self.f.dfg.values_labels.is_none() {
            return;
        }

        debug!("value labeling: block {}", block);
        for &arg in self.f.dfg.block_params(block) {
            self.emit_value_label_marks_for_value(arg);
        }
        self.finish_ir_inst(SourceLoc::default());
    }

    fn finish_ir_inst(&mut self, loc: SourceLoc) {
        // `bb_insts` is kept in reverse order, so emit the instructions in
        // reverse order.
        for mut tuple in self.ir_insts.drain(..).rev() {
            tuple.loc = loc;
            self.bb_insts.push(tuple);
        }
    }

    fn finish_bb(&mut self) {
        let start = self.block_insts.len();
        for tuple in self.bb_insts.drain(..).rev() {
            self.block_insts.push(tuple);
        }
        let end = self.block_insts.len();
        self.block_ranges.push((start, end));
    }

    fn copy_bbs_to_vcode(&mut self) {
        for &(start, end) in self.block_ranges.iter().rev() {
            for &InstTuple {
                loc,
                is_safepoint,
                ref inst,
            } in &self.block_insts[start..end]
            {
                self.vcode.set_srcloc(loc);
                self.vcode.push(inst.clone(), is_safepoint);
            }
            self.vcode.end_bb();
        }
    }

    fn lower_clif_branches<B: LowerBackend<MInst = I>>(
        &mut self,
        backend: &B,
        block: Block,
        branches: &SmallVec<[Inst; 2]>,
        targets: &SmallVec<[MachLabel; 2]>,
    ) -> CodegenResult<()> {
        debug!(
            "lower_clif_branches: block {} branches {:?} targets {:?}",
            block, branches, targets,
        );
        // When considering code-motion opportunities, consider the current
        // program point to be the first branch.
        self.cur_inst = Some(branches[0]);
        backend.lower_branch_group(self, branches, targets)?;
        let loc = self.srcloc(branches[0]);
        self.finish_ir_inst(loc);
        Ok(())
    }

    fn collect_branches_and_targets(
        &self,
        bindex: BlockIndex,
        _bb: Block,
        branches: &mut SmallVec<[Inst; 2]>,
        targets: &mut SmallVec<[MachLabel; 2]>,
    ) {
        branches.clear();
        targets.clear();
        let mut last_inst = None;
        for &(inst, succ) in self.vcode.block_order().succ_indices(bindex) {
            // Avoid duplicates: this ensures a br_table is only inserted once.
            if last_inst != Some(inst) {
                branches.push(inst);
            } else {
                debug_assert!(self.f.dfg[inst].opcode() == Opcode::BrTable);
                debug_assert!(branches.len() == 1);
            }
            last_inst = Some(inst);
            targets.push(MachLabel::from_block(succ));
        }
    }

    /// Lower the function.
    pub fn lower<B: LowerBackend<MInst = I>>(mut self, backend: &B) -> CodegenResult<VCode<I>> {
        debug!("about to lower function: {:?}", self.f);

        // Initialize the ABI object.
        self.vcode.abi().init(&mut self);

        self.vcode.set_entry(0);

        // Reused vectors for branch lowering.
        let mut branches: SmallVec<[Inst; 2]> = SmallVec::new();
        let mut targets: SmallVec<[MachLabel; 2]> = SmallVec::new();

        // get a copy of the lowered order; we hold this separately because we
        // need a mut ref to the vcode to mutate it below.
        let lowered_order: SmallVec<[LoweredBlock; 64]> = self
            .vcode
            .block_order()
            .lowered_order()
            .iter()
            .cloned()
            .collect();

        // Add block params for each block in lowered order.
        for (bindex, lb) in lowered_order.iter().enumerate() {
            let params: SmallVec<[VReg; 16]> = smallvec![];
            if let Some(bb) = lb.orig_block() {
                for blockparam in self.f.dfg.block_params(bb) {
                    for vreg in self.value_regs[blockparam].regs() {
                        params.push(vreg);
                    }
                }
            } else {
                // This is an edge block only: it needs to carry the
                // block params of the target. The succesor *must*
                // have an orig block if we do not.
                let param_block = if let Some((_, _, b)) = lb.in_edge() {
                    assert!(lb.out_edge().is_none());
                    b
                } else if let Some((_, _, b)) = lb.out_edge() {
                    assert!(lb.in_edge().is_none());
                    b
                } else {
                    panic!("Empty lowered block: {}", bindex);
                };
                for blockparam in self.f.dfg.block_params(param_block) {
                    for vreg in self.value_regs[blockparam].regs() {
                        params.push(vreg);
                    }
                }
            }
            self.vcode.add_blockparams(&params[..]);
        }

        // Main lowering loop over lowered blocks.
        for (bindex, lb) in lowered_order.iter().enumerate().rev() {
            let bindex = bindex as BlockIndex;

            // Lower the block body in reverse order (see comment in
            // `lower_clif_block()` for rationale).

            // End branches.
            if let Some(bb) = lb.orig_block() {
                self.collect_branches_and_targets(bindex, bb, &mut branches, &mut targets);
                if branches.len() > 0 {
                    self.lower_clif_branches(backend, bb, &branches, &targets)?;
                    self.finish_ir_inst(self.srcloc(branches[0]));
                }
            } else {
                // If no orig block, this must be a pure edge block;
                // get the successor and emit a jump.
                let (_, succ) = self.vcode.block_order().succ_indices(bindex)[0];
                // Get the params too.
                let orig_branch = if let Some((_, br, _)) = lb.in_edge() {
                    br
                } else if let Some((_, br, _)) = lb.out_edge() {
                    br
                } else {
                    panic!("Empty lowered block: {}", bindex);
                };
                let mut param_operands = vec![];
                for param in self.f.dfg.inst_variable_args(orig_branch) {
                    let vregs = self.put_value_in_regs(param);
                    for &vreg in vregs.regs() {
                        param_operands.push(Reg::reg_use(vreg));
                    }
                }
                self.emit(I::gen_jump(MachLabel::from_block(succ), param_operands));
                self.finish_ir_inst(SourceLoc::default());
            }

            // If there is an original block body (not just an edge
            // block), then lower it.
            if let Some(bb) = lb.orig_block() {
                self.lower_clif_block(backend, bb)?;
                self.emit_value_label_markers_for_block_args(bb);
            }

            if bindex == 0 {
                // Set up the function with arg vreg inits.
                self.gen_prologue();
                self.finish_ir_inst(SourceLoc::default());
            }

            self.finish_bb();
        }

        self.copy_bbs_to_vcode();

        // Now that we've emitted all instructions into the VCodeBuilder, let's build the VCode.
        let (vcode, stack_map_info) = self.vcode.build();
        debug!("built vcode: {:?}", vcode);

        Ok((vcode, stack_map_info))
    }

    fn put_value_in_regs(&mut self, val: Value) -> ValueRegs<VReg> {
        debug!("put_value_in_reg: val {}", val);
        let mut regs = self.value_regs[val];
        debug!(" -> regs {:?}", regs);
        assert!(regs.is_valid());

        self.value_lowered_uses[val] += 1;

        regs
    }

    /// Get the actual inputs for a value. This is the implementation for
    /// `get_input()` but starting from the SSA value, which is not exposed to
    /// the backend.
    fn get_value_as_source_or_const(&self, val: Value) -> NonRegInput {
        debug!(
            "get_input_for_val: val {} at cur_inst {:?} cur_scan_entry_color {:?}",
            val, self.cur_inst, self.cur_scan_entry_color,
        );
        let inst = match self.f.dfg.value_def(val) {
            // OK to merge source instruction if (i) we have a source
            // instruction, and:
            // - It has no side-effects, OR
            // - It has a side-effect, has one output value, that one output has
            //   only one use (this one), and the instruction's color is *one less
            //   than* the current scan color.
            //
            //   This latter set of conditions is testing whether a
            //   side-effecting instruction can sink to the current scan
            //   location; this is possible if the in-color of this inst is
            //   equal to the out-color of the producing inst, so no other
            //   side-effecting ops occur between them (which will only be true
            //   if they are in the same BB, because color increments at each BB
            //   start).
            //
            //   If it is actually sunk, then in `merge_inst()`, we update the
            //   scan color so that as we scan over the range past which the
            //   instruction was sunk, we allow other instructions (that came
            //   prior to the sunk instruction) to sink.
            ValueDef::Result(src_inst, result_idx) => {
                let src_side_effect = has_lowering_side_effect(self.f, src_inst);
                debug!(" -> src inst {}", src_inst);
                debug!(" -> has lowering side effect: {}", src_side_effect);
                if !src_side_effect {
                    // Pure instruction: always possible to sink.
                    Some((src_inst, result_idx))
                } else {
                    // Side-effect: test whether this is the only use of the
                    // only result of the instruction, and whether colors allow
                    // the code-motion.
                    if self.cur_scan_entry_color.is_some()
                        && self.value_uses[val] == 1
                        && self.value_lowered_uses[val] == 0
                        && self.num_outputs(src_inst) == 1
                        && self
                            .side_effect_inst_entry_colors
                            .get(&src_inst)
                            .unwrap()
                            .get()
                            + 1
                            == self.cur_scan_entry_color.unwrap().get()
                    {
                        Some((src_inst, 0))
                    } else {
                        None
                    }
                }
            }
            _ => None,
        };
        let constant = inst.and_then(|(inst, _)| self.get_constant(inst));

        NonRegInput { inst, constant }
    }

    fn abi(&mut self) -> &mut dyn ABICallee<I = I> {
        self.vcode.abi()
    }
}

impl<'func, I: VCodeInst> LowerCtx for Lower<'func, I> {
    type I = I;

    fn data(&self, ir_inst: Inst) -> &InstructionData {
        &self.f.dfg[ir_inst]
    }

    fn ty(&self, ir_inst: Inst) -> Type {
        self.f.dfg.ctrl_typevar(ir_inst)
    }

    fn call_target<'b>(&'b self, ir_inst: Inst) -> Option<(&'b ExternalName, RelocDistance)> {
        match &self.f.dfg[ir_inst] {
            &InstructionData::Call { func_ref, .. }
            | &InstructionData::FuncAddr { func_ref, .. } => {
                let funcdata = &self.f.dfg.ext_funcs[func_ref];
                let dist = funcdata.reloc_distance();
                Some((&funcdata.name, dist))
            }
            _ => None,
        }
    }

    fn call_sig<'b>(&'b self, ir_inst: Inst) -> Option<&'b Signature> {
        match &self.f.dfg[ir_inst] {
            &InstructionData::Call { func_ref, .. } => {
                let funcdata = &self.f.dfg.ext_funcs[func_ref];
                Some(&self.f.dfg.signatures[funcdata.signature])
            }
            &InstructionData::CallIndirect { sig_ref, .. } => Some(&self.f.dfg.signatures[sig_ref]),
            _ => None,
        }
    }

    fn symbol_value<'b>(&'b self, ir_inst: Inst) -> Option<(&'b ExternalName, RelocDistance, i64)> {
        match &self.f.dfg[ir_inst] {
            &InstructionData::UnaryGlobalValue { global_value, .. } => {
                let gvdata = &self.f.global_values[global_value];
                match gvdata {
                    &GlobalValueData::Symbol {
                        ref name,
                        ref offset,
                        ..
                    } => {
                        let offset = offset.bits();
                        let dist = gvdata.maybe_reloc_distance().unwrap();
                        Some((name, dist, offset))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn memflags(&self, ir_inst: Inst) -> Option<MemFlags> {
        match &self.f.dfg[ir_inst] {
            &InstructionData::AtomicCas { flags, .. } => Some(flags),
            &InstructionData::AtomicRmw { flags, .. } => Some(flags),
            &InstructionData::Load { flags, .. }
            | &InstructionData::LoadComplex { flags, .. }
            | &InstructionData::LoadNoOffset { flags, .. }
            | &InstructionData::Store { flags, .. }
            | &InstructionData::StoreComplex { flags, .. } => Some(flags),
            &InstructionData::StoreNoOffset { flags, .. } => Some(flags),
            _ => None,
        }
    }

    fn srcloc(&self, ir_inst: Inst) -> SourceLoc {
        self.f.srclocs[ir_inst]
    }

    fn num_inputs(&self, ir_inst: Inst) -> usize {
        self.f.dfg.inst_args(ir_inst).len()
    }

    fn num_outputs(&self, ir_inst: Inst) -> usize {
        self.f.dfg.inst_results(ir_inst).len()
    }

    fn input_ty(&self, ir_inst: Inst, idx: usize) -> Type {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        let val = self.f.dfg.resolve_aliases(val);
        self.f.dfg.value_type(val)
    }

    fn output_ty(&self, ir_inst: Inst, idx: usize) -> Type {
        self.f.dfg.value_type(self.f.dfg.inst_results(ir_inst)[idx])
    }

    fn get_constant(&self, ir_inst: Inst) -> Option<u64> {
        self.inst_constants.get(&ir_inst).cloned()
    }

    fn get_input_as_source_or_const(&self, ir_inst: Inst, idx: usize) -> NonRegInput {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        let val = self.f.dfg.resolve_aliases(val);
        self.get_value_as_source_or_const(val)
    }

    fn input_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Reg> {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        let val = self.f.dfg.resolve_aliases(val);
        let vregs = self.put_value_in_vregs(val);
        vregs.map(|v| Reg::reg_use(v))
    }

    fn input_at_end_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Reg> {
        let val = self.f.dfg.inst_args(ir_inst)[idx];
        let val = self.f.dfg.resolve_aliases(val);
        let vregs = self.put_value_in_vregs(val);
        vregs.map(|v| Reg::reg_use_at_end(v))
    }

    fn output_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Writable<Reg>> {
        let val = self.f.dfg.inst_results(ir_inst)[idx];
        let vregs = self.value_regs[val];
        vregs.map(|v| Writable::from_reg(Operand::reg_def(v)))
    }

    fn output_at_start_regs(&mut self, ir_inst: Inst, idx: usize) -> ValueRegs<Writable<Reg>> {
        let val = self.f.dfg.inst_results(ir_inst)[idx];
        let vregs = self.value_regs[val];
        vregs.map(|v| Writable::from_reg(Operand::reg_def(v)))
    }

    fn reused_input_output_regs(
        &mut self,
        ir_inst: Inst,
        input_idx: usize,
        output_idx: usize,
    ) -> (Reg, Writable<Reg>) {
        let input_val = self.f.dfg.inst_args(ir_inst)[input_idx];
        let input_val = self.f.dfg.resolve_aliases(input_val);
        let input_vregs = self.put_value_in_vregs(input_val);
        let input_vreg = input_vregs
            .only_reg()
            .expect("reuse semantics not well-defined for multi-reg case");
        let output_val = self.f.dfg.inst_results(ir_inst)[output_idx];
        let output_val = self.f.dfg.resolve_aliases(output_val);
        let output_vregs = self.value_regs[output_val];
        let output_vreg = output_vregs
            .only_reg()
            .expect("reuse semantics not well-defined for multi-reg case");
        (
            Reg::reg_use(input_vreg),
            Writable::from_reg(Reg::reg_reuse_def(output_vreg, input_idx)),
        )
    }

    fn alloc_tmp(&mut self, ty: Type) -> ValueRegs<Writable<Reg>> {
        let vregs = alloc_vregs(ty, &mut self.next_vreg, &mut self.vcode).unwrap();
        vregs.map(|v| Writable::from_reg(Reg::reg_temp(v)))
    }

    fn emit(&mut self, mach_inst: I) {
        self.ir_insts.push(InstTuple {
            loc: SourceLoc::default(),
            is_safepoint: false,
            inst: mach_inst,
        });
    }

    fn emit_safepoint(&mut self, mach_inst: I) {
        self.ir_insts.push(InstTuple {
            loc: SourceLoc::default(),
            is_safepoint: true,
            inst: mach_inst,
        });
    }

    fn sink_inst(&mut self, ir_inst: Inst) {
        assert!(has_lowering_side_effect(self.f, ir_inst));
        assert!(self.cur_scan_entry_color.is_some());

        let sunk_inst_entry_color = self
            .side_effect_inst_entry_colors
            .get(&ir_inst)
            .cloned()
            .unwrap();
        let sunk_inst_exit_color = InstColor::new(sunk_inst_entry_color.get() + 1);
        assert!(sunk_inst_exit_color == self.cur_scan_entry_color.unwrap());
        self.cur_scan_entry_color = Some(sunk_inst_entry_color);
        self.inst_sunk.insert(ir_inst);
    }

    fn get_constant_data(&self, constant_handle: Constant) -> &ConstantData {
        self.f.dfg.constants.get(constant_handle)
    }

    fn use_constant(&mut self, constant: VCodeConstantData) -> VCodeConstant {
        self.vcode.constants().insert(constant)
    }

    fn get_immediate(&self, ir_inst: Inst) -> Option<DataValue> {
        let inst_data = self.data(ir_inst);
        match inst_data {
            InstructionData::Shuffle { mask, .. } => {
                let buffer = self.f.dfg.immediates.get(mask.clone()).unwrap().as_slice();
                let value = DataValue::V128(buffer.try_into().expect("a 16-byte data buffer"));
                Some(value)
            }
            InstructionData::UnaryConst {
                constant_handle, ..
            } => {
                let buffer = self.f.dfg.constants.get(constant_handle.clone()).as_slice();
                let value = DataValue::V128(buffer.try_into().expect("a 16-byte data buffer"));
                Some(value)
            }
            _ => inst_data.imm_value(),
        }
    }
}

/// Visit all successors of a block with a given visitor closure.
pub(crate) fn visit_block_succs<F: FnMut(Inst, Block)>(f: &Function, block: Block, mut visit: F) {
    for inst in f.layout.block_likely_branches(block) {
        if f.dfg[inst].opcode().is_branch() {
            visit_branch_targets(f, block, inst, &mut visit);
        }
    }
}

fn visit_branch_targets<F: FnMut(Inst, Block)>(
    f: &Function,
    block: Block,
    inst: Inst,
    visit: &mut F,
) {
    if f.dfg[inst].opcode() == Opcode::Fallthrough {
        visit(inst, f.layout.next_block(block).unwrap());
    } else {
        match f.dfg[inst].analyze_branch(&f.dfg.value_lists) {
            BranchInfo::NotABranch => {}
            BranchInfo::SingleDest(dest, _) => {
                visit(inst, dest);
            }
            BranchInfo::Table(table, maybe_dest) => {
                if let Some(dest) = maybe_dest {
                    visit(inst, dest);
                }
                for &dest in f.jump_tables[table].as_slice() {
                    visit(inst, dest);
                }
            }
        }
    }
}
