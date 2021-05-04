//! This implements the VCode container: a CFG of Insts that have been lowered.
//!
//! VCode is virtual-register code. An instruction in VCode is almost a machine
//! instruction; however, its register slots can refer to virtual registers in
//! addition to real machine registers.
//!
//! TODO: update description to refer to SSA
//!
//! VCode is structured with traditional basic blocks, and
//! each block must be terminated by an unconditional branch (one target), a
//! conditional branch (two targets), or a return (no targets). Note that this
//! slightly differs from the machine code of most ISAs: in most ISAs, a
//! conditional branch has one target (and the not-taken case falls through).
//! However, we expect that machine backends will elide branches to the following
//! block (i.e., zero-offset jumps), and will be able to codegen a branch-cond /
//! branch-uncond pair if *both* targets are not fallthrough. This allows us to
//! play with layout prior to final binary emission, as well, if we want.
//!
//! See the main module comment in `mod.rs` for more details on the VCode-based
//! backend pipeline.

use super::Reg;
use crate::ir::{self, types, Constant, ConstantData, SourceLoc};
use crate::machinst::*;
use crate::timing;
use cranelift_entity::{entity_impl, Keys, PrimaryMap};
use regalloc2::{Operand, OperandKind, PReg, VReg};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

/// Index referring to an instruction in VCode.
pub type InsnIndex = u32;
/// Index referring to a basic block in VCode.
pub type BlockIndex = u32;
/// Range of an instructions in VCode.
pub type InsnRange = core::ops::Range<InsnIndex>;

/// VCodeInst wraps all requirements for a MachInst to be in VCode: it must be
/// a `MachInst` and it must be able to emit machine code.
pub trait VCodeInst: MachInst + MachInstEmit {}
impl<I: MachInst + MachInstEmit> VCodeInst for I {}

/// A function in "VCode" (virtual-register code) form, after
/// lowering.  This is essentially a standard CFG of basic blocks,
/// where each basic block consists of lowered instructions produced
/// by the machine-specific backend.
///
/// Note that VCode is still SSA; its main difference from CLIF is
/// that it is a *close-to-machine-code* representation, where
/// instruction selection has made decisions specific to our target
/// architecture to match its ISA.
pub struct VCode<I: VCodeInst, A: ABICallee<I = I>> {
    /// Callee-side ABI implementation.
    ///
    /// The VCode owns the ABI object because it needs to be able to
    /// generate frame-specific code, such as spill and reload code,
    /// on its own as it implements the regalloc's trait
    /// interface. Otherwise, it would have made more sense to keep
    /// this in the Lower context, because the ABI implementation is
    /// really logically part of the lowering process, not the final
    /// code representation.
    ///
    /// Note that we've statically parameterized the VCode type on the
    /// particular ABI implementation so that we do not have any
    /// dynamic dispatch here. In practice, there is only one ABI
    /// trait implementation per architecture, so this doesn't really
    /// add any code-size overhead.
    abi: A,

    /// VReg IR-level types.
    vreg_types: Vec<Type>,

    /// Which vregs are ref-typed?
    reftype_vregs: Vec<VReg>,

    /// Lowered machine instructions in order corresponding to the original IR.
    insts: Vec<I>,

    /// Operands extracted from `insts`.
    operands: Vec<Operand>,

    /// Operand ranges for each instruction.
    operand_ranges: Vec<(usize, usize)>,

    /// Source locations for each instruction. (`SourceLoc` is a `u32`, so it is
    /// reasonable to keep one of these per instruction.)
    srclocs: Vec<SourceLoc>,

    /// Entry block.
    entry: BlockIndex,

    /// Block instruction indices.
    block_ranges: Vec<(InsnIndex, InsnIndex)>,

    /// Block successors: index range in the successor-list below.
    block_succ_range: Vec<(usize, usize)>,

    /// Block successor lists, concatenated into one Vec. The `block_succ_range`
    /// list of tuples above gives (start, end) ranges within this list that
    /// correspond to each basic block's successors.
    block_succs: Vec<regalloc2::Block>,

    /// Block predecessors: index range in the predecessor-list below
    block_pred_range: Vec<(usize, usize)>,

    /// Block predecessor lists, concatenated into one Vec. The
    /// `block_succ_range` list of tuples above gives (start, end)
    /// ranges within this list that correspond to each basic block's
    /// successors.
    block_preds: Vec<regalloc2::Block>,

    /// Block parameters, stored contiguously.
    block_params: Vec<VReg>,

    /// Index ranges in block-param list, indexed by BB.
    block_param_range: Vec<(usize, usize)>,

    /// Block-order information.
    block_order: BlockLoweringOrder,

    /// Constant information used during code emission. This should be
    /// immutable across function compilations within the same module.
    emit_info: I::Info,

    /// A list of SpillSlots at instruction indices corresponding to
    /// safepoints that contain references.
    safepoint_slots: Vec<(InsnIndex, SpillSlot)>,

    /// Do we generate debug info?
    generate_debug_info: bool,

    /// Instruction end offsets, instruction indices at each label, and total
    /// buffer size.  Only present if `generate_debug_info` is set.
    insts_layout: RefCell<(Vec<u32>, Vec<u32>, u32)>,

    /// Constants.
    constants: VCodeConstants,

    /// Are any debug value-labels present? If not, we can skip the
    /// post-emission analysis.
    has_value_labels: bool,
}

/// A builder for a VCode function body. This builder is designed for the
/// lowering approach that we take: we traverse basic blocks in forward
/// (original IR) order, but within each basic block, we generate code from
/// bottom to top; and within each IR instruction that we visit in this reverse
/// order, we emit machine instructions in *forward* order again.
///
/// Hence, to produce the final instructions in proper order, we perform two
/// swaps.  First, the machine instructions (`I` instances) are produced in
/// forward order for an individual IR instruction. Then these are *reversed*
/// and concatenated to `bb_insns` at the end of the IR instruction lowering.
/// The `bb_insns` vec will thus contain all machine instructions for a basic
/// block, in reverse order. Finally, when we're done with a basic block, we
/// reverse the whole block's vec of instructions again, and concatenate onto
/// the VCode's insts.
pub struct VCodeBuilder<I: VCodeInst, A: ABICallee<I = I>> {
    /// In-progress VCode.
    vcode: VCode<I, A>,

    /// Index of the last block-start in the vcode.
    block_start: InsnIndex,

    /// Start of succs for the current block in the concatenated succs list.
    succ_start: usize,

    /// Current source location.
    cur_srcloc: SourceLoc,
}

impl<I: VCodeInst, A: ABICallee<I = I>> VCodeBuilder<I, A> {
    /// Create a new VCodeBuilder.
    pub fn new(
        abi: A,
        emit_info: I::Info,
        block_order: BlockLoweringOrder,
        constants: VCodeConstants,
    ) -> VCodeBuilder<I, A> {
        let vcode = VCode::new(
            abi,
            emit_info,
            block_order,
            constants,
            /* generate_debug_info = */ true,
        );

        VCodeBuilder {
            vcode,
            block_start: 0,
            succ_start: 0,
            cur_srcloc: SourceLoc::default(),
        }
    }

    /// Access to the BlockLoweringOrder object.
    pub fn block_order(&self) -> &BlockLoweringOrder {
        &self.vcode.block_order
    }

    /// Set the type of a VReg.
    pub fn set_vreg_type(&mut self, vreg: VReg, ty: Type) {
        if self.vcode.vreg_types.len() <= vreg.vreg() {
            self.vcode.vreg_types.resize(vreg.vreg() + 1, ir::types::I8);
        }
        self.vcode.vreg_types[vreg.vreg()] = ty;
        if is_reftype(ty) {
            self.vcode.reftype_vregs.push(vreg);
        }
    }

    /// Are there any reference-typed values at all among the vregs?
    pub fn have_ref_values(&self) -> bool {
        self.vcode.reftype_vregs.len() > 0
    }

    /// Set the current block as the entry block.
    pub fn set_entry(&mut self, block: BlockIndex) {
        self.vcode.entry = block;
    }

    /// End the current basic block. Must be called after emitting vcode insts
    /// for IR insts and prior to ending the function (building the VCode).
    pub fn end_bb(&mut self) {
        let start_idx = self.block_start;
        let end_idx = self.vcode.insts.len() as InsnIndex;
        self.block_start = end_idx;
        // Add the instruction index range to the list of blocks.
        self.vcode.block_ranges.push((start_idx, end_idx));
        // End the successors list.
        let succ_end = self.vcode.block_succs.len();
        self.vcode
            .block_succ_range
            .push((self.succ_start, succ_end));
        self.succ_start = succ_end;
    }

    /// Push an instruction for the current BB and current IR inst within the BB.
    pub fn push(&mut self, mut insn: I) {
        match insn.is_term() {
            MachTerminator::None | MachTerminator::Ret => {}
            MachTerminator::Uncond(target) => {
                self.vcode
                    .block_succs
                    .push(regalloc2::Block::new(target.get() as usize));
            }
            MachTerminator::Cond(true_branch, false_branch) => {
                self.vcode
                    .block_succs
                    .push(regalloc2::Block::new(true_branch.get() as usize));
                self.vcode
                    .block_succs
                    .push(regalloc2::Block::new(false_branch.get() as usize));
            }
            MachTerminator::Indirect(targets) => {
                for target in targets {
                    self.vcode
                        .block_succs
                        .push(regalloc2::Block::new(target.get() as usize));
                }
            }
        }
        if insn.defines_value_label().is_some() {
            self.vcode.has_value_labels = true;
        }
        let ops_start = self.vcode.operands.len();
        insn.visit_regs(|reg| {
            self.vcode.operands.push(reg.as_operand().unwrap());
        });
        let ops_end = self.vcode.operands.len();
        self.vcode.operand_ranges.push((ops_start, ops_end));
        self.vcode.insts.push(insn);
        self.vcode.srclocs.push(self.cur_srcloc);
    }

    /// Get the current source location.
    pub fn get_srcloc(&self) -> SourceLoc {
        self.cur_srcloc
    }

    /// Set the current source location.
    pub fn set_srcloc(&mut self, srcloc: SourceLoc) {
        self.cur_srcloc = srcloc;
    }

    /// Add block params for the current BB. Must be called for every BB.
    pub fn add_blockparams(&mut self, params: &[VReg]) {
        let start = self.vcode.block_params.len();
        self.vcode.block_params.extend(params.iter().cloned());
        let end = self.vcode.block_params.len();
        self.vcode.block_param_range.push((start, end));
    }

    /// Access the constants.
    pub fn constants(&mut self) -> &mut VCodeConstants {
        &mut self.vcode.constants
    }

    fn compute_preds(&mut self) {
        // Collect predecessors for each block.
        let mut preds: Vec<SmallVec<[regalloc2::Block; 4]>> =
            vec![smallvec![]; self.vcode.num_blocks()];
        for (block, &(start, end)) in self.vcode.block_succ_range.iter().enumerate() {
            let block = regalloc2::Block::new(block);
            let succs = &self.vcode.block_succs[start..end];
            for &succ in succs {
                preds[succ.index()].push(block);
            }
        }

        for block in 0..preds.len() {
            let block = regalloc2::Block::new(block);
            let start = self.vcode.block_preds.len();
            self.vcode
                .block_preds
                .extend(preds[block.index()].iter().cloned());
            let end = self.vcode.block_preds.len();
            self.vcode.block_pred_range.push((start, end));
        }
    }

    /// Build the final VCode, returning the vcode itself as well as auxiliary
    /// information, such as the stack map request information.
    pub fn build(mut self) -> VCode<I, A> {
        self.compute_preds();
        self.vcode
    }

    pub fn abi(&mut self) -> &mut A {
        &mut self.vcode.abi
    }
}

fn is_redundant_move<I: VCodeInst>(insn: &I) -> bool {
    if let Some((to, from)) = insn.is_move() {
        to == from
    } else {
        false
    }
}

/// Is this type a reference type?
fn is_reftype(ty: Type) -> bool {
    ty == types::R64 || ty == types::R32
}

impl<I: VCodeInst, A: ABICallee<I = I>> VCode<I, A> {
    /// New empty VCode.
    fn new(
        abi: A,
        emit_info: I::Info,
        block_order: BlockLoweringOrder,
        constants: VCodeConstants,
        generate_debug_info: bool,
    ) -> VCode<I, A> {
        VCode {
            abi,
            vreg_types: vec![],
            reftype_vregs: vec![],
            insts: vec![],
            operands: vec![],
            operand_ranges: vec![],
            srclocs: vec![],
            entry: 0,
            block_ranges: vec![],
            block_succ_range: vec![],
            block_succs: vec![],
            block_pred_range: vec![],
            block_preds: vec![],
            block_params: vec![],
            block_param_range: vec![],
            block_order,
            emit_info,
            safepoint_slots: vec![],
            generate_debug_info,
            insts_layout: RefCell::new((vec![], vec![], 0)),
            constants,
            has_value_labels: false,
        }
    }

    /// Get the IR-level type of a VReg.
    pub fn vreg_type(&self, vreg: VReg) -> Type {
        self.vreg_types[vreg.vreg()]
    }

    /// Are there any reference-typed values at all among the vregs?
    pub fn have_ref_values(&self) -> bool {
        self.reftype_vregs.len() > 0
    }

    /// Get the entry block.
    pub fn entry(&self) -> BlockIndex {
        self.entry
    }

    /// Get the number of blocks. Block indices will be in the range `0 ..
    /// (self.num_blocks() - 1)`.
    pub fn num_blocks(&self) -> usize {
        self.block_ranges.len()
    }

    /// Stack frame size for the full function's body.
    pub fn frame_size(&self) -> u32 {
        self.abi.frame_size()
    }

    /// Inbound stack-args size.
    pub fn stack_args_size(&self) -> u32 {
        self.abi.stack_args_size()
    }

    /// Get the successors for a block.
    pub fn succs(&self, block: BlockIndex) -> &[regalloc2::Block] {
        let (start, end) = self.block_succ_range[block as usize];
        &self.block_succs[start..end]
    }

    /// Get the predecessors for a block.
    pub fn preds(&self, block: BlockIndex) -> &[regalloc2::Block] {
        let (start, end) = self.block_pred_range[block as usize];
        &self.block_preds[start..end]
    }

    /// Get the blockparams for a block.
    pub fn block_params(&self, block: BlockIndex) -> &[VReg] {
        let (start, end) = self.block_param_range[block as usize];
        &self.block_params[start..end]
    }

    fn insns_for_edit(&self, edit: &regalloc2::Edit) -> SmallInstVec<I> {
        match edit {
            regalloc2::Edit::Move { from, to } => {
                let class = from.class();
                let ty = I::type_for_rc(class);
                if from.is_reg() && to.is_reg() {
                    smallvec![I::gen_move(
                        to.as_reg().unwrap(),
                        from.as_reg().unwrap(),
                        ty,
                    )]
                } else if from.is_reg() && to.is_stack() {
                    self.abi
                        .gen_spill(to.as_stack().unwrap(), from.as_reg().unwrap())
                } else if from.is_stack() && to.is_reg() {
                    self.abi
                        .gen_reload(to.as_reg().unwrap(), from.as_stack().unwrap())
                } else {
                    assert!(from.is_stack() && to.is_stack());
                    self.abi
                        .gen_stack_move(to.as_stack().unwrap(), from.as_stack().unwrap(), ty)
                }
            }
            regalloc2::Edit::BlockParams { .. } => smallvec![],
        }
    }

    /// Take the results of register allocation, with a sequence of
    /// instructions including spliced fill/reload/move instructions, and replace
    /// the VCode with them.
    pub fn finalize_with_regalloc_output(&mut self, out: &regalloc2::Output) {
        // Record the spillslot count and clobbered registers for the ABI/stack
        // setup code.
        self.abi.set_num_spillslots(out.num_spillslots);
        let mut clobbered = regalloc2::bitvec::BitVec::new();
        for i in 0..self.insts.len() {
            if !self.insts[i].is_included_in_clobbers() {
                continue;
            }
            let (start, end) = self.operand_ranges[i];
            let allocs = &out.allocs[start..end];
            let operands = &self.operands[start..end];
            for (op, alloc) in operands.iter().zip(allocs.iter()) {
                if op.kind() == OperandKind::Def && alloc.is_reg() {
                    let preg = alloc.as_reg().unwrap();
                    clobbered.set(preg.index(), true);
                }
            }
        }
        let mut clobbered_vec: Vec<PReg> = vec![];
        for i in clobbered.iter() {
            clobbered_vec.push(PReg::from_index(i));
        }
        self.abi.set_clobbered(clobbered_vec);

        // Rewrite the instruction stream, inserting edits as
        // necessary and updating operands.

        let mut final_insns = vec![];
        let mut final_block_ranges = vec![(0, 0); self.num_blocks()];
        let mut final_srclocs = vec![];
        self.operands.clear();
        let nop = I::gen_nop(0);
        let mut last_inst = 0;
        let mut edit_idx = 0;
        for block in 0..self.num_blocks() {
            let (orig_start, orig_end) = self.block_ranges[block];
            let orig_start = orig_start as usize;
            let orig_end = orig_end as usize;
            // Make sure we stream through instructions in
            // order. Lowering should have ensured this.
            assert_eq!(orig_start, last_inst);
            last_inst = orig_end;

            let block_start = final_insns.len() as InsnIndex;
            for i in orig_start..orig_end {
                // Are there any edits to perform prior to this
                // instruction? Insert them if so.
                let before_pos = regalloc2::ProgPoint::before(regalloc2::Inst::new(i));
                assert!(edit_idx >= out.edits.len() || out.edits[edit_idx].0 >= before_pos);
                while edit_idx < out.edits.len() && out.edits[edit_idx].0 == before_pos {
                    for edit_insn in self.insns_for_edit(&out.edits[edit_idx].1) {
                        final_insns.push(edit_insn);
                        final_srclocs.push(SourceLoc::default());
                    }
                    edit_idx += 1;
                }

                // Take the instruction; we won't be using `insns` again.
                let mut insn = std::mem::replace(&mut self.insts[i], nop.clone());
                // Get the allocations from the regalloc result.
                let (start, end) = self.operand_ranges[i];
                let allocs = &out.allocs[start..end];
                // Rewrite the Operands in the instruction to Allocations.
                let mut alloc_idx = 0;
                insn.visit_regs(|reg| {
                    if let Some(op) = reg.as_operand() {
                        *reg = Reg::alloc(allocs[alloc_idx], op.kind());
                    }
                    alloc_idx += 1;
                });

                // If this is the entry-point or a return, generate
                // the true {prologue, epilogue} now, respectively.
                if block == self.entry as usize && i == orig_start {
                    for insn in self.abi.gen_prologue() {
                        final_insns.push(insn);
                        final_srclocs.push(self.srclocs[i]);
                    }
                } else if insn.is_ret() {
                    for insn in self.abi.gen_epilogue() {
                        final_insns.push(insn);
                        final_srclocs.push(self.srclocs[i]);
                    }
                } else {
                    // Elide redundant moves at this point (we only know what is
                    // redundant once registers are allocated).
                    if !is_redundant_move(&insn) {
                        final_insns.push(insn);
                        final_srclocs.push(self.srclocs[i]);
                    }
                }

                // Are there any edits to perform after this
                // instruction? Insert them if so.
                let after_pos = regalloc2::ProgPoint::after(regalloc2::Inst::new(i));
                assert!(edit_idx >= out.edits.len() || out.edits[edit_idx].0 >= after_pos);
                while edit_idx < out.edits.len() && out.edits[edit_idx].0 == after_pos {
                    for edit_insn in self.insns_for_edit(&out.edits[edit_idx].1) {
                        final_insns.push(edit_insn);
                        final_srclocs.push(SourceLoc::default());
                    }
                    edit_idx += 1;
                }
            }
            let block_end = final_insns.len() as InsnIndex;
            final_block_ranges[block] = (block_start, block_end);
        }

        debug_assert!(final_insns.len() == final_srclocs.len());

        self.insts = final_insns;
        self.srclocs = final_srclocs;
        self.block_ranges = final_block_ranges;

        self.safepoint_slots = out
            .safepoint_slots
            .iter()
            .map(|(progpoint, slot)| (progpoint.inst.index() as u32, *slot))
            .collect();
    }

    /// Emit the instructions to a `MachBuffer`, containing fixed-up code and external
    /// reloc/trap/etc. records ready for use.
    pub fn emit(&self) -> MachBuffer<I>
    where
        I: MachInstEmit,
    {
        let _tt = timing::vcode_emit();
        let mut buffer = MachBuffer::new();
        let mut state = I::State::new(&self.abi);

        // The first M MachLabels are reserved for block indices, the next N MachLabels for
        // constants.
        buffer.reserve_labels_for_blocks(self.num_blocks() as BlockIndex);
        buffer.reserve_labels_for_constants(&self.constants);

        let mut inst_ends = vec![0; self.insts.len()];
        let mut label_insn_iix = vec![0; self.num_blocks()];

        let mut safepoint_idx = 0;
        let mut cur_srcloc = None;
        for block in 0..self.num_blocks() {
            let block = block as BlockIndex;
            let new_offset = I::align_basic_block(buffer.cur_offset());
            while new_offset > buffer.cur_offset() {
                // Pad with NOPs up to the aligned block offset.
                let nop = I::gen_nop((new_offset - buffer.cur_offset()) as usize);
                nop.emit(&mut buffer, &self.emit_info, &mut Default::default());
            }
            assert_eq!(buffer.cur_offset(), new_offset);

            let (start, end) = self.block_ranges[block as usize];
            buffer.bind_label(MachLabel::from_block(block));
            label_insn_iix[block as usize] = start;
            for iix in start..end {
                let srcloc = self.srclocs[iix as usize];
                if cur_srcloc != Some(srcloc) {
                    if cur_srcloc.is_some() {
                        buffer.end_srcloc();
                    }
                    buffer.start_srcloc(srcloc);
                    cur_srcloc = Some(srcloc);
                }
                state.pre_sourceloc(cur_srcloc.unwrap_or(SourceLoc::default()));

                if self.insts[iix as usize].is_safepoint() {
                    let mut slots = vec![];
                    while safepoint_idx < self.safepoint_slots.len()
                        && self.safepoint_slots[safepoint_idx].0 == iix
                    {
                        slots.push(self.safepoint_slots[safepoint_idx].1);
                        safepoint_idx += 1;
                    }
                    if slots.len() > 0 {
                        let stack_map = self.abi.spillslots_to_stack_map(&slots[..], &state);
                        state.pre_safepoint(stack_map);
                    }
                }

                self.insts[iix as usize].emit(&mut buffer, &self.emit_info, &mut state);

                if self.generate_debug_info {
                    // Buffer truncation may have happened since last inst append; trim inst-end
                    // layout info as appropriate.
                    let l = &mut inst_ends[0..iix as usize];
                    for end in l.iter_mut().rev() {
                        if *end > buffer.cur_offset() {
                            *end = buffer.cur_offset();
                        } else {
                            break;
                        }
                    }
                    inst_ends[iix as usize] = buffer.cur_offset();
                }
            }

            if cur_srcloc.is_some() {
                buffer.end_srcloc();
                cur_srcloc = None;
            }

            // Do we need an island? Get the worst-case size of the next BB and see if, having
            // emitted that many bytes, we will be beyond the deadline.
            if block < (self.num_blocks() - 1) as BlockIndex {
                let next_block = block + 1;
                let next_block_range = self.block_ranges[next_block as usize];
                let next_block_size = next_block_range.1 - next_block_range.0;
                let worst_case_next_bb = I::worst_case_size() * next_block_size;
                if buffer.island_needed(worst_case_next_bb) {
                    buffer.emit_island();
                }
            }
        }

        // Emit the constants used by the function.
        for (constant, data) in self.constants.iter() {
            let label = buffer.get_label_for_constant(constant);
            buffer.defer_constant(label, data.alignment(), data.as_slice(), u32::max_value());
        }

        if self.generate_debug_info {
            for end in inst_ends.iter_mut().rev() {
                if *end > buffer.cur_offset() {
                    *end = buffer.cur_offset();
                } else {
                    break;
                }
            }
            *self.insts_layout.borrow_mut() = (inst_ends, label_insn_iix, buffer.cur_offset());
        }

        buffer
    }

    /// Generates value-label ranges.
    pub fn value_labels_ranges(&self) -> ValueLabelsRanges {
        if !self.has_value_labels {
            return ValueLabelsRanges::default();
        }

        let layout = &self.insts_layout.borrow();
        debug::compute(&self.insts, &layout.0[..], &layout.1[..])
    }

    /// Get the offsets of stackslots.
    pub fn stackslot_offsets(&self) -> &PrimaryMap<StackSlot, u32> {
        self.abi.stackslot_offsets()
    }

    /// Get the IR block for a BlockIndex, if one exists.
    pub fn bindex_to_bb(&self, block: BlockIndex) -> Option<ir::Block> {
        self.block_order.lowered_order()[block as usize].orig_block()
    }
}

impl<I: VCodeInst, A: ABICallee<I = I>> regalloc2::Function for VCode<I, A> {
    fn insts(&self) -> usize {
        self.insts.len()
    }

    fn blocks(&self) -> usize {
        self.num_blocks()
    }

    fn entry_block(&self) -> regalloc2::Block {
        regalloc2::Block::new(self.entry as usize)
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        let (start, end) = self.block_ranges[block.index()];
        regalloc2::InstRange::forward(
            regalloc2::Inst::new(start as usize),
            regalloc2::Inst::new(end as usize),
        )
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let (start, end) = self.block_succ_range[block.index()];
        &self.block_succs[start..end]
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let (start, end) = self.block_pred_range[block.index()];
        &self.block_preds[start..end]
    }

    fn block_params(&self, block: regalloc2::Block) -> &[regalloc2::VReg] {
        let (start, end) = self.block_param_range[block.index()];
        &self.block_params[start..end]
    }

    fn is_call(&self, insn: regalloc2::Inst) -> bool {
        self.insts[insn.index()].is_call()
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        match self.insts[insn.index()].is_term() {
            MachTerminator::Ret => true,
            _ => false,
        }
    }

    fn is_move(&self, insn: regalloc2::Inst) -> Option<(VReg, VReg)> {
        match self.insts[insn.index()].is_move() {
            Some((r1, r2)) if r1.is_operand() && r2.is_operand() => Some((
                r1.as_operand().unwrap().vreg(),
                r2.as_operand().unwrap().vreg(),
            )),
            _ => None,
        }
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        match self.insts[insn.index()].is_term() {
            MachTerminator::Uncond(..)
            | MachTerminator::Cond(..)
            | MachTerminator::Indirect(..) => true,
            _ => false,
        }
    }

    fn branch_blockparam_arg_offset(&self, _: regalloc2::Block, inst: regalloc2::Inst) -> usize {
        self.insts[inst.index()].blockparam_offset()
    }

    fn is_safepoint(&self, insn: regalloc2::Inst) -> bool {
        self.insts[insn.index()].is_safepoint()
    }

    fn num_vregs(&self) -> usize {
        self.vreg_types.len()
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        let range = self.operand_ranges[insn.index()];
        &self.operands[range.0..range.1]
    }

    fn inst_clobbers(&self, insn: regalloc2::Inst) -> &[PReg] {
        self.insts[insn.index()].clobbers()
    }

    fn spillslot_size(&self, regclass: RegClass, vreg: VReg) -> usize {
        let ty = self.vreg_type(vreg);
        self.abi.get_spillslot_size(regclass, ty) as usize
    }

    fn reftype_vregs(&self) -> &[VReg] {
        &self.reftype_vregs[..]
    }

    fn multi_spillslot_named_by_last_slot(&self) -> bool {
        false
    }
}

/// Pretty-printing.
impl<I: VCodeInst, A: ABICallee<I = I>> fmt::Debug for VCode<I, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VCode {{{{\n")?;
        write!(f, "  Entry block: {}\n", self.entry)?;

        let mut state = Default::default();
        let mut safepoint_idx = 0;
        for i in 0..self.num_blocks() {
            let block = i as BlockIndex;

            write!(f, "Block {}:\n", block)?;
            if let Some(bb) = self.bindex_to_bb(block) {
                write!(f, "  (original IR block: {})\n", bb)?;
            }
            for succ in self.succs(block) {
                write!(f, "  (successor: Block {})\n", succ.index())?;
            }
            let (start, end) = self.block_ranges[block as usize];
            write!(f, "  (instruction range: {} .. {})\n", start, end)?;
            for inst in start..end {
                while safepoint_idx < self.safepoint_slots.len()
                    && self.safepoint_slots[safepoint_idx].0 == inst
                {
                    write!(
                        f,
                        "      (safepoint: slot {})\n",
                        self.safepoint_slots[safepoint_idx].1
                    )?;
                    safepoint_idx += 1;
                }
                write!(
                    f,
                    "  Inst {}:   {}",
                    inst,
                    self.insts[inst as usize].pretty_print(&mut state)
                )?;
            }
        }

        write!(f, "}}}}\n")?;

        Ok(())
    }
}

/// This structure tracks the large constants used in VCode that will be emitted separately by the
/// [MachBuffer].
///
/// First, during the lowering phase, constants are inserted using
/// [VCodeConstants.insert]; an intermediate handle, [VCodeConstant], tracks what constants are
/// used in this phase. Some deduplication is performed, when possible, as constant
/// values are inserted.
///
/// Secondly, during the emission phase, the [MachBuffer] assigns [MachLabel]s for each of the
/// constants so that instructions can refer to the value's memory location. The [MachBuffer]
/// then writes the constant values to the buffer.
#[derive(Default)]
pub struct VCodeConstants {
    constants: PrimaryMap<VCodeConstant, VCodeConstantData>,
    pool_uses: HashMap<Constant, VCodeConstant>,
    well_known_uses: HashMap<*const [u8], VCodeConstant>,
}
impl VCodeConstants {
    /// Initialize the structure with the expected number of constants.
    pub fn with_capacity(expected_num_constants: usize) -> Self {
        Self {
            constants: PrimaryMap::with_capacity(expected_num_constants),
            pool_uses: HashMap::with_capacity(expected_num_constants),
            well_known_uses: HashMap::new(),
        }
    }

    /// Insert a constant; using this method indicates that a constant value will be used and thus
    /// will be emitted to the `MachBuffer`. The current implementation can deduplicate constants
    /// that are [VCodeConstantData::Pool] or [VCodeConstantData::WellKnown] but not
    /// [VCodeConstantData::Generated].
    pub fn insert(&mut self, data: VCodeConstantData) -> VCodeConstant {
        match data {
            VCodeConstantData::Generated(_) => self.constants.push(data),
            VCodeConstantData::Pool(constant, _) => match self.pool_uses.get(&constant) {
                None => {
                    let vcode_constant = self.constants.push(data);
                    self.pool_uses.insert(constant, vcode_constant);
                    vcode_constant
                }
                Some(&vcode_constant) => vcode_constant,
            },
            VCodeConstantData::WellKnown(data_ref) => {
                match self.well_known_uses.get(&(data_ref as *const [u8])) {
                    None => {
                        let vcode_constant = self.constants.push(data);
                        self.well_known_uses
                            .insert(data_ref as *const [u8], vcode_constant);
                        vcode_constant
                    }
                    Some(&vcode_constant) => vcode_constant,
                }
            }
        }
    }

    /// Retrieve a byte slice for the given [VCodeConstant], if available.
    pub fn get(&self, constant: VCodeConstant) -> Option<&[u8]> {
        self.constants.get(constant).map(|d| d.as_slice())
    }

    /// Return the number of constants inserted.
    pub fn len(&self) -> usize {
        self.constants.len()
    }

    /// Iterate over the [VCodeConstant] keys inserted in this structure.
    pub fn keys(&self) -> Keys<VCodeConstant> {
        self.constants.keys()
    }

    /// Iterate over the [VCodeConstant] keys and the data (as a byte slice) inserted in this
    /// structure.
    pub fn iter(&self) -> impl Iterator<Item = (VCodeConstant, &VCodeConstantData)> {
        self.constants.iter()
    }
}

/// A use of a constant by one or more VCode instructions; see [VCodeConstants].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VCodeConstant(u32);
entity_impl!(VCodeConstant);

/// Identify the different types of constant that can be inserted into [VCodeConstants]. Tracking
/// these separately instead of as raw byte buffers allows us to avoid some duplication.
pub enum VCodeConstantData {
    /// A constant already present in the Cranelift IR
    /// [ConstantPool](crate::ir::constant::ConstantPool).
    Pool(Constant, ConstantData),
    /// A reference to a well-known constant value that is statically encoded within the compiler.
    WellKnown(&'static [u8]),
    /// A constant value generated during lowering; the value may depend on the instruction context
    /// which makes it difficult to de-duplicate--if possible, use other variants.
    Generated(ConstantData),
}
impl VCodeConstantData {
    /// Retrieve the constant data as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            VCodeConstantData::Pool(_, d) | VCodeConstantData::Generated(d) => d.as_slice(),
            VCodeConstantData::WellKnown(d) => d,
        }
    }

    /// Calculate the alignment of the constant data.
    pub fn alignment(&self) -> u32 {
        if self.as_slice().len() <= 8 {
            8
        } else {
            16
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn size_of_constant_structs() {
        assert_eq!(size_of::<Constant>(), 4);
        assert_eq!(size_of::<VCodeConstant>(), 4);
        assert_eq!(size_of::<ConstantData>(), 24);
        assert_eq!(size_of::<VCodeConstantData>(), 32);
        assert_eq!(
            size_of::<PrimaryMap<VCodeConstant, VCodeConstantData>>(),
            24
        );
        // TODO The VCodeConstants structure's memory size could be further optimized.
        // With certain versions of Rust, each `HashMap` in `VCodeConstants` occupied at
        // least 48 bytes, making an empty `VCodeConstants` cost 120 bytes.
    }
}
