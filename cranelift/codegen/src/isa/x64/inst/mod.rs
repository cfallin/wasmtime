//! This module defines x86_64-specific machine instruction types.

use crate::binemit::{CodeOffset, StackMap};
use crate::ir::{types, ExternalName, Opcode, SourceLoc, TrapCode, Type, ValueLabel};
use crate::isa::unwind::UnwindInst;
use crate::isa::x64::abi::{X64ABICallee, X64ABIMachineSpec};
use crate::isa::x64::settings as x64_settings;
use crate::isa::CallConv;
use crate::machinst::*;
use crate::{settings, settings::Flags, CodegenError, CodegenResult};
use alloc::boxed::Box;
use alloc::vec::Vec;
use regalloc2::{Operand, PReg, VReg};
use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::string::{String, ToString};

pub mod args;
mod emit;
#[cfg(test)]
mod emit_tests;
pub(crate) mod encoding;
pub mod regs;
pub mod unwind;

use args::*;

//=============================================================================
// Instructions (top level): definition

// Don't build these directly.  Instead use the Inst:: functions to create them.

/// Instructions.  Destinations are on the RIGHT (a la AT&T syntax).
#[derive(Clone)]
pub enum Inst {
    /// Nops of various sizes, including zero.
    Nop { len: u8 },

    // =====================================
    // Integer instructions.
    /// Integer arithmetic/bit-twiddling: (add sub and or xor mul adc? sbb?) (32 64) (reg addr imm) reg
    AluRmiR {
        size: OperandSize, // 4 or 8
        op: AluRmiROpcode,
        src1: RegMemImm,
        src2: Reg,
        dst: Reg,
    },

    /// Instructions on GPR that only read src and defines dst (dst is not modified): bsr, etc.
    UnaryRmR {
        size: OperandSize, // 2, 4 or 8
        op: UnaryRmROpcode,
        src: RegMem,
        dst: Reg,
    },

    /// Bitwise not
    Not {
        size: OperandSize, // 1, 2, 4 or 8
        src: Reg,
        dst: Reg,
    },

    /// Integer negation
    Neg {
        size: OperandSize, // 1, 2, 4 or 8
        src: Reg,
        dst: Reg,
    },

    /// Integer quotient and remainder: (div idiv) $rax $rdx (reg addr)
    Div {
        size: OperandSize, // 1, 2, 4 or 8
        signed: bool,
        divisor: RegMem,
        in_lo: Reg,
        in_hi: Reg,
        out_div: Reg,
        out_rem: Reg,
    },

    /// The low and high bits of a (un)signed multiply: RDX:RAX := RAX * rhs.
    Mul {
        size: OperandSize, // 2, 4, or 8
        signed: bool,
        rhs: RegMem,
        src: Reg,
        out_lo: Reg,
        out_hi: Reg,
    },

    /// A synthetic sequence to implement the right inline checks for remainder and division,
    /// assuming the dividend is in %rax.
    /// Puts the result back into %rax if is_div, %rdx if !is_div, to mimic what the div
    /// instruction does.
    /// The generated code sequence is described in the emit's function match arm for this
    /// instruction.
    CheckedDivOrRemSeq {
        kind: DivOrRemKind,
        size: OperandSize,
        divisor: Reg,  // input
        tmp: Reg,      // tmp
        dividend: Reg, // input
        out_lo: Reg,   // output
        out_hi: Reg,   // output
    },

    /// Do a sign-extend based on the sign of the value in rax into rdx: (cwd cdq cqo)
    /// or al into ah: (cbw)
    SignExtendData {
        size: OperandSize, // 1, 2, 4 or 8
        src: Reg,
        dst: Reg,
    },

    /// Constant materialization: (imm32 imm64) reg.
    /// Either: movl $imm32, %reg32 or movabsq $imm64, %reg32.
    Imm {
        dst_size: OperandSize, // 4 or 8
        simm64: u64,
        dst: Reg,
    },

    /// GPR to GPR move: mov (64 32) reg reg.
    MovRR {
        size: OperandSize, // 4 or 8
        src: Reg,
        dst: Reg,
    },

    /// Zero-extended loads, except for 64 bits: movz (bl bq wl wq lq) addr reg.
    /// Note that the lq variant doesn't really exist since the default zero-extend rule makes it
    /// unnecessary. For that case we emit the equivalent "movl AM, reg32".
    MovzxRmR {
        ext_mode: ExtMode,
        src: RegMem,
        dst: Reg,
    },

    /// A plain 64-bit integer load, since MovZX_RM_R can't represent that.
    Mov64MR { src: SyntheticAmode, dst: Reg },

    /// Loads the memory address of addr into dst.
    LoadEffectiveAddress { addr: SyntheticAmode, dst: Reg },

    /// Sign-extended loads and moves: movs (bl bq wl wq lq) addr reg.
    MovsxRmR {
        ext_mode: ExtMode,
        src: RegMem,
        dst: Reg,
    },

    /// Integer stores: mov (b w l q) reg addr.
    MovRM {
        size: OperandSize, // 1, 2, 4 or 8.
        src: Reg,
        dst: SyntheticAmode,
    },

    /// Arithmetic shifts: (shl shr sar) (b w l q) imm reg.
    ShiftRImm {
        size: OperandSize, // 1, 2, 4 or 8
        kind: ShiftKind,
        /// shift count: 0 .. #bits-in-type - 1.
        num_bits: u8,
        src: Reg,
        dst: Reg, // reuses src
    },

    /// Arithmetic shifts: like `ShiftRImm` but with a variable shift-count input.
    ShiftRVar {
        size: OperandSize, // 1, 2, 4 or 8
        kind: ShiftKind,
        src: Reg,
        count: Reg,
        dst: Reg, // reuses src
    },

    /// Arithmetic SIMD shifts.
    XmmRmiReg {
        opcode: SseOpcode,
        src1: RegMemImm,
        src2: Reg,
        dst: Reg, // reuses src2
    },

    /// Integer comparisons/tests: cmp or test (b w l q) (reg addr imm) reg.
    CmpRmiR {
        size: OperandSize, // 1, 2, 4 or 8
        opcode: CmpOpcode,
        src: RegMemImm,
        dst: Reg,
    },

    /// Materializes the requested condition code in the destination reg.
    Setcc { cc: CC, dst: Reg },

    /// Integer conditional move.
    /// Overwrites the destination register.
    Cmove {
        size: OperandSize, // 2, 4, or 8
        cc: CC,
        src1: RegMem,
        src2: Reg,
        dst: Reg,
    },

    // =====================================
    // Stack manipulation.
    /// pushq (reg addr imm)
    Push64 { src: RegMemImm },

    /// popq reg
    Pop64 { dst: Reg },

    // =====================================
    // Floating-point operations.
    /// XMM (scalar or vector) binary op: (add sub and or xor mul adc? sbb?) (32 64) (reg addr) reg
    XmmRmR {
        op: SseOpcode,
        src1: RegMem,
        src2: Reg,
        dst: Reg,
    },

    /// XMM (scalar or vector) unary op: mov between XMM registers (32 64) (reg addr) reg, sqrt,
    /// etc.
    ///
    /// This differs from XMM_RM_R in that the dst register of XmmUnaryRmR is not used in the
    /// computation of the instruction dst value and so does not have to be a previously valid
    /// value. This is characteristic of mov instructions.
    XmmUnaryRmR {
        op: SseOpcode,
        src: RegMem,
        dst: Reg,
    },

    /// XMM (scalar or vector) unary op (from xmm to reg/mem): stores, movd, movq
    XmmMovRM {
        op: SseOpcode,
        src: Reg,
        dst: SyntheticAmode,
    },

    /// XMM (vector) unary op (to move a constant value into an xmm register): movups
    XmmLoadConst {
        src: VCodeConstant,
        dst: Reg,
        ty: Type,
    },

    /// XMM (scalar) unary op (from xmm to integer reg): movd, movq, cvtts{s,d}2si
    XmmToGpr {
        op: SseOpcode,
        src: Reg,
        dst: Reg,
        dst_size: OperandSize,
    },

    /// XMM (scalar) unary op (from integer to float reg): movd, movq, cvtsi2s{s,d}
    GprToXmm {
        op: SseOpcode,
        src: RegMem,
        dst: Reg,
        src_size: OperandSize,
    },

    /// Converts an unsigned int64 to a float32/float64.
    CvtUint64ToFloatSeq {
        dst_size: OperandSize, // 4 or 8
        src: Reg,
        dst: Reg,
        tmp_gpr1: Reg,
        tmp_gpr2: Reg,
    },

    /// Converts a scalar xmm to a signed int32/int64.
    CvtFloatToSintSeq {
        dst_size: OperandSize,
        src_size: OperandSize,
        is_saturating: bool,
        src: Reg,
        dst: Reg,
        tmp_gpr: Reg,
        tmp_xmm: Reg,
    },

    /// Converts a scalar xmm to an unsigned int32/int64.
    CvtFloatToUintSeq {
        src_size: OperandSize,
        dst_size: OperandSize,
        is_saturating: bool,
        src: Reg,
        dst: Reg,
        tmp_gpr: Reg,
        tmp_xmm: Reg,
    },

    /// A sequence to compute min/max with the proper NaN semantics for xmm registers.
    XmmMinMaxSeq {
        size: OperandSize,
        is_min: bool,
        lhs: Reg,
        rhs: Reg,
        dst: Reg,
    },

    /// XMM (scalar) conditional move.
    /// Overwrites the destination register if cc is set.
    XmmCmove {
        size: OperandSize, // 4 or 8
        cc: CC,
        src1: RegMem,
        src2: Reg,
        dst: Reg, // reuses src2
    },

    /// Float comparisons/tests: cmp (b w l q) (reg addr imm) reg.
    XmmCmpRmR {
        op: SseOpcode,
        src: RegMem,
        dst: Reg,
    },

    /// A binary XMM instruction with an 8-bit immediate: e.g. cmp (ps pd) imm (reg addr) reg
    XmmRmRImm {
        op: SseOpcode,
        src1: RegMem,
        src2: Reg,
        dst: Reg,
        imm: u8,
        size: OperandSize, // 4 or 8
    },

    // =====================================
    // Control flow instructions.
    /// Direct call: call simm32.
    CallKnown {
        dest: ExternalName,
        opcode: Opcode,
        operands: Vec<Reg>,
        clobbers: &'static [PReg],
    },

    /// Indirect call: callq (reg mem).
    CallUnknown {
        dest: RegMem,
        opcode: Opcode,
        operands: Vec<Reg>,
        clobbers: &'static [PReg],
    },

    /// Return.
    Ret,

    /// Jump to a known target: jmp simm32.
    JmpKnown { dst: MachLabel, args: Vec<Reg> },

    /// One-way conditional branch: jcond cond target.
    ///
    /// This instruction is useful when we have conditional jumps depending on more than two
    /// conditions, see for instance the lowering of Brz/brnz with Fcmp inputs.
    ///
    /// A note of caution: in contexts where the branch target is another block, this has to be the
    /// same successor as the one specified in the terminator branch of the current block.
    /// Otherwise, this might confuse register allocation by creating new invisible edges.
    JmpIf { cc: CC, taken: MachLabel },

    /// Two-way conditional branch: jcond cond target target.
    /// Emitted as a compound sequence; the MachBuffer will shrink it as appropriate.
    JmpCond {
        cc: CC,
        taken: MachLabel,
        not_taken: MachLabel,
        /// Blockparam args. Used to communicate to the regalloc but
        /// not used during code emission; the blockparam dataflow is
        /// reified into explicit moves by regalloc edits.
        args: Vec<Reg>,
    },

    /// Jump-table sequence, as one compound instruction (see note in lower.rs for rationale).
    /// The generated code sequence is described in the emit's function match arm for this
    /// instruction.
    /// See comment in lowering about the temporaries signedness.
    JmpTableSeq {
        idx: Reg,
        tmp1: Reg,
        tmp2: Reg,
        default_target: MachLabel,
        targets: Vec<MachLabel>,
        targets_for_term: Vec<MachLabel>,
        args: Vec<Reg>,
    },

    /// Indirect jump: jmpq (reg mem).
    JmpUnknown { target: RegMem, args: Vec<Reg> },

    /// Traps if the condition code is set.
    TrapIf { cc: CC, trap_code: TrapCode },

    /// A debug trap.
    Hlt,

    /// An instruction that will always trigger the illegal instruction exception.
    Ud2 { trap_code: TrapCode },

    /// Loads an external symbol in a register, with a relocation:
    ///
    /// movq $name@GOTPCREL(%rip), dst    if PIC is enabled, or
    /// movabsq $name, dst                otherwise.
    LoadExtName {
        dst: Reg,
        name: Box<ExternalName>,
        offset: i64,
    },

    // =====================================
    // Instructions pertaining to atomic memory accesses.
    /// A standard (native) `lock cmpxchg src, (amode)`.
    LockCmpxchg {
        ty: Type, // I8, I16, I32 or I64
        src: Reg,
        dst: SyntheticAmode,
        expected: Reg, // input
        actual: Reg,   // output
    },

    /// A synthetic instruction, based on a loop around a native `lock cmpxchg` instruction.
    /// This atomically modifies a value in memory and returns the old value.  The sequence
    /// consists of an initial "normal" load from `dst`, followed by a loop which computes the
    /// new value and tries to compare-and-swap ("CAS") it into `dst`, using the native
    /// instruction `lock cmpxchg{b,w,l,q}` .  The loop iterates until the CAS is successful.
    /// If there is no contention, there will be only one pass through the loop body.  The
    /// sequence does *not* perform any explicit memory fence instructions
    /// (mfence/sfence/lfence).
    ///
    /// Note that the transaction is atomic in the sense that, as observed by some other thread,
    /// `dst` either has the initial or final value, but no other.  It isn't atomic in the sense
    /// of guaranteeing that no other thread writes to `dst` in between the initial load and the
    /// CAS -- but that would cause the CAS to fail unless the other thread's last write before
    /// the CAS wrote the same value that was already there.  In other words, this
    /// implementation suffers (unavoidably) from the A-B-A problem.
    AtomicRmwSeq {
        ty: Type, // I8, I16, I32 or I64
        op: inst_common::AtomicRmwOp,
        addr: Reg,    // input
        src: Reg,     // input
        scratch: Reg, // temp
        old_out: Reg, // output
    },

    /// A memory fence (mfence, lfence or sfence).
    Fence { kind: FenceKind },

    // =====================================
    // Meta-instructions generating no code.
    /// Marker, no-op in generated code: SP "virtual offset" is adjusted. This
    /// controls how MemArg::NominalSPOffset args are lowered.
    VirtualSPOffsetAdj { offset: i64 },

    /// Provides a way to tell the register allocator that the upcoming sequence of instructions
    /// will overwrite `dst` so it should be considered as a `def`; use this with care.
    ///
    /// This is useful when we have a sequence of instructions whose register usages are nominally
    /// `mod`s, but such that the combination of operations creates a result that is independent of
    /// the initial register value. It's thus semantically a `def`, not a `mod`, when all the
    /// instructions are taken together, so we want to ensure the register is defined (its
    /// live-range starts) prior to the sequence to keep analyses happy.
    ///
    /// One alternative would be a compound instruction that somehow encapsulates the others and
    /// reports its own `def`s/`use`s/`mod`s; this adds complexity (the instruction list is no
    /// longer flat) and requires knowledge about semantics and initial-value independence anyway.
    XmmUninitializedValue { dst: Reg },

    /// A call to the `ElfTlsGetAddr` libcall. Returns address
    /// of TLS symbol in rax.
    ElfTlsGetAddr { symbol: ExternalName, dst: Reg },

    /// A Mach-O TLS symbol access. Returns address of the TLS
    /// symbol in rax.
    MachOTlsGetAddr { symbol: ExternalName, dst: Reg },

    /// A definition of a value label.
    ValueLabelMarker { reg: Reg, label: ValueLabel },

    /// An unwind pseudoinstruction describing the state of the
    /// machine at this program point.
    Unwind { inst: UnwindInst },

    /// A set of register operands with constraints that generates no
    /// machine code. This can be used to tie vregs to registers at
    /// certain program points, e.g. at function entry/exit.
    RegConstraints { args: Vec<Reg> },
}

pub(crate) fn low32_will_sign_extend_to_64(x: u64) -> bool {
    let xs = x as i64;
    xs == ((xs << 32) >> 32)
}

impl Inst {
    /// Retrieve a list of ISA feature sets in which the instruction is available. An empty list
    /// indicates that the instruction is available in the baseline feature set (i.e. SSE2 and
    /// below); more than one `InstructionSet` in the list indicates that the instruction is present
    /// *any* of the included ISA feature sets.
    fn available_in_any_isa(&self) -> SmallVec<[InstructionSet; 2]> {
        match self {
            // These instructions are part of SSE2, which is a basic requirement in Cranelift, and
            // don't have to be checked.
            Inst::AluRmiR { .. }
            | Inst::AtomicRmwSeq { .. }
            | Inst::CallKnown { .. }
            | Inst::CallUnknown { .. }
            | Inst::CheckedDivOrRemSeq { .. }
            | Inst::Cmove { .. }
            | Inst::CmpRmiR { .. }
            | Inst::CvtFloatToSintSeq { .. }
            | Inst::CvtFloatToUintSeq { .. }
            | Inst::CvtUint64ToFloatSeq { .. }
            | Inst::Div { .. }
            | Inst::EpiloguePlaceholder
            | Inst::Fence { .. }
            | Inst::Hlt
            | Inst::Imm { .. }
            | Inst::JmpCond { .. }
            | Inst::JmpIf { .. }
            | Inst::JmpKnown { .. }
            | Inst::JmpTableSeq { .. }
            | Inst::JmpUnknown { .. }
            | Inst::LoadEffectiveAddress { .. }
            | Inst::LoadExtName { .. }
            | Inst::LockCmpxchg { .. }
            | Inst::Mov64MR { .. }
            | Inst::MovRM { .. }
            | Inst::MovRR { .. }
            | Inst::MovsxRmR { .. }
            | Inst::MovzxRmR { .. }
            | Inst::MulHi { .. }
            | Inst::Neg { .. }
            | Inst::Not { .. }
            | Inst::Nop { .. }
            | Inst::Pop64 { .. }
            | Inst::Push64 { .. }
            | Inst::Ret
            | Inst::Setcc { .. }
            | Inst::ShiftRImm { .. }
            | Inst::ShiftRVar { .. }
            | Inst::SignExtendData { .. }
            | Inst::TrapIf { .. }
            | Inst::Ud2 { .. }
            | Inst::VirtualSPOffsetAdj { .. }
            | Inst::XmmCmove { .. }
            | Inst::XmmCmpRmR { .. }
            | Inst::XmmLoadConst { .. }
            | Inst::XmmMinMaxSeq { .. }
            | Inst::XmmUninitializedValue { .. }
            | Inst::ElfTlsGetAddr { .. }
            | Inst::MachOTlsGetAddr { .. }
            | Inst::ValueLabelMarker { .. }
            | Inst::Unwind { .. } => smallvec![],

            Inst::UnaryRmR { op, .. } => op.available_from(),

            // These use dynamic SSE opcodes.
            Inst::GprToXmm { op, .. }
            | Inst::XmmMovRM { op, .. }
            | Inst::XmmRmiReg { opcode: op, .. }
            | Inst::XmmRmR { op, .. }
            | Inst::XmmRmRImm { op, .. }
            | Inst::XmmToGpr { op, .. }
            | Inst::XmmUnaryRmR { op, .. } => smallvec![op.available_from()],
        }
    }
}

// Handy constructors for Insts.

impl Inst {
    pub(crate) fn nop(len: u8) -> Self {
        debug_assert!(len <= 15);
        Self::Nop { len }
    }

    pub(crate) fn alu_rmi_r(
        size: OperandSize,
        op: AluRmiROpcode,
        src1: RegMemImm,
        src2: VReg,
        dst: VReg,
    ) -> Self {
        debug_assert!(size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src1 = src1.adjust_operand(|vreg| Reg::reg_use(vreg));
        let src2 = Reg::reg_use(src2);
        // `src2` always comes first when presenting operands to the
        // regalloc so that we can know its index for sure, because
        // `src1` may or may not contain a `VReg`. This allows us to
        // name the reused register operand index correctly here.
        let dst = Reg::reg_reuse_def(dst, 0);
        Self::AluRmiR {
            size,
            op,
            src1,
            src2,
            dst,
        }
    }

    pub(crate) fn unary_rm_r(
        size: OperandSize,
        op: UnaryRmROpcode,
        src: RegMem,
        dst: VReg,
    ) -> Self {
        debug_assert!(size.is_one_of(&[
            OperandSize::Size16,
            OperandSize::Size32,
            OperandSize::Size64
        ]));
        let src = src.adjust_operand(|vreg| Reg::reg_use(vreg));
        let dst = Reg::reg_def(dst);
        Self::UnaryRmR { size, op, src, dst }
    }

    pub(crate) fn not(size: OperandSize, src: VReg, dst: VReg) -> Inst {
        let src = Reg::reg_use(src);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::Not { size, src, dst }
    }

    pub(crate) fn neg(size: OperandSize, src: VReg, dst: VReg) -> Inst {
        let src = Reg::reg_use(src);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::Neg { size, src, dst }
    }

    pub(crate) fn div(
        size: OperandSize,
        signed: bool,
        divisor: RegMem,
        in_lo: VReg,
        in_hi: VReg,
        out_div: VReg,
        out_rem: VReg,
    ) -> Inst {
        let divisor = divisor.adjust_operand(|vreg| Reg::reg_use(vreg));
        let in_lo = Reg::reg_fixed_use(in_lo, regs::rax());
        let in_hi = Reg::reg_fixed_use(in_hi, regs::rdx());
        let out_div = Reg::reg_fixed_def(out_div, regs::rax());
        let out_rem = Reg::reg_fixed_def(out_rem, regs::rdx());
        Inst::Div {
            size,
            signed,
            divisor,
            in_lo,
            in_hi,
            out_div,
            out_rem,
        }
    }

    pub(crate) fn mul(
        size: OperandSize,
        signed: bool,
        rhs: RegMem,
        src: VReg,
        out_lo: VReg,
        out_hi: VReg,
    ) -> Inst {
        debug_assert!(size.is_one_of(&[
            OperandSize::Size16,
            OperandSize::Size32,
            OperandSize::Size64
        ]));
        let rhs = rhs.adjust_operand(|vreg| Reg::reg_use(vreg));
        let out_lo = Reg::reg_fixed_def(out_lo, regs::rax());
        let out_hi = Reg::reg_fixed_def(out_hi, regs::rdx());
        let src = Reg::reg_fixed_use(src, regs::rax());
        Inst::Mul {
            size,
            signed,
            rhs,
            src,
            out_lo,
            out_hi,
        }
    }

    pub(crate) fn checked_div_or_rem_seq(
        kind: DivOrRemKind,
        size: OperandSize,
        divisor: VReg,
        dividend: VReg,
        tmp: VReg,
        out_lo: VReg,
        out_hi: VReg,
    ) -> Inst {
        let divisor = Reg::reg_use(divisor);
        let dividend = Reg::reg_fixed_use(dividend, regs::rax());
        let divisor = Reg::reg_use(divisor);
        let tmp = Reg::reg_temp(tmp);
        let out_lo = Reg::reg_fixed_use(out_lo, regs::rax());
        let out_hi = Reg::reg_fixed_use(out_hi, regs::rax());

        Inst::CheckedDivOrRemSeq {
            kind,
            size,
            divisor,
            tmp,
            dividend,
            out_lo,
            out_hi,
        }
    }

    pub(crate) fn sign_extend_data(size: OperandSize, src: VReg, dst: VReg) -> Inst {
        let src = Reg::reg_fixed_use(src, regs::rax());
        let dst_reg = match size {
            OperandSize::Size8 => regs::rax(), // result into AH
            _ => regs::rdx(),
        };
        let dst = Reg::reg_fixed_def(dst, dst_reg);
        Inst::SignExtendData { size, src, dst }
    }

    pub(crate) fn imm(dst_size: OperandSize, simm64: u64, dst: VReg) -> Inst {
        // Try to generate a 32-bit immediate when the upper high bits are zeroed (which matches
        // the semantics of movl).
        let dst_size = match dst_size {
            OperandSize::Size64 if simm64 > u32::max_value() as u64 => OperandSize::Size64,
            _ => OperandSize::Size32,
        };
        let dst = Reg::reg_def(dst);
        Inst::Imm {
            dst_size,
            simm64,
            dst,
        }
    }

    pub(crate) fn mov_r_r(size: OperandSize, src: VReg, dst: VReg) -> Inst {
        debug_assert!(size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src = Reg::reg_use(src);
        let dst = Reg::reg_def(dst);
        Inst::MovRR { size, src, dst }
    }

    // TODO Can be replaced by `Inst::move` (high-level) and `Inst::unary_rm_r` (low-level)
    pub(crate) fn xmm_mov(op: SseOpcode, src: RegMem, dst: VReg) -> Inst {
        let src = src.adjust_operand(|vreg| Reg::reg_use(vreg));
        let dst = Reg::reg_def(dst);
        Inst::XmmUnaryRmR { op, src, dst }
    }

    pub(crate) fn xmm_load_const(src: VCodeConstant, dst: VReg, ty: Type) -> Inst {
        debug_assert!(ty.is_vector() && ty.bits() == 128);
        let dst = Reg::reg_def(dst);
        Inst::XmmLoadConst { src, dst, ty }
    }

    /// Convenient helper for unary float operations.
    pub(crate) fn xmm_unary_rm_r(op: SseOpcode, src: RegMem, dst: VReg) -> Inst {
        let src = src.adjust_operand(|vreg| Reg::reg_use(vreg));
        let dst = Reg::reg_def(dst);
        Inst::XmmUnaryRmR { op, src, dst }
    }

    pub(crate) fn xmm_rm_r(op: SseOpcode, src1: RegMem, src2: VReg, dst: VReg) -> Self {
        let src1 = src1.adjust_operand(|vreg| Reg::reg_use(src1));
        let src2 = Reg::reg_use(src2);
        // as with alu_rmi_r case, src2 comes first so we can use `0` index here.
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::XmmRmR {
            op,
            src1,
            src2,
            dst,
        }
    }

    pub(crate) fn xmm_uninit_value(dst: VReg) -> Self {
        let dst = Reg::reg_def(dst);
        Inst::XmmUninitializedValue { dst }
    }

    pub(crate) fn xmm_mov_r_m(op: SseOpcode, src: VReg, dst: impl Into<SyntheticAmode>) -> Inst {
        let src = Reg::reg_use(src);
        Inst::XmmMovRM {
            op,
            src,
            dst: dst.into(),
        }
    }

    pub(crate) fn xmm_to_gpr(op: SseOpcode, src: VReg, dst: VReg, dst_size: OperandSize) -> Inst {
        debug_assert!(dst_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src = Reg::reg_use(src);
        let dst = Reg::reg_def(dst);
        Inst::XmmToGpr {
            op,
            src,
            dst,
            dst_size,
        }
    }

    pub(crate) fn gpr_to_xmm(op: SseOpcode, src: RegMem, src_size: OperandSize, dst: VReg) -> Inst {
        debug_assert!(src_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let dst = Reg::reg_def(dst);
        Inst::GprToXmm {
            op,
            src,
            dst,
            src_size,
        }
    }

    pub(crate) fn xmm_cmp_rm_r(op: SseOpcode, src: RegMem, dst: VReg) -> Inst {
        // N.B.: a *use*, not def -- `dst` is actually a source.
        let dst = Reg::reg_use(dst);
        Inst::XmmCmpRmR { op, src, dst }
    }

    pub(crate) fn cvt_u64_to_float_seq(
        dst_size: OperandSize,
        src: VReg,
        tmp_gpr1: VReg,
        tmp_gpr2: VReg,
        dst: VReg,
    ) -> Inst {
        debug_assert!(dst_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src = Reg::reg_use(src);
        let dst = Reg::reg_def(dst);
        let tmp_gpr1 = Reg::reg_temp(tmp_gpr1);
        let tmp_gpr2 = Reg::reg_temp(tmp_gpr2);
        Inst::CvtUint64ToFloatSeq {
            src,
            dst,
            tmp_gpr1,
            tmp_gpr2,
            dst_size,
        }
    }

    pub(crate) fn cvt_float_to_sint_seq(
        src_size: OperandSize,
        dst_size: OperandSize,
        is_saturating: bool,
        src: VReg,
        dst: VReg,
        tmp_gpr: VReg,
        tmp_xmm: VReg,
    ) -> Inst {
        debug_assert!(src_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        debug_assert!(dst_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src = Reg::reg_use(src);
        let dst = Reg::reg_def(dst);
        let tmp_gpr = Reg::reg_temp(tmp_gpr);
        let tmp_xmm = Reg::reg_temp(tmp_xmm);
        Inst::CvtFloatToSintSeq {
            src_size,
            dst_size,
            is_saturating,
            src,
            dst,
            tmp_gpr,
            tmp_xmm,
        }
    }

    pub(crate) fn cvt_float_to_uint_seq(
        src_size: OperandSize,
        dst_size: OperandSize,
        is_saturating: bool,
        src: VReg,
        dst: VReg,
        tmp_gpr: VReg,
        tmp_xmm: VReg,
    ) -> Inst {
        debug_assert!(src_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        debug_assert!(dst_size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src = Reg::reg_use(src);
        let dst = Reg::reg_def(dst);
        let tmp_gpr = Reg::reg_temp(tmp_gpr);
        let tmp_xmm = Reg::reg_temp(tmp_xmm);
        Inst::CvtFloatToUintSeq {
            src_size,
            dst_size,
            is_saturating,
            src,
            dst,
            tmp_gpr,
            tmp_xmm,
        }
    }

    pub(crate) fn xmm_min_max_seq(
        size: OperandSize,
        is_min: bool,
        lhs: VReg,
        rhs: VReg,
        dst: VReg,
    ) -> Inst {
        debug_assert!(size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let lhs = Reg::reg_use(lhs);
        let rhs = Reg::reg_use(rhs);
        let dst = Reg::reg_reuse_def(dst, 1);
        Inst::XmmMinMaxSeq {
            size,
            is_min,
            lhs,
            rhs,
            dst,
        }
    }

    pub(crate) fn xmm_rm_r_imm(
        op: SseOpcode,
        src1: RegMem,
        src2: VReg,
        dst: VReg,
        imm: u8,
        size: OperandSize,
    ) -> Inst {
        debug_assert!(size.is_one_of(&[OperandSize::Size32, OperandSize::Size64]));
        let src1 = src1.adjust_operand(|vreg| Reg::reg_use(vreg));
        let src2 = Reg::reg_use(src2);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::XmmRmRImm {
            op,
            src1,
            src2,
            dst,
            imm,
            size,
        }
    }

    pub(crate) fn movzx_rm_r(ext_mode: ExtMode, src: RegMem, dst: VReg) -> Inst {
        let src = src.adjust_operand(|vreg| Reg::reg_use(vreg));
        let dst = Reg::reg_def(dst);
        Inst::MovzxRmR { ext_mode, src, dst }
    }

    pub(crate) fn xmm_rmi_reg(opcode: SseOpcode, src1: RegMemImm, src2: VReg, dst: VReg) -> Inst {
        let src1 = src1.adjust_operand(|vreg| Reg::reg_use(vreg));
        let src2 = Reg::reg_use(src2);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::XmmRmiReg {
            opcode,
            src1,
            src2,
            dst,
        }
    }

    pub(crate) fn movsx_rm_r(ext_mode: ExtMode, src: RegMem, dst: VReg) -> Inst {
        let src = src.adjust_operand(|vreg| Reg::reg_use(vreg));
        let dst = Reg::reg_def(dst);
        Inst::MovsxRmR { ext_mode, src, dst }
    }

    pub(crate) fn mov64_m_r(src: impl Into<SyntheticAmode>, dst: VReg) -> Inst {
        let dst = Reg::reg_def(dst);
        Inst::Mov64MR {
            src: src.into(),
            dst,
        }
    }

    /// A convenience function to be able to use a RegMem as the source of a move.
    pub(crate) fn mov64_rm_r(src: RegMem, dst: VReg) -> Inst {
        match src {
            RegMem::Reg { reg } => Self::mov_r_r(OperandSize::Size64, reg, dst),
            RegMem::Mem { addr } => Self::mov64_m_r(addr, dst),
        }
    }

    pub(crate) fn mov_r_m(size: OperandSize, src: VReg, dst: impl Into<SyntheticAmode>) -> Inst {
        let src = Reg::reg_use(src);
        Inst::MovRM {
            size,
            src,
            dst: dst.into(),
        }
    }

    pub(crate) fn lea(addr: impl Into<SyntheticAmode>, dst: VReg) -> Inst {
        let dst = Reg::reg_def(dst);
        Inst::LoadEffectiveAddress {
            addr: addr.into(),
            dst,
        }
    }

    pub(crate) fn shift_r_imm(
        size: OperandSize,
        kind: ShiftKind,
        num_bits: u8,
        src: VReg,
        dst: VReg,
    ) -> Inst {
        debug_assert!(num_bits < size.to_bits());
        let src = Reg::reg_use(src);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::ShiftRImm {
            size,
            kind,
            num_bits,
            src,
            dst,
        }
    }

    pub(crate) fn shift_r_var(
        size: OperandSize,
        kind: ShiftKind,
        src: VReg,
        count: VReg,
        dst: VReg,
    ) -> Inst {
        let src = Reg::reg_use(src);
        let count = Reg::reg_fixed_use(count, regs::rcx());
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::ShiftRVar {
            size,
            kind,
            src,
            count,
            dst,
        }
    }

    /// Does a comparison of dst - src for operands of size `size`, as stated by the machine
    /// instruction semantics. Be careful with the order of parameters!
    pub(crate) fn cmp_rmi_r(size: OperandSize, src: RegMemImm, dst: VReg) -> Inst {
        let dst = Reg::reg_use(dst); // `dst` is actually a source.
        Inst::CmpRmiR {
            size,
            src,
            dst,
            opcode: CmpOpcode::Cmp,
        }
    }

    /// Does a comparison of dst & src for operands of size `size`.
    pub(crate) fn test_rmi_r(size: OperandSize, src: RegMemImm, dst: VReg) -> Inst {
        let dst = Reg::reg_use(dst); // `dst` is actually a source.
        Inst::CmpRmiR {
            size,
            src,
            dst,
            opcode: CmpOpcode::Test,
        }
    }

    pub(crate) fn trap(trap_code: TrapCode) -> Inst {
        Inst::Ud2 {
            trap_code: trap_code,
        }
    }

    pub(crate) fn setcc(cc: CC, dst: VReg) -> Inst {
        let dst = Reg::reg_def(dst);
        Inst::Setcc { cc, dst }
    }

    pub(crate) fn cmove(size: OperandSize, cc: CC, src1: RegMem, src2: VReg, dst: VReg) -> Inst {
        debug_assert!(size.is_one_of(&[
            OperandSize::Size16,
            OperandSize::Size32,
            OperandSize::Size64
        ]));
        let src1 = src1.adjust_operand(|vreg| Reg::reg_use(vreg));
        let src2 = Reg::reg_use(src2);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::Cmove {
            size,
            cc,
            src1,
            src2,
            dst,
        }
    }

    pub(crate) fn xmm_cmove(
        size: OperandSize,
        cc: CC,
        src1: RegMem,
        src2: VReg,
        dst: VReg,
    ) -> Inst {
        let src1 = src1.adjust_operand(|vreg| Reg::reg_use(vreg));
        let src2 = Reg::reg_use(src2);
        let dst = Reg::reg_reuse_def(dst, 0);
        Inst::XmmCmove {
            size,
            cc,
            src1,
            src2,
            dst,
        }
    }

    pub(crate) fn push64(src: RegMemImm) -> Inst {
        Inst::Push64 { src }
    }

    pub(crate) fn pop64(dst: VReg) -> Inst {
        let dst = Reg::reg_def(dst);
        Inst::Pop64 { dst }
    }

    pub(crate) fn call_known(
        dest: ExternalName,
        opcode: Opcode,
        operands: Vec<Operand>,
        clobbers: &'static [PReg],
    ) -> Inst {
        Inst::CallKnown {
            dest,
            opcode,
            operands,
            clobbers,
        }
    }

    pub(crate) fn call_unknown(
        dest: RegMem,
        opcode: Opcode,
        operands: Vec<Operand>,
        clobbers: &'static [PReg],
    ) -> Inst {
        dest.assert_regclass_is(RegClass::I64);
        Inst::CallUnknown {
            dest,
            opcode,
            operands,
            clobbers,
        }
    }

    pub(crate) fn ret() -> Inst {
        Inst::Ret
    }

    pub(crate) fn epilogue_placeholder() -> Inst {
        Inst::EpiloguePlaceholder
    }

    pub(crate) fn jmp_known(dst: MachLabel) -> Inst {
        Inst::JmpKnown { dst }
    }

    pub(crate) fn jmp_if(cc: CC, taken: MachLabel) -> Inst {
        Inst::JmpIf { cc, taken }
    }

    pub(crate) fn jmp_cond(
        cc: CC,
        taken: MachLabel,
        not_taken: MachLabel,
        args: Vec<VReg>,
    ) -> Inst {
        let args = args.map(|vreg| Reg::reg_use_at_end(vreg)).collect();
        Inst::JmpCond {
            cc,
            taken,
            not_taken,
            args,
        }
    }

    pub(crate) fn jmp_unknown(target: RegMem) -> Inst {
        target.assert_regclass_is(RegClass::I64);
        Inst::JmpUnknown { target }
    }

    pub(crate) fn trap_if(cc: CC, trap_code: TrapCode) -> Inst {
        Inst::TrapIf { cc, trap_code }
    }

    /// Choose which instruction to use for loading a register value from memory. For loads smaller
    /// than 64 bits, this method expects a way to extend the value (i.e. [ExtKind::SignExtend],
    /// [ExtKind::ZeroExtend]); loads with no extension necessary will ignore this.
    pub(crate) fn load(
        ty: Type,
        from_addr: impl Into<SyntheticAmode>,
        to_reg: VReg,
        ext_kind: ExtKind,
    ) -> Inst {
        let rc = to_reg.to_reg().get_class();
        match rc {
            RegClass::I64 => {
                let ext_mode = match ty.bytes() {
                    1 => Some(ExtMode::BQ),
                    2 => Some(ExtMode::WQ),
                    4 => Some(ExtMode::LQ),
                    8 => None,
                    _ => unreachable!("the type should never use a scalar load: {}", ty),
                };
                if let Some(ext_mode) = ext_mode {
                    // Values smaller than 64 bits must be extended in some way.
                    match ext_kind {
                        ExtKind::SignExtend => {
                            Inst::movsx_rm_r(ext_mode, RegMem::mem(from_addr), to_reg)
                        }
                        ExtKind::ZeroExtend => {
                            Inst::movzx_rm_r(ext_mode, RegMem::mem(from_addr), to_reg)
                        }
                        ExtKind::None => panic!(
                            "expected an extension kind for extension mode: {:?}",
                            ext_mode
                        ),
                    }
                } else {
                    // 64-bit values can be moved directly.
                    Inst::mov64_m_r(from_addr, to_reg)
                }
            }
            RegClass::V128 => {
                let opcode = match ty {
                    types::F32 => SseOpcode::Movss,
                    types::F64 => SseOpcode::Movsd,
                    types::F32X4 => SseOpcode::Movups,
                    types::F64X2 => SseOpcode::Movupd,
                    _ if ty.is_vector() && ty.bits() == 128 => SseOpcode::Movdqu,
                    _ => unimplemented!("unable to load type: {}", ty),
                };
                Inst::xmm_unary_rm_r(opcode, RegMem::mem(from_addr), to_reg)
            }
            _ => panic!("unable to generate load for register class: {:?}", rc),
        }
    }

    /// Choose which instruction to use for storing a register value to memory.
    pub(crate) fn store(ty: Type, from_reg: VReg, to_addr: impl Into<SyntheticAmode>) -> Inst {
        let rc = from_reg.get_class();
        match rc {
            RegClass::I64 => Inst::mov_r_m(
                match ty {
                    types::B1 => OperandSize::Size8,
                    types::I32 | types::R32 => OperandSize::Size32,
                    types::I64 | types::R64 => OperandSize::Size64,
                    _ => unimplemented!("integer store of type: {}", ty),
                },
                from_reg,
                to_addr,
            ),
            RegClass::V128 => {
                let opcode = match ty {
                    types::F32 => SseOpcode::Movss,
                    types::F64 => SseOpcode::Movsd,
                    types::F32X4 => SseOpcode::Movups,
                    types::F64X2 => SseOpcode::Movupd,
                    _ if ty.is_vector() && ty.bits() == 128 => SseOpcode::Movdqu,
                    _ => unimplemented!("unable to store type: {}", ty),
                };
                Inst::xmm_mov_r_m(opcode, from_reg, to_addr)
            }
            _ => panic!("unable to generate store for register class: {:?}", rc),
        }
    }
}

// Inst helpers.

impl Inst {
    /// In certain cases, instructions of this format can act as a definition of an XMM register,
    /// producing a value that is independent of its initial value.
    ///
    /// For example, a vector equality comparison (`cmppd` or `cmpps`) that compares a register to
    /// itself will generate all ones as a result, regardless of its value. From the register
    /// allocator's point of view, we should (i) record the first register, which is normally a
    /// mod, as a def instead; and (ii) not record the second register as a use, because it is the
    /// same as the first register (already handled).
    fn produces_const(&self) -> bool {
        match self {
            Self::AluRmiR {
                op,
                src1,
                src2,
                dst,
                ..
            } => {
                src1.to_reg() == Some(src2.to_reg())
                    && (*op == AluRmiROpcode::Xor || *op == AluRmiROpcode::Sub)
            }

            Self::XmmRmR {
                op,
                src1,
                src2,
                dst,
                ..
            } => {
                src1.to_reg() == Some(src2.to_reg())
                    && (*op == SseOpcode::Xorps
                        || *op == SseOpcode::Xorpd
                        || *op == SseOpcode::Pxor
                        || *op == SseOpcode::Pcmpeqb
                        || *op == SseOpcode::Pcmpeqw
                        || *op == SseOpcode::Pcmpeqd
                        || *op == SseOpcode::Pcmpeqq)
            }

            Self::XmmRmRImm {
                op,
                src1,
                src2,
                dst,
                imm,
                ..
            } => {
                src1.to_reg() == Some(src2.to_reg())
                    && (*op == SseOpcode::Cmppd || *op == SseOpcode::Cmpps)
                    && *imm == FcmpImm::Equal.encode()
            }

            _ => false,
        }
    }

    /// Choose which instruction to use for comparing two values for equality.
    pub(crate) fn equals(ty: Type, from: RegMem, to: VReg) -> Inst {
        match ty {
            types::I8X16 | types::B8X16 => Inst::xmm_rm_r(SseOpcode::Pcmpeqb, from, to),
            types::I16X8 | types::B16X8 => Inst::xmm_rm_r(SseOpcode::Pcmpeqw, from, to),
            types::I32X4 | types::B32X4 => Inst::xmm_rm_r(SseOpcode::Pcmpeqd, from, to),
            types::I64X2 | types::B64X2 => Inst::xmm_rm_r(SseOpcode::Pcmpeqq, from, to),
            types::F32X4 => Inst::xmm_rm_r_imm(
                SseOpcode::Cmpps,
                from,
                to,
                FcmpImm::Equal.encode(),
                OperandSize::Size32,
            ),
            types::F64X2 => Inst::xmm_rm_r_imm(
                SseOpcode::Cmppd,
                from,
                to,
                FcmpImm::Equal.encode(),
                OperandSize::Size32,
            ),
            _ => unimplemented!("unimplemented type for Inst::equals: {}", ty),
        }
    }

    /// Choose which instruction to use for computing a bitwise AND on two values.
    pub(crate) fn and(ty: Type, src1: RegMem, src2: VReg, dst: VReg) -> Inst {
        match ty {
            types::F32X4 => Inst::xmm_rm_r(SseOpcode::Andps, src1, src2, dst),
            types::F64X2 => Inst::xmm_rm_r(SseOpcode::Andpd, src1, src2, dst),
            _ if ty.is_vector() && ty.bits() == 128 => {
                Inst::xmm_rm_r(SseOpcode::Pand, src1, src2, dst)
            }
            _ => unimplemented!("unimplemented type for Inst::and: {}", ty),
        }
    }

    /// Choose which instruction to use for computing a bitwise AND NOT on two values.
    pub(crate) fn and_not(ty: Type, src1: RegMem, src2: VReg, dst: VReg) -> Inst {
        match ty {
            types::F32X4 => Inst::xmm_rm_r(SseOpcode::Andnps, src1, src2, dst),
            types::F64X2 => Inst::xmm_rm_r(SseOpcode::Andnpd, src1, src2, dst),
            _ if ty.is_vector() && ty.bits() == 128 => {
                Inst::xmm_rm_r(SseOpcode::Pandn, src1, src2, dst)
            }
            _ => unimplemented!("unimplemented type for Inst::and_not: {}", ty),
        }
    }

    /// Choose which instruction to use for computing a bitwise OR on two values.
    pub(crate) fn or(ty: Type, src1: RegMem, src2: VReg, dst: VReg) -> Inst {
        match ty {
            types::F32X4 => Inst::xmm_rm_r(SseOpcode::Orps, src1, src2, dst),
            types::F64X2 => Inst::xmm_rm_r(SseOpcode::Orpd, src1, src2, dst),
            _ if ty.is_vector() && ty.bits() == 128 => {
                Inst::xmm_rm_r(SseOpcode::Por, src1, src2, dst)
            }
            _ => unimplemented!("unimplemented type for Inst::or: {}", ty),
        }
    }

    /// Choose which instruction to use for computing a bitwise XOR on two values.
    pub(crate) fn xor(ty: Type, src1: RegMem, src2: VReg, dst: VReg) -> Inst {
        match ty {
            types::F32X4 => Inst::xmm_rm_r(SseOpcode::Xorps, src1, src2, dst),
            types::F64X2 => Inst::xmm_rm_r(SseOpcode::Xorpd, src1, src2, dst),
            _ if ty.is_vector() && ty.bits() == 128 => {
                Inst::xmm_rm_r(SseOpcode::Pxor, src1, src2, dst)
            }
            _ => unimplemented!("unimplemented type for Inst::xor: {}", ty),
        }
    }
}

//=============================================================================
// Instructions: printing

fn show_reg(reg: Reg, size: u8) -> String {
    if let Some(preg) = reg.as_preg() {
        Inst::reg_name(preg, size).to_string()
    } else if let Some(slot) = reg.as_spillslot() {
        format!("{}", slot)
    } else if let Some(op) = reg.as_operand() {
        format!("{}", op)
    }
}

impl fmt::Debug for Inst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn suffix_lq(size: OperandSize) -> &'static str {
            match size {
                OperandSize::Size32 => "l",
                OperandSize::Size64 => "q",
                _ => unreachable!(),
            }
        }

        fn suffix_lqb(size: OperandSize, is_8: bool) -> &'static str {
            match (size, is_8) {
                (_, true) => "b",
                (OperandSize::Size32, false) => "l",
                (OperandSize::Size64, false) => "q",
                _ => unreachable!(),
            }
        }

        fn size_lqb(size: OperandSize, is_8: bool) -> u8 {
            if is_8 {
                return 1;
            }
            size.to_bytes()
        }

        fn suffix_bwlq(size: OperandSize) -> &'static str {
            match size {
                OperandSize::Size8 => "b",
                OperandSize::Size16 => "w",
                OperandSize::Size32 => "l",
                OperandSize::Size64 => "q",
            }
        }

        match self {
            Inst::Nop { len } => write!(f, "nop len={}", len),

            Inst::AluRmiR {
                size,
                op,
                src1,
                src2,
                dst,
            } => write!(
                f,
                "{:?}{} {}, {}, {}",
                op,
                src1,
                show_reg(*src1, size_lqb(*size, op.is_8bit())),
                show_reg(*src2, size_lqb(*size, op.is_8bit())),
                show_reg(*dst, size_lqb(*size, op.is_8bit())),
            ),

            Inst::UnaryRmR { src, dst, op, size } => write!(
                f,
                "{:?}{} {}, {}",
                op,
                suffix_bwlq(*size),
                src,
                show_reg(*dst, size.to_bytes()),
            ),

            Inst::Not { size, src, dst } => write!(
                f,
                "not{} {}, {}",
                suffix_bwlq(*size),
                show_reg(*src, size.to_bytes()),
                show_reg(*dst, size.to_bytes()),
            ),

            Inst::Neg { size, src, dst } => write!(
                f,
                "neg{} {}, {}",
                suffix_bwlq(*size),
                show_reg(*src, size.to_bytes()),
                show_reg(*dst, size.to_bytes()),
            ),

            Inst::Div {
                size,
                signed,
                divisor,
                in_lo,
                in_hi,
                out_div,
                out_rem,
                ..
            } => write!(
                f,
                "{} divisor={}, in_lo={} in_hi={} out_div={} out_rem={}",
                if *signed { "idiv" } else { "div" },
                divisor,
                show_reg(*in_lo, size.to_bytes()),
                show_reg(*in_hi, size.to_bytes()),
                show_reg(*out_div, size.to_bytes()),
                show_reg(*out_rem, size.to_bytes()),
            ),

            Inst::Mul {
                size,
                signed,
                rhs,
                src,
                out_lo,
                out_hi,
                ..
            } => write!(
                f,
                "{} {}, {}, out_lo={}, out_hi={}",
                if *signed { "imul" } else { "mul" },
                rhs,
                show_reg(*src, size.to_bytes()),
                show_reg(*out_lo, size.to_bytes()),
                show_reg(*out_hi, size.to_bytes()),
            ),

            Inst::CheckedDivOrRemSeq {
                kind,
                size,
                divisor,
                tmp,
                dividend,
                out_lo,
                out_hi,
                ..
            } => write!(
                f,
                "{} dividend={}, divisor={}, out_lo={}, out_hi={}, tmp={}",
                match kind {
                    DivOrRemKind::SignedDiv => "sdiv",
                    DivOrRemKind::UnsignedDiv => "udiv",
                    DivOrRemKind::SignedRem => "srem",
                    DivOrRemKind::UnsignedRem => "urem",
                },
                show_reg(*dividend, size.to_bytes()),
                show_reg(*divisor, size.to_bytes()),
                show_reg(*out_lo, size.to_bytes()),
                show_reg(*out_hi, size.to_bytes()),
                show_reg(*tmp, size.to_bytes()),
            ),

            Inst::SignExtendData { size, src, dst } => write!(
                f,
                "{} {}, {}",
                match size {
                    OperandSize::Size8 => "cbw",
                    OperandSize::Size16 => "cwd",
                    OperandSize::Size32 => "cdq",
                    OperandSize::Size64 => "cqo",
                },
                show_reg(*src, size.to_bytes()),
                show_reg(*dst, size.to_bytes()),
            ),

            Inst::XmmUnaryRmR { op, src, dst, .. } => {
                write!(f, "{:?} {}, {}", op, src, show_reg(*dst, 0),)
            }

            Inst::XmmMovRM { op, src, dst, .. } => {
                write!(f, "{:?} {}, {}", op, show_reg(*src, 0), dst)
            }

            Inst::XmmRmR {
                op,
                src1,
                src2,
                dst,
                ..
            } => {
                write!(
                    f,
                    "{:?} {}, {}, {}",
                    op,
                    src1,
                    show_reg(*src2, 0),
                    show_reg(*dst, 0),
                )
            }

            Inst::XmmMinMaxSeq {
                size,
                is_min,
                lhs,
                rhs,
                dst,
            } => write!(
                f,
                "{} f{} {}, {}, {}",
                if *is_min {
                    "xmm min seq "
                } else {
                    "xmm max seq "
                },
                size,
                show_reg(*lhs, 0),
                show_reg(*rhs, 0),
                show_reg(*dst, 0),
            ),

            Inst::XmmRmRImm {
                op,
                src1,
                src2,
                dst,
                imm,
                size,
                ..
            } => write!(
                f,
                "{:?}{} ${}, {}, {}, {}",
                op,
                if *size == OperandSize::Size64 {
                    ".w"
                } else {
                    ""
                },
                imm,
                src1,
                show_reg(*src2, 0),
                show_reg(*dst, 0),
            ),

            Inst::XmmUninitializedValue { dst } => {
                write!(f, "uninit {}", show_reg(*dst, 0))
            }

            Inst::XmmLoadConst { src, dst, .. } => {
                write!(f, "load_const {:?}, {}", src, show_reg(*dst, 0))
            }

            Inst::XmmToGpr {
                op,
                src,
                dst,
                dst_size,
            } => {
                let dst_size = dst_size.to_bytes();
                write!(
                    f,
                    "{:?} {}, {}",
                    op,
                    show_reg(*src, 0),
                    show_reg(*dst, dst_size),
                )
            }

            Inst::GprToXmm {
                op,
                src,
                src_size,
                dst,
            } => write!(
                f,
                "{:?} {}, {}",
                op,
                show_reg(*src, src_size.to_bytes()),
                show_reg(*dst, 0),
            ),

            Inst::XmmCmpRmR { op, src, dst } => {
                write!(f, "{:?} {}, {}", op, show_reg(*src, 0), show_reg(*dst, 0),)
            }

            Inst::CvtUint64ToFloatSeq {
                src, dst, dst_size, ..
            } => write!(
                f,
                "u64_to_{}_seq {}, {}",
                if *dst_size == OperandSize::Size64 {
                    "f64"
                } else {
                    "f32"
                },
                show_reg(*src, 0),
                show_reg(*dst, 0),
            ),

            Inst::CvtFloatToSintSeq {
                src,
                dst,
                src_size,
                dst_size,
                ..
            } => write!(
                f,
                "cvt_float{}_to_sint{}_seq {}, {}",
                src_size.to_bits(),
                dst_size.to_bits(),
                show_reg(*src, 0),
                show_reg(*dst, 0),
            ),

            Inst::CvtFloatToUintSeq {
                src,
                dst,
                src_size,
                dst_size,
                ..
            } => write!(
                f,
                "cvt_float{}_to_uint{}_seq {}, {}",
                src_size.to_bits(),
                dst_size.to_bits(),
                show_reg(*src, 8),
                show_reg(*dst, dst_size.to_bytes()),
            ),

            Inst::Imm {
                dst_size,
                simm64,
                dst,
            } => {
                if *dst_size == OperandSize::Size64 {
                    write!(f, "movabsq ${}, {}", *simm64 as i64, show_reg(*dst, 8))
                } else {
                    write!(
                        f,
                        "movl ${}, {}",
                        (*simm64 as u32) as i32,
                        show_reg(*dst, 4)
                    )
                }
            }

            Inst::MovRR { size, src, dst } => write!(
                f,
                "mov{} {}, {}",
                suffix_lq(*size),
                show_reg(*src, size.to_bytes()),
                show_reg(*dst, size.to_bytes())
            ),

            Inst::MovzxRmR {
                ext_mode, src, dst, ..
            } => {
                if *ext_mode == ExtMode::LQ {
                    write!(
                        f,
                        "movl {}, {}",
                        show_reg(*src, ext_mode.src_size()),
                        show_reg(*dst, 4),
                    )
                } else {
                    write!(
                        f,
                        "movz {}, {}",
                        show_reg(*src, ext_mode.src_size()),
                        show_reg(*dst, ext_mode.dst_size()),
                    )
                }
            }

            Inst::Mov64MR { src, dst, .. } => {
                write!(f, "movq {}, {}", show_reg(*src, 0), show_reg(*dst, 0),)
            }

            Inst::LoadEffectiveAddress { addr, dst } => {
                write!(f, "lea {:?}, {}", addr, show_reg(*dst, 0),)
            }

            Inst::MovsxRmR {
                ext_mode, src, dst, ..
            } => write!(
                f,
                "movs{:?} {}, {}",
                ext_mode,
                show_reg(*src, ext_mode.src_size()),
                show_reg(*dst, ext_mode.dst_size())
            ),

            Inst::MovRM { size, src, dst, .. } => write!(
                f,
                "mov{} {}, {:?}",
                suffix_bwlq(*size),
                show_reg(*src, size.to_bytes()),
                dst,
            ),

            Inst::ShiftRImm {
                size,
                kind,
                num_bits,
                src,
                dst,
            } => {
                write!(
                    f,
                    "{:?}{} ${}, {}, {}",
                    kind,
                    suffix_bwlq(*size),
                    num_bits,
                    show_reg(src, size.to_bytes()),
                    show_reg(dst, size.to_bytes())
                )
            }

            Inst::ShiftRVar {
                size,
                kind,
                src,
                count,
                dst,
            } => {
                write!(
                    f,
                    "{:?}{} {}, {}, {}",
                    kind,
                    suffix_bwlq(*size),
                    show_reg(*src, size.to_bytes()),
                    show_reg(*count, size.to_bytes()),
                    show_reg(*dst, size.to_bytes()),
                )
            }

            Inst::XmmRmiReg {
                opcode,
                src1,
                src2,
                dst,
            } => {
                write!(
                    f,
                    "{:?} {:?}, {}, {}",
                    opcode,
                    src1,
                    show_reg(*src2, 0),
                    show_reg(*dst, 0),
                )
            }

            Inst::CmpRmiR {
                size,
                src,
                dst,
                opcode,
            } => {
                let op = match opcode {
                    CmpOpcode::Cmp => "cmp",
                    CmpOpcode::Test => "test",
                };
                write!(
                    f,
                    "{:?}{} {}, {}",
                    op,
                    suffix_bwlq(*size),
                    show_reg(*src, size.to_bytes()),
                    show_reg(*dst, size.to_bytes())
                )
            }

            Inst::Setcc { cc, dst } => write!(f, "set{:?} {}", cc, show_reg(*dst, 1)),

            Inst::Cmove {
                size,
                cc,
                src1,
                src2,
                dst,
            } => write!(
                f,
                "cmov{:?}{} {}, {}, {}",
                cc,
                suffix_bwlq(*size),
                src1,
                show_reg(*src2, size.to_bytes()),
                show_reg(*dst, size.to_bytes()),
            ),

            Inst::XmmCmove {
                size,
                cc,
                src1,
                src2,
                dst,
            } => {
                write!(
                    f,
                    "use {}; j{:?} $next; mov{} {}, {}; $next: ",
                    show_reg(*src2, size.to_bytes()),
                    cc.invert(),
                    if *size == OperandSize::Size64 {
                        "sd"
                    } else {
                        "ss"
                    },
                    src1,
                    show_reg(*dst, size.to_bytes()),
                )
            }

            Inst::Push64 { src } => {
                write!(f, "pushq {}", show_reg(*src, 0))
            }

            Inst::Pop64 { dst } => {
                write!(f, "popq {}", show_reg(*dst, 0))
            }

            Inst::CallKnown { dest, .. } => write!(f, "call {:?}", dest),

            Inst::CallUnknown { dest, .. } => write!(f, "call *{}", show_reg(*dest, 0),),

            Inst::Ret => write!(f, "ret"),

            Inst::EpiloguePlaceholder => write!(f, "epilogue placeholder"),

            Inst::JmpKnown { dst } => {
                write!(f, "jmp {:?}", dst)
            }

            Inst::JmpIf { cc, taken } => write!(f, "j{:?} {:?}", cc, taken,),

            Inst::JmpCond {
                cc,
                taken,
                not_taken,
                ..
            } => write!(f, "j{} {:?}; j {:?}", cc, taken, not_taken,),

            Inst::JmpTableSeq { idx, .. } => {
                write!(f, "br_table {}", show_reg(*idx, 0))
            }

            Inst::JmpUnknown { target } => write!(f, "jmp *{}", show_reg(*target, 0),),

            Inst::TrapIf { cc, trap_code, .. } => {
                write!(f, "j{:?} ; ud2 {} ;", cc.invert(), trap_code)
            }

            Inst::LoadExtName {
                dst, name, offset, ..
            } => write!(
                f,
                "load_ext_name {}+{}, {}",
                name,
                offset,
                show_reg(*dst, 8),
            ),

            Inst::LockCmpxchg {
                ty,
                src,
                dst,
                expected,
                actual,
                ..
            } => {
                let size = ty.bytes() as u8;
                write!(
                    f,
                    "lock cmpxchg{} {}, {}, expected_in={}, actual_out={}",
                    suffix_bwlq(OperandSize::from_bytes(size as u32)),
                    show_reg(*src, size),
                    show_reg(*dst, size),
                    show_reg(*expected, size),
                    show_reg(*actual, size),
                )
            }

            Inst::AtomicRmwSeq {
                ty,
                op,
                addr,
                src,
                scratch,
                old_out,
            } => {
                write!(
                    f,
                    "atomic_rmw_seq {:?} addr={} src={} scratch={} old_out={}",
                    op, addr, src, scratch, old_out,
                )
            }

            Inst::Fence { kind } => write!(
                f,
                "{}",
                match kind {
                    FenceKind::MFence => "mfence",
                    FenceKind::LFence => "lfence",
                    FenceKind::SFence => "sfence",
                }
            ),

            Inst::VirtualSPOffsetAdj { offset } => write!(f, "virtual_sp_offset_adjust {}", offset),

            Inst::Hlt => write!(f, "hlt"),

            Inst::Ud2 { trap_code } => write!(f, "ud2 {}", trap_code),

            Inst::ElfTlsGetAddr { ref symbol, dst } => {
                write!(f, "elf_tls_get_addr {:?}, {}", symbol, show_reg(*dst, 0))
            }

            Inst::MachOTlsGetAddr { ref symbol, dst } => {
                write!(f, "macho_tls_get_addr {:?}, {}", symbol, show_reg(*dst, 0))
            }

            Inst::ValueLabelMarker { label, reg } => {
                write!(f, "value_label {:?}, {}", label, show_reg(reg, 0))
            }

            Inst::Unwind { inst } => {
                write!(f, "unwind {:?}", inst)
            }

            Inst::RegConstraints { ref args } => {
                write!(f, "reg_constraints {:?}", args)
            }
        }
    }
}

fn x64_visit_regs<F: FnMut(&mut Reg)>(inst: &mut Inst, mut f: F) {
    match inst {
        &mut Inst::AluRmiR {
            ref mut src1,
            ref mut src2,
            ref mut dst,
            ..
        } => {
            if inst.produces_const() {
                // No need to account for src, since src == dst.
                f(dst);
            } else {
                // src2 first, so that dst can refer to it with
                // constant operand index 0 to reuse its register.
                f(src2);
                src1.visit_regs(&mut f);
                f(dst);
            }
        }
        &mut Inst::Not {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        &mut Inst::Neg {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        Inst::Div {
            ref mut divisor,
            ref mut in_lo,
            ref mut in_hi,
            ref mut out_div,
            ref mut out_rem,
            ..
        } => {
            divisor.visit_regs(&mut f);
            f(in_lo);
            f(in_hi);
            f(out_div);
            f(out_rem);
        }
        &mut Inst::Mul {
            ref mut rhs,
            ref mut src,
            ref mut out_lo,
            ref mut out_hi,
            ..
        } => {
            rhs.visit_regs(&mut f);
            f(src);
            f(out_lo);
            f(out_hi);
        }
        &mut Inst::CheckedDivOrRemSeq {
            ref mut divisor,
            ref mut tmp,
            ref mut dividend,
            ref mut out_lo,
            ref mut out_hi,
            ..
        } => {
            f(divisor);
            f(tmp);
            f(dividend);
            f(out_lo);
            f(out_hi);
        }
        &mut Inst::SignExtendData {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        &mut Inst::UnaryRmR {
            ref mut src,
            ref mut dst,
            ..
        }
        | &mut Inst::XmmUnaryRmR {
            ref mut src,
            ref mut dst,
            ..
        } => {
            src.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::XmmRmR {
            ref mut src1,
            ref mut src2,
            ref mut dst,
            ..
        } => {
            if inst.produces_const() {
                // No need to account for src, since src == dst.
                f(dst);
            } else {
                f(src2);
                src1.visit_regs(&mut f);
                f(dst);
            }
        }
        &mut Inst::XmmRmRImm {
            op,
            ref mut src1,
            ref mut src2,
            ref mut dst,
            ..
        } => {
            if inst.produces_const() {
                // No need to account for src, since src == dst.
                f(dst);
            } else {
                f(src2);
                src1.visit_regs(&mut f);
                f(dst);
            }
        }
        &mut Inst::XmmUninitializedValue { ref mut dst } => {
            f(dst);
        }
        &mut Inst::XmmLoadConst { ref mut dst, .. } => {
            f(dst);
        }
        &mut Inst::XmmMinMaxSeq {
            ref mut lhs,
            ref mut rhs,
            ref mut dst,
            ..
        } => {
            f(lhs);
            f(rhs);
            f(dst);
        }
        &mut Inst::XmmRmiReg {
            ref mut src1,
            ref mut src2,
            ref mut dst,
            ..
        } => {
            f(src2);
            src1.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::XmmMovRM {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            dst.visit_regs(&mut f);
        }
        &mut Inst::XmmCmpRmR {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        &mut Inst::Imm { ref mut dst, .. } => {
            f(dst);
        }
        &mut Inst::MovRR {
            ref mut src,
            ref mut dst,
            ..
        }
        | &mut Inst::XmmToGpr {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        &mut Inst::GprToXmm {
            ref mut src,
            ref mut dst,
            ..
        } => {
            src.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::CvtUint64ToFloatSeq {
            ref mut src,
            ref mut dst,
            ref mut tmp_gpr1,
            ref mut tmp_gpr2,
            ..
        } => {
            f(src);
            f(dst);
            f(tmp_gpr1);
            f(tmp_gpr2);
        }
        &mut Inst::CvtFloatToSintSeq {
            ref mut src,
            ref mut dst,
            ref mut tmp_xmm,
            ref mut tmp_gpr,
            ..
        }
        | &mut Inst::CvtFloatToUintSeq {
            ref mut src,
            ref mut dst,
            ref mut tmp_gpr,
            ref mut tmp_xmm,
            ..
        } => {
            f(src);
            f(dst);
            f(tmp_gpr);
            f(tmp_xmm);
        }
        &mut Inst::MovzxRmR {
            ref mut src,
            ref mut dst,
            ..
        } => {
            src.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::Mov64MR {
            ref mut src,
            ref mut dst,
            ..
        }
        | &mut Inst::LoadEffectiveAddress {
            addr: ref mut src,
            ref mut dst,
        } => {
            src.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::MovsxRmR {
            ref mut src,
            ref mut dst,
            ..
        } => {
            src.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::MovRM {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            dst.visit_regs(&mut f);
        }
        &mut Inst::ShiftRImm {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        &mut Inst::ShiftRVar {
            ref mut src,
            ref mut count,
            ref mut dst,
            ..
        } => {
            f(src);
            f(count);
            f(dst);
        }
        &mut Inst::CmpRmiR {
            ref mut src,
            ref mut dst,
            ..
        } => {
            f(src);
            f(dst);
        }
        &mut Inst::Setcc { ref mut dst, .. } => {
            f(dst);
        }
        &mut Inst::Cmove {
            ref mut src1,
            ref mut src2,
            ref mut dst,
            ..
        }
        | &mut Inst::XmmCmove {
            ref mut src1,
            ref mut src2,
            ref mut dst,
            ..
        } => {
            f(src2);
            src1.visit_regs(&mut f);
            f(dst);
        }
        &mut Inst::Push64 { ref mut src } => {
            f(src);
        }
        &mut Inst::Pop64 { ref mut dst } => {
            f(dst);
        }

        &mut Inst::CallKnown { ref mut operands } => {
            for op in operands {
                f(op);
            }
        }

        &mut Inst::CallUnknown {
            ref mut dest,
            ref mut operands,
            ..
        } => {
            dest.visit_regs(&mut f);
            for op in operands {
                f(op);
            }
        }

        &mut Inst::JmpTableSeq {
            ref mut idx,
            ref mut tmp1,
            ref mut tmp2,
            ref mut args,
            ..
        } => {
            f(idx);
            f(tmp1);
            f(tmp2);
            for arg in args {
                f(arg);
            }
        }

        &mut Inst::JmpUnknown {
            ref mut target,
            ref mut args,
        } => {
            target.visit_regs(&mut f);
            for arg in args {
                f(arg);
            }
        }

        &mut Inst::LoadExtName { ref mut dst, .. } => {
            f(dst);
        }

        &mut Inst::LockCmpxchg {
            ref mut src,
            ref mut dst,
            ref mut expected,
            ref mut actual,
            ..
        } => {
            f(src);
            dst.visit_regs(&mut f);
            f(expected);
            f(actual);
        }

        &mut Inst::AtomicRmwSeq {
            ref mut addr,
            ref mut src,
            ref mut scratch,
            ref mut old_out,
            ..
        } => {
            f(addr);
            f(src);
            f(scratch);
            f(old_out);
        }

        &mut Inst::JmpCond { ref mut args, .. } => {
            for arg in args {
                f(arg);
            }
        }

        &mut Inst::JmpKnown { ref mut args, .. } => {
            for arg in args {
                f(arg);
            }
        }

        &mut Inst::Ret
        | &mut Inst::EpiloguePlaceholder
        | &mut Inst::JmpIf { .. }
        | &mut Inst::Nop { .. }
        | &mut Inst::TrapIf { .. }
        | &mut Inst::VirtualSPOffsetAdj { .. }
        | &mut Inst::Hlt
        | &mut Inst::Ud2 { .. }
        | &mut Inst::Fence { .. } => {
            // No registers are used.
        }

        &mut Inst::ElfTlsGetAddr { ref mut dst, .. }
        | &mut Inst::MachOTlsGetAddr { ref mut dst, .. } => {
            f(dst);
        }

        &mut Inst::ValueLabelMarker { ref mut reg, .. } => {
            f(reg);
        }

        &mut Inst::Unwind { .. } => {}

        &mut Inst::RegConstraints { ref mut args } => {
            for arg in args {
                f(arg);
            }
        }
    }
}

impl Amode {
    /// Offset the amode by a fixed offset.
    pub(crate) fn offset(&self, offset: u32) -> Self {
        let mut ret = self.clone();
        match &mut ret {
            &mut Amode::ImmReg { ref mut simm32, .. } => *simm32 += offset,
            &mut Amode::ImmRegRegShift { ref mut simm32, .. } => *simm32 += offset,
            _ => panic!("Cannot offset amode: {:?}", self),
        }
        ret
    }
}

//=============================================================================
// Instructions: misc functions and external interface

impl MachInst for Inst {
    fn visit_regs<F: FnMut(&mut Reg)>(&mut self, f: F) {
        x64_visit_regs(self, f)
    }

    fn clobbers(&self) -> &[PReg] {
        match self {
            &Inst::CallKnown { ref clobbers, .. } | &Inst::CallUnknown { ref clobbers, .. } => {
                &clobbers[..]
            }
            &Inst::ElfTlsGetAddr { .. } | &Inst::MachOTlsGetAddr { .. } => {
                // All caller-saves are clobbered.
                //
                // We use the SysV calling convention here because the
                // pseudoinstruction (and relocation that it emits) is specific to
                // ELF systems; other x86-64 targets with other conventions (i.e.,
                // Windows) use different TLS strategies.
                X64ABIMachineSpec::get_clobbers(CallConv::SystemV)
            }
            _ => &[],
        }
    }

    fn is_move(&self) -> Option<(Reg, Reg)> {
        match self {
            // Note (carefully!) that a 32-bit mov *isn't* a no-op since it zeroes
            // out the upper 32 bits of the destination.  For example, we could
            // conceivably use `movl %reg, %reg` to zero out the top 32 bits of
            // %reg.
            Self::MovRR { size, src, dst, .. } if *size == OperandSize::Size64 => {
                Some((*dst, *src))
            }
            // Note as well that MOVS[S|D] when used in the `XmmUnaryRmR` context are pure moves of
            // scalar floating-point values (and annotate `dst` as `def`s to the register allocator)
            // whereas the same operation in a packed context, e.g. `XMM_RM_R`, is used to merge a
            // value into the lowest lane of a vector (not a move).
            Self::XmmUnaryRmR { op, src, dst, .. }
                if *op == SseOpcode::Movss
                    || *op == SseOpcode::Movsd
                    || *op == SseOpcode::Movaps
                    || *op == SseOpcode::Movapd
                    || *op == SseOpcode::Movups
                    || *op == SseOpcode::Movupd
                    || *op == SseOpcode::Movdqa
                    || *op == SseOpcode::Movdqu =>
            {
                if let RegMem::Reg { reg } = src {
                    Some((*dst, *reg))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn is_term<'a>(&'a self) -> MachTerminator<'a> {
        match self {
            // Interesting cases.
            &Self::Ret | &Self::EpiloguePlaceholder => MachTerminator::Ret,
            &Self::JmpKnown { dst } => MachTerminator::Uncond(dst),
            &Self::JmpCond {
                taken, not_taken, ..
            } => MachTerminator::Cond(taken, not_taken),
            &Self::JmpTableSeq {
                ref targets_for_term,
                ..
            } => MachTerminator::Indirect(&targets_for_term[..]),
            // All other cases are boring.
            _ => MachTerminator::None,
        }
    }

    fn blockparam_offset(&self) -> usize {
        match self {
            &Self::JmpCond { .. } => 0,
            &Self::JmpKnown { .. } => 0,
            &Self::JmpTableSeq { .. } => {
                // Three args before the blockparams: idx, tmp1, tmp2
                3
            }
            // No other insts are terminators according to the above
            // `is_term()`.
            _ => 0,
        }
    }

    fn is_ret(&self) -> bool {
        match self {
            &Self::Ret => true,
            _ => false,
        }
    }

    fn is_call(&self) -> bool {
        match self {
            &Self::CallKnown { .. } | &Self::CallUnknown { .. } => true,
            _ => false,
        }
    }

    fn is_safepoint(&self) -> bool {
        self.is_call()
    }

    fn stack_op_info(&self) -> Option<MachInstStackOpInfo> {
        match self {
            Self::VirtualSPOffsetAdj { offset } => Some(MachInstStackOpInfo::NomSPAdj(*offset)),
            Self::MovRM {
                size: OperandSize::Size8,
                src,
                dst: SyntheticAmode::NominalSPOffset { simm32 },
            } => Some(MachInstStackOpInfo::StoreNomSPOff(*src, *simm32 as i64)),
            Self::Mov64MR {
                src: SyntheticAmode::NominalSPOffset { simm32 },
                dst,
            } => Some(MachInstStackOpInfo::LoadNomSPOff(
                dst.to_reg(),
                *simm32 as i64,
            )),
            _ => None,
        }
    }

    fn gen_move(dst_reg: Reg, src_reg: Reg, ty: Type) -> Inst {
        let rc_dst = dst_reg.class();
        let rc_src = src_reg.class();
        // If this isn't true, we have gone way off the rails.
        debug_assert!(rc_dst == rc_src);
        match rc_dst {
            RegClass::Int => Inst::MovRR {
                size: OperandSize::Size64,
                src: src_reg,
                dst: dst_reg,
            },
            RegClass::Float => {
                // The Intel optimization manual, in "3.5.1.13 Zero-Latency MOV Instructions",
                // doesn't include MOVSS/MOVSD as instructions with zero-latency. Use movaps for
                // those, which may write more lanes that we need, but are specified to have
                // zero-latency.
                let op = match ty {
                    types::F32 | types::F64 | types::F32X4 => SseOpcode::Movaps,
                    types::F64X2 => SseOpcode::Movapd,
                    _ if ty.is_vector() && ty.bits() == 128 => SseOpcode::Movdqa,
                    _ => unimplemented!("unable to move type: {}", ty),
                };
                Inst::XmmUnaryRmR {
                    op,
                    src: src_reg,
                    dst: dst_reg,
                }
            }
            _ => panic!("gen_move(x64): unhandled regclass {:?}", rc_dst),
        }
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        Inst::nop(std::cmp::min(preferred_size, 15) as u8)
    }

    fn gen_reg_constraint_inst(args: Vec<Operand>) -> Inst {
        Inst::RegConstraints { args }
    }

    fn rc_for_type(ty: Type) -> CodegenResult<(&'static [RegClass], &'static [Type])> {
        match ty {
            types::I8 => Ok((&[RegClass::I64], &[types::I8])),
            types::I16 => Ok((&[RegClass::I64], &[types::I16])),
            types::I32 => Ok((&[RegClass::I64], &[types::I32])),
            types::I64 => Ok((&[RegClass::I64], &[types::I64])),
            types::B1 => Ok((&[RegClass::I64], &[types::B1])),
            types::B8 => Ok((&[RegClass::I64], &[types::B8])),
            types::B16 => Ok((&[RegClass::I64], &[types::B16])),
            types::B32 => Ok((&[RegClass::I64], &[types::B32])),
            types::B64 => Ok((&[RegClass::I64], &[types::B64])),
            types::R32 => panic!("32-bit reftype pointer should never be seen on x86-64"),
            types::R64 => Ok((&[RegClass::I64], &[types::R64])),
            types::F32 => Ok((&[RegClass::V128], &[types::F32])),
            types::F64 => Ok((&[RegClass::V128], &[types::F64])),
            types::I128 => Ok((&[RegClass::I64, RegClass::I64], &[types::I64, types::I64])),
            types::B128 => Ok((&[RegClass::I64, RegClass::I64], &[types::B64, types::B64])),
            _ if ty.is_vector() => {
                assert!(ty.bits() <= 128);
                Ok((&[RegClass::V128], &[types::I8X16]))
            }
            types::IFLAGS | types::FFLAGS => Ok((&[RegClass::I64], &[types::I64])),
            _ => Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {}",
                ty
            ))),
        }
    }

    fn gen_jump(label: MachLabel) -> Inst {
        Inst::jmp_known(label)
    }

    fn gen_constant<F: FnMut(Type) -> VReg>(
        to_regs: ValueRegs<VReg>,
        value: u128,
        ty: Type,
        mut alloc_tmp: F,
    ) -> SmallVec<[Self; 4]> {
        let mut ret = SmallVec::new();
        if ty == types::I128 {
            ret.push(Inst::imm(
                OperandSize::Size64,
                value as u64,
                to_regs.regs()[0],
            ));
            ret.push(Inst::imm(
                OperandSize::Size64,
                (value >> 64) as u64,
                to_regs.regs()[1],
            ));
        } else {
            let to_reg = to_regs
                .only_reg()
                .expect("multi-reg values not supported on x64");
            if ty == types::F32 {
                if value == 0 {
                    ret.push(Inst::xmm_rm_r(
                        SseOpcode::Xorps,
                        RegMem::reg(to_reg),
                        to_reg,
                    ));
                } else {
                    let tmp = alloc_tmp(types::I32);
                    ret.push(Inst::imm(OperandSize::Size32, value as u64, tmp));

                    ret.push(Inst::gpr_to_xmm(
                        SseOpcode::Movd,
                        RegMem::reg(tmp),
                        OperandSize::Size32,
                        to_reg,
                    ));
                }
            } else if ty == types::F64 {
                if value == 0 {
                    ret.push(Inst::xmm_rm_r(
                        SseOpcode::Xorpd,
                        RegMem::reg(to_reg),
                        to_reg,
                    ));
                } else {
                    let tmp = alloc_tmp(types::I64);
                    ret.push(Inst::imm(OperandSize::Size64, value as u64, tmp));

                    ret.push(Inst::gpr_to_xmm(
                        SseOpcode::Movq,
                        RegMem::reg(tmp),
                        OperandSize::Size64,
                        to_reg,
                    ));
                }
            } else {
                // Must be an integer type.
                debug_assert!(
                    ty == types::B1
                        || ty == types::I8
                        || ty == types::B8
                        || ty == types::I16
                        || ty == types::B16
                        || ty == types::I32
                        || ty == types::B32
                        || ty == types::I64
                        || ty == types::B64
                        || ty == types::R32
                        || ty == types::R64
                );
                // Immediates must be 32 or 64 bits.
                // Smaller types are widened.
                let size = match OperandSize::from_ty(ty) {
                    OperandSize::Size64 => OperandSize::Size64,
                    _ => OperandSize::Size32,
                };
                if value == 0 {
                    ret.push(Inst::alu_rmi_r(
                        size,
                        AluRmiROpcode::Xor,
                        RegMemImm::reg(to_reg),
                        to_reg,
                    ));
                } else {
                    let value = value as u64;
                    ret.push(Inst::imm(size, value.into(), to_reg));
                }
            }
        }
        ret
    }

    fn worst_case_size() -> CodeOffset {
        15
    }

    fn ref_type_regclass(_: &settings::Flags) -> RegClass {
        RegClass::I64
    }

    fn gen_value_label_marker(label: ValueLabel, reg: Reg) -> Self {
        Inst::ValueLabelMarker { label, reg }
    }

    fn defines_value_label(&self) -> Option<(ValueLabel, Reg)> {
        match self {
            Inst::ValueLabelMarker { label, reg } => Some((*label, *reg)),
            _ => None,
        }
    }

    type LabelUse = LabelUse;
}

/// State carried between emissions of a sequence of instructions.
#[derive(Default, Clone, Debug)]
pub struct EmitState {
    /// Addend to convert nominal-SP offsets to real-SP offsets at the current
    /// program point.
    pub(crate) virtual_sp_offset: i64,
    /// Offset of FP from nominal-SP.
    pub(crate) nominal_sp_to_fp: i64,
    /// Safepoint stack map for upcoming instruction, as provided to `pre_safepoint()`.
    stack_map: Option<StackMap>,
    /// Current source location.
    cur_srcloc: SourceLoc,
}

/// Constant state used during emissions of a sequence of instructions.
pub struct EmitInfo {
    flags: settings::Flags,
    isa_flags: x64_settings::Flags,
}

impl EmitInfo {
    pub(crate) fn new(flags: settings::Flags, isa_flags: x64_settings::Flags) -> Self {
        Self { flags, isa_flags }
    }
}

impl MachInstEmitInfo for EmitInfo {
    fn flags(&self) -> &Flags {
        &self.flags
    }
}

impl MachInstEmit for Inst {
    type State = EmitState;
    type Info = EmitInfo;

    fn emit(&self, sink: &mut MachBuffer<Inst>, info: &Self::Info, state: &mut Self::State) {
        emit::emit(self, sink, info, state);
    }

    fn reg_name(preg: PReg, size: u8) -> &'static str {
        regs::reg_name(preg, size)
    }

    fn pretty_print(&self, _: &mut Self::State) -> String {
        format!("{:?}", self)
    }
}

impl MachInstEmitState<Inst> for EmitState {
    fn new<A: ABICallee<I = Inst>>(abi: &A) -> Self {
        EmitState {
            virtual_sp_offset: 0,
            nominal_sp_to_fp: abi.frame_size() as i64,
            stack_map: None,
            cur_srcloc: SourceLoc::default(),
        }
    }

    fn pre_safepoint(&mut self, stack_map: StackMap) {
        self.stack_map = Some(stack_map);
    }

    fn pre_sourceloc(&mut self, srcloc: SourceLoc) {
        self.cur_srcloc = srcloc;
    }
}

impl EmitState {
    fn take_stack_map(&mut self) -> Option<StackMap> {
        self.stack_map.take()
    }

    fn clear_post_insn(&mut self) {
        self.stack_map = None;
    }

    fn cur_srcloc(&self) -> SourceLoc {
        self.cur_srcloc
    }
}

/// A label-use (internal relocation) in generated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {
    /// A 32-bit offset from location of relocation itself, added to the existing value at that
    /// location. Used for control flow instructions which consider an offset from the start of the
    /// next instruction (so the size of the payload -- 4 bytes -- is subtracted from the payload).
    JmpRel32,

    /// A 32-bit offset from location of relocation itself, added to the existing value at that
    /// location.
    PCRel32,
}

impl MachInstLabelUse for LabelUse {
    const ALIGN: CodeOffset = 1;

    fn max_pos_range(self) -> CodeOffset {
        match self {
            LabelUse::JmpRel32 | LabelUse::PCRel32 => 0x7fff_ffff,
        }
    }

    fn max_neg_range(self) -> CodeOffset {
        match self {
            LabelUse::JmpRel32 | LabelUse::PCRel32 => 0x8000_0000,
        }
    }

    fn patch_size(self) -> CodeOffset {
        match self {
            LabelUse::JmpRel32 | LabelUse::PCRel32 => 4,
        }
    }

    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        let pc_rel = (label_offset as i64) - (use_offset as i64);
        debug_assert!(pc_rel <= self.max_pos_range() as i64);
        debug_assert!(pc_rel >= -(self.max_neg_range() as i64));
        let pc_rel = pc_rel as u32;
        match self {
            LabelUse::JmpRel32 => {
                let addend = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
                let value = pc_rel.wrapping_add(addend).wrapping_sub(4);
                buffer.copy_from_slice(&value.to_le_bytes()[..]);
            }
            LabelUse::PCRel32 => {
                let addend = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
                let value = pc_rel.wrapping_add(addend);
                buffer.copy_from_slice(&value.to_le_bytes()[..]);
            }
        }
    }

    fn supports_veneer(self) -> bool {
        match self {
            LabelUse::JmpRel32 | LabelUse::PCRel32 => false,
        }
    }

    fn veneer_size(self) -> CodeOffset {
        match self {
            LabelUse::JmpRel32 | LabelUse::PCRel32 => 0,
        }
    }

    fn generate_veneer(self, _: &mut [u8], _: CodeOffset) -> (CodeOffset, LabelUse) {
        match self {
            LabelUse::JmpRel32 | LabelUse::PCRel32 => {
                panic!("Veneer not supported for JumpRel32 label-use.");
            }
        }
    }
}
