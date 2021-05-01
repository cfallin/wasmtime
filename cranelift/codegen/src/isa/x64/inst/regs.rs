//! Registers, the Universe thereof, and printing.
//!
//! These are ordered by sequence number, as required in the Universe.
//!
//! The caller-saved registers are placed first in order to prefer not to clobber (requiring
//! saves/restores in prologue/epilogue code) when possible. Note that there is no other heuristic
//! in the backend that will apply such pressure; the register allocator's cost heuristics are not
//! aware of the cost of clobber-save/restore code.
//!
//! One might worry that this pessimizes code with many callsites, where using caller-saves causes
//! us to have to save them (as we are the caller) frequently. However, the register allocator
//! *should be* aware of *this* cost, because it sees that the call instruction modifies all of the
//! caller-saved (i.e., callee-clobbered) registers.
//!
//! Hence, this ordering encodes pressure in one direction (prefer not to clobber registers that we
//! ourselves have to save) and this is balanaced against the RA's pressure in the other direction
//! at callsites.

use crate::machinst::Reg;
use crate::settings;
use regalloc2::{MachineEnv, PReg, RegClass};

// Hardware encodings (note the special rax, rcx, rdx, rbx order).

pub const ENC_RAX: u8 = 0;
pub const ENC_RCX: u8 = 1;
pub const ENC_RDX: u8 = 2;
pub const ENC_RBX: u8 = 3;
pub const ENC_RSP: u8 = 4;
pub const ENC_RBP: u8 = 5;
pub const ENC_RSI: u8 = 6;
pub const ENC_RDI: u8 = 7;
pub const ENC_R8: u8 = 8;
pub const ENC_R9: u8 = 9;
pub const ENC_R10: u8 = 10;
pub const ENC_R11: u8 = 11;
pub const ENC_R12: u8 = 12;
pub const ENC_R13: u8 = 13;
pub const ENC_R14: u8 = 14;
pub const ENC_R15: u8 = 15;

fn gpr(enc: u8) -> PReg {
    PReg::new(enc, RegClass::Int)
}

pub(crate) fn rax() -> PReg {
    gpr(ENC_RAX)
}
pub(crate) fn rcx() -> PReg {
    gpr(ENC_RCX)
}
pub(crate) fn rdx() -> PReg {
    gpr(ENC_RDX)
}
pub(crate) fn rbx() -> PReg {
    gpr(ENC_RBX)
}
pub(crate) fn rsp() -> PReg {
    gpr(ENC_RSP)
}
pub(crate) fn rbp() -> PReg {
    gpr(ENC_RBP)
}
pub(crate) fn rsi() -> PReg {
    gpr(ENC_RSI)
}
pub(crate) fn rdi() -> PReg {
    gpr(ENC_RDI)
}
pub(crate) fn r8() -> PReg {
    gpr(ENC_R8)
}
pub(crate) fn r9() -> PReg {
    gpr(ENC_R9)
}
pub(crate) fn r10() -> PReg {
    gpr(ENC_R10)
}
pub(crate) fn r11() -> PReg {
    gpr(ENC_R11)
}
pub(crate) fn r12() -> PReg {
    gpr(ENC_R12)
}
pub(crate) fn r13() -> PReg {
    gpr(ENC_R13)
}
pub(crate) fn r14() -> PReg {
    gpr(ENC_R14)
}
pub(crate) fn r15() -> PReg {
    gpr(ENC_R15)
}

/// The pinned register on this architecture.
/// It must be the same as Spidermonkey's HeapReg, as found in this file.
/// https://searchfox.org/mozilla-central/source/js/src/jit/x64/Assembler-x64.h#99
pub(crate) fn pinned_reg() -> PReg {
    r15()
}

fn fpr(enc: u8) -> Reg {
    PReg::new(enc, RegClass::Float)
}

pub(crate) fn xmm0() -> PReg {
    fpr(0)
}
pub(crate) fn xmm1() -> PReg {
    fpr(1)
}
pub(crate) fn xmm2() -> PReg {
    fpr(2)
}
pub(crate) fn xmm3() -> PReg {
    fpr(3)
}
pub(crate) fn xmm4() -> PReg {
    fpr(4)
}
pub(crate) fn xmm5() -> PReg {
    fpr(5)
}
pub(crate) fn xmm6() -> PReg {
    fpr(6)
}
pub(crate) fn xmm7() -> PReg {
    fpr(7)
}
pub(crate) fn xmm8() -> PReg {
    fpr(8)
}
pub(crate) fn xmm9() -> PReg {
    fpr(9)
}
pub(crate) fn xmm10() -> PReg {
    fpr(10)
}
pub(crate) fn xmm11() -> PReg {
    fpr(11)
}
pub(crate) fn xmm12() -> PReg {
    fpr(12)
}
pub(crate) fn xmm13() -> PReg {
    fpr(13)
}
pub(crate) fn xmm14() -> PReg {
    fpr(14)
}
pub(crate) fn xmm15() -> PReg {
    fpr(15)
}

/// Create the MachineEnv for x64.
pub(crate) fn create_machine_env(flags: &settings::Flags) -> MachineEnv {}

/// Create the register universe for X64.
///
/// The ordering of registers matters, as commented in the file doc comment: assumes the
/// calling-convention is SystemV, at the moment.
pub(crate) fn create_reg_universe_systemv(flags: &settings::Flags) -> MachineEnv {
    let mut regs = vec![];
    let mut regs_by_class = vec![vec![], vec![]];
    let mut scratch_by_class = vec![];

    // Add all PRegs. Every PReg that appears in VCode (even e.g. as a
    // constraint) must be here, even those that are not allocatable.
    regs.push(rax());
    regs.push(rcx());
    regs.push(rdx());
    regs.push(rbx());
    regs.push(rsp());
    regs.push(rbp());
    regs.push(rsi());
    regs.push(rdi());
    regs.push(r8());
    regs.push(r9());
    regs.push(r10());
    regs.push(r11());
    regs.push(r12());
    regs.push(r13());
    regs.push(r14());
    regs.push(r15());
    regs.push(xmm0());
    regs.push(xmm1());
    regs.push(xmm2());
    regs.push(xmm3());
    regs.push(xmm4());
    regs.push(xmm5());
    regs.push(xmm6());
    regs.push(xmm7());
    regs.push(xmm8());
    regs.push(xmm9());
    regs.push(xmm10());
    regs.push(xmm11());
    regs.push(xmm12());
    regs.push(xmm13());
    regs.push(xmm14());
    regs.push(xmm15());

    let use_pinned_reg = flags.enable_pinned_reg();

    // Add allocatable PRegs by class. TODO: two priorities within
    // each class -- preferred and non-preferred.
    regs_by_class[0].push(rax());
    regs_by_class[0].push(rcx());
    regs_by_class[0].push(rdx());
    regs_by_class[0].push(rbx());
    regs_by_class[0].push(rsi());
    regs_by_class[0].push(rdi());
    regs_by_class[0].push(r8());
    regs_by_class[0].push(r9());
    regs_by_class[0].push(r10());
    regs_by_class[0].push(r11());
    regs_by_class[0].push(r12());
    regs_by_class[0].push(r13());
    if !use_pinned_reg {
        regs_by_class[0].push(r15());
    }
    scratch_by_class.push(r14());

    regs_by_class[1].push(xmm0());
    regs_by_class[1].push(xmm1());
    regs_by_class[1].push(xmm2());
    regs_by_class[1].push(xmm3());
    regs_by_class[1].push(xmm4());
    regs_by_class[1].push(xmm5());
    regs_by_class[1].push(xmm6());
    regs_by_class[1].push(xmm7());
    regs_by_class[1].push(xmm8());
    regs_by_class[1].push(xmm9());
    regs_by_class[1].push(xmm10());
    regs_by_class[1].push(xmm11());
    regs_by_class[1].push(xmm12());
    regs_by_class[1].push(xmm13());
    regs_by_class[1].push(xmm14());
    scratch_by_class.push(xmm15());

    MachineEnv {
        regs,
        regs_by_class,
        scratch_by_class,
    }
}

/// If `ireg` denotes a PReg, make a best-effort attempt to show its
/// name with the appropriate size-based declension.
pub fn show_ireg_sized(reg: PReg, size: u8) -> &'static str {
    let sizes = match (reg.class(), reg.hw_enc()) {
        (RegClass::Int, 0) => ["%rax", "%eax", "%ax", "%al"],
        (RegClass::Int, 1) => ["%rcx", "%ecx", "%cx", "%cl"],
        (RegClass::Int, 2) => ["%rdx", "%edx", "%dx", "%dl"],
        (RegClass::Int, 3) => ["%rbx", "%ebx", "%bx", "%bl"],
        (RegClass::Int, 4) => ["%rsp", "%esp", "%sp", "%spl"],
        (RegClass::Int, 5) => ["%rbp", "%ebp", "%bp", "%bpl"],
        (RegClass::Int, 6) => ["%rsi", "%esi", "%si", "%sil"],
        (RegClass::Int, 7) => ["%rdi", "%edi", "%di", "%dil"],
        (RegClass::Int, 8) => ["%r8", "%r8d", "%r8w", "%r8b"],
        (RegClass::Int, 9) => ["%r9", "%r9d", "%r9w", "%r9b"],
        (RegClass::Int, 10) => ["%r10", "%r10d", "%r10w", "%r10b"],
        (RegClass::Int, 11) => ["%r11", "%r11d", "%r11w", "%r11b"],
        (RegClass::Int, 12) => ["%r12", "%r12d", "%r12w", "%r12b"],
        (RegClass::Int, 13) => ["%r13", "%r13d", "%r13w", "%r13b"],
        (RegClass::Int, 14) => ["%r14", "%r14d", "%r14w", "%r14b"],
        (RegClass::Int, 15) => ["%r15", "%r15d", "%r15w", "%r15b"],

        (RegClass::Float, 0) => ["%xmm0", "%xmm0", "%xmm0", "%xmm0"],
        (RegClass::Float, 1) => ["%xmm1", "%xmm1", "%xmm1", "%xmm1"],
        (RegClass::Float, 2) => ["%xmm2", "%xmm2", "%xmm2", "%xmm2"],
        (RegClass::Float, 3) => ["%xmm3", "%xmm3", "%xmm3", "%xmm3"],
        (RegClass::Float, 4) => ["%xmm4", "%xmm4", "%xmm4", "%xmm4"],
        (RegClass::Float, 5) => ["%xmm5", "%xmm5", "%xmm5", "%xmm5"],
        (RegClass::Float, 6) => ["%xmm6", "%xmm6", "%xmm6", "%xmm6"],
        (RegClass::Float, 7) => ["%xmm7", "%xmm7", "%xmm7", "%xmm7"],
        (RegClass::Float, 8) => ["%xmm8", "%xmm8", "%xmm8", "%xmm8"],
        (RegClass::Float, 9) => ["%xmm9", "%xmm9", "%xmm9", "%xmm9"],
        (RegClass::Float, 10) => ["%xmm10", "%xmm10", "%xmm10", "%xmm10"],
        (RegClass::Float, 11) => ["%xmm11", "%xmm11", "%xmm11", "%xmm11"],
        (RegClass::Float, 12) => ["%xmm12", "%xmm12", "%xmm12", "%xmm12"],
        (RegClass::Float, 13) => ["%xmm13", "%xmm13", "%xmm13", "%xmm13"],
        (RegClass::Float, 14) => ["%xmm14", "%xmm14", "%xmm14", "%xmm14"],
        (RegClass::Float, 15) => ["%xmm15", "%xmm15", "%xmm15", "%xmm15"],
    };

    match size {
        1 => sizes[3],
        2 => sizes[2],
        4 => sizes[1],
        _ => sizes[0],
    }
}
