//! Regalloc interface: layer of types that wrap regalloc concepts for use in MachInst backends.

use regalloc2::{Allocation, Operand, OperandKind, OperandOrAllocation, OperandPolicy, OperandPos};

pub use regalloc2::{MachineEnv, PReg, RegClass, SpillSlot, VReg};

/// A `Reg` encompasses everything that an instruction needs to record
/// for a register mention. Internally, it contains either an
/// `Operand` (before regalloc) or an `Allocation` (after regalloc).
///
/// Ordinarily, backend code should deal in `VReg`s when generating
/// and lowering into VCode. `Reg` should appear only in in the
/// instruction enum itself, so that we do not need separate types for
/// before- and after-regalloc data.
///
/// Backends' instruction implementations will ordinarily provide two
/// constructors for a given instruction: one that takes purely VRegs,
/// and builds operands from them based on the usage characteristics
/// (def/use, before/after/both, reuse of input, fixed reg, etc), and
/// one that takes PRegs for use only when generating post-regalloc
/// code (e.g. prologues and epilogues).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg {
    inner: OperandOrAllocation,
}

impl Reg {
    /// Create a `Reg` that wraps an `Operand`: this is the
    /// pre-regalloc form of an instruction's register slot.
    pub fn operand(op: Operand) -> Self {
        Self {
            inner: OperandOrAllocation::from_operand(op),
        }
    }

    /// Create a `Reg` that represents an invalid value.
    pub fn invalid() -> Self {
        Reg::alloc(Allocation::none())
    }

    /// Create a `Reg` that wraps a `VReg` as a use (at the
    /// before-point of its instruction).
    pub fn reg_use(vreg: VReg) -> Self {
        Reg::operand(Operand::reg_use(vreg))
    }

    /// Create a `Reg` that wraps a `VReg` as a use (at the
    /// after-point of its instruction).
    pub fn reg_use_at_end(vreg: VReg) -> Self {
        Reg::operand(Operand::reg_use_at_end(vreg))
    }

    /// Create a `Reg` that wraps a `VReg` as a def (at the
    /// before-point of its instruction).
    pub fn reg_def_at_start(vreg: VReg) -> Self {
        Reg::operand(Operand::reg_def_at_start(vreg))
    }

    /// Create a `Reg` that wraps a `VReg` as a def (at the
    /// after-point of its instruction).
    pub fn reg_def(vreg: VReg) -> Self {
        Reg::operand(Operand::reg_def(vreg))
    }

    /// Create a `Reg` that wraps a `VReg` as a temp def (valid for
    /// the whole instruction).
    pub fn reg_temp(vreg: VReg) -> Self {
        Reg::operand(Operand::reg_temp(vreg))
    }

    /// Create a `Reg` that wraps a `VReg` as a use with a fixed-reg
    /// constraint.
    pub fn reg_fixed_use(vreg: VReg, preg: PReg) -> Self {
        Reg::operand(Operand::reg_fixed_use(vreg, preg))
    }

    /// Create a `Reg` that wraps a `VReg` as a def with a fixed-reg
    /// constraint.
    pub fn reg_fixed_def(vreg: VReg, preg: PReg) -> Self {
        Reg::operand(Operand::reg_fixed_def(vreg, preg))
    }

    /// Create a `Reg` that wraps a `VReg` with the constraint that it
    /// reuse the register of another Reg.
    pub fn reg_reuse_def(vreg: VReg, reused_idx: usize) -> Self {
        Reg::operand(Operand::new(
            vreg,
            OperandPolicy::Reuse(reused_idx),
            OperandKind::Def,
            OperandPos::After,
        ))
    }

    /// Create a `Reg` that wraps an `Allocation`: this is the
    /// post-regalloc form of an instruction's register slot.
    pub fn alloc(alloc: Allocation, kind: OperandKind) -> Self {
        Self {
            inner: OperandOrAllocation::from_alloc_and_kind(alloc, kind),
        }
    }

    /// Create a `Reg` that wraps an allocated `PReg`.
    pub fn preg_use(preg: PReg) -> Self {
        Reg::alloc(Allocation::reg(preg), OperandKind::Use)
    }

    /// Create a `Reg` that wraps an allocated `PReg`.
    pub fn preg_def(preg: PReg) -> Self {
        Reg::alloc(Allocation::reg(preg), OperandKind::Def)
    }

    /// Create a `Reg` that wraps an allocated `SpillSlot`.
    pub fn spillslot_use(slot: SpillSlot) -> Self {
        Reg::alloc(Allocation::stack(slot), OperandKind::Use)
    }

    /// Create a `Reg` that wraps an allocated `SpillSlot`.
    pub fn spillslot_def(slot: SpillSlot) -> Self {
        Reg::alloc(Allocation::stack(slot), OperandKind::Def)
    }

    /// Is this an Operand (an unallocated register-mention specification)?
    pub fn is_operand(self) -> bool {
        self.inner.as_operand().is_some()
    }

    /// Is this an Allocation (a post-regalloc location)?
    pub fn is_alloc(self) -> bool {
        self.inner.as_alloc().is_some()
    }

    /// Convert to and return the inner `Operand`, if in that mode.
    pub fn as_operand(self) -> Option<Operand> {
        self.inner.as_operand()
    }

    /// Convert to and return the inner `Allocation`, if in that mode.
    pub fn as_alloc(self) -> Option<Allocation> {
        self.inner.as_alloc()
    }

    pub fn is_def(self) -> bool {
        self.inner.kind() == OperandKind::Def
    }

    /// Convert to and return the inner `PReg` inside an `Allocation`,
    /// if in that mode.
    pub fn as_preg(self) -> Option<PReg> {
        self.as_alloc().filter_map(|a| a.as_reg())
    }

    /// Convert to and return the inner `SpillSlot` inside an
    /// `Allocation`, if in that mode.
    pub fn as_spillslot(self) -> Option<SpillSlot> {
        self.as_alloc().filter_map(|a| a.as_spillslot())
    }
}

impl std::fmt::Debug for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(preg) = self.as_reg() {
            write!(f, "{}", preg)
        } else if let Some(slot) = self.as_spillslot() {
            write!(f, "{}", slot)
        } else if let Some(op) = self.as_operand() {
            write!(f, "{}", op)
        }
    }
}
