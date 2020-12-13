//! Data structure for tracking the (possibly multiple) registers that hold one
//! SSA `Value`.

use regalloc::{RealReg, Reg, VirtualReg, Writable};
use std::fmt::Debug;

/// Location at which a `Value` is stored in register(s): the value is located
/// in one or more registers, depending on its width. A value may be stored in
/// more than one register if the machine has no registers wide enough
/// otherwise: for example, on a 32-bit architecture, we may store `I64` values
/// in two registers, and `I128` values in four.
///
/// By convention, the register parts are kept in machine-endian order here.
///
/// N.B.: we cap the capacity of this at four, and we use special in-band
/// sentinal `Reg` values (`Reg::invalid()`) to avoid the need to carry a
/// separate length. This allows the struct to be `Copy` (no heap or drop
/// overhead) and be only 16 bytes, which is important for compiler performance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueRegs<R: Clone + Copy + Debug + PartialEq + Eq + InvalidSentinel> {
    // N.B.: this will be 16 bytes on x86-64 / aarch64: the SmallVec will
    // combine its length pointer
    parts: [R; 4],
}

/// A type with an "invalid" sentinel value.
pub trait InvalidSentinel: Copy + Eq {
    /// The invalid sentinel value.
    fn invalid_sentinel() -> Self;
    /// Is this the invalid sentinel?
    fn is_invalid_sentinel(self) -> bool {
        self == Self::invalid_sentinel()
    }
}
impl InvalidSentinel for Reg {
    fn invalid_sentinel() -> Self {
        Reg::invalid()
    }
}
impl InvalidSentinel for VirtualReg {
    fn invalid_sentinel() -> Self {
        VirtualReg::invalid()
    }
}
impl InvalidSentinel for RealReg {
    fn invalid_sentinel() -> Self {
        RealReg::invalid()
    }
}
impl InvalidSentinel for Writable<Reg> {
    fn invalid_sentinel() -> Self {
        Writable::from_reg(Reg::invalid_sentinel())
    }
}

impl<R: Clone + Copy + Debug + PartialEq + Eq + InvalidSentinel> ValueRegs<R> {
    /// Create an invalid Value-in-Reg.
    pub fn invalid() -> Self {
        ValueRegs {
            parts: [R::invalid_sentinel(); 4],
        }
    }
    /// Create a Value-in-R location for a value stored in one register.
    pub fn one(reg: R) -> Self {
        ValueRegs {
            parts: [
                reg,
                R::invalid_sentinel(),
                R::invalid_sentinel(),
                R::invalid_sentinel(),
            ],
        }
    }
    /// Create a Value-in-R location for a value stored in two registers.
    pub fn two(r1: R, r2: R) -> Self {
        ValueRegs {
            parts: [r1, r2, R::invalid_sentinel(), R::invalid_sentinel()],
        }
    }
    /// Create a Value-in-R location for a value stored in four registers.
    pub fn four(r1: R, r2: R, r3: R, r4: R) -> Self {
        ValueRegs {
            parts: [r1, r2, r3, r4],
        }
    }

    /// Is this Value-to-Reg mapping valid?
    pub fn is_valid(self) -> bool {
        !self.parts[0].is_invalid_sentinel()
    }
    /// Is this Value-to-Reg mapping invalid?
    pub fn is_invalid(self) -> bool {
        self.parts[0].is_invalid_sentinel()
    }

    /// Return the number of registers used.
    pub fn len(self) -> usize {
        // If rustc/LLVM is smart enough, this might even be vectorized...
        (self.parts[0] != R::invalid_sentinel()) as usize
            + (self.parts[1] != R::invalid_sentinel()) as usize
            + (self.parts[2] != R::invalid_sentinel()) as usize
            + (self.parts[3] != R::invalid_sentinel()) as usize
    }

    /// Return the single register used for this value, if any.
    pub fn only_reg(self) -> Option<R> {
        if self.len() == 1 {
            Some(self.parts[0])
        } else {
            None
        }
    }

    /// Return an iterator over the registers storing this value.
    pub fn regs(&self) -> &[R] {
        &self.parts[0..self.len()]
    }

    /// Map individual registers via a map function.
    pub fn map<NewR, F>(self, f: F) -> ValueRegs<NewR>
    where
        NewR: Clone + Copy + Debug + PartialEq + Eq + InvalidSentinel,
        F: Fn(R) -> NewR,
    {
        ValueRegs {
            parts: [
                f(self.parts[0]),
                f(self.parts[1]),
                f(self.parts[2]),
                f(self.parts[3]),
            ],
        }
    }
}

/// Create a writable ValueRegs.
pub(crate) fn writable_value_regs(regs: ValueRegs<Reg>) -> ValueRegs<Writable<Reg>> {
    ValueRegs {
        parts: [
            Writable::from_reg(regs.parts[0]),
            Writable::from_reg(regs.parts[1]),
            Writable::from_reg(regs.parts[2]),
            Writable::from_reg(regs.parts[3]),
        ],
    }
}

/// Strip a writable ValueRegs down to a readonly ValueRegs.
pub(crate) fn non_writable_value_regs(regs: ValueRegs<Writable<Reg>>) -> ValueRegs<Reg> {
    ValueRegs {
        parts: [
            regs.parts[0].to_reg(),
            regs.parts[1].to_reg(),
            regs.parts[2].to_reg(),
            regs.parts[3].to_reg(),
        ],
    }
}
