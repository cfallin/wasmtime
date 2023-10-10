//! Proof-carrying code. We attach "facts" to values and then check
//! that they remain true after compilation.
//!
//! A few key design principle of this approach are:
//!
//! - The producer of the IR provides the axioms. All "ground truth",
//!   such as what memory is accessible -- is meant to come by way of
//!   facts on the function arguments and global values. In some
//!   sense, all we are doing here is validating the "internal
//!   consistency" of the facts that are provided on values, and the
//!   actions performed on those values.
//!
//! - We do not derive and forward-propagate facts eagerly. Rather,
//!   the producer needs to provide breadcrumbs -- a "proof witness"
//!   of sorts -- to allow the checking to complete. That means that
//!   as an address is computed, or pointer chains are dereferenced,
//!   each intermediate value will likely have some fact attached.
//!
//!   This does create more verbose IR, but a significant positive
//!   benefit is that it avoids unnecessary work: we do not build up a
//!   knowledge base that effectively encodes the integer ranges of
//!   many or most values in the program. Rather, we only check
//!   specifically the memory-access sequences. In practice, each such
//!   sequence is likely to be a carefully-controlled sequence of IR
//!   operations from, e.g., a sandboxing compiler (such as
//!   `cranelift-wasm`) so adding annotations here to communicate
//!   intent (ranges, bounds-checks, and the like) is no problem.
//!
//! Facts are attached to SSA values in CLIF, and are maintained
//! through optimizations and through lowering. They are thus also
//! present on VRegs in the VCode. In theory, facts could be checked
//! at either level, though in practice it is most useful to check
//! them at the VCode level if the goal is an end-to-end verification
//! of certain properties (e.g., memory sandboxing).
//!
//! Checking facts entails visiting each instruction that defines a
//! value with a fact, and checking the result's fact against the
//! facts on arguments and the operand. For VCode, this is
//! fundamentally a question of the target ISA's semantics, so we call
//! into the `LowerBackend` for this. Note that during checking there
//! is also limited forward propagation / inference, but only within
//! an instruction: for example, an addressing mode commonly can
//! include an addition, multiplication/shift, or extend operation,
//! and there is no way to attach facts to the intermediate values
//! "inside" the instruction, so instead the backend can use
//! `FactContext::add()` and friends to forward-propagate facts.
//!
//! TODO:
//! - Check blockparams' preds against blockparams' facts.
//! - Propagate facts through optimization (egraph layer).
//! - Generate facts in cranelift-wasm frontend when lowering memory ops.
//! - Implement richer "points-to" facts that describe the pointed-to
//!   memory, so the loaded values can also have facts.
//! - Support bounds-checking-type operations for dynamic memories and
//!   tables.
//! - Implement checking at the CLIF level as well.
//! - Check instructions that can trap as well?

use crate::ir;
use crate::isa::TargetIsa;
use crate::machinst::{InsnIndex, LowerBackend, MachInst, VCode};
use std::fmt;

#[cfg(feature = "enable-serde")]
use serde_derive::{Deserialize, Serialize};

/// The result of checking proof-carrying-code facts.
pub type PccResult<T> = std::result::Result<T, PccError>;

/// An error or inconsistency discovered when checking proof-carrying
/// code.
#[derive(Debug, Clone)]
pub enum PccError {
    /// An operation wraps around, invalidating the stated value
    /// range.
    Overflow,
    /// An input to an operator that produces a fact-annotated value
    /// does not have a fact describing it, and one is needed.
    MissingFact,
    /// A derivation of an output fact is unsupported (incorrect or
    /// not derivable).
    UnsupportedFact,
    /// A memory access is out of bounds.
    OutOfBounds,
    /// Proof-carrying-code checking is not implemented for a
    /// particular compiler backend.
    UnimplementedBackend,
    /// Proof-carrying-code checking is not implemented for a
    /// particular instruction that instruction-selection chose. This
    /// is an internal compiler error.
    UnimplementedInst,
}

/// A fact on a value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub enum Fact {
    /// A bitslice of a value (up to a bitwidth) is less than or equal
    /// to a given maximum value.
    ///
    /// The slicing behavior is needed because this fact can describe
    /// both an SSA `Value`, whose entire value is well-defined, and a
    /// `VReg` in VCode, whose bits beyond the type stored in that
    /// register are don't-care (undefined).
    ValueMax {
        /// The bitwidth of bits we care about, from the LSB upward.
        bit_width: u16,
        /// The maximum value that the bitslice can take
        /// (inclusive). The range is unsigned: the bits of the value
        /// will be within the range `0..=max`.
        max: u64,
    },

    /// A pointer value to a memory region that can be accessed.
    PointsTo {
        /// A description of the memory region this pointer is allowed
        /// to access (size, etc).
        region: MemoryRegion,
    },

    /// A pointer to a memory type.
    Mem {
        /// The memory type.
        ty: ir::MemoryType,
        /// The offset into the memory type.
        offset: i64,
    },
}

/// A memory region that can be accessed. This description is attached
/// to a particular base pointer.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct MemoryRegion {
    /// Includes both the actual memory bound as well as any guard
    /// pages. Inclusive, so we can represent the full range of a
    /// `u64`. The range is unsigned.
    pub max: u64,
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Fact::ValueMax { bit_width, max } => write!(f, "max({}, {:#x})", bit_width, max),
            Fact::PointsTo {
                region: MemoryRegion { max },
            } => write!(f, "points_to(0x{:x})", max),
            Fact::Mem { ty, offset } => write!(f, "mem({}, {:#x})", ty, offset),
        }
    }
}

macro_rules! ensure {
    ( $condition:expr, $err:tt $(,)? ) => {
        if !$condition {
            return Err(PccError::$err);
        }
    };
}

macro_rules! bail {
    ( $err:tt ) => {{
        return Err(PccError::$err);
    }};
}

/// A "context" in which we can evaluate and derive facts. This
/// context carries environment/global properties, such as the machine
/// pointer width.
#[derive(Debug)]
pub struct FactContext {
    pointer_width: u16,
}

impl FactContext {
    /// Create a new "fact context" in which to evaluate facts.
    pub fn new(pointer_width: u16) -> Self {
        FactContext { pointer_width }
    }

    /// Computes whether `lhs` "subsumes" (implies) `rhs`.
    pub fn subsumes(&self, lhs: &Fact, rhs: &Fact) -> bool {
        match (lhs, rhs) {
            (
                Fact::ValueMax {
                    bit_width: bw_lhs,
                    max: max_lhs,
                },
                Fact::ValueMax {
                    bit_width: bw_rhs,
                    max: max_rhs,
                },
            ) => {
                // If the bitwidths we're claiming facts about are the
                // same, and if the value is less than or equal to
                // `max_lhs`, and if `max_rhs` is less than `max_lhs`,
                // then it is certainly less than or equal to
                // `max_rhs`.
                //
                // In other words, we can always expand the claimed
                // possible value range.
                bw_lhs == bw_rhs && max_lhs <= max_rhs
            }
            (
                Fact::PointsTo {
                    region: MemoryRegion { max: max_lhs },
                },
                Fact::PointsTo {
                    region: MemoryRegion { max: max_rhs },
                },
            ) => {
                // If the pointer is valid up to `max_lhs`, and
                // `max_rhs` is less than or equal to `max_lhs`, then
                // it is certainly valid up to `max_rhs`.
                //
                // In other words, we can always shrink the valid
                // addressable region.
                max_rhs <= max_lhs
            }
            _ => false,
        }
    }

    /// Computes whatever fact can be known about the sum of two
    /// values with attached facts. The add is performed to the given
    /// bit-width. Note that this is distinct from the machine or
    /// pointer width: e.g., many 64-bit machines can still do 32-bit
    /// adds that wrap at 2^32.
    pub fn add(&self, lhs: &Fact, rhs: &Fact, add_width: u16) -> Option<Fact> {
        match (lhs, rhs) {
            (
                Fact::ValueMax {
                    bit_width: bw_lhs,
                    max: lhs,
                },
                Fact::ValueMax {
                    bit_width: bw_rhs,
                    max: rhs,
                },
            ) if bw_lhs == bw_rhs && add_width >= *bw_lhs => {
                let computed_max = lhs.checked_add(*rhs)?;
                Some(Fact::ValueMax {
                    bit_width: *bw_lhs,
                    max: computed_max,
                })
            }

            (
                Fact::ValueMax {
                    bit_width: bw_max,
                    max,
                },
                Fact::PointsTo { region },
            )
            | (
                Fact::PointsTo { region },
                Fact::ValueMax {
                    bit_width: bw_max,
                    max,
                },
            ) if *bw_max >= self.pointer_width && add_width >= *bw_max => {
                let region = MemoryRegion {
                    max: region.max.checked_sub(*max)?,
                };
                Some(Fact::PointsTo { region })
            }

            _ => None,
        }
    }

    /// Computes the `uextend` of a value with the given facts.
    pub fn uextend(&self, fact: &Fact, from_width: u16, to_width: u16) -> Option<Fact> {
        match fact {
            // If we have a defined value in bits 0..bit_width, and we
            // are filling zeroes into from_bits..to_bits, and
            // bit_width and from_bits are exactly contiguous, then we
            // have defined values in 0..to_bits (and because this is
            // a zero-extend, the max value is the same).
            Fact::ValueMax { bit_width, max } if *bit_width == from_width => Some(Fact::ValueMax {
                bit_width: to_width,
                max: *max,
            }),
            // Otherwise, we can at least claim that the value is
            // within the range of `to_width`.
            Fact::ValueMax { .. } => Some(Fact::ValueMax {
                bit_width: to_width,
                max: max_value_for_width(to_width),
            }),
            _ => None,
        }
    }

    /// Computes the `sextend` of a value with the given facts.
    pub fn sextend(&self, fact: &Fact, from_width: u16, to_width: u16) -> Option<Fact> {
        match fact {
            // If we have a defined value in bits 0..bit_width, and
            // the MSB w.r.t. `from_width` is *not* set, then we can
            // do the same as `uextend`.
            Fact::ValueMax { bit_width, max }
                if *bit_width == from_width && (*max & (1 << (*bit_width - 1)) == 0) =>
            {
                self.uextend(fact, from_width, to_width)
            }
            _ => None,
        }
    }

    /// Scales a value with a fact by a known constant.
    pub fn scale(&self, fact: &Fact, width: u16, factor: u32) -> Option<Fact> {
        // The minimal (loosest) fact we can claim: the value will be
        // within the range implied by its bitwidth.
        let minimal_fact = Fact::ValueMax {
            bit_width: width,
            max: max_value_for_width(width),
        };
        match fact {
            Fact::ValueMax { bit_width, max } if *bit_width == width => {
                let max = match max.checked_mul(u64::from(factor)) {
                    Some(max) => max,
                    None => return Some(minimal_fact),
                };
                if *bit_width < 64 && max > max_value_for_width(width) {
                    return Some(minimal_fact);
                }
                Some(Fact::ValueMax {
                    bit_width: *bit_width,
                    max,
                })
            }
            _ => None,
        }
    }

    /// Left-shifts a value with a fact by a known constant.
    pub fn shl(&self, fact: &Fact, width: u16, amount: u16) -> Option<Fact> {
        if amount >= 32 {
            return None;
        }
        let factor: u32 = 1 << amount;
        self.scale(fact, width, factor)
    }

    /// Offsets a value with a fact by a known amount.
    pub fn offset(&self, fact: &Fact, width: u16, offset: i64) -> Option<Fact> {
        // If we eventually support two-sided ranges, we can
        // represent (0..n) + m -> ((0+m)..(n+m)). However,
        // right now, all ranges start with zero, so any
        // negative offset could underflow, and removes all
        // claims of constrained range.
        let offset = u64::try_from(offset).ok()?;

        match fact {
            Fact::ValueMax { bit_width, max } if *bit_width == width => {
                let max = match max.checked_add(offset) {
                    Some(max) => max,
                    None => {
                        return Some(Fact::ValueMax {
                            bit_width: width,
                            max: max_value_for_width(width),
                        })
                    }
                };

                Some(Fact::ValueMax {
                    bit_width: *bit_width,
                    max,
                })
            }
            Fact::PointsTo {
                region: MemoryRegion { max },
            } => {
                let max = max.checked_sub(offset)?;
                Some(Fact::PointsTo {
                    region: MemoryRegion { max },
                })
            }
            _ => None,
        }
    }

    /// Check that accessing memory via a pointer with this fact, with
    /// a memory access of the given size, is valid.
    pub fn check_address(&self, fact: &Fact, size: u32) -> PccResult<()> {
        match fact {
            Fact::PointsTo {
                region: MemoryRegion { max },
            } => ensure!(u64::from(size) <= *max, OutOfBounds),
            _ => bail!(OutOfBounds),
        }

        Ok(())
    }
}

fn max_value_for_width(bits: u16) -> u64 {
    assert!(bits <= 64);
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Top-level entry point after compilation: this checks the facts in
/// VCode.
pub fn check_vcode_facts<B: LowerBackend + TargetIsa>(
    _f: &ir::Function,
    vcode: &VCode<B::MInst>,
    backend: &B,
) -> PccResult<()> {
    for inst in 0..vcode.num_insts() {
        let inst = InsnIndex::new(inst);
        if vcode.inst_defines_facts(inst) || vcode[inst].is_mem_access() {
            // This instruction defines a register with a new fact, or
            // has some side-effect we want to be careful to
            // verify. We'll call into the backend to validate this
            // fact with respect to the instruction and the input
            // facts.
            backend.check_fact(&vcode[inst], vcode)?;
        }
    }
    Ok(())
}
