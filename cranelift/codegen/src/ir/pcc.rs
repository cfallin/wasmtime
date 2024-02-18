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
//!
//! Correctness:
//! - Underflow/overflow: clear min and max respectively on all adds
//!   and subs
//!
//! Deployment:
//! - Add to fuzzing
//! - Turn on during wasm spec-tests
//!
//! More checks:
//! - Check that facts on `vmctx` GVs are subsumed by the actual facts
//!   on the vmctx arg in block0 (function arg).
//!
//! Generality:
//! - facts on outputs (in func signature)?
//! - Implement checking at the CLIF level as well.
//! - Check instructions that can trap as well?
//!
//! Nicer errors:
//! - attach instruction index or some other identifier to errors
//!
//! Text format cleanup:
//! - make the bitwidth on `max` facts optional in the CLIF text
//!   format?
//! - make offset in `mem` fact optional in the text format?
//!
//! Bikeshed colors (syntax):
//! - Put fact bang-annotations after types?
//!   `v0: i64 ! fact(..)` vs. `v0 ! fact(..): i64`

use crate::ir;
use crate::ir::types::*;
use crate::isa::TargetIsa;
use crate::machinst::{BlockIndex, LowerBackend, VCode};
use crate::trace;
use regalloc2::Function as _;
use smallvec::{smallvec, SmallVec};
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
    /// A block parameter claims a fact that one of its predecessors
    /// does not support.
    UnsupportedBlockparam,
    /// A memory access is out of bounds.
    OutOfBounds,
    /// Proof-carrying-code checking is not implemented for a
    /// particular compiler backend.
    UnimplementedBackend,
    /// Proof-carrying-code checking is not implemented for a
    /// particular instruction that instruction-selection chose. This
    /// is an internal compiler error.
    UnimplementedInst,
    /// Access to an invalid or undefined field offset in a struct.
    InvalidFieldOffset,
    /// Access to a field via the wrong type.
    BadFieldType,
    /// Store to a read-only field.
    WriteToReadOnlyField,
    /// Store of data to a field with a fact that does not subsume the
    /// field's fact.
    InvalidStoredFact,
}

/// A range in an integer space. This can be used to describe a value
/// or an offset into a memtype.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub enum ValueRange {
    /// Exactly the value(s) given.
    Exact(SmallVec<[Expr; 1]>),
    /// A value that lies within all specified lower and upper bounds
    /// (inclusive).
    Inclusive {
        /// Lower bounds (inclusive). The list specifies a set of
        /// bounds; the concrete value is greater than or equal to
        /// *all* of these bounds. If the list is empty, then there is
        /// no lower bound.
        min: SmallVec<[Expr; 1]>,
        /// Upper bounds (inclusive). The list specifies a set of
        /// bounds; the concrete value is less than or equal to *all*
        /// of these bounds. If the list is empty, then there is no
        /// upper bound.
        max: SmallVec<[Expr; 1]>,
    },
}

/// A fact on a value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub enum Fact {
    /// A bitslice of a value (up to a bitwidth) is within the given
    /// integer range.
    ///
    /// The slicing behavior is needed because this fact can describe
    /// both an SSA `Value`, whose entire value is well-defined, and a
    /// `VReg` in VCode, whose bits beyond the type stored in that
    /// register are don't-care (undefined).
    Range {
        /// The bitwidth of bits we care about, from the LSB upward.
        bit_width: u16,
        /// The actual range.
        range: ValueRange,
    },

    /// A pointer to a memory type, with an offset inside the memory
    /// type specified as a range, and optionally nullable (can take
    /// on a zero/NULL pointer value) as well.
    Mem {
        /// The memory type.
        ty: ir::MemoryType,
        /// The range of offsets into this type.
        range: ValueRange,
        /// This pointer can also be null.
        nullable: bool,
    },

    /// A definition of a value to be used as a symbol in
    /// BaseExprs. There can only be one of these per value number.
    ///
    /// Note that this differs from a `DynamicRange` specifying that
    /// some value in the program is the same as `value`. A `def(v1)`
    /// fact is propagated to machine code and serves as a source of
    /// truth: the value or location labeled with this fact *defines*
    /// what `v1` is, and any `dynamic_range(64, v1, v1)`-labeled
    /// values elsewhere are claiming to be equal to this value.
    ///
    /// This is necessary because we don't propagate SSA value labels
    /// down to machine code otherwise; so when referring symbolically
    /// to addresses and expressions derived from addresses, we need
    /// to introduce the symbol first.
    Def {
        /// The SSA value this value defines.
        value: ir::Value,
    },

    /// A comparison result between two dynamic values with a
    /// comparison of a certain kind.
    Compare {
        /// The kind of comparison.
        kind: ir::condcodes::IntCC,
        /// The left-hand side of the comparison.
        lhs: Expr,
        /// The right-hand side of the comparison.
        rhs: Expr,
    },

    /// A "conflict fact": this fact results from merging two other
    /// facts, and it can never be satisfied -- checking any value
    /// against this fact will fail.
    Conflict,
}

/// A bound expression.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct Expr {
    /// The dynamic (base) part.
    pub base: BaseExpr,
    /// The static (offset) part.
    pub offset: i128,
}

/// The base part of a bound expression.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub enum BaseExpr {
    /// No dynamic part (i.e., zero).
    None,
    /// A global value.
    GlobalValue(ir::GlobalValue),
    /// An SSA Value as a symbolic value. This can be referenced in
    /// facts even after we've lowered out of SSA: it becomes simply
    /// some symbolic value.
    Value(ir::Value),
    /// Top of the address space. This is "saturating": the offset
    /// doesn't matter.
    Max,
}

impl BaseExpr {
    /// Is one base less than or equal to another? (We can't always
    /// know; in such cases, returns `false`.)
    fn le(lhs: &BaseExpr, rhs: &BaseExpr) -> bool {
        // (i) reflexivity; (ii) 0 <= x for all (unsigned) x; (iii) x <= max for all x.
        lhs == rhs || *lhs == BaseExpr::None || *rhs == BaseExpr::Max
    }
}

impl Expr {
    /// Constant value.
    pub const fn constant(value: u64) -> Self {
        Expr {
            base: BaseExpr::None,
            // Safety: `i128::from(u64)` is not const, but this will never overflow.
            offset: value as i128,
        }
    }

    /// Constant value, full 128-bit width.
    pub const fn constant128(value: i128) -> Self {
        Expr {
            base: BaseExpr::None,
            offset: value,
        }
    }

    /// Maximum (saturated) value.
    pub const fn max_value() -> Self {
        Expr {
            base: BaseExpr::Max,
            offset: 0,
        }
    }

    /// The value of an SSA value.
    pub const fn value(value: ir::Value) -> Self {
        Expr {
            base: BaseExpr::Value(value),
            offset: 0,
        }
    }

    /// The value of an SSA value plus some offset.
    pub const fn value_offset(value: ir::Value, offset: i128) -> Self {
        Expr {
            base: BaseExpr::Value(value),
            offset,
        }
    }

    /// The value of a global value.
    pub const fn global_value(gv: ir::GlobalValue) -> Self {
        Expr {
            base: BaseExpr::GlobalValue(gv),
            offset: 0,
        }
    }

    /// The value of a global value plus some offset.
    pub const fn global_value_offset(gv: ir::GlobalValue, offset: i128) -> Self {
        Expr {
            base: BaseExpr::GlobalValue(gv),
            offset,
        }
    }

    /// Is one expression definitely less than or equal to another?
    /// (We can't always know; in such cases, returns `false`.)
    fn le(lhs: &Expr, rhs: &Expr) -> bool {
        let result = if rhs.base == BaseExpr::Max {
            true
        } else if lhs == &Expr::constant(0) && rhs.base != BaseExpr::None {
            true
        } else {
            BaseExpr::le(&lhs.base, &rhs.base) && lhs.offset <= rhs.offset
        };
        trace!("Expr::le: {lhs:?} {rhs:?} -> {result}");
        result
    }

    /// Add one expression to another.
    fn add(lhs: &Expr, rhs: &Expr) -> Expr {
        let Some(offset) = lhs.offset.checked_add(rhs.offset) else {
            return Expr::max_value();
        };
        let result = if lhs.base == rhs.base {
            Expr {
                base: lhs.base.clone(),
                offset,
            }
        } else if lhs.base == BaseExpr::None {
            Expr {
                base: rhs.base.clone(),
                offset,
            }
        } else if rhs.base == BaseExpr::None {
            Expr {
                base: lhs.base.clone(),
                offset,
            }
        } else {
            Expr {
                base: BaseExpr::Max,
                offset: 0,
            }
        };
        trace!("Expr::add: {lhs:?} + {rhs:?} -> {result:?}");
        result
    }

    /// Add a static offset to an expression.
    pub fn offset(lhs: &Expr, rhs: i64) -> Option<Expr> {
        let offset = lhs.offset.checked_add(rhs.into())?;
        Some(Expr {
            base: lhs.base.clone(),
            offset,
        })
    }

    /// Determine if we can know the difference between two expressions.
    pub fn difference(lhs: &Expr, rhs: &Expr) -> Option<i64> {
        match (lhs.base, rhs.base) {
            (BaseExpr::Max, _) | (_, BaseExpr::Max) => None,
            (a, b) if a == b => i64::try_from(lhs.offset.checked_sub(rhs.offset)?).ok(),
            _ => None,
        }
    }

    /// Is this Expr a BaseExpr with no offset? Return it if so.
    pub fn without_offset(&self) -> Option<&BaseExpr> {
        if self.offset == 0 {
            Some(&self.base)
        } else {
            None
        }
    }

    /// Multiply an expression by a constant, if possible.
    fn scale(&self, factor: u32) -> Option<Expr> {
        let offset = self.offset.checked_mul(i128::from(factor))?;
        match self.base {
            BaseExpr::None => Some(Expr {
                base: BaseExpr::None,
                offset,
            }),
            BaseExpr::Max => Some(Expr {
                base: BaseExpr::Max,
                offset: 0,
            }),
            _ => None,
        }
    }

    /// Multiply an expression by a constant, rounding downward if we
    /// must approximate.
    fn scale_downward(&self, factor: u32) -> Expr {
        self.scale(factor).unwrap_or(Expr::constant(0))
    }

    /// Multiply an expression by a constant, rounding upward if we
    /// must approximate.
    fn scale_upward(&self, factor: u32) -> Expr {
        self.scale(factor).unwrap_or(Expr::max_value())
    }

    /// Is this Expr an integer constant?
    fn as_const(&self) -> Option<i128> {
        match self.base {
            BaseExpr::None => Some(self.offset),
            _ => None,
        }
    }
}

impl fmt::Display for BaseExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BaseExpr::None => Ok(()),
            BaseExpr::Max => write!(f, "max"),
            BaseExpr::GlobalValue(gv) => write!(f, "{gv}"),
            BaseExpr::Value(value) => write!(f, "{value}"),
        }
    }
}

impl BaseExpr {
    /// Does this dynamic_expression take an offset?
    pub fn is_some(&self) -> bool {
        match self {
            BaseExpr::None => false,
            _ => true,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.base)?;
        match self.offset {
            offset if offset > 0 && self.base.is_some() => write!(f, "+{offset:#x}"),
            offset if offset > 0 => write!(f, "{offset:#x}"),
            offset if offset < 0 => {
                let negative_offset = -i128::from(offset); // upcast to support i64::MIN.
                write!(f, "-{negative_offset:#x}")
            }
            0 if self.base.is_some() => Ok(()),
            0 => write!(f, "0"),
            _ => unreachable!(),
        }
    }
}

struct DisplayExprs<'a>(&'a [Expr]);

impl<'a> fmt::Display for DisplayExprs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0.len() {
            0 => write!(f, "{{}}"),
            1 => write!(f, "{}", self.0[0]),
            _ => {
                write!(f, "{{")?;

                let mut first = true;
                for expr in self.0 {
                    if first {
                        write!(f, " {expr}")?;
                        first = false;
                    } else {
                        write!(f, ", {expr}")?;
                    }
                }

                write!(f, " }}")?;
                Ok(())
            }
        }
    }
}

impl fmt::Display for ValueRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ValueRange::Exact(exprs) => write!(f, "{}", DisplayExprs(exprs)),
            ValueRange::Inclusive { min, max } => {
                write!(f, "{}, {}", DisplayExprs(&min[..]), DisplayExprs(&max[..]))
            }
        }
    }
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Fact::Range { bit_width, range } => {
                write!(f, "range({bit_width}, {range})")
            }
            Fact::Mem {
                ty,
                range,
                nullable,
            } => {
                let nullable_flag = if *nullable { ", nullable" } else { "" };
                write!(f, "mem({ty}, {range}{nullable_flag})")
            }
            Fact::Def { value } => write!(f, "def({value})"),
            Fact::Compare { kind, lhs, rhs } => {
                write!(f, "compare({kind}, {lhs}, {rhs})")
            }
            Fact::Conflict => write!(f, "conflict"),
        }
    }
}

impl ValueRange {
    /// Is this ValueRange an exact integer constant?
    pub fn as_const(&self) -> Option<i128> {
        match self {
            ValueRange::Exact(exact) => exact.iter().find_map(|&e| e.as_const()),
            ValueRange::Inclusive { .. } => None,
        }
    }

    /// Is this ValueRange definitely less than or equal to the given expression?
    pub fn le_expr(&self, expr: &Expr) -> bool {
        let result = match self {
            ValueRange::Exact(exact) => exact.iter().any(|equiv| Expr::le(equiv, expr)),
            // The range is <= the expr if *any* of its upper bounds
            // are <= the expr, because each upper bound constrains
            // the whole range (i.e., the range is the intersection of
            // all combinations of bounds).
            ValueRange::Inclusive { max, .. } => {
                max.iter().any(|upper_bound| Expr::le(upper_bound, expr))
            }
        };
        trace!("ValueRange::le_expr: {self:?} {expr:?} -> {result}");
        result
    }

    /// Is the expression definitely within the ValueRange?
    pub fn contains_expr(&self, expr: &Expr) -> bool {
        let result = match self {
            ValueRange::Exact(exact) => exact.iter().any(|equiv| equiv == expr),
            ValueRange::Inclusive { min, max } => {
                min.iter().all(|lower_bound| Expr::le(lower_bound, expr))
                    && max.iter().all(|upper_bound| Expr::le(expr, upper_bound))
            }
        };
        trace!("ValueRange::contains_expr: {self:?} {expr:?} -> {result}");
        result
    }

    /// Simplify a ValueRange by removing redundant bounds. Any lower
    /// bound greater than another lower bound, or any upper bound
    /// less than another upper bound, can be removed.
    pub fn simplify(&mut self) {
        trace!("simplify: {self:?}");
        match self {
            ValueRange::Exact(equivs) => {
                equivs.sort();
                equivs.dedup();
            }
            ValueRange::Inclusive { min, max } => {
                let mut out_min = 0;
                for i in 0..min.len() {
                    let redundant = min[0..out_min]
                        .iter()
                        .any(|other_bound| Expr::le(other_bound, &min[i]));
                    if !redundant {
                        if i > out_min {
                            min.swap(out_min, i);
                        }
                        out_min += 1;
                    }
                }
                min.truncate(out_min);
                // Canonicalize to allow fast-path equality checks to more likely succeed.
                min.sort();

                let mut out_max = 0;
                for i in 0..max.len() {
                    let redundant = max[0..out_max]
                        .iter()
                        .any(|other_bound| Expr::le(&max[i], other_bound));
                    if !redundant {
                        if i > out_max {
                            max.swap(out_max, i);
                        }
                        out_max += 1;
                    }
                }
                max.truncate(out_max);
                // Canonicalize to allow fast-path equality checks to more likely succeed.
                max.sort();

                // If `min` and `max` are exactly one element each,
                // and the same element, then we can narrow. Note that
                // we don't sipmlify e.g. `range(32, {v1}, {v1, v2})`
                // to `range(32, v1)` because this loses information
                // (namely, the upper bound `v2`).
                if min.len() == 1 && max.len() == 1 && min[0] == max[0] {
                    *self = ValueRange::Exact(smallvec![min[0].clone()]);
                }
            }
        }
        trace!("simplify: produced {self:?}");
    }

    /// Does one ValueRange contain another? Assumes both sides are already simplified.
    pub fn contains(&self, other: &ValueRange) -> bool {
        let result = match (self, other) {
            (a, b) if a == b => true,
            (ValueRange::Exact(expr1), ValueRange::Exact(expr2)) => {
                expr1.iter().any(|e| expr2.contains(e))
            }
            (range @ ValueRange::Inclusive { .. }, ValueRange::Exact(expr)) => {
                expr.iter().any(|e| range.contains_expr(e))
            }
            (ValueRange::Exact(..), ValueRange::Inclusive { .. }) => false,
            (
                range1 @ ValueRange::Inclusive { .. },
                ValueRange::Inclusive {
                    min: min2,
                    max: max2,
                },
            ) => {
                // *Some* lower bound and *some* upper bound of the
                // RHS must be contained in the LHS. Either those
                // lower and upper bounds are tight, in which case all
                // values between them are then contained in the LHS;
                // or they are loose, and the true range is contained
                // within them, which in turn is contained in the LHS.
                (min2
                    .iter()
                    .any(|lower_bound2| range1.contains_expr(lower_bound2))
                    || range1.contains_expr(&Expr::constant(0)))
                    && (max2
                        .iter()
                        .any(|upper_bound2| range1.contains_expr(upper_bound2))
                        || range1.contains_expr(&Expr::max_value()))
            }
        };
        trace!("ValueRange::contains: {self:?} {other:?} -> {result}");
        result
    }

    /// Intersect two ValueRanges.
    pub fn intersect(lhs: &ValueRange, rhs: &ValueRange) -> ValueRange {
        match (lhs, rhs) {
            (ValueRange::Exact(e1), ValueRange::Exact(e2)) => {
                let mut result =
                    ValueRange::Exact(e1.iter().cloned().chain(e2.iter().cloned()).collect());
                result.simplify();
                result
            }
            (ValueRange::Exact(equiv), ValueRange::Inclusive { min, max })
            | (ValueRange::Inclusive { min, max }, ValueRange::Exact(equiv)) => {
                let mut min = min.clone();
                let mut max = max.clone();
                min.extend(equiv.iter().cloned());
                max.extend(equiv.iter().cloned());
                let mut result = ValueRange::Inclusive { min, max };
                result.simplify();
                result
            }
            (
                ValueRange::Inclusive {
                    min: min1,
                    max: max1,
                },
                ValueRange::Inclusive {
                    min: min2,
                    max: max2,
                },
            ) => {
                let mut min = min1.clone();
                let mut max = max1.clone();
                min.extend(min2.iter().cloned());
                max.extend(max2.iter().cloned());
                let mut result = ValueRange::Inclusive { min, max };
                result.simplify();
                result
            }
        }
    }

    /// Take the union of two ranges.
    pub fn union(lhs: &ValueRange, rhs: &ValueRange) -> ValueRange {
        match (lhs, rhs) {
            (ValueRange::Exact(e1), ValueRange::Exact(e2)) if lhs.contains(rhs) => {
                let combined = e1.iter().cloned().chain(e2.iter().cloned()).collect();
                let mut result = ValueRange::Exact(combined);
                result.simplify();
                result
            }
            (ValueRange::Exact(_), ValueRange::Exact(_)) => {
                // TODO: we could check whether one Exact is less than
                // the other, and if so, create a range from one to
                // the other. For now, let's just return a range that
                // covers everything.
                ValueRange::Inclusive {
                    min: smallvec![],
                    max: smallvec![],
                }
            }
            (e @ ValueRange::Exact(equiv), r @ ValueRange::Inclusive { min, max })
            | (r @ ValueRange::Inclusive { min, max }, e @ ValueRange::Exact(equiv)) => {
                if r.contains(e) {
                    r.clone()
                } else {
                    let min = min
                        .iter()
                        .filter(|&e| equiv.iter().any(|eq| Expr::le(e, eq)))
                        .cloned()
                        .collect();
                    let max = max
                        .iter()
                        .filter(|&e| equiv.iter().any(|eq| Expr::le(eq, e)))
                        .cloned()
                        .collect();
                    // No need to simplify -- we are only removing
                    // constraints, so no bounds will be newly
                    // subsumed.
                    ValueRange::Inclusive { min, max }
                }
            }
            (
                ValueRange::Inclusive {
                    min: min1,
                    max: max1,
                },
                ValueRange::Inclusive {
                    min: min2,
                    max: max2,
                },
            ) => {
                // Take lower bounds from LHS that are less than all
                // lower bounds on the RHS; and likewise the other
                // way; and likewise for upper bounds.
                let min = min1
                    .iter()
                    .filter(|&e| min2.iter().all(|e2| Expr::le(e, e2)))
                    .cloned()
                    .chain(
                        min2.iter()
                            .filter(|e| min1.iter().all(|e2| Expr::le(e, e2)))
                            .cloned(),
                    )
                    .collect();
                let max = max1
                    .iter()
                    .filter(|&e| max2.iter().all(|e2| Expr::le(e2, e)))
                    .cloned()
                    .chain(
                        max2.iter()
                            .filter(|e| max1.iter().all(|e2| Expr::le(e2, e)))
                            .cloned(),
                    )
                    .collect();
                let mut result = ValueRange::Inclusive { min, max };
                result.simplify();
                result
            }
        }
    }

    /// Scale a range by a factor.
    pub fn scale(&self, factor: u32) -> ValueRange {
        match self {
            ValueRange::Exact(equivs) => {
                let equivs = equivs
                    .iter()
                    .filter_map(|e| e.scale(factor))
                    .collect::<SmallVec<[Expr; 1]>>();
                if equivs.is_empty() {
                    let mut result = ValueRange::Inclusive {
                        min: smallvec![],
                        max: smallvec![],
                    };
                    result.simplify();
                    result
                } else {
                    ValueRange::Exact(equivs)
                }
            }
            ValueRange::Inclusive { min, max } => {
                let min = min.iter().map(|e| e.scale_downward(factor)).collect();
                let max = max.iter().map(|e| e.scale_upward(factor)).collect();
                let mut result = ValueRange::Inclusive { min, max };
                result.simplify();
                result
            }
        }
    }

    /// Add an offset to the lower and upper bounds of a range.
    pub fn offset(&self, offset: i64) -> ValueRange {
        match self {
            ValueRange::Exact(equivs) => {
                let equivs = equivs
                    .iter()
                    .flat_map(|e| Expr::offset(e, offset))
                    .collect();
                let mut result = ValueRange::Exact(equivs);
                result.simplify();
                result
            }
            ValueRange::Inclusive { min, max } => {
                let min = min.iter().flat_map(|e| Expr::offset(e, offset)).collect();
                let max = max.iter().flat_map(|e| Expr::offset(e, offset)).collect();
                let mut result = ValueRange::Inclusive { min, max };
                result.simplify();
                result
            }
        }
    }

    /// Find the range of the sum of two values described by ranges.
    pub fn add(lhs: &ValueRange, rhs: &ValueRange) -> ValueRange {
        trace!("ValueRange::add: {lhs:?} + {rhs:?}");
        match (lhs, rhs) {
            (ValueRange::Exact(e1), ValueRange::Exact(e2)) => {
                let equivs = e1
                    .iter()
                    .flat_map(|e1| e2.iter().map(|e2| Expr::add(e1, e2)))
                    .collect();
                let mut result = ValueRange::Exact(equivs);
                trace!(" -> exact + exact: {result:?}");
                result.simplify();
                trace!(" -> {result:?}");
                result
            }
            (ValueRange::Exact(exact), ValueRange::Inclusive { min, max })
            | (ValueRange::Inclusive { min, max }, ValueRange::Exact(exact)) => {
                let min = min
                    .iter()
                    .flat_map(|m| exact.iter().map(|e| Expr::add(m, e)))
                    .collect();
                let max = max
                    .iter()
                    .flat_map(|m| exact.iter().map(|e| Expr::add(m, e)))
                    .collect();
                let mut result = ValueRange::Inclusive { min, max };
                trace!(" -> exact + inclusive: {result:?}");
                result.simplify();
                trace!(" -> {result:?}");
                result
            }
            (
                ValueRange::Inclusive {
                    min: min1,
                    max: max1,
                },
                ValueRange::Inclusive {
                    min: min2,
                    max: max2,
                },
            ) => {
                let min = min1
                    .iter()
                    .flat_map(|m1| min2.iter().map(|m2| Expr::add(m1, m2)))
                    .collect();
                let max = max1
                    .iter()
                    .flat_map(|m1| max2.iter().map(|m2| Expr::add(m1, m2)))
                    .collect();
                let mut result = ValueRange::Inclusive { min, max };
                trace!(" -> inclusive + inclusive: {result:?}");
                result.simplify();
                trace!(" -> {result:?}");
                result
            }
        }
    }

    /// Clamp a ValueRange given a bit-width for the result.
    fn clamp(self, width: u16) -> ValueRange {
        trace!("ValueRange::clamp: {self:?} width {width}");
        let result = if self.contains_expr(&Expr::constant128(
            i128::from(max_value_for_width(width)) + 1,
        )) {
            // Underflow or overflow is possible!
            ValueRange::Inclusive {
                min: smallvec![],
                max: smallvec![Expr::constant(max_value_for_width(width))],
            }
        } else {
            self
        };
        trace!("ValueRange::clamp: -> {result:?}");
        result
    }
}

impl Fact {
    /// Create a range fact that specifies a single known constant value.
    pub fn constant(bit_width: u16, value: u64) -> Self {
        debug_assert!(value <= max_value_for_width(bit_width));
        // `min` and `max` are inclusive, so this specifies a range of
        // exactly one value.
        Fact::Range {
            bit_width,
            range: ValueRange::Exact(smallvec![Expr::constant(value)]),
        }
    }

    /// Create a range fact that points to the base of a memory type.
    pub fn dynamic_base_ptr(ty: ir::MemoryType) -> Self {
        Fact::Mem {
            ty,
            range: ValueRange::Exact(smallvec![Expr::constant(0)]),
            nullable: false,
        }
    }

    /// Create a fact that specifies the value is exactly an SSA value.
    ///
    /// Note that this differs from a `def` fact: it is not *defining*
    /// a symbol to have the value that this fact is attached to;
    /// rather it is claiming that this value is the same as whatever
    /// that symbol is. (In other words, the def should be elsewhere,
    /// and we are tying ourselves to it.)
    pub fn value(bit_width: u16, value: ir::Value) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Exact(smallvec![Expr::value(value)]),
        }
    }

    /// Create a fact that specifies the value is exactly an SSA value plus some offset.
    pub fn value_offset(bit_width: u16, value: ir::Value, offset: i64) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Exact(smallvec![Expr::value_offset(value, offset.into())]),
        }
    }

    /// Create a fact that specifies the value is exactly the value of a GV.
    pub fn global_value(bit_width: u16, gv: ir::GlobalValue) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Exact(smallvec![Expr::global_value(gv)]),
        }
    }

    /// Create a fact that specifies the value is exactly the value of a GV plus some offset.
    pub fn global_value_offset(bit_width: u16, gv: ir::GlobalValue, offset: i64) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Exact(smallvec![Expr::global_value_offset(gv, offset.into())]),
        }
    }

    /// Create a fact that expresses a given static range, from zero
    /// up to `max` (inclusive).
    pub fn static_value_range(bit_width: u16, max: u64) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Inclusive {
                min: smallvec![],
                max: smallvec![Expr::constant(max)],
            },
        }
    }

    /// Create a fact that expresses a given static range, from `min`
    /// (inclusive) up to `max` (inclusive).
    pub fn static_value_two_ended_range(bit_width: u16, min: u64, max: u64) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Inclusive {
                min: smallvec![Expr::constant(min)],
                max: smallvec![Expr::constant(max)],
            },
        }
    }

    /// Create a fact that expresses a given dynamic range, from zero up to `expr`.
    pub fn dynamic_value_range(bit_width: u16, max: Expr) -> Self {
        Fact::Range {
            bit_width,
            range: ValueRange::Inclusive {
                min: smallvec![],
                max: smallvec![max],
            },
        }
    }

    /// Create a range fact that specifies the maximum range for a
    /// value of the given bit-width.
    pub fn max_range_for_width(bit_width: u16) -> Self {
        let min = smallvec![];
        let max = smallvec![Expr::constant(max_value_for_width(bit_width))];
        let range = ValueRange::Inclusive { min, max };
        Fact::Range { bit_width, range }
    }

    /// Create a fact that describes the base pointer for a memory
    /// type.
    pub fn memory_base(ty: ir::MemoryType) -> Self {
        Fact::Mem {
            ty,
            range: ValueRange::Exact(smallvec![Expr::constant(0)]),
            nullable: false,
        }
    }

    /// Create a fact that describes a pointer to the given memory
    /// type with an offset described by the given fact.
    pub fn memory_with_range(
        ty: ir::MemoryType,
        offset_fact: Fact,
        nullable: bool,
    ) -> Option<Self> {
        let Fact::Range {
            bit_width: _,
            range,
        } = offset_fact
        else {
            return None;
        };
        Some(Fact::Mem {
            ty,
            range,
            nullable,
        })
    }

    /// Create a range fact that specifies the maximum range for a
    /// value of the given bit-width, zero-extended into a wider
    /// width.
    pub fn max_range_for_width_extended(from_width: u16, to_width: u16) -> Self {
        debug_assert!(from_width <= to_width);
        let min = smallvec![];
        let max = smallvec![Expr::constant(max_value_for_width(from_width))];
        let range = ValueRange::Inclusive { min, max };
        Fact::Range {
            bit_width: to_width,
            range,
        }
    }

    /// Try to infer a minimal fact for a value of the given IR type.
    pub fn infer_from_type(ty: ir::Type) -> Option<Self> {
        match ty {
            I8 | I16 | I32 | I64 => {
                Some(Self::max_range_for_width(u16::try_from(ty.bits()).unwrap()))
            }
            _ => None,
        }
    }

    /// Does this fact "propagate" automatically, i.e., cause
    /// instructions that process it to infer their own output facts?
    /// Not all facts propagate automatically; otherwise, verification
    /// would be much slower.
    pub fn propagates(&self) -> bool {
        match self {
            Fact::Mem { .. } => true,
            _ => false,
        }
    }

    /// Merge two facts. We take the *intersection*: that is, we know
    /// both facts to be true, so we can intersect ranges. (This
    /// differs from the usual static analysis approach, where we are
    /// merging multiple possibilities into a generalized / widened
    /// fact. We want to narrow here.)
    pub fn intersect(a: &Fact, b: &Fact) -> Fact {
        match (a, b) {
            (
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range1,
                },
                Fact::Range {
                    bit_width: bw_rhs,
                    range: range2,
                },
            ) if bw_lhs == bw_rhs => Fact::Range {
                bit_width: *bw_lhs,
                range: ValueRange::intersect(range1, range2),
            },

            (
                Fact::Mem {
                    ty: ty_lhs,
                    range: range1,
                    nullable: nullable_lhs,
                },
                Fact::Mem {
                    ty: ty_rhs,
                    range: range2,
                    nullable: nullable_rhs,
                },
            ) if ty_lhs == ty_rhs => Fact::Mem {
                ty: *ty_lhs,
                range: ValueRange::intersect(range1, range2),
                nullable: *nullable_lhs && *nullable_rhs,
            },

            _ => Fact::Conflict,
        }
    }

    /// Take the union of two facts: produce a fact that applies to a
    /// value that has either one fact or another (e.g., at a
    /// control-flow merge point or a conditional-select operator).
    pub fn union(a: &Fact, b: &Fact) -> Fact {
        match (a, b) {
            (
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range1,
                },
                Fact::Range {
                    bit_width: bw_rhs,
                    range: range2,
                },
            ) if bw_lhs == bw_rhs => Fact::Range {
                bit_width: *bw_lhs,
                range: ValueRange::union(range1, range2),
            },

            (
                Fact::Mem {
                    ty: ty_lhs,
                    range: range1,
                    nullable: nullable_lhs,
                },
                Fact::Mem {
                    ty: ty_rhs,
                    range: range2,
                    nullable: nullable_rhs,
                },
            ) if ty_lhs == ty_rhs => Fact::Mem {
                ty: *ty_lhs,
                range: ValueRange::union(range1, range2),
                nullable: *nullable_lhs || *nullable_rhs,
            },

            _ => Fact::Conflict,
        }
    }

    /// Does this fact describe an exact expression?
    pub fn as_expr(&self) -> Option<&Expr> {
        match self {
            Fact::Range {
                range: ValueRange::Exact(equiv),
                ..
            } => equiv.first(),
            _ => None,
        }
    }

    /// Does this fact describe a constant?
    pub fn as_const(&self) -> Option<i128> {
        match self {
            Fact::Range { range, .. } => range.as_const(),
            _ => None,
        }
    }

    /// Offsets a value with a fact by a known amount.
    pub fn offset(&self, width: u16, offset: i64) -> Option<Fact> {
        if offset == 0 {
            return Some(self.clone());
        }

        let result = match self {
            Fact::Range { bit_width, range } if *bit_width == width => Some(Fact::Range {
                bit_width: *bit_width,
                range: range.offset(offset.into()).clamp(width),
            }),
            Fact::Mem {
                ty,
                range,
                nullable: false,
            } => Some(Fact::Mem {
                ty: *ty,
                range: range.offset(offset.into()).clamp(width),
                nullable: false,
            }),
            _ => None,
        };
        trace!("offset: {self:?} + {offset} in width {width} -> {result:?}");
        result
    }

    /// Get the range of a fact: either the actual value range, or the
    /// range of offsets into a memory type.
    pub fn range(&self) -> Option<&ValueRange> {
        match self {
            Fact::Range { range, .. } | Fact::Mem { range, .. } => Some(range),
            _ => None,
        }
    }

    /// Update the range in either a Range or Mem fact.
    pub fn with_range(&self, range: ValueRange) -> Fact {
        match self {
            Fact::Range { bit_width, .. } => Fact::Range {
                bit_width: *bit_width,
                range,
            },
            Fact::Mem { ty, nullable, .. } => Fact::Mem {
                ty: *ty,
                nullable: *nullable,
                range,
            },
            f => f.clone(),
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

/// The two kinds of inequalities: "strict" (`<`, `>`) and "loose"
/// (`<=`, `>=`), the latter of which admit equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InequalityKind {
    /// Strict inequality: {less,greater}-than.
    Strict,
    /// Loose inequality: {less,greater}-than-or-equal.
    Loose,
}

/// A "context" in which we can evaluate and derive facts. This
/// context carries environment/global properties, such as the machine
/// pointer width.
pub struct FactContext<'a> {
    function: &'a ir::Function,
    pointer_width: u16,
}

impl<'a> FactContext<'a> {
    /// Create a new "fact context" in which to evaluate facts.
    pub fn new(function: &'a ir::Function, pointer_width: u16) -> Self {
        FactContext {
            function,
            pointer_width,
        }
    }

    /// Computes whether `lhs` "subsumes" (implies) `rhs`.
    pub fn subsumes(&self, lhs: &Fact, rhs: &Fact) -> bool {
        trace!("subsumes {lhs:?} {rhs:?}");
        match (lhs, rhs) {
            // Reflexivity.
            (l, r) if l == r => true,

            (
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range_lhs,
                },
                Fact::Range {
                    bit_width: bw_rhs,
                    range: range_rhs,
                },
            ) if bw_lhs == bw_rhs => range_rhs.contains(range_lhs),

            (
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range_lhs,
                },
                Fact::Range {
                    bit_width: bw_rhs,
                    range: range_rhs,
                },
            ) if bw_lhs > bw_rhs => {
                // If the LHS makes a claim about a larger bitwidth,
                // then it can still imply the RHS if the RHS claims
                // the full range of its width.
                let rhs_is_trivially_true = range_rhs.contains_expr(&Expr::constant(0))
                    && range_rhs.contains_expr(&Expr::constant(max_value_for_width(*bw_rhs)));
                // It can also still imply the RHS if the LHS's range
                // is within the bitwidth of the RHS, so we don't have
                // to worry about truncation/aliasing effects.
                let lhs_is_in_rhs_width_range =
                    range_lhs.le_expr(&Expr::constant(max_value_for_width(*bw_rhs)));

                rhs_is_trivially_true || lhs_is_in_rhs_width_range
            }

            (
                Fact::Mem {
                    ty: ty_lhs,
                    range: range_lhs,
                    nullable: nullable_lhs,
                },
                Fact::Mem {
                    ty: ty_rhs,
                    range: range_rhs,
                    nullable: nullable_rhs,
                },
            ) => {
                ty_lhs == ty_rhs
                    && range_rhs.contains(range_lhs)
                    && (*nullable_lhs || !*nullable_rhs)
            }

            // Constant zero subsumes nullable DynamicMem pointers.
            (
                Fact::Range {
                    bit_width, range, ..
                },
                Fact::Mem { nullable: true, .. },
            ) if *bit_width == self.pointer_width && range.le_expr(&Expr::constant(0)) => true,

            // Any fact subsumes a Def, because the Def makes no
            // claims about the actual value (it ties a symbol to that
            // value, but the value is fed to the symbol, not the
            // other way around).
            (_, Fact::Def { .. }) => true,

            _ => false,
        }
    }

    /// Computes whether the optional fact `lhs` subsumes (implies)
    /// the optional fact `lhs`. A `None` never subsumes any fact, and
    /// is always subsumed by any fact at all (or no fact).
    pub fn subsumes_fact_optionals(&self, lhs: Option<&Fact>, rhs: Option<&Fact>) -> bool {
        match (lhs, rhs) {
            (None, None) => true,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(lhs), Some(rhs)) => self.subsumes(lhs, rhs),
        }
    }

    /// Computes whatever fact can be known about the sum of two
    /// values with attached facts. The add is performed to the given
    /// bit-width. Note that this is distinct from the machine or
    /// pointer width: e.g., many 64-bit machines can still do 32-bit
    /// adds that wrap at 2^32.
    pub fn add(&self, lhs: &Fact, rhs: &Fact, add_width: u16) -> Option<Fact> {
        let result = match (lhs, rhs) {
            (
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range_lhs,
                },
                Fact::Range {
                    bit_width: bw_rhs,
                    range: range_rhs,
                },
            ) if bw_lhs == bw_rhs && add_width >= *bw_lhs => Some(Fact::Range {
                bit_width: *bw_lhs,
                range: ValueRange::add(range_lhs, range_rhs).clamp(add_width),
            }),

            (
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range_lhs,
                },
                Fact::Mem {
                    ty,
                    range: range_rhs,
                    nullable,
                },
            )
            | (
                Fact::Mem {
                    ty,
                    range: range_rhs,
                    nullable,
                },
                Fact::Range {
                    bit_width: bw_lhs,
                    range: range_lhs,
                },
            ) if *bw_lhs >= self.pointer_width
                && add_width >= *bw_lhs
                // A null pointer doesn't remain a null pointer unless
                // the right-hand side is constant zero.
                && (!*nullable || range_lhs.le_expr(&Expr::constant(0))) =>
            {
                Some(Fact::Mem {
                    ty: *ty,
                    range: ValueRange::add(range_lhs, range_rhs).clamp(add_width),
                    nullable: *nullable,
                })
            }

            _ => None,
        };

        trace!("add({add_width}): {lhs:?} + {rhs:?} -> {result:?}");
        result
    }

    /// Computes the `uextend` of a value with the given facts.
    pub fn uextend(&self, fact: &Fact, from_width: u16, to_width: u16) -> Option<Fact> {
        if from_width == to_width {
            return Some(fact.clone());
        }

        let result = match fact {
            Fact::Range { bit_width, range } if *bit_width == from_width => Some(Fact::Range {
                bit_width: to_width,
                range: range.clone(),
            }),

            // If the claim is a definition of a value, we can say
            // that the output has a range of exactly that value.
            Fact::Def { value } => Some(Fact::value(to_width, *value)),

            // Otherwise, we can at least claim that the value is
            // within the range of `from_width`.
            Fact::Range { .. } => Some(Fact::max_range_for_width_extended(from_width, to_width)),

            _ => None,
        };
        trace!("uextend: fact {fact:?} from {from_width} to {to_width} -> {result:?}");
        result
    }

    /// Computes the `sextend` of a value with the given facts.
    pub fn sextend(&self, fact: &Fact, from_width: u16, to_width: u16) -> Option<Fact> {
        let max_positive_value = 1u64 << (from_width - 1);
        match fact {
            // If we have a defined value in bits 0..bit_width, and
            // the MSB w.r.t. `from_width` is *not* set, then we can
            // do the same as `uextend`.
            Fact::Range {
                bit_width, range, ..
            } if *bit_width == from_width && range.le_expr(&Expr::constant(max_positive_value)) => {
                self.uextend(fact, from_width, to_width)
            }
            _ => None,
        }
    }

    /// Computes the bit-truncation of a value with the given fact.
    pub fn truncate(&self, fact: &Fact, from_width: u16, to_width: u16) -> Option<Fact> {
        if from_width == to_width {
            return Some(fact.clone());
        }

        trace!(
            "truncate: fact {:?} from {} to {}",
            fact,
            from_width,
            to_width
        );

        match fact {
            Fact::Range { bit_width, range } if *bit_width == from_width => {
                let max_val = (1u64 << to_width) - 1;
                if range.le_expr(&Expr::constant(max_val)) {
                    Some(Fact::Range {
                        bit_width: to_width,
                        range: range.clone(),
                    })
                } else {
                    Some(Fact::max_range_for_width(to_width))
                }
            }
            _ => None,
        }
    }

    /// Scales a value with a fact by a known constant.
    pub fn scale(&self, fact: &Fact, width: u16, factor: u32) -> Option<Fact> {
        let result = match fact {
            x if factor == 1 => Some(x.clone()),
            Fact::Range { bit_width, range } if *bit_width == width => Some(Fact::Range {
                bit_width: *bit_width,
                range: range.scale(factor).clamp(width),
            }),
            _ => None,
        };
        trace!("scale: {fact:?} * {factor} at width {width} -> {result:?}");
        result
    }

    /// Left-shifts a value with a fact by a known constant.
    pub fn shl(&self, fact: &Fact, width: u16, amount: u16) -> Option<Fact> {
        if amount >= 32 {
            return None;
        }
        let factor: u32 = 1 << amount;
        self.scale(fact, width, factor)
    }

    /// Check that accessing memory via a pointer with this fact, with
    /// a memory access of the given size, is valid.
    ///
    /// If valid, returns the memory type and offset into that type
    /// that this address accesses, if known, or `None` if the range
    /// doesn't constrain the access to exactly one location.
    fn check_address(
        &self,
        fact: &Fact,
        access_size: u32,
    ) -> PccResult<Option<(ir::MemoryType, u64)>> {
        trace!("check_address: fact {:?} access_size {}", fact, access_size);

        match fact {
            Fact::Mem {
                ty,
                range,
                nullable: _,
            } => {
                trace!(" -> memory type: {}", self.function.memory_types[*ty]);
                match &self.function.memory_types[*ty] {
                    ir::MemoryTypeData::Struct { size, .. }
                    | ir::MemoryTypeData::Memory { size } => {
                        ensure!(u64::from(access_size) <= *size, OutOfBounds);
                        let effective_size = *size - u64::from(access_size);
                        ensure!(range.le_expr(&Expr::constant(effective_size)), OutOfBounds);
                    }
                    ir::MemoryTypeData::DynamicMemory {
                        gv,
                        size: mem_static_size,
                    } => {
                        let effective_size = i128::from(*mem_static_size) - i128::from(access_size);
                        let end = Expr::global_value_offset(*gv, effective_size);
                        ensure!(range.le_expr(&end), OutOfBounds)
                    }
                    ir::MemoryTypeData::Empty => bail!(OutOfBounds),
                }
                let specific_ty_and_offset =
                    if let Some(constant) = range.as_const().and_then(|i| u64::try_from(i).ok()) {
                        Some((*ty, constant))
                    } else {
                        None
                    };
                trace!(" -> specific type and offset: {specific_ty_and_offset:?}");
                Ok(specific_ty_and_offset)
            }

            _ => bail!(OutOfBounds),
        }
    }

    /// Get the access struct field, if any, by a pointer with the
    /// given fact and an access of the given type.
    pub fn struct_field<'b>(
        &'b self,
        fact: &Fact,
        access_ty: ir::Type,
    ) -> PccResult<Option<&'b ir::MemoryTypeField>> {
        let (ty, offset) = match self.check_address(fact, access_ty.bytes())? {
            Some((ty, offset)) => (ty, offset),
            None => return Ok(None),
        };

        if let ir::MemoryTypeData::Struct { fields, .. } = &self.function.memory_types[ty] {
            let field = fields
                .iter()
                .find(|field| field.offset == offset)
                .ok_or(PccError::InvalidFieldOffset)?;
            if field.ty != access_ty {
                bail!(BadFieldType);
            }
            Ok(Some(field))
        } else {
            // Access to valid memory, but not a struct: no facts can
            // be attached to the result.
            Ok(None)
        }
    }

    /// Check a load, and determine what fact, if any, the result of
    /// the load might have.
    pub fn load<'b>(&'b self, fact: &Fact, access_ty: ir::Type) -> PccResult<Option<&'b Fact>> {
        Ok(self
            .struct_field(fact, access_ty)?
            .and_then(|field| field.fact()))
    }

    /// Check a store.
    pub fn store(
        &self,
        fact: &Fact,
        access_ty: ir::Type,
        data_fact: Option<&Fact>,
    ) -> PccResult<()> {
        if let Some(field) = self.struct_field(fact, access_ty)? {
            // If it's a read-only field, disallow.
            if field.readonly {
                bail!(WriteToReadOnlyField);
            }
            // Check that the fact on the stored data subsumes the
            // field's fact.
            if !self.subsumes_fact_optionals(data_fact, field.fact()) {
                bail!(InvalidStoredFact);
            }
        }
        Ok(())
    }

    /// Apply a known inequality to rewrite dynamic bounds using
    /// transitivity, if possible.
    ///
    /// Given that `lhs >= rhs` (if `kind` is not `strict`) or `lhs >
    /// rhs` (if `kind` is `strict`), update `fact`.
    pub fn apply_inequality(
        &self,
        fact: &Fact,
        lhs: &Fact,
        rhs: &Fact,
        kind: InequalityKind,
    ) -> Fact {
        trace!("apply_inequality: fact {fact:?} lhs {lhs:?} rhs {rhs:?} kind {kind:?}");
        // If `rhs` is an exact value, and we have an expression for
        // it, and if `fact` is a range, look for that expression in
        // the upper bounds of `fact`; if present (or offset), add the
        // lower and upper bounds of `lhs` (properly offset if needed)
        // to `fact`'s upper bounds.
        let result = match (fact.range(), lhs, rhs) {
            (
                Some(range),
                Fact::Range {
                    range: ValueRange::Exact(equiv_lhs),
                    ..
                },
                Fact::Range {
                    range: ValueRange::Exact(equiv_rhs),
                    ..
                },
            ) => {
                trace!(" -> range {range:?} LHS equiv {equiv_lhs:?} RHS equiv {equiv_rhs:?}");
                let new_range = match range {
                    ValueRange::Inclusive { min, max } => {
                        let offset = max
                            .iter()
                            .flat_map(|m| equiv_rhs.iter().flat_map(|e| Expr::difference(m, e)))
                            .max();
                        trace!(" -> offset {offset:?}");
                        if let Some(offset) = offset {
                            let offset = match kind {
                                InequalityKind::Loose => offset,
                                InequalityKind::Strict => offset - 1,
                            };
                            let new_upper_bounds =
                                equiv_lhs.iter().flat_map(|e| Expr::offset(e, offset));
                            let max = max.iter().cloned().chain(new_upper_bounds).collect();
                            let min = min.clone();
                            ValueRange::Inclusive { min, max }
                        } else {
                            range.clone()
                        }
                    }
                    ValueRange::Exact(equivs) => {
                        let offset = equivs
                            .iter()
                            .flat_map(|m| equiv_rhs.iter().flat_map(|e| Expr::difference(m, e)))
                            .max();
                        trace!(" -> offset {offset:?}");
                        if let Some(offset) = offset {
                            let offset = match kind {
                                InequalityKind::Loose => offset,
                                InequalityKind::Strict => offset - 1,
                            };
                            let new_upper_bounds = equiv_lhs
                                .iter()
                                .flat_map(|e| Expr::offset(e, offset))
                                .chain(equivs.iter().cloned())
                                .collect();
                            trace!(" -> new_upper_bounds {new_upper_bounds:?}");
                            ValueRange::Inclusive {
                                min: equivs.clone(),
                                max: new_upper_bounds,
                            }
                        } else {
                            ValueRange::Inclusive {
                                min: smallvec![],
                                max: smallvec![],
                            }
                        }
                    }
                };
                fact.with_range(new_range)
            }
            _ => fact.clone(),
        };

        trace!("apply_inequality({fact:?}, {lhs:?}, {rhs:?}, {kind:?} -> {result:?}");
        result
    }
}

const fn max_value_for_width(bits: u16) -> u64 {
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
    f: &ir::Function,
    vcode: &mut VCode<B::MInst>,
    backend: &B,
) -> PccResult<()> {
    let ctx = FactContext::new(f, backend.triple().pointer_width().unwrap().bits().into());

    // Check that individual instructions are valid according to input
    // facts, and support the stated output facts.
    for block in 0..vcode.num_blocks() {
        let block = BlockIndex::new(block);
        let mut flow_state = B::FactFlowState::default();
        for inst in vcode.block_insns(block).iter() {
            // Check any output facts on this inst.
            if let Err(e) = backend.check_fact(&ctx, vcode, inst, &mut flow_state) {
                log::error!("Error checking instruction: {:?}", vcode[inst]);
                return Err(e);
            }

            // If this is a branch, check that all block arguments subsume
            // the assumed facts on the blockparams of successors.
            if vcode.is_branch(inst) {
                for (succ_idx, succ) in vcode.block_succs(block).iter().enumerate() {
                    for (arg, param) in vcode
                        .branch_blockparams(block, inst, succ_idx)
                        .iter()
                        .zip(vcode.block_params(*succ).iter())
                    {
                        let arg_fact = vcode.vreg_fact(*arg);
                        let param_fact = vcode.vreg_fact(*param);
                        if !ctx.subsumes_fact_optionals(arg_fact, param_fact) {
                            return Err(PccError::UnsupportedBlockparam);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
