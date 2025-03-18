//! Exception tables: catch handlers on `try_call` instructions.
//!
//! An exception table describes where execution flows after returning
//! from `try_call`. It contains both the "normal" destination -- the
//! block to branch to when the function returns without throwing an
//! exception -- and any "catch" destinations associated with
//! particular exception tags. Each target indicates the arguments to
//! pass to the block that receives control.
//!
//! Like other side-tables (e.g., jump tables), each exception table
//! must be used by only one instruction. Sharing is not permitted
//! because it can complicate transforms (how does one change the
//! table used by only one instruction if others also use it?).
//!
//! In order to allow the `try_call` instruction itself to remain
//! small, the exception table also contains the signature ID of the
//! called function.

use crate::ir::entities::{ExceptionTag, SigRef};
use crate::ir::instructions::ValueListPool;
use crate::ir::BlockCall;
use alloc::vec::Vec;
use core::fmt::{self, Display, Formatter};
use cranelift_entity::packed_option::PackedOption;
#[cfg(feature = "enable-serde")]
use serde_derive::{Deserialize, Serialize};

/// Contents of an exception table.
///
/// The "no exception" target for is stored as the first element of the
/// underlying vector.  It can be accessed through the `normal_block`
/// and `normal_block_mut` functions. All blocks may be iterated using
/// the `all_branches` and `all_branches_mut` functions, which will
/// both iterate over the default block first.
#[derive(Debug, Clone, PartialEq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct ExceptionTableData {
    /// All BlockCalls packed together. This is necessary because the
    /// rest of the compiler expects to be able to grab a slice of
    /// branch targets for any branch instruction. The first BlockCall
    /// is the normal-return destination, and the rest correspond to
    /// the tags in `tags` below. Thus, we have the invariant that
    /// `targets.len() == tags.len() + 1`.
    ///
    /// The BlockCall for the ordinary return case, at index 0, must
    /// have a signature such that the returns of the invoked function
    /// (in `sig`) plus with the arguments in the `BlockCall` match
    /// the block parameter signature of the named block.
    ///
    /// The BlockCalls for the remainder of the cases must have a
    /// signature such that the exception-payload pointer (a pointer
    /// type according to the platform) plus the arguments in the
    /// `BlockCall` match the parameter signature of the named block.
    pub targets: Vec<BlockCall>,

    /// Tags corresponding to targets other than the first one.
    ///
    /// A tag value of `None` indicates a catch-all handler. The
    /// catch-all handler matches only if no other handler matches,
    /// regardless of the order in this vector.
    ///
    /// `tags[i]` corresponds to `targets[i + 1]`. See above regarding
    /// the invariant that relates their lengths.
    pub tags: Vec<PackedOption<ExceptionTag>>,

    /// The signature of the function whose invocation is associated
    /// with this handler table.
    pub sig: SigRef,
}

impl ExceptionTableData {
    /// Return a value that can display the contents of this exception
    /// table.
    pub fn display<'a>(&'a self, pool: &'a ValueListPool) -> DisplayExceptionTable<'a> {
        DisplayExceptionTable { table: self, pool }
    }

    /// Get the default target for the non-exceptional return case.
    pub fn normal_return(&self) -> &BlockCall {
        &self.targets[0]
    }

    /// Get the targets for exceptional return cases, together with
    /// their tags.
    pub fn catches(&self) -> impl Iterator<Item = (Option<ExceptionTag>, &BlockCall)> + '_ {
        self.tags
            .iter()
            .map(|tag| tag.expand())
            .zip(self.targets.iter().skip(1))
    }
}

/// A wrapper for the context required to display a
/// [ExceptionTableData].
pub struct DisplayExceptionTable<'a> {
    table: &'a ExceptionTableData,
    pool: &'a ValueListPool,
}

impl<'a> Display for DisplayExceptionTable<'a> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "{}, {}, [",
            self.table.sig,
            self.table.normal_return().display(self.pool)
        )?;
        let mut first = true;
        for (tag, block_call) in self.table.catches() {
            if first {
                write!(fmt, " ")?;
                first = false;
            } else {
                write!(fmt, ", ")?;
            }
            if let Some(tag) = tag {
                write!(fmt, "{}: {}", tag, block_call.display(self.pool))?;
            } else {
                write!(fmt, "default: {}", block_call.display(self.pool))?;
            }
        }
        let space = if first { "" } else { " " };
        write!(fmt, "{}]", space)?;
        Ok(())
    }
}
