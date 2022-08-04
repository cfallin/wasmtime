//! Data for callsite labels.

use crate::ir::Inst;
use cranelift_entity::packed_option::PackedOption;
use std::fmt;

/// Data associated with a callsite label.
#[derive(Clone, Default)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct CallSiteData {
    /// The call instruction that this callsite label refers to.
    pub call_inst: PackedOption<Inst>,
}

impl fmt::Display for CallSiteData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // We don't write out the instruction index in the textual
        // form of CLIF; we fill it in automatically when parsing.
        write!(f, "callsite")
    }
}
