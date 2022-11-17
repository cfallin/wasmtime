//! Support for egraphs represented in the DataFlowGraph.

use crate::cursor::{Cursor, FuncCursor};
use crate::inst_predicates::is_pure_for_egraph;
use crate::ir::Function;

/// Pass over a Function that does the whole aegraph thing.
///
/// - Removes non-skeleton nodes from the Layout.
/// - Performs a GVN-and-rule-application pass over all Values
///   reachable from the skeleton, potentially creating new Union
///   nodes (i.e., an aegraph) so that some values have multiple
///   representations.
/// - Does "extraction" on the aegraph: selects the best value out of
///   the tree-of-Union nodes for each used value.
/// - Does "scoped elaboration" on the aegraph: chooses one or more
///   locations for pure nodes to become instructions again in the
///   layout, as forced by the skeleton.
///
/// At the beginning and end of this pass, the CLIF should be in a
/// state that passes the verifier and, additionally, has no Union
/// nodes. During the pass, Union nodes may exist, and instructions in
/// the layout may refer to results of instructions that are not
/// placed in the layout.
pub struct EgraphPass<'a> {
    func: &'a mut Function,
}

impl<'a> EgraphPass<'a> {
    /// Create a new EgraphPass.
    pub fn new(func: &'a mut Function) -> Self {
        Self { func }
    }

    /// Run the process.
    pub fn run(&mut self) {
        self.remove_pure();
    }

    /// Remove pure nodes from the `Layout` of the function, ensuring
    /// that only the "side-effect skeleton" remains. This is the
    /// first step of egraph-based processing and allows the egraph to
    /// reason about pure nodes and move them freely.
    fn remove_pure(&mut self) {
        let mut cursor = FuncCursor::new(self.func);
        while let Some(_block) = cursor.next_block() {
            while let Some(inst) = cursor.next_inst() {
                if is_pure_for_egraph(cursor.func, inst) {
                    cursor.remove_inst();
                }
            }
        }
    }
}
