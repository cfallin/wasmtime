//! Support for egraphs represented in the DataFlowGraph.

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::egraph::elaborate::Elaborator;
use crate::egraph::Stats;
use crate::flowgraph::ControlFlowGraph;
use crate::fx::FxHashSet;
use crate::inst_predicates::is_pure_for_egraph;
use crate::ir::{Function, Value};
use crate::loop_analysis::{LoopAnalysis, LoopLevel};

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
    /// The function we're operating on.
    func: &'a mut Function,
    /// Dominator tree, used for elaboration pass.
    domtree: &'a DominatorTree,
    /// Loop analysis results, used for built-in LICM during elaboration.
    loop_analysis: &'a LoopAnalysis,
    /// Which canonical Values do we want to rematerialize in each
    /// block where they're used?
    ///
    /// (A canonical Value is the *oldest* Value in an eclass,
    /// i.e. tree of union value-nodes).
    pub(crate) remat_values: FxHashSet<Value>,
    stats: Stats,
}

impl<'a> EgraphPass<'a> {
    /// Create a new EgraphPass.
    pub fn new(
        func: &'a mut Function,
        domtree: &'a DominatorTree,
        loop_analysis: &'a LoopAnalysis,
        // Only used for AliasAnalysis (TODO).
        _cfg: &ControlFlowGraph,
    ) -> Self {
        Self {
            func,
            domtree,
            loop_analysis,
            cfg,
            stats: Stats::default(),
        }
    }

    /// Run the process.
    pub fn run(&mut self) {
        self.remove_pure();
        self.elaborate();
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

    /// Scoped elaboration: compute a final ordering of op computation
    /// for each block and update the given Func body. After this
    /// runs, the function body is back into the state where every
    /// Inst with an used result is placed in the layout (possibly
    /// duplicated, if our code-motion logic decides this is the best
    /// option).
    ///
    /// This works in concert with the domtree. We do a preorder
    /// traversal of the domtree, tracking a scoped map from Id to
    /// (new) Value. The map's scopes correspond to levels in the
    /// domtree.
    ///
    /// At each block, we iterate forward over the side-effecting
    /// eclasses, and recursively generate their arg eclasses, then
    /// emit the ops themselves.
    ///
    /// To use an eclass in a given block, we first look it up in the
    /// scoped map, and get the Value if already present. If not, we
    /// need to generate it. We emit the extracted enode for this
    /// eclass after recursively generating its args. Eclasses are
    /// thus computed "as late as possible", but then memoized into
    /// the Id-to-Value map and available to all dominated blocks and
    /// for the rest of this block. (This subsumes GVN.)
    fn elaborate(&mut self) {
        let elaborator = Elaborator::new(
            self.func,
            self.domtree,
            self.loop_analysis,
            &mut self.remat_ids,
            &mut self.stats,
        );
        elab.elaborate();
    }
}
