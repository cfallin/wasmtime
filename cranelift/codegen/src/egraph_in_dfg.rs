//! Support for egraphs represented in the DataFlowGraph.

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::egraph::elaborate::Elaborator;
use crate::egraph::Stats;
use crate::flowgraph::ControlFlowGraph;
use crate::fx::FxHashSet;
use crate::inst_predicates::is_pure_for_egraph;
use crate::ir::{Block, Function, Value, ValueDef};
use crate::loop_analysis::{LoopAnalysis, LoopLevel};
use crate::trace;
use crate::unionfind::UnionFind;
use cranelift_entity::SecondaryMap;

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
    remat_values: FxHashSet<Value>,
    /// Stats collected while we run this pass.
    pub(crate) stats: Stats,
    /// Union-find that maps all members of a Union tree (eclass) back
    /// to the *oldest* (lowest-numbered) `Value`.
    eclasses: UnionFind<Value>,
    /// Analysis values per `Value`.
    analysis_values: SecondaryMap<Value, AnalysisValue>,
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
        let num_values = func.dfg.num_values();
        Self {
            func,
            domtree,
            loop_analysis,
            stats: Stats::default(),
            eclasses: UnionFind::with_capacity(num_values),
            remat_values: FxHashSet::default(),
            analysis_values: SecondaryMap::with_capacity(num_values),
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
        while let Some(block) = cursor.next_block() {
            for &param in cursor.func.dfg.block_params(block) {
                trace!("creating initial singleton eclass for {}", param);
                self.eclasses.add(param);
                Self::compute_analysis_value(
                    cursor.func,
                    &self.loop_analysis,
                    &mut self.analysis_values,
                    param,
                );
            }
            while let Some(inst) = cursor.next_inst() {
                // While we're passing over all insts, create initial
                // singleton eclasses for all result and blockparam
                // values.  Also do initial analysis of all inst
                // results.
                for &result in cursor.func.dfg.inst_results(inst) {
                    trace!("creating initial singleton eclass for {}", result);
                    self.eclasses.add(result);
                    Self::compute_analysis_value(
                        cursor.func,
                        &self.loop_analysis,
                        &mut self.analysis_values,
                        result,
                    );
                }

                if is_pure_for_egraph(cursor.func, inst) {
                    cursor.remove_inst_and_step_back();
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
        let mut elaborator = Elaborator::new(
            self.func,
            self.domtree,
            self.loop_analysis,
            &mut self.remat_values,
            &self.analysis_values,
            &self.eclasses,
            &mut self.stats,
        );
        elaborator.elaborate();
    }

    /// Compute analysis values for a given Value.
    fn compute_analysis_value(
        func: &Function,
        loop_analysis: &LoopAnalysis,
        analysis_values: &mut SecondaryMap<Value, AnalysisValue>,
        value: Value,
    ) {
        let aval = match func.dfg.value_def(value) {
            ValueDef::Result(inst, _result_idx) => {
                // TODO: get max loop level from args, rather than
                // taking original block's loop level.
                let block = func.layout.inst_block(inst).unwrap();
                AnalysisValue::for_block(&loop_analysis, block)
            }
            ValueDef::Param(block, _idx) => AnalysisValue::for_block(&loop_analysis, block),
            ValueDef::Union(x, y) => {
                // Meet the two analysis values.
                let x_val = &analysis_values[x];
                let y_val = &analysis_values[y];
                AnalysisValue::meet(x_val, y_val)
            }
        };

        analysis_values[value] = aval;
    }
}

/// Analysis results for each eclass id.
#[derive(Clone, Debug)]
pub(crate) struct AnalysisValue {
    pub(crate) loop_level: LoopLevel,
}

impl Default for AnalysisValue {
    fn default() -> Self {
        Self {
            loop_level: LoopLevel::root(),
        }
    }
}

impl AnalysisValue {
    fn meet(x: &AnalysisValue, y: &AnalysisValue) -> AnalysisValue {
        AnalysisValue {
            loop_level: std::cmp::max(x.loop_level, y.loop_level),
        }
    }

    fn for_block(loop_analysis: &LoopAnalysis, block: Block) -> AnalysisValue {
        let loop_level = loop_analysis.loop_level(block);
        AnalysisValue { loop_level }
    }
}
