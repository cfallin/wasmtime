//! Support for egraphs represented in the DataFlowGraph.

use crate::ctxhash::{CtxEq, CtxHash, CtxHashMap};
use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::egraph::domtree::DomTreeWithChildren;
use crate::egraph::elaborate::Elaborator;
use crate::egraph::Stats;
use crate::flowgraph::ControlFlowGraph;
use crate::fx::FxHashSet;
use crate::inst_predicates::is_pure_for_egraph;
use crate::ir::{Block, Function, Inst, InstructionData, Layout, Value, ValueDef, ValueListPool};
use crate::trace;
use crate::unionfind::UnionFind;
use cranelift_entity::packed_option::ReservedValue;
use cranelift_entity::SecondaryMap;
use std::hash::Hasher;

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
    /// "Domtree with children": like `domtree`, but with an explicit
    /// list of children, rather than just parent pointers.
    domtree_children: DomTreeWithChildren,
    /// Loop analysis results, used for built-in LICM during
    /// elaboration. See below about "pseudo-loop-levels".
    pseudo_loop_levels: PseudoLoopLevels,
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
    pub fn new(func: &'a mut Function, domtree: &'a DominatorTree, cfg: &ControlFlowGraph) -> Self {
        let num_values = func.dfg.num_values();
        let domtree_children = DomTreeWithChildren::new(func, domtree);
        let entry = func
            .layout
            .entry_block()
            .expect("Function must have an entry block");
        let pseudo_loop_levels =
            PseudoLoopLevels::compute(domtree, &func.layout, &domtree_children, cfg, entry);
        Self {
            func,
            domtree,
            domtree_children,
            pseudo_loop_levels,
            stats: Stats::default(),
            eclasses: UnionFind::with_capacity(num_values),
            remat_values: FxHashSet::default(),
            analysis_values: SecondaryMap::with_capacity(num_values),
        }
    }

    /// Run the process.
    pub fn run(&mut self) {
        self.remove_pure_and_optimize();
        self.elaborate();
    }

    /// Remove pure nodes from the `Layout` of the function, ensuring
    /// that only the "side-effect skeleton" remains, and also
    /// optimize the pure nodes. This is the first step of
    /// egraph-based processing and turns the pure CFG-based CLIF into
    /// a CFG skeleton with a sea of (optimized) nodes tying it
    /// together.
    ///
    /// As we walk through the code, we eagerly apply optimization
    /// rules; at any given point we have a "latest version" of an
    /// eclass of possible representations for a `Value` in the
    /// original program, which is itself a `Value` at the root of a
    /// union-tree. We keep a map from the original values to these
    /// optimized values. When we encounter any instruction (pure or
    /// side-effecting skeleton) we rewrite its arguments to capture
    /// the "latest" optimized forms of these values. (We need to do
    /// this as part of this pass, and not later using a finished map,
    /// because the eclass can continue to be updated and we need to
    /// only refer to its subset that exists at this stage, to
    /// maintain acyclicity.)
    fn remove_pure_and_optimize(&mut self) {
        let mut cursor = FuncCursor::new(self.func);
        let mut value_to_opt_value: SecondaryMap<Value, Value> =
            SecondaryMap::with_default(Value::reserved_value());
        let mut gvn_map: CtxHashMap<InstructionData, Value> =
            CtxHashMap::with_capacity(cursor.func.dfg.num_values());

        while let Some(block) = cursor.next_block() {
            for &param in cursor.func.dfg.block_params(block) {
                trace!("creating initial singleton eclass for {}", param);
                self.eclasses.add(param);
                Self::compute_analysis_value(
                    cursor.func,
                    &self.pseudo_loop_levels,
                    &mut self.analysis_values,
                    param,
                );
                value_to_opt_value[param] = param;
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
                        &self.pseudo_loop_levels,
                        &mut self.analysis_values,
                        result,
                    );
                }

                // Rewrite args of *all* instructions using the
                // value-to-opt-value map.
                for arg in cursor.func.dfg.inst_args_mut(inst) {
                    *arg = value_to_opt_value[*arg];
                }

                if is_pure_for_egraph(cursor.func, inst) {
                    // Insert into GVN map and optimize any new nodes
                    // inserted (recursively performing this work for
                    // any nodes the optimization rules produce).
                    Self::insert_pure_enode(
                        cursor.func,
                        inst,
                        &mut value_to_opt_value,
                        &mut gvn_map,
                        &mut self.eclasses,
                    );
                    // We've now rewritten all uses, or will when we
                    // see them, and the instruction exists as a pure
                    // enode in the eclass, so we can remove it.
                    cursor.remove_inst_and_step_back();
                } else {
                    // Not pure, but may still be a store: add it to
                    // the store-map if so so store-to-load forwarding
                    // can work properly.
                    todo!("store-map update");
                }
            }
        }
    }

    /// Optimization of a single instruction.
    ///
    /// This does a few things:
    /// - Looks up the instruction in the GVN deduplication map. If we
    ///   already have the same instruction somewhere else, with the
    ///   same args, then we can alias the original instruction's
    ///   results and omit this instruction entirely.
    ///   - Note that we do this canonicalization based on the
    ///     instruction with its arguments as *canonical* eclass IDs,
    ///     that is, the oldest (smallest index) `Value` reachable in
    ///     the tree-of-unions (whole eclass). This ensures that we
    ///     properly canonicalize newer nodes that use newer "versions"
    ///     of a value that are still equal to the older versions.
    /// - If the instruction is "new" (not deduplicated), then apply
    ///   optimization rules:
    ///   - All of the mid-end rules written in ISLE.
    ///   - Store-to-load forwarding.
    /// - Update the value-to-opt-value map, and update the eclass
    ///   union-find, if we rewrote the value to different form(s).
    fn insert_pure_enode(
        func: &mut Function,
        inst: Inst,
        value_to_opt_value: &mut SecondaryMap<Value, Value>,
        gvn_map: &mut CtxHashMap<InstructionData, Value>,
        eclasses: &mut UnionFind<Value>,
    ) {
        // Create the external context for looking up and updating the
        // GVN map. This is necessary so that instructions themselves
        // do not have to carry all the references or data for a full
        // `Eq` or `Hash` impl.
        let gvn_context = GVNContext {
            union_find: eclasses,
            value_lists: &func.dfg.value_lists,
        };

        // Required for proper logic below.
        debug_assert_eq!(func.dfg.inst_results(inst).len(), 1);
        let result = func.dfg.inst_results(inst)[0];

        // Does this instruction already exist? If so, add entries to
        // the value-map to rewrite uses of its results to the results
        // of the original (existing) instruction. If not, optimize
        // the new instruction.
        if let Some(&orig_result) = gvn_map.get(&func.dfg[inst], &gvn_context) {
            value_to_opt_value[result] = orig_result;
            eclasses.union(result, orig_result);
        } else {
            let opt_value =
                Self::optimize_pure_enode(func, inst, value_to_opt_value, gvn_map, eclasses);
            let gvn_context = GVNContext {
                union_find: eclasses,
                value_lists: &func.dfg.value_lists,
            };
            gvn_map.insert(func.dfg[inst].clone(), opt_value, &gvn_context);
            value_to_opt_value[result] = opt_value;
        }
    }

    /// Optimizes an enode by applying any matching mid-end rewrite
    /// rules (or store-to-load forwarding, which is a special case),
    /// unioning together all possible optimized (or rewritten) forms
    /// of this expression into an eclass and returning the `Value`
    /// that represents that eclass.
    ///
    /// TODO: wrap up args into a context struct (here and above in
    /// insert_pure_enode).
    fn optimize_pure_enode(
        _func: &mut Function,
        _inst: Inst,
        _value_to_opt_value: &mut SecondaryMap<Value, Value>,
        _gvn_map: &mut CtxHashMap<InstructionData, Value>,
        _eclasses: &mut UnionFind<Value>,
    ) -> Value {
        todo!()
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
            &self.domtree_children,
            &self.pseudo_loop_levels,
            &mut self.remat_values,
            &self.analysis_values,
            &mut self.eclasses,
            &mut self.stats,
        );
        elaborator.elaborate();
    }

    /// Compute analysis values for a given Value.
    fn compute_analysis_value(
        func: &Function,
        pseudo_loop_levels: &PseudoLoopLevels,
        analysis_values: &mut SecondaryMap<Value, AnalysisValue>,
        value: Value,
    ) {
        let aval = match func.dfg.value_def(value) {
            ValueDef::Result(inst, _result_idx) => {
                // TODO: get max loop level from args, rather than
                // taking original block's loop level.
                let block = func.layout.inst_block(inst).unwrap();
                AnalysisValue::for_block(pseudo_loop_levels, block)
            }
            ValueDef::Param(block, _idx) => AnalysisValue::for_block(pseudo_loop_levels, block),
            ValueDef::Union(x, y) => {
                // Meet the two analysis values.
                let x_val = &analysis_values[x];
                let y_val = &analysis_values[y];
                AnalysisValue::meet(x_val, y_val)
            }
        };

        trace!("Analysis value for {} is {:?}", value, aval);
        analysis_values[value] = aval;
    }
}

/// Implementation of external-context equality and hashing on
/// InstructionData. This allows us to deduplicate instructions given
/// some context that lets us see its value lists and the mapping from
/// any value to "canonical value" (in an eclass).
struct GVNContext<'a> {
    value_lists: &'a ValueListPool,
    union_find: &'a UnionFind<Value>,
}

impl<'a> CtxEq<InstructionData, InstructionData> for GVNContext<'a> {
    fn ctx_eq(&self, a: &InstructionData, b: &InstructionData) -> bool {
        a.eq(b, self.value_lists, |value| self.union_find.find(value))
    }
}

impl<'a> CtxHash<InstructionData> for GVNContext<'a> {
    fn ctx_hash(&self, inst: &InstructionData) -> u64 {
        let mut state = crate::fx::FxHasher::default();
        inst.hash(&mut state, self.value_lists, |value| {
            self.union_find.find(value)
        });
        state.finish()
    }
}

/// Pseudo-loop-level is like LoopLevel, but for domtrees.
///
/// More specifically: the pseudo-loop-level of a node (block) in the
/// domtree is the number of loop headers that exist in the path from
/// the root to that node.
///
/// The difference between this and an actual loop-nest-based "loop
/// level" arises because execution can leave a loop but remain in a
/// part of the CFG that is dominated by the loop body (if the exit is
/// at the end of the loop). Then we could use a value defined
/// *inside* the loop, statically (and computed in the last iteration
/// of the loop) when we are outside the loop.
///
/// In order to make our heads hurt less, we do LICM in terms of
/// pseudo-loop-level. This allows us to track the loop nest more
/// naturally alongside the domtree preorder traversal for
/// elaboration: the loop stack is a sub-stack of the total domtree
/// path for a given node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) struct PseudoLoopLevel(u32);
impl PseudoLoopLevel {
    pub(crate) fn inc(self) -> Self {
        Self(
            self.0
                .checked_add(1)
                .expect("Too many loops! (Limit of 2^32.)"),
        )
    }
    pub(crate) fn level(self) -> usize {
        self.0 as usize
    }
}

pub(crate) struct PseudoLoopLevels {
    pub(crate) levels: SecondaryMap<Block, PseudoLoopLevel>,
    pub(crate) headers: FxHashSet<Block>,
}
impl PseudoLoopLevels {
    pub(crate) fn compute(
        domtree: &DominatorTree,
        layout: &Layout,
        domtree_children: &DomTreeWithChildren,
        cfg: &ControlFlowGraph,
        entry: Block,
    ) -> Self {
        let mut stack = vec![];
        struct StackEntry {
            block: Block,
            level: PseudoLoopLevel,
        }
        stack.push(StackEntry {
            block: entry,
            level: PseudoLoopLevel(0),
        });

        let mut levels = SecondaryMap::default();
        let mut headers = FxHashSet::default();
        while let Some(entry) = stack.pop() {
            // Determine whether `entry.block` is a loop header: check
            // all preds to see if any are dominated by `entry.block`.
            let is_header = cfg
                .pred_iter(entry.block)
                .any(|pred| domtree.dominates(entry.block, pred.block, layout));
            let this_level = if is_header {
                headers.insert(entry.block);
                entry.level.inc()
            } else {
                entry.level
            };

            levels[entry.block] = this_level;
            trace!(
                "PseudoLoopLevels::compute: block {} level {:?} is_header {}",
                entry.block,
                this_level,
                is_header
            );

            for child in domtree_children.children(entry.block) {
                stack.push(StackEntry {
                    block: child,
                    level: this_level,
                });
            }
        }
        Self { levels, headers }
    }

    pub(crate) fn is_loop_header(&self, block: Block) -> bool {
        self.headers.contains(&block)
    }
    pub(crate) fn pseudo_loop_level(&self, block: Block) -> PseudoLoopLevel {
        self.levels[block]
    }
}

/// Analysis results for each eclass id.
#[derive(Clone, Debug)]
pub(crate) struct AnalysisValue {
    pub(crate) pseudo_loop_level: PseudoLoopLevel,
}

impl Default for AnalysisValue {
    fn default() -> Self {
        Self {
            pseudo_loop_level: PseudoLoopLevel(0),
        }
    }
}

impl AnalysisValue {
    fn meet(x: &AnalysisValue, y: &AnalysisValue) -> AnalysisValue {
        AnalysisValue {
            pseudo_loop_level: std::cmp::max(x.pseudo_loop_level, y.pseudo_loop_level),
        }
    }

    fn for_block(loop_analysis: &PseudoLoopLevels, block: Block) -> AnalysisValue {
        let pseudo_loop_level = loop_analysis.pseudo_loop_level(block);
        trace!(
            "AnalysisValue::for_block: block {} -> level {:?}",
            block,
            pseudo_loop_level
        );
        AnalysisValue { pseudo_loop_level }
    }
}
