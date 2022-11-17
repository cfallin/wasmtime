//! Elaboration phase: lowers EGraph back to sequences of operations
//! in CFG nodes.

use super::domtree::DomTreeWithChildren;
use super::node::{pure_op_cost, Cost};
use super::Analysis;
use super::AnalysisValue;
use super::Stats;
use crate::dominator_tree::DominatorTree;
use crate::fx::FxHashSet;
use crate::ir::ValueDef;
use crate::ir::{Block, Function, Inst, Opcode, RelSourceLoc, Type, Value, ValueList};
use crate::loop_analysis::LoopAnalysis;
use crate::scoped_hash_map::ScopedHashMap;
use crate::trace;
use alloc::vec::Vec;
use cranelift_entity::{packed_option::PackedOption, packed_option::ReservedValue, SecondaryMap};
use smallvec::{smallvec, SmallVec};
use std::ops::Add;

type LoopDepth = u32;

pub(crate) struct Elaborator<'a> {
    func: &'a mut Function,
    domtree: &'a DominatorTree,
    loop_analysis: &'a LoopAnalysis,
    analysis_values: &'a SecondaryMap<Value, AnalysisValue>,
    /// Map from Inst that is pure (and was thus not in the
    /// side-effecting skeleton) to the elaborated inst (placed in the
    /// layout) to whose results we refer in the final code.
    ///
    /// The first time we use some result of an instruction during
    /// elaboration, we can place it and insert an identity map (inst
    /// to that same inst) in this scoped map. Within that block and
    /// its dom-tree children, that mapping is visible and we can
    /// continue to use it. This allows us to avoid cloning the
    /// instruction. However, if we pop that scope and use it
    /// somewhere else as well, we will need to duplicate. We detect
    /// this case by checking, when an Inst is not present in this
    /// map, whether it is already placed in the Layout. If so, we
    /// duplicate, and insert non-identity mappings from the original
    /// inst to the cloned inst.
    inst_to_elaborated_inst: ScopedHashMap<Inst, Inst>,
    /// Map from Value to the best (lowest-cost) Value in its eclass
    /// (tree of union value-nodes).
    value_to_best_value: SecondaryMap<Value, (Cost, Value)>,
    /// Stack of blocks and loops in current elaboration path.
    loop_stack: SmallVec<[LoopStackEntry; 8]>,
    cur_block: Option<Block>,
    first_branch: SecondaryMap<Block, PackedOption<Inst>>,
    /// Values that opt rules have indicated should be rematerialized
    /// in every block they are used (e.g., immediates or other
    /// "cheap-to-compute" ops).
    remat_values: &'a FxHashSet<Value>,
    /// Explicitly-unrolled value elaboration stack.
    elab_stack: Vec<ElabStackEntry>,
    elab_result_stack: Vec<Value>,
    /// Explicitly-unrolled block elaboration stack.
    block_stack: Vec<BlockStackEntry>,
    stats: &'a mut Stats,
}

#[derive(Clone, Debug)]
struct LoopStackEntry {
    /// The hoist point: a block that immediately dominates this
    /// loop. May not be an immediate predecessor, but will be a valid
    /// point to place all loop-invariant ops: they must depend only
    /// on inputs that dominate the loop, so are available at (the end
    /// of) this block.
    hoist_block: Block,
    /// The depth in the scope map.
    scope_depth: u32,
}

#[derive(Clone, Debug)]
enum ElabStackEntry {
    /// Next action is to resolve this inst into an elaborated inst
    /// (placed into the layout) recursively elaborate the insts that
    /// produce its args.
    ///
    /// Any inserted ops should be inserted before `before`, which is
    /// the instruction demanding this value.
    Start {
        inst: Inst,
        result_idx: usize,
        before: Inst,
    },
    /// Args have been pushed; waiting for results.
    PendingInst {
        inst: Inst,
        remat: bool,
        num_args: usize,
        result_idx: usize,
        before: Inst,
    },
}

#[derive(Clone, Debug)]
enum BlockStackEntry {
    Elaborate { block: Block, idom: Option<Block> },
    Pop,
}

impl<'a> Elaborator<'a> {
    pub(crate) fn new(
        func: &'a mut Function,
        domtree: &'a DominatorTree,
        loop_analysis: &'a LoopAnalysis,
        remat_values: &'a FxHashSet<Value>,
        analysis_values: &'a SecondaryMap<Value, AnalysisValue>,
        stats: &'a mut Stats,
    ) -> Self {
        let num_blocks = func.dfg.num_blocks();
        let num_insts = func.dfg.num_insts();
        let num_values = func.dfg.num_values();
        let mut value_to_best_value =
            SecondaryMap::with_default((Cost::infinity(), Value::reserved_value()));
        value_to_best_value.resize(num_values);
        Self {
            func,
            domtree,
            loop_analysis,
            inst_to_elaborated_inst: ScopedHashMap::with_capacity(num_insts),
            value_to_best_value,
            loop_stack: smallvec![],
            cur_block: None,
            first_branch: SecondaryMap::with_capacity(num_blocks),
            remat_values,
            elab_stack: vec![],
            elab_result_stack: vec![],
            block_stack: vec![],
            analysis_values,
            stats,
        }
    }

    fn cur_loop_depth(&self) -> LoopDepth {
        self.loop_stack.len() as LoopDepth
    }

    fn start_block(&mut self, idom: Option<Block>, block: Block) {
        trace!(
            "start_block: block {:?} with idom {:?} at loop depth {} scope depth {}",
            block,
            idom,
            self.cur_loop_depth(),
            self.inst_to_elaborated_inst.depth()
        );

        // Note that if the *entry* block is a loop header, we will
        // not make note of the loop here because it will not have an
        // immediate dominator. We must disallow this case because we
        // will skip adding the `LoopStackEntry` here but our
        // `LoopAnalysis` will otherwise still make note of this loop
        // and loop depths will not match.
        if let Some(idom) = idom {
            if self.loop_analysis.is_loop_header(block).is_some() {
                self.loop_stack.push(LoopStackEntry {
                    // Any code hoisted out of this loop will have code
                    // placed in `idom`, and will have def mappings
                    // inserted in to the scoped hashmap at that block's
                    // level.
                    hoist_block: idom,
                    scope_depth: (self.inst_to_elaborated_inst.depth() - 1) as u32,
                });
                trace!(
                    " -> loop header, pushing; depth now {}",
                    self.loop_stack.len()
                );
            }
        } else {
            debug_assert!(
                self.loop_analysis.is_loop_header(block).is_none(),
                "Entry block (domtree root) cannot be a loop header!"
            );
        }

        self.cur_block = Some(block);
    }

    fn compute_best_nodes(&mut self) {
        let best = &mut self.value_to_best_value;
        for (value, def) in self.func.dfg.values_and_defs() {
            trace!("computing best for value {:?} def {:?}", value, def);
            match def {
                ValueDef::Union(x, y) => {
                    // Pick the best of the two options based on
                    // min-cost. This works because each element of `best`
                    // is a `(cost, value)` tuple; `cost` comes first so
                    // the natural comparison works based on cost, and
                    // breaks ties based on value number.
                    trace!(" -> best of {:?} and {:?}", best[x], best[y]);
                    best[value] = std::cmp::min(best[x], best[y]);
                    trace!(" -> {:?}", best[value]);
                }
                ValueDef::Param(_, _) => {
                    best[value] = (Cost::zero(), value);
                }
                // If the Inst is inserted into the layout (which is,
                // at this point, only the side-effecting skeleton),
                // then it must be computed and thus we give it zero
                // cost.
                ValueDef::Result(inst, _) if self.func.layout.inst_block(inst).is_some() => {
                    best[value] = (Cost::zero(), value);
                }
                ValueDef::Result(inst, _) => {
                    trace!(" -> value {}: result, computing cost", value);
                    let inst_data = &self.func.dfg[inst];
                    let level = self.analysis_values[value].loop_level;
                    // N.B.: at this point we know that the opcode is
                    // pure, so `pure_op_cost`'s precondition is
                    // satisfied.
                    let cost = pure_op_cost(inst_data.opcode()).at_level(level)
                        + self
                            .func
                            .dfg
                            .inst_args(inst)
                            .iter()
                            .map(|value| best[*value].0)
                            // Can't use `.sum()` for `Cost` types; do
                            // an explicit reduce instead.
                            .fold(Cost::zero(), Cost::add);
                    best[value] = (cost, value);
                }
            };
            debug_assert_ne!(best[value].0, Cost::infinity());
            debug_assert_ne!(best[value].1, Value::reserved_value());
            trace!("best for eclass {:?}: {:?}", value, best[value]);
        }
    }

    fn elaborate_eclass_use(&mut self, value: Value) {
        self.elab_stack.push(ElabStackEntry::Start { id });
        self.process_elab_stack();
        debug_assert_eq!(self.elab_result_stack.len(), 1);
        self.elab_result_stack.clear();
    }

    fn process_elab_stack(&mut self) {
        while let Some(entry) = self.elab_stack.last() {
            match entry {
                &ElabStackEntry::Start { id } => {
                    // We always replace the Start entry, so pop it now.
                    self.elab_stack.pop();

                    self.stats.elaborate_visit_node += 1;
                    let canonical = self.egraph.canonical_id(id);
                    trace!("elaborate: id {}", id);

                    let remat = if let Some(val) = self.id_to_value.get(&canonical) {
                        // Look at the defined block, and determine whether this
                        // node kind allows rematerialization if the value comes
                        // from another block. If so, ignore the hit and recompute
                        // below.
                        let remat = val.block() != self.cur_block.unwrap()
                            && self.remat_ids.contains(&canonical);
                        if !remat {
                            trace!("elaborate: id {} -> {:?}", id, val);
                            self.stats.elaborate_memoize_hit += 1;
                            self.elab_result_stack.push(val.clone());
                            continue;
                        }
                        trace!("elaborate: id {} -> remat", id);
                        self.stats.elaborate_memoize_miss_remat += 1;
                        // The op is pure at this point, so it is always valid to
                        // remove from this map.
                        self.id_to_value.remove(&canonical);
                        true
                    } else {
                        self.remat_ids.contains(&canonical)
                    };
                    self.stats.elaborate_memoize_miss += 1;

                    // Get the best option; we use `id` (latest id) here so we
                    // have a full view of the eclass.
                    let (_, best_node_eclass) = self.id_to_best_cost_and_node[id];
                    debug_assert_ne!(best_node_eclass, Id::invalid());

                    trace!(
                        "elaborate: id {} -> best {} -> eclass node {:?}",
                        id,
                        best_node_eclass,
                        self.egraph.classes[best_node_eclass]
                    );
                    let node_key = self.egraph.classes[best_node_eclass].get_node().unwrap();
                    let node = node_key.node(&self.egraph.nodes);
                    trace!(" -> enode {:?}", node);

                    // Is the node a block param? We should never get here if so
                    // (they are inserted when first visiting the block).
                    if matches!(node, Node::Param { .. }) {
                        unreachable!("Param nodes should already be inserted");
                    }

                    // Is the node a result projection? If so, resolve
                    // the value we are projecting a part of, then
                    // eventually return here (saving state with a
                    // PendingProjection).
                    if let Node::Result {
                        value, result, ty, ..
                    } = node
                    {
                        trace!(" -> result; pushing arg value {}", value);
                        self.elab_stack.push(ElabStackEntry::PendingProjection {
                            index: *result,
                            canonical,
                            ty: *ty,
                        });
                        self.elab_stack.push(ElabStackEntry::Start { id: *value });
                        continue;
                    }

                    // We're going to need to emit this
                    // operator. First, enqueue all args to be
                    // elaborated. Push state to receive the results
                    // and later elab this node.
                    let num_args = self.node_ctx.children(&node).len();
                    self.elab_stack.push(ElabStackEntry::PendingNode {
                        canonical,
                        node_key,
                        remat,
                        num_args,
                    });
                    // Push args in reverse order so we process the
                    // first arg first.
                    for &arg_id in self.node_ctx.children(&node).iter().rev() {
                        self.elab_stack.push(ElabStackEntry::Start { id: arg_id });
                    }
                }

                &ElabStackEntry::PendingNode {
                    canonical,
                    node_key,
                    remat,
                    num_args,
                } => {
                    self.elab_stack.pop();

                    let node = node_key.node(&self.egraph.nodes);

                    // We should have all args resolved at this point.
                    let arg_idx = self.elab_result_stack.len() - num_args;
                    let args = &self.elab_result_stack[arg_idx..];

                    // Gather the individual output-CLIF `Value`s.
                    let arg_values: SmallVec<[Value; 8]> = args
                        .iter()
                        .map(|idvalue| match idvalue {
                            IdValue::Value { value, .. } => *value,
                            IdValue::Values { .. } => {
                                panic!("enode depends directly on multi-value result")
                            }
                        })
                        .collect();

                    // Compute max loop depth.
                    let max_loop_depth = args
                        .iter()
                        .map(|idvalue| match idvalue {
                            IdValue::Value { depth, .. } => *depth,
                            IdValue::Values { .. } => unreachable!(),
                        })
                        .max()
                        .unwrap_or(0);

                    // Remove args from result stack.
                    self.elab_result_stack.truncate(arg_idx);

                    // Determine the location at which we emit it. This is the
                    // current block *unless* we hoist above a loop when all args
                    // are loop-invariant (and this op is pure).
                    let (loop_depth, scope_depth, block) = if node.is_non_pure() {
                        // Non-pure op: always at the current location.
                        (
                            self.cur_loop_depth(),
                            self.id_to_value.depth(),
                            self.cur_block.unwrap(),
                        )
                    } else if max_loop_depth == self.cur_loop_depth() || remat {
                        // Pure op, but depends on some value at the current loop
                        // depth, or remat forces it here: as above.
                        (
                            self.cur_loop_depth(),
                            self.id_to_value.depth(),
                            self.cur_block.unwrap(),
                        )
                    } else {
                        // Pure op, and does not depend on any args at current
                        // loop depth: hoist out of loop.
                        self.stats.elaborate_licm_hoist += 1;
                        let data = &self.loop_stack[max_loop_depth as usize];
                        (max_loop_depth, data.scope_depth as usize, data.hoist_block)
                    };
                    // Loop scopes are a subset of all scopes.
                    debug_assert!(scope_depth >= loop_depth as usize);

                    // This is an actual operation; emit the node in sequence now.
                    let results = self.add_node(node, &arg_values[..], block);
                    let results_slice = results.as_slice(&self.func.dfg.value_lists);

                    // Build the result and memoize in the id-to-value map.
                    let result = if results_slice.len() == 1 {
                        IdValue::Value {
                            depth: loop_depth,
                            block,
                            value: results_slice[0],
                        }
                    } else {
                        IdValue::Values {
                            depth: loop_depth,
                            block,
                            values: results,
                        }
                    };

                    self.id_to_value.insert_if_absent_with_depth(
                        canonical,
                        result.clone(),
                        scope_depth,
                    );

                    // Push onto the elab-results stack.
                    self.elab_result_stack.push(result)
                }
                &ElabStackEntry::PendingProjection {
                    ty,
                    index,
                    canonical,
                } => {
                    self.elab_stack.pop();

                    // Grab the input from the elab-result stack.
                    let value = self.elab_result_stack.pop().expect("Should have result");

                    let (depth, block, values) = match value {
                        IdValue::Values {
                            depth,
                            block,
                            values,
                            ..
                        } => (depth, block, values),
                        IdValue::Value { .. } => {
                            unreachable!("Projection nodes should not be used on single results");
                        }
                    };
                    let values = values.as_slice(&self.func.dfg.value_lists);
                    let value = values[index];
                    self.func.dfg.fill_in_value_type(value, ty);
                    let value = IdValue::Value {
                        depth,
                        block,
                        value,
                    };
                    self.id_to_value.insert_if_absent(canonical, value.clone());

                    self.elab_result_stack.push(value);
                }
            }
        }
    }

    fn elaborate_block<'b, PF: Fn(Block) -> &'b [(Id, Type)], SEF: Fn(Block) -> &'b [Id]>(
        &mut self,
        idom: Option<Block>,
        block: Block,
        block_params_fn: &PF,
        block_side_effects_fn: &SEF,
    ) {
        let blockparam_ids_tys = (block_params_fn)(block);
        self.start_block(idom, block, blockparam_ids_tys);
        for &id in (block_side_effects_fn)(block) {
            self.elaborate_eclass_use(id);
        }
    }

    fn elaborate_domtree<'b, PF: Fn(Block) -> &'b [(Id, Type)], SEF: Fn(Block) -> &'b [Id]>(
        &mut self,
        block_params_fn: &PF,
        block_side_effects_fn: &SEF,
        domtree: &DomTreeWithChildren,
    ) {
        let root = domtree.root();
        self.block_stack.push(BlockStackEntry::Elaborate {
            block: root,
            idom: None,
        });
        while let Some(top) = self.block_stack.pop() {
            match top {
                BlockStackEntry::Elaborate { block, idom } => {
                    self.block_stack.push(BlockStackEntry::Pop);
                    self.id_to_value.increment_depth();

                    self.elaborate_block(idom, block, block_params_fn, block_side_effects_fn);

                    // Push children. We are doing a preorder
                    // traversal so we do this after processing this
                    // block above.
                    let block_stack_end = self.block_stack.len();
                    for child in domtree.children(block) {
                        self.block_stack.push(BlockStackEntry::Elaborate {
                            block: child,
                            idom: Some(block),
                        });
                    }
                    // Reverse what we just pushed so we elaborate in
                    // original block order. (The domtree iter is a
                    // single-ended iter over a singly-linked list so
                    // we can't `.rev()` above.)
                    self.block_stack[block_stack_end..].reverse();
                }
                BlockStackEntry::Pop => {
                    self.id_to_value.decrement_depth();
                    if let Some(innermost_loop) = self.loop_stack.last() {
                        if innermost_loop.scope_depth as usize == self.id_to_value.depth() {
                            self.loop_stack.pop();
                        }
                    }
                }
            }
        }
    }

    fn clear_func_body(&mut self) {
        // Clear all instructions and args/results from the DFG. We
        // rebuild them entirely during elaboration. (TODO: reuse the
        // existing inst for the *first* copy of a given node.)
        self.func.dfg.clear_insts();
        // Clear the instructions in every block, but leave the list
        // of blocks and their layout unmodified.
        self.func.layout.clear_insts();
        self.func.srclocs.clear();
    }

    pub(crate) fn elaborate<'b, PF: Fn(Block) -> &'b [(Id, Type)], SEF: Fn(Block) -> &'b [Id]>(
        &mut self,
        block_params_fn: PF,
        block_side_effects_fn: SEF,
    ) {
        let domtree = DomTreeWithChildren::new(self.func, self.domtree);
        self.stats.elaborate_func += 1;
        self.stats.elaborate_func_pre_insts += self.func.dfg.num_insts() as u64;
        self.clear_func_body();
        self.compute_best_nodes();
        self.elaborate_domtree(&block_params_fn, &block_side_effects_fn, &domtree);
        self.stats.elaborate_func_post_insts += self.func.dfg.num_insts() as u64;
    }
}
