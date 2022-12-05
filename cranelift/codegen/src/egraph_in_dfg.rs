//! Support for egraphs represented in the DataFlowGraph.

use crate::ctxhash::{CtxEq, CtxHash, CtxHashMap};
use crate::cursor::{Cursor, CursorPosition, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::egraph::domtree::DomTreeWithChildren;
use crate::egraph::elaborate::Elaborator;
use crate::egraph::Stats;
use crate::fx::FxHashSet;
use crate::inst_predicates::is_pure_for_egraph;
use crate::ir::{DataFlowGraph, Function, Inst, InstructionData, Type, Value, ValueListPool};
use crate::loop_analysis::LoopAnalysis;
use crate::opts::generated_code::ContextIter;
use crate::opts::IsleContext;
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
    /// elaboration.
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
}

/// Context passed through node insertion and optimization.
pub(crate) struct OptimizeCtx<'a> {
    // Borrowed from EgraphPass:
    pub(crate) func: &'a mut Function,
    pub(crate) value_to_opt_value: &'a mut SecondaryMap<Value, Value>,
    pub(crate) gvn_map: &'a mut CtxHashMap<(Type, InstructionData), Value>,
    pub(crate) eclasses: &'a mut UnionFind<Value>,
    pub(crate) remat_values: &'a mut FxHashSet<Value>,
    // Held locally during optimization of one node (recursively):
    pub(crate) rewrite_depth: usize,
    pub(crate) subsume_values: FxHashSet<Value>,
}

/// For passing to `insert_pure_enode`. Sometimes the enode already
/// exists as an Inst (from the original CLIF), and sometimes we're in
/// the middle of creating it and want to avoid inserting it if
/// possible until we know we need it.
pub(crate) enum NewOrExistingInst {
    New(InstructionData, Type),
    Existing(Inst),
}

impl NewOrExistingInst {
    fn get_inst_key<'a>(&'a self, dfg: &'a DataFlowGraph) -> (Type, InstructionData) {
        match self {
            NewOrExistingInst::New(data, ty) => (*ty, data.clone()),
            NewOrExistingInst::Existing(inst) => {
                let ty = dfg.ctrl_typevar(*inst);
                (ty, dfg[*inst].clone())
            }
        }
    }
}

impl<'a> OptimizeCtx<'a> {
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
    pub(crate) fn insert_pure_enode(&mut self, inst: NewOrExistingInst) -> Value {
        // Create the external context for looking up and updating the
        // GVN map. This is necessary so that instructions themselves
        // do not have to carry all the references or data for a full
        // `Eq` or `Hash` impl.
        let gvn_context = GVNContext {
            union_find: self.eclasses,
            value_lists: &self.func.dfg.value_lists,
        };

        // Does this instruction already exist? If so, add entries to
        // the value-map to rewrite uses of its results to the results
        // of the original (existing) instruction. If not, optimize
        // the new instruction.
        if let Some(&orig_result) = self
            .gvn_map
            .get(&inst.get_inst_key(&self.func.dfg), &gvn_context)
        {
            if let NewOrExistingInst::Existing(inst) = inst {
                debug_assert_eq!(self.func.dfg.inst_results(inst).len(), 1);
                let result = self.func.dfg.inst_results(inst)[0];
                self.value_to_opt_value[result] = orig_result;
                self.eclasses.union(result, orig_result);
                result
            } else {
                orig_result
            }
        } else {
            // Now actually insert the InstructionData and attach
            // result value (exactly one).
            let (inst, result, ty) = match inst {
                NewOrExistingInst::New(data, typevar) => {
                    let inst = self.func.dfg.make_inst(data);
                    // TODO: reuse return value?
                    self.func.dfg.make_inst_results(inst, typevar);
                    let result = self.func.dfg.inst_results(inst)[0];
                    // Add to eclass unionfind.
                    self.eclasses.add(result);
                    // New inst. We need to do the analysis of its result.
                    (inst, result, typevar)
                }
                NewOrExistingInst::Existing(inst) => {
                    let result = self.func.dfg.inst_results(inst)[0];
                    let ty = self.func.dfg.ctrl_typevar(inst);
                    (inst, result, ty)
                }
            };

            let opt_value = self.optimize_pure_enode(inst);
            let gvn_context = GVNContext {
                union_find: self.eclasses,
                value_lists: &self.func.dfg.value_lists,
            };
            self.gvn_map
                .insert((ty, self.func.dfg[inst].clone()), opt_value, &gvn_context);
            self.value_to_opt_value[result] = opt_value;
            opt_value
        }
    }

    /// Optimizes an enode by applying any matching mid-end rewrite
    /// rules (or store-to-load forwarding, which is a special case),
    /// unioning together all possible optimized (or rewritten) forms
    /// of this expression into an eclass and returning the `Value`
    /// that represents that eclass.
    fn optimize_pure_enode(&mut self, inst: Inst) -> Value {
        // TODO: integrate store-to-load forwarding.

        // A pure node always has exactly one result.
        let orig_value = self.func.dfg.inst_results(inst)[0];

        let mut isle_ctx = IsleContext { ctx: self };

        // Limit rewrite depth. When we apply optimization rules, they
        // may create new nodes (values) and those are, recursively,
        // optimized eagerly as soon as they are created. So we may
        // have more than one ISLE invocation on the stack. (This is
        // necessary so that as the toplevel builds the
        // right-hand-side expression bottom-up, it uses the "latest"
        // optimized values for all the constituent parts.) To avoid
        // infinite or problematic recursion, we bound the rewrite
        // depth to a small constant here.
        const REWRITE_LIMIT: usize = 5;
        if isle_ctx.ctx.rewrite_depth > REWRITE_LIMIT {
            return orig_value;
        }
        isle_ctx.ctx.rewrite_depth += 1;

        // Invoke the ISLE toplevel constructor, getting all new
        // values produced as equivalents to this value.
        trace!("Calling into ISLE with original value {}", orig_value);
        let optimized_values =
            crate::opts::generated_code::constructor_simplify(&mut isle_ctx, orig_value);

        // Create a union of all new values with the original (or
        // maybe just one new value marked as "subsuming" the
        // original, if present.)
        let mut union_value = orig_value;
        if let Some(mut optimized_values) = optimized_values {
            while let Some(optimized_value) = optimized_values.next(&mut isle_ctx) {
                trace!(
                    "Returned from ISLE for {}, got {:?}",
                    orig_value,
                    optimized_value
                );
                if isle_ctx.ctx.subsume_values.contains(&optimized_value) {
                    // Merge in the unionfind so canonicalization
                    // still works, but take *only* the subsuming
                    // value, and break now.
                    isle_ctx.ctx.eclasses.union(optimized_value, union_value);
                    union_value = optimized_value;
                    break;
                }

                let old_union_value = union_value;
                union_value = isle_ctx
                    .ctx
                    .func
                    .dfg
                    .union(old_union_value, optimized_value);
                isle_ctx.ctx.eclasses.add(union_value);
                isle_ctx
                    .ctx
                    .eclasses
                    .union(old_union_value, optimized_value);
                isle_ctx.ctx.eclasses.union(old_union_value, union_value);
            }
        }

        isle_ctx.ctx.rewrite_depth -= 1;

        union_value
    }
}

impl<'a> EgraphPass<'a> {
    /// Create a new EgraphPass.
    pub fn new(
        func: &'a mut Function,
        domtree: &'a DominatorTree,
        loop_analysis: &'a LoopAnalysis,
    ) -> Self {
        let num_values = func.dfg.num_values();
        let domtree_children = DomTreeWithChildren::new(func, domtree);
        Self {
            func,
            domtree,
            domtree_children,
            loop_analysis,
            stats: Stats::default(),
            eclasses: UnionFind::with_capacity(num_values),
            remat_values: FxHashSet::default(),
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
        let mut gvn_map: CtxHashMap<(Type, InstructionData), Value> =
            CtxHashMap::with_capacity(cursor.func.dfg.num_values());

        // In domtree preorder, visit blocks. (TODO: factor out an
        // iterator from this and elaborator.)
        let root = self.domtree_children.root();
        let mut block_stack = vec![root];
        while let Some(block) = block_stack.pop() {
            // We popped this block; push children
            // immediately, then process this block.
            for child in self.domtree_children.children(block) {
                block_stack.push(child);
            }

            trace!("Processing block {}", block);
            cursor.set_position(CursorPosition::Before(block));

            for &param in cursor.func.dfg.block_params(block) {
                trace!("creating initial singleton eclass for blockparam {}", param);
                self.eclasses.add(param);
                value_to_opt_value[param] = param;
            }
            while let Some(inst) = cursor.next_inst() {
                trace!("Processing inst {}", inst);

                // While we're passing over all insts, create initial
                // singleton eclasses for all result and blockparam
                // values.  Also do initial analysis of all inst
                // results.
                for &result in cursor.func.dfg.inst_results(inst) {
                    trace!("creating initial singleton eclass for {}", result);
                    self.eclasses.add(result);
                }

                // Rewrite args of *all* instructions using the
                // value-to-opt-value map.
                cursor.func.dfg.resolve_aliases_in_arguments(inst);
                for arg in cursor.func.dfg.inst_args_mut(inst) {
                    let new_value = value_to_opt_value[*arg];
                    trace!("rewriting arg {} of inst {} to {}", arg, inst, new_value);
                    debug_assert_ne!(new_value, Value::reserved_value());
                    *arg = new_value;
                }

                if is_pure_for_egraph(cursor.func, inst) {
                    // Insert into GVN map and optimize any new nodes
                    // inserted (recursively performing this work for
                    // any nodes the optimization rules produce).
                    let mut ctx = OptimizeCtx {
                        func: cursor.func,
                        value_to_opt_value: &mut value_to_opt_value,
                        gvn_map: &mut gvn_map,
                        eclasses: &mut self.eclasses,
                        rewrite_depth: 0,
                        subsume_values: FxHashSet::default(),
                        remat_values: &mut self.remat_values,
                    };
                    let inst = NewOrExistingInst::Existing(inst);
                    ctx.insert_pure_enode(inst);
                    // We've now rewritten all uses, or will when we
                    // see them, and the instruction exists as a pure
                    // enode in the eclass, so we can remove it.
                    cursor.remove_inst_and_step_back();
                } else {
                    // Not pure, but may still be a store: add it to
                    // the store-map if so so store-to-load forwarding
                    // can work properly.
                    //todo!("store-map update");

                    // Set all results to identity-map to themselves
                    // in the value-to-opt-value map.
                    for &result in cursor.func.dfg.inst_results(inst) {
                        value_to_opt_value[result] = result;
                        self.eclasses.add(result);
                    }
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
            &self.domtree_children,
            self.loop_analysis,
            &mut self.remat_values,
            &mut self.eclasses,
            &mut self.stats,
        );
        elaborator.elaborate();
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

impl<'a> CtxEq<(Type, InstructionData), (Type, InstructionData)> for GVNContext<'a> {
    fn ctx_eq(
        &self,
        (a_ty, a_inst): &(Type, InstructionData),
        (b_ty, b_inst): &(Type, InstructionData),
    ) -> bool {
        a_ty == b_ty
            && a_inst.eq(b_inst, self.value_lists, |value| {
                self.union_find.find(value)
            })
    }
}

impl<'a> CtxHash<(Type, InstructionData)> for GVNContext<'a> {
    fn ctx_hash(&self, (ty, inst): &(Type, InstructionData)) -> u64 {
        let mut state = crate::fx::FxHasher::default();
        std::hash::Hash::hash(&ty, &mut state);
        inst.hash(&mut state, self.value_lists, |value| {
            self.union_find.find(value)
        });
        state.finish()
    }
}
