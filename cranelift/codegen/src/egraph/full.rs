//! Full congruence mode for the (a)egraph.
//!
//! TODO: integrate alias analysis.

use super::*;
use crate::trace;
use cranelift_entity::SecondaryMap;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::vec::Vec;

/// State for a full-congruence optimization pass over the egraph.
pub struct FullCongruence<'a, 'b> {
    egraph: &'b mut EgraphPass<'a>,
    /// Map from instruction (with canonicalized args) to
    /// canonicalized eclass.
    dedup: FxHashMap<(Type, InstructionData), Value>,
    /// Map from canonicalized eclass to list of all enodes.
    eclasses: SecondaryMap<Value, SmallVec<[Value; 8]>>,
    /// List of canonicalized eclass IDs.
    eclass_list: Vec<Value>,
    /// List of enode IDs that are non-pure.
    non_pure: FxHashSet<Value>,
}

impl<'a, 'b> FullCongruence<'a, 'b> {
    /// Build a new full-congruence runner.
    pub fn new(egraph: &'b mut EgraphPass<'a>) -> Self {
        Self {
            egraph,
            dedup: FxHashMap::default(),
            eclasses: SecondaryMap::default(),
            eclass_list: vec![],
            non_pure: FxHashSet::default(),
        }
    }

    /// Run the overall optimization fixpoint loop.
    pub fn run(&mut self) {
        self.remove_pure_ops_from_skeleton();
        self.build_eclasses();
        let limit_nodes =
            self.dedup.len() * usize::from(self.egraph.flags.full_egraph_max_inflation());
        for _ in 0..self.egraph.flags.full_egraph_max_passes() {
            self.alias_analysis();
            self.batch_rewrite();
            if !self.batch_congruence() {
                break;
            }
            if self.dedup.len() > limit_nodes {
                break;
            }
        }
        self.remove_cycles();
        self.latest_args();
        trace!("after full egraph:\n{:?}", self.egraph.func);
    }

    /// Remove pure ops from the CFG, leaving the skeleton.
    fn remove_pure_ops_from_skeleton(&mut self) {
        let mut pos = FuncCursor::new(self.egraph.func);
        while let Some(block) = pos.next_block() {
            for &param in pos.func.dfg.block_params(block) {
                self.non_pure.insert(param);
            }
            while let Some(inst) = pos.next_inst() {
                if is_pure_for_egraph(pos.func, inst) {
                    log::trace!("pure inst removed from skeleton: {}", inst);
                    pos.remove_inst_and_step_back();
                } else {
                    for &result in pos.func.dfg.inst_results(inst) {
                        self.non_pure.insert(result);
                    }
                }
            }
        }
    }

    /// Build data structures that represent eclasses, to allow batch
    /// congruence and cycle removal to work.
    fn build_eclasses(&mut self) {
        // Collect:
        // - The full set of nodes per canonical node (eclass);
        // - The full list of eclasses.
        for (value, def) in self.egraph.func.dfg.values_and_defs() {
            if self.egraph.func.dfg.resolve_aliases(value) != value {
                continue;
            }
            self.egraph.eclasses.add(value);
            self.eclasses[value].push(value);
            self.eclass_list.push(value);
            match def {
                ValueDef::Result(inst, ..) => {
                    if is_pure_for_egraph(self.egraph.func, inst) {
                        log::trace!(
                            "pure op result {} of inst {} becomes singleton eclass",
                            value,
                            inst
                        );
                        let ty = self.egraph.func.dfg.ctrl_typevar(inst);
                        let old = self
                            .dedup
                            .insert((ty, self.egraph.func.dfg.insts[inst]), value);
                        if let Some(old) = old {
                            self.egraph.eclasses.union(old, value);
                        }
                    } else {
                        log::trace!(
                            "non-pure result {} of inst {} is terminal singleton eclass",
                            value,
                            inst
                        );
                    }
                }
                ValueDef::Param(..) => {
                    log::trace!("blockparam {} is terminal singleton eclass", value);
                }
                ValueDef::Union(..) => {}
            }
        }
    }

    /// Apply alias analysis across the skeleton.
    fn alias_analysis(&mut self) {
        let mut cursor = FuncCursor::new(self.egraph.func);
        while let Some(block) = cursor.next_block() {
            let mut state = self.egraph.alias_analysis.block_starting_state(block);
            self.egraph.alias_analysis.start_block();
            while let Some(inst) = cursor.next_inst() {
                log::trace!(
                    "running alias analysis at skeleton inst {}: state {:?}",
                    inst,
                    state
                );

                if let Some(new_value) =
                    self.egraph
                        .alias_analysis
                        .process_inst(cursor.func, &mut state, inst)
                {
                    let result = cursor.func.dfg.first_result(inst);
                    log::trace!(
                        "replacing load {} result {} with {}",
                        inst,
                        result,
                        new_value
                    );
                    cursor.func.dfg.clear_results(inst);
                    cursor.func.dfg.change_to_alias(result, new_value);
                    self.egraph.eclasses.union(result, new_value);
                    cursor.remove_inst_and_step_back();
                }
            }
        }
    }

    /// Apply rewrite rules in bulk, without doing nested rewriting.
    ///
    /// To be run after the egraph has been constructed, and before it
    /// is elaborated.
    fn batch_rewrite(&mut self) {
        // For every value node, invoke the mid-end rewrite rule entry
        // point with a context that creates new nodes but does not
        // recursively rewrite (add a flag for this on OptimizeCtx).
        let mut optimized_values: SmallVec<[Value; 5]> = SmallVec::new();
        for i in 0..self.eclass_list.len() {
            let eclass = self.eclass_list[i];
            for j in 0..self.eclasses[eclass].len() {
                let enode = self.eclasses[eclass][j];
                if self.non_pure.contains(&enode) {
                    continue;
                }
                log::trace!("batch rewrite: eclass {} enode {}", eclass, enode);
                crate::opts::generated_code::constructor_simplify(
                    &mut IsleContext { ctx: self },
                    enode,
                    &mut optimized_values,
                );
                for result in optimized_values.drain(..) {
                    if result == enode {
                        continue;
                    }
                    log::trace!(
                        "batch rewrite of eclass {} enode {} got new enode {}",
                        eclass,
                        enode,
                        result
                    );
                    self.egraph.eclasses.add(result);
                    self.egraph.eclasses.union(eclass, result);
                }
            }
        }

        // For every skeleton node, canonicalize args.
        let mut cursor = FuncCursor::new(self.egraph.func);
        while let Some(_block) = cursor.next_block() {
            while let Some(inst) = cursor.next_inst() {
                let pos = cursor.position();
                let mut op = self.egraph.func.dfg.insts[inst];
                self.canonicalize_args(&mut op);
                self.egraph.func.dfg.insts[inst] = op;
                cursor = FuncCursor::new(self.egraph.func).at_position(pos);
            }
        }
    }

    /// Perform a full-congruence pass: find all eclass members,
    /// reduce that set by removing nodes that have lower "available
    /// blocks" in the domtree (to preserve acyclicity), then update
    /// args to the "latest" version of each eclass.
    ///
    /// To be run after the egraph has been constructed, and before it
    /// is elaborated.
    ///
    /// Returns `true` if any changes were made.
    fn batch_congruence(&mut self) -> bool {
        // In a fixpoint loop, edit args and re-intern nodes:
        // - For each node, canonicalize args according to union-find.
        // - Look up in GVN map; if maps to another canonical node
        //   (eclass), union them.
        let mut any_changed = false;
        let mut changed = true;
        while changed {
            changed = false;
            log::trace!("batch congruence: new iteration");

            for i in 0..self.eclass_list.len() {
                let eclass = self.eclass_list[i];
                let canonical_eclass = self.egraph.eclasses.find_and_update(eclass);
                log::trace!(
                    "eclass {} canonical {}: {:?}",
                    eclass,
                    canonical_eclass,
                    self.eclasses[canonical_eclass]
                );
                if eclass != canonical_eclass {
                    changed = true;
                }

                for j in 0..self.eclasses[eclass].len() {
                    let enode = self.eclasses[eclass][j];
                    if self.non_pure.contains(&enode) {
                        continue;
                    }
                    log::trace!("enode {}", enode);
                    // Get the InstructionData and Type; remove from
                    // dedup map; re-canonicalize args; re-insert,
                    // pointing to canonical eclass.
                    let ValueDef::Result(inst, ..) = self.egraph.func.dfg.value_def(enode) else {
                        continue;
                    };
                    let ty = self.egraph.func.dfg.ctrl_typevar(inst);
                    let mut op = self.egraph.func.dfg.insts[inst];
                    self.dedup.remove(&(ty, op));
                    log::trace!("enode {} has ty {:?} op {:?}", enode, ty, op);
                    self.canonicalize_args(&mut op);
                    log::trace!(" -> canonicalized: {:?}", op);
                    match self.dedup.entry((ty, op)) {
                        Entry::Vacant(v) => {
                            log::trace!("inserting {} back into dedup map", canonical_eclass);
                            v.insert(canonical_eclass);
                        }
                        Entry::Occupied(o) => {
                            log::trace!(
                                "already in dedup map as {}; unioning with {}",
                                o.get(),
                                canonical_eclass
                            );
                            self.egraph.eclasses.union(*o.get(), canonical_eclass);
                            changed = true;
                        }
                    }
                    self.egraph.func.dfg.insts[inst] = op;
                }
            }

            // Now that we've performed unions in the union-find,
            // iterate over all eclasses in the `eclass_list`; if
            // merged into another eclass, move the enodes over to the
            // new canonical eclass's node list, and create a
            // union-node and upate `latest`.
            let mut new_eclass_list = vec![];
            for i in 0..self.eclass_list.len() {
                let eclass = self.eclass_list[i];
                let canonical_eclass = self.egraph.eclasses.find_and_update(eclass);
                if canonical_eclass != eclass {
                    log::trace!("merging eclasses {} and {}", eclass, canonical_eclass);
                    let members = std::mem::take(&mut self.eclasses[eclass]);
                    self.eclasses[canonical_eclass].extend(members.into_iter());
                    log::trace!(" -> enode list: {:?}", self.eclasses[canonical_eclass]);
                    changed = true;
                } else {
                    new_eclass_list.push(eclass);
                }

                self.eclasses[eclass].sort_unstable();
                self.eclasses[eclass].dedup();
            }
            log::trace!("new eclass list: {:?}", new_eclass_list);
            self.eclass_list = new_eclass_list;
            self.eclass_list.sort_unstable();
            self.eclass_list.dedup();

            if changed {
                any_changed = true;
            }
        }

        any_changed
    }

    /// Remove cycles, after doing a fixpoint of full batch-congruence
    /// and rewriting. Unnecessary if the egraph is used only in
    /// single-pass eager rewrite mode.
    fn remove_cycles(&mut self) {
        // Collect an "available generation" for every eclass and
        // enode. The available generation for an eclass is the
        // minimum of that of its enodes; for an enode, the maximum of
        // that of its arguments. Skeleton enodes and eclasses have an
        // available generation of 0.
        //
        // We compute this in a fixpoint. It is guaranteed to
        // terminate, even in the presence of cycles, as long as there
        // is *some* non-cyclic path from roots to compute any given
        // value. The graph starts this way (from the input CFG), and
        // we never remove nodes during rewriting/congruence, so this
        // must be the case.
        let mut gen_eclass: SecondaryMap<Value, u32> = SecondaryMap::with_default(u32::MAX);
        let mut gen_enode: SecondaryMap<Value, u32> = SecondaryMap::with_default(u32::MAX);

        let mut cursor = FuncCursor::new(self.egraph.func);
        while let Some(block) = cursor.next_block() {
            for &param in cursor.func.dfg.block_params(block) {
                let eclass = self.egraph.eclasses.find(param);
                gen_eclass[eclass] = 0;
                gen_enode[param] = 0;
                log::trace!("eclass {} blockparam {} avail at gen 0", eclass, param);
            }
            while let Some(inst) = cursor.next_inst() {
                for &result in cursor.func.dfg.inst_results(inst) {
                    let eclass = self.egraph.eclasses.find(result);
                    gen_eclass[eclass] = 0;
                    gen_enode[result] = 0;
                    log::trace!(
                        "eclass {} result {} in skeleton avail at gen 0",
                        eclass,
                        result
                    );
                }
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for &eclass in &self.eclass_list {
                let mut eclass_avail = u32::MAX;
                for &enode in &self.eclasses[eclass] {
                    if let ValueDef::Result(inst, ..) = self.egraph.func.dfg.value_def(enode) {
                        let gen = self
                            .egraph
                            .func
                            .dfg
                            .inst_args(inst)
                            .iter()
                            .map(|&arg| {
                                let arg = self.egraph.func.dfg.resolve_aliases(arg);
                                let eclass = self.egraph.eclasses.find(arg);
                                gen_eclass[eclass]
                            })
                            .fold(0, std::cmp::max)
                            .saturating_add(1);
                        log::trace!(
                            "enode {} (eclass {}): available at generation {}",
                            enode,
                            eclass,
                            gen
                        );
                        let gen = std::cmp::min(gen, gen_enode[enode]);
                        gen_enode[enode] = gen;
                        eclass_avail = std::cmp::min(eclass_avail, gen);
                    }
                }
                if eclass_avail < gen_eclass[eclass] {
                    gen_eclass[eclass] = eclass_avail;
                    changed = true;
                    log::trace!("eclass {}: available at {}", eclass, eclass_avail);
                }
            }
        }

        // Now filter each eclass to enodes with available generation
        // at the minimum for the eclass.
        for &eclass in &self.eclass_list {
            let mut enodes = std::mem::take(&mut self.eclasses[eclass]);
            enodes.retain(|&mut enode| {
                log::trace!(
                    "testing eclass {} enode {}: enode gen {} eclass gen {}",
                    eclass,
                    enode,
                    gen_enode[enode],
                    gen_eclass[eclass]
                );
                gen_enode[enode] == gen_eclass[eclass]
            });
            log::trace!("eclass {}: trimmed to {:?}", eclass, enodes);
            self.eclasses[eclass] = enodes;
        }
    }

    /// Final step: create union-node spines, and update arguments of
    /// all enodes to use latest IDs in each eclass.
    fn latest_args(&mut self) {
        let mut latest: SecondaryMap<Value, Value> =
            SecondaryMap::with_default(Value::reserved_value());
        for &eclass in &self.eclass_list {
            let eclass = self.egraph.eclasses.find_and_update(eclass);
            log::trace!(
                "computing latest for eclass {} with enodes {:?}",
                eclass,
                self.eclasses[eclass]
            );
            latest[eclass] = self.eclasses[eclass][0];
            for &enode in self.eclasses[eclass].iter().skip(1) {
                let union = self.egraph.func.dfg.union(latest[eclass], enode);
                self.egraph.eclasses.add(union);
                self.egraph.eclasses.union(latest[eclass], union);
                self.egraph.eclasses.union(enode, union);
                self.egraph.func.dfg.merge_facts(latest[eclass], enode);
                self.egraph.func.dfg.merge_facts(enode, union);
                latest[eclass] = union;
            }
            log::trace!("eclass {}: latest is {}", eclass, latest[eclass]);
        }

        let rewrite_arg = |func: &mut Function, arg: Value| -> Value {
            let arg = func.dfg.resolve_aliases(arg);
            let arg = self.egraph.eclasses.find(arg);
            let value = if latest[arg].is_reserved_value() {
                arg
            } else {
                latest[arg]
            };
            log::trace!(" -> orig arg {} to latest {}", arg, value);
            value
        };

        let rewrite_inst = |func: &mut Function, inst: Inst| {
            let mut op = func.dfg.insts[inst];
            for i in 0..op.arguments(&mut func.dfg.value_lists).len() {
                let arg = op.arguments(&mut func.dfg.value_lists)[i];
                let value = rewrite_arg(func, arg);
                op.arguments_mut(&mut func.dfg.value_lists)[i] = value;
            }
            for i in 0..op.branch_destination(&func.dfg.jump_tables).len() {
                let mut call = op.branch_destination(&func.dfg.jump_tables)[i];
                for j in 0..call.args_slice(&func.dfg.value_lists).len() {
                    let arg = call.args_slice(&func.dfg.value_lists)[j];
                    let value = rewrite_arg(func, arg);
                    call.args_slice_mut(&mut func.dfg.value_lists)[j] = value;
                }
                op.branch_destination_mut(&mut func.dfg.jump_tables)[i] = call;
            }
            func.dfg.insts[inst] = op;
        };

        for &eclass in &self.eclass_list {
            log::trace!("eclass {}: nodes: {:?}", eclass, self.eclasses[eclass]);
            for &enode in &self.eclasses[eclass] {
                let ValueDef::Result(inst, _) = self.egraph.func.dfg.value_def(enode) else {
                    continue;
                };
                log::trace!(
                    "setting latest values in eclass {} enode {} inst {}",
                    eclass,
                    enode,
                    inst
                );
                rewrite_inst(self.egraph.func, inst);
            }
        }

        let mut cursor = FuncCursor::new(self.egraph.func);
        while let Some(_block) = cursor.next_block() {
            while let Some(inst) = cursor.next_inst() {
                log::trace!("setting latest values in skeleton inst {}", inst);
                rewrite_inst(cursor.func, inst);
            }
        }
    }

    fn canonicalize_args(&mut self, op: &mut InstructionData) {
        // Canonicalize arguments to eclass IDs.
        for i in 0..op.arguments(&self.egraph.func.dfg.value_lists).len() {
            let arg = op.arguments(&self.egraph.func.dfg.value_lists)[i];
            let arg = self.egraph.func.dfg.resolve_aliases(arg);
            op.arguments_mut(&mut self.egraph.func.dfg.value_lists)[i] =
                self.egraph.eclasses.find_and_update(arg);
        }
    }
}

impl<'a, 'b> crate::opts::EgraphImpl for FullCongruence<'a, 'b> {
    fn insert_node(&mut self, mut op: InstructionData, ty: Type) -> Value {
        self.canonicalize_args(&mut op);
        // Intern in the GVN map.
        match self.dedup.entry((ty, op)) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                // Create the node.
                let inst = self.egraph.func.dfg.make_inst(op);
                self.egraph.func.dfg.make_inst_results(inst, ty);
                let result = self.egraph.func.dfg.first_result(inst);
                // New singleton eclass.
                self.egraph.eclasses.add(result);
                self.eclasses[result].push(result);
                self.eclass_list.push(result);
                log::trace!(
                    "insert_node: created {} with ty {:?} op {:?}",
                    result,
                    ty,
                    op
                );
                *v.insert(result)
            }
        }
    }
    fn func(&mut self) -> &mut Function {
        &mut self.egraph.func
    }
    fn remat(&mut self, value: Value) -> Value {
        self.egraph.remat_values.insert(value);
        self.egraph.stats.remat += 1;
        value
    }
    fn subsume(&mut self, value: Value) -> Value {
        value
    }
    fn eclass_members_direct() -> bool {
        true
    }
    fn eclass_members(&self, value: Value) -> SmallVec<[Value; 8]> {
        let mut ret = self.eclasses[value].clone();
        ret.retain(|e| {
            log::trace!("eclass_member: {}", e);
            matches!(self.egraph.func.dfg.value_def(*e), ValueDef::Result(..))
        });
        ret
    }
}
