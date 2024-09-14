//! Full congruence mode for the (a)egraph.
//!
//! TODO: integrate alias analysis.

use super::*;
use crate::trace;
use cranelift_entity::SecondaryMap;
use rustc_hash::FxHashMap;
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
}

impl<'a, 'b> FullCongruence<'a, 'b> {
    /// Build a new full-congruence runner.
    pub fn new(egraph: &'b mut EgraphPass<'a>) -> Self {
        Self {
            egraph,
            dedup: FxHashMap::default(),
            eclasses: SecondaryMap::default(),
            eclass_list: vec![],
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
        while let Some(_block) = pos.next_block() {
            while let Some(inst) = pos.next_inst() {
                if is_pure_for_egraph(pos.func, inst) {
                    log::trace!("pure inst removed from skeleton: {}", inst);
                    pos.remove_inst_and_step_back();
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
                    log::trace!("enode {}", enode);
                    // Get the InstructionData and Type; remove from
                    // dedup map; re-canonicalize args; re-insert,
                    // pointing to canonical eclass.
                    let ValueDef::Result(inst, ..) = self.egraph.func.dfg.value_def(enode) else {
                        log::trace!(" -> non-pure, skipping");
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
            }
            log::trace!("new eclass list: {:?}", new_eclass_list);
            self.eclass_list = new_eclass_list;

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
        // Step 1: collect the "available block" (highest point in the
        // domtree) at which an eclass is available. We seed this with
        // locations from the skeleton in this step and complete it
        // for pure nodes with a fixpoint below.
        let mut available_block_eclass: SecondaryMap<Value, Block> =
            SecondaryMap::with_default(Block::reserved_value());
        let mut available_block_enode: SecondaryMap<Value, Block> =
            SecondaryMap::with_default(Block::reserved_value());

        for (value, def) in self.egraph.func.dfg.values_and_defs() {
            if self.egraph.func.dfg.resolve_aliases(value) != value {
                continue;
            }
            match def {
                ValueDef::Result(inst, ..) => {
                    // Skeleton insts remain in the layout; set
                    // their available block.
                    if let Some(block) = self.egraph.func.layout.inst_block(inst) {
                        available_block_eclass[value] = block;
                        available_block_enode[value] = block;
                        log::trace!(
                            "remove_cycles: non-pure inst {}: result {} is available in {}",
                            inst,
                            value,
                            block
                        );
                    }
                }
                ValueDef::Param(block, ..) => {
                    available_block_eclass[value] = block;
                    available_block_enode[value] = block;
                    log::trace!(
                        "remove_cycles: blockparam {} is available in {}",
                        value,
                        block
                    );
                }
                ValueDef::Union(..) => {}
            }
        }

        let entry = self.egraph.func.layout.entry_block().unwrap();
        let meet_up_domtree = |b1: Block, b2: Block| {
            if b2.is_reserved_value() {
                b1
            } else if b1.is_reserved_value() {
                b2
            } else if b1 == b2 {
                b1
            } else if self.egraph.domtree.dominates(b1, b2) {
                b1
            } else {
                assert!(self.egraph.domtree.dominates(b2, b1));
                b2
            }
        };
        let meet_down_domtree = |b1: Block, b2: Block| {
            if b2.is_reserved_value() || b1.is_reserved_value() {
                Block::reserved_value()
            } else if b1 == b2 {
                b1
            } else if self.egraph.domtree.dominates(b1, b2) {
                b2
            } else {
                assert!(self.egraph.domtree.dominates(b2, b1));
                b1
            }
        };

        let mut avail_changed = true;
        while avail_changed {
            avail_changed = false;

            for &eclass in &self.eclass_list {
                // Iterate over all enodes in the class, finding
                // the available-block for each (or not yet
                // available). Then find the highest-in-domtree
                // available block.
                let mut best_avail = Block::reserved_value();
                for &member in &self.eclasses[eclass] {
                    let ValueDef::Result(inst, ..) = self.egraph.func.dfg.value_def(member) else {
                        continue;
                    };
                    let node_args = self.egraph.func.dfg.inst_args(inst);
                    log::trace!("eclass {} current best_avail {}", eclass, best_avail);
                    let best_node = node_args
                        .iter()
                        .map(|&arg| self.egraph.func.dfg.resolve_aliases(arg))
                        .map(|arg| self.egraph.eclasses.find_and_update(arg))
                        .map(|arg| available_block_eclass[arg])
                        .fold(entry, meet_down_domtree);
                    available_block_enode[member] = best_node;
                    best_avail = meet_up_domtree(best_avail, best_node);
                    log::trace!(
                        "eclass {} enode {} best_node {} -> eclass best_avail {}",
                        eclass,
                        member,
                        best_node,
                        best_avail
                    );
                }
                log::trace!("eclass {}: best avail {}", eclass, best_avail);

                if best_avail != available_block_eclass[eclass] {
                    avail_changed = true;
                    available_block_eclass[eclass] = best_avail;
                }
            }
        }

        // Step 2: trim eclasses according to dominance to preserve
        // acyclicity:
        // - For each eclass, iterate over members and their
        //   "available blocks";
        // - Make note of nodes that must be removed;
        // - Filter the enode list.
        for &eclass in &self.eclass_list {
            let mut enodes = std::mem::take(&mut self.eclasses[eclass]);
            enodes.retain(|&mut enode| {
                available_block_enode[enode]
                    == available_block_eclass[self.egraph.eclasses.find_and_update(enode)]
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

        let rewrite_inst = |func: &mut Function, inst: Inst| {
            let mut op = func.dfg.insts[inst];
            for i in 0..op.arguments(&mut func.dfg.value_lists).len() {
                let arg = op.arguments(&mut func.dfg.value_lists)[i];
                let arg = func.dfg.resolve_aliases(arg);
                let arg = self.egraph.eclasses.find(arg);
                let value = if latest[arg].is_reserved_value() {
                    arg
                } else {
                    latest[arg]
                };
                log::trace!(" -> orig arg {} to latest {}", arg, value);
                op.arguments_mut(&mut func.dfg.value_lists)[i] = value;
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
