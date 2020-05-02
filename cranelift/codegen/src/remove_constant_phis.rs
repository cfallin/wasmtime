//! A Constant-Phi-Node removal pass.

#![allow(unused_imports)]

use log::info;

use crate::cursor::{Cursor, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::entity::EntityList;
use crate::entity::EntityRef;
use crate::inst_predicates::{any_inst_results_used, has_side_effect};
use crate::ir::Function;
use crate::timing;

use crate::ir;
use crate::ir::types;
use crate::ir::{Block, Inst, Opcode, Type, Value};
use crate::ir::{DataFlowGraph, InstructionData};

use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::vec::Vec;

#[derive(Clone, Copy, Debug)]
enum LatticePoint {
    None,
    One(Value),
    Many,
}

impl LatticePoint {
    fn join(&self, other: &LatticePoint) -> LatticePoint {
        match (self, other) {
            // Joining with None has no effect
            (LatticePoint::None, p2) => *p2,
            (p1, LatticePoint::None) => *p1,
            // Joining with Many produces Many
            (LatticePoint::Many, _p2) => LatticePoint::Many,
            (_p1, LatticePoint::Many) => LatticePoint::Many,
            // The only interesting case
            (LatticePoint::One(v1), LatticePoint::One(v2)) => {
                if v1 == v2 {
                    LatticePoint::One(*v1)
                } else {
                    LatticePoint::Many
                }
            }
        }
    }
    fn equals(&self, other: &LatticePoint) -> bool {
        match (self, other) {
            (LatticePoint::None, LatticePoint::None) => true,
            (LatticePoint::Many, LatticePoint::Many) => true,
            (LatticePoint::One(v1), LatticePoint::One(v2)) => v1 == v2,
            (_, _) => false,
        }
    }
    fn is_one(&self) -> bool {
        if let LatticePoint::One(_) = self {
            true
        } else {
            false
        }
    }
}

// For some block, a useful bundle of info.
#[derive(Debug)]
struct BlockSummary {
    block: Block,
    formals: Vec<Value>,
    dests: Vec<(Block, Vec<Value>)>,
}
impl BlockSummary {
    fn new(block: Block, formals: Vec<Value>) -> Self {
        Self {
            block,
            formals,
            dests: vec![],
        }
    }
}

// Solver state.  This holds a LatticePoint for each block formal parameter,
// except for the entry block.
struct SolverState {
    points: HashMap<Value, LatticePoint>,
}
impl SolverState {
    fn new() -> Self {
        Self {
            points: HashMap::new(),
        }
    }
}

/// Remove constant phis in `func`.
#[inline(never)]
pub fn do_remove_constant_phis(func: &mut Function, domtree: &mut DominatorTree) {
    info!("BEGIN do_remove_constant_phis");

    //println!("QQQQ num values = {}", func.dfg.num_values());

    // Get the blocks, in reverse postorder
    let mut blocks_reverse_postorder: Vec<Block> = vec![];
    for block in domtree.cfg_postorder() {
        blocks_reverse_postorder.push(*block);
    }
    blocks_reverse_postorder.reverse();
    //println!("QQQQ blocks = {:?}", blocks_reverse_postorder);
    //println!("");

    // For each block, get the formals
    //for b in &blocks_reverse_postorder {
    //    let formals: &[Value] = func.dfg.block_params(*b);
    //println!("QQQQ {:?} has formals {:?}", b, formals);
    //}
    //println!("");

    // For each block, print the destinations and actuals
    let mut summaries = HashMap::<Block, BlockSummary>::new();

    for b in &blocks_reverse_postorder {
        let formals = func.dfg.block_params(*b);
        let mut summary = BlockSummary::new(*b, formals.to_vec());

        for inst in func.layout.block_insts(*b) {
            let idetails = &func.dfg[inst];
            //println!("QQQQ {:?} {:?} {:?}", b, inst, idetails);
            match idetails {
                InstructionData::Jump {
                    opcode: _,
                    args: _,
                    destination,
                } => {
                    let mut actuals = Vec::<Value>::new();
                    //print!("QQQQ   {:?} -> {:?} args ", b, *destination);
                    for arg in func.dfg.inst_variable_args(inst) {
                        let arg = func.dfg.resolve_aliases(*arg);
                        //print!("{:?} ", arg);
                        actuals.push(arg);
                    }
                    //println!("");
                    summary.dests.push((*destination, actuals));
                }
                InstructionData::Branch {
                    opcode: _,
                    args: _,
                    destination,
                } => {
                    let mut actuals = Vec::<Value>::new();
                    //print!("QQQQ   {:?} -> {:?} args ", b, *destination);
                    for arg in func.dfg.inst_variable_args(inst) {
                        let arg = func.dfg.resolve_aliases(*arg);
                        //print!("{:?} ", arg);
                        actuals.push(arg);
                    }
                    //println!("");
                    summary.dests.push((*destination, actuals));
                }
                InstructionData::BranchTable { .. } => {
                    // Ignore.  These edges carry no parameters.
                }
                // BranchInt, BranchIcmp, BranchFloat, BranchTable
                // Opcode::FallThrough
                // See fn branch_targets in machinst/lower.rs
                other => {
                    assert!(!other.opcode().is_branch());
                }
            }
        }
        // is_branch, is_indirect_branch

        summaries.insert(*b, summary);
    }
    //println!("");

    //for summary in &summaries {
    //println!("QQQQ summary: {:?}", summary);
    //}
    //println!("");

    // Initial solver state
    let entry_block = func
        .layout
        .entry_block()
        .expect("remove_constant_phis: entry block unknown");

    let mut state = SolverState::new();

    // For each block, get the formals
    for b in &blocks_reverse_postorder {
        if *b == entry_block {
            continue;
        }
        let formals: &[Value] = func.dfg.block_params(*b);
        for formal in formals {
            let mb_old_point = state.points.insert(*formal, LatticePoint::None);
            assert!(mb_old_point.is_none());
        }
    }
    //for (val, point) in &state.points {
    //println!("QQQQ initial {:?} = {:?}", val, point);
    //}
    //println!("");

    // Solve
    // Repeatedly traverse the blocks in reverse postorder, until there are no changes
    let mut iter_no = 0;
    loop {
        iter_no += 1;
        //println!("QQQQ iteration {}", iter_no);
        let mut changed = false;

        for src in &blocks_reverse_postorder {
            //println!("QQQQ   src = {:?}", src);
            let src_summary = summaries
                .get(src)
                .expect("remove_constant_phis: block has no summary #1");
            for (dst, src_actuals) in &src_summary.dests {
                //println!("QQQQ     {:?} -> {:?} {:?}", src, dst, src_actuals);
                assert!(*dst != entry_block);
                let dst_summary = summaries
                    .get(dst)
                    .expect("remove_constant_phis: block has no summary #2");
                let dst_formals = &dst_summary.formals;
                assert!(src_actuals.len() == dst_formals.len());
                for i in 0..src_actuals.len() {
                    let formal = &dst_formals[i];
                    let actual = &src_actuals[i];
                    //println!("QQQQ       formal {:?} actual {:?}", formal, actual);

                    // Find the abstract value for `actual`.  If it is a block
                    // formal parameter then the most recent abstract value is
                    // to be found in the running environment.  If it's not,
                    // then it's a real value defining point (not a phi), in
                    // which case return it itself.
                    let actual_point = match state.points.get(actual) {
                        Some(pt) => *pt,
                        None => LatticePoint::One(*actual),
                    };

                    let formal_point_old = state.points.get(formal).unwrap();
                    let formal_point_new = formal_point_old.join(&actual_point);
                    if !formal_point_new.equals(formal_point_old) {
                        changed = true;
                        state.points.insert(*formal, formal_point_new);
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
    let mut n_consts = 0;
    for (_val, point) in &state.points {
        //println!("QQQQ final {:?} = {:?}", val, point);
        if point.is_one() {
            n_consts += 1;
        }
    }
    //println!("QQQQ final: {} formals, of which {} const", state.points.len(), n_consts);

    // Make up a set of blocks that need editing.
    let mut need_editing = HashSet::<Block>::new();
    for (block, summary) in &summaries {
        if *block == entry_block {
            continue;
        }
        let mut n_const = 0;
        for formal in &summary.formals {
            let formal_point = state.points.get(&formal).unwrap();
            if formal_point.is_one() {
                n_const += 1;
            }
        }
        if n_const > 0 {
            //println!("QQQQ {:?} has {} const out of {}", block, n_const, summary.formals.len());
            need_editing.insert(*block);
        }
    }

    // Edit the block set, part 1: deal with the formals.  For each formal
    // which is redundant, remove it, and also add a reroute from it to the
    // constant value which it we know it to be.
    for b in &need_editing {
        let mut del_these = SmallVec::<[(Value, Value); 32]>::new();
        let formals: &[Value] = func.dfg.block_params(*b);
        for val in formals {
            // The state must give an absval for `val`.
            if let LatticePoint::One(replacement_val) = state.points.get(val).unwrap() {
                del_these.push((*val, *replacement_val));
            }
        }
        for (redundant_val, replacement_val) in del_these {
            func.dfg.remove_block_param(redundant_val);
            func.dfg.change_to_alias(redundant_val, replacement_val);
        }
    }

    // Edit the block set, part 2: visit all branch insns.  If the destination
    // has had its formals changed, change the actuals accordingly.
    for b in &blocks_reverse_postorder {
        for inst in func.layout.block_insts(*b) {
            let idetails = &func.dfg[inst];
            //println!("QQQQ {:?} {:?} {:?}", b, inst, idetails);
            let mut mash_this_one: Option<(Block, usize)> = None;
            match idetails {
                InstructionData::Jump {
                    opcode: _,
                    args: _,
                    destination,
                } => {
                    if need_editing.contains(destination) {
                        mash_this_one = Some((*destination, 0));
                    }
                }
                InstructionData::Branch {
                    opcode: _,
                    args: _,
                    destination,
                } => {
                    if need_editing.contains(destination) {
                        mash_this_one = Some((*destination, 1));
                    }
                }
                InstructionData::BranchTable { .. } => {
                    // Ignore.  These edges carry no parameters.
                }
                // BranchInt, BranchIcmp, BranchFloat, BranchTable
                // Opcode::FallThrough
                // See fn branch_targets in machinst/lower.rs
                ref other => {
                    assert!(!other.opcode().is_branch());
                }
            }
            if let Some((destination, num_fixed_args)) = mash_this_one {
                //println!("QQQQ");
                //println!("QQQQ   -- processing");
                //println!("QQQQ   {:?} {:?} {:?}", b, inst, idetails);
                let summary = summaries.get(&destination).unwrap();
                //println!("QQQQ   summary {:?}", summary);

                // Should not fail on Branch or Jump insns
                let old_vals = func.dfg[inst].take_value_list().unwrap();
                let num_old_vals = old_vals.len(&func.dfg.value_lists);
                //print!("QQQQ   old_vals: ");
                //for i in 0 .. num_old_vals {
                //    print!("{:?} ", old_vals.get(i, &func.dfg.value_lists).unwrap());
                //}
                //println!("");

                //let num_total_args = old_value_list.len(&func.dfg.value_lists);
                //let num_varia_args = func.dfg.inst_variable_args(inst).len();
                //let num_fixed_args = num_total_args - num_varia_args; //func.dfg.inst_fixed_args(inst).len();
                //println!("QQQQ   num_total_args {} num_varia_args {} num_fixed_args {}",
                //         num_total_args, num_varia_args, num_fixed_args);
                //println!("QQQQ   num_formals {}", summary.formals.len());
                // Check that the numbers of arguments make sense.
                assert!(num_fixed_args <= num_old_vals);
                assert!(num_fixed_args + summary.formals.len() == num_old_vals);
                // Create a new value list.
                let mut new_value_list = EntityList::<Value>::new();
                // Copy the fixed args to the new list
                for i in 0..num_fixed_args {
                    let val = old_vals.get(i, &func.dfg.value_lists).unwrap();
                    new_value_list.push(val, &mut func.dfg.value_lists);
                }
                // Copy the variable args (the actual block params) to the new
                // list, filtering out redundant ones.
                for i in 0..summary.formals.len() {
                    let actual_i = old_vals
                        .get(num_fixed_args + i, &func.dfg.value_lists)
                        .unwrap();
                    let formal_i = summary.formals[i];
                    let is_redundant = state.points.get(&formal_i).unwrap().is_one();
                    if !is_redundant {
                        new_value_list.push(actual_i, &mut func.dfg.value_lists);
                    }
                }
                //let num_args_after = new_value_list.len(&func.dfg.value_lists);
                //println!("QQQQ   args before {} after {}", num_old_vals, num_args_after);
                func.dfg[inst].put_value_list(new_value_list);
            }
        }
    }
    //println!("QQQQ");

    //let _tt = timing::dce();
    //debug_assert!(domtree.is_valid());
    //
    //let mut live = vec![false; func.dfg.num_values()];
    //for &block in domtree.cfg_postorder() {
    //    let mut pos = FuncCursor::new(func).at_bottom(block);
    //    while let Some(inst) = pos.prev_inst() {
    //        {
    //            if has_side_effect(pos.func, inst)
    //                || any_inst_results_used(inst, &live, &pos.func.dfg)
    //            {
    //                for arg in pos.func.dfg.inst_args(inst) {
    //                    let v = pos.func.dfg.resolve_aliases(*arg);
    //                    live[v.index()] = true;
    //                }
    //                continue;
    //            }
    //        }
    //        pos.remove_inst();
    //    }
    //}
    info!(
        "END do_remove_constant_phis: {} iters.  {} formals, of which {} const.",
        iter_no,
        state.points.len(),
        n_consts
    );
    //panic!("Meh.");
}
