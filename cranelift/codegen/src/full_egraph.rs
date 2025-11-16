//! Full-egraph optimization pass.
//!
//! Unlike the aegraph functionality (see [`crate::egraph`]), this
//! builds a fully separate representaion of the program, and performs
//! a conventional egraph batch-rewrite optimization (as in [1]).
//!
//! [1]: M Willsey, C Nandi, YR Wang, O Flatt, Z Tatlock, P
//!      Pancheckha. egg: Fast and Extensible Equality Saturation. In
//!      POPL 2021. <https://dl.acm.org/doi/10.1145/3434304>
//!
//! TODO:
//! - Full egraph pass using egg. Elaboration as in aegraphs (do a
//!   simple recursive thing).
//! - repeat passes in aegraph.
//! - sweep parameters in aegraph (no rewrites up to N rewrites).
//! - single (best) rewrite only in aegraph.

use crate::cursor::{Cursor, CursorPosition, FuncCursor};
use crate::dominator_tree::DominatorTree;
use crate::flowgraph::ControlFlowGraph;
use crate::ir::{DataFlowGraph, Function, Inst, InstructionData, Type, Value, ValueListPool};
use alloc::vec::Vec;
use cranelift_entity::SecondaryMap;
use cranelift_entity::packed_option::ReservedValue;
use egg::{EGraph, Id, Language};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Node {
    Insn {
        data: InstructionData,
        ty: Type,
        args: Vec<Id>,
    },
    Root(Value),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NodeDiscriminant {
    Insn(InstructionData, Type),
    Root(Value),
}

impl Node {
    fn new(dfg: &mut DataFlowGraph, inst: Inst, value_to_id: &SecondaryMap<Value, Id>) -> Node {
        // Translate args to Ids and null out the values in the
        // InstructionData so we can use `self` as the discriminant.
        let mut args = vec![];
        dfg.map_inst_values(inst, |value| {
            args.push(value_to_id[value]);
            Value::reserved_value()
        });
        let data = dfg.insts[inst];
        let ty = dfg.ctrl_typevar(inst);
        Node::Insn { data, ty, args }
    }
}

impl Language for Node {
    type Discriminant = NodeDiscriminant;
    fn discriminant(&self) -> Self::Discriminant {
        match self {
            Node::Insn { data, ty, .. } => NodeDiscriminant::Insn(*data, *ty),
            Node::Root(value) => NodeDiscriminant::Root(*value),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        self == other
    }

    fn children(&self) -> &[Id] {
        match self {
            Node::Insn { args, .. } => &args[..],
            _ => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Node::Insn { args, .. } => &mut args[..],
            _ => &mut [],
        }
    }
}

pub(crate) struct FullEgraphPass<'a> {
    func: &'a mut Function,
    egraph: EGraph<Node, ()>,
    value_to_id: SecondaryMap<Value, Id>,
    cfg: ControlFlowGraph,
    domtree: DominatorTree,
}

impl<'a> FullEgraphPass<'a> {
    fn new(func: &'a mut Function) -> Self {
        let cfg = ControlFlowGraph::with_function(func);
        let domtree = DominatorTree::with_function(func, &cfg);
        FullEgraphPass {
            func,
            egraph: EGraph::new(()),
            value_to_id: SecondaryMap::default(),
            cfg,
            domtree,
        }
    }

    fn func_to_egraph(&mut self) {
        // Traverse the function in RPO so we always see defs before uses.
        let mut cursor = FuncCursor::new(self.func);
        for &block in self.domtree.cfg_rpo() {
            // Create entries for the blockparams.
            for &blockparam in cursor.func.dfg.block_params(block) {
                let id = self.egraph.add(Node::Root(blockparam));
                self.value_to_id[blockparam] = id;
            }

            // Create entries for pure instructions, removing them
            // from the function.
            cursor.set_position(CursorPosition::Before(block));
            while let Some(insn) = cursor.next_inst() {
                if crate::inst_predicates::is_pure_for_egraph(cursor.func, insn) {
                    let result = *cursor
                        .func
                        .dfg
                        .inst_results(insn)
                        .get(0)
                        .expect("egraph-pure instructions have exactly one result");
                    let node = Node::new(&mut cursor.func.dfg, insn, &self.value_to_id);
                    let id = self.egraph.add(node);
                    self.value_to_id[result] = id;
                    //cursor.remove_inst_and_step_back();
                } else {
                    // Create a new Root node for each result value.
                    for &result in cursor.func.dfg.inst_results(insn) {
                        let id = self.egraph.add(Node::Root(result));
                        self.value_to_id[result] = id;
                    }
                }
            }
        }
    }

    fn egraph_rewrite(&mut self) {
        
    }
    
    fn egraph_to_func(&mut self) {}
}

pub(crate) fn run(func: &mut Function) {
    let mut pass = FullEgraphPass::new(func);
    pass.func_to_egraph();
    pass.egraph_rewrite();
    pass.egraph_to_func();
}
