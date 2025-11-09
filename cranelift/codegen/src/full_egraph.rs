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

use crate::cursor::{Cursor, FuncCursor};
use crate::ir::{Function, InstructionData, Type, Value, ValueListPool};
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
    fn new(
        mut data: InstructionData,
        ty: Type,
        pool: &mut ValueListPool,
        value_to_id: &SecondaryMap<Value, Id>,
    ) -> Node {
        // Translate args to Ids and null out the values in the
        let mut args = vec![];
        // InstructionData so we can use `self` as the discriminant.
        for arg in data.arguments_mut(pool) {
            args.push(value_to_id[*arg]);
            *arg = Value::reserved_value();
        }
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

struct FullEgraphPass<'a> {
    func: &'a mut Function,
    egraph: EGraph<Node, ()>,
    value_to_id: SecondaryMap<Value, Id>,
}

pub(crate) fn run_full_egraph(func: &mut Function) {
    // First, convert the full CFG to an egraph.
    let mut value_to_id = SecondaryMap::new();

    let mut cursor = FuncCursor::new(func);
    while let Some(block) = cursor.next_block() {
        // Create entries for the blockparams.

        while let Some(insn) = cursor.next_inst() {}
    }
}
