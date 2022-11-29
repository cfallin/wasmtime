//! Optimization driver using ISLE rewrite rules on an egraph.

use crate::egraph::Analysis;
use crate::egraph::FuncEGraph;
pub use crate::egraph::{Node, NodeCtx};
use crate::egraph_in_dfg::{NewOrExistingInst, OptimizeCtx};
use crate::ir::condcodes;
pub use crate::ir::condcodes::{FloatCC, IntCC};
pub use crate::ir::immediates::{Ieee32, Ieee64, Imm64, Offset32, Uimm32, Uimm64, Uimm8};
pub use crate::ir::types::*;
pub use crate::ir::{
    dynamic_to_fixed, AtomicRmwOp, Block, Constant, DynamicStackSlot, FuncRef, GlobalValue, Heap,
    Immediate, InstructionData, JumpTable, MemFlags, Opcode, StackSlot, Table, TrapCode, Type,
    Value, ValueData,
};
use crate::isle_common_prelude_methods;
use crate::machinst::isle::*;
use crate::trace;
use cranelift_entity::{EntityList, EntityRef};
use smallvec::{smallvec, SmallVec};

#[allow(dead_code)]
pub type Unit = ();
pub type Range = (usize, usize);
pub type ValueArray2 = [Value; 2];
pub type ValueArray3 = [Value; 3];

pub type ConstructorVec<T> = SmallVec<[T; 8]>;

pub(crate) mod generated_code;
use generated_code::ContextIter;

pub(crate) struct IsleContext<'a, 'b> {
    ctx: &'a mut OptimizeCtx<'b>,
}

/*
pub(crate) fn store_to_load<'a>(id: Id, egraph: &mut FuncEGraph<'a>) -> Id {
    // Note that we only examine the latest enode in the eclass: opts
    // are invoked for every new enode added to an eclass, so
    // traversing the whole eclass would be redundant.
    let load_key = egraph.egraph.classes[id].get_node().unwrap();
    if let Node::Load {
        op:
            InstructionImms::Load {
                opcode: Opcode::Load,
                offset: load_offset,
                ..
            },
        ty: load_ty,
        addr: load_addr,
        mem_state,
        ..
    } = load_key.node(&egraph.egraph.nodes)
    {
        if let Some(store_inst) = mem_state.as_store() {
            trace!(" -> got load op for id {}", id);
            if let Some((store_ty, store_id)) = egraph.store_nodes.get(&store_inst) {
                trace!(" -> got store id: {} ty: {}", store_id, store_ty);
                let store_key = egraph.egraph.classes[*store_id].get_node().unwrap();
                if let Node::Inst {
                    op:
                        InstructionImms::Store {
                            opcode: Opcode::Store,
                            offset: store_offset,
                            ..
                        },
                    args: store_args,
                    ..
                } = store_key.node(&egraph.egraph.nodes)
                {
                    let store_args = store_args.as_slice(&egraph.node_ctx.args);
                    let store_data = store_args[0];
                    let store_addr = store_args[1];
                    if *load_offset == *store_offset
                        && *load_ty == *store_ty
                        && egraph.egraph.unionfind.equiv_id_mut(*load_addr, store_addr)
                    {
                        trace!(" -> same offset, type, address; forwarding");
                        egraph.stats.store_to_load_forward += 1;
                        return store_data;
                    }
                }
            }
        }
    }

    id
}
*/

struct NodesEtorIter {
    stack: SmallVec<[Value; 8]>,
}
impl NodesEtorIter {
    fn new(root: Value) -> Self {
        Self {
            stack: smallvec![root],
        }
    }
}

impl ContextIter for NodesEtorIter {
    type Context<'a> = OptimizeCtx<'a>;
    type Output = (Type, InstructionData);

    fn next(&mut self, ctx: &mut IsleContext<'_, '_>) -> Option<Self::Output> {
        while let Some(value) = self.stack.pop() {
            let value = ctx.ctx.func.dfg.resolve_aliases(value);
            trace!("iter: value {:?}", value);
            match &ctx.ctx.func.dfg[value] {
                &ValueData::Union { x, y, .. } => {
                    self.stack.push(x);
                    self.stack.push(y);
                    continue;
                }
                &ValueData::Inst { ty, inst, .. }
                    if ctx.ctx.func.dfg.inst_results(inst).len() == 1 =>
                {
                    return Some((ty, &ctx.ctx.func.dfg[inst]));
                }
                _ => {}
            }
        }
        None
    }
}

impl<'a, 'b> generated_code::Context for IsleContext<'a, 'b> {
    isle_common_prelude_methods!();

    fn eclass_type(&mut self, eclass: Value) -> Option<Type> {
        Some(self.ctx.func.dfg.value_type(eclass))
    }

    fn at_loop_level(&mut self, eclass: Value) -> (u8, Value) {
        (
            self.ctx.analysis_values[eclass].loop_level.level() as u8,
            eclass,
        )
    }

    type enodes_etor_iter = NodesEtorIter;

    fn enodes_etor(&mut self, eclass: Value) -> Option<NodesEtorIter> {
        Some(NodesEtorIter::new(eclass))
    }

    fn pure_enode_ctor(&mut self, ty: Type, op: &InstructionData) -> Value {
        self.ctx
            .insert_pure_enode(NewOrExistingInst::New(op.clone()))
    }

    fn value_array_2_ctor(&mut self, arg0: Value, arg1: Value) -> ValueArray2 {
        [arg0, arg1]
    }

    fn value_array_3_ctor(&mut self, arg0: Value, arg1: Value, arg2: Value) -> ValueArray3 {
        [arg0, arg1, arg2]
    }

    fn remat(&mut self, value: Value) -> Value {
        trace!("remat: {}", value);
        self.ctx.remat_values.insert(value);
        value
    }

    fn subsume(&mut self, value: Value) -> Value {
        trace!("subsume: {}", value);
        self.ctx.subsume_values.insert(value);
        value
    }
}
