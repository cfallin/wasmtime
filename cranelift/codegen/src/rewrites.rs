//! Simple single-pass rewrite transform.

use crate::ir::*;
use crate::opts::{EgraphImpl, IsleContext};
use crate::{
    cursor::{Cursor, FuncCursor},
    inst_predicates::is_pure_for_egraph,
};
use smallvec::{smallvec, SmallVec};

pub(crate) struct Rewriter<'a> {
    cursor: FuncCursor<'a>,
}

impl<'a> EgraphImpl for Rewriter<'a> {
    fn insert_node(&mut self, op: InstructionData, ty: Type) -> Value {
        let (inst, dfg) = self.cursor.ins().build(op, ty);
        let result = dfg.inst_results(inst)[0];
        log::trace!(
            "insert_node {:?} ty {:?} -> inst {} result {}",
            op,
            ty,
            inst,
            result
        );
        result
    }
    fn func(&mut self) -> &mut Function {
        &mut self.cursor.func
    }
    fn remat(&mut self, value: Value) -> Value {
        value
    }
    fn subsume(&mut self, value: Value) -> Value {
        value
    }
    fn eclass_members_direct() -> bool {
        true
    }
    fn eclass_members(&self, value: Value) -> SmallVec<[Value; 8]> {
        smallvec![value]
    }
}

impl<'a> Rewriter<'a> {
    pub(crate) fn new(func: &'a mut Function) -> Rewriter<'a> {
        Rewriter {
            cursor: FuncCursor::new(func),
        }
    }

    pub(crate) fn run(&mut self) {
        log::trace!("running rewriter on body:\n{}", self.cursor.func.display());
        let mut optimized_values: SmallVec<[Value; 5]> = SmallVec::new();
        while let Some(_block) = self.cursor.next_block() {
            while let Some(inst) = self.cursor.next_inst() {
                if is_pure_for_egraph(self.cursor.func, inst) {
                    let value = self.cursor.func.dfg.inst_results(inst)[0];
                    log::trace!("considering inst {} with result {}", inst, value);
                    optimized_values.clear();
                    crate::opts::generated_code::constructor_simplify(
                        &mut IsleContext { ctx: self },
                        value,
                        &mut optimized_values,
                    );
                    log::trace!(" -> rewritten values: {:?}", optimized_values);

                    if let Some(new_value) = optimized_values
                        .iter()
                        .cloned()
                        .filter(|&v| v != value)
                        .next()
                    {
                        // Remove the original inst and rewrite its
                        // result to be an alias of this value.
                        match self.cursor.func.dfg.value_def(new_value) {
                            ValueDef::Result(new_inst, 0) => {
                                self.cursor.func.dfg.replace_with_aliases(inst, new_inst);
                                self.cursor.remove_inst_and_step_back();
                            }
                            _ => {}
                        };
                    }
                }
            }
        }
        log::trace!("after rewriter:\n{}", self.cursor.func.display());
    }
}
