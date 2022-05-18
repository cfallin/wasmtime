//! Computation of "facts": mid-end analysis results from rules
//! specified in ISLE.
//!
//! To add an analysis:
//!
//! - Add a field to `FactTables` to hold the state per instruction or
//!   value.
//! - Add a macro invocation in the `impl generated::Context` block
//!   below using `fact_table_per_inst_ctors` or
//!   `fact_table_per_value_ctors`.
//! - Add an invocation to the toplevel `compute_*` ISLE entry point
//!   in the `for inst in ...` loop in `analyze()` below.
//! - Write the analysis in a new file in `facts/*.isle`, providing
//!   the analysis type (e.g. `LastStoreState`), the default-value
//!   ctor (e.g. `last_store_default`), and the main evaluation
//!   entry point (e.g. `last_store`).
//! - Add the analysis ISLE file to cranelift/codegen/build.rs
//!   near where e.g. `last_store.isle` is named.
//!
//! And that's it!

use crate::dominator_tree::DominatorTree;
use crate::fx::FxHashMap;
use crate::machinst::isle::{analysis::*, *};
use crate::{ir::*, settings};
use cranelift_entity::SecondaryMap;

isle_common_prelude_uses!();

#[derive(Clone, Debug, Default)]
pub struct FactTables {
    /// Single predecessor, if any.
    pub single_pred: SecondaryMap<Inst, PackedOption<Inst>>,
    /// Last-store indices: per instruction, if a memory op, the
    /// instruction of the last prior instruction that may store to
    /// its accessed memory.
    pub last_store: SecondaryMap<Inst, Option<generated::LastStoreState>>,
    /// Known memory values: per (last_store, addr_value, ty), a
    /// value, if any, known to be equal to the value stored in that
    /// location.
    ///
    /// TODO: right now this is inverted (per value, the last
    /// store/value/type) to build but we need a better way to handle
    /// sparse relations
    pub mem_values: FxHashMap<Value, (Value, Inst, Type)>,
    
    //    pub mem_values: FxHashMap<(Inst, Value, Type), Value>,
}

/// Perform an analysis of the function.
pub fn analyze(
    func: &Function,
    flags: &settings::Flags,
    domtree: &mut DominatorTree,
) -> FactTables {
    let mut ctx = IsleContext {
        func,
        flags,
        facts: FactTables::default(),
    };

    // For each block, in RPO (so we see defs before uses except for
    // cycles via blockparams), compute facts for each instruction.
    for &block in domtree.cfg_postorder().iter().rev() {
        let mut last_inst = None;
        for inst in func.layout.block_insts(block) {
            ctx.facts.single_pred[inst] = last_inst.into();
            last_inst = Some(inst);

            ctx.facts.last_store[inst] = generated::compute_last_store(&mut ctx, inst);

            // TODO: compute_mem_values() per value...  PROBLEM: need
            // a way for a "fact" relation/constructor to populate
            // sparsely. compute_mem_values needs to *produce* the key
            // (last_store, type, inst) with which its memoized tuple
            // is stored.
        }
    }

    ctx.facts
}

macro_rules! fact_table_per_inst_ctors {
    ($name:tt, $ty:ty) => {
        fn $name(&mut self, inst: Inst) -> Option<$ty> {
            // Return the already-computed value, or `None` if not. We
            // do not recurse on a missing fact because the top-level
            // eager evaluation strategy should fill in any facts that
            // we are allowed to rely on before we reach this inst.
            self.facts.$name[inst].clone()
        }
    };
}

macro_rules! fact_table_per_value_ctors {
    ($name:tt, $ty:ty) => {
        fn $name(&mut self, value: Value) -> Option<$ty> {
            self.facts.$name.get(&value).cloned()
        }
    };
}

/// Implementations of extern constructors usable from analyses in
/// ISLE.
impl<'a> generated::Context for IsleContext<'a> {
    isle_common_prelude_methods!();

    fn inst_single_pred(&mut self, inst: Inst) -> Option<Inst> {
        self.facts.single_pred[inst].into()
    }

    fn inst_no_single_pred(&mut self, inst: Inst) -> Option<()> {
        if self.facts.single_pred[inst].is_none() {
            Some(())
        } else {
            None
        }
    }

    fact_table_per_inst_ctors!(last_store, generated::LastStoreState);
//    fact_table_per_value_ctors!(mem_values, (Value, Inst, Type));
}

mod generated {
    // See https://github.com/rust-lang/rust/issues/47995: we cannot
    // use `#![...]` attributes inside of the generated ISLE source
    // below because we include!() it. We must include!() it because
    // its path depends on an environment variable; and also because
    // of this, we can't do the `#[path = "..."]  mod generated_code;`
    // trick either.
    #![allow(dead_code, unreachable_code, unreachable_patterns)]
    #![allow(unused_imports, unused_variables, non_snake_case, unused_mut)]
    #![allow(irrefutable_let_patterns)]

    include!(concat!(env!("ISLE_DIR"), "/isle_facts.rs"));
}
