//! Optimization driver using ISLE rewrite rules on an egraph.

pub use crate::ir::condcodes::{FloatCC, IntCC};
use crate::ir::dfg::ValueDef;
pub use crate::ir::immediates::{Ieee128, Ieee16, Ieee32, Ieee64, Imm64, Offset32, Uimm8, V128Imm};
use crate::ir::instructions::InstructionFormat;
pub use crate::ir::types::*;
pub use crate::ir::{
    AtomicRmwOp, BlockCall, Constant, DynamicStackSlot, FuncRef, GlobalValue, Immediate,
    InstructionData, MemFlags, Opcode, StackSlot, TrapCode, Type, Value,
};
use crate::isle_common_prelude_methods;
use crate::machinst::isle::*;
use crate::trace;
use cranelift_entity::packed_option::ReservedValue;
use smallvec::{smallvec, SmallVec};
use std::marker::PhantomData;

#[allow(dead_code)]
pub type Unit = ();
pub type Range = (usize, usize);
pub type ValueArray2 = [Value; 2];
pub type ValueArray3 = [Value; 3];

const MAX_ISLE_RETURNS: usize = 8;

pub type ConstructorVec<T> = SmallVec<[T; MAX_ISLE_RETURNS]>;

type TypeAndInstructionData = (Type, InstructionData);

impl<T: smallvec::Array> generated_code::Length for SmallVec<T> {
    #[inline]
    fn len(&self) -> usize {
        SmallVec::len(self)
    }
}

pub(crate) mod generated_code;
use generated_code::{ContextIter, IntoContextIter};

pub(crate) trait EgraphImpl {
    fn insert_node(&mut self, op: InstructionData, ty: Type) -> Value;
    fn func(&mut self) -> &mut Function;
    fn remat(&mut self, value: Value) -> Value;
    fn subsume(&mut self, value: Value) -> Value;
    fn eclass_members_direct() -> bool {
        false
    }
    fn eclass_members(&self, _value: Value) -> SmallVec<[Value; 8]> {
        smallvec![]
    }
}

pub(crate) struct IsleContext<'a, E: EgraphImpl>
where
    E: 'a,
{
    pub(crate) ctx: &'a mut E,
}

pub(crate) struct InstDataEtorIter<'a, E: EgraphImpl>
where
    E: 'a,
{
    stack: SmallVec<[Value; 8]>,
    _phantom1: PhantomData<&'a E>,
}

impl<'a, E: EgraphImpl> Default for InstDataEtorIter<'a, E> {
    fn default() -> Self {
        InstDataEtorIter::<'a, E> {
            stack: SmallVec::default(),
            _phantom1: PhantomData,
        }
    }
}

impl<'a, E: EgraphImpl> InstDataEtorIter<'a, E> {
    fn new(egraph: &E, root: Value) -> Self {
        debug_assert_ne!(root, Value::reserved_value());
        if E::eclass_members_direct() {
            Self {
                stack: egraph.eclass_members(root),
                _phantom1: PhantomData,
            }
        } else {
            Self {
                stack: smallvec![root],
                _phantom1: PhantomData,
            }
        }
    }
}

impl<'a, E: EgraphImpl> ContextIter for InstDataEtorIter<'a, E>
where
    E: 'a,
{
    type Context = IsleContext<'a, E>;
    type Output = (Type, InstructionData);

    fn next(&mut self, ctx: &mut IsleContext<'a, E>) -> Option<Self::Output> {
        if E::eclass_members_direct() {
            self.stack
                .pop()
                .and_then(|val| match ctx.ctx.func().dfg.value_def(val) {
                    ValueDef::Result(inst, _) => {
                        let ty = ctx.ctx.func().dfg.value_type(val);
                        trace!(" -> value of type {}", ty);
                        Some((ty, ctx.ctx.func().dfg.insts[inst]))
                    }
                    _ => None,
                })
        } else {
            while let Some(value) = self.stack.pop() {
                debug_assert!(ctx.ctx.func().dfg.value_is_real(value));
                trace!("iter: value {:?}", value);
                match ctx.ctx.func().dfg.value_def(value) {
                    ValueDef::Union(x, y) => {
                        debug_assert_ne!(x, Value::reserved_value());
                        debug_assert_ne!(y, Value::reserved_value());
                        trace!(" -> {}, {}", x, y);
                        self.stack.push(x);
                        self.stack.push(y);
                        continue;
                    }
                    ValueDef::Result(inst, _)
                        if ctx.ctx.func().dfg.inst_results(inst).len() == 1 =>
                    {
                        let ty = ctx.ctx.func().dfg.value_type(value);
                        trace!(" -> value of type {}", ty);
                        return Some((ty, ctx.ctx.func().dfg.insts[inst]));
                    }
                    _ => {}
                }
            }
            None
        }
    }
}

impl<'a, E: EgraphImpl> IntoContextIter for InstDataEtorIter<'a, E>
where
    E: 'a,
{
    type Context = IsleContext<'a, E>;
    type Output = (Type, InstructionData);
    type IntoIter = Self;

    fn into_context_iter(self) -> Self {
        self
    }
}

pub(crate) struct MaybeUnaryEtorIter<'a, E: EgraphImpl>
where
    E: 'a,
{
    opcode: Option<Opcode>,
    inner: InstDataEtorIter<'a, E>,
    fallback: Option<Value>,
}

impl<'a, E: EgraphImpl> Default for MaybeUnaryEtorIter<'a, E> {
    fn default() -> Self {
        MaybeUnaryEtorIter::<'a, E> {
            opcode: None,
            inner: Default::default(),
            fallback: None,
        }
    }
}

impl<'a, E: EgraphImpl> MaybeUnaryEtorIter<'a, E>
where
    E: 'a,
{
    fn new(egraph: &E, opcode: Opcode, value: Value) -> Self {
        debug_assert_eq!(opcode.format(), InstructionFormat::Unary);
        Self {
            opcode: Some(opcode),
            inner: InstDataEtorIter::new(egraph, value),
            fallback: Some(value),
        }
    }
}

impl<'a, E: EgraphImpl> ContextIter for MaybeUnaryEtorIter<'a, E>
where
    E: 'a,
{
    type Context = IsleContext<'a, E>;
    type Output = (Type, Value);

    fn next(&mut self, ctx: &mut IsleContext<'a, E>) -> Option<Self::Output> {
        debug_assert_ne!(self.opcode, None);
        while let Some((ty, inst_def)) = self.inner.next(ctx) {
            let InstructionData::Unary { opcode, arg } = inst_def else {
                continue;
            };
            if Some(opcode) == self.opcode {
                self.fallback = None;
                return Some((ty, arg));
            }
        }

        self.fallback.take().map(|value| {
            let ty = generated_code::Context::value_type(ctx, value);
            (ty, value)
        })
    }
}

impl<'a, E: EgraphImpl> IntoContextIter for MaybeUnaryEtorIter<'a, E>
where
    E: 'a,
{
    type Context = IsleContext<'a, E>;
    type Output = (Type, Value);
    type IntoIter = Self;

    fn into_context_iter(self) -> Self {
        self
    }
}

impl<'a, E: EgraphImpl> generated_code::Context for IsleContext<'a, E>
where
    E: 'a,
{
    isle_common_prelude_methods!();

    type inst_data_etor_returns = InstDataEtorIter<'a, E>;

    fn inst_data_etor(&mut self, eclass: Value, returns: &mut InstDataEtorIter<'a, E>) {
        *returns = InstDataEtorIter::<'a, E>::new(self.ctx, eclass);
    }

    type inst_data_tupled_etor_returns = InstDataEtorIter<'a, E>;

    fn inst_data_tupled_etor(&mut self, eclass: Value, returns: &mut InstDataEtorIter<'a, E>) {
        // Literally identical to `inst_data_etor`, just a different nominal type in ISLE
        self.inst_data_etor(eclass, returns);
    }

    fn make_inst_ctor(&mut self, ty: Type, op: &InstructionData) -> Value {
        let value = self.ctx.insert_node(*op, ty);
        trace!("make_inst_ctor: {:?} -> {}", op, value);
        value
    }

    fn value_array_2_ctor(&mut self, arg0: Value, arg1: Value) -> ValueArray2 {
        [arg0, arg1]
    }

    fn value_array_3_ctor(&mut self, arg0: Value, arg1: Value, arg2: Value) -> ValueArray3 {
        [arg0, arg1, arg2]
    }

    #[inline]
    fn value_type(&mut self, val: Value) -> Type {
        self.ctx.func().dfg.value_type(val)
    }

    fn iconst_sextend_etor(
        &mut self,
        (ty, inst_data): (Type, InstructionData),
    ) -> Option<(Type, i64)> {
        if let InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            imm,
        } = inst_data
        {
            Some((ty, self.i64_sextend_imm64(ty, imm)))
        } else {
            None
        }
    }

    fn remat(&mut self, value: Value) -> Value {
        trace!("remat: {}", value);
        self.ctx.remat(value)
    }

    fn subsume(&mut self, value: Value) -> Value {
        trace!("subsume: {}", value);
        self.ctx.subsume(value)
    }

    fn splat64(&mut self, val: u64) -> Constant {
        let val = u128::from(val);
        let val = val | (val << 64);
        let imm = V128Imm(val.to_le_bytes());
        self.ctx.func().dfg.constants.insert(imm.into())
    }

    type sextend_maybe_etor_returns = MaybeUnaryEtorIter<'a, E>;
    fn sextend_maybe_etor(&mut self, value: Value, returns: &mut Self::sextend_maybe_etor_returns) {
        *returns = MaybeUnaryEtorIter::new(self.ctx, Opcode::Sextend, value);
    }

    type uextend_maybe_etor_returns = MaybeUnaryEtorIter<'a, E>;
    fn uextend_maybe_etor(&mut self, value: Value, returns: &mut Self::uextend_maybe_etor_returns) {
        *returns = MaybeUnaryEtorIter::new(self.ctx, Opcode::Uextend, value);
    }

    // NB: Cranelift's defined semantics for `fcvt_from_{s,u}int` match Rust's
    // own semantics for converting an integer to a float, so these are all
    // implemented with `as` conversions in Rust.
    fn f32_from_uint(&mut self, n: u64) -> Ieee32 {
        Ieee32::with_float(n as f32)
    }

    fn f64_from_uint(&mut self, n: u64) -> Ieee64 {
        Ieee64::with_float(n as f64)
    }

    fn f32_from_sint(&mut self, n: i64) -> Ieee32 {
        Ieee32::with_float(n as f32)
    }

    fn f64_from_sint(&mut self, n: i64) -> Ieee64 {
        Ieee64::with_float(n as f64)
    }

    fn u64_bswap16(&mut self, n: u64) -> u64 {
        (n as u16).swap_bytes() as u64
    }

    fn u64_bswap32(&mut self, n: u64) -> u64 {
        (n as u32).swap_bytes() as u64
    }

    fn u64_bswap64(&mut self, n: u64) -> u64 {
        n.swap_bytes()
    }

    fn ieee128_constant_extractor(&mut self, n: Constant) -> Option<Ieee128> {
        self.ctx.func().dfg.constants.get(n).try_into().ok()
    }

    fn ieee128_constant(&mut self, n: Ieee128) -> Constant {
        self.ctx.func().dfg.constants.insert(n.into())
    }
}
