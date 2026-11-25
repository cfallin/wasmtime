//! Replaying core object construction. Host functions become stubs, and
//! modules are compiled from the bytecode that the embedder provides.

use super::super::*;
use super::Trampolines;
use crate::runtime::vm;
use crate::{
    Extern, FuncType, Global, GlobalType, HeapType, Instance, MemoryType, Module, Mutability, Ref,
    RefType, StoreContextMut, Table, TableType, Val,
};
use objects::{RecordedGlobal, RecordedModule, RecordedTable};
use wasmtime_environ::{EntityIndex, Initializer};

/// Replays a record of the construction of an object other than a module or an
/// instance.
pub(super) fn replay_object(
    store: &mut StoreInner<()>,
    trampolines: &Trampolines,
    tag: u8,
    body: &mut Reader<'_>,
) -> Result<()> {
    let id = body.u32()?;
    match tag {
        codec::HOST => {
            let params = Kind::decode_list(body.blob()?)?;
            let results = Kind::decode_list(body.blob()?)?;
            body.end()?;
            let tys = |kinds: &[Kind]| kinds.iter().map(|k| k.ty()).collect::<Result<Vec<_>>>();
            let ty = FuncType::new(store.engine(), tys(&params)?, tys(&results)?);
            let func = trampolines.host_stub(store, ty)?;
            let mut session = store.rr.session.take().unwrap();
            let added = session.objects.add_func(store, func);
            store.rr.session = Some(session);
            check_id(RecordedFunc::from_u32(id), added?)?;
        }
        codec::GLOBAL => {
            let kind = body.kind()?;
            let mutability = if body.bool()? {
                Mutability::Var
            } else {
                Mutability::Const
            };
            let objects = store.rr_objects();
            check_id(RecordedGlobal::from_u32(id), objects.globals.next_key())?;
            let raw = body.value(kind, store, objects)?;
            body.end()?;
            let value = match kind {
                Kind::I32 => Val::I32(raw.get_i32()),
                Kind::I64 => Val::I64(raw.get_i64()),
                Kind::F32 => Val::F32(raw.get_f32()),
                Kind::F64 => Val::F64(raw.get_f64()),
                Kind::V128 => Val::V128(raw.get_v128().into()),
                // SAFETY: `Reader::value` returned this store's function.
                Kind::FuncRef { .. } => Val::FuncRef(unsafe {
                    NonNull::new(raw.get_funcref().cast())
                        .map(|f| Func::from_vm_func_ref(store.id(), f))
                }),
                Kind::Unsupported => unreachable!(),
            };
            Global::new(
                StoreContextMut(store),
                GlobalType::new(kind.ty()?, mutability),
                value,
            )?;
        }
        codec::MEMORY => {
            let memory64 = body.bool()?;
            let page_size_log2 = body.u8()?;
            let (min, max) = limits(body)?;
            body.end()?;
            let next = store.rr_objects().memories.next_key();
            check_id(RecordedMemory::from_u32(id), next)?;
            let ty = MemoryType::builder()
                .memory64(memory64)
                .min(min)
                .max(max)
                .page_size_log2(page_size_log2)
                .build()?;
            Memory::new(StoreContextMut(store), ty)?;
        }
        codec::TABLE => {
            let Kind::FuncRef { nullable } = body.kind()? else {
                bail!("invalid table type in trace");
            };
            let table64 = body.bool()?;
            let (min, max) = limits(body)?;
            let objects = store.rr_objects();
            let init = Ref::Func(objects.decode_ref(body.u32()?)?);
            body.end()?;
            check_id(RecordedTable::from_u32(id), objects.tables.next_key())?;
            let element = RefType::new(nullable, HeapType::Func);
            let ty = if table64 {
                TableType::new64(element, min, max)?
            } else {
                TableType::new(
                    element,
                    u32::try_from(min)?,
                    max.map(u32::try_from).transpose()?,
                )?
            };
            Table::new(StoreContextMut(store), ty, init)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Replays an instance's construction, returning it and its startup function,
/// which a separate activation runs.
pub(super) fn replay_instance(
    store: &mut StoreOpaque,
    body: &mut Reader<'_>,
) -> Result<(Instance, Option<RecordedFunc>)> {
    fn find<T: Clone>(object: Option<&T>) -> Result<T> {
        object
            .cloned()
            .ok_or_else(|| format_err!("invalid object ID in trace"))
    }

    let objects = store.rr_objects();
    let module = find(objects.modules.get(RecordedModule::from_u32(body.u32()?)))?;
    let mut imports = TryVec::new();
    for initializer in &module.env_module().initializers {
        let Initializer::Import { index, .. } = *initializer;
        let id = body.u32()?;
        imports.push(match index {
            EntityIndex::Function(_) => {
                Extern::Func(objects.referable_func(RecordedFunc::from_u32(id))?)
            }
            EntityIndex::Memory(_) => {
                Extern::Memory(find(objects.memories.get(RecordedMemory::from_u32(id)))?)
            }
            EntityIndex::Table(_) => {
                Extern::Table(find(objects.tables.get(RecordedTable::from_u32(id)))?)
            }
            EntityIndex::Global(_) => {
                Extern::Global(find(objects.globals.get(RecordedGlobal::from_u32(id)))?)
            }
            _ => bail!("unsupported import in trace"),
        })?;
    }
    body.end()?;
    let imports = Instance::typecheck_externs(store, &module, &imports)?;
    // SAFETY: imports were checked against module. Allocation has no async
    // limiter.
    let (instance, _) =
        vm::assert_ready(unsafe { Instance::new_raw(store, None, &module, imports.as_ref()) })?;
    let store_id = store.id();
    let (inst, registry) = instance.id.get_mut_and_module_registry(store);
    // SAFETY: the new instance belongs to this store.
    let startup = unsafe { inst.get_startup_func(registry, store_id) };
    let startup = startup
        .map(|func| store.rr_objects().find_func(store, func))
        .transpose()?;
    Ok((instance, startup))
}

fn limits(body: &mut Reader<'_>) -> Result<(u64, Option<u64>)> {
    let min = body.u64()?;
    let has_max = body.bool()?;
    let max = body.u64()?;
    Ok((min, has_max.then_some(max)))
}

fn check_id<T: PartialEq>(recorded: T, actual: T) -> Result<()> {
    ensure!(recorded == actual, "invalid object ID in trace");
    Ok(())
}

#[cfg(any(feature = "cranelift", feature = "winch"))]
pub(super) fn compile_module(engine: &Engine, wasm: &[u8]) -> Result<Module> {
    Module::from_binary(engine, wasm)
}

#[cfg(not(any(feature = "cranelift", feature = "winch")))]
pub(super) fn compile_module(_engine: &Engine, _wasm: &[u8]) -> Result<Module> {
    bail!("replay requires a compiler")
}
