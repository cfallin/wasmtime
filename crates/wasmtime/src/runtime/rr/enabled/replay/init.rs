//! Replaying core object construction. Host functions become stubs, and
//! modules are compiled from the bytecode that the embedder provides.

use super::super::*;
use super::Trampolines;
use crate::runtime::vm;
use crate::{
    ArrayType, Extern, FuncType, Global, GlobalType, HeapType, Instance, MemoryType, Module,
    Mutability, Ref, StoreContextMut, StructType, Table, TableType, Val, ValType,
};
use core::mem::MaybeUninit;
use objects::{RecordedGlobal, RecordedModule, RecordedTable, RecordedType};
use wasmtime_environ::{EntityIndex, Initializer, TypeIndex, VMSharedTypeIndex};

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
            let objects = store.rr_objects();
            let params = Reader::new(body.blob()?).val_types(objects)?;
            let results = Reader::new(body.blob()?).val_types(objects)?;
            body.end()?;
            let ty = FuncType::new(store.engine(), params, results);
            let func = trampolines.host_stub(store, ty)?;
            let mut session = store.rr.session.take().unwrap();
            let added = session.objects.add_func(store, func);
            store.rr.session = Some(session);
            check_id(RecordedFunc::from_u32(id), added?)?;
        }
        codec::TYPE => {
            let module = RecordedModule::from_u32(body.u32()?);
            let index = TypeIndex::from_u32(body.u32()?);
            body.end()?;
            let modules = &store.rr_objects().modules;
            let ty = modules
                .get(module)
                .and_then(|module| super::super::init::module_type(module, index))
                .and_then(|index| heap_type(store.engine(), index))
                .ok_or_else(|| format_err!("invalid type in trace"))?;
            let added = store.rr_session().objects.types.push(ty)?;
            check_id(RecordedType::from_u32(id), added)?;
        }
        codec::GLOBAL => {
            let ty = body.val_type(store.rr_objects())?;
            let mutability = if body.bool()? {
                Mutability::Var
            } else {
                Mutability::Const
            };
            let objects = store.rr_objects();
            check_id(RecordedGlobal::from_u32(id), objects.globals.next_key())?;
            let mut raw = [MaybeUninit::uninit()];
            body.values(core::slice::from_ref(&ty), &mut raw, store)?;
            body.end()?;
            // SAFETY: `values` initialized this.
            let raw = unsafe { raw[0].assume_init() };
            let value = match Kind::new(&ty) {
                Kind::I32 => Val::I32(raw.get_i32()),
                Kind::I64 => Val::I64(raw.get_i64()),
                Kind::F32 => Val::F32(raw.get_f32()),
                Kind::F64 => Val::F64(raw.get_f64()),
                Kind::V128 => Val::V128(raw.get_v128().into()),
                // SAFETY: `values` decoded this store's function.
                Kind::FuncRef { .. } => Val::FuncRef(unsafe {
                    NonNull::new(raw.get_funcref().cast())
                        .map(|f| Func::from_vm_func_ref(store.id(), f))
                }),
                Kind::GcRef { .. } => {
                    ensure!(raw.get_anyref() == 0, "invalid global in trace");
                    ty.default_value()
                        .ok_or_else(|| format_err!("invalid global in trace"))?
                }
                Kind::Unsupported => bail!(codec::UNSUPPORTED),
            };
            Global::new(
                StoreContextMut(store),
                GlobalType::new(ty, mutability),
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
            let ValType::Ref(element) = body.val_type(store.rr_objects())? else {
                bail!("invalid table type in trace");
            };
            let table64 = body.bool()?;
            let (min, max) = limits(body)?;
            let objects = store.rr_objects();
            let init = match objects.decode_ref(body.u32()?)? {
                Some(func) => Ref::Func(Some(func)),
                None => Ref::null(element.heap_type()),
            };
            body.end()?;
            check_id(RecordedTable::from_u32(id), objects.tables.next_key())?;
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

fn heap_type(engine: &Engine, index: VMSharedTypeIndex) -> Option<HeapType> {
    FuncType::from_shared_type_index(engine, index)
        .map(HeapType::ConcreteFunc)
        .or_else(|| StructType::from_shared_type_index(engine, index).map(HeapType::ConcreteStruct))
        .or_else(|| ArrayType::from_shared_type_index(engine, index).map(HeapType::ConcreteArray))
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
