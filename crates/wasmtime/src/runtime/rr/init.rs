//! Core object construction. Components are flattened into the core modules
//! they instantiate; host and component imports become typed replay stubs.
//! Only validated Wasm bytecode is loaded from the trace, never native code.

use super::*;
use crate::runtime::vm;
use crate::{Extern, Global, Instance, Module, StoreContextMut};
use wasmtime_environ::packed_option::ReservedValue;
use wasmtime_environ::{DefinedGlobalIndex, DefinedMemoryIndex, DefinedTableIndex, EntityIndex};

impl Session {
    fn import_func(&mut self, store: &StoreOpaque, func: Func) -> Result<usize> {
        if let Ok(id) = self.objects.find_func(func.vm_func_ref(store)) {
            return Ok(id);
        }
        let id = self.objects.add_func(store, func)?;
        let f = &self.objects.funcs[id];
        ensure!(f.host, "unregistered guest function in instance imports");
        let mut body = u32::try_from(id)?.to_le_bytes().to_vec();
        codec::blob(
            &mut body,
            &f.params.iter().map(|k| *k as u8).collect::<Vec<_>>(),
        )?;
        codec::blob(
            &mut body,
            &f.results.iter().map(|k| *k as u8).collect::<Vec<_>>(),
        )?;
        self.append(codec::HOST, &body)?;
        Ok(id)
    }

    fn import_global(&mut self, store: &mut StoreOpaque, global: Global) -> Result<usize> {
        if let Some(id) = self.objects.globals_by_key.get(&global.rr_key(store)) {
            return Ok(*id);
        }
        let ty = global._ty(store);
        let value = global.rr_read(store);
        let id = self.objects.globals.len();
        let mut body = u32::try_from(id)?.to_le_bytes().to_vec();
        let mutable = u8::from(ty.mutability() == crate::Mutability::Var);
        if let crate::ValType::Ref(reference) = ty.content() {
            ensure!(
                reference.heap_type().is_func(),
                "record/replay requires an abstract function global"
            );
            let crate::Val::FuncRef(func) = value else {
                unreachable!()
            };
            let func = match func {
                None => 0,
                Some(func) => 1 + self.import_func(store, func)?,
            };
            body.extend_from_slice(&[codec::FUNCREF, mutable, u8::from(reference.is_nullable())]);
            body.extend_from_slice(&u32::try_from(func)?.to_le_bytes());
        } else {
            let kind = Kind::new(ty.content().clone())?;
            let raw = match value {
                crate::Val::I32(v) => ValRaw::i32(v),
                crate::Val::I64(v) => ValRaw::i64(v),
                crate::Val::F32(v) => ValRaw::f32(v),
                crate::Val::F64(v) => ValRaw::f64(v),
                crate::Val::V128(v) => ValRaw::v128(v.as_u128()),
                _ => unreachable!(),
            };
            body.extend_from_slice(&[kind as u8, mutable]);
            codec::reserve(&mut body, kind.size())?;
            // SAFETY: kind and raw both came from the global's checked type.
            unsafe {
                codec::values(&mut body, &[kind], &raw, |_| unreachable!())?;
            }
        }
        self.append(codec::GLOBAL, &body)?;
        self.objects
            .globals_by_key
            .insert(global.rr_key(store), id)?;
        self.objects.globals.push(global);
        if global.rr_is_component_flag() {
            self.objects.flags.push(id);
        }
        Ok(id)
    }
}

impl StoreOpaque {
    // Detach metadata while enumerating runtime objects; no guest code or
    // embedder callbacks run in these closures. Always restore it on error.
    fn rr_register(&mut self, f: impl FnOnce(&mut Self, &mut Session) -> Result<()>) -> Result<()> {
        if !self.rr.active() {
            return Ok(());
        }
        self.rr_flush()?;
        let mut session = self.rr.session.take().unwrap();
        let result = f(self, &mut session);
        if result.is_err() {
            session.fail(format_err!("failed to record object construction"));
        }
        self.rr.session = Some(session);
        result
    }

    /// Register a reference supplied by the host before recording its value.
    pub(super) fn rr_register_func_reference(&mut self, raw: NonNull<VMFuncRef>) -> Result<()> {
        if self
            .rr
            .session
            .as_ref()
            .unwrap()
            .objects
            .find_func(raw)
            .is_ok()
        {
            return Ok(());
        }
        self.rr_register(|store, session| {
            // SAFETY: the boundary's typed slot contains a store-rooted reference.
            let func = unsafe { Func::from_vm_func_ref(store.id(), raw) };
            session.import_func(store, func)?;
            Ok(())
        })
    }

    pub(crate) fn rr_created_global(&mut self, global: Global) -> Result<()> {
        self.rr_register(|store, session| {
            session.import_global(store, global)?;
            Ok(())
        })
    }

    pub(crate) fn rr_created_memory(&mut self, memory: Memory) -> Result<()> {
        self.rr_register(|store, session| {
            let ty = memory.wasmtime_ty(store);
            let id = session.objects.memories.len();
            let mut body = u32::try_from(id)?.to_le_bytes().to_vec();
            body.extend_from_slice(&[
                u8::from(ty.idx_type == wasmtime_environ::IndexType::I64),
                ty.page_size_log2,
            ]);
            body.extend_from_slice(&ty.limits.min.to_le_bytes());
            body.push(u8::from(ty.limits.max.is_some()));
            body.extend_from_slice(&ty.limits.max.unwrap_or_default().to_le_bytes());
            session.append(codec::MEMORY, &body)?;
            session
                .objects
                .memories_by_key
                .insert(memory.rr_key(store), id)?;
            session.objects.memories.push(memory);
            Ok(())
        })
    }

    pub(crate) fn rr_created_table(&mut self, table: crate::Table, init: crate::Ref) -> Result<()> {
        self.rr_register(|store, session| {
            let ty = table.ty_(store);
            ensure!(
                ty.element().heap_type().is_func(),
                "record/replay requires a function table"
            );
            let func = match init {
                crate::Ref::Func(None) => 0,
                crate::Ref::Func(Some(func)) => 1 + session.import_func(store, func)?,
                _ => bail!("unsupported initial table element"),
            };
            let id = session.objects.tables.len();
            let mut body = u32::try_from(id)?.to_le_bytes().to_vec();
            body.extend_from_slice(&[u8::from(ty.is_64()), u8::from(ty.element().is_nullable())]);
            body.extend_from_slice(&ty.minimum().to_le_bytes());
            body.push(u8::from(ty.maximum().is_some()));
            body.extend_from_slice(&ty.maximum().unwrap_or_default().to_le_bytes());
            body.extend_from_slice(&u32::try_from(func)?.to_le_bytes());
            session.append(codec::TABLE, &body)?;
            session.objects.tables_by_key.insert(table.rr_key(), id)?;
            session.objects.tables.push(table);
            Ok(())
        })
    }

    pub(crate) fn rr_created_instance(
        &mut self,
        instance: Instance,
        module: &Module,
    ) -> Result<()> {
        self.rr_register(|store, session| {
            if matches!(session.mode, Mode::Recording { .. }) {
                let wasm = module.debug_bytecode().ok_or_else(|| {
                    format_err!("module has no retained bytecode for initialization replay")
                })?;
                let module_id = match session
                    .objects
                    .modules
                    .iter()
                    .position(|m| Module::same(m, module))
                {
                    Some(id) => id,
                    None => {
                        let id = session.objects.modules.len();
                        let mut body = u32::try_from(id)?.to_le_bytes().to_vec();
                        codec::blob(&mut body, wasm)?;
                        session.append(codec::MODULE, &body)?;
                        session.objects.modules.push(module.clone());
                        id
                    }
                };
                let mut body = u32::try_from(module_id)?.to_le_bytes().to_vec();
                for initializer in &module.env_module().initializers {
                    let wasmtime_environ::Initializer::Import { index: ty, .. } = *initializer;
                    let object = match ty {
                        EntityIndex::Function(index) => {
                            let store_id = store.id();
                            let (inst, registry) = instance.id.get_mut_and_module_registry(store);
                            // SAFETY: the new instance and its imports belong to store.
                            let func = unsafe { inst.get_exported_func(registry, store_id, index) };
                            session.import_func(store, func)?
                        }
                        EntityIndex::Memory(index) => {
                            let memory = store[instance.id]
                                .get_exported_memory(store.id(), index)
                                .unshared()
                                .ok_or_else(|| format_err!("shared memory import in recording"))?;
                            *session
                                .objects
                                .memories_by_key
                                .get(&memory.rr_key(store))
                                .ok_or_else(|| format_err!("unregistered imported memory"))?
                        }
                        EntityIndex::Table(index) => {
                            let table = store[instance.id].get_exported_table(store.id(), index);
                            *session
                                .objects
                                .tables_by_key
                                .get(&table.rr_key())
                                .ok_or_else(|| format_err!("unregistered imported table"))?
                        }
                        EntityIndex::Global(index) => {
                            let global = store[instance.id].get_exported_global(store.id(), index);
                            session.import_global(store, global)?
                        }
                        _ => bail!("unsupported record/replay import"),
                    };
                    body.extend_from_slice(&u32::try_from(object)?.to_le_bytes());
                }
                session.append(codec::INSTANCE, &body)?;
            }
            session.objects.register_instance(store, instance)?;
            Ok(())
        })
    }

    /// Component instance flags are ordinary imported i32 globals. Host-side
    /// canonical-call bookkeeping can update them between any two crossings.
    /// Recording their values at those crossings also covers generated adapter
    /// code without adding a component-specific event to the protocol.
    pub(super) fn rr_flush_flags(&mut self) -> Result<()> {
        let session = self.rr.session.as_ref().unwrap();
        let count = session.objects.flags.len();
        for i in 0..count {
            let objects = &self.rr.session.as_ref().unwrap().objects;
            let id = objects.flags[i];
            let global = objects.globals[id];
            let value = global.rr_read(self).unwrap_i32();
            let mut body = [0; 8];
            body[..4].copy_from_slice(&u32::try_from(id)?.to_le_bytes());
            body[4..].copy_from_slice(&value.to_le_bytes());
            self.rr
                .session
                .as_mut()
                .unwrap()
                .append(codec::GLOBAL_WRITE, &body)?;
        }
        Ok(())
    }
}

impl Objects {
    pub(super) fn importable_func(&self, id: usize) -> Result<Func> {
        let func = self
            .funcs
            .get(id)
            .ok_or_else(|| format_err!("invalid function id"))?;
        ensure!(
            !func.startup,
            "instance startup cannot be used as a function reference"
        );
        Ok(func.func)
    }

    fn register_instance(&mut self, store: &mut StoreOpaque, instance: Instance) -> Result<()> {
        let id = instance.id;
        let functions = store[id]
            .env_module()
            .functions
            .iter()
            .filter(|(i, f)| {
                !store[id].env_module().is_imported_function(*i) && !f.func_ref.is_reserved_value()
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        for index in functions {
            let store_id = store.id();
            let (inst, registry) = id.get_mut_and_module_registry(store);
            // SAFETY: this instance belongs to store.
            let func = unsafe { inst.get_exported_func(registry, store_id, index) };
            self.add_func(store, func)?;
        }
        let store_id = store.id();
        let (inst, registry) = id.get_mut_and_module_registry(store);
        // SAFETY: this instance belongs to store.
        if let Some(startup) = unsafe { inst.get_startup_func(registry, store_id) } {
            let id = self.add_func(store, startup)?;
            self.funcs[id].startup = true;
        }
        for i in 0..store[id].env_module().num_defined_memories() {
            // SAFETY: shared memories have been rejected before allocation.
            let memory =
                unsafe { Memory::from_raw(id, DefinedMemoryIndex::from_u32(u32::try_from(i)?)) };
            self.memories_by_key
                .insert(memory.rr_key(store), self.memories.len())?;
            self.memories.push(memory);
        }
        for i in 0..store[id].env_module().num_defined_tables() {
            let table = crate::Table::from_raw(id, DefinedTableIndex::from_u32(u32::try_from(i)?));
            self.tables_by_key
                .insert(table.rr_key(), self.tables.len())?;
            self.tables.push(table);
        }
        for i in 0..store[id].env_module().num_defined_globals() {
            let global = Global::new_instance(
                store,
                id.instance(),
                DefinedGlobalIndex::from_u32(u32::try_from(i)?),
            );
            self.globals_by_key
                .insert(global.rr_key(store), self.globals.len())?;
            self.globals.push(global);
        }
        Ok(())
    }
}

pub(super) fn replay_event<T: 'static>(
    store: &mut StoreInner<T>,
    trampolines: &super::replay::Trampolines,
    tag: u8,
    body: &mut Reader<'_>,
) -> Result<Option<(Instance, Option<usize>)>> {
    match tag {
        codec::HOST => {
            let id = usize::try_from(body.u32()?)?;
            let params = body
                .blob()?
                .iter()
                .map(|v| Kind::read(*v).map(Kind::ty))
                .collect::<Result<Vec<_>>>()?;
            let results = body
                .blob()?
                .iter()
                .map(|v| Kind::read(*v).map(Kind::ty))
                .collect::<Result<Vec<_>>>()?;
            body.end()?;
            ensure!(
                id == store.rr.session.as_ref().unwrap().objects.funcs.len(),
                "invalid host function id"
            );
            let ty = crate::FuncType::new(store.engine(), params, results);
            let func = trampolines.host_stub(store, ty)?;
            let mut session = store.rr.session.take().unwrap();
            let added = session.objects.add_func(store, func);
            store.rr.session = Some(session);
            ensure!(added? == id, "duplicate replay host");
        }
        codec::MODULE => {
            let id = usize::try_from(body.u32()?)?;
            let wasm = body.blob()?;
            body.end()?;
            ensure!(
                id == store.rr.session.as_ref().unwrap().objects.modules.len(),
                "invalid module id"
            );
            let module = compile_module(store.engine(), wasm)?;
            store
                .rr
                .session
                .as_mut()
                .unwrap()
                .objects
                .modules
                .push(module);
        }
        codec::INSTANCE => {
            let id = usize::try_from(body.u32()?)?;
            let objects = &store.rr.session.as_ref().unwrap().objects;
            let module = objects
                .modules
                .get(id)
                .ok_or_else(|| format_err!("invalid instance module"))?
                .clone();
            let mut imports = Vec::new();
            for initializer in &module.env_module().initializers {
                let wasmtime_environ::Initializer::Import { index: ty, .. } = *initializer;
                let id = usize::try_from(body.u32()?)?;
                imports.push(match ty {
                    EntityIndex::Function(_) => Extern::Func(objects.importable_func(id)?),
                    EntityIndex::Memory(_) => Extern::Memory(
                        *objects
                            .memories
                            .get(id)
                            .ok_or_else(|| format_err!("invalid memory import"))?,
                    ),
                    EntityIndex::Table(_) => Extern::Table(
                        *objects
                            .tables
                            .get(id)
                            .ok_or_else(|| format_err!("invalid table import"))?,
                    ),
                    EntityIndex::Global(_) => Extern::Global(
                        *objects
                            .globals
                            .get(id)
                            .ok_or_else(|| format_err!("invalid global import"))?,
                    ),
                    _ => bail!("unsupported replay import"),
                });
            }
            body.end()?;
            let imports = Instance::typecheck_externs(store, &module, &imports)?;
            // SAFETY: imports were checked against module. Allocation has no
            // async limiter; startup is a separate recorded guest activation.
            let (instance, _) = vm::assert_ready(unsafe {
                Instance::new_raw(store, None, &module, imports.as_ref())
            })?;
            let store_id = store.id();
            let (inst, registry) = instance.id.get_mut_and_module_registry(store);
            // SAFETY: the new instance belongs to this store.
            let startup = unsafe { inst.get_startup_func(registry, store_id) }
                .map(|func| {
                    store
                        .rr
                        .session
                        .as_ref()
                        .unwrap()
                        .objects
                        .find_func(func.vm_func_ref(store))
                })
                .transpose()?;
            return Ok(Some((instance, startup)));
        }
        codec::GLOBAL => {
            let id = usize::try_from(body.u32()?)?;
            let kind = body.u8()?;
            let mutable = body.u8()?;
            ensure!(mutable <= 1, "invalid global mutability");
            let (ty, value) = if kind == codec::FUNCREF {
                let nullable = body.u8()?;
                ensure!(nullable <= 1, "invalid global nullability");
                let func = usize::try_from(body.u32()?)?;
                let objects = &store.rr.session.as_ref().unwrap().objects;
                let func = if func == 0 {
                    None
                } else {
                    Some(objects.importable_func(func - 1)?)
                };
                (
                    crate::ValType::Ref(crate::RefType::new(nullable == 1, crate::HeapType::Func)),
                    crate::Val::FuncRef(func),
                )
            } else {
                let kind = Kind::read(kind)?;
                let mut raw = [ValRaw::v128(0)];
                body.values(&[kind], &mut raw, |_| unreachable!())?;
                let value = match kind {
                    Kind::I32 => crate::Val::I32(raw[0].get_i32()),
                    Kind::I64 => crate::Val::I64(raw[0].get_i64()),
                    Kind::F32 => crate::Val::F32(raw[0].get_f32()),
                    Kind::F64 => crate::Val::F64(raw[0].get_f64()),
                    Kind::V128 => crate::Val::V128(raw[0].get_v128().into()),
                    Kind::FuncRef => unreachable!("reference globals are decoded separately"),
                };
                (kind.ty(), value)
            };
            body.end()?;
            let ty = crate::GlobalType::new(
                ty,
                if mutable == 1 {
                    crate::Mutability::Var
                } else {
                    crate::Mutability::Const
                },
            );
            ensure!(
                id == store.rr.session.as_ref().unwrap().objects.globals.len(),
                "invalid global id"
            );
            Global::new(StoreContextMut(&mut *store), ty, value)?;
        }
        codec::MEMORY | codec::TABLE => {
            let id = usize::try_from(body.u32()?)?;
            let is64 = body.u8()?;
            let flag = body.u8()?;
            let min = body.u64()?;
            let has_max = body.u8()?;
            ensure!(has_max <= 1, "invalid object maximum");
            let max = body.u64()?;
            let max = (has_max == 1).then_some(max);
            ensure!(is64 <= 1, "invalid object index type");
            if tag == codec::MEMORY {
                body.end()?;
                ensure!(
                    id == store.rr.session.as_ref().unwrap().objects.memories.len(),
                    "invalid memory id"
                );
                let ty = crate::MemoryType::builder()
                    .memory64(is64 == 1)
                    .min(min)
                    .max(max)
                    .page_size_log2(flag)
                    .build()?;
                Memory::new(StoreContextMut(&mut *store), ty)?;
            } else {
                ensure!(flag <= 1, "invalid table nullability");
                let func = usize::try_from(body.u32()?)?;
                body.end()?;
                let objects = &store.rr.session.as_ref().unwrap().objects;
                ensure!(id == objects.tables.len(), "invalid table id");
                let init = crate::Ref::Func(if func == 0 {
                    None
                } else {
                    Some(objects.importable_func(func - 1)?)
                });
                let element = crate::RefType::new(flag == 1, crate::HeapType::Func);
                let ty = if is64 == 1 {
                    crate::TableType::new64(element, min, max)
                } else {
                    crate::TableType::new(
                        element,
                        u32::try_from(min)?,
                        max.map(u32::try_from).transpose()?,
                    )
                };
                crate::Table::new(StoreContextMut(&mut *store), ty, init)?;
            }
        }
        codec::GLOBAL_WRITE => {
            let id = usize::try_from(body.u32()?)?;
            let value = body.u32()? as i32;
            body.end()?;
            let global = *store
                .rr
                .session
                .as_ref()
                .unwrap()
                .objects
                .globals
                .get(id)
                .ok_or_else(|| format_err!("invalid global write"))?;
            global._set(store, crate::Val::I32(value))?;
        }
        _ => unreachable!(),
    }
    Ok(None)
}

fn compile_module(engine: &crate::Engine, wasm: &[u8]) -> Result<Module> {
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    {
        Module::from_binary(engine, wasm)
    }
    #[cfg(not(any(feature = "cranelift", feature = "winch")))]
    {
        let _ = (engine, wasm);
        bail!("initialization replay requires a compiler")
    }
}
