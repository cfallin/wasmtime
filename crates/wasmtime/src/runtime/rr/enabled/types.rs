//! Recording concrete types.
//!
//! The identity of a type depends on its whole recursion group, so that is the
//! unit in which types are recorded, in the encoding of a `rectype` in the Wasm
//! binary format. The trace's types are the index space of types, so together
//! these records are a type section.

use super::*;
use crate::type_registry::RegisteredType;
use crate::{ContType, HeapType, ValType};
use alloc::collections::BTreeSet;
use wasmtime_environ::{
    EngineOrModuleTypeIndex, TypeTrace, VMSharedTypeIndex, WasmCompositeInnerType, WasmFieldType,
    WasmStorageType, WasmSubType,
};

impl Session {
    /// Records the definition of a concrete type that a value type refers to.
    pub(super) fn record_type(&mut self, ty: &ValType) -> Result<()> {
        match ty {
            ValType::Ref(ty) => self.record_heap_type(ty.heap_type()),
            _ => Ok(()),
        }
    }

    /// Records the recursion group of a concrete type, and all those that it
    /// refers to, if they have not been recorded.
    pub(super) fn record_heap_type(&mut self, ty: &HeapType) -> Result<()> {
        // Replay adds all types when it reaches the first.
        let (Some(ty), Mode::Recording(_)) = (ty.as_registered_type(), &self.mode) else {
            return Ok(());
        };
        // The guest chooses how deeply types refer to each other.
        let mut pending = vec![ty.index()];
        while let Some(&index) = pending.last() {
            if self.objects.has_type(index) {
                pending.pop();
                continue;
            }
            // Types keep those that they refer to alive.
            let ty = RegisteredType::try_root(ty.engine(), index)
                .ok_or_else(|| format_err!("unregistered type"))?;
            let group = ty.rec_group_types();
            let members = group
                .iter()
                .map(|(index, _)| *index)
                .collect::<BTreeSet<_>>();
            let pending_before = pending.len();
            for (_, ty) in &group {
                ty.trace(&mut |index| {
                    if let EngineOrModuleTypeIndex::Engine(index) = index
                        && !members.contains(&index)
                        && !self.objects.has_type(index)
                    {
                        pending.push(index);
                    }
                    Ok::<_, Error>(())
                })?;
            }
            // Groups come after those that they refer to.
            if pending.len() == pending_before {
                self.record_rec_group(ty.engine(), &group)?;
                pending.pop();
            }
        }
        Ok(())
    }

    fn record_rec_group(
        &mut self,
        engine: &Engine,
        group: &[(VMSharedTypeIndex, Arc<WasmSubType>)],
    ) -> Result<()> {
        for (index, _) in group {
            self.objects.add_type(heap_type(engine, *index)?)?;
        }
        let mut body = TryVec::new();
        if group.len() != 1 {
            body.push(0x4e)?;
            codec::leb128(&mut body, u32::try_from(group.len())?)?;
        }
        for (_, ty) in group {
            self.objects.encode_sub_type(&mut body, ty)?;
        }
        self.append(codec::TYPE, &body)
    }
}

impl Objects {
    /// Appends a type in the encoding of a `subtype` in the Wasm binary format.
    fn encode_sub_type(&self, bytes: &mut TryVec<u8>, ty: &WasmSubType) -> Result<()> {
        ensure!(
            !ty.composite_type.shared,
            "record/replay does not support shared types"
        );
        if !ty.is_final || ty.supertype.is_some() {
            bytes.push(if ty.is_final { 0x4f } else { 0x50 })?;
            match ty.supertype {
                Some(supertype) => {
                    bytes.push(1)?;
                    codec::leb128(bytes, self.type_id(supertype)?.as_u32())?;
                }
                None => bytes.push(0)?,
            }
        }
        match &ty.composite_type.inner {
            WasmCompositeInnerType::Func(ty) => {
                bytes.push(0x60)?;
                for types in [ty.params(), ty.results()] {
                    codec::leb128(bytes, u32::try_from(types.len())?)?;
                    for ty in types {
                        codec::wasm_val_type(bytes, ty, self)?;
                    }
                }
            }
            WasmCompositeInnerType::Struct(ty) => {
                bytes.push(0x5f)?;
                codec::leb128(bytes, u32::try_from(ty.fields.len())?)?;
                for field in &ty.fields {
                    self.encode_field_type(bytes, field)?;
                }
            }
            WasmCompositeInnerType::Array(ty) => {
                bytes.push(0x5e)?;
                self.encode_field_type(bytes, &ty.0)?;
            }
            WasmCompositeInnerType::Cont(ty) => {
                bytes.push(0x5d)?;
                codec::signed_leb128(bytes, self.type_id(ty.type_index())?.as_u32())?;
            }
            // Wasm has no such types.
            WasmCompositeInnerType::Exn(_) => bail!("exception type in recursion group"),
        }
        Ok(())
    }

    fn encode_field_type(&self, bytes: &mut TryVec<u8>, ty: &WasmFieldType) -> Result<()> {
        match &ty.element_type {
            WasmStorageType::I8 => bytes.push(0x78)?,
            WasmStorageType::I16 => bytes.push(0x77)?,
            WasmStorageType::Val(ty) => codec::wasm_val_type(bytes, ty, self)?,
        }
        bytes.push(u8::from(ty.mutable))?;
        Ok(())
    }
}

fn heap_type(engine: &Engine, index: VMSharedTypeIndex) -> Result<HeapType> {
    use crate::{ArrayType, FuncType, StructType};
    FuncType::from_shared_type_index(engine, index)
        .map(HeapType::ConcreteFunc)
        .or_else(|| StructType::from_shared_type_index(engine, index).map(HeapType::ConcreteStruct))
        .or_else(|| ArrayType::from_shared_type_index(engine, index).map(HeapType::ConcreteArray))
        .or_else(|| ContType::from_shared_type_index(engine, index).map(HeapType::ConcreteCont))
        .ok_or_else(|| format_err!("unsupported type in record/replay"))
}

/// Reads all of the types in a trace.
#[cfg(any(feature = "cranelift", feature = "winch"))]
pub(super) fn read_types(engine: &Engine, trace: &[u8]) -> Result<Vec<HeapType>> {
    use wasmtime_environ::{ModuleEnvironment, ModuleTypesBuilder, StaticModuleIndex};

    let mut types = TryVec::new();
    let mut count = 0;
    let mut reader = Reader::at(trace, codec::MAGIC.len());
    // If the trace is malformed then replay fails when it gets there.
    while let Ok((tag, mut body)) = reader.record() {
        if tag == codec::TYPE {
            types.extend_from_slice(body.rest())?;
            count += 1;
        }
    }
    let mut section = TryVec::new();
    codec::leb128(&mut section, count)?;
    section.extend_from_slice(&types)?;
    let mut wasm = TryVec::new();
    wasm.extend_from_slice(b"\0asm\x01\0\0\0\x01")?;
    codec::leb128(&mut wasm, u32::try_from(section.len())?)?;
    wasm.extend_from_slice(&section)?;

    // This is a module, so it can be validated and its types registered as any
    // other's are. No code needs to be compiled for it.
    //
    // The host can create non-final function types regardless of the engine's
    // features.
    let features = engine.features() | wasmparser::WasmFeatures::GC;
    let mut parser = wasmparser::Parser::new(0);
    parser.set_features(features);
    let mut validator = wasmparser::Validator::new_with_features(features);
    let mut types = ModuleTypesBuilder::new(&validator);
    let module = StaticModuleIndex::from_u32(0);
    let mut module = ModuleEnvironment::new(engine.tunables(), &mut validator, &mut types, module)
        .translate(parser, &wasm)
        .context("invalid types in trace")?
        .module;
    let mut types = types.finish();
    ensure!(
        engine.gc_runtime().is_some() || types.wasm_types().all(|(_, ty)| ty.is_func()),
        "invalid types in trace"
    );
    // This keeps the types registered until the results do.
    let registered =
        engine.register_and_canonicalize_types(&mut types, core::iter::once(&mut module))?;
    let types = module.types.values().map(|ty| {
        let ty = match *ty {
            EngineOrModuleTypeIndex::Engine(index) => Some(index),
            EngineOrModuleTypeIndex::Module(index) => registered.shared_type(index),
            EngineOrModuleTypeIndex::RecGroup(_) => None,
        };
        heap_type(engine, ty.ok_or_else(|| format_err!("unregistered type"))?)
    });
    types.collect()
}

#[cfg(not(any(feature = "cranelift", feature = "winch")))]
pub(super) fn read_types(_engine: &Engine, _trace: &[u8]) -> Result<Vec<HeapType>> {
    bail!("replay requires a compiler")
}
