//! Replaying the host's operations on GC objects.

use super::super::*;
use super::Replayer;
use super::init::value;
use crate::runtime::vm::VMGcRef;
use crate::{
    AnyRef, ArrayRef, ArrayRefPre, AsContext, AsContextMut, ExnRef, ExnRefPre, ExnType, ExternRef,
    HeapType, RootScope, StructRef, StructRefPre, Unrooted, Val, ValType,
};
use objects::{RecordedGlobal, RecordedTable, RecordedTag};
use wasmtime_environ::VMGcKind;

fn values(
    store: &mut StoreOpaque,
    body: &mut Reader<'_>,
    types: impl Iterator<Item = ValType>,
) -> Result<Vec<Val>> {
    types.map(|ty| value(store, body, &ty)).collect()
}

/// Decodes a handle.
fn object(store: &StoreOpaque, handle: u32) -> Result<(VMGcRef, VMGcKind)> {
    let object = store.rr_handles().get(handle)?;
    let kind = store.require_gc_store()?.kind(&object)?;
    Ok((object, kind))
}

impl Replayer {
    /// Replays the host's allocation of an object.
    pub(super) fn allocate(&mut self, body: &mut Reader<'_>) -> Result<()> {
        // This may collect garbage.
        self.publish_parked(None)?;
        let mut scope = RootScope::new(&mut self.store);
        let store: &mut StoreOpaque = scope.as_context_mut().0;
        let heap_type = |store: &StoreOpaque, body: &mut Reader<'_>| {
            let ty = RecordedType::from_u32(body.u32()?);
            let ty = store.rr_objects().types.get(ty).cloned();
            ty.ok_or_else(|| format_err!("invalid type ID in trace"))
        };
        let raw = match body.u8()? {
            codec::ALLOC_EXTERN => {
                body.end()?;
                let object = ExternRef::new(&mut scope, ())?;
                object.try_gc_ref(scope.as_context().0)?.as_raw_u32()
            }
            codec::ALLOC_STRUCT => {
                let HeapType::ConcreteStruct(ty) = heap_type(store, body)? else {
                    bail!("invalid struct type in trace");
                };
                let types = ty.fields().map(|f| f.element_type().unpack().clone());
                let fields = values(store, body, types)?;
                body.end()?;
                let pre = StructRefPre::new(&mut scope, ty);
                let object = StructRef::new(&mut scope, &pre, &fields)?;
                object.try_gc_ref(scope.as_context().0)?.as_raw_u32()
            }
            codec::ALLOC_ARRAY => {
                let HeapType::ConcreteArray(ty) = heap_type(store, body)? else {
                    bail!("invalid array type in trace");
                };
                let len = body.u32()?;
                let elem = ty.element_type().unpack().clone();
                let mut elems = Vec::new();
                while body.end().is_err() {
                    elems.push(value(store, body, &elem)?);
                }
                let pre = ArrayRefPre::new(&mut scope, ty);
                let object = if let [elem] = &elems[..] {
                    ArrayRef::new(&mut scope, &pre, elem, len)?
                } else {
                    ensure!(
                        elems.len() == usize::try_from(len)?,
                        "invalid array length in trace"
                    );
                    ArrayRef::new_fixed(&mut scope, &pre, &elems)?
                };
                object.try_gc_ref(scope.as_context().0)?.as_raw_u32()
            }
            codec::ALLOC_BYTES => {
                let HeapType::ConcreteArray(ty) = heap_type(store, body)? else {
                    bail!("invalid array type in trace");
                };
                let pre = ArrayRefPre::new(&mut scope, ty);
                let object = ArrayRef::new_from_i8_slice(&mut scope, &pre, body.rest())?;
                object.try_gc_ref(scope.as_context().0)?.as_raw_u32()
            }
            codec::ALLOC_EXN => {
                let tag = RecordedTag::from_u32(body.u32()?);
                let tag = store.rr_objects().tags.get(tag).copied();
                let tag = tag.ok_or_else(|| format_err!("invalid tag ID in trace"))?;
                let ty = ExnType::from_tag_type(&tag._ty(store))?;
                let types = ty.fields().map(|f| f.element_type().unpack().clone());
                let fields = values(store, body, types)?;
                body.end()?;
                let pre = ExnRefPre::new(&mut scope, ty);
                let object = ExnRef::new(&mut scope, &pre, &tag, &fields)?;
                object.try_gc_ref(scope.as_context().0)?.as_raw_u32()
            }
            _ => bail!("invalid allocation in trace"),
        };
        scope.as_context_mut().0.rr_create_handle(raw)
    }

    /// Replays the host's read of a reference.
    pub(super) fn read_ref(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let kind = body.u8()?;
        let id = body.u32()?;
        let index = body.u64()?;
        body.end()?;
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let objects = store.rr_objects();
        let value = match kind {
            codec::READ_GLOBAL => {
                let global = objects.globals.get(RecordedGlobal::from_u32(id)).copied();
                let global = global.ok_or_else(|| format_err!("invalid global ID in trace"))?;
                global._get(store)
            }
            codec::READ_TABLE => {
                let table = objects.tables.get(RecordedTable::from_u32(id)).copied();
                let table = table.ok_or_else(|| format_err!("invalid table ID in trace"))?;
                let value = table._get_if_initialized(store, index);
                value
                    .ok_or_else(|| format_err!("invalid table index in trace"))?
                    .into()
            }
            codec::READ_OBJECT => {
                let (object, kind) = object(store, id)?;
                let context = self.store.as_context();
                let index = u32::try_from(index)?;
                match kind {
                    VMGcKind::StructRef => Unrooted::<StructRef>::new(context.0, &object)
                        .field(context, usize::try_from(index)?)?,
                    VMGcKind::ArrayRef => {
                        Unrooted::<ArrayRef>::new(context.0, &object).get(context, index)?
                    }
                    VMGcKind::ExnRef => Unrooted::<ExnRef>::new(context.0, &object)
                        .field(context, usize::try_from(index)?)?,
                    _ => bail!("read of an invalid object in trace"),
                }
            }
            _ => bail!("invalid read in trace"),
        };
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let (Kind::GcRef { .. }, raw) = store.rr_val_raw(&value)? else {
            bail!("invalid read in trace");
        };
        store.rr_create_handle(raw.get_anyref())
    }

    /// Replays the host's write to a field or element.
    pub(super) fn object_write(&mut self, body: &mut Reader<'_>) -> Result<()> {
        let handle = body.u32()?;
        let index = body.u32()?;
        let mut scope = RootScope::new(&mut self.store);
        let store: &mut StoreOpaque = scope.as_context_mut().0;
        let (object, kind) = object(store, handle)?;
        let invalid = || format_err!("invalid write to an object in trace");
        ensure!(kind.matches(VMGcKind::AnyRef), invalid());
        let object = Unrooted::<AnyRef>::new(store, &object).to_rooted_(store)?;
        match kind {
            VMGcKind::StructRef => {
                let object = object.unwrap_struct(&scope)?;
                let index = usize::try_from(index)?;
                let store: &mut StoreOpaque = scope.as_context_mut().0;
                let ty = object._ty(store)?.field(index).ok_or_else(invalid)?;
                let value = value(store, body, ty.element_type().unpack())?;
                body.end()?;
                object._set_field(store, index, value)
            }
            VMGcKind::ArrayRef => {
                let object = object.unwrap_array(&scope)?;
                let store: &mut StoreOpaque = scope.as_context_mut().0;
                let ty = object._ty(store)?.element_type();
                let value = value(store, body, ty.unpack())?;
                body.end()?;
                object._set(store, index, value)
            }
            _ => Err(invalid()),
        }
    }
}
