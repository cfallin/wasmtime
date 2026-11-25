//! Recording the host's operations on GC objects, so that it can pass any
//! object to the guest.
//!
//! The objects that the host allocates have handles from the beginning. Reading
//! a reference out of the guest's state needs only shared access to the store,
//! and gives an unrooted reference, which the host can do nothing with but read
//! further until it roots it. So the recorder notes where references were read
//! from, and records the read if the reference is rooted.

use super::*;
use crate::rr::{Allocation, GcSource};
use crate::runtime::epoch_state::EpochStateHandle;
use crate::runtime::vm::{GcHeap, VMGcRef};
use crate::{GcRef, HeapType, Rooted, Val};
use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use wasmtime_environ::VMGcKind;

type HeapVersion = EpochStateHandle<Box<dyn GcHeap>>;

/// Where the host read references from, by their raw values, in the current
/// version of the GC heap.
#[derive(Default)]
pub(super) struct Reads {
    heap: Option<HeapVersion>,
    sources: BTreeMap<u32, GcSource>,
}

impl StoreOpaque {
    /// Notes that the host read `gc_ref` from `source`.
    pub(crate) fn rr_note_read(&self, gc_ref: Option<&VMGcRef>, source: GcSource) {
        let Some(gc_ref) = gc_ref.filter(|r| !r.is_i31() && self.rr.recording()) else {
            return;
        };
        let heap = self.unwrap_gc_store().gc_heap.handle();
        let mut reads = self.rr.session.as_ref().unwrap().reads.write();
        if reads.heap != Some(heap) {
            reads.heap = Some(heap);
            reads.sources.clear();
        }
        // Keep the first source, so that sources do not form cycles.
        reads.sources.entry(gc_ref.as_raw_u32()).or_insert(source);
    }

    /// Creates a handle for an object that the host is rooting, if it read the
    /// reference in version `heap` of the GC heap.
    pub(crate) fn rr_rooting(&mut self, raw: u32, heap: HeapVersion) {
        if !self.rr.recording() {
            return;
        }
        let mut reads = core::mem::take(&mut *self.rr_session().reads.write());
        if reads.heap == Some(heap) {
            let result = self.rr_record_read(raw, &mut reads.sources);
            let _ = self.rr_poison_on_err(result);
        }
        *self.rr_session().reads.write() = reads;
    }

    fn rr_record_read(&mut self, raw: u32, sources: &mut BTreeMap<u32, GcSource>) -> Result<()> {
        // Objects that lead to this one need handles first.
        let mut path = vec![raw];
        while let Some(raw) = path.last() {
            let source = match sources.entry(*raw) {
                Entry::Occupied(entry) if self.rr_handles().find(*raw).is_none() => entry.remove(),
                _ => {
                    path.pop();
                    continue;
                }
            };
            let raw = *raw;
            let mut body = TryVec::new();
            let (kind, id, index) = match source {
                GcSource::Global(global) => {
                    let id = self.rr_objects().find_global(self, global);
                    let id = id.ok_or_else(|| format_err!("read of an unregistered global"))?;
                    (codec::READ_GLOBAL, id.as_u32(), 0)
                }
                GcSource::Table(table, index) => {
                    let id = self.rr_objects().find_table(self, table);
                    let id = id.ok_or_else(|| format_err!("read of an unregistered table"))?;
                    (codec::READ_TABLE, id.as_u32(), index)
                }
                GcSource::Object { object, offset } => {
                    let Some(id) = self.rr_handles().find(object) else {
                        if !sources.contains_key(&object) {
                            // The host got the object in some way that is not
                            // recorded, so it can't be passed to the guest.
                            path.pop();
                            continue;
                        }
                        sources.insert(raw, GcSource::Object { object, offset });
                        path.push(object);
                        continue;
                    };
                    let index = self.rr_field_index(object, offset)?;
                    (codec::READ_OBJECT, id, index)
                }
            };
            body.push(kind)?;
            body.extend_from_slice(&id.to_le_bytes())?;
            body.extend_from_slice(&index.to_le_bytes())?;
            self.rr_create_handle(raw)?;
            self.rr_session().append(codec::READ, &body)?;
            path.pop();
        }
        Ok(())
    }

    /// The index of the field or element at `offset`, which depends on the
    /// collector, in an object.
    fn rr_field_index(&self, raw: u32, offset: u32) -> Result<u64> {
        let gc_ref = VMGcRef::from_raw_u32(raw).ok_or_else(|| format_err!("read of null"))?;
        let header = self.require_gc_store()?.header(&gc_ref)?;
        let ty = header
            .ty()
            .ok_or_else(|| format_err!("object without a type"))?;
        if header.matches_kind(VMGcKind::ArrayRef) {
            let layout = crate::runtime::gc::gc_array_layout(self.engine(), ty)?;
            return Ok(u64::from((offset - layout.base_size) / layout.elem_size));
        }
        let layout = crate::runtime::gc::gc_struct_layout(self.engine(), ty)?;
        let index = layout.fields.iter().position(|f| f.offset == offset);
        let index = index.ok_or_else(|| format_err!("read of an invalid field"))?;
        Ok(u64::try_from(index)?)
    }

    /// Records the host's allocation of `object`.
    pub(crate) fn rr_allocated<T: GcRef>(
        &mut self,
        object: &Rooted<T>,
        allocation: Allocation<'_>,
    ) -> Result<()> {
        if !self.rr.recording() {
            return Ok(());
        }
        let result = (|| {
            self.rr_check()?;
            let mut body = TryVec::new();
            match allocation {
                Allocation::Extern => body.push(codec::ALLOC_EXTERN)?,
                Allocation::Struct(ty, fields) => {
                    let ty = self.rr_type(HeapType::ConcreteStruct(ty.clone()))?;
                    body.push(codec::ALLOC_STRUCT)?;
                    body.extend_from_slice(&ty.as_u32().to_le_bytes())?;
                    self.rr_encode_vals(&mut body, fields)?;
                }
                Allocation::Array(ty, len, elems) => {
                    let ty = self.rr_type(HeapType::ConcreteArray(ty.clone()))?;
                    body.push(codec::ALLOC_ARRAY)?;
                    body.extend_from_slice(&ty.as_u32().to_le_bytes())?;
                    body.extend_from_slice(&len.to_le_bytes())?;
                    self.rr_encode_vals(&mut body, elems)?;
                }
                Allocation::Bytes(ty, bytes) => {
                    let ty = self.rr_type(HeapType::ConcreteArray(ty.clone()))?;
                    body.push(codec::ALLOC_BYTES)?;
                    body.extend_from_slice(&ty.as_u32().to_le_bytes())?;
                    body.extend_from_slice(bytes)?;
                }
                Allocation::Exn(tag, fields) => {
                    let tag = self.rr_objects().find_tag(tag);
                    let tag = tag.ok_or_else(|| format_err!("unregistered tag"))?;
                    body.push(codec::ALLOC_EXN)?;
                    body.extend_from_slice(&tag.as_u32().to_le_bytes())?;
                    self.rr_encode_vals(&mut body, fields)?;
                }
            }
            let raw = object.try_gc_ref(self)?.as_raw_u32();
            self.rr_create_handle(raw)?;
            self.rr_session().append(codec::ALLOC, &body)
        })();
        self.rr_poison_on_err(result)
    }

    /// Records the host's write to a field or element of the object whose raw
    /// reference is `object`.
    pub(crate) fn rr_object_written(&mut self, object: u32, index: u32, value: &Val) -> Result<()> {
        self.rr_mutated(codec::OBJECT_WRITE, |store, body| {
            let handle = store.rr_handles().find(object);
            let handle = handle.ok_or_else(|| format_err!(handles::NO_HANDLE))?;
            body.extend_from_slice(&handle.to_le_bytes())?;
            body.extend_from_slice(&index.to_le_bytes())?;
            store.rr_encode_vals(body, core::slice::from_ref(value))
        })
    }

    fn rr_type(&mut self, ty: HeapType) -> Result<RecordedType> {
        self.rr_register(|_, session| session.record_heap_type(&ty))?;
        self.rr_objects().find_type(&ty)
    }
}
