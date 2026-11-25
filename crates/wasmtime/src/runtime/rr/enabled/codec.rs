//! Versioned wire format. All lengths and numbers are little endian.
//!
//! Each record has a one-byte tag and a four-byte payload length. The format is
//! versioned independently of Wasmtime: changing it requires changing MAGIC.

use super::{Handles, Objects, RecordedCall};
use crate::prelude::*;
use crate::store::StoreOpaque;
use crate::{HeapTopType, HeapType, RefType, Trap, ValRaw, ValType};
use core::mem::MaybeUninit;
use wasmtime_environ::{WasmHeapType, WasmValType};

pub(super) const MAGIC: &[u8; 8] = b"WTRR\0\0\0\x12";
pub(super) const END: u8 = 0;
pub(super) const ENTER_WASM: u8 = 1;
pub(super) const LEAVE_WASM: u8 = 2;
pub(super) const ENTER_HOST: u8 = 3;
pub(super) const LEAVE_HOST: u8 = 4;
pub(super) const WRITE: u8 = 5;
/// A memory's growth by the host: `[id, new size in bytes]`.
pub(super) const RESIZE: u8 = 6;

/// A host function: `[id, type]`.
pub(super) const HOST: u8 = 7;
/// A module's first instantiation: `[id, SHA-256 of its bytecode]`.
pub(super) const MODULE: u8 = 8;
pub(super) const INSTANCE: u8 = 9;
pub(super) const GLOBAL: u8 = 10;
pub(super) const MEMORY: u8 = 11;
pub(super) const TABLE: u8 = 12;
/// A guest's failed `memory.grow` or `table.grow`; see `GrowthFailure`.
pub(super) const GROWTH_FAILED: u8 = 13;
/// The host's write to a global: `[id, value]`.
pub(super) const GLOBAL_WRITE: u8 = 14;
/// An embedder-defined event: `[tag, postcard payload]`.
pub(super) const EVENT: u8 = 15;
/// A recursion group of types, with the next IDs: a `rectype` in the Wasm binary
/// format.
pub(super) const TYPE: u8 = 16;
/// The release of a GC handle: `[handle]`.
pub(super) const RELEASE: u8 = 18;
/// A tag that the host created: `[id, type]`.
pub(super) const TAG: u8 = 19;

/// The host's write of one value to a table: `[id, start, length, value]`.
pub(super) const TABLE_FILL: u8 = 22;
/// A table's growth by the host: `[id, new size, value of new elements]`.
pub(super) const TABLE_GROW: u8 = 23;
/// The host's copy between tables:
/// `[destination, start, source, start, length]`.
pub(super) const TABLE_COPY: u8 = 24;

/// The host's allocation of an object, for the next handle: `[kind]` and
/// - `ALLOC_EXTERN`: nothing, as the guest can't observe host data
/// - `ALLOC_STRUCT`: `[type, fields]`
/// - `ALLOC_ARRAY`: `[type, length, value of all elements or of each]`
/// - `ALLOC_BYTES`: `[type, bytes]`
/// - `ALLOC_EXN`: `[tag, fields]`
#[cfg(feature = "gc")]
pub(super) const ALLOC: u8 = 17;
/// The host's read of a reference to an object, for the next handle:
/// `[kind, global, table, or handle, index]`.
#[cfg(feature = "gc")]
pub(super) const READ: u8 = 20;
/// The host's write to a field or element: `[handle, index, value]`.
#[cfg(feature = "gc")]
pub(super) const OBJECT_WRITE: u8 = 21;

#[cfg(feature = "gc")]
mod host_gc {
    pub const ALLOC_EXTERN: u8 = 0;
    pub const ALLOC_STRUCT: u8 = 1;
    pub const ALLOC_ARRAY: u8 = 2;
    pub const ALLOC_BYTES: u8 = 3;
    pub const ALLOC_EXN: u8 = 4;

    pub const READ_GLOBAL: u8 = 0;
    pub const READ_TABLE: u8 = 1;
    pub const READ_OBJECT: u8 = 2;
}
#[cfg(feature = "gc")]
pub(super) use host_gc::*;

pub(super) const UNSUPPORTED: &str =
    "record/replay does not support continuation references at boundaries";

/// The types of values that cross the record/replay boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    I32,
    I64,
    F32,
    F64,
    V128,
    /// A function reference.
    FuncRef {
        nullable: bool,
    },
    /// A reference to a GC object, or an `i31ref`.
    GcRef {
        nullable: bool,
    },
    /// Continuation references, which functions may use internally but which
    /// cannot cross the record/replay boundary.
    Unsupported,
}

impl Kind {
    pub fn new(ty: &ValType) -> Self {
        match ty {
            ValType::I32 => Self::I32,
            ValType::I64 => Self::I64,
            ValType::F32 => Self::F32,
            ValType::F64 => Self::F64,
            ValType::V128 => Self::V128,
            ValType::Ref(r) => {
                let nullable = r.is_nullable();
                match r.heap_type().top() {
                    HeapTopType::Func => Self::FuncRef { nullable },
                    HeapTopType::Any | HeapTopType::Extern | HeapTopType::Exn => {
                        Self::GcRef { nullable }
                    }
                    HeapTopType::Cont => Self::Unsupported,
                }
            }
        }
    }

    /// The size of a value of this type in a trace. References are represented
    /// by IDs.
    pub fn size(self) -> usize {
        match self {
            Self::I32 | Self::F32 | Self::FuncRef { .. } => 4,
            Self::I64 | Self::F64 | Self::GcRef { .. } => 8,
            Self::V128 => 16,
            Self::Unsupported => 0,
        }
    }
}

/// Abstract heap types, as encoded in the Wasm binary format.
fn abstract_heap_types() -> [(u8, HeapType); 14] {
    [
        (0x70, HeapType::Func),
        (0x6f, HeapType::Extern),
        (0x6e, HeapType::Any),
        (0x6d, HeapType::Eq),
        (0x6c, HeapType::I31),
        (0x6b, HeapType::Struct),
        (0x6a, HeapType::Array),
        (0x69, HeapType::Exn),
        (0x73, HeapType::NoFunc),
        (0x72, HeapType::NoExtern),
        (0x71, HeapType::None),
        (0x74, HeapType::NoExn),
        (0x68, HeapType::Cont),
        (0x75, HeapType::NoCont),
    ]
}

/// Appends a type in the encoding of a `valtype` in the Wasm binary format, in
/// which the trace's types are the index space of types.
pub(super) fn val_type(bytes: &mut TryVec<u8>, ty: &ValType, objects: &Objects) -> Result<()> {
    wasm_val_type(bytes, &ty.to_wasm_type(), objects)
}

/// Like [`val_type`].
pub(super) fn wasm_val_type(
    bytes: &mut TryVec<u8>,
    ty: &WasmValType,
    objects: &Objects,
) -> Result<()> {
    let ty = match ty {
        WasmValType::I32 => return Ok(bytes.push(0x7f)?),
        WasmValType::I64 => return Ok(bytes.push(0x7e)?),
        WasmValType::F32 => return Ok(bytes.push(0x7d)?),
        WasmValType::F64 => return Ok(bytes.push(0x7c)?),
        WasmValType::V128 => return Ok(bytes.push(0x7b)?),
        WasmValType::Ref(ty) => ty,
    };
    let abstract_ = abstract_heap_types()
        .into_iter()
        .find(|(_, abstract_)| abstract_.to_wasm_type() == ty.heap_type);
    if let Some((code, _)) = abstract_ {
        if !ty.nullable {
            bytes.push(0x64)?;
        }
        return Ok(bytes.push(code)?);
    }
    let (WasmHeapType::ConcreteFunc(index)
    | WasmHeapType::ConcreteStruct(index)
    | WasmHeapType::ConcreteArray(index)
    | WasmHeapType::ConcreteCont(index)) = ty.heap_type
    else {
        bail!("unsupported type in record/replay");
    };
    let id = objects.type_id(index)?.as_u32();
    bytes.push(if ty.nullable { 0x63 } else { 0x64 })?;
    signed_leb128(bytes, id)
}

/// Appends a number in unsigned LEB128.
pub(super) fn leb128(bytes: &mut TryVec<u8>, mut value: u32) -> Result<()> {
    while value >= 0x80 {
        bytes.push(u8::try_from(value & 0x7f)? | 0x80)?;
        value >>= 7;
    }
    Ok(bytes.push(u8::try_from(value)?)?)
}

/// Appends a nonnegative number in signed LEB128.
pub(super) fn signed_leb128(bytes: &mut TryVec<u8>, mut value: u32) -> Result<()> {
    while value >= 0x40 {
        bytes.push(u8::try_from(value & 0x7f)? | 0x80)?;
        value >>= 7;
    }
    Ok(bytes.push(u8::try_from(value)?)?)
}

/// Begins a record whose payload, of `len` bytes, the caller then appends.
pub(super) fn record(bytes: &mut TryVec<u8>, tag: u8, len: usize) -> Result<()> {
    let len32 = u32::try_from(len)?;
    bytes.reserve(
        len.checked_add(5)
            .ok_or_else(|| format_err!("trace length overflow"))?,
    )?;
    bytes.push(tag)?;
    bytes.extend_from_slice(&len32.to_le_bytes())?;
    Ok(())
}

/// Encodes references.
#[derive(Clone, Copy)]
pub(super) struct Refs<'a> {
    pub objects: &'a Objects,
    pub handles: &'a Handles,
}

/// Appends values, reading only the part of each slot that its type specifies.
///
/// # Safety
///
/// `raw` must point to an initialized slot of the corresponding type for each
/// element of `kinds`, whose references are valid.
pub(super) unsafe fn values(
    bytes: &mut TryVec<u8>,
    kinds: &[Kind],
    raw: *const ValRaw,
    refs: Refs<'_>,
) -> Result<()> {
    for (i, kind) in kinds.iter().enumerate() {
        // SAFETY: the caller guarantees these slots have the specified types.
        let value = unsafe { &*raw.add(i) };
        match kind {
            Kind::I32 => bytes.extend_from_slice(&value.get_i32().to_le_bytes())?,
            Kind::I64 => bytes.extend_from_slice(&value.get_i64().to_le_bytes())?,
            Kind::F32 => bytes.extend_from_slice(&value.get_f32().to_le_bytes())?,
            Kind::F64 => bytes.extend_from_slice(&value.get_f64().to_le_bytes())?,
            Kind::V128 => bytes.extend_from_slice(&value.get_v128().to_le_bytes())?,
            Kind::FuncRef { .. } => {
                // SAFETY: inherited from this function's contract.
                let id = unsafe { refs.objects.encode_ref(value.get_funcref())? };
                bytes.extend_from_slice(&id.to_le_bytes())?
            }
            Kind::GcRef { .. } => {
                // All GC references have the same representation.
                let id = refs.handles.encode(value.get_anyref())?;
                bytes.extend_from_slice(&id.to_le_bytes())?
            }
            Kind::Unsupported => bail!(UNSUPPORTED),
        }
    }
    Ok(())
}

pub(super) fn values_len(kinds: &[Kind]) -> usize {
    kinds.iter().map(|k| k.size()).sum()
}

/// Records the return of `call`, which may have thrown `exception`.
///
/// # Safety
///
/// If `result` is successful then the contract of `values` applies to `kinds`,
/// and `raw`.
pub(super) unsafe fn outcome(
    bytes: &mut TryVec<u8>,
    tag: u8,
    call: RecordedCall,
    result: &Result<()>,
    kinds: &[Kind],
    raw: *const ValRaw,
    refs: Refs<'_>,
    exception: u32,
) -> Result<()> {
    let call = call.as_u32().to_le_bytes();
    match result {
        Ok(()) => {
            record(bytes, tag, 5 + values_len(kinds))?;
            bytes.extend_from_slice(&call)?;
            bytes.push(0)?;
            // SAFETY: inherited from this function's contract.
            unsafe { values(bytes, kinds, raw, refs)? };
        }
        Err(e) => {
            let thrown = e.is::<crate::ThrownException>();
            if thrown || (tag == LEAVE_WASM && is_exception(e)) {
                let exception = refs.handles.encode(if thrown { exception } else { 0 })?;
                record(bytes, tag, 13)?;
                bytes.extend_from_slice(&call)?;
                bytes.push(3)?;
                bytes.extend_from_slice(&exception.to_le_bytes())?;
            } else if let Some(trap) = e.downcast_ref::<Trap>() {
                record(bytes, tag, 6)?;
                bytes.extend_from_slice(&call)?;
                bytes.extend_from_slice(&[1, *trap as u8])?;
            } else {
                // Context, such as a Wasm backtrace, depends on the activation's
                // callers and runtime configuration, so only the root cause of
                // an error is reproduced.
                let message = e.root_cause().to_string();
                record(bytes, tag, 5 + message.len())?;
                bytes.extend_from_slice(&call)?;
                bytes.push(2)?;
                bytes.extend_from_slice(message.as_bytes())?;
            }
        }
    }
    Ok(())
}

/// Whether a call into Wasm failed as the guest threw an exception. How the host
/// sees that, as an exception or a trap, depends on how it called the guest.
fn is_exception(e: &Error) -> bool {
    e.is::<crate::ThrownException>() || e.downcast_ref() == Some(&Trap::UncaughtException)
}

/// A cursor over the bytes of a trace or of a record.
pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::at(bytes, 0)
    }

    pub fn at(bytes: &'a [u8], position: usize) -> Self {
        Self { bytes, position }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// All bytes, regardless of what has been read.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Reads all remaining bytes.
    pub fn rest(&mut self) -> &'a [u8] {
        let rest = &self.bytes[self.position..];
        self.position = self.bytes.len();
        rest
    }

    /// The tag of the next record, if any.
    pub fn peek_tag(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    pub fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| format_err!("trace length overflow"))?;
        let result = self.bytes.get(self.position..end).ok_or_else(|| {
            format_err!("truncated record/replay trace at byte {}", self.position)
        })?;
        self.position = end;
        Ok(result)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => bail!("invalid boolean in trace"),
        }
    }

    pub fn blob(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.u32()?)?;
        self.take(len)
    }

    pub fn checksum(&mut self) -> Result<super::Checksum> {
        Ok(self.take(32)?.try_into().unwrap())
    }

    pub fn record(&mut self) -> Result<(u8, Reader<'a>)> {
        let tag = self.u8()?;
        let body = self.blob()?;
        Ok((tag, Reader::new(body)))
    }

    pub fn end(&self) -> Result<()> {
        ensure!(
            self.position == self.bytes.len(),
            "unexpected trailing trace data"
        );
        Ok(())
    }

    /// Reads a type from [`val_type`].
    pub fn val_type(&mut self, objects: &Objects) -> Result<ValType> {
        let ref_type = |nullable, heap| ValType::Ref(RefType::new(nullable, heap));
        let find_abstract = |code| {
            let mut types = abstract_heap_types().into_iter();
            types.find_map(|(c, ty)| (c == code).then_some(ty))
        };
        let code = self.u8()?;
        Ok(match code {
            0x7f => ValType::I32,
            0x7e => ValType::I64,
            0x7d => ValType::F32,
            0x7c => ValType::F64,
            0x7b => ValType::V128,
            0x63 | 0x64 => {
                let next = self.u8()?;
                if let Some(heap) = find_abstract(next) {
                    ensure!(code == 0x64, "invalid value type in trace");
                    return Ok(ref_type(false, heap));
                }
                let mut byte = next;
                let mut id = u32::from(byte & 0x7f);
                let mut shift = 0;
                while byte & 0x80 != 0 {
                    byte = self.u8()?;
                    shift += 7;
                    let bits = u32::from(byte & 0x7f);
                    ensure!(shift < 28 || bits < 8, "invalid value type in trace");
                    id |= bits.checked_shl(shift).unwrap_or(0);
                }
                let ty = objects.types.get(super::RecordedType::from_u32(id));
                let ty = ty.ok_or_else(|| format_err!("invalid type ID in trace"))?;
                ref_type(code == 0x63, ty.clone())
            }
            code => match find_abstract(code) {
                Some(heap) => ref_type(true, heap),
                None => bail!("invalid value type in trace"),
            },
        })
    }

    /// Reads a value. A GC reference has yet to be exposed to Wasm, and only
    /// the nullability of a reference has been checked.
    fn value(&mut self, kind: Kind, store: &StoreOpaque) -> Result<ValRaw> {
        Ok(match kind {
            Kind::I32 => ValRaw::u32(self.u32()?),
            Kind::I64 => ValRaw::u64(self.u64()?),
            Kind::F32 => ValRaw::f32(self.u32()?),
            Kind::F64 => ValRaw::f64(self.u64()?),
            Kind::V128 => ValRaw::v128(u128::from_le_bytes(self.take(16)?.try_into().unwrap())),
            Kind::FuncRef { nullable } => {
                let func = store.rr_objects().decode_ref(self.u32()?)?;
                ensure!(
                    nullable || func.is_some(),
                    "null value of non-nullable type in trace"
                );
                ValRaw::funcref(match func {
                    Some(func) => func.vm_func_ref(store).as_ptr().cast(),
                    None => core::ptr::null_mut(),
                })
            }
            Kind::GcRef { nullable } => {
                let gc_ref = store.rr_handles().decode(self.u64()?)?;
                ensure!(
                    nullable || gc_ref.is_some(),
                    "null value of non-nullable type in trace"
                );
                ValRaw::anyref(gc_ref.map_or(0, |r| r.as_raw_u32()))
            }
            Kind::Unsupported => bail!(UNSUPPORTED),
        })
    }

    /// Reads values into the first of `slots`.
    pub fn values(
        &mut self,
        types: &[ValType],
        slots: &mut [MaybeUninit<ValRaw>],
        store: &mut StoreOpaque,
    ) -> Result<()> {
        ensure!(slots.len() >= types.len(), "not enough value slots");
        for (ty, slot) in types.iter().zip(slots) {
            let kind = Kind::new(ty);
            slot.write(self.value(kind, store)?);
            // SAFETY: the slot was just initialized with a value of this kind.
            let raw = unsafe { slot.assume_init_ref() };
            store.rr_check_ref_type(ty, raw)?;
            store.rr_expose_refs(&[kind], core::slice::from_mut(slot))?;
        }
        Ok(())
    }

    /// Whether this is the record of a call into Wasm that threw an exception
    /// that the host did not see.
    pub fn has_null_exception(&self) -> bool {
        self.bytes.len() == 13 && self.bytes[4..] == [3, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    /// Reads how a call returned, from [`outcome`], after its call ID. If it
    /// succeeded, its results are written to the first of `slots`.
    pub fn outcome(
        &mut self,
        types: &[ValType],
        slots: &mut [MaybeUninit<ValRaw>],
        store: &mut StoreOpaque,
    ) -> Result<Result<()>> {
        let result = match self.u8()? {
            0 => Ok(self.values(types, slots, store)?),
            1 => Err(Trap::from_u8(self.u8()?)
                .ok_or_else(|| format_err!("invalid trap code in trace"))?
                .into()),
            2 => Err(format_err!("{}", core::str::from_utf8(self.rest())?)),
            3 => Err(store.rr_throw(self.u64()?)?),
            _ => bail!("invalid outcome in trace"),
        };
        self.end()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_use_the_wasm_binary_encoding() -> Result<()> {
        use wasmparser::{RefType as R, ValType as V};

        let objects = Objects::default();
        for (ty, wasm) in [
            (ValType::I32, V::I32),
            (ValType::I64, V::I64),
            (ValType::F32, V::F32),
            (ValType::F64, V::F64),
            (ValType::V128, V::V128),
            (ValType::FUNCREF, V::FUNCREF),
            (
                ValType::Ref(RefType::new(false, HeapType::Func)),
                V::Ref(R::FUNC),
            ),
            (ValType::EXTERNREF, V::EXTERNREF),
            (ValType::ANYREF, V::Ref(R::ANYREF)),
            (
                ValType::Ref(RefType::new(false, HeapType::Struct)),
                V::Ref(R::STRUCT),
            ),
            (ValType::EXNREF, V::EXNREF),
            (ValType::NULLREF, V::Ref(R::NULLREF)),
        ] {
            let mut bytes = TryVec::new();
            val_type(&mut bytes, &ty, &objects)?;
            let mut reader = Reader::new(&bytes);
            assert!(ValType::eq(&reader.val_type(&objects)?, &ty));
            reader.end()?;
            let mut reader = wasmparser::BinaryReader::new(&bytes, 0);
            assert_eq!(reader.read::<V>()?, wasm);
            assert!(reader.eof());
        }
        Ok(())
    }
}
