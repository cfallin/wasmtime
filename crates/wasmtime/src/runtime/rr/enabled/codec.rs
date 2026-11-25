//! Versioned wire format. All lengths and numbers are little endian.
//!
//! Each record has a one-byte tag and a four-byte payload length. The format is
//! versioned independently of Wasmtime: changing it requires changing MAGIC.

use super::{Objects, RecordedCall};
use crate::prelude::*;
use crate::store::StoreOpaque;
use crate::{HeapType, RefType, Trap, ValRaw, ValType};
use core::mem::MaybeUninit;

pub(super) const MAGIC: &[u8; 8] = b"WTRR\0\0\0\x0e";
pub(super) const END: u8 = 0;
pub(super) const ENTER_WASM: u8 = 1;
pub(super) const LEAVE_WASM: u8 = 2;
pub(super) const ENTER_HOST: u8 = 3;
pub(super) const LEAVE_HOST: u8 = 4;
pub(super) const WRITE: u8 = 5;
/// A memory's growth by the host: `[id, new size in bytes]`.
pub(super) const RESIZE: u8 = 6;

pub(super) const HOST: u8 = 7;
/// A module's first instantiation: `[id, SHA-256 of its bytecode]`.
pub(super) const MODULE: u8 = 8;
pub(super) const INSTANCE: u8 = 9;
pub(super) const GLOBAL: u8 = 10;
pub(super) const MEMORY: u8 = 11;
pub(super) const TABLE: u8 = 12;
/// A guest's failed `memory.grow` or `table.grow`; see `GrowthFailure`.
pub(super) const GROWTH_FAILED: u8 = 13;
/// The host's write to a component instance flag: `[id, value]`.
pub(super) const GLOBAL_WRITE: u8 = 14;
/// An embedder-defined event: `[tag, postcard payload]`.
pub(super) const EVENT: u8 = 15;

pub(super) const UNSUPPORTED: &str =
    "record/replay does not support GC or typed reference boundaries";

/// The types of values that cross the record/replay boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    I32,
    I64,
    F32,
    F64,
    V128,
    /// An abstract function reference.
    FuncRef {
        nullable: bool,
    },
    /// GC and typed references, which functions may use internally but which
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
            ValType::Ref(r) if r.heap_type().is_func() => Self::FuncRef {
                nullable: r.is_nullable(),
            },
            ValType::Ref(_) => Self::Unsupported,
        }
    }

    /// Appends the type in the encoding of a `valtype` in the Wasm binary
    /// format.
    pub fn encode(self, bytes: &mut TryVec<u8>) -> Result<()> {
        let encoded: &[u8] = match self {
            Self::I32 => &[0x7f],
            Self::I64 => &[0x7e],
            Self::F32 => &[0x7d],
            Self::F64 => &[0x7c],
            Self::V128 => &[0x7b],
            Self::FuncRef { nullable: true } => &[0x70],
            Self::FuncRef { nullable: false } => &[0x64, 0x70],
            Self::Unsupported => bail!(UNSUPPORTED),
        };
        bytes.extend_from_slice(encoded)?;
        Ok(())
    }

    /// Decodes a list of types from [`Kind::encode`].
    pub fn decode_list(bytes: &[u8]) -> Result<TryVec<Self>> {
        let mut reader = Reader::new(bytes);
        let mut kinds = TryVec::new();
        while reader.end().is_err() {
            kinds.push(reader.kind()?)?;
        }
        Ok(kinds)
    }

    pub fn ty(self) -> Result<ValType> {
        Ok(match self {
            Self::I32 => ValType::I32,
            Self::I64 => ValType::I64,
            Self::F32 => ValType::F32,
            Self::F64 => ValType::F64,
            Self::V128 => ValType::V128,
            Self::FuncRef { nullable } => ValType::Ref(RefType::new(nullable, HeapType::Func)),
            Self::Unsupported => bail!(UNSUPPORTED),
        })
    }

    /// The size of a value of this type in a trace. Function references are
    /// represented by IDs.
    pub fn size(self) -> usize {
        match self {
            Self::I32 | Self::F32 | Self::FuncRef { .. } => 4,
            Self::I64 | Self::F64 => 8,
            Self::V128 => 16,
            Self::Unsupported => 0,
        }
    }
}

pub(super) fn blob(bytes: &mut TryVec<u8>, value: &[u8]) -> Result<()> {
    let len = u32::try_from(value.len())?;
    bytes.extend_from_slice(&len.to_le_bytes())?;
    bytes.extend_from_slice(value)?;
    Ok(())
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

/// Appends values, reading only the part of each slot that its type specifies.
///
/// # Safety
///
/// `raw` must point to an initialized slot of the corresponding type for each
/// element of `kinds`, and it must be sound to call `encode_ref` with the
/// function references among them.
pub(super) unsafe fn values(
    bytes: &mut TryVec<u8>,
    kinds: &[Kind],
    raw: *const ValRaw,
    mut encode_ref: impl FnMut(*mut core::ffi::c_void) -> Result<u32>,
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
                bytes.extend_from_slice(&encode_ref(value.get_funcref())?.to_le_bytes())?
            }
            Kind::Unsupported => bail!(UNSUPPORTED),
        }
    }
    Ok(())
}

pub(super) fn values_len(kinds: &[Kind]) -> usize {
    kinds.iter().map(|k| k.size()).sum()
}

/// Records the return of `call`.
///
/// # Safety
///
/// If `result` is successful then the contract of `values` applies to `kinds`,
/// `raw`, and `encode_ref`.
pub(super) unsafe fn outcome(
    bytes: &mut TryVec<u8>,
    tag: u8,
    call: RecordedCall,
    result: &Result<()>,
    kinds: &[Kind],
    raw: *const ValRaw,
    encode_ref: impl FnMut(*mut core::ffi::c_void) -> Result<u32>,
) -> Result<()> {
    let call = call.as_u32().to_le_bytes();
    match result {
        Ok(()) => {
            record(bytes, tag, 5 + values_len(kinds))?;
            bytes.extend_from_slice(&call)?;
            bytes.push(0)?;
            // SAFETY: inherited from this function's contract.
            unsafe { values(bytes, kinds, raw, encode_ref)? };
        }
        Err(e) => {
            if tag == LEAVE_WASM && is_exception(e) {
                record(bytes, tag, 5)?;
                bytes.extend_from_slice(&call)?;
                bytes.push(3)?;
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

    /// Reads a type from [`Kind::encode`].
    pub fn kind(&mut self) -> Result<Kind> {
        Ok(match self.u8()? {
            0x7f => Kind::I32,
            0x7e => Kind::I64,
            0x7d => Kind::F32,
            0x7c => Kind::F64,
            0x7b => Kind::V128,
            0x70 => Kind::FuncRef { nullable: true },
            0x64 if self.u8()? == 0x70 => Kind::FuncRef { nullable: false },
            _ => bail!("invalid value type in trace"),
        })
    }

    pub fn value(&mut self, kind: Kind, store: &StoreOpaque, objects: &Objects) -> Result<ValRaw> {
        Ok(match kind {
            Kind::I32 => ValRaw::u32(self.u32()?),
            Kind::I64 => ValRaw::u64(self.u64()?),
            Kind::F32 => ValRaw::f32(self.u32()?),
            Kind::F64 => ValRaw::f64(self.u64()?),
            Kind::V128 => ValRaw::v128(u128::from_le_bytes(self.take(16)?.try_into().unwrap())),
            Kind::FuncRef { nullable } => {
                ValRaw::funcref(match objects.decode_ref(self.u32()?)? {
                    Some(func) => func.vm_func_ref(store).as_ptr().cast(),
                    None => {
                        ensure!(nullable, "null value of non-nullable type in trace");
                        core::ptr::null_mut()
                    }
                })
            }
            Kind::Unsupported => bail!(UNSUPPORTED),
        })
    }

    /// Reads values into the first of `slots`.
    pub fn values(
        &mut self,
        kinds: &[Kind],
        slots: &mut [MaybeUninit<ValRaw>],
        store: &StoreOpaque,
        objects: &Objects,
    ) -> Result<()> {
        ensure!(slots.len() >= kinds.len(), "not enough value slots");
        for (kind, slot) in kinds.iter().zip(slots) {
            slot.write(self.value(*kind, store, objects)?);
        }
        Ok(())
    }

    /// Reads how a call returned, from [`outcome`], after its call ID. If it
    /// succeeded, its results are written to the first of `slots`.
    pub fn outcome(
        &mut self,
        kinds: &[Kind],
        slots: &mut [MaybeUninit<ValRaw>],
        store: &StoreOpaque,
        objects: &Objects,
    ) -> Result<Result<()>> {
        let result = match self.u8()? {
            0 => Ok(self.values(kinds, slots, store, objects)?),
            1 => Err(Trap::from_u8(self.u8()?)
                .ok_or_else(|| format_err!("invalid trap code in trace"))?
                .into()),
            2 => Err(format_err!("{}", core::str::from_utf8(self.rest())?)),
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
    fn kinds_use_the_wasm_binary_encoding() -> Result<()> {
        use wasmparser::{RefType, ValType};

        for (kind, ty) in [
            (Kind::I32, ValType::I32),
            (Kind::I64, ValType::I64),
            (Kind::F32, ValType::F32),
            (Kind::F64, ValType::F64),
            (Kind::V128, ValType::V128),
            (Kind::FuncRef { nullable: true }, ValType::FUNCREF),
            (
                Kind::FuncRef { nullable: false },
                ValType::Ref(RefType::FUNC),
            ),
        ] {
            let mut bytes = TryVec::new();
            kind.encode(&mut bytes)?;
            assert_eq!(*Kind::decode_list(&bytes)?, [kind]);
            let mut reader = wasmparser::BinaryReader::new(&bytes, 0);
            assert_eq!(reader.read::<ValType>()?, ty);
            assert!(reader.eof());
        }
        assert!(Kind::Unsupported.encode(&mut TryVec::new()).is_err());
        Ok(())
    }
}
