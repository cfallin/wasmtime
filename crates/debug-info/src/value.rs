//! Values: typed views of debuggee data, and the operations used by
//! access paths (member access, indexing, dereference, address-of).

use crate::dwarf::{DebugInfo, Variable};
use crate::eval::Location;
use crate::location::{self, WasmSlot};
use crate::pretty;
use crate::target::Target;
use crate::types::*;
use anyhow::{Result, anyhow, bail};

/// Where a value's bytes come from.
#[derive(Clone, Debug)]
pub enum Data {
    /// An object in linear memory at this address (read lazily).
    Memory(u64),
    /// Captured bytes (from a Wasm local, a constant, or a snapshot),
    /// with the original memory address if the value was an lvalue in
    /// memory, or the Wasm slot (and byte offset within it) it came
    /// from.
    Bytes {
        bytes: Vec<u8>,
        addr: Option<u64>,
        slot: Option<(WasmSlot, u64)>,
    },
}

/// A typed value.
#[derive(Clone, Debug)]
pub struct Value {
    pub ty: TypeId,
    pub data: Data,
    /// For bitfields: (bit offset within the data, bit size).
    pub bitfield: Option<(u64, u64)>,
}

/// Context for operating on values: debug info plus debuggee access.
pub struct Ctx<'a> {
    pub info: &'a DebugInfo,
    pub target: &'a dyn Target,
    /// Nesting depth of "look-through" fallbacks (member/index access
    /// via a pretty-printer's `deref`), to bound recursion.
    look_through: std::cell::Cell<u32>,
}

impl Value {
    /// A value in memory.
    pub fn in_memory(ty: TypeId, addr: u64) -> Value {
        Value {
            ty,
            data: Data::Memory(addr),
            bitfield: None,
        }
    }

    /// A value from captured bytes.
    pub fn from_bytes(ty: TypeId, bytes: Vec<u8>) -> Value {
        Value {
            ty,
            data: Data::Bytes {
                bytes,
                addr: None,
                slot: None,
            },
            bitfield: None,
        }
    }

    /// A value held in a Wasm slot (local, global, or operand-stack
    /// entry), whose current contents are `bytes`.
    pub fn in_slot(ty: TypeId, bytes: Vec<u8>, slot: WasmSlot) -> Value {
        Value {
            ty,
            data: Data::Bytes {
                bytes,
                addr: None,
                slot: Some((slot, 0)),
            },
            bitfield: None,
        }
    }

    /// The memory address of this value, if it is (or was) an lvalue.
    pub fn address(&self) -> Option<u64> {
        if self.bitfield.is_some() {
            return None;
        }
        match &self.data {
            Data::Memory(a) => Some(*a),
            Data::Bytes { addr, .. } => *addr,
        }
    }

    /// A sub-object at `byte_offset` with type `ty`.
    pub fn sub(&self, ty: TypeId, byte_offset: u64, bitfield: Option<(u64, u64)>) -> Value {
        let data = match &self.data {
            Data::Memory(a) => Data::Memory(a + byte_offset),
            Data::Bytes { bytes, addr, slot } => {
                let start = usize::try_from(byte_offset).unwrap_or(usize::MAX);
                Data::Bytes {
                    bytes: bytes.get(start..).map(|b| b.to_vec()).unwrap_or_default(),
                    addr: addr.map(|a| a + byte_offset),
                    slot: slot.map(|(s, o)| (s, o + byte_offset)),
                }
            }
        };
        Value { ty, data, bitfield }
    }
}

impl<'a> Ctx<'a> {
    pub fn new(info: &'a DebugInfo, target: &'a dyn Target) -> Self {
        Ctx {
            info,
            target,
            look_through: std::cell::Cell::new(0),
        }
    }

    /// Pointer size for values of the given type's language/unit.
    pub fn ptr_size(&self) -> u64 {
        self.info.default_address_size()
    }

    /// Read `len` bytes of the value (from its start).
    pub fn read_bytes(&self, v: &Value, len: u64) -> Result<Vec<u8>> {
        match &v.data {
            Data::Memory(a) => self.target.read_memory(0, *a, len),
            Data::Bytes { bytes, .. } => {
                let len = usize::try_from(len)?;
                if bytes.len() < len {
                    bail!("value bytes unavailable");
                }
                Ok(bytes[..len].to_vec())
            }
        }
    }

    /// Read the value's full contents.
    pub fn contents(&self, v: &Value) -> Result<Vec<u8>> {
        let size = self
            .info
            .type_size(v.ty)
            .ok_or_else(|| anyhow!("type `{}` has unknown size", self.info.type_name(v.ty)))?;
        if size > 1 << 24 {
            bail!("value too large ({size} bytes)");
        }
        self.read_bytes(v, size)
    }

    /// Read the value as an unsigned integer (up to 8 bytes), handling
    /// bitfields.
    pub fn read_uint(&self, v: &Value) -> Result<u64> {
        if let Some((off, size)) = v.bitfield {
            let nbytes = (off + size).div_ceil(8);
            let bytes = self.read_bytes(v, nbytes)?;
            let mut raw: u128 = 0;
            for (i, b) in bytes.iter().enumerate() {
                raw |= u128::from(*b) << (8 * i);
            }
            let mask = if size >= 64 {
                u64::MAX
            } else {
                (1u64 << size) - 1
            };
            return Ok(((raw >> off) as u64) & mask);
        }
        let size = self.info.type_size(v.ty).unwrap_or(self.ptr_size()).min(8);
        let bytes = self.read_bytes(v, size)?;
        let mut b = [0u8; 8];
        b[..bytes.len()].copy_from_slice(&bytes);
        Ok(u64::from_le_bytes(b))
    }

    /// Read the value as a signed integer, sign-extending from its size.
    pub fn read_int(&self, v: &Value) -> Result<i64> {
        let raw = self.read_uint(v)?;
        let bits = match v.bitfield {
            Some((_, s)) => s,
            None => self.info.type_size(v.ty).unwrap_or(8).min(8) * 8,
        };
        if bits == 0 || bits >= 64 {
            return Ok(raw as i64);
        }
        let shift = 64 - bits;
        Ok(((raw << shift) as i64) >> shift)
    }

    /// Snapshot a value: capture its bytes now so that later printing
    /// shows the value as of now (pointers inside still refer to live
    /// memory).
    pub fn snapshot(&self, v: &Value) -> Result<Value> {
        if v.bitfield.is_some() {
            let x = self.read_uint(v)?;
            return Ok(Value::from_bytes(v.ty, x.to_le_bytes().to_vec()));
        }
        match &v.data {
            Data::Bytes { .. } => Ok(v.clone()),
            Data::Memory(a) => {
                let bytes = self.contents(v)?;
                Ok(Value {
                    ty: v.ty,
                    data: Data::Bytes {
                        bytes,
                        addr: Some(*a),
                        slot: None,
                    },
                    bitfield: None,
                })
            }
        }
    }

    /// Materialize an evaluated DWARF location as a value of type `ty`.
    pub fn from_location(&self, ty: TypeId, loc: Location) -> Result<Value> {
        let size = self.info.type_size(ty).unwrap_or(8);
        match loc {
            Location::Memory(a) => Ok(Value::in_memory(ty, a)),
            Location::Value(mut bytes) => {
                bytes.resize(usize::try_from(size)?.max(bytes.len()), 0);
                Ok(Value::from_bytes(ty, bytes))
            }
            Location::WasmLocal(i) => Ok(Value::in_slot(
                ty,
                self.target.local(i)?.to_bytes(),
                WasmSlot::Local(i),
            )),
            Location::WasmGlobal(i) => Ok(Value::in_slot(
                ty,
                self.target.global(i)?.to_bytes(),
                WasmSlot::Global(i),
            )),
            Location::WasmStack(i) => Ok(Value::in_slot(
                ty,
                self.target.operand(i)?.to_bytes(),
                WasmSlot::Stack(i),
            )),
            Location::Pieces(pieces) => {
                let mut bytes = vec![];
                for p in pieces {
                    let n = p.size_bits.div_ceil(8);
                    let mut b = match p.loc {
                        Location::Memory(a) => self.target.read_memory(0, a, n)?,
                        Location::Value(b) => b,
                        Location::WasmLocal(i) => self.target.local(i)?.to_bytes(),
                        Location::WasmGlobal(i) => self.target.global(i)?.to_bytes(),
                        Location::WasmStack(i) => self.target.operand(i)?.to_bytes(),
                        Location::Unavailable(r) => bail!("<{r}>"),
                        Location::Pieces(_) => bail!("nested pieces"),
                    };
                    b.resize(usize::try_from(n)?, 0);
                    bytes.extend(b);
                }
                Ok(Value::from_bytes(ty, bytes))
            }
            Location::Unavailable(reason) => bail!("<{reason}>"),
        }
    }

    /// Read a variable's value at DWARF address `addr`, given the frame
    /// base of its function.
    pub fn variable(&self, var: &Variable, addr: u64, frame_base: Option<u64>) -> Result<Value> {
        let ty = var
            .ty
            .ok_or_else(|| anyhow!("variable `{}` has no type", var.name))?;
        let loc = self
            .info
            .eval_loc(var.die.unit, &var.loc, addr, frame_base, self.target)?;
        self.from_location(ty, loc)
    }

    // -----------------------------------------------------------------
    // Access-path operations.

    /// Dereference a pointer value.
    pub fn deref(&self, v: &Value) -> Result<Value> {
        let sty = self.info.strip(v.ty);
        let t = self.info.ty(sty);
        match &t.kind {
            TypeKind::Pointer { pointee, .. } => {
                let pointee =
                    pointee.ok_or_else(|| anyhow!("cannot dereference a `void *` pointer"))?;
                let addr = self.read_uint(v)?;
                if addr == 0 {
                    bail!("cannot dereference a null pointer");
                }
                Ok(Value::in_memory(pointee, addr))
            }
            _ => {
                if let Some(p) = pretty::find(self, v.ty) {
                    if let Some(r) = p.deref(self, v) {
                        return r;
                    }
                }
                bail!(
                    "cannot dereference a value of type `{}`",
                    self.info.type_name(v.ty)
                )
            }
        }
    }

    /// Take the address of a value.
    pub fn address_of(&self, v: &Value) -> Result<Value> {
        let addr = match v.address() {
            Some(a) => a,
            None => match self.location(v) {
                Ok(loc) => bail!("cannot take the address of a value in {loc}"),
                Err(_) if v.bitfield.is_some() => bail!("cannot take the address of a bitfield"),
                Err(e) => bail!("cannot take the address: {e}"),
            },
        };
        let ptr_size = self.ptr_size();
        let pty = self.info.pointer_to(v.ty, ptr_size);
        let bytes = addr.to_le_bytes()[..usize::try_from(ptr_size)?].to_vec();
        Ok(Value::from_bytes(pty, bytes))
    }

    /// Where a value lives: a range of linear memory, or (part of) a Wasm
    /// local, global or operand-stack slot. Fails for values that were
    /// computed rather than stored (constants, and values whose DWARF
    /// location is a computed expression or split into pieces).
    pub fn location(&self, v: &Value) -> Result<location::Location> {
        let size = match v.bitfield {
            Some((off, bits)) => (off + bits).div_ceil(8),
            None => self.info.type_size(v.ty).unwrap_or(0),
        };
        match &v.data {
            Data::Memory(base) => Ok(location::Location::Mem { base: *base, size }),
            Data::Bytes {
                addr: Some(base), ..
            } => Ok(location::Location::Mem { base: *base, size }),
            Data::Bytes {
                slot: Some((slot, offset)),
                ..
            } => Ok(location::Location::in_slot(*slot, *offset, size)),
            Data::Bytes { .. } => {
                bail!("the value has no storage location (it is computed, not stored)")
            }
        }
    }

    /// If `v` is a Rust-style enum (struct with a variant part), find
    /// the active variant: returns (variant name, payload value).
    pub fn active_variant(&self, v: &Value) -> Result<Option<(String, Value)>> {
        let sty = self.info.strip(v.ty);
        let t = self.info.ty(sty);
        let TypeKind::Struct(st) = &t.kind else {
            return Ok(None);
        };
        let Some(part) = &st.variant_part else {
            return Ok(None);
        };
        let discr = match &part.discr {
            Some(m) => {
                let dv = v.sub(
                    m.ty,
                    m.byte_offset(),
                    m.bit_size.map(|s| (m.bit_offset % 8, s)),
                );
                Some(self.read_uint(&dv)?)
            }
            None => None,
        };
        let mut chosen = None;
        if let Some(d) = discr {
            let dsize = part
                .discr
                .as_ref()
                .and_then(|m| self.info.type_size(m.ty))
                .unwrap_or(8);
            let mask = if dsize >= 8 {
                u64::MAX
            } else {
                (1u64 << (dsize * 8)) - 1
            };
            chosen = part
                .variants
                .iter()
                .find(|var| var.discr_value.is_some_and(|x| x & mask == d & mask));
        }
        if chosen.is_none() {
            chosen = part.variants.iter().find(|var| var.discr_value.is_none());
        }
        let Some(variant) = chosen else {
            bail!("invalid enum discriminant {discr:?}");
        };
        let Some(m) = variant.members.first() else {
            return Ok(None);
        };
        let name = m.name.clone().unwrap_or_else(|| "?".into());
        Ok(Some((name, v.sub(m.ty, m.byte_offset(), None))))
    }

    /// Access a member by name. Pointers (and references) are
    /// dereferenced automatically; for Rust enums the member is looked
    /// up in the active variant. Numeric names (`0`, `1`) select tuple
    /// fields.
    pub fn member(&self, v: &Value, name: &str) -> Result<Value> {
        self.member_impl(v, name, true)
    }

    /// Like [`Ctx::member`], but never looks through pretty-printed
    /// wrappers. Pretty-printers use this to access library internals.
    pub fn field(&self, v: &Value, name: &str) -> Result<Value> {
        self.member_impl(v, name, false)
    }

    fn member_impl(&self, v: &Value, name: &str, look_through: bool) -> Result<Value> {
        let sty = self.info.strip(v.ty);
        let t = self.info.ty(sty);
        match &t.kind {
            TypeKind::Pointer { .. } => {
                let d = self.deref(v)?;
                self.member_impl(&d, name, look_through)
            }
            TypeKind::Struct(st) => {
                if st.is_enum_like() {
                    if let Some((vname, payload)) = self.active_variant(v)? {
                        if vname == name {
                            return Ok(payload);
                        }
                        if let Ok(v) = self.member_impl(&payload, name, look_through) {
                            return Ok(v);
                        }
                        // As a convenience, look through single-field
                        // tuple variants (e.g. `opt.field` for
                        // `Some(Box<T>)`), as if auto-dereferencing.
                        if let Ok(inner) = self.member_impl(&payload, "__0", look_through) {
                            let single = match &self.info.ty(self.info.strip(payload.ty)).kind {
                                TypeKind::Struct(pst) => pst.members.len() == 1,
                                _ => false,
                            };
                            if single {
                                if let Ok(v) = self.member_impl(&inner, name, look_through) {
                                    return Ok(v);
                                }
                            }
                        }
                        bail!("active variant `{vname}` has no field `{name}`");
                    }
                }
                if let Some(r) = self.find_member(v, st, name)? {
                    return Ok(r);
                }
                if name.chars().all(|c| c.is_ascii_digit()) {
                    if let Some(r) = self.find_member(v, st, &format!("__{name}"))? {
                        return Ok(r);
                    }
                }
                // Look through "transparent" pretty-printed wrappers
                // (e.g. `Arc<T>`, `RefCell<T>`, `std::unique_ptr<T>`).
                if look_through {
                    if let Some(r) = self.look_through(v, |inner| self.member(inner, name)) {
                        return r;
                    }
                }
                bail!(
                    "type `{}` has no member `{name}`",
                    self.info.type_name(v.ty)
                )
            }
            _ => bail!(
                "cannot access member `{name}` of non-struct type `{}`",
                self.info.type_name(v.ty)
            ),
        }
    }

    fn find_member(&self, v: &Value, st: &StructType, name: &str) -> Result<Option<Value>> {
        for m in &st.members {
            let mv = || {
                v.sub(
                    m.ty,
                    m.byte_offset(),
                    m.bit_size.map(|s| (m.bit_offset % 8, s)),
                )
            };
            if m.name.as_deref() == Some(name) {
                return Ok(Some(mv()));
            }
            // Anonymous struct/union members, and C++ base classes
            // (named `<Base>`), are searched recursively.
            if m.name.as_deref().is_none_or(|n| n.starts_with('<')) {
                let inner = mv();
                let it = self.info.ty(self.info.strip(m.ty));
                if let TypeKind::Struct(ist) = &it.kind {
                    if let Some(r) = self.find_member(&inner, ist, name)? {
                        return Ok(Some(r));
                    }
                }
            }
        }
        Ok(None)
    }

    /// If `v` has a pretty-printer that dereferences it (smart pointers,
    /// cells), apply `f` to the dereferenced value. Bounded in depth.
    fn look_through(
        &self,
        v: &Value,
        f: impl FnOnce(&Value) -> Result<Value>,
    ) -> Option<Result<Value>> {
        let depth = self.look_through.get();
        if depth >= 8 {
            return None;
        }
        let p = pretty::find(self, v.ty)?;
        self.look_through.set(depth + 1);
        let r = match p.deref(self, v) {
            Some(Ok(inner)) => Some(f(&inner)),
            _ => None,
        };
        self.look_through.set(depth);
        r
    }

    /// Index into an array, pointer, or indexable pretty-printed type.
    pub fn index(&self, v: &Value, i: u64) -> Result<Value> {
        if let Some(p) = pretty::find(self, v.ty) {
            if let Some(r) = p.index(self, v, i) {
                return r;
            }
            if let Some(r) = self.look_through(v, |inner| self.index(inner, i)) {
                return r;
            }
        }
        let sty = self.info.strip(v.ty);
        let t = self.info.ty(sty);
        match &t.kind {
            TypeKind::Array { elem, len } => {
                if let Some(len) = len {
                    if i >= *len {
                        bail!("index {i} out of bounds (length {len})");
                    }
                }
                let esize = self
                    .info
                    .type_size(*elem)
                    .ok_or_else(|| anyhow!("element type has unknown size"))?;
                Ok(v.sub(*elem, i * esize, None))
            }
            TypeKind::Pointer { pointee, kind } => {
                // A Rust reference (or C++ reference) is indexed through
                // to its referent (`r[1]` for `r: &[i32; 3]` or
                // `&Vec<i32>`); raw pointers do pointer arithmetic.
                let is_ref = *kind != PtrKind::Pointer
                    || (t.lang == Lang::Rust
                        && t.name.as_deref().is_some_and(|n| n.starts_with('&')));
                if is_ref {
                    return self.index(&self.deref(v)?, i);
                }
                let pointee = pointee.ok_or_else(|| anyhow!("cannot index a `void *` pointer"))?;
                let esize = self
                    .info
                    .type_size(pointee)
                    .ok_or_else(|| anyhow!("pointee type has unknown size"))?;
                let addr = self.read_uint(v)?;
                Ok(Value::in_memory(pointee, addr + i * esize))
            }
            _ => bail!(
                "cannot index a value of type `{}`",
                self.info.type_name(v.ty)
            ),
        }
    }
}
