//! Pretty-printers for well-known standard-library types.
//!
//! This is a deliberately small, closed framework: each printer matches
//! on a type's (qualified) name and/or structure and produces a
//! [`Summary`], which the generic formatter renders. Printers may also
//! provide synthetic indexing (so that `v[2]` works on a `Vec`) and
//! dereference.

use crate::format::{FormatOptions, quote_bytes};
use crate::types::*;
use crate::value::{Ctx, Value};
use anyhow::{Result, anyhow, bail};

/// A pretty-printer's view of a value.
pub enum Summary {
    /// A complete textual rendering (e.g., a string literal).
    Text(String),
    /// A sequence of `len` elements of type `elem` stored contiguously
    /// at `addr`, rendered as `prefix open e0, e1, ... close`.
    Seq {
        prefix: String,
        open: &'static str,
        close: &'static str,
        elem: TypeId,
        addr: u64,
        len: u64,
    },
    /// Key/value entries (only the first few may be included); `total`
    /// is the full number of entries.
    Map {
        prefix: String,
        entries: Vec<(Value, Value)>,
        total: u64,
    },
    /// Set elements (only the first few may be included).
    Set {
        prefix: String,
        elems: Vec<Value>,
        total: u64,
    },
    /// A wrapper around one inner value, rendered `prefix(inner)`.
    Wrap { prefix: String, inner: Value },
}

/// A pretty-printer for a family of types.
pub trait PrettyPrinter: Sync {
    /// Name of this printer (for diagnostics / `info pretty-printers`).
    fn name(&self) -> &'static str;

    /// Whether this printer handles type `ty` (whose display name is
    /// `name`, if any).
    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool;

    /// Summarize the value.
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary>;

    /// Synthetic indexing, if supported.
    fn index(&self, _ctx: &Ctx<'_>, _v: &Value, _i: u64) -> Option<Result<Value>> {
        None
    }

    /// Synthetic dereference, if supported.
    fn deref(&self, _ctx: &Ctx<'_>, _v: &Value) -> Option<Result<Value>> {
        None
    }
}

static PRINTERS: &[&dyn PrettyPrinter] = &[
    &RustStr,
    &RustSlice,
    &RustString,
    &RustVec,
    &RustBox,
    &RustHashMap,
    &RustHashSet,
    &CString,
    &CCharArray,
];

/// All registered printers.
pub fn printers() -> &'static [&'static dyn PrettyPrinter] {
    PRINTERS
}

/// Find a printer for `ty` (trying the type as written, then with
/// typedefs and qualifiers stripped).
pub fn find(ctx: &Ctx<'_>, ty: TypeId) -> Option<&'static dyn PrettyPrinter> {
    let mut cands = vec![ty];
    let s = ctx.info.strip(ty);
    if s != ty {
        cands.push(s);
    }
    for id in cands {
        let t = ctx.info.ty(id);
        let name = t.name.clone().unwrap_or_default();
        for p in PRINTERS {
            if p.matches(ctx, &t, &name) {
                return Some(*p);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------
// Helpers.

/// Find the first pointer-typed leaf inside a value (depth-first through
/// struct members). Used to dig the data pointer out of
/// `RawVec`/`Unique`/`NonNull` wrappers, whose layouts vary between Rust
/// versions.
pub fn first_pointer_leaf(ctx: &Ctx<'_>, v: &Value) -> Result<Option<Value>> {
    first_pointer_leaf_depth(ctx, v, 0)
}

fn first_pointer_leaf_depth(ctx: &Ctx<'_>, v: &Value, depth: usize) -> Result<Option<Value>> {
    if depth > 8 {
        return Ok(None);
    }
    let t = ctx.info.ty(ctx.info.strip(v.ty));
    match &t.kind {
        TypeKind::Pointer { .. } => Ok(Some(v.clone())),
        TypeKind::Struct(st) => {
            for m in &st.members {
                let mv = v.sub(m.ty, m.byte_offset(), None);
                if let Some(p) = first_pointer_leaf_depth(ctx, &mv, depth + 1)? {
                    return Ok(Some(p));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn struct_of(ctx: &Ctx<'_>, ty: TypeId) -> Option<StructType> {
    match &ctx.info.ty(ctx.info.strip(ty)).kind {
        TypeKind::Struct(st) => Some(st.clone()),
        _ => None,
    }
}

fn has_members(t: &Type, names: &[&str]) -> bool {
    match &t.kind {
        TypeKind::Struct(st) => names
            .iter()
            .all(|n| st.members.iter().any(|m| m.name.as_deref() == Some(*n))),
        _ => false,
    }
}

/// Read a fat pointer (`data_ptr`, `length`).
fn fat_ptr(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, Option<TypeId>)> {
    let p = ctx.member(v, "data_ptr")?;
    let len = ctx.read_uint(&ctx.member(v, "length")?)?;
    let addr = ctx.read_uint(&p)?;
    let pointee = match ctx.info.ty(ctx.info.strip(p.ty)).kind {
        TypeKind::Pointer { pointee, .. } => pointee,
        _ => None,
    };
    Ok((addr, len, pointee))
}

/// Read up to `max` bytes at `addr` as a (possibly truncated) string.
fn read_str(ctx: &Ctx<'_>, addr: u64, len: u64, max: usize, rust: bool) -> Result<String> {
    let n = len.min(max as u64);
    let bytes = if n == 0 {
        vec![]
    } else {
        ctx.target.read_memory(0, addr, n)?
    };
    let mut s = quote_bytes(&bytes, rust);
    if n < len {
        s.push_str("...");
    }
    Ok(s)
}

/// (data pointer, len, element type) of a Rust `Vec<T>`.
fn vec_parts(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, TypeId)> {
    let st = struct_of(ctx, v.ty).ok_or_else(|| anyhow!("Vec is not a struct"))?;
    let elem = st
        .template_param("T")
        .ok_or_else(|| anyhow!("Vec has no `T` parameter"))?;
    let len = ctx.read_uint(&ctx.member(v, "len")?)?;
    let buf = ctx.member(v, "buf")?;
    let ptr = first_pointer_leaf(ctx, &buf)?.ok_or_else(|| anyhow!("Vec has no data pointer"))?;
    Ok((ctx.read_uint(&ptr)?, len, elem))
}

fn char_like(ctx: &Ctx<'_>, ty: Option<TypeId>) -> bool {
    let Some(ty) = ty else { return false };
    matches!(
        ctx.info.ty(ctx.info.strip(ty)).kind,
        TypeKind::Base(BaseEnc::SignedChar | BaseEnc::UnsignedChar)
    )
}

// ---------------------------------------------------------------------
// Rust printers.

/// `&str` (and `&mut str`, `Box<str>`).
struct RustStr;
impl PrettyPrinter for RustStr {
    fn name(&self) -> &'static str {
        "rust-str"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust && matches!(name, "&str" | "&mut str" | "*const str" | "*mut str")
            || name.starts_with("alloc::boxed::Box<str") && has_members(ty, &["data_ptr", "length"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (addr, len, _) = fat_ptr(ctx, v)?;
        Ok(Summary::Text(read_str(
            ctx,
            addr,
            len,
            opts.max_string,
            true,
        )?))
    }
}

/// `&[T]` and friends.
struct RustSlice;
impl PrettyPrinter for RustSlice {
    fn name(&self) -> &'static str {
        "rust-slice"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust
            && (name.starts_with("&[")
                || name.starts_with("&mut [")
                || name.starts_with("*const [")
                || name.starts_with("*mut [")
                || name.starts_with("alloc::boxed::Box<["))
            && has_members(ty, &["data_ptr", "length"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let (addr, len, elem) = fat_ptr(ctx, v)?;
        let elem = elem.ok_or_else(|| anyhow!("slice without element type"))?;
        Ok(Summary::Seq {
            prefix: "&".into(),
            open: "[",
            close: "]",
            elem,
            addr,
            len,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, len, elem) = fat_ptr(ctx, v)?;
            let elem = elem.ok_or_else(|| anyhow!("slice without element type"))?;
            if i >= len {
                bail!("index {i} out of bounds (length {len})");
            }
            let size = ctx.info.type_size(elem).unwrap_or(0);
            Ok(Value::in_memory(elem, addr + i * size))
        })())
    }
}

/// `alloc::string::String`.
struct RustString;
impl PrettyPrinter for RustString {
    fn name(&self) -> &'static str {
        "rust-string"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust && name == "alloc::string::String"
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let vec = ctx.member(v, "vec")?;
        let (addr, len, _) = vec_parts(ctx, &vec)?;
        Ok(Summary::Text(read_str(
            ctx,
            addr,
            len,
            opts.max_string,
            true,
        )?))
    }
}

/// `alloc::vec::Vec<T>`.
struct RustVec;
impl PrettyPrinter for RustVec {
    fn name(&self) -> &'static str {
        "rust-vec"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust && name.starts_with("alloc::vec::Vec<")
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let (addr, len, elem) = vec_parts(ctx, v)?;
        Ok(Summary::Seq {
            prefix: "vec!".into(),
            open: "[",
            close: "]",
            elem,
            addr,
            len,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, len, elem) = vec_parts(ctx, v)?;
            if i >= len {
                bail!("index {i} out of bounds (length {len})");
            }
            let size = ctx.info.type_size(elem).unwrap_or(0);
            Ok(Value::in_memory(elem, addr + i * size))
        })())
    }
}

/// `alloc::boxed::Box<T>` (a thin pointer for sized `T`).
struct RustBox;
impl PrettyPrinter for RustBox {
    fn name(&self) -> &'static str {
        "rust-box"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust
            && name.starts_with("alloc::boxed::Box<")
            && matches!(ty.kind, TypeKind::Pointer { .. })
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(Summary::Wrap {
            prefix: "Box".into(),
            inner: ctx.deref(v)?,
        })
    }
}

/// Iterate the full buckets of a hashbrown `RawTable<T>` value, returning
/// up to `max` element values and the total item count.
fn raw_table_elems(ctx: &Ctx<'_>, raw_table: &Value, max: usize) -> Result<(Vec<Value>, u64)> {
    let st = struct_of(ctx, raw_table.ty).ok_or_else(|| anyhow!("RawTable is not a struct"))?;
    let elem = st
        .template_param("T")
        .ok_or_else(|| anyhow!("RawTable has no `T` parameter"))?;
    let inner = ctx.member(raw_table, "table")?;
    let bucket_mask = ctx.read_uint(&ctx.member(&inner, "bucket_mask")?)?;
    let items = ctx.read_uint(&ctx.member(&inner, "items")?)?;
    let ctrl_field = ctx.member(&inner, "ctrl")?;
    let ctrl = first_pointer_leaf(ctx, &ctrl_field)?
        .ok_or_else(|| anyhow!("RawTableInner has no ctrl pointer"))?;
    let ctrl = ctx.read_uint(&ctrl)?;
    if items == 0 {
        return Ok((vec![], 0));
    }
    let buckets = bucket_mask + 1;
    if buckets > 1 << 24 {
        bail!("implausible hash table size {buckets}");
    }
    let size = ctx
        .info
        .type_size(elem)
        .ok_or_else(|| anyhow!("hash table element has unknown size"))?;
    let ctrl_bytes = ctx.target.read_memory(0, ctrl, buckets)?;
    let mut out = vec![];
    for (i, b) in ctrl_bytes.iter().enumerate() {
        if out.len() >= max {
            break;
        }
        // Top bit clear means the bucket is full.
        if b & 0x80 == 0 {
            let addr = ctrl - (i as u64 + 1) * size;
            out.push(Value::in_memory(elem, addr));
        }
    }
    Ok((out, items))
}

/// Find the hashbrown `RawTable` inside a std or hashbrown `HashMap`.
fn map_raw_table(ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
    let base = ctx.member(v, "base").unwrap_or_else(|_| v.clone());
    ctx.member(&base, "table")
}

/// `std::collections::HashMap<K, V>` (and `hashbrown::HashMap`).
struct RustHashMap;
impl PrettyPrinter for RustHashMap {
    fn name(&self) -> &'static str {
        "rust-hashmap"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust
            && (name.starts_with("std::collections::hash::map::HashMap<")
                || name.starts_with("hashbrown::map::HashMap<"))
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let table = map_raw_table(ctx, v)?;
        let (elems, total) = raw_table_elems(ctx, &table, opts.max_elems)?;
        let mut entries = vec![];
        for e in elems {
            entries.push((ctx.member(&e, "__0")?, ctx.member(&e, "__1")?));
        }
        Ok(Summary::Map {
            prefix: "HashMap".into(),
            entries,
            total,
        })
    }
}

/// `std::collections::HashSet<T>` (and `hashbrown::HashSet`).
struct RustHashSet;
impl PrettyPrinter for RustHashSet {
    fn name(&self) -> &'static str {
        "rust-hashset"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        ty.lang == Lang::Rust
            && (name.starts_with("std::collections::hash::set::HashSet<")
                || name.starts_with("hashbrown::set::HashSet<"))
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let base = ctx.member(v, "base").unwrap_or_else(|_| v.clone());
        let map = ctx.member(&base, "map")?;
        let table = map_raw_table(ctx, &map)?;
        let (elems, total) = raw_table_elems(ctx, &table, opts.max_elems)?;
        let mut keys = vec![];
        for e in elems {
            keys.push(ctx.member(&e, "__0")?);
        }
        Ok(Summary::Set {
            prefix: "HashSet".into(),
            elems: keys,
            total,
        })
    }
}

// ---------------------------------------------------------------------
// C printers.

/// `char *` / `const char *` (and signed/unsigned variants): show the
/// pointer and the NUL-terminated string it points to.
struct CString;
impl PrettyPrinter for CString {
    fn name(&self) -> &'static str {
        "c-string"
    }
    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, _name: &str) -> bool {
        ty.lang.is_c_like()
            && match ty.kind {
                TypeKind::Pointer {
                    pointee,
                    kind: PtrKind::Pointer,
                } => char_like(ctx, pointee),
                _ => false,
            }
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let addr = ctx.read_uint(v)?;
        if addr == 0 {
            return Ok(Summary::Text("0x0".into()));
        }
        let mut bytes = vec![];
        let mut truncated = true;
        let mut cur = addr;
        'outer: while bytes.len() < opts.max_string {
            let chunk = match ctx.target.read_memory(0, cur, 64) {
                Ok(c) => c,
                Err(_) => {
                    // Possibly near the end of memory: fall back to
                    // byte-at-a-time reads.
                    match ctx.target.read_memory(0, cur, 1) {
                        Ok(c) => c,
                        Err(_) => {
                            return Ok(Summary::Text(format!("{addr:#x} <error reading string>")));
                        }
                    }
                }
            };
            for b in chunk {
                if b == 0 {
                    truncated = false;
                    break 'outer;
                }
                bytes.push(b);
                if bytes.len() >= opts.max_string {
                    break 'outer;
                }
            }
            cur += 64;
        }
        let mut s = quote_bytes(&bytes, false);
        if truncated {
            s.push_str("...");
        }
        Ok(Summary::Text(format!("{addr:#x} {s}")))
    }
}

/// `char[N]`: show as a string up to the first NUL.
struct CCharArray;
impl PrettyPrinter for CCharArray {
    fn name(&self) -> &'static str {
        "c-char-array"
    }
    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, _name: &str) -> bool {
        ty.lang.is_c_like()
            && match ty.kind {
                TypeKind::Array { elem, len: Some(_) } => char_like(ctx, Some(elem)),
                _ => false,
            }
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let bytes = ctx.contents(v)?;
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        let n = end.min(opts.max_string);
        let mut s = quote_bytes(&bytes[..n], false);
        if n < end {
            s.push_str("...");
        }
        Ok(Summary::Text(s))
    }
}
