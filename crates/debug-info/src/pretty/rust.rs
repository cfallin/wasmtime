//! Pretty-printers for Rust standard-library types.
//!
//! These match on the qualified type names that rustc emits in DWARF
//! (e.g. `alloc::vec::Vec<i32, alloc::alloc::Global>`) and dig through
//! the library's internal layouts by member name, preferring robust
//! strategies (e.g. "first pointer field") where internals have changed
//! between Rust versions.

use super::*;
use crate::format::quote_bytes;
use anyhow::bail;

pub(super) static PRINTERS: &[&dyn PrettyPrinter] = &[
    &RustStr,
    &StrData,
    &TraitObject,
    &OnceLock,
    &LazyCell,
    &LazyLock,
    &Guard,
    &RustSlice,
    &RustString,
    &OsStrings,
    &CStrings,
    &RustVec,
    &VecDeque,
    &BinaryHeap,
    &LinkedList,
    &BTreeMap,
    &BTreeSet,
    &RustHashMap,
    &RustHashSet,
    &RustBox,
    &RefCounted,
    &RefCountedWeak,
    &Wrapper {
        name: "rust-mutex",
        label: "Mutex",
        matches: |n| {
            template_base(n).is_some_and(|b| b.starts_with("std::sync::") && b.ends_with("::Mutex"))
        },
        steps: &[("", &["data"]), ("core::cell::UnsafeCell<", &["value"])],
    },
    &Wrapper {
        name: "rust-rwlock",
        label: "RwLock",
        matches: |n| {
            template_base(n)
                .is_some_and(|b| b.starts_with("std::sync::") && b.ends_with("::RwLock"))
        },
        steps: &[("", &["data"]), ("core::cell::UnsafeCell<", &["value"])],
    },
    &Wrapper {
        name: "rust-refcell",
        label: "RefCell",
        matches: |n| is_one_of(n, &["core::cell::RefCell"]),
        steps: &[("", &["value"]), ("core::cell::UnsafeCell<", &["value"])],
    },
    &Wrapper {
        name: "rust-cell",
        label: "Cell",
        matches: |n| is_one_of(n, &["core::cell::Cell"]),
        steps: &[("", &["value"]), ("core::cell::UnsafeCell<", &["value"])],
    },
    &Wrapper {
        name: "rust-unsafecell",
        label: "UnsafeCell",
        matches: |n| is_one_of(n, &["core::cell::UnsafeCell"]),
        steps: &[("", &["value"])],
    },
    &Wrapper {
        name: "rust-syncunsafecell",
        label: "SyncUnsafeCell",
        matches: |n| is_one_of(n, &["core::cell::SyncUnsafeCell"]),
        steps: &[("", &["value"]), ("core::cell::UnsafeCell<", &["value"])],
    },
    &Wrapper {
        name: "rust-manuallydrop",
        label: "ManuallyDrop",
        matches: |n| is_one_of(n, &["core::mem::manually_drop::ManuallyDrop"]),
        steps: &[
            ("", &["value"]),
            ("core::mem::maybe_dangling::MaybeDangling<", &["__0"]),
        ],
    },
    &Wrapper {
        name: "rust-pin",
        label: "Pin",
        matches: |n| is_one_of(n, &["core::pin::Pin"]),
        steps: &[("", &["__pointer", "pointer"])],
    },
    &OnceCell,
    &Scalar {
        name: "rust-nonzero",
        matches: |n| is_one_of(n, &["core::num::nonzero::NonZero"]),
    },
    &Scalar {
        name: "rust-atomic",
        matches: |n| template_base(n).is_some_and(|b| b.starts_with("core::sync::atomic::Atomic")),
    },
];

/// Is this a Rust type?
fn rust(t: &Type) -> bool {
    t.lang == Lang::Rust
}

/// Read a fat pointer (`data_ptr`, `length`): (address, length, element
/// type).
fn fat_ptr(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, Option<TypeId>)> {
    let p = ctx.field(v, "data_ptr")?;
    let len = ctx.read_uint(&ctx.field(v, "length")?)?;
    let addr = ctx.read_uint(&p)?;
    Ok((addr, len, pointee(ctx, &p).ok()))
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

/// (data pointer, capacity) of a `RawVec`.
fn raw_vec_parts(ctx: &Ctx<'_>, buf: &Value) -> Result<(u64, u64)> {
    let ptr = first_pointer_leaf(ctx, buf)?.ok_or_else(|| anyhow!("RawVec has no pointer"))?;
    let cap = member_chain(ctx, buf, &[&["inner"], &["cap"]]);
    Ok((ctx.read_uint(&ptr)?, first_uint_leaf(ctx, &cap)?))
}

/// (data pointer, len, element type) of a `Vec<T>`.
fn vec_parts(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, TypeId)> {
    let elem = template_param(ctx, v.ty, "T")?;
    let len = ctx.read_uint(&ctx.field(v, "len")?)?;
    let (ptr, _) = raw_vec_parts(ctx, &ctx.field(v, "buf")?)?;
    Ok((ptr, len, elem))
}

/// If `v` is an `Option<P>` for a pointer-like `P`, the pointer value
/// inside it, if `Some`.
fn option_ptr(ctx: &Ctx<'_>, v: &Value) -> Result<Option<Value>> {
    match ctx.active_variant(v)? {
        Some((name, payload)) if name == "Some" => first_pointer_leaf(ctx, &payload),
        Some(_) => Ok(None),
        None => first_pointer_leaf(ctx, v),
    }
}

// ---------------------------------------------------------------------
// Strings.

/// `&str` (and `&mut str`, `Box<str>`).
struct RustStr;
impl PrettyPrinter for RustStr {
    fn name(&self) -> &'static str {
        "rust-str"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && (matches!(name, "&str" | "&mut str" | "*const str" | "*mut str")
                || (is_one_of(name, &["alloc::boxed::Box"])
                    && name.starts_with("alloc::boxed::Box<str")))
            && has_members(ty, &["data_ptr", "length"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (addr, len, _) = fat_ptr(ctx, v)?;
        let s = read_str(ctx, addr, len, opts.max_string, true)?;
        let name = ctx.info.ty(ctx.info.strip(v.ty)).name.clone();
        if name.is_some_and(|n| n.starts_with("alloc::boxed::Box<")) {
            Ok(Summary::Text(format!("Box({s})")))
        } else {
            Ok(Summary::Text(s))
        }
    }
}

/// `alloc::string::String`.
struct RustString;
impl PrettyPrinter for RustString {
    fn name(&self) -> &'static str {
        "rust-string"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && name == "alloc::string::String"
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let vec = ctx.field(v, "vec")?;
        let (addr, len, _) = vec_parts(ctx, &vec)?;
        Ok(Summary::Text(read_str(
            ctx,
            addr,
            len,
            opts.max_string,
            true,
        )?))
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, len, elem) = vec_parts(ctx, &ctx.field(v, "vec")?)?;
            check_index(i, len)?;
            Ok(element(ctx, elem, addr, i))
        })())
    }
}

/// `PathBuf`, `OsString` (owned, wrapping a `Vec<u8>`) and `&Path`,
/// `&OsStr` (fat pointers).
struct OsStrings;
impl PrettyPrinter for OsStrings {
    fn name(&self) -> &'static str {
        "rust-os-string"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && matches!(
                name,
                "std::path::PathBuf"
                    | "std::ffi::os_str::OsString"
                    | "&std::path::Path"
                    | "&std::ffi::os_str::OsStr"
            )
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let t = ctx.info.ty(ctx.info.strip(v.ty));
        let name = t.name.as_deref().unwrap_or("");
        let (addr, len) = if name.starts_with('&') {
            let (a, l, _) = fat_ptr(ctx, v)?;
            (a, l)
        } else {
            // PathBuf { inner: OsString { inner: Buf { inner: Vec<u8> } } }
            let vec = member_chain(ctx, v, &[&["inner"], &["inner"], &["inner"]]);
            let (a, l, _) = vec_parts(ctx, &vec)?;
            (a, l)
        };
        let s = read_str(ctx, addr, len, opts.max_string, true)?;
        let label = crate::format::rust_short_name(name.trim_start_matches('&'));
        Ok(Summary::Text(format!("{label}({s})")))
    }
}

/// `CString` (a `Box<[u8]>` including the NUL) and `&CStr`.
struct CStrings;
impl PrettyPrinter for CStrings {
    fn name(&self) -> &'static str {
        "rust-c-string"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && matches!(
                name,
                "alloc::ffi::c_str::CString" | "&core::ffi::c_str::CStr"
            )
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let fat = if has_members(&ctx.info.ty(ctx.info.strip(v.ty)), &["data_ptr", "length"]) {
            v.clone()
        } else {
            ctx.field(v, "inner")?
        };
        let (addr, len, _) = fat_ptr(ctx, &fat)?;
        // Omit the terminating NUL.
        let s = read_str(ctx, addr, len.saturating_sub(1), opts.max_string, true)?;
        Ok(Summary::Text(format!("c{s}")))
    }
}

// ---------------------------------------------------------------------
// Sequences.

/// `&[T]` and friends (fat pointers), including `Box<[T]>`.
struct RustSlice;
impl PrettyPrinter for RustSlice {
    fn name(&self) -> &'static str {
        "rust-slice"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && (name.starts_with("&[")
                || name.starts_with("&mut [")
                || name.starts_with("*const [")
                || name.starts_with("*mut [")
                || (is_one_of(name, &["alloc::boxed::Box"])
                    && name.starts_with("alloc::boxed::Box<[")))
            && has_members(ty, &["data_ptr", "length"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let (addr, len, elem) = fat_ptr(ctx, v)?;
        let elem = elem.ok_or_else(|| anyhow!("slice without element type"))?;
        let name = ctx.info.ty(ctx.info.strip(v.ty)).name.clone();
        let boxed = name.is_some_and(|n| n.starts_with("alloc::boxed::Box<"));
        Ok(Summary::Seq {
            prefix: if boxed { "Box(" } else { "&" }.into(),
            open: "[",
            close: if boxed { "])" } else { "]" },
            elem,
            addr,
            len,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, len, elem) = fat_ptr(ctx, v)?;
            let elem = elem.ok_or_else(|| anyhow!("slice without element type"))?;
            check_index(i, len)?;
            Ok(element(ctx, elem, addr, i))
        })())
    }
}

/// `alloc::vec::Vec<T>`.
struct RustVec;
impl PrettyPrinter for RustVec {
    fn name(&self) -> &'static str {
        "rust-vec"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::vec::Vec"])
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
            check_index(i, len)?;
            Ok(element(ctx, elem, addr, i))
        })())
    }
}

/// `VecDeque<T>`: a ring buffer.
struct VecDeque;
impl VecDeque {
    /// (buffer, capacity, head, len, element type).
    fn parts(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, u64, u64, TypeId)> {
        let elem = template_param(ctx, v.ty, "T")?;
        let head = first_uint_leaf(ctx, &ctx.field(v, "head")?)?;
        let len = ctx.read_uint(&ctx.field(v, "len")?)?;
        let (ptr, cap) = raw_vec_parts(ctx, &ctx.field(v, "buf")?)?;
        Ok((ptr, cap, head, len, elem))
    }
    fn get(ctx: &Ctx<'_>, v: &Value, i: u64) -> Result<Value> {
        let (ptr, cap, head, len, elem) = Self::parts(ctx, v)?;
        check_index(i, len)?;
        Ok(element(ctx, elem, ptr, (head + i) % cap.max(1)))
    }
}
impl PrettyPrinter for VecDeque {
    fn name(&self) -> &'static str {
        "rust-vecdeque"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::collections::vec_deque::VecDeque"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (ptr, cap, head, len, elem) = Self::parts(ctx, v)?;
        let n = len.min(opts.max_elems as u64);
        let elems = (0..n)
            .map(|i| element(ctx, elem, ptr, (head + i) % cap.max(1)))
            .collect();
        Ok(Summary::List {
            prefix: "VecDeque".into(),
            open: "[",
            close: "]",
            elems,
            total: len,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some(Self::get(ctx, v, i))
    }
}

/// `BinaryHeap<T>`: a `Vec<T>` in heap order.
struct BinaryHeap;
impl PrettyPrinter for BinaryHeap {
    fn name(&self) -> &'static str {
        "rust-binaryheap"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::collections::binary_heap::BinaryHeap"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let (addr, len, elem) = vec_parts(ctx, &ctx.field(v, "data")?)?;
        Ok(Summary::Seq {
            prefix: "BinaryHeap".into(),
            open: "[",
            close: "]",
            elem,
            addr,
            len,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, len, elem) = vec_parts(ctx, &ctx.field(v, "data")?)?;
            check_index(i, len)?;
            Ok(element(ctx, elem, addr, i))
        })())
    }
}

/// `LinkedList<T>`: a doubly-linked list of `Node<T>`.
struct LinkedList;
impl LinkedList {
    fn elems(ctx: &Ctx<'_>, v: &Value, max: u64) -> Result<(Vec<Value>, u64)> {
        let len = ctx.read_uint(&ctx.field(v, "len")?)?;
        let mut out = vec![];
        let mut next = option_ptr(ctx, &ctx.field(v, "head")?)?;
        while let Some(p) = next {
            if out.len() as u64 >= max.min(len) {
                break;
            }
            let addr = ctx.read_uint(&p)?;
            let node = Value::in_memory(pointee(ctx, &p)?, addr);
            out.push(ctx.field(&node, "element")?);
            next = option_ptr(ctx, &ctx.field(&node, "next")?)?;
        }
        Ok((out, len))
    }
}
impl PrettyPrinter for LinkedList {
    fn name(&self) -> &'static str {
        "rust-linkedlist"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::collections::linked_list::LinkedList"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (elems, total) = Self::elems(ctx, v, opts.max_elems as u64)?;
        Ok(Summary::List {
            prefix: "LinkedList".into(),
            open: "[",
            close: "]",
            elems,
            total,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (mut elems, len) = Self::elems(ctx, v, i + 1)?;
            check_index(i, len)?;
            elems
                .pop()
                .ok_or_else(|| anyhow!("list is shorter than its length"))
        })())
    }
}

// ---------------------------------------------------------------------
// Maps and sets.

/// In-order traversal of a `BTreeMap<K, V>`: up to `max` (key, value)
/// pairs, and the total length.
fn btree_entries(ctx: &Ctx<'_>, map: &Value, max: usize) -> Result<(Vec<(Value, Value)>, u64)> {
    let k = template_param(ctx, map.ty, "K")?;
    let v = template_param(ctx, map.ty, "V")?;
    let length = ctx.read_uint(&ctx.field(map, "length")?)?;
    let mut out = vec![];
    let root = match ctx.active_variant(&ctx.field(map, "root")?)? {
        Some((name, payload)) if name == "Some" => ctx.field(&payload, "__0")?,
        _ => return Ok((out, length)),
    };
    let height = ctx.read_uint(&ctx.field(&root, "height")?)?;
    let node_ptr = first_pointer_leaf(ctx, &ctx.field(&root, "node")?)?
        .ok_or_else(|| anyhow!("BTreeMap root has no node pointer"))?;
    let leaf_ty = pointee(ctx, &node_ptr)?;
    // Internal nodes are `#[repr(C)] InternalNode { data: LeafNode,
    // edges: [..] }`; find the edges' offset from DWARF if possible.
    let ptr_size = ctx.ptr_size();
    let edges_off = ctx
        .info
        .ty(leaf_ty)
        .name
        .as_ref()
        .and_then(|n| {
            let internal = n.replacen("LeafNode<", "InternalNode<", 1);
            ctx.info.find_type(&internal).ok().flatten()
        })
        .and_then(|t| struct_of(ctx, t))
        .and_then(|st| {
            st.members
                .iter()
                .find(|m| m.name.as_deref() == Some("edges"))
                .map(|m| m.byte_offset())
        })
        .unwrap_or_else(|| {
            let s = ctx.info.type_size(leaf_ty).unwrap_or(0);
            s.div_ceil(ptr_size) * ptr_size
        });
    let ksize = ctx.info.type_size(k).unwrap_or(0);
    let vsize = ctx.info.type_size(v).unwrap_or(0);

    struct Walk<'a, 'b> {
        ctx: &'a Ctx<'b>,
        leaf_ty: TypeId,
        edges_off: u64,
        ptr_size: u64,
        k: TypeId,
        v: TypeId,
        ksize: u64,
        vsize: u64,
        max: usize,
    }
    impl Walk<'_, '_> {
        fn walk(&self, addr: u64, height: u64, out: &mut Vec<(Value, Value)>) -> Result<()> {
            if out.len() >= self.max || height > 32 {
                return Ok(());
            }
            let node = Value::in_memory(self.leaf_ty, addr);
            let len = self.ctx.read_uint(&self.ctx.field(&node, "len")?)?;
            let keys = self.ctx.field(&node, "keys")?.address().unwrap_or(0);
            let vals = self.ctx.field(&node, "vals")?.address().unwrap_or(0);
            for i in 0..=len {
                if height > 0 {
                    let edge = self.ctx.target.read_memory(
                        0,
                        addr + self.edges_off + i * self.ptr_size,
                        self.ptr_size,
                    )?;
                    let mut b = [0u8; 8];
                    b[..edge.len()].copy_from_slice(&edge);
                    self.walk(u64::from_le_bytes(b), height - 1, out)?;
                }
                if i < len && out.len() < self.max {
                    out.push((
                        Value::in_memory(self.k, keys + i * self.ksize),
                        Value::in_memory(self.v, vals + i * self.vsize),
                    ));
                }
            }
            Ok(())
        }
    }
    let w = Walk {
        ctx,
        leaf_ty,
        edges_off,
        ptr_size,
        k,
        v,
        ksize,
        vsize,
        max,
    };
    w.walk(ctx.read_uint(&node_ptr)?, height, &mut out)?;
    Ok((out, length))
}

/// `BTreeMap<K, V>`.
struct BTreeMap;
impl PrettyPrinter for BTreeMap {
    fn name(&self) -> &'static str {
        "rust-btreemap"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::collections::btree::map::BTreeMap"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (entries, total) = btree_entries(ctx, v, opts.max_elems)?;
        Ok(Summary::Map {
            prefix: "BTreeMap".into(),
            entries,
            total,
            cpp: false,
        })
    }
}

/// `BTreeSet<T>`: a `BTreeMap<T, SetValZST>`.
struct BTreeSet;
impl PrettyPrinter for BTreeSet {
    fn name(&self) -> &'static str {
        "rust-btreeset"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::collections::btree::set::BTreeSet"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (entries, total) = btree_entries(ctx, &ctx.field(v, "map")?, opts.max_elems)?;
        Ok(Summary::Set {
            prefix: "BTreeSet".into(),
            elems: entries.into_iter().map(|(k, _)| k).collect(),
            total,
        })
    }
}

/// Iterate the full buckets of a hashbrown `RawTable<T>` value, returning
/// up to `max` element values and the total item count.
fn raw_table_elems(ctx: &Ctx<'_>, raw_table: &Value, max: usize) -> Result<(Vec<Value>, u64)> {
    let elem = template_param(ctx, raw_table.ty, "T")?;
    let inner = ctx.field(raw_table, "table")?;
    let bucket_mask = ctx.read_uint(&ctx.field(&inner, "bucket_mask")?)?;
    let items = ctx.read_uint(&ctx.field(&inner, "items")?)?;
    let ctrl_field = ctx.field(&inner, "ctrl")?;
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
        // Top bit clear means the bucket is full. Buckets are stored
        // backwards from the control bytes.
        if b & 0x80 == 0 {
            let addr = ctrl - (i as u64 + 1) * size;
            out.push(Value::in_memory(elem, addr));
        }
    }
    Ok((out, items))
}

/// Find the hashbrown `RawTable` inside a std or hashbrown `HashMap`.
fn map_raw_table(ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
    let base = ctx.field(v, "base").unwrap_or_else(|_| v.clone());
    ctx.field(&base, "table")
}

/// `std::collections::HashMap<K, V>` (and `hashbrown::HashMap`).
struct RustHashMap;
impl PrettyPrinter for RustHashMap {
    fn name(&self) -> &'static str {
        "rust-hashmap"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && is_one_of(
                name,
                &[
                    "std::collections::hash::map::HashMap",
                    "hashbrown::map::HashMap",
                ],
            )
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let table = map_raw_table(ctx, v)?;
        let (elems, total) = raw_table_elems(ctx, &table, opts.max_elems)?;
        let mut entries = vec![];
        for e in elems {
            entries.push((ctx.field(&e, "__0")?, ctx.field(&e, "__1")?));
        }
        Ok(Summary::Map {
            prefix: "HashMap".into(),
            entries,
            total,
            cpp: false,
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
        rust(ty)
            && is_one_of(
                name,
                &[
                    "std::collections::hash::set::HashSet",
                    "hashbrown::set::HashSet",
                ],
            )
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let base = ctx.field(v, "base").unwrap_or_else(|_| v.clone());
        let map = ctx.field(&base, "map")?;
        let table = map_raw_table(ctx, &map)?;
        let (elems, total) = raw_table_elems(ctx, &table, opts.max_elems)?;
        let mut keys = vec![];
        for e in elems {
            keys.push(ctx.field(&e, "__0")?);
        }
        Ok(Summary::Set {
            prefix: "HashSet".into(),
            elems: keys,
            total,
        })
    }
}

// ---------------------------------------------------------------------
// Pointers and wrappers.

/// `alloc::boxed::Box<T>` (a thin pointer for sized `T`).
struct RustBox;
impl PrettyPrinter for RustBox {
    fn name(&self) -> &'static str {
        "rust-box"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && is_one_of(name, &["alloc::boxed::Box"])
            && matches!(ty.kind, TypeKind::Pointer { .. })
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(Summary::Wrap {
            prefix: "Box".into(),
            inner: ctx.deref(v)?,
            note: None,
        })
    }
}

/// The target of a pointer that may be a fat pointer: a thin pointer to
/// a sized `T`, a `{data_ptr, length}` pointer to a slice or `str`, or a
/// `{pointer, vtable}` pointer to a trait object.
enum PtrTarget {
    Thin { addr: u64, ty: TypeId },
    Slice { addr: u64, ty: TypeId, len: u64 },
    Dyn { addr: u64, ty: TypeId, vtable: u64 },
}

impl PtrTarget {
    fn addr_and_type(&self) -> (u64, TypeId) {
        match *self {
            PtrTarget::Thin { addr, ty }
            | PtrTarget::Slice { addr, ty, .. }
            | PtrTarget::Dyn { addr, ty, .. } => (addr, ty),
        }
    }
}

/// Classify the pointer inside `v` (e.g. a `NonNull<T>` or raw pointer
/// field), which may be thin or fat, digging through single-field
/// wrappers such as `NonNull` and `Unique`.
fn ptr_target(ctx: &Ctx<'_>, v: &Value) -> Result<PtrTarget> {
    let mut cur = v.clone();
    for _ in 0..8 {
        let t = ctx.info.ty(ctx.info.strip(cur.ty));
        if let TypeKind::Pointer { .. } = t.kind {
            return Ok(PtrTarget::Thin {
                addr: ctx.read_uint(&cur)?,
                ty: pointee(ctx, &cur)?,
            });
        }
        if has_members(&t, &["data_ptr", "length"]) {
            let p = ctx.field(&cur, "data_ptr")?;
            return Ok(PtrTarget::Slice {
                addr: ctx.read_uint(&p)?,
                ty: pointee(ctx, &p)?,
                len: ctx.read_uint(&ctx.field(&cur, "length")?)?,
            });
        }
        if has_members(&t, &["pointer", "vtable"]) {
            let p = ctx.field(&cur, "pointer")?;
            return Ok(PtrTarget::Dyn {
                addr: ctx.read_uint(&p)?,
                ty: pointee(ctx, &p)?,
                vtable: ctx.read_uint(&ctx.field(&cur, "vtable")?)?,
            });
        }
        match &t.kind {
            TypeKind::Struct(st) if !st.members.is_empty() => {
                let m = &st.members[0];
                cur = cur.sub(m.ty, m.byte_offset(), None);
            }
            _ => break,
        }
    }
    bail!("no pointer in `{}`", ctx.info.type_name(v.ty))
}

/// The concrete object behind a trait-object pointer, using its vtable
/// to find the concrete type.
fn dyn_object(ctx: &Ctx<'_>, addr: u64, vtable: u64) -> Result<Value> {
    let ty = ctx
        .info
        .vtable_concrete_type(vtable)?
        .ok_or_else(|| anyhow!("unknown vtable {vtable:#x}"))?;
    Ok(Value::in_memory(ty, addr))
}

/// Read the `align` entry of a Rust vtable (`[drop, size, align, ...]`).
fn vtable_align(ctx: &Ctx<'_>, vtable: u64) -> Result<u64> {
    let ps = ctx.ptr_size();
    let b = ctx.target.read_memory(0, vtable + 2 * ps, ps)?;
    let mut buf = [0u8; 8];
    buf[..b.len()].copy_from_slice(&b);
    Ok(u64::from_le_bytes(buf))
}

/// The (strong, weak) counts and the contents of an `Arc`/`Rc`'s
/// `ArcInner`/`RcInner`. The weak count excludes the implicit weak
/// reference held by the strong references collectively, matching
/// `Rc::weak_count`. Handles unsized contents (`Arc<str>`, `Rc<[T]>`,
/// `Arc<dyn Trait>`).
fn rc_inner(ctx: &Ctx<'_>, rc: &Value) -> Result<Option<(u64, u64, Value)>> {
    let target = ptr_target(ctx, &ctx.field(rc, "ptr")?)?;
    let (addr, ty) = target.addr_and_type();
    let dangling = if ctx.ptr_size() >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * ctx.ptr_size())) - 1
    };
    if addr == 0 || addr == dangling {
        return Ok(None);
    }
    let inner = Value::in_memory(ty, addr);
    let strong = first_uint_leaf(ctx, &ctx.field(&inner, "strong")?)?;
    let weak = first_uint_leaf(ctx, &ctx.field(&inner, "weak")?)?;
    let data = ctx
        .member(&inner, "data")
        .or_else(|_| ctx.field(&inner, "value"))?;
    let data = match target {
        PtrTarget::Thin { .. } => data,
        PtrTarget::Slice { len, .. } => {
            let elem = match ctx.info.ty(ctx.info.strip(data.ty)).kind {
                TypeKind::Array { elem, .. } => elem,
                _ => bail!("unexpected unsized `Rc`/`Arc` contents"),
            };
            let is_str = ctx
                .info
                .ty(ty)
                .name
                .as_deref()
                .is_some_and(|n| n.ends_with("<str>"));
            let arr = if is_str {
                ctx.info.named_array_of(elem, len, "str")
            } else {
                ctx.info.array_of(elem, len)
            };
            Value::in_memory(arr, data.address().unwrap_or(0))
        }
        PtrTarget::Dyn { vtable, .. } => {
            // The payload follows the two counts, aligned for the
            // concrete type.
            let counts = 2 * ctx.ptr_size();
            let align = vtable_align(ctx, vtable)?.max(1);
            dyn_object(ctx, addr + counts.div_ceil(align) * align, vtable)?
        }
    };
    let weak = if strong > 0 {
        weak.saturating_sub(1)
    } else {
        weak
    };
    Ok(Some((strong, weak, data)))
}

fn rc_label(name: &str) -> &'static str {
    if name.starts_with("alloc::sync::") {
        "Arc"
    } else {
        "Rc"
    }
}

/// `Arc<T>` and `Rc<T>`.
struct RefCounted;
impl PrettyPrinter for RefCounted {
    fn name(&self) -> &'static str {
        "rust-rc"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::sync::Arc", "alloc::rc::Rc"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let name = match_name(&ctx.info.ty(ctx.info.strip(v.ty)));
        let (strong, weak, data) = rc_inner(ctx, v)?.ok_or_else(|| anyhow!("dangling `{name}`"))?;
        Ok(Summary::Wrap {
            prefix: rc_label(&name).into(),
            inner: data,
            note: Some(format!("strong={strong}, weak={weak}")),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some((|| {
            Ok(rc_inner(ctx, v)?
                .ok_or_else(|| anyhow!("dangling pointer"))?
                .2)
        })())
    }
}

/// `sync::Weak<T>` and `rc::Weak<T>`.
struct RefCountedWeak;
impl PrettyPrinter for RefCountedWeak {
    fn name(&self) -> &'static str {
        "rust-weak"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["alloc::sync::Weak", "alloc::rc::Weak"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(match rc_inner(ctx, v)? {
            Some((strong, weak, data)) if strong > 0 => Summary::Wrap {
                prefix: "Weak".into(),
                inner: data,
                // Count this `Weak` itself among the weak references.
                note: Some(format!("strong={strong}, weak={weak}")),
            },
            Some(_) => Summary::Text("Weak(<dropped>)".into()),
            None => Summary::Text("Weak(<dangling>)".into()),
        })
    }
}

/// A transparent wrapper (cells, locks, `ManuallyDrop`, `Pin`): shown as
/// `Label(contents)`, and dereferenced (for `*`, `.field` and `[i]`) to
/// its contents. `steps` lead from the wrapper to the contents; each is
/// (required type-name prefix of the current value, or "" for any;
/// alternative member names).
struct Wrapper {
    name: &'static str,
    label: &'static str,
    matches: fn(&str) -> bool,
    steps: &'static [(&'static str, &'static [&'static str])],
}
impl Wrapper {
    fn contents(&self, ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
        let mut cur = v.clone();
        for (guard, names) in self.steps {
            let tn = ctx
                .info
                .ty(ctx.info.strip(cur.ty))
                .name
                .clone()
                .unwrap_or_default();
            if !tn.starts_with(guard) {
                continue;
            }
            cur = names
                .iter()
                .find_map(|n| ctx.field(&cur, n).ok())
                .ok_or_else(|| anyhow!("unexpected layout of `{tn}`"))?;
        }
        Ok(cur)
    }
}
impl PrettyPrinter for Wrapper {
    fn name(&self) -> &'static str {
        self.name
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && (self.matches)(name)
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(Summary::Wrap {
            prefix: self.label.into(),
            inner: self.contents(ctx, v)?,
            note: None,
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(self.contents(ctx, v))
    }
}

/// `OnceCell<T>`: an `UnsafeCell<Option<T>>`.
struct OnceCell;
impl PrettyPrinter for OnceCell {
    fn name(&self) -> &'static str {
        "rust-oncecell"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["core::cell::once::OnceCell"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let opt = member_chain(ctx, v, &[&["inner"], &["value"]]);
        match ctx.active_variant(&opt)? {
            Some((name, payload)) if name == "Some" => Ok(Summary::Wrap {
                prefix: "OnceCell".into(),
                inner: ctx.field(&payload, "__0")?,
                note: None,
            }),
            _ => Ok(Summary::Text("OnceCell(<uninit>)".into())),
        }
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some((|| {
            let opt = member_chain(ctx, v, &[&["inner"], &["value"]]);
            match ctx.active_variant(&opt)? {
                Some((name, payload)) if name == "Some" => ctx.field(&payload, "__0"),
                _ => bail!("OnceCell is uninitialized"),
            }
        })())
    }
}

/// Wrappers around a single scalar (`NonZero<T>`, atomics), shown as
/// just the scalar.
struct Scalar {
    name: &'static str,
    matches: fn(&str) -> bool,
}
impl Scalar {
    fn contents(ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
        super::leaf(ctx, v, 0, &|k| {
            matches!(k, TypeKind::Base(_) | TypeKind::Pointer { .. })
        })?
        .ok_or_else(|| anyhow!("no scalar in `{}`", ctx.info.type_name(v.ty)))
    }
}
impl PrettyPrinter for Scalar {
    fn name(&self) -> &'static str {
        self.name
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && (self.matches)(name)
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(Summary::Wrap {
            prefix: String::new(),
            inner: Self::contents(ctx, v)?,
            note: None,
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(Self::contents(ctx, v))
    }
}

/// The contents of a synthesized `str` (e.g. inside an `Arc<str>`).
struct StrData;
impl PrettyPrinter for StrData {
    fn name(&self) -> &'static str {
        "rust-str-data"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && name == "str" && matches!(ty.kind, TypeKind::Array { .. })
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let addr = v.address().ok_or_else(|| anyhow!("str not in memory"))?;
        let len = ctx.info.type_size(v.ty).unwrap_or(0);
        Ok(Summary::Text(read_str(
            ctx,
            addr,
            len,
            opts.max_string,
            true,
        )?))
    }
}

/// Trait objects: `Box<dyn Trait>`, `&dyn Trait`, `&mut dyn Trait`, and
/// raw `dyn` pointers. The concrete type is found via the vtable.
struct TraitObject;
impl TraitObject {
    fn object(ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
        match ptr_target(ctx, v)? {
            PtrTarget::Dyn { addr, vtable, .. } => dyn_object(ctx, addr, vtable),
            _ => bail!("not a trait object"),
        }
    }
}
impl PrettyPrinter for TraitObject {
    fn name(&self) -> &'static str {
        "rust-dyn"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty)
            && has_members(ty, &["pointer", "vtable"])
            && (name.starts_with("&dyn ")
                || name.starts_with("&mut dyn ")
                || name.starts_with("*const dyn ")
                || name.starts_with("*mut dyn ")
                || (is_one_of(name, &["alloc::boxed::Box"])
                    && name.starts_with("alloc::boxed::Box<dyn ")))
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let name = match_name(&ctx.info.ty(ctx.info.strip(v.ty)));
        let inner = Self::object(ctx, v)?;
        Ok(if name.starts_with("alloc::boxed::Box<") {
            Summary::Wrap {
                prefix: "Box".into(),
                inner,
                note: None,
            }
        } else if name.starts_with('&') {
            Summary::Prefixed {
                prefix: "&".into(),
                inner,
            }
        } else {
            let addr = inner.address().unwrap_or(0);
            Summary::Text(format!("({name}) {addr:#x}"))
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(Self::object(ctx, v))
    }
}

/// Whether a `std::sync::Once` has completed: the single-threaded
/// implementation uses a `State` enum, and the futex/queue
/// implementations use 0 for "complete".
fn once_completed(ctx: &Ctx<'_>, once: &Value) -> Result<bool> {
    let leaf = super::leaf(ctx, once, 0, &|k| {
        matches!(
            k,
            TypeKind::Enum { .. } | TypeKind::Base(_) | TypeKind::Pointer { .. }
        )
    })?
    .ok_or_else(|| anyhow!("unexpected `Once` layout"))?;
    let raw = ctx.read_uint(&leaf)?;
    match &ctx.info.ty(ctx.info.strip(leaf.ty)).kind {
        TypeKind::Enum { enumerators, .. } => Ok(enumerators
            .iter()
            .any(|(n, x)| n == "Complete" && *x == i128::from(raw))),
        _ => Ok(raw == 0),
    }
}

/// Look through `UnsafeCell`, `MaybeUninit` and `ManuallyDrop`/
/// `MaybeDangling` wrappers to the stored value.
fn unwrap_storage(ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
    let mut cur = v.clone();
    for _ in 0..8 {
        let name = match_name(&ctx.info.ty(ctx.info.strip(cur.ty)));
        cur = match template_base(&name) {
            Some("core::cell::UnsafeCell")
            | Some("core::mem::maybe_uninit::MaybeUninit")
            | Some("core::mem::manually_drop::ManuallyDrop") => ctx.field(&cur, "value")?,
            Some("core::mem::maybe_dangling::MaybeDangling") => ctx.field(&cur, "__0")?,
            _ => return Ok(cur),
        };
    }
    Ok(cur)
}

/// `OnceLock<T>`: a `Once` plus possibly-uninitialized storage.
struct OnceLock;
impl OnceLock {
    fn contents(ctx: &Ctx<'_>, v: &Value) -> Result<Option<Value>> {
        if !once_completed(ctx, &ctx.field(v, "once")?)? {
            return Ok(None);
        }
        Ok(Some(unwrap_storage(ctx, &ctx.field(v, "value")?)?))
    }
}
impl PrettyPrinter for OnceLock {
    fn name(&self) -> &'static str {
        "rust-oncelock"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["std::sync::once_lock::OnceLock"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(match Self::contents(ctx, v)? {
            Some(inner) => Summary::Wrap {
                prefix: "OnceLock".into(),
                inner,
                note: None,
            },
            None => Summary::Text("OnceLock(<uninit>)".into()),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(
            Self::contents(ctx, v)
                .and_then(|c| c.ok_or_else(|| anyhow!("OnceLock is uninitialized"))),
        )
    }
}

/// `LazyCell<T, F>`: an `UnsafeCell<State<T, F>>` where `State` is one
/// of `Uninit(F)`, `Init(T)` or `Poisoned`.
struct LazyCell;
impl LazyCell {
    fn state(ctx: &Ctx<'_>, v: &Value) -> Result<(String, Option<Value>)> {
        let state = unwrap_storage(ctx, &ctx.field(v, "state")?)?;
        let (name, payload) = ctx
            .active_variant(&state)?
            .ok_or_else(|| anyhow!("unexpected LazyCell layout"))?;
        let value = if name == "Init" {
            Some(ctx.field(&payload, "__0")?)
        } else {
            None
        };
        Ok((name, value))
    }
}
impl PrettyPrinter for LazyCell {
    fn name(&self) -> &'static str {
        "rust-lazycell"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["core::cell::lazy::LazyCell"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(match Self::state(ctx, v)? {
            (_, Some(inner)) => Summary::Wrap {
                prefix: "LazyCell".into(),
                inner,
                note: None,
            },
            (name, None) if name == "Poisoned" => Summary::Text("LazyCell(<poisoned>)".into()),
            _ => Summary::Text("LazyCell(<uninit>)".into()),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(
            Self::state(ctx, v)
                .and_then(|(_, c)| c.ok_or_else(|| anyhow!("LazyCell is uninitialized"))),
        )
    }
}

/// `LazyLock<T, F>`: a `Once` plus a union of the value and the
/// initializer.
struct LazyLock;
impl LazyLock {
    fn contents(ctx: &Ctx<'_>, v: &Value) -> Result<Option<Value>> {
        if !once_completed(ctx, &ctx.field(v, "once")?)? {
            return Ok(None);
        }
        let data = unwrap_storage(ctx, &ctx.field(v, "data")?)?;
        Ok(Some(unwrap_storage(ctx, &ctx.field(&data, "value")?)?))
    }
}
impl PrettyPrinter for LazyLock {
    fn name(&self) -> &'static str {
        "rust-lazylock"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && is_one_of(name, &["std::sync::lazy_lock::LazyLock"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(match Self::contents(ctx, v)? {
            Some(inner) => Summary::Wrap {
                prefix: "LazyLock".into(),
                inner,
                note: None,
            },
            None => Summary::Text("LazyLock(<uninit>)".into()),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(
            Self::contents(ctx, v)
                .and_then(|c| c.ok_or_else(|| anyhow!("LazyLock is uninitialized"))),
        )
    }
}

/// Lock and borrow guards: `MutexGuard`, `RwLockReadGuard`,
/// `RwLockWriteGuard` (and mapped variants), `Ref` and `RefMut`. Shown
/// as `Guard(contents)` and dereferenced to the protected data.
struct Guard;
impl Guard {
    fn label(name: &str) -> Option<&str> {
        let base = template_base(name)?;
        let short = base.rsplit("::").next()?;
        let ok = if base.starts_with("std::sync::") {
            matches!(
                short,
                "MutexGuard"
                    | "MappedMutexGuard"
                    | "RwLockReadGuard"
                    | "RwLockWriteGuard"
                    | "MappedRwLockReadGuard"
                    | "MappedRwLockWriteGuard"
            )
        } else {
            matches!(base, "core::cell::Ref" | "core::cell::RefMut")
        };
        ok.then_some(short)
    }

    fn contents(ctx: &Ctx<'_>, v: &Value) -> Result<Value> {
        if let Ok(lock) = ctx.field(v, "lock") {
            // `&Mutex<T>` / `&RwLock<T>`: the lock's contents.
            let lock = ctx.deref(&lock)?;
            return ctx.deref(&lock);
        }
        let ptr = ctx.field(v, "data").or_else(|_| ctx.field(v, "value"))?;
        match ptr_target(ctx, &ptr)? {
            PtrTarget::Thin { addr, ty } => Ok(Value::in_memory(ty, addr)),
            PtrTarget::Dyn { addr, vtable, .. } => dyn_object(ctx, addr, vtable),
            PtrTarget::Slice { addr, ty, len } => {
                Ok(Value::in_memory(ctx.info.array_of(ty, len), addr))
            }
        }
    }
}
impl PrettyPrinter for Guard {
    fn name(&self) -> &'static str {
        "rust-guard"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        rust(ty) && Self::label(name).is_some()
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let name = match_name(&ctx.info.ty(ctx.info.strip(v.ty)));
        Ok(Summary::Wrap {
            prefix: Self::label(&name).unwrap_or("Guard").into(),
            inner: Self::contents(ctx, v)?,
            note: None,
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(Self::contents(ctx, v))
    }
}
