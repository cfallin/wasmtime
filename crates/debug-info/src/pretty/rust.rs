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
    let p = ctx.member(v, "data_ptr")?;
    let len = ctx.read_uint(&ctx.member(v, "length")?)?;
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
    let len = ctx.read_uint(&ctx.member(v, "len")?)?;
    let (ptr, _) = raw_vec_parts(ctx, &ctx.member(v, "buf")?)?;
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
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, len, elem) = vec_parts(ctx, &ctx.member(v, "vec")?)?;
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
            ctx.member(v, "inner")?
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
        let head = first_uint_leaf(ctx, &ctx.member(v, "head")?)?;
        let len = ctx.read_uint(&ctx.member(v, "len")?)?;
        let (ptr, cap) = raw_vec_parts(ctx, &ctx.member(v, "buf")?)?;
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
        let (addr, len, elem) = vec_parts(ctx, &ctx.member(v, "data")?)?;
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
            let (addr, len, elem) = vec_parts(ctx, &ctx.member(v, "data")?)?;
            check_index(i, len)?;
            Ok(element(ctx, elem, addr, i))
        })())
    }
}

/// `LinkedList<T>`: a doubly-linked list of `Node<T>`.
struct LinkedList;
impl LinkedList {
    fn elems(ctx: &Ctx<'_>, v: &Value, max: u64) -> Result<(Vec<Value>, u64)> {
        let len = ctx.read_uint(&ctx.member(v, "len")?)?;
        let mut out = vec![];
        let mut next = option_ptr(ctx, &ctx.member(v, "head")?)?;
        while let Some(p) = next {
            if out.len() as u64 >= max.min(len) {
                break;
            }
            let addr = ctx.read_uint(&p)?;
            let node = Value::in_memory(pointee(ctx, &p)?, addr);
            out.push(ctx.member(&node, "element")?);
            next = option_ptr(ctx, &ctx.member(&node, "next")?)?;
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
    let length = ctx.read_uint(&ctx.member(map, "length")?)?;
    let mut out = vec![];
    let root = match ctx.active_variant(&ctx.member(map, "root")?)? {
        Some((name, payload)) if name == "Some" => ctx.member(&payload, "__0")?,
        _ => return Ok((out, length)),
    };
    let height = ctx.read_uint(&ctx.member(&root, "height")?)?;
    let node_ptr = first_pointer_leaf(ctx, &ctx.member(&root, "node")?)?
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
            let len = self.ctx.read_uint(&self.ctx.member(&node, "len")?)?;
            let keys = self.ctx.member(&node, "keys")?.address().unwrap_or(0);
            let vals = self.ctx.member(&node, "vals")?.address().unwrap_or(0);
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
        let (entries, total) = btree_entries(ctx, &ctx.member(v, "map")?, opts.max_elems)?;
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
            entries.push((ctx.member(&e, "__0")?, ctx.member(&e, "__1")?));
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

/// The (strong, weak) counts and the contents of an `ArcInner`/`RcInner`.
/// The weak count excludes the implicit weak reference held by the
/// strong references collectively, matching `Rc::weak_count`.
fn rc_inner(ctx: &Ctx<'_>, ptr: &Value) -> Result<Option<(u64, u64, Value)>> {
    let addr = ctx.read_uint(ptr)?;
    let dangling = if ctx.ptr_size() >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * ctx.ptr_size())) - 1
    };
    if addr == 0 || addr == dangling {
        return Ok(None);
    }
    let inner = Value::in_memory(pointee(ctx, ptr)?, addr);
    let strong = first_uint_leaf(ctx, &ctx.member(&inner, "strong")?)?;
    let weak = first_uint_leaf(ctx, &ctx.member(&inner, "weak")?)?;
    let data = ctx
        .member(&inner, "data")
        .or_else(|_| ctx.member(&inner, "value"))?;
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
        let ptr = first_pointer_leaf(ctx, &ctx.member(v, "ptr")?)?
            .ok_or_else(|| anyhow!("no pointer in `{name}`"))?;
        let (strong, weak, data) =
            rc_inner(ctx, &ptr)?.ok_or_else(|| anyhow!("dangling `{name}`"))?;
        Ok(Summary::Wrap {
            prefix: rc_label(&name).into(),
            inner: data,
            note: Some(format!("strong={strong}, weak={weak}")),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some((|| {
            let ptr = first_pointer_leaf(ctx, &ctx.member(v, "ptr")?)?
                .ok_or_else(|| anyhow!("no pointer"))?;
            Ok(rc_inner(ctx, &ptr)?
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
        let ptr = first_pointer_leaf(ctx, &ctx.member(v, "ptr")?)?
            .ok_or_else(|| anyhow!("no pointer in Weak"))?;
        Ok(match rc_inner(ctx, &ptr)? {
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
                .find_map(|n| ctx.member(&cur, n).ok())
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
                inner: ctx.member(&payload, "__0")?,
                note: None,
            }),
            _ => Ok(Summary::Text("OnceCell(<uninit>)".into())),
        }
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some((|| {
            let opt = member_chain(ctx, v, &[&["inner"], &["value"]]);
            match ctx.active_variant(&opt)? {
                Some((name, payload)) if name == "Some" => ctx.member(&payload, "__0"),
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
