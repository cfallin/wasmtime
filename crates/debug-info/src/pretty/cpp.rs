//! Pretty-printers for C++ standard-library types, for LLVM's libc++
//! (as shipped with wasi-sdk).
//!
//! Template instantiations are handled generically: element and node
//! types are recovered from DWARF (pointer member types, template
//! parameters, and node types found by name), so no per-instantiation
//! knowledge is needed.

use super::*;
use crate::format::quote_bytes;
use anyhow::bail;

pub(super) static PRINTERS: &[&dyn PrettyPrinter] =
    &[&String, &Vector, &Tree, &HashTable, &UniquePtr, &SharedPtr];

fn cpp(t: &Type) -> bool {
    t.lang == Lang::Cpp
}

/// Find a libc++ internal type (e.g. `__tree_node<`) satisfying `pred`,
/// whichever inline ABI namespace it lives in.
fn find_libcxx_type(ctx: &Ctx<'_>, name: &str, pred: &dyn Fn(TypeId) -> bool) -> Option<TypeId> {
    ["std::__2::", "std::__1::", "std::"]
        .iter()
        .find_map(|ns| ctx.info.find_type_where(&format!("{ns}{name}"), pred))
}

/// Find a node type `__<kind>_node<_Tp, ...>` whose `_Tp` is `value`.
fn node_type(ctx: &Ctx<'_>, kind: &str, value: TypeId) -> Result<TypeId> {
    find_libcxx_type(ctx, &format!("__{kind}_node<"), &|t| {
        template_param(ctx, t, "_Tp").is_ok_and(|tp| same_type(ctx, tp, value))
    })
    .ok_or_else(|| {
        anyhow!(
            "cannot find the libc++ {kind} node type for `{}` in the debug info",
            ctx.info.type_name(value)
        )
    })
}

/// Read a pointer-valued member (looking through compressed pairs).
fn ptr_member(ctx: &Ctx<'_>, v: &Value, names: &[&str]) -> Result<u64> {
    let m = names
        .iter()
        .find_map(|n| ctx.field(v, n).ok())
        .ok_or_else(|| anyhow!("no member {names:?} in `{}`", ctx.info.type_name(v.ty)))?;
    let p = first_pointer_leaf(ctx, &m)?.ok_or_else(|| anyhow!("{names:?} is not a pointer"))?;
    ctx.read_uint(&p)
}

/// Read an integer-valued member (looking through compressed pairs).
fn uint_member(ctx: &Ctx<'_>, v: &Value, names: &[&str]) -> Result<u64> {
    let m = names
        .iter()
        .find_map(|n| ctx.field(v, n).ok())
        .ok_or_else(|| anyhow!("no member {names:?} in `{}`", ctx.info.type_name(v.ty)))?;
    first_uint_leaf(ctx, &m)
}

/// Split a map's value (`__value_type<K, V>` / `pair<const K, V>`) into
/// key and value.
fn key_value(ctx: &Ctx<'_>, v: &Value) -> Result<(Value, Value)> {
    let pair = ctx.field(v, "__cc_").unwrap_or_else(|_| v.clone());
    Ok((ctx.field(&pair, "first")?, ctx.field(&pair, "second")?))
}

fn pluralize(n: u64) -> &'static str {
    if n == 1 { "element" } else { "elements" }
}

// ---------------------------------------------------------------------

/// `std::string` (`basic_string<char>`), with the short-string
/// optimization.
struct String;
impl String {
    /// (data address, size, char type).
    fn parts(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, TypeId)> {
        let rep = ctx.field(v, "__rep_")?;
        let s = ctx.field(&rep, "__s")?;
        let l = ctx.field(&rep, "__l")?;
        let is_long = match ctx.field(&s, "__is_long_") {
            Ok(b) => ctx.read_uint(&b)? != 0,
            Err(_) => ctx.read_uint(&ctx.field(&l, "__is_long_")?)? != 0,
        };
        if is_long {
            let data = ctx.field(&l, "__data_")?;
            let size = ctx.read_uint(&ctx.field(&l, "__size_")?)?;
            Ok((ctx.read_uint(&data)?, size, pointee(ctx, &data)?))
        } else {
            let data = ctx.field(&s, "__data_")?;
            let size = ctx.read_uint(&ctx.field(&s, "__size_")?)?;
            let elem = match ctx.info.ty(ctx.info.strip(data.ty)).kind {
                TypeKind::Array { elem, .. } => elem,
                _ => bail!("unexpected std::string layout"),
            };
            let addr = data
                .address()
                .ok_or_else(|| anyhow!("std::string not in memory"))?;
            Ok((addr, size, elem))
        }
    }
}
impl PrettyPrinter for String {
    fn name(&self) -> &'static str {
        "cpp-string"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        cpp(ty)
            && is_one_of(name, &["std::basic_string"])
            && name.starts_with("std::basic_string<char,")
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let (addr, size, _) = Self::parts(ctx, v)?;
        let n = size.min(opts.max_string as u64);
        let bytes = if n == 0 {
            vec![]
        } else {
            ctx.target.read_memory(0, addr, n)?
        };
        let mut s = quote_bytes(&bytes, false);
        if n < size {
            s.push_str("...");
        }
        Ok(Summary::Text(s))
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (addr, size, elem) = Self::parts(ctx, v)?;
            check_index(i, size)?;
            Ok(element(ctx, elem, addr, i))
        })())
    }
}

/// `std::vector<T>` (not `vector<bool>`).
struct Vector;
impl Vector {
    /// (begin, len, capacity, element type).
    fn parts(ctx: &Ctx<'_>, v: &Value) -> Result<(u64, u64, u64, TypeId)> {
        let begin_field = ctx.field(v, "__begin_")?;
        let elem = pointee(ctx, &begin_field)?;
        let size = ctx.info.type_size(elem).unwrap_or(1).max(1);
        let begin = ctx.read_uint(&begin_field)?;
        let end = ptr_member(ctx, v, &["__end_"])?;
        let cap = ptr_member(ctx, v, &["__cap_", "__end_cap_"]).unwrap_or(end);
        Ok((
            begin,
            end.saturating_sub(begin) / size,
            cap.saturating_sub(begin) / size,
            elem,
        ))
    }
}
impl PrettyPrinter for Vector {
    fn name(&self) -> &'static str {
        "cpp-vector"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        cpp(ty) && is_one_of(name, &["std::vector"]) && !name.starts_with("std::vector<bool,")
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let (begin, len, cap, elem) = Self::parts(ctx, v)?;
        Ok(Summary::Seq {
            prefix: format!("std::vector of length {len}, capacity {cap} = "),
            open: "{",
            close: "}",
            elem,
            addr: begin,
            len,
        })
    }
    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        Some((|| {
            let (begin, len, _, elem) = Self::parts(ctx, v)?;
            check_index(i, len)?;
            Ok(element(ctx, elem, begin, i))
        })())
    }
}

/// `std::map`, `std::multimap`, `std::set`, `std::multiset`: red-black
/// trees (`__tree`).
struct Tree;
impl Tree {
    /// In-order traversal: up to `max` node values, and the size.
    fn values(ctx: &Ctx<'_>, v: &Value, max: usize) -> Result<(Vec<Value>, u64)> {
        let tree = ctx.field(v, "__tree_")?;
        let size = uint_member(ctx, &tree, &["__size_", "__pair3_"])?;
        let value_ty = template_param(ctx, tree.ty, "_Tp")?;
        let end = ctx
            .member(&tree, "__end_node_")
            .or_else(|_| ctx.field(&ctx.field(&tree, "__pair1_")?, "__value_"))?;
        let root = ctx.read_uint(&ctx.field(&end, "__left_")?)?;
        let mut out = vec![];
        if size == 0 || root == 0 {
            return Ok((out, size));
        }
        let node_ty = node_type(ctx, "tree", value_ty)?;
        fn walk(
            ctx: &Ctx<'_>,
            node_ty: TypeId,
            addr: u64,
            depth: usize,
            max: usize,
            out: &mut Vec<Value>,
        ) -> Result<()> {
            if addr == 0 || depth > 64 || out.len() >= max {
                return Ok(());
            }
            let node = Value::in_memory(node_ty, addr);
            let left = ctx.read_uint(&ctx.field(&node, "__left_")?)?;
            walk(ctx, node_ty, left, depth + 1, max, out)?;
            if out.len() < max {
                out.push(ctx.field(&node, "__value_")?);
            }
            let right = ctx.read_uint(&ctx.field(&node, "__right_")?)?;
            walk(ctx, node_ty, right, depth + 1, max, out)
        }
        walk(ctx, node_ty, root, 0, max, &mut out)?;
        Ok((out, size))
    }
}
impl PrettyPrinter for Tree {
    fn name(&self) -> &'static str {
        "cpp-tree"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        cpp(ty)
            && is_one_of(
                name,
                &["std::map", "std::multimap", "std::set", "std::multiset"],
            )
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let name = match_name(&ctx.info.ty(ctx.info.strip(v.ty)));
        let kind = &name[..name.find('<').unwrap_or(name.len())];
        let (values, total) = Self::values(ctx, v, opts.max_elems)?;
        let prefix = format!("{kind} with {total} {}", pluralize(total));
        if kind.ends_with("map") {
            let entries = values
                .iter()
                .map(|v| key_value(ctx, v))
                .collect::<Result<Vec<_>>>()?;
            Ok(Summary::Map {
                prefix: format!("{prefix} ="),
                entries,
                total,
                cpp: true,
            })
        } else {
            Ok(Summary::List {
                prefix: format!("{prefix} = "),
                open: "{",
                close: "}",
                elems: values,
                total,
            })
        }
    }
}

/// `std::unordered_map`, `std::unordered_set` (and multi- variants):
/// hash tables (`__hash_table`) whose nodes form a singly-linked list.
struct HashTable;
impl HashTable {
    fn values(ctx: &Ctx<'_>, v: &Value, max: usize) -> Result<(Vec<Value>, u64)> {
        let table = ctx.field(v, "__table_")?;
        let size = uint_member(ctx, &table, &["__size_", "__p2_"])?;
        let value_ty = template_param(ctx, table.ty, "_Tp")?;
        let first = ctx
            .member(&table, "__first_node_")
            .or_else(|_| ctx.field(&ctx.field(&table, "__p1_")?, "__value_"))?;
        let mut addr = ctx.read_uint(&ctx.field(&first, "__next_")?)?;
        let mut out = vec![];
        if size == 0 || addr == 0 {
            return Ok((out, size));
        }
        let node_ty = node_type(ctx, "hash", value_ty)?;
        while addr != 0 && out.len() < max.min(usize::try_from(size)?) {
            let node = Value::in_memory(node_ty, addr);
            out.push(ctx.field(&node, "__value_")?);
            addr = ctx.read_uint(&ctx.field(&node, "__next_")?)?;
        }
        Ok((out, size))
    }
}
impl PrettyPrinter for HashTable {
    fn name(&self) -> &'static str {
        "cpp-hash-table"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        cpp(ty)
            && is_one_of(
                name,
                &[
                    "std::unordered_map",
                    "std::unordered_multimap",
                    "std::unordered_set",
                    "std::unordered_multiset",
                ],
            )
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let name = match_name(&ctx.info.ty(ctx.info.strip(v.ty)));
        let kind = &name[..name.find('<').unwrap_or(name.len())];
        let (values, total) = Self::values(ctx, v, opts.max_elems)?;
        let prefix = format!("{kind} with {total} {}", pluralize(total));
        if kind.ends_with("map") {
            let entries = values
                .iter()
                .map(|v| key_value(ctx, v))
                .collect::<Result<Vec<_>>>()?;
            Ok(Summary::Map {
                prefix: format!("{prefix} ="),
                entries,
                total,
                cpp: true,
            })
        } else {
            Ok(Summary::List {
                prefix: format!("{prefix} = "),
                open: "{",
                close: "}",
                elems: values,
                total,
            })
        }
    }
}

/// The pointee of a smart pointer, if non-null.
fn smart_ptr_target(ctx: &Ctx<'_>, v: &Value) -> Result<Option<Value>> {
    let m = ctx.field(v, "__ptr_")?;
    let p = first_pointer_leaf(ctx, &m)?.ok_or_else(|| anyhow!("__ptr_ is not a pointer"))?;
    let addr = ctx.read_uint(&p)?;
    if addr == 0 {
        return Ok(None);
    }
    Ok(Some(Value::in_memory(pointee(ctx, &p)?, addr)))
}

/// `std::unique_ptr<T>`.
struct UniquePtr;
impl PrettyPrinter for UniquePtr {
    fn name(&self) -> &'static str {
        "cpp-unique-ptr"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        cpp(ty) && is_one_of(name, &["std::unique_ptr"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        Ok(match smart_ptr_target(ctx, v)? {
            Some(t) => Summary::Labeled {
                label: format!("std::unique_ptr({:#x})", t.address().unwrap_or(0)),
                inner: Some(t),
            },
            None => Summary::Text("std::unique_ptr(nullptr)".into()),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(smart_ptr_target(ctx, v).and_then(|t| t.ok_or_else(|| anyhow!("null unique_ptr"))))
    }
}

/// `std::shared_ptr<T>` and `std::weak_ptr<T>`.
struct SharedPtr;
impl SharedPtr {
    /// (use count, weak count) from the control block, if any.
    fn counts(ctx: &Ctx<'_>, v: &Value) -> Result<Option<(u64, u64)>> {
        let cntrl = ctx.field(v, "__cntrl_")?;
        if ctx.read_uint(&cntrl)? == 0 {
            return Ok(None);
        }
        let block = ctx.deref(&cntrl)?;
        // libc++ stores both counts minus one, and the owners
        // collectively hold one implicit weak reference.
        let uses = ctx.read_int(&ctx.field(&block, "__shared_owners_")?)? + 1;
        let weak = ctx.read_int(&ctx.field(&block, "__shared_weak_owners_")?)? + 1;
        let weak = if uses > 0 { weak - 1 } else { weak };
        Ok(Some((uses.max(0) as u64, weak.max(0) as u64)))
    }
}
impl PrettyPrinter for SharedPtr {
    fn name(&self) -> &'static str {
        "cpp-shared-ptr"
    }
    fn matches(&self, _ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        cpp(ty) && is_one_of(name, &["std::shared_ptr", "std::weak_ptr"])
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, _opts: &FormatOptions) -> Result<Summary> {
        let name = match_name(&ctx.info.ty(ctx.info.strip(v.ty)));
        let kind = &name[..name.find('<').unwrap_or(name.len())];
        let target = smart_ptr_target(ctx, v)?;
        let counts = Self::counts(ctx, v)?;
        Ok(match (target, counts) {
            (Some(t), Some((uses, weak))) if uses > 0 => Summary::Labeled {
                label: format!("{kind}(use_count={uses}, weak_count={weak})"),
                inner: Some(t),
            },
            (Some(_), Some(_)) => Summary::Text(format!("{kind}(expired)")),
            _ => Summary::Text(format!("{kind}(nullptr)")),
        })
    }
    fn deref(&self, ctx: &Ctx<'_>, v: &Value) -> Option<Result<Value>> {
        Some(smart_ptr_target(ctx, v).and_then(|t| t.ok_or_else(|| anyhow!("null pointer"))))
    }
}
