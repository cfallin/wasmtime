//! Pretty-printers for well-known standard-library types.
//!
//! This is a deliberately small, closed framework: each printer matches
//! on a type's (qualified) name and/or structure and produces a
//! [`Summary`], which the generic formatter renders. Printers may also
//! provide synthetic indexing (so that `v[2]` works on a `Vec`) and
//! dereference (so that `*arc`, `arc.field` and `uptr->x` see through
//! smart pointers and cells).
//!
//! Printers are grouped by language: [`rust`] (Rust's `std`), [`cpp`]
//! (LLVM's libc++, as shipped with wasi-sdk) and [`c`] (C strings).

mod c;
mod cpp;
mod rust;

use crate::format::FormatOptions;
use crate::types::*;
use crate::value::{Ctx, Value};
use anyhow::{Result, anyhow};

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
    /// A sequence of (possibly non-contiguous) elements, of which only
    /// the first few may be included; `total` is the full count.
    List {
        prefix: String,
        open: &'static str,
        close: &'static str,
        elems: Vec<Value>,
        total: u64,
    },
    /// Key/value entries (only the first few may be included); `total`
    /// is the full number of entries. Rendered Rust-style
    /// (`prefix(len=N) { k: v }`) or C++/gdb-style (`prefix {[k] = v}`).
    Map {
        prefix: String,
        entries: Vec<(Value, Value)>,
        total: u64,
        cpp: bool,
    },
    /// Set elements (only the first few may be included).
    Set {
        prefix: String,
        elems: Vec<Value>,
        total: u64,
    },
    /// A wrapper around one inner value, rendered `prefix(inner)`
    /// followed by an optional `[note]`; with an empty prefix and no
    /// note, just the inner value is shown.
    Wrap {
        prefix: String,
        inner: Value,
        note: Option<String>,
    },
    /// A label, optionally followed by ` = inner` (C++ smart pointers).
    Labeled { label: String, inner: Option<Value> },
    /// A prefix immediately followed by the inner value (e.g. `&` for a
    /// reference to a trait object).
    Prefixed { prefix: String, inner: Value },
}

/// A pretty-printer for a family of types.
pub trait PrettyPrinter: Sync {
    /// Name of this printer (for diagnostics).
    fn name(&self) -> &'static str;

    /// Whether this printer handles type `ty`. `name` is the type's
    /// qualified name, normalized for matching (for C++, libc++'s inline
    /// namespace is removed: `std::vector<...>`).
    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool;

    /// Summarize the value.
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary>;

    /// Synthetic indexing, if supported.
    fn index(&self, _ctx: &Ctx<'_>, _v: &Value, _i: u64) -> Option<Result<Value>> {
        None
    }

    /// Synthetic dereference, if supported: the value a smart pointer
    /// points to, or the contents of a cell/lock.
    fn deref(&self, _ctx: &Ctx<'_>, _v: &Value) -> Option<Result<Value>> {
        None
    }
}

/// All registered printers.
pub fn printers() -> Vec<&'static dyn PrettyPrinter> {
    let mut v = vec![];
    v.extend_from_slice(rust::PRINTERS);
    v.extend_from_slice(cpp::PRINTERS);
    v.extend_from_slice(c::PRINTERS);
    v
}

/// Normalize a type name for matching by printers.
fn match_name(t: &Type) -> String {
    let name = t.name.clone().unwrap_or_default();
    match t.lang {
        Lang::Cpp => simplify_cpp_type_name(&name),
        _ => name,
    }
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
        let name = match_name(&t);
        let lists: [&[&'static dyn PrettyPrinter]; 3] = match t.lang {
            Lang::Rust => [rust::PRINTERS, &[], &[]],
            Lang::Cpp => [cpp::PRINTERS, c::PRINTERS, &[]],
            _ => [c::PRINTERS, &[], &[]],
        };
        for list in lists {
            for p in list {
                if p.matches(ctx, &t, &name) {
                    return Some(*p);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------
// Helpers shared by printers.

/// Find the first pointer-typed leaf inside a value (depth-first through
/// struct members). Used to dig data pointers out of wrappers such as
/// `RawVec`/`Unique`/`NonNull` or libc++'s compressed pairs, whose
/// layouts vary between library versions.
pub fn first_pointer_leaf(ctx: &Ctx<'_>, v: &Value) -> Result<Option<Value>> {
    leaf(ctx, v, 0, &|k| matches!(k, TypeKind::Pointer { .. }))
}

/// Find the first integer leaf (through single-field wrappers such as
/// `Cell<usize>`, `AtomicUsize` or `UsizeNoHighBit`) and read it.
pub fn first_uint_leaf(ctx: &Ctx<'_>, v: &Value) -> Result<u64> {
    let l = leaf(ctx, v, 0, &|k| matches!(k, TypeKind::Base(_)))?
        .ok_or_else(|| anyhow!("no integer field in `{}`", ctx.info.type_name(v.ty)))?;
    ctx.read_uint(&l)
}

fn leaf(
    ctx: &Ctx<'_>,
    v: &Value,
    depth: usize,
    want: &dyn Fn(&TypeKind) -> bool,
) -> Result<Option<Value>> {
    if depth > 8 {
        return Ok(None);
    }
    let t = ctx.info.ty(ctx.info.strip(v.ty));
    if want(&t.kind) {
        return Ok(Some(v.clone()));
    }
    if let TypeKind::Struct(st) = &t.kind {
        for m in &st.members {
            let mv = v.sub(m.ty, m.byte_offset(), None);
            if let Some(p) = leaf(ctx, &mv, depth + 1, want)? {
                return Ok(Some(p));
            }
        }
    }
    Ok(None)
}

/// The base name of a generic type: `a::B` for `a::B<T, U>`. Returns
/// `None` if anything follows the generic arguments (e.g. for the
/// nested `a::B<T>::C`), so that printers don't match associated types;
/// non-generic names are returned as-is.
fn template_base(name: &str) -> Option<&str> {
    let Some(open) = name.find('<') else {
        return Some(name);
    };
    let bytes = name.as_bytes();
    let mut depth = 0;
    for i in open..bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            // `->` (in `fn() -> T`) is not a closing bracket.
            b'>' if bytes[i - 1] == b'-' => {}
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return (i == name.len() - 1).then(|| &name[..open]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `name` is an instantiation of one of the generic types
/// `bases` (or exactly one of those names, if not generic).
fn is_one_of(name: &str, bases: &[&str]) -> bool {
    template_base(name).is_some_and(|b| bases.contains(&b))
}

/// The struct type of a value, if it is a struct.
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

/// A template (generic) type parameter of a struct type.
fn template_param(ctx: &Ctx<'_>, ty: TypeId, name: &str) -> Result<TypeId> {
    struct_of(ctx, ty)
        .and_then(|st| st.template_param(name))
        .ok_or_else(|| {
            anyhow!(
                "`{}` has no template parameter `{name}`",
                ctx.info.type_name(ty)
            )
        })
}

/// Follow a chain of members; each step lists alternative names (to
/// cope with layout differences between library versions). Steps with
/// no matching member are skipped.
fn member_chain(ctx: &Ctx<'_>, v: &Value, steps: &[&[&str]]) -> Value {
    let mut cur = v.clone();
    for step in steps {
        for name in *step {
            if let Ok(next) = ctx.field(&cur, name) {
                cur = next;
                break;
            }
        }
    }
    cur
}

/// The pointee type of a pointer-typed value.
fn pointee(ctx: &Ctx<'_>, p: &Value) -> Result<TypeId> {
    match ctx.info.ty(ctx.info.strip(p.ty)).kind {
        TypeKind::Pointer {
            pointee: Some(t), ..
        } => Ok(t),
        _ => Err(anyhow!("not a typed pointer")),
    }
}

/// Element `i` of a contiguous array of `elem` at `addr`.
fn element(ctx: &Ctx<'_>, elem: TypeId, addr: u64, i: u64) -> Value {
    let size = ctx.info.type_size(elem).unwrap_or(0);
    Value::in_memory(elem, addr + i * size)
}

/// Whether two types have the same display name.
fn same_type(ctx: &Ctx<'_>, a: TypeId, b: TypeId) -> bool {
    a == b || ctx.info.type_name(a) == ctx.info.type_name(b)
}

/// Bounds-check an index.
fn check_index(i: u64, len: u64) -> Result<()> {
    if i >= len {
        anyhow::bail!("index {i} out of bounds (length {len})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_bases() {
        assert_eq!(template_base("a::B<T, c::D<E>>"), Some("a::B"));
        assert_eq!(template_base("a::B"), Some("a::B"));
        assert_eq!(template_base("std::shared_ptr<Point>::element_type"), None);
        assert!(is_one_of(
            "std::vector<int, std::allocator<int> >",
            &["std::vector"]
        ));
        assert!(!is_one_of("std::vectorish<int>", &["std::vector"]));
        assert_eq!(
            template_base("std::sync::LazyLock<u32, fn() -> u32>"),
            Some("std::sync::LazyLock")
        );
    }
}
