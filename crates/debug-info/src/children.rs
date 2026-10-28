//! Structured children of values, mirroring what the pretty-printer
//! shows: sequence elements, map entries, struct fields, enum payload
//! fields, and pointees. Used by debugger front-ends (such as DAP) that
//! present values as expandable trees.

use crate::format::{FormatOptions, format_value};
use crate::pretty::{self, Summary};
use crate::types::*;
use crate::value::{Ctx, Value};

/// How to name a child with an access path, relative to its parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChildPath {
    /// Append a suffix (`.field`, `->field`, `[3]`).
    Suffix(String),
    /// Dereference the parent (`*parent`).
    Deref,
    /// Not expressible as an access path (e.g. a map entry).
    None,
}

/// One child of a value.
#[derive(Clone, Debug)]
pub struct Child {
    /// Display name (`x`, `[3]`, `"key"`, ...).
    pub name: String,
    pub value: Value,
    pub path: ChildPath,
}

impl Child {
    /// The access path naming this child, given its parent's path.
    pub fn eval_name(&self, parent: &str) -> Option<String> {
        let simple = parent
            .chars()
            .all(|c| c.is_alphanumeric() || "_.[]:$%>-".contains(c));
        let parent = if simple {
            parent.to_string()
        } else {
            format!("({parent})")
        };
        match &self.path {
            ChildPath::Suffix(s) => Some(format!("{parent}{s}")),
            ChildPath::Deref => Some(format!("*{parent}")),
            ChildPath::None => None,
        }
    }
}

/// Children of `v`, skipping the first `start` and returning at most
/// `count`. Never fails: values that can't be expanded have no
/// children.
pub fn children(
    ctx: &Ctx<'_>,
    v: &Value,
    opts: &FormatOptions,
    start: usize,
    count: usize,
) -> Vec<Child> {
    let limit = start.saturating_add(count);
    let mut kids = children_inner(ctx, v, opts, limit, 0);
    if start >= kids.len() {
        return vec![];
    }
    kids.drain(..start);
    kids.truncate(count);
    kids
}

/// Whether `v` has any children.
pub fn has_children(ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> bool {
    !children_inner(ctx, v, opts, 1, 0).is_empty()
}

/// The number of indexed children (array/sequence elements), if `v` is
/// a sequence; used for paging large arrays.
pub fn indexed_len(ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Option<u64> {
    if !opts.raw {
        if let Some(p) = pretty::find(ctx, v.ty) {
            if let Ok(s) = p.summarize(ctx, v, opts) {
                return match s {
                    Summary::Seq { len, .. } => Some(len),
                    Summary::List { total, .. } => Some(total),
                    _ => None,
                };
            }
        }
    }
    match ctx.info.ty(ctx.info.strip(v.ty)).kind {
        TypeKind::Array { len, .. } => len,
        _ => None,
    }
}

fn indexed(name: u64, path: bool) -> (String, ChildPath) {
    (
        format!("[{name}]"),
        if path {
            ChildPath::Suffix(format!("[{name}]"))
        } else {
            ChildPath::None
        },
    )
}

fn children_inner(
    ctx: &Ctx<'_>,
    v: &Value,
    opts: &FormatOptions,
    limit: usize,
    depth: usize,
) -> Vec<Child> {
    if depth > 8 || limit == 0 {
        return vec![];
    }
    if !opts.raw {
        if let Some(p) = pretty::find(ctx, v.ty) {
            let popts = FormatOptions {
                max_elems: limit,
                ..opts.clone()
            };
            if let Ok(s) = p.summarize(ctx, v, &popts) {
                return summary_children(ctx, s, opts, limit, depth);
            }
            // Fall back to the raw view.
        }
    }
    raw_children(ctx, v, opts, limit, depth)
}

fn summary_children(
    ctx: &Ctx<'_>,
    s: Summary,
    opts: &FormatOptions,
    limit: usize,
    depth: usize,
) -> Vec<Child> {
    match s {
        Summary::Text(_) | Summary::Labeled { inner: None, .. } => vec![],
        Summary::Seq {
            elem, addr, len, ..
        } => {
            let size = ctx.info.type_size(elem).unwrap_or(0);
            (0..len.min(limit as u64))
                .map(|i| {
                    let (name, path) = indexed(i, true);
                    Child {
                        name,
                        value: Value::in_memory(elem, addr + i * size),
                        path,
                    }
                })
                .collect()
        }
        Summary::List { elems, .. } => elems
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, value)| {
                let (name, path) = indexed(i as u64, true);
                Child { name, value, path }
            })
            .collect(),
        Summary::Set { elems, .. } => elems
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, value)| {
                let (name, path) = indexed(i as u64, false);
                Child { name, value, path }
            })
            .collect(),
        Summary::Map { entries, .. } => {
            let kopts = FormatOptions {
                max_depth: 2,
                max_elems: 8,
                max_string: 64,
                ..opts.clone()
            };
            entries
                .into_iter()
                .take(limit)
                .map(|(k, value)| Child {
                    name: format_value(ctx, &k, &kopts),
                    value,
                    path: ChildPath::None,
                })
                .collect()
        }
        Summary::Fields {
            fields, internal, ..
        } => fields
            .into_iter()
            .chain(internal)
            .take(limit)
            .map(|(name, value)| {
                let path = crate::pretty::js::property_path(&name);
                Child {
                    path: if path.is_empty() {
                        ChildPath::None
                    } else {
                        ChildPath::Suffix(path)
                    },
                    name,
                    value,
                }
            })
            .collect(),
        // Wrappers are transparent: their contents' children (paths
        // through wrappers work because member access and indexing look
        // through them).
        Summary::Wrap { inner, .. }
        | Summary::Prefixed { inner, .. }
        | Summary::Labeled {
            inner: Some(inner), ..
        } => children_inner(ctx, &inner, opts, limit, depth + 1),
    }
}

fn member_children(
    ctx: &Ctx<'_>,
    v: &Value,
    st: &StructType,
    arrow: bool,
    limit: usize,
    out: &mut Vec<Child>,
) {
    for m in &st.members {
        if out.len() >= limit {
            return;
        }
        if m.artificial {
            continue;
        }
        let mv = v.sub(
            m.ty,
            m.byte_offset(),
            m.bit_size.map(|s| (m.bit_offset % 8, s)),
        );
        match m.name.as_deref() {
            None => {
                // Anonymous struct/union: flatten its members (member
                // lookup searches inside them).
                if let TypeKind::Struct(ist) = &ctx.info.ty(ctx.info.strip(m.ty)).kind {
                    member_children(ctx, &mv, ist, arrow, limit, out);
                }
            }
            Some(n) if n.starts_with('<') => out.push(Child {
                name: n.to_string(),
                value: mv,
                path: ChildPath::None,
            }),
            Some(n) => {
                let name = n.strip_prefix("__").filter(|r| r.parse::<u32>().is_ok());
                let name = name.unwrap_or(n).to_string();
                let sep = if arrow { "->" } else { "." };
                out.push(Child {
                    path: ChildPath::Suffix(format!("{sep}{name}")),
                    name,
                    value: mv,
                });
            }
        }
    }
}

fn raw_children(
    ctx: &Ctx<'_>,
    v: &Value,
    opts: &FormatOptions,
    limit: usize,
    depth: usize,
) -> Vec<Child> {
    let t = ctx.info.ty(ctx.info.strip(v.ty));
    let mut out = vec![];
    match &t.kind {
        TypeKind::Struct(st) if st.is_enum_like() => {
            if let Ok(Some((_, payload))) = ctx.active_variant(v) {
                if let TypeKind::Struct(pst) = &ctx.info.ty(ctx.info.strip(payload.ty)).kind {
                    member_children(ctx, &payload, pst, false, limit, &mut out);
                }
            }
        }
        TypeKind::Struct(st) => member_children(ctx, v, st, false, limit, &mut out),
        TypeKind::Array { elem, len } => {
            let size = ctx.info.type_size(*elem).unwrap_or(0);
            for i in 0..len.unwrap_or(0).min(limit as u64) {
                let (name, path) = indexed(i, true);
                out.push(Child {
                    name,
                    value: v.sub(*elem, i * size, None),
                    path,
                });
            }
        }
        TypeKind::Pointer {
            pointee: Some(_), ..
        } => {
            if ctx.read_uint(v).unwrap_or(0) == 0 {
                return out;
            }
            let Ok(target) = ctx.deref(v) else {
                return out;
            };
            let pt = ctx.info.ty(ctx.info.strip(target.ty));
            match &pt.kind {
                TypeKind::Void | TypeKind::Function { .. } | TypeKind::Unknown => {}
                TypeKind::Struct(st) if !st.is_enum_like() && t.lang.is_c_like() => {
                    // C/C++ pointer to a struct: its members, named with
                    // `->`.
                    if pretty::find(ctx, target.ty).is_some() && !opts.raw {
                        return children_inner(ctx, &target, opts, limit, depth + 1);
                    }
                    member_children(ctx, &target, st, true, limit, &mut out);
                }
                TypeKind::Struct(_) | TypeKind::Array { .. } => {
                    return children_inner(ctx, &target, opts, limit, depth + 1);
                }
                _ => out.push(Child {
                    name: "*".into(),
                    value: target,
                    path: ChildPath::Deref,
                }),
            }
        }
        _ => {}
    }
    out
}
