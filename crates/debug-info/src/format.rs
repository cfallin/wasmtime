//! Formatting of values and types for display.

use crate::demangle::simple_name;
use crate::pretty::{self, Summary};
use crate::types::*;
use crate::value::{Ctx, Value};
use anyhow::Result;
use std::fmt::Write;

/// Options controlling value formatting.
#[derive(Clone, Debug)]
pub struct FormatOptions {
    /// Print integers in hexadecimal.
    pub hex: bool,
    /// Bypass pretty-printers.
    pub raw: bool,
    /// Maximum nesting depth of aggregates and dereferenced references.
    pub max_depth: usize,
    /// Maximum number of elements printed for arrays/sequences/maps.
    pub max_elems: usize,
    /// Maximum number of bytes printed for strings.
    pub max_string: usize,
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions {
            hex: false,
            raw: false,
            max_depth: 8,
            max_elems: 100,
            max_string: 200,
        }
    }
}

/// Quote a byte string, escaping non-printable characters. Rust strings
/// are decoded as UTF-8 (lossily); C strings are treated as bytes.
pub fn quote_bytes(bytes: &[u8], rust: bool) -> String {
    let mut out = String::from("\"");
    if rust {
        for c in String::from_utf8_lossy(bytes).chars() {
            match c {
                '"' => out.push_str("\\\""),
                c => out.extend(c.escape_debug()),
            }
        }
    } else {
        for &b in bytes {
            match b {
                b'"' => out.push_str("\\\""),
                b'\\' => out.push_str("\\\\"),
                b'\n' => out.push_str("\\n"),
                b'\t' => out.push_str("\\t"),
                b'\r' => out.push_str("\\r"),
                0x20..=0x7e => out.push(b as char),
                _ => {
                    let _ = write!(out, "\\{b:03o}");
                }
            }
        }
    }
    out.push('"');
    out
}

fn quote_char(c: u32, rust: bool) -> String {
    match char::from_u32(c) {
        Some('\'') => "'\\''".into(),
        Some(ch) if rust => format!("'{}'", ch.escape_debug()),
        Some(ch) if (0x20..0x7f).contains(&c) => format!("'{ch}'"),
        _ if !rust => match c {
            0 => "'\\0'".into(),
            10 => "'\\n'".into(),
            9 => "'\\t'".into(),
            _ => format!("'\\{c:03o}'"),
        },
        _ => format!("'\\u{{{c:x}}}'"),
    }
}

/// The short, generic-free display name of a Rust type: for example,
/// `core::option::Option<i32>` becomes `Option`.
pub fn rust_short_name(name: &str) -> &str {
    simple_name(name)
}

/// Format a value as a string.
pub fn format_value(ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> String {
    let mut out = String::new();
    Formatter { ctx, opts }.value(v, 0, &mut out);
    out
}

struct Formatter<'a, 'b> {
    ctx: &'a Ctx<'b>,
    opts: &'a FormatOptions,
}

impl Formatter<'_, '_> {
    fn value(&self, v: &Value, depth: usize, out: &mut String) {
        if let Err(e) = self.value_inner(v, depth, out) {
            let _ = write!(out, "<error: {e}>");
        }
    }

    fn value_inner(&self, v: &Value, depth: usize, out: &mut String) -> Result<()> {
        if depth > self.opts.max_depth {
            out.push_str("...");
            return Ok(());
        }
        let info = self.ctx.info;
        if !self.opts.raw {
            if let Some(p) = pretty::find(self.ctx, v.ty) {
                let s = p.summarize(self.ctx, v, self.opts)?;
                return self.summary(s, depth, out);
            }
        }
        let sty = info.strip(v.ty);
        let t = info.ty(sty);
        let rust = t.lang == Lang::Rust;
        match &t.kind {
            TypeKind::Base(enc) => self.base(v, *enc, t.size.unwrap_or(0), rust, out)?,
            TypeKind::Pointer { pointee, kind } => {
                let addr = self.ctx.read_uint(v)?;
                let is_ref = if rust {
                    t.name.as_deref().is_some_and(|n| n.starts_with('&'))
                        || *kind != PtrKind::Pointer
                } else {
                    *kind != PtrKind::Pointer
                };
                match pointee {
                    Some(p) if is_ref && addr != 0 => {
                        if rust {
                            out.push('&');
                        }
                        self.value(&Value::in_memory(*p, addr), depth + 1, out);
                    }
                    _ => {
                        if depth == 0 && !rust {
                            let _ = write!(out, "({}) ", info.type_name(v.ty));
                        }
                        let _ = write!(out, "{addr:#x}");
                    }
                }
            }
            TypeKind::Struct(st) => {
                if st.is_enum_like() {
                    self.rust_enum(v, &t, depth, out)?;
                } else if rust {
                    self.rust_struct(v, &t, st, depth, out)?;
                } else {
                    self.c_struct(v, st, depth, out)?;
                }
            }
            TypeKind::Enum { enumerators, .. } => {
                let raw = self.ctx.read_int(v)?;
                let unsigned = self.ctx.read_uint(v)?;
                let found = enumerators
                    .iter()
                    .find(|(_, x)| *x == i128::from(raw) || *x == i128::from(unsigned));
                match found {
                    Some((n, _)) if rust => {
                        let tn = t.name.as_deref().map(rust_short_name).unwrap_or("?");
                        let _ = write!(out, "{tn}::{n}");
                    }
                    Some((n, _)) => out.push_str(n),
                    None => {
                        let _ = write!(out, "{raw}");
                    }
                }
            }
            TypeKind::Array { elem, len } => {
                let (open, close) = if rust { ("[", "]") } else { ("{", "}") };
                let len = len.unwrap_or(0);
                let esize = info.type_size(*elem).unwrap_or(0);
                out.push_str(open);
                let n = len.min(self.opts.max_elems as u64);
                for i in 0..n {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.value(&v.sub(*elem, i * esize, None), depth + 1, out);
                }
                if n < len {
                    let _ = write!(out, ", ...({} more)", len - n);
                }
                out.push_str(close);
            }
            TypeKind::Void => out.push_str("void"),
            TypeKind::Function { .. } => out.push_str("<function>"),
            TypeKind::Typedef(None) | TypeKind::Qualified { target: None, .. } => {
                out.push_str("void")
            }
            TypeKind::Typedef(_) | TypeKind::Qualified { .. } => unreachable!(),
            TypeKind::Unknown => out.push_str("<unknown type>"),
        }
        Ok(())
    }

    fn base(&self, v: &Value, enc: BaseEnc, size: u64, rust: bool, out: &mut String) -> Result<()> {
        if size == 0 && v.bitfield.is_none() {
            out.push_str("()");
            return Ok(());
        }
        match enc {
            BaseEnc::Bool => {
                let x = self.ctx.read_uint(v)?;
                match x {
                    0 => out.push_str("false"),
                    1 => out.push_str("true"),
                    x => {
                        let _ = write!(out, "<invalid bool {x}>");
                    }
                }
            }
            BaseEnc::Float => {
                let x = self.ctx.read_uint(v)?;
                match size {
                    4 => {
                        let _ = write!(out, "{:?}", f32::from_bits(x as u32));
                    }
                    8 => {
                        let _ = write!(out, "{:?}", f64::from_bits(x));
                    }
                    _ => {
                        let _ = write!(out, "<{size}-byte float {x:#x}>");
                    }
                }
            }
            BaseEnc::Utf => {
                let x = self.ctx.read_uint(v)? as u32;
                out.push_str(&quote_char(x, rust));
            }
            BaseEnc::SignedChar | BaseEnc::UnsignedChar if !rust => {
                let x = if enc == BaseEnc::SignedChar {
                    self.ctx.read_int(v)?
                } else {
                    self.ctx.read_uint(v)? as i64
                };
                if self.opts.hex {
                    let _ = write!(out, "{:#x}", x as u8);
                } else {
                    let _ = write!(out, "{x} {}", quote_char(u32::from(x as u8), false));
                }
            }
            BaseEnc::Signed | BaseEnc::SignedChar => {
                if size > 8 && v.bitfield.is_none() {
                    let b = self.ctx.read_bytes(v, 16)?;
                    let x = i128::from_le_bytes(b.try_into().unwrap());
                    if self.opts.hex {
                        let _ = write!(out, "{x:#x}");
                    } else {
                        let _ = write!(out, "{x}");
                    }
                } else if self.opts.hex {
                    let x = self.ctx.read_uint(v)?;
                    let _ = write!(out, "{x:#x}");
                } else {
                    let _ = write!(out, "{}", self.ctx.read_int(v)?);
                }
            }
            BaseEnc::Unsigned | BaseEnc::UnsignedChar | BaseEnc::Other => {
                if size > 8 && v.bitfield.is_none() {
                    let b = self.ctx.read_bytes(v, 16)?;
                    let x = u128::from_le_bytes(b.try_into().unwrap());
                    if self.opts.hex {
                        let _ = write!(out, "{x:#x}");
                    } else {
                        let _ = write!(out, "{x}");
                    }
                } else {
                    let x = self.ctx.read_uint(v)?;
                    if self.opts.hex {
                        let _ = write!(out, "{x:#x}");
                    } else {
                        let _ = write!(out, "{x}");
                    }
                }
            }
        }
        Ok(())
    }

    fn members(&self, v: &Value, st: &StructType) -> Vec<(Option<String>, Value)> {
        st.members
            .iter()
            .filter(|m| !m.artificial)
            .map(|m| {
                (
                    m.name.clone(),
                    v.sub(
                        m.ty,
                        m.byte_offset(),
                        m.bit_size.map(|s| (m.bit_offset % 8, s)),
                    ),
                )
            })
            .collect()
    }

    fn c_struct(&self, v: &Value, st: &StructType, depth: usize, out: &mut String) -> Result<()> {
        if depth == 0 {
            let _ = write!(out, "{} ", self.ctx.info.type_name(v.ty));
        }
        if st.declaration {
            out.push_str("<incomplete type>");
            return Ok(());
        }
        let members = self.members(v, st);
        if members.is_empty() {
            out.push_str("{}");
            return Ok(());
        }
        out.push_str("{ ");
        for (i, (name, mv)) in members.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            if let Some(n) = name {
                let _ = write!(out, ".{n} = ");
            }
            self.value(mv, depth + 1, out);
        }
        out.push_str(" }");
        Ok(())
    }

    fn rust_fields(&self, v: &Value, st: &StructType, depth: usize, out: &mut String) {
        let members = self.members(v, st);
        if st.is_tuple_like() {
            out.push('(');
            for (i, (_, mv)) in members.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                self.value(mv, depth + 1, out);
            }
            out.push(')');
        } else if !members.is_empty() {
            out.push_str(" { ");
            for (i, (name, mv)) in members.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}: ", name.as_deref().unwrap_or("?"));
                self.value(mv, depth + 1, out);
            }
            out.push_str(" }");
        }
    }

    fn rust_struct(
        &self,
        v: &Value,
        t: &Type,
        st: &StructType,
        depth: usize,
        out: &mut String,
    ) -> Result<()> {
        let name = t.name.as_deref().unwrap_or("");
        if name.starts_with('(') {
            // A tuple type.
            out.push('(');
            let members = self.members(v, st);
            for (i, (_, mv)) in members.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                self.value(mv, depth + 1, out);
            }
            if members.len() == 1 {
                out.push(',');
            }
            out.push(')');
            return Ok(());
        }
        out.push_str(rust_short_name(name));
        self.rust_fields(v, st, depth, out);
        Ok(())
    }

    fn rust_enum(&self, v: &Value, t: &Type, depth: usize, out: &mut String) -> Result<()> {
        let ename = t.name.as_deref().map(rust_short_name).unwrap_or("?");
        let Some((vname, payload)) = self.ctx.active_variant(v)? else {
            let _ = write!(out, "{ename}::<unknown variant>");
            return Ok(());
        };
        if !matches!(ename, "Option" | "Result") {
            let _ = write!(out, "{ename}::");
        }
        out.push_str(&vname);
        let pt = self.ctx.info.ty(self.ctx.info.strip(payload.ty));
        if let TypeKind::Struct(pst) = &pt.kind {
            self.rust_fields(&payload, pst, depth, out);
        }
        Ok(())
    }

    fn summary(&self, s: Summary, depth: usize, out: &mut String) -> Result<()> {
        match s {
            Summary::Text(t) => out.push_str(&t),
            Summary::Seq {
                prefix,
                open,
                close,
                elem,
                addr,
                len,
            } => {
                out.push_str(&prefix);
                out.push_str(open);
                let esize = self.ctx.info.type_size(elem).unwrap_or(0);
                let n = len.min(self.opts.max_elems as u64);
                for i in 0..n {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.value(&Value::in_memory(elem, addr + i * esize), depth + 1, out);
                }
                if n < len {
                    let _ = write!(out, ", ...({} more)", len - n);
                }
                out.push_str(close);
            }
            Summary::Map {
                prefix,
                entries,
                total,
            } => {
                let _ = write!(out, "{prefix}(len={total}) {{");
                for (i, (k, v)) in entries.iter().enumerate() {
                    out.push_str(if i > 0 { ", " } else { " " });
                    self.value(k, depth + 1, out);
                    out.push_str(": ");
                    self.value(v, depth + 1, out);
                }
                if (entries.len() as u64) < total {
                    out.push_str(", ...");
                }
                out.push_str(if entries.is_empty() { "}" } else { " }" });
            }
            Summary::Set {
                prefix,
                elems,
                total,
            } => {
                let _ = write!(out, "{prefix}(len={total}) {{");
                for (i, k) in elems.iter().enumerate() {
                    out.push_str(if i > 0 { ", " } else { " " });
                    self.value(k, depth + 1, out);
                }
                if (elems.len() as u64) < total {
                    out.push_str(", ...");
                }
                out.push_str(if elems.is_empty() { "}" } else { " }" });
            }
            Summary::Wrap { prefix, inner } => {
                let _ = write!(out, "{prefix}(");
                self.value(&inner, depth + 1, out);
                out.push(')');
            }
        }
        Ok(())
    }
}

/// Describe a type in detail (as for `ptype`): struct members, enum
/// variants, etc.
pub fn describe_type(info: &crate::DebugInfo, ty: TypeId) -> String {
    let sty = info.strip(ty);
    let t = info.ty(sty);
    let rust = t.lang == Lang::Rust;
    let mut out = String::new();
    match &t.kind {
        TypeKind::Struct(st) if st.is_enum_like() => {
            let _ = writeln!(out, "enum {} {{", info.type_name(sty));
            for var in &st.variant_part.as_ref().unwrap().variants {
                for m in &var.members {
                    let _ = write!(out, "    {}", m.name.as_deref().unwrap_or("?"));
                    let pt = info.ty(info.strip(m.ty));
                    if let TypeKind::Struct(pst) = &pt.kind {
                        if pst.is_tuple_like() {
                            let fields: Vec<_> =
                                pst.members.iter().map(|f| info.type_name(f.ty)).collect();
                            let _ = write!(out, "({})", fields.join(", "));
                        } else if !pst.members.is_empty() {
                            let fields: Vec<_> = pst
                                .members
                                .iter()
                                .map(|f| {
                                    format!(
                                        "{}: {}",
                                        f.name.as_deref().unwrap_or("?"),
                                        info.type_name(f.ty)
                                    )
                                })
                                .collect();
                            let _ = write!(out, " {{ {} }}", fields.join(", "));
                        }
                    }
                    out.push_str(",\n");
                }
            }
            out.push('}');
        }
        TypeKind::Struct(st) => {
            let _ = writeln!(
                out,
                "{}{} {{",
                if rust { "struct " } else { "" },
                info.type_name(sty)
            );
            for m in &st.members {
                let name = m.name.as_deref().unwrap_or("<anonymous>");
                let bits = m.bit_size.map(|b| format!(" : {b}")).unwrap_or_default();
                if rust {
                    let _ = writeln!(out, "    {name}: {},", info.type_name(m.ty));
                } else {
                    let _ = writeln!(out, "    {};", c_decl(info, m.ty, name) + &bits);
                }
            }
            if let Some(part) = &st.variant_part {
                let _ = writeln!(out, "    <variant part with {} variants>", part.variants.len());
            }
            out.push('}');
        }
        TypeKind::Enum { enumerators, .. } => {
            let _ = writeln!(out, "{} {{", if rust { format!("enum {}", info.type_name(sty)) } else { info.type_name(sty) });
            for (n, v) in enumerators {
                let _ = writeln!(out, "    {n} = {v},");
            }
            out.push('}');
        }
        _ => out.push_str(&info.type_name(ty)),
    }
    if sty != ty {
        out = format!("{} = {out}", info.type_name(ty));
    }
    out
}

/// A C-style declaration of `name` with type `ty` (handling arrays).
fn c_decl(info: &crate::DebugInfo, ty: TypeId, name: &str) -> String {
    let t = info.ty(ty);
    match &t.kind {
        TypeKind::Array { elem, len } if t.name.is_none() => {
            let l = len.map(|l| l.to_string()).unwrap_or_default();
            c_decl(info, *elem, &format!("{name}[{l}]"))
        }
        _ => {
            let tn = info.type_name(ty);
            if tn.ends_with('*') {
                format!("{tn}{name}")
            } else {
                format!("{tn} {name}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting() {
        assert_eq!(quote_bytes(b"hi\n\"x\"", false), "\"hi\\n\\\"x\\\"\"");
        assert_eq!(quote_bytes(b"\x01", false), "\"\\001\"");
        assert_eq!(quote_bytes("héllo\n".as_bytes(), true), "\"héllo\\n\"");
        assert_eq!(quote_char(97, false), "'a'");
        assert_eq!(quote_char(0, false), "'\\0'");
        assert_eq!(quote_char(0x1F600, true), "'😀'");
    }

    #[test]
    fn short_names() {
        assert_eq!(rust_short_name("core::option::Option<i32>"), "Option");
        assert_eq!(rust_short_name("debug_tui_rust::Shape"), "Shape");
    }
}
