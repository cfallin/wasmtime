//! Wasm-level values: numbers and references, including GC objects.

use crate::api::{self, AnyrefKind, Debuggee, WasmType};
use anyhow::{Result, anyhow, bail};
use std::fmt::Write;
use wasmtime_debug_info::WasmVal;

/// A Wasm value captured from the debuggee.
pub enum WVal {
    Num(WasmVal),
    Func(Option<api::WasmFunc>),
    Exn(Option<api::WasmException>),
    Any(Option<api::WasmAnyref>),
    Extern {
        null: bool,
    },
    /// A table, which can be indexed (`$table0[3]`).
    Table(api::Table),
}

impl WVal {
    pub fn from_api(v: &api::WasmValue) -> WVal {
        match v.get_type() {
            WasmType::WasmI32 => WVal::Num(WasmVal::I32(v.unwrap_i32())),
            WasmType::WasmI64 => WVal::Num(WasmVal::I64(v.unwrap_i64())),
            WasmType::WasmF32 => WVal::Num(WasmVal::F32(v.unwrap_f32().to_bits())),
            WasmType::WasmF64 => WVal::Num(WasmVal::F64(v.unwrap_f64().to_bits())),
            WasmType::WasmV128 => {
                let b: [u8; 16] = v.unwrap_v128().try_into().unwrap_or([0; 16]);
                WVal::Num(WasmVal::V128(u128::from_le_bytes(b)))
            }
            WasmType::WasmFuncref => WVal::Func(v.unwrap_func()),
            WasmType::WasmExnref => WVal::Exn(v.unwrap_exception()),
            WasmType::WasmAnyref => WVal::Any(v.unwrap_anyref()),
            WasmType::WasmExternref => WVal::Extern { null: v.is_null() },
        }
    }

    /// Clone (resources are cloned via their `clone` methods).
    pub fn duplicate(&self, d: &Debuggee) -> WVal {
        match self {
            WVal::Num(n) => WVal::Num(*n),
            WVal::Func(f) => WVal::Func(f.as_ref().map(|f| f.clone())),
            WVal::Exn(e) => WVal::Exn(e.as_ref().map(|e| e.clone(d))),
            WVal::Any(a) => WVal::Any(a.as_ref().map(|a| a.clone())),
            WVal::Extern { null } => WVal::Extern { null: *null },
            WVal::Table(t) => WVal::Table(t.clone()),
        }
    }

    /// The scalar value, if this is a number.
    pub fn as_num(&self) -> Option<WasmVal> {
        match self {
            WVal::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Select a field (struct field index, for GC structs).
    pub fn field(&self, d: &Debuggee, name: &str) -> Result<WVal> {
        let idx: u32 = name.parse().map_err(|_| {
            anyhow!("Wasm GC struct fields are selected by index (`.0`, `.1`, ...)")
        })?;
        match self {
            WVal::Any(Some(a)) => match a.kind(d)? {
                AnyrefKind::StructRef(n) => {
                    if idx >= n {
                        bail!("struct has only {n} fields");
                    }
                    let fields = a.struct_fields(d)?;
                    Ok(WVal::from_api(&fields[usize::try_from(idx)?]))
                }
                _ => bail!("not a GC struct"),
            },
            WVal::Exn(Some(e)) => {
                let vals = e.get_values(d)?;
                let v = vals
                    .get(usize::try_from(idx)?)
                    .ok_or_else(|| anyhow!("exception has only {} values", vals.len()))?;
                Ok(WVal::from_api(v))
            }
            WVal::Any(None) | WVal::Exn(None) => bail!("null reference"),
            _ => bail!("cannot select a field of this Wasm value"),
        }
    }

    /// Index an array or table.
    pub fn index(&self, d: &Debuggee, i: u64) -> Result<WVal> {
        match self {
            WVal::Any(Some(a)) => match a.kind(d)? {
                AnyrefKind::ArrayRef(n) => {
                    let i = u32::try_from(i)?;
                    if i >= n {
                        bail!("index {i} out of bounds (array length {n})");
                    }
                    let elems = a.array_elements(d, i, 1)?;
                    Ok(WVal::from_api(&elems[0]))
                }
                _ => bail!("not a GC array"),
            },
            WVal::Table(t) => {
                let len = t.len(d);
                if i >= len {
                    bail!("index {i} out of bounds (table length {len})");
                }
                Ok(WVal::from_api(&t.get_element(d, i)?))
            }
            WVal::Any(None) => bail!("null reference"),
            _ => bail!("cannot index this Wasm value"),
        }
    }
}

/// Format a type-tagged scalar.
pub fn format_num(v: WasmVal, hex: bool) -> String {
    match v {
        WasmVal::I32(x) if hex => format!("i32 {x:#x}"),
        WasmVal::I32(x) => format!("i32 {}", x as i32),
        WasmVal::I64(x) if hex => format!("i64 {x:#x}"),
        WasmVal::I64(x) => format!("i64 {}", x as i64),
        WasmVal::F32(x) => format!("f32 {:?}", f32::from_bits(x)),
        WasmVal::F64(x) => format!("f64 {:?}", f64::from_bits(x)),
        WasmVal::V128(x) => format!("v128 {x:#034x}"),
    }
}

fn sig(d: &Debuggee, f: &api::WasmFunc) -> String {
    let tys = |r: std::result::Result<Vec<WasmType>, api::Error>| -> String {
        match r {
            Ok(v) => v.iter().map(type_name).collect::<Vec<_>>().join(", "),
            Err(_) => "?".into(),
        }
    };
    format!("({}) -> ({})", tys(f.params(d)), tys(f.results(d)))
}

pub fn type_name(t: &WasmType) -> &'static str {
    match t {
        WasmType::WasmI32 => "i32",
        WasmType::WasmI64 => "i64",
        WasmType::WasmF32 => "f32",
        WasmType::WasmF64 => "f64",
        WasmType::WasmV128 => "v128",
        WasmType::WasmFuncref => "funcref",
        WasmType::WasmExnref => "exnref",
        WasmType::WasmAnyref => "anyref",
        WasmType::WasmExternref => "externref",
    }
}

/// Format a Wasm value, expanding GC objects up to `depth` levels.
pub fn format(d: &Debuggee, v: &WVal, hex: bool, depth: usize) -> String {
    let mut out = String::new();
    format_into(d, v, hex, depth, &mut out);
    out
}

const MAX_GC_ELEMS: u32 = 32;

fn format_into(d: &Debuggee, v: &WVal, hex: bool, depth: usize, out: &mut String) {
    match v {
        WVal::Num(n) => out.push_str(&format_num(*n, hex)),
        WVal::Func(None) => out.push_str("(ref.null func)"),
        WVal::Func(Some(f)) => {
            let _ = write!(out, "funcref {}", sig(d, f));
        }
        WVal::Exn(None) => out.push_str("(ref.null exn)"),
        WVal::Exn(Some(e)) => {
            out.push_str("exnref");
            if depth > 0 {
                match e.get_values(d) {
                    Ok(vals) => {
                        out.push_str(" (");
                        for (i, x) in vals.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            format_into(d, &WVal::from_api(x), hex, depth - 1, out);
                        }
                        out.push(')');
                    }
                    Err(e) => {
                        let _ = write!(out, " <error: {e:?}>");
                    }
                }
            }
        }
        WVal::Extern { null: true } => out.push_str("(ref.null extern)"),
        WVal::Extern { null: false } => out.push_str("externref <host object>"),
        WVal::Any(None) => out.push_str("(ref.null any)"),
        WVal::Any(Some(a)) => match a.kind(d) {
            Err(e) => {
                let _ = write!(out, "anyref <error: {e:?}>");
            }
            Ok(AnyrefKind::I31(x)) => {
                let _ = write!(out, "i31 {x}");
            }
            Ok(AnyrefKind::Other) => out.push_str("anyref <host object>"),
            Ok(AnyrefKind::StructRef(n)) => {
                let _ = write!(out, "struct ({n} fields)");
                if depth == 0 {
                    return;
                }
                match a.struct_fields(d) {
                    Ok(fields) => {
                        out.push_str(" {");
                        for (i, f) in fields.iter().enumerate() {
                            let _ = write!(out, "{}{i}: ", if i > 0 { ", " } else { " " });
                            format_into(d, &WVal::from_api(f), hex, depth - 1, out);
                        }
                        out.push_str(if fields.is_empty() { "}" } else { " }" });
                    }
                    Err(e) => {
                        let _ = write!(out, " <error: {e:?}>");
                    }
                }
            }
            Ok(AnyrefKind::ArrayRef(n)) => {
                let _ = write!(out, "array (len {n})");
                if depth == 0 {
                    return;
                }
                match a.array_elements(d, 0, n.min(MAX_GC_ELEMS)) {
                    Ok(elems) => {
                        out.push_str(" [");
                        for (i, e) in elems.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            format_into(d, &WVal::from_api(e), hex, depth - 1, out);
                        }
                        if n > MAX_GC_ELEMS {
                            out.push_str(", ...");
                        }
                        out.push(']');
                    }
                    Err(e) => {
                        let _ = write!(out, " <error: {e:?}>");
                    }
                }
            }
        },
        WVal::Table(t) => {
            let _ = write!(out, "table (len {})", t.len(d));
        }
    }
}
