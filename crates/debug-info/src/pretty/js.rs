//! Pretty-printers for JavaScript values (`JS::Value`), as in programs
//! compiled to Wasm by NightMonkey (SpiderMonkey's AOT tier) with
//! `--debug-info`.
//!
//! Nothing about the engine's heap is built in here: the producer
//! describes it in DWARF, as structure types (`js::NativeObject`,
//! `js::Shape`, `js::PropMap`, `JSString`, ...) whose members give field
//! offsets, and enumeration types (`JSValueTag`, `js::ShapeFlags`,
//! `js::PropertyFlags`, ...) whose enumerators give tags and flag bits.
//! If any of that is missing, the printer does not match and values are
//! shown raw.
//!
//! Values are shown as a JS console would: primitives as literals,
//! arrays as `[1, 2, 3]`, plain objects as `{x: 1, y: "a"}`, class
//! instances as `Point {x: 3, y: 4}`, and functions as `[Function: f]`.
//! Expanding an object lists its own properties and its
//! `[[Prototype]]`; `obj.name` and `arr[i]` access paths follow
//! JavaScript lookup (own properties, then the prototype chain).

use super::*;
use anyhow::bail;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;

pub(super) static PRINTERS: &[&dyn PrettyPrinter] = &[&JsValuePrinter];

/// The engine's heap layout, as described by the DWARF.
pub(crate) struct Layout {
    value_ty: TypeId,

    tag_shift: u64,
    tag_clear: u64,
    tag_int32: u64,
    tag_boolean: u64,
    tag_undefined: u64,
    tag_null: u64,
    tag_magic: u64,
    tag_string: u64,
    tag_symbol: u64,
    tag_bigint: u64,
    tag_object: u64,
    magic_names: HashMap<u64, String>,

    obj_shape: u64,
    obj_slots: u64,
    obj_elements: u64,
    obj_fixed: u64,
    elems_header: u64,
    elems_initlen: u64,
    elems_length: u64,

    shape_base: u64,
    shape_flags: u64,
    shape_propmap: u64,
    map_length_mask: u64,
    shape_native_bit: u64,
    fixed_slots_shift: u64,
    fixed_slots_mask: u64,
    base_clasp: u64,
    base_proto: u64,
    clasp_name: u64,

    map_flags: u64,
    map_keys: u64,
    map_capacity: u64,
    map_is_compact: u64,
    map_has_prev: u64,
    compact_infos: u64,
    linked_prev: u64,
    linked_infos: u64,
    prop_slot_shift: u64,
    prop_accessor: u64,
    prop_custom: u64,
    jsid_void: u64,
    jsid_type_mask: u64,
    jsid_int_bit: u64,
    jsid_symbol: u64,

    str_flags: u64,
    str_length: u64,
    str_inline: u64,
    str_chars: u64,
    str_left: u64,
    str_right: u64,
    str_linear: u64,
    str_inline_bit: u64,
    str_latin1: u64,
    sym_desc: u64,
    bigint_flags: u64,
    bigint_len: u64,
    bigint_digits: u64,

    fun_flags: u64,
    fun_atom: u64,
    fun_kind_mask: u64,
    fun_class_ctor: u64,
    fun_arrow: u64,
    getter: u64,
    setter: u64,
}

impl Layout {
    fn load(info: &crate::DebugInfo) -> Result<Layout> {
        let ty = |name: &str| -> Result<TypeId> {
            info.find_type(name)?
                .ok_or_else(|| anyhow!("no type `{name}`"))
        };
        let off = |sname: &str, member: &str| -> Result<u64> {
            let t = info.ty(ty(sname)?);
            let TypeKind::Struct(st) = &t.kind else {
                bail!("`{sname}` is not a struct");
            };
            st.members
                .iter()
                .find(|m| m.name.as_deref() == Some(member))
                .map(|m| m.byte_offset())
                .ok_or_else(|| anyhow!("`{sname}` has no member `{member}`"))
        };
        let size = |sname: &str| -> Result<u64> {
            info.type_size(ty(sname)?)
                .ok_or_else(|| anyhow!("`{sname}` has no size"))
        };
        let enumerators = |ename: &str| -> Result<Vec<(String, i128)>> {
            match &info.ty(ty(ename)?).kind {
                TypeKind::Enum { enumerators, .. } => Ok(enumerators.clone()),
                _ => bail!("`{ename}` is not an enumeration"),
            }
        };
        let k = |ename: &str, name: &str| -> Result<u64> {
            enumerators(ename)?
                .into_iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v as u64)
                .ok_or_else(|| anyhow!("`{ename}` has no enumerator `{name}`"))
        };
        let magic_names = enumerators("JSWhyMagic")?
            .into_iter()
            .map(|(n, v)| (v as u64, n))
            .collect();
        let elems_header = size("js::ObjectElements")?;
        Ok(Layout {
            value_ty: ty("JS::Value")?,
            tag_shift: k("JSValueTag", "JSVAL_TAG_SHIFT")?,
            tag_clear: k("JSValueTag", "JSVAL_TAG_CLEAR")?,
            tag_int32: k("JSValueTag", "JSVAL_TAG_INT32")?,
            tag_boolean: k("JSValueTag", "JSVAL_TAG_BOOLEAN")?,
            tag_undefined: k("JSValueTag", "JSVAL_TAG_UNDEFINED")?,
            tag_null: k("JSValueTag", "JSVAL_TAG_NULL")?,
            tag_magic: k("JSValueTag", "JSVAL_TAG_MAGIC")?,
            tag_string: k("JSValueTag", "JSVAL_TAG_STRING")?,
            tag_symbol: k("JSValueTag", "JSVAL_TAG_SYMBOL")?,
            tag_bigint: k("JSValueTag", "JSVAL_TAG_BIGINT")?,
            tag_object: k("JSValueTag", "JSVAL_TAG_OBJECT")?,
            magic_names,
            obj_shape: off("js::NativeObject", "shape_")?,
            obj_slots: off("js::NativeObject", "slots_")?,
            obj_elements: off("js::NativeObject", "elements_")?,
            obj_fixed: off("js::NativeObject", "fixedSlots_")?,
            elems_header,
            elems_initlen: off("js::ObjectElements", "initializedLength")?,
            elems_length: off("js::ObjectElements", "length")?,
            shape_base: off("js::Shape", "base_")?,
            shape_flags: off("js::Shape", "immutableFlags")?,
            shape_propmap: off("js::Shape", "propMap_")?,
            map_length_mask: k("js::ShapeFlags", "MAP_LENGTH_MASK")?,
            shape_native_bit: k("js::ShapeFlags", "IS_NATIVE_BIT")?,
            fixed_slots_shift: k("js::ShapeFlags", "FIXED_SLOTS_SHIFT")?,
            fixed_slots_mask: k("js::ShapeFlags", "FIXED_SLOTS_MASK")?,
            base_clasp: off("js::BaseShape", "clasp_")?,
            base_proto: off("js::BaseShape", "proto_")?,
            clasp_name: off("JSClass", "name")?,
            map_flags: off("js::PropMap", "flags_")?,
            map_keys: off("js::PropMap", "keys_")?,
            map_capacity: k("js::PropMapFlags", "Capacity")?,
            map_is_compact: k("js::PropMapFlags", "IsCompact")?,
            map_has_prev: k("js::PropMapFlags", "HasPrev")?,
            compact_infos: off("js::CompactPropMap", "propInfos_")?,
            linked_prev: off("js::LinkedPropMap", "previous_")?,
            linked_infos: off("js::LinkedPropMap", "propInfos_")?,
            prop_slot_shift: k("js::PropertyFlags", "SlotShift")?,
            prop_accessor: k("js::PropertyFlags", "AccessorProperty")?,
            prop_custom: k("js::PropertyFlags", "CustomDataProperty")?,
            jsid_void: k("js::PropertyKeyTags", "VoidTypeTag")?,
            jsid_type_mask: k("js::PropertyKeyTags", "TypeMask")?,
            jsid_int_bit: k("js::PropertyKeyTags", "IntTagBit")?,
            jsid_symbol: k("js::PropertyKeyTags", "SymbolTypeTag")?,
            str_flags: off("JSString", "flags_")?,
            str_length: off("JSString", "length_")?,
            str_inline: off("JSString", "inlineStorage_")?,
            str_chars: off("JSString", "nonInlineChars_")?,
            str_left: off("JSString", "left_")?,
            str_right: off("JSString", "right_")?,
            str_linear: k("js::StringFlags", "LINEAR_BIT")?,
            str_inline_bit: k("js::StringFlags", "INLINE_CHARS_BIT")?,
            str_latin1: k("js::StringFlags", "LATIN1_CHARS_BIT")?,
            sym_desc: off("JS::Symbol", "description_")?,
            bigint_flags: off("JS::BigInt", "flags_")?,
            bigint_len: off("JS::BigInt", "digitLength_")?,
            bigint_digits: off("JS::BigInt", "inlineDigits_")?,
            fun_flags: off("JSFunction", "flagsAndArgCount_")?,
            fun_atom: off("JSFunction", "atom_")?,
            fun_kind_mask: k("js::FunctionFlags", "FUNCTION_KIND_MASK")?,
            fun_class_ctor: k("js::FunctionFlags", "CLASSCONSTRUCTOR_KIND")?,
            fun_arrow: k("js::FunctionFlags", "ARROW_KIND")?,
            getter: off("js::GetterSetter", "getter_")?,
            setter: off("js::GetterSetter", "setter_")?,
        })
    }

    /// The layout for `ctx`'s debug info (loaded once), if it describes
    /// one.
    fn of(ctx: &Ctx<'_>) -> Option<Rc<Layout>> {
        let mut cache = ctx.info.js_layout.borrow_mut();
        cache
            .get_or_insert_with(|| Layout::load(ctx.info).ok().map(Rc::new))
            .clone()
    }
}

/// A decoded `JS::Value`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Js {
    Double(f64),
    Int32(i32),
    Bool(bool),
    Undefined,
    Null,
    Magic(u64),
    String(u64),
    Symbol(u64),
    BigInt(u64),
    Object(u64),
    Other(u64),
}

/// An own property's key.
enum Key {
    Name(String),
    Index(u64),
    Symbol(String),
}

impl Key {
    fn display(&self) -> String {
        match self {
            Key::Name(n) => n.clone(),
            Key::Index(i) => i.to_string(),
            Key::Symbol(d) => format!("[Symbol({d})]"),
        }
    }

    /// The key as an object literal would write it (quoted unless it is
    /// an identifier or an index).
    fn label(&self) -> String {
        match self {
            Key::Name(n) if !is_identifier(n) => {
                js_quote(&n.encode_utf16().collect::<Vec<_>>(), false)
            }
            _ => self.display(),
        }
    }
}

/// An own property: its key and where its value is (or its accessors).
struct Prop {
    key: Key,
    loc: PropLoc,
}

enum PropLoc {
    Slot(u64),
    Accessor { getter: u64, setter: u64 },
    /// No slot (an engine-computed property, such as an array's
    /// `length`).
    Custom,
}

struct Heap<'a, 'b> {
    ctx: &'a Ctx<'b>,
    l: &'a Layout,
}

impl Heap<'_, '_> {
    fn r32(&self, addr: u64) -> Result<u64> {
        let b = self.ctx.target.read_memory(0, addr, 4)?;
        Ok(u64::from(u32::from_le_bytes(b[..4].try_into()?)))
    }

    fn r16(&self, addr: u64) -> Result<u64> {
        let b = self.ctx.target.read_memory(0, addr, 2)?;
        Ok(u64::from(u16::from_le_bytes(b[..2].try_into()?)))
    }

    fn r64(&self, addr: u64) -> Result<u64> {
        let b = self.ctx.target.read_memory(0, addr, 8)?;
        Ok(u64::from_le_bytes(b[..8].try_into()?))
    }

    /// A GC cell's header word as a pointer (the low bits are GC flags).
    fn header_ptr(&self, cell: u64) -> Result<u64> {
        Ok(self.r32(cell)? & !7)
    }

    fn decode(&self, bits: u64) -> Js {
        let l = self.l;
        let tag = bits >> l.tag_shift;
        let payload = bits & 0xffff_ffff;
        if tag <= l.tag_clear {
            return Js::Double(f64::from_bits(bits));
        }
        match tag {
            t if t == l.tag_int32 => Js::Int32(payload as u32 as i32),
            t if t == l.tag_boolean => Js::Bool(payload != 0),
            t if t == l.tag_undefined => Js::Undefined,
            t if t == l.tag_null => Js::Null,
            t if t == l.tag_magic => Js::Magic(payload),
            t if t == l.tag_string => Js::String(payload),
            t if t == l.tag_symbol => Js::Symbol(payload),
            t if t == l.tag_bigint => Js::BigInt(payload),
            t if t == l.tag_object => Js::Object(payload),
            _ => Js::Other(bits),
        }
    }

    /// A synthetic `JS::Value` (for values not stored in memory as such).
    fn synth(&self, bits: u64) -> Value {
        Value::from_bytes(self.l.value_ty, bits.to_le_bytes().to_vec())
    }

    fn object_value(&self, obj: u64) -> Value {
        self.synth((self.l.tag_object << self.l.tag_shift) | obj)
    }

    fn int32_value(&self, i: i32) -> Value {
        self.synth((self.l.tag_int32 << self.l.tag_shift) | u64::from(i as u32))
    }

    fn null_value(&self) -> Value {
        self.synth(self.l.tag_null << self.l.tag_shift)
    }

    /// The characters of string `s` (flattening ropes), up to `max`;
    /// whether it was truncated.
    fn string(&self, s: u64, max: usize) -> Result<(Vec<u16>, bool)> {
        let mut out = vec![];
        let mut stack = vec![s];
        let mut truncated = false;
        while let Some(s) = stack.pop() {
            if out.len() >= max {
                truncated = true;
                break;
            }
            let flags = self.r32(s + self.l.str_flags)?;
            let len = self.r32(s + self.l.str_length)?;
            if flags & self.l.str_linear == 0 {
                // A rope: left, then right.
                stack.push(self.r32(s + self.l.str_right)?);
                stack.push(self.r32(s + self.l.str_left)?);
                continue;
            }
            let chars = if flags & self.l.str_inline_bit != 0 {
                s + self.l.str_inline
            } else {
                self.r32(s + self.l.str_chars)?
            };
            let take = len.min((max - out.len()) as u64);
            if take < len {
                truncated = true;
            }
            if flags & self.l.str_latin1 != 0 {
                let b = self.ctx.target.read_memory(0, chars, take)?;
                out.extend(b.iter().map(|&c| u16::from(c)));
            } else {
                let b = self.ctx.target.read_memory(0, chars, 2 * take)?;
                out.extend(b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
            }
        }
        Ok((out, truncated))
    }

    fn string_lossy(&self, s: u64, max: usize) -> Result<String> {
        let (chars, truncated) = self.string(s, max)?;
        let mut out = String::from_utf16_lossy(&chars);
        if truncated {
            out.push_str("...");
        }
        Ok(out)
    }

    /// The name of the class of object `obj` (`JSClass::name`).
    fn class_name(&self, obj: u64) -> Result<String> {
        let shape = self.header_ptr(obj + self.l.obj_shape)?;
        let base = self.header_ptr(shape + self.l.shape_base)?;
        let clasp = self.header_ptr(base + self.l.base_clasp)?;
        let name = self.r32(clasp + self.l.clasp_name)?;
        let bytes = self.ctx.target.read_memory(0, name, 64)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    /// The `[[Prototype]]` of object `obj`: `Some(0)` for null, `None`
    /// for a lazily computed one (proxies).
    fn proto(&self, obj: u64) -> Result<Option<u64>> {
        let shape = self.header_ptr(obj + self.l.obj_shape)?;
        let base = self.header_ptr(shape + self.l.shape_base)?;
        let proto = self.r32(base + self.l.base_proto)?;
        Ok(if proto == 1 { None } else { Some(proto) })
    }

    /// Whether `obj` is a native object (not a proxy or Wasm GC object):
    /// only those have slots, elements and a property map.
    fn is_native(&self, obj: u64) -> Result<bool> {
        let shape = self.header_ptr(obj + self.l.obj_shape)?;
        Ok(self.r32(shape + self.l.shape_flags)? & self.l.shape_native_bit != 0)
    }

    fn nfixed(&self, obj: u64) -> Result<u64> {
        let shape = self.header_ptr(obj + self.l.obj_shape)?;
        let flags = self.r32(shape + self.l.shape_flags)?;
        Ok((flags >> self.l.fixed_slots_shift) & self.l.fixed_slots_mask)
    }

    /// The address of slot `slot` of object `obj`.
    fn slot_addr(&self, obj: u64, slot: u64) -> Result<u64> {
        let nfixed = self.nfixed(obj)?;
        Ok(if slot < nfixed {
            obj + self.l.obj_fixed + 8 * slot
        } else {
            self.r32(obj + self.l.obj_slots)? + 8 * (slot - nfixed)
        })
    }

    /// Object `obj`'s own properties in definition order (at most
    /// `limit`, newest dropped first; `None` = all).
    fn props(&self, obj: u64, limit: Option<usize>) -> Result<Vec<Prop>> {
        let l = self.l;
        if !self.is_native(obj)? {
            return Ok(vec![]);
        }
        let shape = self.header_ptr(obj + l.obj_shape)?;
        let flags = self.r32(shape + l.shape_flags)?;
        let mut map = self.r32(shape + l.shape_propmap)?;
        let mut len = flags & l.map_length_mask;
        let mut out = vec![];
        // Walk newest to oldest, then reverse. Bounded, in case of a
        // corrupt (or concurrently mutated) chain.
        let mut entries = 0;
        while map != 0 && len != 0 && entries < 100_000 {
            let mflags = self.r32(map + l.map_flags)?;
            for i in (0..len).rev() {
                entries += 1;
                let key = self.r32(map + l.map_keys + 4 * i)?;
                if key == l.jsid_void {
                    continue;
                }
                let info = if mflags & l.map_is_compact != 0 {
                    self.r16(map + l.compact_infos + 2 * i)?
                } else {
                    self.r32(map + l.linked_infos + 4 * i)?
                };
                let key = if key & l.jsid_int_bit != 0 {
                    Key::Index(key >> 1)
                } else if key & l.jsid_type_mask == l.jsid_symbol {
                    let sym = key & !l.jsid_type_mask;
                    let desc = self.r32(sym + l.sym_desc)?;
                    Key::Symbol(if desc == 0 {
                        String::new()
                    } else {
                        self.string_lossy(desc, 64)?
                    })
                } else if key & l.jsid_type_mask == 0 {
                    Key::Name(self.string_lossy(key, 256)?)
                } else {
                    continue;
                };
                let slot = info >> l.prop_slot_shift;
                let loc = if info & l.prop_custom != 0 {
                    PropLoc::Custom
                } else if info & l.prop_accessor != 0 {
                    // The slot holds a GetterSetter (a private GC thing).
                    let gs = self.r32(self.slot_addr(obj, slot)?)?;
                    PropLoc::Accessor {
                        getter: self.r32(gs + l.getter)?,
                        setter: self.r32(gs + l.setter)?,
                    }
                } else {
                    PropLoc::Slot(self.slot_addr(obj, slot)?)
                };
                out.push(Prop { key, loc });
            }
            if mflags & l.map_has_prev == 0 {
                break;
            }
            map = self.r32(map + l.linked_prev)?;
            len = l.map_capacity;
        }
        out.reverse();
        if let Some(limit) = limit {
            out.truncate(limit);
        }
        Ok(out)
    }

    /// Dense elements: (address of the first, initialized length, length).
    fn elements(&self, obj: u64) -> Result<(u64, u64, u64)> {
        let l = self.l;
        let elems = self.r32(obj + l.obj_elements)?;
        let header = elems - l.elems_header;
        Ok((
            elems,
            self.r32(header + l.elems_initlen)?,
            self.r32(header + l.elems_length)?,
        ))
    }

    /// Own data property `name` of `obj`, then along the prototype chain.
    fn lookup(&self, mut obj: u64, name: &str) -> Result<Option<Value>> {
        if name == "__proto__" {
            // An accessor on `Object.prototype`: show what it would return.
            return Ok(match self.proto(obj)? {
                Some(0) => Some(self.null_value()),
                Some(p) => Some(self.object_value(p)),
                None => None,
            });
        }
        for _ in 0..64 {
            if !self.is_native(obj)? {
                return Ok(None);
            }
            if self.class_name(obj)? == "Array" && name == "length" {
                let (_, _, len) = self.elements(obj)?;
                return Ok(Some(self.int32_value(len as i32)));
            }
            if let Ok(i) = name.parse::<u64>() {
                let (elems, initlen, _) = self.elements(obj)?;
                if i < initlen {
                    return Ok(Some(Value::in_memory(self.l.value_ty, elems + 8 * i)));
                }
            }
            for p in self.props(obj, None)? {
                if p.key.display() != name {
                    continue;
                }
                return Ok(match p.loc {
                    PropLoc::Slot(addr) => Some(Value::in_memory(self.l.value_ty, addr)),
                    PropLoc::Accessor { getter, .. } if getter != 0 => {
                        Some(self.object_value(getter))
                    }
                    _ => None,
                });
            }
            match self.proto(obj)? {
                Some(p) if p != 0 => obj = p,
                _ => return Ok(None),
            }
        }
        Ok(None)
    }

    /// A function's name (its atom), if it has one.
    fn function_name(&self, f: u64) -> Result<Option<String>> {
        match self.decode(self.r64(f + self.l.fun_atom)?) {
            Js::String(s) => Ok(Some(self.string_lossy(s, 128)?)),
            _ => Ok(None),
        }
    }

    fn function_kind(&self, f: u64) -> Result<u64> {
        let flags = self.r64(f + self.l.fun_flags)? & 0xffff;
        Ok(flags & self.l.fun_kind_mask)
    }

    /// The name of the constructor of instances whose prototype is
    /// `proto` (its own `constructor` property's function name).
    fn constructor_name(&self, proto: u64) -> Result<Option<String>> {
        for p in self.props(proto, None)? {
            if !matches!(&p.key, Key::Name(n) if n == "constructor") {
                continue;
            }
            let PropLoc::Slot(addr) = p.loc else {
                return Ok(None);
            };
            if let Js::Object(f) = self.decode(self.r64(addr)?) {
                if self.class_name(f)? == "Function" {
                    return self.function_name(f);
                }
            }
        }
        Ok(None)
    }

    fn bigint(&self, b: u64) -> Result<String> {
        let l = self.l;
        let flags = self.r32(b + l.bigint_flags)?;
        let len = self.r32(b + l.bigint_len)?;
        let digits_addr = if len <= 2 {
            b + l.bigint_digits
        } else {
            self.r32(b + l.bigint_digits)?
        };
        if len > 4 {
            return Ok("<bigint>n".into());
        }
        let mut v: u128 = 0;
        for i in (0..len).rev() {
            v = (v << 32) | u128::from(self.r32(digits_addr + 4 * i)?);
        }
        // BigInt's sign is flag bit 3 (`SignBit`).
        let sign = if flags & 8 != 0 { "-" } else { "" };
        Ok(format!("{sign}{v}n"))
    }
}

/// Format a double as JavaScript does (`1`, `1.5`, `NaN`, `-Infinity`).
fn js_number(d: f64) -> String {
    if d.is_nan() {
        "NaN".into()
    } else if d.is_infinite() {
        if d > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else if d == 0.0 {
        if d.is_sign_negative() { "-0" } else { "0" }.into()
    } else if d.abs() < 1e-6 || d.abs() >= 1e21 {
        // Exponent notation, as JS writes it: `1.5e-7`, `1e+21`.
        let s = format!("{d:e}");
        match s.split_once('e') {
            Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
            _ => s,
        }
    } else {
        // Shortest round-trip digits, as JS does.
        format!("{d}")
    }
}

/// Quote a JS string.
fn js_quote(chars: &[u16], truncated: bool) -> String {
    let mut out = String::from("\"");
    for c in char::decode_utf16(chars.iter().copied()) {
        match c {
            Ok('"') => out.push_str("\\\""),
            Ok('\\') => out.push_str("\\\\"),
            Ok('\n') => out.push_str("\\n"),
            Ok('\r') => out.push_str("\\r"),
            Ok('\t') => out.push_str("\\t"),
            Ok(c) if (c as u32) < 0x20 => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            Ok(c) => out.push(c),
            Err(e) => {
                let _ = write!(out, "\\u{:04x}", e.unpaired_surrogate());
            }
        }
    }
    out.push('"');
    if truncated {
        out.push_str("...");
    }
    out
}

/// Whether `name` can be written after a `.` in an access path.
fn is_identifier(name: &str) -> bool {
    let mut cs = name.chars();
    cs.next()
        .is_some_and(|c| c.is_alphabetic() || c == '_' || c == '$')
        && cs.all(|c| c.is_alphanumeric() || c == '_' || c == '$')
}

/// `JS::Value`.
struct JsValuePrinter;

impl JsValuePrinter {
    fn object(&self, h: &Heap<'_, '_>, obj: u64, opts: &FormatOptions) -> Result<Summary> {
        let class = h.class_name(obj)?;
        if !h.is_native(obj)? {
            // A proxy (e.g. the global's WindowProxy): no slots to show.
            return Ok(Summary::Text(format!("[object {class}]")));
        }
        let limit = opts.max_elems;
        let proto = h.proto(obj)?;
        let proto_value = match proto {
            Some(0) => Some(h.null_value()),
            Some(p) => Some(h.object_value(p)),
            None => None,
        };
        let mut internal = vec![];
        let mut fields = vec![];
        let named = |h: &Heap<'_, '_>, fields: &mut Vec<(String, Value)>| -> Result<u64> {
            let props = h.props(obj, None)?;
            let total = props.len() as u64;
            for p in props.into_iter() {
                if fields.len() >= limit {
                    break;
                }
                let v = match p.loc {
                    PropLoc::Slot(addr) => Value::in_memory(h.l.value_ty, addr),
                    PropLoc::Accessor { getter, setter } => {
                        if getter != 0 {
                            fields.push((format!("get {}", p.key.label()), h.object_value(getter)));
                        }
                        if setter != 0 {
                            fields.push((format!("set {}", p.key.label()), h.object_value(setter)));
                        }
                        continue;
                    }
                    PropLoc::Custom => continue,
                };
                fields.push((p.key.label(), v));
            }
            Ok(total)
        };
        let (prefix, array, collapsed, total) = match class.as_str() {
            "Array" => {
                let (elems, initlen, len) = h.elements(obj)?;
                for i in 0..initlen.min(limit as u64) {
                    fields.push((i.to_string(), Value::in_memory(h.l.value_ty, elems + 8 * i)));
                }
                internal.push(("length".to_string(), h.int32_value(len as i32)));
                let mut extra = vec![];
                named(h, &mut extra)?;
                internal.extend(extra);
                (String::new(), true, false, len)
            }
            "Function" => {
                let name = h.function_name(obj)?.unwrap_or_default();
                let kind = h.function_kind(obj)?;
                let prefix = if kind == h.l.fun_class_ctor {
                    format!("[class {name}]")
                } else if kind == h.l.fun_arrow {
                    format!("[Function (arrow): {name}]")
                } else if name.is_empty() {
                    "[Function (anonymous)]".to_string()
                } else {
                    format!("[Function: {name}]")
                };
                let total = named(h, &mut fields)?;
                (prefix, false, true, total)
            }
            _ => {
                // Indexed properties first, as JS orders them (skipping
                // holes: they are absent properties).
                let (elems, initlen, _) = h.elements(obj)?;
                let mut present = 0;
                for i in 0..initlen {
                    let addr = elems + 8 * i;
                    if matches!(h.decode(h.r64(addr)?), Js::Magic(_)) {
                        continue;
                    }
                    present += 1;
                    if fields.len() < limit {
                        fields.push((i.to_string(), Value::in_memory(h.l.value_ty, addr)));
                    }
                }
                let total = named(h, &mut fields)? + present;
                // Name instances by their constructor, as a console does;
                // a plain object (`Object`'s) gets no prefix.
                let ctor = match proto {
                    Some(p) if p != 0 => h.constructor_name(p).ok().flatten(),
                    _ => None,
                };
                let prefix = match (class.as_str(), ctor.as_deref()) {
                    ("Object", Some("Object")) => String::new(),
                    ("Object", None) if proto == Some(0) => "[Object: null prototype] ".into(),
                    ("Object", Some(c)) if !c.is_empty() => format!("{c} "),
                    (c, _) => format!("{c} "),
                };
                (prefix, false, false, total)
            }
        };
        if let Some(p) = proto_value {
            internal.push(("[[Prototype]]".to_string(), p));
        }
        Ok(Summary::Fields {
            prefix,
            array,
            collapsed,
            fields,
            total,
            internal,
        })
    }
}

impl PrettyPrinter for JsValuePrinter {
    fn name(&self) -> &'static str {
        "js-value"
    }

    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, name: &str) -> bool {
        name == "JS::Value" && ty.size == Some(8) && Layout::of(ctx).is_some()
    }

    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let l = Layout::of(ctx).ok_or_else(|| anyhow!("no JS heap layout"))?;
        let h = Heap { ctx, l: &l };
        let bits = u64::from_le_bytes(ctx.read_bytes(v, 8)?[..8].try_into()?);
        Ok(match h.decode(bits) {
            Js::Double(d) => Summary::Text(js_number(d)),
            Js::Int32(i) => Summary::Text(i.to_string()),
            Js::Bool(b) => Summary::Text(b.to_string()),
            Js::Undefined => Summary::Text("undefined".into()),
            Js::Null => Summary::Text("null".into()),
            Js::Magic(why) => Summary::Text(match l.magic_names.get(&why).map(|s| s.as_str()) {
                Some("JS_UNINITIALIZED_LEXICAL") => "<uninitialized>".into(),
                Some("JS_ELEMENTS_HOLE") => "<empty>".into(),
                Some("JS_OPTIMIZED_OUT") => "<optimized out>".into(),
                Some(n) => format!("<magic {n}>"),
                None => format!("<magic {why}>"),
            }),
            Js::String(s) => {
                let (chars, truncated) = h.string(s, opts.max_string)?;
                Summary::Text(js_quote(&chars, truncated))
            }
            Js::Symbol(s) => {
                let desc = h.r32(s + l.sym_desc)?;
                Summary::Text(if desc == 0 {
                    "Symbol()".into()
                } else {
                    format!("Symbol({})", h.string_lossy(desc, opts.max_string)?)
                })
            }
            Js::BigInt(b) => Summary::Text(h.bigint(b)?),
            Js::Object(obj) => self.object(&h, obj, opts)?,
            Js::Other(bits) => Summary::Text(format!("<value {bits:#018x}>")),
        })
    }

    fn dynamic_type(&self, ctx: &Ctx<'_>, v: &Value) -> Option<String> {
        let l = Layout::of(ctx)?;
        let h = Heap { ctx, l: &l };
        let bits = u64::from_le_bytes(ctx.read_bytes(v, 8).ok()?[..8].try_into().ok()?);
        Some(match h.decode(bits) {
            Js::Double(_) | Js::Int32(_) => "number".into(),
            Js::Bool(_) => "boolean".into(),
            Js::Undefined => "undefined".into(),
            Js::Null => "null".into(),
            Js::String(_) => "string".into(),
            Js::Symbol(_) => "symbol".into(),
            Js::BigInt(_) => "bigint".into(),
            Js::Magic(_) | Js::Other(_) => return None,
            Js::Object(obj) => {
                let class = h.class_name(obj).ok()?;
                if class != "Object" || !h.is_native(obj).ok()? {
                    return Some(class);
                }
                match h.proto(obj).ok()? {
                    Some(p) if p != 0 => h
                        .constructor_name(p)
                        .ok()
                        .flatten()
                        .filter(|n| !n.is_empty())
                        .unwrap_or(class),
                    _ => class,
                }
            }
        })
    }

    fn index(&self, ctx: &Ctx<'_>, v: &Value, i: u64) -> Option<Result<Value>> {
        self.member(ctx, v, &i.to_string())
    }

    fn member(&self, ctx: &Ctx<'_>, v: &Value, name: &str) -> Option<Result<Value>> {
        let l = Layout::of(ctx)?;
        let h = Heap { ctx, l: &l };
        let r = (|| {
            let bits = u64::from_le_bytes(ctx.read_bytes(v, 8)?[..8].try_into()?);
            match h.decode(bits) {
                Js::Object(obj) => h
                    .lookup(obj, name)?
                    .ok_or_else(|| anyhow!("no property `{name}`")),
                Js::String(s) if name == "length" => {
                    let len = h.r32(s + l.str_length)?;
                    Ok(h.int32_value(len as i32))
                }
                _ => bail!("not an object"),
            }
        })();
        Some(r)
    }
}

/// The child naming used for a JS property in access paths.
pub(crate) fn property_path(name: &str) -> String {
    if is_identifier(name) {
        format!(".{name}")
    } else if name.parse::<u64>().is_ok() {
        format!("[{name}]")
    } else {
        // Not expressible in the access-path syntax.
        String::new()
    }
}
