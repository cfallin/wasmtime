//! Wasm module structure: names, function bodies, disassembly.

use crate::demangle::demangle;
use crate::dwarf::DebugInfo;
use anyhow::{Context, Result, bail};
use std::cell::OnceCell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use wasmparser::{
    CompositeInnerType, KnownCustom, Name, Operator, Parser, Payload, TypeRef, ValType,
};

/// Render a value type in WAT syntax, e.g. `i32`, `funcref`,
/// `(ref null $3)`.
pub fn valtype_str(t: &ValType) -> String {
    use wasmparser::{HeapType, UnpackedIndex};
    match t {
        ValType::Ref(rt) => {
            #[allow(unreachable_patterns, reason = "`Id` exists only with some features")]
            let idx = |i: &UnpackedIndex| match i {
                UnpackedIndex::Module(i) => format!("${i}"),
                UnpackedIndex::RecGroup(i) => format!("$rec{i}"),
                other => format!("{other:?}"),
            };
            let (heap, abstract_) = match rt.heap_type() {
                HeapType::Abstract { shared, ty } => {
                    let name = format!("{ty:?}").to_lowercase();
                    let name = if shared {
                        format!("shared {name}")
                    } else {
                        name
                    };
                    (name, true)
                }
                HeapType::Concrete(i) => (idx(&i), false),
                HeapType::Exact(i) => (format!("(exact {})", idx(&i)), false),
            };
            if abstract_ && rt.is_nullable() && !heap.contains(' ') {
                format!("{heap}ref")
            } else if rt.is_nullable() {
                format!("(ref null {heap})")
            } else {
                format!("(ref {heap})")
            }
        }
        t => t.to_string(),
    }
}

/// A function signature, rendered for display.
#[derive(Clone, Debug, Default)]
pub struct FuncSig {
    pub params: Vec<String>,
    pub results: Vec<String>,
}

impl std::fmt::Display for FuncSig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.params.join(", "))?;
        if !self.results.is_empty() {
            write!(f, " -> ({})", self.results.join(", "))?;
        }
        Ok(())
    }
}

/// A function defined in the module (i.e., with a body).
#[derive(Clone, Debug)]
pub struct FuncBody {
    /// Function index in the module's function index space.
    pub index: u32,
    /// Byte range of the body (including local declarations) within
    /// the module binary.
    pub range: Range<u32>,
    /// Offset of the first instruction of the body.
    pub code_start: u32,
    /// Types of declared (non-parameter) locals.
    pub locals: Vec<ValType>,
}

/// An import, for display.
#[derive(Clone, Debug)]
pub struct ImportInfo {
    pub module: String,
    pub name: String,
    pub kind: &'static str,
}

/// One line of disassembly.
#[derive(Clone, Debug)]
pub struct DisasmLine {
    /// Offset of this instruction in the original module binary.
    pub offset: u32,
    /// The text of the instruction (WAT syntax, without indentation).
    pub text: String,
    /// Nesting depth (for indentation).
    pub indent: usize,
}

/// Parsed information about one core Wasm module.
pub struct ModuleInfo {
    /// The full module bytecode.
    pub bytes: Arc<[u8]>,
    /// Module name, from the name section, if any.
    pub name: Option<String>,
    /// Offset of the code section's contents in the module binary. DWARF
    /// addresses are relative to this offset.
    pub code_section_offset: Option<u32>,
    /// Number of imported functions (these precede defined functions in
    /// the function index space).
    pub num_imported_funcs: u32,
    /// Number of imported globals/memories/tables.
    pub num_imported_globals: u32,
    pub num_imported_memories: u32,
    pub num_imported_tables: u32,
    /// Type index for every function (imported and defined).
    pub func_types: Vec<u32>,
    /// Signatures, by type index (`None` for non-function types).
    pub sigs: Vec<Option<FuncSig>>,
    /// Defined function bodies, in order.
    pub bodies: Vec<FuncBody>,
    /// Imports, in order.
    pub imports: Vec<ImportInfo>,
    /// Export names for functions, by function index.
    pub func_exports: HashMap<u32, String>,
    /// Function names from the name section (raw, possibly mangled).
    pub func_names: HashMap<u32, String>,
    /// Local names from the name section: (func, local) -> name.
    pub local_names: HashMap<(u32, u32), String>,
    /// Global names from the name section.
    pub global_names: HashMap<u32, String>,
    /// Table names from the name section.
    pub table_names: HashMap<u32, String>,
    /// Number of defined globals, memories, tables.
    pub num_defined_globals: u32,
    pub num_defined_memories: u32,
    pub num_defined_tables: u32,
    /// Parsed DWARF debug info, if the module carries any.
    pub debug: Option<DebugInfo>,
    /// A load error for DWARF, if one occurred, to report to the user.
    pub debug_error: Option<String>,
    disasm: OnceCell<Result<Vec<DisasmLine>, String>>,
}

impl ModuleInfo {
    /// Parse a module from its bytecode.
    pub fn parse(bytes: impl Into<Arc<[u8]>>) -> Result<ModuleInfo> {
        let bytes: Arc<[u8]> = bytes.into();
        let mut info = ModuleInfo {
            bytes: bytes.clone(),
            name: None,
            code_section_offset: None,
            num_imported_funcs: 0,
            num_imported_globals: 0,
            num_imported_memories: 0,
            num_imported_tables: 0,
            func_types: vec![],
            sigs: vec![],
            bodies: vec![],
            imports: vec![],
            func_exports: HashMap::new(),
            func_names: HashMap::new(),
            local_names: HashMap::new(),
            global_names: HashMap::new(),
            table_names: HashMap::new(),
            num_defined_globals: 0,
            num_defined_memories: 0,
            num_defined_tables: 0,
            debug: None,
            debug_error: None,
            disasm: OnceCell::new(),
        };
        let mut dwarf_sections: HashMap<String, Range<usize>> = HashMap::new();

        for payload in Parser::new(0).parse_all(&bytes) {
            match payload.context("failed to parse Wasm module")? {
                Payload::TypeSection(reader) => {
                    for group in reader {
                        for sub in group?.into_types() {
                            info.sigs.push(match &sub.composite_type.inner {
                                CompositeInnerType::Func(ft) => Some(FuncSig {
                                    params: ft.params().iter().map(valtype_str).collect(),
                                    results: ft.results().iter().map(valtype_str).collect(),
                                }),
                                _ => None,
                            });
                        }
                    }
                }
                Payload::ImportSection(reader) => {
                    for import in reader.into_imports() {
                        let import = import?;
                        let kind = match import.ty {
                            TypeRef::Func(ty) | TypeRef::FuncExact(ty) => {
                                info.num_imported_funcs += 1;
                                info.func_types.push(ty);
                                "func"
                            }
                            TypeRef::Global(_) => {
                                info.num_imported_globals += 1;
                                "global"
                            }
                            TypeRef::Memory(_) => {
                                info.num_imported_memories += 1;
                                "memory"
                            }
                            TypeRef::Table(_) => {
                                info.num_imported_tables += 1;
                                "table"
                            }
                            TypeRef::Tag(_) => "tag",
                        };
                        info.imports.push(ImportInfo {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                            kind,
                        });
                    }
                }
                Payload::FunctionSection(reader) => {
                    for ty in reader {
                        info.func_types.push(ty?);
                    }
                }
                Payload::GlobalSection(reader) => info.num_defined_globals = reader.count(),
                Payload::MemorySection(reader) => info.num_defined_memories = reader.count(),
                Payload::TableSection(reader) => info.num_defined_tables = reader.count(),
                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        if export.kind == wasmparser::ExternalKind::Func {
                            info.func_exports
                                .entry(export.index)
                                .or_insert_with(|| export.name.to_string());
                        }
                    }
                }
                Payload::CodeSectionStart {
                    unchecked_range, ..
                } => {
                    info.code_section_offset = Some(u32::try_from(unchecked_range.start)?);
                }
                Payload::CodeSectionEntry(body) => {
                    let index = info.num_imported_funcs + u32::try_from(info.bodies.len())?;
                    let range = body.range();
                    let mut locals = vec![];
                    let mut locals_reader = body.get_locals_reader()?;
                    for _ in 0..locals_reader.get_count() {
                        let (count, ty) = locals_reader.read()?;
                        for _ in 0..count {
                            locals.push(ty);
                        }
                    }
                    let code_start = u32::try_from(locals_reader.original_position())?;
                    info.bodies.push(FuncBody {
                        index,
                        range: u32::try_from(range.start)?..u32::try_from(range.end)?,
                        code_start,
                        locals,
                    });
                }
                Payload::CustomSection(reader) => {
                    let name = reader.name();
                    if name.starts_with(".debug_") {
                        let r = reader.data_range();
                        dwarf_sections.insert(
                            name.to_string(),
                            usize::try_from(r.start)?..usize::try_from(r.end)?,
                        );
                    }
                    if let KnownCustom::Name(names) = reader.as_known() {
                        // Ignore errors in the name section: it's
                        // advisory only.
                        let _ = info.parse_names(names);
                    }
                }
                _ => {}
            }
        }

        if !dwarf_sections.is_empty() {
            match DebugInfo::load(&bytes, &dwarf_sections) {
                Ok(d) => info.debug = Some(d),
                Err(e) => info.debug_error = Some(format!("{e:#}")),
            }
        }

        Ok(info)
    }

    fn parse_names(&mut self, names: wasmparser::NameSectionReader<'_>) -> Result<()> {
        for name in names {
            match name? {
                Name::Module { name, .. } => self.name = Some(name.to_string()),
                Name::Function(map) => {
                    for naming in map {
                        let naming = naming?;
                        self.func_names
                            .insert(naming.index, naming.name.to_string());
                    }
                }
                Name::Local(map) => {
                    for f in map {
                        let f = f?;
                        for naming in f.names {
                            let naming = naming?;
                            self.local_names
                                .insert((f.index, naming.index), naming.name.to_string());
                        }
                    }
                }
                Name::Global(map) => {
                    for naming in map {
                        let naming = naming?;
                        self.global_names
                            .insert(naming.index, naming.name.to_string());
                    }
                }
                Name::Table(map) => {
                    for naming in map {
                        let naming = naming?;
                        self.table_names
                            .insert(naming.index, naming.name.to_string());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Total number of functions (imported plus defined).
    pub fn num_funcs(&self) -> u32 {
        u32::try_from(self.func_types.len()).unwrap()
    }

    /// Total number of globals.
    pub fn num_globals(&self) -> u32 {
        self.num_imported_globals + self.num_defined_globals
    }

    /// Total number of memories.
    pub fn num_memories(&self) -> u32 {
        self.num_imported_memories + self.num_defined_memories
    }

    /// Total number of tables.
    pub fn num_tables(&self) -> u32 {
        self.num_imported_tables + self.num_defined_tables
    }

    /// A human-readable (demangled) name for function `index`, if any.
    pub fn func_name(&self, index: u32) -> Option<String> {
        if let Some(n) = self.func_names.get(&index) {
            return Some(demangle(n));
        }
        if let Some(n) = self.func_exports.get(&index) {
            return Some(n.clone());
        }
        if index < self.num_imported_funcs {
            let mut i = 0;
            for imp in &self.imports {
                if imp.kind == "func" {
                    if i == index {
                        return Some(format!("{}::{}", imp.module, imp.name));
                    }
                    i += 1;
                }
            }
        }
        None
    }

    /// A display name for function `index`: the demangled name if one
    /// is known, otherwise `func[N]`.
    pub fn func_display_name(&self, index: u32) -> String {
        self.func_name(index)
            .unwrap_or_else(|| format!("func[{index}]"))
    }

    /// Signature of function `index`.
    pub fn func_sig(&self, index: u32) -> Option<&FuncSig> {
        let ty = *self.func_types.get(usize::try_from(index).ok()?)?;
        self.sigs.get(usize::try_from(ty).ok()?)?.as_ref()
    }

    /// The body of defined function `index`, if it is defined here.
    pub fn body(&self, index: u32) -> Option<&FuncBody> {
        let defined = index.checked_sub(self.num_imported_funcs)?;
        self.bodies.get(usize::try_from(defined).ok()?)
    }

    /// The function whose body contains module offset `pc`.
    pub fn func_at_pc(&self, pc: u32) -> Option<&FuncBody> {
        let i = self.bodies.partition_point(|b| b.range.end <= pc);
        let b = self.bodies.get(i)?;
        if b.range.contains(&pc) { Some(b) } else { None }
    }

    /// Convert a module offset to a DWARF (code-section-relative)
    /// address.
    pub fn pc_to_dwarf(&self, pc: u32) -> Option<u64> {
        let base = self.code_section_offset?;
        Some(u64::from(pc.checked_sub(base)?))
    }

    /// Convert a DWARF address to a module offset.
    pub fn dwarf_to_pc(&self, addr: u64) -> Option<u32> {
        let base = self.code_section_offset?;
        u32::try_from(addr).ok()?.checked_add(base)
    }

    /// Names of the Wasm locals of function `index` (params first),
    /// with their types.
    pub fn func_locals(&self, index: u32) -> Vec<(Option<String>, String)> {
        let mut out = vec![];
        let mut i = 0;
        if let Some(sig) = self.func_sig(index) {
            for p in &sig.params {
                out.push((self.local_names.get(&(index, i)).cloned(), p.clone()));
                i += 1;
            }
        }
        if let Some(body) = self.body(index) {
            for l in &body.locals {
                out.push((self.local_names.get(&(index, i)).cloned(), valtype_str(l)));
                i += 1;
            }
        }
        out
    }

    /// Instruction offsets (module-relative) of each instruction in
    /// function `index`, along with whether each is a call.
    pub fn instructions(&self, index: u32) -> Result<Vec<(u32, InstrKind)>> {
        let Some(body) = self.body(index) else {
            bail!("function {index} has no body in this module");
        };
        let start = usize::try_from(body.range.start)?;
        let end = usize::try_from(body.range.end)?;
        let reader = wasmparser::BinaryReader::new(&self.bytes[start..end], u64::try_from(start)?);
        let body = wasmparser::FunctionBody::new(reader);
        let mut ops = body.get_operators_reader()?;
        let mut out = vec![];
        while !ops.eof() {
            let (op, offset) = ops.read_with_offset()?;
            let kind = match op {
                Operator::Call { .. }
                | Operator::CallIndirect { .. }
                | Operator::CallRef { .. } => InstrKind::Call,
                Operator::ReturnCall { .. }
                | Operator::ReturnCallIndirect { .. }
                | Operator::ReturnCallRef { .. } => InstrKind::TailCall,
                Operator::Return => InstrKind::Return,
                _ => InstrKind::Other,
            };
            out.push((u32::try_from(offset)?, kind));
        }
        Ok(out)
    }

    /// The offset of the instruction immediately following the one at
    /// `pc`, if any.
    pub fn next_instruction(&self, pc: u32) -> Option<u32> {
        let f = self.func_at_pc(pc)?;
        let instrs = self.instructions(f.index).ok()?;
        let i = instrs.iter().position(|(o, _)| *o == pc)?;
        instrs.get(i + 1).map(|(o, _)| *o)
    }

    /// The offset of the instruction immediately preceding the one at
    /// `pc` in the same function, if any. For a caller frame's return
    /// address, this is the call instruction.
    pub fn prev_instruction(&self, pc: u32) -> Option<u32> {
        let f = self.func_at_pc(pc)?;
        let instrs = self.instructions(f.index).ok()?;
        let i = instrs.iter().position(|(o, _)| *o == pc)?;
        instrs.get(i.checked_sub(1)?).map(|(o, _)| *o)
    }

    /// Kind of the instruction at `pc`.
    pub fn instruction_kind(&self, pc: u32) -> Option<InstrKind> {
        let f = self.func_at_pc(pc)?;
        let instrs = self.instructions(f.index).ok()?;
        instrs.iter().find(|(o, _)| *o == pc).map(|(_, k)| *k)
    }

    /// Get the full-module disassembly (computed lazily and cached).
    pub fn disassembly(&self) -> Result<&[DisasmLine]> {
        match self
            .disasm
            .get_or_init(|| self.compute_disassembly().map_err(|e| format!("{e:#}")))
        {
            Ok(v) => Ok(&v[..]),
            Err(e) => bail!("disassembly failed: {e}"),
        }
    }

    /// Disassembly of function `index`.
    pub fn disassemble_func(&self, index: u32) -> Result<&[DisasmLine]> {
        let Some(body) = self.body(index) else {
            bail!("function {index} has no body in this module");
        };
        let all = self.disassembly()?;
        let start = all.partition_point(|l| l.offset < body.code_start);
        let end = all.partition_point(|l| l.offset < body.range.end);
        Ok(&all[start..end])
    }

    fn compute_disassembly(&self) -> Result<Vec<DisasmLine>> {
        // Strip custom sections other than `name` before printing:
        // DWARF can be many megabytes and would otherwise be printed
        // as escaped strings. Keep a map from offsets in the stripped
        // binary back to the original.
        let (stripped, segments) = strip_custom_sections(&self.bytes)?;
        let map_offset = |o: u64| -> Option<u32> {
            let i = segments.partition_point(|s| s.0 <= o);
            let (new_start, orig_start) = segments.get(i.checked_sub(1)?)?;
            u32::try_from(o - new_start + orig_start).ok()
        };

        let mut config = wasmprinter::Config::new();
        config.fold_instructions(false);
        let mut storage = String::new();
        let mut out = vec![];
        let code_start = self.code_section_offset.unwrap_or(u32::MAX);
        for (offset, line) in config.offsets_and_lines(&stripped, &mut storage)? {
            let Some(offset) = offset.and_then(map_offset) else {
                continue;
            };
            if offset < code_start {
                continue;
            }
            let trimmed = line.trim_start();
            let indent = (line.len() - trimmed.len()) / 2;
            let mut text = trimmed.trim_end().to_string();
            // The function's final `end` is printed as the closing
            // paren of the `(func ...)` form.
            if text == ")" {
                text = "end".to_string();
            }
            if text.starts_with("(func") || text.starts_with("(local") || text.is_empty() {
                continue;
            }
            out.push(DisasmLine {
                offset,
                text,
                indent,
            });
        }
        out.sort_by_key(|l| l.offset);
        Ok(out)
    }
}

/// Coarse classification of instructions, for stepping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrKind {
    Call,
    TailCall,
    Return,
    Other,
}

/// Remove custom sections (other than `name`) from a core module,
/// returning the new bytes and a list of `(new_offset, orig_offset)`
/// segment starts that map positions in the new binary back to the
/// original.
fn strip_custom_sections(bytes: &[u8]) -> Result<(Vec<u8>, Vec<(u64, u64)>)> {
    if bytes.len() < 8 {
        bail!("module too short");
    }
    let mut out = bytes[..8].to_vec();
    let mut segments = vec![(0u64, 0u64)];
    let mut pos = 8;
    while pos < bytes.len() {
        let sec_start = pos;
        let id = bytes[pos];
        pos += 1;
        let mut reader = wasmparser::BinaryReader::new(&bytes[pos..], u64::try_from(pos)?);
        let size = usize::try_from(reader.read_var_u32()?)?;
        let contents_start = usize::try_from(reader.original_position())?;
        let end = contents_start + size;
        if end > bytes.len() {
            bail!("section extends past end of module");
        }
        let keep = if id == 0 {
            let mut r = wasmparser::BinaryReader::new(
                &bytes[contents_start..end],
                u64::try_from(contents_start)?,
            );
            r.read_string().map(|n| n == "name").unwrap_or(false)
        } else {
            true
        };
        if keep {
            segments.push((u64::try_from(out.len())?, u64::try_from(sec_start)?));
            out.extend_from_slice(&bytes[sec_start..end]);
        }
        pos = end;
    }
    Ok((out, segments))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        let wasm = wat::parse_str(
            r#"
            (module $m
              (import "env" "f" (func $imp (param i32)))
              (func $add (export "add") (param $a i32) (param $b i32) (result i32)
                (local $t i32)
                local.get $a
                local.get $b
                i32.add
                call $helper)
              (func $helper (param i32) (result i32)
                local.get 0
                call $imp
                i32.const 1)
              (@custom "producers" "xyz")
            )
            "#,
        )
        .unwrap();
        let m = ModuleInfo::parse(wasm).unwrap();
        assert_eq!(m.name.as_deref(), Some("m"));
        assert_eq!(m.num_imported_funcs, 1);
        assert_eq!(m.num_funcs(), 3);
        assert_eq!(m.func_display_name(1), "add");
        assert_eq!(m.func_sig(1).unwrap().to_string(), "(i32, i32) -> (i32)");
        let locals = m.func_locals(1);
        assert_eq!(locals.len(), 3);
        assert_eq!(locals[2].0.as_deref(), Some("t"));

        let instrs = m.instructions(1).unwrap();
        assert_eq!(instrs.len(), 5); // 4 + `end`
        assert_eq!(instrs[3].1, InstrKind::Call);
        assert_eq!(m.next_instruction(instrs[3].0), Some(instrs[4].0));

        let dis = m.disassemble_func(1).unwrap();
        let texts: Vec<_> = dis.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "local.get $a",
                "local.get $b",
                "i32.add",
                "call $helper",
                "end"
            ]
        );
        assert_eq!(dis[0].offset, instrs[0].0);
        assert_eq!(m.func_at_pc(instrs[2].0).unwrap().index, 1);
    }
}
