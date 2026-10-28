//! DWARF debug information: line tables, functions, inline frames,
//! scopes, variables and types.
//!
//! All addresses in this module are DWARF addresses, which for Wasm are
//! offsets relative to the start of the code section's contents. Use
//! [`ModuleInfo::pc_to_dwarf`](crate::ModuleInfo::pc_to_dwarf) and
//! [`ModuleInfo::dwarf_to_pc`](crate::ModuleInfo::dwarf_to_pc) to convert.

use crate::demangle::demangle;
use crate::eval::{self, Location};
use crate::target::Target;
use crate::types::*;
use anyhow::{Result, bail};
use gimli::{AttributeValue, DebuggingInformationEntry, Reader, UnitOffset};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

/// The gimli reader type used throughout.
pub type R = gimli::EndianArcSlice<gimli::LittleEndian>;

/// Addresses at or above this are linker tombstones for discarded code.
const TOMBSTONE_MIN: u64 = 0xffff_fff0;

/// A reference to a DIE: compilation unit index plus offset in unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DieRef {
    pub unit: usize,
    pub offset: UnitOffset<usize>,
}

struct UnitInfo {
    unit: gimli::Unit<R>,
    lang: Lang,
    /// Map from this unit's line-program file indices to global file
    /// ids.
    files: Vec<Option<u32>>,
}

/// One row of a line table.
#[derive(Clone, Copy, Debug)]
pub struct LineRow {
    pub addr: u64,
    pub file: u32,
    pub line: u32,
    pub column: u32,
    pub is_stmt: bool,
    pub prologue_end: bool,
}

struct Sequence {
    start: u64,
    end: u64,
    rows: Vec<LineRow>,
}

/// A function (concrete subprogram with code).
#[derive(Clone, Debug)]
pub struct Function {
    /// Display name (demangled, qualified).
    pub name: String,
    /// Address ranges of the function's code.
    pub ranges: Vec<Range<u64>>,
    /// Lowest address (entry point).
    pub low: u64,
    pub die: DieRef,
    pub decl_file: Option<u32>,
    pub decl_line: Option<u32>,
    /// Whether this is the program's main subprogram
    /// (`DW_AT_main_subprogram`, emitted e.g. for Rust's user `main`).
    pub main_subprogram: bool,
}

/// A global (static) variable.
#[derive(Clone, Debug)]
pub struct GlobalVar {
    /// Qualified name.
    pub name: String,
    pub die: DieRef,
}

/// Where a variable lives.
#[derive(Clone, Debug)]
pub enum VarLoc {
    Expr(gimli::Expression<R>),
    List(gimli::LocationListsOffset<usize>),
    Const(Vec<u8>),
    None,
}

/// A variable or parameter visible in some scope.
#[derive(Clone, Debug)]
pub struct Variable {
    pub name: String,
    pub ty: Option<TypeId>,
    pub loc: VarLoc,
    pub is_param: bool,
    /// Lexical nesting depth within its frame (deeper shadows
    /// shallower).
    pub scope_depth: usize,
    pub decl_line: Option<u32>,
    pub die: DieRef,
}

/// A "virtual" frame at a given address: either a concrete function or
/// an inlined subroutine within it.
#[derive(Clone, Debug)]
pub struct ScopeFrame {
    pub name: String,
    pub vars: Vec<Variable>,
    /// Whether this frame represents an inlined call.
    pub inlined: bool,
    /// For inlined frames, the location of the call in the caller
    /// (file id, line).
    pub call_site: Option<(u32, u32)>,
}

impl ScopeFrame {
    /// Find a variable by name, preferring the innermost declaration.
    pub fn find_var(&self, name: &str) -> Option<&Variable> {
        self.vars
            .iter()
            .filter(|v| v.name == name)
            .max_by_key(|v| v.scope_depth)
    }
}

/// A resolved source location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLoc {
    pub file: u32,
    pub line: u32,
    pub column: u32,
}

/// Parsed DWARF for one module.
pub struct DebugInfo {
    dwarf: gimli::Dwarf<R>,
    units: Vec<UnitInfo>,
    /// All source file paths referenced by line tables.
    pub files: Vec<String>,
    sequences: Vec<Sequence>,
    /// All functions, sorted by `low`.
    pub functions: Vec<Function>,
    /// (start, end, function index) for every function range, sorted.
    func_ranges: Vec<(u64, u64, usize)>,
    pub globals: Vec<GlobalVar>,
    /// Qualified-name prefixes for named DIEs nested in namespaces or
    /// types.
    namespaces: HashMap<DieRef, Rc<str>>,
    /// Complete type definitions by qualified name, used to resolve
    /// declarations.
    type_defs: HashMap<String, DieRef>,
    types: RefCell<TypeArena>,
    type_cache: RefCell<HashMap<DieRef, TypeId>>,
    synth_cache: RefCell<HashMap<String, TypeId>>,
    /// Rust trait-object vtables: address -> the vtable variable's DIE
    /// (built lazily).
    vtables: RefCell<Option<HashMap<u64, DieRef>>>,
    /// The JavaScript heap layout the DWARF describes, if any (loaded on
    /// first use by the JS pretty-printer).
    pub(crate) js_layout: RefCell<Option<Option<Rc<crate::pretty::js::Layout>>>>,
}

impl DebugInfo {
    /// Load DWARF from the given module bytes and custom-section ranges
    /// (keyed by section name, e.g. `.debug_info`).
    pub fn load(bytes: &Arc<[u8]>, sections: &HashMap<String, Range<usize>>) -> Result<DebugInfo> {
        let load = |id: gimli::SectionId| -> Result<R, gimli::Error> {
            let data = match sections.get(id.name()) {
                Some(range) => Arc::from(&bytes[range.clone()]),
                None => Arc::from(&[][..]),
            };
            Ok(R::new(data, gimli::LittleEndian))
        };
        let dwarf = gimli::Dwarf::load(load)?;

        let mut info = DebugInfo {
            dwarf,
            units: vec![],
            files: vec![],
            sequences: vec![],
            functions: vec![],
            func_ranges: vec![],
            globals: vec![],
            namespaces: HashMap::new(),
            type_defs: HashMap::new(),
            types: RefCell::new(TypeArena::default()),
            type_cache: RefCell::new(HashMap::new()),
            synth_cache: RefCell::new(HashMap::new()),
            vtables: RefCell::new(None),
            js_layout: RefCell::new(None),
        };

        let mut file_ids: HashMap<String, u32> = HashMap::new();
        let mut headers = info.dwarf.units();
        while let Some(header) = headers.next()? {
            let unit = info.dwarf.unit(header)?;
            let mut lang = Lang::Other;
            {
                let mut cursor = unit.entries();
                if let Some(root) = cursor.next_dfs()? {
                    if let Some(AttributeValue::Language(l)) =
                        root.attr_value(gimli::DW_AT_language)
                    {
                        lang = Lang::from_dwarf(l);
                    }
                }
            }
            let unit_idx = info.units.len();
            let files = info.load_lines(&unit, &mut file_ids)?;
            info.units.push(UnitInfo { unit, lang, files });
            info.scan_unit(unit_idx)?;
        }

        info.sequences.sort_by_key(|s| s.start);
        info.functions.sort_by_key(|f| f.low);
        for (i, f) in info.functions.iter().enumerate() {
            for r in &f.ranges {
                info.func_ranges.push((r.start, r.end, i));
            }
        }
        info.func_ranges.sort();
        Ok(info)
    }

    fn load_lines(
        &mut self,
        unit: &gimli::Unit<R>,
        file_ids: &mut HashMap<String, u32>,
    ) -> Result<Vec<Option<u32>>> {
        let Some(program) = unit.line_program.clone() else {
            return Ok(vec![]);
        };
        let header = program.header().clone();
        let comp_dir = match &unit.comp_dir {
            Some(d) => d.to_string_lossy()?.into_owned(),
            None => String::new(),
        };
        // Map file indices to global ids.
        let base = if header.version() >= 5 { 0 } else { 1 };
        let mut files = vec![None; header.file_names().len() + base];
        for (i, file) in header.file_names().iter().enumerate() {
            let name = self
                .dwarf
                .attr_string(unit, file.path_name())?
                .to_string_lossy()?
                .into_owned();
            let path = if name.starts_with('/') || name.contains("://") {
                name
            } else {
                let dir = match file.directory(&header) {
                    Some(d) => self
                        .dwarf
                        .attr_string(unit, d)?
                        .to_string_lossy()?
                        .into_owned(),
                    None => String::new(),
                };
                let dir = if dir.starts_with('/') || dir.contains("://") || comp_dir.is_empty() {
                    dir
                } else if dir.is_empty() {
                    comp_dir.clone()
                } else {
                    format!("{comp_dir}/{dir}")
                };
                if dir.is_empty() {
                    name
                } else {
                    format!("{dir}/{name}")
                }
            };
            let path = normalize_path(&path);
            let id = *file_ids.entry(path.clone()).or_insert_with(|| {
                self.files.push(path);
                u32::try_from(self.files.len() - 1).unwrap()
            });
            files[i + base] = Some(id);
        }

        let mut rows = program.rows();
        let mut current: Vec<LineRow> = vec![];
        while let Some((_, row)) = rows.next_row()? {
            if row.end_sequence() {
                if let Some(first) = current.first() {
                    let start = first.addr;
                    if start < TOMBSTONE_MIN {
                        self.sequences.push(Sequence {
                            start,
                            end: row.address(),
                            rows: std::mem::take(&mut current),
                        });
                    }
                }
                current.clear();
                continue;
            }
            let file = files
                .get(usize::try_from(row.file_index())?)
                .copied()
                .flatten()
                .unwrap_or(u32::MAX);
            current.push(LineRow {
                addr: row.address(),
                file,
                line: row.line().map(|l| l.get() as u32).unwrap_or(0),
                column: match row.column() {
                    gimli::ColumnType::LeftEdge => 0,
                    gimli::ColumnType::Column(c) => c.get() as u32,
                },
                is_stmt: row.is_stmt(),
                prologue_end: row.prologue_end(),
            });
        }
        Ok(files)
    }

    /// Scan a unit's DIE tree for functions, global variables, and
    /// namespace information.
    fn scan_unit(&mut self, unit_idx: usize) -> Result<()> {
        let mut ns_stack: Vec<(isize, String)> = vec![];
        let mut subprogram_depth: Option<isize> = None;
        let mut new_functions: Vec<(DieRef, Vec<Range<u64>>, Option<String>, String)> = vec![];
        let mut new_globals = vec![];
        let mut new_namespaces = vec![];
        let mut new_type_defs = vec![];
        {
            let unit = &self.units[unit_idx].unit;
            let mut cursor = unit.entries();
            while let Some(entry) = cursor.next_dfs()? {
                let depth = entry.depth();
                while ns_stack.last().is_some_and(|(d, _)| *d >= depth) {
                    ns_stack.pop();
                }
                if subprogram_depth.is_some_and(|d| depth <= d) {
                    subprogram_depth = None;
                }
                let die = DieRef {
                    unit: unit_idx,
                    offset: entry.offset(),
                };
                let tag = entry.tag();
                let prefix = || {
                    ns_stack
                        .iter()
                        .map(|(_, n)| n.as_str())
                        .collect::<Vec<_>>()
                        .join("::")
                };
                let name = entry
                    .attr_value(gimli::DW_AT_name)
                    .and_then(|v| self.dwarf.attr_string(unit, v).ok())
                    .map(|s| s.to_string_lossy().map(|s| s.into_owned()))
                    .transpose()?;

                match tag {
                    gimli::DW_TAG_subprogram => {
                        if subprogram_depth.is_none() && entry.has_children() {
                            subprogram_depth = Some(depth);
                        }
                        let mut ranges = vec![];
                        let mut iter = self.dwarf.die_ranges(unit, entry)?;
                        while let Some(r) = iter.next()? {
                            if r.begin < TOMBSTONE_MIN && r.end > r.begin {
                                ranges.push(r.begin..r.end);
                            }
                        }
                        if !ranges.is_empty() {
                            new_functions.push((die, ranges, name.clone(), prefix()));
                        }
                    }
                    gimli::DW_TAG_variable if subprogram_depth.is_none() => {
                        if entry.attr_value(gimli::DW_AT_location).is_some() {
                            if let Some(name) = &name {
                                let p = prefix();
                                let qualified = if p.is_empty() {
                                    name.clone()
                                } else {
                                    format!("{p}::{name}")
                                };
                                new_globals.push(GlobalVar {
                                    name: qualified,
                                    die,
                                });
                            } else if let Some(spec) = entry.attr_value(gimli::DW_AT_specification)
                            {
                                // C++ out-of-line static member
                                // definitions: name via specification.
                                let _ = spec;
                            }
                        }
                    }
                    gimli::DW_TAG_structure_type
                    | gimli::DW_TAG_class_type
                    | gimli::DW_TAG_union_type
                    | gimli::DW_TAG_enumeration_type
                    | gimli::DW_TAG_typedef
                    | gimli::DW_TAG_base_type => {
                        let p = prefix();
                        if !p.is_empty() {
                            new_namespaces.push((die, p.clone()));
                        }
                        if tag != gimli::DW_TAG_typedef
                            && tag != gimli::DW_TAG_base_type
                            && entry.attr_value(gimli::DW_AT_declaration).is_none()
                        {
                            if let Some(name) = &name {
                                let q = if p.is_empty() {
                                    name.clone()
                                } else {
                                    format!("{p}::{name}")
                                };
                                new_type_defs.push((q, die));
                            }
                        }
                    }
                    _ => {}
                }

                let is_scope = matches!(
                    tag,
                    gimli::DW_TAG_namespace
                        | gimli::DW_TAG_structure_type
                        | gimli::DW_TAG_class_type
                        | gimli::DW_TAG_union_type
                        | gimli::DW_TAG_enumeration_type
                );
                if is_scope && entry.has_children() {
                    if let Some(name) = name {
                        ns_stack.push((depth, name));
                    } else {
                        ns_stack.push((depth, String::new()));
                        // Anonymous namespaces/types: keep depth
                        // tracking but don't contribute a name.
                        ns_stack.pop();
                    }
                }
            }
        }

        for (die, p) in new_namespaces {
            self.namespaces.insert(die, Rc::from(p));
        }
        for (q, die) in new_type_defs {
            self.type_defs.entry(q).or_insert(die);
        }
        self.globals.extend(new_globals);
        for (die, ranges, name, prefix) in new_functions {
            let name = self.function_name(die, name, &prefix);
            let low = ranges.iter().map(|r| r.start).min().unwrap();
            let e = self.entry(die)?;
            let decl_file = self
                .attr_follow(die, gimli::DW_AT_decl_file)
                .and_then(|(r, v)| self.file_attr(r.unit, v));
            let decl_line = self
                .attr_follow(die, gimli::DW_AT_decl_line)
                .and_then(|(_, v)| v.udata_value())
                .map(|l| l as u32);
            let main_subprogram = e.attr_value(gimli::DW_AT_main_subprogram).is_some();
            self.functions.push(Function {
                name,
                ranges,
                low,
                die,
                decl_file,
                decl_line,
                main_subprogram,
            });
        }
        Ok(())
    }

    fn function_name(&self, die: DieRef, name: Option<String>, prefix: &str) -> String {
        if let Some(linkage) = self.die_string_follow(die, gimli::DW_AT_linkage_name) {
            let d = demangle(&linkage);
            if d != linkage || name.is_none() {
                return d;
            }
        }
        let name = name
            .or_else(|| self.die_string_follow(die, gimli::DW_AT_name))
            .unwrap_or_else(|| "<unknown>".to_string());
        if prefix.is_empty() {
            name
        } else {
            format!("{prefix}::{name}")
        }
    }

    // ---------------------------------------------------------------
    // Low-level DIE helpers.

    fn entry(&self, r: DieRef) -> Result<DebuggingInformationEntry<R>> {
        Ok(self.units[r.unit].unit.entry(r.offset)?)
    }

    /// Language of the unit containing `r`.
    pub fn lang(&self, r: DieRef) -> Lang {
        self.units[r.unit].lang
    }

    fn resolve_ref(&self, unit: usize, v: &AttributeValue<R>) -> Option<DieRef> {
        match v {
            AttributeValue::UnitRef(o) => Some(DieRef { unit, offset: *o }),
            AttributeValue::DebugInfoRef(o) => {
                for (i, u) in self.units.iter().enumerate() {
                    if let Some(off) = o.to_unit_offset(&u.unit.header) {
                        return Some(DieRef {
                            unit: i,
                            offset: off,
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Get attribute `at` of `r`, following `DW_AT_abstract_origin` and
    /// `DW_AT_specification` links if not present directly.
    fn attr_follow(&self, r: DieRef, at: gimli::DwAt) -> Option<(DieRef, AttributeValue<R>)> {
        let mut cur = r;
        for _ in 0..8 {
            let e = self.entry(cur).ok()?;
            if let Some(v) = e.attr_value(at) {
                return Some((cur, v));
            }
            let next = e
                .attr_value(gimli::DW_AT_abstract_origin)
                .or_else(|| e.attr_value(gimli::DW_AT_specification))?;
            cur = self.resolve_ref(cur.unit, &next)?;
        }
        None
    }

    fn die_string_follow(&self, r: DieRef, at: gimli::DwAt) -> Option<String> {
        let (r, v) = self.attr_follow(r, at)?;
        let s = self.dwarf.attr_string(&self.units[r.unit].unit, v).ok()?;
        Some(s.to_string_lossy().ok()?.into_owned())
    }

    fn file_attr(&self, unit: usize, v: AttributeValue<R>) -> Option<u32> {
        let idx = match v {
            AttributeValue::FileIndex(i) => i,
            v => v.udata_value()?,
        };
        self.units[unit]
            .files
            .get(usize::try_from(idx).ok()?)
            .copied()
            .flatten()
    }

    fn die_ranges(&self, r: DieRef, e: &DebuggingInformationEntry<R>) -> Result<Vec<Range<u64>>> {
        let mut out = vec![];
        let mut iter = self.dwarf.die_ranges(&self.units[r.unit].unit, e)?;
        while let Some(range) = iter.next()? {
            if range.begin < TOMBSTONE_MIN && range.end > range.begin {
                out.push(range.begin..range.end);
            }
        }
        Ok(out)
    }

    fn children(&self, r: DieRef) -> Result<Vec<(DieRef, DebuggingInformationEntry<R>)>> {
        let unit = &self.units[r.unit].unit;
        let mut tree = unit.entries_tree(Some(r.offset))?;
        let root = tree.root()?;
        let mut out = vec![];
        let mut children = root.children();
        while let Some(child) = children.next()? {
            let e = child.entry().clone();
            out.push((
                DieRef {
                    unit: r.unit,
                    offset: e.offset(),
                },
                e,
            ));
        }
        Ok(out)
    }

    // ---------------------------------------------------------------
    // Line tables.

    /// Look up the line-table row covering `addr`.
    pub fn line_row(&self, addr: u64) -> Option<&LineRow> {
        let i = self.sequences.partition_point(|s| s.start <= addr);
        // Sequences may in principle overlap; search backward a little.
        for s in self.sequences[..i].iter().rev().take(4) {
            if addr >= s.start && addr < s.end {
                let j = s.rows.partition_point(|r| r.addr <= addr);
                return s.rows.get(j.checked_sub(1)?);
            }
        }
        None
    }

    /// Source location for `addr`, if known (and not line 0).
    pub fn source_loc(&self, addr: u64) -> Option<SourceLoc> {
        let row = self.line_row(addr)?;
        if row.line == 0 || row.file == u32::MAX {
            return None;
        }
        Some(SourceLoc {
            file: row.file,
            line: row.line,
            column: row.column,
        })
    }

    /// File path for a file id.
    pub fn file_name(&self, file: u32) -> &str {
        self.files
            .get(usize::try_from(file).unwrap_or(usize::MAX))
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }

    /// Find file ids matching a user query: an exact path, or a path
    /// suffix at a `/` boundary.
    pub fn find_files(&self, query: &str) -> Vec<u32> {
        let mut out = vec![];
        for (i, f) in self.files.iter().enumerate() {
            if f == query || f.ends_with(&format!("/{query}")) {
                out.push(u32::try_from(i).unwrap());
            }
        }
        out
    }

    /// Find breakpoint addresses for `line` in any of `files`. If no
    /// statement exists on exactly that line, the next line with code is
    /// used. Returns the addresses and the actual line used.
    pub fn line_addresses(&self, files: &[u32], line: u32) -> Option<(Vec<u64>, u32)> {
        // Find the best line: exact, or the smallest greater line.
        let mut best: Option<u32> = None;
        for s in &self.sequences {
            for r in &s.rows {
                if r.is_stmt && files.contains(&r.file) && r.line >= line {
                    if best.is_none_or(|b| r.line < b) {
                        best = Some(r.line);
                    }
                }
            }
        }
        let target = best?;
        // Collect entry addresses: rows with the target line whose
        // predecessor in the sequence is on a different line.
        let mut per_func: HashMap<usize, u64> = HashMap::new();
        let mut no_func = vec![];
        for s in &self.sequences {
            for (i, r) in s.rows.iter().enumerate() {
                if !(r.is_stmt && files.contains(&r.file) && r.line == target) {
                    continue;
                }
                if i > 0 {
                    let p = &s.rows[i - 1];
                    if p.line == r.line && p.file == r.file {
                        continue;
                    }
                }
                match self.function_index_at(r.addr) {
                    Some(f) => {
                        let e = per_func.entry(f).or_insert(r.addr);
                        *e = (*e).min(r.addr);
                    }
                    None => no_func.push(r.addr),
                }
            }
        }
        let mut addrs: Vec<u64> = per_func.into_values().chain(no_func).collect();
        // Avoid stopping inside a function's prologue: if the chosen
        // address is the function entry, move to the prologue end.
        for a in addrs.iter_mut() {
            if let Some(f) = self.function_at(*a) {
                if f.low == *a {
                    let after = self.prologue_end(f);
                    if self.source_loc(after).is_some_and(|l| l.line == target) {
                        *a = after;
                    }
                }
            }
        }
        addrs.sort();
        addrs.dedup();
        Some((addrs, target))
    }

    /// All line-table rows within `range`, sorted by address.
    pub fn rows_in(&self, range: &Range<u64>) -> Vec<LineRow> {
        let mut out = vec![];
        for s in &self.sequences {
            if s.end <= range.start || s.start >= range.end {
                continue;
            }
            for r in &s.rows {
                if range.contains(&r.addr) {
                    out.push(*r);
                }
            }
        }
        out.sort_by_key(|r| r.addr);
        out
    }

    /// The address just past `f`'s prologue: a row marked
    /// `prologue_end`, else the first row on a different line from the
    /// entry row, else the entry itself.
    pub fn prologue_end(&self, f: &Function) -> u64 {
        let Some(range) = f.ranges.iter().find(|r| r.start == f.low) else {
            return f.low;
        };
        let rows = self.rows_in(range);
        if let Some(r) = rows.iter().find(|r| r.prologue_end) {
            return r.addr;
        }
        let Some(first) = rows.first() else {
            return f.low;
        };
        rows.iter()
            .find(|r| r.addr > first.addr && r.line != 0 && r.line != first.line)
            .map(|r| r.addr)
            .unwrap_or(f.low)
    }

    // ---------------------------------------------------------------
    // Functions and scopes.

    fn function_index_at(&self, addr: u64) -> Option<usize> {
        let i = self.func_ranges.partition_point(|(s, _, _)| *s <= addr);
        for (s, e, f) in self.func_ranges[..i].iter().rev().take(8) {
            if addr >= *s && addr < *e {
                return Some(*f);
            }
        }
        None
    }

    /// The function containing `addr`.
    pub fn function_at(&self, addr: u64) -> Option<&Function> {
        self.function_index_at(addr).map(|i| &self.functions[i])
    }

    /// Find functions matching `query`: exact qualified name, or a
    /// `::`-suffix match, or a match ignoring generic arguments.
    pub fn find_functions(&self, query: &str) -> Vec<&Function> {
        if query == "main" {
            let mains: Vec<_> = self
                .functions
                .iter()
                .filter(|f| f.main_subprogram)
                .collect();
            if !mains.is_empty() {
                return mains;
            }
        }
        let exact: Vec<_> = self.functions.iter().filter(|f| f.name == query).collect();
        if !exact.is_empty() {
            return exact;
        }
        let suffix = format!("::{query}");
        let query_base = function_base_name(query);
        self.functions
            .iter()
            .filter(|f| {
                let base = function_base_name(&f.name);
                base == query
                    || base == query_base
                    || base.ends_with(&suffix)
                    || f.name.ends_with(&suffix)
            })
            .collect()
    }

    /// Compute the virtual frames (innermost first) at `addr`: inlined
    /// subroutines followed by the containing function. Returns an
    /// empty vector if no function covers `addr`.
    pub fn frames_at(&self, addr: u64) -> Result<Vec<ScopeFrame>> {
        let Some(f) = self.function_at(addr) else {
            return Ok(vec![]);
        };
        let mut frames = vec![ScopeFrame {
            name: f.name.clone(),
            vars: vec![],
            inlined: false,
            call_site: None,
        }];
        self.walk_scope(f.die, addr, &mut frames, 0, 0)?;
        frames.reverse();
        Ok(frames)
    }

    fn walk_scope(
        &self,
        scope: DieRef,
        addr: u64,
        frames: &mut Vec<ScopeFrame>,
        frame: usize,
        depth: usize,
    ) -> Result<()> {
        for (r, e) in self.children(scope)? {
            match e.tag() {
                gimli::DW_TAG_formal_parameter | gimli::DW_TAG_variable => {
                    if let Some(v) = self.make_variable(r, &e, depth)? {
                        frames[frame].vars.push(v);
                    }
                }
                gimli::DW_TAG_lexical_block => {
                    let ranges = self.die_ranges(r, &e)?;
                    if ranges.is_empty() || ranges.iter().any(|x| x.contains(&addr)) {
                        if ranges.is_empty() {
                            // A block with no ranges covers its parent's
                            // extent.
                        }
                        self.walk_scope(r, addr, frames, frame, depth + 1)?;
                    }
                }
                gimli::DW_TAG_inlined_subroutine => {
                    let ranges = self.die_ranges(r, &e)?;
                    if ranges.iter().any(|x| x.contains(&addr)) {
                        let name = self.function_name(r, None, "");
                        let call_file = e
                            .attr_value(gimli::DW_AT_call_file)
                            .and_then(|v| self.file_attr(r.unit, v));
                        let call_line = e
                            .attr_value(gimli::DW_AT_call_line)
                            .and_then(|v| v.udata_value())
                            .map(|l| l as u32);
                        frames.push(ScopeFrame {
                            name,
                            vars: vec![],
                            inlined: true,
                            call_site: call_file.zip(call_line),
                        });
                        let idx = frames.len() - 1;
                        self.walk_scope(r, addr, frames, idx, 0)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn make_variable(
        &self,
        r: DieRef,
        e: &DebuggingInformationEntry<R>,
        depth: usize,
    ) -> Result<Option<Variable>> {
        let Some(name) = self.die_string_follow(r, gimli::DW_AT_name) else {
            return Ok(None);
        };
        let ty = match self.attr_follow(r, gimli::DW_AT_type) {
            Some((tr, v)) => match self.resolve_ref(tr.unit, &v) {
                Some(t) => Some(self.type_of(t)?),
                None => None,
            },
            None => None,
        };
        let loc = self.var_loc(r, e)?;
        let decl_line = self
            .attr_follow(r, gimli::DW_AT_decl_line)
            .and_then(|(_, v)| v.udata_value())
            .map(|l| l as u32);
        Ok(Some(Variable {
            name,
            ty,
            loc,
            is_param: e.tag() == gimli::DW_TAG_formal_parameter,
            scope_depth: depth,
            decl_line,
            die: r,
        }))
    }

    fn var_loc(&self, r: DieRef, e: &DebuggingInformationEntry<R>) -> Result<VarLoc> {
        if let Some(v) = e.attr_value(gimli::DW_AT_location) {
            return Ok(match v {
                AttributeValue::Exprloc(expr) => VarLoc::Expr(expr),
                AttributeValue::LocationListsRef(o) => VarLoc::List(o),
                AttributeValue::SecOffset(o) => VarLoc::List(gimli::LocationListsOffset(o)),
                v => match self
                    .dwarf
                    .attr_locations_offset(&self.units[r.unit].unit, v)?
                {
                    Some(o) => VarLoc::List(o),
                    None => VarLoc::None,
                },
            });
        }
        if let Some(v) = e.attr_value(gimli::DW_AT_const_value) {
            let bytes = match v {
                AttributeValue::Block(b) => b.to_slice()?.into_owned(),
                AttributeValue::Sdata(x) => x.to_le_bytes().to_vec(),
                AttributeValue::Udata(x) => x.to_le_bytes().to_vec(),
                AttributeValue::Data1(x) => vec![x],
                AttributeValue::Data2(x) => x.to_le_bytes().to_vec(),
                AttributeValue::Data4(x) => x.to_le_bytes().to_vec(),
                AttributeValue::Data8(x) => x.to_le_bytes().to_vec(),
                _ => return Ok(VarLoc::None),
            };
            return Ok(VarLoc::Const(bytes));
        }
        Ok(VarLoc::None)
    }

    /// The declared return type of a function, if any (`None` for
    /// `void`).
    pub fn return_type(&self, f: &Function) -> Result<Option<TypeId>> {
        match self.attr_follow(f.die, gimli::DW_AT_type) {
            Some((tr, v)) => match self.resolve_ref(tr.unit, &v) {
                Some(t) => Ok(Some(self.type_of(t)?)),
                None => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// Find a global variable by (qualified or simple) name.
    pub fn find_global(&self, name: &str) -> Option<&GlobalVar> {
        if let Some(g) = self.globals.iter().find(|g| g.name == name) {
            return Some(g);
        }
        let suffix = format!("::{name}");
        self.globals.iter().find(|g| g.name.ends_with(&suffix))
    }

    /// The type of a global variable.
    pub fn global_type(&self, g: &GlobalVar) -> Result<Option<TypeId>> {
        match self.attr_follow(g.die, gimli::DW_AT_type) {
            Some((tr, v)) => match self.resolve_ref(tr.unit, &v) {
                Some(t) => Ok(Some(self.type_of(t)?)),
                None => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// The location of a global variable.
    pub fn global_loc(&self, g: &GlobalVar) -> Result<VarLoc> {
        let e = self.entry(g.die)?;
        self.var_loc(g.die, &e)
    }

    // ---------------------------------------------------------------
    // Location evaluation.

    /// Compute the frame base for function `f` (the value used by
    /// `DW_OP_fbreg`).
    pub fn frame_base(&self, f: &Function, addr: u64, target: &dyn Target) -> Result<Option<u64>> {
        let e = self.entry(f.die)?;
        let loc = match e.attr_value(gimli::DW_AT_frame_base) {
            Some(AttributeValue::Exprloc(expr)) => VarLoc::Expr(expr),
            Some(AttributeValue::LocationListsRef(o)) => VarLoc::List(o),
            Some(AttributeValue::SecOffset(o)) => VarLoc::List(gimli::LocationListsOffset(o)),
            _ => return Ok(None),
        };
        let loc = self.eval_loc(f.die.unit, &loc, addr, None, target)?;
        Ok(match loc {
            Location::WasmLocal(i) => Some(target.local(i)?.as_u64()),
            Location::WasmGlobal(i) => Some(target.global(i)?.as_u64()),
            Location::WasmStack(i) => Some(target.operand(i)?.as_u64()),
            Location::Memory(a) => Some(a),
            Location::Value(bytes) => {
                let mut b = [0u8; 8];
                let n = bytes.len().min(8);
                b[..n].copy_from_slice(&bytes[..n]);
                Some(u64::from_le_bytes(b))
            }
            _ => None,
        })
    }

    /// Evaluate a variable location at `addr`.
    pub fn eval_loc(
        &self,
        unit: usize,
        loc: &VarLoc,
        addr: u64,
        frame_base: Option<u64>,
        target: &dyn Target,
    ) -> Result<Location> {
        let u = &self.units[unit].unit;
        let encoding = u.encoding();
        match loc {
            VarLoc::Expr(expr) => eval::evaluate(expr, encoding, frame_base, target),
            VarLoc::List(offset) => {
                let mut iter = self.dwarf.locations(u, *offset)?;
                while let Some(entry) = iter.next()? {
                    if addr >= entry.range.begin && addr < entry.range.end {
                        return eval::evaluate(&entry.data, encoding, frame_base, target);
                    }
                }
                Ok(Location::Unavailable("optimized out".into()))
            }
            VarLoc::Const(bytes) => Ok(Location::Value(bytes.clone())),
            VarLoc::None => Ok(Location::Unavailable("no location".into())),
        }
    }

    /// Address size (in bytes) of the unit containing `r`.
    pub fn address_size(&self, unit: usize) -> u8 {
        self.units[unit].unit.encoding().address_size
    }

    // ---------------------------------------------------------------
    // Types.

    /// Get a type by id.
    pub fn ty(&self, id: TypeId) -> Rc<Type> {
        self.types.borrow().types[id.0].clone()
    }

    /// Get (building if necessary) the type for a type DIE.
    pub fn type_of(&self, r: DieRef) -> Result<TypeId> {
        if let Some(id) = self.type_cache.borrow().get(&r) {
            return Ok(*id);
        }
        // Resolve declarations to definitions where possible.
        let e = self.entry(r)?;
        if e.attr_value(gimli::DW_AT_declaration).is_some()
            && matches!(
                e.tag(),
                gimli::DW_TAG_structure_type
                    | gimli::DW_TAG_class_type
                    | gimli::DW_TAG_union_type
                    | gimli::DW_TAG_enumeration_type
            )
        {
            if let Some(name) = self.qualified_name(r, &e) {
                if let Some(def) = self.type_defs.get(&name).copied() {
                    if def != r {
                        let id = self.type_of(def)?;
                        self.type_cache.borrow_mut().insert(r, id);
                        return Ok(id);
                    }
                }
            }
        }
        // Allocate a placeholder to break cycles.
        let id = self.types.borrow_mut().alloc(Type {
            name: None,
            size: None,
            kind: TypeKind::Unknown,
            lang: self.lang(r),
        });
        self.type_cache.borrow_mut().insert(r, id);
        let ty = self.build_type(r, &e)?;
        self.types.borrow_mut().types[id.0] = Rc::new(ty);
        Ok(id)
    }

    fn qualified_name(&self, r: DieRef, e: &DebuggingInformationEntry<R>) -> Option<String> {
        let name = e
            .attr_value(gimli::DW_AT_name)
            .and_then(|v| self.dwarf.attr_string(&self.units[r.unit].unit, v).ok())?
            .to_string_lossy()
            .ok()?
            .into_owned();
        Some(match self.namespaces.get(&r) {
            Some(p) => format!("{p}::{name}"),
            None => name,
        })
    }

    fn type_attr(&self, r: DieRef, e: &DebuggingInformationEntry<R>) -> Result<Option<TypeId>> {
        match e.attr_value(gimli::DW_AT_type) {
            Some(v) => match self.resolve_ref(r.unit, &v) {
                Some(t) => Ok(Some(self.type_of(t)?)),
                None => Ok(None),
            },
            None => Ok(None),
        }
    }

    fn build_type(&self, r: DieRef, e: &DebuggingInformationEntry<R>) -> Result<Type> {
        let lang = self.lang(r);
        let name = self.qualified_name(r, e);
        let size = e
            .attr_value(gimli::DW_AT_byte_size)
            .and_then(|v| v.udata_value());
        let addr_size = u64::from(self.address_size(r.unit));
        let kind = match e.tag() {
            gimli::DW_TAG_base_type => {
                let enc = match e.attr_value(gimli::DW_AT_encoding) {
                    Some(AttributeValue::Encoding(enc)) => match enc {
                        gimli::DW_ATE_signed => BaseEnc::Signed,
                        gimli::DW_ATE_unsigned => BaseEnc::Unsigned,
                        gimli::DW_ATE_signed_char => BaseEnc::SignedChar,
                        gimli::DW_ATE_unsigned_char => BaseEnc::UnsignedChar,
                        gimli::DW_ATE_UTF => BaseEnc::Utf,
                        gimli::DW_ATE_float => BaseEnc::Float,
                        gimli::DW_ATE_boolean => BaseEnc::Bool,
                        _ => BaseEnc::Other,
                    },
                    _ => BaseEnc::Other,
                };
                TypeKind::Base(enc)
            }
            gimli::DW_TAG_pointer_type
            | gimli::DW_TAG_reference_type
            | gimli::DW_TAG_rvalue_reference_type => {
                let kind = match e.tag() {
                    gimli::DW_TAG_pointer_type => PtrKind::Pointer,
                    gimli::DW_TAG_reference_type => PtrKind::Reference,
                    _ => PtrKind::RvalueReference,
                };
                let pointee = self.type_attr(r, e)?;
                return Ok(Type {
                    name,
                    size: Some(size.unwrap_or(addr_size)),
                    kind: TypeKind::Pointer { pointee, kind },
                    lang,
                });
            }
            gimli::DW_TAG_structure_type | gimli::DW_TAG_class_type | gimli::DW_TAG_union_type => {
                let kind = match e.tag() {
                    gimli::DW_TAG_structure_type => StructKind::Struct,
                    gimli::DW_TAG_class_type => StructKind::Class,
                    _ => StructKind::Union,
                };
                let mut st = StructType {
                    kind,
                    members: vec![],
                    variant_part: None,
                    template_params: vec![],
                    declaration: e.attr_value(gimli::DW_AT_declaration).is_some(),
                };
                for (cr, ce) in self.children(r)? {
                    match ce.tag() {
                        gimli::DW_TAG_member => {
                            if ce.attr_value(gimli::DW_AT_external).is_some()
                                || ce.attr_value(gimli::DW_AT_declaration).is_some()
                            {
                                // Static member; not part of layout.
                                continue;
                            }
                            if let Some(m) = self.make_member(cr, &ce)? {
                                st.members.push(m);
                            }
                        }
                        gimli::DW_TAG_inheritance => {
                            if let Some(mut m) = self.make_member(cr, &ce)? {
                                m.name = Some(format!("<{}>", self.type_name(m.ty)));
                                st.members.push(m);
                            }
                        }
                        gimli::DW_TAG_variant_part => {
                            st.variant_part = Some(self.make_variant_part(cr, &ce)?);
                        }
                        gimli::DW_TAG_template_type_parameter => {
                            let pname = self
                                .die_string_follow(cr, gimli::DW_AT_name)
                                .unwrap_or_default();
                            if let Some(t) = self.type_attr(cr, &ce)? {
                                st.template_params.push((pname, t));
                            }
                        }
                        _ => {}
                    }
                }
                TypeKind::Struct(st)
            }
            gimli::DW_TAG_enumeration_type => {
                let underlying = self.type_attr(r, e)?;
                let signed = underlying.is_some_and(|u| {
                    matches!(
                        self.ty(self.strip(u)).kind,
                        TypeKind::Base(BaseEnc::Signed | BaseEnc::SignedChar)
                    )
                });
                let mut enumerators = vec![];
                for (cr, ce) in self.children(r)? {
                    if ce.tag() != gimli::DW_TAG_enumerator {
                        continue;
                    }
                    let n = self
                        .die_string_follow(cr, gimli::DW_AT_name)
                        .unwrap_or_default();
                    let v = ce
                        .attr_value(gimli::DW_AT_const_value)
                        .map(|v| const_to_i128(&v, signed, size))
                        .unwrap_or(0);
                    enumerators.push((n, v));
                }
                TypeKind::Enum {
                    underlying,
                    enumerators,
                }
            }
            gimli::DW_TAG_array_type => {
                let Some(elem) = self.type_attr(r, e)? else {
                    bail!("array type without element type");
                };
                let mut dims = vec![];
                for (_, ce) in self.children(r)? {
                    if ce.tag() != gimli::DW_TAG_subrange_type {
                        continue;
                    }
                    let count = if let Some(c) = ce
                        .attr_value(gimli::DW_AT_count)
                        .and_then(|v| v.udata_value())
                    {
                        Some(c)
                    } else {
                        ce.attr_value(gimli::DW_AT_upper_bound)
                            .and_then(|v| match v {
                                AttributeValue::Sdata(s) if s < 0 => None,
                                v => v.udata_value(),
                            })
                            .map(|u| {
                                let lower = ce
                                    .attr_value(gimli::DW_AT_lower_bound)
                                    .and_then(|v| v.udata_value())
                                    .unwrap_or(0);
                                u + 1 - lower
                            })
                    };
                    dims.push(count);
                }
                if dims.is_empty() {
                    dims.push(None);
                }
                // Build nested arrays from the innermost dimension out.
                let mut cur = elem;
                for count in dims[1..].iter().rev() {
                    let esize = self.type_size(cur);
                    cur = self.types.borrow_mut().alloc(Type {
                        name: None,
                        size: esize.zip(*count).map(|(s, c)| s * c),
                        kind: TypeKind::Array {
                            elem: cur,
                            len: *count,
                        },
                        lang,
                    });
                }
                let esize = self.type_size(cur);
                return Ok(Type {
                    name,
                    size: size.or(esize.zip(dims[0]).map(|(s, c)| s * c)),
                    kind: TypeKind::Array {
                        elem: cur,
                        len: dims[0],
                    },
                    lang,
                });
            }
            gimli::DW_TAG_typedef => TypeKind::Typedef(self.type_attr(r, e)?),
            gimli::DW_TAG_const_type
            | gimli::DW_TAG_volatile_type
            | gimli::DW_TAG_restrict_type
            | gimli::DW_TAG_atomic_type => {
                let qual = match e.tag() {
                    gimli::DW_TAG_const_type => Qual::Const,
                    gimli::DW_TAG_volatile_type => Qual::Volatile,
                    gimli::DW_TAG_restrict_type => Qual::Restrict,
                    _ => Qual::Atomic,
                };
                TypeKind::Qualified {
                    qual,
                    target: self.type_attr(r, e)?,
                }
            }
            gimli::DW_TAG_subroutine_type => {
                let ret = self.type_attr(r, e)?;
                let mut params = vec![];
                for (cr, ce) in self.children(r)? {
                    if ce.tag() == gimli::DW_TAG_formal_parameter {
                        if let Some(t) = self.type_attr(cr, &ce)? {
                            params.push(t);
                        }
                    }
                }
                TypeKind::Function { ret, params }
            }
            gimli::DW_TAG_unspecified_type => TypeKind::Void,
            _ => TypeKind::Unknown,
        };
        Ok(Type {
            name,
            size,
            kind,
            lang,
        })
    }

    fn make_member(&self, r: DieRef, e: &DebuggingInformationEntry<R>) -> Result<Option<Member>> {
        let Some(ty) = self.type_attr(r, e)? else {
            return Ok(None);
        };
        let name = self.die_string_follow(r, gimli::DW_AT_name);
        let bit_size = e
            .attr_value(gimli::DW_AT_bit_size)
            .and_then(|v| v.udata_value());
        let mut bit_offset = match e.attr_value(gimli::DW_AT_data_member_location) {
            Some(AttributeValue::Exprloc(expr)) => {
                // Typically `DW_OP_plus_uconst N`.
                let mut reader = expr.0.clone();
                let enc = self.units[r.unit].unit.encoding();
                match gimli::Operation::parse(&mut reader, enc) {
                    Ok(gimli::Operation::PlusConstant { value }) => value * 8,
                    Ok(gimli::Operation::UnsignedConstant { value }) => value * 8,
                    _ => 0,
                }
            }
            Some(v) => v.udata_value().unwrap_or(0) * 8,
            None => 0,
        };
        if let Some(dbo) = e
            .attr_value(gimli::DW_AT_data_bit_offset)
            .and_then(|v| v.udata_value())
        {
            bit_offset = dbo;
        } else if let (Some(bo), Some(bs)) = (
            e.attr_value(gimli::DW_AT_bit_offset)
                .and_then(|v| v.udata_value()),
            bit_size,
        ) {
            // DWARF 2/3 big-endian-style bit offset within a storage unit
            // of `byte_size` bytes.
            let storage = e
                .attr_value(gimli::DW_AT_byte_size)
                .and_then(|v| v.udata_value())
                .or_else(|| self.type_size(ty))
                .unwrap_or(4);
            bit_offset += storage * 8 - bo - bs;
        }
        Ok(Some(Member {
            name,
            ty,
            bit_offset,
            bit_size,
            artificial: e.attr_value(gimli::DW_AT_artificial).is_some(),
        }))
    }

    fn make_variant_part(
        &self,
        r: DieRef,
        e: &DebuggingInformationEntry<R>,
    ) -> Result<VariantPart> {
        let discr_ref = e
            .attr_value(gimli::DW_AT_discr)
            .and_then(|v| self.resolve_ref(r.unit, &v));
        let mut part = VariantPart {
            discr: None,
            variants: vec![],
        };
        for (cr, ce) in self.children(r)? {
            match ce.tag() {
                gimli::DW_TAG_member => {
                    if Some(cr) == discr_ref || discr_ref.is_none() {
                        part.discr = self.make_member(cr, &ce)?;
                    }
                }
                gimli::DW_TAG_variant => {
                    let discr_value = ce
                        .attr_value(gimli::DW_AT_discr_value)
                        .map(|v| const_to_i128(&v, false, None) as u64);
                    let mut members = vec![];
                    for (mr, me) in self.children(cr)? {
                        if me.tag() == gimli::DW_TAG_member {
                            if let Some(m) = self.make_member(mr, &me)? {
                                members.push(m);
                            }
                        }
                    }
                    part.variants.push(Variant {
                        discr_value,
                        members,
                    });
                }
                _ => {}
            }
        }
        Ok(part)
    }

    /// Strip typedefs and qualifiers.
    pub fn strip(&self, mut id: TypeId) -> TypeId {
        for _ in 0..64 {
            let t = self.ty(id);
            match t.kind {
                TypeKind::Typedef(Some(next))
                | TypeKind::Qualified {
                    target: Some(next), ..
                } => id = next,
                _ => return id,
            }
        }
        id
    }

    /// Size of a type in bytes (following typedefs/qualifiers).
    pub fn type_size(&self, id: TypeId) -> Option<u64> {
        let t = self.ty(id);
        if let Some(s) = t.size {
            return Some(s);
        }
        match t.kind {
            TypeKind::Typedef(Some(next))
            | TypeKind::Qualified {
                target: Some(next), ..
            } => self.type_size(next),
            TypeKind::Array {
                elem,
                len: Some(len),
            } => Some(self.type_size(elem)? * len),
            TypeKind::Void => Some(0),
            _ => None,
        }
    }

    /// A display name for a type.
    pub fn type_name(&self, id: TypeId) -> String {
        self.type_name_depth(id, 0)
    }

    fn type_name_depth(&self, id: TypeId, depth: usize) -> String {
        if depth > 16 {
            return "...".into();
        }
        let t = self.ty(id);
        let c = t.lang.is_c_like();
        let sub = |o: Option<TypeId>| match o {
            Some(x) => self.type_name_depth(x, depth + 1),
            None => "void".to_string(),
        };
        match &t.kind {
            TypeKind::Struct(st) if c => {
                let kw = match st.kind {
                    StructKind::Struct => "struct",
                    StructKind::Class => "class",
                    StructKind::Union => "union",
                };
                match &t.name {
                    Some(n) if t.lang == Lang::C => format!("{kw} {n}"),
                    Some(n) => simplify_cpp_type_name(n),
                    None => format!("{kw} <anonymous>"),
                }
            }
            TypeKind::Enum { .. } if c => match &t.name {
                Some(n) if t.lang == Lang::C => format!("enum {n}"),
                Some(n) => n.clone(),
                None => "enum <anonymous>".into(),
            },
            _ if t.name.is_some() => {
                let n = t.name.clone().unwrap();
                match t.lang {
                    Lang::Rust => simplify_rust_type_name(&n),
                    Lang::Cpp => simplify_cpp_type_name(&n),
                    _ => n,
                }
            }
            TypeKind::Void => "void".into(),
            TypeKind::Pointer { pointee, kind } => {
                if c {
                    let s = match kind {
                        PtrKind::Pointer => "*",
                        PtrKind::Reference => "&",
                        PtrKind::RvalueReference => "&&",
                    };
                    match pointee.map(|p| self.ty(p)) {
                        Some(pt) if matches!(pt.kind, TypeKind::Function { .. }) => {
                            self.function_type_name(pointee.unwrap(), "(*)", depth)
                        }
                        _ => format!("{} {s}", sub(*pointee)),
                    }
                } else {
                    match kind {
                        PtrKind::Pointer => format!("*const {}", sub(*pointee)),
                        _ => format!("&{}", sub(*pointee)),
                    }
                }
            }
            TypeKind::Array { elem, len } => {
                let l = len.map(|l| l.to_string()).unwrap_or_default();
                if c {
                    format!("{} [{l}]", sub(Some(*elem)))
                } else {
                    format!("[{}; {l}]", sub(Some(*elem)))
                }
            }
            TypeKind::Qualified { qual, target } => {
                format!("{} {}", qual.keyword(), sub(*target))
            }
            TypeKind::Function { .. } => self.function_type_name(id, "", depth),
            TypeKind::Typedef(t) => sub(*t),
            TypeKind::Base(_) | TypeKind::Struct(_) | TypeKind::Enum { .. } => "<anonymous>".into(),
            TypeKind::Unknown => "<unknown type>".into(),
        }
    }

    fn function_type_name(&self, id: TypeId, mid: &str, depth: usize) -> String {
        let t = self.ty(id);
        let TypeKind::Function { ret, params } = &t.kind else {
            return "?".into();
        };
        let ret = match ret {
            Some(r) => self.type_name_depth(*r, depth + 1),
            None => "void".into(),
        };
        let params: Vec<String> = params
            .iter()
            .map(|p| self.type_name_depth(*p, depth + 1))
            .collect();
        format!("{ret} {mid}({})", params.join(", "))
    }

    /// Get or synthesize a base type with the given name, encoding and
    /// size.
    pub fn synth_base(&self, name: &str, enc: BaseEnc, size: u64, lang: Lang) -> TypeId {
        let key = format!("base:{name}:{size}");
        if let Some(id) = self.synth_cache.borrow().get(&key) {
            return *id;
        }
        let id = self.types.borrow_mut().alloc(Type {
            name: Some(name.to_string()),
            size: Some(size),
            kind: TypeKind::Base(enc),
            lang,
        });
        self.synth_cache.borrow_mut().insert(key, id);
        id
    }

    /// Get or synthesize a pointer type to `pointee`.
    pub fn pointer_to(&self, pointee: TypeId, size: u64) -> TypeId {
        let key = format!("ptr:{}", pointee.0);
        if let Some(id) = self.synth_cache.borrow().get(&key) {
            return *id;
        }
        let lang = self.ty(pointee).lang;
        let id = self.types.borrow_mut().alloc(Type {
            name: None,
            size: Some(size),
            kind: TypeKind::Pointer {
                pointee: Some(pointee),
                kind: PtrKind::Pointer,
            },
            lang,
        });
        self.synth_cache.borrow_mut().insert(key, id);
        id
    }

    /// Get or synthesize an array type of `len` elements.
    pub fn array_of(&self, elem: TypeId, len: u64) -> TypeId {
        let key = format!("arr:{}:{len}", elem.0);
        if let Some(id) = self.synth_cache.borrow().get(&key) {
            return *id;
        }
        let lang = self.ty(elem).lang;
        let size = self.type_size(elem).map(|s| s * len);
        let id = self.types.borrow_mut().alloc(Type {
            name: None,
            size,
            kind: TypeKind::Array {
                elem,
                len: Some(len),
            },
            lang,
        });
        self.synth_cache.borrow_mut().insert(key, id);
        id
    }

    /// Get or synthesize a named array type, e.g. a `str` of `len`
    /// bytes for the contents of an `Arc<str>`.
    pub fn named_array_of(&self, elem: TypeId, len: u64, name: &str) -> TypeId {
        let key = format!("narr:{name}:{}:{len}", elem.0);
        if let Some(id) = self.synth_cache.borrow().get(&key) {
            return *id;
        }
        let lang = self.ty(elem).lang;
        let size = self.type_size(elem).map(|s| s * len);
        let id = self.types.borrow_mut().alloc(Type {
            name: Some(name.to_string()),
            size,
            kind: TypeKind::Array {
                elem,
                len: Some(len),
            },
            lang,
        });
        self.synth_cache.borrow_mut().insert(key, id);
        id
    }

    /// For a Rust trait object, find the concrete type behind a vtable
    /// at linear-memory address `addr`, using the `<T as Trait>::{vtable}`
    /// variables (whose types name `T` via `DW_AT_containing_type`) that
    /// rustc emits.
    pub fn vtable_concrete_type(&self, addr: u64) -> Result<Option<TypeId>> {
        if self.vtables.borrow().is_none() {
            let mut map = HashMap::new();
            for g in self
                .globals
                .iter()
                .filter(|g| g.name.ends_with("::{vtable}"))
            {
                let Ok(loc) = self.global_loc(g) else {
                    continue;
                };
                if let Ok(eval::Location::Memory(a)) =
                    self.eval_loc(g.die.unit, &loc, 0, None, &crate::target::NoTarget)
                {
                    map.insert(a, g.die);
                }
            }
            *self.vtables.borrow_mut() = Some(map);
        }
        let Some(die) = self.vtables.borrow().as_ref().unwrap().get(&addr).copied() else {
            return Ok(None);
        };
        let Some((r, v)) = self.attr_follow(die, gimli::DW_AT_type) else {
            return Ok(None);
        };
        let Some(vt) = self.resolve_ref(r.unit, &v) else {
            return Ok(None);
        };
        let e = self.entry(vt)?;
        match e
            .attr_value(gimli::DW_AT_containing_type)
            .and_then(|v| self.resolve_ref(vt.unit, &v))
        {
            Some(t) => Ok(Some(self.type_of(t)?)),
            None => Ok(None),
        }
    }

    /// Default address size across the module (from the first unit).
    pub fn default_address_size(&self) -> u64 {
        self.units
            .first()
            .map(|u| u64::from(u.unit.encoding().address_size))
            .unwrap_or(4)
    }

    /// Find a type definition whose qualified name starts with `prefix`
    /// and which satisfies `pred`. Used by pretty-printers to locate
    /// helper types (e.g. a container's node type) from DWARF.
    pub fn find_type_where(&self, prefix: &str, pred: &dyn Fn(TypeId) -> bool) -> Option<TypeId> {
        let mut names: Vec<&String> = self
            .type_defs
            .keys()
            .filter(|k| k.starts_with(prefix))
            .collect();
        names.sort();
        for n in names {
            if let Ok(id) = self.type_of(self.type_defs[n]) {
                if pred(id) {
                    return Some(id);
                }
            }
        }
        None
    }

    /// Find a type definition by qualified name.
    pub fn find_type(&self, name: &str) -> Result<Option<TypeId>> {
        match self.type_defs.get(name) {
            Some(r) => Ok(Some(self.type_of(*r)?)),
            None => Ok(None),
        }
    }
}

fn const_to_i128(v: &AttributeValue<R>, signed: bool, size: Option<u64>) -> i128 {
    let (raw, bits): (u64, u32) = match *v {
        AttributeValue::Sdata(x) => return i128::from(x),
        AttributeValue::Udata(x) => return i128::from(x),
        AttributeValue::Data1(x) => (u64::from(x), 8),
        AttributeValue::Data2(x) => (u64::from(x), 16),
        AttributeValue::Data4(x) => (u64::from(x), 32),
        AttributeValue::Data8(x) => (x, 64),
        _ => return 0,
    };
    let bits = size.map(|s| (s * 8) as u32).unwrap_or(bits).min(bits);
    if signed && bits < 64 {
        let shift = 64 - bits;
        i128::from(((raw << shift) as i64) >> shift)
    } else if signed {
        i128::from(raw as i64)
    } else {
        i128::from(raw)
    }
}

/// Reduce a (demangled) function name to its qualified base name for
/// matching: drop C++ parameter lists and qualifiers, template/generic
/// arguments, and C++ template return types. For example,
/// `int geo::twice<int>(int)` becomes `geo::twice` and
/// `geo::Vec2::dot(geo::Vec2 const&) const` becomes `geo::Vec2::dot`.
fn function_base_name(name: &str) -> String {
    let mut s = name.trim();
    // Drop a trailing parameter list (and any qualifiers after it).
    if let Some(close) = s.rfind(')') {
        let mut depth = 0;
        let mut open = None;
        for (i, c) in s[..=close].char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        open = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(open) = open {
            // Only treat it as a parameter list if it follows a name
            // (not e.g. a Rust tuple type).
            if open > 0 && !s[..open].ends_with(['<', ' ']) {
                s = &s[..open];
            }
        }
    }
    // Strip generics, including Rust turbofish (`f::<T>`).
    let base = strip_generics(s);
    let base = base.strip_suffix("::").unwrap_or(&base).to_string();
    // Drop a leading return type (C++ templates).
    match base.rsplit_once(' ') {
        Some((_, last)) if !last.is_empty() => last.to_string(),
        _ => base,
    }
}

/// Strip generic arguments (`<...>`) from a name.
fn strip_generics(name: &str) -> String {
    // Note: `operator<`-style names are rare enough to ignore here.
    let mut out = String::new();
    let mut depth = 0;
    let mut prev = ' ';
    for c in name.chars() {
        let after_dash = prev == '-';
        prev = c;
        match c {
            '<' => depth += 1,
            // `->` (in `fn() -> T`) is not a closing bracket.
            '>' if after_dash => {
                if depth == 0 {
                    out.push(c);
                }
            }
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Normalize `.` and `..` components in a slash-separated path.
fn normalize_path(p: &str) -> String {
    let abs = p.starts_with('/');
    let mut parts: Vec<&str> = vec![];
    for comp in p.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|l| *l != "..") {
                    parts.pop();
                } else if !abs {
                    parts.push("..");
                }
            }
            c => parts.push(c),
        }
    }
    // Preserve URL-like schemes (e.g. `wasisdk://`) as-is.
    if p.contains("://") {
        return p.to_string();
    }
    let joined = parts.join("/");
    if abs { format!("/{joined}") } else { joined }
}

impl std::fmt::Debug for DebugInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebugInfo")
            .field("units", &self.units.len())
            .field("functions", &self.functions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_names() {
        assert_eq!(function_base_name("int geo::twice<int>(int)"), "geo::twice");
        assert_eq!(
            function_base_name("geo::Vec2::dot(geo::Vec2 const&) const"),
            "geo::Vec2::dot"
        );
        assert_eq!(function_base_name("main"), "main");
        assert_eq!(
            function_base_name("krate::consume::<alloc::vec::Vec<u32>>"),
            "krate::consume"
        );
        assert_eq!(
            function_base_name("alloc::vec::Vec<T>::push"),
            "alloc::vec::Vec::push"
        );
        assert_eq!(
            function_base_name("<core::option::Option<T>>::as_ref"),
            "::as_ref"
        );
    }
}
