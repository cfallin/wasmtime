//! Command parsing and execution.

use crate::session::{
    FrameTarget, HistEntry, Outcome, RunState, Session, StopReason, WatchReport, to_wasm_val,
};
use crate::wasmval::{self, WVal};
use anyhow::{Result, anyhow, bail};
use std::fmt::Write;
use wasmtime_debug_info::cli::{
    BreakpointSpec, ExamineFormat, HELP, basename, demangle_wat, format_examine_string,
    format_watched_bytes, hexdump, parse_u64, truncate,
};
use wasmtime_debug_info::format::{FormatOptions, describe_type, format_value};
use wasmtime_debug_info::location::{Location, WasmSlot};
use wasmtime_debug_info::path::{self, HistoryRef, Path};
use wasmtime_debug_info::types::{BaseEnc, Lang, TypeKind};
use wasmtime_debug_info::value::{Ctx, Value};
use wasmtime_debug_info::{Target, WasmVal};

/// Whether `word` names a command of the text interface.
pub fn is_command(word: &str) -> bool {
    matches!(
        word,
        "help"
            | "h"
            | "quit"
            | "q"
            | "detach"
            | "start"
            | "continue"
            | "c"
            | "run"
            | "r"
            | "step"
            | "s"
            | "next"
            | "n"
            | "stepi"
            | "si"
            | "nexti"
            | "ni"
            | "finish"
            | "fin"
            | "interrupt"
            | "reverse-continue"
            | "rc"
            | "reverse-step"
            | "rs"
            | "reverse-next"
            | "rn"
            | "reverse-finish"
            | "reverse-stepi"
            | "rsi"
            | "reverse-nexti"
            | "rni"
            | "watch"
            | "break"
            | "b"
            | "br"
            | "tbreak"
            | "delete"
            | "d"
            | "backtrace"
            | "bt"
            | "where"
            | "frame"
            | "f"
            | "up"
            | "down"
            | "list"
            | "l"
            | "print"
            | "p"
            | "output"
            | "ptype"
            | "location"
            | "loc"
            | "whatis"
            | "x"
            | "info"
            | "i"
            | "disassemble"
            | "disas"
            | "wasm-locals"
            | "wasm-stack"
            | "wasm-globals"
            | "wasm-memory"
            | "wasm-table"
            | "wasm-instances"
            | "wasm-modules"
            | "wasm-functions"
    )
}

/// Memory to watch, with the type of the value it holds.
#[derive(Clone, Copy)]
pub struct WatchTarget {
    pub inst: usize,
    pub addr: u64,
    pub len: u64,
    pub ty: Option<(usize, wasmtime_debug_info::types::TypeId)>,
}

/// Result of executing a command.
pub enum Flow {
    Continue,
    Quit,
}

fn compact_opts() -> FormatOptions {
    FormatOptions {
        max_depth: 2,
        max_elems: 4,
        max_string: 32,
        ..Default::default()
    }
}

/// Options parsed from `-i N` / `-m N` / `-t N` flags.
struct Flags {
    inst: Option<usize>,
    index: Option<u32>,
    rest: Vec<String>,
}

fn parse_flags(args: &str, index_flag: &str) -> Result<Flags> {
    let mut f = Flags {
        inst: None,
        index: None,
        rest: vec![],
    };
    let mut it = args.split_whitespace();
    while let Some(a) = it.next() {
        if a == "-i" {
            let v = it
                .next()
                .ok_or_else(|| anyhow!("-i requires an instance number"))?;
            f.inst = Some(usize::try_from(parse_u64(v)?)?);
        } else if a == index_flag {
            let v = it
                .next()
                .ok_or_else(|| anyhow!("{index_flag} requires an index"))?;
            f.index = Some(u32::try_from(parse_u64(v)?)?);
        } else {
            f.rest.push(a.to_string());
        }
    }
    Ok(f)
}

impl Session<'_> {
    /// Print the banner shown on connection.
    pub fn banner(&mut self) {
        self.refresh_entities();
        let mut out = String::from("wasmtime-debug: interactive WebAssembly debugger.\n");
        let _ = writeln!(out, "Modules in store:");
        for (i, m) in self.modules.iter().enumerate() {
            let dbg = match (&m.info.debug, &m.info.debug_error) {
                (Some(d), _) => format!("debug info: {} functions", d.functions.len()),
                (None, Some(e)) => format!("debug info error: {e}"),
                (None, None) => "no debug info".into(),
            };
            let _ = writeln!(
                out,
                "  [{i}] {} ({} functions; {dbg})",
                m.label,
                m.info.num_funcs()
            );
        }
        out.push_str(
            "The program has not started. Type `start` to run to `main`, `continue` to run,\n\
             or `help` for a list of commands.\n",
        );
        self.print(&out);
    }

    pub fn prompt(&mut self) {
        self.print("(wasmtime-debug) ");
    }

    /// Execute one command line.
    pub async fn execute(&mut self, line: &str) -> Flow {
        let line = line.trim();
        if line.is_empty() {
            return Flow::Continue;
        }
        let (cmd, args) = match line.find(|c: char| c.is_whitespace()) {
            Some(i) => (&line[..i], line[i..].trim()),
            None => (line, ""),
        };
        let (cmd, fmt) = match cmd.split_once('/') {
            Some((c, f)) => (c, f),
            None => (cmd, ""),
        };
        let r = match cmd {
            "help" | "h" => {
                self.print(HELP);
                Ok(())
            }
            "quit" | "q" | "detach" => return Flow::Quit,
            "start" => self.cmd_start().await,
            "continue" | "c" | "run" | "r" => self.cmd_resume(Resume::Continue).await,
            "step" | "s" => self.cmd_resume(Resume::Step).await,
            "next" | "n" => self.cmd_resume(Resume::Next).await,
            "stepi" | "si" => self.cmd_resume(Resume::StepI).await,
            "nexti" | "ni" => self.cmd_resume(Resume::NextI).await,
            "finish" | "fin" => self.cmd_resume(Resume::Finish).await,
            "reverse-continue" | "rc" => self.cmd_resume(Resume::ReverseContinue).await,
            "reverse-step" | "rs" => self.cmd_resume(Resume::ReverseStep).await,
            "reverse-next" | "rn" => self.cmd_resume(Resume::ReverseNext).await,
            "reverse-finish" => self.cmd_resume(Resume::ReverseFinish).await,
            "reverse-stepi" | "rsi" => self.cmd_resume(Resume::ReverseStepI).await,
            "reverse-nexti" | "rni" => self.cmd_resume(Resume::ReverseNextI).await,
            "watch" => self.cmd_watch(args),
            "interrupt" => {
                self.println("The program is not running.");
                Ok(())
            }
            "break" | "b" | "br" => self.cmd_break(args, false),
            "tbreak" => self.cmd_break(args, true),
            "delete" | "d" => self.cmd_delete(args),
            "backtrace" | "bt" | "where" => self.cmd_backtrace(),
            "frame" | "f" => self.cmd_frame(args),
            "up" => self.cmd_up_down(args, true),
            "down" => self.cmd_up_down(args, false),
            "list" | "l" => self.cmd_list(args),
            "print" | "p" | "output" => self.cmd_print(args, fmt),
            "ptype" => self.cmd_ptype(args, true),
            "location" | "loc" => self.cmd_location(args),
            "whatis" => self.cmd_ptype(args, false),
            "x" => self.cmd_examine(args, fmt),
            "info" | "i" => self.cmd_info(args),
            "disassemble" | "disas" => self.cmd_disas(args),
            "wasm-locals" => self.cmd_wasm_locals(),
            "wasm-stack" => self.cmd_wasm_stack(),
            "wasm-globals" => self.cmd_wasm_globals(args),
            "wasm-memory" => self.cmd_wasm_memory(args),
            "wasm-table" => self.cmd_wasm_table(args),
            "wasm-instances" => self.cmd_wasm_instances(),
            "wasm-modules" => self.cmd_wasm_modules(),
            "wasm-functions" => self.cmd_wasm_functions(args),
            _ => Err(anyhow!("Undefined command: \"{cmd}\". Try \"help\".")),
        };
        if let Err(e) = r {
            self.println(&format!("{e}"));
        }
        if self.detached {
            return Flow::Quit;
        }
        Flow::Continue
    }

    // -----------------------------------------------------------------
    // Helpers.

    pub(crate) fn require_stopped(&self) -> Result<()> {
        match self.state {
            RunState::Stopped if !self.vframes.is_empty() => Ok(()),
            RunState::NotStarted => bail!("The program has not been started."),
            RunState::Exited => bail!("The program is not being run."),
            _ => bail!("No stack."),
        }
    }

    /// The instance to use for instance-scoped commands by default.
    pub(crate) fn default_inst(&self) -> Result<usize> {
        if let Some(v) = self.vframes.get(self.selected) {
            return Ok(self.phys[v.phys].inst);
        }
        // Prefer an instance of a module with debug info.
        if let Some(i) = self
            .instances
            .iter()
            .position(|i| self.modules[i.module].info.debug.is_some())
        {
            return Ok(i);
        }
        if self.instances.is_empty() {
            bail!("No instances exist yet.");
        }
        Ok(0)
    }

    fn inst_arg(&self, inst: Option<usize>) -> Result<usize> {
        match inst {
            Some(i) if i < self.instances.len() => Ok(i),
            Some(i) => bail!("No instance {i}."),
            None => self.default_inst(),
        }
    }

    /// The module to resolve source-level names in by default.
    pub(crate) fn default_module(&self) -> Option<usize> {
        if let Some(v) = self.vframes.get(self.selected) {
            let m = self.phys[v.phys].module;
            if self.modules[m].info.debug.is_some() {
                return Some(m);
            }
        }
        self.modules.iter().position(|m| m.info.debug.is_some())
    }

    fn file_display(&self, module: usize, file: u32) -> String {
        match &self.modules[module].info.debug {
            Some(d) => basename(d.file_name(file)).to_string(),
            None => "??".into(),
        }
    }

    pub(crate) fn source_lines(&mut self, path: &str) -> Option<&Vec<String>> {
        if !self.source_cache.contains_key(path) {
            let lines = std::fs::read_to_string(path)
                .ok()
                .map(|s| s.lines().map(|l| l.to_string()).collect());
            self.source_cache.insert(path.to_string(), lines);
        }
        self.source_cache.get(path).unwrap().as_ref()
    }

    /// Print a source line (`N\ttext`).
    fn print_source_line(&mut self, module: usize, file: u32, line: u32) {
        let Some(path) = self.modules[module]
            .info
            .debug
            .as_ref()
            .map(|d| d.file_name(file).to_string())
        else {
            return;
        };
        let text = self
            .source_lines(&path)
            .and_then(|ls| ls.get(usize::try_from(line).ok()?.checked_sub(1)?))
            .cloned();
        match text {
            Some(t) => self.println(&format!("{line}\t{t}")),
            None => self.println(&format!("{line}\tin {path} (source unavailable)")),
        }
    }

    /// A one-line description of logical frame `vi`: `name (args) at
    /// file:line`.
    pub(crate) fn frame_desc(&self, vi: usize) -> String {
        let v = &self.vframes[vi];
        let p = &self.phys[v.phys];
        let args = self.frame_args(vi);
        let mut s = format!("{} ({args})", v.name);
        match &v.loc {
            Some(loc) => {
                let _ = write!(
                    s,
                    " at {}:{}",
                    self.file_display(p.module, loc.file),
                    loc.line
                );
            }
            None => {
                let _ = write!(s, " in {} at pc {:#x}", self.modules[p.module].label, p.pc);
            }
        }
        s
    }

    pub(crate) fn frame_args(&self, vi: usize) -> String {
        let v = &self.vframes[vi];
        let Some(scope) = &v.scope else {
            return self.wasm_frame_args(vi);
        };
        let p = &self.phys[v.phys];
        let m = &self.modules[p.module];
        let (Some(dbg), Some(addr)) = (&m.info.debug, m.info.pc_to_dwarf(p.lookup_pc)) else {
            return String::new();
        };
        let inst = &self.instances[p.inst].handle;
        let target = FrameTarget::new(self.d, Some(&p.frame), inst);
        let fb = self.frame_base(v.phys, &target);
        let ctx = Ctx::new(dbg, &target);
        let mut out = vec![];
        for var in scope.vars.iter().filter(|v| v.is_param) {
            let val = match ctx.variable(var, addr, fb) {
                Ok(val) => truncate(format_value(&ctx, &val, &compact_opts()), 40),
                Err(e) => format!("{e}"),
            };
            out.push(format!("{}={val}", var.name));
        }
        out.join(", ")
    }

    /// Without debug info, show the Wasm-level parameters of a frame.
    pub(crate) fn wasm_frame_args(&self, vi: usize) -> String {
        let p = &self.phys[self.vframes[vi].phys];
        let Some(func) = p.func else {
            return String::new();
        };
        let info = &self.modules[p.module].info;
        let nparams = info.func_sig(func).map(|s| s.params.len()).unwrap_or(0);
        if nparams == 0 {
            return String::new();
        }
        let Ok(locals) = p.frame.get_locals(self.d) else {
            return String::new();
        };
        let names = info.func_locals(func);
        locals
            .iter()
            .take(nparams)
            .enumerate()
            .map(|(i, l)| {
                let name = names
                    .get(i)
                    .and_then(|(n, _)| n.clone())
                    .unwrap_or_else(|| format!("%local{i}"));
                let val = truncate(wasmval::format(self.d, &WVal::from_api(l), false, 0), 40);
                format!("{name}={val}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Print the selected frame's header and current line.
    fn print_frame_header(&mut self, vi: usize, with_number: bool) {
        let desc = self.frame_desc(vi);
        if with_number {
            self.println(&format!("#{vi}  {desc}"));
        } else {
            self.println(&desc);
        }
        self.print_current_line(vi);
    }

    fn print_current_line(&mut self, vi: usize) {
        let v = &self.vframes[vi];
        let p = &self.phys[v.phys];
        let module = p.module;
        let pc = p.pc;
        if let Some(loc) = v.loc.clone() {
            // A following `list` centers on this line.
            self.list_pos = Some((module, loc.file, loc.line.saturating_sub(5).max(1)));
            self.print_source_line(module, loc.file, loc.line);
        } else {
            self.print_disas_line(module, pc);
        }
    }

    fn print_disas_line(&mut self, module: usize, pc: u32) {
        let text = self.modules[module]
            .info
            .func_at_pc(pc)
            .and_then(|f| self.modules[module].info.disassemble_func(f.index).ok())
            .and_then(|lines| {
                lines
                    .iter()
                    .find(|l| l.offset == pc)
                    .map(|l| demangle_wat(&l.text))
            });
        match text {
            Some(t) => self.println(&format!("=> {pc:#x}: {t}")),
            None => self.println(&format!("=> {pc:#x}")),
        }
    }

    // -----------------------------------------------------------------
    // Execution.

    async fn cmd_start(&mut self) -> Result<()> {
        if self.state != RunState::NotStarted {
            bail!("The program is already running.");
        }
        let locs = self.resolve_location("main")?;
        self.add_breakpoint("main".into(), locs, true)?;
        self.cmd_resume(Resume::Continue).await
    }

    async fn cmd_resume(&mut self, how: Resume) -> Result<()> {
        if self.state == RunState::Exited {
            bail!("The program is not being run.");
        }
        let before = self
            .vframes
            .get(self.selected)
            .map(|v| (v.name.clone(), self.vframes.len() - self.selected));
        if let Resume::Finish = how {
            self.require_stopped()?;
            let desc = self.frame_desc(self.selected);
            self.println(&format!("Run till exit from #{}  {desc}", self.selected));
        }
        if let Resume::ReverseFinish = how {
            self.require_stopped()?;
            if !self.reversible {
                bail!("{}", crate::session::NOT_REVERSIBLE);
            }
            let desc = self.frame_desc(self.selected);
            self.println(&format!("Run back to call of #{}  {desc}", self.selected));
        }
        // Remember the callee (for return values) when finishing.
        let finishing = match how {
            Resume::Finish => {
                let v = &self.vframes[self.selected];
                let p = &self.phys[v.phys];
                Some((p.module, p.lookup_pc, v.inlined))
            }
            _ => None,
        };
        let _ = self.flush_output().await;
        let outcome = match how {
            Resume::Continue => {
                if self.state == RunState::NotStarted {
                    self.state = RunState::Stopped;
                }
                self.cont().await
            }
            Resume::Step => self.step_line(false).await,
            Resume::Next => self.step_line(true).await,
            Resume::StepI => {
                if self.state == RunState::NotStarted {
                    self.state = RunState::Stopped;
                }
                self.stepi().await
            }
            Resume::NextI => {
                if self.state == RunState::NotStarted {
                    self.state = RunState::Stopped;
                }
                self.nexti().await
            }
            Resume::Finish => self.finish().await,
            Resume::ReverseContinue => self.reverse_cont().await,
            Resume::ReverseStep => self.reverse_step_line(false).await,
            Resume::ReverseNext => self.reverse_step_line(true).await,
            Resume::ReverseFinish => self.reverse_finish().await,
            Resume::ReverseStepI => self.reverse_stepi().await,
            Resume::ReverseNextI => self.reverse_nexti().await,
        }?;
        self.report(
            outcome,
            before,
            finishing,
            matches!(
                how,
                Resume::StepI | Resume::NextI | Resume::ReverseStepI | Resume::ReverseNextI
            ),
        );
        Ok(())
    }

    fn report(
        &mut self,
        outcome: Outcome,
        before: Option<(String, usize)>,
        finishing: Option<(usize, u32, bool)>,
        instruction_step: bool,
    ) {
        let reason = match outcome {
            Outcome::Exited => {
                self.println("[Program exited]");
                return;
            }
            Outcome::Stopped(r) => r,
        };
        match &reason {
            StopReason::ReplayBegin => {
                self.println(
                    "\nNo more reverse-execution history: at the beginning of the recording.",
                );
            }
            StopReason::ReplayEnd => {
                self.println(
                    "\nNo more execution history: at the end of the recording; \
                     continuing will complete it.",
                );
            }
            _ => {}
        }
        if self.vframes.is_empty() {
            if matches!(reason, StopReason::ReplayBegin | StopReason::ReplayEnd) {
                return;
            }
            self.println("Stopped (no Wasm frames on the stack).");
            return;
        }
        let same_frame = before
            .as_ref()
            .is_some_and(|(n, depth)| *n == self.vframes[0].name && *depth == self.vframes.len());
        match reason {
            StopReason::Breakpoint(Some(id)) => {
                let temp = self
                    .breakpoints
                    .iter()
                    .find(|b| b.id == id)
                    .is_some_and(|b| b.temporary);
                if let Some(b) = self.breakpoints.iter_mut().find(|b| b.id == id) {
                    b.hits += 1;
                }
                let desc = self.frame_desc(0);
                if temp {
                    self.delete_breakpoint(id);
                    self.println(&format!("\nTemporary breakpoint {id}, {desc}"));
                } else {
                    self.println(&format!("\nBreakpoint {id}, {desc}"));
                }
                self.print_current_line(0);
            }
            StopReason::Breakpoint(None) => {
                self.println(&format!("\nStopped at {}", self.frame_desc(0)));
                self.print_current_line(0);
            }
            StopReason::Interrupted => {
                self.println(&format!("\nProgram interrupted.\n{}", self.frame_desc(0)));
                self.print_current_line(0);
            }
            StopReason::Trap => {
                self.println(&format!(
                    "\nProgram trapped (Wasm trap); continuing will terminate it.\n{}",
                    self.frame_desc(0)
                ));
                self.print_current_line(0);
            }
            StopReason::Exception => {
                self.println(&format!("\nWasm exception thrown.\n{}", self.frame_desc(0)));
                self.print_current_line(0);
            }
            StopReason::Step => {
                if !same_frame {
                    self.println(&self.frame_desc(0));
                }
                if instruction_step {
                    let p = &self.phys[0];
                    let (m, pc) = (p.module, p.pc);
                    if let Some(loc) = self.vframes[0].loc.clone() {
                        self.list_pos = Some((m, loc.file, loc.line.saturating_sub(5).max(1)));
                        self.print_source_line(m, loc.file, loc.line);
                    }
                    self.print_disas_line(m, pc);
                } else {
                    self.print_current_line(0);
                }
            }
            StopReason::Watchpoint(report) => {
                self.report_watch(&report);
                self.println(&self.frame_desc(0));
                self.print_current_line(0);
            }
            StopReason::ReplayBegin | StopReason::ReplayEnd => {
                self.println(&self.frame_desc(0));
                self.print_current_line(0);
            }
            StopReason::Finished => {
                self.print_frame_header(0, false);
                if let Some((module, pc, inlined)) = finishing {
                    if !inlined {
                        self.print_return_value(module, pc);
                    }
                }
            }
        }
    }

    /// Describe a watchpoint stop: the watchpoints hit, with their old and
    /// new values.
    pub(crate) fn report_watch(&mut self, r: &WatchReport) {
        let text = self.watch_report_text(r);
        self.print(&text);
    }

    /// The text describing a watchpoint stop.
    pub(crate) fn watch_report_text(&self, r: &WatchReport) -> String {
        let mut out = String::new();
        if r.ids.is_empty() {
            out.push_str("\nWatched memory is being written.\n");
        }
        for (k, id) in r.ids.iter().enumerate() {
            let spec = self
                .watchpoints
                .iter()
                .find(|w| w.id == *id)
                .map(|w| w.spec.clone())
                .unwrap_or_default();
            let old = self.format_watch_value(*id, r.old[k].as_deref());
            let new = self.format_watch_value(*id, r.new[k].as_deref());
            if r.reverse {
                let _ = write!(
                    out,
                    "\nWatchpoint {id}: {spec} (running backward: stopped before the write)\n\n\
                     Value before = {old}\nValue after = {new}\n"
                );
            } else {
                let _ = write!(
                    out,
                    "\nWatchpoint {id}: {spec}\n\nOld value = {old}\nNew value = {new}\n"
                );
            }
        }
        out
    }

    /// Format the contents of watchpoint `id`'s memory, with the watched
    /// value's type if known.
    fn format_watch_value(&self, id: u32, bytes: Option<&[u8]>) -> String {
        let Some(bytes) = bytes else {
            return "<unknown>".into();
        };
        let w = self.watchpoints.iter().find(|w| w.id == id);
        if let Some((module, ty, inst)) = w.and_then(|w| w.ty.map(|(m, t)| (m, t, w.inst))) {
            if let Some(dbg) = self.modules[module].info.debug.as_ref() {
                let target = FrameTarget::new(self.d, None, &self.instances[inst].handle);
                let ctx = Ctx::new(dbg, &target);
                let v = Value::from_bytes(ty, bytes.to_vec());
                return format_value(&ctx, &v, &FormatOptions::default());
            }
        }
        format_watched_bytes(bytes)
    }

    /// After `finish`, show the callee's return value(s), which are on
    /// top of the caller's operand stack.
    pub(crate) fn print_return_value(&mut self, callee_module: usize, callee_pc: u32) {
        let info = &self.modules[callee_module].info;
        let Some(func) = info.func_at_pc(callee_pc).map(|b| b.index) else {
            return;
        };
        let nresults = info.func_sig(func).map(|s| s.results.len()).unwrap_or(0);
        if nresults == 0 {
            return;
        }
        let Ok(stack) = self.phys[0].frame.get_stack(self.d) else {
            return;
        };
        if stack.len() < nresults {
            return;
        }
        let results = &stack[stack.len() - nresults..];
        // If the callee has a scalar DWARF return type, show a typed
        // value too.
        let typed = (|| {
            let dbg = info.debug.as_ref()?;
            let f = dbg.function_at(info.pc_to_dwarf(callee_pc)?)?;
            let ty = dbg.return_type(f).ok()??;
            let size = dbg.type_size(ty)?;
            // Values that are returned directly (rather than through a
            // return-area pointer) occupy a single Wasm result.
            if nresults != 1 || size > 8 {
                return None;
            }
            let raw = to_wasm_val(&results[0]).ok()?;
            let mut bytes = raw.to_bytes();
            bytes.resize(usize::try_from(size).ok()?, 0);
            Some(Value::from_bytes(ty, bytes))
        })();
        match typed {
            Some(v) => {
                let inst = self.phys[0].inst;
                let text = {
                    let dbg = info.debug.as_ref().unwrap();
                    let target = FrameTarget::new(self.d, None, &self.instances[inst].handle);
                    let ctx = Ctx::new(dbg, &target);
                    format_value(&ctx, &v, &FormatOptions::default())
                };
                self.history.push(HistEntry::Src {
                    module: callee_module,
                    inst: Some(inst),
                    value: v,
                });
                let n = self.history.len();
                self.println(&format!("Value returned is %{n} = {text}"));
            }
            None => {
                let vals: Vec<String> = results
                    .iter()
                    .map(|r| wasmval::format(self.d, &WVal::from_api(r), false, 1))
                    .collect();
                self.println(&format!("Wasm results: {}", vals.join(", ")));
            }
        }
    }

    // -----------------------------------------------------------------
    // Breakpoints.

    /// Resolve a location spec into (module, pc) pairs.
    pub(crate) fn resolve_location(&self, spec: &str) -> Result<Vec<(usize, u32)>> {
        let (file, line) = match BreakpointSpec::parse(spec)? {
            BreakpointSpec::Pc { module, pc } => {
                let module = match module {
                    Some(m) => m,
                    None => self
                        .default_module()
                        .or_else(|| self.vframes.first().map(|v| self.phys[v.phys].module))
                        .ok_or_else(|| anyhow!("No default module."))?,
                };
                if module >= self.modules.len() {
                    bail!("No module {module}.");
                }
                return Ok(vec![(module, pc)]);
            }
            BreakpointSpec::Line { file, line } => (file, Some(line)),
            BreakpointSpec::Function(_) => (None, None),
        };
        let spec = spec.trim();
        if let Some(line) = line {
            let mut out = vec![];
            let mut found_file = false;
            for (mi, m) in self.modules.iter().enumerate() {
                let Some(dbg) = &m.info.debug else { continue };
                let files = match &file {
                    Some(f) => dbg.find_files(f),
                    None => match self.list_pos {
                        Some((lm, lf, _)) if lm == mi => vec![lf],
                        _ => match dbg.find_functions("main").first() {
                            Some(f) => f.decl_file.into_iter().collect(),
                            None => vec![],
                        },
                    },
                };
                if files.is_empty() {
                    continue;
                }
                found_file = true;
                if let Some((addrs, _)) = dbg.line_addresses(&files, line) {
                    for a in addrs {
                        if let Some(pc) = m.info.dwarf_to_pc(a) {
                            out.push((mi, pc));
                        }
                    }
                }
            }
            if !found_file {
                match file {
                    Some(f) => bail!("No source file named {f}."),
                    None => bail!("No default source file; use FILE:LINE."),
                }
            }
            if out.is_empty() {
                bail!("No line {line} in the given file.");
            }
            return Ok(out);
        }
        // A function name: DWARF functions first, then Wasm names.
        let mut out = vec![];
        for (mi, m) in self.modules.iter().enumerate() {
            if let Some(dbg) = &m.info.debug {
                for f in dbg.find_functions(spec) {
                    if let Some(pc) = m.info.dwarf_to_pc(dbg.prologue_end(f)) {
                        out.push((mi, pc));
                    }
                }
            }
        }
        if out.is_empty() {
            for (mi, m) in self.modules.iter().enumerate() {
                for b in &m.info.bodies {
                    let name = m.info.func_display_name(b.index);
                    if name == spec || name.ends_with(&format!("::{spec}")) {
                        out.push((mi, b.code_start));
                    }
                }
            }
        }
        if out.is_empty() {
            bail!("Function \"{spec}\" not defined.");
        }
        Ok(out)
    }

    pub(crate) fn describe_pc(&self, module: usize, pc: u32) -> String {
        let m = &self.modules[module];
        let mut s = format!("pc {pc:#x}");
        if let (Some(dbg), Some(addr)) = (&m.info.debug, m.info.pc_to_dwarf(pc)) {
            if let Some(f) = dbg.function_at(addr) {
                let _ = write!(s, " in {}", f.name);
            }
            if let Some(loc) = dbg.source_loc(addr) {
                let _ = write!(s, " at {}:{}", basename(dbg.file_name(loc.file)), loc.line);
            }
        } else if let Some(f) = m.info.func_at_pc(pc) {
            let _ = write!(s, " in {}", m.info.func_display_name(f.index));
        }
        if self.modules.len() > 1 {
            let _ = write!(s, " [{}]", m.label);
        }
        s
    }

    fn cmd_break(&mut self, args: &str, temporary: bool) -> Result<()> {
        let spec = if args.is_empty() {
            // Break at the current location.
            let v = self
                .vframes
                .get(self.selected)
                .ok_or_else(|| anyhow!("No default breakpoint location."))?;
            let p = &self.phys[v.phys];
            format!("*{}:{:#x}", p.module, p.pc)
        } else {
            args.to_string()
        };
        let locs = self.resolve_location(&spec)?;
        let descs: Vec<String> = locs
            .iter()
            .map(|(m, pc)| self.describe_pc(*m, *pc))
            .collect();
        let id = self.add_breakpoint(spec, locs, temporary)?;
        let kind = if temporary {
            "Temporary breakpoint"
        } else {
            "Breakpoint"
        };
        if descs.len() == 1 {
            self.println(&format!("{kind} {id} at {}", descs[0]));
        } else {
            self.println(&format!("{kind} {id} at {} locations:", descs.len()));
            for d in descs {
                self.println(&format!("  {d}"));
            }
        }
        Ok(())
    }

    fn cmd_watch(&mut self, args: &str) -> Result<()> {
        if args.is_empty() {
            bail!("Argument required (expression to watch, or `-m ADDR LEN`).");
        }
        if let Some(rest) = args.strip_prefix("-m ") {
            let mut parts = rest.split_whitespace();
            let (Some(a), Some(l), None) = (parts.next(), parts.next(), parts.next()) else {
                bail!("Usage: watch -m ADDR LEN");
            };
            let addr = self.eval_address(a)?;
            let len = parse_u64(l)?;
            let inst = self.default_inst()?;
            let id = self.add_watchpoint(format!("-m {addr:#x} {len}"), inst, addr, len, None)?;
            self.println(&format!(
                "Watchpoint {id}: memory {addr:#x} length {len:#x}"
            ));
            return Ok(());
        }
        let t = self.watch_target(args)?;
        let id = self.add_watchpoint(args.to_string(), t.inst, t.addr, t.len, t.ty)?;
        self.println(&format!(
            "Watchpoint {id}: {args} (memory {:#x} length {:#x})",
            t.addr, t.len
        ));
        Ok(())
    }

    /// The memory holding the value of access path `expr`, to watch.
    pub(crate) fn watch_target(&mut self, expr: &str) -> Result<WatchTarget> {
        let p = path::parse(expr)?;
        if matches!(p.root(), Path::WasmName(_) | Path::Builtin(_)) {
            let loc = self.wasm_location(&p)?;
            bail!("Only values in linear memory can be watched; `{expr}` is in {loc}.");
        }
        let (module, inst, value) = match self.eval_opts(expr, false)? {
            HistEntry::Src {
                module,
                inst,
                value,
            } => (module, inst.unwrap_or(0), value),
            HistEntry::Wasm(_) => bail!("Only values in linear memory can be watched."),
        };
        self.value_watch_target(module, inst, &value)
            .map_err(|e| anyhow!("{e}; `{expr}` cannot be watched."))
    }

    /// The memory holding a source-level value, to watch.
    pub(crate) fn value_watch_target(
        &self,
        module: usize,
        inst: usize,
        value: &Value,
    ) -> Result<WatchTarget> {
        let dbg = self.modules[module].info.debug.as_ref().unwrap();
        let target = FrameTarget::new(self.d, None, &self.instances[inst].handle);
        let loc = Ctx::new(dbg, &target).location(value)?;
        let Location::Mem { base, size } = loc else {
            bail!("Only values in linear memory can be watched; the value is in {loc}");
        };
        Ok(WatchTarget {
            inst,
            addr: base,
            len: size,
            ty: Some((module, value.ty)),
        })
    }

    fn cmd_delete(&mut self, args: &str) -> Result<()> {
        let args = args.strip_prefix("breakpoints").unwrap_or(args).trim();
        if args.is_empty() {
            let ids: Vec<u32> = self
                .breakpoints
                .iter()
                .map(|b| b.id)
                .chain(self.watchpoints.iter().map(|w| w.id))
                .collect();
            for id in ids {
                self.delete_breakpoint(id);
            }
            self.println("Deleted all breakpoints.");
            return Ok(());
        }
        for a in args.split_whitespace() {
            let id: u32 = a
                .parse()
                .map_err(|_| anyhow!("Bad breakpoint number '{a}'"))?;
            if !self.delete_breakpoint(id) {
                self.println(&format!("No breakpoint number {id}."));
            }
        }
        Ok(())
    }

    fn info_breakpoints(&mut self) {
        if self.breakpoints.is_empty() && self.watchpoints.is_empty() {
            self.println("No breakpoints or watchpoints.");
            return;
        }
        let mut out = String::from("Num  Type          Hits  Where\n");
        for b in &self.breakpoints {
            let kind = if b.temporary {
                "tbreakpoint"
            } else {
                "breakpoint"
            };
            let _ = writeln!(out, "{:<4} {:<13} {:<5} {}", b.id, kind, b.hits, b.spec);
            for (m, pc) in &b.locs {
                let _ = writeln!(
                    out,
                    "                         {}",
                    self.describe_pc(*m, *pc)
                );
            }
        }
        for w in &self.watchpoints {
            let _ = writeln!(
                out,
                "{:<4} {:<13} {:<5} {}",
                w.id, "watchpoint", w.hits, w.spec
            );
            if w.ty.is_some() {
                let _ = writeln!(
                    out,
                    "                         memory {:#x} length {:#x}",
                    w.addr, w.len
                );
            }
        }
        // Keep the listing in id order.
        let mut lines: Vec<&str> = out.lines().collect();
        let header = lines.remove(0);
        let mut entries: Vec<(u32, Vec<&str>)> = vec![];
        for l in lines {
            match l.split_whitespace().next().and_then(|w| w.parse().ok()) {
                Some(id) if !l.starts_with(' ') => entries.push((id, vec![l])),
                _ => entries.last_mut().unwrap().1.push(l),
            }
        }
        entries.sort_by_key(|(id, _)| *id);
        let mut sorted = format!("{header}\n");
        for (_, ls) in entries {
            for l in ls {
                sorted.push_str(l);
                sorted.push('\n');
            }
        }
        self.print(&sorted);
    }

    // -----------------------------------------------------------------
    // Stack.

    fn cmd_backtrace(&mut self) -> Result<()> {
        self.require_stopped()?;
        let mut out = String::new();
        for vi in 0..self.vframes.len() {
            let v = &self.vframes[vi];
            let p = &self.phys[v.phys];
            let marker = if vi == self.selected { "*" } else { " " };
            let pcs = if v.inlined {
                "[inlined]".to_string()
            } else {
                format!("{:#08x}", p.pc)
            };
            let _ = writeln!(out, "{marker}#{vi:<3} {pcs} in {}", self.frame_desc(vi));
        }
        self.print(&out);
        Ok(())
    }

    fn select_frame(&mut self, vi: usize) -> Result<()> {
        if vi >= self.vframes.len() {
            bail!("No frame at level {vi}.");
        }
        self.selected = vi;
        self.print_frame_header(vi, true);
        Ok(())
    }

    fn cmd_frame(&mut self, args: &str) -> Result<()> {
        self.require_stopped()?;
        let vi = if args.is_empty() {
            self.selected
        } else {
            usize::try_from(parse_u64(args)?)?
        };
        self.select_frame(vi)
    }

    fn cmd_up_down(&mut self, args: &str, up: bool) -> Result<()> {
        self.require_stopped()?;
        let n = if args.is_empty() {
            1
        } else {
            usize::try_from(parse_u64(args)?)?
        };
        let vi = if up {
            let vi = self.selected + n;
            if vi >= self.vframes.len() {
                bail!("Initial frame selected; you cannot go up.");
            }
            vi
        } else {
            self.selected
                .checked_sub(n)
                .ok_or_else(|| anyhow!("Bottom (innermost) frame selected; you cannot go down."))?
        };
        self.select_frame(vi)
    }

    // -----------------------------------------------------------------
    // Source listing.

    fn cmd_list(&mut self, args: &str) -> Result<()> {
        let (module, file, center) = if args.is_empty() {
            match self.list_pos {
                // `list_pos` holds the next line to show.
                Some((m, f, l)) => (m, f, l + 5),
                None => {
                    let m = self
                        .default_module()
                        .ok_or_else(|| anyhow!("No debug info."))?;
                    let dbg = self.modules[m].info.debug.as_ref().unwrap();
                    let f = dbg
                        .find_functions("main")
                        .first()
                        .copied()
                        .ok_or_else(|| anyhow!("No default source file."))?;
                    let loc = dbg
                        .source_loc(f.low)
                        .ok_or_else(|| anyhow!("No line info for main."))?;
                    (m, loc.file, loc.line)
                }
            }
        } else if args.chars().all(|c| c.is_ascii_digit()) {
            let (m, f, _) = self
                .list_pos
                .ok_or_else(|| anyhow!("No default source file; use FILE:LINE."))?;
            (m, f, args.parse()?)
        } else {
            let locs = self.resolve_location(args)?;
            let (m, pc) = locs[0];
            let dbg = self.modules[m]
                .info
                .debug
                .as_ref()
                .ok_or_else(|| anyhow!("No debug info for that location."))?;
            let addr = self.modules[m].info.pc_to_dwarf(pc).unwrap_or(0);
            let loc = dbg
                .source_loc(addr)
                .ok_or_else(|| anyhow!("No line info for that location."))?;
            (m, loc.file, loc.line)
        };
        let start = center.saturating_sub(5).max(1);
        let path = self.modules[module]
            .info
            .debug
            .as_ref()
            .unwrap()
            .file_name(file)
            .to_string();
        let lines = self.source_lines(&path).cloned().ok_or_else(|| {
            anyhow!(
                "Cannot read source file {path}.\n\
                     (Give the debugger read-only access with `-Ddebug-fs=<dir>`.)"
            )
        })?;
        let mut last = start;
        for n in start..start + 10 {
            let Some(text) = lines.get(usize::try_from(n)? - 1) else {
                break;
            };
            self.println(&format!("{n}\t{text}"));
            last = n;
        }
        // The next `list` (without arguments) continues after the last
        // line shown.
        self.list_pos = Some((module, file, last + 1));
        Ok(())
    }

    // -----------------------------------------------------------------
    // Values.

    /// Evaluate a path to either a source-level value (with its module
    /// and instance) or a Wasm-level value.
    pub(crate) fn eval(&mut self, expr: &str) -> Result<HistEntry> {
        self.eval_opts(expr, true)
    }

    /// Evaluate a path. With `snapshot`, source-level values are captured
    /// (as for the value history); otherwise they keep referring to
    /// their storage, which preserves their location for `location`.
    pub(crate) fn eval_opts(&mut self, expr: &str, snapshot: bool) -> Result<HistEntry> {
        let p = path::parse(expr)?;
        match p.root() {
            Path::History(h) => {
                let idx = self.history_index(h)?;
                match &self.history[idx] {
                    HistEntry::Wasm(w) => {
                        let w = w.duplicate(self.d);
                        self.eval_wasm_path(&p, w)
                    }
                    HistEntry::Src {
                        module,
                        inst,
                        value,
                    } => {
                        let (module, inst, value) = (*module, *inst, value.clone());
                        let inst = inst.map(|i| &self.instances[i].handle);
                        let dbg = self.modules[module].info.debug.as_ref().unwrap();
                        let fallback;
                        let inst = match inst {
                            Some(i) => i,
                            None => {
                                fallback = &self.instances[self.default_inst()?].handle;
                                fallback
                            }
                        };
                        let target = FrameTarget::new(self.d, None, inst);
                        let ctx = Ctx::new(dbg, &target);
                        let v = path::eval(&ctx, &p, &mut |_| Ok(value.clone()))?;
                        let v = if snapshot {
                            ctx.snapshot(&v).unwrap_or(v)
                        } else {
                            v
                        };
                        Ok(HistEntry::Src {
                            module,
                            inst: self.inst_index_of(inst),
                            value: v,
                        })
                    }
                }
            }
            Path::WasmName(name) => {
                let root = self.named_wasm_entity(name)?.ok_or_else(|| {
                    anyhow!(
                        "No Wasm local, global or table named `${name}` in the current context \
                         (indexed entities are `%localN`, `%globalN`, etc.)."
                    )
                })?;
                self.eval_wasm_path(&p, root)
            }
            Path::Builtin(name) => {
                let root = self.builtin_root(name)?;
                self.eval_wasm_path(&p, root)
            }
            _ => self.eval_source(&p, snapshot),
        }
    }

    fn inst_index_of(&self, inst: &crate::api::Instance) -> Option<usize> {
        let id = inst.unique_id();
        self.instances
            .iter()
            .position(|i| i.handle.unique_id() == id)
    }

    fn history_index(&self, h: &HistoryRef) -> Result<usize> {
        let n = self.history.len();
        match *h {
            HistoryRef::Absolute(i) if i >= 1 && i <= n => Ok(i - 1),
            HistoryRef::Absolute(i) => bail!("History has not yet reached %{i}."),
            HistoryRef::Relative(k) if k < n => Ok(n - 1 - k),
            HistoryRef::Relative(_) => bail!("History is empty or too short."),
        }
    }

    /// Resolve a `%name` built-in: indexed Wasm entities and the PC.
    fn builtin_root(&self, name: &str) -> Result<WVal> {
        let num = |prefix: &str| -> Option<u32> { name.strip_prefix(prefix)?.parse().ok() };
        if name == "pc" {
            self.require_stopped()?;
            let pc = self.phys[self.vframes[self.selected].phys].pc;
            return Ok(WVal::Num(WasmVal::I32(pc)));
        }
        if let Some(i) = num("local") {
            self.require_stopped()?;
            let f = &self.phys[self.vframes[self.selected].phys].frame;
            let locals = f.get_locals(self.d).map_err(|e| anyhow!("{e:?}"))?;
            let v = locals
                .get(usize::try_from(i)?)
                .ok_or_else(|| anyhow!("No local {i} (frame has {}).", locals.len()))?;
            return Ok(WVal::from_api(v));
        }
        if let Some(i) = num("stack") {
            self.require_stopped()?;
            let f = &self.phys[self.vframes[self.selected].phys].frame;
            let stack = f.get_stack(self.d).map_err(|e| anyhow!("{e:?}"))?;
            let v = stack
                .get(usize::try_from(i)?)
                .ok_or_else(|| anyhow!("No operand-stack slot {i} (depth {}).", stack.len()))?;
            return Ok(WVal::from_api(v));
        }
        if let Some(i) = num("global") {
            let inst = &self.instances[self.default_inst()?].handle;
            let g = inst
                .get_global(self.d, i)
                .map_err(|_| anyhow!("No global {i}."))?;
            let v = g.get(self.d).map_err(|e| anyhow!("{e:?}"))?;
            return Ok(WVal::from_api(&v));
        }
        if let Some(i) = num("table") {
            let inst = &self.instances[self.default_inst()?].handle;
            let t = inst
                .get_table(self.d, i)
                .map_err(|_| anyhow!("No table {i}."))?;
            return Ok(WVal::Table(t));
        }
        bail!("Unknown built-in `%{name}` (try %localN, %stackN, %globalN, %tableN, %pc).")
    }

    /// Look up a Wasm entity by its name-section name: a local of the
    /// selected frame's function, else a global or table of the selected
    /// frame's instance (or the default instance).
    fn named_wasm_entity(&self, name: &str) -> Result<Option<WVal>> {
        Ok(match self.resolve_wasm_name(name) {
            Some(NamedEntity::Local(i)) => Some(self.builtin_root(&format!("local{i}"))?),
            Some(NamedEntity::Global(i)) => Some(self.builtin_root(&format!("global{i}"))?),
            Some(NamedEntity::Table(i)) => Some(self.builtin_root(&format!("table{i}"))?),
            None => None,
        })
    }

    /// Resolve a name-section name: a local of the selected frame's
    /// function, else a table or global of the default instance.
    fn resolve_wasm_name(&self, name: &str) -> Option<NamedEntity> {
        if let Some(v) = self.vframes.get(self.selected) {
            let p = &self.phys[v.phys];
            let info = &self.modules[p.module].info;
            if let Some(func) = p.func {
                if let Some(((_, l), _)) = info
                    .local_names
                    .iter()
                    .find(|((f, _), n)| *f == func && n.as_str() == name)
                {
                    return Some(NamedEntity::Local(*l));
                }
            }
        }
        let ii = self.default_inst().ok()?;
        let info = &self.modules[self.instances[ii].module].info;
        let find = |names: &std::collections::HashMap<u32, String>| {
            names
                .iter()
                .find(|(_, n)| n.as_str() == name)
                .map(|(i, _)| *i)
        };
        if let Some(t) = find(&info.table_names) {
            return Some(NamedEntity::Table(t));
        }
        find(&info.global_names).map(NamedEntity::Global)
    }

    /// The storage location of a Wasm-level path (`%local3`, `$x`,
    /// `%global0`, `%stack1`). Only whole slots are locations; GC object
    /// fields and table elements are not (yet) supported.
    pub(crate) fn wasm_location(&self, p: &Path) -> Result<Location> {
        let slot = match p {
            Path::Builtin(name) => {
                let num = |prefix: &str| -> Option<u32> { name.strip_prefix(prefix)?.parse().ok() };
                if let Some(i) = num("local") {
                    WasmSlot::Local(i)
                } else if let Some(i) = num("global") {
                    WasmSlot::Global(i)
                } else if let Some(i) = num("stack") {
                    WasmSlot::Stack(i)
                } else if name == "pc" {
                    bail!("`%pc` is not stored in the Wasm machine's state.")
                } else {
                    bail!("`%{name}` has no storage location.")
                }
            }
            Path::WasmName(name) => match self.resolve_wasm_name(name) {
                Some(NamedEntity::Local(i)) => WasmSlot::Local(i),
                Some(NamedEntity::Global(i)) => WasmSlot::Global(i),
                Some(NamedEntity::Table(_)) => bail!("`${name}` is a table, not a storage slot."),
                None => bail!("No Wasm local or global named `${name}`."),
            },
            _ => bail!("Locations of Wasm GC object fields and table elements are not supported."),
        };
        let size = match self.builtin_root(&match slot {
            WasmSlot::Local(i) => format!("local{i}"),
            WasmSlot::Global(i) => format!("global{i}"),
            WasmSlot::Stack(i) => format!("stack{i}"),
        })? {
            WVal::Num(WasmVal::I32(_) | WasmVal::F32(_)) => 4,
            WVal::Num(WasmVal::I64(_) | WasmVal::F64(_)) => 8,
            WVal::Num(WasmVal::V128(_)) => 16,
            // References are opaque; report a nominal size.
            _ => 4,
        };
        Ok(Location::in_slot(slot, 0, size))
    }

    fn cmd_location(&mut self, args: &str) -> Result<()> {
        if args.is_empty() {
            bail!("Argument required (path whose location to show).");
        }
        let p = path::parse(args)?;
        let loc = match p.root() {
            Path::WasmName(_) | Path::Builtin(_) => self.wasm_location(&p)?,
            _ => match self.eval_opts(args, false)? {
                HistEntry::Src {
                    module,
                    inst,
                    value,
                } => {
                    let dbg = self.modules[module].info.debug.as_ref().unwrap();
                    let inst = inst.unwrap_or(0);
                    let target = FrameTarget::new(self.d, None, &self.instances[inst].handle);
                    Ctx::new(dbg, &target).location(&value)?
                }
                HistEntry::Wasm(_) => bail!(
                    "Locations of Wasm GC object fields and table elements are not supported."
                ),
            },
        };
        self.println(&loc.to_string());
        Ok(())
    }

    fn eval_wasm_path(&self, p: &Path, root: WVal) -> Result<HistEntry> {
        fn go(s: &Session<'_>, p: &Path, root: &WVal) -> Result<WVal> {
            match p {
                Path::Field(inner, f) => go(s, inner, root)?.field(s.d, f),
                Path::Index(inner, i) => go(s, inner, root)?.index(s.d, *i),
                Path::Deref(_) | Path::AddrOf(_) | Path::Arrow(..) => {
                    bail!("`*`, `&` and `->` are not supported on Wasm-level values")
                }
                _ => Ok(root.duplicate(s.d)),
            }
        }
        Ok(HistEntry::Wasm(go(self, p, &root)?))
    }

    fn eval_source(&mut self, p: &Path, snapshot: bool) -> Result<HistEntry> {
        let module = self
            .default_module()
            .ok_or_else(|| anyhow!("No debug info available."))?;
        let dbg = self.modules[module].info.debug.as_ref().unwrap();
        let sel = self
            .vframes
            .get(self.selected)
            .filter(|v| self.phys[v.phys].module == module);
        let (frame, inst) = match sel {
            Some(v) => {
                let p = &self.phys[v.phys];
                (Some(&p.frame), p.inst)
            }
            None => {
                let inst = self
                    .instances
                    .iter()
                    .position(|i| i.module == module)
                    .ok_or_else(|| anyhow!("No instance of the module exists yet."))?;
                (None, inst)
            }
        };
        let target = FrameTarget::new(self.d, frame, &self.instances[inst].handle);
        let ctx = Ctx::new(dbg, &target);
        let addr = sel.and_then(|v| self.dwarf_addr(v.phys));
        let fb = sel.and_then(|v| self.frame_base(v.phys, &target));
        let scope = sel.and_then(|v| v.scope.as_ref());
        let mut root = |r: &Path| -> Result<Value> {
            match r {
                Path::Var(name) => {
                    if let (Some(scope), Some(addr)) = (scope, addr) {
                        if let Some(var) = scope.find_var(name) {
                            return ctx.variable(var, addr, fb);
                        }
                    }
                    if let Some(g) = dbg.find_global(name) {
                        let ty = dbg
                            .global_type(g)?
                            .ok_or_else(|| anyhow!("global `{name}` has no type"))?;
                        let loc = dbg.global_loc(g)?;
                        let l = dbg.eval_loc(g.die.unit, &loc, addr.unwrap_or(0), None, &target)?;
                        return ctx.from_location(ty, l);
                    }
                    bail!("No symbol \"{name}\" in current context.")
                }
                Path::Int(i) => {
                    let lang = scope
                        .and_then(|s| s.vars.first())
                        .map(|v| dbg.lang(v.die))
                        .unwrap_or(Lang::C);
                    let ty = dbg.synth_base("long", BaseEnc::Signed, 8, lang);
                    Ok(Value::from_bytes(ty, i.to_le_bytes().to_vec()))
                }
                _ => bail!("unsupported path root"),
            }
        };
        let v = path::eval(&ctx, p, &mut root)?;
        let v = if snapshot {
            ctx.snapshot(&v).unwrap_or(v)
        } else {
            v
        };
        Ok(HistEntry::Src {
            module,
            inst: Some(inst),
            value: v,
        })
    }

    fn format_entry(&self, e: &HistEntry, opts: &FormatOptions) -> String {
        match e {
            HistEntry::Wasm(w) => wasmval::format(self.d, w, opts.hex, 3),
            HistEntry::Src {
                module,
                inst,
                value,
            } => {
                let dbg = self.modules[*module].info.debug.as_ref().unwrap();
                let inst = inst.unwrap_or(0);
                let target = FrameTarget::new(self.d, None, &self.instances[inst].handle);
                let ctx = Ctx::new(dbg, &target);
                format_value(&ctx, value, opts)
            }
        }
    }

    fn cmd_print(&mut self, args: &str, fmt: &str) -> Result<()> {
        if args.is_empty() {
            bail!("Argument required (expression to print).");
        }
        let opts = FormatOptions {
            hex: fmt.contains('x'),
            raw: fmt.contains('r'),
            ..Default::default()
        };
        let e = self.eval(args)?;
        let text = self.format_entry(&e, &opts);
        self.history.push(e);
        let n = self.history.len();
        self.println(&format!("%{n} = {text}"));
        Ok(())
    }

    fn cmd_ptype(&mut self, args: &str, detailed: bool) -> Result<()> {
        let e = self.eval(args)?;
        let text = match &e {
            HistEntry::Wasm(w) => match w {
                WVal::Num(n) => wasmval::format_num(*n, false)
                    .split(' ')
                    .next()
                    .unwrap()
                    .to_string(),
                WVal::Func(_) => "funcref".into(),
                WVal::Exn(_) => "exnref".into(),
                WVal::Any(_) => "anyref".into(),
                WVal::Extern { .. } => "externref".into(),
                WVal::Table(_) => "table".into(),
            },
            HistEntry::Src { module, value, .. } => {
                let dbg = self.modules[*module].info.debug.as_ref().unwrap();
                if detailed {
                    describe_type(dbg, value.ty)
                } else {
                    dbg.type_name(value.ty)
                }
            }
        };
        self.println(&format!("type = {text}"));
        Ok(())
    }

    fn cmd_examine(&mut self, args: &str, fmt: &str) -> Result<()> {
        let ef = ExamineFormat::parse(fmt)?;
        let (count, format) = (ef.count, ef.format);
        let addr = self.eval_address(args)?;
        let inst = self.default_inst()?;
        let target = FrameTarget::new(self.d, None, &self.instances[inst].handle);
        let mut out = String::new();
        if format == 's' {
            let mut a = addr;
            for _ in 0..count {
                let mut bytes = vec![];
                loop {
                    let b = target.read_memory(0, a, 1)?;
                    a += 1;
                    if b[0] == 0 || bytes.len() >= 1024 {
                        break;
                    }
                    bytes.push(b[0]);
                }
                out.push_str(&format_examine_string(a - bytes.len() as u64 - 1, &bytes));
            }
            self.print(&out);
            return Ok(());
        }
        let bytes = target.read_memory(0, addr, ef.byte_len())?;
        out.push_str(&ef.format(addr, &bytes));
        self.print(&out);
        Ok(())
    }

    /// Evaluate an address: a number, a pointer or integer value, or the
    /// address of a value in memory.
    pub(crate) fn eval_address(&mut self, args: &str) -> Result<u64> {
        Ok(if let Ok(a) = parse_u64(args) {
            a
        } else {
            let e = self.eval(args)?;
            match &e {
                HistEntry::Src {
                    module,
                    inst,
                    value,
                } => {
                    let dbg = self.modules[*module].info.debug.as_ref().unwrap();
                    let t = dbg.ty(dbg.strip(value.ty));
                    let target =
                        FrameTarget::new(self.d, None, &self.instances[inst.unwrap_or(0)].handle);
                    let ctx = Ctx::new(dbg, &target);
                    match t.kind {
                        TypeKind::Pointer { .. } | TypeKind::Base(_) => ctx.read_uint(value)?,
                        _ => value
                            .address()
                            .ok_or_else(|| anyhow!("Value has no address."))?,
                    }
                }
                HistEntry::Wasm(w) => match w.as_num() {
                    Some(n) => n.as_u64(),
                    None => bail!("Not an address."),
                },
            }
        })
    }

    // -----------------------------------------------------------------
    // `info`.

    fn cmd_info(&mut self, args: &str) -> Result<()> {
        let (sub, rest) = match args.split_once(' ') {
            Some((s, r)) => (s, r.trim()),
            None => (args, ""),
        };
        match sub {
            "breakpoints" | "break" | "b" => {
                self.info_breakpoints();
                Ok(())
            }
            "locals" => self.info_vars(false),
            "args" => self.info_vars(true),
            "frame" => self.info_frame(),
            "functions" => self.info_functions(rest),
            "variables" | "globals" => self.info_variables(rest),
            "source" => self.info_source(),
            "modules" => self.cmd_wasm_modules(),
            "instances" => self.cmd_wasm_instances(),
            "" => bail!(
                "\"info\" must be followed by a subcommand (breakpoints, locals, args, frame, functions, variables, source)."
            ),
            _ => bail!("Undefined info command: \"{sub}\"."),
        }
    }

    fn info_vars(&mut self, params: bool) -> Result<()> {
        self.require_stopped()?;
        let v = &self.vframes[self.selected];
        let Some(scope) = &v.scope else {
            self.println("No symbol table info available.");
            return Ok(());
        };
        let p = &self.phys[v.phys];
        let m = &self.modules[p.module];
        let dbg = m.info.debug.as_ref().unwrap();
        let addr = m.info.pc_to_dwarf(p.lookup_pc).unwrap();
        let target = FrameTarget::new(self.d, Some(&p.frame), &self.instances[p.inst].handle);
        let fb = self.frame_base(v.phys, &target);
        let ctx = Ctx::new(dbg, &target);
        let mut out = String::new();
        let mut any = false;
        for var in scope.vars.iter().filter(|x| x.is_param == params) {
            any = true;
            let text = match ctx.variable(var, addr, fb) {
                Ok(val) => format_value(&ctx, &val, &FormatOptions::default()),
                Err(e) => format!("{e}"),
            };
            let _ = writeln!(out, "{} = {text}", var.name);
        }
        if !any {
            out.push_str(if params {
                "No arguments.\n"
            } else {
                "No locals.\n"
            });
        }
        self.print(&out);
        Ok(())
    }

    fn info_frame(&mut self) -> Result<()> {
        self.require_stopped()?;
        let vi = self.selected;
        let v = &self.vframes[vi];
        let p = &self.phys[v.phys];
        let m = &self.modules[p.module];
        let mut out = format!("Stack level {vi}, {}\n", self.frame_desc(vi));
        let _ = writeln!(
            out,
            "  module [{}] {}, instance #{}, pc {:#x}",
            p.module, m.label, p.inst, p.pc
        );
        if let Some(f) = p.func {
            let sig = m
                .info
                .func_sig(f)
                .map(|s| s.to_string())
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "  Wasm function {f}: {} {sig}",
                m.info.func_display_name(f)
            );
        }
        if v.inlined {
            out.push_str("  inlined into its caller\n");
        }
        let target = FrameTarget::new(self.d, Some(&p.frame), &self.instances[p.inst].handle);
        if let Some(fb) = self.frame_base(v.phys, &target) {
            let _ = writeln!(out, "  frame base: {fb:#x}");
        }
        self.print(&out);
        Ok(())
    }

    fn info_functions(&mut self, filter: &str) -> Result<()> {
        let mut out = String::new();
        for (mi, m) in self.modules.iter().enumerate() {
            let Some(dbg) = &m.info.debug else { continue };
            let mut names: Vec<&str> = dbg
                .functions
                .iter()
                .map(|f| f.name.as_str())
                .filter(|n| n.contains(filter))
                .collect();
            names.sort();
            names.dedup();
            let _ = writeln!(
                out,
                "Module [{mi}] {}: {} matching functions",
                m.label,
                names.len()
            );
            for n in names.iter().take(200) {
                let _ = writeln!(out, "  {n}");
            }
            if names.len() > 200 {
                out.push_str("  ...\n");
            }
        }
        self.print(&out);
        Ok(())
    }

    fn info_variables(&mut self, filter: &str) -> Result<()> {
        let mut out = String::new();
        for (mi, m) in self.modules.iter().enumerate() {
            let Some(dbg) = &m.info.debug else { continue };
            let mut names: Vec<&str> = dbg
                .globals
                .iter()
                .map(|g| g.name.as_str())
                .filter(|n| n.contains(filter))
                .collect();
            names.sort();
            names.dedup();
            let _ = writeln!(
                out,
                "Module [{mi}] {}: {} matching globals",
                m.label,
                names.len()
            );
            for n in names.iter().take(200) {
                let _ = writeln!(out, "  {n}");
            }
        }
        self.print(&out);
        Ok(())
    }

    fn info_source(&mut self) -> Result<()> {
        let (m, f, _) = self
            .list_pos
            .ok_or_else(|| anyhow!("No current source file."))?;
        let path = self.modules[m]
            .info
            .debug
            .as_ref()
            .unwrap()
            .file_name(f)
            .to_string();
        let readable = self.source_lines(&path).is_some();
        self.println(&format!(
            "Current source file is {path}{}",
            if readable { "" } else { " (not readable)" }
        ));
        Ok(())
    }

    // -----------------------------------------------------------------
    // Wasm machine level.

    fn cmd_disas(&mut self, args: &str) -> Result<()> {
        let mut all = false;
        let mut target_spec = None;
        for a in args.split_whitespace() {
            if a == "all" || a == "/a" {
                all = true;
            } else {
                target_spec = Some(a.to_string());
            }
        }
        let (module, pc, func) = match target_spec {
            Some(spec) => {
                let (m, pc) = self.resolve_location(&spec)?[0];
                let f = self.modules[m]
                    .info
                    .func_at_pc(pc)
                    .ok_or_else(|| anyhow!("No function at {pc:#x}."))?
                    .index;
                all = all || !spec.starts_with('*');
                (m, Some(pc), f)
            }
            None => {
                self.require_stopped()?;
                let p = &self.phys[self.vframes[self.selected].phys];
                let f = p.func.ok_or_else(|| anyhow!("No function at pc."))?;
                (p.module, Some(p.pc), f)
            }
        };
        let info = &self.modules[module].info;
        let lines = info.disassemble_func(func)?;
        let cur = self
            .vframes
            .get(self.selected)
            .map(|v| &self.phys[v.phys])
            .filter(|p| p.module == module)
            .map(|p| p.pc);
        let center = lines
            .iter()
            .position(|l| Some(l.offset) == pc.or(cur))
            .unwrap_or(0);
        let (start, end) = if all {
            (0, lines.len())
        } else {
            (center.saturating_sub(10), (center + 15).min(lines.len()))
        };
        let mut out = format!(
            "Dump of Wasm code for function {} (func {func}):\n",
            info.func_display_name(func)
        );
        if start > 0 {
            out.push_str("   ...\n");
        }
        for l in &lines[start..end] {
            let marker = if Some(l.offset) == cur { "=>" } else { "  " };
            let indent = "  ".repeat(l.indent.saturating_sub(2));
            let _ = writeln!(
                out,
                "{marker} {:#08x}: {indent}{}",
                l.offset,
                demangle_wat(&l.text)
            );
        }
        if end < lines.len() {
            out.push_str("   ...\n");
        }
        out.push_str("End of dump.\n");
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_locals(&mut self) -> Result<()> {
        self.require_stopped()?;
        let p = &self.phys[self.vframes[self.selected].phys];
        let names = p
            .func
            .map(|f| self.modules[p.module].info.func_locals(f))
            .unwrap_or_default();
        let locals = p.frame.get_locals(self.d).map_err(|e| anyhow!("{e:?}"))?;
        let mut out = String::new();
        for (i, v) in locals.iter().enumerate() {
            let name = names
                .get(i)
                .and_then(|(n, _)| n.clone())
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "local {i}{name}: {}",
                wasmval::format(self.d, &WVal::from_api(v), false, 1)
            );
        }
        if locals.is_empty() {
            out.push_str("No locals.\n");
        }
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_stack(&mut self) -> Result<()> {
        self.require_stopped()?;
        let p = &self.phys[self.vframes[self.selected].phys];
        let stack = p.frame.get_stack(self.d).map_err(|e| anyhow!("{e:?}"))?;
        let mut out = String::new();
        if stack.is_empty() {
            out.push_str("Operand stack is empty.\n");
        }
        for (i, v) in stack.iter().enumerate() {
            let top = if i + 1 == stack.len() { "  <- top" } else { "" };
            let _ = writeln!(
                out,
                "stack[{i}]: {}{top}",
                wasmval::format(self.d, &WVal::from_api(v), false, 1)
            );
        }
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_globals(&mut self, args: &str) -> Result<()> {
        let f = parse_flags(args, "-g")?;
        let ii = self.inst_arg(f.inst)?;
        let module = self.instances[ii].module;
        let n = self.modules[module].info.num_globals();
        let mut out = format!(
            "Globals of instance #{ii} ({}):\n",
            self.modules[module].label
        );
        for g in 0..n {
            let inst = &self.instances[ii].handle;
            let text = match inst.get_global(self.d, g).and_then(|g| g.get(self.d)) {
                Ok(v) => wasmval::format(self.d, &WVal::from_api(&v), false, 1),
                Err(e) => format!("<{e:?}>"),
            };
            let name = self.modules[module]
                .info
                .global_names
                .get(&g)
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            let _ = writeln!(out, "  global {g}{name}: {text}");
        }
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_memory(&mut self, args: &str) -> Result<()> {
        let f = parse_flags(args, "-m")?;
        let ii = self.inst_arg(f.inst)?;
        let mi = f.index.unwrap_or(0);
        let mem = self.instances[ii]
            .handle
            .get_memory(self.d, mi)
            .map_err(|_| anyhow!("Instance #{ii} has no memory {mi}."))?;
        let size = mem.size_bytes(self.d);
        let page = mem.page_size_bytes(self.d);
        if f.rest.is_empty() {
            self.println(&format!(
                "Memory {mi} of instance #{ii}: {size} bytes ({} pages of {page} bytes)",
                size / page.max(1)
            ));
            return Ok(());
        }
        let addr = parse_u64(&f.rest[0])?;
        let len = match f.rest.get(1) {
            Some(l) => parse_u64(l)?,
            None => 64,
        }
        .min(64 * 1024);
        let bytes = mem
            .get_bytes(self.d, addr, len)
            .map_err(|_| anyhow!("Cannot access memory at {addr:#x}+{len} (size {size})."))?;
        let out = hexdump(addr, &bytes);
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_table(&mut self, args: &str) -> Result<()> {
        let f = parse_flags(args, "-t")?;
        let ii = self.inst_arg(f.inst)?;
        let ti = f.index.unwrap_or(0);
        let table = self.instances[ii]
            .handle
            .get_table(self.d, ti)
            .map_err(|_| anyhow!("Instance #{ii} has no table {ti}."))?;
        let len = table.len(self.d);
        let start = match f.rest.first() {
            Some(s) => parse_u64(s)?,
            None => 0,
        };
        let count = match f.rest.get(1) {
            Some(c) => parse_u64(c)?,
            None => 16,
        };
        let mut out = format!("Table {ti} of instance #{ii}: {len} elements\n");
        for i in start..(start + count).min(len) {
            let text = match table.get_element(self.d, i) {
                Ok(v) => wasmval::format(self.d, &WVal::from_api(&v), false, 1),
                Err(e) => format!("<{e:?}>"),
            };
            let _ = writeln!(out, "  [{i}] {text}");
        }
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_instances(&mut self) -> Result<()> {
        self.refresh_entities();
        let mut out = String::new();
        for (i, inst) in self.instances.iter().enumerate() {
            let m = &self.modules[inst.module];
            let _ = writeln!(
                out,
                "#{i}  module [{}] {}: {} funcs, {} memories, {} tables, {} globals",
                inst.module,
                m.label,
                m.info.num_funcs(),
                m.info.num_memories(),
                m.info.num_tables(),
                m.info.num_globals()
            );
        }
        if self.instances.is_empty() {
            out.push_str("No instances.\n");
        }
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_modules(&mut self) -> Result<()> {
        self.refresh_entities();
        let mut out = String::new();
        for (i, m) in self.modules.iter().enumerate() {
            let _ = writeln!(
                out,
                "[{i}] {}: {} bytes, {} funcs ({} imported), debug info: {}",
                m.label,
                m.info.bytes.len(),
                m.info.num_funcs(),
                m.info.num_imported_funcs,
                if m.info.debug.is_some() { "yes" } else { "no" }
            );
        }
        self.print(&out);
        Ok(())
    }

    fn cmd_wasm_functions(&mut self, args: &str) -> Result<()> {
        let f = parse_flags(args, "-m")?;
        let filter = f.rest.join(" ");
        // Functions are a property of the module; select it via an
        // instance (`-i`), directly (`-m`), or by default.
        let (mi, what) = match (f.inst, f.index) {
            (_, Some(m)) => (usize::try_from(m)?, format!("module [{m}]")),
            (Some(_), None) => {
                let ii = self.inst_arg(f.inst)?;
                (self.instances[ii].module, format!("instance #{ii}"))
            }
            (None, None) => match self.default_inst() {
                Ok(ii) => (self.instances[ii].module, format!("instance #{ii}")),
                Err(_) => {
                    let m = self.default_module().unwrap_or(0);
                    (m, format!("module [{m}]"))
                }
            },
        };
        let m = self
            .modules
            .get(mi)
            .ok_or_else(|| anyhow!("No module {mi}."))?;
        let mut out = format!("Functions of {what} ({}):\n", m.label);
        let mut shown = 0;
        for idx in 0..m.info.num_funcs() {
            let name = m.info.func_display_name(idx);
            if !name.contains(&filter) {
                continue;
            }
            shown += 1;
            if shown > 500 {
                out.push_str("  ...\n");
                break;
            }
            let sig = m
                .info
                .func_sig(idx)
                .map(|s| s.to_string())
                .unwrap_or_default();
            let kind = match m.info.body(idx) {
                Some(b) => format!("@{:#x}, {} bytes", b.range.start, b.range.len()),
                None => "import".to_string(),
            };
            let _ = writeln!(out, "  {idx:>5}  {name} {sig}  [{kind}]");
        }
        self.print(&out);
        Ok(())
    }
}

/// A Wasm entity named in the name section.
enum NamedEntity {
    Local(u32),
    Global(u32),
    Table(u32),
}

#[derive(Clone, Copy)]
enum Resume {
    Continue,
    Step,
    Next,
    StepI,
    NextI,
    Finish,
    ReverseContinue,
    ReverseStep,
    ReverseNext,
    ReverseFinish,
    ReverseStepI,
    ReverseNextI,
}
