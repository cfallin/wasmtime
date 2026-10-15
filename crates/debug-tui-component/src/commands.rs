//! Command parsing and execution.

use crate::session::{FrameTarget, HistEntry, Outcome, RunState, Session, StopReason, to_wasm_val};
use crate::wasmval::{self, WVal};
use anyhow::{Result, anyhow, bail};
use std::fmt::Write;
use wasmtime_debug_info::demangle::demangle;
use wasmtime_debug_info::format::{FormatOptions, describe_type, format_value};
use wasmtime_debug_info::path::{self, HistoryRef, Path};
use wasmtime_debug_info::types::{BaseEnc, Lang, TypeKind};
use wasmtime_debug_info::value::{Ctx, Value};
use wasmtime_debug_info::{Target, WasmVal};

const HELP: &str = "\
Execution:
  start                    Run to the beginning of `main`.
  continue (c), run (r)    Continue execution.
  step (s)                 Step one source line, entering calls.
  next (n)                 Step one source line, stepping over calls.
  finish (fin)             Run until the selected frame returns.
  stepi (si)               Step one Wasm instruction.
  nexti (ni)               Step one Wasm instruction, stepping over calls.
  interrupt, Ctrl-C        Stop a running program.
Breakpoints:
  break (b) LOC            Set a breakpoint. LOC is FILE:LINE, LINE, FUNCTION,
                           *PC (module offset in the current module), or
                           *MODULE:PC.
  tbreak LOC               Set a temporary (one-shot) breakpoint.
  delete (d) [N...]        Delete breakpoints (all if none given).
  info breakpoints         List breakpoints.
Stack:
  backtrace (bt)           Show the call stack (including inlined frames).
  frame (f) [N]            Select / show a frame.
  up [N], down [N]         Move up (toward callers) / down the stack.
Source and data:
  list (l) [LOC]           List source lines.
  print (p)[/x|/r] PATH    Print a value. PATH is an access path: `x`, `p->x`,
                           `arr[2].y`, `*p`, `&x`, `t.0`, `$1->next`, `$`, `$$2`.
                           /x prints integers in hex; /r disables pretty-printers.
                           Wasm-level roots: $NAME (a Wasm local, global or
                           table named in the name section), and built-ins
                           %localN, %stackN, %globalN, %tableN[I], %pc.
                           GC refs: `.N` (field), `[I]`.
  ptype PATH, whatis PATH  Show the type of a value.
  info locals|args         Show variables of the selected frame.
  info frame               Describe the selected frame.
  info functions [TEXT]    List functions (with debug info) matching TEXT.
  info variables [TEXT]    List global variables matching TEXT.
  info source              Show the current source file.
  x/NFU ADDR|PATH          Examine linear memory: N count, F format
                           (x,d,u,c,s), U unit size (b,h,w,g).
Wasm machine level:
  disassemble (disas) [FUNC|*PC] [all]
                           Disassemble around the PC (or a whole function).
  wasm-locals              Wasm locals of the selected frame.
  wasm-stack               Wasm operand stack of the selected frame.
  wasm-globals [-i INST]   Globals of an instance.
  wasm-memory [-i INST] [-m MEM] [ADDR [LEN]]
                           Show a memory's size, or hexdump its contents.
  wasm-table [-i INST] [-t TABLE] [START [COUNT]]
                           Show a table's size and elements.
  wasm-instances           List all instances in the store.
  wasm-modules             List all modules in the store.
  wasm-functions [-i INST | -m MODULE] [TEXT]
                           List functions of a module.
Other:
  help                     Show this help.
  quit (q), detach         Detach (the program runs to completion).
";

/// Result of executing a command.
pub enum Flow {
    Continue,
    Quit,
}

fn basename(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

fn compact_opts() -> FormatOptions {
    FormatOptions {
        max_depth: 2,
        max_elems: 4,
        max_string: 32,
        ..Default::default()
    }
}

fn truncate(mut s: String, n: usize) -> String {
    if s.chars().count() > n {
        s = s.chars().take(n).collect::<String>() + "...";
    }
    s
}

fn parse_u64(s: &str) -> Result<u64> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Ok(u64::from_str_radix(h, 16)?)
    } else {
        Ok(s.parse()?)
    }
}

/// Demangle `$`-prefixed mangled symbol names in a line of WAT.
fn demangle_wat(line: &str) -> String {
    line.split(' ')
        .map(|tok| match tok.strip_prefix('$') {
            Some(sym) if sym.starts_with("_Z") || sym.starts_with("_R") => {
                format!("<{}>", demangle(sym))
            }
            _ => tok.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
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
        self.conn.print(&out);
    }

    pub fn prompt(&mut self) {
        self.conn.print("(wasmtime-debug) ");
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
                self.conn.print(HELP);
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
            "interrupt" => {
                self.conn.println("The program is not running.");
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
            self.conn.println(&format!("{e}"));
        }
        if self.detached {
            return Flow::Quit;
        }
        Flow::Continue
    }

    // -----------------------------------------------------------------
    // Helpers.

    fn require_stopped(&self) -> Result<()> {
        match self.state {
            RunState::Stopped if !self.vframes.is_empty() => Ok(()),
            RunState::NotStarted => bail!("The program has not been started."),
            RunState::Exited => bail!("The program is not being run."),
            _ => bail!("No stack."),
        }
    }

    /// The instance to use for instance-scoped commands by default.
    fn default_inst(&self) -> Result<usize> {
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
    fn default_module(&self) -> Option<usize> {
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

    fn source_lines(&mut self, path: &str) -> Option<&Vec<String>> {
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
            Some(t) => self.conn.println(&format!("{line}\t{t}")),
            None => self
                .conn
                .println(&format!("{line}\tin {path} (source unavailable)")),
        }
    }

    /// A one-line description of logical frame `vi`: `name (args) at
    /// file:line`.
    fn frame_desc(&self, vi: usize) -> String {
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

    fn frame_args(&self, vi: usize) -> String {
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
    fn wasm_frame_args(&self, vi: usize) -> String {
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
                    .unwrap_or_else(|| format!("${i}"));
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
            self.conn.println(&format!("#{vi}  {desc}"));
        } else {
            self.conn.println(&desc);
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
            Some(t) => self.conn.println(&format!("=> {pc:#x}: {t}")),
            None => self.conn.println(&format!("=> {pc:#x}")),
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
            self.conn
                .println(&format!("Run till exit from #{}  {desc}", self.selected));
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
        let _ = self.conn.flush().await;
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
        }?;
        self.report(
            outcome,
            before,
            finishing,
            matches!(how, Resume::StepI | Resume::NextI),
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
                self.conn.println("[Program exited]");
                return;
            }
            Outcome::Stopped(r) => r,
        };
        if self.vframes.is_empty() {
            self.conn.println("Stopped (no Wasm frames on the stack).");
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
                    self.conn
                        .println(&format!("\nTemporary breakpoint {id}, {desc}"));
                } else {
                    self.conn.println(&format!("\nBreakpoint {id}, {desc}"));
                }
                self.print_current_line(0);
            }
            StopReason::Breakpoint(None) => {
                self.conn
                    .println(&format!("\nStopped at {}", self.frame_desc(0)));
                self.print_current_line(0);
            }
            StopReason::Interrupted => {
                self.conn
                    .println(&format!("\nProgram interrupted.\n{}", self.frame_desc(0)));
                self.print_current_line(0);
            }
            StopReason::Trap => {
                self.conn.println(&format!(
                    "\nProgram trapped (Wasm trap); continuing will terminate it.\n{}",
                    self.frame_desc(0)
                ));
                self.print_current_line(0);
            }
            StopReason::Exception => {
                self.conn
                    .println(&format!("\nWasm exception thrown.\n{}", self.frame_desc(0)));
                self.print_current_line(0);
            }
            StopReason::Step => {
                if !same_frame {
                    self.conn.println(&self.frame_desc(0));
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

    /// After `finish`, show the callee's return value(s), which are on
    /// top of the caller's operand stack.
    fn print_return_value(&mut self, callee_module: usize, callee_pc: u32) {
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
                self.conn
                    .println(&format!("Value returned is ${n} = {text}"));
            }
            None => {
                let vals: Vec<String> = results
                    .iter()
                    .map(|r| wasmval::format(self.d, &WVal::from_api(r), false, 1))
                    .collect();
                self.conn
                    .println(&format!("Wasm results: {}", vals.join(", ")));
            }
        }
    }

    // -----------------------------------------------------------------
    // Breakpoints.

    /// Resolve a location spec into (module, pc) pairs.
    fn resolve_location(&self, spec: &str) -> Result<Vec<(usize, u32)>> {
        let spec = spec.trim();
        if spec.is_empty() {
            bail!("Argument required (location).");
        }
        // `*PC` or `*MODULE:PC`.
        if let Some(rest) = spec.strip_prefix('*') {
            let (module, pc) = match rest.split_once(':') {
                Some((m, pc)) => (usize::try_from(parse_u64(m)?)?, parse_u64(pc)?),
                None => (
                    self.default_module()
                        .or_else(|| self.vframes.first().map(|v| self.phys[v.phys].module))
                        .ok_or_else(|| anyhow!("No default module."))?,
                    parse_u64(rest)?,
                ),
            };
            if module >= self.modules.len() {
                bail!("No module {module}.");
            }
            return Ok(vec![(module, u32::try_from(pc)?)]);
        }
        // `FILE:LINE` or `LINE`.
        let (file, line) = match spec.rsplit_once(':') {
            Some((f, l)) if l.chars().all(|c| c.is_ascii_digit()) && !l.is_empty() => {
                (Some(f.to_string()), Some(l.parse::<u32>()?))
            }
            _ if spec.chars().all(|c| c.is_ascii_digit()) => (None, Some(spec.parse::<u32>()?)),
            _ => (None, None),
        };
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

    fn describe_pc(&self, module: usize, pc: u32) -> String {
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
            self.conn.println(&format!("{kind} {id} at {}", descs[0]));
        } else {
            self.conn
                .println(&format!("{kind} {id} at {} locations:", descs.len()));
            for d in descs {
                self.conn.println(&format!("  {d}"));
            }
        }
        Ok(())
    }

    fn cmd_delete(&mut self, args: &str) -> Result<()> {
        let args = args.strip_prefix("breakpoints").unwrap_or(args).trim();
        if args.is_empty() {
            let ids: Vec<u32> = self.breakpoints.iter().map(|b| b.id).collect();
            for id in ids {
                self.delete_breakpoint(id);
            }
            self.conn.println("Deleted all breakpoints.");
            return Ok(());
        }
        for a in args.split_whitespace() {
            let id: u32 = a
                .parse()
                .map_err(|_| anyhow!("Bad breakpoint number '{a}'"))?;
            if !self.delete_breakpoint(id) {
                self.conn.println(&format!("No breakpoint number {id}."));
            }
        }
        Ok(())
    }

    fn info_breakpoints(&mut self) {
        if self.breakpoints.is_empty() {
            self.conn.println("No breakpoints.");
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
            self.conn.println(&format!("{n}\t{text}"));
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
    fn eval(&mut self, expr: &str) -> Result<HistEntry> {
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
                        let v = ctx.snapshot(&v).unwrap_or(v);
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
            _ => self.eval_source(&p),
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
            HistoryRef::Absolute(i) => bail!("History has not yet reached ${i}."),
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
        if let Some(v) = self.vframes.get(self.selected) {
            let p = &self.phys[v.phys];
            let info = &self.modules[p.module].info;
            if let Some(func) = p.func {
                let index = info
                    .local_names
                    .iter()
                    .find(|((f, _), n)| *f == func && n.as_str() == name)
                    .map(|((_, l), _)| *l);
                if let Some(i) = index {
                    let locals = p.frame.get_locals(self.d).map_err(|e| anyhow!("{e:?}"))?;
                    let v = locals
                        .get(usize::try_from(i)?)
                        .ok_or_else(|| anyhow!("No local {i} (frame has {}).", locals.len()))?;
                    return Ok(Some(WVal::from_api(v)));
                }
            }
        }
        let Ok(ii) = self.default_inst() else {
            return Ok(None);
        };
        let info = &self.modules[self.instances[ii].module].info;
        if let Some(t) = info
            .table_names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(t, _)| *t)
        {
            let table = self.instances[ii]
                .handle
                .get_table(self.d, t)
                .map_err(|_| anyhow!("No table {t}."))?;
            return Ok(Some(WVal::Table(table)));
        }
        let Some(g) = info
            .global_names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(g, _)| *g)
        else {
            return Ok(None);
        };
        let global = self.instances[ii]
            .handle
            .get_global(self.d, g)
            .map_err(|_| anyhow!("No global {g}."))?;
        let v = global.get(self.d).map_err(|e| anyhow!("{e:?}"))?;
        Ok(Some(WVal::from_api(&v)))
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

    fn eval_source(&mut self, p: &Path) -> Result<HistEntry> {
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
        let v = ctx.snapshot(&v).unwrap_or(v);
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
        self.conn.println(&format!("${n} = {text}"));
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
        self.conn.println(&format!("type = {text}"));
        Ok(())
    }

    fn cmd_examine(&mut self, args: &str, fmt: &str) -> Result<()> {
        // Parse N, F, U from e.g. `16xb`.
        let digits: String = fmt.chars().take_while(|c| c.is_ascii_digit()).collect();
        let count: u64 = if digits.is_empty() {
            1
        } else {
            digits.parse()?
        };
        let letters = &fmt[digits.len()..];
        let mut format = 'x';
        let mut unit = 0u64;
        for c in letters.chars() {
            match c {
                'x' | 'd' | 'u' | 'c' | 's' => format = c,
                'b' => unit = 1,
                'h' => unit = 2,
                'w' => unit = 4,
                'g' => unit = 8,
                _ => bail!("Invalid format letter '{c}'."),
            }
        }
        if unit == 0 {
            unit = if format == 'c' { 1 } else { 4 };
        }
        // Determine the address.
        let addr = if let Ok(a) = parse_u64(args) {
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
        };
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
                let _ = writeln!(
                    out,
                    "{:#x}: {}",
                    a - bytes.len() as u64 - 1,
                    wasmtime_debug_info::format::quote_bytes(&bytes, false)
                );
            }
            self.conn.print(&out);
            return Ok(());
        }
        let bytes = target.read_memory(0, addr, count * unit)?;
        let per_line = (16 / unit).max(1);
        for (i, chunk) in bytes.chunks(usize::try_from(unit)?).enumerate() {
            let i = i as u64;
            if i % per_line == 0 {
                if i > 0 {
                    out.push('\n');
                }
                let _ = write!(out, "{:#010x}:", addr + i * unit);
            }
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            let v = u64::from_le_bytes(b);
            let w = usize::try_from(unit * 2)?;
            let _ = match format {
                'x' => write!(out, " 0x{v:0w$x}"),
                'u' => write!(out, " {v}"),
                'd' => {
                    let shift = 64 - unit * 8;
                    write!(out, " {}", ((v << shift) as i64) >> shift)
                }
                'c' => write!(out, " {:?}", (v as u8) as char),
                _ => unreachable!(),
            };
        }
        out.push('\n');
        self.conn.print(&out);
        Ok(())
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
            self.conn.println("No symbol table info available.");
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.println(&format!(
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
            self.conn.println(&format!(
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
        let mut out = String::new();
        for (i, chunk) in bytes.chunks(16).enumerate() {
            let _ = write!(out, "{:#010x}:", addr + 16 * i as u64);
            for j in 0..16 {
                match chunk.get(j) {
                    Some(b) => {
                        let _ = write!(out, " {b:02x}");
                    }
                    None => out.push_str("   "),
                }
            }
            out.push_str("  |");
            for b in chunk {
                out.push(if (0x20..0x7f).contains(b) {
                    *b as char
                } else {
                    '.'
                });
            }
            out.push_str("|\n");
        }
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
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
        self.conn.print(&out);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Resume {
    Continue,
    Step,
    Next,
    StepI,
    NextI,
    Finish,
}
