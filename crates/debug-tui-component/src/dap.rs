//! A Debug Adapter Protocol (DAP) server front-end.
//!
//! This exposes the same debugger session as the text interface through
//! DAP requests and events:
//!
//! - Threads: a single thread (the debuggee store).
//! - Stack frames: source-level frames, including inlined frames, with
//!   instruction-pointer references for disassembly.
//! - Scopes: source-level arguments and locals, plus the Wasm machine
//!   state: Wasm locals, the operand stack, and globals. Values expand
//!   into trees mirroring the pretty-printer's view (struct fields,
//!   sequence elements, map entries, enum payloads), including Wasm GC
//!   objects.
//! - Breakpoints: source lines, functions, and instructions; data
//!   breakpoints (watchpoints) on values in linear memory, which stop on
//!   writes.
//! - Execution: continue, pause, next/step-in (by line or, with
//!   `granularity: "instruction"`, by Wasm instruction), step-out. When
//!   debugging a replay (`wasmtime replay`), also `stepBack` and
//!   `reverseContinue`.
//! - Memory: `readMemory` on linear memory, and `disassemble`.
//! - `evaluate`: access paths for watches and hovers; in the `repl`
//!   context, any command of the text interface (e.g. `wasm-table`,
//!   `x/16xb &buf`, `info functions`) can be run, so every feature is
//!   reachable from a DAP client's debug console.
//!
//! Memory references are hexadecimal linear-memory addresses (in memory 0
//! of the selected frame's instance). Instruction references encode a
//! module index and bytecode offset (see
//! [`wasmtime_debug_dap::code_address`]).

use crate::api;
use crate::commands::WatchTarget;
use crate::session::{FrameTarget, Front, HistEntry, Outcome, RunState, Session, StopReason};
use crate::wasmval::{self, WVal};
use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::collections::HashMap;
use wasmtime_debug_dap::types::{
    Breakpoint, DisassembledInstruction, Module, Scope, Source, StackFrame, StoppedEvent, Variable,
};
use wasmtime_debug_dap::{
    Decoder, base64_encode, code_address, encode, memory_reference, parse_memory_reference,
    split_code_address,
};
use wasmtime_debug_info::children::{children, has_children, indexed_len};
use wasmtime_debug_info::cli::{basename, demangle_wat};
use wasmtime_debug_info::format::{FormatOptions, format_value};
use wasmtime_debug_info::value::{Ctx, Value as DValue};
use wstd::io::{AsyncRead, AsyncWrite};
use wstd::net::TcpStream;

/// The DAP connection: message framing over a TCP stream.
pub struct Transport {
    stream: TcpStream,
    decoder: Decoder,
    seq: i64,
}

impl Transport {
    pub fn new(stream: TcpStream) -> Self {
        Transport {
            stream,
            decoder: Decoder::new(),
            seq: 0,
        }
    }

    /// Read the next message; `None` at end of stream. Cancel-safe.
    pub async fn read_message(&mut self) -> Result<Option<Value>> {
        loop {
            if let Some(m) = self.decoder.next()? {
                return Ok(Some(m));
            }
            let mut buf = [0u8; 4096];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                return Ok(None);
            }
            self.decoder.push(&buf[..n]);
        }
    }

    async fn send(&mut self, mut msg: Value) -> Result<()> {
        self.seq += 1;
        msg["seq"] = json!(self.seq);
        self.stream.write_all(&encode(&msg)).await?;
        self.stream.flush().await?;
        Ok(())
    }

    pub async fn respond(&mut self, req: &Value, result: Result<Value, String>) -> Result<()> {
        let seq = req["seq"].as_i64().unwrap_or(0);
        let command = req["command"].as_str().unwrap_or("");
        self.send(wasmtime_debug_dap::response(0, seq, command, result))
            .await
    }

    pub async fn event(&mut self, name: &str, body: Value) -> Result<()> {
        self.send(wasmtime_debug_dap::event(0, name, body)).await
    }
}

/// What a DAP `variablesReference` refers to.
enum VarRef {
    /// A scope of a frame.
    Scope { frame: usize, kind: ScopeKind },
    /// A source-level value (with its module, instance, and access path).
    Src {
        module: usize,
        inst: usize,
        value: DValue,
        path: Option<String>,
    },
    /// A Wasm-level value (GC object or table).
    Wasm { value: WVal, path: Option<String> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Args,
    Locals,
    WasmLocals,
    WasmStack,
    WasmGlobals,
}

/// DAP-specific state.
#[derive(Default)]
struct DapState {
    refs: Vec<VarRef>,
    /// Breakpoint ids set via `setBreakpoints`, by source path.
    source_bps: HashMap<String, Vec<u32>>,
    function_bps: Vec<u32>,
    instruction_bps: Vec<u32>,
    /// Watchpoint ids set via `setDataBreakpoints`.
    data_bps: Vec<u32>,
    /// Watch targets handed out by `dataBreakpointInfo`, indexed by
    /// `dataId`.
    data_targets: Vec<(String, WatchTarget)>,
    stop_on_entry: bool,
}

impl DapState {
    fn alloc(&mut self, r: VarRef) -> i64 {
        self.refs.push(r);
        self.refs.len() as i64
    }
}

/// How to resume.
#[derive(Clone, Copy)]
enum Run {
    Continue,
    Next,
    Step,
    NextI,
    StepI,
    Finish,
    ReverseContinue,
    ReverseNext,
    ReverseStep,
    ReverseNextI,
    ReverseStepI,
    ReverseFinish,
}

impl Run {
    fn is_reverse(self) -> bool {
        matches!(
            self,
            Run::ReverseContinue
                | Run::ReverseNext
                | Run::ReverseStep
                | Run::ReverseNextI
                | Run::ReverseStepI
                | Run::ReverseFinish
        )
    }
}

fn args(req: &Value) -> &Value {
    &req["arguments"]
}

impl Session<'_> {
    fn transport(&mut self) -> &mut Transport {
        match &mut self.front {
            Front::Dap(t) => t,
            Front::Tui(_) => unreachable!("DAP request in text mode"),
        }
    }

    /// Handle a request that arrives while the debuggee is running.
    pub(crate) async fn dap_request_while_running(&mut self, req: Value) -> Result<()> {
        let result = match req["command"].as_str().unwrap_or("") {
            "pause" => {
                self.d.interrupt();
                Ok(json!({}))
            }
            "threads" => Ok(threads()),
            "disconnect" | "terminate" => {
                self.detached = true;
                self.remove_all_breakpoints();
                Ok(json!({}))
            }
            _ => Err("The program is running.".to_string()),
        };
        self.transport().respond(&req, result).await
    }
}

fn threads() -> Value {
    json!({ "threads": [{ "id": 1, "name": "main" }] })
}

fn capabilities(reversible: bool) -> Value {
    json!({
        "supportsStepBack": reversible,
        "supportsDataBreakpoints": true,
        "supportsDataBreakpointBytes": true,
        "supportsConfigurationDoneRequest": true,
        "supportsFunctionBreakpoints": true,
        "supportsInstructionBreakpoints": true,
        "supportsSteppingGranularity": true,
        "supportsReadMemoryRequest": true,
        "supportsDisassembleRequest": true,
        "supportsEvaluateForHovers": true,
        "supportsValueFormattingOptions": true,
        "supportsModulesRequest": true,
        "supportsTerminateRequest": true,
        "supportTerminateDebuggee": false,
        "supportsSetVariable": false,
        "supportsDelayedStackTraceLoading": true,
        "exceptionBreakpointFilters": [],
    })
}

/// Commands of the text interface that resume execution; in the REPL
/// they are mapped to the corresponding DAP execution flow.
fn repl_run_command(word: &str) -> Option<Run> {
    Some(match word {
        "continue" | "c" | "run" | "r" => Run::Continue,
        "next" | "n" => Run::Next,
        "step" | "s" => Run::Step,
        "nexti" | "ni" => Run::NextI,
        "stepi" | "si" => Run::StepI,
        "finish" | "fin" => Run::Finish,
        "reverse-continue" | "rc" => Run::ReverseContinue,
        "reverse-next" | "rn" => Run::ReverseNext,
        "reverse-step" | "rs" => Run::ReverseStep,
        "reverse-nexti" | "rni" => Run::ReverseNextI,
        "reverse-stepi" | "rsi" => Run::ReverseStepI,
        "reverse-finish" => Run::ReverseFinish,
        _ => return None,
    })
}

/// Serve DAP until the client disconnects.
pub async fn run(s: &mut Session<'_>) -> Result<()> {
    let mut st = DapState::default();
    s.refresh_entities();
    loop {
        let Some(req) = s.transport().read_message().await? else {
            break;
        };
        if req["type"].as_str() != Some("request") {
            continue;
        }
        let command = req["command"].as_str().unwrap_or("").to_string();
        match command.as_str() {
            "initialize" => {
                let caps = capabilities(s.reversible);
                s.transport().respond(&req, Ok(caps)).await?;
                s.transport().event("initialized", json!({})).await?;
            }
            "launch" | "attach" => {
                st.stop_on_entry = args(&req)["stopOnEntry"].as_bool().unwrap_or(false);
                s.transport().respond(&req, Ok(json!({}))).await?;
            }
            "configurationDone" => {
                s.transport().respond(&req, Ok(json!({}))).await?;
                if st.stop_on_entry {
                    let ev = StoppedEvent {
                        reason: "entry".into(),
                        description: Some("The program has not started.".into()),
                        thread_id: Some(1),
                        all_threads_stopped: true,
                        ..Default::default()
                    };
                    s.transport()
                        .event("stopped", serde_json::to_value(ev)?)
                        .await?;
                } else {
                    run_and_report(s, &mut st, Run::Continue).await?;
                }
            }
            "continue" | "next" | "stepIn" | "stepOut" | "stepBack" | "reverseContinue" => {
                let instr = args(&req)["granularity"].as_str() == Some("instruction");
                let how = match (command.as_str(), instr) {
                    ("continue", _) => Run::Continue,
                    ("next", false) => Run::Next,
                    ("next", true) => Run::NextI,
                    ("stepIn", false) => Run::Step,
                    ("stepIn", true) => Run::StepI,
                    ("stepBack", false) => Run::ReverseNext,
                    ("stepBack", true) => Run::ReverseNextI,
                    ("reverseContinue", _) => Run::ReverseContinue,
                    _ => Run::Finish,
                };
                if how.is_reverse() && !s.reversible {
                    s.transport()
                        .respond(&req, Err(crate::session::NOT_REVERSIBLE.into()))
                        .await?;
                    continue;
                }
                if s.state == RunState::Exited {
                    s.transport()
                        .respond(&req, Err("The program is not being run.".into()))
                        .await?;
                    continue;
                }
                let body = if command == "continue" {
                    json!({ "allThreadsContinued": true })
                } else {
                    json!({})
                };
                s.transport().respond(&req, Ok(body)).await?;
                run_and_report(s, &mut st, how).await?;
            }
            "pause" => {
                // Already stopped.
                s.transport().respond(&req, Ok(json!({}))).await?;
            }
            "disconnect" | "terminate" => {
                s.transport().respond(&req, Ok(json!({}))).await?;
                break;
            }
            "evaluate"
                if args(&req)["context"].as_str() == Some("repl")
                    && repl_run_command(
                        args(&req)["expression"]
                            .as_str()
                            .unwrap_or("")
                            .split_whitespace()
                            .next()
                            .unwrap_or(""),
                    )
                    .is_some() =>
            {
                let expr = args(&req)["expression"].as_str().unwrap_or("").to_string();
                let how = repl_run_command(expr.split_whitespace().next().unwrap()).unwrap();
                if how.is_reverse() && !s.reversible {
                    s.transport()
                        .respond(&req, Err(crate::session::NOT_REVERSIBLE.into()))
                        .await?;
                    continue;
                }
                s.transport()
                    .respond(&req, Ok(json!({ "result": "", "variablesReference": 0 })))
                    .await?;
                run_and_report(s, &mut st, how).await?;
            }
            _ => {
                let result = handle(s, &mut st, &command, args(&req)).await;
                let result = result.map_err(|e| format!("{e:#}"));
                s.transport().respond(&req, result).await?;
            }
        }
        if s.detached {
            break;
        }
    }
    Ok(())
}

/// Resume execution and report the outcome with events.
async fn run_and_report(s: &mut Session<'_>, st: &mut DapState, how: Run) -> Result<()> {
    st.refs.clear();
    if s.state == RunState::NotStarted && matches!(how, Run::Continue | Run::StepI | Run::NextI) {
        s.state = RunState::Stopped;
    }
    let finishing = match how {
        Run::Finish => s.vframes.get(s.selected).map(|v| {
            let p = &s.phys[v.phys];
            (p.module, p.lookup_pc, v.inlined)
        }),
        _ => None,
    };
    let outcome = match how {
        Run::Continue => s.cont().await,
        Run::Next => s.step_line(true).await,
        Run::Step => s.step_line(false).await,
        Run::NextI => s.nexti().await,
        Run::StepI => s.stepi().await,
        Run::Finish => s.finish().await,
        Run::ReverseContinue => s.reverse_cont().await,
        Run::ReverseNext => s.reverse_step_line(true).await,
        Run::ReverseStep => s.reverse_step_line(false).await,
        Run::ReverseNextI => s.reverse_nexti().await,
        Run::ReverseStepI => s.reverse_stepi().await,
        Run::ReverseFinish => s.reverse_finish().await,
    };
    let outcome = match outcome {
        Ok(o) => o,
        Err(e) => {
            // Report the failure, and leave the program stopped.
            s.transport()
                .event(
                    "output",
                    json!({ "category": "console", "output": format!("{e:#}\n") }),
                )
                .await?;
            Outcome::Stopped(StopReason::Step)
        }
    };
    if s.detached {
        return Ok(());
    }
    match outcome {
        Outcome::Exited => {
            s.transport()
                .event("exited", json!({ "exitCode": 0 }))
                .await?;
            s.transport().event("terminated", json!({})).await?;
        }
        Outcome::Stopped(reason) => {
            if let (StopReason::Finished, Some((module, pc, false))) = (&reason, finishing) {
                s.out.clear();
                s.print_return_value(module, pc);
                let text = std::mem::take(&mut s.out);
                if !text.is_empty() {
                    s.transport()
                        .event("output", json!({ "category": "console", "output": text }))
                        .await?;
                }
            }
            if let StopReason::Breakpoint(Some(id)) = reason {
                if let Some(b) = s.breakpoints.iter_mut().find(|b| b.id == id) {
                    b.hits += 1;
                }
            }
            // Describe watchpoint and history-boundary stops in the
            // console too.
            let note = match &reason {
                StopReason::Watchpoint(r) => s.watch_report_text(r),
                StopReason::ReplayBegin => {
                    "No more reverse-execution history: at the beginning of the recording.\n".into()
                }
                StopReason::ReplayEnd => "No more execution history: at the end of the \
                                          recording; continuing will complete it.\n"
                    .into(),
                _ => String::new(),
            };
            let note = note.trim_start_matches('\n');
            if !note.is_empty() {
                s.transport()
                    .event("output", json!({ "category": "console", "output": note }))
                    .await?;
            }
            let (reason, description, hit) = match reason {
                StopReason::Breakpoint(Some(id)) => (
                    "breakpoint",
                    format!("Breakpoint {id}"),
                    vec![i64::from(id)],
                ),
                StopReason::Breakpoint(None) => ("breakpoint", "Breakpoint".into(), vec![]),
                StopReason::Step | StopReason::Finished => ("step", "Step".into(), vec![]),
                StopReason::Interrupted => ("pause", "Program interrupted".into(), vec![]),
                StopReason::Trap => (
                    "exception",
                    "Program trapped (Wasm trap); continuing will terminate it".into(),
                    vec![],
                ),
                StopReason::Exception => ("exception", "Wasm exception thrown".into(), vec![]),
                StopReason::Watchpoint(r) => {
                    let ids: Vec<String> = r.ids.iter().map(|i| i.to_string()).collect();
                    (
                        "data breakpoint",
                        format!("Watchpoint {}", ids.join(", ")),
                        r.ids.iter().map(|&i| i64::from(i)).collect(),
                    )
                }
                StopReason::ReplayBegin => (
                    "beginning of recording",
                    "At the beginning of the recording".into(),
                    vec![],
                ),
                StopReason::ReplayEnd => (
                    "end of recording",
                    "At the end of the recording".into(),
                    vec![],
                ),
            };
            let ev = StoppedEvent {
                reason: reason.into(),
                description: Some(description),
                thread_id: Some(1),
                all_threads_stopped: true,
                hit_breakpoint_ids: hit,
                ..Default::default()
            };
            s.transport()
                .event("stopped", serde_json::to_value(ev)?)
                .await?;
        }
    }
    Ok(())
}

/// Handle a request that does not resume execution.
async fn handle(s: &mut Session<'_>, st: &mut DapState, command: &str, a: &Value) -> Result<Value> {
    match command {
        "threads" => Ok(threads()),
        "stackTrace" => stack_trace(s, a),
        "scopes" => scopes(s, st, a),
        "variables" => variables(s, st, a),
        "evaluate" => evaluate(s, st, a).await,
        "setBreakpoints" => set_breakpoints(s, st, a),
        "setFunctionBreakpoints" => set_function_breakpoints(s, st, a),
        "setInstructionBreakpoints" => set_instruction_breakpoints(s, st, a),
        "setExceptionBreakpoints" => Ok(json!({})),
        "dataBreakpointInfo" => data_breakpoint_info(s, st, a),
        "setDataBreakpoints" => set_data_breakpoints(s, st, a),
        "readMemory" => read_memory(s, a),
        "disassemble" => disassemble(s, a),
        "source" => source(s, a),
        "modules" => modules(s),
        "loadedSources" => Ok(json!({ "sources": [] })),
        _ => Err(anyhow!("unsupported request `{command}`")),
    }
}

fn frame_index(s: &Session<'_>, a: &Value) -> Result<usize> {
    let id = a["frameId"].as_i64().unwrap_or(1);
    let vi = usize::try_from(id - 1).map_err(|_| anyhow!("invalid frame id"))?;
    if vi >= s.vframes.len() {
        return Err(anyhow!("invalid frame id {id}"));
    }
    Ok(vi)
}

fn source_of(s: &Session<'_>, module: usize, file: u32) -> Option<Source> {
    let dbg = s.modules[module].info.debug.as_ref()?;
    let path = dbg.file_name(file).to_string();
    Some(Source {
        name: Some(basename(&path).to_string()),
        path: Some(path),
        source_reference: None,
    })
}

fn stack_trace(s: &mut Session<'_>, a: &Value) -> Result<Value> {
    let start = a["startFrame"].as_u64().unwrap_or(0) as usize;
    let levels = match a["levels"].as_u64().unwrap_or(0) {
        0 => usize::MAX,
        n => n as usize,
    };
    let with_params = a["format"]["parameters"].as_bool().unwrap_or(false);
    let mut frames = vec![];
    for vi in start..s.vframes.len().min(start.saturating_add(levels)) {
        let v = &s.vframes[vi];
        let p = &s.phys[v.phys];
        let name = if with_params {
            format!("{} ({})", v.name, s.frame_args(vi))
        } else {
            v.name.clone()
        };
        let (source, line, column) = match &v.loc {
            Some(loc) => (
                source_of(s, p.module, loc.file),
                i64::from(loc.line),
                i64::from(loc.column.max(1)),
            ),
            None => (None, 0, 0),
        };
        frames.push(StackFrame {
            id: vi as i64 + 1,
            name,
            presentation_hint: source.is_none().then(|| "subtle".to_string()),
            source,
            line,
            column,
            instruction_pointer_reference: Some(memory_reference(code_address(p.module, p.pc))),
            module_id: Some(json!(p.module)),
            physical_frame: Some(v.phys as i64),
        });
    }
    Ok(json!({ "stackFrames": frames, "totalFrames": s.vframes.len() }))
}

fn scopes(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    let vi = frame_index(s, a)?;
    let mut scopes = vec![];
    let mut push = |st: &mut DapState, name: &str, hint: Option<&str>, kind| {
        scopes.push(Scope {
            name: name.into(),
            presentation_hint: hint.map(|h| h.into()),
            variables_reference: st.alloc(VarRef::Scope { frame: vi, kind }),
            expensive: false,
            ..Default::default()
        });
    };
    if s.vframes[vi].scope.is_some() {
        push(st, "Arguments", Some("arguments"), ScopeKind::Args);
        push(st, "Locals", Some("locals"), ScopeKind::Locals);
    }
    push(st, "Wasm Locals", Some("registers"), ScopeKind::WasmLocals);
    push(
        st,
        "Wasm Operand Stack",
        Some("registers"),
        ScopeKind::WasmStack,
    );
    push(st, "Wasm Globals", None, ScopeKind::WasmGlobals);
    Ok(json!({ "scopes": scopes }))
}

fn format_opts(a: &Value) -> FormatOptions {
    FormatOptions {
        hex: a["format"]["hex"].as_bool().unwrap_or(false),
        ..Default::default()
    }
}

/// Build a DAP variable for a source-level value.
fn src_variable(
    s: &Session<'_>,
    st: &mut DapState,
    name: String,
    eval_name: Option<String>,
    module: usize,
    inst: usize,
    value: DValue,
    opts: &FormatOptions,
) -> Variable {
    let dbg = s.modules[module].info.debug.as_ref().unwrap();
    let target = FrameTarget::new(s.d, None, &s.instances[inst].handle);
    let ctx = Ctx::new(dbg, &target);
    let text = format_value(&ctx, &value, opts);
    let ty = dbg.type_name(value.ty);
    // For pointers, the memory reference is the address pointed to;
    // otherwise, the value's own address.
    let is_ptr = matches!(
        dbg.ty(dbg.strip(value.ty)).kind,
        wasmtime_debug_info::types::TypeKind::Pointer { .. }
    );
    let memory_reference = if is_ptr {
        ctx.read_uint(&value).ok().map(memory_reference)
    } else {
        value.address().map(memory_reference)
    };
    let expandable = has_children(&ctx, &value, opts);
    let indexed = if expandable {
        indexed_len(&ctx, &value, opts)
    } else {
        None
    };
    let variables_reference = if expandable {
        st.alloc(VarRef::Src {
            module,
            inst,
            value,
            path: eval_name.clone(),
        })
    } else {
        0
    };
    Variable {
        name,
        value: text,
        ty: Some(ty),
        variables_reference,
        evaluate_name: eval_name,
        memory_reference,
        indexed_variables: indexed.map(|n| n as i64),
    }
}

/// Whether a Wasm value can be expanded.
fn wasm_expandable(d: &api::Debuggee, v: &WVal) -> bool {
    match v {
        WVal::Any(Some(a)) => matches!(
            a.kind(d),
            Ok(api::AnyrefKind::StructRef(n) | api::AnyrefKind::ArrayRef(n)) if n > 0
        ),
        WVal::Exn(Some(_)) | WVal::Table(_) => true,
        _ => false,
    }
}

/// Build a DAP variable for a Wasm-level value.
fn wasm_variable(
    s: &Session<'_>,
    st: &mut DapState,
    name: String,
    eval_name: Option<String>,
    value: WVal,
    hex: bool,
) -> Variable {
    let text = wasmval::format(s.d, &value, hex, 1);
    let variables_reference = if wasm_expandable(s.d, &value) {
        st.alloc(VarRef::Wasm {
            value: value.duplicate(s.d),
            path: eval_name.clone(),
        })
    } else {
        0
    };
    Variable {
        name,
        value: text,
        variables_reference,
        evaluate_name: eval_name,
        ..Default::default()
    }
}

fn variables(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    let r = a["variablesReference"].as_i64().unwrap_or(0);
    let idx = usize::try_from(r - 1).map_err(|_| anyhow!("invalid variables reference"))?;
    if idx >= st.refs.len() {
        return Err(anyhow!("invalid variables reference {r}"));
    }
    let start = a["start"].as_u64().unwrap_or(0) as usize;
    let count = match a["count"].as_u64().unwrap_or(0) {
        0 => 1000,
        n => n as usize,
    };
    let opts = format_opts(a);
    let mut out = vec![];
    // Take the reference out temporarily to avoid borrowing `st`.
    let r = std::mem::replace(
        &mut st.refs[idx],
        VarRef::Scope {
            frame: 0,
            kind: ScopeKind::Args,
        },
    );
    let result = (|| -> Result<()> {
        match &r {
            VarRef::Scope { frame, kind } => {
                scope_variables(s, st, *frame, *kind, &opts, &mut out)?;
            }
            VarRef::Src {
                module,
                inst,
                value,
                path,
            } => {
                let dbg = s.modules[*module].info.debug.as_ref().unwrap();
                let target = FrameTarget::new(s.d, None, &s.instances[*inst].handle);
                let ctx = Ctx::new(dbg, &target);
                let kids = children(&ctx, value, &opts, start, count);
                drop(ctx);
                for k in kids {
                    let eval_name = path.as_deref().and_then(|p| k.eval_name(p));
                    out.push(src_variable(
                        s, st, k.name, eval_name, *module, *inst, k.value, &opts,
                    ));
                }
            }
            VarRef::Wasm { value, path } => {
                for (name, suffix, v) in wasm_children(s, value, start, count)? {
                    let eval_name = path.as_ref().map(|p| format!("{p}{suffix}"));
                    out.push(wasm_variable(s, st, name, eval_name, v, opts.hex));
                }
            }
        }
        Ok(())
    })();
    st.refs[idx] = r;
    result?;
    Ok(json!({ "variables": out }))
}

/// Children of a Wasm value: (name, path suffix, value).
fn wasm_children(
    s: &Session<'_>,
    v: &WVal,
    start: usize,
    count: usize,
) -> Result<Vec<(String, String, WVal)>> {
    let d = s.d;
    let err = |e: api::Error| anyhow!("{e:?}");
    let mut out = vec![];
    match v {
        WVal::Any(Some(a)) => match a.kind(d).map_err(err)? {
            api::AnyrefKind::StructRef(_) => {
                for (i, f) in a.struct_fields(d).map_err(err)?.iter().enumerate() {
                    if i >= start && out.len() < count {
                        out.push((i.to_string(), format!(".{i}"), WVal::from_api(f)));
                    }
                }
            }
            api::AnyrefKind::ArrayRef(n) => {
                let start = u32::try_from(start)?.min(n);
                let count = u32::try_from(count)?.min(n - start);
                for (i, e) in a
                    .array_elements(d, start, count)
                    .map_err(err)?
                    .iter()
                    .enumerate()
                {
                    let i = start as usize + i;
                    out.push((format!("[{i}]"), format!("[{i}]"), WVal::from_api(e)));
                }
            }
            _ => {}
        },
        WVal::Exn(Some(e)) => {
            for (i, f) in e.get_values(d).map_err(err)?.iter().enumerate() {
                if i >= start && out.len() < count {
                    out.push((i.to_string(), format!(".{i}"), WVal::from_api(f)));
                }
            }
        }
        WVal::Table(t) => {
            let len = t.len(d);
            for i in (start as u64)..len.min(start as u64 + count as u64) {
                let e = t.get_element(d, i).map_err(err)?;
                out.push((format!("[{i}]"), format!("[{i}]"), WVal::from_api(&e)));
            }
        }
        _ => {}
    }
    Ok(out)
}

fn scope_variables(
    s: &mut Session<'_>,
    st: &mut DapState,
    vi: usize,
    kind: ScopeKind,
    opts: &FormatOptions,
    out: &mut Vec<Variable>,
) -> Result<()> {
    let v = &s.vframes[vi];
    let phys = v.phys;
    let p = &s.phys[phys];
    let (module, inst) = (p.module, p.inst);
    match kind {
        ScopeKind::Args | ScopeKind::Locals => {
            let Some(scope) = v.scope.clone() else {
                return Ok(());
            };
            let addr = s.dwarf_addr(phys).unwrap_or(0);
            let frame = &s.phys[phys].frame;
            let target = FrameTarget::new(s.d, Some(frame), &s.instances[inst].handle);
            let fb = s.frame_base(phys, &target);
            let dbg = s.modules[module].info.debug.as_ref().unwrap();
            let ctx = Ctx::new(dbg, &target);
            let mut vals = vec![];
            for var in scope
                .vars
                .iter()
                .filter(|x| x.is_param == (kind == ScopeKind::Args))
            {
                // Snapshot values held in Wasm locals etc. so that they
                // remain valid without the frame.
                let val = ctx.variable(var, addr, fb);
                vals.push((var.name.clone(), val));
            }
            drop(ctx);
            for (name, val) in vals {
                match val {
                    Ok(value) => out.push(src_variable(
                        s,
                        st,
                        name.clone(),
                        Some(name),
                        module,
                        inst,
                        value,
                        opts,
                    )),
                    Err(e) => out.push(Variable {
                        evaluate_name: Some(name.clone()),
                        name,
                        value: format!("{e}"),
                        ..Default::default()
                    }),
                }
            }
        }
        ScopeKind::WasmLocals => {
            let names = p
                .func
                .map(|f| s.modules[module].info.func_locals(f))
                .unwrap_or_default();
            let locals = p.frame.get_locals(s.d).map_err(|e| anyhow!("{e:?}"))?;
            let vals: Vec<WVal> = locals.iter().map(WVal::from_api).collect();
            for (i, val) in vals.into_iter().enumerate() {
                let label = names
                    .get(i)
                    .and_then(|(n, _)| n.clone())
                    .map(|n| format!("local {i} ({n})"))
                    .unwrap_or_else(|| format!("local {i}"));
                out.push(wasm_variable(
                    s,
                    st,
                    label,
                    Some(format!("%local{i}")),
                    val,
                    opts.hex,
                ));
            }
        }
        ScopeKind::WasmStack => {
            let stack = p.frame.get_stack(s.d).map_err(|e| anyhow!("{e:?}"))?;
            let vals: Vec<WVal> = stack.iter().map(WVal::from_api).collect();
            for (i, val) in vals.into_iter().enumerate() {
                out.push(wasm_variable(
                    s,
                    st,
                    format!("stack[{i}]"),
                    Some(format!("%stack{i}")),
                    val,
                    opts.hex,
                ));
            }
        }
        ScopeKind::WasmGlobals => {
            let n = s.modules[module].info.num_globals();
            for g in 0..n {
                let h = &s.instances[inst].handle;
                let val = match h.get_global(s.d, g).and_then(|gl| gl.get(s.d)) {
                    Ok(v) => WVal::from_api(&v),
                    Err(_) => continue,
                };
                let label = match s.modules[module].info.global_names.get(&g) {
                    Some(n) => format!("global {g} ({n})"),
                    None => format!("global {g}"),
                };
                out.push(wasm_variable(
                    s,
                    st,
                    label,
                    Some(format!("%global{g}")),
                    val,
                    opts.hex,
                ));
            }
        }
    }
    Ok(())
}

async fn evaluate(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    let expr = a["expression"].as_str().unwrap_or("").trim().to_string();
    let context = a["context"].as_str().unwrap_or("repl");
    let saved = s.selected;
    if a.get("frameId").is_some_and(|f| !f.is_null()) && !s.vframes.is_empty() {
        s.selected = frame_index(s, a)?;
    }
    let result = evaluate_inner(s, st, &expr, context, a).await;
    if context != "repl" {
        s.selected = saved;
    }
    result
}

async fn evaluate_inner(
    s: &mut Session<'_>,
    st: &mut DapState,
    expr: &str,
    context: &str,
    a: &Value,
) -> Result<Value> {
    let word = expr.split_whitespace().next().unwrap_or("");
    let word = word.split('/').next().unwrap_or("");
    if context == "repl" && crate::commands::is_command(word) {
        // Run a text-interface command and return its output.
        s.out.clear();
        let _ = s.execute(expr).await;
        let text = std::mem::take(&mut s.out);
        return Ok(json!({
            "result": text.trim_end_matches('\n'),
            "variablesReference": 0,
        }));
    }
    let opts = format_opts(a);
    let entry = s.eval(expr)?;
    let var = match entry {
        HistEntry::Src {
            module,
            inst,
            value,
        } => src_variable(
            s,
            st,
            expr.to_string(),
            Some(expr.to_string()),
            module,
            inst.unwrap_or(0),
            value,
            &opts,
        ),
        HistEntry::Wasm(w) => {
            wasm_variable(s, st, expr.to_string(), Some(expr.to_string()), w, opts.hex)
        }
    };
    let mut body = json!({
        "result": var.value,
        "variablesReference": var.variables_reference,
    });
    if let Some(t) = var.ty {
        body["type"] = json!(t);
    }
    if let Some(m) = var.memory_reference {
        body["memoryReference"] = json!(m);
    }
    if let Some(n) = var.indexed_variables {
        body["indexedVariables"] = json!(n);
    }
    Ok(body)
}

/// Add breakpoints for `specs`, returning DAP breakpoint descriptions.
fn add_breakpoints(s: &mut Session<'_>, specs: &[String], ids: &mut Vec<u32>) -> Vec<Breakpoint> {
    let mut out = vec![];
    for spec in specs {
        match s
            .resolve_location(spec)
            .and_then(|locs| Ok((s.add_breakpoint(spec.clone(), locs.clone(), false)?, locs)))
        {
            Ok((id, locs)) => {
                ids.push(id);
                let (module, pc) = locs[0];
                let info = &s.modules[module].info;
                let loc = info
                    .debug
                    .as_ref()
                    .zip(info.pc_to_dwarf(pc))
                    .and_then(|(d, addr)| d.source_loc(addr));
                out.push(Breakpoint {
                    id: Some(i64::from(id)),
                    verified: true,
                    line: loc.as_ref().map(|l| i64::from(l.line)),
                    source: loc.and_then(|l| source_of(s, module, l.file)),
                    instruction_reference: Some(memory_reference(code_address(module, pc))),
                    message: Some(s.describe_pc(module, pc)),
                });
            }
            Err(e) => out.push(Breakpoint {
                verified: false,
                message: Some(format!("{e}")),
                ..Default::default()
            }),
        }
    }
    out
}

fn set_breakpoints(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    let path = a["source"]["path"]
        .as_str()
        .or_else(|| a["source"]["name"].as_str())
        .ok_or_else(|| anyhow!("source path required"))?
        .to_string();
    for id in st.source_bps.remove(&path).unwrap_or_default() {
        s.delete_breakpoint(id);
    }
    let specs: Vec<String> = a["breakpoints"]
        .as_array()
        .map(|bps| {
            bps.iter()
                .map(|b| format!("{path}:{}", b["line"].as_u64().unwrap_or(0)))
                .collect()
        })
        .unwrap_or_default();
    let mut ids = vec![];
    let bps = add_breakpoints(s, &specs, &mut ids);
    st.source_bps.insert(path, ids);
    Ok(json!({ "breakpoints": bps }))
}

fn set_function_breakpoints(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    for id in std::mem::take(&mut st.function_bps) {
        s.delete_breakpoint(id);
    }
    let specs: Vec<String> = a["breakpoints"]
        .as_array()
        .map(|bps| {
            bps.iter()
                .filter_map(|b| b["name"].as_str().map(|n| n.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let mut ids = vec![];
    let bps = add_breakpoints(s, &specs, &mut ids);
    st.function_bps = ids;
    Ok(json!({ "breakpoints": bps }))
}

fn set_instruction_breakpoints(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    for id in std::mem::take(&mut st.instruction_bps) {
        s.delete_breakpoint(id);
    }
    let mut specs = vec![];
    for b in a["breakpoints"].as_array().cloned().unwrap_or_default() {
        let base = b["instructionReference"]
            .as_str()
            .and_then(parse_memory_reference)
            .ok_or_else(|| anyhow!("invalid instruction reference"))?;
        let addr = base.wrapping_add(b["offset"].as_i64().unwrap_or(0) as u64);
        let (module, pc) = split_code_address(addr);
        specs.push(format!("*{module}:{pc:#x}"));
    }
    let mut ids = vec![];
    let bps = add_breakpoints(s, &specs, &mut ids);
    st.instruction_bps = ids;
    Ok(json!({ "breakpoints": bps }))
}

/// Resolve the target of a data breakpoint: a variable (a child of a
/// `variablesReference`), an expression, or an address range.
fn data_target(s: &mut Session<'_>, st: &DapState, a: &Value) -> Result<(String, WatchTarget)> {
    let name = a["name"].as_str().unwrap_or("").to_string();
    if a["asAddress"].as_bool() == Some(true) {
        let addr = parse_memory_reference(&name)
            .map(Ok)
            .unwrap_or_else(|| s.eval_address(&name))?;
        let len = a["bytes"].as_u64().unwrap_or(1);
        let t = WatchTarget {
            inst: s.default_inst()?,
            addr,
            len,
            ty: None,
        };
        return Ok((format!("memory {addr:#x} length {len:#x}"), t));
    }
    let reference = a["variablesReference"].as_i64().filter(|r| *r > 0);
    let Some(r) = reference else {
        // An expression, evaluated in the given frame.
        let saved = s.selected;
        if a.get("frameId").is_some_and(|f| !f.is_null()) && !s.vframes.is_empty() {
            s.selected = frame_index(s, a)?;
        }
        let t = s.watch_target(&name);
        s.selected = saved;
        return Ok((name, t?));
    };
    let r = st
        .refs
        .get(usize::try_from(r - 1)?)
        .ok_or_else(|| anyhow!("invalid variables reference {r}"))?;
    match r {
        VarRef::Scope {
            frame,
            kind: ScopeKind::Args | ScopeKind::Locals,
        } => {
            let saved = s.selected;
            s.selected = *frame;
            let t = s.watch_target(&name);
            s.selected = saved;
            Ok((name, t?))
        }
        VarRef::Src {
            module,
            inst,
            value,
            path,
        } => {
            let (module, inst) = (*module, *inst);
            let dbg = s.modules[module].info.debug.as_ref().unwrap();
            let target = FrameTarget::new(s.d, None, &s.instances[inst].handle);
            let ctx = Ctx::new(dbg, &target);
            let opts = FormatOptions::default();
            let mut start = 0;
            loop {
                let kids = children(&ctx, value, &opts, start, 1000);
                if kids.is_empty() {
                    break Err(anyhow!("no member `{name}`"));
                }
                start += kids.len();
                if let Some(k) = kids.into_iter().find(|k| k.name == name) {
                    let desc = path
                        .as_deref()
                        .and_then(|p| k.eval_name(p))
                        .unwrap_or_else(|| name.clone());
                    drop(ctx);
                    break Ok((desc, s.value_watch_target(module, inst, &k.value)?));
                }
            }
        }
        _ => Err(anyhow!(
            "Only values in linear memory can be watched, not Wasm locals, globals or operands."
        )),
    }
}

fn data_breakpoint_info(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    Ok(match data_target(s, st, a) {
        Ok((desc, t)) => {
            st.data_targets.push((desc.clone(), t));
            json!({
                "dataId": (st.data_targets.len() - 1).to_string(),
                "description": desc,
                "accessTypes": ["write"],
                "canPersist": false,
            })
        }
        Err(e) => json!({ "dataId": null, "description": format!("{e}") }),
    })
}

fn set_data_breakpoints(s: &mut Session<'_>, st: &mut DapState, a: &Value) -> Result<Value> {
    for id in std::mem::take(&mut st.data_bps) {
        s.delete_breakpoint(id);
    }
    let mut out = vec![];
    for b in a["breakpoints"].as_array().cloned().unwrap_or_default() {
        let result = (|| -> Result<u32> {
            if b["accessType"].as_str().is_some_and(|t| t != "write") {
                return Err(anyhow!("Only writes can be watched."));
            }
            let (desc, t) = b["dataId"]
                .as_str()
                .and_then(|d| d.parse::<usize>().ok())
                .and_then(|i| st.data_targets.get(i))
                .ok_or_else(|| anyhow!("unknown dataId"))?;
            let t = *t;
            s.add_watchpoint(desc.clone(), t.inst, t.addr, t.len, t.ty)
        })();
        match result {
            Ok(id) => {
                st.data_bps.push(id);
                let w = s.watchpoints.iter().find(|w| w.id == id).unwrap();
                out.push(Breakpoint {
                    id: Some(i64::from(id)),
                    verified: true,
                    message: Some(if w.ty.is_some() {
                        format!("{} (memory {:#x} length {:#x})", w.spec, w.addr, w.len)
                    } else {
                        w.spec.clone()
                    }),
                    ..Default::default()
                });
            }
            Err(e) => out.push(Breakpoint {
                verified: false,
                message: Some(format!("{e}")),
                ..Default::default()
            }),
        }
    }
    Ok(json!({ "breakpoints": out }))
}

fn read_memory(s: &mut Session<'_>, a: &Value) -> Result<Value> {
    let base = a["memoryReference"]
        .as_str()
        .and_then(parse_memory_reference)
        .ok_or_else(|| anyhow!("invalid memory reference"))?;
    let addr = base.wrapping_add(a["offset"].as_i64().unwrap_or(0) as u64);
    let count = a["count"].as_u64().unwrap_or(0).min(1 << 20);
    let inst = s.default_inst()?;
    let mem = s.instances[inst]
        .handle
        .get_memory(s.d, 0)
        .map_err(|_| anyhow!("instance has no memory"))?;
    let size = mem.size_bytes(s.d);
    let readable = size.saturating_sub(addr).min(count);
    let data = if readable == 0 {
        vec![]
    } else {
        mem.get_bytes(s.d, addr, readable)
            .map_err(|e| anyhow!("{e:?}"))?
    };
    Ok(json!({
        "address": memory_reference(addr),
        "data": base64_encode(&data),
        "unreadableBytes": count - readable,
    }))
}

fn disassemble(s: &mut Session<'_>, a: &Value) -> Result<Value> {
    let base = a["memoryReference"]
        .as_str()
        .and_then(parse_memory_reference)
        .ok_or_else(|| anyhow!("invalid memory reference"))?;
    let addr = base.wrapping_add(a["offset"].as_i64().unwrap_or(0) as u64);
    let (module, pc) = split_code_address(addr);
    let info = &s
        .modules
        .get(module)
        .ok_or_else(|| anyhow!("no module {module}"))?
        .info;
    let lines = info.disassembly()?;
    let at = lines.partition_point(|l| l.offset < pc) as i64;
    let first = at + a["instructionOffset"].as_i64().unwrap_or(0);
    let count = a["instructionCount"].as_i64().unwrap_or(0).clamp(0, 10_000);
    let mut out = vec![];
    for i in first..first + count {
        let Some(l) = usize::try_from(i).ok().and_then(|i| lines.get(i)) else {
            out.push(DisassembledInstruction {
                address: memory_reference(code_address(module, 0)),
                instruction: "<invalid>".into(),
                presentation_hint: Some("invalid".into()),
                ..Default::default()
            });
            continue;
        };
        let func = info.func_at_pc(l.offset);
        let symbol = func
            .filter(|f| f.code_start == l.offset)
            .map(|f| info.func_display_name(f.index));
        let loc = info
            .debug
            .as_ref()
            .zip(info.pc_to_dwarf(l.offset))
            .and_then(|(d, a)| d.line_row(a).map(|r| (r.file, r.line)))
            .filter(|(_, line)| *line != 0);
        out.push(DisassembledInstruction {
            address: memory_reference(code_address(module, l.offset)),
            instruction: format!(
                "{}{}",
                "  ".repeat(l.indent.saturating_sub(2)),
                demangle_wat(&l.text)
            ),
            symbol,
            location: loc.and_then(|(f, _)| source_of(s, module, f)),
            line: loc.map(|(_, l)| i64::from(l)),
            presentation_hint: None,
        });
    }
    Ok(json!({ "instructions": out }))
}

fn source(s: &mut Session<'_>, a: &Value) -> Result<Value> {
    let path = a["source"]["path"]
        .as_str()
        .ok_or_else(|| anyhow!("source path required"))?
        .to_string();
    let lines = s.source_lines(&path).cloned().ok_or_else(|| {
        anyhow!("cannot read {path} (give the debugger access with `-Ddebug-fs=<dir>`)")
    })?;
    Ok(json!({ "content": lines.join("\n") + "\n" }))
}

fn modules(s: &mut Session<'_>) -> Result<Value> {
    s.refresh_entities();
    let modules: Vec<Module> = s
        .modules
        .iter()
        .enumerate()
        .map(|(i, m)| Module {
            id: json!(i),
            name: m.label.clone(),
            symbol_status: Some(
                if m.info.debug.is_some() {
                    "Symbols loaded"
                } else {
                    "No symbols"
                }
                .into(),
            ),
        })
        .collect();
    Ok(json!({ "modules": modules, "totalModules": modules.len() }))
}
