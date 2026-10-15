//! End-to-end tests for the built-in debugger's Debug Adapter Protocol
//! server (`wasmtime run --debug-dap`).
//!
//! These mirror the text-interface tests in `debug_tui.rs`, driving the
//! same scenarios through DAP requests with a simple DAP client and
//! checking the structured results (frames, scopes, variable trees,
//! events, memory, disassembly) as well as the output of text commands
//! run through the `evaluate` request's `repl` context.
//!
//! Like the text-interface tests these are `#[ignore]`d by default; run
//! them with
//! `cargo test --features debug-dap-tui --test all -- --ignored debug_dap::`.

use super::debug_tui::{GC_WAT, check_output, free_port, record, wasmtime_binary, wat_file};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use test_programs_artifacts::*;
use wasmtime::{Result, bail, format_err};
use wasmtime_debug_dap::types::{Breakpoint, Scope, StackFrame, Variable};
use wasmtime_debug_dap::{Client, base64_decode, parse_memory_reference};

type DapResult<T> = std::result::Result<T, wasmtime::Error>;

fn err(e: impl std::fmt::Display) -> wasmtime::Error {
    format_err!("{e}")
}

/// A running `wasmtime run --debug-dap` with a connected client, stopped
/// on entry.
struct Dap {
    child: Child,
    client: Client,
    /// Console `output` events received so far.
    console: String,
    /// The server's capabilities.
    caps: Value,
}

impl Dap {
    fn spawn(wasm: &str, extra_args: &[&str]) -> Result<Dap> {
        Dap::spawn_subcommand("run", wasm, extra_args)
    }

    /// Debug a replay of the recorded execution `trace`.
    fn spawn_replay(trace: &str) -> Result<Dap> {
        Dap::spawn_subcommand("replay", trace, &[])
    }

    fn spawn_subcommand(subcommand: &str, wasm: &str, extra_args: &[&str]) -> Result<Dap> {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/test-programs/src/bin");
        let port = free_port();
        let mut cmd = Command::new(wasmtime_binary());
        cmd.arg(subcommand)
            .arg("-Ccache=n")
            .arg(format!("--debug-dap={port}"))
            .arg(format!("-Ddebug-fs={src}"))
            .args(extra_args)
            .arg(wasm)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        eprintln!("spawning: {cmd:?}");
        let mut child = cmd.spawn()?;
        let mut stderr = BufReader::new(child.stderr.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear();
            if stderr.read_line(&mut line)? == 0 {
                let _ = child.kill();
                bail!("wasmtime exited before the debugger was ready");
            }
            if line.contains("Debugger listening on") {
                break;
            }
        }
        std::thread::spawn(move || {
            let mut sink = String::new();
            let _ = stderr.read_to_string(&mut sink);
            eprintln!("wasmtime stderr (rest): {sink}");
        });
        let mut client = Client::connect(("127.0.0.1", port)).map_err(err)?;
        client.timeout = Duration::from_secs(120);
        let caps = client
            .request("initialize", json!({ "adapterID": "wasmtime" }))
            .map_err(err)?;
        assert_eq!(caps["supportsDisassembleRequest"], true);
        assert_eq!(caps["supportsInstructionBreakpoints"], true);
        client
            .request("launch", json!({ "stopOnEntry": true }))
            .map_err(err)?;
        client
            .request("configurationDone", json!({}))
            .map_err(err)?;
        let mut dap = Dap {
            child,
            client,
            console: String::new(),
            caps,
        };
        let stopped = dap.wait_stop()?;
        assert_eq!(stopped.0, "stopped");
        assert_eq!(stopped.1["reason"], "entry");
        Ok(dap)
    }

    fn req(&mut self, command: &str, args: Value) -> Result<Value> {
        self.client.request(command, args).map_err(err)
    }

    /// Wait for `stopped` or `terminated`, collecting console output.
    fn wait_stop(&mut self) -> Result<(String, Value)> {
        let console = &mut self.console;
        self.client
            .wait_any_event(&["stopped", "terminated"], &mut |e| {
                if e["event"] == "output" {
                    console.push_str(e["body"]["output"].as_str().unwrap_or(""));
                }
            })
            .map_err(err)
    }

    /// Resume with `command` and wait for the program to stop.
    fn resume(&mut self, command: &str, granularity: Option<&str>) -> Result<(String, Value)> {
        let mut args = json!({ "threadId": 1 });
        if let Some(g) = granularity {
            args["granularity"] = json!(g);
        }
        self.req(command, args)?;
        self.wait_stop()
    }

    fn frames(&mut self) -> Result<Vec<StackFrame>> {
        let body = self.req(
            "stackTrace",
            json!({ "threadId": 1, "format": { "parameters": true } }),
        )?;
        Ok(serde_json::from_value(body["stackFrames"].clone())?)
    }

    fn scopes(&mut self, frame: i64) -> Result<Vec<Scope>> {
        let body = self.req("scopes", json!({ "frameId": frame }))?;
        Ok(serde_json::from_value(body["scopes"].clone())?)
    }

    fn vars(&mut self, r: i64) -> Result<Vec<Variable>> {
        assert!(r > 0, "not expandable");
        let body = self.req("variables", json!({ "variablesReference": r }))?;
        Ok(serde_json::from_value(body["variables"].clone())?)
    }

    fn scope_vars(&mut self, frame: i64, scope: &str) -> Result<Vec<Variable>> {
        let scopes = self.scopes(frame)?;
        let s = scopes
            .iter()
            .find(|s| s.name == scope)
            .ok_or_else(|| format_err!("no scope {scope}"))?;
        self.vars(s.variables_reference)
    }

    fn eval(&mut self, expr: &str, frame: i64) -> Result<Value> {
        self.req(
            "evaluate",
            json!({ "expression": expr, "frameId": frame, "context": "watch" }),
        )
    }

    fn repl(&mut self, cmd: &str, frame: i64) -> Result<String> {
        let body = self.req(
            "evaluate",
            json!({ "expression": cmd, "frameId": frame, "context": "repl" }),
        )?;
        Ok(body["result"].as_str().unwrap_or("").to_string())
    }

    fn read_memory(&mut self, addr: u64, n: u64) -> Result<Vec<u8>> {
        let body = self.req(
            "readMemory",
            json!({ "memoryReference": format!("{addr:#x}"), "count": n }),
        )?;
        base64_decode(body["data"].as_str().unwrap()).ok_or_else(|| format_err!("bad base64"))
    }

    /// Disconnect and return the program's stdout.
    fn finish(mut self) -> Result<String> {
        let _ = self.client.request("disconnect", json!({}));
        let output = self.child.wait_with_output()?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

fn var<'a>(vars: &'a [Variable], name: &str) -> &'a Variable {
    vars.iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("no variable `{name}` in {vars:#?}"))
}

fn names(vars: &[Variable]) -> Vec<&str> {
    vars.iter().map(|v| v.name.as_str()).collect()
}

fn breakpoints(body: &Value) -> Vec<Breakpoint> {
    serde_json::from_value(body["breakpoints"].clone()).unwrap()
}

#[test]
#[ignore]
fn debug_dap_c_values_and_stepping() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_C, &[])?;
    let bps = breakpoints(&dap.req(
        "setBreakpoints",
        json!({ "source": { "path": "debug_tui_c.c" }, "breakpoints": [{ "line": 60 }] }),
    )?);
    assert!(bps[0].verified);
    assert_eq!(bps[0].line, Some(60));
    assert!(
        bps[0]
            .source
            .as_ref()
            .and_then(|s| s.path.as_deref())
            .is_some_and(|p| p.ends_with("/debug_tui_c.c"))
    );
    let bp_id = bps[0].id.unwrap();

    let (ev, body) = dap.resume("continue", None)?;
    assert_eq!(ev, "stopped");
    assert_eq!(body["reason"], "breakpoint");
    assert_eq!(body["hitBreakpointIds"], json!([bp_id]));

    // Stack.
    let frames = dap.frames()?;
    assert_eq!(frames[0].name, "main ()");
    assert_eq!(frames[0].line, 60);
    assert_eq!(frames[1].name, "_start ()");

    // Scopes and variable trees.
    let scope_names: Vec<String> = dap.scopes(1)?.into_iter().map(|s| s.name).collect();
    assert_eq!(
        scope_names,
        [
            "Arguments",
            "Locals",
            "Wasm Locals",
            "Wasm Operand Stack",
            "Wasm Globals"
        ]
    );
    let locals = dap.scope_vars(1, "Locals")?;
    assert_eq!(names(&locals), ["tri", "other", "n", "heap", "arr", "r"]);
    let tri = var(&locals, "tri").clone();
    assert!(tri.value.starts_with("struct Shape { .name = 0x"));
    assert!(
        tri.value
            .contains(".tag = \"tri\", .scale = 1.5, .flags = 5 }")
    );
    assert_eq!(tri.ty.as_deref(), Some("struct Shape"));
    assert!(tri.memory_reference.is_some());
    let fields = dap.vars(tri.variables_reference)?;
    assert_eq!(
        names(&fields),
        ["name", "color", "points", "next", "tag", "scale", "flags"]
    );
    assert_eq!(var(&fields, "color").value, "GREEN");
    assert!(var(&fields, "name").value.ends_with("\"triangle\""));
    let points = var(&fields, "points").clone();
    assert_eq!(points.indexed_variables, Some(3));
    let elems = dap.vars(points.variables_reference)?;
    assert_eq!(names(&elems), ["[0]", "[1]", "[2]"]);
    assert_eq!(elems[1].evaluate_name.as_deref(), Some("tri.points[1]"));
    let p1 = dap.vars(elems[1].variables_reference)?;
    assert_eq!(var(&p1, "x").value, "4");
    assert_eq!(
        var(&p1, "x").evaluate_name.as_deref(),
        Some("tri.points[1].x")
    );
    // A pointer to a struct expands to the pointee's fields, named with
    // `->`.
    let next = var(&fields, "next").clone();
    let pointee = dap.vars(next.variables_reference)?;
    assert_eq!(
        var(&pointee, "name").evaluate_name.as_deref(),
        Some("tri.next->name")
    );
    assert!(var(&pointee, "name").value.ends_with("\"other\""));
    assert_eq!(var(&pointee, "color").value, "BLUE");

    // Evaluation, including the evaluate names from the tree.
    assert_eq!(dap.eval("tri.points[1].x", 1)?["result"], "4");
    assert!(
        dap.eval("tri.next->name", 1)?["result"]
            .as_str()
            .unwrap()
            .ends_with("\"other\"")
    );
    assert_eq!(dap.eval("global_counter", 1)?["result"], "43");
    assert!(dap.eval("nope", 1).is_err());

    // Text commands through the REPL, with value history.
    assert_eq!(dap.repl("print tri.tag", 1)?, "%1 = \"tri\"");
    assert_eq!(dap.repl("print %1", 1)?, "%2 = \"tri\"");
    check_output(
        &dap.repl("ptype tri", 1)?,
        "check: type = struct Shape {\ncheck: const char *name;\ncheck: unsigned int flags : 3;",
    )?;
    check_output(
        &dap.repl("info breakpoints", 1)?,
        "check: 1    breakpoint    1     debug_tui_c.c:60\ncheck: in main at debug_tui_c.c:60",
    )?;

    // Memory: `&arr` has a memory reference; read it.
    let arr = dap.eval("&arr", 1)?;
    let addr = parse_memory_reference(arr["memoryReference"].as_str().unwrap()).unwrap();
    let bytes = dap.read_memory(addr, 20)?;
    let ints: Vec<u32> = bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(ints, [10, 20, 30, 40, 50]);

    // Disassembly around the PC, with source locations.
    let ip = frames[0].instruction_pointer_reference.clone().unwrap();
    let body = dap.req(
        "disassemble",
        json!({ "memoryReference": ip, "instructionOffset": -1, "instructionCount": 3 }),
    )?;
    let instrs = body["instructions"].as_array().unwrap();
    assert_eq!(instrs.len(), 3);
    assert_eq!(instrs[1]["address"], json!(ip));
    assert_eq!(instrs[1]["instruction"], "local.get 0");
    assert_eq!(instrs[1]["line"], 60);

    // Step into `area`.
    let (_, body) = dap.resume("stepIn", None)?;
    assert_eq!(body["reason"], "step");
    let frames = dap.frames()?;
    assert!(frames[0].name.starts_with("area (s=(struct Shape *) 0x"));
    assert_eq!(frames[0].line, 33);
    let args = dap.scope_vars(1, "Arguments")?;
    assert_eq!(names(&args), ["s"]);
    let s = dap.vars(args[0].variables_reference)?;
    assert_eq!(
        var(&s, "points").evaluate_name.as_deref(),
        Some("s->points")
    );
    assert_eq!(dap.eval("s->points[2].y", 1)?["result"], "3");
    // Frame 2 is `main`.
    assert_eq!(dap.eval("r", 2)?["result"], "0");

    // Step out, reporting the return value.
    dap.console.clear();
    dap.resume("stepOut", None)?;
    assert!(
        dap.console.contains("Value returned is %3 = 6"),
        "{}",
        dap.console
    );
    assert_eq!(dap.frames()?[0].line, 60);
    dap.resume("next", None)?;
    assert_eq!(dap.frames()?[0].line, 61);
    assert_eq!(dap.eval("r", 1)?["result"], "6");

    // Run to completion.
    dap.req(
        "setBreakpoints",
        json!({ "source": { "path": "debug_tui_c.c" }, "breakpoints": [] }),
    )?;
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    let stdout = dap.finish()?;
    assert!(stdout.contains("area = 6 on the heap 30"));
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_rust_values() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_RUST, &[])?;
    let bps = breakpoints(&dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "describe" }] }),
    )?);
    assert!(bps[0].verified);
    dap.resume("continue", None)?;
    let frames = dap.frames()?;
    assert_eq!(
        frames[0].name,
        "debug_tui_rust::describe (shape=&Shape::Rect { w: 3, h: 4 }, color=Color::Blue, point=&Point { x: 1, y: 2 })"
    );
    assert!(
        frames
            .iter()
            .any(|f| f.name.contains("lang_start_internal"))
    );

    let args = dap.scope_vars(1, "Arguments")?;
    assert_eq!(names(&args), ["shape", "color", "point"]);
    assert_eq!(var(&args, "shape").value, "&Shape::Rect { w: 3, h: 4 }");
    // A reference expands to its referent's fields; an enum to its active
    // variant's fields.
    let shape = dap.vars(var(&args, "shape").variables_reference)?;
    assert_eq!(names(&shape), ["w", "h"]);
    assert_eq!(var(&shape, "w").evaluate_name.as_deref(), Some("shape.w"));
    assert_eq!(dap.eval("shape.w", 1)?["result"], "3");

    // The caller's locals, in frame 2.
    let locals = dap.scope_vars(2, "Locals")?;
    assert_eq!(var(&locals, "name").value, "\"wasmtime\"");
    assert_eq!(var(&locals, "nums").value, "vec![1, 2, 3, 4]");
    let shapes = var(&locals, "shapes").clone();
    assert_eq!(shapes.indexed_variables, Some(3));
    let elems = dap.vars(shapes.variables_reference)?;
    assert_eq!(names(&elems), ["[0]", "[1]", "[2]"]);
    assert_eq!(elems[0].value, "Shape::Circle(1.5)");
    let circle = dap.vars(elems[0].variables_reference)?;
    assert_eq!(names(&circle), ["0"]);
    assert_eq!(circle[0].evaluate_name.as_deref(), Some("shapes[0].0"));
    assert_eq!(dap.eval("shapes[0].0", 2)?["result"], "1.5");
    let map = var(&locals, "map").clone();
    let mut entries: Vec<(String, String)> = dap
        .vars(map.variables_reference)?
        .into_iter()
        .map(|v| (v.name, v.value))
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        [
            ("\"one\"".to_string(), "1".to_string()),
            ("\"two\"".to_string(), "2".to_string())
        ]
    );
    let list = dap.vars(var(&locals, "list").variables_reference)?;
    assert_eq!(names(&list), ["value", "next"]);
    assert_eq!(dap.eval("list.next.value", 2)?["result"], "2");
    assert_eq!(var(&locals, "tuple").value, "(1, 'x', true)");
    assert_eq!(dap.eval("GREETING", 2)?["result"], "\"hello, world\"");

    dap.resume("stepOut", None)?;
    assert!(dap.frames()?[0].name.starts_with("debug_tui_rust::main"));
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    assert!(dap.finish()?.contains("wasmtime [1, 2, 3, 4]"));
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_rust_std_containers() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_RUST_STD, &[])?;
    dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "consume" }] }),
    )?;
    dap.resume("continue", None)?;
    let locals = dap.scope_vars(2, "Locals")?;
    for (name, value) in [
        ("deque", "VecDeque[0, 3, 4, 5]"),
        ("btree_set", "BTreeSet(len=3) { 'x', 'y', 'z' }"),
        (
            "arc",
            "Arc(Inner { id: 3, label: \"arc\" }) [strong=2, weak=0]",
        ),
        ("refcell", "RefCell(vec![1, -1])"),
        ("once_lock", "OnceLock(\"set once\")"),
        ("mutex_guard", "MutexGuard(vec![3, 4])"),
        ("rc_slice", "Rc([11, 12]) [strong=1, weak=0]"),
        ("boxed_dyn", "Box(Inner { id: 6, label: \"dyn\" })"),
        ("custom", "MyCustomPointer { _inner: Box(vec![1, 2, 3]) }"),
        ("s", "S { ptr: &123, name: \"s\" }"),
    ] {
        assert_eq!(var(&locals, name).value, value, "{name}");
    }
    // Containers expand to their elements, through wrappers.
    let deque = dap.vars(var(&locals, "deque").variables_reference)?;
    assert_eq!(names(&deque), ["[0]", "[1]", "[2]", "[3]"]);
    assert_eq!(deque[3].value, "5");
    assert_eq!(
        dap.eval(deque[3].evaluate_name.as_deref().unwrap(), 2)?["result"],
        "5"
    );
    let btree = dap.vars(var(&locals, "btree_map").variables_reference)?;
    assert_eq!(btree.len(), 20);
    assert_eq!(
        (btree[3].name.as_str(), btree[3].value.as_str()),
        ("3", "\"odd\"")
    );
    let arc = dap.vars(var(&locals, "arc").variables_reference)?;
    assert_eq!(names(&arc), ["id", "label"]);
    assert_eq!(arc[0].evaluate_name.as_deref(), Some("arc.id"));
    assert_eq!(dap.eval("arc.id", 2)?["result"], "3");
    let refcell = dap.vars(var(&locals, "refcell").variables_reference)?;
    assert_eq!(names(&refcell), ["[0]", "[1]"]);
    assert_eq!(dap.eval("refcell[1]", 2)?["result"], "-1");
    let nested = dap.vars(var(&locals, "nested").variables_reference)?;
    assert_eq!(nested[1].value, "None");
    let dynamic = dap.vars(var(&locals, "boxed_dyn").variables_reference)?;
    assert_eq!(names(&dynamic), ["id", "label"]);
    let custom = dap.vars(var(&locals, "custom").variables_reference)?;
    let inner = dap.vars(custom[0].variables_reference)?;
    assert_eq!(names(&inner), ["[0]", "[1]", "[2]"]);
    assert_eq!(inner[2].evaluate_name.as_deref(), Some("custom._inner[2]"));
    assert_eq!(dap.eval("custom._inner[2]", 2)?["result"], "3");
    // References expand to their referents.
    let s = dap.vars(var(&locals, "s").variables_reference)?;
    let ptr = dap.vars(var(&s, "ptr").variables_reference)?;
    assert_eq!((ptr[0].name.as_str(), ptr[0].value.as_str()), ("*", "123"));
    assert_eq!(dap.eval("*s.ptr", 2)?["result"], "123");
    dap.finish()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_cpp_std() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_CPP, &[])?;
    dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "consume" }] }),
    )?;
    dap.resume("continue", None)?;
    let locals = dap.scope_vars(2, "Locals")?;
    assert_eq!(
        var(&locals, "vec").value,
        "std::vector of length 3, capacity 3 = {1, 2, 3}"
    );
    assert_eq!(
        var(&locals, "map").value,
        "std::map with 3 elements = {[1] = \"one\", [2] = \"two\", [3] = \"three\"}"
    );
    assert_eq!(var(&locals, "short_str").value, "\"hi\"");
    let map = dap.vars(var(&locals, "map").variables_reference)?;
    assert_eq!(
        map.iter()
            .map(|v| (v.name.as_str(), v.value.as_str()))
            .collect::<Vec<_>>(),
        [("1", "\"one\""), ("2", "\"two\""), ("3", "\"three\"")]
    );
    let points = dap.vars(var(&locals, "points").variables_reference)?;
    let p1 = dap.vars(points[1].variables_reference)?;
    assert_eq!(var(&p1, "y").evaluate_name.as_deref(), Some("points[1].y"));
    assert_eq!(dap.eval("points[1].y", 2)?["result"], "4");
    let uptr = dap.vars(var(&locals, "uptr").variables_reference)?;
    assert_eq!(names(&uptr), ["x", "y"]);
    assert_eq!(dap.eval("uptr->y", 2)?["result"], "6");
    assert_eq!(dap.eval("sptr->x", 2)?["result"], "7");
    assert_eq!(
        var(&locals, "weird").value,
        "std::vector<Weird, std::allocator<Weird> > { .not_a_vector = 7 }"
    );
    dap.finish()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_wasm_level_and_gc() -> DapResult<()> {
    let wat = wat_file(GC_WAT)?;
    let mut dap = Dap::spawn(
        wat.path().to_str().unwrap(),
        &["-Wgc=y,function-references=y"],
    )?;
    let bps = breakpoints(&dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "use" }] }),
    )?);
    assert!(bps[0].verified);
    dap.resume("continue", None)?;
    let frames = dap.frames()?;
    assert_eq!(
        frames[0].name,
        "use (p=struct (3 fields), a=array (len 4), i=i31 -42)"
    );
    assert!(frames[0].source.is_none());
    // No debug info: only the Wasm scopes.
    let scope_names: Vec<String> = dap.scopes(1)?.into_iter().map(|s| s.name).collect();
    assert_eq!(
        scope_names,
        ["Wasm Locals", "Wasm Operand Stack", "Wasm Globals"]
    );
    let locals = dap.scope_vars(1, "Wasm Locals")?;
    assert_eq!(
        names(&locals),
        ["local 0 (p)", "local 1 (a)", "local 2 (i)", "local 3 (tmp)"]
    );
    assert_eq!(locals[2].value, "i31 -42");
    assert_eq!(locals[0].evaluate_name.as_deref(), Some("%local0"));
    // GC objects expand into fields and elements.
    let p = dap.vars(locals[0].variables_reference)?;
    assert_eq!(names(&p), ["0", "1", "2"]);
    assert_eq!(p[1].value, "f64 2.5");
    let inner = dap.vars(p[2].variables_reference)?;
    assert_eq!(inner[1].evaluate_name.as_deref(), Some("%local0.2.1"));
    assert_eq!(dap.eval("%local0.2.1", 1)?["result"], "f64 4.5");
    assert_eq!(dap.eval("$p.2.1", 1)?["result"], "f64 4.5");
    let a = dap.vars(locals[1].variables_reference)?;
    assert_eq!(names(&a), ["[0]", "[1]", "[2]", "[3]"]);

    // Instruction stepping, and the operand stack.
    dap.resume("stepIn", Some("instruction"))?;
    dap.resume("stepIn", Some("instruction"))?;
    let stack = dap.scope_vars(1, "Wasm Operand Stack")?;
    assert_eq!(
        stack
            .iter()
            .map(|v| (v.name.as_str(), v.value.as_str()))
            .collect::<Vec<_>>(),
        [("stack[0]", "i32 65"), ("stack[1]", "i32 3")]
    );
    for _ in 0..7 {
        dap.resume("next", Some("instruction"))?;
    }
    let globals = dap.scope_vars(1, "Wasm Globals")?;
    assert_eq!(names(&globals), ["global 0 (g)"]);
    let g = dap.vars(globals[0].variables_reference)?;
    assert_eq!(g[0].value, "i32 1");
    assert_eq!(dap.eval("$t[1][3]", 1)?["result"], "i32 7");
    check_output(
        &dap.repl("wasm-table", 1)?,
        "check: Table 0 of instance #0: 3 elements\ncheck: [1] array (len 4)",
    )?;
    let mem = dap.read_memory(16, 14)?;
    assert_eq!(mem, b"hello, memory!");
    // Disassembly with function symbols.
    let ip = dap.frames()?[0]
        .instruction_pointer_reference
        .clone()
        .unwrap();
    let body = dap.req(
        "disassemble",
        json!({ "memoryReference": ip, "instructionOffset": 0, "instructionCount": 2 }),
    )?;
    assert_eq!(body["instructions"][0]["instruction"], "local.get $p");
    assert_eq!(
        body["instructions"][1]["instruction"],
        "struct.get $point $x"
    );

    dap.console.clear();
    dap.resume("stepOut", None)?;
    assert!(dap.console.contains("Wasm results: i32 8"));
    assert_eq!(dap.frames()?[0].name, "_start ()");
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    dap.finish()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_instruction_stepping_and_breakpoints() -> DapResult<()> {
    let wat = wat_file(
        r#"
        (module $steps
          (func $add (param $a i32) (param $b i32) (result i32)
            local.get $a
            local.get $b
            i32.add)
          (func $main (export "_start")
            i32.const 2
            i32.const 3
            call $add
            i32.const 10
            i32.mul
            drop))
        "#,
    )?;
    let mut dap = Dap::spawn(wat.path().to_str().unwrap(), &[])?;
    let bps = breakpoints(&dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "main" }] }),
    )?);
    let main_ref = bps[0].instruction_reference.clone().unwrap();
    dap.resume("continue", None)?;
    // Replace the function breakpoint with an instruction breakpoint two
    // instructions later (at the `call`).
    dap.req("setFunctionBreakpoints", json!({ "breakpoints": [] }))?;
    let bps = breakpoints(&dap.req(
        "setInstructionBreakpoints",
        json!({ "breakpoints": [{ "instructionReference": main_ref, "offset": 4 }] }),
    )?);
    assert!(bps[0].verified);
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "breakpoint");
    let stack = dap.scope_vars(1, "Wasm Operand Stack")?;
    assert_eq!(stack.len(), 2);
    // Step into the call.
    dap.resume("stepIn", Some("instruction"))?;
    let frames = dap.frames()?;
    assert_eq!(frames[0].name, "add (a=i32 2, b=i32 3)");
    assert_eq!(frames.len(), 2);
    for _ in 0..4 {
        dap.resume("stepIn", Some("instruction"))?;
    }
    assert_eq!(dap.frames()?[0].name, "main ()");
    let stack = dap.scope_vars(1, "Wasm Operand Stack")?;
    assert_eq!(stack[0].value, "i32 5");
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    dap.finish()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_pause() -> DapResult<()> {
    let wat = wat_file(
        r#"
        (module
          (func $spin (export "_start")
            (loop $l (br $l))))
        "#,
    )?;
    let mut dap = Dap::spawn(wat.path().to_str().unwrap(), &[])?;
    dap.req("continue", json!({ "threadId": 1 }))?;
    std::thread::sleep(Duration::from_millis(500));
    // Requests that need a stopped program fail while running.
    assert!(dap.req("stackTrace", json!({ "threadId": 1 })).is_err());
    dap.req("pause", json!({ "threadId": 1 }))?;
    let (ev, body) = dap.wait_stop()?;
    assert_eq!(ev, "stopped");
    assert_eq!(body["reason"], "pause");
    assert_eq!(dap.frames()?[0].name, "spin ()");
    let mut child = dap.child;
    child.kill()?;
    child.wait()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_trap() -> DapResult<()> {
    let wat = wat_file(
        r#"
        (module
          (func $divide (param $a i32) (param $b i32) (result i32)
            (i32.div_s (local.get $a) (local.get $b)))
          (func (export "_start")
            (drop (call $divide (i32.const 10) (i32.const 0)))))
        "#,
    )?;
    let mut dap = Dap::spawn(wat.path().to_str().unwrap(), &[])?;
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "exception");
    assert_eq!(dap.frames()?[0].name, "divide (a=i32 10, b=i32 0)");
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    let _ = dap.client.request("disconnect", json!({}));
    let status = dap.child.wait()?;
    assert!(!status.success());
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_named_entities_and_locations() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_C_OPT, &[])?;
    dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "checksum" }] }),
    )?;
    dap.resume("continue", None)?;
    assert_eq!(dap.repl("location data", 1)?, "local 0");
    assert_eq!(dap.repl("location n", 1)?, "local 1");
    assert_eq!(dap.repl("location %global0", 1)?, "global 0");
    // `&` gives a memory reference that can be read.
    let buf = dap.eval("&global_buf", 1)?;
    let addr = parse_memory_reference(buf["memoryReference"].as_str().unwrap()).unwrap();
    assert_eq!(dap.read_memory(addr, 16)?, b"0123456789abcdef");
    // A variable's memory reference.
    let first = dap.eval("global_buf[0]", 1)?;
    assert_eq!(
        parse_memory_reference(first["memoryReference"].as_str().unwrap()),
        Some(addr)
    );
    // `x` through the REPL.
    check_output(&dap.repl("x/4cb &global_buf", 1)?, "check: '0' '1' '2' '3'")?;
    // Wasm locals by index and by name agree.
    assert_eq!(
        dap.eval("%local1", 1)?["result"],
        dap.scope_vars(1, "Wasm Locals")?[1].value.as_str()
    );
    dap.finish()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_source_request() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_C, &[])?;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/crates/test-programs/src/bin/debug_tui_c.c"
    );
    let body = dap.req("source", json!({ "source": { "path": path } }))?;
    assert!(
        body["content"]
            .as_str()
            .unwrap()
            .contains("int area(struct Shape *s)")
    );
    // Outside the `-Ddebug-fs` directory, sources are unreadable.
    assert!(
        dap.req("source", json!({ "source": { "path": "/etc/hostname" } }))
            .is_err()
    );
    let modules = dap.req("modules", json!({}))?;
    assert_eq!(modules["modules"][0]["symbolStatus"], "Symbols loaded");
    dap.finish()?;
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_data_breakpoints() -> DapResult<()> {
    let mut dap = Dap::spawn(DEBUG_TUI_C_HISTORY, &[])?;
    assert_eq!(dap.caps["supportsDataBreakpoints"], true);
    assert_eq!(dap.caps["supportsStepBack"], false);
    dap.req(
        "setFunctionBreakpoints",
        json!({ "breakpoints": [{ "name": "add" }] }),
    )?;
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "breakpoint");

    // A member of a variable, by its parent's `variablesReference`.
    let args = dap.scope_vars(1, "Arguments")?;
    let acc = var(&args, "acc").variables_reference;
    let fields = dap.vars(acc)?;
    assert_eq!(names(&fields), ["total", "count"]);
    let total = dap.req(
        "dataBreakpointInfo",
        json!({ "variablesReference": acc, "name": "total" }),
    )?;
    assert_eq!(total["description"], "acc->total");
    assert_eq!(total["accessTypes"], json!(["write"]));
    // An expression.
    let counter = dap.req(
        "dataBreakpointInfo",
        json!({ "name": "counter", "frameId": 1 }),
    )?;
    assert_eq!(counter["description"], "counter");
    // An address range.
    let count_addr = var(&fields, "count").memory_reference.clone().unwrap();
    let count = dap.req(
        "dataBreakpointInfo",
        json!({ "name": count_addr, "asAddress": true, "bytes": 4 }),
    )?;
    assert!(count["dataId"].is_string());
    // Values outside linear memory cannot be watched.
    let scopes = dap.scopes(1)?;
    let wasm_locals = scopes
        .iter()
        .find(|s| s.name == "Wasm Locals")
        .unwrap()
        .variables_reference;
    let local = dap.req(
        "dataBreakpointInfo",
        json!({ "variablesReference": wasm_locals, "name": "local 0" }),
    )?;
    assert!(local["dataId"].is_null());
    assert!(
        local["description"]
            .as_str()
            .unwrap()
            .contains("Only values in linear memory can be watched")
    );

    // Watch `acc->total` and `counter`.
    let body = dap.req(
        "setDataBreakpoints",
        json!({ "breakpoints": [
            { "dataId": total["dataId"], "accessType": "write" },
            { "dataId": counter["dataId"] },
            { "dataId": count["dataId"], "accessType": "read" },
        ] }),
    )?;
    let bps = breakpoints(&body);
    assert!(bps[0].verified && bps[1].verified && !bps[2].verified);
    let (total_id, counter_id) = (bps[0].id.unwrap(), bps[1].id.unwrap());
    dap.req("setFunctionBreakpoints", json!({ "breakpoints": [] }))?;

    dap.console.clear();
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "data breakpoint");
    assert_eq!(body["hitBreakpointIds"], json!([total_id]));
    check_output(
        &dap.console,
        r#"
check: Watchpoint
sameln: acc->total
check: Old value = 0
check: New value = 1
"#,
    )?;
    // Stopped after the write.
    assert_eq!(dap.frames()?[0].line, 22);
    assert_eq!(dap.eval("acc->total", 1)?["result"], "1");
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["hitBreakpointIds"], json!([counter_id]));
    assert_eq!(dap.eval("counter", 1)?["result"], "1");

    // Replacing the data breakpoints removes the old ones.
    dap.req(
        "setDataBreakpoints",
        json!({ "breakpoints": [{ "dataId": total["dataId"] }] }),
    )?;
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "data breakpoint");
    assert_eq!(dap.eval("acc->total", 1)?["result"], "5");
    // `watch` also works in the debug console.
    let out = dap.repl("info breakpoints", 1)?;
    assert!(out.contains("watchpoint"), "{out}");

    // This is not a replay, so it cannot run backward.
    assert!(
        dap.req("stepBack", json!({ "threadId": 1 }))
            .unwrap_err()
            .to_string()
            .contains("requires a recorded execution")
    );
    dap.req("setDataBreakpoints", json!({ "breakpoints": [] }))?;
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    let stdout = dap.finish()?;
    assert!(stdout.contains("total = 30 count = 4 counter = 4"));
    Ok(())
}

#[test]
#[ignore]
fn debug_dap_reverse_execution() -> DapResult<()> {
    let (_dir, trace) = record(DEBUG_TUI_C_HISTORY)?;
    let mut dap = Dap::spawn_replay(&trace)?;
    assert_eq!(dap.caps["supportsStepBack"], true);
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/crates/test-programs/src/bin/debug_tui_c_history.c"
    );
    let bps = breakpoints(&dap.req(
        "setBreakpoints",
        json!({ "source": { "path": path }, "breakpoints": [{ "line": 29 }] }),
    )?);
    let bp = bps[0].id.unwrap();
    dap.resume("continue", None)?;
    dap.resume("continue", None)?;
    assert_eq!(dap.eval("i", 1)?["result"], "2");

    // Step back over lines and instructions.
    let (_, body) = dap.resume("stepBack", None)?;
    assert_eq!(body["reason"], "step");
    assert_eq!(dap.frames()?[0].line, 28);
    let (_, body) = dap.resume("stepBack", None)?;
    assert_eq!(body["reason"], "step");
    let frames = dap.frames()?;
    assert_eq!(frames[0].line, 30);
    assert!(frames[0].name.starts_with("main"));
    let before = dap.frames()?[0].instruction_pointer_reference.clone();
    dap.resume("stepBack", Some("instruction"))?;
    assert_ne!(dap.frames()?[0].instruction_pointer_reference, before);

    // Back to the previous breakpoint hit.
    let (_, body) = dap.resume("reverseContinue", None)?;
    assert_eq!(body["reason"], "breakpoint");
    assert_eq!(body["hitBreakpointIds"], json!([bp]));
    assert_eq!(dap.eval("i", 1)?["result"], "1");

    // Reverse commands in the debug console.
    dap.req(
        "evaluate",
        json!({ "expression": "reverse-step", "frameId": 1, "context": "repl" }),
    )?;
    dap.wait_stop()?;
    assert_eq!(dap.frames()?[0].line, 28);

    // Data breakpoints, forward and backward.
    let total = dap.req(
        "dataBreakpointInfo",
        json!({ "name": "acc.total", "frameId": 1 }),
    )?;
    let body = dap.req(
        "setDataBreakpoints",
        json!({ "breakpoints": [{ "dataId": total["dataId"] }] }),
    )?;
    let watch = breakpoints(&body)[0].id.unwrap();
    dap.req(
        "setBreakpoints",
        json!({ "source": { "path": path }, "breakpoints": [] }),
    )?;
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "data breakpoint");
    assert!(dap.frames()?[0].name.starts_with("add ("));
    dap.console.clear();
    let (_, body) = dap.resume("reverseContinue", None)?;
    assert_eq!(body["reason"], "data breakpoint");
    assert_eq!(body["hitBreakpointIds"], json!([watch]));
    check_output(
        &dap.console,
        r#"
check: acc.total (running backward: stopped before the write)
check: Value before = 0
check: Value after = 1
"#,
    )?;
    assert_eq!(dap.frames()?[0].line, 21);

    // To the beginning of the recording.
    dap.req("setDataBreakpoints", json!({ "breakpoints": [] }))?;
    let (_, body) = dap.resume("reverseContinue", None)?;
    assert_eq!(body["reason"], "beginning of recording");
    assert!(dap.frames()?.is_empty());
    // And to its end, after which the program completes.
    let (_, body) = dap.resume("continue", None)?;
    assert_eq!(body["reason"], "end of recording");
    let (ev, _) = dap.resume("continue", None)?;
    assert_eq!(ev, "terminated");
    let stdout = dap.finish()?;
    assert!(stdout.contains("total = 30 count = 4 counter = 4"));
    Ok(())
}
