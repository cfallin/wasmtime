//! End-to-end tests for the built-in interactive debugger
//! (`wasmtime run --debug-tui`).
//!
//! These tests launch `wasmtime run --debug-tui <port>`, connect to the
//! debugger over TCP, drive it with a script of commands, and check the
//! transcript with FileCheck-style directives.
//!
//! The debugger is given read-only access to the test programs' sources
//! with `-Ddebug-fs`.
//!
//! Requirements:
//!   - Built with `--features debug-dap-tui`.
//!   - `WASI_SDK_PATH` set at build time (for the C test program).
//!
//! These are `#[ignore]`d by default, like the gdbstub tests; run them with
//! `cargo test --features debug-dap-tui --test all -- --ignored debug_tui::`.

use filecheck::{CheckerBuilder, NO_VARIABLES};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use test_programs_artifacts::*;
use wasmtime::{Result, bail, format_err};

const PROMPT: &str = "(wasmtime-debug) ";

/// Find the wasmtime binary built alongside the test binary.
pub(crate) fn wasmtime_binary() -> std::path::PathBuf {
    let mut me = std::env::current_exe().expect("current_exe specified");
    me.pop(); // chop off file name
    me.pop(); // chop off `deps`
    if cfg!(target_os = "windows") {
        me.push("wasmtime.exe");
    } else {
        me.push("wasmtime");
    }
    me
}

pub(crate) fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A running `wasmtime run --debug-tui` process with a connected client.
struct Tui {
    child: Child,
    stream: TcpStream,
    buf: Vec<u8>,
    transcript: String,
}

impl Tui {
    /// Spawn with read-only access to the test programs' sources (so
    /// that source lines are shown).
    fn spawn(wasm: &str, extra_args: &[&str]) -> Result<Tui> {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/test-programs/src/bin");
        let mut args = vec![format!("-Ddebug-fs={src}")];
        args.extend(extra_args.iter().map(|s| s.to_string()));
        Tui::spawn_raw(wasm, &args)
    }

    fn spawn_raw(wasm: &str, extra_args: &[String]) -> Result<Tui> {
        Tui::spawn_subcommand("run", wasm, extra_args)
    }

    /// Spawn a debugger on a replay of the recorded execution `trace`.
    fn spawn_replay(trace: &str) -> Result<Tui> {
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/test-programs/src/bin");
        Tui::spawn_subcommand("replay", trace, &[format!("-Ddebug-fs={src}")])
    }

    fn spawn_subcommand(subcommand: &str, wasm: &str, extra_args: &[String]) -> Result<Tui> {
        let port = free_port();
        let mut cmd = Command::new(wasmtime_binary());
        cmd.arg(subcommand)
            .arg("-Ccache=n")
            .arg(format!("--debug-tui={port}"))
            .args(extra_args)
            .arg(wasm)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        eprintln!("spawning: {cmd:?}");
        let mut child = cmd.spawn()?;

        // Wait for the readiness message on stderr.
        let mut stderr = BufReader::new(child.stderr.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear();
            if stderr.read_line(&mut line)? == 0 {
                let _ = child.kill();
                bail!("wasmtime exited before the debugger was ready");
            }
            eprintln!("wasmtime stderr: {}", line.trim_end());
            if line.contains("Debugger listening on") {
                break;
            }
        }
        // Keep draining stderr in the background so the child never
        // blocks on a full pipe.
        std::thread::spawn(move || {
            let mut sink = String::new();
            let _ = stderr.read_to_string(&mut sink);
            eprintln!("wasmtime stderr (rest): {sink}");
        });

        let stream = TcpStream::connect(("127.0.0.1", port))?;
        stream.set_read_timeout(Some(Duration::from_secs(120)))?;
        let mut tui = Tui {
            child,
            stream,
            buf: vec![],
            transcript: String::new(),
        };
        let banner = tui.read_until(PROMPT)?;
        tui.transcript.push_str(&banner);
        Ok(tui)
    }

    /// Read until `marker` appears; return everything before it.
    fn read_until(&mut self, marker: &str) -> Result<String> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if let Some(pos) = find(&self.buf, marker.as_bytes()) {
                let out = String::from_utf8_lossy(&self.buf[..pos]).replace("\r\n", "\n");
                self.buf.drain(..pos + marker.len());
                return Ok(out);
            }
            if Instant::now() > deadline {
                bail!(
                    "timed out waiting for {marker:?}; got:\n{}",
                    String::from_utf8_lossy(&self.buf)
                );
            }
            let mut chunk = [0u8; 4096];
            let n = self.stream.read(&mut chunk)?;
            if n == 0 {
                bail!(
                    "connection closed while waiting for {marker:?}; got:\n{}\ntranscript:\n{}",
                    String::from_utf8_lossy(&self.buf),
                    self.transcript
                );
            }
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }

    fn send(&mut self, line: &str) -> Result<()> {
        self.transcript.push_str(PROMPT);
        self.transcript.push_str(line);
        self.transcript.push('\n');
        self.stream.write_all(line.as_bytes())?;
        self.stream.write_all(b"\n")?;
        Ok(())
    }

    /// Run one command and wait for the next prompt.
    fn cmd(&mut self, line: &str) -> Result<String> {
        self.send(line)?;
        let out = self.read_until(PROMPT)?;
        self.transcript.push_str(&out);
        Ok(out)
    }

    /// Run each non-empty line of `script` as a command.
    fn script(&mut self, script: &str) -> Result<()> {
        for line in script.lines() {
            let line = line.trim();
            if !line.is_empty() {
                self.cmd(line)?;
            }
        }
        Ok(())
    }

    /// Detach, wait for the program to finish, and return the
    /// transcript and the program's stdout.
    fn finish(mut self) -> Result<(String, String)> {
        self.send("quit")?;
        let mut rest = String::new();
        let _ = self.stream.read_to_string(&mut rest);
        self.transcript.push_str(&rest.replace("\r\n", "\n"));
        let output = self.child.wait_with_output()?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        eprintln!("--- transcript ---\n{}", self.transcript);
        eprintln!("--- program stdout ---\n{stdout}");
        if !output.status.success() {
            bail!("wasmtime exited with {}", output.status);
        }
        Ok((self.transcript, stdout))
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Validate output against FileCheck-style directives. A `$` in the
/// directives matches a literal `$` (as printed for value-history
/// numbers).
pub(crate) fn check_output(output: &str, directives: &str) -> Result<()> {
    let directives = directives.replace('$', "$$");
    let mut builder = CheckerBuilder::new();
    builder
        .text(&directives)
        .map_err(|e| format_err!("unable to build checker: {e:?}"))?;
    let checker = builder.finish();
    let check = checker
        .explain(output, NO_VARIABLES)
        .map_err(|e| format_err!("{e:?}"))?;
    assert!(
        check.0,
        "didn't pass check {}\n\ntranscript:\n{output}",
        check.1
    );
    Ok(())
}

/// Record an execution of `wasm` with `wasmtime run --record`, returning
/// the directory holding the trace (to keep it alive) and the trace's
/// path.
pub(crate) fn record(wasm: &str) -> Result<(tempfile::TempDir, String)> {
    let dir = tempfile::tempdir()?;
    let trace = dir.path().join("trace").to_str().unwrap().to_string();
    let output = Command::new(wasmtime_binary())
        .args(["run", "-Ccache=n", &format!("--record={trace}"), wasm])
        .output()?;
    if !output.status.success() {
        bail!(
            "recording failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok((dir, trace))
}

pub(crate) fn wat_file(text: &str) -> Result<tempfile::NamedTempFile> {
    let mut f = tempfile::Builder::new().suffix(".wat").tempfile()?;
    f.write_all(text.as_bytes())?;
    Ok(f)
}

#[test]
#[ignore]
fn debug_tui_c_values_and_stepping() -> Result<()> {
    let mut tui = Tui::spawn(DEBUG_TUI_C, &[])?;
    tui.script(
        r#"
        b debug_tui_c.c:60
        c
        bt
        p tri
        p tri.points[1].x
        p *tri.next
        p %1.next->name
        p %
        p tri.tag
        p tri.flags
        p tri.color
        p heap
        p n
        p arr
        p/x arr
        p global_counter
        p origin
        p &tri
        ptype tri
        whatis tri.next
        info locals
        s
        info args
        p s->points[2]
        up
        p r
        down
        finish
        n
        p r
        x/5dw &arr
        x/s heap
        list
        info breakpoints
        delete 1
        c
        "#,
    )?;
    let (transcript, stdout) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: [0] debug_tui_c.wasm
check: (wasmtime-debug) b debug_tui_c.c:60
check: Breakpoint 1 at pc
check: in main at debug_tui_c.c:60
check: Breakpoint 1, main () at debug_tui_c.c:60
check: 60	  int r = area(&tri);
check: #0
check: in main () at debug_tui_c.c:60
check: in _start ()
check: %1 = struct Shape { .name =
sameln: "triangle", .color = GREEN, .points = {{ .x = 0, .y = 0 }, { .x = 4, .y = 0 }, { .x = 0, .y = 3 }}, .next =
sameln: .tag = "tri", .scale = 1.5, .flags = 5 }
check: %2 = 4
check: %3 = struct Shape { .name =
sameln: "other", .color = BLUE,
check: %4 =
sameln: "other"
check: %5 =
sameln: "other"
check: %6 = "tri"
check: %7 = 5
check: %8 = GREEN
check: %9 =
sameln: "on the heap"
check: %10 = union Num { .i = 1065353216, .f = 1.0 }
check: %11 = {10, 20, 30, 40, 50}
check: %12 = {0xa, 0x14, 0x1e, 0x28, 0x32}
check: %13 = 43
check: %14 = struct Point { .x = 0, .y = 0 }
check: %15 = (struct Shape *)
check: type = struct Shape {
check:     const char *name;
check:     enum Color color;
check:     struct Point points[3];
check:     struct Shape *next;
check:     char tag[8];
check:     double scale;
check:     unsigned int flags : 3;
check: }
check: type = struct Shape *
check: (wasmtime-debug) info locals
check: tri = struct Shape
check: heap =
sameln: "on the heap"
check: arr = {10, 20, 30, 40, 50}
check: (wasmtime-debug) s
check: area (s=(struct Shape *)
sameln: at debug_tui_c.c:33
check: 33	  int a = s->points[1].x - s->points[0].x;
check: s = (struct Shape *)
check: = struct Point { .x = 0, .y = 3 }
check: (wasmtime-debug) up
check: #1  main () at debug_tui_c.c:60
check: (wasmtime-debug) finish
check: Run till exit from #0  area
check: main () at debug_tui_c.c:60
check: Value returned is %
sameln: = 6
check: (wasmtime-debug) n
check: 61	  printf(
check: (wasmtime-debug) p r
check: = 6
check: (wasmtime-debug) x/5dw &arr
check: 10 20 30 40
check: 50
check: (wasmtime-debug) x/s heap
check: "on the heap"
check: (wasmtime-debug) list
check: 56	  char *heap = malloc(16);
check: 63	}
check: (wasmtime-debug) info breakpoints
check: 1    breakpoint    1     debug_tui_c.c:60
check: (wasmtime-debug) c
check: [Program exited]
"#,
    )?;
    assert!(stdout.contains("area = 6 on the heap 30"));
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_rust_values() -> Result<()> {
    let mut tui = Tui::spawn(DEBUG_TUI_RUST, &[])?;
    tui.script(
        r#"
        start
        b debug_tui_rust.rs:62
        c
        p name
        p nums
        p words
        p shapes
        p shapes[1]
        p shapes[1].w
        p maybe
        p nothing
        p res
        p boxed
        p boxed.y
        p *boxed
        p list
        p list.next.value
        p w
        p w.0
        p tuple
        p tuple.1
        p slice
        p slice[1]
        p map
        p color
        p point
        p GREETING
        p/r name
        ptype shapes[0]
        whatis maybe
        p %3[1]
        p %%1
        b describe
        c
        bt
        info args
        p *shape
        p point.x
        finish
        c
        "#,
    )?;
    let (transcript, stdout) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Temporary breakpoint 1, debug_tui_rust::main () at debug_tui_rust.rs:39
check: Breakpoint 2, debug_tui_rust::main () at debug_tui_rust.rs:62
check: %1 = "wasmtime"
check: %2 = vec![1, 2, 3, 4]
check: %3 = vec!["a", "bc"]
check: %4 = vec![Shape::Circle(1.5), Shape::Rect { w: 3, h: 4 }, Shape::Empty]
check: %5 = Shape::Rect { w: 3, h: 4 }
check: %6 = 3
check: %7 = Some(7)
check: %8 = None
check: %9 = Err("bad")
check: %10 = Box(Point { x: 5, y: -6 })
check: %11 = -6
check: %12 = Point { x: 5, y: -6 }
check: %13 = Node { value: 1, next: Some(Box(Node { value: 2, next: None })) }
check: %14 = 2
check: %15 = Wrapper(99)
check: %16 = 99
check: %17 = (1, 'x', true)
check: %18 = 'x'
check: %19 = &[2, 3]
check: %20 = 3
check: %21 = HashMap(len=2) {
unordered: "one": 1
unordered: "two": 2
check: %22 = Color::Blue
check: %23 = Point { x: 1, y: 2 }
check: %24 = "hello, world"
check: %25 = String { vec: Vec { buf: RawVec {
check: type = enum debug_tui_rust::Shape {
check:     Circle(f64),
check:     Rect { w: u32, h: u32 },
check:     Empty,
check: }
check: type = core::option::Option<i32>
check: %26 = "bc"
check: %27 = "wasmtime"
check: Breakpoint 3 at
check: Breakpoint 3, debug_tui_rust::describe (shape=&Shape::Rect { w: 3, h: 4 }, color=Color::Blue, point=&Point { x: 1, y: 2 }) at debug_tui_rust.rs:34
check: #0
sameln: debug_tui_rust::describe
check: #1
sameln: debug_tui_rust::main () at debug_tui_rust.rs:62
check: [inlined]
check: shape = &Shape::Rect { w: 3, h: 4 }
check: color = Color::Blue
check: point = &Point { x: 1, y: 2 }
check: = Shape::Rect { w: 3, h: 4 }
check: = 1
check: Run till exit from #0
check: debug_tui_rust::main () at debug_tui_rust.rs:
check: println!(
check: [Program exited]
"#,
    )?;
    assert!(stdout.contains("wasmtime [1, 2, 3, 4]"));
    Ok(())
}

pub(crate) const GC_WAT: &str = r#"
(module $gcdemo
  (type $point (struct (field $x (mut i32)) (field $y f64) (field $next (ref null $point))))
  (type $arr (array (mut i32)))
  (global $g (mut (ref null $point)) (ref.null $point))
  (table $t 3 anyref)
  (memory 1)
  (data (i32.const 16) "hello, memory!")
  (func $use (param $p (ref null $point)) (param $a (ref null $arr)) (param $i i31ref) (result i32)
    (local $tmp anyref)
    (local.set $tmp (array.new $arr (i32.const 65) (i32.const 3)))
    (global.set $g (local.get $p))
    (table.set $t (i32.const 1) (local.get $a))
    local.get $p
    struct.get $point $x
    local.get $a
    i32.const 2
    array.get $arr
    i32.add)
  (func (export "_start")
    (call $use
      (struct.new $point (i32.const 1) (f64.const 2.5)
        (struct.new $point (i32.const 3) (f64.const 4.5) (ref.null $point)))
      (array.new $arr (i32.const 7) (i32.const 4))
      (ref.i31 (i32.const -42)))
    drop)
)
"#;

#[test]
#[ignore]
fn debug_tui_wasm_level_and_gc() -> Result<()> {
    let wat = wat_file(GC_WAT)?;
    let mut tui = Tui::spawn(
        wat.path().to_str().unwrap(),
        &["-Wgc=y,function-references=y"],
    )?;
    tui.script(
        r#"
        wasm-modules
        wasm-functions
        b use
        c
        wasm-instances
        bt
        wasm-locals
        p %local0
        p %local0.2.1
        p %local1[2]
        p %local2
        p $p.2.1
        p $i
        stepi
        stepi
        wasm-stack
        p %stack1
        nexti
        nexti
        nexti
        nexti
        nexti
        nexti
        nexti
        wasm-locals
        wasm-globals
        wasm-table
        p %table0[1][3]
        p %global0.0
        p $g.1
        p $t[1][0]
        wasm-memory
        wasm-memory 16 16
        x/4xb 16
        disas
        finish
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: [0] gcdemo:
sameln: bytes, 2 funcs (0 imported), debug info: no
check: 0  use ((ref null $0), (ref null $1), i31ref) -> (i32)
check: Breakpoint 1, use (p=struct (3 fields), a=array (len 4), i=i31 -42) in gcdemo at pc
check: => 
sameln: : i32.const 65
check: #0  module [0] gcdemo: 2 funcs, 1 memories, 1 tables, 1 globals
check: #0
sameln: in use
check: #1
sameln: in _start
check: local 0 (p): struct (3 fields) { 0: i32 1, 1: f64 2.5, 2: struct (3 fields) }
check: local 1 (a): array (len 4) [i32 7, i32 7, i32 7, i32 7]
check: local 2 (i): i31 -42
check: local 3 (tmp): (ref.null any)
check: %1 = struct (3 fields) { 0: i32 1, 1: f64 2.5, 2: struct (3 fields) { 0: i32 3, 1: f64 4.5, 2: (ref.null any) } }
check: %2 = f64 4.5
check: %3 = i32 7
check: %4 = i31 -42
check: %5 = f64 4.5
check: %6 = i31 -42
check: (wasmtime-debug) wasm-stack
check: stack[0]: i32 65
check: stack[1]: i32 3  <- top
check: %7 = i32 3
check: local 3 (tmp): array (len 3) [i32 65, i32 65, i32 65]
check: global 0 (g): struct (3 fields) { 0: i32 1, 1: f64 2.5, 2: struct (3 fields) }
check: Table 0 of instance #0: 3 elements
check: [0] (ref.null any)
check: [1] array (len 4) [i32 7, i32 7, i32 7, i32 7]
check: %8 = i32 7
check: %9 = i32 1
check: %10 = f64 2.5
check: %11 = i32 7
check: Memory 0 of instance #0: 65536 bytes (1 pages of 65536 bytes)
check: 0x00000010: 68 65 6c 6c 6f 2c 20 6d 65 6d 6f 72 79 21 00 00  |hello, memory!..|
check: 0x00000010: 0x68 0x65 0x6c 0x6c
check: Dump of Wasm code for function use (func 0):
check: => 
sameln: : local.get $p
check: struct.get $point $x
check: Run till exit from #0  use
check: _start () in gcdemo
check: => 
sameln: : drop
check: Wasm results: i32 8
check: [Program exited]
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_interrupt() -> Result<()> {
    let wat = wat_file(
        r#"
        (module
          (func $spin (export "_start")
            (loop $l (br $l))))
        "#,
    )?;
    let mut tui = Tui::spawn(wat.path().to_str().unwrap(), &[])?;
    tui.send("c")?;
    std::thread::sleep(Duration::from_millis(500));
    tui.send("interrupt")?;
    let out = tui.read_until(PROMPT)?;
    tui.transcript.push_str(&out);
    tui.cmd("bt")?;
    tui.cmd("disas")?;
    // Detaching leaves the program running forever; kill it instead.
    let mut child = tui.child;
    child.kill()?;
    child.wait()?;
    eprintln!("--- transcript ---\n{}", tui.transcript);
    check_output(
        &tui.transcript,
        r#"
check: Program interrupted.
check: spin ()
check: #0
sameln: in spin
check: Dump of Wasm code for function spin
check: br $l
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_trap_then_exit() -> Result<()> {
    let wat = wat_file(
        r#"
        (module
          (func $divide (param $a i32) (param $b i32) (result i32)
            (i32.div_s (local.get $a) (local.get $b)))
          (func (export "_start")
            (drop (call $divide (i32.const 10) (i32.const 0)))))
        "#,
    )?;
    let mut tui = Tui::spawn(wat.path().to_str().unwrap(), &[])?;
    tui.script(
        r#"
        c
        bt
        c
        bt
        "#,
    )?;
    tui.send("quit")?;
    let mut rest = String::new();
    let _ = tui.stream.read_to_string(&mut rest);
    let output = tui.child.wait_with_output()?;
    eprintln!("--- transcript ---\n{}", tui.transcript);
    check_output(
        &tui.transcript,
        r#"
check: Program trapped (Wasm trap)
check: divide (a=i32 10, b=i32 0)
check: => 
sameln: i32.div_s
check: #0
sameln: in divide
check: #1
sameln: in _start
check: (wasmtime-debug) c
check: [Program exited]
check: The program is not being run.
"#,
    )?;
    // The trap is still reported, and reflected in the exit status,
    // once the debugger detaches.
    assert!(!output.status.success());
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_watchpoints() -> Result<()> {
    let mut tui = Tui::spawn(DEBUG_TUI_C_HISTORY, &[])?;
    tui.script(
        r#"
        b add
        c
        watch counter
        watch acc->total
        watch %local0
        watch -m &acc->count 4
        info breakpoints
        c
        c
        c
        delete 2 3
        c
        c
        p acc->count
        delete
        info breakpoints
        "#,
    )?;
    let (transcript, stdout) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Breakpoint 1, add (acc=
sameln: v=1)
check: (wasmtime-debug) watch counter
check: Watchpoint 2: counter (memory
sameln: length 0x4)
check: (wasmtime-debug) watch acc->total
check: Watchpoint 3: acc->total (memory
check: (wasmtime-debug) watch %local0
check: Only values in linear memory can be watched; `%local0` is in local 0.
check: (wasmtime-debug) watch -m &acc->count 4
check: Watchpoint 4: memory
sameln: length 0x4
check: (wasmtime-debug) info breakpoints
check: 1    breakpoint    1     add
check: 2    watchpoint    0     counter
check: 3    watchpoint    0     acc->total
check: 4    watchpoint    0     -m
check: (wasmtime-debug) c
check: Watchpoint 3: acc->total
check: Old value = 0
check: New value = 1
check: add (acc=
check: 22	  acc->count++;
check: (wasmtime-debug) c
check: Watchpoint 4: -m
check: Old value = 0 (0x0)
check: New value = 1 (0x1)
check: 23	  counter++;
check: (wasmtime-debug) c
check: Watchpoint 2: counter
check: Old value = 0
check: New value = 1
check: 24	}
check: (wasmtime-debug) delete 2 3
check: (wasmtime-debug) c
check: Breakpoint 1, add (acc=
sameln: v=4)
check: (wasmtime-debug) c
check: Watchpoint 4:
check: Old value = 1 (0x1)
check: New value = 2 (0x2)
check: (wasmtime-debug) p acc->count
check: = 2
check: (wasmtime-debug) info breakpoints
check: No breakpoints or watchpoints.
"#,
    )?;
    assert!(stdout.contains("total = 30 count = 4 counter = 4"));
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_reverse_requires_replay() -> Result<()> {
    let mut tui = Tui::spawn(DEBUG_TUI_C_HISTORY, &[])?;
    tui.script(
        r#"
        start
        reverse-continue
        rn
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: (wasmtime-debug) reverse-continue
check: Reverse execution requires a recorded execution
check: (wasmtime-debug) rn
check: Reverse execution requires a recorded execution
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_reverse_execution() -> Result<()> {
    let (_dir, trace) = record(DEBUG_TUI_C_HISTORY)?;
    let mut tui = Tui::spawn_replay(&trace)?;
    tui.script(
        r#"
        rc
        b debug_tui_c_history.c:29
        c
        c
        p i
        rs
        rs
        bt
        rs
        rs
        rs
        rs
        rs
        rs
        rs
        rn
        rn
        p i
        c
        n
        n
        rn
        rn
        rni
        rsi
        rsi
        b square
        c
        c
        bt
        reverse-finish
        bt
        delete
        watch acc.total
        c
        rc
        rc
        rc
        delete
        c
        rc
        start
        p acc
        c
        c
        "#,
    )?;
    let (transcript, stdout) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: (wasmtime-debug) rc
check: No more reverse-execution history: at the beginning of the recording.
check: Breakpoint 1, main () at debug_tui_c_history.c:29
check: Breakpoint 1, main () at debug_tui_c_history.c:29
check: (wasmtime-debug) p i
check: = 2

From line 29 of the second iteration, back through the `for` and into
the call of `add` made by the first iteration's line 30.
check: (wasmtime-debug) rs
check: 28	  for (int i = 1; i <= 4; i++) {
check: (wasmtime-debug) rs
check: add (acc=
sameln: v=1) at debug_tui_c_history.c:24
check: 24	}
check: (wasmtime-debug) bt
check: #0
sameln: in add
check: #1
sameln: in main () at debug_tui_c_history.c:30
check: (wasmtime-debug) rs
check: 23	  counter++;
check: (wasmtime-debug) rs
check: 22	  acc->count++;
check: (wasmtime-debug) rs
check: 21	  acc->total += v;

Back out of `add` (past its prologue) to its call, at the call
instruction; then to the beginning of that line, into `square`, and
back out of it.
check: (wasmtime-debug) rs
check: main () at debug_tui_c_history.c:30
check: 30	    add(&acc, sq);
check: (wasmtime-debug) rs
check: 30	    add(&acc, sq);
check: (wasmtime-debug) rs
check: square (x=1) at debug_tui_c_history.c:17
check: 17	  return r;
check: (wasmtime-debug) rs
check: 16	  int r = x * x;
check: (wasmtime-debug) rn
check: main () at debug_tui_c_history.c:29
check: (wasmtime-debug) rn
check: 29	    int sq = square(i);
check: (wasmtime-debug) p i
check: = 1

Forward again, then back over calls.
check: (wasmtime-debug) c
check: Breakpoint 1, main () at debug_tui_c_history.c:29
check: (wasmtime-debug) n
check: 30	    add(&acc, sq);
check: (wasmtime-debug) n
check: 28	  for (int i = 1; i <= 4; i++) {
check: (wasmtime-debug) rn
check: 30	    add(&acc, sq);
check: (wasmtime-debug) rn
check: 29	    int sq = square(i);
check: (wasmtime-debug) rni
check: => 
check: (wasmtime-debug) rsi
check: => 
check: (wasmtime-debug) rsi
check: => 

Back to the call of `square`.
check: (wasmtime-debug) b square
check: Breakpoint 2 at
check: Breakpoint 1, main ()
check: Breakpoint 2, square (x=2)
check: (wasmtime-debug) reverse-finish
check: Run back to call of #0  square (x=
check: main () at debug_tui_c_history.c:29
check: 29	    int sq = square(i);
check: (wasmtime-debug) bt
check: #0
sameln: in main ()
check: #1
sameln: in _start

Watchpoints, forward and backward.
check: (wasmtime-debug) watch acc.total
check: Watchpoint 3: acc.total
check: (wasmtime-debug) c
check: Watchpoint 3: acc.total
check: Old value = 1
check: New value = 5
check: (wasmtime-debug) rc
check: Watchpoint 3: acc.total (running backward: stopped before the write)
check: Value before = 1
check: Value after = 5
check: 21	  acc->total += v;
check: (wasmtime-debug) rc
check: Watchpoint 3: acc.total (running backward: stopped before the write)
check: Value before = 0
check: Value after = 1
check: (wasmtime-debug) rc
check: Watchpoint 3: acc.total (running backward: stopped before the write)
check: Value before = 0
check: Value after = 0
check: 27	  struct Acc acc = {0, 0};

To the end of the recording, back to the beginning, and through again.
check: (wasmtime-debug) c
check: No more execution history: at the end of the recording; continuing will complete it.
check: (wasmtime-debug) rc
check: No more reverse-execution history: at the beginning of the recording.
check: (wasmtime-debug) start
check: Temporary breakpoint 4, main ()
check: (wasmtime-debug) p acc
check: (wasmtime-debug) c
check: No more execution history
check: (wasmtime-debug) c
check: [Program exited]
"#,
    )?;
    assert!(stdout.contains("total = 30 count = 4 counter = 4"));
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_source_requires_debug_fs() -> Result<()> {
    // Without `-Ddebug-fs`, the debugger cannot read source files.
    let mut tui = Tui::spawn_raw(DEBUG_TUI_C, &[])?;
    tui.script(
        r#"
        b debug_tui_c.c:60
        c
        list
        p tri.points[1].x
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Breakpoint 1, main () at debug_tui_c.c:60
check: 60	in
sameln: debug_tui_c.c (source unavailable)
check: Cannot read source file
check: -Ddebug-fs=<dir>
check: = 4
check: [Program exited]
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_instruction_stepping() -> Result<()> {
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
            drop
            i32.const 4
            i32.const 5
            call $add
            drop))
        "#,
    )?;
    let mut tui = Tui::spawn(wat.path().to_str().unwrap(), &[])?;
    tui.script(
        r#"
        b main
        c
        stepi
        wasm-stack
        stepi
        wasm-stack
        stepi
        bt
        stepi
        stepi
        stepi
        stepi
        wasm-stack
        stepi
        stepi
        wasm-stack
        stepi
        stepi
        stepi
        nexti
        wasm-stack
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Breakpoint 1, main () in steps
check: : i32.const 2
check: (wasmtime-debug) stepi
check: : i32.const 3
check: stack[0]: i32 2  <- top
check: (wasmtime-debug) stepi
check: : call $add
check: stack[0]: i32 2
nextln: stack[1]: i32 3  <- top
check: (wasmtime-debug) stepi
check: add (a=i32 2, b=i32 3) in steps
check: : local.get $a
check: #0
sameln: in add
check: #1
sameln: in main
check: (wasmtime-debug) stepi
check: : local.get $b
check: (wasmtime-debug) stepi
check: : i32.add
check: (wasmtime-debug) stepi
check: : end
check: (wasmtime-debug) stepi
check: main () in steps
check: : i32.const 10
check: stack[0]: i32 5  <- top
check: (wasmtime-debug) stepi
check: : i32.mul
check: (wasmtime-debug) stepi
check: : drop
check: stack[0]: i32 50  <- top
check: (wasmtime-debug) stepi
check: : i32.const 4
check: (wasmtime-debug) stepi
check: : i32.const 5
check: (wasmtime-debug) stepi
check: : call $add
check: (wasmtime-debug) nexti
check: : drop
check: stack[0]: i32 9  <- top
check: [Program exited]
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_c_instruction_stepping() -> Result<()> {
    // Instruction-level stepping within source code: each step shows the
    // current source line and instruction, and stepping into a call
    // enters the callee.
    let mut tui = Tui::spawn(DEBUG_TUI_C, &[])?;
    tui.script(
        r#"
        b debug_tui_c.c:60
        c
        si
        si
        si
        si
        si
        bt
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Breakpoint 1, main () at debug_tui_c.c:60
check: (wasmtime-debug) si
check: 60	  int r = area(&tri);
check: : local.get 0
check: (wasmtime-debug) si
check: : i32.const 136
check: (wasmtime-debug) si
check: : i32.add
check: (wasmtime-debug) si
check: : call $area
check: (wasmtime-debug) si
check: area (
sameln: at debug_tui_c.c:32
check: 32	int area(struct Shape *s) {
check: #0
sameln: in area
check: #1
sameln: in main () at debug_tui_c.c:60
check: [Program exited]
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_named_wasm_entities() -> Result<()> {
    // `$name` refers to entities named in the name section; `%localN`
    // etc. refer to entities by index. A local deliberately named
    // `$local0` (which is local 2) checks that the two don't collide.
    let wat = wat_file(
        r#"
        (module $fact
          (global $calls (mut i32) (i32.const 0))
          (table $tab 2 funcref)
          (func $fact (param $n i32) (result i32)
            (local $acc i32)
            (local $local0 i32)
            (local.set $local0 (i32.const 99))
            (local.set $acc (i32.const 1))
            (block $done
              (loop $top
                (br_if $done (i32.eqz (local.get $n)))
                (local.set $acc (i32.mul (local.get $acc) (local.get $n)))
                (local.set $n (i32.sub (local.get $n) (i32.const 1)))
                (global.set $calls (i32.add (global.get $calls) (i32.const 1)))
                (br $top)))
            (local.get $acc))
          (func (export "_start") (drop (call $fact (i32.const 5)))))
        "#,
    )?;
    let mut tui = Tui::spawn(wat.path().to_str().unwrap(), &[])?;
    tui.script(
        r#"
        b fact
        c
        p $n
        p $acc
        p $local0
        p %local0
        p %local2
        stepi
        stepi
        stepi
        stepi
        p $local0
        p $acc
        p $calls
        p $tab
        p %pc
        p $nope
        p %nope
        finish
        p %
        p $calls
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Breakpoint 1, fact (n=i32 5)
check: %1 = i32 5
check: %2 = i32 0
check: %3 = i32 0
check: %4 = i32 5
check: %5 = i32 0
check: : local.set $acc
check: %6 = i32 99
check: %7 = i32 1
check: %8 = i32 0
check: %9 = table (len 2)
check: %10 = i32
check: No Wasm local, global or table named `$nope`
check: Unknown built-in `%nope`
check: Wasm results: i32 120
check: (wasmtime-debug) p %
check: %11 = i32
check: %12 = i32 5
check: [Program exited]
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_rust_std_containers() -> Result<()> {
    let mut tui = Tui::spawn(DEBUG_TUI_RUST_STD, &[])?;
    tui.script(
        r#"
        b consume
        c
        up
        info locals
        p arc.id
        p *arc
        p refcell[1]
        p deque[3]
        p linked_list[2]
        p mutex.label
        p nested[0]
        p/r cell
        p GLOBAL_LAZY
        p GLOBAL_LAZY_UNFORCED
        p boxed_dyn.id
        p mutex_guard[1]
        p rc_slice[1]
        p *s.ptr
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: (wasmtime-debug) info locals
check: vec = vec![10, 20, 30]
check: deque = VecDeque[0, 3, 4, 5]
check: hash_map = HashMap(len=2) {
unordered: "a": 1
unordered: "b": 2
check: btree_map = BTreeMap(len=20) { 0: "even", 1: "odd", 2: "even",
sameln: 18: "even", 19: "odd" }
check: hash_set = HashSet(len=3) {
check: btree_set = BTreeSet(len=3) { 'x', 'y', 'z' }
check: tokens = vec![Token::Num(-5), Token::Ident("foo"), Token::Pair { a: 1, b: 2 }, Token::Eof]
check: string = "hello"
check: cow_borrowed = Cow::Borrowed("borrowed")
check: cow_owned = Cow::Owned("owned")
check: cow_slice = Cow::Borrowed(&[1, 2, 3])
check: boxed = Box(Inner { id: 1, label: "boxed" })
check: boxed_slice = Box([4, 5, 6])
check: boxed_str = Box("boxed str")
check: mutex = Mutex(Inner { id: 2, label: "mutex" })
check: rwlock = RwLock(vec![1, 2])
check: arc = Arc(Inner { id: 3, label: "arc" }) [strong=2, weak=0]
check: arc2 = Arc(Inner { id: 3, label: "arc" }) [strong=2, weak=0]
check: rc = Rc("rc") [strong=1, weak=1]
check: weak = Weak("rc") [strong=1, weak=1]
check: refcell = RefCell(vec![1, -1])
check: cell = Cell(42)
check: unsafe_cell = UnsafeCell(7)
check: once_cell = OnceCell(99)
check: option_box = Some(Box(Inner { id: 4, label: "opt" }))
check: result = Ok(vec![1])
check: manually_drop = ManuallyDrop(Inner { id: 5, label: "md" })
check: wrapping = Wrapping(250)
check: non_zero = 17
check: range = Range { start: 3, end: 9 }
check: linked_list = LinkedList[100, 200, 300]
check: binary_heap = BinaryHeap[8, 1, 5]
check: arc_mutex = Arc(Mutex(HashMap(len=1) { 1: 'a' })) [strong=1, weak=0]
check: nested = vec![Some(Rc(RefCell(Token::Num(3))) [strong=1, weak=0]), None]
check: array = [1, 2, 3]
check: tuple = (1, "two", 3.0)
check: once_lock = OnceLock("set once")
check: once_lock_empty = OnceLock(<uninit>)
check: lazy_cell = LazyCell(42)
check: lazy_cell_unforced = LazyCell(<uninit>)
check: mutex_guard = MutexGuard(vec![3, 4])
check: read_guard = RwLockReadGuard(5)
check: ref_guard = Ref(Token::Ident("cell"))
check: arc_str = Arc("arc str") [strong=1, weak=0]
check: rc_slice = Rc([11, 12]) [strong=1, weak=0]
check: rc_str = Rc("rc str") [strong=1, weak=0]
check: boxed_dyn = Box(Inner { id: 6, label: "dyn" })
check: ref_dyn = &(1, "two", 3.0)
check: custom = MyCustomPointer { _inner: Box(vec![1, 2, 3]) }
check: s = S { ptr: &123, name: "s" }
check: %1 = 3
check: %2 = Inner { id: 3, label: "arc" }
check: %3 = -1
check: %4 = 5
check: %5 = 300
check: %6 = "mutex"
check: %7 = Some(Rc(RefCell(Token::Num(3))) [strong=1, weak=0])
check: %8 = Cell { value: UnsafeCell { value: 42 } }
check: %9 = LazyLock(vec![7, 8, 9])
check: %10 = LazyLock(<uninit>)
check: %11 = 6
check: %12 = 4
check: %13 = 12
check: %14 = 123
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_cpp_std() -> Result<()> {
    let mut tui = Tui::spawn(DEBUG_TUI_CPP, &[])?;
    tui.script(
        r#"
        b consume
        c
        up
        info locals
        p vec[1]
        p points[1].y
        p long_str[2]
        p *uptr
        p uptr->y
        p sptr->x
        p sptr2.y
        p cstr
        p weird
        p weird.not_a_vector
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: (wasmtime-debug) info locals
check: vec = std::vector of length 3, capacity 3 = {1, 2, 3}
check: points = std::vector of length 2, capacity 2 = {{ .x = 1, .y = 2 }, { .x = 3, .y = 4 }}
check: short_str = "hi"
check: long_str = "a string too long for the small-string buffer"
check: map = std::map with 3 elements = {[1] = "one", [2] = "two", [3] = "three"}
check: set = std::set with 4 elements = {1, 3, 5, 9}
check: umap = std::unordered_map with 2 elements = {
unordered: ["x"] = 10
unordered: ["y"] = 20
check: uset = std::unordered_set with 2 elements = {
check: uptr = std::unique_ptr(
sameln: ) = { .x = 5, .y = 6 }
check: null_uptr = std::unique_ptr(nullptr)
check: sptr = std::shared_ptr(use_count=2, weak_count=0) = { .x = 7, .y = 8 }
check: sptr2 = std::shared_ptr(use_count=2, weak_count=0) = { .x = 7, .y = 8 }
check: cstr =
sameln: "a C string"
check: %1 = 2
check: %2 = 4
check: %3 = 115 's'
check: %4 = Point { .x = 5, .y = 6 }
check: %5 = 6
check: %6 = 7
check: %7 = 8
check: %8 =
sameln: "a C string"
check: %9 = std::vector<Weird, std::allocator<Weird> > { .not_a_vector = 7 }
check: %10 = 7
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn debug_tui_locations() -> Result<()> {
    // Optimized code keeps some variables in Wasm locals.
    let mut tui = Tui::spawn(DEBUG_TUI_C_OPT, &[])?;
    tui.script(
        r#"
        b checksum
        c
        location data
        location n
        p &n
        location data[2]
        location global_buf
        location global_buf[3]
        p &global_buf
        x/16xb &global_buf
        x/16cb global_buf
        location %local1
        location %global0
        location %pc
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: Breakpoint 1, checksum (data=
check: (wasmtime-debug) location data
nextln: local 0
check: (wasmtime-debug) location n
nextln: local 1
check: cannot take the address of a value in local 1
check: (wasmtime-debug) location data[2]
nextln: memory
sameln: length 0x1
check: (wasmtime-debug) location global_buf
nextln: memory
sameln: length 0x10
check: (wasmtime-debug) location global_buf[3]
nextln: memory
sameln: length 0x1
check: %1 = (unsigned char [16] *)
check: 0x30 0x31 0x32 0x33 0x34 0x35 0x36 0x37 0x38 0x39 0x61 0x62 0x63 0x64 0x65 0x66
check: '0' '1' '2' '3' '4' '5' '6' '7' '8' '9' 'a' 'b' 'c' 'd' 'e' 'f'
check: (wasmtime-debug) location %local1
nextln: local 1
check: (wasmtime-debug) location %global0
nextln: global 0
check: `%pc` is not stored
check: [Program exited]
"#,
    )?;

    // Unoptimized code, and Wasm-level names.
    let mut tui = Tui::spawn(DEBUG_TUI_C, &[])?;
    tui.script(
        r#"
        b debug_tui_c.c:60
        c
        location tri
        location tri.points[1]
        location tri.flags
        p &tri.points[1]
        x/2dw &tri.points[1]
        location %1
        c
        "#,
    )?;
    let (transcript, _) = tui.finish()?;
    check_output(
        &transcript,
        r#"
check: (wasmtime-debug) location tri
nextln: memory
sameln: length 0x40
check: (wasmtime-debug) location tri.points[1]
nextln: memory
sameln: length 0x8
check: (wasmtime-debug) location tri.flags
nextln: memory
sameln: length 0x1
check: %1 = (struct Point *)
check: (wasmtime-debug) x/2dw &tri.points[1]
nextln: 4 0
check: (wasmtime-debug) location %1
nextln: the value has no storage location
"#,
    )?;
    Ok(())
}

macro_rules! assert_test_exists {
    ($name:ident) => {
        #[expect(unused_imports, reason = "here to assert the test exists")]
        use self::$name as _;
    };
}
foreach_debug_tui!(assert_test_exists);

// The test programs are exercised by the tests above.
#[expect(
    dead_code,
    reason = "test-program names checked by `assert_test_exists`"
)]
fn debug_tui_c() {}
#[expect(
    dead_code,
    reason = "test-program names checked by `assert_test_exists`"
)]
fn debug_tui_rust() {}
#[expect(
    dead_code,
    reason = "test-program names checked by `assert_test_exists`"
)]
fn debug_tui_rust_std() {}
#[expect(
    dead_code,
    reason = "test-program names checked by `assert_test_exists`"
)]
fn debug_tui_cpp() {}
#[expect(
    dead_code,
    reason = "test-program names checked by `assert_test_exists`"
)]
fn debug_tui_c_opt() {}
#[expect(
    dead_code,
    reason = "test-program names checked by `assert_test_exists`"
)]
fn debug_tui_c_history() {}
