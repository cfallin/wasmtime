//! End-to-end tests for the built-in interactive debugger
//! (`wasmtime run --debug-tui`).
//!
//! These tests launch `wasmtime run --debug-tui <port>`, connect to the
//! debugger over TCP, drive it with a script of commands, and check the
//! transcript with FileCheck-style directives.
//!
//! Requirements:
//!   - Built with `--features debug-tui`.
//!   - `WASI_SDK_PATH` set at build time (for the C test program).
//!
//! These are `#[ignore]`d by default, like the gdbstub tests; run them with
//! `cargo test --features debug-tui --test all -- --ignored debug_tui::`.

use filecheck::{CheckerBuilder, NO_VARIABLES};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use test_programs_artifacts::*;
use wasmtime::{Result, bail, format_err};

const PROMPT: &str = "(wasmtime-debug) ";

/// Find the wasmtime binary built alongside the test binary.
fn wasmtime_binary() -> std::path::PathBuf {
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

fn free_port() -> u16 {
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
    fn spawn(wasm: &str, extra_args: &[&str]) -> Result<Tui> {
        let port = free_port();
        let mut cmd = Command::new(wasmtime_binary());
        cmd.arg("run")
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
fn check_output(output: &str, directives: &str) -> Result<()> {
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

fn wat_file(text: &str) -> Result<tempfile::NamedTempFile> {
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
        p $1.next->name
        p $
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
check: $1 = struct Shape { .name =
sameln: "triangle", .color = GREEN, .points = {{ .x = 0, .y = 0 }, { .x = 4, .y = 0 }, { .x = 0, .y = 3 }}, .next =
sameln: .tag = "tri", .scale = 1.5, .flags = 5 }
check: $2 = 4
check: $3 = struct Shape { .name =
sameln: "other", .color = BLUE,
check: $4 =
sameln: "other"
check: $5 =
sameln: "other"
check: $6 = "tri"
check: $7 = 5
check: $8 = GREEN
check: $9 =
sameln: "on the heap"
check: $10 = union Num { .i = 1065353216, .f = 1.0 }
check: $11 = {10, 20, 30, 40, 50}
check: $12 = {0xa, 0x14, 0x1e, 0x28, 0x32}
check: $13 = 43
check: $14 = struct Point { .x = 0, .y = 0 }
check: $15 = (struct Shape *)
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
check: Value returned is $
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
        b debug_tui_rust.rs:66
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
        p $3[1]
        p $$1
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
check: Breakpoint 2, debug_tui_rust::main () at debug_tui_rust.rs:66
check: $1 = "wasmtime"
check: $2 = vec![1, 2, 3, 4]
check: $3 = vec!["a", "bc"]
check: $4 = vec![Shape::Circle(1.5), Shape::Rect { w: 3, h: 4 }, Shape::Empty]
check: $5 = Shape::Rect { w: 3, h: 4 }
check: $6 = 3
check: $7 = Some(7)
check: $8 = None
check: $9 = Err("bad")
check: $10 = Box(Point { x: 5, y: -6 })
check: $11 = -6
check: $12 = Point { x: 5, y: -6 }
check: $13 = Node { value: 1, next: Some(Box(Node { value: 2, next: None })) }
check: $14 = 2
check: $15 = Wrapper(99)
check: $16 = 99
check: $17 = (1, 'x', true)
check: $18 = 'x'
check: $19 = &[2, 3]
check: $20 = 3
check: $21 = HashMap(len=2) {
unordered: "one": 1
unordered: "two": 2
check: $22 = Color::Blue
check: $23 = Point { x: 1, y: 2 }
check: $24 = "hello, world"
check: $25 = String { vec: Vec { buf: RawVec {
check: type = enum debug_tui_rust::Shape {
check:     Circle(f64),
check:     Rect { w: u32, h: u32 },
check:     Empty,
check: }
check: type = core::option::Option<i32>
check: $26 = "bc"
check: $27 = "wasmtime"
check: Breakpoint 3 at
check: Breakpoint 3, debug_tui_rust::describe (shape=&Shape::Rect { w: 3, h: 4 }, color=Color::Blue, point=&Point { x: 1, y: 2 }) at debug_tui_rust.rs:34
check: #0
sameln: debug_tui_rust::describe
check: #1
sameln: debug_tui_rust::main () at debug_tui_rust.rs:66
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

const GC_WAT: &str = r#"
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
        p $local0
        p $local0.2.1
        p $local1[2]
        p $local2
        stepi
        stepi
        wasm-stack
        p $stack1
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
        p $table0[1][3]
        p $global0.0
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
check: $1 = struct (3 fields) { 0: i32 1, 1: f64 2.5, 2: struct (3 fields) { 0: i32 3, 1: f64 4.5, 2: (ref.null any) } }
check: $2 = f64 4.5
check: $3 = i32 7
check: $4 = i31 -42
check: (wasmtime-debug) wasm-stack
check: stack[0]: i32 65
check: stack[1]: i32 3  <- top
check: $5 = i32 3
check: local 3 (tmp): array (len 3) [i32 65, i32 65, i32 65]
check: global 0 (g): struct (3 fields) { 0: i32 1, 1: f64 2.5, 2: struct (3 fields) }
check: Table 0 of instance #0: 3 elements
check: [0] (ref.null any)
check: [1] array (len 4) [i32 7, i32 7, i32 7, i32 7]
check: $6 = i32 7
check: $7 = i32 1
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
