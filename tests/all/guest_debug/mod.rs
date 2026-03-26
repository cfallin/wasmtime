//! Integration tests for guest-debug support (gdbstub + LLDB).
//!
//! These tests launch `wasmtime run` or `wasmtime serve` with `-g <port>`,
//! connect LLDB via the wasm remote protocol, execute debug scripts
//! (set breakpoints, continue, inspect variables, etc.), and validate output.
//!
//! Requirements:
//!   - `/opt/wasi-sdk/bin/lldb` (LLDB with wasm plugin support)
//!   - `WASI_SDK_PATH` env var set (for C test programs)
//!   - Built with `--features gdbstub`

use filecheck::{CheckerBuilder, NO_VARIABLES};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use test_programs_artifacts::*;
use wasmtime::{Result, bail, format_err};

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

/// Find an available TCP port by binding to port 0.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Path to the wasm-aware LLDB.
fn lldb_path() -> String {
    std::env::var("LLDB").unwrap_or("/opt/wasi-sdk/bin/lldb".to_string())
}

/// The readiness marker printed by the gdbstub to stderr.
const GDBSTUB_READY_MARKER: &str = "Debugger listening on";

/// A running wasmtime process with a gdbstub endpoint.
struct WasmtimeWithGdbstub {
    child: Child,
    /// Keeps the stderr pipe alive to avoid SIGPIPE on the child.
    /// Also used by serve tests to read the HTTP address.
    stderr_reader: BufReader<std::process::ChildStderr>,
}

impl WasmtimeWithGdbstub {
    /// Spawn wasmtime with `-D inherit-stderr` and wait for stderr to
    /// contain the gdbstub readiness marker.
    fn spawn(
        subcmd: &str,
        gdbstub_port: u16,
        extra_args: &[&str],
        timeout: Duration,
    ) -> Result<Self> {
        let mut cmd = Command::new(wasmtime_binary());
        cmd.arg(subcmd)
            .arg(format!("-g{gdbstub_port}"))
            .arg("-Dinherit-stderr")
            .args(extra_args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        eprintln!("spawning: {cmd:?}");
        let mut child = cmd.spawn()?;

        let stderr = child.stderr.take().unwrap();
        let mut reader = BufReader::new(stderr);
        let deadline = std::time::Instant::now() + timeout;
        let mut line = String::new();
        loop {
            if std::time::Instant::now() > deadline {
                child.kill().ok();
                bail!("timed out waiting for gdbstub readiness");
            }
            line.clear();
            reader.read_line(&mut line)?;
            eprintln!("wasmtime stderr: {}", line.trim_end());
            if line.contains(GDBSTUB_READY_MARKER) {
                return Ok(Self {
                    child,
                    stderr_reader: reader,
                });
            }
            if line.is_empty() {
                child.kill().ok();
                let status = child.wait()?;
                bail!("wasmtime exited ({status}) without readiness marker");
            }
        }
    }

    /// Read stderr lines until one contains `marker`, returning it.
    fn wait_for_stderr(&mut self, marker: &str, timeout: Duration) -> Result<String> {
        let deadline = std::time::Instant::now() + timeout;
        let mut line = String::new();
        loop {
            if std::time::Instant::now() > deadline {
                bail!("timed out waiting for '{marker}' on stderr");
            }
            line.clear();
            self.stderr_reader.read_line(&mut line)?;
            eprintln!("wasmtime stderr: {}", line.trim_end());
            if line.contains(marker) {
                return Ok(line);
            }
            if line.is_empty() {
                bail!("wasmtime stderr closed before finding '{marker}'");
            }
        }
    }
}

/// Run an LLDB debug script against a gdbstub endpoint.
///
/// Connects LLDB to `127.0.0.1:<port>` using the wasm plugin, executes
/// the given script commands, and returns LLDB's stdout.
fn lldb_with_gdbstub_script(port: u16, script: &str) -> Result<String> {
    let _ = env_logger::try_init();

    let mut cmd = Command::new(lldb_path());
    cmd.arg("--batch");
    // Connect to the gdbstub
    cmd.arg("-o").arg(format!(
        "process connect --plugin wasm connect://127.0.0.1:{port}"
    ));
    // Add each script line as a separate -o argument
    for line in script.lines() {
        let line = line.trim();
        if !line.is_empty() {
            cmd.arg("-o").arg(line);
        }
    }

    eprintln!("Running LLDB: {cmd:?}");
    let output = cmd.output()?;

    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8(output.stderr)?;
    eprintln!("--- LLDB stdout ---\n{stdout}");
    eprintln!("--- LLDB stderr ---\n{stderr}");

    // LLDB may exit non-zero when the process exits; that's fine for our
    // purposes as long as we got output.
    Ok(stdout)
}

/// Validate output against FileCheck-style directives.
fn check_output(output: &str, directives: &str) -> Result<()> {
    let mut builder = CheckerBuilder::new();
    builder
        .text(directives)
        .map_err(|e| format_err!("unable to build checker: {e:?}"))?;
    let checker = builder.finish();
    let check = checker
        .explain(output, NO_VARIABLES)
        .map_err(|e| format_err!("{e:?}"))?;
    assert!(check.0, "didn't pass check {}", check.1);
    Ok(())
}

// ---- CLI (wasmtime run) tests ----

#[test]
#[ignore]
fn guest_debug_cli_fib_continue() -> Result<()> {
    let port = free_port();
    let mut wt = WasmtimeWithGdbstub::spawn(
        "run",
        port,
        &[
            "-Ccache=n",
            "-Ddebug-info",
            "-Oopt-level=0",
            GUEST_DEBUG_FIB,
        ],
        Duration::from_secs(30),
    )?;

    let output = lldb_with_gdbstub_script(
        port,
        r#"
b fib
c
fr v
c
"#,
    )?;
    wt.child.kill().ok();
    wt.child.wait()?;

    check_output(
        &output,
        r#"
check: stop reason
check: fib
check: n =
"#,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn guest_debug_cli_fib_step() -> Result<()> {
    let port = free_port();
    let mut wt = WasmtimeWithGdbstub::spawn(
        "run",
        port,
        &[
            "-Ccache=n",
            "-Ddebug-info",
            "-Oopt-level=0",
            GUEST_DEBUG_FIB,
        ],
        Duration::from_secs(30),
    )?;

    let output = lldb_with_gdbstub_script(
        port,
        r#"
b fib
c
n
n
n
fr v
c
"#,
    )?;
    wt.child.kill().ok();
    wt.child.wait()?;

    check_output(
        &output,
        r#"
check: stop reason
check: fib
"#,
    )?;
    Ok(())
}

// ---- HTTP (wasmtime serve) tests ----

/// Start serve under debugger, connect LLDB, continue, make HTTP request.
#[test]
#[ignore]
fn guest_debug_serve_hello_with_request() -> Result<()> {
    let gdb_port = free_port();
    let http_component = guest_debug_test_programs_artifact::HTTP_HELLO_COMPONENT;

    let mut wt = WasmtimeWithGdbstub::spawn(
        "serve",
        gdb_port,
        &["--addr=127.0.0.1:0", "-Scli", http_component],
        Duration::from_secs(30),
    )?;

    // Connect LLDB and continue in background so the HTTP server starts.
    let lldb_handle = std::thread::spawn(move || lldb_with_gdbstub_script(gdb_port, "c\n"));

    // Read the HTTP address from wasmtime's stderr.
    let line = wt.wait_for_stderr("Serving HTTP", Duration::from_secs(15))?;
    let http_addr = line
        .find("127.0.0.1")
        .and_then(|start| {
            let addr = &line[start..];
            let end = addr.find('/')?;
            addr[..end].parse::<SocketAddr>().ok()
        })
        .ok_or_else(|| format_err!("failed to parse HTTP address from: {line}"))?;
    eprintln!("HTTP address: {http_addr}");

    // Make a raw HTTP request.
    let mut tcp = TcpStream::connect_timeout(&http_addr, Duration::from_secs(5))?;
    tcp.set_read_timeout(Some(Duration::from_secs(5)))?;
    write!(tcp, "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
    let mut response = String::new();
    let _ = std::io::Read::read_to_string(&mut tcp, &mut response);

    assert!(
        response.contains("Hello from guest-debug test!"),
        "unexpected response: {response}"
    );

    wt.child.kill().ok();
    let _ = lldb_handle.join();
    wt.child.wait()?;
    Ok(())
}
