//! Recording guest output with `wasmtime_wasi::rr` and replaying it.

use std::io::Write;
use std::sync::{Arc, Mutex};
use test_programs_artifacts::*;
use wasmtime::component::{Component, ResourceTable};
use wasmtime::rr::{ReplayStop, Trace};
use wasmtime::{Engine, RRConfig, Result, Store, format_err};
use wasmtime_wasi::cli::{IsTerminal, StdoutStream};
use wasmtime_wasi::p2::pipe::MemoryOutputPipe;
use wasmtime_wasi::rr::{Output, OutputKind, RecordedOutput, replay_output};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

fn engine(mode: RRConfig) -> Engine {
    test_programs_artifacts::engine(|config| {
        config.rr(mode);
        config.wasm_component_model_async(true);
    })
}

/// Store data for recording. The WASI context is built after recording
/// starts, since it needs the store's event sink.
#[derive(Default)]
struct Host {
    p1: Option<wasmtime_wasi::p1::WasiP1Ctx>,
    wasi: Option<WasiCtx>,
    table: ResourceTable,
}

impl WasiView for Host {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: self.wasi.as_mut().unwrap(),
            table: &mut self.table,
        }
    }
}

/// What a recording wrote to stdout and stderr.
struct Recorded {
    trace: Trace,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Starts recording in a new store, returning it with a WASI context builder
/// whose output is recorded, and the pipes the output also goes to.
fn start_recording(
    args: &[&str],
    async_only: bool,
) -> Result<(
    Store<Host>,
    WasiCtxBuilder,
    MemoryOutputPipe,
    MemoryOutputPipe,
)> {
    let mut store = Store::new(&engine(RRConfig::Recording), Host::default());
    store.start_recording()?;
    let sink = store.rr_event_sink().unwrap();
    let stdout = MemoryOutputPipe::new(1 << 20);
    let stderr = MemoryOutputPipe::new(1 << 20);
    let mut builder = WasiCtxBuilder::new();
    builder.args(args);
    if async_only {
        builder
            .stdout(RecordedOutput::new(
                AsyncOnly(stdout.clone()),
                sink.clone(),
                OutputKind::Stdout,
            ))
            .stderr(RecordedOutput::new(
                AsyncOnly(stderr.clone()),
                sink,
                OutputKind::Stderr,
            ));
    } else {
        builder
            .stdout(RecordedOutput::new(
                stdout.clone(),
                sink.clone(),
                OutputKind::Stdout,
            ))
            .stderr(RecordedOutput::new(
                stderr.clone(),
                sink,
                OutputKind::Stderr,
            ));
    }
    Ok((store, builder, stdout, stderr))
}

/// An output stream with only an `AsyncWrite` implementation, whose WASIp2
/// stream is the default adapter around it.
struct AsyncOnly(MemoryOutputPipe);

impl IsTerminal for AsyncOnly {
    fn is_terminal(&self) -> bool {
        false
    }
}

impl StdoutStream for AsyncOnly {
    fn async_stream(&self) -> Box<dyn tokio::io::AsyncWrite + Send + Sync> {
        self.0.async_stream()
    }
}

fn finish(
    mut store: Store<Host>,
    stdout: MemoryOutputPipe,
    stderr: MemoryOutputPipe,
) -> Result<Recorded> {
    let trace = store.finish_recording()?;
    Ok(Recorded {
        trace,
        stdout: stdout.contents().to_vec(),
        stderr: stderr.contents().to_vec(),
    })
}

async fn record_p1(path: &str, args: &[&str]) -> Result<Recorded> {
    let (mut store, mut builder, stdout, stderr) = start_recording(args, false)?;
    store.data_mut().p1 = Some(builder.build_p1());
    let engine = store.engine().clone();
    let mut linker = wasmtime::Linker::<Host>::new(&engine);
    wasmtime_wasi::p1::add_to_linker_async(&mut linker, |h| h.p1.as_mut().unwrap())?;
    let module = wasmtime::Module::from_file(&engine, path)?;
    let instance = linker.instantiate_async(&mut store, &module).await?;
    let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
    start.call_async(&mut store, ()).await?;
    finish(store, stdout, stderr)
}

async fn record_p2(path: &str, args: &[&str], async_only: bool) -> Result<Recorded> {
    let (mut store, mut builder, stdout, stderr) = start_recording(args, async_only)?;
    store.data_mut().wasi = Some(builder.build());
    let engine = store.engine().clone();
    let mut linker = wasmtime::component::Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    let component = Component::from_file(&engine, path)?;
    let command =
        wasmtime_wasi::p2::bindings::Command::instantiate_async(&mut store, &component, &linker)
            .await?;
    command
        .wasi_cli_run()
        .call_run(&mut store)
        .await?
        .map_err(|()| format_err!("run failed"))?;
    finish(store, stdout, stderr)
}

async fn record_p3(path: &str, args: &[&str]) -> Result<Recorded> {
    let (mut store, mut builder, stdout, stderr) = start_recording(args, false)?;
    store.data_mut().wasi = Some(builder.build());
    let engine = store.engine().clone();
    let mut linker = wasmtime::component::Linker::new(&engine);
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
    wasmtime_wasi::p3::add_to_linker(&mut linker)?;
    let component = Component::from_file(&engine, path)?;
    let command =
        wasmtime_wasi::p3::bindings::Command::instantiate_async(&mut store, &component, &linker)
            .await?;
    store
        .run_concurrent(async move |store| command.wasi_cli_run().call_run(store).await)
        .await??
        .map_err(|()| format_err!("run failed"))?;
    finish(store, stdout, stderr)
}

/// A `Write` whose contents can be inspected.
#[derive(Clone, Default)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl SharedBuf {
    fn take(&self) -> Vec<u8> {
        std::mem::take(&mut self.0.lock().unwrap())
    }
}

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Replays `trace` to the end, returning its output events, and what
/// `replay_output` wrote to stdout and stderr.
async fn replay(trace: &Trace) -> Result<(Vec<Output>, Vec<u8>, Vec<u8>)> {
    let mut store = Store::new(&engine(RRConfig::Replaying), ());
    let mut replayer = store.replayer(trace)?;
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = events.clone();
    replayer.on_event(move |output: Output| observed.lock().unwrap().push(output));
    let (stdout, stderr) = (SharedBuf::default(), SharedBuf::default());
    replay_output(&mut replayer, stdout.clone(), stderr.clone());
    assert_eq!(replayer.run().await?, ReplayStop::Finished);
    let events = std::mem::take(&mut *events.lock().unwrap());
    Ok((events, stdout.take(), stderr.take()))
}

/// Checks that replaying `recorded` produces exactly the recorded output,
/// returning the replayed events.
async fn check_replay(recorded: &Recorded) -> Result<Vec<Output>> {
    let (events, stdout, stderr) = replay(&recorded.trace).await?;
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&recorded.stdout)
    );
    assert_eq!(
        String::from_utf8_lossy(&stderr),
        String::from_utf8_lossy(&recorded.stderr)
    );
    // The events are the same output, split into writes.
    let concat = |kind| {
        events
            .iter()
            .filter(|e| e.kind == kind)
            .flat_map(|e| e.bytes.iter().copied())
            .collect::<Vec<_>>()
    };
    assert_eq!(concat(OutputKind::Stdout), recorded.stdout);
    assert_eq!(concat(OutputKind::Stderr), recorded.stderr);
    assert!(events.iter().all(|e| !e.bytes.is_empty()));
    Ok(events)
}

fn output(kind: OutputKind, bytes: &str) -> Output {
    Output {
        kind,
        bytes: bytes.as_bytes().to_vec(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn rr_p1_stdout_and_stderr() -> Result<()> {
    let recorded = record_p1(P2_CLI_HELLO_STDOUT, &["hello"]).await?;
    assert_eq!(recorded.stdout, b"hello, world\n");
    assert_eq!(recorded.stderr, b"hello, world\n");
    let events = check_replay(&recorded).await?;
    assert_eq!(
        events,
        [
            output(OutputKind::Stdout, "hello, world\n"),
            output(OutputKind::Stderr, "hello, world\n"),
        ]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rr_p1_many_writes() -> Result<()> {
    let recorded = record_p1(P1_CLI_MUCH_STDOUT, &["much", "abc", "50"]).await?;
    assert_eq!(recorded.stdout, "abc".repeat(50).as_bytes());
    let events = check_replay(&recorded).await?;
    assert_eq!(events, vec![output(OutputKind::Stdout, "abc"); 50]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rr_p2_stdout_and_stderr() -> Result<()> {
    let recorded = record_p2(P2_CLI_HELLO_STDOUT_COMPONENT, &["hello"], false).await?;
    assert_eq!(recorded.stdout, b"hello, world\n");
    assert_eq!(recorded.stderr, b"hello, world\n");
    let events = check_replay(&recorded).await?;
    assert_eq!(
        events,
        [
            output(OutputKind::Stdout, "hello, world\n"),
            output(OutputKind::Stderr, "hello, world\n"),
        ]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rr_p2_many_writes() -> Result<()> {
    let recorded = record_p2(P1_CLI_MUCH_STDOUT_COMPONENT, &["much", "abc", "50"], false).await?;
    assert_eq!(recorded.stdout, "abc".repeat(50).as_bytes());
    let events = check_replay(&recorded).await?;
    assert_eq!(events, vec![output(OutputKind::Stdout, "abc"); 50]);
    Ok(())
}

// With an inner stream whose WASIp2 stream is the default adapter around its
// `AsyncWrite`, each write is still recorded once.
#[tokio::test(flavor = "multi_thread")]
async fn rr_p2_default_adapter_records_once() -> Result<()> {
    let recorded = record_p2(P2_CLI_HELLO_STDOUT_COMPONENT, &["hello"], true).await?;
    assert_eq!(recorded.stdout, b"hello, world\n");
    assert_eq!(recorded.stderr, b"hello, world\n");
    let events = check_replay(&recorded).await?;
    assert_eq!(
        events,
        [
            output(OutputKind::Stdout, "hello, world\n"),
            output(OutputKind::Stderr, "hello, world\n"),
        ]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rr_p3_stdout() -> Result<()> {
    let recorded = record_p3(P3_CLI_HELLO_STDOUT_COMPONENT, &["hello"]).await?;
    assert_eq!(recorded.stdout, b"hello, world\n");
    check_replay(&recorded).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn rr_p3_many_writes() -> Result<()> {
    let recorded = record_p3(P3_CLI_MUCH_STDOUT_COMPONENT, &["much", "abc", "50"]).await?;
    assert_eq!(recorded.stdout, "abc".repeat(50).as_bytes());
    check_replay(&recorded).await?;
    Ok(())
}
