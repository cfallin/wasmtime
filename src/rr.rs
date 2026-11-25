//! Support for recording executions (`-R`) and `wasmtime replay`.

use crate::common::RunCommon;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmtime::rr::{
    Checksum, EventReceiver, EventSink, Trace, TraceEvent, checksum_hex, core_modules,
    event_channel, record_event,
};
use wasmtime::{Engine, Result, Store, bail, error::Context as _, format_err};
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::rr::{OutputKind, RecordedOutput};

/// Core module bytecode by checksum.
pub(crate) type BundleModules = BTreeMap<Checksum, Arc<[u8]>>;

/// How a recorded program exited, recorded at the end of `-R` traces so
/// that `wasmtime replay` exits the same way.
#[derive(serde_derive::Serialize, serde_derive::Deserialize, Debug, Clone)]
pub struct RecordedExit {
    /// The WASI exit code, if the program exited explicitly.
    code: Option<i32>,
    /// Whether the program trapped.
    trap: bool,
    /// The error the program failed with, if any, as Wasmtime printed it.
    error: Option<String>,
}

impl TraceEvent for RecordedExit {
    const TAG: u32 = 0x8000_0100;
}

/// Creates a store, which records its execution and the events sent to
/// `receiver` if there is one.
pub fn new_store<T: 'static>(
    engine: &Engine,
    data: T,
    receiver: Option<EventReceiver>,
) -> Result<Store<T>> {
    let Some(receiver) = receiver else {
        return Ok(Store::new(engine, data));
    };
    let mut store = Store::new_recording(engine, data)?;
    store.rr_attach_events(receiver)?;
    Ok(store)
}

/// Records WASI stdout and stderr (when inherited) through `sink`.
pub fn record_output(builder: &mut WasiCtxBuilder, sink: &EventSink, stdout: bool, stderr: bool) {
    if stdout {
        builder.stdout(RecordedOutput::new(
            wasmtime_wasi::cli::stdout(),
            sink.clone(),
            OutputKind::Stdout,
        ));
    }
    if stderr {
        builder.stderr(RecordedOutput::new(
            wasmtime_wasi::cli::stderr(),
            sink.clone(),
            OutputKind::Stderr,
        ));
    }
}

/// The name of the trace in a trace bundle directory.
const BUNDLE_TRACE: &str = "trace";

/// The directory of a trace bundle holding the bytecode of the modules the
/// trace instantiates, each in `<checksum>.wasm`.
const BUNDLE_MODULES: &str = "modules";

/// Where `-R` writes a recording.
#[derive(Clone)]
pub struct RecordTarget {
    /// The trace file or, with `-R bundle`, the bundle directory. Nothing is
    /// written if this is empty.
    pub path: PathBuf,
    /// With `-R bundle`, the bytecode of the program's modules.
    bundle: Option<BundleModules>,
}

impl RunCommon {
    /// Whether `-R` records the execution.
    pub fn recording(&self) -> bool {
        self.common.record.path.is_some()
    }

    /// Creates a channel through which to record output, if `-R` records the
    /// execution.
    pub fn record_channel(&self) -> (Option<EventSink>, Option<EventReceiver>) {
        self.recording().then(event_channel).unzip()
    }

    /// Where `-R` writes the recording, once the program has been loaded.
    pub fn record_target(&self) -> Option<RecordTarget> {
        let path = self.common.record.path.as_ref()?;
        Some(RecordTarget {
            path: path.into(),
            bundle: (self.common.record.bundle == Some(true))
                .then(|| self.bundle_modules.lock().unwrap().clone()),
        })
    }
}

/// Finishes `store`'s recording, recording how the program exited (`error`,
/// if it failed), and writes the trace to `target`.
pub fn finish_recording<T: 'static>(
    store: &mut Store<T>,
    target: &RecordTarget,
    error: Option<&wasmtime::Error>,
) -> Result<()> {
    let exit = match error {
        None => RecordedExit {
            code: None,
            trap: false,
            error: None,
        },
        Some(e) => match e.downcast_ref::<wasmtime_wasi::I32Exit>() {
            Some(exit) => RecordedExit {
                code: Some(exit.0),
                trap: false,
                error: None,
            },
            None => RecordedExit {
                code: None,
                trap: e.is::<wasmtime::Trap>(),
                error: Some(format!("{e:?}")),
            },
        },
    };
    record_event(&mut *store, &exit)?;
    let trace = store
        .finish_recording()
        .context("failed to record the execution")?;
    let path = &target.path;
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    let Some(modules) = &target.bundle else {
        return std::fs::write(path, trace.as_bytes())
            .with_context(|| format!("failed to write trace to `{}`", Path::display(path)));
    };
    let dir = path.join(BUNDLE_MODULES);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create trace bundle `{}`", Path::display(path)))?;
    for checksum in trace.module_checksums()? {
        let name = format!("{}.wasm", checksum_hex(&checksum));
        let wasm = modules.get(&checksum).ok_or_else(|| {
            format_err!("the trace instantiates a module ({name}) that is not part of the program")
        })?;
        std::fs::write(dir.join(&name), wasm)
            .with_context(|| format!("failed to write `{name}` to trace bundle"))?;
    }
    let trace_path = path.join(BUNDLE_TRACE);
    std::fs::write(&trace_path, trace.as_bytes())
        .with_context(|| format!("failed to write trace to `{}`", Path::display(&trace_path)))?;
    Ok(())
}

/// Reads the trace at `path`, a trace file or a trace bundle directory, and
/// the bytecode of the modules it instantiates, from the bundle or from the
/// modules and components in `modules`, which take precedence.
pub fn load_replay(
    engine: &Engine,
    path: &Path,
    modules: &[PathBuf],
) -> Result<(Trace, Vec<Vec<u8>>)> {
    let bundle = path.is_dir();
    let trace_path = if bundle {
        path.join(BUNDLE_TRACE)
    } else {
        path.to_path_buf()
    };
    let bytes = std::fs::read(&trace_path)
        .with_context(|| format!("failed to read trace `{}`", Path::display(&trace_path)))?;
    let trace = Trace::from_bytes(bytes)?;
    let mut provided = BundleModules::new();
    for file in modules {
        let wasm = std::fs::read(file)
            .with_context(|| format!("failed to read `{}`", Path::display(file)))?;
        let modules = core_modules(engine, &wasm)
            .with_context(|| format!("failed to find the modules of `{}`", Path::display(file)))?;
        for (checksum, wasm) in modules {
            provided.insert(checksum, wasm.into());
        }
    }
    let mut bytecode = Vec::new();
    let mut missing = Vec::new();
    for checksum in trace.module_checksums()? {
        if let Some(wasm) = provided.get(&checksum) {
            bytecode.push(wasm.to_vec());
            continue;
        }
        let name = format!("{}.wasm", checksum_hex(&checksum));
        let file = path.join(BUNDLE_MODULES).join(&name);
        if bundle && file.exists() {
            bytecode.push(
                std::fs::read(&file)
                    .with_context(|| format!("failed to read `{}`", Path::display(&file)))?,
            );
            continue;
        }
        missing.push(name);
    }
    if !missing.is_empty() {
        bail!(
            "the trace instantiates modules that were not provided ({}); pass the \
             program's Wasm with `--module`, or record a trace bundle with `-R bundle`",
            missing.join(", ")
        );
    }
    Ok((trace, bytecode))
}

/// Exits as a recorded program did, once its replay has finished.
pub fn replay_exit(exit: Option<RecordedExit>) -> Result<()> {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    let Some(exit) = exit else {
        return Ok(());
    };
    if let Some(code) = exit.code {
        std::process::exit(code);
    }
    match exit.error {
        Some(error) if exit.trap => {
            eprintln!("Error: {error}");
            cfg_select! {
                unix => {
                    std::process::exit(rustix::process::EXIT_SIGNALED_SIGABRT);
                }
                windows => {
                    std::process::exit(3);
                }
            }
        }
        Some(error) => Err(format_err!("{error}")),
        None => Ok(()),
    }
}
