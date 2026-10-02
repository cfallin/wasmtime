//! The module that implements the `wasmtime replay` command.

use crate::common::{RecordedExit, RunCommon, load_replay, replay_exit};
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use wasmtime::rr::ReplayStop;
use wasmtime::{Engine, RRConfig, Result, Store};

/// Replays an execution recorded with `wasmtime run -R path=...` or
/// `wasmtime serve -R path=...`.
///
/// Replay reproduces the recorded execution deterministically, without
/// running any host code: everything the guest received from the outside
/// world comes from the trace, and the guest's recorded output is printed
/// again.
///
/// Traces refer to the modules they instantiate by checksum. Replay finds
/// them in a trace bundle (recorded with `-R bundle`), or in the Wasm given
/// with `--module`.
#[derive(Parser)]
pub struct ReplayCommand {
    #[command(flatten)]
    #[expect(missing_docs, reason = "don't want to mess with clap doc-strings")]
    pub run: RunCommon,

    /// A Wasm module or component, such as the recorded program, whose core
    /// modules the trace instantiates. This can be given more than once, and
    /// takes precedence over the modules in a trace bundle. Components are
    /// translated with this command's `-W` options, which must match those
    /// of the recording.
    #[arg(long = "module", value_name = "WASM")]
    pub modules: Vec<PathBuf>,

    /// The trace to replay: a trace file, or a trace bundle directory.
    #[arg(value_name = "TRACE")]
    pub trace: PathBuf,
}

impl ReplayCommand {
    /// Executes the command.
    pub fn execute(self) -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_time()
            .enable_io()
            .build()?;
        runtime.block_on(self.replay())
    }

    async fn replay(mut self) -> Result<()> {
        if self.run.recording() {
            wasmtime::bail!("a replay cannot be recorded");
        }
        let trace_path = self.trace;
        self.run.common.init_logging()?;

        let mut config = self.run.common.config(None)?;
        config.rr(RRConfig::Replaying);
        let engine = Engine::new(&config)?;
        let (trace, bytecode) = load_replay(&engine, &trace_path, &self.modules)?;
        let mut store = Store::new(&engine, ());
        let mut replayer = store.replayer(&trace)?;
        for wasm in &bytecode {
            replayer.add_bytecode(wasm)?;
        }

        // Print the recorded output again, and note how the program exited.
        let exit = Arc::new(Mutex::new(None));
        wasmtime_wasi::rr::replay_output(&mut replayer, std::io::stdout(), std::io::stderr());
        replayer.on_event({
            let exit = exit.clone();
            move |e: RecordedExit| *exit.lock().unwrap() = Some(e)
        });
        while replayer.run().await? != ReplayStop::Finished {}
        drop(replayer);
        replay_exit(exit.lock().unwrap().take())
    }
}
