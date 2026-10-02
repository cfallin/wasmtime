//! The module that implements the `wasmtime replay` command.

use crate::common::{RecordedExit, RunCommon, replay_exit};
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use wasmtime::rr::{ReplayStop, Trace};
use wasmtime::{Engine, RRConfig, Result, Store, error::Context as _};

/// Replays an execution recorded with `wasmtime run --record` or
/// `wasmtime serve --record`.
///
/// Replay reproduces the recorded execution deterministically, without
/// running any host code: everything the guest received from the outside
/// world comes from the trace, and the guest's recorded output is printed
/// again.
#[derive(Parser)]
pub struct ReplayCommand {
    #[command(flatten)]
    #[expect(missing_docs, reason = "don't want to mess with clap doc-strings")]
    pub run: RunCommon,

    /// The trace to replay.
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
        if self.run.record.is_some() {
            wasmtime::bail!("a replay cannot be recorded");
        }
        let trace_path = self.trace;
        self.run.common.init_logging()?;

        let bytes = std::fs::read(&trace_path)
            .with_context(|| format!("failed to read trace `{}`", trace_path.display()))?;
        let trace = Trace::from_bytes(bytes)?;
        let mut config = self.run.common.config(None)?;
        config.rr(RRConfig::Replaying);
        let engine = Engine::new(&config)?;
        let mut store = Store::new(&engine, ());
        let mut replayer = store.replayer(&trace)?;

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
