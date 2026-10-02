//! Replaying the recordings of `*.wast` tests.

use wasmtime::{Engine, Result, Store, rr};

/// Replays `trace` with `engine`.
pub async fn check_replay(engine: &Engine, trace: &rr::Trace) -> Result<()> {
    let mut store = Store::new(engine, ());
    store.replay(trace).await?;
    Ok(())
}
