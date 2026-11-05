//! Replaying the recordings of `*.wast` tests.

use wasmtime::{Engine, Result, Store, rr};

/// Replays `trace` with `engine`, compiling its modules from `bytecode`.
pub async fn check_replay(
    engine: &Engine,
    trace: &rr::Trace,
    bytecode: impl IntoIterator<Item = &[u8]>,
) -> Result<()> {
    let mut store = Store::new(engine, ());
    let bytecode = bytecode.into_iter().collect::<Vec<_>>();
    store.replay(trace, &bytecode).await?;
    Ok(())
}
