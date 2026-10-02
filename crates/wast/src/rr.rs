//! Replaying the recordings of `*.wast` tests.

use wasmtime::{Engine, Result, Store, bail, rr};

/// Single-step at most this many guest instructions per trace.
const MAX_STEPS: usize = 2000;
/// Take at most this many checkpoints per trace; each copies guest memory.
const MAX_CHECKPOINTS: usize = 16;

/// Replays `trace` with `engine`, compiling its modules from `bytecode`. If
/// the engine enables guest debugging, replay single-steps the first guest
/// instructions, checkpointing evenly among them, and is then rewound to each
/// checkpoint and replayed to the end again.
pub async fn check_replay(
    engine: &Engine,
    trace: &rr::Trace,
    bytecode: impl IntoIterator<Item = &[u8]>,
) -> Result<()> {
    let mut store = Store::new(engine, ());
    let mut replayer = store.replayer(trace)?;
    for wasm in bytecode {
        replayer.add_bytecode(wasm)?;
    }
    let Some(mut breakpoints) = replayer.edit_breakpoints() else {
        return expect_end(&mut replayer).await;
    };
    breakpoints.single_step(true)?;
    drop(breakpoints);
    let mut checkpoints = vec![replayer.checkpoint()?];
    let mut stops = 0_usize;
    loop {
        match replayer.run().await? {
            rr::ReplayStop::Breakpoint => {
                stops += 1;
                if stops % (MAX_STEPS / MAX_CHECKPOINTS) == 0 {
                    checkpoints.push(replayer.checkpoint()?);
                }
                if stops == MAX_STEPS {
                    replayer.edit_breakpoints().unwrap().single_step(false)?;
                }
            }
            rr::ReplayStop::Finished => break,
            stop => bail!("unexpected replay stop {stop:?}"),
        }
    }
    replayer.edit_breakpoints().unwrap().single_step(false)?;
    for checkpoint in checkpoints.iter().rev() {
        replayer.restore(checkpoint)?;
        expect_end(&mut replayer).await?;
    }
    Ok(())
}

async fn expect_end(replayer: &mut rr::Replayer<'_, ()>) -> Result<()> {
    match replayer.run().await? {
        rr::ReplayStop::Finished => Ok(()),
        stop => bail!("unexpected replay stop {stop:?}"),
    }
}
