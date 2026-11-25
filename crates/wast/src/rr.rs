//! Replaying the recordings of `*.wast` tests.

use wasmtime::rr::{Checkpoint, ReplayStop, Replayer};
use wasmtime::{Result, bail};

/// Single-step at most this many guest instructions.
const MAX_STEPS: usize = 2000;

/// Take a checkpoint each time that this many have executed.
const CHECKPOINT_STEPS: usize = 700;

/// Replays a recording to the end.
///
/// If the engine enables guest debugging, this single-steps the first
/// instructions, taking checkpoints, and then rewinds to each and replays from
/// there.
pub async fn replay(replayer: &mut Replayer) -> Result<()> {
    if replayer.edit_breakpoints(|b| b.single_step(true)).is_err() {
        return finish(replayer).await;
    }
    let checkpoints = step(replayer).await?;
    for checkpoint in checkpoints.iter().rev() {
        replayer.restore(checkpoint)?;
        finish(replayer).await?;
    }
    Ok(())
}

async fn step(replayer: &mut Replayer) -> Result<Vec<Checkpoint>> {
    let mut checkpoints = vec![replayer.checkpoint()?];
    for step in 1..=MAX_STEPS {
        match replayer.run().await? {
            ReplayStop::Breakpoint if step % CHECKPOINT_STEPS == 0 => {
                checkpoints.push(replayer.checkpoint()?);
            }
            ReplayStop::Breakpoint => {}
            ReplayStop::Finished => break,
            stop => bail!("unexpected replay stop {stop:?}"),
        }
    }
    replayer.edit_breakpoints(|b| b.single_step(false))?;
    finish(replayer).await?;
    Ok(checkpoints)
}

async fn finish(replayer: &mut Replayer) -> Result<()> {
    match replayer.run().await? {
        ReplayStop::Finished => Ok(()),
        stop => bail!("unexpected replay stop {stop:?}"),
    }
}
