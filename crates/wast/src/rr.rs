//! Replaying the recordings of `*.wast` tests.

use wasmtime::rr::{ReplayStop, Replayer};
use wasmtime::{Result, bail};

/// Single-step at most this many guest instructions.
const MAX_STEPS: usize = 2000;

/// Replays a recording to the end, single-stepping the first instructions if
/// the engine enables guest debugging.
pub async fn replay(replayer: &mut Replayer) -> Result<()> {
    let mut stepping = replayer.edit_breakpoints(|b| b.single_step(true)).is_ok();
    let mut steps = 0;
    loop {
        match replayer.run().await? {
            ReplayStop::Breakpoint if stepping => {
                steps += 1;
                if steps == MAX_STEPS {
                    replayer.edit_breakpoints(|b| b.single_step(false))?;
                    stepping = false;
                }
            }
            ReplayStop::Finished => return Ok(()),
            stop => bail!("unexpected replay stop {stop:?}"),
        }
    }
}
