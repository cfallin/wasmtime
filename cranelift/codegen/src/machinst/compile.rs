//! Compilation backend pipeline: optimized IR to VCode / binemit.

use crate::ir::Function;
use crate::machinst::*;
use crate::settings;
use crate::timing;

use log::{debug, log_enabled, Level};
use regalloc2;

/// Compile the given function down to VCode with allocated registers, ready
/// for binary emission.
pub fn compile<'a, Backend, ABI>(
    f: &Function,
    b: &Backend,
    emit_info: <Backend::MInst as MachInstEmit>::Info,
) -> CodegenResult<VCode<Backend::MInst, ABI>>
where
    Backend: LowerBackend + MachBackend,
    ABI: ABICallee<I = Backend::MInst>,
{
    // Compute lowered block order.
    let block_order = BlockLoweringOrder::new(f);
    // Build the lowering context.
    let lower = Lower::new(f, emit_info, block_order)?;
    // Lower the IR.
    let mut vcode = {
        let _tt = timing::vcode_lower();
        lower.lower(b)?
    };

    // Creating the vcode string representation may be costly for large functions, so don't do it
    // if the Debug level hasn't been statically (through features) or dynamically (through
    // RUST_LOG) enabled.
    if log_enabled!(Level::Debug) {
        debug!(
            "vcode from lowering: \n{}",
            vcode.show_rru(Some(b.reg_universe()))
        );
    }

    // Perform register allocation.
    let result = {
        let _tt = timing::regalloc();
        regalloc2::run(&vcode, b.reg_env())
            .map_err(|err| {
                debug!(
                    "Register allocation error for vcode\n{:?}\nError: {:?}",
                    vcode, err
                );
                err
            })
            .expect("register allocation")
    };

    // Reorder vcode into final order and copy out final instruction sequence
    // all at once. This also inserts prologues/epilogues.
    {
        let _tt = timing::vcode_post_ra();
        vcode.finalize_with_regalloc_output(&result);
    }

    if log_enabled!(Level::Debug) {
        debug!(
            "vcode after regalloc: final version:\n{}",
            vcode.show_rru(Some(b.reg_universe()))
        );
    }

    Ok(vcode)
}
