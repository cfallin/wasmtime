//! Compilation backend pipeline: optimized IR to VCode / binemit.

use crate::ir::Function;
use crate::machinst::*;
use crate::settings;
use crate::timing;

use log::{debug, log_enabled, Level};
use regalloc::{allocate_registers, RegAllocAlgorithm};
use std::env;

/// Compile the given function down to VCode with allocated registers, ready
/// for binary emission.
pub fn compile<B: LowerBackend>(
    f: &Function,
    b: &B,
    abi: Box<dyn ABIBody<B::MInst>>,
    flags: &settings::Flags,
) -> VCode<B::MInst>
where
    B::MInst: ShowWithRRU,
{
    let call_conv = f.signature.call_conv;

    // This lowers the CL IR.
    let mut vcode = Lower::new(f, abi).lower(b);

    let universe = &B::MInst::reg_universe(call_conv);

    debug!(
        "vcode from lowering: \n{}",
        show_vcode(&vcode, Some(universe), &None)
    );

    // Perform register allocation.
    let algorithm = match env::var("REGALLOC") {
        Ok(str) => match str.as_str() {
            "lsrac" => RegAllocAlgorithm::LinearScanChecked,
            "lsra" => RegAllocAlgorithm::LinearScan,
            // to wit: btc doesn't mean "bitcoin" here
            "btc" => RegAllocAlgorithm::BacktrackingChecked,
            _ => RegAllocAlgorithm::Backtracking,
        },
        // By default use backtracking, which is the fastest.
        Err(_) => RegAllocAlgorithm::Backtracking,
    };

    let want_annotations = log_enabled!(Level::Debug);

    let result = {
        let _tt = timing::regalloc();
        allocate_registers(
            &mut vcode,
            algorithm,
            universe,
            /*request_block_annotations=*/ want_annotations,
        )
        .map_err(|err| {
            debug!(
                "Register allocation error for vcode\n{}\nError: {:?}",
                show_vcode(&vcode, Some(universe), &None),
                err
            );
            err
        })
        .expect("register allocation")
    };

    // Clone the anns, if we asked for any, since
    // |replace_insns_from_regalloc| will take ownership of the entire
    // allocation result.  This is zero-cost in non-debug-logging configs.
    let mb_annotations = if want_annotations {
        result.block_annotations.clone()
    } else {
        None
    };

    // Reorder vcode into final order and copy out final instruction sequence
    // all at once. This also inserts prologues/epilogues.
    vcode.replace_insns_from_regalloc(result, flags);

    vcode.remove_redundant_branches();

    // Do final passes over code to finalize branches.
    vcode.finalize_branches();

    debug!(
        "vcode after regalloc: final version:\n{}",
        show_vcode(&vcode, Some(universe), &mb_annotations)
    );

    //println!("{}\n", vcode.show_rru(Some(&B::MInst::reg_universe())));

    vcode
}
