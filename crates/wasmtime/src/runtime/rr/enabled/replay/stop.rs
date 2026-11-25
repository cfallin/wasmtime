//! Stopping the running activation, at a debug event or to yield to the async
//! executor.
//!
//! Compiled code reaches these points through builtins, which note the stop in
//! the store and the activation's control block. The builtin's trampoline then
//! yields to the replayer.

use super::super::*;
use core::sync::atomic::{AtomicBool, Ordering};
use wasmtime_environ::RrYield;

/// Why the running activation stops.
pub(in super::super) enum Stop {
    #[cfg(feature = "debug")]
    Breakpoint,
    #[cfg(feature = "debug")]
    Watchpoint(crate::WatchpointHit),
    Interrupt,
    /// The count of debug events reached its target.
    #[cfg(feature = "debug")]
    EventTarget,
    /// To let the replayer yield to the async executor.
    Yield,
}

/// What replay does at epoch checks.
#[derive(Default)]
pub(in super::super) struct Epochs {
    /// Set by the embedder to interrupt replay.
    pub interrupt: Option<Arc<AtomicBool>>,
    /// How many epochs apart replay yields to the async executor, and the epoch
    /// at which it next does.
    pub yields: Option<(u64, u64)>,
}

impl StoreOpaque {
    /// Stops the running replay activation.
    fn rr_stop(&mut self, stop: Stop) {
        let control = self.vm_store_context().replay_control.unwrap();
        // SAFETY: the running activation's control block is live, and no
        // reference to it is live while it runs.
        unsafe {
            (*control.as_ptr()).reason = RrYield::Stopped as u32;
        }
        self.rr_session().mode.replaying().stop = Some(stop);
    }

    /// Handles a breakpoint during replay, returning whether replaying.
    #[cfg(feature = "debug")]
    pub(crate) fn rr_stop_at_breakpoint(&mut self) -> bool {
        if !self.rr.replaying() {
            return false;
        }
        if let Some(stop) = self.rr_breakpoint_stop() {
            self.rr_stop(stop);
        }
        true
    }

    /// Handles a watchpoint during replay, returning whether replaying.
    #[cfg(feature = "debug")]
    pub(crate) fn rr_stop_at_watchpoint(&mut self, hit: crate::WatchpointHit) -> bool {
        if !self.rr.replaying() {
            return false;
        }
        if self.rr_watchpoint_stops() {
            self.rr_stop(Stop::Watchpoint(hit));
        }
        true
    }

    /// Handles an epoch deadline during replay, returning the new deadline, or
    /// `None` when not replaying. Replay never traps or runs callbacks at epoch
    /// deadlines, which the recording did not.
    #[cfg(target_has_atomic = "64")]
    pub(crate) fn rr_replay_epoch(&mut self) -> Option<u64> {
        if !self.rr.replaying() {
            return None;
        }
        let epoch = self.engine().current_epoch();
        // Interrupts wait until replay moves forward.
        #[cfg(feature = "debug")]
        let moving_backward = self.rr_moving_backward();
        #[cfg(not(feature = "debug"))]
        let moving_backward = false;
        let epochs = &mut self.rr_session().mode.replaying().epochs;
        let interrupt = epochs
            .interrupt
            .as_ref()
            .filter(|_| !moving_backward)
            .is_some_and(|flag| flag.swap(false, Ordering::SeqCst));
        let due = epochs.yields.as_mut().filter(|(_, next)| epoch >= *next);
        let yields = match due {
            Some((delta, next)) => {
                *next = epoch.saturating_add(*delta);
                true
            }
            None => false,
        };
        if interrupt {
            self.rr_stop(Stop::Interrupt);
        } else if yields {
            self.rr_stop(Stop::Yield);
        }
        // Check for interrupts at every epoch.
        self.set_epoch_deadline(1);
        Some(self.get_epoch_deadline())
    }

    /// Handles running out of fuel during replay, returning whether replaying.
    /// Replay never runs out of fuel, which the recording did not: it refuels,
    /// with the yield interval's worth of fuel if there is one.
    pub(crate) fn rr_replay_fuel(&mut self) -> Result<bool> {
        if !self.rr.replaying() {
            return Ok(false);
        }
        self.set_fuel(u64::MAX)?;
        if self.fuel_yield_interval.is_some() {
            self.rr_stop(Stop::Yield);
        }
        Ok(true)
    }
}
