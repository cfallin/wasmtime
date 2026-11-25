//! Counting debug events, so that replay can return to a point in the guest's
//! execution.
//!
//! To count events, every breakpoint check is patched in. The trampoline that
//! they call counts them, and only calls the `breakpoint` builtin if its filter
//! matches, for checks that may be at a breakpoint, or if it reaches a target.

use super::super::*;
use super::reverse::Position;
use super::stop::Stop;
use alloc::collections::BTreeSet;
use wasmtime_environ::DEBUG_EVENT_FILTER_LEN;

/// The state of counting debug events.
pub(in super::super) struct EventCounter {
    filter: Box<[u8; DEBUG_EVENT_FILTER_LEN]>,
    /// The return addresses of checks at breakpoints.
    breakpoints: BTreeSet<usize>,
    single_step: bool,
    mode: CountMode,
}

/// What breakpoints and watchpoints do while debug events are counted.
pub(in super::super) enum CountMode {
    /// They stop replay.
    Stop,
    /// They are ignored.
    Ignore,
    /// They are noted, without stopping replay.
    Search {
        /// Only those before this position are.
        before: Position,
        /// The last that was.
        last: Option<Position>,
    },
}

impl EventCounter {
    fn update_filter(&mut self) {
        match self.mode {
            CountMode::Ignore => self.filter.fill(0),
            _ if self.single_step => self.filter.fill(1),
            _ => {
                self.filter.fill(0);
                for pc in &self.breakpoints {
                    self.filter[pc % DEBUG_EVENT_FILTER_LEN] = 1;
                }
            }
        }
    }

    /// Returns whether to stop at a breakpoint or watchpoint.
    fn hit(&mut self, position: Position) -> bool {
        match &mut self.mode {
            CountMode::Stop => true,
            CountMode::Ignore => false,
            CountMode::Search { before, last } => {
                if position < *before {
                    *last = Some(position);
                }
                false
            }
        }
    }
}

impl StoreOpaque {
    fn rr_counter(&mut self) -> Option<&mut EventCounter> {
        if !self.rr.replaying() {
            return None;
        }
        self.rr_session().mode.replaying().counter.as_mut()
    }

    /// Starts counting debug events. Every breakpoint check must be patched in.
    pub(super) fn rr_count_events(&mut self) -> Result<()> {
        let mut counter = EventCounter {
            filter: try_new::<Box<_>>([0; DEBUG_EVENT_FILTER_LEN])?,
            breakpoints: BTreeSet::new(),
            single_step: false,
            mode: CountMode::Stop,
        };
        self.vm_store_context_mut().debug_event_filter =
            Some(NonNull::from(&mut *counter.filter).cast().into());
        self.rr_session().mode.replaying().counter = Some(counter);
        self.rr_breakpoints_edited();
        Ok(())
    }

    /// The number of debug events that have been counted.
    pub(super) fn rr_event_count(&mut self) -> usize {
        *self.vm_store_context_mut().debug_event_count.get_mut()
    }

    pub(super) fn rr_set_event_count(&mut self, count: usize) {
        *self.vm_store_context_mut().debug_event_count.get_mut() = count;
    }

    /// Sets the count at which replay stops, if any, and what breakpoints and
    /// watchpoints do until then. Returns the previous mode.
    pub(super) fn rr_set_count_mode(
        &mut self,
        target: Option<usize>,
        mode: CountMode,
    ) -> Option<CountMode> {
        *self.vm_store_context_mut().debug_event_target.get_mut() = target.unwrap_or(usize::MAX);
        let counter = self.rr_counter()?;
        let previous = core::mem::replace(&mut counter.mode, mode);
        counter.update_filter();
        Some(previous)
    }

    /// Updates the state of counting to changed breakpoints.
    pub(crate) fn rr_breakpoints_edited(&mut self) {
        if self.rr_counter().is_none() {
            return;
        }
        let breakpoints = self.breakpoint_return_addresses().into_iter().collect();
        let single_step = self.breakpoints_and_registry().0.is_single_step();
        let counter = self.rr_counter().unwrap();
        counter.breakpoints = breakpoints;
        counter.single_step = single_step;
        counter.update_filter();
    }

    /// Why a check that called the `breakpoint` builtin stops, if it does.
    pub(super) fn rr_breakpoint_stop(&mut self) -> Option<Stop> {
        let pc = *self.vm_store_context_mut().last_wasm_exit_pc.get_mut();
        let events = self.rr_event_count();
        let target = *self.vm_store_context_mut().debug_event_target.get_mut();
        let Some(counter) = self.rr_counter() else {
            return Some(Stop::Breakpoint);
        };
        let hit = counter.single_step || counter.breakpoints.contains(&pc);
        if hit && counter.hit(Position::at_check(events)) {
            return Some(Stop::Breakpoint);
        }
        (events >= target).then_some(Stop::EventTarget)
    }

    /// Whether a watchpoint stops.
    pub(super) fn rr_watchpoint_stops(&mut self) -> bool {
        let events = self.rr_event_count();
        match self.rr_counter() {
            Some(counter) => counter.hit(Position::after_check(events)),
            None => true,
        }
    }

    /// Whether replay is re-executing to move backward.
    pub(super) fn rr_moving_backward(&mut self) -> bool {
        self.rr_counter()
            .is_some_and(|c| !matches!(c.mode, CountMode::Stop))
    }
}
