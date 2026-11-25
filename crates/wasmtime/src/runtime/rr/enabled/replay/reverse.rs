//! Reverse execution of a replay, for reversible debugging.
//!
//! Replay is deterministic, so moving backward means restoring an earlier
//! snapshot and replaying forward to the target, as in Boothe, "Efficient
//! Algorithms for Bidirectional Debugging" (PLDI 2000). Debug events are counted
//! to find it.
//!
//! * Reverse stepping moves to the previous event.
//! * Reverse continuing replays from the latest snapshot to the present, noting
//!   breakpoints and watchpoints, and then again to the last of them. If there
//!   are none, it searches from the snapshot before to there, and so on.
//!
//! While replay runs forward, it takes snapshots at each stop and after each
//! interval of events. It keeps fewer of them the further back they are.
//!
//! Moving backward does not deliver embedder events.

use super::count::CountMode;
use super::*;

/// A point in a replay. These are ordered by execution, and are the same in
/// every replay of a trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position {
    /// The number of debug events that were counted.
    events: usize,
    place: Place,
}

/// Where a position is, relative to its last debug event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Place {
    /// At the event's check, before the instruction that follows.
    Check,
    /// During that instruction, such as at a watchpoint.
    Instruction,
    /// After all execution.
    End,
}

impl Position {
    /// The beginning of a replay.
    const BEGINNING: Position = Position::at_check(0);

    pub(super) const fn at_check(events: usize) -> Self {
        Position {
            events,
            place: Place::Check,
        }
    }

    pub(super) const fn after_check(events: usize) -> Self {
        Position {
            events,
            place: Place::Instruction,
        }
    }

    /// The number of debug events, which are at each instruction, that have been
    /// counted.
    pub fn events(&self) -> usize {
        self.events
    }

    /// Whether this is the beginning of the replay.
    pub fn is_beginning(&self) -> bool {
        *self == Self::BEGINNING
    }

    /// Whether this is the end of the replay.
    pub fn is_end(&self) -> bool {
        self.place == Place::End
    }
}

/// The state of a replayer with reverse execution.
pub(super) struct Reverse {
    interval: usize,
    /// In order of position. The first is at the beginning.
    snapshots: Vec<(Position, Checkpoint)>,
}

impl Replayer {
    /// Enables [`Replayer::reverse_step`] and [`Replayer::reverse_continue`].
    ///
    /// This requires an engine with guest debugging, and must be called before
    /// replay starts. Replay then counts debug events, which slows it down, and
    /// takes a checkpoint at each stop and after each `interval` of events.
    pub fn enable_reverse_execution(&mut self, interval: usize) -> Result<()> {
        ensure!(
            self.offset == codec::MAGIC.len() && self.reverse.is_none(),
            "reverse execution must be enabled before replay starts"
        );
        self.edit_breakpoints(|b| b.count_events())?;
        self.store.as_context_mut().0.rr_count_events()?;
        let beginning = self.checkpoint()?;
        self.reverse = Some(Reverse {
            interval: interval.max(1),
            snapshots: vec![(Position::BEGINNING, beginning)],
        });
        Ok(())
    }

    /// The current position, if reverse execution is enabled.
    pub fn position(&mut self) -> Option<Position> {
        self.reverse.as_ref()?;
        Some(self.current_position())
    }

    fn current_position(&mut self) -> Position {
        Position {
            events: self.store.as_context_mut().0.rr_event_count(),
            place: if self.finished && self.paused.is_none() {
                Place::End
            } else if self.at_check || self.offset == codec::MAGIC.len() {
                Place::Check
            } else {
                Place::Instruction
            },
        }
    }

    /// Moves back to the previous instruction, where a forward single step would
    /// stop.
    ///
    /// Returns [`ReplayStop::Breakpoint`] there, or [`ReplayStop::Beginning`] at
    /// the beginning of the replay.
    ///
    /// If the future is dropped, the replay is left at some earlier point, from
    /// which it can continue in either direction.
    pub async fn reverse_step(&mut self) -> Result<ReplayStop> {
        self.require_reverse()?;
        let present = self.current_position();
        let events = match present.place {
            Place::Check => present.events.saturating_sub(1),
            Place::Instruction | Place::End => present.events,
        };
        if events == 0 {
            return self.reverse_to_beginning();
        }
        self.seek(events).await?;
        self.stop_moving_backward()?;
        Ok(ReplayStop::Breakpoint)
    }

    /// Moves back to the last breakpoint or watchpoint, as they are now set, and
    /// returns it, or [`ReplayStop::Beginning`] if there is none.
    ///
    /// If the future is dropped, the replay is left at some earlier point, from
    /// which it can continue in either direction.
    pub async fn reverse_continue(&mut self) -> Result<ReplayStop> {
        self.require_reverse()?;
        if self.store().is_single_step() {
            return self.reverse_step().await;
        }
        let mut end = self.current_position();
        // Whether the end itself is searched, as it is for a search up to
        // where a later one began.
        let mut inclusive = false;
        loop {
            let snapshots = &self.reverse.as_ref().unwrap().snapshots;
            let Some(index) = snapshots.iter().rposition(|(p, _)| *p < end) else {
                return self.reverse_to_beginning();
            };
            let start = snapshots[index].0;
            if let Some(last) = self.search(index, end, inclusive).await? {
                return self.move_to_hit(last).await;
            }
            end = start;
            inclusive = true;
        }
    }

    /// Finds the last breakpoint or watchpoint between a snapshot and `end`.
    async fn search(
        &mut self,
        snapshot: usize,
        end: Position,
        inclusive: bool,
    ) -> Result<Option<Position>> {
        self.restore_snapshot(snapshot)?;
        let before = match (end.place, inclusive) {
            (Place::Check, true) => Position::after_check(end.events),
            (Place::Instruction, true) => Position::at_check(end.events + 1),
            _ => end,
        };
        // Include the instruction after the last event.
        let target = (end.place != Place::End).then_some(end.events + 1);
        let mode = CountMode::Search { before, last: None };
        self.set_count_mode(target, mode);
        while let Some(stop) = self.run_to_stop().await? {
            if stop == ReplayStop::Finished {
                break;
            }
        }
        match self.set_count_mode(None, CountMode::Ignore) {
            Some(CountMode::Search { last, .. }) => Ok(last),
            _ => unreachable!(),
        }
    }

    /// Moves to a breakpoint or watchpoint at `position`.
    async fn move_to_hit(&mut self, position: Position) -> Result<ReplayStop> {
        self.seek(position.events).await?;
        self.set_count_mode(None, CountMode::Stop);
        let mut stop = ReplayStop::Breakpoint;
        if position.place == Place::Instruction {
            let next = self.run_to_stop().await?;
            stop = next.ok_or_else(|| format_err!("replay did not reach a watchpoint again"))?;
        }
        ensure!(
            self.current_position() == position,
            "replay did not reach a breakpoint or watchpoint again"
        );
        self.stop_moving_backward()?;
        Ok(stop)
    }

    /// Moves to the given debug event.
    async fn seek(&mut self, events: usize) -> Result<()> {
        let target = Position::at_check(events);
        let snapshots = &self.reverse.as_ref().unwrap().snapshots;
        let index = snapshots.iter().rposition(|(p, _)| *p <= target).unwrap();
        self.restore_snapshot(index)?;
        if self.current_position() == target {
            return Ok(());
        }
        self.set_count_mode(Some(events), CountMode::Ignore);
        while let Some(stop) = self.run_to_stop().await? {
            ensure!(
                stop != ReplayStop::Finished,
                "replay ended before a debug event that it reached before"
            );
        }
        Ok(())
    }

    fn reverse_to_beginning(&mut self) -> Result<ReplayStop> {
        self.restore_snapshot(0)?;
        self.set_count_mode(None, CountMode::Stop);
        Ok(ReplayStop::Beginning)
    }

    fn set_count_mode(&mut self, target: Option<usize>, mode: CountMode) -> Option<CountMode> {
        self.store
            .as_context_mut()
            .0
            .rr_set_count_mode(target, mode)
    }

    fn stop_moving_backward(&mut self) -> Result<()> {
        self.set_count_mode(None, CountMode::Stop);
        self.take_snapshot()
    }

    fn require_reverse(&self) -> Result<()> {
        ensure!(
            self.reverse.is_some(),
            "reverse execution is not enabled for this replay"
        );
        Ok(())
    }

    fn restore_snapshot(&mut self, index: usize) -> Result<()> {
        let reverse = self.reverse.take().unwrap();
        let result = self.restore(&reverse.snapshots[index].1);
        self.reverse = Some(reverse);
        result
    }

    /// Prepares to run forward, if reverse execution is enabled, until the next
    /// snapshot is due.
    pub(super) fn prepare_to_run_forward(&mut self) {
        let Some(reverse) = &self.reverse else {
            return;
        };
        let interval = reverse.interval;
        let events = self.store.as_context_mut().0.rr_event_count();
        let next = (events / interval + 1).saturating_mul(interval);
        self.set_count_mode(Some(next), CountMode::Stop);
    }

    /// Takes a snapshot of the current position if reverse execution is enabled,
    /// and discards those that are no longer needed.
    pub(super) fn take_snapshot(&mut self) -> Result<()> {
        if self.reverse.is_none() {
            return Ok(());
        }
        let position = self.current_position();
        let checkpoint = self.checkpoint()?;
        let reverse = self.reverse.as_mut().unwrap();
        // Replay takes later snapshots again if it reaches them.
        reverse.snapshots.retain(|(p, _)| *p < position);
        // Going back, keep those at least twice as far away as the one after.
        let mut kept = None;
        let mut thinned = Vec::new();
        while let Some((p, checkpoint)) = reverse.snapshots.pop() {
            let distance = position.events - p.events;
            if p.is_beginning() || kept.is_none_or(|k| distance >= reverse.interval.max(k * 2)) {
                kept = Some(distance);
                thinned.push((p, checkpoint));
            }
        }
        thinned.reverse();
        thinned.push((position, checkpoint));
        reverse.snapshots = thinned;
        Ok(())
    }
}
