//! Reverse execution of a replay, for reversible debugging.
//!
//! Replay is deterministic, so moving backward means restoring an earlier
//! snapshot and replaying forward to the target. Every stop has an exact
//! [`ReplayPosition`]: guest code compiled for debugging a replay counts the
//! Wasm operators it executes, and can stop when the count reaches a target.
//!
//! * Reverse stepping seeks one operator back: it restores the latest
//!   snapshot at or before the target and replays to it with a step target.
//! * Reverse continuing finds the last breakpoint or watchpoint stop before
//!   the present: it restores the latest snapshot before the present, replays
//!   forward to the present while noting such stops, and then restores the
//!   snapshot again and replays to the last one. If there was none, it
//!   searches the window before the previous snapshot, and so on back to the
//!   beginning.
//!
//! While running forward, the replayer takes a snapshot every
//! `snapshot_interval` steps and at every stop, keeping fewer the further back
//! they are: each snapshot kept is at least twice as far from the present as
//! the next later one. The beginning of the replay is always kept.
//!
//! Re-execution that only moves replay backward is quiet: it delivers no
//! embedder events. Forward execution delivers them every time it passes them,
//! as the recorded execution did, including after moving backward.

use super::*;

/// A point in a replay, ordered by execution.
///
/// Positions are deterministic: the same point has the same position in
/// every replay of a trace. A position combines the number of Wasm operators
/// the guest has executed (see [`ReplayPosition::steps`]) with where, around
/// the latest operator, replay is stopped: at its breakpoint check, just
/// before it executes, during it (at a watchpoint), or after all execution
/// (at the end of the replay).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplayPosition {
    steps: u64,
    // 0: at an operator's breakpoint check; 1: just before it executes (step
    // targets and interrupts) or between operators; 2: during it
    // (watchpoints); 3: the end of the replay.
    rank: u8,
}

impl ReplayPosition {
    /// The beginning of a replay, before any guest code has executed.
    pub const BEGINNING: ReplayPosition = ReplayPosition { steps: 0, rank: 1 };

    pub(super) fn new(steps: u64, rank: u8) -> ReplayPosition {
        ReplayPosition { steps, rank }
    }

    pub(super) fn end(steps: u64) -> ReplayPosition {
        ReplayPosition { steps, rank: 3 }
    }

    /// The number of Wasm operators the guest has executed, counting the
    /// operator replay is stopped at. Only engines with guest debugging
    /// count operators; otherwise this is zero.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Whether this is the beginning of the replay.
    pub fn is_beginning(&self) -> bool {
        *self == ReplayPosition::BEGINNING
    }

    /// Whether this is the end of the replay.
    pub fn is_end(&self) -> bool {
        self.rank == 3
    }
}

/// The snapshots of a replayer with reverse execution.
pub(super) struct Reverse {
    interval: u64,
    // Ascending by position; the first is the beginning.
    snapshots: Vec<(ReplayPosition, Checkpoint)>,
}

impl<'a, T: Send + 'static> Replayer<'a, T> {
    /// Enables reverse execution: [`Replayer::reverse_step`] and
    /// [`Replayer::reverse_continue`].
    ///
    /// This requires an engine with guest debugging, and must be called
    /// before replay starts. Forward execution then takes snapshots (see
    /// [`Replayer::checkpoint`]) at stops and every `snapshot_interval` guest
    /// steps, so that moving backward replays at most a few intervals.
    /// Moving backward delivers no embedder events; running forward again
    /// delivers them again as replay passes them.
    pub fn enable_reverse_execution(&mut self, snapshot_interval: u64) -> Result<()> {
        ensure!(
            self.driver.store.engine().tunables().debug_step_counter,
            "reverse execution requires a replaying engine with guest debugging"
        );
        ensure!(
            self.driver.reader.position() == codec::MAGIC.len()
                && self.driver.activations.is_empty(),
            "reverse execution must be enabled before replay starts"
        );
        let initial = self.checkpoint()?;
        self.reverse = Some(Reverse {
            interval: snapshot_interval.max(1),
            snapshots: alloc::vec![(ReplayPosition::BEGINNING, initial)],
        });
        self.driver.creation = Some(Vec::new());
        Ok(())
    }

    /// Moves the replay back to the previous operator: to where a forward
    /// single step from there would stop, just before the operator executes.
    ///
    /// Returns [`ReplayStop::Breakpoint`] when stopped there, or
    /// [`ReplayStop::Beginning`] at the beginning of the replay. Breakpoints
    /// between are not reported.
    pub async fn reverse_step(&mut self) -> Result<ReplayStop> {
        self.require_reverse()?;
        let present = self.position();
        // Stops at or just before an operator's execution move to the previous
        // one; stops during or after it, to just before it.
        let target = if present.rank >= 2 {
            present.steps
        } else {
            present.steps.saturating_sub(1)
        };
        if target == 0 {
            return self.reverse_to_beginning();
        }
        let result = self.seek(ReplayPosition::new(target, 1)).await;
        self.driver.quiet = false;
        result?;
        self.snapshot()?;
        Ok(ReplayStop::Breakpoint)
    }

    /// Moves the replay back to the last breakpoint or watchpoint stop before
    /// the present, as configured now, and returns it, or
    /// [`ReplayStop::Beginning`] if there is none.
    pub async fn reverse_continue(&mut self) -> Result<ReplayStop> {
        self.require_reverse()?;
        let result = self.reverse_continue_inner().await;
        self.driver.quiet = false;
        let stop = result?;
        if stop != ReplayStop::Beginning {
            self.snapshot()?;
        }
        Ok(stop)
    }

    async fn reverse_continue_inner(&mut self) -> Result<ReplayStop> {
        let mut window_end = self.position();
        // The first window excludes the present. Each later one ends at the
        // snapshot that began the window after it, whose own stop (if any)
        // replaying from that snapshot does not reach again, so it includes
        // its end.
        let mut inclusive = false;
        loop {
            // Search the window from the latest snapshot before its end.
            let snapshots = &self.reverse.as_ref().unwrap().snapshots;
            let Some(index) = snapshots.iter().rposition(|(pos, _)| *pos < window_end) else {
                return self.reverse_to_beginning();
            };
            let start = snapshots[index].0;
            self.restore_snapshot(index)?;
            self.driver.quiet = true;
            if !window_end.is_end() {
                self.driver.store.set_debug_step_target(window_end.steps);
            }
            let mut last = None;
            loop {
                let stop = self.run_raw().await?;
                let position = self.position();
                match stop {
                    ReplayStop::Finished => break,
                    _ if position > window_end || (!inclusive && position == window_end) => break,
                    // The window ends during this operator: stop after it.
                    ReplayStop::StepTarget => {
                        self.driver.store.set_debug_step_target(position.steps + 1)
                    }
                    ReplayStop::Breakpoint => last = Some(position),
                    #[cfg(feature = "debug")]
                    ReplayStop::Watchpoint(_) => last = Some(position),
                    _ => {}
                }
            }
            let Some(last) = last else {
                // Nothing in this window: search the one before it.
                window_end = start;
                inclusive = true;
                continue;
            };
            self.restore_snapshot(index)?;
            loop {
                let stop = self.run_raw().await?;
                ensure!(
                    stop != ReplayStop::Finished && self.position() <= last,
                    "reverse execution diverged: a stop was not reproduced"
                );
                if self.position() == last {
                    return Ok(stop);
                }
            }
        }
    }

    /// Replays forward to `target`, from the latest snapshot at or before
    /// it.
    async fn seek(&mut self, target: ReplayPosition) -> Result<()> {
        let snapshots = &self.reverse.as_ref().unwrap().snapshots;
        let index = snapshots
            .iter()
            .rposition(|(pos, _)| *pos <= target)
            .expect("the beginning precedes every position");
        let at = snapshots[index].0;
        self.restore_snapshot(index)?;
        if at == target {
            return Ok(());
        }
        self.driver.quiet = true;
        self.driver.store.set_debug_step_target(target.steps);
        loop {
            match self.run_raw().await? {
                ReplayStop::StepTarget if self.position() >= target => return Ok(()),
                ReplayStop::Finished => {
                    bail!("reverse execution diverged: replay ended before its target")
                }
                _ => {}
            }
        }
    }

    fn reverse_to_beginning(&mut self) -> Result<ReplayStop> {
        self.restore_snapshot(0)?;
        Ok(ReplayStop::Beginning)
    }

    fn restore_snapshot(&mut self, index: usize) -> Result<()> {
        let reverse = self.reverse.take().unwrap();
        let result = self.restore(&reverse.snapshots[index].1);
        self.reverse = Some(reverse);
        result
    }

    fn require_reverse(&mut self) -> Result<()> {
        ensure!(
            self.reverse.is_some(),
            "reverse execution is not enabled for this replay"
        );
        // An earlier reverse operation's future may have been dropped.
        self.driver.quiet = false;
        Ok(())
    }

    /// Runs forward to the next stop, taking snapshots on the way.
    pub(super) async fn run_recording_snapshots(&mut self) -> Result<ReplayStop> {
        self.driver.quiet = false;
        loop {
            let steps = self.driver.store.debug_steps();
            let interval = self.reverse.as_ref().unwrap().interval;
            let next_snapshot = (steps / interval + 1).saturating_mul(interval);
            let target = self
                .driver
                .user_target
                .map_or(next_snapshot, |t| t.min(next_snapshot));
            self.driver.store.set_debug_step_target(target);
            let stop = self.run_raw().await?;
            match stop {
                ReplayStop::StepTarget
                    if self
                        .driver
                        .user_target
                        .is_none_or(|t| self.position().steps < t) =>
                {
                    self.snapshot()?;
                    continue;
                }
                ReplayStop::StepTarget => self.driver.user_target = None,
                ReplayStop::Interrupted => {
                    if self.settle_interrupt().await? == ReplayStop::Finished {
                        return Ok(ReplayStop::Finished);
                    }
                }
                _ => {}
            }
            self.driver
                .store
                .set_debug_step_target(self.driver.user_target.unwrap_or(u64::MAX));
            if stop != ReplayStop::Finished && !matches!(stop, ReplayStop::Event(_)) {
                self.snapshot()?;
            }
            return Ok(stop);
        }
    }

    /// Takes a snapshot at the present position, and thins out the others.
    fn snapshot(&mut self) -> Result<()> {
        let present = self.position();
        let reverse = self.reverse.as_ref().unwrap();
        let interval = reverse.interval;
        if !reverse.snapshots.iter().any(|(pos, _)| *pos == present) {
            let checkpoint = self.checkpoint()?;
            let snapshots = &mut self.reverse.as_mut().unwrap().snapshots;
            snapshots.try_reserve(1)?;
            let index = snapshots.partition_point(|(pos, _)| *pos < present);
            snapshots.insert(index, (present, checkpoint));
        }
        let snapshots = &mut self.reverse.as_mut().unwrap().snapshots;
        // Snapshots after the present belong to a future that replay will
        // recompute anyway.
        snapshots.retain(|(pos, _)| *pos <= present);
        // Keep the beginning, and from the present backward, snapshots that
        // are at least twice as far away as the last one kept.
        let mut last_kept = None;
        let mut keep = alloc::vec![true; snapshots.len()];
        for i in (1..snapshots.len()).rev() {
            let distance = present.steps - snapshots[i].0.steps;
            keep[i] = match last_kept {
                None => true,
                Some(last) => distance >= interval.max(last * 2),
            };
            if keep[i] {
                last_kept = Some(distance);
            }
        }
        let mut keep = keep.into_iter();
        snapshots.retain(|_| keep.next().unwrap());
        Ok(())
    }
}
