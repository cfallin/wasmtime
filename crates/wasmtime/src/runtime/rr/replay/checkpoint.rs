//! Checkpoints of a replay, for rewinding it.
//!
//! A checkpoint is taken between trace events, or while an activation is
//! stopped at a debug event. It captures:
//!
//! * the driver's protocol state: the trace position, the activations and
//!   their parked host calls, the startup and growth-failure state, and the
//!   number of objects and instances constructed so far;
//! * each activation's raw fiber snapshot, control block, saved
//!   `VMStoreContext` state, protection-key mask, and value buffer;
//! * the guest state of every object constructed so far: memory sizes and
//!   contents, function table sizes and elements, and mutable globals
//!   (including component instance flags).
//!
//! Restoring puts all of this back in place. Activations keep their fiber
//! stacks, control blocks, and buffers at the same addresses, so the restored
//! stacks remain valid. An activation that completed after a checkpoint is
//! therefore retained, rather than freed, while a checkpoint can restore it.
//! Objects constructed after a checkpoint stay in the store but become
//! unreachable once it is restored; replay constructs them again.

use super::*;
use crate::Val;
use crate::runtime::vm::FuncTableElem;
use alloc::sync::{Arc, Weak};
use core::sync::atomic::{AtomicUsize, Ordering};
use wasmtime_fiber::RawFiberSnapshot;

/// The checkpoints of one replayer, and the completed activations they need.
pub(super) struct Checkpoints {
    replayer: usize,
    // Live checkpoints, by the activations each can restore.
    live: Vec<(Weak<()>, Vec<u64>)>,
    // Completed activations that a live checkpoint can restore.
    retired: Vec<Activation>,
}

impl Default for Checkpoints {
    fn default() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Checkpoints {
            replayer: NEXT.fetch_add(1, Ordering::Relaxed),
            live: Vec::new(),
            retired: Vec::new(),
        }
    }
}

impl Checkpoints {
    fn needed(&self, serial: u64) -> bool {
        self.live
            .iter()
            .any(|(live, serials)| live.strong_count() > 0 && serials.contains(&serial))
    }

    /// Keeps a completed activation if a checkpoint can restore it, and
    /// otherwise returns it to be freed.
    pub(super) fn retire(&mut self, activation: Activation) -> Option<Activation> {
        if self.needed(activation.serial) && self.retired.try_reserve(1).is_ok() {
            self.retired.push(activation);
            None
        } else {
            Some(activation)
        }
    }

    /// Forgets dropped checkpoints, returning the activations that only they
    /// needed.
    fn prune(&mut self) -> Vec<Activation> {
        self.live.retain(|(live, _)| live.strong_count() > 0);
        let (keep, free) = core::mem::take(&mut self.retired)
            .into_iter()
            .partition(|a| self.needed(a.serial));
        self.retired = keep;
        free
    }

    pub(super) fn take_retired(&mut self) -> Vec<Activation> {
        core::mem::take(&mut self.retired)
    }
}

/// A point in a replay that the [`Replayer`] that created it can return to.
///
/// See [`Replayer::checkpoint`] and [`Replayer::restore`].
pub struct Checkpoint {
    replayer: usize,
    _live: Arc<()>,
    position: usize,
    instances: usize,
    pending_startup: Option<usize>,
    finished: bool,
    paused: Option<u64>,
    growth_failures: Vec<[u8; codec::GROWTH_FAILED_LEN]>,
    objects: ObjectCounts,
    activations: Vec<ActivationImage>,
    memories: Vec<Vec<u8>>,
    tables: Vec<Vec<FuncTableElem>>,
    globals: Vec<Option<Val>>,
}

// SAFETY: the raw pointers in a checkpoint are only used, by the replayer
// that created it, to restore its own activations and objects.
unsafe impl Send for Checkpoint {}
unsafe impl Sync for Checkpoint {}

struct ActivationImage {
    serial: u64,
    fiber: RawFiberSnapshot,
    control: VMReplayControl,
    context: core::mem::ManuallyDrop<EntryStoreContext>,
    mpk: Option<ProtectionMask>,
    host: Option<HostCall>,
    values: Vec<ValRaw>,
}

/// How many of each kind of object had been constructed.
struct ObjectCounts {
    funcs: usize,
    memories: usize,
    globals: usize,
    tables: usize,
    flags: usize,
    modules: usize,
}

impl Objects {
    fn counts(&self) -> ObjectCounts {
        ObjectCounts {
            funcs: self.funcs.len(),
            memories: self.memories.len(),
            globals: self.globals.len(),
            tables: self.tables.len(),
            flags: self.flags.len(),
            modules: self.modules_defined,
        }
    }

    fn truncate(&mut self, counts: &ObjectCounts) {
        self.funcs.truncate(counts.funcs);
        self.functions_by_key.retain(|_, id| *id < counts.funcs);
        self.memories.truncate(counts.memories);
        self.memories_by_key.retain(|_, id| *id < counts.memories);
        self.globals.truncate(counts.globals);
        self.globals_by_key.retain(|_, id| *id < counts.globals);
        self.tables.truncate(counts.tables);
        self.tables_by_key.retain(|_, id| *id < counts.tables);
        self.flags.truncate(counts.flags);
        self.modules_defined = counts.modules;
    }
}

fn try_copy<T: Clone>(items: &[T]) -> Result<Vec<T>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(items.len())?;
    copy.extend_from_slice(items);
    Ok(copy)
}

impl<'a, T: Send + 'static> Replayer<'a, T> {
    /// Captures the current point of the replay, which [`Replayer::restore`]
    /// can later return to.
    ///
    /// Checkpoints can be taken before replay starts and whenever
    /// [`Replayer::run`] has returned. They copy all guest memories, so they
    /// may be large. Completed guest activations are retained while a
    /// checkpoint that can restore them is alive.
    pub fn checkpoint(&mut self) -> Result<Checkpoint> {
        let driver = &mut self.driver;
        ensure!(
            driver.observed.is_none(),
            "replay is between a guest stop and its trace event"
        );
        let free = driver.checkpoints.prune();
        for activation in free {
            activation.dispose(driver.store);
        }

        let mut activations = Vec::new();
        activations.try_reserve_exact(driver.activations.len())?;
        for activation in &driver.activations {
            activations.push(ActivationImage {
                serial: activation.serial,
                fiber: activation.fiber.as_ref().unwrap().snapshot()?,
                // SAFETY: the activation is not running, and the control
                // block is plain data.
                control: unsafe { core::ptr::read(activation.control.as_ptr()) },
                context: activation.context.rr_clone(),
                mpk: activation.mpk,
                host: activation.host,
                values: try_copy(&activation.values)?,
            });
        }

        let store: &mut StoreOpaque = driver.store;
        let objects = &store.rr.session.as_ref().unwrap().objects;
        let (memories, tables, globals) = (
            try_copy(&objects.memories)?,
            try_copy(&objects.tables)?,
            try_copy(&objects.globals)?,
        );
        let counts = objects.counts();
        let mut memory_images = Vec::new();
        memory_images.try_reserve_exact(memories.len())?;
        for memory in memories {
            memory_images.push(try_copy(memory.rr_data(store))?);
        }
        let mut table_images = Vec::new();
        table_images.try_reserve_exact(tables.len())?;
        for table in tables {
            table_images.push(table.rr_elements(store)?);
        }
        let mut global_images = Vec::new();
        global_images.try_reserve_exact(globals.len())?;
        for global in globals {
            let mutable = global._ty(store).mutability() == crate::Mutability::Var;
            global_images.push(mutable.then(|| global.rr_read(store)));
        }

        let live = try_new::<Arc<_>>(())?;
        let serials = activations.iter().map(|a| a.serial).collect();
        driver.checkpoints.live.try_reserve(1)?;
        driver
            .checkpoints
            .live
            .push((Arc::downgrade(&live), serials));
        let growth_failures = try_copy(driver.growth_failures())?;
        Ok(Checkpoint {
            replayer: driver.checkpoints.replayer,
            _live: live,
            position: driver.reader.position(),
            instances: driver.instances.len(),
            pending_startup: driver.pending_startup,
            finished: driver.finished,
            paused: driver.paused,
            growth_failures,
            objects: counts,
            activations,
            memories: memory_images,
            tables: table_images,
            globals: global_images,
        })
    }

    /// Returns the replay to a checkpoint taken by this replayer.
    ///
    /// Embedder events replayed after the checkpoint are delivered to
    /// observers again as replay proceeds. Breakpoints and other debugger
    /// configuration are not part of the replay and are left unchanged. Frame
    /// handles obtained before restoring become invalid.
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        let driver = &mut self.driver;
        ensure!(
            checkpoint.replayer == driver.checkpoints.replayer,
            "checkpoint belongs to a different replay"
        );

        // Collect the activations to restore before modifying anything.
        let mut current = core::mem::take(&mut driver.activations);
        current.try_reserve(driver.checkpoints.retired.len())?;
        current.append(&mut driver.checkpoints.retired);
        let mut restored = Vec::new();
        restored.try_reserve_exact(checkpoint.activations.len())?;
        for image in &checkpoint.activations {
            match current.iter().position(|a| a.serial == image.serial) {
                Some(index) => restored.push(current.swap_remove(index)),
                None => {
                    driver.activations = restored;
                    driver.activations.append(&mut current);
                    bail!("checkpoint's activations are no longer available");
                }
            }
        }
        // Activations created since the checkpoint may belong to another one.
        for activation in current {
            driver.retire(activation);
        }

        for (activation, image) in restored.iter_mut().zip(&checkpoint.activations) {
            let fiber = activation.fiber.as_mut().unwrap();
            // SAFETY: the snapshot was taken from this fiber at a stop, and
            // everything its continuation refers to is at the same address:
            // the control block, value buffers, store, and module code.
            unsafe {
                fiber.restore(&image.fiber)?;
                core::ptr::write(activation.control.as_ptr(), core::ptr::read(&image.control));
            }
            activation.context = image.context.rr_clone();
            activation.mpk = image.mpk;
            activation.host = image.host;
            activation.values.copy_from_slice(&image.values);
        }
        driver.activations = restored;

        let store: &mut StoreOpaque = driver.store;
        let objects = &mut store.rr.session.as_mut().unwrap().objects;
        objects.truncate(&checkpoint.objects);
        let (memories, tables, globals) = (
            try_copy(&objects.memories)?,
            try_copy(&objects.tables)?,
            try_copy(&objects.globals)?,
        );
        for (memory, bytes) in memories.iter().zip(&checkpoint.memories) {
            memory.rr_restore(store, bytes)?;
        }
        for (table, elements) in tables.iter().zip(&checkpoint.tables) {
            table.rr_restore(store, elements)?;
        }
        for (global, value) in globals.iter().zip(&checkpoint.globals) {
            if let Some(value) = value {
                global._set(store, *value)?;
            }
        }
        // Invalidate frame handles into the replaced stacks.
        store.vm_store_context_mut().execution_version += 1;

        driver.instances.truncate(checkpoint.instances);
        driver.reader.set_position(checkpoint.position);
        driver.pending_startup = checkpoint.pending_startup;
        driver.finished = checkpoint.finished;
        driver.paused = checkpoint.paused;
        driver.observed = None;
        driver.stop = None;
        *driver.growth_failures() = try_copy(&checkpoint.growth_failures)?;
        Ok(())
    }
}
