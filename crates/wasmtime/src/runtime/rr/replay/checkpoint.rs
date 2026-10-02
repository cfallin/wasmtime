//! Checkpoints of a replay, for rewinding it.
//!
//! A checkpoint is taken between trace events, or while an activation is
//! stopped at a debug event. It captures:
//!
//! * the driver's protocol state: the trace position, the activations and
//!   their parked host calls, the startup and growth-failure state, and the
//!   identities of the objects and instances constructed so far;
//! * each activation's raw fiber snapshot, control block, saved
//!   `VMStoreContext` state, protection-key mask, and value buffer;
//! * the guest state of every object constructed so far: memory sizes and
//!   contents, function table sizes and elements, and mutable globals
//!   (including component instance flags).
//!
//! Memory contents are kept as layered images (see `overlay`): a checkpoint
//! copies only the pages written since the previous checkpoint, which the
//! memory's watchpoint shadow detects, and restoring writes only the pages
//! that may differ from the current contents.
//!
//! Restoring puts all of this back in place. Activations keep their fiber
//! stacks, control blocks, and buffers at the same addresses, so the restored
//! stacks remain valid. An activation that completed after a checkpoint is
//! therefore retained, rather than freed, while a checkpoint can restore it.
//! Objects constructed after a checkpoint stay in the store but become
//! unreachable once it is restored; replay constructs new ones again, so each
//! checkpoint keeps the object identities of its own timeline. Compiled
//! modules are immutable and shared between timelines.

use super::*;
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
    globals: overlay::ValuesHistory,
}

impl Default for Checkpoints {
    fn default() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Checkpoints {
            replayer: NEXT.fetch_add(1, Ordering::Relaxed),
            live: Vec::new(),
            retired: Vec::new(),
            globals: Default::default(),
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
    pending_startup: Option<usize>,
    finished: bool,
    paused: Option<u64>,
    pending_write: Option<(Memory, Range<usize>)>,
    growth_failures: Vec<[u8; codec::GROWTH_FAILED_LEN]>,
    objects: Objects,
    instance_list: Vec<crate::Instance>,
    activations: Vec<ActivationImage>,
    memories: Vec<overlay::Image>,
    tables: Vec<overlay::Image>,
    globals: Arc<overlay::ValuesLayer>,
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

impl Checkpoint {
    /// The number of bytes of guest memory contents copied for this
    /// checkpoint: those written since the previous checkpoint, at the
    /// granularity set by [`Replayer::set_checkpoint_page_size`]. Unchanged
    /// contents are shared with earlier checkpoints.
    pub fn memory_bytes(&self) -> usize {
        self.memories.iter().map(|m| m.stored_bytes()).sum()
    }

    /// The number of bytes of table elements copied for this checkpoint:
    /// those in groups of 64 slots written since the previous checkpoint.
    pub fn table_bytes(&self) -> usize {
        self.tables.iter().map(|t| t.stored_bytes()).sum()
    }
}

impl Objects {
    /// A copy of these identities, sharing the compiled modules.
    fn try_clone_identities(&self) -> Result<Objects> {
        Ok(Objects {
            funcs: try_copy(&self.funcs)?,
            functions_by_key: self.functions_by_key.try_clone()?,
            memories: try_copy(&self.memories)?,
            memories_by_key: self.memories_by_key.try_clone()?,
            globals: try_copy(&self.globals)?,
            globals_by_key: self.globals_by_key.try_clone()?,
            tables: try_copy(&self.tables)?,
            tables_by_key: self.tables_by_key.try_clone()?,
            flags: try_copy(&self.flags)?,
            modules: Vec::new(),
            modules_defined: self.modules_defined,
        })
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
    /// [`Replayer::run`] has returned. A checkpoint retains the guest memory
    /// pages that changed since the previous one. Completed guest activations
    /// are retained while a checkpoint that can restore them is alive.
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
        let identities = objects.try_clone_identities()?;
        let mut memory_images = Vec::new();
        memory_images.try_reserve_exact(memories.len())?;
        for memory in memories {
            memory_images.push(checkpoint_object(store, overlay::Tracked::Memory(memory))?);
        }
        if let Some((memory, range)) = &driver.pending_write {
            store.rr_dirty(*memory, range.clone())?;
        }
        let mut table_images = Vec::new();
        table_images.try_reserve_exact(tables.len())?;
        for table in tables {
            table_images.push(checkpoint_object(store, overlay::Tracked::Table(table))?);
        }
        let mut values = Vec::new();
        values.try_reserve_exact(globals.len())?;
        for global in globals {
            values.push((global.rr_key(store), global.rr_raw(store)));
        }
        let global_images = driver.checkpoints.globals.checkpoint(values)?;

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
            instance_list: try_copy(&driver.instances)?,
            pending_startup: driver.pending_startup,
            finished: driver.finished,
            paused: driver.paused,
            pending_write: driver.pending_write.clone(),
            growth_failures,
            objects: identities,
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
        let identities = checkpoint.objects.try_clone_identities()?;
        let objects = &mut store.rr.session.as_mut().unwrap().objects;
        let modules = core::mem::take(&mut objects.modules);
        *objects = identities;
        objects.modules = modules;
        let (memories, tables, globals) = (
            try_copy(&objects.memories)?,
            try_copy(&objects.tables)?,
            try_copy(&objects.globals)?,
        );
        for (memory, image) in memories.iter().zip(&checkpoint.memories) {
            store
                .rr_with_history(overlay::Tracked::Memory(*memory), |history, memory| {
                    history.restore(memory, image)
                })?
                .expect("a checkpointed memory has a history");
        }
        if let Some((memory, range)) = &checkpoint.pending_write {
            store.rr_dirty(*memory, range.clone())?;
        }
        for (table, elements) in tables.iter().zip(&checkpoint.tables) {
            store
                .rr_with_history(overlay::Tracked::Table(*table), |history, table| {
                    history.restore(table, elements)
                })?
                .expect("a checkpointed table has a history");
        }
        let mut by_key = alloc::collections::BTreeMap::new();
        for global in &globals {
            by_key.insert(global.rr_key(store), *global);
        }
        driver.checkpoints.globals.restore(
            &checkpoint.globals,
            by_key.keys().copied(),
            |key| by_key[&key].rr_raw(store),
            |key, value| by_key[&key].rr_set_raw(store, value),
        );
        // Invalidate frame handles into the replaced stacks.
        store.vm_store_context_mut().execution_version += 1;

        driver.instances = try_copy(&checkpoint.instance_list)?;
        driver.reader.set_position(checkpoint.position);
        driver.pending_startup = checkpoint.pending_startup;
        driver.finished = checkpoint.finished;
        driver.paused = checkpoint.paused;
        driver.pending_write = checkpoint.pending_write.clone();
        driver.observed = None;
        driver.stop = None;
        *driver.growth_failures() = try_copy(&checkpoint.growth_failures)?;
        Ok(())
    }
}

/// Captures the contents of `object`, starting to track its writes if this
/// is its first checkpoint.
fn checkpoint_object(store: &mut StoreOpaque, object: overlay::Tracked) -> Result<overlay::Image> {
    let checkpoint = |history: &mut overlay::History, object: &mut dyn overlay::TrackedMemory| {
        history.checkpoint(object)
    };
    if let Some(image) = store.rr_with_history(object, checkpoint)? {
        return Ok(image);
    }
    let page_size = match object {
        overlay::Tracked::Memory(memory) => {
            ensure!(
                memory.vm_shadow(store).is_some(),
                "replay checkpoints do not support this memory"
            );
            let Mode::Replaying { page_size, .. } = store.rr_session().mode else {
                unreachable!()
            };
            page_size
        }
        overlay::Tracked::Table(table) => {
            // Check that the table's elements can be checkpointed.
            table.rr_slots(store)?;
            TABLE_PAGE_SLOTS * table.rr_slot_size(store)
        }
    };
    let key = object.key(store);
    let history = overlay::History::new(page_size, &mut overlay::StoreObject { store, object });
    let Mode::Replaying { histories, .. } = &mut store.rr_session().mode else {
        unreachable!()
    };
    histories.insert(key, history);
    Ok(store.rr_with_history(object, checkpoint)?.unwrap())
}

/// The number of table slots that checkpoints track together.
const TABLE_PAGE_SLOTS: usize = 64;
