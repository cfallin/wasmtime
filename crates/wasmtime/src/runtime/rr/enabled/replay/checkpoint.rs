//! Checkpoints of a replay, for rewinding it.
//!
//! A checkpoint captures the state of the replay itself, including each
//! activation's stack, and of everything that the guest can change.
//!
//! Restoring puts all of this back in place, at the same addresses, so that the
//! restored stacks remain valid. Activations that have finished are therefore
//! kept while a checkpoint needs them.
//!
//! Objects that were created after a checkpoint also stay in the store, and
//! their IDs still refer to them. They are out of reach of the guest, until
//! replay reaches their creation again. Rather than create them again, it then
//! restores a checkpoint from just after they were first created. Handles to
//! objects thus stay valid, and rewinding repeatedly does not leak them.

use super::activation::ActivationImage;
use super::image::{Image, PagedImage};
use super::*;
use crate::runtime::vm::RrSegments;
use alloc::sync::Weak;
use core::sync::atomic::{AtomicUsize, Ordering};
use objects::RecordedTable;
use wasmtime_environ::EntityRef as _;

/// The page size of images of globals and tables.
const PAGE_SIZE: usize = 1024;

/// The checkpoints of a replayer.
pub(super) struct Checkpoints {
    /// Identifies this replayer.
    replayer: usize,
    /// Which activations each live checkpoint can restore.
    live: Vec<(Weak<()>, Vec<Serial>)>,
    /// Finished activations that a live checkpoint can restore.
    retired: TryVec<Activation>,
    /// Once there is a checkpoint, a checkpoint just after each run of records
    /// that creates objects, by the run's offset.
    creations: BTreeMap<usize, Checkpoint>,
    /// The most recent images, to share pages with.
    globals: Arc<PagedImage>,
    tables: Vec<Arc<PagedImage>>,
    segments: Vec<Arc<RrSegments>>,
    #[cfg(feature = "gc")]
    gc: Option<Arc<crate::store::RrGcImage>>,
}

impl Default for Checkpoints {
    fn default() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Checkpoints {
            replayer: NEXT.fetch_add(1, Ordering::Relaxed),
            live: Vec::new(),
            retired: TryVec::new(),
            creations: BTreeMap::new(),
            globals: Arc::default(),
            tables: Vec::new(),
            segments: Vec::new(),
            #[cfg(feature = "gc")]
            gc: None,
        }
    }
}

impl Checkpoints {
    /// Whether there has been a checkpoint.
    fn started(&self) -> bool {
        !self.live.is_empty() || !self.creations.is_empty()
    }

    fn needs(&self, activation: &Activation) -> bool {
        let serial = activation.serial;
        self.live
            .iter()
            .any(|(live, serials)| live.strong_count() > 0 && serials.contains(&serial))
    }

    /// Takes the finished activations, for disposal.
    pub fn take_retired(&mut self) -> TryVec<Activation> {
        core::mem::take(&mut self.retired)
    }
}

/// A point in a replay that the [`Replayer`] that created it can return to.
///
/// See [`Replayer::checkpoint`] and [`Replayer::restore`].
pub struct Checkpoint {
    replayer: usize,
    _live: Arc<()>,
    offset: usize,
    next_module: RecordedModule,
    pending_startup: Option<RecordedFunc>,
    finished: bool,
    paused: Option<RecordedCall>,
    #[cfg(feature = "debug")]
    pending_write: Option<(Memory, Range<usize>)>,
    growth_failures: TryVec<GrowthFailure>,
    activations: TryVec<ActivationImage>,
    memories: TryVec<Image>,
    tables: Vec<Arc<PagedImage>>,
    globals: Arc<PagedImage>,
    segments: Vec<Arc<RrSegments>>,
    #[cfg(feature = "gc")]
    gc: Option<Arc<crate::store::RrGcImage>>,
    copied_bytes: usize,
}

impl Checkpoint {
    /// The number of bytes of guest memories, tables, and the GC heap that were
    /// copied for this checkpoint. Those that were not written since an earlier
    /// checkpoint are shared with it.
    pub fn copied_bytes(&self) -> usize {
        self.copied_bytes
    }
}

impl Replayer {
    /// Captures the current point of the replay, which [`Replayer::restore`] can
    /// return to.
    ///
    /// Checkpoints can be taken before replay starts, and whenever
    /// [`Replayer::run`] has returned without an error.
    pub fn checkpoint(&mut self) -> Result<Checkpoint> {
        ensure!(
            self.observed.is_none() && !self.failed,
            "cannot checkpoint replay in the middle of a step, or after it failed"
        );
        self.capture()
    }

    fn capture(&mut self) -> Result<Checkpoint> {
        self.prune_checkpoints()?;

        let mut activations = TryVec::new();
        let mut serials = Vec::new();
        for activation in &self.activations {
            activations.push(activation.snapshot()?)?;
            serials.push(activation.serial);
        }

        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let mut copied_bytes = 0;
        let mut memories = TryVec::new();
        for i in 0..self.defined.memories {
            let image = store.rr_checkpoint_memory(RecordedMemory::new(i))?;
            copied_bytes += image.copied_bytes();
            memories.push(image)?;
        }
        // The shadow no longer reports a write that has already been checked.
        #[cfg(feature = "debug")]
        if let Some((memory, range)) = &self.pending_write {
            store.rr_memory_written(*memory, range.clone())?;
        }
        let mut tables = Vec::new();
        for i in 0..self.defined.tables {
            let table = store.rr_objects().tables[RecordedTable::new(i)];
            let prev = self.checkpoints.tables.get(i).map(|t| &**t);
            let image = PagedImage::capture(table.rr_slots(store)?, PAGE_SIZE, prev)?;
            copied_bytes += image.copied_bytes();
            tables.push(try_new::<Arc<_>>(image)?);
        }
        self.checkpoints.tables = tables;

        let mut values = Vec::new();
        let globals = &store.rr_objects().globals;
        for global in globals.values().take(self.defined.globals) {
            values.extend_from_slice(&global.rr_raw(store));
        }
        let prev = Some(&*self.checkpoints.globals);
        let globals = PagedImage::capture(&values, PAGE_SIZE, prev)?;
        self.checkpoints.globals = try_new::<Arc<_>>(globals)?;

        let mut segments = Vec::new();
        for (i, instance) in self.instances[..self.defined.instances].iter().enumerate() {
            let prev = self.checkpoints.segments.get(i).map(|s| &**s);
            let image = instance.id.get_mut(store).rr_save_segments(prev)?;
            segments.push(try_new::<Arc<_>>(image)?);
        }
        self.checkpoints.segments = segments;

        #[cfg(feature = "gc")]
        {
            let page_size = store.rr_session().mode.replaying().histories.page_size;
            let prev = self.checkpoints.gc.as_deref();
            let image = store.rr_gc_checkpoint(prev, page_size)?;
            copied_bytes += image.as_ref().map_or(0, |image| image.copied_bytes());
            self.checkpoints.gc = image.map(|i| try_new::<Arc<_>>(i)).transpose()?;
        }

        let live = try_new::<Arc<_>>(())?;
        self.checkpoints.live.push((Arc::downgrade(&live), serials));
        Ok(Checkpoint {
            replayer: self.checkpoints.replayer,
            _live: live,
            offset: self.offset,
            next_module: self.next_module,
            pending_startup: self.pending_startup,
            finished: self.finished,
            paused: self.paused,
            #[cfg(feature = "debug")]
            pending_write: self.pending_write.clone(),
            growth_failures: self.replaying().growth_failures.try_clone()?,
            activations,
            memories,
            tables: self.checkpoints.tables.clone(),
            globals: self.checkpoints.globals.clone(),
            segments: self.checkpoints.segments.clone(),
            #[cfg(feature = "gc")]
            gc: self.checkpoints.gc.clone(),
            copied_bytes,
        })
    }

    /// Returns the replay to a checkpoint that this replayer took.
    ///
    /// Observers see events that were recorded after the checkpoint again as
    /// replay reaches them. Breakpoints, single-stepping, and watchpoints are
    /// left unchanged. Frame handles and unrooted references become invalid.
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        ensure!(
            checkpoint.replayer == self.checkpoints.replayer,
            "checkpoint belongs to a different replay"
        );
        self.failed = true;
        self.restore_activations(checkpoint)?;

        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        for (i, image) in checkpoint.memories.iter().enumerate() {
            store.rr_restore_memory(RecordedMemory::new(i), image)?;
        }
        #[cfg(feature = "debug")]
        if let Some((memory, range)) = &checkpoint.pending_write {
            store.rr_memory_written(*memory, range.clone())?;
        }
        for (i, image) in checkpoint.tables.iter().enumerate() {
            let table = store.rr_objects().tables[RecordedTable::new(i)];
            table.rr_resize(store, image.len())?;
            image.restore_into(table.rr_slots(store)?);
        }

        let mut values = Vec::new();
        values.resize(checkpoint.globals.len(), 0);
        checkpoint.globals.restore_into(&mut values);
        let globals = &store.rr_objects().globals;
        for (global, value) in globals.values().zip(values.as_chunks().0) {
            global.rr_set_raw(store, value);
        }
        for (instance, segments) in self.instances.iter().zip(&checkpoint.segments) {
            instance.id.get_mut(store).rr_restore_segments(segments)?;
        }
        #[cfg(feature = "gc")]
        if let Some(image) = &checkpoint.gc {
            store.rr_gc_restore(image)?;
        }
        store.on_wasm_to_host_transition();

        self.checkpoints.globals = checkpoint.globals.clone();
        self.checkpoints.tables = checkpoint.tables.clone();
        self.checkpoints.segments = checkpoint.segments.clone();
        #[cfg(feature = "gc")]
        {
            self.checkpoints.gc = checkpoint.gc.clone();
        }
        self.defined = Defined {
            memories: checkpoint.memories.len(),
            tables: checkpoint.tables.len(),
            globals: checkpoint.globals.len() / size_of::<[u8; 16]>(),
            instances: checkpoint.segments.len(),
        };
        self.clear_undefined_objects()?;
        self.offset = checkpoint.offset;
        self.next_module = checkpoint.next_module;
        self.pending_startup = checkpoint.pending_startup;
        self.finished = checkpoint.finished;
        self.paused = checkpoint.paused;
        #[cfg(feature = "debug")]
        {
            self.pending_write = checkpoint.pending_write.clone();
        }
        self.replaying().growth_failures = checkpoint.growth_failures.try_clone()?;
        self.observed = None;
        self.stop = None;
        self.failed = false;
        Ok(())
    }

    fn restore_activations(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        let mut available = core::mem::take(&mut self.activations);
        for activation in self.checkpoints.take_retired() {
            available.push(activation)?;
        }
        for image in &checkpoint.activations {
            let index = available.iter().position(|a| a.serial == image.serial);
            let index = index.ok_or_else(|| format_err!("activation is no longer available"))?;
            let mut activation = available.swap_remove(index);
            // SAFETY: everything that the stack refers to is at the same
            // address.
            let restored = unsafe { activation.restore(image) };
            self.activations.push(activation)?;
            restored?;
        }
        // Another checkpoint may need the rest.
        for activation in available {
            self.retire(activation)?;
        }
        Ok(())
    }

    /// Keeps a finished activation if a checkpoint can restore it, and frees it
    /// otherwise.
    pub(super) fn retire(&mut self, activation: Activation) -> Result<()> {
        if self.checkpoints.needs(&activation) {
            self.checkpoints.retired.push(activation)?;
        } else {
            activation.dispose(self.store.as_context_mut().0);
        }
        Ok(())
    }

    /// Frees the activations that only dropped checkpoints needed.
    fn prune_checkpoints(&mut self) -> Result<()> {
        self.checkpoints
            .live
            .retain(|(live, _)| live.strong_count() > 0);
        for activation in self.checkpoints.take_retired() {
            self.retire(activation)?;
        }
        Ok(())
    }

    /// Clears the references to GC objects from objects that have not been
    /// created at this point, which the collector would otherwise trace.
    fn clear_undefined_objects(&mut self) -> Result<()> {
        let store: &mut StoreOpaque = self.store.as_context_mut().0;
        let objects = store.rr_objects();
        for global in objects.globals.values().skip(self.defined.globals) {
            if global
                ._ty(store)
                .content()
                .is_vmgcref_type_and_points_to_object()
            {
                global.rr_set_raw(store, &[0; 16]);
            }
        }
        for i in self.defined.tables..store.rr_objects().tables.len() {
            let table = store.rr_objects().tables[RecordedTable::new(i)];
            if table
                .ty_(store)
                .element()
                .is_vmgcref_type_and_points_to_object()
            {
                table.rr_slots(store)?.fill(0);
            }
        }
        for instance in &self.instances[self.defined.instances..] {
            instance.id.get_mut(store).rr_clear_gc_segments();
        }
        Ok(())
    }

    /// Replays the run of records at the current offset if they create objects
    /// and checkpoints may make them execute more than once. Returns whether
    /// this did anything.
    pub(super) fn replay_creations(&mut self) -> Result<bool> {
        if !self.checkpoints.started() || !self.at_creation() {
            return Ok(false);
        }
        self.require_idle()?;
        let offset = self.offset;
        // Detach these, as restoring one of them borrows the replayer.
        let mut creations = core::mem::take(&mut self.checkpoints.creations);
        let result = match creations.get(&offset) {
            Some(checkpoint) => self.restore(checkpoint),
            None => (|| {
                while self.at_creation() {
                    self.step_record()?;
                }
                creations.insert(offset, self.capture()?);
                Ok(())
            })(),
        };
        self.checkpoints.creations = creations;
        result?;
        Ok(true)
    }

    fn at_creation(&self) -> bool {
        matches!(
            Reader::at(&self.trace.bytes, self.offset).peek_tag(),
            Some(
                codec::HOST
                    | codec::MODULE
                    | codec::INSTANCE
                    | codec::GLOBAL
                    | codec::MEMORY
                    | codec::TABLE
            )
        )
    }

    /// Sets the granularity, in bytes, at which checkpoints track writes to
    /// guest memory. The default is 4096. This must be set before the first
    /// checkpoint.
    pub fn set_checkpoint_page_size(&mut self, size: usize) -> Result<()> {
        ensure!(size > 0, "checkpoint page size must be nonzero");
        let histories = &mut self.replaying().histories;
        ensure!(
            histories.is_empty(),
            "checkpoint page size set after a checkpoint"
        );
        histories.page_size = size;
        Ok(())
    }
}
