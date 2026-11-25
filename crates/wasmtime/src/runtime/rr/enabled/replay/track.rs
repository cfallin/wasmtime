//! Tracking the guest's writes to memories for checkpoints.

use super::super::*;
use super::image::{History, Image, Tracked};
use alloc::collections::BTreeMap;

/// The histories of memories that have been checkpointed.
pub(in super::super) struct Histories {
    memories: BTreeMap<RecordedMemory, History>,
    /// The granularity at which writes to memories are tracked.
    pub page_size: usize,
}

impl Default for Histories {
    fn default() -> Self {
        Histories {
            memories: BTreeMap::new(),
            page_size: 4096,
        }
    }
}

impl Histories {
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }
}

struct TrackedMemory<'a>(&'a mut StoreOpaque, Memory);

impl Tracked for TrackedMemory<'_> {
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.1.data_mut_untracked(self.0)
    }

    fn shadow_mut(&mut self) -> &mut [u8] {
        match self.1.vm_shadow_mut(self.0) {
            Some(shadow) => shadow.bytes_mut(),
            None => &mut [],
        }
    }

    fn set_shadow_fill(&mut self, fill: u8) {
        if let Some(shadow) = self.1.vm_shadow_mut(self.0) {
            shadow.set_fill(fill);
        }
    }

    fn resize(&mut self, len: usize) -> Result<()> {
        self.1.rr_resize(self.0, len)
    }
}

impl StoreOpaque {
    fn rr_histories(&mut self) -> &mut Histories {
        &mut self.rr_session().mode.replaying().histories
    }

    /// Runs `f` on the history of a memory, if it has one.
    fn rr_with_memory<R>(
        &mut self,
        id: RecordedMemory,
        f: impl FnOnce(&mut History, &mut dyn Tracked) -> Result<R>,
    ) -> Result<Option<R>> {
        let Some(mut history) = self.rr_histories().memories.remove(&id) else {
            return Ok(None);
        };
        let memory = self.rr_objects().memories[id];
        let result = f(&mut history, &mut TrackedMemory(self, memory));
        self.rr_histories().memories.insert(id, history);
        result.map(Some)
    }

    /// Notes during replay that `range` of `memory` is about to be written.
    pub(crate) fn rr_memory_written(&mut self, memory: Memory, range: Range<usize>) -> Result<()> {
        if !self.rr.replaying() {
            return Ok(());
        }
        let Some(id) = self.rr_objects().find_memory(self, memory) else {
            bail!("write to an unregistered memory");
        };
        self.rr_with_memory(id, |history, memory| history.write(memory, range))?;
        Ok(())
    }

    /// Captures the contents of a memory, starting to track its writes if this
    /// is its first checkpoint.
    pub(super) fn rr_checkpoint_memory(&mut self, id: RecordedMemory) -> Result<Image> {
        if !self.rr_histories().memories.contains_key(&id) {
            let memory = self.rr_objects().memories[id];
            ensure!(
                memory.vm_shadow(self).is_some(),
                "replay checkpoints do not support this memory"
            );
            let page_size = self.rr_histories().page_size;
            let history = History::new(page_size, &mut TrackedMemory(self, memory));
            self.rr_histories().memories.insert(id, history);
        }
        let image = self.rr_with_memory(id, |history, memory| history.checkpoint(memory))?;
        Ok(image.unwrap())
    }

    pub(super) fn rr_restore_memory(&mut self, id: RecordedMemory, image: &Image) -> Result<()> {
        self.rr_with_memory(id, |history, memory| history.restore(memory, image))?;
        Ok(())
    }
}
