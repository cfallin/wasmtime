//! Layered, reference-counted images of linear memories, for checkpoints.
//!
//! A memory's contents at a checkpoint are an [`Image`]: a chain of immutable
//! [`Layer`]s, each holding the pages written since its parent's checkpoint,
//! as they were at its own checkpoint. A page in no layer of the chain has
//! not been written since the memory's [`History`] began, and still has its
//! contents from then, which the history records the first time the page is
//! written. Checkpoints taken on different branches of a replay share their
//! common ancestor layers.
//!
//! Writes are detected without copying or comparing memory: each tracked
//! memory's shadow marks every byte of a page "clean" (`WATCH_CLEAN`) until
//! the first write to the page after a checkpoint, which compiled code reports
//! through the memory's watchpoint builtin. The page size is an arbitrary
//! software granularity, independent of the host's pages.

use crate::Memory;
use crate::prelude::*;
use crate::runtime::vm::WATCH_CLEAN;
use crate::store::StoreOpaque;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use core::ops::Range;

/// The contents of a memory at a checkpoint.
pub(crate) type Image = Arc<Layer>;

/// The pages written between a checkpoint and its parent, as they were at the
/// checkpoint.
pub(crate) struct Layer {
    parent: Option<Image>,
    depth: usize,
    /// The memory's length, in bytes, at the checkpoint.
    len: usize,
    pages: BTreeMap<usize, Arc<[u8]>>,
}

impl Layer {
    /// The bytes of memory contents this layer holds.
    pub(crate) fn stored_bytes(&self) -> usize {
        self.pages.values().map(|page| page.len()).sum()
    }

    /// The contents of page `index` at this checkpoint, if any layer of the
    /// chain has them.
    fn page(&self, index: usize) -> Option<&Arc<[u8]>> {
        let mut layer = self;
        loop {
            if let Some(page) = layer.pages.get(&index) {
                return Some(page);
            }
            layer = layer.parent.as_deref()?;
        }
    }

    /// The layers of `image` up to, but excluding, `ancestor`.
    fn above<'a>(
        image: Option<&'a Image>,
        ancestor: Option<&Image>,
    ) -> impl Iterator<Item = &'a Layer> {
        let mut layer = image.map(|i| &**i);
        core::iter::from_fn(move || {
            let current = layer?;
            if ancestor.is_some_and(|a| core::ptr::eq(current, &**a)) {
                return None;
            }
            layer = current.parent.as_deref();
            Some(current)
        })
    }

    /// The nearest common ancestor of two images.
    fn common_ancestor<'a>(
        mut a: Option<&'a Image>,
        mut b: Option<&'a Image>,
    ) -> Option<&'a Image> {
        let depth = |i: Option<&Image>| i.map_or(0, |i| i.depth);
        while depth(a) > depth(b) {
            a = a.and_then(|i| i.parent.as_ref());
        }
        while depth(b) > depth(a) {
            b = b.and_then(|i| i.parent.as_ref());
        }
        while let (Some(x), Some(y)) = (a, b) {
            if Arc::ptr_eq(x, y) {
                return Some(x);
            }
            a = x.parent.as_ref();
            b = y.parent.as_ref();
        }
        None
    }

    /// The pages whose contents may differ between two images: those written
    /// on either side since their common ancestor.
    fn changed_between(a: Option<&Image>, b: Option<&Image>) -> BTreeSet<usize> {
        let ancestor = Layer::common_ancestor(a, b);
        Layer::above(a, ancestor)
            .chain(Layer::above(b, ancestor))
            .flat_map(|layer| layer.pages.keys().copied())
            .collect()
    }
}

/// A memory, or another object viewed as bytes, whose contents a [`History`]
/// tracks.
pub(crate) trait TrackedMemory {
    /// The memory's bytes.
    fn bytes(&mut self) -> &[u8];
    /// The memory's bytes.
    fn bytes_mut(&mut self) -> &mut [u8];
    /// The memory's shadow, one byte per byte of the memory.
    fn shadow_mut(&mut self) -> &mut [u8];
    /// Resizes the memory. Bytes added read as zero, and their shadow bytes
    /// are set to the fill value of [`TrackedMemory::set_shadow_fill`].
    fn resize(&mut self, len: usize) -> Result<()>;
    /// Sets the shadow value of bytes added when the memory grows.
    fn set_shadow_fill(&mut self, fill: u8);
}

/// An object of a store whose contents checkpoints track.
#[derive(Clone, Copy)]
pub(crate) enum Tracked {
    /// A linear memory, whose shadow reports guest writes.
    Memory(Memory),
    /// A table, viewed as the bytes of its elements, whose writes are always
    /// reported.
    Table(crate::Table),
}

/// Identifies a tracked object within its store.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TrackedKey {
    Memory(usize),
    Table(u32, u32),
}

impl Tracked {
    pub(crate) fn key(&self, store: &StoreOpaque) -> TrackedKey {
        match self {
            Tracked::Memory(memory) => TrackedKey::Memory(memory.rr_key(store)),
            Tracked::Table(table) => {
                let (instance, index) = table.rr_key();
                TrackedKey::Table(instance, index)
            }
        }
    }
}

/// A tracked object of a store.
pub(crate) struct StoreObject<'a> {
    pub(crate) store: &'a mut StoreOpaque,
    pub(crate) object: Tracked,
}

impl TrackedMemory for StoreObject<'_> {
    fn bytes(&mut self) -> &[u8] {
        self.bytes_mut()
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        match self.object {
            Tracked::Memory(memory) => memory.rr_data_mut(self.store),
            // Only function and GC reference tables are checkpointed.
            Tracked::Table(table) => table.rr_slots(self.store).unwrap_or_default(),
        }
    }
    fn shadow_mut(&mut self) -> &mut [u8] {
        match self.object {
            Tracked::Memory(memory) => match memory.vm_shadow_mut(self.store) {
                Some(shadow) => shadow.bytes_mut(),
                None => &mut [],
            },
            Tracked::Table(_) => &mut [],
        }
    }
    fn resize(&mut self, len: usize) -> Result<()> {
        match self.object {
            Tracked::Memory(memory) => memory.rr_resize(self.store, len),
            Tracked::Table(table) => table.rr_resize(self.store, len),
        }
    }
    fn set_shadow_fill(&mut self, fill: u8) {
        if let Tracked::Memory(memory) = self.object
            && let Some(shadow) = memory.vm_shadow_mut(self.store)
        {
            shadow.set_fill(fill);
        }
    }
}

/// The checkpointed contents of one memory over its lifetime.
pub(crate) struct History {
    page_size: usize,
    /// Each page's contents when it was first written after this history
    /// began, which is when its contents last agreed on every branch.
    base: BTreeMap<usize, Arc<[u8]>>,
    /// The image the memory's contents are based on.
    head: Option<Image>,
    /// The pages written since `head`.
    dirty: BTreeSet<usize>,
}

impl History {
    /// Starts tracking `memory`, whose current contents become its base.
    pub(crate) fn new(page_size: usize, memory: &mut dyn TrackedMemory) -> Self {
        assert!(page_size > 0);
        memory
            .shadow_mut()
            .iter_mut()
            .for_each(|b| *b |= WATCH_CLEAN);
        memory.set_shadow_fill(WATCH_CLEAN);
        History {
            page_size,
            base: BTreeMap::new(),
            head: None,
            dirty: BTreeSet::new(),
        }
    }

    fn pages(&self, range: Range<usize>) -> Range<usize> {
        range.start / self.page_size..range.end.div_ceil(self.page_size)
    }

    fn page_bytes(&self, index: usize, len: usize) -> Range<usize> {
        let start = (index * self.page_size).min(len);
        start..(start + self.page_size).min(len)
    }

    /// A copy of page `index` of `bytes`, zero-padded to the page size.
    fn copy_page(&self, bytes: &[u8], index: usize) -> Result<Arc<[u8]>> {
        let mut page = Vec::new();
        page.try_reserve_exact(self.page_size)?;
        page.extend_from_slice(&bytes[self.page_bytes(index, bytes.len())]);
        page.resize(self.page_size, 0);
        Ok(page.into())
    }

    fn set_clean(&self, memory: &mut dyn TrackedMemory, index: usize, clean: bool) {
        let range = self.page_bytes(index, memory.shadow_mut().len());
        for byte in &mut memory.shadow_mut()[range] {
            if clean {
                *byte |= WATCH_CLEAN;
            } else {
                *byte &= !WATCH_CLEAN;
            }
        }
    }

    /// Records that `range` of the memory is about to be written.
    pub(crate) fn write(
        &mut self,
        memory: &mut dyn TrackedMemory,
        range: Range<usize>,
    ) -> Result<()> {
        for index in self.pages(range) {
            if self.dirty.contains(&index) {
                continue;
            }
            if !self.base.contains_key(&index) {
                let page = self.copy_page(memory.bytes(), index)?;
                self.base.insert(index, page);
            }
            self.dirty.insert(index);
            self.set_clean(memory, index, false);
        }
        Ok(())
    }

    /// Captures the memory's current contents as a new image.
    pub(crate) fn checkpoint(&mut self, memory: &mut dyn TrackedMemory) -> Result<Image> {
        let len = memory.bytes().len();
        let mut pages = BTreeMap::new();
        for &index in &self.dirty {
            if index * self.page_size < len {
                pages.insert(index, self.copy_page(memory.bytes(), index)?);
            }
            self.set_clean(memory, index, true);
        }
        self.dirty.clear();
        let image = try_new::<Arc<_>>(Layer {
            depth: self.head.as_ref().map_or(1, |h| h.depth + 1),
            parent: self.head.take(),
            len,
            pages,
        })?;
        self.head = Some(image.clone());
        Ok(image)
    }

    /// Restores the memory's contents to `image`, which this history created.
    /// Only pages that may differ are written.
    pub(crate) fn restore(&mut self, memory: &mut dyn TrackedMemory, image: &Image) -> Result<()> {
        let mut changed = Layer::changed_between(self.head.as_ref(), Some(image));
        changed.extend(self.dirty.iter().copied());
        memory.resize(image.len)?;
        let len = image.len;
        for index in changed {
            let range = self.page_bytes(index, len);
            if range.is_empty() {
                continue;
            }
            // A page that may have changed has been written, so it has a base.
            let page = image
                .page(index)
                .or_else(|| self.base.get(&index))
                .expect("a written page has a base");
            memory.bytes_mut()[range.clone()].copy_from_slice(&page[..range.len()]);
            self.set_clean(memory, index, true);
        }
        self.dirty.clear();
        self.head = Some(image.clone());
        Ok(())
    }

    /// The number of distinct page copies this history and `images` hold.
    #[cfg(test)]
    fn pages_held(&self, images: &[&Image]) -> usize {
        let mut seen = BTreeSet::new();
        let mut layer_pages = 0;
        for image in images {
            for layer in Layer::above(Some(image), None) {
                if seen.insert(core::ptr::from_ref(layer)) {
                    layer_pages += layer.pages.len();
                }
            }
        }
        layer_pages + self.base.len()
    }
}

/// The contents of untracked bytes, as pages shared with the previous image
/// where they are unchanged. Capturing compares every page with the previous
/// image, and restoring compares every page with the current contents.
pub(crate) struct PagedImage {
    len: usize,
    pages: Vec<Arc<[u8]>>,
}

impl PagedImage {
    /// Captures `bytes`, sharing the pages unchanged since `prev`.
    pub(crate) fn capture(
        bytes: &[u8],
        page_size: usize,
        prev: Option<&PagedImage>,
    ) -> Result<Self> {
        let mut pages = Vec::new();
        pages.try_reserve_exact(bytes.len().div_ceil(page_size))?;
        for (i, chunk) in bytes.chunks(page_size).enumerate() {
            match prev.and_then(|p| p.pages.get(i)) {
                Some(page) if **page == *chunk => pages.push(page.clone()),
                _ => {
                    let mut page = Vec::new();
                    page.try_reserve_exact(chunk.len())?;
                    page.extend_from_slice(chunk);
                    pages.push(page.into());
                }
            }
        }
        Ok(PagedImage {
            len: bytes.len(),
            pages,
        })
    }

    /// The length of the captured bytes.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The bytes of the pages this image does not share with `prev`.
    pub(crate) fn stored_bytes(&self, prev: Option<&PagedImage>) -> usize {
        self.pages
            .iter()
            .enumerate()
            .filter(|(i, page)| {
                !prev
                    .and_then(|p| p.pages.get(*i))
                    .is_some_and(|p| Arc::ptr_eq(p, page))
            })
            .map(|(_, page)| page.len())
            .sum()
    }

    /// Writes the pages of `bytes`, which has this image's length, that
    /// differ from this image.
    pub(crate) fn restore_into(&self, bytes: &mut [u8]) {
        debug_assert_eq!(bytes.len(), self.len);
        let page_size = self.pages.first().map_or(1, |p| p.len());
        for (chunk, page) in bytes.chunks_mut(page_size).zip(&self.pages) {
            if *chunk != **page {
                chunk.copy_from_slice(page);
            }
        }
    }
}

/// The raw value of a global.
pub(crate) type Value = [u8; 16];

/// The globals whose values changed between a checkpoint and its parent, as
/// they were at the checkpoint, by key.
pub(crate) struct ValuesLayer {
    parent: Option<Arc<ValuesLayer>>,
    values: BTreeMap<usize, Value>,
}

impl ValuesLayer {
    /// The number of values this layer holds.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.values.len()
    }
}

/// The checkpointed values of a set of globals. Global writes are not
/// tracked: checkpoints and restores compare values instead, since globals
/// are few and some (such as a shadow stack pointer) are written constantly.
#[derive(Default)]
pub(crate) struct ValuesHistory {
    head: Option<Arc<ValuesLayer>>,
    /// The values at `head`.
    head_values: BTreeMap<usize, Value>,
}

impl ValuesHistory {
    /// Captures the current `values`, keeping only those that differ from the
    /// previous checkpoint's.
    pub(crate) fn checkpoint(
        &mut self,
        values: impl IntoIterator<Item = (usize, Value)>,
    ) -> Result<Arc<ValuesLayer>> {
        let mut changed = BTreeMap::new();
        for (key, value) in values {
            if self.head_values.get(&key) != Some(&value) {
                changed.insert(key, value);
                self.head_values.insert(key, value);
            }
        }
        let layer = try_new::<Arc<_>>(ValuesLayer {
            parent: self.head.take(),
            values: changed,
        })?;
        self.head = Some(layer.clone());
        Ok(layer)
    }

    /// Restores `image`'s values of `keys`, calling `write` for each value
    /// that differs from its current one, as given by `read`.
    pub(crate) fn restore(
        &mut self,
        image: &Arc<ValuesLayer>,
        keys: impl IntoIterator<Item = usize>,
        mut read: impl FnMut(usize) -> Value,
        mut write: impl FnMut(usize, &Value),
    ) {
        let mut wanted = keys.into_iter().collect::<BTreeSet<_>>();
        let mut values = BTreeMap::new();
        let mut layer = Some(&**image);
        while let Some(current) = layer {
            if wanted.is_empty() {
                break;
            }
            for (key, value) in &current.values {
                if wanted.remove(key) {
                    values.insert(*key, *value);
                }
            }
            layer = current.parent.as_deref();
        }
        // Every global of a checkpoint has a value in its chain.
        debug_assert!(wanted.is_empty());
        for (key, value) in &values {
            if read(*key) != *value {
                write(*key, value);
            }
        }
        self.head_values = values;
        self.head = Some(image.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 16;

    /// A memory with a shadow, which reports writes to clean bytes as
    /// compiled code would.
    #[derive(Default)]
    struct TestMemory {
        bytes: Vec<u8>,
        shadow: Vec<u8>,
        fill: u8,
    }

    impl TrackedMemory for TestMemory {
        fn bytes(&mut self) -> &[u8] {
            &self.bytes
        }
        fn bytes_mut(&mut self) -> &mut [u8] {
            &mut self.bytes
        }
        fn shadow_mut(&mut self) -> &mut [u8] {
            &mut self.shadow
        }
        fn resize(&mut self, len: usize) -> Result<()> {
            self.bytes.resize(len, 0);
            let old = self.shadow.len();
            self.shadow.resize(len, 0);
            if len > old {
                self.shadow[old..].fill(self.fill);
            }
            Ok(())
        }
        fn set_shadow_fill(&mut self, fill: u8) {
            self.fill = fill;
        }
    }

    impl TestMemory {
        fn new(len: usize) -> Self {
            let mut memory = TestMemory::default();
            memory.resize(len).unwrap();
            memory
        }

        /// A guest write: reports the write if any byte is clean, then writes.
        fn store(&mut self, history: &mut History, at: usize, data: &[u8]) {
            let range = at..at + data.len();
            if self.shadow[range.clone()]
                .iter()
                .any(|b| b & WATCH_CLEAN != 0)
            {
                history.write(self, range.clone()).unwrap();
            }
            self.bytes[range].copy_from_slice(data);
        }

        fn grow(&mut self, by: usize) {
            let len = self.bytes.len() + by;
            self.resize(len).unwrap();
        }
    }

    #[test]
    fn paged_images_share_unchanged_pages() {
        let mut data = vec![0_u8; 4 * PAGE + 10];
        let first = PagedImage::capture(&data, PAGE, None).unwrap();
        data[2 * PAGE + 3] = 7;
        let second = PagedImage::capture(&data, PAGE, Some(&first)).unwrap();
        assert_eq!(second.stored_bytes(Some(&first)), PAGE);
        for i in 0..first.pages.len() {
            assert_eq!(Arc::ptr_eq(&first.pages[i], &second.pages[i]), i != 2);
        }
        let mut current = data.clone();
        current[PAGE] = 1;
        second.restore_into(&mut current);
        assert_eq!(current, data);
        first.restore_into(&mut current);
        assert!(current.iter().all(|b| *b == 0));
    }

    #[test]
    fn values_keep_only_changes() {
        let mut current = BTreeMap::from([(1, [1; 16]), (2, [2; 16])]);
        let mut history = ValuesHistory::default();
        let first = history.checkpoint(current.clone()).unwrap();
        assert_eq!(first.len(), 2);
        current.insert(2, [3; 16]);
        let second = history.checkpoint(current.clone()).unwrap();
        assert_eq!(second.len(), 1);
        current.insert(3, [4; 16]);
        let third = history.checkpoint(current.clone()).unwrap();
        assert_eq!(third.len(), 1);

        let mut writes = Vec::new();
        history.restore(
            &first,
            [1, 2],
            |key| current[&key],
            |key, value| writes.push((key, *value)),
        );
        assert_eq!(writes, [(2, [2; 16])]);
        current.insert(2, [2; 16]);
        // Unchanged values after a restore are not stored again.
        let again = history.checkpoint(current.clone()).unwrap();
        assert_eq!(again.len(), 1); // Global 3, which `first` did not have.
        writes.clear();
        history.restore(
            &second,
            [1, 2],
            |key| current[&key],
            |key, value| writes.push((key, *value)),
        );
        assert_eq!(writes, [(2, [3; 16])]);
    }

    #[test]
    fn restores_linear_history() {
        let mut memory = TestMemory::new(4 * PAGE);
        memory.bytes[5] = 9; // Contents from before tracking began.
        let mut history = History::new(PAGE, &mut memory);
        let start = history.checkpoint(&mut memory).unwrap();
        let start_bytes = memory.bytes.clone();

        memory.store(&mut history, 0, &[1, 2, 3]);
        memory.store(&mut history, 2 * PAGE + 1, &[4]);
        let first = history.checkpoint(&mut memory).unwrap();
        let first_bytes = memory.bytes.clone();
        // Only the two written pages were copied for the checkpoint, plus
        // their bases.
        assert_eq!(first.pages.len(), 2);
        assert_eq!(history.pages_held(&[&start, &first]), 4);

        memory.store(&mut history, 1, &[7]);
        memory.store(&mut history, 3 * PAGE, &[8; PAGE]);
        let end_bytes = memory.bytes.clone();

        history.restore(&mut memory, &first).unwrap();
        assert_eq!(memory.bytes, first_bytes);
        history.restore(&mut memory, &start).unwrap();
        assert_eq!(memory.bytes, start_bytes);
        assert_eq!(memory.bytes[5], 9);
        assert!(memory.shadow.iter().all(|b| b & WATCH_CLEAN != 0));

        // Restored pages are clean again: writing them is reported.
        memory.store(&mut history, 2 * PAGE + 1, &[4]);
        memory.store(&mut history, 0, &[1, 2, 3]);
        memory.store(&mut history, 1, &[7]);
        memory.store(&mut history, 3 * PAGE, &[8; PAGE]);
        assert_eq!(memory.bytes, end_bytes);
    }

    #[test]
    fn restores_across_branches() {
        let mut memory = TestMemory::new(2 * PAGE);
        let mut history = History::new(PAGE, &mut memory);
        let root = history.checkpoint(&mut memory).unwrap();

        // Branch A writes page 0, then page 1.
        memory.store(&mut history, 0, &[1]);
        let a1 = history.checkpoint(&mut memory).unwrap();
        memory.store(&mut history, PAGE, &[2]);
        let a2 = history.checkpoint(&mut memory).unwrap();
        let a2_bytes = memory.bytes.clone();

        // Branch B, from the root, writes page 1 differently.
        history.restore(&mut memory, &root).unwrap();
        memory.store(&mut history, PAGE + 1, &[3]);
        let b1 = history.checkpoint(&mut memory).unwrap();
        let b1_bytes = memory.bytes.clone();

        // Jump between the branches in every direction.
        for (image, bytes) in [
            (&a2, &a2_bytes),
            (&b1, &b1_bytes),
            (&a1, &{
                let mut b = vec![0; 2 * PAGE];
                b[0] = 1;
                b
            }),
            (&a2, &a2_bytes),
            (&root, &vec![0; 2 * PAGE]),
            (&b1, &b1_bytes),
        ] {
            history.restore(&mut memory, image).unwrap();
            assert_eq!(&memory.bytes, bytes);
        }
    }

    #[test]
    fn restores_across_growth() {
        let mut memory = TestMemory::new(PAGE);
        let mut history = History::new(PAGE, &mut memory);
        memory.store(&mut history, 0, &[1]);
        let small = history.checkpoint(&mut memory).unwrap();

        memory.grow(2 * PAGE);
        memory.store(&mut history, 2 * PAGE, &[2]);
        let big = history.checkpoint(&mut memory).unwrap();
        let big_bytes = memory.bytes.clone();

        history.restore(&mut memory, &small).unwrap();
        assert_eq!(memory.bytes.len(), PAGE);
        assert_eq!(memory.bytes[0], 1);

        // Regrowing on another branch: the grown pages start as zeroes and
        // are tracked, written or not.
        memory.grow(2 * PAGE);
        assert!(memory.bytes[PAGE..].iter().all(|b| *b == 0));
        memory.store(&mut history, PAGE, &[5]);
        let other = history.checkpoint(&mut memory).unwrap();
        let other_bytes = memory.bytes.clone();

        history.restore(&mut memory, &big).unwrap();
        assert_eq!(memory.bytes, big_bytes);
        history.restore(&mut memory, &other).unwrap();
        assert_eq!(memory.bytes, other_bytes);
    }

    #[test]
    fn checkpoints_without_writes_share_everything() {
        let mut memory = TestMemory::new(64 * PAGE);
        let mut history = History::new(PAGE, &mut memory);
        let images = (0..10)
            .map(|_| history.checkpoint(&mut memory).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(history.pages_held(&images.iter().collect::<Vec<_>>()), 0);
        memory.store(&mut history, 3, &[1]);
        // A later write to the same page is not reported again.
        assert!(memory.shadow[..PAGE].iter().all(|b| b & WATCH_CLEAN == 0));
        let last = history.checkpoint(&mut memory).unwrap();
        assert_eq!(last.pages.len(), 1);
    }
}
