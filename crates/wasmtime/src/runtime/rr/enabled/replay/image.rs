//! Images of guest state at checkpoints.
//!
//! An [`Image`] of a memory is a persistent map of the pages written
//! since its [`History`] began, which shares unchanged pages, and unchanged
//! parts of the map, with other images. A page that is not in the map still has
//! its contents from when the history began, which the history records when the
//! page is first written.
//!
//! Writes are detected without copying or comparing contents: a memory's shadow
//! marks every byte of a page as clean (`WATCH_CLEAN`) until the first write to
//! the page after a checkpoint, which compiled code reports through the memory's
//! watchpoint builtin. The size of a page is independent of the host's.
//!
//! A [`PagedImage`] is of state whose writes are not tracked, as there is little
//! of it or as the garbage collector also writes to it. It shares unchanged
//! pages with the previous image, which it is compared with to find them.

use crate::prelude::*;
use crate::runtime::vm::WATCH_CLEAN;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use core::ops::Range;

type Page = Arc<[u8]>;

/// The log2 of the number of children of a [`Node`].
const FANOUT_LOG2: u32 = 4;
const FANOUT: usize = 1 << FANOUT_LOG2;

#[derive(Clone)]
enum Node {
    Branch([Option<Arc<Node>>; FANOUT]),
    Page(Page),
}

/// A persistent map from page indices to their contents: cloning it is cheap,
/// and clones share everything that has not changed since.
#[derive(Clone, Default)]
struct PageMap {
    root: Option<Arc<Node>>,
    /// The number of levels of branches.
    height: u32,
}

impl PageMap {
    fn child_index(index: usize, level: u32) -> usize {
        (index >> (FANOUT_LOG2 * level)) % FANOUT
    }

    /// Whether `index` is within the range of a map of the given height.
    fn fits(index: usize, height: u32) -> bool {
        index
            .checked_shr(FANOUT_LOG2 * height)
            .is_none_or(|rest| rest == 0)
    }

    /// Makes this map at least `height` levels high.
    fn raise(&mut self, height: u32) -> Result<()> {
        while self.height < height {
            if let Some(root) = self.root.take() {
                let mut children = [const { None }; FANOUT];
                children[0] = Some(root);
                self.root = Some(try_new::<Arc<_>>(Node::Branch(children))?);
            }
            self.height += 1;
        }
        Ok(())
    }

    fn insert(&mut self, index: usize, page: Page) -> Result<()> {
        while !Self::fits(index, self.height) {
            self.raise(self.height + 1)?;
        }
        let mut slot = &mut self.root;
        for level in (0..self.height).rev() {
            let node = match slot {
                Some(node) => node,
                None => slot.insert(try_new::<Arc<_>>(Node::Branch([const { None }; FANOUT]))?),
            };
            if Arc::get_mut(node).is_none() {
                *node = try_new::<Arc<_>>(Node::clone(node))?;
            }
            let Some(Node::Branch(children)) = Arc::get_mut(node) else {
                unreachable!()
            };
            slot = &mut children[Self::child_index(index, level)];
        }
        *slot = Some(try_new::<Arc<_>>(Node::Page(page))?);
        Ok(())
    }

    fn get(&self, index: usize) -> Option<&Page> {
        if !Self::fits(index, self.height) {
            return None;
        }
        let mut node = self.root.as_ref()?;
        for level in (0..self.height).rev() {
            let Node::Branch(children) = &**node else {
                unreachable!()
            };
            node = children[Self::child_index(index, level)].as_ref()?;
        }
        let Node::Page(page) = &**node else {
            unreachable!()
        };
        Some(page)
    }

    /// The indices of the pages that differ between two maps.
    fn differences(mut a: PageMap, mut b: PageMap) -> Result<BTreeSet<usize>> {
        a.raise(b.height)?;
        b.raise(a.height)?;
        let mut differences = BTreeSet::new();
        Self::diff(
            a.root.as_ref(),
            b.root.as_ref(),
            a.height,
            0,
            &mut differences,
        );
        Ok(differences)
    }

    fn diff(
        a: Option<&Arc<Node>>,
        b: Option<&Arc<Node>>,
        level: u32,
        prefix: usize,
        differences: &mut BTreeSet<usize>,
    ) {
        let same = match (a, b) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        let Some(level) = level.checked_sub(1) else {
            differences.insert(prefix);
            return;
        };
        let child = |node: Option<&'_ Arc<Node>>, i: usize| match node.map(|n| &**n) {
            Some(Node::Branch(children)) => children[i].clone(),
            Some(Node::Page(_)) => unreachable!(),
            None => None,
        };
        for i in 0..FANOUT {
            let (a, b) = (child(a, i), child(b, i));
            let prefix = prefix << FANOUT_LOG2 | i;
            Self::diff(a.as_ref(), b.as_ref(), level, prefix, differences);
        }
    }
}

/// The contents of a memory at a checkpoint.
pub(in super::super) struct Image {
    /// The length at the checkpoint, in bytes.
    len: usize,
    pages: PageMap,
    /// The number of bytes that were copied for this image.
    copied: usize,
}

impl Image {
    pub fn copied_bytes(&self) -> usize {
        self.copied
    }
}

/// A memory whose contents a [`History`] tracks.
pub(in super::super) trait Tracked {
    fn bytes_mut(&mut self) -> &mut [u8];

    /// The shadow, one byte per byte.
    fn shadow_mut(&mut self) -> &mut [u8];

    /// Sets the shadow value of bytes added when the memory grows.
    fn set_shadow_fill(&mut self, fill: u8);

    /// Resizes the memory to `len` bytes. Bytes that are added read as zero,
    /// and have the fill value in the shadow.
    fn resize(&mut self, len: usize) -> Result<()>;
}

/// Tracks the contents of a memory since its first checkpoint.
pub(in super::super) struct History {
    page_size: usize,
    /// The contents, when this history began, of each page that has been
    /// written since.
    base: BTreeMap<usize, Page>,
    /// The pages of the last checkpoint or restored image.
    pages: PageMap,
    /// The pages written since.
    dirty: BTreeSet<usize>,
}

impl History {
    /// Starts tracking `memory`.
    pub fn new(page_size: usize, memory: &mut dyn Tracked) -> Self {
        assert!(page_size > 0);
        for byte in memory.shadow_mut() {
            *byte |= WATCH_CLEAN;
        }
        memory.set_shadow_fill(WATCH_CLEAN);
        History {
            page_size,
            base: BTreeMap::new(),
            pages: PageMap::default(),
            dirty: BTreeSet::new(),
        }
    }

    fn page_range(&self, index: usize, len: usize) -> Range<usize> {
        let start = index.saturating_mul(self.page_size).min(len);
        start..start.saturating_add(self.page_size).min(len)
    }

    /// A copy of page `index` of `bytes`, padded with zeros.
    fn copy_page(&self, bytes: &[u8], index: usize) -> Result<Page> {
        let mut page = Vec::new();
        page.try_reserve_exact(self.page_size)?;
        page.extend_from_slice(&bytes[self.page_range(index, bytes.len())]);
        page.resize(self.page_size, 0);
        Ok(page.into())
    }

    fn set_clean(&self, memory: &mut dyn Tracked, index: usize, clean: bool) {
        let shadow = memory.shadow_mut();
        let range = self.page_range(index, shadow.len());
        for byte in &mut shadow[range] {
            if clean {
                *byte |= WATCH_CLEAN;
            } else {
                *byte &= !WATCH_CLEAN;
            }
        }
    }

    /// Notes that `range` is about to be written.
    pub fn write(&mut self, memory: &mut dyn Tracked, range: Range<usize>) -> Result<()> {
        for index in range.start / self.page_size..range.end.div_ceil(self.page_size) {
            if self.dirty.contains(&index) {
                continue;
            }
            if !self.base.contains_key(&index) {
                let page = self.copy_page(memory.bytes_mut(), index)?;
                self.base.insert(index, page);
            }
            self.dirty.insert(index);
            self.set_clean(memory, index, false);
        }
        Ok(())
    }

    /// Captures the memory's current contents.
    pub fn checkpoint(&mut self, memory: &mut dyn Tracked) -> Result<Image> {
        let len = memory.bytes_mut().len();
        let mut copied = 0;
        for &index in &self.dirty {
            if !self.page_range(index, len).is_empty() {
                let page = self.copy_page(memory.bytes_mut(), index)?;
                copied += page.len();
                self.pages.insert(index, page)?;
            }
            self.set_clean(memory, index, true);
        }
        self.dirty.clear();
        Ok(Image {
            len,
            pages: self.pages.clone(),
            copied,
        })
    }

    /// Restores the memory's contents to `image`, from this history, by writing
    /// the pages that may differ.
    pub fn restore(&mut self, memory: &mut dyn Tracked, image: &Image) -> Result<()> {
        let mut changed = PageMap::differences(self.pages.clone(), image.pages.clone())?;
        changed.append(&mut self.dirty);
        memory.resize(image.len)?;
        for index in changed {
            let range = self.page_range(index, image.len);
            if range.is_empty() {
                continue;
            }
            let page = image
                .pages
                .get(index)
                .or_else(|| self.base.get(&index))
                .expect("a page that was written has a base");
            let len = range.len();
            memory.bytes_mut()[range].copy_from_slice(&page[..len]);
            self.set_clean(memory, index, true);
        }
        self.pages = image.pages.clone();
        Ok(())
    }
}

/// The contents of untracked state, as pages shared with the previous image.
#[derive(Default)]
pub(crate) struct PagedImage {
    len: usize,
    pages: Vec<Page>,
    /// The number of bytes that were copied for this image.
    copied: usize,
}

impl PagedImage {
    /// Captures `bytes`.
    pub fn capture(bytes: &[u8], page_size: usize, prev: Option<&PagedImage>) -> Result<Self> {
        let mut pages = Vec::new();
        pages.try_reserve_exact(bytes.len().div_ceil(page_size))?;
        let mut copied = 0;
        for (i, chunk) in bytes.chunks(page_size).enumerate() {
            match prev.and_then(|p| p.pages.get(i)) {
                Some(page) if **page == *chunk => pages.push(page.clone()),
                _ => {
                    let mut page = Vec::new();
                    page.try_reserve_exact(chunk.len())?;
                    page.extend_from_slice(chunk);
                    copied += chunk.len();
                    pages.push(page.into());
                }
            }
        }
        Ok(PagedImage {
            len: bytes.len(),
            pages,
            copied,
        })
    }

    /// The length of the captured bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn copied_bytes(&self) -> usize {
        self.copied
    }

    /// Writes the pages of `bytes`, which has this image's length, that differ.
    pub fn restore_into(&self, bytes: &mut [u8]) {
        debug_assert_eq!(bytes.len(), self.len);
        let mut start = 0;
        for page in &self.pages {
            let chunk = &mut bytes[start..start + page.len()];
            if *chunk != **page {
                chunk.copy_from_slice(page);
            }
            start += page.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 16;

    /// A memory with a shadow, which reports writes to clean bytes as compiled
    /// code would.
    #[derive(Default)]
    struct TestMemory {
        bytes: Vec<u8>,
        shadow: Vec<u8>,
        fill: u8,
    }

    impl Tracked for TestMemory {
        fn bytes_mut(&mut self) -> &mut [u8] {
            &mut self.bytes
        }

        fn shadow_mut(&mut self) -> &mut [u8] {
            &mut self.shadow
        }

        fn set_shadow_fill(&mut self, fill: u8) {
            self.fill = fill;
        }

        fn resize(&mut self, len: usize) -> Result<()> {
            self.bytes.resize(len, 0);
            self.shadow.resize(len, self.fill);
            Ok(())
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
            let clean = |b: &u8| b & WATCH_CLEAN != 0;
            if self.shadow[range.clone()].iter().any(clean) {
                history.write(self, range.clone()).unwrap();
            }
            self.bytes[range].copy_from_slice(data);
        }

        fn grow(&mut self, by: usize) {
            self.resize(self.bytes.len() + by).unwrap();
        }
    }

    #[test]
    fn page_maps() {
        let page = |b: u8| Page::from(vec![b]);
        let mut a = PageMap::default();
        a.insert(0, page(1)).unwrap();
        assert_eq!(a.height, 0);
        a.insert(17, page(2)).unwrap();
        let mut b = a.clone();
        let far = usize::MAX / PAGE;
        b.insert(far, page(3)).unwrap();
        b.insert(17, page(4)).unwrap();
        for (map, index, value) in [
            (&a, 0, Some(1)),
            (&a, 1, None),
            (&a, 17, Some(2)),
            (&a, far, None),
            (&b, 0, Some(1)),
            (&b, 17, Some(4)),
            (&b, far, Some(3)),
        ] {
            assert_eq!(map.get(index).map(|p| p[0]), value);
        }
        let differences = |a: &PageMap, b: &PageMap| {
            PageMap::differences(a.clone(), b.clone())
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>()
        };
        assert_eq!(differences(&a, &b), [17, far]);
        assert_eq!(differences(&b, &a), [17, far]);
        assert_eq!(differences(&b, &b), [] as [usize; 0]);
        assert_eq!(differences(&PageMap::default(), &b), [0, 17, far]);
    }

    #[test]
    fn restores_linear_history() {
        let mut memory = TestMemory::new(4 * PAGE);
        memory.bytes[5] = 9; // Contents from before tracking began.
        let mut history = History::new(PAGE, &mut memory);
        let start = history.checkpoint(&mut memory).unwrap();
        let start_bytes = memory.bytes.clone();
        assert_eq!(start.copied_bytes(), 0);

        memory.store(&mut history, 0, &[1, 2, 3]);
        memory.store(&mut history, 2 * PAGE + 1, &[4]);
        let first = history.checkpoint(&mut memory).unwrap();
        let first_bytes = memory.bytes.clone();
        // Only the two written pages were copied for the checkpoint.
        assert_eq!(first.copied_bytes(), 2 * PAGE);
        assert_eq!(history.base.len(), 2);

        memory.store(&mut history, 1, &[7]);
        memory.store(&mut history, 3 * PAGE, &[8; PAGE]);
        let end_bytes = memory.bytes.clone();

        history.restore(&mut memory, &first).unwrap();
        assert_eq!(memory.bytes, first_bytes);
        history.restore(&mut memory, &start).unwrap();
        assert_eq!(memory.bytes, start_bytes);
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

        // One branch writes page 0, then page 1.
        memory.store(&mut history, 0, &[1]);
        let a1 = history.checkpoint(&mut memory).unwrap();
        let a1_bytes = memory.bytes.clone();
        memory.store(&mut history, PAGE, &[2]);
        let a2 = history.checkpoint(&mut memory).unwrap();
        let a2_bytes = memory.bytes.clone();

        // Another, from the root, writes page 1 differently.
        history.restore(&mut memory, &root).unwrap();
        memory.store(&mut history, PAGE + 1, &[3]);
        let b1 = history.checkpoint(&mut memory).unwrap();
        let b1_bytes = memory.bytes.clone();

        // Jump between the branches in every direction.
        for (image, bytes) in [
            (&a2, &a2_bytes),
            (&b1, &b1_bytes),
            (&a1, &a1_bytes),
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

        // Growing again on another branch: pages start as zeros and are
        // tracked.
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
    fn dropped_images_free_their_pages() {
        let mut memory = TestMemory::new(PAGE);
        let mut history = History::new(PAGE, &mut memory);
        memory.store(&mut history, 0, &[1]);
        let first = history.checkpoint(&mut memory).unwrap();
        let page = Arc::downgrade(first.pages.get(0).unwrap());
        memory.store(&mut history, 0, &[2]);
        let _second = history.checkpoint(&mut memory).unwrap();
        assert!(page.upgrade().is_some());
        drop(first);
        assert!(page.upgrade().is_none());
    }

    #[test]
    fn paged_images_share_unchanged_pages() {
        let mut data = vec![0_u8; 4 * PAGE + 10];
        let first = PagedImage::capture(&data, PAGE, None).unwrap();
        data[2 * PAGE + 3] = 7;
        let second = PagedImage::capture(&data, PAGE, Some(&first)).unwrap();
        assert_eq!(second.copied_bytes(), PAGE);
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
}
