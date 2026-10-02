//! The shadow bytes of linear memories, for watchpoints.
//!
//! When `Tunables::memory_watchpoints` is enabled, compiled code checks a
//! memory's shadow before every store: a nonzero shadow byte calls a
//! watchpoint builtin before the corresponding memory byte is written. The
//! bits of a shadow byte say why it is watched.

use crate::prelude::*;
use crate::runtime::vm::VmPtr;
use crate::runtime::vm::vmcontext::VMMemoryShadow;
use core::ptr::NonNull;

/// A shadow bit set for each byte a debugger watches.
#[cfg(feature = "debug")]
pub const WATCH_DEBUG: u8 = 1 << 0;

/// A shadow bit set for each byte of a page that record/replay has not seen
/// written since its last checkpoint.
#[cfg(feature = "rr")]
pub const WATCH_CLEAN: u8 = 1 << 1;

/// The shadow of a linear memory: one byte per byte of the memory.
pub struct MemoryShadow {
    // Compiled code reaches the bytes through this cell, which has a stable
    // address even when the bytes are reallocated.
    cell: Box<VMMemoryShadow>,
    bytes: Box<[u8]>,
    len: usize,
    // The value of the bytes added when the memory grows.
    fill: u8,
}

// SAFETY: the cell only points to `bytes`, which the shadow owns.
unsafe impl Send for MemoryShadow {}
unsafe impl Sync for MemoryShadow {}

impl MemoryShadow {
    /// Creates an unwatched shadow for a memory of `len` bytes.
    pub fn new(len: usize) -> Result<Self> {
        let bytes = zeroed(len)?;
        let cell = try_new::<Box<_>>(VMMemoryShadow {
            base: VmPtr::from(base(&bytes)),
        })?;
        Ok(MemoryShadow {
            cell,
            bytes,
            len,
            fill: 0,
        })
    }

    /// The cell through which compiled code finds the shadow bytes.
    pub fn cell(&self) -> NonNull<VMMemoryShadow> {
        NonNull::from(&*self.cell)
    }

    /// The shadow bytes, one per byte of the memory.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// The shadow bytes, one per byte of the memory.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[..self.len]
    }

    /// Sets the value of the bytes added when the memory grows.
    pub fn set_fill(&mut self, fill: u8) {
        self.fill = fill;
    }

    /// Follows the memory's size, which is now `len` bytes.
    pub fn resize(&mut self, len: usize) -> Result<()> {
        if len > self.bytes.len() {
            let mut bytes = zeroed(len.max(self.bytes.len().saturating_mul(2)))?;
            bytes[..self.len].copy_from_slice(&self.bytes[..self.len]);
            self.cell.base = VmPtr::from(base(&bytes));
            self.bytes = bytes;
        }
        if len > self.len {
            self.bytes[self.len..len].fill(self.fill);
        }
        self.len = len;
        Ok(())
    }
}

fn base(bytes: &[u8]) -> NonNull<u8> {
    NonNull::new(bytes.as_ptr().cast_mut()).unwrap_or(NonNull::dangling())
}

/// Allocates zeroed bytes, which the allocator can provide lazily.
fn zeroed(len: usize) -> Result<Box<[u8]>, OutOfMemory> {
    if len == 0 {
        return Ok(Box::default());
    }
    let layout = core::alloc::Layout::array::<u8>(len).map_err(|_| OutOfMemory::new(len))?;
    // SAFETY: the layout has a nonzero size, and a successful allocation of
    // `len` zeroed bytes is a valid `Box<[u8]>`.
    unsafe {
        let ptr = alloc::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            return Err(OutOfMemory::new(len));
        }
        Ok(Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, len)))
    }
}
