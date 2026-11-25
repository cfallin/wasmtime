//! Guest `memory.grow` and `table.grow` instructions that failed.
//!
//! Whether growth fails depends on the host, for example on its resource
//! limiter, so replay must fail exactly the growth that failed when recorded.
//! Successful growth is not recorded.

use super::*;
use crate::rr::Growable;
use crate::store::StoreInstanceId;
use objects::RecordedTable;
use wasmtime_environ::collections::TryClone;

/// A failed guest growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GrowthFailure {
    object: RecordedGrowable,
    /// The object's size, in bytes or elements, which tells this apart from
    /// growth by the same amount that succeeded.
    size: u64,
    /// The amount that the guest requested, in pages or elements.
    delta: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecordedGrowable {
    Memory(RecordedMemory),
    Table(RecordedTable),
}

impl TryClone for GrowthFailure {
    fn try_clone(&self) -> Result<Self, OutOfMemory> {
        Ok(*self)
    }
}

impl GrowthFailure {
    fn new(store: &StoreOpaque, object: Growable, delta: u64) -> Result<Self> {
        let objects = store.rr_objects();
        let (id, size) = match object {
            Growable::Memory(instance, index) => {
                // SAFETY: guest code grows memories through `memory.grow`,
                // which shared memories do not use in a recording as they are
                // rejected.
                let memory =
                    unsafe { Memory::from_raw(StoreInstanceId::new(store.id(), instance), index) };
                (
                    objects
                        .find_memory(store, memory)
                        .map(RecordedGrowable::Memory),
                    u64::try_from(memory.internal_data_size(store))?,
                )
            }
            Growable::Table(instance, index) => {
                let table =
                    crate::Table::from_raw(StoreInstanceId::new(store.id(), instance), index);
                (
                    objects
                        .find_table(store, table)
                        .map(RecordedGrowable::Table),
                    table.size_(store),
                )
            }
        };
        Ok(GrowthFailure {
            object: id.ok_or_else(|| format_err!("guest grew an unregistered object"))?,
            size,
            delta,
        })
    }

    fn encode(&self) -> [u8; 21] {
        let (kind, id) = match self.object {
            RecordedGrowable::Memory(id) => (0, id.as_u32()),
            RecordedGrowable::Table(id) => (1, id.as_u32()),
        };
        let mut body = [0; 21];
        body[0] = kind;
        body[1..5].copy_from_slice(&id.to_le_bytes());
        body[5..13].copy_from_slice(&self.size.to_le_bytes());
        body[13..].copy_from_slice(&self.delta.to_le_bytes());
        body
    }

    pub fn decode(body: &mut Reader<'_>) -> Result<Self> {
        let kind = body.u8()?;
        let id = body.u32()?;
        let failure = GrowthFailure {
            object: match kind {
                0 => RecordedGrowable::Memory(RecordedMemory::from_u32(id)),
                1 => RecordedGrowable::Table(RecordedTable::from_u32(id)),
                _ => bail!("invalid kind of growth in trace"),
            },
            size: body.u64()?,
            delta: body.u64()?,
        };
        body.end()?;
        Ok(failure)
    }
}

impl StoreOpaque {
    /// Whether replay must fail this guest growth, because it failed when it
    /// was recorded. Called before attempting the growth.
    pub(crate) fn rr_replay_growth_fails(&mut self, object: Growable, delta: u64) -> Result<bool> {
        if !self.rr.replaying() {
            return Ok(false);
        }
        let failure = GrowthFailure::new(self, object, delta)?;
        let failures = &mut self.rr_session().mode.replaying().growth_failures;
        let fails = failures.last() == Some(&failure);
        if fails {
            failures.pop();
        }
        Ok(fails)
    }

    /// Records a failed guest growth, or reports a divergence if it succeeded
    /// when recorded.
    pub(crate) fn rr_growth_failed(&mut self, object: Growable, delta: u64) -> Result<()> {
        if self.rr.replaying() {
            let error = || format_err!("guest growth failed that succeeded when recorded");
            self.rr_session().fail(error());
            return Err(error());
        }
        if !self.rr.recording() {
            return Ok(());
        }
        let result = GrowthFailure::new(self, object, delta)
            .and_then(|f| self.rr_session().append(codec::GROWTH_FAILED, &f.encode()));
        self.rr_poison_on_err(result)
    }
}
