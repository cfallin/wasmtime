//! Recording the host's writes to globals and tables, and parts in common with
//! its writes to objects.
//!
//! Unlike its writes to memories, these are recorded as they happen. Later
//! records, such as of references that the host reads, may depend on them.

use super::*;
use crate::{Global, Ref, Table, Val};
use objects::RecordedTable;

impl StoreOpaque {
    /// Encodes values that the host is storing, which must be rooted.
    pub(super) fn rr_encode_vals(&mut self, body: &mut TryVec<u8>, vals: &[Val]) -> Result<()> {
        for val in vals {
            if let Val::FuncRef(Some(func)) = val {
                let raw = func.vm_func_ref(self);
                // SAFETY: the store owns the function.
                unsafe { self.rr_register_func_reference(raw)? };
            }
            let (kind, raw) = self.rr_val_raw(val)?;
            // SAFETY: the value is valid and of this kind.
            unsafe { codec::values(body, &[kind], &raw, self.rr_session().refs())? };
        }
        Ok(())
    }

    /// Records a mutation, the encoding of which `f` appends, that just took
    /// place.
    pub(super) fn rr_mutated(
        &mut self,
        tag: u8,
        f: impl FnOnce(&mut Self, &mut TryVec<u8>) -> Result<()>,
    ) -> Result<()> {
        if !self.rr.recording() {
            return Ok(());
        }
        let result = (|| {
            self.rr_check()?;
            let mut body = TryVec::new();
            f(self, &mut body)?;
            self.rr_session().append(tag, &body)
        })();
        self.rr_poison_on_err(result)
    }

    fn rr_table_id(&self, table: Table) -> Result<RecordedTable> {
        let id = self.rr_objects().find_table(self, table);
        id.ok_or_else(|| format_err!("host modified an unregistered table"))
    }

    /// Records that the host set `global` to `value`.
    pub(crate) fn rr_global_written(&mut self, global: Global, value: &Val) -> Result<()> {
        self.rr_mutated(codec::GLOBAL_WRITE, |store, body| {
            let id = store.rr_objects().find_global(store, global);
            let id = id.ok_or_else(|| format_err!("host modified an unregistered global"))?;
            body.extend_from_slice(&id.as_u32().to_le_bytes())?;
            store.rr_encode_vals(body, core::slice::from_ref(value))
        })
    }

    /// Records that the host set the `len` elements of `table` from `start` to
    /// `value`.
    pub(crate) fn rr_table_filled(
        &mut self,
        table: Table,
        start: u64,
        len: u64,
        value: Ref,
    ) -> Result<()> {
        self.rr_mutated(codec::TABLE_FILL, |store, body| {
            let id = store.rr_table_id(table)?;
            body.extend_from_slice(&id.as_u32().to_le_bytes())?;
            body.extend_from_slice(&start.to_le_bytes())?;
            body.extend_from_slice(&len.to_le_bytes())?;
            store.rr_encode_vals(body, &[value.into()])
        })
    }

    /// Records that the host grew `table`, whose new elements are `init`.
    pub(crate) fn rr_table_grown(&mut self, table: Table, init: Ref) -> Result<()> {
        self.rr_mutated(codec::TABLE_GROW, |store, body| {
            let id = store.rr_table_id(table)?;
            body.extend_from_slice(&id.as_u32().to_le_bytes())?;
            body.extend_from_slice(&table.size_(store).to_le_bytes())?;
            store.rr_encode_vals(body, &[init.into()])
        })
    }

    /// Records that the host copied `len` elements between tables.
    pub(crate) fn rr_table_copied(
        &mut self,
        dst: (Table, u64),
        src: (Table, u64),
        len: u64,
    ) -> Result<()> {
        self.rr_mutated(codec::TABLE_COPY, |store, body| {
            for (table, index) in [dst, src] {
                let id = store.rr_table_id(table)?;
                body.extend_from_slice(&id.as_u32().to_le_bytes())?;
                body.extend_from_slice(&index.to_le_bytes())?;
            }
            Ok(body.extend_from_slice(&len.to_le_bytes())?)
        })
    }
}
