//! Replaying the host's writes to globals and tables.

use super::super::*;
use super::init::value;
use crate::{Ref, StoreContextMut, Table, ValType};
use objects::{RecordedGlobal, RecordedTable};

fn table(store: &StoreOpaque, body: &mut Reader<'_>) -> Result<Table> {
    let id = RecordedTable::from_u32(body.u32()?);
    let table = store.rr_objects().tables.get(id).copied();
    table.ok_or_else(|| format_err!("invalid table ID in trace"))
}

/// Reads an element of `table`, which is rooted in the current scope.
fn element(store: &mut StoreOpaque, body: &mut Reader<'_>, table: Table) -> Result<Ref> {
    let ty = ValType::Ref(table.ty_(store).element().clone());
    let value = value(store, body, &ty)?;
    Ok(value.ref_().unwrap())
}

/// Replays a write to a global or a table. The values that it stores are rooted
/// in the current scope.
pub(super) fn replay_mutation(
    store: &mut StoreInner<()>,
    tag: u8,
    body: &mut Reader<'_>,
) -> Result<()> {
    match tag {
        codec::GLOBAL_WRITE => global_write(store, body),
        codec::TABLE_FILL => table_fill(store, body),
        codec::TABLE_GROW => table_grow(store, body),
        codec::TABLE_COPY => table_copy(store, body),
        _ => unreachable!(),
    }
}

/// Replays the host's write to a global.
fn global_write(store: &mut StoreOpaque, body: &mut Reader<'_>) -> Result<()> {
    let id = RecordedGlobal::from_u32(body.u32()?);
    let global = store.rr_objects().globals.get(id).copied();
    let global = global.ok_or_else(|| format_err!("invalid global ID in trace"))?;
    let ty = global._ty(store);
    ensure!(
        ty.mutability().is_var(),
        "write to an immutable global in trace"
    );
    let value = value(store, body, ty.content())?;
    body.end()?;
    global._set(store, value)
}

/// Replays the host's write of one value to elements of a table.
fn table_fill(store: &mut StoreOpaque, body: &mut Reader<'_>) -> Result<()> {
    let table = table(store, body)?;
    let start = body.u64()?;
    let len = body.u64()?;
    let value = element(store, body, table)?;
    body.end()?;
    table._fill(store, start, value, len)
}

/// Replays a table's growth by the host.
fn table_grow(store: &mut StoreInner<()>, body: &mut Reader<'_>) -> Result<()> {
    let table = table(store, body)?;
    let new_size = body.u64()?;
    let init = element(store, body, table)?;
    body.end()?;
    let delta = new_size.checked_sub(table.size_(store));
    let delta = delta.ok_or_else(|| format_err!("invalid table size in trace"))?;
    table.grow(StoreContextMut(store), delta, init)?;
    Ok(())
}

/// Replays the host's copy of elements between tables.
fn table_copy(store: &mut StoreOpaque, body: &mut Reader<'_>) -> Result<()> {
    let dst = table(store, body)?;
    let dst_index = body.u64()?;
    let src = table(store, body)?;
    let src_index = body.u64()?;
    let len = body.u64()?;
    body.end()?;
    Table::copy_(store, (dst, dst_index), (src, src_index), len)
}
