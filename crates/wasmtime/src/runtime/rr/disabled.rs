//! The crate-private record/replay API without the `rr` feature: no store ever
//! records.

use crate::prelude::*;
use crate::rr::Growable;
use crate::runtime::vm::VMFuncRef;
use crate::store::{InstanceId, StoreOpaque};
use crate::{Global, Instance, Memory, Module, Ref, Table, ValRaw, bail_bug};
use core::ops::Range;
use core::ptr::NonNull;
use wasmtime_environ::RrCallKind;

#[derive(Default)]
pub(crate) struct State {}

impl State {
    #[inline]
    pub(crate) fn poison(&mut self, _operation: &'static str) {}

    #[inline]
    pub(crate) fn reject(&mut self, _operation: &'static str) -> Result<()> {
        Ok(())
    }
}

#[inline]
#[cfg(feature = "component-model")]
pub(crate) fn wrap_destructor<T: 'static>(
    _store: &mut crate::StoreContextMut<'_, T>,
    dtor: NonNull<VMFuncRef>,
) -> Result<NonNull<VMFuncRef>> {
    Ok(dtor)
}

impl StoreOpaque {
    #[inline]
    pub(crate) fn rr_flush(&mut self) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_track_memory(&mut self, _memory: Memory, _range: Range<usize>) {}

    #[inline]
    #[cfg(feature = "component-model")]
    pub(crate) fn rr_track_memory_definition(
        &mut self,
        _definition: NonNull<crate::runtime::vm::VMMemoryDefinition>,
        _written: Range<usize>,
    ) {
    }

    #[inline]
    #[cfg(feature = "component-model")]
    pub(crate) fn rr_track_raw_memory(&mut self, _ptr: *mut u8, _len: usize) {}

    #[inline]
    #[cfg(feature = "gc")]
    pub(crate) fn rr_trace_parked_roots(&self, _roots: &mut crate::vm::GcRootsList) {}

    #[inline]
    #[cfg(feature = "gc")]
    pub(crate) fn rr_trace_handles(&mut self, _roots: &mut crate::vm::GcRootsList) {}

    #[inline]
    #[cfg(feature = "gc")]
    pub(crate) fn rr_created_externref(&mut self, _gc_ref: &crate::vm::VMGcRef) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_memory_grown(&mut self, _memory: Memory) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_created_global(&mut self, _global: Global) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_created_memory(&mut self, _memory: Memory) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_created_table(&mut self, _table: Table, _init: Ref) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_created_instance(
        &mut self,
        _instance: Instance,
        _module: &Module,
    ) -> Result<()> {
        Ok(())
    }

    #[inline]
    #[cfg(feature = "debug")]
    pub(crate) fn rr_breakpoints_edited(&mut self) {}

    #[inline]
    #[cfg(feature = "debug")]
    pub(crate) fn rr_stop_at_breakpoint(&mut self) -> bool {
        false
    }

    #[inline]
    #[cfg(feature = "debug")]
    pub(crate) fn rr_stop_at_watchpoint(&mut self, _hit: crate::WatchpointHit) -> bool {
        false
    }

    #[inline]
    #[cfg(target_has_atomic = "64")]
    pub(crate) fn rr_replay_epoch(&mut self) -> Option<u64> {
        None
    }

    #[inline]
    pub(crate) fn rr_replay_fuel(&mut self) -> Result<bool> {
        Ok(false)
    }

    #[inline]
    pub(crate) fn rr_memory_written(
        &mut self,
        _memory: Memory,
        _range: Range<usize>,
    ) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(crate) fn rr_replay_growth_fails(
        &mut self,
        _object: Growable,
        _delta: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    #[inline]
    pub(crate) fn rr_growth_failed(&mut self, _object: Growable, _delta: u64) -> Result<()> {
        Ok(())
    }

    /// See the enabled version's safety contract.
    pub(crate) unsafe fn rr_enter(
        &mut self,
        _caller: InstanceId,
        _func: NonNull<VMFuncRef>,
        _raw: *const ValRaw,
        _kind: RrCallKind,
    ) -> Result<u32> {
        bail_bug!("recording without record/replay support")
    }

    /// See the enabled version's safety contract.
    pub(crate) unsafe fn rr_leave(
        &mut self,
        _call: u32,
        _raw: *const ValRaw,
        _kind: RrCallKind,
        _succeeded: bool,
    ) -> Result<()> {
        bail_bug!("recording without record/replay support")
    }
}
