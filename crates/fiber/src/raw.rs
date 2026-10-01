//! Raw fibers: fibers whose entry point and suspension points are machine code
//! supplied by the caller rather than a Rust closure.
//!
//! A raw fiber starts by calling an `extern "C"` entry point with an opaque
//! argument. The code running on the fiber suspends itself by calling the
//! audited assembly routine from [`RawFiber::switch_routine`] with
//! [`RawFiber::switch_arg`]. The fiber library contributes no Rust frames to
//! the fiber's stack: only the platform's fiber-start assembly frame sits
//! beneath the entry point.
//!
//! Raw fibers never complete by returning. The entry point instead signals
//! completion through its own protocol and then suspends a final time; its
//! owner records this with [`RawFiber::finish`]. A finished fiber can be
//! destroyed without resuming it, since no destructors or cleanup obligations
//! may be live on its stack.
//!
//! # Snapshots
//!
//! A stopped raw fiber's stack can be captured with [`RawFiber::snapshot`] and
//! later restored into the same fiber with [`RawFiber::restore`]. Snapshots
//! capture the stack bytes between the saved stack pointer and the top of the
//! stack, which includes the callee-saved registers spilled by the switch
//! routine and therefore the complete suspended register context. Restoring
//! happens at the original addresses, so a snapshot is only valid for the
//! fiber that created it.
//!
//! The switch routine records where to resume the *host* in the reserved slot
//! at the top of the stack on every [`RawFiber::resume`]. Restoring a snapshot
//! therefore never resumes into the host continuation that existed when the
//! snapshot was taken: the next resume always returns to its own caller.
//!
//! It is the caller's responsibility that a snapshot is only taken while the
//! fiber is stopped at a point that is eligible to be restored: no Rust frames
//! with destructors or other cleanup obligations may be suspended on the
//! stack, and every pointer that the suspended code can dereference must still
//! refer to live storage, at the same address, whenever the snapshot is
//! restored and resumed. This is a contract of the code running on the fiber,
//! not something the fiber library can verify by inspecting stack bytes.

use crate::FiberStack;
use crate::stackswitch::{
    RAW_FIBERS, wasmtime_fiber_init, wasmtime_fiber_switch, wasmtime_fiber_switch_,
};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(any(not(feature = "std"), not(any(miri, windows)))))]
use unsupported::*;
use wasmtime_environ::error::{Error, Result, bail, ensure};

/// The entry point of a raw fiber.
///
/// This is called on the fiber's stack with the `arg` given to
/// [`RawFiber::new`] and the fiber's [`RawFiber::switch_arg`]. It must never
/// return.
pub type RawFiberEntry = unsafe extern "C" fn(arg: *mut u8, switch_arg: *mut u8);

/// The lifecycle of a [`RawFiber`].
///
/// A raw fiber is only ever observed while stopped; it runs only for the
/// duration of a call to [`RawFiber::resume`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RawFiberState {
    /// The fiber has not yet been resumed; resuming it calls its entry point.
    Initial,
    /// The fiber suspended itself and may be resumed.
    Suspended,
    /// The fiber suspended for the last time, as recorded by
    /// [`RawFiber::finish`]. It may not be resumed unless an earlier snapshot
    /// is restored first.
    Terminal,
}

/// A fiber running caller-supplied machine code. See the [module
/// documentation](self) for its contract.
pub struct RawFiber {
    stack: FiberStack,
    state: RawFiberState,
    id: usize,
}

/// A copy of a stopped [`RawFiber`]'s stack and lifecycle state.
pub struct RawFiberSnapshot {
    fiber: usize,
    sp: usize,
    bytes: Vec<u8>,
    state: RawFiberState,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

impl RawFiber {
    /// Whether raw fibers are supported on this platform and configuration.
    ///
    /// Raw fibers require one of the stack-switching routines in this crate
    /// (not an embedder's `custom` one) and are not supported on Windows,
    /// under Miri, or with AddressSanitizer, whose fiber-switch handshakes
    /// would require Rust code at every suspension point.
    pub fn is_supported() -> bool {
        RAW_FIBERS && !cfg!(asan)
    }

    /// Creates a new raw fiber that calls `entry(arg, switch_arg)` when first
    /// resumed.
    ///
    /// On error the stack is handed back to the caller.
    ///
    /// # Safety
    ///
    /// `entry` must uphold the contract of [`RawFiberEntry`] and suspend only
    /// through [`RawFiber::switch_routine`], passing [`RawFiber::switch_arg`].
    /// `arg` must remain valid for as long as the fiber may run.
    pub unsafe fn new(
        stack: FiberStack,
        entry: RawFiberEntry,
        arg: *mut u8,
    ) -> Result<Self, (Error, FiberStack)> {
        if !Self::is_supported() {
            return Err((
                wasmtime_environ::error::format_err!(
                    "raw fibers are not supported on this platform"
                ),
                stack,
            ));
        }
        let Some(top) = stack.top() else {
            return Err((
                wasmtime_environ::error::format_err!("raw fibers require a known stack top"),
                stack,
            ));
        };
        // SAFETY: the stack is owned by this fiber and large enough for the
        // initial frame, which the existing closure-based fibers rely on too.
        // The fiber-start trampoline only forwards both arguments to `entry`.
        unsafe {
            wasmtime_fiber_init(
                top,
                core::mem::transmute::<RawFiberEntry, extern "C" fn(*mut u8, *mut u8) -> *mut u8>(
                    entry,
                ),
                arg,
            );
        }
        Ok(RawFiber {
            stack,
            state: RawFiberState::Initial,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// The audited assembly routine which code on a raw fiber calls, with the
    /// C calling convention and [`RawFiber::switch_arg`] as its only argument,
    /// to suspend itself. It returns when the fiber is next resumed.
    pub fn switch_routine() -> unsafe extern "C" fn(*mut u8) {
        wasmtime_fiber_switch_
    }

    /// The argument which code on this fiber passes to
    /// [`RawFiber::switch_routine`].
    pub fn switch_arg(&self) -> *mut u8 {
        self.stack.top().unwrap()
    }

    /// The current lifecycle state of this fiber.
    pub fn state(&self) -> RawFiberState {
        self.state
    }

    /// The stack this fiber runs on.
    pub fn stack(&self) -> &FiberStack {
        &self.stack
    }

    /// Destroys this fiber, returning its stack. Nothing on the fiber's stack
    /// is run or unwound.
    pub fn into_stack(self) -> FiberStack {
        self.stack
    }

    /// Runs this fiber until it next suspends itself.
    ///
    /// Returns an error, without running anything, if the fiber is terminal.
    ///
    /// # Safety
    ///
    /// Everything the fiber's code requires must be valid for the duration of
    /// the call. In particular after a [`RawFiber::restore`], every pointer the
    /// restored continuation can dereference must refer to live storage.
    pub unsafe fn resume(&mut self) -> Result<()> {
        ensure!(
            self.state != RawFiberState::Terminal,
            "cannot resume a terminal raw fiber"
        );
        let top = self.switch_arg();
        // SAFETY: the saved stack pointer of a stopped fiber is in the
        // reserved slot. The switch stores this host continuation in the same
        // slot, so the fiber always suspends back to here.
        unsafe { wasmtime_fiber_switch(top) };
        self.state = RawFiberState::Suspended;
        Ok(())
    }

    /// Records that this fiber's most recent suspension was its last one.
    ///
    /// The fiber's owner learns this through its own protocol with the code
    /// running on the fiber.
    pub fn finish(&mut self) -> Result<()> {
        ensure!(
            self.state == RawFiberState::Suspended,
            "only a suspended raw fiber can finish"
        );
        self.state = RawFiberState::Terminal;
        Ok(())
    }

    fn saved_sp_slot(&self) -> *mut usize {
        // See the layout diagram in `unix.rs`: the slot below the run-result
        // slot holds the stack pointer to resume the fiber at.
        self.switch_arg().cast::<usize>().wrapping_sub(2)
    }

    fn saved_sp(&self) -> usize {
        // SAFETY: the reserved slot is initialized by `wasmtime_fiber_init`
        // and maintained by the switch routine.
        unsafe { self.saved_sp_slot().read() }
    }

    /// Copies this stopped fiber's stack and lifecycle state.
    pub fn snapshot(&self) -> Result<RawFiberSnapshot> {
        let top = self.switch_arg() as usize;
        let sp = self.saved_sp();
        let range = self.stack.range();
        ensure!(
            range.is_none_or(|r| r.start <= sp) && sp < top,
            "raw fiber has an invalid saved stack pointer"
        );
        let len = top - sp;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| wasmtime_environ::error::OutOfMemory::new(len))?;
        // SAFETY: `sp..top` is within this fiber's stack, which is not
        // running because `self` is borrowed here.
        bytes.extend_from_slice(unsafe { core::slice::from_raw_parts(sp as *const u8, len) });
        Ok(RawFiberSnapshot {
            fiber: self.id,
            sp,
            bytes,
            state: self.state,
        })
    }

    /// Restores a snapshot previously taken from this fiber, replacing its
    /// current continuation and lifecycle state. The next [`RawFiber::resume`]
    /// continues from the snapshot's suspension point (or the entry point, for
    /// a snapshot of an initial fiber), returning to that resume's caller.
    ///
    /// Snapshots taken from a different fiber are rejected without modifying
    /// this fiber.
    ///
    /// # Safety
    ///
    /// The snapshot must have been taken at a point eligible for restoration,
    /// as described in the [module documentation](self).
    pub unsafe fn restore(&mut self, snapshot: &RawFiberSnapshot) -> Result<()> {
        if snapshot.fiber != self.id {
            bail!("raw fiber snapshot belongs to a different fiber");
        }
        // SAFETY: the snapshot came from this fiber's stack, so the range is
        // in bounds. It includes the saved stack pointer slot, which this
        // overwrites with `snapshot.sp`, and the stack is not running.
        unsafe {
            core::ptr::copy_nonoverlapping(
                snapshot.bytes.as_ptr(),
                snapshot.sp as *mut u8,
                snapshot.bytes.len(),
            );
        }
        debug_assert_eq!(self.saved_sp(), snapshot.sp);
        self.state = snapshot.state;
        Ok(())
    }

    /// The return address and frame pointer of the code that called the
    /// switch routine, for walking a suspended fiber's frames.
    ///
    /// Returns `None` if the fiber has not suspended or if this is not
    /// implemented for the current architecture.
    pub fn suspended_frame(&self) -> Option<(usize, usize)> {
        // Indices, in words from the saved stack pointer, of the frame
        // pointer and return address that the switch routine spills. Keep in
        // sync with `wasmtime_fiber_switch_` in `stackswitch/*.rs`.
        let (fp, pc) = if cfg!(target_arch = "x86_64") {
            (5, 6)
        } else if cfg!(target_arch = "aarch64") {
            (18, 19)
        } else {
            return None;
        };
        if self.state == RawFiberState::Initial {
            return None;
        }
        let sp = self.saved_sp() as *const usize;
        // SAFETY: a suspended fiber's saved stack pointer addresses the
        // registers spilled by the switch routine.
        unsafe { Some((sp.add(pc).read(), sp.add(fp).read())) }
    }
}

impl RawFiberSnapshot {
    /// The lifecycle state the fiber will have once this is restored.
    pub fn state(&self) -> RawFiberState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    /// State shared between a test and the code on its raw fiber.
    #[derive(Default)]
    struct Shared {
        observed: Cell<u64>,
        finished: Cell<bool>,
    }

    /// Yields the values 1, 2, 3 and then finishes. The counter is local to
    /// the fiber, so a restored snapshot rewinds it.
    unsafe extern "C" fn count_to_three(arg: *mut u8, switch_arg: *mut u8) {
        // SAFETY: tests pass a `Shared` that outlives the fiber's use.
        let shared = unsafe { &*arg.cast::<Shared>() };
        let switch = RawFiber::switch_routine();
        let mut count = 0;
        while count < 3 {
            count += 1;
            shared.observed.set(count);
            // SAFETY: this is the switch routine for this fiber.
            unsafe { switch(switch_arg) };
        }
        shared.finished.set(true);
        // SAFETY: as above.
        unsafe { switch(switch_arg) };
        // Resuming a terminal fiber is rejected by `RawFiber::resume`.
        std::process::abort();
    }

    fn fiber(shared: &Shared) -> Option<RawFiber> {
        if !RawFiber::is_supported() {
            return None;
        }
        let stack = FiberStack::new(64 * 1024, false).unwrap();
        let arg = core::ptr::from_ref(shared).cast_mut().cast();
        Some(
            unsafe { RawFiber::new(stack, count_to_three, arg) }
                .map_err(|e| e.0)
                .unwrap(),
        )
    }

    /// Resumes `fiber` from a host stack frame `depth` calls deeper than the
    /// caller, so that each resume has a different host continuation.
    #[inline(never)]
    fn resume_at_depth(fiber: &mut RawFiber, depth: usize) {
        if depth == 0 {
            unsafe { fiber.resume().unwrap() };
        } else {
            resume_at_depth(fiber, depth - 1);
            core::hint::black_box(depth);
        }
    }

    #[test]
    fn lifecycle() {
        let shared = Shared::default();
        let Some(mut fiber) = fiber(&shared) else {
            return;
        };
        assert_eq!(fiber.state(), RawFiberState::Initial);
        for i in 1..=3 {
            resume_at_depth(&mut fiber, i as usize);
            assert_eq!(shared.observed.get(), i);
            assert_eq!(fiber.state(), RawFiberState::Suspended);
        }
        unsafe { fiber.resume().unwrap() };
        assert!(shared.finished.get());
        fiber.finish().unwrap();
        assert_eq!(fiber.state(), RawFiberState::Terminal);
        assert!(unsafe { fiber.resume() }.is_err());
        assert!(fiber.finish().is_err());
        // A terminal fiber is destroyed without being resumed.
        drop(fiber.into_stack());
    }

    #[test]
    fn rewind() {
        let shared = Shared::default();
        let Some(mut fiber) = fiber(&shared) else {
            return;
        };
        let initial = fiber.snapshot().unwrap();
        unsafe { fiber.resume().unwrap() };
        let first = fiber.snapshot().unwrap();
        assert_eq!(first.state(), RawFiberState::Suspended);

        // Repeatedly rewind to the first yield, resuming from different host
        // frames each time.
        for depth in 0..3 {
            unsafe { fiber.resume().unwrap() };
            assert_eq!(shared.observed.get(), 2);
            unsafe { fiber.restore(&first).unwrap() };
            resume_at_depth(&mut fiber, depth);
            assert_eq!(shared.observed.get(), 2);
            unsafe { fiber.restore(&first).unwrap() };
        }

        // Run to completion, then rewind after the final yield.
        for _ in 0..3 {
            unsafe { fiber.resume().unwrap() };
        }
        assert!(shared.finished.get());
        fiber.finish().unwrap();
        unsafe { fiber.restore(&first).unwrap() };
        assert_eq!(fiber.state(), RawFiberState::Suspended);
        unsafe { fiber.resume().unwrap() };
        assert_eq!(shared.observed.get(), 2);

        // Restart from the very beginning.
        unsafe { fiber.restore(&initial).unwrap() };
        assert_eq!(fiber.state(), RawFiberState::Initial);
        unsafe { fiber.resume().unwrap() };
        assert_eq!(shared.observed.get(), 1);
    }

    #[test]
    fn snapshots_belong_to_one_fiber() {
        let a = Shared::default();
        let b = Shared::default();
        let (Some(mut fa), Some(mut fb)) = (fiber(&a), fiber(&b)) else {
            return;
        };
        unsafe { fa.resume().unwrap() };
        let snapshot = fa.snapshot().unwrap();
        assert!(unsafe { fb.restore(&snapshot) }.is_err());
        assert_eq!(fb.state(), RawFiberState::Initial);

        // Destroying a fiber leaves its outstanding snapshots inert, even if a
        // new fiber reuses the same stack memory.
        unsafe { fa.resume().unwrap() };
        let stack = fa.into_stack();
        let arg = core::ptr::from_ref(&a).cast_mut().cast();
        let mut fc = unsafe { RawFiber::new(stack, count_to_three, arg) }
            .map_err(|e| e.0)
            .unwrap();
        assert!(unsafe { fc.restore(&snapshot) }.is_err());
        unsafe { fc.resume().unwrap() };
        assert_eq!(a.observed.get(), 1);
        drop(snapshot);
        unsafe { fb.resume().unwrap() };
        assert_eq!(b.observed.get(), 1);
    }

    #[test]
    fn resume_on_another_thread() {
        let shared = std::sync::Arc::new(SharedSync::default());
        if !RawFiber::is_supported() {
            return;
        }
        unsafe extern "C" fn entry(arg: *mut u8, switch_arg: *mut u8) {
            let shared = unsafe { &*arg.cast::<SharedSync>() };
            let switch = RawFiber::switch_routine();
            loop {
                shared.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                unsafe { switch(switch_arg) };
            }
        }
        #[derive(Default)]
        struct SharedSync(core::sync::atomic::AtomicUsize);

        let stack = FiberStack::new(64 * 1024, false).unwrap();
        let arg = std::sync::Arc::as_ptr(&shared).cast_mut().cast();
        let mut fiber = unsafe { RawFiber::new(stack, entry, arg) }
            .map_err(|e| e.0)
            .unwrap();
        unsafe { fiber.resume().unwrap() };
        let snapshot = fiber.snapshot().unwrap();
        let mut fiber = std::thread::spawn(move || {
            unsafe { fiber.resume().unwrap() };
            fiber
        })
        .join()
        .unwrap();
        unsafe { fiber.restore(&snapshot).unwrap() };
        unsafe { fiber.resume().unwrap() };
        assert_eq!(shared.0.load(core::sync::atomic::Ordering::Relaxed), 3);
    }

    #[test]
    fn suspended_frame_points_into_entry() {
        let shared = Shared::default();
        let Some(mut fiber) = fiber(&shared) else {
            return;
        };
        assert!(fiber.suspended_frame().is_none());
        unsafe { fiber.resume().unwrap() };
        if let Some((pc, fp)) = fiber.suspended_frame() {
            // Without frame pointers in this Rust entry, `fp` may still be
            // the initial one: the top of the stack.
            let range = fiber.stack().range().unwrap();
            assert!(
                range.start <= fp && fp <= range.end,
                "{fp:#x} vs {range:x?}"
            );
            let entry = count_to_three as *const () as usize;
            assert!(pc > entry && pc < entry + 4096, "{pc:#x} vs {entry:#x}");
        }
    }
}
