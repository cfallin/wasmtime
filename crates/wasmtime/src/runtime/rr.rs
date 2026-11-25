//! Record/replay for a whole store.
//!
//! Recording begins with an empty store, created by
//! [`Store::new_recording`](crate::Store::new_recording). Objects receive
//! numeric identities automatically; all construction is part of the trace.
//! Traces identify core modules only by the checksums of their bytecode, so
//! that they stay small. Components are recorded as the core instances that they
//! instantiate.
//!
//! A `Replayer` replays a trace: it compiles each module from bytecode that the
//! embedder provides, and executes guest code without invoking the original
//! host functions. It requires a compiler and a native (non-Pulley) target, and
//! is unsupported under Miri, with AddressSanitizer, and with hardware-enforced
//! shadow stacks.
//!
//! The embedder can record its own events, such as the program's output, for
//! replay to report. Replay can also stop at breakpoints, watchpoints, and
//! single steps, with guest debugging enabled on the replaying engine, and be
//! rewound to checkpoints.
//!
//! Core function boundaries support numbers, vectors, and abstract function
//! references. Guest code may use GC and typed function references, but they may
//! not cross the host boundary, and the host may not write to GC objects. It may
//! throw and catch exceptions, including those that reach the host, but host
//! functions may not throw them. Importing tags is unsupported.
//! Shared memory, host table/global mutation, call hooks,
//! custom signal handlers, Wasm stack switching, epoch deadline callbacks, and
//! guest debugging are unsupported. Fuel and epoch interruption may yield to
//! the async executor, which the guest cannot observe, but a recording in
//! which they stop the guest (by trapping, or by the call being cancelled
//! while it yields) cannot be replayed. Host writes through the
//! memory APIs, including slices from
//! [`Memory::data_mut`](crate::Memory::data_mut), are recorded; writes through
//! raw pointers such as [`Memory::data_ptr`](crate::Memory::data_ptr) are not.
//!
//! Traces describe execution at the level of WebAssembly semantics, and their
//! format is versioned independently of Wasmtime, so that a trace does not have
//! to be replayed by the version of Wasmtime that recorded it. They contain
//! host-supplied data and can be large; applications should impose their own
//! storage limits.

use crate::store::InstanceId;
use wasmtime_environ::{DefinedMemoryIndex, DefinedTableIndex};

// The rest of the runtime uses record/replay through the crate-private API that
// both of these modules provide, which does nothing when the feature is
// disabled.

#[cfg(feature = "rr")]
mod enabled;
#[cfg(feature = "rr")]
pub use enabled::*;

#[cfg(not(feature = "rr"))]
mod disabled;
#[cfg(not(feature = "rr"))]
pub(crate) use disabled::*;

/// An object that guest code can grow.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "rr"), expect(dead_code, reason = "never recorded"))]
pub(crate) enum Growable {
    Memory(InstanceId, DefinedMemoryIndex),
    Table(InstanceId, DefinedTableIndex),
}
