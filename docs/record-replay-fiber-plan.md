Record/replay fibers: plan
==========================

**Status (2026-10-01).** Steps 1–5 and 7 are implemented; step 6 is deferred.
[record-replay.md](record-replay.md) describes the result.

- Raw fibers (`wasmtime_fiber::RawFiber`): entry point plus context, no Rust
  closure, explicit `Initial`/`Suspended`/`Terminal` lifecycle, terminal resume
  rejected, destruction without unwinding. Fiber-owned snapshots restore at
  the original address, are bound to their fiber by identity, and always
  return to the current resume's caller. Tested on x86-64 Linux: multiple
  yields, repeated rewind from different host frames, rewind after the final
  yield and to the initial state, cross-thread resume, and snapshot ownership.
  Other Unix architectures share the reserved-slot layout and compile, but
  were not run here. Windows, Miri, and AddressSanitizer report raw fibers as
  unsupported.
- `FuncKey::ReplayStart` and `FuncKey::ReplayHostCall` are compiled into every
  module of a record/replay engine for a native target (`Tunables::recording`);
  the driver takes them from an empty module that it compiles. They reach the switch routine through a function
  pointer in `VMReplayControl`, so no process address is serialized. A test
  compiles them for x86-64, aarch64 (Linux and macOS), s390x, and riscv64.
- Traps, recorded host errors, and synchronous libcalls run through the
  existing array-to-Wasm landing pad and `raise`; the driver installs a
  `CallThreadState` and the activation's `VMStoreContext` state around each
  resume. After every yield the driver verifies (x86-64 and aarch64, debug
  and release) that the suspended stack contains only generated code.
- Replay host stubs, the old Rust interception in `HostFunc`,
  `StoreFiberYield::ReplayHost`, and `resume_replay_fiber` are replaced or
  removed. The trace format is unchanged. Pulley engines reject replay.
- Since then (see record-replay.md): whole-store checkpoints, breakpoint and
  single-step stops yielded through the control block (part of step 6), and
  inspection of parked activations are implemented, and the raw-fiber tests
  also pass on aarch64, s390x, and riscv64 under qemu. Still deferred from
  step 6: trap/host-error debug events and ordinary asynchronous execution
  through the yield contract.


The rest of this document is the design brief agreed on 2026-09-21, before
implementation. References to the "current" or "existing" implementation
describe the closure-based replay fibers that it replaced.

The objective is to make replay guest fibers contain zero Rust frames at every
permitted snapshot point. Replace the Rust entry closure and replay hostcall
function with generated trampolines. Put raw fiber creation, stack snapshots,
and restore fixups in `wasmtime-internal-fiber`. Preserve the existing core,
component, and initialization replay behavior.

**Decisions already made**

- Zero Rust frames means at snapshot points. Synchronous runtime libcalls may
  execute Rust on the guest stack, provided they finish before suspension.
  Audited fiber-switch assembly may remain on the suspended stack.
- Every asynchronous call must leave the guest through the generated yield
  mechanism. Anything that can produce a debug event is an asynchronous call
  for this purpose. No debug callback, future, or Rust adapter may suspend on
  the snapshot-capable guest stack.
- During replay, the driver supplies recorded host results and effects without
  executing original host functions or component builtins. Outside replay,
  asynchronous calls execute on an owned child fiber; their Rust frames are
  separate from the guest continuation. This is the architecture to support,
  not a reason to replace the existing replay behavior with host execution.
- A generated start trampoline wraps the guest invocation. Guest return causes
  a final yield meaning "guest returned; you can drop me". Such a fiber never
  returns to its creator through an ordinary function return. Resuming its
  terminal continuation without restoring an earlier snapshot is invalid.
- The fiber library owns fiber snapshots and the platform-specific restore
  rules. The replay driver must not manipulate stack-layout offsets or copy
  stacks itself. Wasmtime still owns store state and replay protocol state.
- Recording and replay continue to start with empty stores, reconstructing
  modules solely from bytecode in the trace. Keep numeric object identities,
  module identity deduplication, and component builtins as ordinary calls.
- Full guest-memory/table/global rollback and a public reversible-debugger API
  are separate work. Implement and test fiber snapshots here, and describe the
  additional state a whole-store checkpoint must coordinate. Do not advertise
  fiber restoration alone as whole-store restoration.

**Where the implementation stands**

Code layout when this plan was written (since replaced):

| Location | Relevant current behavior / planned work |
| --- | --- |
| `crates/wasmtime/src/runtime/rr/replay.rs` | `Activation` owns a `StoreFiber`; `Driver::step` creates a Rust closure that calls `Func::call_unchecked_raw`; `host_call` suspends via `with_blocking`; cleanup calls `dispose`. Replace this guest-side path. |
| `crates/wasmtime/src/runtime/rr/init.rs` | HOST records create `Func::new` replay stubs. Construct typed replay functions whose array-call entry targets generated code directly. |
| `crates/wasmtime/src/runtime/func.rs` | The Rust array-call wrapper intercepts replay and calls `rr::replay::host_call`. Replay must bypass that wrapper. |
| `crates/wasmtime/src/runtime/rr.rs` | `Mode::Replaying` has `yielded`, `response`, `completed`, and `entering` state. Move ABI communication into fixed-layout storage; retain driver-owned protocol/error state as needed. |
| `crates/wasmtime/src/runtime/fiber.rs` | `StoreFiber`, `FiberResumeState`, `resume_replay_fiber`, and `make_fiber_unchecked` provide the current lifecycle and context switching. Extract/adapt the host-side discipline without retaining the closure-based guest path. |
| `crates/fiber/src/lib.rs`, `unix.rs`, `windows.rs`, `stackswitch/*` | Closure entry, generic `RunResult`, completion bookkeeping, stack allocation, and switch assembly. Add a distinct raw lifecycle and snapshot abstraction. |
| `crates/environ/src/key.rs`, `compile/mod.rs` | Add trampoline keys and compilation plumbing, including key encoding/decoding. |
| `crates/environ/src/vmtypes.rs`, `vmoffsets.rs`; `crates/wasmtime/src/runtime/vm/vmcontext.rs` | Define fixed-layout communication fields and generated-code offsets/defaults. |
| `crates/cranelift/src/compiler.rs`, `alias_region.rs`; `crates/wasmtime/src/compile.rs` | Compile, link, root, and expose the generated trampolines; model their memory accesses correctly. |
| `crates/wasmtime/src/runtime/vm/traphandlers.rs` | Trap registration, `CallThreadState`, error propagation, and unwinding need a raw replay entry/exit contract. |
| `crates/wasmtime/tests/record_replay.rs`, `rr_async_component.wat` | Existing behavior coverage to preserve and extend. |

The x86-64 switch currently has an ordinary Rust
`wasmtime_fiber_switch` wrapper around naked-assembly
`wasmtime_fiber_switch_`. Calling the wrapper is not sufficient for the new
invariant. Expose an audited assembly entry through the fiber library and check
other architectures individually. Avoid depending on optimizer inlining or a
particular release build to eliminate Rust frames.

**Target control flow and ownership**

```text
Rust driver, running on the host stack
  -> fiber library raw resume / assembly switch
    -> generated start trampoline
      -> existing array-to-Wasm trampoline -> guest
        -> existing wasm-to-array trampoline
          -> new generated array-ABI yield trampoline
            -> fiber library assembly switch -> driver

Guest return -> generated start trampoline -> terminal yield -> driver
```

Use an activation-owned, fixed-address control block, reachable through a
pointer in `VMStoreContext`. Proposed fields include entry function/context,
caller context, entry array pointer/length, suspension reason, observed callee
context or identity, hostcall array pointer/length, and resume disposition.
Exact names and layout are implementation choices. Use explicit integer tags,
raw pointers, and generated offsets; no Rust enums with implicit layout,
references, trait objects, `Result`, or owning values in the JIT-facing ABI.
Rust-owned errors and futures belong outside the guest continuation.

The hostcall trampoline has the generic array calling convention; the existing
wasm-to-array trampoline handles signature-specific argument spilling. The new
trampoline publishes the call data, invokes the assembly switch directly, and
on resumption returns the appropriate ABI status. Callee lookup, signature and
buffer validation, trace decoding, argument comparison, result construction,
and call-ID bookkeeping run on the driver stack. Results-only slots can be
uninitialized until the driver writes them; do not form initialized Rust values
from the whole array.

Install the selected activation's control pointer on every resume. Copy or
retain suspension metadata per activation before running another activation.
Nested callbacks and concurrent component activations can complete out of
creation order, so a single durable store-wide mailbox or a LIFO-only protocol
is insufficient. No Rust borrow into the stack or control block may survive a
resume or restoration that can change the borrowed storage.

**Implementation sequence**

1. Define the raw fiber lifecycle and ABI before changing the replay driver.
   Provide an unsafe entry-point-and-context constructor with no Rust closure
   on the guest stack. Define states for initial entry, running, suspended,
   terminal yield, and disposed. A terminal fiber can be destroyed without
   resuming/unwinding it; restoring a retained earlier snapshot makes its prior
   suspended continuation runnable again. Unexpected entry return or terminal
   resume must fail deterministically. Preserve the ordinary closure-based API
   for its existing callers.

   Specify the snapshot safety contract: only stopped fibers at eligible yield
   points, no suspended Rust frames or cleanup obligations, and live backing
   storage for all pointers the continuation can dereference. Eligibility is
   guaranteed by the caller/entry ABI, not inferred by scanning bytes.

2. Implement fiber-owned snapshots and restoration for the first supported
   native target, then cover the intended platform matrix. An opaque snapshot
   preserves stack bytes, guest register/resume context, and fiber lifecycle
   state. Initially restore at the original stack address; snapshots keep the
   required stack allocation alive or require an explicit retained owner.
   Reject incompatible fiber/stack images before mutating the destination.

   Completion-by-yield avoids restoring a vanished caller through an ordinary
   return. Still make the switch bookkeeping explicit: the saved guest context
   belongs to the snapshot, while the host return context belongs to the
   current resume invocation. Restore or reinitialize reserved switch slots as
   required inside the fiber library. Never resume into a host continuation
   from the snapshot's creation. Test this rather than imposing driver-side
   stack fixups. Include restore after final yield, repeated restoration, and
   destruction with outstanding snapshot ownership.

3. Add generated start and hostcall-yield trampolines. Use new `FuncKey`s
   (working names `ReplayStart` and `ReplayHostCall`, not fixed API names), wire
   their encoding, compiler dispatch, symbol/linking metadata, and code lifetime.
   Keep the yield trampoline signature-independent. Use runtime relocation or
   an explicit function pointer for the audited fiber switch entry; do not
   serialize a process address into an artifact.

   The start trampoline invokes the supplied array-to-Wasm entry and reports
   successful completion with a final yield. Use the same path for generated
   module startup activations. Provide correct unwind/stack-walking metadata
   and a defined invalid-resume path. Check generated code for accidental Rust
   helpers on any suspension path, including debug builds.

4. Establish trap, error, and runtime-context behavior before migrating all
   replay calls. Keep host-side TLS/trap registration alive around each resume,
   and rebuild it for that invocation/thread. Preserve stack limits, guard
   ranges, executor context, protection keys where enabled, and guest stack
   walking. Do not snapshot pointers to expired host-side `CallThreadState`s.

   Audit native faults, explicit Wasm traps, recorded host errors, synchronous
   libcalls that trap, and host panic cleanup. The normal failed array-call
   path can invoke `raise`; routing success through generated code alone is
   insufficient. Use an appropriate generated/assembly trap landing or exit
   path so any snapshot-eligible stop has no live Rust frames. Temporary Rust
   trap processing is allowed under the synchronous rule, but it must finish
   before such a yield. Construct errors/backtraces on the host side with the
   required guest state still available. No Rust unwinding may be necessary
   to dispose a suspended raw guest fiber.

5. Migrate replay to the new path. Replace `Func::new` replay stubs with rooted,
   correctly typed contexts that point directly to the generated array entry.
   Replace closure-created `StoreFiber`s with raw activations and consume the
   control block in `Driver::resume`. Preserve trace validation, startup
   sequencing, nullable funcrefs, memory effects, traps, cancellation, and
   concurrent call-ID routing. Remove obsolete replay-only Rust interception
   and `ReplayHost` suspension plumbing once unused; retain ordinary runtime
   functionality. Keep trace-format changes out of this refactor unless an
   actual serialization change requires a version bump.

6. Route asynchronous/debug boundaries through the same yield contract.
   Debug stops publish an event to the driver without consuming or inventing
   hostcall trace records. Poll handlers outside the guest fiber and inspect
   the selected parked activation using its saved stack metadata. For ordinary
   execution, use an owned child fiber for asynchronous work; define ownership,
   cancellation, nested guest entry, and teardown before permitting that path
   to participate in snapshots. Child fibers containing Rust state are not
   copied as guest snapshots. Restoring a guest checkpoint must dispose or
   reconstruct external async work according to the controller policy.

   Keep the current guest-debug restriction until the event and inspection
   paths meet this contract. If full debug/non-replay integration is a later
   increment, leave an explicit remaining-work item and do not describe it as
   supported merely because the yield reason exists.

7. Specify the checkpoint boundary above fibers and update documentation.
   A replay checkpoint also needs trace cursor, dynamic call IDs, activation
   set and statuses, pending startup/observed event state, control blocks,
   driver-owned entry/result buffers, and guest store state. Shared references
   must resolve to live storage at the same addresses. Keep module code,
   contexts, and completed activation storage alive while snapshots need them.
   Define handling of objects created after a checkpoint instead of assuming
   restoring bytes removes them. At a hostcall checkpoint, state whether the
   EnterHost record has been matched; use one consistent convention. Debug
   checkpoints also preserve the stopped PC and next expected trace event.

   Test fiber rewind using controlled state, and state clearly which higher
   level restoration remains unimplemented. Rewrite the outdated remaining
   work in `record-replay.md` and remove comments describing the old Rust-frame
   replay path once that path is gone.

**Validation and completion criteria**

- Verify actual suspended stacks at hostcall and terminal yields contain only
  generated code and the audited assembly path. Use emitted-code/stack-walk
  checks in debug and optimized builds; ordinary functional tests alone do not
  prove the absence of Rust frames.
- At the fiber layer, test multiple yields, retained snapshots, repeated
  rewind, rewind after terminal yield, different host resume invocations, and
  compatible executor-thread migration. Verify the restored fiber yields to
  the current host continuation. Cover invalid lifecycle transitions and
  snapshot/fiber ownership without relying on Rust destructors on the guest.
- Retain all RR integration coverage: nested/caught traps, asynchronous host
  errors, component resources and callbacks, out-of-order activations, imported
  objects and aliases, initialization/startup, malformed traces, and cancellation.
  Add terminal-yield and trap-path cleanup regressions. Callbacks during replay
  must still run as separate guest activations without original host execution.
- Test debug suspension/resumption, cancellation, and nested inspection when
  enabling that integration. Check that handlers/async work run off the guest
  stack and do not alter trace position merely by observing execution.
- Handle native backend/platform differences explicitly, including Windows,
  sanitizer stack-switch handshakes, unwind metadata, and architecture-specific
  register state. Pulley needs separate treatment because interpretation runs
  Rust; do not silently claim the native snapshot guarantee there. Unsupported
  configurations should report an explicit limitation, not use a hidden
  closure-based fallback for snapshot-capable replay.
- Preserve compilation with RR disabled, without component support, and in
  compiler-free runtime builds. Compiler-free builds can retain the existing
  error for traces requiring module compilation. Add targeted compiler tests
  for new keys, trampoline generation, and fixed-layout offsets.

Useful baseline commands (add focused raw-fiber/compiler tests as implemented):

```sh
cargo test -p wasmtime-internal-fiber
cargo test -p wasmtime --features rr --test record_replay
cargo test -p wasmtime --no-default-features \
  --features cranelift,runtime,std,rr,wat --test record_replay
cargo test -p wasmtime --release --no-default-features \
  --features cranelift,runtime,std,rr,wat,component-model-async --test record_replay
cargo check -p wasmtime --no-default-features --features runtime,rr
cargo check -p wasmtime --no-default-features --features runtime,std,cranelift
cargo clippy -p wasmtime --features rr --test record_replay -- -D warnings
cargo fmt --all -- --check
```
