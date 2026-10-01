The next implementation step is described in the
[replay fiber kickoff plan](record-replay-fiber-plan.md). Its agreed design
supersedes the fiber/snapshot roadmap below; this document describes the current
implementation.

The `wasmtime/rr` feature records core Wasm execution for a whole
store and replays it with a top-level driver of independent Wasm activations.
Recording includes core and component execution, object construction, and
startup. Components replay as their constituent core modules; replay does not
reconstruct the component runtime or require a component linker. Snapshot-ready
fiber images remain implementation work. Reversible guest debugging is an
intended consumer of this design; its current runtime guard is temporary.

The entrypoints are `Store::start_recording()`,
`Store::finish_recording() -> Result<rr::Trace>`, and
`Store::replay(&rr::Trace).await -> Result<rr::Replay>`. An engine uses
`Config::rr(RRConfig::Recording)` or `Config::rr(RRConfig::Replaying)`.
Recording can use synchronous or asynchronous calls; replay always uses fibers.
The feature does not require component-model support.

```rust,ignore
// Recording begins with an empty store, before imports or instances exist.
store.start_recording()?;
let instance = linker.instantiate(&mut store, &component)?;
let run = instance.get_typed_func::<(), ()>(&mut store, "run")?;
run.call(&mut store, ())?;
let trace = store.finish_recording()?;
std::fs::write("execution.rr", trace.as_bytes())?;

// Replay requires an empty store with an RRConfig::Replaying engine.
let replay = replay_store.replay(&trace).await?;
// Core instance exports are available for inspection, in construction order.
let core_instances = replay.instances();
```

Objects are identified automatically. Functions receive compact u32 IDs by
recorded construction order; imported aliases share an ID with their
definition. Only escaping Wasm functions have function references
and can cross the host boundary. This includes private functions reachable
through tables. Memory IDs follow defining instance/memory order, including
host-created memories' dummy instances. Export names and lookup order do not
participate. Runtime addresses are only reverse-lookup keys within one store;
no address or process-wide StoreId is serialized.

Recording and replay require an empty store: no core or component instances,
host functions, memories, globals, tables, or GC objects. Application data `T`
may already be initialized. Modules and linker definitions prepared outside the
store do not populate it. All guest initialization is recorded, and replay uses
only modules and object definitions from the trace. Debugger interaction is a
separate observation/control API concern, described below.

Construction records contain module bytecode, host function signatures, object
types and initial values, and core instance imports by object ID. Repeated
instantiation of the same `Module` reuses its trace entry by object identity;
separately compiled modules have separate entries. Replay validates and
compiles the original Wasm, typechecks imports, allocates the instance, and
drives its generated startup function as an ordinary guest activation. Start-function host calls and nested
instantiation use the same protocol as later execution. Even a startup that
traps is replayed and checked. A missing or repeated startup is rejected.
Replaying module construction requires a compiler. Traces never load serialized
native code. Newly created core instances are returned in `rr::Replay`,
including the constituent modules and synthetic adapters of components.

In addition to construction, the execution stream contains:

* Calls: `EnterWasm`, `LeaveWasm`, `EnterHost`, and `LeaveHost`. Entries identify
  the callee and carry core arguments. Each call has a unique dynamic ID;
  returns identify the call they complete, so concurrent component activations
  can finish in any order. Returns carry either core results, a
  Wasm trap code, or a host error message. Outgoing host arguments and guest
  results are included for divergence detection. Host-to-host `Func` calls
  produce no call events; their memory effects and guest callbacks are still
  recorded. Ordinary Wasm-to-Wasm calls stay within an activation.
* Memory effects: byte writes and host-initiated growth. Growth checks the old
  byte length before extending the memory. Guest stores and successful guest
  growth execute normally and are not recorded as host effects. Component
  instance flags are ordinary mutable i32 globals in the core import graph;
  global-write records reproduce the canonical runtime's changes at guest
  entry and host return boundaries.

All component core-callable trampolines and intrinsics are wrapped as ordinary
opaque host calls. The wrapper includes generated entry checks and resource
operations. Known-intrinsic inlining is disabled for RR compilation so those
calls cannot bypass interception. Replay supplies their recorded core results
and memory effects without executing the component host runtime. No string,
list, resource, future, or stream event types are needed.

Generated guest resource destructors are routed through the ordinary guest
entry path, as are realloc, post-return, and asynchronous canonical callbacks.
Component memory access uses the same write tracker as core host calls, and
transcoders register their destination ranges. Component-to-component adapters
are replayed as ordinary core modules. The driver also supports suspended
component activations that complete out of creation order.

The private versioned binary format uses a one-byte tag, a little-endian u32
payload length, and a payload. Values have widths determined by their core
signature (4, 8, or 16 bytes); it never copies padding from `ValRaw`. Function
references are u32 function IDs, with zero for null, and are resolved against
the replay store before entering guest code. This also covers internal async
adapter signatures; raw function pointers are never serialized. Memory
offsets and lengths of grown memories are u64. An explicit end record is
required. `Trace::from_bytes` validates framing; replay validates object
construction and the event protocol as well. The format has no cross-version
compatibility promise.

Recording appends directly to a store-owned `Vec<u8>`. Function and memory
identities are resolved through hash maps populated during construction.
Ordinary call events require no temporary event allocation and no per-event
I/O or writer dispatch.
Buffer growth is fallible. Chunk rotation, configurable size limits,
compression, and an inline VMStoreContext append cursor are future work.
The trace currently stays in memory until finalization.

`Memory::data_mut_tracked(store, range)` returns an `rr::MemoryMut` guard. It
registers a pending write when mutably dereferenced and commits on drop.
`Memory::write` registers only its destination range. Legacy `data_mut` and
`data_and_store_mut` register the full memory. Overlapping pending ranges can
be coalesced within a host segment. Pending writes are flushed before guest
entry, host return, host growth, and finalization. Consequently, forgetting a
guard does not lose writes or accidentally record later guest writes as host
effects. Infallible accessors/destructors poison the recording on failure;
`finish_recording` reports the error and discards the session.

Writes through `Memory::data_ptr` or other untracked pointers are outside the
supported contract. A pointer obtained through a tracked guard may be used
only within the guard's borrowing rules. Shared/external concurrent writes
cannot be represented by the current serial trace.

The replay driver owns all activations and resumes one at a time. A guest call
to a host import yields before constructing `Caller`, entering the host GC
scope, or invoking the host closure/future. The driver matches `EnterHost`,
applies memory effects, and starts callbacks on separate fibers. `LeaveHost`
decodes results directly into the parked activation's array-call slots and
resumes it. These slots are treated as potentially uninitialized storage.
The entry value buffers belong to the driver, with stable addresses for the
entire activation lifetime. Suspended entry/host-boundary frames don't own
these allocations; the driver reuses scratch storage for comparisons. Guest
return values and traps are checked against the trace; a correctly reproduced
guest trap is a successful replay.

Dropping the replay future, including after a decoding/divergence error,
disposes activations in reverse order before freeing their buffers or releasing
the store. Replay yields to the async executor after a bounded number of trace
events. An activation that never reaches a boundary does not yield;
deterministic interruption is not implemented. Failed or cancelled replay
does not roll back guest state; retry with a fresh store.

Current explicit restrictions include GC and typed function references in
core signatures, GC-using modules, shared memories, guest stack switching, resource
limiters, call hooks, custom signal handlers, fuel, and epochs. Component-level
resources, futures, and streams cross these boundaries as numeric core handles
and are supported. Guest debug events are temporarily disabled pending the
integration below. Host table/global mutation is rejected; construction of
numeric or abstract funcref globals and abstract funcref tables is supported.
Failed guest memory/table growth invalidates a recording until allocation decisions have a replay policy. Host panics leave
unmatched calls and cannot be finalized into a complete trace. Arbitrary Rust
error objects are represented by messages, not recreated.

The remaining implementation work is:

1. Complete the replay fiber image contract. `StoreFiber` keeps
   `FiberResumeState` (saved TLS, stack limit, and protection-key state) and a
   completion bit outside its stack. A future fiber snapshot primitive must
   preserve or reconstruct these alongside the stack, with stable addresses.
   Activation descriptors and entry arguments can be reconstructed from the
   trace's call IDs and entry records, but their backing allocations must
   survive while snapshots refer to them. Generic trap/catch-unwind entry and
   exit frames also need an ownership audit and a minimal replay-specific path
   where required. The current implementation is not safe to restore by
   copying stack bytes alone. No snapshot API or stack-copy primitive is
   introduced here. Guest memory/table/global snapshots remain separate from
   control-stack restoration.

2. Integrate async debug hooks with the driver as an observation/control path.
   Breakpoints, single-step stops, traps, and hostcall errors must pause the
   current replay activation without consuming or inventing host-call records.
   The replay debug trampoline should yield a debug stop to the driver; the
   driver should run/poll the async `DebugHandler` outside the guest fiber,
   then resume that same activation. No arbitrary debugger future may remain
   on a stack that is snapshotted. Borrowed error payloads need driver-owned
   storage or a stable borrow of the parked activation. Exception events will
   additionally require the currently unsupported GC/exception policy.

   Debug inspection must explicitly select the parked activation and use its
   saved stack/TLS context: the existing API walks the currently entered Wasm
   stack and cannot simply be called after yielding without this adaptation.
   Nested activations should be exposed in logical call order even though
   their stacks are separate. The ordinary runtime context-switch discipline
   must still apply across every await and executor thread migration.

   A debug stop can occur between trace events, so a snapshot checkpoint must
   preserve that paused PC and the cursor of the next expected event. At a
   host stop, matching the observed host call must precede a checkpoint.
   Restoring should discard any in-progress debugger future and produce a
   fresh stop notification. Breakpoint configuration and debugger/controller
   state are outside the reproduced execution. Read-only inspection can
   continue replay unchanged; state edits and debugger-invoked guest calls
   need an explicit fork/new-recording policy (or rejection), since silently
   modifying state would invalidate deterministic replay. Tests must cover
   async hook yields, nested activations, changing breakpoints, cancellation,
   trap stops, and eventually repeated backward/forward restoration. Remove
   the temporary debug guard when this path and its inspection semantics are
   implemented, not merely when generic async hooks can be polled.

3. Specify allocation failure and deterministic interruption behavior, extend
   backend/platform coverage, and measure append overhead and trace volume.

The integration tests are in `crates/wasmtime/tests/record_replay.rs`. They
cover nested callbacks, private table callbacks and Linker imports, memory
aliases, writes around callbacks, forgotten guards, growth, numeric/vector
bits, caught traps, async host errors, malformed traces, cancellation, direct
host calls, rejection of nonempty stores, segment drops, and restrictions. The
initialization tests replay from empty stores, including start-function host
calls and traps, imported host objects, component strings and post-return,
resource destructors, cross-component transcoding, concurrent activations,
asynchronous canonical callbacks and lowering, future transfers, subtask
cancellation, stream wait results, and function references across boundaries.

```sh
cargo test -p wasmtime --features rr --test record_replay
cargo test -p wasmtime --release --no-default-features \
  --features cranelift,runtime,std,rr,wat,component-model-async --test record_replay
cargo check -p wasmtime --no-default-features --features runtime,rr
```
