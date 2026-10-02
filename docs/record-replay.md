# Record and replay

The `wasmtime/rr` feature records core Wasm execution for a whole
store and replays it with a top-level driver of independent Wasm activations.
Recording includes core and component execution, object construction, and
startup. Components replay as their constituent core modules; replay does not
reconstruct the component runtime or require a component linker. Suspended
replay activations contain no Rust frames, so a replay can be checkpointed and
rewound, and guest debugging (breakpoints, single-stepping, frame inspection)
works on replay. Together these support reversible debugging.

The entrypoints are `Store::start_recording()`,
`Store::finish_recording() -> Result<rr::Trace>`, and
`Store::replay(&rr::Trace).await -> Result<rr::Replay>`. `Store::replayer`
returns an `rr::Replayer` that replays one stop at a time. An engine uses
`Config::rr(RRConfig::Recording)` or `Config::rr(RRConfig::Replaying)`.
Recording can use synchronous or asynchronous calls; replay always uses fibers.
The feature does not require component-model support. Replay requires a
compiler; see the restrictions below.

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
// Core instances are available for inspection (not calls), in construction
// order.
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
* Embedder events: `rr::record_event` appends a value of an embedder type
  implementing `rr::TraceEvent` (a stable `u32` tag plus serde, encoded with
  postcard) at the current point of the recording, e.g. a WASI
  implementation's output. During replay, `Replayer::on_event` observers
  receive them as replay reaches them, including again after rewinding;
  observers cannot affect the replay.
* Guest growth failures: a failed `memory.grow` or `table.grow` records the
  object, its old size, and the delta. Such records directly follow the event
  that resumed the guest; the driver queues them, and replay fails exactly the
  matching growth, even where the replaying engine could allocate more. An
  unrecorded failure, or a recorded one that does not recur, is a divergence.

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

Host writes to guest memory are recorded through the ordinary APIs, so that an
embedding runtime needs no record/replay-specific code. Handing out a mutable
view registers its range as pending: `Memory::data_mut` and
`data_and_store_mut` register the whole memory, `Memory::write` its
destination, and component lowering (the code that `bindgen!` host bindings
use) and builtins only the bytes they write. The public
`LowerContext::as_slice_mut` registers the whole memory. Registration is
constant time; at the next guest entry, host return, host growth, or
finalization, pending ranges are sorted, merged, and their current contents
appended to the trace. A write is therefore never lost, and a later guest
write is never mistaken for a host effect. Infallible accessors poison the
recording on failure; `finish_recording` reports the error and discards the
session.

Writes through `Memory::data_ptr` or other raw pointers are outside the
supported contract. Shared/external concurrent writes cannot be represented by
the current serial trace.

The replay driver owns all activations and resumes one at a time. Each guest
activation runs on a raw fiber (`wasmtime_fiber::RawFiber`) whose stack, while
suspended, holds only generated code and the fiber library's audited assembly:

```text
fiber start (asm) -> ReplayStart -> array-to-Wasm -> guest
    -> Wasm-to-array -> ReplayHostCall -> fiber switch (asm)
```

`ReplayStart` and `ReplayHostCall` are signature-independent trampolines
(`FuncKey`s of their own). Engines configured for record/replay compile them
into every module for a native target; the driver takes them from an
otherwise empty module that it compiles when replay starts and that keeps
their code alive. Replay host stubs are typed
`VMArrayCallHostFuncContext`s whose array-call entry is `ReplayHostCall`, so
a guest call to a host import never enters Rust: the trampoline publishes the
callee and its array-call buffer in the activation's `VMReplayControl` and
calls the fiber switch routine directly. Each activation owns its control
block at a fixed address; `VMStoreContext::replay_control` points at the
running activation's. The driver identifies and validates the callee, matches
`EnterHost`, applies memory effects, and starts callbacks as separate
activations. `LeaveHost` decodes results directly into the parked activation's
array-call slots (treated as potentially uninitialized storage) and resumes
it, or records the host error for the Wasm-to-array trampoline to raise.

`ReplayStart` calls the activation's entry function with the array calling
convention. A return or a trap (which lands in the array-to-Wasm trampoline)
is reported by a final yield; the driver records the fiber as terminal and
destroys it without resuming it. The entry value buffers belong to the driver,
with stable addresses for the entire activation lifetime; the driver reuses
scratch storage for comparisons. Guest return values and traps are checked
against the trace; a correctly reproduced guest trap is a successful replay.

State that an ordinary call keeps on the host stack is installed by the driver
around each resumption instead: the `CallThreadState` that signal handlers and
`raise` use, the activation's entry/exit registers and stack chain, its stack
limit and guard range, and its protection-key mask. Synchronous libcalls,
including trap raising, still run Rust code on the guest stack, but always
finish before the next yield. After every yield the driver walks the
suspended frame-pointer chain and fails replay if any frame is not generated
code (x86-64 and aarch64). The driver accesses control blocks only through raw
pointers and holds no borrow of them, or of a fiber stack, across a resume.

Dropping the replay future, including after a decoding/divergence error,
frees activations in reverse order before releasing the store; nothing runs or
unwinds on their fibers. Replay yields to the async executor after a bounded
number of trace events. An activation that never reaches a boundary does not
yield; deterministic interruption is not implemented. Failed or cancelled
replay does not roll back guest state; retry with a fresh store.

A recording may end while guest calls are unfinished, for example with
component-model tasks suspended in host calls, or after a host panic. The
trace then ends with those activations parked at their host calls, and replay
reproduces execution up to that point.

## Checkpoints

`Replayer::checkpoint` captures the replay at a stop, and `Replayer::restore`
returns to it; see `rr/replay/checkpoint.rs`. A checkpoint holds the driver's
protocol state (trace position, activations and parked host calls, startup and
growth-failure state, object identities and instances), each activation's raw
fiber snapshot, control block, parked `VMStoreContext` state, protection-key
mask, and value buffer, and the guest state of every object: memory sizes and
contents, function table sizes and raw elements, and mutable globals
(including component instance flags).

Restoring puts all of this back in place. Fibers, control blocks, and buffers
keep their addresses, so completed activations are retained while a live
checkpoint can restore them. Memories and tables shrink back when restored;
discarded pages become inaccessible and read as zero once regrown. Objects
constructed after a checkpoint remain in the store but are unreachable after
restoring, and replaying forward constructs new ones, so each checkpoint keeps
its own timeline's object identities. Compiled modules are shared between
timelines, so breakpoints on them persist. Embedder events are delivered to
observers again as replay passes them. Frame handles become invalid.

Memory images are 4 KiB pages shared with the previous checkpoint's image of
the same memory, so a checkpoint retains only changed pages, and restoring
writes only pages that differ from the current contents. Changed pages are
still found by comparing memory; dirty-page tracking is future work.

To keep traces independent of the replaying engine's configuration, record/
replay engines give every module an unconditional startup function and treat
every function as escaping, so function identities and startup activations do
not depend on memory-initialization strategy or on guest debugging.

## Debugging on replay

A replaying engine may enable `Config::guest_debug`, and breakpoints and
single-stepping are configured through the store as usual
(`Replayer::store().edit_breakpoints()`). In record/replay engines the
breakpoint trampoline checks, after the breakpoint libcall returns, whether
that libcall requested a debug stop of the running replay activation, and if
so yields to the driver through the control block (`VM_REPLAY_DEBUG`). The
stopped stack therefore holds no host frames and can be checkpointed.
`Replayer::run` returns `ReplayStop::Breakpoint`; the next `run` resumes the
stopped activation before processing further trace events.

While stopped, `Replayer::debug_exit_frames` installs the parked activation's
`VMStoreContext` state and a `CallThreadState` just long enough to collect
frame handles, which are then inspected through `Replayer::store`. It returns
the stopped activation's exit frame followed by those of activations parked
at host calls, most recent first: for a callback beneath a host frame, its
guest callers. `ReplayStop::Event` (enabled by `Replayer::stop_at_events`)
stops after embedder events.

Debug events other than breakpoints and steps (traps, host errors,
exceptions) are not reported on replay, and the store's async
`DebugHandler` is not invoked. Recording with guest debugging is rejected,
since a debug handler would run unrecorded host code.

Restrictions:

* Core signatures may not contain GC or typed function references, and
  modules may not use GC, exceptions, shared memories, or stack switching.
  Component-level resources, futures, and streams cross the boundary as
  numeric core handles and are supported.
* Hosts may not mutate tables or globals; constructing numeric or abstract
  funcref globals and abstract funcref tables is supported.
* Resource limiters, call hooks, custom signal handlers, fuel, and epochs are
  rejected, as is recording with guest debugging. Installing a limiter, hook,
  or handler during a recording poisons the recording.
* Rust error objects are represented by their root cause's message.
* Replay requires a native compilation target: Pulley interprets guest code
  in Rust. Raw fibers require this crate's own stack switching, which Windows
  (OS fibers) and Miri lack, and AddressSanitizer's fiber handshake would
  require Rust code at every switch. These report an error rather than
  falling back to closure-based fibers.
* After a replay, the store can be inspected but not called: its host
  functions are replay stubs.

The remaining implementation work is:

1. Dirty-page tracking for checkpoints, instead of comparing memory, e.g.
   with `PAGEMAP_SCAN` or write protection.
2. Trap, host-error, and exception debug events on replay. Traps are raised
   from the synchronous `raise` libcall, which would need to yield a stop
   (with driver-owned payload storage) before unwinding.
3. Embedder integrations of `rr::record_event`, such as WASI output.
   Host bindings usually see only the store's data, not the store, so this may
   need an event sink that is flushed into the trace at the next boundary.
4. Ordinary (non-replay) asynchronous execution does not use raw fibers. To
   participate in snapshots, asynchronous host work would run on an owned
   child fiber whose Rust frames are never copied as guest snapshots.
5. Deterministic interruption, Windows and sanitizer support, verifying
   suspended stacks on architectures other than x86-64 and aarch64, and
   measuring append overhead and trace volume.

Tests are in `crates/wasmtime/tests/record_replay.rs` (record/replay
behavior, checkpoints, and debugging), `crates/fiber/src/raw.rs` (raw fiber
lifecycle and snapshots), and the `.wast` runner, which with `--features rr`
records the component-model async suites, replays them, and single-steps and
rewinds the replays:

```sh
cargo test -p wasmtime-internal-fiber
cargo test -p wasmtime --features rr --test record_replay
cargo test -p wasmtime --features rr,all-arch,pulley --test record_replay replay_
cargo test -p wasmtime --release --no-default-features \
  --features cranelift,runtime,std,rr,wat,component-model-async --test record_replay
cargo check -p wasmtime --no-default-features --features runtime,rr
cargo test --test wast --features rr -- component-model
```

The raw-fiber design and its rationale are described in the
[replay fiber plan](record-replay-fiber-plan.md).
