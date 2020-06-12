//! Support for reference types (GC pointers) and safepoints for precise GC.
//!
//! Some values in the program dataflow are reference-typed (or reftyped for
//! short). These values have types `R32` or `R64`.
//!
//! To allow for precise GC, the codegen inserts "safepoints" at lowering. Every
//! instruction conceptually has a bit indicating whether it is a safepoint. If
//! an instruction is a safepoint, then no reftyped value may be live across the
//! instruction in a register. Instead, each such value must pass through memory
//! in a stack storage sloe. At emission time, the backend provides metadata to
//! the `CodeSink` that includes "stackmaps" at each safepoint. These stackmaps
//! provide a *precise* indication of which stack slots have reftyped values; no
//! reftyped value is missing (or else GC might miss a root) and no slot is
//! marked as reftyped that is not actually so (or else precise GC, and
//! especially moving/compacting GC, would be impossible).
//!
//! The basic idea here is to do some analysis before lowering from CLIF to
//! VCode, and then use these analysis results during lowering to insert the
//! necessary stores and loads (which are like spills and reloads, but not
//! managed by the register allocator). In essence, we implement the following
//! dataflow algorithm:
//!
//! - At every program point, for every reftyped vreg, we track whether the
//!   vreg is "in register" or "on stack" or "uninit". These states form a
//!   lattice:
//!   
//!        Dead (undefined)
//!             |
//!           On stack
