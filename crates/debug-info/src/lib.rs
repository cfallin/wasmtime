//! Source-level and Wasm-level program understanding for debugger
//! components.
//!
//! This crate is deliberately independent of any particular debugger
//! transport or host API: it consumes raw Wasm module bytes (including
//! DWARF custom sections) and talks to the paused debuggee only through
//! the small [`Target`] trait. That makes it usable both from inside a
//! Wasmtime debugger component (where `Target` is implemented over the
//! `wasmtime:debugger` WIT API) and from native unit tests.
//!
//! Major pieces:
//!
//! - [`module`]: Wasm module structure (function names, body ranges,
//!   disassembly, instruction boundaries).
//! - [`dwarf`]: DWARF line tables, functions, inline frames, scopes and
//!   variables.
//! - [`types`]: a language-agnostic model of DWARF types, including Rust
//!   enums (variant parts).
//! - [`eval`]: DWARF location-expression evaluation with the Wasm
//!   extensions (`DW_OP_WASM_location`).
//! - [`value`] and [`path`]: values and simple "access path"
//!   expressions (`p->x`, `arr[2].y`, `*p`, `$1.field`).
//! - [`format`] and [`pretty`]: value formatting and a small set of
//!   pretty-printers for well-known standard-library types.
//! - [`demangle`]: Rust and C++ symbol demangling.

pub mod demangle;
pub mod dwarf;
pub mod eval;
pub mod format;
pub mod module;
pub mod path;
pub mod pretty;
pub mod target;
pub mod types;
pub mod value;

pub use dwarf::DebugInfo;
pub use module::ModuleInfo;
pub use target::{Target, WasmVal};
