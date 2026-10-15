//! Abstraction over a paused debuggee, as seen from one stack frame.

use anyhow::Result;

/// A Wasm-level scalar value, as found in a local, global, or on the
/// operand stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WasmVal {
    I32(u32),
    I64(u64),
    F32(u32),
    F64(u64),
    V128(u128),
}

impl WasmVal {
    /// Interpret this value as an unsigned integer (zero-extended, or
    /// raw bits for floats), as used by DWARF expression evaluation.
    pub fn as_u64(&self) -> u64 {
        match *self {
            WasmVal::I32(x) | WasmVal::F32(x) => u64::from(x),
            WasmVal::I64(x) | WasmVal::F64(x) => x,
            WasmVal::V128(x) => x as u64,
        }
    }

    /// The little-endian byte representation of this value.
    pub fn to_bytes(&self) -> Vec<u8> {
        match *self {
            WasmVal::I32(x) | WasmVal::F32(x) => x.to_le_bytes().to_vec(),
            WasmVal::I64(x) | WasmVal::F64(x) => x.to_le_bytes().to_vec(),
            WasmVal::V128(x) => x.to_le_bytes().to_vec(),
        }
    }
}

/// Access to debuggee state needed to evaluate DWARF locations and read
/// values: linear memory plus the Wasm locals, globals and operand stack
/// of one particular frame.
pub trait Target {
    /// Read `len` bytes of linear memory `memory` at `addr`.
    fn read_memory(&self, memory: u32, addr: u64, len: u64) -> Result<Vec<u8>>;

    /// Get the value of Wasm local `index` in the frame.
    fn local(&self, index: u32) -> Result<WasmVal>;

    /// Get the value of Wasm global `index` in the frame's instance.
    fn global(&self, index: u32) -> Result<WasmVal>;

    /// Get the value at operand-stack slot `index` (counted from the
    /// bottom of the frame's operand stack).
    fn operand(&self, index: u32) -> Result<WasmVal>;
}

/// A `Target` with no state at all; every access fails. Useful for
/// formatting values whose bytes are already fully captured.
pub struct NoTarget;

impl Target for NoTarget {
    fn read_memory(&self, _memory: u32, addr: u64, _len: u64) -> Result<Vec<u8>> {
        anyhow::bail!("cannot read memory at {addr:#x}: no debuggee")
    }
    fn local(&self, index: u32) -> Result<WasmVal> {
        anyhow::bail!("cannot read local {index}: no frame")
    }
    fn global(&self, index: u32) -> Result<WasmVal> {
        anyhow::bail!("cannot read global {index}: no frame")
    }
    fn operand(&self, index: u32) -> Result<WasmVal> {
        anyhow::bail!("cannot read operand {index}: no frame")
    }
}
