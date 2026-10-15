//! Storage locations of values: where an access path's value lives in
//! the Wasm machine. Used to describe values to the user and, in the
//! future, to set watchpoints.

use std::fmt;

/// A Wasm-VM storage slot outside linear memory that can hold (part of)
/// a source-level value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmSlot {
    Local(u32),
    Global(u32),
    Stack(u32),
}

/// Where a value lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    /// `size` bytes of linear memory (memory 0) at `base`.
    Mem { base: u64, size: u64 },
    /// Bytes `offset..offset + size` of Wasm local `index` (of the
    /// frame in which the path was evaluated).
    Local { index: u32, offset: u64, size: u64 },
    /// Bytes `offset..offset + size` of Wasm global `index`.
    Global { index: u32, offset: u64, size: u64 },
    /// Bytes `offset..offset + size` of operand-stack slot `index`.
    Stack { index: u32, offset: u64, size: u64 },
}

impl Location {
    /// A location for a sub-range of a Wasm slot.
    pub fn in_slot(slot: WasmSlot, offset: u64, size: u64) -> Location {
        match slot {
            WasmSlot::Local(index) => Location::Local {
                index,
                offset,
                size,
            },
            WasmSlot::Global(index) => Location::Global {
                index,
                offset,
                size,
            },
            WasmSlot::Stack(index) => Location::Stack {
                index,
                offset,
                size,
            },
        }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (what, index, offset, size) = match *self {
            Location::Mem { base, size } => {
                return write!(f, "memory {base:#x} length {size:#x}");
            }
            Location::Local {
                index,
                offset,
                size,
            } => ("local", index, offset, size),
            Location::Global {
                index,
                offset,
                size,
            } => ("global", index, offset, size),
            Location::Stack {
                index,
                offset,
                size,
            } => ("stack", index, offset, size),
        };
        write!(f, "{what} {index}")?;
        if offset != 0 {
            write!(f, " (bytes {offset}..{})", offset + size)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        assert_eq!(
            Location::Mem {
                base: 0x1234,
                size: 8
            }
            .to_string(),
            "memory 0x1234 length 0x8"
        );
        assert_eq!(
            Location::in_slot(WasmSlot::Local(3), 0, 4).to_string(),
            "local 3"
        );
        assert_eq!(
            Location::in_slot(WasmSlot::Global(1), 4, 4).to_string(),
            "global 1 (bytes 4..8)"
        );
    }
}
