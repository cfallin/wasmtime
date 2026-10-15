//! Evaluation of DWARF location expressions, including the Wasm
//! extension operator `DW_OP_WASM_location`.
//!
//! gimli parses `DW_OP_WASM_location` but its evaluator does not support
//! it, so we implement the (small) subset of DWARF expressions that
//! compilers actually emit for Wasm targets here.
//!
//! Semantics of `DW_OP_WASM_location` follow LLVM/LLDB: when it is the
//! entire location (or an entire piece), it denotes a register-like
//! location (the variable lives in that Wasm local/global/operand);
//! otherwise it pushes the current value of that Wasm entity onto the
//! expression stack.

use crate::dwarf::R;
use crate::target::Target;
use anyhow::{Result, bail};
use gimli::{Encoding, Expression, Operation, Reader};

/// The result of evaluating a location expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Location {
    /// The object lives in linear memory (memory 0) at this address.
    Memory(u64),
    /// The object has no location, but its value is known (bytes in
    /// little-endian order; may be longer than the object).
    Value(Vec<u8>),
    /// The object lives in a Wasm local of the current frame.
    WasmLocal(u32),
    /// The object lives in a Wasm global.
    WasmGlobal(u32),
    /// The object lives in a Wasm operand-stack slot.
    WasmStack(u32),
    /// The object is split into pieces.
    Pieces(Vec<Piece>),
    /// The object's location is unavailable (with a reason).
    Unavailable(String),
}

/// One piece of a composite location.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub loc: Location,
    pub size_bits: u64,
}

/// Evaluate `expr`.
pub fn evaluate(
    expr: &Expression<R>,
    encoding: Encoding,
    frame_base: Option<u64>,
    target: &dyn Target,
) -> Result<Location> {
    // Parse all operations up front, recording byte offsets for
    // branches.
    let total = expr.0.len();
    let mut ops = vec![];
    let mut reader = expr.0.clone();
    while !reader.is_empty() {
        let start = total - reader.len();
        let op = Operation::parse(&mut reader, encoding)?;
        let end = total - reader.len();
        ops.push((start, end, op));
    }

    let addr_mask = if encoding.address_size >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * u32::from(encoding.address_size))) - 1
    };

    let mut stack: Vec<u64> = vec![];
    let mut pieces: Vec<Piece> = vec![];
    // A register-like location established by the last operation.
    let mut reg_loc: Option<Location> = None;
    let mut stack_value = false;
    let mut implicit: Option<Vec<u8>> = None;

    macro_rules! pop {
        () => {
            match stack.pop() {
                Some(v) => v,
                None => bail!("DWARF expression stack underflow"),
            }
        };
    }
    macro_rules! binop {
        ($f:expr) => {{
            let b = pop!();
            let a = pop!();
            #[allow(clippy::redundant_closure_call)]
            stack.push(($f)(a, b));
        }};
    }

    let mut i = 0;
    while i < ops.len() {
        let (_, end, ref op) = ops[i];
        let next_is_end_or_piece =
            i + 1 == ops.len() || matches!(ops[i + 1].2, Operation::Piece { .. });
        i += 1;
        match *op {
            Operation::WasmLocal { index } => {
                if next_is_end_or_piece {
                    reg_loc = Some(Location::WasmLocal(index));
                } else {
                    stack.push(target.local(index)?.as_u64());
                }
            }
            Operation::WasmGlobal { index } => {
                if next_is_end_or_piece {
                    reg_loc = Some(Location::WasmGlobal(index));
                } else {
                    stack.push(target.global(index)?.as_u64());
                }
            }
            Operation::WasmStack { index } => {
                if next_is_end_or_piece {
                    reg_loc = Some(Location::WasmStack(index));
                } else {
                    stack.push(target.operand(index)?.as_u64());
                }
            }
            Operation::Address { address } => stack.push(address),
            Operation::UnsignedConstant { value } => stack.push(value),
            Operation::SignedConstant { value } => stack.push(value as u64),
            Operation::FrameOffset { offset } => {
                let Some(fb) = frame_base else {
                    bail!("DW_OP_fbreg without a frame base");
                };
                stack.push(fb.wrapping_add(offset as u64) & addr_mask);
            }
            Operation::PlusConstant { value } => {
                let a = pop!();
                stack.push(a.wrapping_add(value));
            }
            Operation::Plus => binop!(|a: u64, b: u64| a.wrapping_add(b)),
            Operation::Minus => binop!(|a: u64, b: u64| a.wrapping_sub(b)),
            Operation::Mul => binop!(|a: u64, b: u64| a.wrapping_mul(b)),
            Operation::Div => {
                let b = pop!() as i64;
                let a = pop!() as i64;
                if b == 0 {
                    bail!("division by zero in DWARF expression");
                }
                stack.push(a.wrapping_div(b) as u64);
            }
            Operation::Mod => {
                let b = pop!();
                let a = pop!();
                if b == 0 {
                    bail!("division by zero in DWARF expression");
                }
                stack.push(a % b);
            }
            Operation::And => binop!(|a: u64, b: u64| a & b),
            Operation::Or => binop!(|a: u64, b: u64| a | b),
            Operation::Xor => binop!(|a: u64, b: u64| a ^ b),
            Operation::Shl => binop!(|a: u64, b: u64| a.checked_shl(b as u32).unwrap_or(0)),
            Operation::Shr => binop!(|a: u64, b: u64| a.checked_shr(b as u32).unwrap_or(0)),
            Operation::Shra => binop!(|a: u64, b: u64| ((a as i64) >> (b.min(63) as u32)) as u64),
            Operation::Eq => binop!(|a: u64, b: u64| u64::from(a == b)),
            Operation::Ne => binop!(|a: u64, b: u64| u64::from(a != b)),
            Operation::Lt => binop!(|a: u64, b: u64| u64::from((a as i64) < (b as i64))),
            Operation::Le => binop!(|a: u64, b: u64| u64::from((a as i64) <= (b as i64))),
            Operation::Gt => binop!(|a: u64, b: u64| u64::from((a as i64) > (b as i64))),
            Operation::Ge => binop!(|a: u64, b: u64| u64::from((a as i64) >= (b as i64))),
            Operation::Neg => {
                let a = pop!();
                stack.push((a as i64).wrapping_neg() as u64);
            }
            Operation::Not => {
                let a = pop!();
                stack.push(!a);
            }
            Operation::Abs => {
                let a = pop!();
                stack.push((a as i64).unsigned_abs());
            }
            Operation::Drop => {
                pop!();
            }
            Operation::Pick { index } => {
                let idx = usize::from(index);
                if idx >= stack.len() {
                    bail!("DW_OP_pick out of range");
                }
                stack.push(stack[stack.len() - 1 - idx]);
            }
            Operation::Swap => {
                let b = pop!();
                let a = pop!();
                stack.push(b);
                stack.push(a);
            }
            Operation::Rot => {
                let c = pop!();
                let b = pop!();
                let a = pop!();
                stack.push(c);
                stack.push(a);
                stack.push(b);
            }
            Operation::Deref { size, space, .. } => {
                if space {
                    bail!("DW_OP_xderef is not supported");
                }
                let addr = pop!() & addr_mask;
                let bytes = target.read_memory(0, addr, u64::from(size))?;
                let mut b = [0u8; 8];
                let n = bytes.len().min(8);
                b[..n].copy_from_slice(&bytes[..n]);
                stack.push(u64::from_le_bytes(b));
            }
            Operation::Skip { target: t } => {
                i = branch(&ops, end, t)?;
            }
            Operation::Bra { target: t } => {
                if pop!() != 0 {
                    i = branch(&ops, end, t)?;
                }
            }
            Operation::Nop => {}
            Operation::StackValue => stack_value = true,
            Operation::ImplicitValue { ref data } => {
                implicit = Some(data.to_slice()?.into_owned());
            }
            Operation::Piece {
                size_in_bits,
                bit_offset,
            } => {
                if bit_offset.is_some_and(|b| b != 0) {
                    bail!("bit-offset pieces are not supported");
                }
                let loc = finish(
                    &mut stack,
                    &mut reg_loc,
                    &mut stack_value,
                    &mut implicit,
                    addr_mask,
                );
                pieces.push(Piece {
                    loc,
                    size_bits: size_in_bits,
                });
            }
            Operation::EntryValue { .. } | Operation::ParameterRef { .. } => {
                return Ok(Location::Unavailable("optimized out (entry value)".into()));
            }
            ref other => bail!("unsupported DWARF operation: {other:?}"),
        }
    }

    if !pieces.is_empty() {
        return Ok(Location::Pieces(pieces));
    }
    Ok(finish(
        &mut stack,
        &mut reg_loc,
        &mut stack_value,
        &mut implicit,
        addr_mask,
    ))
}

fn branch(ops: &[(usize, usize, Operation<R>)], end: usize, target: i16) -> Result<usize> {
    let dest = end as i64 + i64::from(target);
    if dest < 0 {
        bail!("DWARF branch out of range");
    }
    let dest = dest as usize;
    match ops.iter().position(|(start, _, _)| *start == dest) {
        Some(i) => Ok(i),
        None if ops.last().is_some_and(|(_, e, _)| *e == dest) => Ok(ops.len()),
        None => bail!("DWARF branch to invalid offset"),
    }
}

fn finish(
    stack: &mut Vec<u64>,
    reg_loc: &mut Option<Location>,
    stack_value: &mut bool,
    implicit: &mut Option<Vec<u8>>,
    addr_mask: u64,
) -> Location {
    let loc = if let Some(r) = reg_loc.take() {
        r
    } else if let Some(v) = implicit.take() {
        Location::Value(v)
    } else if *stack_value {
        match stack.last() {
            Some(v) => Location::Value(v.to_le_bytes().to_vec()),
            None => Location::Unavailable("empty expression".into()),
        }
    } else {
        match stack.last() {
            Some(v) => Location::Memory(v & addr_mask),
            None => Location::Unavailable("optimized out".into()),
        }
    };
    stack.clear();
    *stack_value = false;
    loc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::WasmVal;
    use std::sync::Arc;

    struct T;
    impl Target for T {
        fn read_memory(&self, _m: u32, addr: u64, len: u64) -> Result<Vec<u8>> {
            Ok((0..len).map(|i| (addr + i) as u8).collect())
        }
        fn local(&self, index: u32) -> Result<WasmVal> {
            Ok(WasmVal::I32(1000 + index))
        }
        fn global(&self, index: u32) -> Result<WasmVal> {
            Ok(WasmVal::I32(2000 + index))
        }
        fn operand(&self, index: u32) -> Result<WasmVal> {
            Ok(WasmVal::I32(3000 + index))
        }
    }

    fn eval(bytes: &[u8], fb: Option<u64>) -> Location {
        let expr = Expression(R::new(Arc::from(bytes), gimli::LittleEndian));
        let enc = Encoding {
            format: gimli::Format::Dwarf32,
            version: 4,
            address_size: 4,
        };
        evaluate(&expr, enc, fb, &T).unwrap()
    }

    #[test]
    fn wasm_local_register() {
        // DW_OP_WASM_location 0x0 0x3
        assert_eq!(eval(&[0xed, 0x00, 0x03], None), Location::WasmLocal(3));
    }

    #[test]
    fn wasm_local_stack_value() {
        // DW_OP_WASM_location 0x0 0x3, DW_OP_stack_value
        assert_eq!(
            eval(&[0xed, 0x00, 0x03, 0x9f], None),
            Location::Value(1003u64.to_le_bytes().to_vec())
        );
    }

    #[test]
    fn global_plus_addr() {
        // DW_OP_WASM_location 0x3 0x1 (u32 global index), DW_OP_addr 0x10, DW_OP_plus
        assert_eq!(
            eval(
                &[0xed, 0x03, 0x01, 0x00, 0x00, 0x00, 0x03, 0x10, 0, 0, 0, 0x22],
                None
            ),
            Location::Memory(2001 + 0x10)
        );
    }

    #[test]
    fn fbreg() {
        // DW_OP_fbreg +40
        assert_eq!(eval(&[0x91, 40], Some(1000)), Location::Memory(1040));
    }

    #[test]
    fn pieces() {
        // DW_OP_WASM_location 0 1, DW_OP_piece 4, DW_OP_WASM_location 0 2, DW_OP_piece 4
        assert_eq!(
            eval(&[0xed, 0, 1, 0x93, 4, 0xed, 0, 2, 0x93, 4], None),
            Location::Pieces(vec![
                Piece {
                    loc: Location::WasmLocal(1),
                    size_bits: 32
                },
                Piece {
                    loc: Location::WasmLocal(2),
                    size_bits: 32
                },
            ])
        );
    }
}
