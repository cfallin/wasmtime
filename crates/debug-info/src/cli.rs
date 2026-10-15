//! Helpers shared by command-line debugger front-ends: the built-in
//! text-mode debugger and native clients such as `dapper`. Keeping these
//! here makes the front-ends accept the same syntax and produce the same
//! output.

use crate::demangle::demangle;
use crate::format::quote_bytes;
use anyhow::{Result, bail};
use std::fmt::Write;

/// The command reference.
pub const HELP: &str = "\
Execution:
  start                    Run to the beginning of `main`.
  continue (c), run (r)    Continue execution.
  step (s)                 Step one source line, entering calls.
  next (n)                 Step one source line, stepping over calls.
  finish (fin)             Run until the selected frame returns.
  stepi (si)               Step one Wasm instruction.
  nexti (ni)               Step one Wasm instruction, stepping over calls.
  interrupt, Ctrl-C        Stop a running program.
Breakpoints:
  break (b) LOC            Set a breakpoint. LOC is FILE:LINE, LINE, FUNCTION,
                           *PC (module offset in the current module), or
                           *MODULE:PC.
  tbreak LOC               Set a temporary (one-shot) breakpoint.
  delete (d) [N...]        Delete breakpoints (all if none given).
  info breakpoints         List breakpoints.
Stack:
  backtrace (bt)           Show the call stack (including inlined frames).
  frame (f) [N]            Select / show a frame.
  up [N], down [N]         Move up (toward callers) / down the stack.
Source and data:
  list (l) [LOC]           List source lines.
  print (p)[/x|/r] PATH    Print a value. PATH is an access path: `x`, `p->x`,
                           `arr[2].y`, `*p`, `&x`, `t.0`, `%1->next`, `%`, `%%2`.
                           /x prints integers in hex; /r disables pretty-printers.
                           Wasm-level roots: $NAME (a Wasm local, global or
                           table named in the name section), and built-ins
                           %localN, %stackN, %globalN, %tableN[I], %pc.
                           GC refs: `.N` (field), `[I]`.
  ptype PATH, whatis PATH  Show the type of a value.
  location (loc) PATH      Show where a value is stored: `memory ADDR length N`,
                           `local N`, `global N` or `stack N`. `&PATH` gives
                           the address of a value in memory, e.g. for
                           `x/16xb &buf`.
  info locals|args         Show variables of the selected frame.
  info frame               Describe the selected frame.
  info functions [TEXT]    List functions (with debug info) matching TEXT.
  info variables [TEXT]    List global variables matching TEXT.
  info source              Show the current source file.
  x/NFU ADDR|PATH          Examine linear memory: N count, F format
                           (x,d,u,c,s), U unit size (b,h,w,g).
Wasm machine level:
  disassemble (disas) [FUNC|*PC] [all]
                           Disassemble around the PC (or a whole function).
  wasm-locals              Wasm locals of the selected frame.
  wasm-stack               Wasm operand stack of the selected frame.
  wasm-globals [-i INST]   Globals of an instance.
  wasm-memory [-i INST] [-m MEM] [ADDR [LEN]]
                           Show a memory's size, or hexdump its contents.
  wasm-table [-i INST] [-t TABLE] [START [COUNT]]
                           Show a table's size and elements.
  wasm-instances           List all instances in the store.
  wasm-modules             List all modules in the store.
  wasm-functions [-i INST | -m MODULE] [TEXT]
                           List functions of a module.
Other:
  help                     Show this help.
  quit (q), detach         Detach (the program runs to completion).
";

/// The final path component of a file path.
pub fn basename(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// Truncate a string to `n` characters, adding `...` if truncated.
pub fn truncate(mut s: String, n: usize) -> String {
    if s.chars().count() > n {
        s = s.chars().take(n).collect::<String>() + "...";
    }
    s
}

/// Parse a decimal or `0x`-prefixed hexadecimal integer.
pub fn parse_u64(s: &str) -> Result<u64> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Ok(u64::from_str_radix(h, 16)?)
    } else {
        Ok(s.parse()?)
    }
}

/// Demangle `$`-prefixed mangled symbol names in a line of WAT.
pub fn demangle_wat(line: &str) -> String {
    line.split(' ')
        .map(|tok| match tok.strip_prefix('$') {
            Some(sym) if sym.starts_with("_Z") || sym.starts_with("_R") => {
                format!("<{}>", demangle(sym))
            }
            _ => tok.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A breakpoint location as written by the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BreakpointSpec {
    /// `*PC` or `*MODULE:PC`: a Wasm bytecode offset.
    Pc { module: Option<usize>, pc: u32 },
    /// `FILE:LINE` or `LINE` (in the current file).
    Line { file: Option<String>, line: u32 },
    /// A function name.
    Function(String),
}

impl BreakpointSpec {
    pub fn parse(spec: &str) -> Result<BreakpointSpec> {
        let spec = spec.trim();
        if spec.is_empty() {
            bail!("Argument required (location).");
        }
        if let Some(rest) = spec.strip_prefix('*') {
            return Ok(match rest.split_once(':') {
                Some((m, pc)) => BreakpointSpec::Pc {
                    module: Some(usize::try_from(parse_u64(m)?)?),
                    pc: u32::try_from(parse_u64(pc)?)?,
                },
                None => BreakpointSpec::Pc {
                    module: None,
                    pc: u32::try_from(parse_u64(rest)?)?,
                },
            });
        }
        if let Some((f, l)) = spec.rsplit_once(':') {
            if !l.is_empty() && l.chars().all(|c| c.is_ascii_digit()) {
                return Ok(BreakpointSpec::Line {
                    file: Some(f.to_string()),
                    line: l.parse()?,
                });
            }
        }
        if spec.chars().all(|c| c.is_ascii_digit()) {
            return Ok(BreakpointSpec::Line {
                file: None,
                line: spec.parse()?,
            });
        }
        Ok(BreakpointSpec::Function(spec.to_string()))
    }
}

/// An `x/NFU` memory-examination format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExamineFormat {
    /// Number of units (or strings, for `s`).
    pub count: u64,
    /// `x`, `d`, `u`, `c` or `s`.
    pub format: char,
    /// Unit size in bytes.
    pub unit: u64,
}

impl ExamineFormat {
    /// Parse the part after `x/`, e.g. `16xb`.
    pub fn parse(fmt: &str) -> Result<ExamineFormat> {
        let digits: String = fmt.chars().take_while(|c| c.is_ascii_digit()).collect();
        let count: u64 = if digits.is_empty() {
            1
        } else {
            digits.parse()?
        };
        let mut format = 'x';
        let mut unit = 0u64;
        for c in fmt[digits.len()..].chars() {
            match c {
                'x' | 'd' | 'u' | 'c' | 's' => format = c,
                'b' => unit = 1,
                'h' => unit = 2,
                'w' => unit = 4,
                'g' => unit = 8,
                _ => bail!("Invalid format letter '{c}'."),
            }
        }
        if unit == 0 {
            unit = if format == 'c' { 1 } else { 4 };
        }
        Ok(ExamineFormat {
            count,
            format,
            unit,
        })
    }

    /// The number of bytes to read (for non-string formats).
    pub fn byte_len(&self) -> u64 {
        self.count * self.unit
    }

    /// Format memory contents read at `addr` (non-string formats).
    pub fn format(&self, addr: u64, bytes: &[u8]) -> String {
        let unit = self.unit.max(1);
        let mut out = String::new();
        let per_line = (16 / unit).max(1);
        for (i, chunk) in bytes.chunks(unit as usize).enumerate() {
            let i = i as u64;
            if i % per_line == 0 {
                if i > 0 {
                    out.push('\n');
                }
                let _ = write!(out, "{:#010x}:", addr + i * unit);
            }
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            let v = u64::from_le_bytes(b);
            let w = (unit * 2) as usize;
            let _ = match self.format {
                'u' => write!(out, " {v}"),
                'd' => {
                    let shift = 64 - unit * 8;
                    write!(out, " {}", ((v << shift) as i64) >> shift)
                }
                'c' => write!(out, " {:?}", (v as u8) as char),
                _ => write!(out, " 0x{v:0w$x}"),
            };
        }
        out.push('\n');
        out
    }
}

/// Format one `x/s` line: a NUL-terminated string read at `addr`.
pub fn format_examine_string(addr: u64, bytes: &[u8]) -> String {
    format!("{addr:#x}: {}\n", quote_bytes(bytes, false))
}

/// A classic 16-bytes-per-line hex dump with an ASCII column.
pub fn hexdump(addr: u64, bytes: &[u8]) -> String {
    let mut out = String::new();
    for (i, chunk) in bytes.chunks(16).enumerate() {
        let _ = write!(out, "{:#010x}:", addr + 16 * i as u64);
        for j in 0..16 {
            match chunk.get(j) {
                Some(b) => {
                    let _ = write!(out, " {b:02x}");
                }
                None => out.push_str("   "),
            }
        }
        out.push_str("  |");
        for b in chunk {
            out.push(if (0x20..0x7f).contains(b) {
                *b as char
            } else {
                '.'
            });
        }
        out.push_str("|\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        assert_eq!(
            BreakpointSpec::parse("main.c:12").unwrap(),
            BreakpointSpec::Line {
                file: Some("main.c".into()),
                line: 12
            }
        );
        assert_eq!(
            BreakpointSpec::parse("12").unwrap(),
            BreakpointSpec::Line {
                file: None,
                line: 12
            }
        );
        assert_eq!(
            BreakpointSpec::parse("*1:0x20").unwrap(),
            BreakpointSpec::Pc {
                module: Some(1),
                pc: 0x20
            }
        );
        assert_eq!(
            BreakpointSpec::parse("a::b").unwrap(),
            BreakpointSpec::Function("a::b".into())
        );
    }

    #[test]
    fn examine() {
        let f = ExamineFormat::parse("4xb").unwrap();
        assert_eq!(
            f.format(0x10, &[1, 2, 3, 0xff]),
            "0x00000010: 0x01 0x02 0x03 0xff\n"
        );
        let f = ExamineFormat::parse("2dw").unwrap();
        assert_eq!(
            f.format(0, &[0xff, 0xff, 0xff, 0xff, 5, 0, 0, 0]),
            "0x00000000: -1 5\n"
        );
    }
}
