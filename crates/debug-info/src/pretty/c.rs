//! Pretty-printers for C (and C-family) strings.

use super::*;
use crate::format::quote_bytes;

pub(super) static PRINTERS: &[&dyn PrettyPrinter] = &[&CString, &CCharArray];

fn char_like(ctx: &Ctx<'_>, ty: Option<TypeId>) -> bool {
    let Some(ty) = ty else { return false };
    matches!(
        ctx.info.ty(ctx.info.strip(ty)).kind,
        TypeKind::Base(BaseEnc::SignedChar | BaseEnc::UnsignedChar)
    )
}

/// `char *` / `const char *` (and signed/unsigned variants): show the
/// pointer and the NUL-terminated string it points to.
struct CString;
impl PrettyPrinter for CString {
    fn name(&self) -> &'static str {
        "c-string"
    }
    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, _name: &str) -> bool {
        ty.lang.is_c_like()
            && match ty.kind {
                TypeKind::Pointer {
                    pointee,
                    kind: PtrKind::Pointer,
                } => char_like(ctx, pointee),
                _ => false,
            }
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let addr = ctx.read_uint(v)?;
        if addr == 0 {
            return Ok(Summary::Text("0x0".into()));
        }
        let mut bytes = vec![];
        let mut truncated = true;
        let mut cur = addr;
        'outer: while bytes.len() < opts.max_string {
            let chunk = match ctx.target.read_memory(0, cur, 64) {
                Ok(c) => c,
                Err(_) => {
                    // Possibly near the end of memory: fall back to
                    // byte-at-a-time reads.
                    match ctx.target.read_memory(0, cur, 1) {
                        Ok(c) => c,
                        Err(_) => {
                            return Ok(Summary::Text(format!("{addr:#x} <error reading string>")));
                        }
                    }
                }
            };
            for b in &chunk {
                if *b == 0 {
                    truncated = false;
                    break 'outer;
                }
                bytes.push(*b);
                if bytes.len() >= opts.max_string {
                    break 'outer;
                }
            }
            cur += chunk.len() as u64;
        }
        let mut s = quote_bytes(&bytes, false);
        if truncated {
            s.push_str("...");
        }
        Ok(Summary::Text(format!("{addr:#x} {s}")))
    }
}

/// `char[N]`: show as a string up to the first NUL.
struct CCharArray;
impl PrettyPrinter for CCharArray {
    fn name(&self) -> &'static str {
        "c-char-array"
    }
    fn matches(&self, ctx: &Ctx<'_>, ty: &Type, _name: &str) -> bool {
        ty.lang.is_c_like()
            && match ty.kind {
                TypeKind::Array { elem, len: Some(_) } => char_like(ctx, Some(elem)),
                _ => false,
            }
    }
    fn summarize(&self, ctx: &Ctx<'_>, v: &Value, opts: &FormatOptions) -> Result<Summary> {
        let bytes = ctx.contents(v)?;
        let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
        let n = end.min(opts.max_string);
        let mut s = quote_bytes(&bytes[..n], false);
        if n < end {
            s.push_str("...");
        }
        Ok(Summary::Text(s))
    }
}
