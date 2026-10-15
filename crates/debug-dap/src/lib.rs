//! The Debug Adapter Protocol (DAP): message framing, message
//! construction, the subset of protocol types used by Wasmtime's debugger,
//! and a simple blocking client.
//!
//! DAP messages are JSON objects framed with HTTP-style headers
//! (`Content-Length: N\r\n\r\n` followed by `N` bytes of JSON). See
//! <https://microsoft.github.io/debug-adapter-protocol/specification>.
//!
//! The framing ([`Decoder`], [`encode`]) is sans-I/O so that it can be
//! used both by the debugger component (with async WASI sockets) and by
//! native clients ([`Client`]).

mod client;
mod codec;
pub mod types;

pub use client::Client;
pub use codec::{Decoder, encode};

use serde_json::{Value, json};

/// Build a request message.
pub fn request(seq: i64, command: &str, arguments: Value) -> Value {
    json!({
        "seq": seq,
        "type": "request",
        "command": command,
        "arguments": arguments,
    })
}

/// Build a response message.
pub fn response(seq: i64, request_seq: i64, command: &str, result: Result<Value, String>) -> Value {
    match result {
        Ok(body) => json!({
            "seq": seq,
            "type": "response",
            "request_seq": request_seq,
            "command": command,
            "success": true,
            "body": body,
        }),
        Err(message) => json!({
            "seq": seq,
            "type": "response",
            "request_seq": request_seq,
            "command": command,
            "success": false,
            "message": message,
            "body": { "error": { "id": 1, "format": message } },
        }),
    }
}

/// Build an event message.
pub fn event(seq: i64, event: &str, body: Value) -> Value {
    json!({
        "seq": seq,
        "type": "event",
        "event": event,
        "body": body,
    })
}

/// Encode a linear-memory or code location as a DAP memory reference (a
/// hexadecimal number, as clients such as VS Code expect).
pub fn memory_reference(addr: u64) -> String {
    format!("{addr:#x}")
}

/// Parse a DAP memory reference produced by [`memory_reference`].
pub fn parse_memory_reference(r: &str) -> Option<u64> {
    let r = r.trim();
    match r.strip_prefix("0x").or_else(|| r.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => r.parse().ok(),
    }
}

/// Code addresses in instruction references and disassembly combine a
/// module index (in the debugger's numbering) with an offset in that
/// module's bytecode.
pub fn code_address(module: usize, pc: u32) -> u64 {
    ((module as u64) << 32) | u64::from(pc)
}

/// Split a code address into (module index, pc).
pub fn split_code_address(addr: u64) -> (usize, u32) {
    ((addr >> 32) as usize, addr as u32)
}

/// Standard base64 encoding (as used by DAP `readMemory`).
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Standard base64 decoding.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let bytes: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    let mut out = vec![];
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        let mut valid: u32 = 0;
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                break;
            }
            n |= val(c)? << (18 - 6 * i);
            valid += 1;
        }
        for i in 0..valid.saturating_sub(1) {
            out.push((n >> (16 - 8 * i)) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64() {
        for data in [&b""[..], b"a", b"ab", b"abc", b"abcd", b"hello, world"] {
            let e = base64_encode(data);
            assert_eq!(base64_decode(&e).unwrap(), data);
        }
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
    }

    #[test]
    fn code_addresses() {
        let a = code_address(3, 0x1234);
        assert_eq!(split_code_address(a), (3, 0x1234));
        assert_eq!(parse_memory_reference(&memory_reference(a)), Some(a));
    }
}
