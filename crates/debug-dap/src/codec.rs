//! DAP base-protocol framing.

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

/// Encode a message with its `Content-Length` header.
pub fn encode(msg: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(msg).expect("JSON values always serialize");
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend(body);
    out
}

/// An incremental decoder for framed DAP messages.
#[derive(Default)]
pub struct Decoder {
    buf: Vec<u8>,
}

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add received bytes.
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Decode the next complete message, if one has been received.
    pub fn next(&mut self) -> Result<Option<Value>> {
        let Some(end) = self.buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            if self.buf.len() > 64 * 1024 {
                bail!("DAP header too long");
            }
            return Ok(None);
        };
        let header = std::str::from_utf8(&self.buf[..end])?;
        let mut len = None;
        for line in header.split("\r\n") {
            if let Some((k, v)) = line.split_once(':') {
                if k.trim().eq_ignore_ascii_case("content-length") {
                    len = Some(v.trim().parse::<usize>()?);
                }
            }
        }
        let len = len.ok_or_else(|| anyhow!("DAP message without Content-Length"))?;
        let start = end + 4;
        if self.buf.len() < start + len {
            return Ok(None);
        }
        let msg = serde_json::from_slice(&self.buf[start..start + len])?;
        self.buf.drain(..start + len);
        Ok(Some(msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_split() {
        let a = json!({"seq": 1, "type": "request", "command": "initialize"});
        let b = json!({"seq": 2, "type": "event", "event": "stopped", "body": {"x": "é"}});
        let mut bytes = encode(&a);
        bytes.extend(encode(&b));
        let mut d = Decoder::new();
        let mut out = vec![];
        // Feed one byte at a time.
        for byte in bytes {
            d.push(&[byte]);
            while let Some(m) = d.next().unwrap() {
                out.push(m);
            }
        }
        assert_eq!(out, vec![a, b]);
    }
}
