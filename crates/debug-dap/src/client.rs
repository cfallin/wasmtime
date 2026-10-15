//! A simple blocking DAP client.

use crate::{Decoder, encode, request};
use anyhow::{Result, bail};
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// A connection to a DAP server. Events received while waiting for
/// responses are queued and can be retrieved with [`Client::next_event`].
pub struct Client {
    stream: TcpStream,
    decoder: Decoder,
    seq: i64,
    events: VecDeque<Value>,
    responses: VecDeque<Value>,
    /// Default timeout for waiting on responses.
    pub timeout: Duration,
}

impl Client {
    /// Connect to a DAP server.
    pub fn connect(addr: impl ToSocketAddrs) -> Result<Client> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        Ok(Client {
            stream,
            decoder: Decoder::new(),
            seq: 0,
            events: VecDeque::new(),
            responses: VecDeque::new(),
            timeout: Duration::from_secs(120),
        })
    }

    /// Send a request without waiting for the response; returns its
    /// sequence number.
    pub fn send_request(&mut self, command: &str, arguments: Value) -> Result<i64> {
        self.seq += 1;
        let msg = request(self.seq, command, arguments);
        self.stream.write_all(&encode(&msg))?;
        Ok(self.seq)
    }

    /// Read one message, waiting up to `timeout` (`None` waits
    /// indefinitely). Returns `None` on timeout.
    fn read_message(&mut self, timeout: Option<Duration>) -> Result<Option<Value>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if let Some(m) = self.decoder.next()? {
                return Ok(Some(m));
            }
            let remaining = match deadline {
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Ok(None);
                    }
                    Some(d - now)
                }
                None => None,
            };
            self.stream.set_read_timeout(remaining)?;
            let mut buf = [0u8; 16384];
            match self.stream.read(&mut buf) {
                Ok(0) => bail!("DAP connection closed"),
                Ok(n) => self.decoder.push(&buf[..n]),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    return Ok(None);
                }
                // A signal (e.g. Ctrl-C, which a client may handle by
                // pausing the debuggee) interrupted the read: report a
                // timeout so that the caller can check for that.
                Err(e) if e.kind() == ErrorKind::Interrupted => return Ok(None),
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Wait for the response to request `seq`; returns its body, or an
    /// error with the server's message if the request failed.
    pub fn wait_response(&mut self, seq: i64) -> Result<Value> {
        let deadline = Instant::now() + self.timeout;
        loop {
            if let Some(i) = self
                .responses
                .iter()
                .position(|r| r["request_seq"].as_i64() == Some(seq))
            {
                let r = self.responses.remove(i).unwrap();
                if r["success"].as_bool() == Some(true) {
                    return Ok(r.get("body").cloned().unwrap_or(Value::Null));
                }
                bail!(
                    "{}",
                    r["message"]
                        .as_str()
                        .unwrap_or("request failed")
                        .to_string()
                );
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("timed out waiting for a DAP response");
            }
            // `None` is a timeout or an interrupted read; the deadline is
            // rechecked above.
            if let Some(m) = self.read_message(Some(remaining))? {
                self.dispatch(m);
            }
        }
    }

    fn dispatch(&mut self, m: Value) {
        match m["type"].as_str() {
            Some("response") => self.responses.push_back(m),
            Some("event") => self.events.push_back(m),
            // Reverse requests (e.g. `runInTerminal`) are not supported.
            _ => {}
        }
    }

    /// Send a request and wait for its response body.
    pub fn request(&mut self, command: &str, arguments: Value) -> Result<Value> {
        let seq = self.send_request(command, arguments)?;
        self.wait_response(seq)
    }

    /// Get the next event: a queued one, or one read within `timeout`
    /// (`None` waits indefinitely).
    pub fn next_event(&mut self, timeout: Option<Duration>) -> Result<Option<Value>> {
        if let Some(e) = self.events.pop_front() {
            return Ok(Some(e));
        }
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            if remaining.is_some_and(|r| r.is_zero()) {
                return Ok(None);
            }
            match self.read_message(remaining)? {
                Some(m) => {
                    self.dispatch(m);
                    if let Some(e) = self.events.pop_front() {
                        return Ok(Some(e));
                    }
                }
                None => return Ok(None),
            }
        }
    }

    /// Wait for an event named `name`, returning its body. Other events
    /// are passed to `other` (e.g. to print `output` events).
    pub fn wait_event(&mut self, name: &str, other: &mut dyn FnMut(&Value)) -> Result<Value> {
        let deadline = Instant::now() + self.timeout;
        loop {
            let Some(e) =
                self.next_event(Some(deadline.saturating_duration_since(Instant::now())))?
            else {
                if Instant::now() >= deadline {
                    bail!("timed out waiting for a `{name}` event");
                }
                continue;
            };
            if e["event"].as_str() == Some(name) {
                return Ok(e.get("body").cloned().unwrap_or(Value::Null));
            }
            other(&e);
        }
    }

    /// Wait for any one of the named events, returning (name, body).
    pub fn wait_any_event(
        &mut self,
        names: &[&str],
        other: &mut dyn FnMut(&Value),
    ) -> Result<(String, Value)> {
        let deadline = Instant::now() + self.timeout;
        loop {
            let Some(e) =
                self.next_event(Some(deadline.saturating_duration_since(Instant::now())))?
            else {
                if Instant::now() >= deadline {
                    bail!("timed out waiting for one of {names:?}");
                }
                continue;
            };
            let name = e["event"].as_str().unwrap_or("").to_string();
            if names.contains(&name.as_str()) {
                return Ok((name, e.get("body").cloned().unwrap_or(Value::Null)));
            }
            other(&e);
        }
    }

    /// Whether events are queued (without blocking).
    pub fn has_queued_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Access the underlying stream (e.g. for polling).
    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }
}
