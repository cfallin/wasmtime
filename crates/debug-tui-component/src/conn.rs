//! A line-oriented terminal connection over TCP.
//!
//! This works with plain `nc` as well as `telnet`: Telnet option
//! negotiation (IAC sequences) is filtered out, and an "interrupt
//! process" (`IAC IP`) or a raw Ctrl-C byte is surfaced as an
//! interrupt request.

use anyhow::Result;
use std::collections::VecDeque;
use wstd::io::{AsyncRead, AsyncWrite};
use wstd::net::TcpStream;

/// One unit of user input.
#[derive(Debug, PartialEq, Eq)]
pub enum Input {
    /// A complete line (without line terminator).
    Line(String),
    /// An interrupt request (Ctrl-C).
    Interrupt,
    /// The connection was closed.
    Eof,
}

#[derive(Clone, Copy)]
enum Telnet {
    Normal,
    Iac,
    Opt,
    Sub,
    SubIac,
}

pub struct Conn {
    stream: TcpStream,
    line: Vec<u8>,
    pending: VecDeque<Input>,
    telnet: Telnet,
    out: String,
    eof: bool,
}

impl Conn {
    pub fn new(stream: TcpStream) -> Self {
        Conn {
            stream,
            line: vec![],
            pending: VecDeque::new(),
            telnet: Telnet::Normal,
            out: String::new(),
            eof: false,
        }
    }

    /// Queue output text.
    pub fn print(&mut self, s: &str) {
        self.out.push_str(s);
    }

    /// Queue output text followed by a newline.
    pub fn println(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// Write all queued output to the connection.
    pub async fn flush(&mut self) -> Result<()> {
        if self.out.is_empty() {
            return Ok(());
        }
        // Use CRLF line endings so that Telnet clients render
        // correctly; `nc` does not mind.
        let text = std::mem::take(&mut self.out).replace('\n', "\r\n");
        self.stream.write_all(text.as_bytes()).await?;
        self.stream.flush().await?;
        Ok(())
    }

    fn process(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.telnet = match (self.telnet, b) {
                (Telnet::Normal, 255) => Telnet::Iac,
                (Telnet::Normal, 3) => {
                    self.pending.push_back(Input::Interrupt);
                    Telnet::Normal
                }
                (Telnet::Normal, b'\n') => {
                    let line = String::from_utf8_lossy(&self.line).into_owned();
                    self.line.clear();
                    self.pending.push_back(Input::Line(line));
                    Telnet::Normal
                }
                (Telnet::Normal, b'\r') | (Telnet::Normal, 0) => Telnet::Normal,
                (Telnet::Normal, b) => {
                    self.line.push(b);
                    Telnet::Normal
                }
                (Telnet::Iac, 255) => {
                    self.line.push(255);
                    Telnet::Normal
                }
                // IP (interrupt process).
                (Telnet::Iac, 244) => {
                    self.pending.push_back(Input::Interrupt);
                    Telnet::Normal
                }
                // WILL/WONT/DO/DONT take one option byte.
                (Telnet::Iac, 251..=254) => Telnet::Opt,
                (Telnet::Iac, 250) => Telnet::Sub,
                (Telnet::Iac, _) => Telnet::Normal,
                (Telnet::Opt, _) => Telnet::Normal,
                (Telnet::Sub, 255) => Telnet::SubIac,
                (Telnet::Sub, _) => Telnet::Sub,
                (Telnet::SubIac, 240) => Telnet::Normal,
                (Telnet::SubIac, _) => Telnet::Sub,
            };
        }
    }

    /// Wait for the next unit of input. Cancel-safe: if the returned
    /// future is dropped before completion, no input is lost.
    pub async fn read_input(&mut self) -> Result<Input> {
        loop {
            if let Some(i) = self.pending.pop_front() {
                return Ok(i);
            }
            if self.eof {
                return Ok(Input::Eof);
            }
            let mut buf = [0u8; 512];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                self.eof = true;
                if !self.line.is_empty() {
                    let line = String::from_utf8_lossy(&self.line).into_owned();
                    self.line.clear();
                    self.pending.push_back(Input::Line(line));
                }
                continue;
            }
            self.process(&buf[..n]);
        }
    }
}
