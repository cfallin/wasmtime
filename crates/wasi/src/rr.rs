//! Capturing WASI output in record/replay traces.
//!
//! Wasmtime's record/replay ([`wasmtime::rr`]) replays guest execution without
//! running host code, so a replayed guest's writes to stdout and stderr go
//! nowhere. This module records that output as [`Output`] events in the trace
//! and replays it with [`replay_output`] or a [`Replayer::on_event`] observer.
//!
//! To record, wrap the stdout and stderr streams given to
//! [`WasiCtxBuilder`](crate::WasiCtxBuilder) in [`RecordedOutput`], using the
//! sink of an [`event_channel`](wasmtime::rr::event_channel), and attach the
//! channel's receiver to the store once it starts recording:
//!
//! ```no_run
//! # use wasmtime::{Config, Engine, RRConfig, Result, Store};
//! # use wasmtime_wasi::WasiCtxBuilder;
//! # use wasmtime_wasi::rr::{OutputKind, RecordedOutput};
//! # fn main() -> Result<()> {
//! let mut config = Config::new();
//! config.rr(RRConfig::Recording);
//! let engine = Engine::new(&config)?;
//! let (sink, receiver) = wasmtime::rr::event_channel();
//! let wasi = WasiCtxBuilder::new()
//!     .stdout(RecordedOutput::new(
//!         wasmtime_wasi::cli::stdout(),
//!         sink.clone(),
//!         OutputKind::Stdout,
//!     ))
//!     .stderr(RecordedOutput::new(
//!         wasmtime_wasi::cli::stderr(),
//!         sink,
//!         OutputKind::Stderr,
//!     ))
//!     .build();
//! let mut store = Store::new(&engine, wasi);
//! store.start_recording()?;
//! store.rr_attach_events(receiver)?;
//! // ... instantiate and run the guest, then `store.finish_recording()`.
//! # Ok(())
//! # }
//! ```
//!
//! Output is recorded when the guest's write is accepted by the stream: for
//! WASIp1 and WASIp2 streams that is inside the host call performing the
//! write, so the event appears in the trace right after that call. WASIp3
//! output is written by host tasks, and enters the trace at the store's next
//! boundary after the write completes, or at the end of the recording. Output
//! written after the recording has finished is not recorded.

use crate::cli::{IsTerminal, StdoutStream};
use bytes::Bytes;
use serde_derive::{Deserialize, Serialize};
use std::io::Write;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::AsyncWrite;
use wasmtime::rr::{EventSink, Replayer, TraceEvent};
use wasmtime_wasi_io::poll::Pollable;
use wasmtime_wasi_io::streams::{OutputStream, StreamResult};

/// Which standard output stream an [`Output`] event was written to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputKind {
    /// The guest's standard output.
    Stdout,
    /// The guest's standard error.
    Stderr,
}

/// A trace event for bytes a guest wrote to its standard output or error.
///
/// Recorded by [`RecordedOutput`]; one event is recorded for each write the
/// stream accepted, in the order they were accepted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// The stream written to.
    pub kind: OutputKind,
    /// The bytes written.
    pub bytes: Vec<u8>,
}

impl TraceEvent for Output {
    /// This tag, within [`wasmtime::rr::RESERVED_TAGS`], is stable.
    const TAG: u32 = 0x8000_0001;
}

/// A [`StdoutStream`] that records everything written to it as [`Output`]
/// events, and forwards it to an inner stream.
///
/// Each write is recorded exactly once, whichever WASI version performs it.
pub struct RecordedOutput<S> {
    inner: S,
    sink: EventSink,
    kind: OutputKind,
}

impl<S> RecordedOutput<S> {
    /// Records writes to `inner` through `sink` (from
    /// [`event_channel`](wasmtime::rr::event_channel)) as output of kind
    /// `kind`.
    pub fn new(inner: S, sink: EventSink, kind: OutputKind) -> Self {
        RecordedOutput { inner, sink, kind }
    }
}

impl<S: IsTerminal> IsTerminal for RecordedOutput<S> {
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }
}

impl<S: StdoutStream> StdoutStream for RecordedOutput<S> {
    // Both methods wrap the corresponding inner stream, so that an inner
    // `p2_stream` implemented with `async_stream` cannot record twice.
    fn p2_stream(&self) -> Box<dyn OutputStream> {
        Box::new(RecordedP2Stream {
            inner: self.inner.p2_stream(),
            recorder: self.recorder(),
        })
    }

    fn async_stream(&self) -> Box<dyn AsyncWrite + Send + Sync> {
        Box::new(RecordedAsyncWrite {
            inner: Pin::from(self.inner.async_stream()),
            recorder: self.recorder(),
        })
    }
}

impl<S> RecordedOutput<S> {
    fn recorder(&self) -> Recorder {
        Recorder {
            sink: self.sink.clone(),
            kind: self.kind,
        }
    }
}

struct Recorder {
    sink: EventSink,
    kind: OutputKind,
}

impl Recorder {
    fn record(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        self.sink.record(&Output {
            kind: self.kind,
            bytes: bytes.to_vec(),
        });
    }
}

struct RecordedP2Stream {
    inner: Box<dyn OutputStream>,
    recorder: Recorder,
}

#[async_trait::async_trait]
impl Pollable for RecordedP2Stream {
    async fn ready(&mut self) {
        self.inner.ready().await
    }
}

// The provided methods `blocking_write_and_flush`, `write_zeroes` and
// `write_ready` are deliberately not forwarded: their default implementations
// write through `write`, which records.
#[async_trait::async_trait]
impl OutputStream for RecordedP2Stream {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        self.inner.write(bytes.clone())?;
        // The guest observes the write as accepted here.
        self.recorder.record(&bytes);
        Ok(())
    }

    fn flush(&mut self) -> StreamResult<()> {
        self.inner.flush()
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        self.inner.check_write()
    }

    async fn cancel(&mut self) {
        self.inner.cancel().await
    }
}

struct RecordedAsyncWrite {
    inner: Pin<Box<dyn AsyncWrite + Send + Sync>>,
    recorder: Recorder,
}

// Vectored writes use the default implementation, through `poll_write`.
impl AsyncWrite for RecordedAsyncWrite {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let result = self.inner.as_mut().poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = result {
            self.recorder.record(&buf[..n.min(buf.len())]);
        }
        result
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.inner.as_mut().poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.inner.as_mut().poll_shutdown(cx)
    }
}

/// Writes the [`Output`] events of a replay to `stdout` and `stderr` as
/// replay reaches them, including again after rewinding to a checkpoint.
///
/// Each event is written and flushed in full; errors writing it are ignored,
/// since they cannot affect the replay. For other handling, observe
/// [`Output`] events with [`Replayer::on_event`] directly.
pub fn replay_output<T: Send>(
    replayer: &mut Replayer<'_, T>,
    mut stdout: impl Write + Send + 'static,
    mut stderr: impl Write + Send + 'static,
) {
    replayer.on_event(move |output: Output| {
        let out: &mut dyn Write = match output.kind {
            OutputKind::Stdout => &mut stdout,
            OutputKind::Stderr => &mut stderr,
        };
        let _ = out.write_all(&output.bytes).and_then(|()| out.flush());
    });
}
