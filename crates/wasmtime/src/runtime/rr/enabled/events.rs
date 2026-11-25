//! Embedder-defined events in traces.

use super::*;
use crate::sync::RwLock;

/// An embedder-defined event that can be recorded in a trace with
/// [`record_event`], and observed during replay with [`Replayer::on_event`].
///
/// Events carry information about the recorded execution, such as its output,
/// that the embedder wants to see again when replaying. They cannot affect the
/// replay itself.
pub trait TraceEvent: serde::Serialize + serde::de::DeserializeOwned + 'static {
    /// Identifies this type of event in traces. It must be unique among an
    /// embedding's event types, and stable between the processes that record
    /// and replay a trace. [`RESERVED_TAGS`] are reserved for Wasmtime's own
    /// crates, such as `wasmtime-wasi`.
    const TAG: u32;
}

/// The event tags reserved for Wasmtime's own crates; see [`TraceEvent::TAG`].
pub const RESERVED_TAGS: core::ops::RangeInclusive<u32> = 0x8000_0000..=u32::MAX;

/// Records `event` at the current point of a recording. This does nothing if
/// the store is not recording.
pub fn record_event<E: TraceEvent>(mut store: impl AsContextMut, event: &E) -> Result<()> {
    let store = store.as_context_mut().0;
    if !store.rr.recording() {
        return Ok(());
    }
    let result = encode(Vec::new(), event).and_then(|bytes| {
        let trace = &mut store.rr_session().mode.recording().bytes;
        Ok(trace.extend_from_slice(&bytes)?)
    });
    store.rr_poison_on_err(result)
}

/// Appends a record of `event` to `bytes`.
fn encode<E: TraceEvent>(mut bytes: Vec<u8>, event: &E) -> Result<Vec<u8>> {
    let start = bytes.len();
    bytes.push(codec::EVENT);
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&E::TAG.to_le_bytes());
    let mut bytes = postcard::to_extend(event, bytes)?;
    let len = u32::try_from(bytes.len() - start - 5)?;
    bytes[start + 1..start + 5].copy_from_slice(&len.to_le_bytes());
    Ok(bytes)
}

/// Creates a channel through which code without access to a store, such as the
/// implementation of a host stream, records events.
///
/// The sink can be handed to event producers (for example, built into a WASI
/// context) before the store is created. Events that it records are buffered
/// until the receiver is attached to a store's recording, with
/// [`Store::rr_attach_events`], from when they enter the trace when the host
/// next calls or returns to Wasm, or at the end of the recording. Once the
/// receiver is dropped, as it is when its recording finishes, recording through
/// the sink does nothing.
pub fn event_channel() -> (EventSink, EventReceiver) {
    let buffer = Arc::new(RwLock::new(Buffer::default()));
    (
        EventSink {
            buffer: buffer.clone(),
        },
        EventReceiver { buffer },
    )
}

/// The producing end of an [`event_channel`], which can be cloned and sent to
/// other threads.
#[derive(Clone)]
pub struct EventSink {
    buffer: Arc<RwLock<Buffer>>,
}

/// The receiving end of an [`event_channel`].
pub struct EventReceiver {
    buffer: Arc<RwLock<Buffer>>,
}

#[derive(Default)]
struct Buffer {
    /// Encoded event records, in the order they were recorded.
    bytes: Vec<u8>,
    /// Set when the receiver is dropped.
    closed: bool,
    /// The first encoding failure, which fails the recording.
    failure: Option<Error>,
}

impl EventSink {
    /// Records `event`, as for [`record_event`].
    pub fn record<E: TraceEvent>(&self, event: &E) {
        let mut buffer = self.buffer.write();
        if buffer.closed || buffer.failure.is_some() {
            return;
        }
        match encode(core::mem::take(&mut buffer.bytes), event) {
            Ok(bytes) => buffer.bytes = bytes,
            Err(e) => buffer.failure = Some(e),
        }
    }
}

impl EventReceiver {
    /// Appends the events recorded since the last call to `trace`.
    pub(super) fn drain(&self, trace: &mut TryVec<u8>) -> Result<()> {
        let mut buffer = self.buffer.write();
        if let Some(e) = buffer.failure.take() {
            return Err(e);
        }
        trace.extend_from_slice(&buffer.bytes)?;
        buffer.bytes.clear();
        Ok(())
    }
}

impl Drop for EventReceiver {
    fn drop(&mut self) {
        let mut buffer = self.buffer.write();
        buffer.closed = true;
        buffer.bytes = Vec::new();
    }
}

impl<T: 'static> Store<T> {
    /// Attaches the receiving end of an [`event_channel`] to this store's
    /// recording, so that events recorded through its sinks enter the trace.
    /// Fails if the store is not recording.
    pub fn rr_attach_events(&mut self, receiver: EventReceiver) -> Result<()> {
        let store = self.as_context_mut().0;
        ensure!(store.rr.recording(), "store is not recording");
        let recording = store.rr_session().mode.recording();
        recording.receivers.push(receiver)?;
        Ok(())
    }
}
