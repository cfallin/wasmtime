//! Bindings for Wasmtime's debugger API.

use wstd::runtime::AsyncPollable;

wit_bindgen::generate!({
    world: "bytecodealliance:wasmtime/debug-main",
    path: "../debugger/wit",
    with: {
        "wasi:io/poll@0.2.12": wasip2::io::poll,
    }
});
pub(crate) use bytecodealliance::wasmtime::debuggee::*;

/// One "resumption", or period of execution, in the debuggee.
pub struct Resumption {
    future: EventFuture,
    pollable: Option<AsyncPollable>,
}

impl Resumption {
    pub fn continue_(d: &Debuggee) -> Self {
        let future = d.continue_(ResumptionValue::Normal);
        let pollable = Some(AsyncPollable::new(future.subscribe()));
        Resumption { future, pollable }
    }

    pub fn single_step(d: &Debuggee) -> Self {
        let future = d.single_step(ResumptionValue::Normal);
        let pollable = Some(AsyncPollable::new(future.subscribe()));
        Resumption { future, pollable }
    }

    /// Run backward by one instruction (`step`) or to the previous
    /// breakpoint or watchpoint stop. Fails for debuggees that are not
    /// replays.
    pub fn reverse(d: &Debuggee, step: bool) -> std::result::Result<Self, Error> {
        let future = if step {
            d.reverse_step()?
        } else {
            d.reverse_continue()?
        };
        let pollable = Some(AsyncPollable::new(future.subscribe()));
        Ok(Resumption { future, pollable })
    }

    pub async fn wait(&mut self) {
        if let Some(pollable) = self.pollable.as_mut() {
            pollable.wait_for().await;
        }
    }

    pub fn result(mut self, d: &Debuggee) -> std::result::Result<Event, Error> {
        // Drop the pollable first, since it's a child resource.
        let _ = self.pollable.take();
        EventFuture::finish(self.future, d)
    }
}
