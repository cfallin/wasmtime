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
    state: ResumptionState,
}

enum ResumptionState {
    Running {
        future: EventFuture,
        pollable: Option<AsyncPollable>,
    },
    /// A resumption that did not run, with the event to report.
    Done(Event),
}

impl Resumption {
    fn running(future: EventFuture) -> Self {
        let pollable = Some(AsyncPollable::new(future.subscribe()));
        Resumption {
            state: ResumptionState::Running { future, pollable },
        }
    }

    pub fn continue_(d: &Debuggee, r: ResumptionValue) -> Self {
        Self::running(d.continue_(r))
    }

    pub fn single_step(d: &Debuggee, r: ResumptionValue) -> Self {
        Self::running(d.single_step(r))
    }

    /// Runs backward, by one step or to the previous breakpoint or
    /// watchpoint stop. A debuggee that cannot run backward has no
    /// history to run back through, so it immediately reports reaching
    /// the beginning of its history.
    pub fn reverse(d: &Debuggee, step: bool) -> Self {
        let future = if step {
            d.reverse_step()
        } else {
            d.reverse_continue()
        };
        match future {
            Ok(future) => Self::running(future),
            Err(_) => Resumption {
                state: ResumptionState::Done(Event::ReplayBegin),
            },
        }
    }

    pub async fn wait(&mut self) {
        if let ResumptionState::Running {
            pollable: Some(pollable),
            ..
        } = &mut self.state
        {
            pollable.wait_for().await;
        }
    }

    pub fn result(self, d: &Debuggee) -> std::result::Result<Event, Error> {
        match self.state {
            ResumptionState::Running {
                future,
                mut pollable,
            } => {
                // Drop the pollable first, since it's a child resource.
                let _ = pollable.take();
                EventFuture::finish(future, d)
            }
            ResumptionState::Done(event) => Ok(event),
        }
    }
}
