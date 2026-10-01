//! The only driver of replay activations. Host boundaries yield control here;
//! no host callback recursively runs another activation.
//!
//! This first runner uses StoreFiber for its TLS/trap/stack-limit discipline.
//! It is NOT yet stack-copyable: StoreFiber's resume state and completion bit,
//! must be included in a replay activation image before implementing snapshots.
//! Value buffers belong to the driver; suspended entry and exit frames borrow
//! them and do not own allocations.

use super::*;
use crate::StoreContextMut;
use crate::runtime::fiber::{self, StoreFiber, StoreFiberYield};
use crate::runtime::vm::{SendSyncPtr, VMArrayCallHostFuncContext, VMOpaqueContext};
use core::mem::MaybeUninit;
use core::task::{Context, Poll};

struct Activation {
    fiber: StoreFiber<'static>,
    func: usize,
    call: usize,
    host: Option<HostCall>,
    // Never resize this buffer while the fiber can access it.
    values: Vec<ValRaw>,
}

/// A borrowed array-call buffer on a suspended activation's stack. Only the
/// driver accesses it, and only before resuming or disposing that activation.
pub(super) struct HostCall {
    func: usize,
    call: usize,
    values: SendSyncPtr<[MaybeUninit<ValRaw>]>,
}

enum Observed {
    Host(usize, HostCall),
    Complete(usize, Result<()>),
}

struct Driver<'a, T: 'static> {
    store: &'a mut StoreInner<T>,
    reader: Reader<'a>,
    activations: Vec<Activation>,
    observed: Option<Observed>,
    scratch: Vec<u8>,
    instances: Vec<crate::Instance>,
    // A newly allocated instance must run its startup before any other code
    // can observe it, including when the trace is malformed.
    pending_startup: Option<usize>,
}

pub(super) async fn run<T: Send>(store: &mut StoreInner<T>, trace: &Trace) -> Result<Replay> {
    store.rr_validate()?;
    ensure!(
        store.engine().is_replaying(),
        "replay requires RRConfig::Replaying"
    );
    let objects = Objects::default();
    let mut reader = Reader::new(&trace.bytes);
    ensure!(
        reader.take(codec::MAGIC.len())? == codec::MAGIC,
        "unsupported trace version"
    );
    store.rr.session = Some(try_new::<Box<_>>(Session {
        objects,
        mode: Mode::Replaying {
            yielded: None,
            response: None,
            completed: None,
            entering: false,
        },
        pending: Vec::new(),
        failure: None,
    })?);
    let mut driver = Driver {
        store,
        reader,
        activations: Vec::new(),
        observed: None,
        scratch: Vec::new(),
        instances: Vec::new(),
        pending_startup: None,
    };
    core::future::poll_fn(|cx| {
        // Give the executor a chance to cancel long traces between events.
        for _ in 0..256 {
            match driver.step(cx) {
                Ok(true) => return Poll::Ready(Ok(())),
                Ok(false) => {}
                Err(e) => {
                    return Poll::Ready(Err(e.context(format!(
                        "replaying trace at byte {}",
                        driver.reader.position
                    ))));
                }
            }
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    })
    .await?;
    Ok(Replay {
        instances: core::mem::take(&mut driver.instances),
    })
}

impl<T: 'static> Driver<'_, T> {
    fn step(&mut self, cx: &mut Context<'_>) -> Result<bool> {
        let (tag, mut body) = self.reader.record()?;
        if self.pending_startup.is_some() {
            ensure!(
                matches!(tag, codec::ENTER_WASM | codec::GLOBAL_WRITE | codec::WRITE),
                "instance startup missing from trace"
            );
        }
        match tag {
            codec::END => {
                ensure!(
                    self.activations.is_empty() && self.observed.is_none(),
                    "trace ended with outstanding activations"
                );
                body.end()?;
                self.reader.end()?;
                return Ok(true);
            }
            codec::HOST
            | codec::MODULE
            | codec::INSTANCE
            | codec::GLOBAL
            | codec::GLOBAL_WRITE
            | codec::MEMORY
            | codec::TABLE => {
                ensure!(
                    self.observed.is_none(),
                    "initialization before matching guest execution"
                );
                ensure!(
                    self.activations.iter().all(|a| a.host.is_some()),
                    "initialization while guest is running"
                );
                if let Some((instance, startup)) =
                    super::init::replay_event(self.store, tag, &mut body)?
                {
                    self.instances.push(instance);
                    self.pending_startup = startup;
                }
            }
            codec::ENTER_WASM => {
                ensure!(
                    self.observed.is_none(),
                    "guest execution diverged before EnterWasm"
                );
                ensure!(
                    self.activations.iter().all(|a| a.host.is_some()),
                    "nested guest entry without a host boundary"
                );
                let id = usize::try_from(body.u32()?)?;
                let call = usize::try_from(body.u32()?)?;
                ensure!(
                    !self.activations.iter().any(|a| a.call == call),
                    "duplicate activation id"
                );
                let session = self.store.rr.session.as_ref().unwrap();
                let bound = session
                    .objects
                    .funcs
                    .get(id)
                    .ok_or_else(|| format_err!("unknown function id"))?;
                ensure!(!bound.host, "EnterWasm names a host function");
                match self.pending_startup.take() {
                    Some(startup) => ensure!(id == startup, "expected instance startup"),
                    None => ensure!(!bound.startup, "unexpected repeated instance startup"),
                }
                let mut values = value_slots(bound)?;
                body.values(&bound.params, &mut values, |id| {
                    session.objects.decode_ref(self.store, id)
                })?;
                body.end()?;
                let func = bound.func;
                self.activations
                    .try_reserve(1)
                    .map_err(|_| OutOfMemory::new(core::mem::size_of::<Activation>()))?;
                let raw = SendSyncPtr::from(NonNull::from(values.as_mut_slice()));
                // SAFETY: the driver owns the argument buffer, which is never
                // resized or released until the fiber finishes or is disposed.
                // Driver exclusively borrows this store until
                // Drop disposes every fiber, including on error/cancellation.
                // T is Send at the public entrypoint; no user T is touched by
                // the replay boundary trampoline in any case.
                let fiber = unsafe {
                    fiber::make_fiber_unchecked(self.store, move |store| {
                        let func_ref = func.vm_func_ref(store);
                        let result = Func::call_unchecked_raw(
                            &mut StoreContextMut(store),
                            func_ref,
                            raw.as_non_null(),
                        );
                        let Mode::Replaying { completed, .. } =
                            &mut store.rr.session.as_mut().unwrap().mode
                        else {
                            unreachable!()
                        };
                        *completed = Some(result);
                        Ok(())
                    })?
                };
                self.activations.push(Activation {
                    fiber,
                    func: id,
                    call,
                    host: None,
                    values,
                });
                let Mode::Replaying { entering, .. } =
                    &mut self.store.rr.session.as_mut().unwrap().mode
                else {
                    unreachable!()
                };
                *entering = true;
                self.resume(self.activations.len() - 1, cx)?;
            }
            codec::ENTER_HOST => {
                let Some(Observed::Host(activation, mut call)) = self.observed.take() else {
                    bail!("expected guest to reach a host boundary");
                };
                let id = usize::try_from(body.u32()?)?;
                call.call = usize::try_from(body.u32()?)?;
                ensure!(
                    !self
                        .activations
                        .iter()
                        .filter_map(|a| a.host.as_ref())
                        .any(|h| h.call == call.call),
                    "duplicate host call id"
                );
                ensure!(
                    call.func == id,
                    "host call target diverged: expected {id}, observed {}",
                    call.func
                );
                let bound = &self.store.rr.session.as_ref().unwrap().objects.funcs[id];
                self.scratch.clear();
                codec::reserve(&mut self.scratch, codec::values_len(&bound.params))?;
                // SAFETY: the activation is suspended and initialized the
                // parameter slots. The trampoline checked the buffer size.
                unsafe {
                    codec::values(
                        &mut self.scratch,
                        &bound.params,
                        call.values.as_ptr().cast(),
                        |ptr| {
                            self.store
                                .rr
                                .session
                                .as_ref()
                                .unwrap()
                                .objects
                                .encode_ref(ptr)
                        },
                    )?
                };
                ensure!(
                    body.take(self.scratch.len())? == self.scratch,
                    "host call arguments diverged"
                );
                body.end()?;
                self.activations
                    .iter_mut()
                    .find(|a| a.call == activation)
                    .unwrap()
                    .host = Some(call);
            }
            codec::LEAVE_HOST => {
                ensure!(self.observed.is_none(), "unexpected LeaveHost");
                let id = usize::try_from(body.u32()?)?;
                let index = self
                    .activations
                    .iter()
                    .position(|a| a.host.as_ref().is_some_and(|h| h.call == id))
                    .ok_or_else(|| format_err!("LeaveHost without a pending host call"))?;
                let mut call = self.activations[index].host.take().unwrap();
                let session = self.store.rr.session.as_ref().unwrap();
                let bound = &session.objects.funcs[call.func];
                // SAFETY: the activation is parked at this host call. Its
                // array-call storage is live and exclusively borrowed here.
                let values = unsafe { call.values.as_mut() };
                let outcome = body.outcome(&bound.results, values, |id| {
                    session.objects.decode_ref(self.store, id)
                })?;
                let session = self.store.rr.session.as_mut().unwrap();
                let Mode::Replaying { response, .. } = &mut session.mode else {
                    unreachable!()
                };
                *response = Some(outcome);
                self.resume(index, cx)?;
            }
            codec::LEAVE_WASM => {
                let Some(Observed::Complete(actual_call, result)) = self.observed.take() else {
                    bail!("expected guest to return or trap");
                };
                let call = usize::try_from(body.u32()?)?;
                ensure!(call == actual_call, "guest return activation diverged");
                let index = self
                    .activations
                    .iter()
                    .position(|a| a.call == call)
                    .unwrap();
                let activation = &self.activations[index];
                let bound = &self.store.rr.session.as_ref().unwrap().objects.funcs[activation.func];
                self.scratch.clear();
                // SAFETY: successful guest calls initialize all result slots.
                unsafe {
                    codec::outcome(
                        &mut self.scratch,
                        codec::LEAVE_WASM,
                        call,
                        &result,
                        &bound.results,
                        activation.values.as_ptr(),
                        |ptr| {
                            self.store
                                .rr
                                .session
                                .as_ref()
                                .unwrap()
                                .objects
                                .encode_ref(ptr)
                        },
                    )?;
                }
                ensure!(
                    body.bytes == &self.scratch[5..],
                    "guest return values or trap diverged"
                );
                self.activations.remove(index);
            }
            codec::WRITE => {
                ensure!(
                    self.observed.is_none(),
                    "memory update before matching guest execution"
                );
                ensure!(
                    self.activations.iter().all(|a| a.host.is_some()),
                    "memory update while guest is running"
                );
                let id = usize::try_from(body.u32()?)?;
                let offset = usize::try_from(body.u64()?)?;
                let memory = *self
                    .store
                    .rr
                    .session
                    .as_ref()
                    .unwrap()
                    .objects
                    .memories
                    .get(id)
                    .ok_or_else(|| format_err!("unknown memory id"))?;
                let bytes = body.take(body.bytes.len() - body.position)?;
                let end = offset
                    .checked_add(bytes.len())
                    .ok_or_else(|| format_err!("memory range overflow"))?;
                memory
                    .rr_data_mut(self.store)
                    .get_mut(offset..end)
                    .ok_or_else(|| format_err!("trace memory write out of bounds"))?
                    .copy_from_slice(bytes);
            }
            codec::RESIZE => {
                ensure!(
                    self.observed.is_none(),
                    "memory growth before matching guest execution"
                );
                ensure!(
                    self.activations.iter().all(|a| a.host.is_some()),
                    "memory growth while guest is running"
                );
                let id = usize::try_from(body.u32()?)?;
                let old = body.u64()?;
                let new = body.u64()?;
                body.end()?;
                let memory = *self
                    .store
                    .rr
                    .session
                    .as_ref()
                    .unwrap()
                    .objects
                    .memories
                    .get(id)
                    .ok_or_else(|| format_err!("unknown memory id"))?;
                let page_size = memory.wasmtime_ty(self.store).page_size();
                ensure!(
                    old == u64::try_from(memory.internal_data_size(self.store))?
                        && new >= old
                        && (new - old) % page_size == 0,
                    "invalid replay memory growth"
                );
                memory.grow(StoreContextMut(self.store), (new - old) / page_size)?;
            }
            _ => bail!("unknown trace event {tag}"),
        }
        Ok(false)
    }

    fn resume(&mut self, index: usize, cx: &mut Context<'_>) -> Result<()> {
        let activation = &mut self.activations[index];
        match fiber::resume_replay_fiber(self.store, &mut activation.fiber, cx) {
            Ok(result) => {
                result?;
                let Mode::Replaying { completed, .. } =
                    &mut self.store.rr.session.as_mut().unwrap().mode
                else {
                    unreachable!()
                };
                let result = completed
                    .take()
                    .ok_or_else(|| format_err!("replay activation did not report completion"))?;
                self.observed = Some(Observed::Complete(activation.call, result));
            }
            Err(StoreFiberYield::ReplayHost) => {
                let Mode::Replaying { yielded, .. } =
                    &mut self.store.rr.session.as_mut().unwrap().mode
                else {
                    unreachable!()
                };
                let call = yielded.take().ok_or_else(|| {
                    format_err!("replay activation did not report its host boundary")
                })?;
                self.observed = Some(Observed::Host(activation.call, call));
            }
            Err(_) => bail!("unsupported suspension during replay"),
        }
        Ok(())
    }
}

impl<T: 'static> Drop for Driver<'_, T> {
    fn drop(&mut self) {
        // Unwind in reverse activation order while the store and session still
        // exist. No original host implementation runs during cancellation.
        while let Some(mut activation) = self.activations.pop() {
            activation.fiber.dispose(self.store);
        }
        self.store.rr.session = None;
    }
}

fn value_slots(bound: &RecordedFunc) -> Result<Vec<ValRaw>> {
    let len = bound.params.len().max(bound.results.len());
    let mut values = Vec::new();
    values
        .try_reserve(len)
        .map_err(|_| OutOfMemory::new(len.saturating_mul(core::mem::size_of::<ValRaw>())))?;
    values.resize(len, ValRaw::v128(0));
    Ok(values)
}

/// The replay-only host boundary. The caller guarantees the context/signature
/// and raw slots came from a live, correctly typed Wasm-to-host trampoline.
pub(crate) unsafe fn host_call(
    store: &mut StoreOpaque,
    callee: NonNull<VMOpaqueContext>,
    args: NonNull<ValRaw>,
    args_len: usize,
) -> Result<()> {
    // SAFETY: this is the context type of the calling array-call trampoline.
    let func = unsafe {
        NonNull::from(
            &VMArrayCallHostFuncContext::from_opaque(callee)
                .as_ref()
                .func_ref,
        )
    };
    let session = store.rr.session.as_ref().unwrap();
    let id = session.objects.find_func(func)?;
    let bound = &session.objects.funcs[id];
    ensure!(bound.host, "replay boundary is not a host function");
    ensure!(
        args_len >= bound.params.len().max(bound.results.len()),
        "invalid replay value storage"
    );
    let values = SendSyncPtr::from(NonNull::slice_from_raw_parts(
        args.cast::<MaybeUninit<ValRaw>>(),
        args_len,
    ));
    let Mode::Replaying { yielded, .. } = &mut store.rr.session.as_mut().unwrap().mode else {
        unreachable!()
    };
    *yielded = Some(HostCall {
        func: id,
        call: 0,
        values,
    });

    // No references into Store or its session survive this yield. The store
    // itself is !Unpin, like the existing ReleaseStore suspension mechanism.
    // This suspends directly, without a user future or callback on the stack.
    store.with_blocking(|_, cx| cx.suspend(StoreFiberYield::ReplayHost))?;

    let session = store.rr.session.as_mut().unwrap();
    let Mode::Replaying { response, .. } = &mut session.mode else {
        unreachable!()
    };
    // The driver has already filled result slots in this activation's array
    // buffer. No result allocation or user-owned value lives in this frame.
    response
        .take()
        .ok_or_else(|| format_err!("replay resumed without host results"))?
}
