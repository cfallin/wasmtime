#![cfg(all(feature = "rr", feature = "cranelift", feature = "wat"))]

use wasmtime::{rr::Trace, *};

fn engine(mode: RRConfig) -> Result<Engine> {
    let mut config = Config::new();
    config.rr(mode);
    Engine::new(&config)
}

// Trace frame tags, for tests that edit traces (see `rr/codec.rs`).
const ENTER_WASM: u8 = 1;
const LEAVE_HOST: u8 = 4;
const WRITE: u8 = 5;
const MODULE: u8 = 8;
const GROWTH_FAILED: u8 = 14;

/// The `(tag, body offset)` of each frame of a serialized trace.
fn frames(bytes: &[u8]) -> Vec<(u8, usize)> {
    let mut frames = Vec::new();
    // Skip the version magic.
    let mut offset = 8;
    while offset < bytes.len() {
        let len = u32::from_le_bytes(bytes[offset + 1..offset + 5].try_into().unwrap());
        frames.push((bytes[offset], offset + 5));
        offset += 5 + len as usize;
    }
    frames
}

/// The body offset of the first frame with `tag`.
fn first_frame(bytes: &[u8], tag: u8) -> usize {
    frames(bytes)
        .into_iter()
        .find(|(t, _)| *t == tag)
        .expect("trace frame")
        .1
}

const NESTED: &str = r#"
(module
  (import "" "host" (func $host (result i32)))
  (import "" "notify" (func $notify (result i32)))
  (memory (export "memory") 1)
  (func (export "callback") (result i32)
    call $notify drop
    i32.const 0 i32.load)
  (func (export "fail") unreachable)
  (func (export "run") (result i32)
    i32.const 0 i32.const 1 i32.store
    call $host
    i32.const 0 i32.load i32.add))
"#;

fn nested() -> Result<(Store<usize>, Func, Memory)> {
    let engine = engine(RRConfig::Recording)?;
    let mut store = Store::new(&engine, 0);
    store.start_recording()?;
    let host = Func::wrap(
        &mut store,
        move |mut caller: Caller<'_, usize>| -> Result<i32> {
            *caller.data_mut() += 1;
            let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
            // A raw slice handed to the host is recorded at the next boundary.
            memory.data_mut(&mut caller)[..4].copy_from_slice(&10_i32.to_le_bytes());
            let callback = caller
                .get_export("callback")
                .unwrap()
                .into_func()
                .unwrap()
                .typed::<(), i32>(&caller)?;
            let value = callback.call(&mut caller, ())?;
            assert_eq!(value, 10);
            let fail = caller.get_export("fail").unwrap().into_func().unwrap();
            let error = fail.call(&mut caller, &[], &mut []).unwrap_err();
            assert_eq!(
                error.downcast_ref::<Trap>(),
                Some(&Trap::UnreachableCodeReached)
            );
            // The legacy slice API also participates in recording.
            memory.data_mut(&mut caller)[..4].copy_from_slice(&20_i32.to_le_bytes());
            Ok(value + 5)
        },
    );
    let notify = Func::wrap(&mut store, move |mut caller: Caller<'_, usize>| {
        *caller.data_mut() += 1;
        7_i32
    });
    let module = Module::new(&engine, NESTED)?;
    let instance = Instance::new(&mut store, &module, &[host.into(), notify.into()])?;
    let run = instance.get_func(&mut store, "run").unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    Ok((store, run, memory))
}

#[tokio::test]
async fn nested_callbacks_memory_and_caught_trap() -> Result<()> {
    let (mut record, run, memory) = nested()?;
    assert_eq!(run.typed::<(), i32>(&record)?.call(&mut record, ())?, 35);
    assert_eq!(*record.data(), 2);
    let trace = Trace::from_bytes(record.finish_recording()?.as_bytes().to_vec())?;
    let expected = memory.data(&record).to_vec();

    let mut replay = Store::new(&engine(RRConfig::Replaying)?, 0_usize);
    let output = replay.replay(&trace).await?;
    let memory = output.instances()[0]
        .get_memory(&mut replay, "memory")
        .unwrap();
    assert_eq!(memory.data(&replay), expected);
    assert_eq!(*replay.data(), 0);

    // Replayed host functions only run under the replay driver, so a replayed
    // store can be inspected but not called.
    let run = output.instances()[0].get_typed_func::<(), i32>(&mut replay, "run")?;
    let error = run.call(&mut replay, ()).unwrap_err();
    assert!(error.to_string().contains("inspected"), "{error:#}");
    Ok(())
}

fn identity() -> Result<(Store<()>, Func, Memory)> {
    let engine = engine(RRConfig::Recording)?;
    let mut store = Store::new(&engine, ());
    store.start_recording()?;
    let module = Module::new(
        &engine,
        r#"(module
        (memory (export "memory") 1)
        (func (export "id") (param i32 i64 f32 f64 v128) (result i32 i64 f32 f64 v128)
            local.get 0 local.get 1 local.get 2 local.get 3 local.get 4))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[])?;
    let func = instance.get_func(&mut store, "id").unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    Ok((store, func, memory))
}

#[tokio::test]
async fn scalar_bits_vectors_and_top_level_memory() -> Result<()> {
    let (mut store, func, memory) = identity()?;
    memory.write(&mut store, 5, b"before")?;
    let args = [
        Val::I32(-17),
        Val::I64(i64::MIN),
        Val::F32(0xffc0_1234),
        Val::F64(0x8000_0000_0000_0000),
        Val::V128(u128::MAX.into()),
    ];
    let mut results = [Val::I32(0); 5];
    func.call(&mut store, &args, &mut results)?;
    memory.data_and_store_mut(&mut store).0[11..16].copy_from_slice(b"after");
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let memory = output.instances()[0]
        .get_memory(&mut replay, "memory")
        .unwrap();
    assert_eq!(&memory.data(&replay)[5..16], b"beforeafter");
    Ok(())
}

#[test]
fn framing_rejects_truncation_and_trailing_data() -> Result<()> {
    let (mut store, _, _) = identity()?;
    let trace = store.finish_recording()?;
    for len in 0..trace.as_bytes().len() {
        assert!(Trace::from_bytes(trace.as_bytes()[..len].to_vec()).is_err());
    }
    let mut extra = trace.as_bytes().to_vec();
    extra.push(0);
    assert!(Trace::from_bytes(extra).is_err());
    Ok(())
}

#[tokio::test]
async fn recording_and_replay_require_empty_stores() -> Result<()> {
    // Application data and precompiled modules do not populate the store.
    let record_engine = engine(RRConfig::Recording)?;
    let replay_engine = engine(RRConfig::Replaying)?;
    let mut record = Store::new(&record_engine, 123);
    record.start_recording()?;
    let trace = record.finish_recording()?;
    let mut replay = Store::new(&replay_engine, 456);
    assert!(replay.replay(&trace).await?.instances().is_empty());
    assert_eq!(*replay.data(), 456);

    let constructors: &[fn(&mut Store<i32>) -> Result<()>] = &[
        |store| {
            Func::wrap(store, || ());
            Ok(())
        },
        |store| {
            Memory::new(store, MemoryType::new(0, None))?;
            Ok(())
        },
        |store| {
            Global::new(
                store,
                GlobalType::new(ValType::I32, Mutability::Const),
                Val::I32(0),
            )?;
            Ok(())
        },
        |store| {
            Table::new(
                store,
                TableType::new(RefType::FUNCREF, 0, None),
                Ref::Func(None),
            )?;
            Ok(())
        },
        |store| {
            let module = Module::new(store.engine(), "(module)")?;
            Instance::new(store, &module, &[])?;
            Ok(())
        },
    ];
    for construct in constructors {
        let mut record = Store::new(&record_engine, 123);
        construct(&mut record)?;
        assert!(
            record
                .start_recording()
                .unwrap_err()
                .to_string()
                .contains("empty store")
        );
        assert_eq!(*record.data(), 123);
        let mut replay = Store::new(&replay_engine, 456);
        construct(&mut replay)?;
        assert!(
            replay
                .replay(&trace)
                .await
                .unwrap_err()
                .to_string()
                .contains("empty store")
        );
        assert_eq!(*replay.data(), 456);
    }
    Ok(())
}

#[test]
fn forbidden_mutation_poisons_even_if_error_is_caught() -> Result<()> {
    let (mut store, _, _) = identity()?;
    let global = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, Mutability::Var),
        Val::I32(0),
    )?;
    assert!(global.set(&mut store, Val::I32(1)).is_err());
    assert!(store.finish_recording().is_err());

    let (mut store, _, _) = identity()?;
    let table = Table::new(
        &mut store,
        TableType::new(RefType::FUNCREF, 1, None),
        Ref::Func(None),
    )?;
    assert!(table.set(&mut store, 0, Ref::Func(None)).is_err());
    assert!(store.finish_recording().is_err());
    Ok(())
}

#[tokio::test]
async fn memory_growth_flushes_old_borrows_and_records_new_extent() -> Result<()> {
    let (mut store, _, memory) = identity()?;
    memory.data_mut(&mut store)[0] = 13;
    assert_eq!(memory.grow(&mut store, 1)?, 1);
    memory.write(&mut store, 65536, b"new page")?;
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let replay_memory = output.instances()[0]
        .get_memory(&mut replay, "memory")
        .unwrap();
    assert_eq!(replay_memory.data(&replay), memory.data(&store));
    Ok(())
}

#[test]
fn rr_config_enables_determinism_but_rejects_explicit_conflicts() -> Result<()> {
    engine(RRConfig::Recording)?;
    engine(RRConfig::Replaying)?;
    let mut config = Config::new();
    config
        .cranelift_nan_canonicalization(false)
        .rr(RRConfig::Recording);
    assert!(Engine::new(&config).is_err());
    let mut config = Config::new();
    config
        .rr(RRConfig::Replaying)
        .relaxed_simd_deterministic(false);
    assert!(Engine::new(&config).is_err());
    Ok(())
}

#[tokio::test]
async fn malformed_host_result_disposes_nested_activations() -> Result<()> {
    let (mut store, run, _) = nested()?;
    run.typed::<(), i32>(&store)?.call(&mut store, ())?;
    let mut bytes = store.finish_recording()?.as_bytes().to_vec();
    // An invalid outcome for the first LeaveHost, while two activations are
    // parked.
    let body = first_frame(&bytes, LEAVE_HOST);
    bytes[body + 4] = 0xff;
    let trace = Trace::from_bytes(bytes)?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, 0_usize);
    assert!(replay.replay(&trace).await.is_err());
    // Cleanup released the store, including TLS and both parked stacks.
    let memory = Memory::new(&mut replay, MemoryType::new(1, None))?;
    memory.write(&mut replay, 0, &[42])?;
    assert_eq!(memory.data(&replay)[0], 42);
    Ok(())
}

#[tokio::test]
async fn trapping_and_failing_activations_finish_by_yielding() -> Result<()> {
    let recording = engine(RRConfig::Recording)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, |x: i32| -> Result<i32> {
        if x % 2 == 0 {
            wasmtime::bail!("even argument {x}")
        }
        Ok(x)
    });
    let module = Module::new(
        &recording,
        r#"(module
        (import "" "host" (func $host (param i32) (result i32)))
        (func (export "run") (param i32) (result i32)
            (if (i32.eqz (i32.rem_u (local.get 0) (i32.const 3)))
                (then unreachable))
            (call $host (local.get 0))))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[host.into()])?;
    let run = instance.get_typed_func::<i32, i32>(&mut store, "run")?;
    // Guest traps before and after host calls, host errors, and returns, each
    // in a fresh activation whose fiber finishes with its final yield.
    let outcomes = (0..6)
        .map(|i| run.call(&mut store, i).is_ok())
        .collect::<Vec<_>>();
    assert_eq!(outcomes, [false, true, false, false, false, true]);
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    replay.replay(&trace).await?;
    // Every activation's state was released.
    let memory = Memory::new(&mut replay, MemoryType::new(1, None))?;
    memory.write(&mut replay, 0, &[1])?;
    Ok(())
}

#[test]
#[cfg(feature = "all-arch")]
fn replay_trampolines_compile_for_native_targets() -> Result<()> {
    for target in [
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "s390x-unknown-linux-gnu",
        "riscv64gc-unknown-linux-gnu",
    ] {
        let mut config = Config::new();
        config.target(target)?.rr(RRConfig::Replaying);
        Engine::new(&config)?.precompile_module(b"\0asm\x01\0\0\0")?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "pulley")]
fn replay_requires_a_native_target() -> Result<()> {
    let mut config = Config::new();
    config.target("pulley64")?.rr(RRConfig::Replaying);
    let error = Engine::new(&config).unwrap_err();
    assert!(error.to_string().contains("native"), "{error:?}");
    Ok(())
}

/// Program output, as a WASI implementation could record it.
#[derive(Debug, Clone, PartialEq, serde_derive::Serialize, serde_derive::Deserialize)]
struct Output {
    stream: u8,
    text: String,
}

impl rr::TraceEvent for Output {
    const TAG: u32 = 1;
}

/// A guest that prints through a host function which records its output.
fn printing() -> Result<(Store<()>, TypedFunc<i32, ()>)> {
    let recording = engine(RRConfig::Recording)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let print = Func::wrap(
        &mut store,
        |mut caller: Caller<'_, ()>, stream: i32, ptr: i32, len: i32| -> Result<()> {
            let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
            let bytes = &memory.data(&caller)[ptr as usize..][..len as usize];
            let text = String::from_utf8(bytes.to_vec())?;
            rr::record_event(
                &mut caller,
                &Output {
                    stream: stream as u8,
                    text,
                },
            )
        },
    );
    let module = Module::new(
        &recording,
        r#"(module
        (import "" "print" (func $print (param i32 i32 i32)))
        (memory (export "memory") 1)
        (data (i32.const 0) "tick err")
        (func (export "run") (param $n i32)
            (loop $l
                (call $print (i32.const 1) (i32.const 0) (i32.const 4))
                (call $print (i32.const 2) (i32.const 5) (i32.const 3))
                (br_if $l (local.tee $n (i32.sub (local.get $n) (i32.const 1)))))))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[print.into()])?;
    let run = instance.get_typed_func::<i32, ()>(&mut store, "run")?;
    Ok((store, run))
}

#[tokio::test]
async fn embedder_events_are_replayed_in_order() -> Result<()> {
    let (mut store, run) = printing()?;
    let start = Output {
        stream: 0,
        text: "start".to_string(),
    };
    rr::record_event(&mut store, &start)?;
    run.call(&mut store, 2)?;
    let trace = store.finish_recording()?;

    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let mut replayer = replay.replayer(&trace)?;
    let observed = seen.clone();
    replayer.on_event(move |output: Output| observed.lock().unwrap().push(output));
    assert_eq!(replayer.run().await?, rr::ReplayStop::Finished);
    drop(replayer);
    let tick = |stream, text: &str| Output {
        stream,
        text: text.to_string(),
    };
    assert_eq!(
        *seen.lock().unwrap(),
        [
            start,
            tick(1, "tick"),
            tick(2, "err"),
            tick(1, "tick"),
            tick(2, "err"),
        ]
    );
    // Recording outside of a session does nothing.
    rr::record_event(&mut replay, &tick(0, "ignored"))?;
    Ok(())
}

#[tokio::test]
async fn failed_guest_growth_is_replayed() -> Result<()> {
    // Growth beyond a small, immovable reservation fails when recording but
    // would succeed with the replaying engine's default configuration.
    let mut config = Config::new();
    config
        .rr(RRConfig::Recording)
        .memory_reservation(1 << 18)
        .memory_reservation_for_growth(0)
        .memory_may_move(false)
        .memory_guard_size(0)
        .signals_based_traps(false);
    let recording = Engine::new(&config)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let module = Module::new(
        &recording,
        r#"(module
        (memory (export "memory") 1)
        (table 1 2 funcref)
        (func (export "run") (result i32 i32 i32)
            (memory.grow (i32.const 8))
            (memory.grow (i32.const 1))
            (table.grow (ref.null func) (i32.const 2))))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[])?;
    let run = instance.get_typed_func::<(), (i32, i32, i32)>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ())?, (-1, 1, -1));
    let trace = store.finish_recording()?;
    let failures = frames(trace.as_bytes())
        .iter()
        .filter(|(tag, _)| *tag == GROWTH_FAILED)
        .count();
    assert_eq!(failures, 2);
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let memory = output.instances()[0]
        .get_memory(&mut replay, "memory")
        .unwrap();
    assert_eq!(memory.data_size(&replay), 2 << 16);
    Ok(())
}

#[tokio::test]
async fn replay_detects_divergent_results() -> Result<()> {
    let recording = engine(RRConfig::Recording)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || 1_i32);
    let module = Module::new(
        &recording,
        r#"(module
        (import "" "host" (func $host (result i32)))
        (func (export "run") (result i32) (i32.add (call $host) (i32.const 1))))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[host.into()])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ())?, 2);
    let mut bytes = store.finish_recording()?.as_bytes().to_vec();
    // The host result, after the call ID and the success tag.
    let body = first_frame(&bytes, LEAVE_HOST);
    bytes[body + 5..body + 9].copy_from_slice(&5_i32.to_le_bytes());
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let error = replay.replay(&Trace::from_bytes(bytes)?).await.unwrap_err();
    assert!(format!("{error:#}").contains("diverged"), "{error:#}");
    Ok(())
}

#[tokio::test]
async fn sessions_require_matching_engine_modes() -> Result<()> {
    let mut replaying = Store::new(&engine(RRConfig::Replaying)?, ());
    assert!(replaying.start_recording().is_err());
    assert!(replaying.finish_recording().is_err());
    let mut recording = Store::new(&engine(RRConfig::Recording)?, ());
    recording.start_recording()?;
    let trace = recording.finish_recording()?;
    let mut recording = Store::new(recording.engine(), ());
    assert!(recording.replay(&trace).await.is_err());
    Store::new(replaying.engine(), ()).replay(&trace).await?;
    Ok(())
}

fn many_calls() -> Result<(Store<()>, Func, Func)> {
    let engine = engine(RRConfig::Recording)?;
    let mut store = Store::new(&engine, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || ());
    let module = Module::new(
        &engine,
        r#"(module
        (import "" "host" (func $host))
        (func (export "run") (local $i i32)
            (loop $again
                call $host
                local.get $i i32.const 1 i32.add local.tee $i
                i32.const 300 i32.lt_u br_if $again)))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[host.into()])?;
    let run = instance.get_func(&mut store, "run").unwrap();
    Ok((store, run, host))
}

#[tokio::test]
async fn cancelling_a_parked_activation_releases_the_store() -> Result<()> {
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    let (mut store, run, _) = many_calls()?;
    run.call(&mut store, &[], &mut [])?;
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let mut future = Box::pin(replay.replay(&trace));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ));
    drop(future);
    assert!(
        replay
            .replay(&trace)
            .await
            .unwrap_err()
            .to_string()
            .contains("empty store")
    );
    let mut fresh = Store::new(replay.engine(), ());
    fresh.replay(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn asynchronous_recording_and_host_errors() -> Result<()> {
    async fn setup() -> Result<(Store<()>, Func)> {
        let engine = engine(RRConfig::Recording)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let host = Func::wrap_async(&mut store, move |_: Caller<'_, ()>, (): ()| {
            Box::new(async move {
                tokio::task::yield_now().await;
                Err::<(), _>(wasmtime::format_err!("recorded host failure"))
            })
        });
        let module = Module::new(
            &engine,
            r#"(module
            (import "" "host" (func $host))
            (func (export "run") call $host))"#,
        )?;
        let instance = Instance::new_async(&mut store, &module, &[host.into()]).await?;
        let run = instance.get_func(&mut store, "run").unwrap();
        Ok((store, run))
    }
    let (mut store, run) = setup().await?;
    assert!(run.call_async(&mut store, &[], &mut []).await.is_err());
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[test]
fn host_panic_cannot_be_finished_as_a_complete_trace() -> Result<()> {
    let engine = engine(RRConfig::Recording)?;
    let mut store = Store::new(&engine, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || -> () {
        panic!("intentional recording test panic")
    });
    let module = Module::new(
        &engine,
        r#"(module
        (import "" "host" (func $host))
        (func (export "run") call $host))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[host.into()])?;
    let run = instance.get_func(&mut store, "run").unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run.call(&mut store, &[], &mut []);
        }))
        .is_err()
    );
    assert!(store.finish_recording().is_err());
    // Finalization discarded the failed session, but the store is populated.
    assert!(
        store
            .start_recording()
            .unwrap_err()
            .to_string()
            .contains("empty store")
    );
    let mut fresh = Store::new(&engine, ());
    fresh.start_recording()?;
    fresh.finish_recording()?;
    Ok(())
}

#[cfg(feature = "gc")]
#[test]
fn gc_reference_boundaries_are_rejected() -> Result<()> {
    let engine = engine(RRConfig::Recording)?;
    let mut store = Store::new(&engine, ());
    store.start_recording()?;
    let host = Func::new(
        &mut store,
        FuncType::new(&engine, [ValType::EXTERNREF], []),
        |_, _, _| Ok(()),
    );
    let module = Module::new(
        &engine,
        r#"(module (import "" "host" (func (param externref))))"#,
    )?;
    assert!(Instance::new(&mut store, &module, &[host.into()]).is_err());
    assert!(store.finish_recording().is_err());
    Ok(())
}

#[tokio::test]
async fn private_table_callback_linker_import_and_memory_alias() -> Result<()> {
    fn setup() -> Result<(Store<()>, Func, Memory)> {
        let engine = engine(RRConfig::Recording)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let memory = Memory::new(&mut store, MemoryType::new(1, Some(2)))?;
        let mut linker = Linker::new(&engine);
        linker.define(&store, "", "memory", memory)?;
        linker.func_wrap(
            "",
            "host",
            move |mut caller: Caller<'_, ()>| -> Result<i32> {
                let memory = caller.get_export("alias").unwrap().into_memory().unwrap();
                memory.write(&mut caller, 0, &23_i32.to_le_bytes())?;
                let table = caller
                    .get_export("callbacks")
                    .unwrap()
                    .into_table()
                    .unwrap();
                let Ref::Func(Some(callback)) = table.get(&mut caller, 0).unwrap() else {
                    panic!()
                };
                callback.typed::<(), i32>(&caller)?.call(&mut caller, ())
            },
        )?;
        let module = Module::new(
            &engine,
            r#"(module
            (import "" "memory" (memory 1 2))
            (import "" "host" (func $host (result i32)))
            (export "alias" (memory 0))
            (table (export "callbacks") 1 funcref)
            (elem (i32.const 0) $private)
            (func $private (result i32) i32.const 0 i32.load)
            (func (export "run") (result i32) call $host))"#,
        )?;
        let instance = linker.instantiate(&mut store, &module)?;
        let run = instance.get_func(&mut store, "run").unwrap();
        Ok((store, run, memory))
    }
    let (mut record, run, memory) = setup()?;
    assert_eq!(run.typed::<(), i32>(&record)?.call(&mut record, ())?, 23);
    let trace = record.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let replay_memory = output.instances()[0]
        .get_memory(&mut replay, "alias")
        .unwrap();
    assert_eq!(memory.data(&record), replay_memory.data(&replay));
    Ok(())
}

#[tokio::test]
async fn object_ids_do_not_depend_on_export_lookup_order() -> Result<()> {
    fn setup(reverse: bool) -> Result<(Store<()>, Func)> {
        let engine = engine(RRConfig::Recording)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let module = Module::new(
            &engine,
            r#"(module
            (func (export "a") (result i32) i32.const 1)
            (func (export "b") (result i32) i32.const 2))"#,
        )?;
        let instance = Instance::new(&mut store, &module, &[])?;
        if reverse {
            instance.get_func(&mut store, "b").unwrap();
        }
        let a = instance.get_func(&mut store, "a").unwrap();
        Ok((store, a))
    }
    let (mut record, a) = setup(false)?;
    a.typed::<(), i32>(&record)?.call(&mut record, ())?;
    let trace = record.finish_recording()?;
    let (mut other, a) = setup(true)?;
    a.typed::<(), i32>(&other)?.call(&mut other, ())?;
    assert_eq!(trace.as_bytes(), other.finish_recording()?.as_bytes());
    Store::new(&engine(RRConfig::Replaying)?, ())
        .replay(&trace)
        .await?;
    Ok(())
}

#[tokio::test]
async fn segment_drops_are_replayed() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let module = Module::new(
        &e,
        r#"(module
        (memory 1) (table 1 funcref)
        (data $data "initial") (elem $elem func $f) (func $f)
        (func (export "drop") data.drop $data elem.drop $elem)
        (func (export "data") i32.const 0 i32.const 0 i32.const 1 memory.init $data)
        (func (export "elem") i32.const 0 i32.const 0 i32.const 1 table.init $elem))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[])?;
    for name in ["data", "elem", "drop"] {
        instance
            .get_typed_func::<(), ()>(&mut store, name)?
            .call(&mut store, ())?;
    }
    for name in ["data", "elem"] {
        assert!(
            instance
                .get_typed_func::<(), ()>(&mut store, name)?
                .call(&mut store, ())
                .is_err()
        );
    }
    let trace = store.finish_recording()?;
    Store::new(&engine(RRConfig::Replaying)?, ())
        .replay(&trace)
        .await?;
    Ok(())
}

#[tokio::test]
async fn direct_host_calls_only_record_their_guest_effects() -> Result<()> {
    fn setup() -> Result<(Store<()>, Func, Memory)> {
        let engine = engine(RRConfig::Recording)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let module = Module::new(
            &engine,
            r#"(module
            (memory (export "memory") 1)
            (func (export "run") (result i32) i32.const 0 i32.load))"#,
        )?;
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();
        let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
        let host = Func::wrap(
            &mut store,
            move |mut caller: Caller<'_, ()>| -> Result<i32> {
                memory.write(&mut caller, 0, &42_i32.to_le_bytes())?;
                run.call(&mut caller, ())
            },
        );
        Ok((store, host, memory))
    }
    let (mut record, host, _) = setup()?;
    assert_eq!(host.typed::<(), i32>(&record)?.call(&mut record, ())?, 42);
    let trace = record.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let memory = output.instances()[0]
        .get_memory(&mut replay, "memory")
        .unwrap();
    assert_eq!(&memory.data(&replay)[..4], &42_i32.to_le_bytes());
    Ok(())
}

#[tokio::test]
async fn initialization_replays_start_calls_and_retains_instance_exports() -> Result<()> {
    let record_engine = engine(RRConfig::Recording)?;
    let mut record = Store::new(&record_engine, ());
    record.start_recording()?;
    let host = Func::wrap(&mut record, |mut caller: Caller<'_, ()>| -> Result<i32> {
        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
        memory.write(&mut caller, 0, &71_i32.to_le_bytes())?;
        Ok(9)
    });
    let module = Module::new(
        &record_engine,
        r#"(module
        (import "" "host" (func $host (result i32)))
        (memory (export "memory") 1)
        (global $g (export "global") (mut i32) (i32.const 0))
        (func $start call $host global.set $g)
        (start $start)
        (func (export "run") (result i32) global.get $g))"#,
    )?;
    let instance = Instance::new(&mut record, &module, &[host.into()])?;
    assert_eq!(
        instance
            .get_typed_func::<(), i32>(&mut record, "run")?
            .call(&mut record, ())?,
        9
    );
    let trace = Trace::from_bytes(record.finish_recording()?.as_bytes().to_vec())?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let instance = output.instances()[0];
    assert_eq!(
        &instance
            .get_memory(&mut replay, "memory")
            .unwrap()
            .data(&replay)[..4],
        &71_i32.to_le_bytes()
    );
    assert_eq!(
        instance
            .get_global(&mut replay, "global")
            .unwrap()
            .get(&mut replay)
            .unwrap_i32(),
        9
    );
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_initialization_and_lowered_host_calls() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let record_engine = engine(RRConfig::Recording)?;
    let mut record = Store::new(&record_engine, 0_u32);
    record.start_recording()?;
    let component = Component::new(
        &record_engine,
        r#"(component
        (import "host" (func $host (param "x" u32) (result u32)))
        (core func $host (canon lower (func $host)))
        (core module $m
            (import "" "host" (func $host (param i32) (result i32)))
            (global $g (export "g") (mut i32) (i32.const 0))
            (func $start i32.const 10 call $host global.set $g)
            (start $start)
            (func (export "run") (param i32) (result i32)
                local.get 0 call $host global.get $g i32.add))
        (core instance $i (instantiate $m
            (with "" (instance (export "host" (func $host))))))
        (func (export "run") (param "x" u32) (result u32)
            (canon lift (core func $i "run"))))"#,
    )?;
    let serialized = component.serialize()?;
    // SAFETY: these unchanged artifacts were just compiled by the same engine.
    let component = unsafe { Component::deserialize(&record_engine, &serialized)? };
    let mut linker = Linker::new(&record_engine);
    linker.root().func_wrap("host", |mut store, (x,): (u32,)| {
        *store.data_mut() += 1;
        Ok((x + 1,))
    })?;
    let instance = linker.instantiate(&mut record, &component)?;
    let run = instance.get_typed_func::<(u32,), (u32,)>(&mut record, "run")?;
    assert_eq!(run.call(&mut record, (30,))?, (42,));
    assert_eq!(*record.data(), 2);
    let trace = Trace::from_bytes(record.finish_recording()?.as_bytes().to_vec())?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, 0_u32);
    let output = replay.replay(&trace).await?;
    assert_eq!(*replay.data(), 0);
    assert!(!output.instances().is_empty());
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_strings_realloc_and_post_return() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (import "host" (func $host (param "s" string) (result string)))
      (core module $alloc
        (memory (export "memory") 1)
        (global $next (mut i32) (i32.const 64))
        (func (export "realloc") (param i32 i32 i32 i32) (result i32)
          global.get $next
          global.get $next local.get 3 i32.add global.set $next))
      (core instance $a (instantiate $alloc))
      (core func $host (canon lower (func $host) (memory (core memory $a "memory")) (realloc (core func $a "realloc"))))
      (core module $run
        (import "" "host" (func $host (param i32 i32 i32)))
        (import "" "memory" (memory 1))
        (func (export "run") (param i32 i32) (result i32)
          local.get 0 local.get 1 i32.const 16 call $host i32.const 16)
        (func (export "post") (param i32)
          i32.const 32 i32.const 1 i32.store))
      (core instance $r (instantiate $run
        (with "" (instance (export "host" (func $host)) (export "memory" (memory $a "memory"))))))
      (func (export "run") (param "s" string) (result string)
        (canon lift (core func $r "run") (memory (core memory $a "memory"))
          (realloc (core func $a "realloc")) (post-return (core func $r "post")))))"#,
    )?;
    let mut linker = Linker::new(&e);
    linker
        .root()
        .func_wrap("host", |_, (s,): (String,)| Ok((format!("{s}!"),)))?;
    let instance = linker.instantiate(&mut store, &c)?;
    let run = instance.get_typed_func::<(&str,), (String,)>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ("héllo",))?, ("héllo!".to_owned(),));
    let trace = store.finish_recording()?;
    // Lowering records the bytes it writes, not the whole 64 KiB memory.
    let written = frames(trace.as_bytes())
        .windows(2)
        .filter(|f| f[0].0 == WRITE)
        .map(|f| f[1].1 - f[0].1 - 5 - 12)
        .sum::<usize>();
    assert!(written < 64, "{written} bytes of memory recorded");
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    let memory = output
        .instances()
        .iter()
        .find_map(|i| i.get_memory(&mut replay, "memory"))
        .unwrap();
    assert_eq!(&memory.data(&replay)[32..36], &1_i32.to_le_bytes());
    assert!(
        memory
            .data(&replay)
            .windows("héllo!".len())
            .any(|s| s == "héllo!".as_bytes())
    );
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_resources_and_guest_destructor_callback() -> Result<()> {
    use wasmtime::component::{Component, Linker, ResourceAny};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, 0);
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (import "notify" (func $notify (param "x" u32)))
      (core func $notify (canon lower (func $notify)))
      (core module $d
        (import "" "notify" (func $notify (param i32)))
        (global $count (export "count") (mut i32) (i32.const 0))
        (func (export "dtor") (param i32)
          local.get 0 call $notify
          global.get $count i32.const 1 i32.add global.set $count))
      (core instance $d (instantiate $d (with "" (instance (export "notify" (func $notify))))))
      (type $r (resource (rep i32) (dtor (core func $d "dtor"))))
      (export $exported-r "r" (type $r))
      (core func $new (canon resource.new $r))
      (core func $drop (canon resource.drop $r))
      (core module $run
        (import "" "new" (func $new (param i32) (result i32)))
        (import "" "drop" (func $drop (param i32)))
        (func (export "make") (result i32) i32.const 42 call $new)
        (func (export "run") i32.const 42 call $new call $drop))
      (core instance $run (instantiate $run
        (with "" (instance (export "new" (func $new)) (export "drop" (func $drop))))))
      (func (export "make") (result (own $exported-r)) (canon lift (core func $run "make")))
      (func (export "run") (canon lift (core func $run "run"))))"#,
    )?;
    let mut linker = Linker::new(&e);
    linker
        .root()
        .func_wrap("notify", |mut store, (x,): (u32,)| {
            assert_eq!(x, 42);
            *store.data_mut() += 1;
            Ok(())
        })?;
    let instance = linker.instantiate(&mut store, &c)?;
    instance
        .get_typed_func::<(), ()>(&mut store, "run")?
        .call(&mut store, ())?;
    let (resource,) = instance
        .get_typed_func::<(), (ResourceAny,)>(&mut store, "make")?
        .call(&mut store, ())?;
    resource.resource_drop(&mut store)?;
    assert_eq!(*store.data(), 2);
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, 0);
    let output = replay.replay(&trace).await?;
    assert_eq!(*replay.data(), 0);
    let count = output
        .instances()
        .iter()
        .find_map(|i| i.get_global(&mut replay, "count"))
        .unwrap();
    assert_eq!(count.get(&mut replay).unwrap_i32(), 2);
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_to_component_adapter_and_transcoding() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (component $a
        (core module $m
          (memory (export "memory") 1)
          (global $next (mut i32) (i32.const 64))
          (func (export "realloc") (param i32 i32 i32 i32) (result i32)
            local.get 0 if (result i32) local.get 0 else
              global.get $next
              global.get $next local.get 3 i32.add global.set $next
            end)
          (func (export "echo") (param i32 i32) (result i32)
            i32.const 16 local.get 0 i32.store
            i32.const 20 local.get 1 i32.store
            i32.const 16))
        (core instance $m (instantiate $m))
        (func (export "echo") (param "s" string) (result string)
          (canon lift (core func $m "echo") (memory (core memory $m "memory"))
            (realloc (core func $m "realloc")))))
      (component $b
        (import "echo" (func $echo (param "s" string) (result string)))
        (core module $alloc
          (memory (export "memory") 1)
          (global $next (mut i32) (i32.const 64))
          (func (export "realloc") (param i32 i32 i32 i32) (result i32)
            local.get 0 if (result i32) local.get 0 else
              global.get $next
              global.get $next local.get 3 i32.add global.set $next
            end))
        (core instance $alloc (instantiate $alloc))
        (core func $echo (canon lower (func $echo)
          (memory (core memory $alloc "memory")) (realloc (core func $alloc "realloc")) string-encoding=utf16))
        (core module $m
          (import "" "echo" (func $echo (param i32 i32 i32)))
          (func (export "run") (param i32 i32) (result i32)
            local.get 0 local.get 1 i32.const 16 call $echo i32.const 16))
        (core instance $m (instantiate $m (with "" (instance (export "echo" (func $echo))))))
        (func (export "run") (param "s" string) (result string)
          (canon lift (core func $m "run") (memory (core memory $alloc "memory"))
            (realloc (core func $alloc "realloc")) string-encoding=utf16)))
      (instance $a (instantiate $a))
      (instance $b (instantiate $b (with "echo" (func $a "echo"))))
      (export "run" (func $b "run")))"#,
    )?;
    let instance = Linker::new(&e).instantiate(&mut store, &c)?;
    let run = instance.get_typed_func::<(&str,), (String,)>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ("héllo 🌍",))?, ("héllo 🌍".into(),));
    let trace = Trace::from_bytes(store.finish_recording()?.as_bytes().to_vec())?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    assert!(output.instances().len() >= 3);
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_async_host_and_trap_during_startup() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (import "host" (func $host))
      (core func $host (canon lower (func $host)))
      (core module $m
        (import "" "host" (func $host))
        (func $start call $host unreachable)
        (start $start))
      (core instance $m (instantiate $m (with "" (instance (export "host" (func $host)))))))"#,
    )?;
    let mut linker = Linker::new(&e);
    linker.root().func_wrap_async("host", |_, (): ()| {
        Box::new(async {
            tokio::task::yield_now().await;
            Ok(())
        })
    })?;
    let error = linker.instantiate_async(&mut store, &c).await.unwrap_err();
    assert_eq!(
        error.downcast_ref::<Trap>(),
        Some(&Trap::UnreachableCodeReached)
    );
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn host_objects_created_during_initialization() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    // An explicit u64::MAX bound is distinct from an unbounded table.
    let wide = Table::new(
        &mut store,
        TableType::new64(RefType::FUNCREF, 0, Some(u64::MAX)),
        Ref::Func(None),
    )?;
    let memory = Memory::new(&mut store, MemoryType::new(1, Some(2)))?;
    memory.write(&mut store, 0, &3_i32.to_le_bytes())?;
    let global = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, Mutability::Const),
        Val::I32(7),
    )?;
    let host = Func::wrap(&mut store, || 10_i32);
    let function_global = Global::new(
        &mut store,
        GlobalType::new(ValType::FUNCREF, Mutability::Const),
        Val::FuncRef(Some(host)),
    )?;
    let table = Table::new(
        &mut store,
        TableType::new(RefType::FUNCREF, 1, Some(1)),
        Ref::Func(Some(host)),
    )?;
    let module = Module::new(
        &e,
        r#"(module
      (import "" "m" (memory 1 2))
      (import "" "g" (global i32))
      (import "" "fg" (global $fg funcref))
      (import "" "t" (table 1 1 funcref))
      (import "" "wide" (table i64 0 18446744073709551615 funcref))
      (type $f (func (result i32)))
      (func (export "run") (result i32)
        i32.const 0 global.get $fg table.set 0
        i32.const 0 call_indirect (type $f)
        global.get 0 i32.add i32.const 0 i32.load i32.add))"#,
    )?;
    let instance = Instance::new(
        &mut store,
        &module,
        &[
            memory.into(),
            global.into(),
            function_global.into(),
            table.into(),
            wide.into(),
        ],
    )?;
    assert_eq!(
        instance
            .get_typed_func::<(), i32>(&mut store, "run")?
            .call(&mut store, ())?,
        20
    );
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn concurrent_component_activations_can_finish_out_of_order() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let mut config = Config::new();
    config
        .rr(RRConfig::Recording)
        .wasm_component_model_async(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (type $t (func async (param "x" u32) (result u32)))
      (import "host" (func $host (type $t)))
      (core func $host (canon lower (func $host)))
      (core module $m
        (import "" "host" (func $host (param i32) (result i32)))
        (func (export "run") (param i32) (result i32) local.get 0 call $host))
      (core instance $m (instantiate $m (with "" (instance (export "host" (func $host))))))
      (func (export "run") (type $t) (canon lift (core func $m "run"))))"#,
    )?;
    let mut linker = Linker::new(&e);
    linker
        .root()
        .func_wrap_concurrent("host", |_, (x,): (u32,)| {
            Box::pin(async move {
                tokio::task::yield_now().await;
                if x == 2 {
                    tokio::task::yield_now().await;
                }
                Ok((x + 10,))
            })
        })?;
    let a = linker.instantiate_async(&mut store, &c).await?;
    let b = linker.instantiate_async(&mut store, &c).await?;
    let a = a.get_typed_func::<(u32,), (u32,)>(&mut store, "run")?;
    let b = b.get_typed_func::<(u32,), (u32,)>(&mut store, "run")?;
    store
        .run_concurrent(async |accessor| -> Result<()> {
            let (a, b) = tokio::join!(
                a.call_concurrent(accessor, (1,)),
                b.call_concurrent(accessor, (2,))
            );
            assert_eq!(a?, (11,));
            assert_eq!(b?, (12,));
            Ok(())
        })
        .await??;
    let trace = store.finish_recording()?;
    config.rr(RRConfig::Replaying);
    let mut replay = Store::new(&Engine::new(&config)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_adapter_callbacks() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let mut config = Config::new();
    config
        .rr(RRConfig::Recording)
        .wasm_component_model_async(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (type $t (func async (param "x" u32) (result u32)))
      (import "host" (func $host (type $t)))
      (component $a
        (type $t (func async (param "x" u32) (result u32)))
        (import "host" (func $host (type $t)))
        (core func $host (canon lower (func $host)))
        (core module $m
          (import "" "host" (func $host (param i32) (result i32)))
          (func (export "run") (param i32) (result i32) local.get 0 call $host))
        (core instance $m (instantiate $m (with "" (instance (export "host" (func $host))))))
        (func (export "run") (type $t) (canon lift (core func $m "run"))))
      (instance $a (instantiate $a (with "host" (func $host))))
      (instance $b (instantiate $a (with "host" (func $a "run"))))
      (export "run" (func $b "run")))"#,
    )?;
    let mut linker = Linker::new(&e);
    linker
        .root()
        .func_wrap_concurrent("host", |_, (x,): (u32,)| {
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok((x + 1,))
            })
        })?;
    let instance = linker.instantiate_async(&mut store, &c).await?;
    let run = instance.get_typed_func::<(u32,), (u32,)>(&mut store, "run")?;
    assert_eq!(run.call_async(&mut store, (41,)).await?, (42,));
    let trace = store.finish_recording()?;
    config.rr(RRConfig::Replaying);
    let mut replay = Store::new(&Engine::new(&config)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_stream_wait() -> Result<()> {
    use wasmtime::component::{Component, Linker, StreamAny};
    let mut config = Config::new();
    config
        .wasm_component_model_async(true)
        .rr(RRConfig::Recording);
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());
    store.start_recording()?;
    let component = Component::new(
        &engine,
        r#"
(component
    (type $s (stream u8))

    (core module $libc (memory (export "mem") 1))
    (core instance $libc (instantiate $libc))

    (core module $m
        (import "" "stream.new" (func $stream.new (result i64)))
        (import "" "task.return" (func $task.return))
        (import "" "waitable-set.new" (func $waitable-set.new (result i32)))
        (import "" "waitable.join" (func $waitable.join (param i32 i32)))
        (import "" "waitable-set.wait" (func $waitable-set.wait (param i32 i32) (result i32)))
        (import "" "waitable-set.drop" (func $waitable-set.drop (param i32)))
        (import "" "mem" (memory 1))

        (global $w (mut i32) (i32.const 0))

        (func (export "mk") (result i32)
            (local $r i32) (local $tmp i64)
            (local.set $tmp (call $stream.new))
            (local.set $r (i32.wrap_i64 (local.get $tmp)))
            (global.set $w (i32.wrap_i64 (i64.shr_u (local.get $tmp) (i64.const 32))))
            local.get $r
        )

        (func (export "run") (result i32)
            (local $ws i32)
            (local.set $ws (call $waitable-set.new))
            (call $waitable.join (global.get $w) (local.get $ws))
            (call $waitable-set.wait (local.get $ws) (i32.const 0))
            i32.const 3 ;; EVENT_STREAM_WRITE
            i32.ne
            if unreachable end

            (if (i32.ne (i32.load (i32.const 0)) (global.get $w))
              (then unreachable))
            (if (i32.ne (i32.load (i32.const 4)) (i32.const 1)) ;; DROPPED | (0 << 4)
              (then unreachable))

            call $task.return

            i32.const 0 ;; CALLBACK_CODE_EXIT
        )

        (func (export "cb") (param i32 i32 i32) (result i32) unreachable)
    )
    (core func $stream.new (canon stream.new $s))
    (core func $task.return (canon task.return))
    (core func $waitable-set.new (canon waitable-set.new))
    (core func $waitable.join (canon waitable.join))
    (core func $waitable-set.wait (canon waitable-set.wait (memory (core memory $libc "mem"))))
    (core func $waitable-set.drop (canon waitable-set.drop))
    (core instance $i (instantiate $m
        (with "" (instance
            (export "stream.new" (func $stream.new))
            (export "task.return" (func $task.return))
            (export "waitable-set.new" (func $waitable-set.new))
            (export "waitable.join" (func $waitable.join))
            (export "waitable-set.wait" (func $waitable-set.wait))
            (export "waitable-set.drop" (func $waitable-set.drop))
            (export "mem" (memory $libc "mem"))
        ))
    ))
    (func (export "mk") (result (stream u8))
        (canon lift (core func $i "mk")))
    (func (export "run") async
        (canon lift (core func $i "run") async (callback (core func $i "cb"))))
)
        "#,
    )?;
    let instance = Linker::new(&engine).instantiate(&mut store, &component)?;
    let mk = instance.get_typed_func::<(), (StreamAny,)>(&mut store, "mk")?;
    let run = instance.get_typed_func::<(), ()>(&mut store, "run")?;
    store
        .run_concurrent(async |store| {
            let (mut stream,) = mk.call_concurrent(store, ()).await?;
            tokio::try_join! {
                async {
                    run.call_concurrent(store, ()).await?;
                    wasmtime::error::Ok(())
                },
                async {
                    store.with(|store| stream.close(store))?;
                    wasmtime::error::Ok(())
                }
            }?;
            wasmtime::error::Ok(())
        })
        .await??;
    let trace = store.finish_recording()?;
    config.rr(RRConfig::Replaying);
    let mut replay = Store::new(&Engine::new(&config)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_lift_callback() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let mut config = Config::new();
    config
        .rr(RRConfig::Recording)
        .wasm_component_model_async(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (core func $return (canon task.return (result u32)))
      (core module $m
        (import "" "return" (func $return (param i32)))
        (global $calls (export "calls") (mut i32) (i32.const 0))
        (func (export "run") (result i32) i32.const 1)
        (func (export "cb") (param i32 i32 i32) (result i32)
          global.get $calls i32.const 1 i32.add global.set $calls
          i32.const 42 call $return i32.const 0))
      (core instance $m (instantiate $m (with "" (instance (export "return" (func $return))))))
      (func (export "run") async (result u32)
        (canon lift (core func $m "run") async (callback (core func $m "cb")))))"#,
    )?;
    let instance = Linker::new(&e).instantiate_async(&mut store, &c).await?;
    let run = instance.get_typed_func::<(), (u32,)>(&mut store, "run")?;
    assert_eq!(run.call_async(&mut store, ()).await?, (42,));
    let trace = store.finish_recording()?;
    config.rr(RRConfig::Replaying);
    let mut replay = Store::new(&Engine::new(&config)?, ());
    let output = replay.replay(&trace).await?;
    let calls = output.instances()[0]
        .get_global(&mut replay, "calls")
        .unwrap();
    assert_eq!(calls.get(&mut replay).unwrap_i32(), 1);
    Ok(())
}

#[tokio::test]
async fn initialization_inside_host_callback() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let inner = Module::new(
        &e,
        r#"(module
      (import "" "host" (func $host (result i32)))
      (global $g (export "g") (mut i32) (i32.const 0))
      (func $start call $host global.set $g) (start $start))"#,
    )?;
    let host = Func::wrap(
        &mut store,
        move |mut caller: Caller<'_, ()>| -> Result<i32> {
            let host = Func::wrap(&mut caller, || 42_i32);
            let instance = Instance::new(&mut caller, &inner, &[host.into()])?;
            Ok(instance
                .get_global(&mut caller, "g")
                .unwrap()
                .get(&mut caller)
                .unwrap_i32())
        },
    );
    let outer = Module::new(
        &e,
        r#"(module
      (import "" "host" (func $host (result i32)))
      (func (export "run") (result i32) call $host))"#,
    )?;
    let instance = Instance::new(&mut store, &outer, &[host.into()])?;
    assert_eq!(
        instance
            .get_typed_func::<(), i32>(&mut store, "run")?
            .call(&mut store, ())?,
        42
    );
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    assert_eq!(output.instances().len(), 2);
    assert_eq!(
        output.instances()[1]
            .get_global(&mut replay, "g")
            .unwrap()
            .get(&mut replay)
            .unwrap_i32(),
        42
    );
    Ok(())
}

#[tokio::test]
async fn initialization_requires_exactly_one_startup() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let module = Module::new(
        &e,
        r#"(module
      (global $g (mut i32) (i32.const 0))
      (func $start i32.const 1 global.set $g) (start $start))"#,
    )?;
    Instance::new(&mut store, &module, &[])?;
    let bytes = store.finish_recording()?.as_bytes().to_vec();
    // The start of the frame (not its body) of the startup activation.
    let startup = first_frame(&bytes, ENTER_WASM) - 5;
    // Truncate the complete startup activation, retaining a valid End frame.
    let mut missing = bytes[..startup].to_vec();
    missing.extend_from_slice(&[0; 5]);
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let error = replay
        .replay(&Trace::from_bytes(missing)?)
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("startup missing"));

    let mut repeated = bytes[..bytes.len() - 5].to_vec();
    repeated.extend_from_slice(&bytes[startup..]);
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let error = replay
        .replay(&Trace::from_bytes(repeated)?)
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("repeated instance startup"));
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn existing_component_without_core_instances_is_rejected() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let trace = store.finish_recording()?;
    for mode in [RRConfig::Recording, RRConfig::Replaying] {
        let e = engine(mode.clone())?;
        let mut store = Store::new(&e, ());
        let component = Component::new(&e, "(component)")?;
        Linker::new(&e).instantiate(&mut store, &component)?;
        let error = match mode {
            RRConfig::Recording => store.start_recording().unwrap_err(),
            RRConfig::Replaying => store.replay(&trace).await.unwrap_err(),
            _ => unreachable!(),
        };
        assert!(error.to_string().contains("empty store"));
    }
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_imported_core_module() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (import "host" (func $host (param "x" u32) (result u32)))
      (import "module" (core module $m
        (import "" "host" (func (param i32) (result i32)))
        (export "run" (func (param i32) (result i32)))))
      (core func $host (canon lower (func $host)))
      (core instance $m (instantiate $m (with "" (instance (export "host" (func $host))))))
      (func (export "run") (param "x" u32) (result u32) (canon lift (core func $m "run"))))"#,
    )?;
    let module = Module::new(
        &e,
        r#"(module
      (import "" "host" (func $host (param i32) (result i32)))
      (func (export "run") (param i32) (result i32) local.get 0 call $host))"#,
    )?;
    let mut linker = Linker::new(&e);
    linker.root().module("module", &module)?;
    linker
        .root()
        .func_wrap("host", |_, (x,): (u32,)| Ok((x + 1,)))?;
    let instance = linker.instantiate(&mut store, &c)?;
    let run = instance.get_typed_func::<(u32,), (u32,)>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, (41,))?, (42,));
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_without_core_instances() -> Result<()> {
    use wasmtime::component::{Component, Linker, ResourceAny};
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let c = Component::new(
        &e,
        r#"(component
      (type $r (resource (rep i32)))
      (export $r-export "r" (type $r))
      (core func $new (canon resource.new $r))
      (func (export "make") (param "x" u32) (result (own $r-export)) (canon lift (core func $new))))"#,
    )?;
    let instance = Linker::new(&e).instantiate(&mut store, &c)?;
    let make = instance.get_typed_func::<(u32,), (ResourceAny,)>(&mut store, "make")?;
    let (resource,) = make.call(&mut store, (42,))?;
    resource.resource_drop(&mut store)?;
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    assert!(replay.replay(&trace).await?.instances().is_empty());
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_lower_futures_and_cancellation() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let mut config = Config::new();
    config
        .rr(RRConfig::Recording)
        .wasm_component_model_async(true)
        .wasm_component_model_more_async_builtins(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    // The component-model suite's cancellation scenario exercises async
    // lowering, guest callbacks, future transfers, and subtask cancellation.
    let c = Component::new(&e, include_str!("rr_async_component.wat"))?;
    let instance = Linker::new(&e).instantiate_async(&mut store, &c).await?;
    let run = instance.get_typed_func::<(), (u32,)>(&mut store, "run")?;
    assert_eq!(run.call_async(&mut store, ()).await?, (42,));
    let trace = store.finish_recording()?;
    config.rr(RRConfig::Replaying);
    let mut replay = Store::new(&Engine::new(&config)?, ());
    replay.replay(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn function_references_cross_boundaries_by_id() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let exchange = Func::wrap(
        &mut store,
        |mut caller: Caller<'_, ()>, original: Option<Func>| -> Result<Option<Func>> {
            assert_eq!(
                original
                    .unwrap()
                    .typed::<(), i32>(&caller)?
                    .call(&mut caller, ())?,
                10
            );
            Ok(Some(Func::wrap(&mut caller, || 42_i32)))
        },
    );
    let module = Module::new(
        &e,
        r#"(module
      (import "" "exchange" (func $exchange (param funcref) (result funcref)))
      (table 1 funcref)
      (type $f (func (result i32)))
      (global $result (export "result") (mut i32) (i32.const 0))
      (func (export "run") (param funcref) (result funcref) (local $f funcref)
        local.get 0 call $exchange local.set $f
        i32.const 0 local.get $f table.set 0
        i32.const 0 call_indirect (type $f) global.set $result
        local.get $f))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[exchange.into()])?;
    let original = Func::wrap(&mut store, || 10_i32);
    let run = instance.get_typed_func::<Option<Func>, Option<Func>>(&mut store, "run")?;
    assert!(run.call(&mut store, Some(original))?.is_some());
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay(&trace).await?;
    assert_eq!(
        output.instances()[0]
            .get_global(&mut replay, "result")
            .unwrap()
            .get(&mut replay)
            .unwrap_i32(),
        42
    );

    // Raw pointers never enter the trace, and an invalid ID must fail before
    // entering guest code with a fabricated function reference.
    let mut bytes = trace.as_bytes().to_vec();
    // The first argument of the first EnterWasm, after the callee and call IDs.
    let body = first_frame(&bytes, ENTER_WASM);
    bytes[body + 8..body + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let error = replay.replay(&Trace::from_bytes(bytes)?).await.unwrap_err();
    assert!(format!("{error:#}").contains("invalid function id"));
    Ok(())
}

#[tokio::test]
async fn initialization_tracks_module_identity() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let wasm = "(module (func (export \"run\") (result i32) i32.const 1))";
    let module = Module::new(&e, wasm)?;
    let serialized = Module::new(&e, wasm)?.serialize()?;
    // SAFETY: these unchanged artifacts were just compiled by the same engine.
    let restored = unsafe { Module::deserialize(&e, &serialized)? };
    assert!(!Module::same(&module, &restored));
    assert_eq!(module.debug_bytecode(), restored.debug_bytecode());
    let different = Module::new(
        &e,
        "(module (func (export \"run\") (result i32) i32.const 2))",
    )?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    for (module, expected) in [
        (module.clone(), 1),
        (module, 1),
        (restored, 1),
        (different, 2),
    ] {
        let instance = Instance::new(&mut store, &module, &[])?;
        assert_eq!(
            instance
                .get_typed_func::<(), i32>(&mut store, "run")?
                .call(&mut store, ())?,
            expected
        );
    }
    let trace = store.finish_recording()?;
    let definitions = frames(trace.as_bytes())
        .iter()
        .filter(|(tag, _)| *tag == MODULE)
        .count();
    // A clone shares a definition; separately compiled modules have their own
    // trace entries even when they contain identical bytecode.
    assert_eq!(definitions, 3);
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    assert_eq!(replay.replay(&trace).await?.instances().len(), 4);
    Ok(())
}

#[tokio::test]
async fn initialization_preserves_import_aliases_across_instances() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || 17_i32);
    let memory = Memory::new(&mut store, MemoryType::new(1, None))?;
    let global = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, Mutability::Var),
        Val::I32(0),
    )?;
    let table = Table::new(
        &mut store,
        TableType::new(RefType::FUNCREF, 1, None),
        Ref::Func(None),
    )?;
    let module = Module::new(
        &e,
        r#"(module
            (type $f (func (result i32)))
            (import "" "host" (func $host (type $f)))
            (import "" "memory" (memory 1))
            (import "" "global" (global $g (mut i32)))
            (import "" "table" (table 1 funcref))
            (elem declare func $host)
            (func (export "write")
                i32.const 0 i32.const 23 i32.store
                i32.const 5 global.set $g
                i32.const 0 ref.func $host table.set)
            (func (export "read") (result i32)
                i32.const 0 i32.load
                global.get $g i32.add
                i32.const 0 call_indirect (type $f) i32.add))"#,
    )?;
    let imports = [host.into(), memory.into(), global.into(), table.into()];
    let a = Instance::new(&mut store, &module, &imports)?;
    let b = Instance::new(&mut store, &module, &imports)?;
    a.get_typed_func::<(), ()>(&mut store, "write")?
        .call(&mut store, ())?;
    assert_eq!(
        b.get_typed_func::<(), i32>(&mut store, "read")?
            .call(&mut store, ())?,
        45
    );
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    assert_eq!(replay.replay(&trace).await?.instances().len(), 2);
    Ok(())
}

/// A guest that counts in a global and in memory, grows its memory and table
/// part-way, and reports each step through a host function that records an
/// event.
const COUNTER: &str = r#"(module
  (import "" "report" (func $report (param i32)))
  (memory (export "memory") 1)
  (table (export "table") 1 funcref)
  (global $count (export "count") (mut i32) (i32.const 0))
  (func (export "run") (param $n i32) (local $i i32)
    (loop $l
      (global.set $count (i32.add (global.get $count) (i32.const 1)))
      (i32.store (i32.mul (local.get $i) (i32.const 4)) (global.get $count))
      (if (i32.eq (local.get $i) (i32.const 2))
        (then
          (drop (memory.grow (i32.const 1)))
          (drop (table.grow (ref.null func) (i32.const 3)))))
      (call $report (local.get $i))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $l (i32.lt_u (local.get $i) (local.get $n))))))"#;

#[derive(Debug, Clone, PartialEq, serde_derive::Serialize, serde_derive::Deserialize)]
struct Step(i32);

impl rr::TraceEvent for Step {
    const TAG: u32 = 2;
}

fn record_counter(steps: i32) -> Result<rr::Trace> {
    let recording = engine(RRConfig::Recording)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let report = Func::wrap(&mut store, |mut caller: Caller<'_, ()>, i: i32| {
        rr::record_event(&mut caller, &Step(i))
    });
    let module = Module::new(&recording, COUNTER)?;
    let instance = Instance::new(&mut store, &module, &[report.into()])?;
    let run = instance.get_typed_func::<i32, ()>(&mut store, "run")?;
    run.call(&mut store, steps)?;
    store.finish_recording()
}

/// The counter, memory size, and table size of the counter instance.
fn counter_state<T: Send>(replayer: &mut rr::Replayer<'_, T>) -> (i32, usize, u64, Vec<u8>) {
    let instance = replayer.instances()[0];
    let mut store = replayer.store();
    let count = instance
        .get_global(&mut store, "count")
        .unwrap()
        .get(&mut store)
        .unwrap_i32();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let table = instance.get_table(&mut store, "table").unwrap();
    (
        count,
        memory.data_size(&store),
        table.size(&store),
        memory.data(&store)[..32].to_vec(),
    )
}

#[tokio::test]
async fn checkpoints_rewind_guest_state_and_events() -> Result<()> {
    let trace = record_counter(6)?;
    let mut store = Store::new(&engine(RRConfig::Replaying)?, ());
    let mut replayer = store.replayer(&trace)?;
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = seen.clone();
    replayer.on_event(move |Step(i)| observed.lock().unwrap().push(i));
    replayer.stop_at_events(true);

    let initial = replayer.checkpoint()?;
    // Stop at the reports of steps 0 and 3: before and after growth, each
    // with the guest parked in the middle of a host call.
    let mut checkpoints = Vec::new();
    let mut states = Vec::new();
    while let rr::ReplayStop::Event(_) = replayer.run().await? {
        let step = *seen.lock().unwrap().last().unwrap();
        if step == 0 || step == 3 {
            checkpoints.push(replayer.checkpoint()?);
            states.push(counter_state(&mut replayer));
        }
    }
    let end = counter_state(&mut replayer);
    assert_eq!(end.0, 6);
    assert_eq!((end.1, end.2), (2 << 16, 4));
    assert_eq!(states[0].0, 1);
    assert_eq!((states[0].1, states[0].2), (1 << 16, 1));
    assert_eq!(states[1].0, 4);
    assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3, 4, 5]);

    // Rewind after the end, to before growth; then jump forward past growth,
    // back again, and replay to the end from each.
    for &which in &[0, 1, 0, 1] {
        replayer.restore(&checkpoints[which])?;
        assert_eq!(counter_state(&mut replayer), states[which]);
        seen.lock().unwrap().clear();
        replayer.stop_at_events(false);
        assert_eq!(replayer.run().await?, rr::ReplayStop::Finished);
        replayer.stop_at_events(true);
        let expected: Vec<i32> = if which == 0 {
            (1..6).collect()
        } else {
            (4..6).collect()
        };
        assert_eq!(*seen.lock().unwrap(), expected);
        assert_eq!(counter_state(&mut replayer), end);
    }

    // Restart from before any object existed.
    replayer.restore(&initial)?;
    assert!(replayer.instances().is_empty());
    seen.lock().unwrap().clear();
    replayer.stop_at_events(false);
    assert_eq!(replayer.run().await?, rr::ReplayStop::Finished);
    assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3, 4, 5]);
    assert_eq!(counter_state(&mut replayer), end);

    // Checkpoints only restore into their own replay.
    drop(replayer);
    let mut other = Store::new(store.engine(), ());
    let mut other = other.replayer(&trace)?;
    assert!(other.restore(&checkpoints[0]).is_err());
    Ok(())
}

#[cfg(feature = "debug")]
fn debug_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.rr(RRConfig::Replaying).guest_debug(true);
    Engine::new(&config)
}

/// Where the counter guest is stopped: its function, PC, loop index, and the
/// replay's view of the counter.
#[cfg(feature = "debug")]
fn position<T: Send>(replayer: &mut rr::Replayer<'_, T>) -> Result<(u32, u32, i32, i32)> {
    let frames = replayer.debug_exit_frames();
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    let instance = replayer.instances()[0];
    let mut store = replayer.store();
    assert!(frame.parent(&mut store)?.is_none());
    let (func, pc) = frame.wasm_function_index_and_pc(&mut store)?.unwrap();
    let i = frame.local(&mut store, 1)?.unwrap_i32();
    let count = instance
        .get_global(&mut store, "count")
        .unwrap()
        .get(&mut store)
        .unwrap_i32();
    Ok((func.as_u32(), pc.raw(), i, count))
}

#[cfg(feature = "debug")]
#[tokio::test]
async fn reversible_debugging_on_replay() -> Result<()> {
    let trace = record_counter(4)?;
    let mut store = Store::new(&debug_engine()?, ());
    let mut replayer = store.replayer(&trace)?;
    let initial = replayer.checkpoint()?;
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = seen.clone();
    replayer.on_event(move |Step(i)| observed.lock().unwrap().push(i));

    // Run until the guest first reports, then single-step it to the end,
    // recording every stop.
    replayer.stop_at_events(true);
    assert_eq!(replayer.run().await?, rr::ReplayStop::Event(2));
    replayer.stop_at_events(false);
    replayer
        .store()
        .edit_breakpoints()
        .unwrap()
        .single_step(true)?;
    let mut trail = Vec::new();
    let mut checkpoints = Vec::new();
    while replayer.run().await? == rr::ReplayStop::Breakpoint {
        checkpoints.push(replayer.checkpoint()?);
        trail.push(position(&mut replayer)?);
    }
    assert!(trail.len() > 20, "{} steps", trail.len());
    assert_eq!(trail.last().unwrap().3, 4);
    assert!(trail.windows(2).all(|w| w[0].2 <= w[1].2));
    assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3]);

    // Step backwards from the end, one stop at a time, by restoring the
    // previous stop's checkpoint: the guest is exactly where it was.
    for (k, checkpoint) in checkpoints.iter().enumerate().rev() {
        replayer.restore(checkpoint)?;
        assert_eq!(position(&mut replayer)?, trail[k]);
    }

    // From the middle, stepping forward again retraces the same path and
    // re-delivers the same events.
    let middle = trail.len() / 2;
    replayer.restore(&checkpoints[middle])?;
    seen.lock().unwrap().clear();
    let mut again = Vec::new();
    while replayer.run().await? == rr::ReplayStop::Breakpoint {
        again.push(position(&mut replayer)?);
    }
    assert_eq!(again, trail[middle + 1..]);
    let reports = trail[middle + 1..]
        .windows(2)
        .filter(|w| w[0].2 != w[1].2)
        .count();
    assert_eq!(seen.lock().unwrap().len(), reports);

    // Breakpoints persist across a restore to the very beginning: with a
    // breakpoint at one PC in the loop, replay stops there once per
    // iteration.
    let (_, pc, _, _) = trail[0];
    replayer.restore(&initial)?;
    assert!(replayer.instances().is_empty());
    assert_eq!(replayer.run().await?, rr::ReplayStop::Breakpoint);
    let instance = replayer.instances()[0];
    let module = instance.module(replayer.store()).clone();
    {
        let mut breakpoints = replayer.store().edit_breakpoints().unwrap();
        breakpoints.single_step(false)?;
        breakpoints.add_breakpoint(&module, ModulePC::new(pc))?;
    }
    let mut hits = Vec::new();
    while replayer.run().await? == rr::ReplayStop::Breakpoint {
        hits.push(position(&mut replayer)?);
    }
    assert!(hits.iter().all(|h| h.1 == pc));
    let iterations = hits.iter().map(|h| h.2).collect::<Vec<_>>();
    assert!(iterations.windows(2).all(|w| w[0] < w[1]));
    replayer.restore(&initial)?;
    let mut from_start = Vec::new();
    while replayer.run().await? == rr::ReplayStop::Breakpoint {
        from_start.push(position(&mut replayer)?);
    }
    assert_eq!(from_start.len(), 4);
    assert!(from_start.ends_with(&hits));
    Ok(())
}
