use std::sync::Mutex;
use wasmtime::{rr::Trace, *};

fn engine(mode: RRConfig) -> Result<Engine> {
    let mut config = Config::new();
    config.rr(mode);
    Engine::new(&config)
}

/// The bytecode of every module these tests compile. Traces refer to modules
/// only by checksum, so replays are given all of it, as they might be given
/// a directory of a program's modules.
static BYTECODE: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

fn register_bytecode(wasm: &[u8]) {
    let mut bytecode = BYTECODE.lock().unwrap();
    if !bytecode.iter().any(|b| b == wasm) {
        bytecode.push(wasm.to_vec());
    }
}

fn new_module(engine: &Engine, wasm: impl AsRef<[u8]>) -> Result<Module> {
    register_bytecode(wasm.as_ref());
    wasmtime::Module::new(engine, wasm)
}

/// Replays with the bytecode of every module these tests compile.
trait ReplayAll<T: Send + 'static> {
    async fn replay_all(&mut self, trace: &Trace) -> Result<rr::Replay>;
}

impl<T: Send + 'static> ReplayAll<T> for Store<T> {
    async fn replay_all(&mut self, trace: &Trace) -> Result<rr::Replay> {
        let bytecode = BYTECODE.lock().unwrap().clone();
        let bytecode = bytecode.iter().map(|b| &b[..]).collect::<Vec<_>>();
        self.replay(trace, &bytecode).await
    }
}

// Trace frame tags, for tests that edit traces (see `rr/codec.rs`).
const ENTER_WASM: u8 = 1;
const LEAVE_HOST: u8 = 4;
const WRITE: u8 = 5;
const MODULE: u8 = 8;

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
    let module = new_module(&engine, NESTED)?;
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
    let output = replay.replay_all(&trace).await?;
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
    let module = new_module(
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
    let output = replay.replay_all(&trace).await?;
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
    assert!(replay.replay_all(&trace).await?.instances().is_empty());
    assert_eq!(*replay.data(), 456);

    let constructors: &[fn(&mut Store<i32>) -> Result<()>] = &[
        |store| {
            Func::wrap(store, || ());
            Ok(())
        },
        |store| {
            Memory::new(store, MemoryType::new(0, None)?)?;
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
                TableType::new(RefType::FUNCREF, 0, None)?,
                Ref::Func(None),
            )?;
            Ok(())
        },
        |store| {
            let module = new_module(store.engine(), "(module)")?;
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
                .replay_all(&trace)
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
        TableType::new(RefType::FUNCREF, 1, None)?,
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
    let output = replay.replay_all(&trace).await?;
    let replay_memory = output.instances()[0]
        .get_memory(&mut replay, "memory")
        .unwrap();
    assert_eq!(replay_memory.data(&replay), memory.data(&store));
    Ok(())
}

#[tokio::test]
async fn memory_ranges_record_only_what_they_borrow() -> Result<()> {
    let (mut store, _, memory) = identity()?;
    memory
        .data_range_mut(&mut store, 100..104)?
        .copy_from_slice(b"abcd");
    assert!(memory.data_range_mut(&mut store, 65530..65540).is_err());
    {
        let (mut ranges, ()) = memory.data_ranges_and_store_data_mut(&mut store);
        assert_eq!(ranges.len(), 65536);
        assert_eq!(ranges.data_range(100..104)?, b"abcd");
        ranges.data_range_mut(2000..2002)?.copy_from_slice(b"xy");
        ranges.data_range_mut(5000..5001)?[0] = 7;
        assert!(ranges.data_range_mut(65536..65537).is_err());
    }
    let trace = store.finish_recording()?;
    // Only the three borrowed ranges were recorded, not all of memory.
    let writes = frames(trace.as_bytes())
        .into_iter()
        .filter(|(tag, _)| *tag == WRITE)
        .count();
    assert_eq!(writes, 3);
    assert!(trace.as_bytes().len() < 1000, "{}", trace.as_bytes().len());

    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let output = replay.replay_all(&trace).await?;
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

/// The bits of `f32.add`'s and relaxed `madd`'s results for a non-canonical
/// NaN, `0x7fa00000`, under `config`.
fn nan_results(config: &Config) -> Result<[u32; 2]> {
    let engine = Engine::new(config)?;
    let module = Module::new(
        &engine,
        r#"
            (module
                (func (export "add") (param i32) (result i32)
                    (i32.reinterpret_f32
                        (f32.add (f32.reinterpret_i32 (local.get 0)) (f32.const 0))))
                (func (export "madd") (param i32) (result i32)
                    (local v128)
                    (local.set 1 (f32x4.splat (f32.reinterpret_i32 (local.get 0))))
                    (i32.reinterpret_f32
                        (f32x4.extract_lane 0
                            (f32x4.relaxed_madd (local.get 1) (local.get 1) (local.get 1))))))
        "#,
    )?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let mut results = [0; 2];
    for (result, name) in results.iter_mut().zip(["add", "madd"]) {
        let func = instance.get_typed_func::<u32, u32>(&mut store, name)?;
        *result = func.call(&mut store, 0x7fa0_0000)?;
    }
    Ok(results)
}

#[test]
fn rr_config_canonicalizes_nans_regardless_of_option_order() -> Result<()> {
    let mut config = Config::new();
    config.wasm_relaxed_simd(true).rr(RRConfig::Recording);
    // Resetting the compiler configuration does not lose the setting.
    config.enable_compiler(false).enable_compiler(true);
    assert_eq!(nan_results(&config)?, [0x7fc0_0000; 2]);
    Ok(())
}

#[test]
#[cfg(target_arch = "x86_64")]
fn fma_builtin_results_match_native_fma() -> Result<()> {
    let mut config = Config::new();
    config.wasm_relaxed_simd(true).rr(RRConfig::Recording);
    // SAFETY: disabling a CPU feature only makes code use fallbacks.
    unsafe {
        config.cranelift_flag_set("has_fma", "false");
    }
    assert_eq!(nan_results(&config)?, [0x7fc0_0000; 2]);
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
    assert!(replay.replay_all(&trace).await.is_err());
    // Cleanup released the store, including TLS and both parked stacks.
    let memory = Memory::new(&mut replay, MemoryType::new(1, None)?)?;
    memory.write(&mut replay, 0, &[42])?;
    assert_eq!(memory.data(&replay)[0], 42);
    Ok(())
}

#[tokio::test]
async fn trapping_and_failing_activations_finish_by_yielding() -> Result<()> {
    trapping_and_failing_activations(engine(RRConfig::Recording)?).await
}

/// Pulley's trampolines record like native ones, and its recordings replay
/// natively.
#[cfg(feature = "pulley")]
#[tokio::test]
async fn pulley_recordings_replay_natively() -> Result<()> {
    let mut config = Config::new();
    config.target("pulley64")?.rr(RRConfig::Recording);
    trapping_and_failing_activations(Engine::new(&config)?).await
}

async fn trapping_and_failing_activations(recording: Engine) -> Result<()> {
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, |x: i32| -> Result<i32> {
        if x % 2 == 0 {
            wasmtime::bail!("even argument {x}")
        }
        Ok(x)
    });
    let module = new_module(
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
    replay.replay_all(&trace).await?;
    // Every activation's state was released.
    let memory = Memory::new(&mut replay, MemoryType::new(1, None)?)?;
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

#[tokio::test]
async fn replay_detects_divergent_results() -> Result<()> {
    let recording = engine(RRConfig::Recording)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || 1_i32);
    let module = new_module(
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
    let error = replay
        .replay_all(&Trace::from_bytes(bytes)?)
        .await
        .unwrap_err();
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
    assert!(recording.replay_all(&trace).await.is_err());
    Store::new(replaying.engine(), ())
        .replay_all(&trace)
        .await?;
    Ok(())
}

fn many_calls() -> Result<(Store<()>, Func, Func)> {
    let engine = engine(RRConfig::Recording)?;
    let mut store = Store::new(&engine, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || ());
    let module = new_module(
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
    let mut future = Box::pin(replay.replay_all(&trace));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ));
    drop(future);
    assert!(
        replay
            .replay_all(&trace)
            .await
            .unwrap_err()
            .to_string()
            .contains("empty store")
    );
    let mut fresh = Store::new(replay.engine(), ());
    fresh.replay_all(&trace).await?;
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
        let module = new_module(
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
    replay.replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn host_panic_ends_the_trace_with_the_call_unfinished() -> Result<()> {
    let recording = engine(RRConfig::Recording)?;
    let mut store = Store::new(&recording, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || -> () {
        panic!("intentional recording test panic")
    });
    let module = new_module(
        &recording,
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
    // The trace replays up to the panic, leaving the guest suspended in the
    // host call.
    let trace = store.finish_recording()?;
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    replay.replay_all(&trace).await?;
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
    let module = new_module(
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
        let memory = Memory::new(&mut store, MemoryType::new(1, Some(2))?)?;
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
        let module = new_module(
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
    let output = replay.replay_all(&trace).await?;
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
        let module = new_module(
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
        .replay_all(&trace)
        .await?;
    Ok(())
}

#[tokio::test]
async fn segment_drops_are_replayed() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let module = new_module(
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
        .replay_all(&trace)
        .await?;
    Ok(())
}

#[tokio::test]
async fn direct_host_calls_only_record_their_guest_effects() -> Result<()> {
    fn setup() -> Result<(Store<()>, Func, Memory)> {
        let engine = engine(RRConfig::Recording)?;
        let mut store = Store::new(&engine, ());
        store.start_recording()?;
        let module = new_module(
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
    let output = replay.replay_all(&trace).await?;
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
    let module = new_module(
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
    let output = replay.replay_all(&trace).await?;
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

#[tokio::test]
async fn host_objects_created_during_initialization() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    // An explicit u64::MAX bound is distinct from an unbounded table.
    let wide = Table::new(
        &mut store,
        TableType::new64(RefType::FUNCREF, 0, Some(u64::MAX))?,
        Ref::Func(None),
    )?;
    let memory = Memory::new(&mut store, MemoryType::new(1, Some(2))?)?;
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
        TableType::new(RefType::FUNCREF, 1, Some(1))?,
        Ref::Func(Some(host)),
    )?;
    let module = new_module(
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
    replay.replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn initialization_inside_host_callback() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let inner = new_module(
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
    let outer = new_module(
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
    let output = replay.replay_all(&trace).await?;
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
    let module = new_module(
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
        .replay_all(&Trace::from_bytes(missing)?)
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("startup missing"));

    let mut repeated = bytes[..bytes.len() - 5].to_vec();
    repeated.extend_from_slice(&bytes[startup..]);
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let error = replay
        .replay_all(&Trace::from_bytes(repeated)?)
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("repeated instance startup"));
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
    let module = new_module(
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
    let output = replay.replay_all(&trace).await?;
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
    // The first argument of the first EnterWasm with arguments (after the
    // instance startups), after the callee and call IDs.
    let body = frames(&bytes)
        .windows(2)
        .find(|f| f[0].0 == ENTER_WASM && f[1].1 - f[0].1 > 8 + 5)
        .unwrap()[0]
        .1;
    bytes[body + 8..body + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    let error = replay
        .replay_all(&Trace::from_bytes(bytes)?)
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("invalid function id"));
    Ok(())
}

#[tokio::test]
async fn initialization_tracks_module_identity() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let wasm = "(module (func (export \"run\") (result i32) i32.const 1))";
    let module = new_module(&e, wasm)?;
    let serialized = new_module(&e, wasm)?.serialize()?;
    // SAFETY: these unchanged artifacts were just compiled by the same engine.
    let restored = unsafe { Module::deserialize(&e, &serialized)? };
    assert!(!Module::same(&module, &restored));
    let different_wasm = "(module (func (export \"run\") (result i32) i32.const 2))";
    let different = new_module(&e, different_wasm)?;
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
    // trace entries even when they contain identical bytecode, which the
    // trace identifies by the same checksum.
    assert_eq!(definitions, 3);
    let checksums = [wasm, different_wasm]
        .iter()
        .map(|wasm| Ok(rr::core_modules(&e, wasm.as_bytes())?[0].0))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(trace.module_checksums()?, checksums);
    let mut replay = Store::new(&engine(RRConfig::Replaying)?, ());
    assert_eq!(replay.replay_all(&trace).await?.instances().len(), 4);
    Ok(())
}

#[tokio::test]
async fn replay_compiles_modules_from_provided_bytecode() -> Result<()> {
    let wasm = "(module (func (export \"run\") (result i32) i32.const 7))";
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let instance = Instance::new(&mut store, &wasmtime::Module::new(&e, wasm)?, &[])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ())?, 7);
    let trace = store.finish_recording()?;
    let replaying = engine(RRConfig::Replaying)?;

    // Traces refer to modules by checksum only.
    let [(checksum, binary)] = &rr::core_modules(&e, wasm.as_bytes())?[..] else {
        panic!("one core module")
    };
    assert_eq!(trace.module_checksums()?, [*checksum]);
    assert!(trace.as_bytes().len() < binary.len() + 128);

    // Without the module's bytecode, replay fails when it reaches the
    // module's instantiation; unrelated bytecode is ignored.
    let unrelated = "(module (func (export \"run\") (result i32) i32.const 8))";
    let error = Store::new(&replaying, ())
        .replay(&trace, &[unrelated.as_bytes()])
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains(&rr::checksum_hex(checksum)),
        "{error:#}"
    );

    // Bytecode can be provided in either format.
    for wasm in [wasm.as_bytes(), binary] {
        let mut replay = Store::new(&replaying, ());
        let output = replay.replay(&trace, &[unrelated.as_bytes(), wasm]).await?;
        assert_eq!(output.instances().len(), 1);
    }
    Ok(())
}

#[tokio::test]
async fn initialization_preserves_import_aliases_across_instances() -> Result<()> {
    let e = engine(RRConfig::Recording)?;
    let mut store = Store::new(&e, ());
    store.start_recording()?;
    let host = Func::wrap(&mut store, || 17_i32);
    let memory = Memory::new(&mut store, MemoryType::new(1, None)?)?;
    let global = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, Mutability::Var),
        Val::I32(0),
    )?;
    let table = Table::new(
        &mut store,
        TableType::new(RefType::FUNCREF, 1, None)?,
        Ref::Func(None),
    )?;
    let module = new_module(
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
    assert_eq!(replay.replay_all(&trace).await?.instances().len(), 2);
    Ok(())
}
