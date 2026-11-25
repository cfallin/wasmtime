use std::sync::Mutex;
use wasmtime::{
    rr::{ReplayStop, Replayer, Trace},
    *,
};

fn engine(mode: RrConfig) -> Result<Engine> {
    let mut config = Config::new();
    config.rr(Some(mode));
    Engine::new(&config)
}

/// The bytecode of every module these tests compile. Traces refer to modules
/// only by checksum, so replays are given all of it, as they might be given
/// a directory of a program's modules.
static BYTECODE: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

fn new_module(engine: &Engine, wasm: &str) -> Result<Module> {
    let mut bytecode = BYTECODE.lock().unwrap();
    if !bytecode.iter().any(|b| b == wasm.as_bytes()) {
        bytecode.push(wasm.as_bytes().to_vec());
    }
    drop(bytecode);
    Module::new(engine, wasm)
}

/// Starts a replay with the bytecode of every module these tests compile.
fn new_replayer(trace: &Trace) -> Result<Replayer> {
    let mut replayer = Replayer::new(&engine(RrConfig::Replaying)?, trace.clone())?;
    for wasm in BYTECODE.lock().unwrap().iter() {
        replayer.add_bytecode(wasm)?;
    }
    Ok(replayer)
}

/// Replays an entire trace.
async fn replay_all(trace: &Trace) -> Result<Replayer> {
    let mut replayer = new_replayer(trace)?;
    assert_eq!(replayer.run().await?, ReplayStop::Finished);
    Ok(replayer)
}

fn identity() -> Result<(Store<()>, Func, Memory)> {
    let engine = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&engine, ())?;
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

#[test]
fn truncation_and_trailing_data_are_rejected() -> Result<()> {
    // Creating a trace only checks its version; reading it checks its framing.
    let read = |bytes: &[u8]| Trace::from_bytes(bytes.to_vec())?.module_checksums();
    let (mut store, _, _) = identity()?;
    let trace = store.finish_recording()?;
    assert_eq!(read(trace.as_bytes())?.len(), 1);
    for len in 0..trace.as_bytes().len() {
        assert!(read(&trace.as_bytes()[..len]).is_err());
    }
    let mut extra = trace.as_bytes().to_vec();
    extra.push(0);
    assert!(read(&extra).is_err());
    assert!(Trace::from_bytes(b"not a trace".to_vec()).is_err());
    Ok(())
}

#[test]
fn recording_and_replay_require_compatible_engines() -> Result<()> {
    let store = Store::new_recording(&engine(RrConfig::Recording)?, 123)?;
    assert_eq!(*store.data(), 123);

    for engine in [Engine::default(), engine(RrConfig::Replaying)?] {
        let err = Store::new_recording(&engine, ()).err().unwrap().to_string();
        assert!(err.contains("RrConfig::Recording"), "{err}");
    }

    let trace = Store::new_recording(&engine(RrConfig::Recording)?, ())?.finish_recording()?;
    for engine in [Engine::default(), engine(RrConfig::Recording)?] {
        let err = Replayer::new(&engine, trace.clone()).err().unwrap();
        assert!(err.to_string().contains("RrConfig::Replaying"), "{err}");
    }
    Ok(())
}

#[test]
fn artifacts_are_specific_to_the_rr_config() -> Result<()> {
    let modes = [None, Some(RrConfig::Recording), Some(RrConfig::Replaying)];
    for compile in modes {
        let artifact =
            Engine::new(Config::new().rr(compile))?.precompile_module(b"\0asm\x01\0\0\0")?;
        for load in modes {
            let engine = Engine::new(Config::new().rr(load))?;
            // SAFETY: the artifact was just compiled by Wasmtime.
            let result = unsafe { Module::deserialize(&engine, &artifact) };
            assert_eq!(result.is_ok(), compile == load, "{compile:?} {load:?}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn non_nullable_function_references_cross_boundaries() -> Result<()> {
    const MODULE: &str = r#"
        (module
            (import "" "host" (func $host (param (ref func)) (result (ref func))))
            (elem declare func $run)
            (func $run (export "run") (result (ref func))
                (call $host (ref.func $run))))
    "#;
    let mut config = Config::new();
    config
        .wasm_function_references(true)
        .rr(Some(RrConfig::Recording));
    let engine = Engine::new(&config)?;
    let mut store = Store::new_recording(&engine, ())?;
    let ty = ValType::Ref(RefType::new(false, HeapType::Func));
    let host = Func::new(
        &mut store,
        FuncType::new(&engine, [ty.clone()], [ty]),
        |_, args, results| {
            results[0] = args[0];
            Ok(())
        },
    );
    let module = Module::new(&engine, MODULE)?;
    let instance = Instance::new(&mut store, &module, &[host.into()])?;
    let run = instance.get_func(&mut store, "run").unwrap();
    let mut results = [Val::I32(0)];
    run.call(&mut store, &[], &mut results)?;
    assert!(matches!(results[0], Val::FuncRef(Some(_))));
    let trace = store.finish_recording()?;

    config.rr(Some(RrConfig::Replaying));
    let mut replay = Replayer::new(&Engine::new(&config)?, trace)?;
    replay.add_bytecode(MODULE.as_bytes())?;
    assert_eq!(replay.run().await?, ReplayStop::Finished);
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

#[test]
fn rr_config_enables_determinism_but_rejects_explicit_conflicts() -> Result<()> {
    engine(RrConfig::Recording)?;
    engine(RrConfig::Replaying)?;
    let mut config = Config::new();
    config
        .cranelift_nan_canonicalization(false)
        .rr(Some(RrConfig::Recording));
    assert!(Engine::new(&config).is_err());
    let mut config = Config::new();
    config
        .rr(Some(RrConfig::Replaying))
        .relaxed_simd_deterministic(false);
    assert!(Engine::new(&config).is_err());
    Ok(())
}

/// The bits of `f32.add`'s and relaxed `madd`'s results for a non-canonical
/// NaN, `0x7fa00000`, under `config`.
fn nan_results(config: &Config) -> Result<[u32; 2]> {
    let engine = Engine::new(config)?;
    let module = new_module(
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
    config.wasm_relaxed_simd(true).rr(Some(RrConfig::Recording));
    // Resetting the compiler configuration does not lose the setting.
    config.enable_compiler(false).enable_compiler(true);
    assert_eq!(nan_results(&config)?, [0x7fc0_0000; 2]);
    Ok(())
}

#[test]
#[cfg(target_arch = "x86_64")]
fn fma_builtin_results_match_native_fma() -> Result<()> {
    let mut config = Config::new();
    config.wasm_relaxed_simd(true).rr(Some(RrConfig::Recording));
    // SAFETY: disabling a CPU feature only makes code use fallbacks.
    unsafe {
        config.cranelift_flag_set("has_fma", "false");
    }
    assert_eq!(nan_results(&config)?, [0x7fc0_0000; 2]);
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

#[cfg(feature = "gc")]
#[test]
fn gc_reference_boundaries_are_rejected() -> Result<()> {
    let engine = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&engine, ())?;
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

/// A store recording with `config`, after `setup`, and a module whose export
/// `spin` loops as many times as its argument says.
///
/// Instantiation runs Wasm (record/replay engines give every module a startup
/// function), so `setup` provides fuel or an epoch deadline beforehand.
fn spinner(
    config: &mut Config,
    setup: impl FnOnce(&mut Store<()>) -> Result<()>,
) -> Result<(Store<()>, Module)> {
    config.rr(Some(RrConfig::Recording));
    let engine = Engine::new(config)?;
    let mut store = Store::new_recording(&engine, ())?;
    setup(&mut store)?;
    let module = new_module(
        &engine,
        r#"(module
        (func (export "spin") (param i32)
            (loop
                (local.set 0 (i32.sub (local.get 0) (i32.const 1)))
                (br_if 0 (local.get 0)))))"#,
    )?;
    Ok((store, module))
}

fn spin(store: &mut Store<()>, module: &Module) -> Result<TypedFunc<i32, ()>> {
    let instance = Instance::new(&mut *store, module, &[])?;
    instance.get_typed_func(store, "spin")
}

async fn spin_async(store: &mut Store<()>, module: &Module) -> Result<TypedFunc<i32, ()>> {
    let instance = Instance::new_async(&mut *store, module, &[]).await?;
    instance.get_typed_func(store, "spin")
}

/// Runs `future` to completion, returning its output and how many times it
/// yielded.
async fn count_yields<T>(future: impl Future<Output = T>) -> (T, usize) {
    let mut future = std::pin::pin!(future);
    let mut yields = 0;
    let output = std::future::poll_fn(|cx| {
        let poll = future.as_mut().poll(cx);
        if poll.is_pending() {
            yields += 1;
        }
        poll
    })
    .await;
    (output, yields)
}

#[test]
fn fuel_is_recorded_unless_it_runs_out() -> Result<()> {
    let fuel = |fuel| move |store: &mut Store<()>| store.set_fuel(fuel);
    let (mut store, module) = spinner(Config::new().consume_fuel(true), fuel(10_000))?;
    spin(&mut store, &module)?.call(&mut store, 10)?;
    store.finish_recording()?;

    let (mut store, module) = spinner(Config::new().consume_fuel(true), fuel(100))?;
    let err = spin(&mut store, &module)?
        .call(&mut store, 1000)
        .unwrap_err();
    assert_eq!(err.downcast::<Trap>()?, Trap::OutOfFuel);
    let err = store.finish_recording().err().unwrap().to_string();
    assert!(err.contains("fuel check"), "{err}");
    Ok(())
}

#[tokio::test]
async fn fuel_yields_are_recorded() -> Result<()> {
    let (mut store, module) = spinner(Config::new().consume_fuel(true), |store| {
        store.set_fuel(10_000)?;
        store.fuel_async_yield_interval(Some(10))
    })?;
    let spin = spin_async(&mut store, &module).await?;
    let (result, yields) = count_yields(spin.call_async(&mut store, 100)).await;
    result?;
    assert!(yields > 0);
    store.finish_recording()?;
    Ok(())
}

#[test]
fn epoch_traps_poison_the_recording() -> Result<()> {
    let (mut store, module) = spinner(Config::new().epoch_interruption(true), |store| {
        store.set_epoch_deadline(1);
        Ok(())
    })?;
    let spin = spin(&mut store, &module)?;
    spin.call(&mut store, 10)?;
    store.engine().increment_epoch();
    let err = spin.call(&mut store, 10).unwrap_err();
    assert_eq!(err.downcast::<Trap>()?, Trap::Interrupt);
    let err = store.finish_recording().err().unwrap().to_string();
    assert!(err.contains("epoch check"), "{err}");
    Ok(())
}

#[tokio::test]
async fn epoch_yields_are_recorded() -> Result<()> {
    let (mut store, module) = spinner(Config::new().epoch_interruption(true), |store| {
        store.set_epoch_deadline(1);
        store.epoch_deadline_async_yield_and_update(1);
        Ok(())
    })?;
    let spin = spin_async(&mut store, &module).await?;
    store.engine().increment_epoch();
    let (result, yields) = count_yields(spin.call_async(&mut store, 10)).await;
    result?;
    assert_eq!(yields, 1);
    store.finish_recording()?;
    Ok(())
}

#[test]
fn epoch_deadline_callbacks_are_rejected() -> Result<()> {
    let mut config = Config::new();
    config
        .epoch_interruption(true)
        .rr(Some(RrConfig::Recording));
    let engine = Engine::new(&config)?;
    let mut store = Store::new_recording(&engine, ())?;
    store.epoch_deadline_callback(|_| Ok(UpdateDeadline::Continue(1)));
    let err = store.finish_recording().err().unwrap().to_string();
    assert!(err.contains("epoch deadline callbacks"), "{err}");
    Ok(())
}

#[tokio::test]
async fn cancelling_at_a_yield_poisons_the_recording() -> Result<()> {
    let (mut store, module) = spinner(Config::new().epoch_interruption(true), |store| {
        store.set_epoch_deadline(1);
        store.epoch_deadline_async_yield_and_update(1);
        Ok(())
    })?;
    let spin = spin_async(&mut store, &module).await?;
    store.engine().increment_epoch();
    {
        // Poll the call until it yields, then drop it.
        let mut call = std::pin::pin!(spin.call_async(&mut store, 10));
        std::future::poll_fn(|cx| {
            assert!(call.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
    }
    let err = store.finish_recording().err().unwrap().to_string();
    assert!(err.contains("epoch check"), "{err}");
    Ok(())
}

// Trace frame tags, for tests that edit traces (see `rr/codec.rs`).
const ENTER_WASM: u8 = 1;
const LEAVE_WASM: u8 = 2;
const ENTER_HOST: u8 = 3;
const LEAVE_HOST: u8 = 4;
const WRITE: u8 = 5;
const MODULE: u8 = 8;
const GROWTH_FAILED: u8 = 13;

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
    let engine = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&engine, 0)?;
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

    let replay = replay_all(&trace).await?;
    let memory = replay.instances()[0]
        .get_memory(replay.store(), "memory")
        .unwrap();
    assert_eq!(memory.data(replay.store()), expected);
    Ok(())
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
    let replay = replay_all(&trace).await?;
    let memory = replay.instances()[0]
        .get_memory(replay.store(), "memory")
        .unwrap();
    assert_eq!(&memory.data(replay.store())[5..16], b"beforeafter");
    Ok(())
}

#[tokio::test]
async fn memory_growth_flushes_old_borrows_and_records_new_extent() -> Result<()> {
    let (mut store, _, memory) = identity()?;
    memory.data_mut(&mut store)[0] = 13;
    assert_eq!(memory.grow(&mut store, 1)?, 1);
    memory.write(&mut store, 65536, b"new page")?;
    let trace = store.finish_recording()?;
    let replay = replay_all(&trace).await?;
    let replay_memory = replay.instances()[0]
        .get_memory(replay.store(), "memory")
        .unwrap();
    assert_eq!(replay_memory.data(replay.store()), memory.data(&store));
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

    let replay = replay_all(&trace).await?;
    let replay_memory = replay.instances()[0]
        .get_memory(replay.store(), "memory")
        .unwrap();
    assert_eq!(replay_memory.data(replay.store()), memory.data(&store));
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
    let mut replay = new_replayer(&trace)?;
    let error = replay.run().await.unwrap_err();
    assert!(
        format!("{error:#}").contains("invalid outcome"),
        "{error:#}"
    );
    let error = replay.run().await.unwrap_err();
    assert!(format!("{error:#}").contains("failed"), "{error:#}");
    Ok(())
}

#[tokio::test]
async fn trapping_and_failing_activations_finish_by_yielding() -> Result<()> {
    trapping_and_failing_activations(engine(RrConfig::Recording)?).await
}

/// Pulley's trampolines record like native ones, and its recordings replay
/// natively.
#[cfg(feature = "pulley")]
#[tokio::test]
async fn pulley_recordings_replay_natively() -> Result<()> {
    let mut config = Config::new();
    config.target("pulley64")?.rr(Some(RrConfig::Recording));
    trapping_and_failing_activations(Engine::new(&config)?).await
}

async fn trapping_and_failing_activations(recording: Engine) -> Result<()> {
    let mut store = Store::new_recording(&recording, ())?;
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
    replay_all(&trace).await?;
    Ok(())
}

#[test]
#[cfg(feature = "pulley")]
fn replay_requires_a_native_target() -> Result<()> {
    let mut config = Config::new();
    config.target("pulley64")?.rr(Some(RrConfig::Replaying));
    let error = Engine::new(&config).unwrap_err();
    assert!(error.to_string().contains("native"), "{error:?}");
    Ok(())
}

#[tokio::test]
async fn replay_detects_divergent_results() -> Result<()> {
    let recording = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&recording, ())?;
    let host = Func::wrap(&mut store, |x: i32| x);
    let module = new_module(
        &recording,
        r#"(module
        (import "" "host" (func $host (param i32) (result i32)))
        (func (export "run") (result i32) (i32.add (call $host (i32.const 1)) (i32.const 1))))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[host.into()])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ())?, 2);
    let trace = store.finish_recording()?;

    // A trace in which the guest returns where it called the host, and the
    // opposite.
    for (tag, swapped) in [(ENTER_HOST, LEAVE_WASM), (LEAVE_WASM, ENTER_HOST)] {
        let mut bytes = trace.as_bytes().to_vec();
        let last = frames(&bytes)
            .into_iter()
            .rfind(|(t, _)| *t == tag)
            .unwrap()
            .1;
        bytes[last - 5] = swapped;
        let error = replay_all(&Trace::from_bytes(bytes)?).await.err().unwrap();
        assert!(format!("{error:#}").contains("diverged"), "{error:#}");
    }

    // The host was called with different arguments.
    let mut bytes = trace.as_bytes().to_vec();
    let body = first_frame(&bytes, ENTER_HOST);
    bytes[body + 8] ^= 1;
    let error = replay_all(&Trace::from_bytes(bytes)?).await.err().unwrap();
    assert!(
        format!("{error:#}").contains("different arguments"),
        "{error:#}"
    );

    let mut bytes = trace.as_bytes().to_vec();
    // The host result, after the call ID and the success tag.
    let body = first_frame(&bytes, LEAVE_HOST);
    bytes[body + 5..body + 9].copy_from_slice(&5_i32.to_le_bytes());
    let error = replay_all(&Trace::from_bytes(bytes)?).await.err().unwrap();
    assert!(format!("{error:#}").contains("diverged"), "{error:#}");
    Ok(())
}

fn many_calls() -> Result<(Store<()>, Func, Func)> {
    let engine = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&engine, ())?;
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
async fn replay_can_be_cancelled_while_the_guest_is_suspended() -> Result<()> {
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    let (mut store, run, _) = many_calls()?;
    run.call(&mut store, &[], &mut [])?;
    let trace = store.finish_recording()?;
    let mut replay = new_replayer(&trace)?;
    let mut future = Box::pin(replay.run());
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ));
    drop(future);
    // Replay can be resumed, or dropped with the guest suspended.
    drop(replay);
    let mut replay = new_replayer(&trace)?;
    let mut future = Box::pin(replay.run());
    let _ = future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()));
    drop(future);
    assert_eq!(replay.run().await?, ReplayStop::Finished);
    Ok(())
}

#[tokio::test]
async fn asynchronous_recording_and_host_errors() -> Result<()> {
    async fn setup() -> Result<(Store<()>, Func)> {
        let engine = engine(RrConfig::Recording)?;
        let mut store = Store::new_recording(&engine, ())?;
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
    replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn host_panic_ends_the_trace_with_the_call_unfinished() -> Result<()> {
    let recording = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&recording, ())?;
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
    replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn fuel_and_epoch_recordings_replay_without_them() -> Result<()> {
    let (mut store, module) = spinner(Config::new().consume_fuel(true), |store| {
        store.set_fuel(10_000)?;
        store.fuel_async_yield_interval(Some(10))
    })?;
    let spin = spin_async(&mut store, &module).await?;
    spin.call_async(&mut store, 100).await?;
    let fuel = store.finish_recording()?;

    let (mut store, module) = spinner(Config::new().epoch_interruption(true), |store| {
        store.set_epoch_deadline(1);
        store.epoch_deadline_async_yield_and_update(1);
        Ok(())
    })?;
    let spin = spin_async(&mut store, &module).await?;
    store.engine().increment_epoch();
    spin.call_async(&mut store, 10).await?;
    let epochs = store.finish_recording()?;

    for trace in [&fuel, &epochs] {
        replay_all(trace).await?;
    }
    let mut config = Config::new();
    config.rr(Some(RrConfig::Replaying)).consume_fuel(true);
    let err = Replayer::new(&Engine::new(&config)?, fuel).err().unwrap();
    assert!(err.to_string().contains("does not support fuel"), "{err}");
    Ok(())
}

#[tokio::test]
async fn private_table_callback_linker_import_and_memory_alias() -> Result<()> {
    fn setup() -> Result<(Store<()>, Func, Memory)> {
        let engine = engine(RrConfig::Recording)?;
        let mut store = Store::new_recording(&engine, ())?;
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
    let replay = replay_all(&trace).await?;
    let replay_memory = replay.instances()[0]
        .get_memory(replay.store(), "alias")
        .unwrap();
    assert_eq!(memory.data(&record), replay_memory.data(replay.store()));
    Ok(())
}

#[tokio::test]
async fn object_ids_do_not_depend_on_export_lookup_order() -> Result<()> {
    fn setup(reverse: bool) -> Result<(Store<()>, Func)> {
        let engine = engine(RrConfig::Recording)?;
        let mut store = Store::new_recording(&engine, ())?;
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
    replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn segment_drops_are_replayed() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
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
    replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn direct_host_calls_only_record_their_guest_effects() -> Result<()> {
    fn setup() -> Result<(Store<()>, Func, Memory)> {
        let engine = engine(RrConfig::Recording)?;
        let mut store = Store::new_recording(&engine, ())?;
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
    let replay = replay_all(&trace).await?;
    let memory = replay.instances()[0]
        .get_memory(replay.store(), "memory")
        .unwrap();
    assert_eq!(&memory.data(replay.store())[..4], &42_i32.to_le_bytes());
    Ok(())
}

#[tokio::test]
async fn initialization_replays_start_calls_and_retains_instance_exports() -> Result<()> {
    let record_engine = engine(RrConfig::Recording)?;
    let mut record = Store::new_recording(&record_engine, ())?;
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
    let replay = replay_all(&trace).await?;
    let instance = replay.instances()[0];
    assert_eq!(
        &instance
            .get_memory(replay.store(), "memory")
            .unwrap()
            .data(replay.store())[..4],
        &71_i32.to_le_bytes()
    );
    assert_eq!(
        instance
            .get_global(replay.store(), "global")
            .unwrap()
            .get(replay.store())
            .unwrap_i32(),
        9
    );
    Ok(())
}

#[tokio::test]
async fn host_objects_created_during_initialization() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
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
    replay_all(&trace).await?;
    Ok(())
}

#[tokio::test]
async fn initialization_inside_host_callback() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
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
    let replay = replay_all(&trace).await?;
    assert_eq!(replay.instances().len(), 2);
    assert_eq!(
        replay.instances()[1]
            .get_global(replay.store(), "g")
            .unwrap()
            .get(replay.store())
            .unwrap_i32(),
        42
    );
    Ok(())
}

#[tokio::test]
async fn initialization_requires_exactly_one_startup() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
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
    let error = replay_all(&Trace::from_bytes(missing)?)
        .await
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("startup missing"));

    let mut repeated = bytes[..bytes.len() - 5].to_vec();
    repeated.extend_from_slice(&bytes[startup..]);
    let error = replay_all(&Trace::from_bytes(repeated)?)
        .await
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("repeated instance startup"));
    Ok(())
}

#[tokio::test]
async fn function_references_cross_boundaries_by_id() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
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
    let replay = replay_all(&trace).await?;
    assert_eq!(
        replay.instances()[0]
            .get_global(replay.store(), "result")
            .unwrap()
            .get(replay.store())
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
    let error = replay_all(&Trace::from_bytes(bytes)?).await.err().unwrap();
    assert!(format!("{error:#}").contains("invalid function ID"));
    Ok(())
}

#[tokio::test]
async fn initialization_tracks_module_identity() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let wasm = "(module (func (export \"run\") (result i32) i32.const 1))";
    let module = new_module(&e, wasm)?;
    let serialized = new_module(&e, wasm)?.serialize()?;
    // SAFETY: these unchanged artifacts were just compiled by the same engine.
    let restored = unsafe { Module::deserialize(&e, &serialized)? };
    assert!(!Module::same(&module, &restored));
    let different_wasm = "(module (func (export \"run\") (result i32) i32.const 2))";
    let different = new_module(&e, different_wasm)?;
    let mut store = Store::new_recording(&e, ())?;
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
    assert_eq!(replay_all(&trace).await?.instances().len(), 4);
    Ok(())
}

#[tokio::test]
async fn replay_compiles_modules_from_provided_bytecode() -> Result<()> {
    let wasm = "(module (func (export \"run\") (result i32) i32.const 7))";
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
    let instance = Instance::new(&mut store, &wasmtime::Module::new(&e, wasm)?, &[])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    assert_eq!(run.call(&mut store, ())?, 7);
    let trace = store.finish_recording()?;
    let replaying = engine(RrConfig::Replaying)?;

    // Traces refer to modules by checksum only.
    let [(checksum, binary)] = &rr::core_modules(&e, wasm.as_bytes())?[..] else {
        panic!("one core module")
    };
    assert_eq!(trace.module_checksums()?, [*checksum]);
    assert!(trace.as_bytes().len() < binary.len() + 128);

    // Without the module's bytecode, replay fails when it reaches the
    // module's instantiation; unrelated bytecode is ignored.
    let unrelated = "(module (func (export \"run\") (result i32) i32.const 8))";
    let mut replay = Replayer::new(&replaying, trace.clone())?;
    replay.add_bytecode(unrelated.as_bytes())?;
    let error = replay.run().await.unwrap_err();
    assert!(
        format!("{error:#}").contains(&rr::checksum_hex(checksum)),
        "{error:#}"
    );

    // Bytecode can be provided in either format.
    for wasm in [wasm.as_bytes(), binary] {
        let mut replay = Replayer::new(&replaying, trace.clone())?;
        replay.add_bytecode(unrelated.as_bytes())?;
        replay.add_bytecode(wasm)?;
        assert_eq!(replay.run().await?, ReplayStop::Finished);
        assert_eq!(replay.instances().len(), 1);
    }
    Ok(())
}

#[tokio::test]
async fn initialization_preserves_import_aliases_across_instances() -> Result<()> {
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
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
    assert_eq!(replay_all(&trace).await?.instances().len(), 2);
    Ok(())
}

#[tokio::test]
async fn failed_guest_growth_is_replayed() -> Result<()> {
    // Growth beyond a small, immovable reservation fails when recording but
    // would succeed with the replaying engine's default configuration.
    let mut config = Config::new();
    config
        .rr(Some(RrConfig::Recording))
        .memory_reservation(1 << 18)
        .memory_reservation_for_growth(0)
        .memory_may_move(false)
        .memory_guard_size(0)
        .signals_based_traps(false);
    let recording = Engine::new(&config)?;
    let mut store = Store::new_recording(&recording, ())?;
    let module = new_module(
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
    let replay = replay_all(&trace).await?;
    let memory = replay.instances()[0]
        .get_memory(replay.store(), "memory")
        .unwrap();
    assert_eq!(memory.data_size(replay.store()), 2 << 16);
    Ok(())
}

#[tokio::test]
async fn replays_read_function_tables_and_look_up_functions() -> Result<()> {
    let recording = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&recording, ())?;
    let module = new_module(
        &recording,
        r#"(module
        (table (export "table") 2 funcref)
        (elem (i32.const 1) func $f)
        (func $f (export "f") (param i32) (result i32) local.get 0))"#,
    )?;
    Instance::new(&mut store, &module, &[])?;
    let trace = store.finish_recording()?;

    let mut replayer = replay_all(&trace).await?;
    let instance = replayer.instances()[0];
    // Replays initialize tables eagerly, so reading them needs no more than
    // the replayer's shared access to the store.
    let s = replayer.store();
    let table = instance.get_table(&s, "table").unwrap();
    assert!(
        table
            .get_if_initialized(&s, 0)
            .unwrap()
            .unwrap_func()
            .is_none()
    );
    let element = *table
        .get_if_initialized(&s, 1)
        .unwrap()
        .unwrap_func()
        .unwrap();
    // Function exports are initialized lazily, so the replayer looks them up.
    let f = replayer
        .get_export(instance, "f")
        .unwrap()
        .into_func()
        .unwrap();
    let s = replayer.store();
    assert_eq!(f.ty(&s).params().len(), 1);
    assert_eq!(f.to_raw(&s), element.to_raw(&s));
    assert!(replayer.get_export(instance, "missing").is_none());
    Ok(())
}
