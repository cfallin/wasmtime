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

/// The bytecode of every module and component these tests compile. Traces refer to modules
/// only by checksum, so replays are given all of it, as they might be given
/// a directory of a program's modules.
static BYTECODE: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

fn register_bytecode(wasm: &str) {
    let mut bytecode = BYTECODE.lock().unwrap();
    if !bytecode.iter().any(|b| b == wasm.as_bytes()) {
        bytecode.push(wasm.as_bytes().to_vec());
    }
}

fn new_module(engine: &Engine, wasm: &str) -> Result<Module> {
    register_bytecode(wasm);
    Module::new(engine, wasm)
}

fn new_component(engine: &Engine, wasm: &str) -> Result<component::Component> {
    register_bytecode(wasm);
    component::Component::new(engine, wasm)
}

/// Starts a replay with the bytecode of every module and component these tests
/// compile, skipping any that the replaying engine rejects (such as components
/// that need features that it does not enable).
fn new_replayer_with(engine: &Engine, trace: &Trace) -> Result<Replayer> {
    let mut replayer = Replayer::new(engine, trace.clone())?;
    for wasm in BYTECODE.lock().unwrap().iter() {
        let _ = replayer.add_bytecode(wasm);
    }
    Ok(replayer)
}

fn new_replayer(trace: &Trace) -> Result<Replayer> {
    new_replayer_with(&engine(RrConfig::Replaying)?, trace)
}

/// Replays an entire trace.
async fn replay_with(engine: &Engine, trace: &Trace) -> Result<Replayer> {
    let mut replayer = new_replayer_with(engine, trace)?;
    assert_eq!(replayer.run().await?, ReplayStop::Finished);
    Ok(replayer)
}

async fn replay_all(trace: &Trace) -> Result<Replayer> {
    replay_with(&engine(RrConfig::Replaying)?, trace).await
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

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_initialization_and_lowered_host_calls() -> Result<()> {
    use wasmtime::component::{Component, Linker};
    let record_engine = engine(RrConfig::Recording)?;
    let mut record = Store::new_recording(&record_engine, 0_u32)?;
    let component = new_component(
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
    let replay = replay_all(&trace).await?;
    assert!(!replay.instances().is_empty());
    Ok(())
}

/// A lowered host function's error, a failing component intrinsic, and a trap
/// in a component trampoline are each recorded as the outcome of the
/// trampoline's host call.
#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_trampoline_failures() -> Result<()> {
    use wasmtime::component::Linker;
    let wat = r#"(component
        (import "host" (func $host (param "x" u32) (result u32)))
        (type $r (resource (rep i32)))
        (core func $host (canon lower (func $host)))
        (core func $rep (canon resource.rep $r))
        (core module $m
            (import "" "host" (func $host (param i32) (result i32)))
            (import "" "rep" (func $rep (param i32) (result i32)))
            (func (export "host") (param i32) (result i32) (call $host (local.get 0)))
            (func (export "rep") (param i32) (result i32) (call $rep (local.get 0)))
            (func (export "id") (param i32) (result i32) (local.get 0))
            (func (export "post") (param i32) (drop (call $host (local.get 0)))))
        (core instance $i (instantiate $m
            (with "" (instance (export "host" (func $host)) (export "rep" (func $rep))))))
        (func (export "host") (param "x" u32) (result u32) (canon lift (core func $i "host")))
        (func (export "rep") (param "x" u32) (result u32) (canon lift (core func $i "rep")))
        (func (export "leave") (param "x" u32) (result u32)
            (canon lift (core func $i "id") (post-return (core func $i "post")))))"#;
    for (export, expected) in [
        ("host", "host failure"),
        ("rep", "unknown handle index"),
        ("leave", "cannot leave component instance"),
    ] {
        let e = engine(RrConfig::Recording)?;
        let mut store = Store::new_recording(&e, ())?;
        let c = new_component(&e, wat)?;
        let mut linker = Linker::new(&e);
        linker
            .root()
            .func_wrap("host", |_, (_,): (u32,)| -> Result<(u32,)> {
                wasmtime::bail!("host failure")
            })?;
        let instance = linker.instantiate(&mut store, &c)?;
        let f = instance.get_typed_func::<(u32,), (u32,)>(&mut store, export)?;
        let error = f.call(&mut store, (7,)).unwrap_err();
        assert!(
            format!("{error:?}").contains(expected),
            "{export}: {error:?}"
        );
        let trace = store.finish_recording()?;
        replay_all(&trace).await?;
    }
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_strings_realloc_and_post_return() -> Result<()> {
    use wasmtime::component::Linker;
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    let replay = replay_all(&trace).await?;
    let memory = replay
        .instances()
        .iter()
        .find_map(|i| i.get_memory(replay.store(), "memory"))
        .unwrap();
    assert_eq!(&memory.data(replay.store())[32..36], &1_i32.to_le_bytes());
    assert!(
        memory
            .data(replay.store())
            .windows("héllo!".len())
            .any(|s| s == "héllo!".as_bytes())
    );
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_resources_and_guest_destructor_callback() -> Result<()> {
    use wasmtime::component::{Linker, ResourceAny};
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, 0)?;
    let c = new_component(
        &e,
        r#"(component
      (import "notify" (func $notify (param "x" u32)))
      (core func $notify (canon lower (func $notify)))
      (core module $d
        (import "" "notify" (func $notify (param i32)))
        (global $count (export "count") (mut i32) (i32.const 0))
        (func (export "dtor") (param i32)
          (if (i32.eq (local.get 0) (i32.const 13)) (then unreachable))
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
        (func (export "run") i32.const 42 call $new call $drop)
        (func (export "trap") i32.const 13 call $new call $drop))
      (core instance $run (instantiate $run
        (with "" (instance (export "new" (func $new)) (export "drop" (func $drop))))))
      (func (export "make") (result (own $exported-r)) (canon lift (core func $run "make")))
      (func (export "run") (canon lift (core func $run "run")))
      (func (export "trap") (canon lift (core func $run "trap"))))"#,
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
    // A destructor's trap also unwinds the `resource.drop` that called it.
    let trap = instance.get_typed_func::<(), ()>(&mut store, "trap")?;
    assert!(trap.call(&mut store, ()).is_err());
    let trace = store.finish_recording()?;
    let replay = replay_all(&trace).await?;
    let count = replay
        .instances()
        .iter()
        .find_map(|i| i.get_global(replay.store(), "count"))
        .unwrap();
    assert_eq!(count.get(replay.store()).unwrap_i32(), 2);
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_to_component_adapter_and_transcoding() -> Result<()> {
    use wasmtime::component::Linker;
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    let replay = replay_all(&trace).await?;
    assert!(replay.instances().len() >= 3);
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_async_host_and_trap_during_startup() -> Result<()> {
    use wasmtime::component::Linker;
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    replay_all(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn concurrent_component_activations_can_finish_out_of_order() -> Result<()> {
    use wasmtime::component::Linker;
    let mut config = Config::new();
    config
        .rr(Some(RrConfig::Recording))
        .wasm_component_model_async(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    config.rr(Some(RrConfig::Replaying));
    replay_with(&Engine::new(&config)?, &trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_adapter_callbacks() -> Result<()> {
    use wasmtime::component::Linker;
    let mut config = Config::new();
    config
        .rr(Some(RrConfig::Recording))
        .wasm_component_model_async(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    config.rr(Some(RrConfig::Replaying));
    replay_with(&Engine::new(&config)?, &trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_stream_wait() -> Result<()> {
    use wasmtime::component::{Linker, StreamAny};
    let mut config = Config::new();
    config
        .wasm_component_model_async(true)
        .rr(Some(RrConfig::Recording));
    let engine = Engine::new(&config)?;
    let mut store = Store::new_recording(&engine, ())?;
    let component = new_component(
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
    config.rr(Some(RrConfig::Replaying));
    replay_with(&Engine::new(&config)?, &trace).await?;
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_lift_callback() -> Result<()> {
    use wasmtime::component::Linker;
    let mut config = Config::new();
    config
        .rr(Some(RrConfig::Recording))
        .wasm_component_model_async(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    config.rr(Some(RrConfig::Replaying));
    let replay = replay_with(&Engine::new(&config)?, &trace).await?;
    let calls = replay.instances()[0]
        .get_global(replay.store(), "calls")
        .unwrap();
    assert_eq!(calls.get(replay.store()).unwrap_i32(), 1);
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_imported_core_module() -> Result<()> {
    use wasmtime::component::Linker;
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    let module = new_module(
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
    replay_all(&trace).await?;
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_without_core_instances() -> Result<()> {
    use wasmtime::component::{Linker, ResourceAny};
    let e = engine(RrConfig::Recording)?;
    let mut store = Store::new_recording(&e, ())?;
    let c = new_component(
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
    assert!(replay_all(&trace).await?.instances().is_empty());
    Ok(())
}

#[cfg(feature = "component-model-async")]
#[tokio::test]
async fn component_async_lower_futures_and_cancellation() -> Result<()> {
    use wasmtime::component::Linker;
    let mut config = Config::new();
    config
        .rr(Some(RrConfig::Recording))
        .wasm_component_model_async(true)
        .wasm_component_model_more_async_builtins(true);
    let e = Engine::new(&config)?;
    let mut store = Store::new_recording(&e, ())?;
    // The component-model suite's cancellation scenario exercises async
    // lowering, guest callbacks, future transfers, and subtask cancellation.
    let c = new_component(&e, include_str!("rr_async_component.wat"))?;
    let instance = Linker::new(&e).instantiate_async(&mut store, &c).await?;
    let run = instance.get_typed_func::<(), (u32,)>(&mut store, "run")?;
    assert_eq!(run.call_async(&mut store, ()).await?, (42,));
    let trace = store.finish_recording()?;
    config.rr(Some(RrConfig::Replaying));
    replay_with(&Engine::new(&config)?, &trace).await?;
    Ok(())
}

#[cfg(feature = "component-model")]
#[tokio::test]
async fn component_bytecode_provides_its_nested_modules() -> Result<()> {
    use wasmtime::component::Linker;
    let wasm = r#"(component
        (core module $a (func (export "f") (result i32) i32.const 1))
        (component $inner
            (core module $b (func (export "g") (result i32) i32.const 2))
            (core instance (instantiate $b)))
        (core instance (instantiate $a))
        (instance (instantiate $inner)))"#;
    let e = engine(RrConfig::Recording)?;
    let modules = rr::core_modules(&e, wasm.as_bytes())?;
    assert_eq!(modules.len(), 2);
    let mut store = Store::new_recording(&e, ())?;
    Linker::new(&e).instantiate(&mut store, &component::Component::new(&e, wasm)?)?;
    let trace = store.finish_recording()?;
    assert_eq!(
        trace.module_checksums()?,
        modules.iter().map(|(c, _)| *c).collect::<Vec<_>>()
    );
    let mut replay = Replayer::new(&engine(RrConfig::Replaying)?, trace)?;
    replay.add_bytecode(wasm.as_bytes())?;
    assert_eq!(replay.run().await?, ReplayStop::Finished);
    assert_eq!(replay.instances().len(), 2);
    Ok(())
}

#[cfg(feature = "gc")]
#[tokio::test]
async fn gc_during_callbacks_sees_parked_roots() -> Result<()> {
    const WAT: &str = r#"
    (module
      (type $s (struct (field (mut i32)) (field (mut (ref null $s)))))
      (import "" "host" (func $host))
      (func $build (param $n i32) (param $base i32) (result (ref null $s))
        (local $i i32) (local $x (ref null $s))
        loop
          (local.set $x (struct.new $s (i32.add (local.get $base) (local.get $i)) (local.get $x)))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br_if 0 (i32.lt_u (local.get $i) (local.get $n)))
        end
        local.get $x)
      (func $sum (param $x (ref null $s)) (result i32) (local $sum i32)
        block
          loop
            (br_if 1 (ref.is_null (local.get $x)))
            (local.set $sum (i32.add (local.get $sum) (struct.get $s 0 (local.get $x))))
            (local.set $x (struct.get $s 1 (local.get $x)))
            br 0
          end
        end
        local.get $sum)
      (func (export "churn") (result i32)
        (local $k i32) (local $t i32)
        loop
          (local.set $t (i32.add (local.get $t)
            (call $sum (call $build (i32.const 1000) (local.get $k)))))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br_if 0 (i32.lt_u (local.get $k) (i32.const 200)))
        end
        local.get $t)
      (func (export "run") (result i32)
        (local $x (ref null $s))
        (local.set $x (call $build (i32.const 500) (i32.const 7)))
        call $host
        (call $sum (local.get $x))))
    "#;
    let mk = |mode, collector| -> Result<Engine> {
        let mut config = Config::new();
        config
            .rr(Some(mode))
            .wasm_gc(true)
            .wasm_function_references(true)
            .collector(collector);
        Engine::new(&config)
    };
    for collector in [
        Collector::DeferredReferenceCounting,
        Collector::Null,
        Collector::Copying,
    ] {
        let recording = mk(RrConfig::Recording, collector)?;
        let mut store = Store::new_recording(&recording, ())?;
        let host = Func::wrap(&mut store, |mut caller: Caller<'_, ()>| -> Result<()> {
            let churn = caller.get_export("churn").unwrap().into_func().unwrap();
            let churn = churn.typed::<(), i32>(&caller)?;
            churn.call(&mut caller, ())?;
            Ok(())
        });
        let module = new_module(&recording, WAT)?;
        let instance = Instance::new(&mut store, &module, &[host.into()])?;
        let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
        // The callback allocates enough to collect while `run` holds a list.
        assert_eq!(run.call(&mut store, ())?, (7..507).sum::<i32>());
        let trace = store.finish_recording()?;
        // Replay checks the recorded result.
        replay_with(&mk(RrConfig::Replaying, collector)?, &trace).await?;
    }
    Ok(())
}

#[cfg(feature = "gc")]
const GC_LIST: &str = r#"
(module
  (type $s (struct (field (mut i32)) (field (ref null $s))))
  (global $list (export "list") (mut (ref null $s)) (ref.null $s))
  (func $churn (param $n i32)
    (local $i i32)
    loop
      (drop (struct.new $s (local.get $i) (ref.null $s)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if 0 (i32.lt_u (local.get $i) (local.get $n)))
    end)
  (func (export "step") (param $k i32) (result i32)
    (local $x (ref null $s)) (local $sum i32)
    ;; Existing objects change too: the previous head gains 100.
    (if (i32.eqz (ref.is_null (global.get $list)))
      (then (struct.set $s 0 (global.get $list)
              (i32.add (struct.get $s 0 (global.get $list)) (i32.const 100)))))
    (global.set $list (struct.new $s (local.get $k) (global.get $list)))
    (call $churn (i32.const 20000))
    (local.set $x (global.get $list))
    block
      loop
        (br_if 1 (ref.is_null (local.get $x)))
        (local.set $sum (i32.add (local.get $sum) (struct.get $s 0 (local.get $x))))
        (local.set $x (struct.get $s 1 (local.get $x)))
        br 0
      end
    end
    local.get $sum))
"#;

#[cfg(feature = "gc")]
fn gc_engine(mode: RrConfig, collector: Collector) -> Result<Engine> {
    let mut config = Config::new();
    config
        .rr(Some(mode))
        .wasm_gc(true)
        .wasm_function_references(true)
        .collector(collector);
    Engine::new(&config)
}

/// The non-null `anyref` in `val`, as an unrooted reference.
#[cfg(feature = "gc")]
fn unrooted_anyref(val: Val) -> Option<Unrooted<AnyRef>> {
    val.unwrap_anyref().map(|r| r.unwrap_unrooted())
}

/// The values in the list in the `list` global, read through unrooted
/// references, which leave the replay's GC roots alone.
#[cfg(feature = "gc")]
fn gc_list(replayer: &Replayer) -> Result<Vec<i32>> {
    let instance = replayer.instances()[0];
    let store = replayer.store();
    let list = instance.get_global(&store, "list").unwrap();
    let mut values = Vec::new();
    let mut next = unrooted_anyref(list.get(&store));
    while let Some(node) = next {
        let node = node.as_struct(&store)?.unwrap();
        values.push(node.field(&store, 0)?.unwrap_i32());
        next = unrooted_anyref(node.field(&store, 1)?);
    }
    Ok(values)
}

#[cfg(feature = "gc")]
const EXCEPTIONS: &str = r#"
(module
  (tag $e (param i32))
  (func $throw (param i32) (throw $e (local.get 0)))
  ;; Throws and catches within the activation.
  (func (export "catch") (param i32) (result i32)
    (block $h (result i32)
      (try_table (catch $e $h) (call $throw (local.get 0)))
      (i32.const -1))
    (i32.add (i32.const 1)))
  ;; Throws an exception that reaches the host.
  (func (export "throw") (param i32)
    (call $throw (local.get 0))))
"#;

#[cfg(feature = "gc")]
fn exceptions_engine(mode: RrConfig, collector: Collector) -> Result<Engine> {
    let mut config = Config::new();
    config
        .rr(Some(mode))
        .wasm_exceptions(true)
        .collector(collector);
    Engine::new(&config)
}

#[cfg(feature = "gc")]
#[tokio::test]
async fn guest_exceptions_are_replayed() -> Result<()> {
    for collector in [
        Collector::DeferredReferenceCounting,
        Collector::Null,
        Collector::Copying,
    ] {
        let recording = exceptions_engine(RrConfig::Recording, collector)?;
        let mut store = Store::new_recording(&recording, ())?;
        let module = new_module(&recording, EXCEPTIONS)?;
        let instance = Instance::new(&mut store, &module, &[])?;
        let catch = instance.get_typed_func::<i32, i32>(&mut store, "catch")?;
        let throw = instance.get_typed_func::<i32, ()>(&mut store, "throw")?;
        assert_eq!(catch.call(&mut store, 5)?, 6);
        // The host takes the first exception that reaches it and leaves the
        // second one pending.
        let err = throw.call(&mut store, 7).unwrap_err();
        assert!(err.is::<ThrownException>());
        let exn = store.take_pending_exception().unwrap();
        assert_eq!(exn.field(&store, 0)?.unwrap_i32(), 7);
        assert!(
            throw
                .call(&mut store, 8)
                .unwrap_err()
                .is::<ThrownException>()
        );
        assert!(store.has_pending_exception());
        let trace = store.finish_recording()?;

        // Replay computes the exceptions itself, and still has the last one
        // pending.
        let engine = exceptions_engine(RrConfig::Replaying, collector)?;
        let replayer = replay_with(&engine, &trace).await?;
        let exn = replayer.pending_exception().unwrap();
        assert_eq!(exn.field(replayer.store(), 0)?.unwrap_i32(), 8);
    }
    Ok(())
}

#[cfg(feature = "gc")]
#[test]
fn host_functions_throwing_exceptions_poison_the_recording() -> Result<()> {
    let recording = exceptions_engine(RrConfig::Recording, Collector::DeferredReferenceCounting)?;
    let mut store = Store::new_recording(&recording, ())?;
    // The host rethrows the exception that a nested guest call threw, and the
    // outer guest code catches it.
    let rethrow = Func::wrap(&mut store, |mut caller: Caller<'_, ()>| -> Result<()> {
        let throw = caller.get_export("throw").unwrap().into_func().unwrap();
        throw.typed::<(), ()>(&caller)?.call(&mut caller, ())
    });
    let module = new_module(
        &recording,
        r#"(module
        (tag $e)
        (import "" "rethrow" (func $rethrow))
        (func (export "throw") (throw $e))
        (func (export "run") (result i32)
          (block $h
            (try_table (catch $e $h) (call $rethrow))
            (return (i32.const 0)))
          (i32.const 1)))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[rethrow.into()])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    // Recording fails the rest of the call once the host throws.
    assert!(run.call(&mut store, ()).is_err());
    let err = store.finish_recording().err().unwrap().to_string();
    assert!(err.contains("host functions throwing exceptions"), "{err}");
    Ok(())
}

#[cfg(feature = "gc")]
#[tokio::test]
async fn gc_heaps_are_replayed() -> Result<()> {
    for collector in [
        Collector::DeferredReferenceCounting,
        Collector::Null,
        Collector::Copying,
    ] {
        let recording = gc_engine(RrConfig::Recording, collector)?;
        let mut store = Store::new_recording(&recording, ())?;
        let module = new_module(&recording, GC_LIST)?;
        let instance = Instance::new(&mut store, &module, &[])?;
        let step = instance.get_typed_func::<i32, i32>(&mut store, "step")?;
        for k in 0..6 {
            // Fails with the null collector once its heap is exhausted, which
            // replay reproduces.
            if step.call(&mut store, k).is_err() {
                break;
            }
        }
        let trace = store.finish_recording()?;
        let engine = gc_engine(RrConfig::Replaying, collector)?;
        let replayer = replay_with(&engine, &trace).await?;
        let list = gc_list(&replayer)?;
        assert!(list.len() >= 2, "{collector:?}");
        assert!(list[1..].iter().rev().zip(100..).all(|(a, b)| *a == b));
    }
    Ok(())
}

#[cfg(feature = "gc")]
#[test]
fn host_mutation_of_gc_objects_poisons_the_recording() -> Result<()> {
    let recording = gc_engine(RrConfig::Recording, Collector::Auto)?;
    let mut store = Store::new_recording(&recording, ())?;
    let module = new_module(&recording, GC_LIST)?;
    let instance = Instance::new(&mut store, &module, &[])?;
    let step = instance.get_typed_func::<i32, i32>(&mut store, "step")?;
    step.call(&mut store, 1)?;
    let list = instance.get_global(&mut store, "list").unwrap();
    let head = unrooted_anyref(list.get(&store)).unwrap();
    let head = head.to_rooted(&mut store)?;
    let head = head.as_struct(&store)?.unwrap();
    assert!(head.set_field(&mut store, 0, Val::I32(7)).is_err());
    let err = store.finish_recording().err().unwrap().to_string();
    assert!(err.contains("host mutation of GC objects"), "{err}");
    Ok(())
}
