use wasmtime::{rr::Trace, *};

fn engine(mode: RRConfig) -> Result<Engine> {
    let mut config = Config::new();
    config.rr(mode);
    Engine::new(&config)
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

#[test]
fn recording_requires_an_empty_store() -> Result<()> {
    // Application data and precompiled modules do not populate the store.
    let record_engine = engine(RRConfig::Recording)?;
    let mut record = Store::new(&record_engine, 123);
    record.start_recording()?;
    record.finish_recording()?;

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
