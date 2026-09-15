#![cfg(all(feature = "cranelift", feature = "runtime", feature = "wat", not(miri)))]

use wasmtime::*;

fn run_i32(wat: &str, inputs: impl IntoIterator<Item = (i32, i32)>) -> Result<()> {
    let inputs: Vec<_> = inputs.into_iter().collect();
    for opt in [OptLevel::None, OptLevel::Speed, OptLevel::SpeedAndSize] {
        let engine = Engine::new(
            Config::new()
                .wasm_multiloop(true)
                .cranelift_opt_level(opt)
                .cranelift_debug_verifier(true),
        )?;
        let mut store = Store::new(&engine, ());
        let module = Module::new(&engine, wat)?;
        let instance = Instance::new(&mut store, &module, &[])?;
        let func = instance.get_typed_func::<i32, i32>(&mut store, "run")?;
        for &(input, expected) in &inputs {
            assert_eq!(func.call(&mut store, input)?, expected);
        }
    }
    Ok(())
}

#[test]
fn irreducible_cycles() -> Result<()> {
    // Body zero enters either body one or two; both enter each other. Neither
    // cyclic body dominates the other. Values and locals cross all these edges.
    let wat = r#"(module
      (func (export "run") (param $n i32) (result i32) (local $sum i32)
        local.get $n
        multiloop
          label (param i32) (result i32)
            local.get $n i32.const 1 i32.and
            br_if 1
            br 2
          label (param i32) (result i32)
            local.set $n
            local.get $sum i32.const 3 i32.add local.set $sum
            local.get $n i32.const 1 i32.sub local.tee $n
            local.get $n i32.const 0 i32.gt_s br_if 2
          label (param i32) (result i32)
            local.set $n
            local.get $sum i32.const 5 i32.add local.set $sum
            local.get $n i32.const 1 i32.sub local.tee $n
            local.get $n i32.const 0 i32.gt_s br_if 1
            drop local.get $sum
        end))"#;
    let mut cases = Vec::new();
    for n in 1..40 {
        let mut remaining = n;
        let mut body = if n % 2 == 1 { 1 } else { 2 };
        let mut sum = 0;
        loop {
            sum += if body == 1 { 3 } else { 5 };
            remaining -= 1;
            if body == 2 && remaining <= 0 {
                break;
            }
            body = if body == 1 { 2 } else { 1 };
        }
        cases.push((n, sum));
    }
    run_i32(wat, cases)
}

#[test]
fn nested_scopes_and_outer_targets() -> Result<()> {
    run_i32(
        r#"(module (func (export "run") (param i32) (result i32)
      i32.const 100
      block $exit (result i32)
        local.get 0
        multiloop
          label (param i32) (result i32)
            block (param i32) (result i32)
              i32.const 9 local.get 0 br_if $exit drop
              i32.const 0 br_table 2 2
            end
          label (param i32) (result i32)
            multiloop
              label (param i32) (result i32) i32.const 2 i32.add
              label (param i32) (result i32) i32.const 3 i32.mul
            end
        end
      end
      i32.add))"#,
        [(0, 106), (1, 109)],
    )
}

#[test]
fn differing_signatures_and_backward_entry() -> Result<()> {
    run_i32(
        r#"(module (func (export "run") (param i32) (result i32)
      multiloop
        label (result i64)
          local.get 0 i32.const 1 i32.add local.set 0
          i64.const 42
        label (param i64) (result i32)
          drop
          local.get 0 i32.const 3 i32.lt_s br_if 0
          local.get 0
      end))"#,
        [(0, 3), (5, 6)],
    )
}

#[test]
fn unreachable_regions_and_late_predecessors() -> Result<()> {
    run_i32(
        r#"(module (func (export "run") (param i32) (result i32)
      multiloop
        label (result i32) i32.const 7 br 2
        label (param i32) (result i32) i32.const 1 i32.add br 3
        label (param i32) (result i32) br 1
      end
      unreachable
      multiloop label label end))"#,
        [(0, 8)],
    )
}

#[test]
fn feature_gating() -> Result<()> {
    let wasm = wat::parse_str("(module (func multiloop label end))")?;
    assert!(Module::new(&Engine::default(), &wasm).is_err());
    let engine = Engine::new(Config::new().wasm_multiloop(true))?;
    Module::new(&engine, &wasm)?;
    assert!(
        Module::new(
            &engine,
            "(module (func multiloop label (result i32) label end))"
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn entries_consume_fuel() -> Result<()> {
    let engine = Engine::new(Config::new().wasm_multiloop(true).consume_fuel(true))?;
    let mut store = Store::new(&engine, ());
    store.set_fuel(100)?;
    let module = Module::new(
        &engine,
        "(module (func (export \"run\") multiloop label br 1 label br 0 end))",
    )?;
    let instance = Instance::new(&mut store, &module, &[])?;
    let run = instance.get_typed_func::<(), ()>(&mut store, "run")?;
    assert_eq!(
        run.call(&mut store, ()).unwrap_err().downcast::<Trap>()?,
        Trap::OutOfFuel
    );
    Ok(())
}

#[test]
fn multivalue_and_vector_transfers() -> Result<()> {
    run_i32(
        r#"(module (func (export "run") (param i32) (result i32)
      local.get 0 v128.const i32x4 1 2 3 4
      multiloop
        label (param i32 v128) (result i32 v128)
          v128.const i32x4 10 20 30 40 i32x4.add
          i32.const 0 br_table 1 1
        label (param i32 v128) (result i32)
          i32x4.extract_lane 2 i32.add
      end))"#,
        [(0, 33), (9, 42)],
    )
}

#[test]
fn entries_check_epoch() -> Result<()> {
    let engine = Engine::new(Config::new().wasm_multiloop(true).epoch_interruption(true))?;
    let mut store = Store::new(&engine, ());
    store.set_epoch_deadline(1);
    let module = Module::new(
        &engine,
        r#"(module
      (import "" "tick" (func $tick))
      (func (export "run")
        multiloop label call $tick br 1 label br 0 end))"#,
    )?;
    let tick = Func::wrap(&mut store, {
        let engine = engine.clone();
        move || engine.increment_epoch()
    });
    let instance = Instance::new(&mut store, &module, &[tick.into()])?;
    let run = instance.get_typed_func::<(), ()>(&mut store, "run")?;
    assert_eq!(
        run.call(&mut store, ()).unwrap_err().downcast::<Trap>()?,
        Trap::Interrupt
    );
    Ok(())
}

#[test]
#[cfg(feature = "winch")]
fn winch_rejects_multiloop_configuration() {
    assert!(Engine::new(Config::new().strategy(Strategy::Winch).wasm_multiloop(true)).is_err());
}

#[test]
fn fallthrough_fuel_is_charged_once() -> Result<()> {
    let engine = Engine::new(Config::new().wasm_multiloop(true).consume_fuel(true))?;
    let mut fuel = Vec::new();
    for body in [
        "multiloop label i32.const 8 drop label local.get 0 i32.const 1 i32.sub local.set 0 local.get 0 br_if 1 end",
        "i32.const 8 drop loop local.get 0 i32.const 1 i32.sub local.set 0 local.get 0 br_if 0 end",
    ] {
        let mut store = Store::new(&engine, ());
        store.set_fuel(1000)?;
        let module = Module::new(
            &engine,
            format!("(module (func (export \"run\") (param i32) i32.const 9 drop {body}))"),
        )?;
        let instance = Instance::new(&mut store, &module, &[])?;
        instance
            .get_typed_func::<i32, ()>(&mut store, "run")?
            .call(&mut store, 10)?;
        fuel.push(store.get_fuel()?);
    }
    assert_eq!(fuel[0], fuel[1]);
    Ok(())
}

#[test]
fn reference_branches() -> Result<()> {
    run_i32(
        r#"(module (func (export "run") (param i32) (result i32)
      multiloop
        label (result i32)
          i32.const 7 ref.null func br_on_null 1
          drop drop i32.const 99
        label (param i32) (result i32) i32.const 1 i32.add
      end))"#,
        [(0, 8)],
    )
}

#[test]
#[cfg(feature = "gc")]
fn exception_entry_transfer() -> Result<()> {
    let engine = Engine::new(Config::new().wasm_multiloop(true).wasm_exceptions(true))?;
    let mut store = Store::new(&engine, ());
    let module = Module::new(
        &engine,
        r#"(module
      (tag $e (param i32))
      (func $throw i32.const 41 throw $e)
      (func (export "run") (result i32)
        multiloop
          label (result i32)
            try_table (result i32) (catch $e 1)
              call $throw i32.const 99
            end
          label (param i32) (result i32) i32.const 1 i32.add
        end))"#,
    )?;
    let instance = Instance::new(&mut store, &module, &[])?;
    assert_eq!(
        instance
            .get_typed_func::<(), i32>(&mut store, "run")?
            .call(&mut store, ())?,
        42
    );
    Ok(())
}
