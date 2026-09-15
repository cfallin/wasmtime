;;! multiloop = true
;;! simd = true

(module
  ;; i32x4.add produces a different CLIF vector type from the canonical
  ;; Wasm v128 type used at entry transfers, including typed br_table edges.
  (func (export "transfer") (param i32) (result i32)
    local.get 0 v128.const i32x4 1 2 3 4
    multiloop
      label (param i32 v128) (result i32 v128)
        v128.const i32x4 10 20 30 40 i32x4.add
        i32.const 0 br_table 1 1
      label (param i32 v128) (result i32)
        i32x4.extract_lane 2 i32.add
    end))

(assert_return (invoke "transfer" (i32.const 0)) (i32.const 33))
(assert_return (invoke "transfer" (i32.const 9)) (i32.const 42))
