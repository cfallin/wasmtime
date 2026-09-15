;;! multiloop = true

(module
  (type $pair (func (param i32 i64) (result i32 i64)))

  ;; A single body has ordinary loop entry and final fallthrough semantics.
  (func (export "single") (param i32) (result i32)
    local.get 0
    multiloop
      label (param i32) (result i32)
        i32.const 1 i32.add
    end)

  ;; Each delimiter transfers the preceding results to the next parameters.
  (func (export "fallthrough") (param i32 i64) (result i32 i64)
    local.get 0 local.get 1
    multiloop
      label (type $pair)
        i64.const 2 i64.add
      label (type $pair)
        i64.const 3 i64.mul
    end)

  (func (export "folded") (param i32) (result i32)
    local.get 0
    (multiloop
      (label (param i32) (result i32) (i32.const 2) (i32.add))
      (label (param i32) (result i32) (i32.const 3) (i32.mul))))

  ;; br 0 always uses P0, even in a body with a different signature.
  (func (export "different-signatures") (param i32) (result i32)
    multiloop
      label (result i64)
        local.get 0 i32.const 1 i32.add local.set 0
        i64.const 42
      label (param i64) (result i32)
        drop
        local.get 0 i32.const 3 i32.lt_s br_if 0
        local.get 0
    end)

  ;; Body zero enters either cyclic body. Neither body one nor body two
  ;; dominates the other; both locals and entry operands cross cyclic edges.
  (func (export "cycle") (param $n i32) (result i32) (local $sum i32)
    local.get $n
    multiloop
      label (param i32) (result i32)
        local.get $n i32.const 1 i32.and br_if 1
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
    end)

  ;; Nested branches discard temporaries but retain the outer stack prefix.
  ;; The named outer target includes all sibling labels in its relative depth.
  (func (export "nested") (param i32) (result i32)
    i32.const 100
    block $exit (result i32)
      local.get 0
      multiloop
        label (param i32) (result i32)
          block (param i32) (result i32)
            i32.const 9 local.get 0 br_if $exit drop
            i32.const 777 i32.const 8 i32.const 0 br_table 2 2
          end
        label (param i32) (result i32)
          multiloop
            label (param i32) (result i32) i32.const 2 i32.add
            label (param i32) (result i32) i32.const 3 i32.mul
          end
      end
    end
    i32.add)

  ;; The inner region's two labels shift the outer body's target to depth 3.
  (func (export "inner-to-outer") (result i32)
    multiloop
      label (result i32)
        multiloop
          label (result i32) i32.const 12 br 3
          label (param i32) (result i32) unreachable
        end
      label (param i32) (result i32) i32.const 1 i32.add
    end)

  ;; All br_table targets share their entry argument types, including the
  ;; enclosing block. Test each case and the default separately.
  (func (export "table") (param i32) (result i32)
    block $exit (result i32)
      multiloop
        label (result i32)
          i32.const 7 local.get 0 br_table 1 2 $exit
        label (param i32) (result i32)
          i32.const 10 i32.add br $exit
        label (param i32) (result i32)
          i32.const 20 i32.add
      end
    end)

  (func (export "return") (result i32)
    multiloop
      label (result i32) i32.const 17 return
      label (param i32) (result i32) unreachable
    end)

  (func (export "trap")
    multiloop label br 1 label unreachable end))

(assert_return (invoke "single" (i32.const 41)) (i32.const 42))
(assert_return (invoke "fallthrough" (i32.const 7) (i64.const 4)) (i32.const 7) (i64.const 18))
(assert_return (invoke "folded" (i32.const 4)) (i32.const 18))
(assert_return (invoke "different-signatures" (i32.const 0)) (i32.const 3))
(assert_return (invoke "different-signatures" (i32.const 5)) (i32.const 6))
;; Alternating contributions of 3 and 5, with body one's completion falling
;; through body two even when the remaining iteration count has reached zero.
(assert_return (invoke "cycle" (i32.const 1)) (i32.const 8))
(assert_return (invoke "cycle" (i32.const 2)) (i32.const 13))
(assert_return (invoke "cycle" (i32.const 3)) (i32.const 16))
(assert_return (invoke "cycle" (i32.const 4)) (i32.const 21))
(assert_return (invoke "cycle" (i32.const 39)) (i32.const 160))
(assert_return (invoke "nested" (i32.const 0)) (i32.const 130))
(assert_return (invoke "nested" (i32.const 1)) (i32.const 109))
(assert_return (invoke "inner-to-outer") (i32.const 13))
(assert_return (invoke "table" (i32.const 0)) (i32.const 17))
(assert_return (invoke "table" (i32.const 1)) (i32.const 27))
(assert_return (invoke "table" (i32.const 2)) (i32.const 7))
(assert_return (invoke "table" (i32.const -1)) (i32.const 7))
(assert_return (invoke "return") (i32.const 17))
(assert_trap (invoke "trap") "unreachable")
