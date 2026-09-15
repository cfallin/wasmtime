;;! multiloop = true

(module
  ;; Body one has no predecessor when its code is encountered. Body two
  ;; subsequently branches back to it, so it must still be translated.
  (func (export "late-predecessor") (result i32)
    multiloop
      label (result i32) i32.const 7 br 2
      label (param i32) (result i32) i32.const 1 i32.add br 3
      label (param i32) (result i32) br 1
    end)

  ;; A genuinely dead body is still validated, but never executed.
  (func (export "dead-body") (result i32)
    multiloop
      label (result i32) i32.const 9 br 2
      label (param i32) (result i32) unreachable
      label (param i32) (result i32)
    end)

  (func (export "dead-region") (result i32)
    i32.const 5 return
    multiloop
      label (result i32) i32.const 6
      label (param i32) (result i32) i32.const 1 i32.add
    end)

  ;; Stack polymorphism applies after an actual terminator within a body.
  (func (export "polymorphic") (result i32)
    multiloop
      label (result i32) i32.const 11 br 1 i32.add
      label (param i32) (result i32)
    end))

(assert_return (invoke "late-predecessor") (i32.const 8))
(assert_return (invoke "dead-body") (i32.const 9))
(assert_return (invoke "dead-region") (i32.const 5))
(assert_return (invoke "polymorphic") (i32.const 11))

;; Lack of an incoming edge does not make a body's stack polymorphic.
(assert_invalid
  (module (func
    multiloop
      label br 2
      label i32.add drop
    end))
  "type mismatch")

;; Delimiters reset unreachable state: a preceding unreachable fallthrough
;; cannot excuse missing operands or missing results in the next body.
(assert_invalid
  (module (func
    multiloop label unreachable label i32.add drop end))
  "type mismatch")
(assert_invalid
  (module (func (result i32)
    multiloop label unreachable label (result i32) end))
  "type mismatch")

;; As with ordinary Wasm blocks, bodies must be typed independently even
;; when the entire region occurs in an unreachable enclosing context.
(assert_invalid
  (module (func
    unreachable
    multiloop label i32.add drop end))
  "type mismatch")

;; Adjacency requires exact signature equality even without fallthrough.
(assert_invalid
  (module (func
    multiloop
      label (result i32) unreachable
      label (param i64) drop
    end))
  "multiloop fallthrough types differ")
