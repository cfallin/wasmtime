;;! multiloop = true

;; Entry must supply P0.
(assert_invalid
  (module (func multiloop label (param i32) drop end))
  "type mismatch")

;; Final completion uses the last body's result signature.
(assert_invalid
  (module (func (result i32)
    multiloop label label (result i32) i64.const 1 end))
  "type mismatch")

;; Even unreachable fallthrough must have matching adjacent signatures.
(assert_invalid
  (module (func
    multiloop label (result i32) unreachable label end))
  "multiloop fallthrough types differ")

;; Branches to siblings use their parameters, not their results.
(assert_invalid
  (module (func
    multiloop
      label (result i32) i64.const 1 br 1
      label (param i32) drop
    end))
  "type mismatch")

;; Label zero is the first body's entry signature in every body.
(assert_invalid
  (module (func (result i32)
    i32.const 0
    multiloop
      label (param i32) (result i64) drop i64.const 1
      label (param i64) (result i32) br 0
    end))
  "type mismatch")

;; br_if requires arguments even when its condition is statically false.
(assert_invalid
  (module (func
    multiloop
      label (result i32) i32.const 0 br_if 1 i32.const 1
      label (param i32) drop
    end))
  "type mismatch")

;; All br_table targets must agree on their branch signatures.
(assert_invalid
  (module (func
    multiloop
      label (result i32) i32.const 1 i32.const 0 br_table 0 1
      label (param i32) drop
    end))
  "type mismatch")

;; N labels plus the enclosing function; there is no extra region exit label.
(assert_invalid
  (module (func multiloop label br 3 label end))
  "unknown label")

;; Zero bodies.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\07\01\05\00\fc\17\00\0b")
  "invalid multiloop body count")

;; Missing first label.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\09\01\07\00\fc\17\01\40\0b\0b")
  "multiloop header must be followed by label")

;; Instruction before first label.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\0c\01\0a\00\fc\17\01\40\01\fc\18\0b\0b")
  "multiloop header must be followed by label")

;; Missing sibling label.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\0c\01\0a\00\fc\17\02\40\40\fc\18\0b\0b")
  "multiloop has missing labels")

;; Extra sibling label.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\0d\01\0b\00\fc\17\01\40\fc\18\fc\18\0b\0b")
  "label outside multiloop or too many labels")

;; Label outside a region.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\06\01\04\00\fc\18\0b")
  "label outside multiloop or too many labels")

;; Label inside an unclosed nested block.
(assert_malformed
  (module binary "\00\61\73\6d\01\00\00\00\01\04\01\60\00\00\03\02\01\00\0a\10\01\0e\00\fc\17\01\40\fc\18\02\40\fc\18\0b\0b\0b")
  "label outside multiloop or too many labels")
