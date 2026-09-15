;;! multiloop = true
;;! exceptions = true

(module
  (tag $e (param i32))
  (func $throw (param i32) local.get 0 throw $e)

  ;; Catch depths refer to the enclosing labels before the try_table is
  ;; pushed. Depth one therefore targets the second body directly.
  (func (export "catch-entry") (result i32)
    multiloop
      label (result i32)
        try_table (result i32) (catch $e 1)
          i32.const 41 call $throw i32.const 99
        end
      label (param i32) (result i32) i32.const 1 i32.add
    end)

  ;; A transfer out of a nested try_table discards its handler scope.
  ;; The throw in the sibling body must reach the enclosing handler.
  (func (export "handler-scope") (result i32)
    block $outer (result i32)
      try_table (result i32) (catch $e $outer)
        multiloop
          label (result i32)
            block $inner (result i32)
              try_table (result i32) (catch $e $inner)
                i32.const 7 br 3
              end
              i32.const 1000 i32.add
            end
          label (param i32) (result i32)
            call $throw unreachable
        end
      end
    end))

(assert_return (invoke "catch-entry") (i32.const 42))
(assert_return (invoke "handler-scope") (i32.const 7))
