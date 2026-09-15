;;! multiloop = false

(assert_invalid
  (module (func multiloop label end))
  "multiloop feature required")
