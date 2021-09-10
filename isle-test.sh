#!/bin/sh
../isle/target/release/isle \
    -i cranelift/codegen/src/isa/prelude.isle \
    -i cranelift/codegen/src/isa/clif.isle \
    -i cranelift/codegen/src/isa/x64/machine.isle \
    -i cranelift/codegen/src/isa/x64/lower.isle \
    -o lower.rs 
