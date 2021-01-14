#!/bin/bash

set -e

CARGO_VERSION=${CARGO_VERSION:-"+nightly"}

# Get the `all` test binary.

# This seems to be the best way to do it; see
# https://github.com/rust-lang/cargo/issues/1924 for some discussion about how
# to build a test binary and get its filename.
EXE=`cargo $CARGO_VERSION test --quiet --release --test all \
        --features experimental_x64 --features test-programs/test_programs \
        --no-run --message-format=json |
        grep release/deps/all |
        jq .executable |
        sed -e 's/"//g'`

echo "Built test binary: $EXE" >&2

valgrind --sigill-diagnostics=no $EXE wast::Cranelift::spec::
