# Guest (Wasm-only) Debugging with `lldb`

The following steps describe how to use `lldb` to debug the Wasm guest
on its own -- as if it were running on a virtual Wasm-instruction-set
computer with hostcalls as single-step indivisible actions. This
functionality, called "guest debugging", allows for disassembly and
single instruction stepping at the Wasm bytecode level.

1. Compile your WebAssembly with debug info enabled, usually `-g`; for
   example:

    ```console
    clang foo.c -g -o foo.wasm
    ```

2. Ensure that you have a build of Wasmtime that has the `gdbstub`
   feature enabled, which is off by default:
   
   - Published CLI binary releases (downloaded manually, via our install
     script, or `cargo binstall`) already have this feature.
   - If building from source *in-tree*, use `cargo build --features gdbstub`.
     (Note that this feature flag does not work when building from published
     crates with `cargo install`.)

3. Run Wasmtime, enabling the gdbstub server:

    ```console
    wasmtime run -g 1234 foo.wasm
    
    Debugger: Debugger listening on 127.0.0.1:1234
    Debugger: In LLDB, attach with: process connect --plugin wasm connect://127.0.0.1:1234
    ```
    
   This will start a "debug server" waiting on local TCP port 1234 for
   a debugger to connect. Execution will not start until the debugger
   connects and issues a "continue" command.
   
4. Run LLDB, connect and debug.

   You'll need a recent version of LLDB (v32 or later) with Wasm
   support enabled. The [wasi-sdk] distribution provides such a
   build.
   
    ```console
    /opt/wasi-sdk/bin/lldb
    (lldb) process connect --plugin wasm connect://0:1234
    Process 1 stopped
    * thread #1, stop reason = signal SIGTRAP
        frame #0: 0x00000000
    error: memory read failed for 0x0
    (lldb) b my_function
    (lldb) continue
    ```
    
   and use LLDB like normal, setting breakpoints, continuing and
   stepping, examining memory and variable state, etc.
   
This functionality should work on any platform that Wasmtime runs on:
debugging is based on code instrumentation, so does not depend on any
native system debugging interfaces or introspection capabilities; and
is supported on all native-compilation ISAs and on Pulley, Wasmtime's
bytecode platform that runs everywhere.
   
[wasi-sdk]: https://github.com/WebAssembly/wasi-sdk/

## Built-in interactive debugger (`--debug-tui`)

*This is experimental.* Wasmtime also includes a self-contained,
gdb-style debugger that does not require a special LLDB build. It
understands DWARF directly (including C structs, Rust enums, and a few
standard-library types such as Rust's `String`, `Vec`, `Box` and
`HashMap`) and also offers a Wasm "virtual machine" view: the operand
stack, Wasm locals, globals, memories, tables, and GC objects.

1. Build Wasmtime with the `debug-tui` feature
   (`cargo build --features debug-tui`).

2. Run your program with the debugger listening on a TCP port:

    ```console
    wasmtime run --debug-tui 1234 foo.wasm
    Debugger: Debugger listening on 127.0.0.1:1234
    ```

3. Connect with `nc` or `telnet` and type commands:

    ```console
    $ nc localhost 1234
    (wasmtime-debug) b foo.c:42
    Breakpoint 1 at pc 0x3c7 in main at foo.c:42
    (wasmtime-debug) continue

    Breakpoint 1, main () at foo.c:42
    42	  int r = area(&shape);
    (wasmtime-debug) print shape.points[1]
    %1 = struct Point { .x = 4, .y = 0 }
    (wasmtime-debug) print %1.x
    %2 = 4
    (wasmtime-debug) wasm-stack
    Operand stack is empty.
    ```

   Type `help` for the full list of commands. Values are named with
   simple "access paths" (`p->x`, `arr[2].y`, `*p`, `&x`, `t.0`) rooted at
   variables, value-history entries (`%1`, `%`, `%%2`), Wasm-level
   entities named in the module's name section (`$x` for a local
   declared as `(local $x i32)`, or a named global or table), or
   built-in indexed entities (`%local3`, `%stack0`, `%global1`,
   `%table0[4]`, `%pc`).

The debugger is itself a WebAssembly component running in Wasmtime's
debugger world, and by default has no filesystem access. To let it
display source files referenced by the debug info, give it read-only
access to the relevant directories with `-Ddebug-fs` (which may be
repeated):

```console
wasmtime run --debug-tui 1234 -Ddebug-fs=$HOME/src/myproject foo.wasm
```

### Debug Adapter Protocol (`--debug-dap`)

The same built-in debugger can serve the [Debug Adapter Protocol]
(DAP) instead of a text interface, for use with DAP clients such as
editors:

```console
wasmtime run --debug-dap 1234 -Ddebug-fs=$HOME/src/myproject foo.wasm
```

Point a DAP client at port 1234 (for example, a VS Code launch
configuration with `"debugServer": 1234`). The server presents:

- stack frames (including inlined frames), with source locations and
  instruction-pointer references for disassembly views;
- scopes for source-level arguments and locals, and for the Wasm machine
  state: Wasm locals, the operand stack, and globals, all of which expand
  into trees (struct fields, sequence elements, map entries, enum
  payloads, Wasm GC objects);
- source, function, and instruction breakpoints; stepping by line or by
  Wasm instruction (`granularity: "instruction"`);
- watch/hover evaluation of access paths, `readMemory` on linear memory,
  and `disassemble`.

Any command of the text interface can also be typed in the client's
debug console (the `repl` evaluation context), e.g. `wasm-table`,
`x/16xb &buf`, or `info functions`.

[Debug Adapter Protocol]: https://microsoft.github.io/debug-adapter-protocol/
