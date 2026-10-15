//! Developer tool: run a wasip1 program in-process under Wasmtime's
//! debug API, stop at a source location, and run a few inspection
//! commands using this library.
//!
//! ```text
//! cargo run -p wasmtime-internal-debug-info --example native-debug -- \
//!     prog.wasm main.c:42 'print x' 'print p->y' locals bt
//! ```
//!
//! Commands: `print <path>`, `print/x <path>`, `ptype <path>`,
//! `locals`, `bt`.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wasmtime::{
    AsContextMut, Config, DebugEvent, DebugHandler, Engine, FrameHandle, Linker, Module, ModulePC,
    Store, StoreContextMut, Val,
};
use wasmtime_internal_debug_info::{
    ModuleInfo, Target, WasmVal,
    format::{FormatOptions, describe_type, format_value},
    path,
    value::{Ctx, Value},
};
use wasmtime_wasi::p1::WasiP1Ctx;

struct FrameTarget<'a, 'b> {
    store: RefCell<StoreContextMut<'a, WasiP1Ctx>>,
    frame: &'b FrameHandle,
}

fn val(v: Val) -> anyhow::Result<WasmVal> {
    Ok(match v {
        Val::I32(x) => WasmVal::I32(x as u32),
        Val::I64(x) => WasmVal::I64(x as u64),
        Val::F32(x) => WasmVal::F32(x),
        Val::F64(x) => WasmVal::F64(x),
        Val::V128(x) => WasmVal::V128(x.as_u128()),
        _ => anyhow::bail!("reference value"),
    })
}

impl Target for FrameTarget<'_, '_> {
    fn read_memory(&self, memory: u32, addr: u64, len: u64) -> anyhow::Result<Vec<u8>> {
        let mut store = self.store.borrow_mut();
        let inst = self
            .frame
            .instance(&mut *store)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mem = inst
            .debug_memory(&mut *store, memory)
            .ok_or_else(|| anyhow::anyhow!("no memory {memory}"))?;
        let data = mem.data(&*store);
        let start = usize::try_from(addr)?;
        let end = start + usize::try_from(len)?;
        data.get(start..end)
            .map(|d| d.to_vec())
            .ok_or_else(|| anyhow::anyhow!("out of bounds read at {addr:#x}"))
    }
    fn local(&self, index: u32) -> anyhow::Result<WasmVal> {
        let mut store = self.store.borrow_mut();
        val(self
            .frame
            .local(&mut *store, index)
            .map_err(|e| anyhow::anyhow!("{e}"))?)
    }
    fn global(&self, index: u32) -> anyhow::Result<WasmVal> {
        let mut store = self.store.borrow_mut();
        let inst = self
            .frame
            .instance(&mut *store)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let g = inst
            .debug_global(&mut *store, index)
            .ok_or_else(|| anyhow::anyhow!("no global {index}"))?;
        val(g.get(&mut *store))
    }
    fn operand(&self, index: u32) -> anyhow::Result<WasmVal> {
        let mut store = self.store.borrow_mut();
        val(self
            .frame
            .stack(&mut *store, index)
            .map_err(|e| anyhow::anyhow!("{e}"))?)
    }
}

#[derive(Clone)]
struct Handler {
    bytes: Arc<Vec<u8>>,
    commands: Arc<Vec<String>>,
    done: Arc<AtomicBool>,
}

impl DebugHandler for Handler {
    type Data = WasiP1Ctx;
    fn handle(
        &self,
        mut store: StoreContextMut<'_, WasiP1Ctx>,
        event: DebugEvent<'_>,
    ) -> impl Future<Output = ()> + Send {
        if matches!(event, DebugEvent::Breakpoint) && !self.done.swap(true, Ordering::SeqCst) {
            run_commands(&self.bytes, &self.commands, &mut store);
        }
        async {}
    }
}

fn run_commands(bytes: &[u8], commands: &[String], store: &mut StoreContextMut<'_, WasiP1Ctx>) {
    let info = ModuleInfo::parse(bytes.to_vec()).unwrap();
    let debug = info.debug.as_ref().unwrap();
    let frame = store.debug_exit_frames().next().unwrap();
    let (_, pc) = frame
        .wasm_function_index_and_pc(&mut *store)
        .unwrap()
        .unwrap();
    let addr = info.pc_to_dwarf(pc.raw()).unwrap();
    let loc = debug.source_loc(addr).unwrap();
    println!(
        "stopped at {}:{} (pc {:#x})",
        debug.file_name(loc.file),
        loc.line,
        pc.raw()
    );
    let target = FrameTarget {
        store: RefCell::new(store.as_context_mut()),
        frame: &frame,
    };
    let func = debug.function_at(addr).unwrap();
    let fb = debug.frame_base(func, addr, &target).unwrap();
    let frames = debug.frames_at(addr).unwrap();
    let ctx = Ctx::new(debug, &target);
    for cmd in commands {
        println!("> {cmd}");
        let (verb, rest) = cmd.split_once(' ').unwrap_or((cmd.as_str(), ""));
        match verb {
            "bt" => {
                for (i, f) in frames.iter().enumerate() {
                    println!(
                        "  #{i} {}{}",
                        f.name,
                        if f.inlined { " [inlined]" } else { "" }
                    );
                }
            }
            "locals" => {
                for v in &frames[0].vars {
                    let s = match ctx.variable(v, addr, fb) {
                        Ok(val) => format_value(&ctx, &val, &FormatOptions::default()),
                        Err(e) => format!("<{e}>"),
                    };
                    println!("  {} = {s}", v.name);
                }
            }
            "print" | "print/x" | "ptype" | "print/r" => {
                let p = match path::parse(rest) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("  error: {e}");
                        continue;
                    }
                };
                let mut root = |p: &path::Path| -> anyhow::Result<Value> {
                    match p {
                        path::Path::Var(name) => {
                            for f in &frames {
                                if let Some(v) = f.find_var(name) {
                                    return ctx.variable(v, addr, fb);
                                }
                            }
                            if let Some(g) = debug.find_global(name) {
                                let ty = debug.global_type(g)?.unwrap();
                                let loc = debug.global_loc(g)?;
                                let l = debug.eval_loc(g.die.unit, &loc, addr, None, &target)?;
                                return ctx.from_location(ty, l);
                            }
                            anyhow::bail!("no variable `{name}`")
                        }
                        _ => anyhow::bail!("unsupported root"),
                    }
                };
                match path::eval(&ctx, &p, &mut root) {
                    Ok(v) if verb == "ptype" => println!("  {}", describe_type(debug, v.ty)),
                    Ok(v) => {
                        let opts = FormatOptions {
                            hex: verb == "print/x",
                            raw: verb == "print/r",
                            ..Default::default()
                        };
                        println!("  {}", format_value(&ctx, &v, &opts));
                    }
                    Err(e) => println!("  error: {e}"),
                }
            }
            _ => println!("  unknown command"),
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wasm = std::fs::read(&args[1])?;
    let (file, line) = args[2].rsplit_once(':').unwrap();
    let line: u32 = line.parse()?;

    let info = ModuleInfo::parse(wasm.clone())?;
    let debug = info.debug.as_ref().expect("no DWARF");
    let files = debug.find_files(file);
    let (addrs, actual) = debug
        .line_addresses(&files, line)
        .expect("no code at that line");
    println!("breakpoint at line {actual}: {} location(s)", addrs.len());

    let mut config = Config::new();
    config.guest_debug(true);
    let engine = Engine::new(&config).map_err(|e| anyhow::anyhow!("{e}"))?;
    let module = Module::new(&engine, &wasm).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut linker = Linker::new(&engine);
    wasmtime_wasi::p1::add_to_linker_async(&mut linker, |t| t)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let ctx = wasmtime_wasi::WasiCtxBuilder::new()
        .inherit_stdout()
        .build_p1();
    let mut store = Store::new(&engine, ctx);
    store.set_debug_handler(Handler {
        bytes: Arc::new(wasm.clone()),
        commands: Arc::new(args[3..].to_vec()),
        done: Arc::new(AtomicBool::new(false)),
    });
    {
        let mut edit = store.edit_breakpoints().unwrap();
        for a in addrs {
            let pc = info.dwarf_to_pc(a).unwrap();
            edit.add_breakpoint(&module, ModulePC::new(pc))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }
    let instance = linker
        .instantiate_async(&mut store, &module)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let _ = start.call_async(&mut store, ()).await;
    Ok(())
}
