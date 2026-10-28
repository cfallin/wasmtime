//! Debugger session state and execution control.

use crate::api::{self, Debuggee};
use crate::conn::{Conn, Input};
use crate::wasmval::WVal;
use anyhow::{Result, anyhow, bail};
use futures::{FutureExt, select};
use std::cell::RefCell;
use std::collections::HashMap;
use wasmtime_debug_info::dwarf::{ScopeFrame, SourceLoc};
use wasmtime_debug_info::module::InstrKind;
use wasmtime_debug_info::types::TypeId;
use wasmtime_debug_info::value::Value;
use wasmtime_debug_info::{ModuleInfo, Target, WasmVal};

/// A core Wasm module known to the debugger.
pub struct ModEntry {
    pub handle: api::Module,
    pub info: ModuleInfo,
    pub label: String,
}

/// A core Wasm instance known to the debugger.
pub struct InstEntry {
    pub handle: api::Instance,
    pub module: usize,
}

/// A physical (Wasm) stack frame.
pub struct PhysFrame {
    pub frame: api::Frame,
    pub inst: usize,
    pub module: usize,
    /// The frame's PC. For the innermost frame this is the current
    /// instruction; for caller frames it is the return address (the
    /// instruction after the call).
    pub pc: u32,
    /// The PC used for source-level lookups: `pc` for the innermost
    /// frame, `pc - 1` (within the call instruction) for callers.
    pub lookup_pc: u32,
    /// Function index within the module.
    pub func: Option<u32>,
}

/// A logical frame: a physical frame, or an inlined call within one.
pub struct VFrame {
    pub phys: usize,
    /// Number of inlined frames below this one within the same physical
    /// frame (0 for the innermost).
    pub scope: Option<ScopeFrame>,
    pub inlined: bool,
    pub name: String,
    pub loc: Option<SourceLoc>,
}

/// A user breakpoint.
pub struct UserBp {
    pub id: u32,
    pub spec: String,
    pub locs: Vec<(usize, u32)>,
    pub temporary: bool,
    pub hits: u32,
}

/// A user watchpoint: guest writes to `len` bytes at `addr` in memory 0
/// of instance `inst` stop execution.
pub struct UserWatch {
    pub id: u32,
    /// What the user asked to watch (an expression, or an address
    /// range).
    pub spec: String,
    pub inst: usize,
    /// The watched memory's `unique-id`.
    pub memory_id: u64,
    pub addr: u64,
    pub len: u64,
    /// The watched value's type (with the module whose debug info
    /// defines it), for showing old and new values.
    pub ty: Option<(usize, TypeId)>,
    pub hits: u32,
}

/// A watchpoint stop: which watchpoints a write hit, with their
/// contents before and after the write.
pub struct WatchReport {
    pub ids: Vec<u32>,
    pub old: Vec<Option<Vec<u8>>>,
    pub new: Vec<Option<Vec<u8>>>,
    /// Whether the stop was reached running backward, in which case
    /// execution is stopped *before* the write.
    pub reverse: bool,
}

/// An entry in the value history.
pub enum HistEntry {
    /// A source-level value (snapshot), with the module whose debug info
    /// describes its type and the instance whose memory it lives in.
    Src {
        module: usize,
        inst: Option<usize>,
        value: Value,
    },
    /// A Wasm-level value.
    Wasm(WVal),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    NotStarted,
    Stopped,
    Exited,
}

/// Why execution stopped, for reporting.
pub enum StopReason {
    Breakpoint(Option<u32>),
    Step,
    Interrupted,
    Trap,
    Exception,
    Finished,
    Watchpoint(WatchReport),
    /// Reverse execution reached the beginning of a replay.
    ReplayBegin,
    /// Forward execution reached the end of a replay.
    ReplayEnd,
}

/// How to resume execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Motion {
    Continue,
    Step,
    ReverseContinue,
    ReverseStep,
}

/// The outcome of a resumption.
pub enum Outcome {
    Stopped(StopReason),
    Exited,
}

pub struct Session<'a> {
    pub d: &'a Debuggee,
    /// Output produced by commands, not yet delivered to the user.
    pub out: String,
    /// The front-end connection.
    pub front: Front,
    pub modules: Vec<ModEntry>,
    mod_by_id: HashMap<u64, usize>,
    pub instances: Vec<InstEntry>,
    inst_by_id: HashMap<u64, usize>,
    pub breakpoints: Vec<UserBp>,
    pub watchpoints: Vec<UserWatch>,
    next_bp: u32,
    installed: HashMap<(usize, u32), u32>,
    pub phys: Vec<PhysFrame>,
    pub vframes: Vec<VFrame>,
    pub selected: usize,
    pub history: Vec<HistEntry>,
    pub state: RunState,
    /// Last listed (module, file, line), for `list` continuation.
    pub list_pos: Option<(usize, u32, u32)>,
    pub source_cache: HashMap<String, Option<Vec<String>>>,
    /// Set when the user disconnects mid-run.
    pub detached: bool,
    /// Whether the debuggee can run backward (it is a replay).
    pub reversible: bool,
    /// The watchpoint stop being reported by the current event.
    pending_watch: Option<WatchReport>,
}

/// The error for reverse execution of a debuggee that is not a replay.
pub const NOT_REVERSIBLE: &str = "Reverse execution requires a recorded execution: record one with \
     `wasmtime run --record=TRACE ...`, then debug its replay with \
     `wasmtime replay --debug-tui=PORT TRACE` (or `--debug-dap`).";

/// Convert a Wasm value resource into a scalar.
pub fn to_wasm_val(v: &api::WasmValue) -> Result<WasmVal> {
    Ok(match v.get_type() {
        api::WasmType::WasmI32 => WasmVal::I32(v.unwrap_i32()),
        api::WasmType::WasmI64 => WasmVal::I64(v.unwrap_i64()),
        api::WasmType::WasmF32 => WasmVal::F32(v.unwrap_f32().to_bits()),
        api::WasmType::WasmF64 => WasmVal::F64(v.unwrap_f64().to_bits()),
        api::WasmType::WasmV128 => {
            let b: [u8; 16] = v
                .unwrap_v128()
                .try_into()
                .map_err(|_| anyhow!("bad v128"))?;
            WasmVal::V128(u128::from_le_bytes(b))
        }
        t => bail!("{} value is not a number", crate::wasmval::type_name(&t)),
    })
}

/// A `Target` over a frame (or just an instance, for memory/globals).
pub struct FrameTarget<'a> {
    pub d: &'a Debuggee,
    pub frame: Option<&'a api::Frame>,
    pub inst: &'a api::Instance,
    locals: RefCell<Option<Vec<api::WasmValue>>>,
    stack: RefCell<Option<Vec<api::WasmValue>>>,
    mem: RefCell<Option<Option<api::Memory>>>,
}

impl<'a> FrameTarget<'a> {
    pub fn new(d: &'a Debuggee, frame: Option<&'a api::Frame>, inst: &'a api::Instance) -> Self {
        FrameTarget {
            d,
            frame,
            inst,
            locals: RefCell::new(None),
            stack: RefCell::new(None),
            mem: RefCell::new(None),
        }
    }
}

fn api_err(e: api::Error) -> anyhow::Error {
    anyhow!("{e:?}")
}

impl Target for FrameTarget<'_> {
    fn read_memory(&self, memory: u32, addr: u64, len: u64) -> Result<Vec<u8>> {
        let mem = if memory == 0 {
            let mut cache = self.mem.borrow_mut();
            if cache.is_none() {
                *cache = Some(self.inst.get_memory(self.d, 0).ok());
            }
            cache.as_ref().unwrap().as_ref().map(|m| m.clone())
        } else {
            self.inst.get_memory(self.d, memory).ok()
        };
        let mem = mem.ok_or_else(|| anyhow!("instance has no memory {memory}"))?;
        mem.get_bytes(self.d, addr, len)
            .map_err(|_| anyhow!("cannot access memory at address {addr:#x}"))
    }

    fn local(&self, index: u32) -> Result<WasmVal> {
        let frame = self.frame.ok_or_else(|| anyhow!("no frame"))?;
        let mut cache = self.locals.borrow_mut();
        if cache.is_none() {
            *cache = Some(frame.get_locals(self.d).map_err(api_err)?);
        }
        let v = cache
            .as_ref()
            .unwrap()
            .get(usize::try_from(index)?)
            .ok_or_else(|| anyhow!("no local {index}"))?;
        to_wasm_val(v)
    }

    fn global(&self, index: u32) -> Result<WasmVal> {
        let g = self.inst.get_global(self.d, index).map_err(api_err)?;
        to_wasm_val(&g.get(self.d).map_err(api_err)?)
    }

    fn operand(&self, index: u32) -> Result<WasmVal> {
        let frame = self.frame.ok_or_else(|| anyhow!("no frame"))?;
        let mut cache = self.stack.borrow_mut();
        if cache.is_none() {
            *cache = Some(frame.get_stack(self.d).map_err(api_err)?);
        }
        let v = cache
            .as_ref()
            .unwrap()
            .get(usize::try_from(index)?)
            .ok_or_else(|| anyhow!("no operand-stack slot {index}"))?;
        to_wasm_val(v)
    }
}

/// The front-end connection: the line-based text interface, or DAP.
pub enum Front {
    Tui(Conn),
    Dap(crate::dap::Transport),
}

impl<'a> Session<'a> {
    /// Queue output text.
    pub fn print(&mut self, s: &str) {
        self.out.push_str(s);
    }

    /// Queue output text followed by a newline.
    pub fn println(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// In the text interface, deliver queued output to the user now. (In
    /// DAP mode, output is collected by the request that produced it.)
    pub async fn flush_output(&mut self) -> Result<()> {
        if let Front::Tui(conn) = &mut self.front {
            let text = std::mem::take(&mut self.out);
            conn.print(&text);
            conn.flush().await?;
        }
        Ok(())
    }

    pub fn new(d: &'a Debuggee, front: Front) -> Self {
        Session {
            d,
            out: String::new(),
            front,
            modules: vec![],
            mod_by_id: HashMap::new(),
            instances: vec![],
            inst_by_id: HashMap::new(),
            breakpoints: vec![],
            watchpoints: vec![],
            next_bp: 1,
            installed: HashMap::new(),
            phys: vec![],
            vframes: vec![],
            selected: 0,
            history: vec![],
            state: RunState::NotStarted,
            list_pos: None,
            source_cache: HashMap::new(),
            detached: false,
            reversible: d.is_reversible(),
            pending_watch: None,
        }
    }

    // -----------------------------------------------------------------
    // Modules and instances.

    fn module_index(&mut self, m: api::Module) -> usize {
        let id = m.unique_id();
        if let Some(i) = self.mod_by_id.get(&id) {
            return *i;
        }
        let bytes = m.bytecode().unwrap_or_default();
        let info = match ModuleInfo::parse(bytes) {
            Ok(i) => i,
            Err(_) => ModuleInfo::parse(&b"\0asm\x01\0\0\0"[..]).unwrap(),
        };
        let idx = self.modules.len();
        let label = match &info.name {
            Some(n) => n.clone(),
            None => format!("module{idx}"),
        };
        self.modules.push(ModEntry {
            handle: m,
            info,
            label,
        });
        self.mod_by_id.insert(id, idx);
        idx
    }

    fn instance_index(&mut self, inst: api::Instance) -> usize {
        let id = inst.unique_id();
        if let Some(i) = self.inst_by_id.get(&id) {
            return *i;
        }
        let module = self.module_index(inst.get_module(self.d));
        let idx = self.instances.len();
        self.instances.push(InstEntry {
            handle: inst,
            module,
        });
        self.inst_by_id.insert(id, idx);
        idx
    }

    /// Discover any new modules and instances.
    pub fn refresh_entities(&mut self) {
        for m in self.d.all_modules() {
            self.module_index(m);
        }
        for i in self.d.all_instances() {
            self.instance_index(i);
        }
    }

    // -----------------------------------------------------------------
    // Stack.

    /// Rebuild the physical and logical frame lists after a stop.
    pub fn refresh_stack(&mut self) {
        self.refresh_entities();
        self.phys.clear();
        self.vframes.clear();
        self.selected = 0;
        for exit in self.d.exit_frames() {
            let mut next = Some(exit);
            while let Some(f) = next {
                next = f.parent_frame(self.d).ok().flatten();
                let (Ok(inst), Ok(pc)) = (f.get_instance(self.d), f.get_pc(self.d)) else {
                    continue;
                };
                let inst = self.instance_index(inst);
                let module = self.instances[inst].module;
                let lookup_pc = if self.phys.is_empty() {
                    pc
                } else {
                    pc.saturating_sub(1)
                };
                let func = self.modules[module]
                    .info
                    .func_at_pc(lookup_pc)
                    .map(|b| b.index);
                self.phys.push(PhysFrame {
                    frame: f,
                    inst,
                    module,
                    pc,
                    lookup_pc,
                    func,
                });
            }
        }
        for (pi, p) in self.phys.iter().enumerate() {
            let m = &self.modules[p.module];
            let scopes = match (&m.info.debug, m.info.pc_to_dwarf(p.lookup_pc)) {
                (Some(dbg), Some(addr)) => {
                    let scopes = dbg.frames_at(addr).unwrap_or_default();
                    let mut locs = vec![dbg.source_loc(addr)];
                    for s in &scopes[..scopes.len().saturating_sub(1)] {
                        locs.push(s.call_site.map(|(file, line)| SourceLoc {
                            file,
                            line,
                            column: 0,
                        }));
                    }
                    scopes.into_iter().zip(locs).collect::<Vec<_>>()
                }
                _ => vec![],
            };
            if scopes.is_empty() {
                let name = p
                    .func
                    .map(|f| m.info.func_display_name(f))
                    .unwrap_or_else(|| "??".into());
                self.vframes.push(VFrame {
                    phys: pi,
                    scope: None,
                    inlined: false,
                    name,
                    loc: None,
                });
            } else {
                for (s, loc) in scopes {
                    self.vframes.push(VFrame {
                        phys: pi,
                        inlined: s.inlined,
                        name: s.name.clone(),
                        scope: Some(s),
                        loc,
                    });
                }
            }
        }
    }

    /// The DWARF address for a physical frame's PC, if its module has
    /// DWARF.
    pub fn dwarf_addr(&self, phys: usize) -> Option<u64> {
        let p = &self.phys[phys];
        let m = &self.modules[p.module];
        m.info.debug.as_ref()?;
        m.info.pc_to_dwarf(p.lookup_pc)
    }

    /// Compute the frame base of a physical frame.
    pub fn frame_base(&self, phys: usize, target: &dyn Target) -> Option<u64> {
        let p = &self.phys[phys];
        let dbg = self.modules[p.module].info.debug.as_ref()?;
        let addr = self.dwarf_addr(phys)?;
        let f = dbg.function_at(addr)?;
        dbg.frame_base(f, addr, target).ok().flatten()
    }

    // -----------------------------------------------------------------
    // Breakpoints.

    fn install(&mut self, module: usize, pc: u32) -> Result<()> {
        let count = self.installed.entry((module, pc)).or_insert(0);
        if *count == 0 {
            self.modules[module]
                .handle
                .add_breakpoint(self.d, pc)
                .map_err(|e| anyhow!("failed to set breakpoint at {pc:#x}: {e:?}"))?;
        }
        *count += 1;
        Ok(())
    }

    fn uninstall(&mut self, module: usize, pc: u32) {
        if let Some(count) = self.installed.get_mut(&(module, pc)) {
            *count -= 1;
            if *count == 0 {
                self.installed.remove(&(module, pc));
                let _ = self.modules[module].handle.remove_breakpoint(self.d, pc);
            }
        }
    }

    /// Add a user breakpoint at the given locations.
    pub fn add_breakpoint(
        &mut self,
        spec: String,
        locs: Vec<(usize, u32)>,
        temporary: bool,
    ) -> Result<u32> {
        for &(m, pc) in &locs {
            self.install(m, pc)?;
        }
        let id = self.next_bp;
        self.next_bp += 1;
        self.breakpoints.push(UserBp {
            id,
            spec,
            locs,
            temporary,
            hits: 0,
        });
        Ok(id)
    }

    pub fn delete_breakpoint(&mut self, id: u32) -> bool {
        if let Some(i) = self.watchpoints.iter().position(|w| w.id == id) {
            self.remove_watch(i);
            return true;
        }
        let Some(i) = self.breakpoints.iter().position(|b| b.id == id) else {
            return false;
        };
        let bp = self.breakpoints.remove(i);
        for (m, pc) in bp.locs {
            self.uninstall(m, pc);
        }
        true
    }

    /// Add a watchpoint on `len` bytes at `addr` in memory 0 of instance
    /// `inst`.
    pub fn add_watchpoint(
        &mut self,
        spec: String,
        inst: usize,
        addr: u64,
        len: u64,
        ty: Option<(usize, TypeId)>,
    ) -> Result<u32> {
        if len == 0 {
            bail!("Cannot watch zero bytes.");
        }
        let mem = self.instances[inst]
            .handle
            .get_memory(self.d, 0)
            .map_err(|_| anyhow!("The instance has no linear memory."))?;
        mem.add_watchpoint(self.d, addr, len).map_err(|e| match e {
            api::Error::OutOfBounds => {
                anyhow!("Cannot watch {len} bytes at {addr:#x}: beyond the end of linear memory.")
            }
            api::Error::UnsupportedType => anyhow!("Shared memories cannot be watched."),
            e => anyhow!("Cannot watch {addr:#x}: {e:?}"),
        })?;
        let id = self.next_bp;
        self.next_bp += 1;
        self.watchpoints.push(UserWatch {
            id,
            spec,
            inst,
            memory_id: mem.unique_id(),
            addr,
            len,
            ty,
            hits: 0,
        });
        Ok(id)
    }

    /// Remove watchpoint `i`. The debuggee watches bytes, not ranges, so
    /// re-watch the bytes of any remaining overlapping watchpoints.
    fn remove_watch(&mut self, i: usize) {
        let w = self.watchpoints.remove(i);
        let Ok(mem) = self.instances[w.inst].handle.get_memory(self.d, 0) else {
            return;
        };
        let _ = mem.remove_watchpoint(self.d, w.addr, w.len);
        for o in &self.watchpoints {
            if o.memory_id == w.memory_id && o.addr < w.addr + w.len && w.addr < o.addr + o.len {
                let _ = mem.add_watchpoint(self.d, o.addr, o.len);
            }
        }
    }

    /// The current contents of watchpoint `i`'s bytes.
    fn watch_contents(&self, i: usize) -> Option<Vec<u8>> {
        let w = &self.watchpoints[i];
        let mem = self.instances[w.inst].handle.get_memory(self.d, 0).ok()?;
        mem.get_bytes(self.d, w.addr, w.len).ok()
    }

    /// The user breakpoint at the current top-of-stack location, if any.
    fn user_bp_here(&self) -> Option<u32> {
        let p = self.phys.first()?;
        self.breakpoints
            .iter()
            .find(|b| b.locs.contains(&(p.module, p.pc)))
            .map(|b| b.id)
    }

    // -----------------------------------------------------------------
    // Execution control.

    /// Resume and wait for the next event. A watchpoint event is recorded
    /// in `pending_watch` with the watched values before and after the
    /// write; running forward, the write is first stepped over, so that
    /// the stop is after it (as in gdb).
    async fn resume(&mut self, m: Motion) -> Result<api::Event> {
        let ev = self.resume_raw(m).await?;
        let api::Event::Watchpoint(hit) = &ev else {
            return Ok(ev);
        };
        let mem_id = hit.memory.unique_id();
        let (start, end) = (hit.address, hit.address.saturating_add(hit.len));
        let hits: Vec<usize> = (0..self.watchpoints.len())
            .filter(|&i| {
                let w = &self.watchpoints[i];
                w.memory_id == mem_id && w.addr < end && start < w.addr + w.len
            })
            .collect();
        let old: Vec<_> = hits.iter().map(|&i| self.watch_contents(i)).collect();
        let reverse = matches!(m, Motion::ReverseContinue | Motion::ReverseStep);
        let new = if reverse {
            // Stopped before the write: apply the bytes being stored.
            hits.iter()
                .zip(&old)
                .map(|(&i, old)| {
                    let w = &self.watchpoints[i];
                    let mut bytes = old.clone()?;
                    let value = hit.value.as_ref()?;
                    for (k, b) in value.iter().enumerate() {
                        let a = hit.address + k as u64;
                        if a >= w.addr && a < w.addr + w.len {
                            bytes[(a - w.addr) as usize] = *b;
                        }
                    }
                    Some(bytes)
                })
                .collect()
        } else {
            let after = self.resume_raw(Motion::Step).await?;
            if !matches!(after, api::Event::Breakpoint) {
                return Ok(after);
            }
            hits.iter().map(|&i| self.watch_contents(i)).collect()
        };
        for &i in &hits {
            self.watchpoints[i].hits += 1;
        }
        self.pending_watch = Some(WatchReport {
            ids: hits.iter().map(|&i| self.watchpoints[i].id).collect(),
            old,
            new,
            reverse,
        });
        Ok(ev)
    }

    /// Resume and wait for the next event. While continuing, user input
    /// is watched for interrupt requests.
    async fn resume_raw(&mut self, m: Motion) -> Result<api::Event> {
        let mut r = match m {
            Motion::Continue => api::Resumption::continue_(self.d),
            Motion::Step => api::Resumption::single_step(self.d),
            Motion::ReverseContinue | Motion::ReverseStep => {
                api::Resumption::reverse(self.d, m == Motion::ReverseStep).map_err(|e| match e {
                    api::Error::NotReversible => anyhow!("{NOT_REVERSIBLE}"),
                    e => anyhow!("Cannot run backward: {e:?}"),
                })?
            }
        };
        if m == Motion::Step || m == Motion::ReverseStep {
            r.wait().await;
            return r.result(self.d).map_err(api_err);
        }
        enum Wake {
            Done,
            Tui(Input),
            Dap(Option<serde_json::Value>),
        }
        let mut announced = false;
        loop {
            let wake = match &mut self.front {
                Front::Tui(conn) => select! {
                    _ = r.wait().fuse() => Wake::Done,
                    input = conn.read_input().fuse() => Wake::Tui(input?),
                },
                Front::Dap(t) => select! {
                    _ = r.wait().fuse() => Wake::Done,
                    msg = t.read_message().fuse() => Wake::Dap(msg?),
                },
            };
            match wake {
                Wake::Done => break,
                Wake::Tui(Input::Interrupt) => self.d.interrupt(),
                Wake::Tui(Input::Line(l)) if l.trim() == "interrupt" => self.d.interrupt(),
                Wake::Tui(Input::Eof) | Wake::Dap(None) => {
                    // User went away: detach and let the program run to
                    // completion.
                    self.detached = true;
                    self.remove_all_breakpoints();
                }
                Wake::Tui(Input::Line(_)) => {
                    if !announced {
                        if let Front::Tui(conn) = &mut self.front {
                            conn.println(
                                "The program is running; type `interrupt` or press Ctrl-C to stop it.",
                            );
                            let _ = conn.flush().await;
                        }
                        announced = true;
                    }
                }
                Wake::Dap(Some(req)) => self.dap_request_while_running(req).await?,
            }
        }
        r.result(self.d).map_err(api_err)
    }

    pub fn remove_all_breakpoints(&mut self) {
        let keys: Vec<_> = self.installed.keys().copied().collect();
        for (m, pc) in keys {
            let _ = self.modules[m].handle.remove_breakpoint(self.d, pc);
        }
        self.installed.clear();
        self.breakpoints.clear();
        while !self.watchpoints.is_empty() {
            self.remove_watch(0);
        }
    }

    /// Classify an event and update state.
    fn on_event(&mut self, ev: &api::Event) -> Option<Outcome> {
        match ev {
            api::Event::Complete => {
                self.state = RunState::Exited;
                self.phys.clear();
                self.vframes.clear();
                Some(Outcome::Exited)
            }
            api::Event::Trap => {
                self.state = RunState::Stopped;
                self.refresh_stack();
                Some(Outcome::Stopped(StopReason::Trap))
            }
            api::Event::Interrupted => {
                self.state = RunState::Stopped;
                self.refresh_stack();
                Some(Outcome::Stopped(StopReason::Interrupted))
            }
            api::Event::Exception(_) => {
                self.state = RunState::Stopped;
                self.refresh_stack();
                Some(Outcome::Stopped(StopReason::Exception))
            }
            api::Event::InjectedCallReturn(_) => None,
            api::Event::Watchpoint(_) => {
                self.state = RunState::Stopped;
                self.refresh_stack();
                let report = self.pending_watch.take().unwrap_or(WatchReport {
                    ids: vec![],
                    old: vec![],
                    new: vec![],
                    reverse: false,
                });
                Some(Outcome::Stopped(StopReason::Watchpoint(report)))
            }
            api::Event::ReplayBegin => {
                self.refresh_stack();
                // Before any Wasm has run, as if the program had not
                // started.
                self.state = if self.phys.is_empty() {
                    RunState::NotStarted
                } else {
                    RunState::Stopped
                };
                Some(Outcome::Stopped(StopReason::ReplayBegin))
            }
            api::Event::ReplayEnd => {
                self.state = RunState::Stopped;
                self.refresh_stack();
                Some(Outcome::Stopped(StopReason::ReplayEnd))
            }
            api::Event::Breakpoint => {
                self.state = RunState::Stopped;
                self.refresh_stack();
                None
            }
        }
    }

    fn require_running(&self) -> Result<()> {
        match self.state {
            RunState::Exited => bail!("The program is not being run."),
            _ => Ok(()),
        }
    }

    /// Continue until a breakpoint or other event.
    pub async fn cont(&mut self) -> Result<Outcome> {
        self.require_running()?;
        loop {
            let ev = self.resume(Motion::Continue).await?;
            if let Some(o) = self.on_event(&ev) {
                return Ok(o);
            }
            if self.detached {
                continue;
            }
            let bp = self.user_bp_here();
            if bp.is_none() && !matches!(ev, api::Event::Breakpoint) {
                continue;
            }
            return Ok(Outcome::Stopped(StopReason::Breakpoint(bp)));
        }
    }

    /// Single-step one Wasm instruction.
    pub async fn stepi(&mut self) -> Result<Outcome> {
        self.require_running()?;
        let ev = self.resume(Motion::Step).await?;
        if let Some(o) = self.on_event(&ev) {
            return Ok(o);
        }
        Ok(Outcome::Stopped(StopReason::Step))
    }

    /// Run until the physical stack depth is `depth` and the top frame is
    /// at (`module`, `pc`), using a temporary breakpoint there. Stops
    /// early (returning `Some`) for user breakpoints or other events.
    async fn run_to(&mut self, module: usize, pc: u32, depth: usize) -> Result<Option<Outcome>> {
        self.install(module, pc)?;
        let result = loop {
            let ev = match self.resume(Motion::Continue).await {
                Ok(ev) => ev,
                Err(e) => break Err(e),
            };
            if let Some(o) = self.on_event(&ev) {
                break Ok(Some(o));
            }
            let at_target = self
                .phys
                .first()
                .is_some_and(|p| p.module == module && p.pc == pc);
            if at_target && self.phys.len() <= depth {
                break Ok(None);
            }
            if let Some(bp) = self.user_bp_here() {
                break Ok(Some(Outcome::Stopped(StopReason::Breakpoint(Some(bp)))));
            }
            if !matches!(ev, api::Event::Breakpoint) {
                break Ok(Some(Outcome::Stopped(StopReason::Interrupted)));
            }
        };
        if self.state != RunState::Exited {
            self.uninstall(module, pc);
        }
        result
    }

    /// The return point in the caller of physical frame `phys`: (module,
    /// pc of the instruction after the call, resulting stack depth).
    fn return_point(&self, phys: usize) -> Option<(usize, u32, usize)> {
        // Caller frames' PCs are already return addresses.
        let caller = self.phys.get(phys + 1)?;
        Some((caller.module, caller.pc, self.phys.len() - phys - 1))
    }

    /// Step out of the top physical frame.
    async fn step_out_phys(&mut self) -> Result<Option<Outcome>> {
        match self.return_point(0) {
            Some((m, pc, depth)) => self.run_to(m, pc, depth).await,
            None => {
                // No Wasm caller: just continue.
                Ok(Some(self.cont().await?))
            }
        }
    }

    fn top_line(&self) -> Option<(u32, u32)> {
        self.vframes.first()?.loc.as_ref().map(|l| (l.file, l.line))
    }

    /// Whether the top frame's PC is at the start of an `is_stmt`
    /// line-table row.
    fn at_stmt_start(&self) -> bool {
        let Some(addr) = self.dwarf_addr(0) else {
            return false;
        };
        let m = &self.modules[self.phys[0].module];
        let Some(dbg) = &m.info.debug else {
            return false;
        };
        dbg.line_row(addr)
            .is_some_and(|r| r.addr == addr && r.is_stmt)
    }

    /// Source-level step. If `over` is true, calls (including inlined
    /// calls) are stepped over (`next`); otherwise they are stepped into
    /// (`step`).
    pub async fn step_line(&mut self, over: bool) -> Result<Outcome> {
        self.require_running()?;
        if self.state == RunState::NotStarted {
            bail!("The program has not been started; use `start` or `continue`.");
        }
        if over {
            if let Some(o) = self.next_by_ranges().await? {
                return Ok(o);
            }
        }
        let start_line = self.top_line();
        let start_phys = self.phys.len();
        let start_logical = self.vframes.len();
        let mut steps = 0u64;
        loop {
            steps += 1;
            if steps > 1_000_000 {
                return Ok(Outcome::Stopped(StopReason::Step));
            }
            let ev = self.resume(Motion::Step).await?;
            if let Some(o) = self.on_event(&ev) {
                return Ok(o);
            }
            let phys = self.phys.len();
            let logical = self.vframes.len();
            if phys > start_phys {
                // Entered a callee.
                let has_lines = self.top_line().is_some();
                if over {
                    if let Some(o) = self.step_out_phys().await? {
                        return Ok(o);
                    }
                    continue;
                }
                if !has_lines {
                    // A callee without line info: a trampoline, or a
                    // runtime that calls on into code with line info
                    // (e.g. a language runtime invoking a compiled
                    // guest function). Stop in the first function with
                    // line info it reaches, else come back.
                    if let Some(o) = self.step_through_lineless().await? {
                        return Ok(o);
                    }
                    continue;
                }
                // Stepping into a function with debug info: stop after
                // its prologue.
                if let Some(o) = self.skip_prologue().await? {
                    return Ok(o);
                }
                return Ok(Outcome::Stopped(StopReason::Step));
            }
            if over && logical > start_logical && phys == start_phys {
                // Inside an inlined call made from the starting frame.
                continue;
            }
            let Some(line) = self.top_line() else {
                continue;
            };
            if phys < start_phys || logical != start_logical || Some(line) != start_line {
                if self.at_stmt_start() || phys < start_phys {
                    return Ok(Outcome::Stopped(StopReason::Step));
                }
            }
        }
    }

    /// `next` via "range stepping": rather than single-stepping, place
    /// temporary breakpoints at every statement in the current function
    /// that is on a different line, plus the return address, and
    /// continue. Returns `None` (without resuming) if this is not
    /// possible, e.g. without line information.
    async fn next_by_ranges(&mut self) -> Result<Option<Outcome>> {
        let Some(start_line) = self.top_line() else {
            return Ok(None);
        };
        let Some(addr) = self.dwarf_addr(0) else {
            return Ok(None);
        };
        let module = self.phys[0].module;
        let mut targets: Vec<u32> = {
            let info = &self.modules[module].info;
            let dbg = info.debug.as_ref().unwrap();
            let Some(f) = dbg.function_at(addr) else {
                return Ok(None);
            };
            let mut t = vec![];
            for r in &f.ranges {
                for row in dbg.rows_in(r) {
                    if row.is_stmt && row.line != 0 && (row.file, row.line) != start_line {
                        if let Some(pc) = info.dwarf_to_pc(row.addr) {
                            t.push(pc);
                        }
                    }
                }
            }
            t
        };
        targets.sort();
        targets.dedup();
        let ret = self.return_point(0);
        let start_phys = self.phys.len();
        let start_logical = self.vframes.len();

        let mut installed = vec![];
        for &pc in &targets {
            self.install(module, pc)?;
            installed.push((module, pc));
        }
        if let Some((m, pc, _)) = ret {
            self.install(m, pc)?;
            installed.push((m, pc));
        }

        let result = loop {
            let ev = match self.resume(Motion::Continue).await {
                Ok(ev) => ev,
                Err(e) => break Err(e),
            };
            if let Some(o) = self.on_event(&ev) {
                break Ok(Some(o));
            }
            let here = self.phys.first().map(|p| (p.module, p.pc));
            let ours = here.is_some_and(|h| installed.contains(&h));
            if !ours {
                // A user breakpoint or other stop.
                let bp = self.user_bp_here();
                break Ok(Some(Outcome::Stopped(StopReason::Breakpoint(bp))));
            }
            let phys = self.phys.len();
            if phys < start_phys {
                // Returned to the caller.
                break Ok(Some(Outcome::Stopped(StopReason::Step)));
            }
            if phys > start_phys {
                // A recursive invocation of this function.
                continue;
            }
            let logical = self.vframes.len();
            if logical > start_logical {
                // Inside code inlined into this frame.
                continue;
            }
            if logical == start_logical && self.top_line() == Some(start_line) {
                continue;
            }
            break Ok(Some(Outcome::Stopped(StopReason::Step)));
        };
        if self.state != RunState::Exited {
            for (m, pc) in installed {
                self.uninstall(m, pc);
            }
        }
        result
    }

    /// The top frame is a callee without line info, entered by a source
    /// step: run until it returns, or until any function with line info
    /// (in any module) is entered below it, stopping there after its
    /// prologue. Returns `None` once back in the caller.
    async fn step_through_lineless(&mut self) -> Result<Option<Outcome>> {
        let Some(ret) = self.return_point(0) else {
            return Ok(Some(self.cont().await?));
        };
        let entry_depth = self.phys.len();
        let mut targets = vec![];
        for (m, module) in self.modules.iter().enumerate() {
            let Some(dbg) = &module.info.debug else {
                continue;
            };
            for f in &dbg.functions {
                if let Some(pc) = module.info.dwarf_to_pc(dbg.prologue_end(f)) {
                    targets.push((m, pc));
                }
            }
        }
        targets.sort();
        targets.dedup();
        let mut installed = vec![];
        for &(m, pc) in targets.iter().chain(std::iter::once(&(ret.0, ret.1))) {
            self.install(m, pc)?;
            installed.push((m, pc));
        }
        let result = loop {
            let ev = match self.resume(Motion::Continue).await {
                Ok(ev) => ev,
                Err(e) => break Err(e),
            };
            if let Some(o) = self.on_event(&ev) {
                break Ok(Some(o));
            }
            let Some(here) = self.phys.first().map(|p| (p.module, p.pc)) else {
                break Ok(Some(Outcome::Stopped(StopReason::Interrupted)));
            };
            let depth = self.phys.len();
            if here == (ret.0, ret.1) && depth <= ret.2 {
                break Ok(None);
            }
            if let Some(bp) = self.user_bp_here() {
                break Ok(Some(Outcome::Stopped(StopReason::Breakpoint(Some(bp)))));
            }
            if targets.binary_search(&here).is_ok() && depth >= entry_depth {
                break Ok(Some(Outcome::Stopped(StopReason::Step)));
            }
            if !matches!(ev, api::Event::Breakpoint) {
                break Ok(Some(Outcome::Stopped(StopReason::Interrupted)));
            }
        };
        if self.state != RunState::Exited {
            for (m, pc) in installed {
                self.uninstall(m, pc);
            }
        }
        result
    }

    /// If the top frame is in its function's prologue, run to the end
    /// of the prologue.
    async fn skip_prologue(&mut self) -> Result<Option<Outcome>> {
        let Some(addr) = self.dwarf_addr(0) else {
            return Ok(None);
        };
        let p = &self.phys[0];
        let m = &self.modules[p.module];
        let dbg = m.info.debug.as_ref().unwrap();
        let Some(f) = dbg.function_at(addr) else {
            return Ok(None);
        };
        let pe = dbg.prologue_end(f);
        if addr >= pe {
            return Ok(None);
        }
        let Some(pc) = m.info.dwarf_to_pc(pe) else {
            return Ok(None);
        };
        let (module, depth) = (p.module, self.phys.len());
        self.run_to(module, pc, depth).await
    }

    /// Step over one Wasm instruction (stepping over calls).
    pub async fn nexti(&mut self) -> Result<Outcome> {
        self.require_running()?;
        if let Some(p) = self.phys.first() {
            let kind = self.modules[p.module].info.instruction_kind(p.pc);
            if kind == Some(InstrKind::Call) {
                if let Some(next) = self.modules[p.module].info.next_instruction(p.pc) {
                    let (module, depth) = (p.module, self.phys.len());
                    if let Some(o) = self.run_to(module, next, depth).await? {
                        return Ok(o);
                    }
                    return Ok(Outcome::Stopped(StopReason::Step));
                }
            }
        }
        self.stepi().await
    }

    /// Run until the selected frame returns.
    pub async fn finish(&mut self) -> Result<Outcome> {
        self.require_running()?;
        if self.vframes.is_empty() {
            bail!("No stack.");
        }
        let sel = &self.vframes[self.selected];
        if sel.inlined {
            // Step until the logical depth drops below this frame's,
            // stepping over any physical calls.
            let target_logical = self.vframes.len() - self.selected - 1;
            let start_phys = self.phys.len();
            loop {
                let ev = self.resume(Motion::Step).await?;
                if let Some(o) = self.on_event(&ev) {
                    return Ok(o);
                }
                if self.phys.len() > start_phys {
                    if let Some(o) = self.step_out_phys().await? {
                        return Ok(o);
                    }
                    continue;
                }
                if self.vframes.len() <= target_logical {
                    return Ok(Outcome::Stopped(StopReason::Finished));
                }
            }
        }
        let phys = sel.phys;
        let Some((m, pc, depth)) = self.return_point(phys) else {
            bail!("\"finish\" not meaningful in the outermost frame.");
        };
        match self.run_to(m, pc, depth).await? {
            Some(o) => Ok(o),
            None => Ok(Outcome::Stopped(StopReason::Finished)),
        }
    }

    // -----------------------------------------------------------------
    // Reverse execution.

    fn require_reversible(&self) -> Result<()> {
        self.require_running()?;
        if !self.reversible {
            bail!("{NOT_REVERSIBLE}");
        }
        Ok(())
    }

    /// Run backward to the previous breakpoint or watchpoint stop.
    pub async fn reverse_cont(&mut self) -> Result<Outcome> {
        self.require_reversible()?;
        let ev = self.resume(Motion::ReverseContinue).await?;
        if let Some(o) = self.on_event(&ev) {
            return Ok(o);
        }
        Ok(Outcome::Stopped(StopReason::Breakpoint(
            self.user_bp_here(),
        )))
    }

    /// Step backward one Wasm instruction.
    pub async fn reverse_stepi(&mut self) -> Result<Outcome> {
        self.require_reversible()?;
        let ev = self.resume(Motion::ReverseStep).await?;
        if let Some(o) = self.on_event(&ev) {
            return Ok(o);
        }
        Ok(Outcome::Stopped(StopReason::Step))
    }

    /// Step backward one Wasm instruction, stepping back over calls.
    pub async fn reverse_nexti(&mut self) -> Result<Outcome> {
        self.require_reversible()?;
        let depth = self.phys.len();
        if let Some(o) = self.reverse_step_over_calls(depth, true).await? {
            return Ok(o);
        }
        Ok(Outcome::Stopped(StopReason::Step))
    }

    /// Step backward one instruction; if that enters a callee (which
    /// returned there), and `over` is set or the callee has no line
    /// information, run back to its call. Returns `Some` for stops other
    /// than completing the step.
    async fn reverse_step_over_calls(
        &mut self,
        depth: usize,
        over: bool,
    ) -> Result<Option<Outcome>> {
        let ev = self.resume(Motion::ReverseStep).await?;
        if let Some(o) = self.on_event(&ev) {
            return Ok(Some(o));
        }
        if self.phys.len() > depth && (over || self.top_line().is_none()) {
            return self.reverse_out_phys(0).await;
        }
        Ok(None)
    }

    /// Run backward to the call that created physical frame `phys`,
    /// stopping at the call instruction in its caller. Stops early
    /// (returning `Some`) for user breakpoints or other events.
    async fn reverse_out_phys(&mut self, phys: usize) -> Result<Option<Outcome>> {
        let Some(caller) = self.phys.get(phys + 1) else {
            // No Wasm caller: back to the beginning (or a breakpoint).
            return Ok(Some(self.reverse_cont().await?));
        };
        let module = caller.module;
        let Some(call) = self.modules[module].info.prev_instruction(caller.pc) else {
            bail!("Cannot find the call instruction before {:#x}.", caller.pc);
        };
        let depth = self.phys.len() - phys - 1;
        self.install(module, call)?;
        let result = loop {
            let ev = match self.resume(Motion::ReverseContinue).await {
                Ok(ev) => ev,
                Err(e) => break Err(e),
            };
            if let Some(o) = self.on_event(&ev) {
                break Ok(Some(o));
            }
            let at_target = self
                .phys
                .first()
                .is_some_and(|p| p.module == module && p.pc == call);
            if at_target && self.phys.len() <= depth {
                break Ok(None);
            }
            if let Some(bp) = self.user_bp_here() {
                break Ok(Some(Outcome::Stopped(StopReason::Breakpoint(Some(bp)))));
            }
        };
        if self.state != RunState::Exited {
            self.uninstall(module, call);
        }
        result
    }

    /// Source-level reverse step: run backward to the beginning of the
    /// previous line executed (or, from the middle of a line, to the
    /// beginning of that line). With `over` (`reverse-next`), calls are
    /// stepped back over; otherwise (`reverse-step`), stepping back into a
    /// call made by that line, if it has line information, stops at the
    /// beginning of the last line executed in it.
    pub async fn reverse_step_line(&mut self, over: bool) -> Result<Outcome> {
        self.require_reversible()?;
        if self.state == RunState::NotStarted {
            bail!("The program has not been started.");
        }
        let mut phys = self.phys.len();
        let mut logical = self.vframes.len();
        let mut steps = 0u64;
        let mut target = match self.top_line() {
            Some(l) if !self.at_stmt_start() => l,
            _ => {
                // Phase 1: step back until in a different line.
                let start_line = self.top_line();
                let target = loop {
                    steps += 1;
                    if steps > 1_000_000 {
                        return Ok(Outcome::Stopped(StopReason::Step));
                    }
                    if let Some(o) = self.reverse_step_over_calls(phys, over).await? {
                        return Ok(o);
                    }
                    if self.phys.len() < phys {
                        // Stepped back out of the function's beginning, to
                        // its call.
                        return Ok(Outcome::Stopped(StopReason::Step));
                    }
                    if self.phys.len() > phys {
                        // Stepped back into a callee with line information:
                        // find the beginning of its last line.
                        phys = self.phys.len();
                        logical = self.vframes.len();
                        match self.top_line() {
                            Some(l) => break l,
                            None => continue,
                        }
                    }
                    if over && self.vframes.len() > logical {
                        // Inside code inlined into this frame.
                        continue;
                    }
                    let Some(line) = self.top_line() else {
                        continue;
                    };
                    if self.vframes.len() != logical || Some(line) != start_line {
                        logical = self.vframes.len();
                        break line;
                    }
                };
                if self.in_prologue() {
                    // Back in the function's prologue: as for stepping
                    // forward, it has no line of its own, so leave the
                    // function, back to its call.
                    if let Some(o) = self.reverse_out_phys(0).await? {
                        return Ok(o);
                    }
                    return Ok(Outcome::Stopped(StopReason::Step));
                }
                target
            }
        };
        // Phase 2: step back to just before the beginning of the target
        // line, then forward again to it.
        loop {
            steps += 1;
            if steps > 1_000_000 {
                return Ok(Outcome::Stopped(StopReason::Step));
            }
            let ev = self.resume(Motion::ReverseStep).await?;
            if matches!(ev, api::Event::ReplayBegin) {
                // The line is the first executed.
                self.on_event(&ev);
                self.state = RunState::Stopped;
                return self.forward_to_line(phys, logical, target).await;
            }
            if let Some(o) = self.on_event(&ev) {
                return Ok(o);
            }
            if self.phys.len() > phys {
                // Stepped back into a callee. If the target line made the
                // call, the line continues before it.
                let call_line = self
                    .vframes
                    .iter()
                    .find(|v| v.phys == 1)
                    .and_then(|v| v.loc.as_ref())
                    .map(|l| (l.file, l.line));
                if call_line != Some(target) {
                    return self.forward_to_line(phys, logical, target).await;
                }
                if !over && self.top_line().is_some() {
                    // Stepping into the call: continue in its last line.
                    phys = self.phys.len();
                    logical = self.vframes.len();
                    target = self.top_line().unwrap();
                    continue;
                }
                if let Some(o) = self.reverse_out_phys(0).await? {
                    return Ok(o);
                }
                // Now at the call instruction, in the target line.
                continue;
            }
            if self.phys.len() < phys {
                return self.forward_to_line(phys, logical, target).await;
            }
            if over && self.vframes.len() > logical {
                continue;
            }
            let line = self.top_line();
            if line.is_none() && self.vframes.len() == logical {
                continue;
            }
            if self.vframes.len() != logical || line != Some(target) {
                return self.forward_to_line(phys, logical, target).await;
            }
        }
    }

    /// Step forward (over calls) until in `line` at physical depth `phys`
    /// and logical depth `logical`.
    async fn forward_to_line(
        &mut self,
        phys: usize,
        logical: usize,
        line: (u32, u32),
    ) -> Result<Outcome> {
        for _ in 0..1_000_000 {
            let o = if self.phys.is_empty() || self.phys.len() > phys {
                self.stepi().await?
            } else {
                self.nexti().await?
            };
            if !matches!(o, Outcome::Stopped(StopReason::Step)) {
                return Ok(o);
            }
            if self.phys.len() == phys
                && self.vframes.len() == logical
                && self.top_line() == Some(line)
            {
                break;
            }
        }
        Ok(Outcome::Stopped(StopReason::Step))
    }

    /// Whether the top frame is in its function's prologue.
    fn in_prologue(&self) -> bool {
        let Some(addr) = self.dwarf_addr(0) else {
            return false;
        };
        let dbg = self.modules[self.phys[0].module]
            .info
            .debug
            .as_ref()
            .unwrap();
        dbg.function_at(addr)
            .is_some_and(|f| addr < dbg.prologue_end(f))
    }

    /// Run backward to the call of the selected frame.
    pub async fn reverse_finish(&mut self) -> Result<Outcome> {
        self.require_reversible()?;
        if self.vframes.is_empty() {
            bail!("No stack.");
        }
        let sel = &self.vframes[self.selected];
        if sel.inlined {
            let target_logical = self.vframes.len() - self.selected - 1;
            let start_phys = self.phys.len();
            loop {
                if let Some(o) = self.reverse_step_over_calls(start_phys, true).await? {
                    return Ok(o);
                }
                if self.vframes.len() <= target_logical {
                    return Ok(Outcome::Stopped(StopReason::Finished));
                }
            }
        }
        let phys = sel.phys;
        if phys + 1 >= self.phys.len() {
            bail!("\"reverse-finish\" not meaningful in the outermost frame.");
        }
        match self.reverse_out_phys(phys).await? {
            Some(o) => Ok(o),
            None => Ok(Outcome::Stopped(StopReason::Finished)),
        }
    }
}
