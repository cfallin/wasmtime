//! Implementation of the standard ARM64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::StackSlot;
use crate::isa::arm64::inst::*;
use crate::isa::arm64::*;
use crate::machinst::*;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, Writable};

use log::debug;

// A location for an argument or return value.
#[derive(Clone, Debug)]
enum ABIArg {
    // In a real register.
    Reg(RealReg),
    // Arguments only: on stack, at given offset from SP at entry.
    Stack(i64, ir::Type),
    // (first and only) return value only: in memory pointed to by x8 on entry.
    RetMem(ir::Type),
}

/// ARM64 ABI information shared between body (callee) and caller.
struct ABISig {
    args: Vec<ABIArg>,
    rets: Vec<ABIArg>,
    stack_arg_space: i64,
}

/// Process a list of parameters or return values and allocate them to X-regs,
/// V-regs, and stack slots.
///
/// Returns the list of argument locations, and the stack-space used (rounded up
/// to a 16-byte-aligned boundary).
fn compute_arg_locs(params: &[ir::AbiParam]) -> (Vec<ABIArg>, i64) {
    // See AArch64 ABI (https://c9x.me/compile/bib/abi-arm64.pdf), sections 5.4.
    let mut next_xreg = 0;
    let mut next_vreg = 0;
    let mut next_stack: u64 = 0;
    let mut ret = vec![];
    for param in params {
        // Validate "purpose".
        match &param.purpose {
            &ir::ArgumentPurpose::VMContext | &ir::ArgumentPurpose::Normal => {}
            _ => panic!(
                "Unsupported argument purpose {:?} in signature: {:?}",
                param.purpose, params
            ),
        }

        if in_int_reg(param.value_type) {
            if next_xreg < 8 {
                ret.push(ABIArg::Reg(xreg(next_xreg).to_real_reg()));
                next_xreg += 1;
            } else {
                ret.push(ABIArg::Stack(next_stack as i64, param.value_type));
                next_stack += 8;
            }
        } else if in_vec_reg(param.value_type) {
            if next_vreg < 8 {
                ret.push(ABIArg::Reg(vreg(next_vreg).to_real_reg()));
                next_vreg += 1;
            } else {
                let size: u64 = match param.value_type {
                    F32 | F64 => 8,
                    _ => panic!("Unsupported vector-reg argument type"),
                };
                // Align.
                assert!(size.is_power_of_two());
                next_stack = (next_stack + size - 1) & !(size - 1);
                ret.push(ABIArg::Stack(next_stack as i64, param.value_type));
                next_stack += size;
            }
        }
    }

    next_stack = (next_stack + 15) & !15;

    (ret, next_stack as i64)
}

impl ABISig {
    fn from_func_sig(sig: &ir::Signature) -> ABISig {
        // Compute args and retvals from signature.
        // TODO: pass in arg-mode or ret-mode. (Does not matter
        // for the types of arguments/return values that we support.)
        let (args, stack_arg_space) = compute_arg_locs(&sig.params);
        let (rets, _) = compute_arg_locs(&sig.returns);

        // Verify that there are no arguments in return-memory area.
        assert!(args.iter().all(|a| match a {
            &ABIArg::RetMem(..) => false,
            _ => true,
        }));
        // Verify that there are no return values on the stack.
        assert!(rets.iter().all(|a| match a {
            &ABIArg::Stack(..) => false,
            _ => true,
        }));

        ABISig {
            args,
            rets,
            stack_arg_space,
        }
    }
}

/// ARM64 ABI object for a function body.
pub struct ARM64ABIBody {
    sig: ABISig,                       // signature: arg and retval regs
    stackslots: Vec<usize>,            // offsets to each stackslot
    stackslots_size: usize,            // total stack size of all stackslots
    clobbered: Set<Writable<RealReg>>, // clobbered registers, from regalloc.
    spillslots: Option<usize>,         // total number of spillslots, from regalloc.
    frame_size: Option<usize>,
}

fn in_int_reg(ty: ir::Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => true,
        types::B1 | types::B8 | types::B16 | types::B32 | types::B64 => true,
        _ => false,
    }
}

fn in_vec_reg(ty: ir::Type) -> bool {
    match ty {
        types::F32 | types::F64 => true,
        _ => false,
    }
}

impl ARM64ABIBody {
    /// Create a new body ABI instance.
    pub fn new(f: &ir::Function) -> ARM64ABIBody {
        debug!("ARM64 ABI: func signature {:?}", f.signature);

        let sig = ABISig::from_func_sig(&f.signature);

        // Compute stackslot locations and total stackslot size.
        let mut stack_offset: usize = 0;
        let mut stackslots = vec![];
        for (stackslot, data) in f.stack_slots.iter() {
            let off = stack_offset;
            stack_offset += data.size as usize;
            stack_offset = (stack_offset + 7) & !7usize;
            assert_eq!(stackslot.as_u32() as usize, stackslots.len());
            stackslots.push(off);
        }

        ARM64ABIBody {
            sig,
            stackslots,
            stackslots_size: stack_offset,
            clobbered: Set::empty(),
            spillslots: None,
            frame_size: None,
        }
    }
}

fn load_stack(
    fp_offset: i64,
    into_reg: Writable<Reg>,
    ty: Type,
    is_reload: Option<SpillSlot>,
) -> Inst {
    assert!(into_reg.to_reg().get_class() == RegClass::I64);
    let mem = MemArg::FPOffset(fp_offset);

    match ty {
        types::B1
        | types::B8
        | types::I8
        | types::B16
        | types::I16
        | types::B32
        | types::I32
        | types::B64
        | types::I64 => Inst::ULoad64 {
            rd: into_reg,
            mem,
            is_reload,
        },
        _ => unimplemented!(),
    }
}

fn store_stack(fp_offset: i64, from_reg: Reg, ty: Type, is_spill: Option<SpillSlot>) -> Inst {
    assert!(from_reg.get_class() == RegClass::I64);
    let mem = MemArg::FPOffset(fp_offset);

    match ty {
        types::B1
        | types::B8
        | types::I8
        | types::B16
        | types::I16
        | types::B32
        | types::I32
        | types::B64
        | types::I64 => Inst::Store64 {
            rd: from_reg,
            mem,
            is_spill,
        },
        _ => unimplemented!(),
    }
}

fn is_callee_save(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I64 => {
            // x19 - x28 inclusive are callee-saves.
            r.get_hw_encoding() >= 19 && r.get_hw_encoding() <= 28
        }
        RegClass::V128 => {
            // v8 - v15 inclusive are callee-saves.
            r.get_hw_encoding() >= 8 && r.get_hw_encoding() <= 15
        }
        _ => panic!("Unexpected RegClass"),
    }
}

fn is_caller_save(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I64 => {
            // x0 - x17 inclusive are caller-saves.
            r.get_hw_encoding() <= 17
        }
        RegClass::V128 => {
            // v0 - v7 inclusive and v16 - v31 inclusive are caller-saves.
            r.get_hw_encoding() <= 7 || (r.get_hw_encoding() >= 16 && r.get_hw_encoding() <= 31)
        }
        _ => panic!("Unexpected RegClass"),
    }
}

fn get_callee_saves(regs: Vec<Writable<RealReg>>) -> Vec<Writable<RealReg>> {
    regs.into_iter()
        .filter(|r| is_callee_save(r.to_reg()))
        .collect()
}

fn get_caller_saves_set() -> Set<Writable<Reg>> {
    let mut set = Set::empty();
    for i in 0..28 {
        let x = writable_xreg(i);
        if is_caller_save(x.to_reg().to_real_reg()) {
            set.insert(x);
        }
    }
    for i in 0..32 {
        let v = writable_vreg(i);
        if is_caller_save(v.to_reg().to_real_reg()) {
            set.insert(v);
        }
    }
    set
}

impl ABIBody<Inst> for ARM64ABIBody {
    fn liveins(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for arg in &self.sig.args {
            if let &ABIArg::Reg(r) = arg {
                set.insert(r);
            }
        }
        set
    }

    fn liveouts(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for ret in &self.sig.rets {
            if let &ABIArg::Reg(r) = ret {
                set.insert(r);
            }
        }
        set
    }

    fn num_args(&self) -> usize {
        self.sig.args.len()
    }

    fn num_retvals(&self) -> usize {
        self.sig.rets.len()
    }

    fn num_stackslots(&self) -> usize {
        self.stackslots.len()
    }

    fn gen_copy_arg_to_reg(&self, idx: usize, into_reg: Writable<Reg>) -> Inst {
        match &self.sig.args[idx] {
            &ABIArg::Reg(r) => Inst::gen_move(into_reg, r.to_reg()),
            &ABIArg::Stack(off, ty) => load_stack(off + 16, into_reg, ty, None),
            _ => unimplemented!(),
        }
    }

    fn gen_copy_reg_to_retval(&self, idx: usize, from_reg: Reg) -> Inst {
        match &self.sig.rets[idx] {
            &ABIArg::Reg(r) => Inst::gen_move(Writable::from_reg(r.to_reg()), from_reg),
            &ABIArg::Stack(off, ty) => store_stack(off + 16, from_reg, ty, None),
            _ => unimplemented!(),
        }
    }

    fn gen_ret(&self) -> Inst {
        Inst::Ret {}
    }

    fn gen_epilogue_placeholder(&self) -> Inst {
        Inst::EpiloguePlaceholder {}
    }

    fn set_num_spillslots(&mut self, slots: usize) {
        self.spillslots = Some(slots);
    }

    fn set_clobbered(&mut self, clobbered: Set<Writable<RealReg>>) {
        self.clobbered = clobbered;
    }

    fn load_stackslot(
        &self,
        slot: StackSlot,
        offset: usize,
        ty: Type,
        into_reg: Writable<Reg>,
    ) -> Inst {
        // Offset from beginning of stackslot area, which is at FP - stackslots_size.
        let stack_off = self.stackslots[slot.as_u32() as usize] as i64;
        let fp_off: i64 = -(self.stackslots_size as i64) + stack_off + (offset as i64);
        load_stack(fp_off, into_reg, ty, None)
    }

    fn store_stackslot(&self, slot: StackSlot, offset: usize, ty: Type, from_reg: Reg) -> Inst {
        // Offset from beginning of stackslot area, which is at FP - stackslots_size.
        let stack_off = self.stackslots[slot.as_u32() as usize] as i64;
        let fp_off: i64 = -(self.stackslots_size as i64) + stack_off + (offset as i64);
        store_stack(fp_off, from_reg, ty, None)
    }

    // Load from a spillslot.
    fn load_spillslot(&self, slot: SpillSlot, ty: Type, into_reg: Writable<Reg>) -> Inst {
        // Note that when spills/fills are generated, we don't yet know how many
        // spillslots there will be, so we allocate *downward* from the beginning
        // of the stackslot area. Hence: FP - stackslot_size - 8*spillslot -
        // sizeof(ty).
        let islot = slot.get() as i64;
        let ty_size = self.get_spillslot_size(into_reg.to_reg().get_class(), ty) * 8;
        let fp_off: i64 = -(self.stackslots_size as i64) - (8 * islot) - ty_size as i64;
        load_stack(fp_off, into_reg, ty, Some(slot))
    }

    // Store to a spillslot.
    fn store_spillslot(&self, slot: SpillSlot, ty: Type, from_reg: Reg) -> Inst {
        let islot = slot.get() as i64;
        let ty_size = self.get_spillslot_size(from_reg.get_class(), ty) * 8;
        let fp_off: i64 = -(self.stackslots_size as i64) - (8 * islot) - ty_size as i64;
        store_stack(fp_off, from_reg, ty, Some(slot))
    }

    fn gen_prologue(&mut self) -> Vec<Inst> {
        let mut insts = vec![];
        let total_stacksize = self.stackslots_size + 8 * self.spillslots.unwrap();
        let total_stacksize = (total_stacksize + 15) & !15; // 16-align the stack.

        // stp fp (x29), lr (x30), [sp, #-16]!
        insts.push(Inst::StoreP64 {
            rt: fp_reg(),
            rt2: link_reg(),
            mem: PairMemArg::PreIndexed(
                writable_stack_reg(),
                SImm7Scaled::maybe_from_i64(-16, types::I64).unwrap(),
            ),
        });
        // mov fp (x29), sp. This uses the ADDI rd, rs, 0 form of `MOV` because
        // the usual encoding (`ORR`) does not work with SP.
        insts.push(Inst::AluRRImm12 {
            alu_op: ALUOp::Add64,
            rd: writable_fp_reg(),
            rn: stack_reg(),
            imm12: Imm12 {
                bits: 0,
                shift12: false,
            },
        });

        if total_stacksize > 0 {
            // sub sp, sp, #total_stacksize
            if let Some(imm12) = Imm12::maybe_from_u64(total_stacksize as u64) {
                let sub_inst = Inst::AluRRImm12 {
                    alu_op: ALUOp::Sub64,
                    rd: writable_stack_reg(),
                    rn: stack_reg(),
                    imm12,
                };
                insts.push(sub_inst);
            } else {
                let const_data = u64_constant(total_stacksize as u64);
                let tmp = writable_spilltmp_reg();
                let const_inst = Inst::ULoad64 {
                    rd: tmp,
                    mem: MemArg::label(MemLabel::ConstantData(const_data)),
                    is_reload: None,
                };
                let sub_inst = Inst::AluRRRExtend {
                    alu_op: ALUOp::Sub64,
                    rd: writable_stack_reg(),
                    rn: stack_reg(),
                    rm: tmp.to_reg(),
                    extendop: ExtendOp::UXTX,
                };
                insts.push(const_inst);
                insts.push(sub_inst);
            }
        }

        // Save clobbered registers.
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg_pair in clobbered.chunks(2) {
            let (r1, r2) = if reg_pair.len() == 2 {
                // .to_reg().to_reg(): Writable<RealReg> --> RealReg --> Reg
                (reg_pair[0].to_reg().to_reg(), reg_pair[1].to_reg().to_reg())
            } else {
                (reg_pair[0].to_reg().to_reg(), zero_reg())
            };
            assert!(r1.get_class() == RegClass::I64);
            assert!(r2.get_class() == RegClass::I64);

            // stp r1, r2, [sp, #-16]!
            insts.push(Inst::StoreP64 {
                rt: r1,
                rt2: r2,
                mem: PairMemArg::PreIndexed(
                    writable_stack_reg(),
                    SImm7Scaled::maybe_from_i64(-16, types::I64).unwrap(),
                ),
            });
        }

        self.frame_size = Some(total_stacksize);
        insts
    }

    fn gen_epilogue(&self) -> Vec<Inst> {
        let mut insts = vec![];

        // Restore clobbered registers.
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg_pair in clobbered.chunks(2).rev() {
            let (r1, r2) = if reg_pair.len() == 2 {
                (
                    reg_pair[0].map(|r| r.to_reg()),
                    reg_pair[1].map(|r| r.to_reg()),
                )
            } else {
                (reg_pair[0].map(|r| r.to_reg()), writable_zero_reg())
            };

            assert!(r1.to_reg().get_class() == RegClass::I64);
            assert!(r2.to_reg().get_class() == RegClass::I64);

            // ldp r1, r2, [sp], #16
            insts.push(Inst::LoadP64 {
                rt: r1,
                rt2: r2,
                mem: PairMemArg::PostIndexed(
                    writable_stack_reg(),
                    SImm7Scaled::maybe_from_i64(16, types::I64).unwrap(),
                ),
            });
        }

        // The MOV (alias of ORR) interprets x31 as XZR, so use an ADD here.
        // MOV to SP is an alias of ADD.
        insts.push(Inst::AluRRImm12 {
            alu_op: ALUOp::Add64,
            rd: writable_stack_reg(),
            rn: fp_reg(),
            imm12: Imm12 {
                bits: 0,
                shift12: false,
            },
        });
        insts.push(Inst::LoadP64 {
            rt: writable_fp_reg(),
            rt2: writable_link_reg(),
            mem: PairMemArg::PostIndexed(
                writable_stack_reg(),
                SImm7Scaled::maybe_from_i64(16, types::I64).unwrap(),
            ),
        });
        insts.push(Inst::Ret {});
        debug!("Epilogue: {:?}", insts);
        insts
    }

    fn frame_size(&self) -> u32 {
        self.frame_size
            .expect("frame size not computed before prologue generation") as u32
    }

    fn get_spillslot_size(&self, rc: RegClass, ty: Type) -> u32 {
        // We allocate in terms of 8-byte slots.
        match (rc, ty) {
            (RegClass::I64, _) => 1,
            (RegClass::V128, F32) | (RegClass::V128, F64) => 1,
            (RegClass::V128, _) => 2,
            _ => panic!("Unexpected register class!"),
        }
    }

    fn gen_spill(&self, to_slot: SpillSlot, from_reg: RealReg, ty: Type) -> Inst {
        self.store_spillslot(to_slot, ty, from_reg.to_reg())
    }

    fn gen_reload(&self, to_reg: Writable<RealReg>, from_slot: SpillSlot, ty: Type) -> Inst {
        self.load_spillslot(from_slot, ty, to_reg.map(|r| r.to_reg()))
    }
}

enum CallDest {
    ExtName(ir::ExternalName),
    Reg(Reg),
}

/// ARM64 ABI object for a function call.
pub struct ARM64ABICall {
    sig: ABISig,
    uses: Set<Reg>,
    defs: Set<Writable<Reg>>,
    dest: CallDest,
}

fn abisig_to_uses_and_defs(sig: &ABISig) -> (Set<Reg>, Set<Writable<Reg>>) {
    // Compute uses: all arg regs.
    let mut uses = Set::empty();
    for arg in &sig.args {
        match arg {
            &ABIArg::Reg(reg) => uses.insert(reg.to_reg()),
            _ => {}
        }
    }

    // Compute defs: all retval regs, and all caller-save (clobbered) regs.
    let mut defs = get_caller_saves_set();
    for ret in &sig.rets {
        match ret {
            &ABIArg::Reg(reg) => defs.insert(Writable::from_reg(reg.to_reg())),
            _ => {}
        }
    }

    (uses, defs)
}

impl ARM64ABICall {
    /// Create a callsite ABI object for a call directly to the
    /// specified function.
    pub fn from_func(sig: &ir::Signature, extname: &ir::ExternalName) -> ARM64ABICall {
        let sig = ABISig::from_func_sig(sig);
        let (uses, defs) = abisig_to_uses_and_defs(&sig);
        ARM64ABICall {
            sig,
            uses,
            defs,
            dest: CallDest::ExtName(extname.clone()),
        }
    }

    /// Create a callsite ABI object for a call to a function pointer with the
    /// given signature.
    pub fn from_ptr(sig: &ir::Signature, ptr: Reg) -> ARM64ABICall {
        let sig = ABISig::from_func_sig(sig);
        let (uses, defs) = abisig_to_uses_and_defs(&sig);
        ARM64ABICall {
            sig,
            uses,
            defs,
            dest: CallDest::Reg(ptr),
        }
    }
}

fn adjust_stack(amt: u64, is_sub: bool) -> Vec<Inst> {
    if amt > 0 {
        let alu_op = if is_sub { ALUOp::Sub64 } else { ALUOp::Add64 };
        if let Some(imm12) = Imm12::maybe_from_u64(amt) {
            vec![Inst::AluRRImm12 {
                alu_op,
                rd: writable_stack_reg(),
                rn: stack_reg(),
                imm12,
            }]
        } else {
            let const_load = Inst::ULoad64 {
                rd: writable_spilltmp_reg(),
                mem: MemArg::Label(MemLabel::ConstantData(u64_constant(amt))),
                is_reload: None,
            };
            let adj = Inst::AluRRRExtend {
                alu_op,
                rd: writable_stack_reg(),
                rn: stack_reg(),
                rm: spilltmp_reg(),
                extendop: ExtendOp::UXTX,
            };
            vec![const_load, adj]
        }
    } else {
        vec![]
    }
}

impl ABICall<Inst> for ARM64ABICall {
    fn num_args(&self) -> usize {
        self.sig.args.len()
    }

    fn gen_stack_pre_adjust(&self) -> Vec<Inst> {
        adjust_stack(self.sig.stack_arg_space as u64, /* is_sub = */ true)
    }

    fn gen_stack_post_adjust(&self) -> Vec<Inst> {
        adjust_stack(self.sig.stack_arg_space as u64, /* is_sub = */ false)
    }

    fn gen_copy_reg_to_arg(&self, idx: usize, from_reg: Reg) -> Inst {
        match &self.sig.args[idx] {
            &ABIArg::Reg(reg) => Inst::gen_move(Writable::from_reg(reg.to_reg()), from_reg),
            &ABIArg::Stack(off, _) => Inst::Store64 {
                rd: from_reg,
                mem: MemArg::SPOffset(off),
                is_spill: None,
            },
            _ => unimplemented!(),
        }
    }

    fn gen_copy_retval_to_reg(&self, idx: usize, into_reg: Writable<Reg>) -> Inst {
        match &self.sig.rets[idx] {
            &ABIArg::Reg(reg) => Inst::gen_move(into_reg, reg.to_reg()),
            &ABIArg::RetMem(..) => panic!("Return-memory area not yet supported"),
            _ => unimplemented!(),
        }
    }

    fn gen_call(&self) -> Vec<Inst> {
        let (uses, defs) = (self.uses.clone(), self.defs.clone());
        match &self.dest {
            &CallDest::ExtName(ref name) => vec![Inst::Call {
                dest: name.clone(),
                uses,
                defs,
            }],
            &CallDest::Reg(reg) => vec![Inst::CallInd {
                rn: reg,
                uses,
                defs,
            }],
        }
    }
}
