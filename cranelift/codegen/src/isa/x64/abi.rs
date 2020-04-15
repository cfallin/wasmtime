//! Implementation of the standard x64 ABI.

#![allow(dead_code)]

use crate::ir;
use crate::ir::types;
use crate::ir::types::*;
use crate::ir::{ArgumentExtension, StackSlot, Type};
use crate::isa;
use crate::isa::x64::inst::*;
use crate::machinst::*;
use crate::settings;

use alloc::vec::Vec;

use regalloc::{RealReg, Reg, RegClass, Set, SpillSlot, Writable};

#[derive(Clone, Debug)]
enum ABIArg {
    Reg(RealReg),
    Stack, // TODO
}

#[derive(Clone, Debug)]
enum ABIRet {
    Reg(RealReg),
    Mem, // TODO
}

pub struct X64ABIBody {
    args: Vec<ABIArg>,
    rets: Vec<ABIRet>,

    /// Offsets to each stack slot.
    stack_slots: Vec<usize>,

    /// Total stack size of all the stack slots.
    stack_slots_size: usize,

    /// Clobbered registers, as indicated by regalloc.
    clobbered: Set<Writable<RealReg>>,

    /// Total number of spill slots, as indicated by regalloc.
    num_spill_slots: Option<usize>,

    /// Calculated while creating the prologue, and used when creating the epilogue. Amount by
    /// which RSP is adjusted downwards to allocate the spill area.
    frame_size_bytes: Option<usize>,

    call_conv: isa::CallConv,
}

fn in_int_reg(ty: types::Type) -> bool {
    match ty {
        types::I8 | types::I16 | types::I32 | types::I64 => true,
        types::B1 | types::B8 | types::B16 | types::B32 | types::B64 => true,
        _ => false,
    }
}

fn get_intreg_for_arg_systemv(idx: usize) -> Option<Reg> {
    match idx {
        0 => Some(reg_RDI()),
        1 => Some(reg_RSI()),
        2 => Some(reg_RDX()),
        3 => Some(reg_RCX()),
        4 => Some(reg_R8()),
        5 => Some(reg_R9()),
        _ => None,
    }
}

fn get_intreg_for_retval_systemv(idx: usize) -> Option<Reg> {
    match idx {
        0 => Some(reg_RAX()),
        1 => Some(reg_RDX()),
        _ => None,
    }
}

fn is_callee_save_systemv(r: RealReg) -> bool {
    match r.get_class() {
        RegClass::I64 => match r.get_hw_encoding() as u8 {
            ENC_RBX | ENC_RBP | ENC_R12 | ENC_R13 | ENC_R14 | ENC_R15 => true,
            _ => false,
        },
        _ => unimplemented!(),
    }
}

fn get_callee_saves(regs: Vec<Writable<RealReg>>) -> Vec<Writable<RealReg>> {
    regs.into_iter()
        .filter(|r| is_callee_save_systemv(r.to_reg()))
        .collect()
}

impl X64ABIBody {
    /// Create a new body ABI instance.
    pub fn new(f: &ir::Function) -> Self {
        println!("X64 ABI: func signature {:?}", f.signature);

        // Compute args and retvals from signature.
        let mut args = vec![];
        let mut next_int_arg = 0;
        for param in &f.signature.params {
            match param.purpose {
                ir::ArgumentPurpose::Normal => {
                    if in_int_reg(param.value_type) {
                        if let Some(reg) = get_intreg_for_arg_systemv(next_int_arg) {
                            args.push(ABIArg::Reg(reg.to_real_reg()));
                        } else {
                            unimplemented!("passing arg on the stack");
                        }
                        next_int_arg += 1;
                    } else {
                        unimplemented!("non int normal register")
                    }
                }

                ir::ArgumentPurpose::VMContext => {
                    debug_assert!(f.signature.call_conv.extends_baldrdash());
                    // VmContext is r14 in Baldrdash.
                    args.push(ABIArg::Reg(reg_R14().to_real_reg()));
                }

                _ => unimplemented!("other parameter purposes"),
            }
        }

        let mut rets = vec![];
        let mut next_int_retval = 0;
        for ret in &f.signature.returns {
            match ret.purpose {
                ir::ArgumentPurpose::Normal => {
                    if in_int_reg(ret.value_type) {
                        if let Some(reg) = get_intreg_for_retval_systemv(next_int_retval) {
                            rets.push(ABIRet::Reg(reg.to_real_reg()));
                        } else {
                            unimplemented!("passing return on the stack");
                        }
                        next_int_retval += 1;
                    } else {
                        unimplemented!("returning non integer normal value");
                    }
                }

                _ => {
                    unimplemented!("non normal argument purpose");
                }
            }
        }

        // Compute stackslot locations and total stackslot size.
        let mut stack_offset: usize = 0;
        let mut stack_slots = vec![];
        for (stackslot, data) in f.stack_slots.iter() {
            let off = stack_offset;
            stack_offset += data.size as usize;
            stack_offset = (stack_offset + 7) & !7usize;
            debug_assert_eq!(stackslot.as_u32() as usize, stack_slots.len());
            stack_slots.push(off);
        }

        Self {
            args,
            rets,
            stack_slots,
            stack_slots_size: stack_offset,
            clobbered: Set::empty(),
            num_spill_slots: None,
            frame_size_bytes: None,
            call_conv: f.signature.call_conv.clone(),
        }
    }
}

impl ABIBody<Inst> for X64ABIBody {
    fn num_args(&self) -> usize {
        unimplemented!()
    }

    fn num_retvals(&self) -> usize {
        unimplemented!()
    }

    fn num_stackslots(&self) -> usize {
        unimplemented!()
    }

    // Clone of arm64 version
    fn liveins(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for arg in &self.args {
            if let &ABIArg::Reg(r) = arg {
                set.insert(r);
            }
        }
        println!("X64 ABI: liveins {:?}", set);
        set
    }

    // Clone of arm64 version
    fn liveouts(&self) -> Set<RealReg> {
        let mut set: Set<RealReg> = Set::empty();
        for ret in &self.rets {
            if let &ABIRet::Reg(r) = ret {
                set.insert(r);
            }
        }
        println!("X64 ABI: liveouts {:?}", set);
        set
    }

    fn gen_copy_arg_to_reg(&self, idx: usize, to_reg: Writable<Reg>) -> Inst {
        match &self.args[idx] {
            ABIArg::Reg(from_reg) => {
                if from_reg.get_class() == RegClass::I32 || from_reg.get_class() == RegClass::I64 {
                    // TODO do we need a sign extension if it's I32?
                    return i_Mov_R_R(/*is64=*/ true, from_reg.to_reg(), to_reg);
                }
                unimplemented!("moving from non-int arg to vreg");
            }
            ABIArg::Stack => unimplemented!("moving from stack arg to vreg"),
        }
    }

    fn gen_copy_reg_to_retval(
        &self,
        idx: usize,
        from_reg: Reg,
        ext: ArgumentExtension,
    ) -> Vec<Inst> {
        match ext {
            ArgumentExtension::None => {}
            _ => unimplemented!(
                "unimplemented argument extension {:?} is required for baldrdash",
                ext
            ),
        };

        let mut ret = Vec::new();
        match &self.rets[idx] {
            ABIRet::Reg(to_reg) => {
                if to_reg.get_class() == RegClass::I32 || to_reg.get_class() == RegClass::I64 {
                    ret.push(i_Mov_R_R(
                        /*is64=*/ true,
                        from_reg,
                        Writable::<Reg>::from_reg(to_reg.to_reg()),
                    ))
                } else {
                    unimplemented!("moving from vreg to non-int return value");
                }
            }

            ABIRet::Mem => {
                // This shouldn't happen.
                panic!("moving from vreg to memory return value");
            }
        }

        ret
    }

    fn gen_ret(&self) -> Inst {
        i_Ret()
    }

    fn gen_epilogue_placeholder(&self) -> Inst {
        i_epilogue_placeholder()
    }

    // Clone of arm64
    fn set_num_spillslots(&mut self, slots: usize) {
        self.num_spill_slots = Some(slots);
    }

    // Clone of arm64
    fn set_clobbered(&mut self, clobbered: Set<Writable<RealReg>>) {
        self.clobbered = clobbered;
    }

    fn stackslot_addr(&self, _slot: StackSlot, _offset: u32, _into_reg: Writable<Reg>) -> Inst {
        unimplemented!()
    }

    fn load_stackslot(
        &self,
        _slot: StackSlot,
        _offset: usize,
        _ty: Type,
        _into_reg: Writable<Reg>,
    ) -> Inst {
        unimplemented!()
    }

    fn store_stackslot(&self, _slot: StackSlot, _offset: usize, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn load_spillslot(&self, _slot: SpillSlot, _ty: Type, _into_reg: Writable<Reg>) -> Inst {
        unimplemented!()
    }

    fn store_spillslot(&self, _slot: SpillSlot, _ty: Type, _from_reg: Reg) -> Inst {
        unimplemented!()
    }

    fn gen_prologue(&mut self, flags: &settings::Flags) -> Vec<Inst> {
        let r_rsp = reg_RSP();

        let mut insts = vec![];

        // Baldrdash generates its own prologue sequence, so we don't have to.
        if !self.call_conv.extends_baldrdash() {
            let r_rbp = reg_RBP();
            let w_rbp = Writable::<Reg>::from_reg(r_rbp);

            // The "traditional" pre-preamble
            // RSP before the call will be 0 % 16.  So here, it is 8 % 16.
            insts.push(i_Push64(ip_RMI_R(r_rbp)));
            // RSP is now 0 % 16
            insts.push(i_Mov_R_R(true, r_rsp, w_rbp));
        }

        // Save callee saved registers that we trash. Keep track of how much space we've used, so
        // as to know what we have to do to get the base of the spill area 0 % 16.
        let mut callee_saved_used = 0;
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for reg in clobbered {
            let r_reg = reg.to_reg();
            match r_reg.get_class() {
                RegClass::I64 => {
                    insts.push(i_Push64(ip_RMI_R(r_reg.to_reg())));
                    callee_saved_used += 8;
                }
                _ => unimplemented!(),
            }
        }

        let mut total_stacksize = self.stack_slots_size + 8 * self.num_spill_slots.unwrap();
        if self.call_conv.extends_baldrdash() {
            // Baldrdash expects the stack to expect at least the number of words set in
            // baldrdash_prologue_words; count them here.
            debug_assert!(
                !flags.enable_probestack(),
                "baldrdash does not expect cranelift to emit stack probes"
            );
            total_stacksize += flags.baldrdash_prologue_words() as usize * 8;
        }

        debug_assert!(callee_saved_used % 16 == 0 || callee_saved_used % 16 == 8);
        let frame_size = total_stacksize + callee_saved_used % 16;

        // Now make sure the frame stack is aligned, so RSP == 0 % 16 in the function's body.
        let frame_size = (frame_size + 15) & !15;

        if frame_size > 0x7FFF_FFFF {
            panic!("gen_prologue(x86): total_stacksize >= 2G");
        }

        if !self.call_conv.extends_baldrdash() {
            // Explicitly allocate the frame.
            let w_rsp = Writable::<Reg>::from_reg(r_rsp);
            if frame_size > 0 {
                // FIXME JRS 2020Feb16: handle frame_size >= 2G?
                insts.push(i_Alu_RMI_R(
                    true,
                    RMI_R_Op::Sub,
                    ip_RMI_I(frame_size as u32),
                    w_rsp,
                ));
            }
        }

        // Stash this value.  We'll need it for the epilogue.
        debug_assert!(self.frame_size_bytes.is_none());
        self.frame_size_bytes = Some(frame_size);

        insts
    }

    fn gen_epilogue(&self, _flags: &settings::Flags) -> Vec<Inst> {
        let mut insts = vec![];

        // Undo what we did in the prologue.

        // Clear the spill area and the 16-alignment padding below it.
        if !self.call_conv.extends_baldrdash() {
            let frame_size = self.frame_size_bytes.unwrap();
            if frame_size > 0 {
                let r_rsp = reg_RSP();
                let w_rsp = Writable::<Reg>::from_reg(r_rsp);

                // FIXME JRS 2020Feb16: what if frame_size >= 2G?
                insts.push(i_Alu_RMI_R(
                    true,
                    RMI_R_Op::Add,
                    ip_RMI_I(frame_size as u32),
                    w_rsp,
                ));
            }
        }

        // Restore regs.
        let clobbered = get_callee_saves(self.clobbered.to_vec());
        for w_real_reg in clobbered.into_iter().rev() {
            match w_real_reg.to_reg().get_class() {
                RegClass::I64 => {
                    // TODO: make these conversion sequences less cumbersome.
                    insts.push(i_Pop64(Writable::<Reg>::from_reg(
                        w_real_reg.to_reg().to_reg(),
                    )))
                }
                _ => unimplemented!(),
            }
        }

        // Baldrdash generates its own preamble.
        if !self.call_conv.extends_baldrdash() {
            let r_rbp = reg_RBP();
            let w_rbp = Writable::<Reg>::from_reg(r_rbp);

            // Undo the "traditional" pre-preamble
            // RSP before the call will be 0 % 16.  So here, it is 8 % 16.
            insts.push(i_Pop64(w_rbp));
            insts.push(i_Ret());
        }

        insts
    }

    fn frame_size(&self) -> u32 {
        self.frame_size_bytes
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

    fn gen_spill(&self, _to_slot: SpillSlot, _from_reg: RealReg, _ty: Type) -> Inst {
        unimplemented!()
    }

    fn gen_reload(&self, _to_reg: Writable<RealReg>, _from_slot: SpillSlot, _ty: Type) -> Inst {
        unimplemented!()
    }
}
