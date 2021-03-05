//! Unwind information for Windows x64 ABI.

use crate::isa::{unwind::input, unwind::winx64::UnwindInfo};
use crate::result::CodegenResult;
use regalloc::{Reg, RegClass};

pub(crate) fn create_unwind_info(
    unwind: input::UnwindInfo<Reg>,
) -> CodegenResult<Option<UnwindInfo>> {
    Ok(Some(UnwindInfo::build::<Reg, RegisterMapper>(unwind)?))
}

struct RegisterMapper;

impl crate::isa::unwind::winx64::RegisterMapper<Reg> for RegisterMapper {
    fn map(reg: Reg) -> crate::isa::unwind::winx64::MappedRegister {
        use crate::isa::unwind::winx64::MappedRegister;
        match reg.get_class() {
            RegClass::I64 => MappedRegister::Int(reg.get_hw_encoding()),
            RegClass::V128 => MappedRegister::Xmm(reg.get_hw_encoding()),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::{ExternalName, InstBuilder, Signature, StackSlotData, StackSlotKind};
    use crate::isa::unwind::winx64::UnwindCode;
    use crate::isa::x86::registers::RU;
    use crate::isa::{lookup_variant, BackendVariant, CallConv};
    use crate::settings::{builder, Flags};
    use crate::Context;
    use std::str::FromStr;
    use target_lexicon::triple;

    #[test]
    fn test_wrong_calling_convention() {
        let isa = lookup_variant(triple!("x86_64"), BackendVariant::MachInst)
            .expect("expect x86 ISA")
            .finish(Flags::new(builder()));

        let mut context = Context::for_function(create_function(CallConv::SystemV, None));

        context.compile(&*isa).expect("expected compilation");

        assert_eq!(
            create_unwind_info(&context.func, &*isa).expect("can create unwind info"),
            None
        );
    }
}
