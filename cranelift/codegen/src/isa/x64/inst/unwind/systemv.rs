//! Unwind information for System V ABI (x86-64).

use crate::isa::unwind::input;
use crate::isa::unwind::systemv::{RegisterMappingError, UnwindInfo};
use crate::result::CodegenResult;
use gimli::{write::CommonInformationEntry, Encoding, Format, Register, X86_64};
use regalloc::{Reg, RegClass};

/// Creates a new x86-64 common information entry (CIE).
pub fn create_cie() -> CommonInformationEntry {
    use gimli::write::CallFrameInstruction;

    let mut entry = CommonInformationEntry::new(
        Encoding {
            address_size: 8,
            format: Format::Dwarf32,
            version: 1,
        },
        1,  // Code alignment factor
        -8, // Data alignment factor
        X86_64::RA,
    );

    // Every frame will start with the call frame address (CFA) at RSP+8
    // It is +8 to account for the push of the return address by the call instruction
    entry.add_instruction(CallFrameInstruction::Cfa(X86_64::RSP, 8));

    // Every frame will start with the return address at RSP (CFA-8 = RSP+8-8 = RSP)
    entry.add_instruction(CallFrameInstruction::Offset(X86_64::RA, -8));

    entry
}

/// Map Cranelift registers to their corresponding Gimli registers.
pub fn map_reg(reg: Reg) -> Result<Register, RegisterMappingError> {
    // Mapping from https://github.com/bytecodealliance/cranelift/pull/902 by @iximeow
    const X86_GP_REG_MAP: [gimli::Register; 16] = [
        X86_64::RAX,
        X86_64::RCX,
        X86_64::RDX,
        X86_64::RBX,
        X86_64::RSP,
        X86_64::RBP,
        X86_64::RSI,
        X86_64::RDI,
        X86_64::R8,
        X86_64::R9,
        X86_64::R10,
        X86_64::R11,
        X86_64::R12,
        X86_64::R13,
        X86_64::R14,
        X86_64::R15,
    ];
    const X86_XMM_REG_MAP: [gimli::Register; 16] = [
        X86_64::XMM0,
        X86_64::XMM1,
        X86_64::XMM2,
        X86_64::XMM3,
        X86_64::XMM4,
        X86_64::XMM5,
        X86_64::XMM6,
        X86_64::XMM7,
        X86_64::XMM8,
        X86_64::XMM9,
        X86_64::XMM10,
        X86_64::XMM11,
        X86_64::XMM12,
        X86_64::XMM13,
        X86_64::XMM14,
        X86_64::XMM15,
    ];

    match reg.get_class() {
        RegClass::I64 => {
            // x86 GP registers have a weird mapping to DWARF registers, so we use a
            // lookup table.
            Ok(X86_GP_REG_MAP[reg.get_hw_encoding() as usize])
        }
        RegClass::V128 => Ok(X86_XMM_REG_MAP[reg.get_hw_encoding() as usize]),
        _ => Err(RegisterMappingError::UnsupportedRegisterBank("class?")),
    }
}

pub(crate) fn create_unwind_info(
    unwind: input::UnwindInfo<Reg>,
) -> CodegenResult<Option<UnwindInfo>> {
    struct RegisterMapper;
    impl crate::isa::unwind::systemv::RegisterMapper<Reg> for RegisterMapper {
        fn map(&self, reg: Reg) -> Result<u16, RegisterMappingError> {
            Ok(map_reg(reg)?.0)
        }
        fn sp(&self) -> u16 {
            X86_64::RSP.0
        }
    }
    let map = RegisterMapper;

    Ok(Some(UnwindInfo::build(unwind, &map)?))
}
