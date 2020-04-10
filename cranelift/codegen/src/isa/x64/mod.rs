//! X86_64-bit Instruction Set Architecture.

#![allow(dead_code)]

use crate::ir::Function;
use crate::isa::Builder as IsaBuilder;
use crate::isa::{CallConv, TargetIsa};
use crate::machinst::vcode::show_vcode;
use crate::machinst::{compile, MachBackend, MachCompileResult, TargetIsaAdapter, VCode};
use crate::result::CodegenResult;

use super::super::settings as shared_settings;

use alloc::boxed::Box;
use std::str::FromStr;

use regalloc::RealRegUniverse;
use target_lexicon::Triple;

// New backend:
mod abi;
mod inst;
mod lower;
mod settings;

use inst::create_reg_universe;

/// An X64 backend.
pub struct X64Backend {
    shared_flags: shared_settings::Flags,
    isa_flags: settings::Flags,
}

impl X64Backend {
    /// Create a new X64 backend with the given (shared) flags.
    pub fn new_with_flags(
        isa_flags: settings::Flags,
        shared_flags: shared_settings::Flags,
    ) -> Self {
        Self {
            isa_flags,
            shared_flags,
        }
    }

    fn compile_vcode(&self, mut func: Function) -> VCode<inst::Inst> {
        // This performs lowering to VCode, register-allocates the code, computes
        // block layout and finalizes branches. The result is ready for binary emission.
        let abi = Box::new(abi::X64ABIBody::new(&func));
        let flags = self.flags();
        compile::compile::<Self>(&mut func, self, abi, flags)
    }
}

impl MachBackend for X64Backend {
    fn compile_function(
        &self,
        func: Function,
        want_disasm: bool,
    ) -> CodegenResult<MachCompileResult> {
        let call_conv = func.signature.call_conv;

        let vcode = self.compile_vcode(func);
        let sections = vcode.emit();
        let frame_size = vcode.frame_size();

        let disasm = if want_disasm {
            Some(show_vcode(
                &vcode,
                Some(&create_reg_universe(call_conv)),
                &None,
            ))
        } else {
            None
        };

        Ok(MachCompileResult {
            sections,
            frame_size,
            disasm,
        })
    }

    fn flags(&self) -> &shared_settings::Flags {
        &self.shared_flags
    }

    fn name(&self) -> &'static str {
        "x64"
    }

    fn triple(&self) -> Triple {
        FromStr::from_str("x86_64").unwrap()
    }

    fn reg_universe(&self, call_conv: CallConv) -> RealRegUniverse {
        create_reg_universe(call_conv)
    }
}

/// Create a new `isa::Builder`.
pub fn isa_builder(triple: Triple) -> IsaBuilder {
    IsaBuilder {
        triple,
        setup: settings::builder(),
        constructor: isa_constructor,
    }
}

fn isa_constructor(
    _: Triple,
    shared_flags: shared_settings::Flags,
    arch_flag_builder: shared_settings::Builder,
) -> Box<dyn TargetIsa> {
    let isa_flags = settings::Flags::new(&shared_flags, arch_flag_builder);
    let backend = X64Backend::new_with_flags(isa_flags, shared_flags);
    Box::new(TargetIsaAdapter::new(backend))
}
