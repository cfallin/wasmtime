use wasmtime_core::{error::Result, format_err};

include!(concat!(env!("OUT_DIR"), "/gen.rs"));

/// Get the included debug-tui component artifact, or an error if it is
/// not included in this build.
pub fn debug_tui() -> Result<&'static [u8]> {
    crate::DEBUG_TUI_COMPONENT.ok_or_else(|| {
        format_err!(concat!(
            "debug-tui component not included in this build. Please build ",
            "in-tree with `--features debug-dap-tui` or use release binaries.",
        ))
    })
}
