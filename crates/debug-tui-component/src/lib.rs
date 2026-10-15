//! An interactive, text-mode (gdb-style) debugger running in Wasmtime's
//! `debug-main` world.
//!
//! The debugger listens on a TCP port; connect with `nc` or `telnet`
//! and type commands at the `(wasmtime-debug)` prompt. All
//! source-language understanding (DWARF, value formatting,
//! pretty-printers) comes from the `wasmtime-internal-debug-info` crate;
//! this crate provides the command-line front-end and drives the
//! debuggee through the `wasmtime:debugger` WIT API.

mod api;
mod commands;
mod conn;
mod dap;
mod session;
mod wasmval;

use crate::commands::Flow;
use crate::conn::{Conn, Input};
use crate::session::{Front, Session};
use anyhow::Result;
use clap::Parser;
use wstd::iter::AsyncIterator;
use wstd::net::TcpListener;

/// Command-line options.
#[derive(Parser)]
struct Options {
    /// The TCP address to listen on, in `<addr>:<port>` format.
    tcp_address: String,
    /// Serve the Debug Adapter Protocol rather than the interactive text
    /// interface.
    #[arg(long)]
    dap: bool,
}

struct Component;
api::export!(Component with_types_in api);

impl api::exports::bytecodealliance::wasmtime::debugger::Guest for Component {
    fn debug(d: &api::Debuggee, args: Vec<String>) {
        let options = Options::parse_from(args);
        wstd::runtime::block_on(async {
            if let Err(e) = run(d, &options).await {
                api::print_debugger_info(&format!("debugger exited with error: {e:#}"));
            }
        });
    }
}

async fn run(d: &api::Debuggee, options: &Options) -> Result<()> {
    let listener = TcpListener::bind(&options.tcp_address).await?;
    api::print_debugger_info(&format!("Debugger listening on {}", options.tcp_address));
    if options.dap {
        api::print_debugger_info("Serving the Debug Adapter Protocol (DAP); connect a DAP client.");
    } else {
        api::print_debugger_info(&format!(
            "Connect with: nc {}",
            options.tcp_address.replace(':', " ")
        ));
    }

    let Some(stream) = listener.incoming().next().await else {
        return Ok(());
    };
    let stream = stream?;
    if options.dap {
        let mut session = Session::new(d, Front::Dap(dap::Transport::new(stream)));
        dap::run(&mut session).await?;
        session.remove_all_breakpoints();
        return Ok(());
    }
    let mut session = Session::new(d, Front::Tui(Conn::new(stream)));
    session.banner();
    loop {
        session.prompt();
        if session.flush_output().await.is_err() {
            break;
        }
        let Front::Tui(conn) = &mut session.front else {
            unreachable!()
        };
        let line = match conn.read_input().await? {
            Input::Line(l) => l,
            Input::Interrupt => {
                session.println("");
                continue;
            }
            Input::Eof => break,
        };
        match session.execute(&line).await {
            Flow::Continue => {}
            Flow::Quit => {
                session.println("Detaching; the program will run to completion.");
                let _ = session.flush_output().await;
                break;
            }
        }
    }
    // Detach: remove all breakpoints so that the debuggee runs freely
    // once we return.
    session.remove_all_breakpoints();
    Ok(())
}
