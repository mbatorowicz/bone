//! Punkt wejścia: bez argumentów okno, z argumentami bieg wsadowy.
//!
//! Konsola jest odłączana tylko w wydaniu bez argumentów — dokładniej: atrybut
//! `windows_subsystem` odłącza ją zawsze w wydaniu, więc tryb wsadowy na Windowsie
//! wypisuje do konsoli, z której został uruchomiony, o ile taka jest. Alternatywą
//! byłoby okno konsoli migające przy każdym uruchomieniu panelu.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::ExitCode;

use bone_core::cli::{self, Command};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = cli::parse(&args).and_then(|command| match command {
        Command::Gui => bone_ui::run().map_err(|e| e.to_string()),
        other => cli::execute(other),
    });
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("bone: {message}");
            ExitCode::FAILURE
        }
    }
}
