#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod cosmology;
mod engine;
mod ics;
mod pm;
mod power;
mod units;

fn main() -> eframe::Result<()> {
    app::run()
}
