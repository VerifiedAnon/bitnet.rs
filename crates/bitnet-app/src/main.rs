//! Main entry point for bitnet-app (CLI/GUI demo for BitNet inference).

mod cli;
mod generation;
mod sampler;
pub mod gui;

use eframe::egui;
use crate::gui::app::ChatApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "BitNet Chat",
        options,
        Box::new(|_cc| Ok(Box::new(ChatApp::default()))),
    )
}
