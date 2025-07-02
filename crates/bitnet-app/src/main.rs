//! Main entry point for bitnet-app (CLI/GUI demo for BitNet inference).

mod cli;
mod generation;
mod sampler;
pub mod gui;

use rayon::ThreadPoolBuilder;
use num_cpus;

fn main() {
    ThreadPoolBuilder::new().num_threads(num_cpus::get()).build_global().unwrap();
    println!("[BitNet] [CPU] Rayon thread pool size: {}", rayon::current_num_threads());
    // TODO: Parse CLI args and launch CLI or GUI mode.
    println!("bitnet-app: CLI/GUI demo (stub)");
}
