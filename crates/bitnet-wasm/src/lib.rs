//! WASM entry point for BitNet. Exports the BitNetWasm API for JS interop.

mod api;
pub mod tests;

pub use api::*; 