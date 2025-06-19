# Core GUI/Visualization Module (bitnet-core/src/gui/)

This module provides a developer-facing, core-level UI for visualizing and debugging model internals, kernel performance, and training progress in BitNet.

## Purpose
- Visualize model weights, activations, and attention maps
- Profile and debug CPU/GPU kernel performance
- Monitor real-time metrics and training progress
- Facilitate rapid debugging and performance tuning during development and research

## Dependencies
- [egui](https://github.com/emilk/egui): Immediate mode GUI for Rust
- [eframe](https://github.com/emilk/egui/tree/master/crates/eframe): Framework for building desktop apps with egui
- [plotters](https://github.com/38/plotters): High-quality static plots (histograms, heatmaps, etc.)

These are enabled via the `core-gui` feature flag in Cargo:

```sh
cargo run --features core-gui --example core_gui_dashboard
```

## Planned Features/Modules
- `mod.rs`: Entry point for the core GUI/visualization module
- `dashboard.rs`: Minimal dashboard for real-time metrics, kernel timings, and training progress
- `weights_viewer.rs`: Visualize model weights, distributions, and quantization effects
- `kernel_profiler.rs`: Interactive profiling and visualization of CPU/GPU kernel performance and correctness
- `attention_map.rs`: Visualization of attention matrices and activations

## Implementation Notes
- Intended for advanced users, developers, and researchers
- Uses `egui`, `eframe`, and `plotters` for visualization
- All core GUI features are optional and gated behind the `core-gui` feature flag
- The main application GUI (in bitnet-app) remains the user-facing interface; this core GUI is for internal development, debugging, and research 