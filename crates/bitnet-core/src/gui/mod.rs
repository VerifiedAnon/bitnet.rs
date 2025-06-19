// mod.rs
// Entry point for core GUI/visualization features in bitnet-core.

#[cfg(feature = "core-gui")]
pub mod dashboard;
#[cfg(feature = "core-gui")]
pub mod weights_viewer;
#[cfg(feature = "core-gui")]
pub mod kernel_profiler;
#[cfg(feature = "core-gui")]
pub mod attention_map;

#[cfg(feature = "core-gui")]
pub fn launch_core_gui() {
    dashboard::run_dashboard();
}

// TODO: Export submodules for dashboard, weights_viewer, kernel_profiler, attention_map, etc. 