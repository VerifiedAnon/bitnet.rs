// dashboard.rs
// Minimal dashboard for real-time metrics, kernel timings, and training progress.

#[cfg(feature = "core-gui")]
pub fn run_dashboard() {
    eframe::run_native(
        "BitNet Core Dashboard",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::new(DashboardApp::default())),
    );
}

#[cfg(feature = "core-gui")]
#[derive(Default)]
struct DashboardApp;

#[cfg(feature = "core-gui")]
impl eframe::App for DashboardApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("BitNet Core Dashboard");
            ui.label("Metrics, kernel timings, and training progress will appear here.");
        });
    }
}

// TODO: Implement dashboard UI for metrics and progress visualization. 