//! Main egui App implementation and UI layout for bitnet-app.

use eframe::egui;
use bitnet_core::settings::InferenceSettings;
use bitnet_core::pipeline::{Pipeline, PipelineOptions, PipelineBackend};
use std::path::{Path, PathBuf};
use rfd::FileDialog;

pub struct ChatApp {
    input: String,
    chats: Vec<Vec<Message>>,
    current_chat: usize,
    tab: Tab,
    pub settings: InferenceSettings,
    // Model loading and pipeline state
    pub model_loaded: bool,
    pub model_path: Option<PathBuf>,
    pub backend: PipelineBackend,
    pub pipeline: Option<Pipeline>,
    pub loading_error: Option<String>,
    pub show_model_loader: bool,
    pub show_history_popup: bool,
}

#[derive(Clone)]
pub struct Message {
    pub text: String,
    pub is_user: bool,
}

#[derive(PartialEq)]
enum Tab {
    Chat,
    Settings,
}

impl Default for ChatApp {
    fn default() -> Self {
        Self {
            input: String::new(),
            chats: vec![vec![Message {
                text: "Welcome to Tab Agent! Ask me anything or paste a URL to scrape.".to_owned(),
                is_user: false,
            }]],
            current_chat: 0,
            tab: Tab::Chat,
            settings: InferenceSettings::default(),
            model_loaded: false,
            model_path: None,
            backend: PipelineBackend::Auto,
            pipeline: None,
            loading_error: None,
            show_model_loader: false,
            show_history_popup: false,
        }
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("model_controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Model dropdown (for now, just a Browse... button)
                let model_label = if let Some(path) = &self.model_path {
                    path.file_name().map(|n| n.to_string_lossy()).unwrap_or_else(|| "(Unknown)".into())
                } else {
                    "Browse...".into()
                };
                if ui.button(model_label).clicked() {
                    self.show_model_loader = true;
                }
                // Backend dropdown
                egui::ComboBox::from_id_source("backend_selector")
                    .selected_text(format!("{:?}", self.backend))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.backend, PipelineBackend::Auto, "Auto");
                        ui.selectable_value(&mut self.backend, PipelineBackend::Gpu, "GPU");
                        ui.selectable_value(&mut self.backend, PipelineBackend::Cpu, "CPU");
                    });
                // Load Model button
                let load_enabled = self.model_path.is_some();
                if ui.add_enabled(load_enabled, egui::Button::new("Load Model").fill(egui::Color32::GREEN)).clicked() {
                    if let Some(path) = &self.model_path {
                        self.loading_error = None;
                        // Try to load the pipeline (blocking for now, using tokio runtime)
                        let input_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or(PathBuf::from("."));
                        let opts = PipelineOptions {
                            model_id: None,
                            input_dir: Some(input_dir),
                            output_dir: None,
                            reporter: None,
                            backend: self.backend,
                            settings: Some(self.settings.clone()),
                            use_single_file: path.extension().map(|e| e == "safetensors").unwrap_or(false),
                            log_level: None,
                            verbose: false,
                        };
                        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
                        match rt.block_on(Pipeline::new(opts)) {
                            Ok(pipeline) => {
                                self.pipeline = Some(pipeline);
                                self.model_loaded = true;
                                self.show_model_loader = false;
                                self.loading_error = None;
                            }
                            Err(e) => {
                                self.loading_error = Some(format!("Failed to load model: {e}"));
                                self.model_loaded = false;
                            }
                        }
                    }
                }
                // Attach, Google Drive, etc. icons (placeholder)
                // ui.image(egui::include_image!("../icons/attach-svgrepo-com.svg"), ...);
            });
        });

        // Model loader popup
        if self.show_model_loader {
            egui::Window::new("Select Model File or Directory")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Select a safetensors file or model directory:");
                    if ui.button("Browse safetensors file...").clicked() {
                        if let Some(file) = FileDialog::new().add_filter("Safetensors", &["safetensors"]).pick_file() {
                            self.model_path = Some(file);
                        }
                    }
                    if ui.button("Browse model directory...").clicked() {
                        if let Some(dir) = FileDialog::new().pick_folder() {
                            // Try to find a safetensors file in the directory
                            let safetensors = std::fs::read_dir(&dir)
                                .ok()
                                .and_then(|mut it| it.find_map(|e| {
                                    let e = e.ok()?;
                                    let path = e.path();
                                    if path.extension().map(|ext| ext == "safetensors").unwrap_or(false) {
                                        Some(path)
                                    } else {
                                        None
                                    }
                                }));
                            self.model_path = safetensors.or(Some(dir));
                        }
                    }
                    if let Some(path) = &self.model_path {
                        ui.label(format!("Selected: {}", path.display()));
                    }
                    if ui.button("Close").clicked() {
                        self.show_model_loader = false;
                    }
                });
        }

        if let Some(err) = &self.loading_error {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.colored_label(egui::Color32::RED, err);
            });
        }

        if self.model_loaded {
            // Main app UI (chat, settings, history)
            egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.tab == Tab::Chat, "Chat").clicked() {
                        self.tab = Tab::Chat;
                    }
                    if ui.selectable_label(self.tab == Tab::Settings, "Settings").clicked() {
                        self.tab = Tab::Settings;
                    }
                    if ui.button("History").clicked() {
                        self.show_history_popup = true;
                    }
                });
            });
            match self.tab {
                Tab::Chat => self.show_chat(ctx),
                Tab::Settings => self.show_settings(ctx),
            }
            if self.show_history_popup {
                egui::Window::new("Chat History")
                    .collapsible(true)
                    .resizable(true)
                    .open(&mut self.show_history_popup)
                    .show(ctx, |ui| {
                        ui.label("Chat history (placeholder)");
                        // TODO: Render chat history list with actions (star, delete, preview, load, rename, etc.)
                    });
            }
        } else if !self.show_model_loader {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("No model loaded");
                ui.label("Please select and load a model to begin.");
            });
        }
    }
}

impl ChatApp {
    fn show_chat(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Minimal Chat Demo");
            if !self.model_loaded || self.pipeline.is_none() {
                ui.colored_label(egui::Color32::YELLOW, "Model not loaded. Please load a model to chat.");
                return;
            }

            // Input box and send button
            ui.horizontal(|ui| {
                let input = ui.add_sized([
                    ui.available_width() - 40.0,
                    36.0
                ], egui::TextEdit::singleline(&mut self.input).hint_text("Type a message..."));
                let send_btn = ui.add_sized([36.0, 36.0], egui::Button::new("\u{27A4}"));
                if (send_btn.clicked() || (input.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))) && !self.input.trim().is_empty() {
                    let user_msg = self.input.trim().to_owned();
                    self.input.clear();
                    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
                    let response = rt.block_on(self.pipeline.as_mut().unwrap().generate_text(&user_msg, &self.settings));
                    match response {
                        Ok(output) => {
                            ui.label(egui::RichText::new(format!("You: {}", user_msg)).color(egui::Color32::LIGHT_BLUE));
                            ui.label(egui::RichText::new(format!("Model: {}", output)).color(egui::Color32::LIGHT_GREEN));
                        }
                        Err(e) => {
                            ui.label(egui::RichText::new(format!("Model error: {}", e)).color(egui::Color32::RED));
                        }
                    }
                }
            });
        });
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Settings");

            egui::CollapsingHeader::new("Core Sampling")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add(egui::Slider::new(&mut self.settings.temperature, 0.0..=2.0).text("Temperature"));
                    ui.add(egui::Slider::new(&mut self.settings.top_p, 0.0..=1.0).text("Top-p"));
                    ui.add(egui::Slider::new(&mut self.settings.top_k, 0..=1000).text("Top-k"));
                    ui.checkbox(&mut self.settings.do_sample, "Do Sample");
                });

            egui::CollapsingHeader::new("Generation Constraints")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add(egui::Slider::new(&mut self.settings.max_new_tokens, 1..=4096).text("Max New Tokens"));
                    ui.add(egui::Slider::new(&mut self.settings.max_length, 1..=32768).text("Max Length"));
                    ui.add(egui::Slider::new(&mut self.settings.min_length, 0..=4096).text("Min Length"));
                    ui.add(egui::Slider::new(&mut self.settings.min_new_tokens, 0..=4096).text("Min New Tokens"));
                    ui.add(egui::Slider::new(&mut self.settings.batch_size, 1..=128).text("Batch Size"));
                    ui.add(egui::Slider::new(&mut self.settings.num_return_sequences, 1..=16).text("Num Return Sequences"));
                    ui.add(egui::Slider::new(&mut self.settings.num_beams, 1..=16).text("Num Beams"));
                    ui.add(egui::Slider::new(&mut self.settings.num_beam_groups, 1..=16).text("Num Beam Groups"));
                    ui.checkbox(&mut self.settings.early_stopping, "Early Stopping");
                    ui.add(egui::Slider::new(&mut self.settings.length_penalty, 0.0..=5.0).text("Length Penalty"));
                    ui.add(egui::Slider::new(&mut self.settings.diversity_penalty, 0.0..=5.0).text("Diversity Penalty"));
                    ui.add(egui::Slider::new(&mut self.settings.no_repeat_ngram_size, 0..=10).text("No Repeat Ngram Size"));
                    ui.add(egui::Slider::new(&mut self.settings.repetition_penalty, 0.5..=2.0).text("Repetition Penalty"));
                    ui.add(egui::Slider::new(&mut self.settings.penalty_alpha, 0.0..=10.0).text("Penalty Alpha"));
                    ui.add(egui::Slider::new(&mut self.settings.threads, 1..=32).text("Threads"));
                });

            egui::CollapsingHeader::new("Token Control")
                .default_open(false)
                .show(ui, |ui| {
                    macro_rules! option_token_field {
                        ($ui:expr, $label:expr, $field:expr) => {{
                            let mut enabled = $field.is_some();
                            $ui.horizontal(|ui| {
                                ui.checkbox(&mut enabled, $label);
                                if enabled {
                                    let mut val = $field.unwrap_or(0);
                                    if ui.add(egui::DragValue::new(&mut val)).changed() {
                                        *$field = Some(val);
                                    }
                                } else {
                                    *$field = None;
                                }
                            });
                        }};
                    }
                    option_token_field!(ui, "EOS Token ID", &mut self.settings.eos_token_id);
                    option_token_field!(ui, "BOS Token ID", &mut self.settings.bos_token_id);
                    option_token_field!(ui, "Pad Token ID", &mut self.settings.pad_token_id);
                    option_token_field!(ui, "Decoder Start Token ID", &mut self.settings.decoder_start_token_id);
                    option_token_field!(ui, "Forced BOS Token ID", &mut self.settings.forced_bos_token_id);
                    option_token_field!(ui, "Forced EOS Token ID", &mut self.settings.forced_eos_token_id);

                    // For lists (bad_words_ids, suppress_tokens), you can display as text for now:
                    ui.label("Bad Words IDs (comma-separated, advanced):");
                    if let Some(ref ids) = self.settings.bad_words_ids {
                        ui.label(format!("{:?}", ids));
                    } else {
                        ui.label("None");
                    }
                    ui.label("Suppress Tokens (comma-separated, advanced):");
                    if let Some(ref ids) = self.settings.suppress_tokens {
                        ui.label(format!("{:?}", ids));
                    } else {
                        ui.label("None");
                    }
                });

            egui::CollapsingHeader::new("Advanced / Debug")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut self.settings.use_cache, "Use Cache");
                    ui.checkbox(&mut self.settings.attention_mask, "Attention Mask");
                    ui.checkbox(&mut self.settings.output_attentions, "Output Attentions");
                    ui.checkbox(&mut self.settings.output_hidden_states, "Output Hidden States");
                    ui.checkbox(&mut self.settings.output_scores, "Output Scores");
                    ui.checkbox(&mut self.settings.remove_invalid_values, "Remove Invalid Values");
                    ui.checkbox(&mut self.settings.return_dict_in_generate, "Return Dict In Generate");

                    // Max time (Option<f32>)
                    let mut max_time_enabled = self.settings.max_time.is_some();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut max_time_enabled, "Max Time");
                        if max_time_enabled {
                            let mut val = self.settings.max_time.unwrap_or(0.0);
                            if ui.add(egui::DragValue::new(&mut val).speed(0.1).prefix("s ")).changed() {
                                self.settings.max_time = Some(val);
                            }
                        } else {
                            self.settings.max_time = None;
                        }
                    });

                    // Prefix (Option<String>)
                    let mut prefix_enabled = self.settings.prefix.is_some();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut prefix_enabled, "Prefix");
                        if prefix_enabled {
                            let mut val = self.settings.prefix.clone().unwrap_or_default();
                            if ui.text_edit_singleline(&mut val).changed() {
                                self.settings.prefix = Some(val);
                            }
                        } else {
                            self.settings.prefix = None;
                        }
                    });
                });

            ui.separator();
            ui.label("System Prompt:");
            ui.text_edit_multiline(&mut self.settings.system_prompt);
            ui.add(egui::DragValue::new(&mut self.settings.seed).speed(1).prefix("Seed: "));
        });
    }
} 