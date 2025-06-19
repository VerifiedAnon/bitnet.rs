#![windows_subsystem = "windows"]

use eframe::{egui, App};
use rfd::FileDialog;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use bitnet_tools::combine::{is_combined_file, file_matches_filter, combine_files_to_path};
use ignore::WalkBuilder;

// Common binary and non-text file extensions to ignore
const IGNORED_EXTENSIONS: &[&str] = &[
    // Binary formats
    ".bin", ".exe", ".dll", ".so", ".dylib", ".pdb",
    ".safetensors", ".onnx", ".pt", ".pth", ".h5", ".ckpt",
    // Image formats
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".ico", ".svg",
    // Audio/Video
    ".mp3", ".wav", ".mp4", ".avi", ".mov",
    // Archives
    ".zip", ".tar", ".gz", ".7z", ".rar",
    // Other binary formats
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    // Cache and build artifacts
    ".pyc", ".pyo", ".rlib", ".rmeta", ".d",
    // Database files
    ".db", ".sqlite", ".sqlite3",
    // Rust specific
    ".rs.bk", ".lock",
];

// Common directories to ignore (in addition to .gitignore patterns)
const IGNORED_DIRECTORIES: &[&str] = &[
    // Version control
    ".git", ".github", ".gitignore", ".gitattributes", ".gitmodules",
    // IDE and editor
    ".vscode", ".idea", ".vs", ".settings",
    // Build and cache
    "target", "node_modules", "__pycache__", ".cache",
    // Environment and config
    ".env", ".config", ".local",
    // Logs and temporary files
    "logs", "temp", "tmp",
    // Dependencies
    "vendor", "packages", "deps",
    // Documentation build
    "docs/_build", "site", "public",
    // Test coverage
    "coverage", ".coverage", "htmlcov",
    // Project specific (from your .gitignore)
    "References", "models", "Original", "Converted",
];

#[derive(PartialEq)]
enum Tab {
    Explorer,
    Preview,
    Settings,
    About,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckState {
    Unchecked,
    Checked,
    Indeterminate,
}

#[derive(Debug, Clone)]
struct DirEntryNode {
    path: PathBuf,
    is_dir: bool,
    children: Vec<DirEntryNode>,
    checked: CheckState,
}

impl DirEntryNode {
    fn new(path: PathBuf, is_dir: bool) -> Self {
        Self {
            path,
            is_dir,
            children: Vec::new(),
            checked: CheckState::Unchecked,
        }
    }

    /// Recursively sets the checked state for this node and all its children.
    fn set_checked_recursive(&mut self, checked: bool) {
        self.checked = if checked { CheckState::Checked } else { CheckState::Unchecked };
        if self.is_dir {
            for child in &mut self.children {
                child.set_checked_recursive(checked);
            }
        }
    }

    /// Updates this node's checked state based on the state of its children.
    /// Returns true if the state changed.
    fn update_checked_state_from_children(&mut self) -> bool {
        if !self.is_dir {
            return false;
        }

        let original_state = self.checked;
        let mut num_checked = 0;
        let mut num_indeterminate = 0;

        for child in &self.children {
            match child.checked {
                CheckState::Checked => num_checked += 1,
                CheckState::Indeterminate => num_indeterminate += 1,
                CheckState::Unchecked => {}
            }
        }

        if num_indeterminate > 0 || (num_checked > 0 && num_checked < self.children.len()) {
            self.checked = CheckState::Indeterminate;
        } else if num_checked == self.children.len() && !self.children.is_empty() {
            self.checked = CheckState::Checked;
        } else {
            self.checked = CheckState::Unchecked;
        }

        original_state != self.checked
    }

    /// Traverses the tree and collects the paths of all files that are checked.
    fn collect_selected_files(&self, selection: &mut Vec<PathBuf>) {
        if self.is_dir {
            for child in &self.children {
                child.collect_selected_files(selection);
            }
        } else if self.checked == CheckState::Checked {
            selection.push(self.path.clone());
        }
    }
}

struct FileCombinerApp {
    selected_folder: Option<String>,
    root_tree: Option<DirEntryNode>,
    selection: Vec<PathBuf>,
    combine_result: Option<String>,
    output_filename: String,
    output_path: Option<PathBuf>,
    include_headers: bool,
    include_tree_overview: bool,
    file_type_filter: String,
    filter_exts: Vec<String>,
    preview_content: String,
    dark_mode: bool,
    tab: Tab,
    scan_error: Option<String>,
}

impl Default for FileCombinerApp {
    fn default() -> Self {
        Self {
            selected_folder: None,
            root_tree: None,
            selection: Vec::new(),
            combine_result: None,
            output_filename: "combined_output.txt".to_string(),
            output_path: None,
            include_headers: true,
            include_tree_overview: true,
            file_type_filter: ".rs,.toml,.md".to_string(), // Default to something useful
            filter_exts: vec![".rs".to_string(), ".toml".to_string(), ".md".to_string()],
            preview_content: String::new(),
            dark_mode: true,
            tab: Tab::Explorer,
            scan_error: None,
        }
    }
}

impl App for FileCombinerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.dark_mode {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🗂️ Universal File Combiner");
                ui.add_space(16.0);
                ui.selectable_value(&mut self.tab, Tab::Explorer, "Explorer");
                ui.selectable_value(&mut self.tab, Tab::Preview, "Preview");
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.selectable_value(&mut self.tab, Tab::About, "About");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.toggle_value(&mut self.dark_mode, "🌙");
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            match self.tab {
                Tab::Explorer => self.ui_explorer(ctx, ui),
                Tab::Preview => self.ui_preview(ui),
                Tab::Settings => self.ui_settings(ui),
                Tab::About => self.ui_about(ui),
            }
        });
    }
}

impl FileCombinerApp {
    /// Helper to generate default output file name based on selected folder
    fn default_output_filename(folder: &str) -> String {
        let path = Path::new(folder);
        let folder_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("combined");
        format!("{}_combined.txt", folder_name)
    }

    fn ui_explorer(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        // --- Top Controls ---
        ui.horizontal(|ui| {
            if ui.button("\u{1F4C1} Pick Folder").on_hover_text("Choose the root folder to scan").clicked() {
                if let Some(folder) = FileDialog::new().pick_folder() {
                    self.selected_folder = Some(folder.display().to_string());
                    self.root_tree = None;
                    self.selection.clear();
                    self.combine_result = None;
                    self.scan_error = None;
                    // Set output_filename to default for this folder
                    self.output_filename = Self::default_output_filename(&folder.display().to_string());
                }
            }
            if let Some(ref folder) = self.selected_folder {
                ui.label(format!("Selected: {}", folder));
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("File type Filter (e.g. .rs,.txt):");
            if ui.text_edit_singleline(&mut self.file_type_filter).changed() {
                self.filter_exts = self.file_type_filter.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
            }
            if ui.button("🔍 Scan").on_hover_text("Scan the selected folder for files").clicked() {
                if let Some(ref folder) = self.selected_folder {
                    let path = PathBuf::from(folder);
                    match build_tree_with_filter(&path, &self.filter_exts) {
                        Ok(tree) => {
                            self.root_tree = Some(tree);
                            self.scan_error = None;
                        }
                        Err(e) => {
                            self.scan_error = Some(format!("Error scanning folder: {}", e));
                            self.root_tree = None;
                        }
                    }
                }
            }
        });
        
        if let Some(ref error) = self.scan_error {
            ui.colored_label(egui::Color32::RED, error);
        }
        
        ui.separator();

        // --- File Tree View ---
        egui::TopBottomPanel::bottom("bottom_panel_explorer")
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    ui.label(format!("Selected files: {}", self.selection.len()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_enabled(!self.selection.is_empty(), egui::Button::new("➡️ Preview & Combine")).clicked() {
                            self.generate_preview();
                            self.tab = Tab::Preview;
                        }
                    });
                });
                if let Some(ref result) = self.combine_result {
                    ui.separator();
                    ui.label(result);
                }
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                if let Some(ref mut root) = self.root_tree {
                    let mut selection_changed = false;
                    let base_path = Path::new(self.selected_folder.as_deref().unwrap_or_default());
                    if Self::file_tree_ui(ui, root, base_path) {
                        selection_changed = true;
                    }
                    if selection_changed {
                        let mut selection = Vec::new();
                        root.collect_selected_files(&mut selection);
                        self.selection = selection;
                        self.combine_result = None; // Clear previous result on new selection
                    }
                } else {
                    ui.label("No folder scanned yet. Pick a folder and click 'Scan'.");
                }
            });
        });
    }

    /// The recursive function that builds the interactive file tree UI.
    /// Returns true if any check state was changed.
    fn file_tree_ui(ui: &mut egui::Ui, node: &mut DirEntryNode, base_path: &Path) -> bool {
        let mut changed = false;

        if node.is_dir {
            if node.children.is_empty() { return false; } // Don't show empty directories
            
            let id = ui.make_persistent_id(&node.path);
            let header = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
            
            let header_response = header.show_header(ui, |ui| {
                // Tri-state checkbox
                if checkbox_tristate(ui, &mut node.checked) {
                    node.set_checked_recursive(node.checked == CheckState::Checked);
                    changed = true;
                }
                // Folder name
                let name = node.path.strip_prefix(base_path).unwrap_or(&node.path).display();
                ui.label(format!("📁 {}", name));
            });

            header_response.body(|ui| {
                for child in &mut node.children {
                    ui.indent("indent", |ui| {
                       if Self::file_tree_ui(ui, child, base_path) {
                           changed = true;
                       }
                    });
                }
            });

            if changed {
                node.update_checked_state_from_children();
            }

        } else { // It's a file
            ui.horizontal(|ui| {
                if checkbox_tristate(ui, &mut node.checked) {
                    changed = true;
                }
                let name = node.path.file_name().unwrap_or_default().to_string_lossy();
                ui.label(format!("📄 {}", name));
            });
        }
        changed
    }

    // --- FIX IS HERE ---
    fn ui_preview(&mut self, ui: &mut egui::Ui) {
        if self.selection.is_empty() {
            ui.label("No files selected. Go back to 'Explorer' to select some files.");
            if ui.button("⬅️ Back to Explorer").clicked() {
                self.tab = Tab::Explorer;
            }
            return;
        }

        // Bottom panel for the action buttons
        egui::TopBottomPanel::bottom("preview_bottom_panel")
            .resizable(false)
            .show_inside(ui, |ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    if ui.button("⬅️ Back").clicked() {
                        self.tab = Tab::Explorer;
                    }
                    if ui.add_enabled(!self.selection.is_empty(), egui::Button::new("✅ Combine & Save")).clicked() {
                        let output_path = self.output_path.clone().unwrap_or_else(|| {
                            Path::new(self.selected_folder.as_ref().unwrap()).join(&self.output_filename)
                        });

                        let tree_overview = if self.include_tree_overview {
                            Some(self.generate_tree_overview())
                        } else {
                            None
                        };
                        
                        let result = combine_files_to_path(&output_path, &self.selection, self.include_headers, tree_overview.as_deref());
                        self.combine_result = Some(match result {
                            Ok(()) => format!("✅ Successfully combined {} files into {}", self.selection.len(), output_path.display()),
                            Err(e) => format!("Failed to combine files: {}", e),
                        });
                        self.tab = Tab::Explorer;
                    }
                });
            });

        // Central panel for the content, which will fill the remaining space
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Preview & Combine");
            ui.add_space(4.0);
            ui.label(format!("Combining {} files:", self.selection.len()));
            ui.separator();
            
            // This ScrollArea will now fill the available space.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.monospace(&self.preview_content);
                });
        });
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        egui::Grid::new("settings_grid").num_columns(2).spacing([40.0, 4.0]).show(ui, |ui| {
            ui.label("Output Filename:");
            ui.text_edit_singleline(&mut self.output_filename);
            ui.end_row();

            ui.label("Custom Output Path:");
            ui.horizontal(|ui| {
                if ui.button("Choose...").clicked() {
                     if let Some(path) = FileDialog::new().set_file_name(&self.output_filename).add_filter("Text File", &["txt"]).save_file() {
                         self.output_path = Some(path);
                     }
                }
                if let Some(path) = &self.output_path {
                    ui.label(path.display().to_string());
                } else {
                    ui.label("Default (in selected folder)");
                }
            });
            ui.end_row();
            
            ui.label("Theme:");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.dark_mode, true, "Dark");
                ui.radio_value(&mut self.dark_mode, false, "Light");
            });
            ui.end_row();
        });
        
        ui.separator();
        ui.heading("Output Options");
        ui.checkbox(&mut self.include_headers, "Include file path headers in output");
        ui.checkbox(&mut self.include_tree_overview, "Include file tree overview at the top of the output");
    }

    fn ui_about(&mut self, ui: &mut egui::Ui) {
        ui.heading("About Universal File Combiner");
        ui.add_space(8.0);
        ui.label("A modern tool to explore, select, and combine files into a single context, perfect for LLM analysis, code archiving, or documentation.");
        ui.add_space(8.0);
        ui.label("Features:");
        ui.label("• Unified file explorer with hierarchical selection");
        ui.label("• File type filtering (.rs, .py, .txt, etc.)");
        ui.label("• Custom output location and filename");
        ui.label("• Optional file headers and a project tree overview");
        ui.label("• Preview before combining");
        ui.add_space(8.0);
        ui.hyperlink_to("Made with Rust and egui", "https://github.com/emilk/egui");
        ui.label("© 2024");
    }

    fn generate_preview(&mut self) {
        let mut preview = String::new();
        let max_preview_size = 32 * 1024; // 32 KB
        let mut total = 0;

        if self.include_tree_overview {
            let tree = self.generate_tree_overview();
            preview.push_str(&tree);
            preview.push_str("\n\n");
            total += tree.len() + 2;
        }
        
        for file_path in &self.selection {
            if self.include_headers {
                let header = format!("\n--- File: {} ---\n", file_path.display());
                if total + header.len() > max_preview_size { break; }
                preview.push_str(&header);
                total += header.len();
            }

            match fs::read_to_string(file_path) {
                Ok(contents) => {
                    let to_add = if total + contents.len() > max_preview_size {
                        &contents[..max_preview_size.saturating_sub(total)]
                    } else { &contents };
                    preview.push_str(to_add);
                    total += to_add.len();

                    if total >= max_preview_size {
                        preview.push_str("\n\n... (preview truncated) ...");
                        break;
                    }
                }
                Err(e) => {
                    let error_msg = format!("[Error reading file: {}]", e);
                     if total + error_msg.len() > max_preview_size { break; }
                    preview.push_str(&error_msg);
                    total += error_msg.len();
                }
            }
        }
        self.preview_content = preview;
    }

    fn generate_tree_overview(&self) -> String {
        let mut overview = String::new();
        if let Some(root) = &self.root_tree {
            let root_name = root.path.file_name().unwrap_or_default().to_string_lossy();
            overview.push_str(&format!("{}/\n", root_name));
            Self::build_tree_string_recursive(&root, &mut overview, "", true);
        }
        overview
    }

    fn build_tree_string_recursive(node: &DirEntryNode, overview: &mut String, prefix: &str, is_last: bool) {
        // Only include nodes that are checked or indeterminate (i.e., contain a selected file)
        if node.checked == CheckState::Unchecked { return; }

        let (connector, child_prefix) = if is_last {
            ("└── ", "    ")
        } else {
            ("├── ", "│   ")
        };
        
        let name = node.path.file_name().unwrap_or_default().to_string_lossy();
        overview.push_str(&format!("{}{}{}\n", prefix, connector, name));

        if node.is_dir {
            let new_prefix = format!("{}{}", prefix, child_prefix);
            let last_child_idx = node.children.iter().rposition(|c| c.checked != CheckState::Unchecked).unwrap_or(0);
            for (i, child) in node.children.iter().enumerate() {
                Self::build_tree_string_recursive(child, overview, &new_prefix, i == last_child_idx);
            }
        }
    }
}

/// A helper function for a three-state checkbox. Returns true if clicked.
fn checkbox_tristate(ui: &mut egui::Ui, state: &mut CheckState) -> bool {
    let label = match state {
        CheckState::Unchecked => "☐",
        CheckState::Checked => "☑",
        CheckState::Indeterminate => "◼",
    };
    if ui.button(label).clicked() {
        *state = if *state == CheckState::Checked { CheckState::Unchecked } else { CheckState::Checked };
        true
    } else {
        false
    }
}

/// Returns true if the directory should be ignored
fn should_ignore_directory(path: &Path) -> bool {
    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
        IGNORED_DIRECTORIES.iter().any(|ignored| {
            dir_name.eq_ignore_ascii_case(ignored) ||
            // Handle nested cases like "docs/_build"
            ignored.split('/').all(|part| dir_name.eq_ignore_ascii_case(part))
        })
    } else {
        false
    }
}

/// Returns true if the file should be ignored based on extension
fn should_ignore_file(path: &Path) -> bool {
    // Check file extensions
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let ext = format!(".{}", ext.to_lowercase());
        if IGNORED_EXTENSIONS.contains(&ext.as_str()) {
            return true;
        }
    }

    // Check if the file itself is in the ignored list (like .gitignore)
    if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
        if IGNORED_DIRECTORIES.contains(&file_name) {
            return true;
        }
    }

    is_combined_file(path) // Also ignore *_combined.txt files
}

fn build_tree_with_filter(path: &Path, filter_exts: &[String]) -> std::io::Result<DirEntryNode> {
    let mut root = DirEntryNode::new(path.to_path_buf(), true);
    
    // Use WalkBuilder to respect .gitignore
    let walker = WalkBuilder::new(path)
        .hidden(true)        // Skip hidden files
        .git_ignore(true)    // Respect .gitignore
        .build();

    let mut dirs: std::collections::HashMap<PathBuf, Vec<DirEntryNode>> = std::collections::HashMap::new();
    
    for entry in walker.filter_map(Result::ok) {
        let path = entry.path().to_owned();
        if path == root.path {
            continue;
        }

        // Skip ignored directories
        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            if should_ignore_directory(&path) {
                continue;
            }
            let node = DirEntryNode::new(path.clone(), true);
            if let Some(parent) = path.parent() {
                dirs.entry(parent.to_owned())
                    .or_default()
                    .push(node);
            }
        } else if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            // Skip files we want to ignore
            if should_ignore_file(&path) {
                continue;
            }

            // Apply user's extension filter
            if !filter_exts.is_empty() && !file_matches_filter(&path, filter_exts) {
                continue;
            }

            let node = DirEntryNode::new(path.clone(), false);
            if let Some(parent) = path.parent() {
                dirs.entry(parent.to_owned())
                    .or_default()
                    .push(node);
            }
        }
    }

    // Build the tree from bottom up
    fn build_tree_recursive(
        path: &Path,
        dirs: &mut std::collections::HashMap<PathBuf, Vec<DirEntryNode>>,
    ) -> Vec<DirEntryNode> {
        if let Some(children) = dirs.remove(path) {
            let mut result = Vec::new();
            for mut child in children {
                if child.is_dir {
                    child.children = build_tree_recursive(&child.path, dirs);
                    if !child.children.is_empty() {
                        result.push(child);
                    }
                } else {
                    result.push(child);
                }
            }
            result.sort_by(|a, b| {
                // Directories first, then files
                if a.is_dir == b.is_dir {
                    a.path.file_name().cmp(&b.path.file_name())
                } else {
                    b.is_dir.cmp(&a.is_dir)
                }
            });
            result
        } else {
            Vec::new()
        }
    }

    root.children = build_tree_recursive(path, &mut dirs);
    Ok(root)
}

fn main() {
    eprintln!("Starting Universal File Combiner...");
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_title("Universal File Combiner"),
        ..Default::default()
    };

    eprintln!("Initializing GUI...");
    
    match eframe::run_native(
        "Universal File Combiner",
        options,
        Box::new(|_cc| {
            eprintln!("Creating application instance...");
            Box::new(FileCombinerApp::default())
        }),
    ) {
        Ok(_) => eprintln!("GUI closed successfully"),
        Err(e) => eprintln!("Error running GUI: {}", e),
    }
}