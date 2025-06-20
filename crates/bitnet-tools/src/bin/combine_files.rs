use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use bitnet_tools::combine::{collect_files, combine_files_to_path};
use bitnet_tools::constants::workspace_root;

fn print_help() {
    println!("Universal File Combiner CLI\n");
    println!("Usage:");
    println!("  combine_files [--dir <dir>] [--ext <.rs,.toml,...>] [--output <file>] [--no-headers] [--no-tree]");
    println!("If no arguments are given, launches the GUI.");
}

fn build_tree_overview(dir: &Path, files: &[PathBuf]) -> String {
    let mut overview = String::new();
    let root_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
    overview.push_str(&format!("{}/\n", root_name));
    for file in files {
        if let Ok(rel) = file.strip_prefix(dir) {
            overview.push_str(&format!("├── {}\n", rel.display()));
        }
    }
    overview
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 {
        // No args: robustly launch GUI (build if needed)
        #[cfg(target_os = "windows")]
        let gui_rel = "target/release/file_combiner_gui.exe";
        #[cfg(not(target_os = "windows"))]
        let gui_rel = "target/release/file_combiner_gui";
        
        // Use workspace root to find the GUI
        let workspace = workspace_root();
        let gui_path = workspace.join(gui_rel);
        
        if gui_path.exists() {
            println!("Launching GUI: {}", gui_path.display());
            match Command::new(&gui_path).current_dir(&workspace).spawn() {
                Ok(_) => return,
                Err(e) => {
                    eprintln!("Failed to launch GUI: {}", e);
                    return;
                }
            }
        } else {
            println!("GUI not built, building now...");
            let status = Command::new("cargo")
                .arg("build")
                .arg("--release")
                .arg("-p")
                .arg("file_combiner_gui")
                .current_dir(&workspace)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status();
            match status {
                Ok(s) if s.success() => {
                    println!("GUI built successfully.");
                    if gui_path.exists() {
                        println!("Launching GUI: {}", gui_path.display());
                        match Command::new(&gui_path).current_dir(&workspace).spawn() {
                            Ok(_) => return,
                            Err(e) => {
                                eprintln!("Failed to launch GUI after build: {}", e);
                                return;
                            }
                        }
                    } else {
                        eprintln!("GUI binary not found after build: {}", gui_path.display());
                        return;
                    }
                }
                _ => {
                    eprintln!("Failed to build GUI. Please check for errors.");
                    return;
                }
            }
        }
    }
    // CLI mode
    let mut dir = None;
    let mut filter_exts: Vec<String> = vec![];
    let mut output = None;
    let mut include_headers = true;
    let mut include_tree = true;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                i += 1;
                dir = args.get(i).cloned();
            }
            "--ext" => {
                i += 1;
                if let Some(exts) = args.get(i) {
                    filter_exts = exts.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
                }
            }
            "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            "--no-headers" => {
                include_headers = false;
            }
            "--no-tree" => {
                include_tree = false;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                print_help();
                return;
            }
        }
        i += 1;
    }
    let dir = match dir {
        Some(d) => PathBuf::from(d),
        None => {
            println!("--dir <dir> is required in CLI mode");
            print_help();
            return;
        }
    };
    let mut files = vec![];
    collect_files(&dir, &filter_exts, &mut files);
    if files.is_empty() {
        println!("No files found to combine.");
        return;
    }
    let output_path = output.map(PathBuf::from).unwrap_or_else(|| {
        let folder_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("combined");
        dir.join(format!("{}_combined.txt", folder_name))
    });
    let tree_overview = if include_tree {
        Some(build_tree_overview(&dir, &files))
    } else {
        None
    };
    match combine_files_to_path(&output_path, &files, include_headers, tree_overview.as_deref()) {
        Ok(()) => println!("✅ Successfully combined {} files into {}", files.len(), output_path.display()),
        Err(e) => println!("Failed to combine files: {}", e),
    }
} 