use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Returns true if the file ends with _combined.txt (case-insensitive)
pub fn is_combined_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|name| name.to_lowercase().ends_with("_combined.txt"))
        .unwrap_or(false)
}

/// Returns true if the file matches any of the given extensions
pub fn file_matches_filter(path: &Path, filter_exts: &[String]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| filter_exts.contains(&format!(".{}", ext).to_lowercase()))
        .unwrap_or(false)
}

/// Recursively builds a list of files in a directory, filtering by extension and ignoring *_combined.txt
pub fn collect_files(dir: &Path, filter_exts: &[String], files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, filter_exts, files);
            } else if (filter_exts.is_empty() || file_matches_filter(&path, filter_exts)) && !is_combined_file(&path) {
                files.push(path);
            }
        }
    }
}

/// Combines files into a single output file, with optional headers and tree overview
pub fn combine_files_to_path(
    output_path: &Path,
    files: &[PathBuf],
    include_headers: bool,
    tree_overview: Option<&str>,
) -> std::io::Result<()> {
    let mut output = fs::File::create(output_path)?;
    if let Some(tree) = tree_overview {
        writeln!(output, "{}", tree)?;
        writeln!(output, "\n")?;
    }
    for (i, file) in files.iter().enumerate() {
        if include_headers {
            writeln!(output, "--- File: {} ---", file.display())?;
        }
        match fs::read_to_string(file) {
            Ok(contents) => {
                write!(output, "{}", contents)?;
            }
            Err(e) => {
                writeln!(output, "[Error reading file: {}]", e)?;
            }
        }
        if i < files.len() - 1 {
            writeln!(output, "\n")?;
        }
    }
    Ok(())
} 