# BitNet Tools (`bitnet-tools`)

A collection of essential utilities for BitNet projects, including a universal file combiner (CLI & GUI) and robust model download tools. Designed for codebase exploration, LLM context preparation, and model management.

---

## Table of Contents

- [Purpose](#purpose)
- [Features](#features)
- [Universal File Combiner](#universal-file-combiner)
- [Model Download Utilities](#model-download-utilities)
- [CLI Usage](#cli-usage)
- [GUI Usage](#gui-usage)
- [Building Standalone Executables](#building-standalone-executables)
- [Implementation Notes](#implementation-notes)
- [License](#license)

---

## Purpose

- Provide shared tools for BitNet development, codebase management, and LLM context preparation
- Enable easy file combining, codebase archiving, and model file management
- Support both CLI and GUI workflows

## Features

- **Universal file combiner**: Combine files from any directory into a single context file (CLI & GUI)
- **File type filtering**: Select files by extension (e.g., `.rs`, `.py`, `.txt`)
- **Streaming-friendly**: Always ignores files ending with `_combined.txt` to prevent recursion
- **Preview and tree overview**: Preview combined output and include a project tree overview
- **Model download utilities**: Download and validate all required BitNet model files from Hugging Face
- **Robust error handling**: Clear errors for I/O, serialization, and API issues
- **Extensible**: Designed for integration with other BitNet crates

## Universal File Combiner

- **GUI**: Hierarchical file explorer, selection, preview, and combine (see `gui_combiner/`)
- **CLI**: Combine files from a directory with flexible options
- **Always ignores**: Files ending with `_combined.txt` to avoid recursive combining
- **Output**: Customizable output location and filename, with optional headers and tree overview

## Model Download Utilities

- Download all required BitNet model files from Hugging Face with progress reporting
- Ensures all files are present and valid for downstream use
- See `src/hf_loader.rs` for details

## CLI Usage

```sh
# Build
cd crates/bitnet-tools
cargo build --release --bin combine_files

# Run (GUI by default)
cargo run --release --bin combine_files

# CLI mode (combine files from a directory)
cargo run --release --bin combine_files -- --dir <dir> [--ext <.rs,.toml,...>] [--output <file>] [--no-headers] [--no-tree]

# Example: Combine all .rs and .toml files in a crate
target/release/combine_files --dir ../../bitnet-core --ext .rs,.toml
```

- `--dir <dir>`: Directory to combine files from (**required in CLI mode**)
- `--ext <.rs,.toml,...>`: Comma-separated file extensions to include (optional)
- `--output <file>`: Output file path (optional, defaults to `<root>_combined.txt`)
- `--no-headers`: Exclude file headers in output
- `--no-tree`: Exclude tree overview at the top

## GUI Usage

```sh
# From the workspace root
cd crates/bitnet-tools/gui_combiner
cargo run --release
```

- Pick a folder, filter file types, select files, preview, and combine
- Output file defaults to `<root_folder_name>_combined.txt` in the selected folder
- Customize output name and location in Settings

## Building Standalone Executables

- Build a standalone `.exe` (Windows) or binary (Linux/macOS) for the GUI or CLI using `cargo build --release` in the respective crate
- Distribute the resulting binary for use on other systems

## Implementation Notes

- All tools are designed to be robust, streaming-friendly, and easy to integrate
- Error types are shared and extensible (see `src/error.rs`)
- Model download logic ensures files are always placed in the correct workspace root

## License

MIT 