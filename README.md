# BitNet-rs

> **🚀 Modular, blazing-fast Rust engine for BitNet LLMs — conversion, inference, and research, with streaming and GPU/CPU support.**

---

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2021-orange" alt="Rust 2021" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License" />
  <img src="https://img.shields.io/badge/CI-passing-brightgreen" alt="CI Status" />
  <img src="https://img.shields.io/badge/Platform-CPU%20%7C%20GPU%20(WGSL)-purple" alt="Platform" />
</p>

---

## Features

- ⚡ **Pure Rust** — No Python or C++ runtime dependencies
- 🧩 **Modular** — Core, converter, tools, and app crates
- 🖥️ **CPU & GPU** — SIMD and WGSL (via wgpu) support
- 📦 **Streaming/blockwise** model loading and inference
- 🛠️ **Model conversion** — Hugging Face safetensors → BitNet format
- 🧪 **Robust validation** — Golden tests, kernel validation, and CI
- 🖼️ **CLI & GUI** — User-friendly app and tools
- 🌐 **WASM-ready** — (Experimental)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Build Instructions](#build-instructions)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [Checklist & Status](#checklist--status)
- [References](#references)

---

## Overview

BitNet-rs is a Rust-based toolkit for BitNet model conversion, inference, and experimentation. It is designed for:

- **Model conversion**: Convert Hugging Face safetensors to BitNet's custom, quantized, streaming-friendly format.
- **Inference**: Run BitNet models efficiently on CPU and GPU, with per-block streaming and minimal memory usage.
- **Extensibility**: Modular crates for core logic, conversion, tools, and user-facing apps (CLI/GUI).
- **Validation**: Rigorous test coverage, golden tests, and kernel validation.

---

## Quick Start

> **Note:** You need a recent Rust toolchain (nightly recommended) and a supported platform (Linux, macOS, Windows; CPU or GPU).

```sh
# Clone the repo
$ git clone <repo-url>
$ cd bitnet-rs

# Build everything
$ cargo build --workspace

# Download a model and convert it
$ cargo run -p bitnet-converter -- --input-dir models/Original/microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir models/Converted/microsoft/bitnet-b1.58-2B-4T-bf16

# Run the app (CLI/GUI)
$ cargo run -p bitnet-app -- --help
```

---

## Build Instructions

1. **Install Rust (nightly recommended):**
   https://rustup.rs/
2. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd bitnet-rs
   ```
3. **Build all crates:**
   ```sh
   cargo build --workspace
   ```
4. **Run tests:**
   ```sh
   cargo test --workspace --all-features
   ```
5. **Run the converter or app:**
   ```sh
   cargo run -p bitnet-converter -- --help
   cargo run -p bitnet-app -- --help
   ```

---

## Usage Examples

- **Convert a model:**
  ```sh
  cargo run -p bitnet-converter -- --input-dir models/Original/microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir models/Converted/microsoft/bitnet-b1.58-2B-4T-bf16
  ```
- **Run the app (CLI/GUI):**
  ```sh
  cargo run -p bitnet-app -- --help
  ```
- **Combine files (tools):**
  ```sh
  cargo run -p bitnet-tools --bin combine_files -- --help
  ```

---

## Contributing

- See [PROJECT_PLAN.md](PROJECT_PLAN.md) for architecture, module breakdown, and validation strategies.
- See [CHECKLIST.md](CHECKLIST.md) for current implementation status and TODOs.
- Please ensure all tests pass and follow the contribution guidelines in the project plan.

---

## Checklist & Status

- The current implementation status of each module and file is tracked in [CHECKLIST.md](CHECKLIST.md).
- Use this to find stubs, partials, and missing features.
- The checklist is updated regularly to reflect the actual state of the codebase.

---

## References

- [Official BitNet repo (Microsoft)](https://github.com/microsoft/BitNet)
- See `References/official/` for CUDA, Python, and kernel reference code.
- See crate-level READMEs for detailed module documentation.

---

> For more details, see the project plan and individual crate READMEs. 