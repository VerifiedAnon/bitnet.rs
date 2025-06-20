# Burn Custom Kernel Test (`burn-custom-kernel-test-0-17`)

A minimal, standalone workspace for basic validation and prototyping of custom BitNet kernel logic, separate from the main BitNet project. Useful for debugging, rapid iteration, and verifying kernel correctness in isolation.

---

## Table of Contents

- [Purpose](#purpose)
- [Directory Structure](#directory-structure)
- [How to Use](#how-to-use)
- [What You Can Test Here](#what-you-can-test-here)
- [Why Use a Separate Workspace?](#why-use-a-separate-workspace)
- [Integration Notes](#integration-notes)

---

## Purpose

- Provide a clean, minimal environment for validating custom BitNet kernel logic (CPU/GPU)
- Enable rapid prototyping and debugging outside the complexity of the main BitNet workspace
- Serve as a sandbox for kernel, quantization, and serialization experiments

## Directory Structure

- `src/`: Source code for minimal kernel tests and validation logic
- `tests/`: Standalone test files (e.g., WGSL, Rust, etc.)
- `bitnet.pdf`: Reference documentation or notes
- `Cargo.toml`, `Cargo.lock`: Minimal Rust project setup
- `target/`: Build artifacts (ignored in version control)

## How to Use

```sh
cd custom-kernel-test
cargo test
# or run specific experiments as needed
```

- Add your kernel, quantization, or serialization code to `src/` or `tests/`
- Use this workspace to debug, validate, and iterate quickly

## What You Can Test Here

- Custom WGSL or Rust kernel logic (BitNet matmul, quantization, etc.)
- Minimal serialization/deserialization routines
- Isolated performance or correctness experiments
- Integration with minimal dependencies (e.g., Burn, safetensors, etc.)

## Why Use a Separate Workspace?

- Avoids interference from the main BitNet workspace's dependencies and build system
- Enables focused, fast iteration on kernel logic
- Useful for debugging issues that may be masked by workspace complexity
- Great for sharing minimal repros or experiments with collaborators

## Integration Notes

- Once validated here, kernel logic can be ported or integrated into the main BitNet workspace
- Keep this directory minimal and focused on validation/prototyping

---

**For questions or sharing experiments, see the main project README or open an issue.** 