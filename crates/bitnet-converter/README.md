# BitNet Converter (`bitnet-converter`)

A robust, streaming-friendly Rust tool for converting standard model weights (e.g., Hugging Face safetensors) into the optimized, quantized format required by the BitNet engine.

---

## Table of Contents

- [Purpose](#purpose)
- [Features](#features)
- [Conversion Pipeline](#conversion-pipeline)
- [CLI Usage](#cli-usage)
- [Output Format](#output-format)
- [Error Handling](#error-handling)
- [Parallelism & Performance](#parallelism--performance)
- [Test Coverage](#test-coverage)
- [Implementation Notes](#implementation-notes)

---

## Purpose

- Convert model weights from common formats (e.g., Hugging Face safetensors) to BitNet's custom, quantized, streaming-friendly format
- Apply quantization, permutation, packing, and interleaving steps
- Ensure compatibility and performance for BitNet inference

## Features

- **Burn-free, pure Rust**: No dependency on the `burn` framework for conversion
- **Streaming output**: Each model block and top-level module is saved as a separate file for efficient loading
- **Parallelized**: Layer conversion and file writing are parallelized for speed
- **Robust loader**: Handles 1D/2D tensors, shape promotion, and errors gracefully
- **Comprehensive error handling**: Clear, actionable errors for missing tensors, shape mismatches, and more
- **Extensive tests**: Serialization, streaming, loader, and error cases are all covered

## Conversion Pipeline

1. **Load config and safetensors** from disk
2. **Parse tensors** using a minimal, robust loader (BF16 → f32, shape promotion)
3. **Quantize and pack** weights using SIMD-optimized and fallback routines
4. **Structure the model** into serializable records (embedding, blocks, norms, lm_head, metadata)
5. **Serialize each part** to a separate `.bin` file in the output directory

## CLI Usage

```sh
cargo run -p bitnet-converter -- [--input-dir <input_dir>] [--output-dir <output_dir>]
```

- If not provided, defaults to the official BitNet-2B model subdirectory under the workspace's models folder.
- Logs are written to `logs/bitnet-converter-<timestamp>.txt`.

**Example:**

```sh
cargo run -p bitnet-converter -- --input-dir models/Original/microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir models/Converted/microsoft/bitnet-b1.58-2B-4T-bf16
```

## Output Format

- **Per-block files**: Each transformer block is saved as `block_<i>.bin`
- **Top-level files**: `embedding.bin`, `norm.bin`, `lm_head.bin`
- **Metadata**: Each output includes a `ModelMetadata` struct (layer count, vocab size, hidden size, timestamp)
- **Format**: All files are serialized using `bincode` for fast, compact storage

## Error Handling

- Uses a custom `ConversionError` enum for clear, actionable errors
- Handles missing tensors, shape mismatches, and invalid data robustly
- Loader gracefully skips unsupported shapes and corrupt files

## Parallelism & Performance

- Uses `rayon` for parallel layer processing and file writing
- SIMD quantization is used where available, with fallback for other platforms
- Loader is minimal and efficient

## Test Coverage

- Serialization/deserialization of full models and individual blocks
- Streaming (per-block) output and reassembly
- Loader correctness for 1D/2D tensors, error cases, and shape promotion
- Error handling for corrupt files and unsupported shapes

## Implementation Notes

- Designed for extensibility to support new formats and quantization schemes
- See the project plan for details on the conversion pipeline
- For more details on the packing format and quantization, see code comments in `src/packer.rs`

---

**For questions or contributions, see the main project README or open an issue.** 