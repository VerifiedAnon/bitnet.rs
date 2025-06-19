# BitNet-rs Project Checklist

Legend:
✅ Complete
🟡 Partial
🟠 Stub
❌ Missing

---

bitnet-rs/
├── Cargo.toml               ✅ Main workspace definition for all crates
├── .cargo/
│   └── config.toml          ✅ Cargo alias for `combine` (runs the file combiner GUI/CLI)
├── README.md                ❌ Project overview, build instructions, and usage examples
│   - [ ] README with usage examples
│   - [ ] Developer/contributor guide
├── src/
│   └── main.rs              ✅ Workspace-level entry point or test harness (used for integration tests, workspace-wide utilities)
├── crates/
│   ├── bitnet-core/
│   │   ├── Cargo.toml         ✅ Core crate manifest
│   │   │   - [ ] Features: gpu flag; Deps: wgpu, half, thiserror, tokenizers, safetensors, hf-hub; GPU-only: wgpu
│   │   ├── src/
│   │   │   ├── lib.rs         ✅ Main library module declarations
│   │   │   ├── error.rs       🟠 Stub, only TODO present
│   │   │   │   - [ ] Define custom error types using thiserror
│   │   │   ├── model.rs       🟡 Partial, struct and weight loading logic present, forward/generation logic stubbed
│   │   │   │   - [🟡] Group tensors by submodule prefix (function implemented, not fully integrated)
│   │   │   │   - [🟡] Pass grouped tensors to submodule constructors (logic present, not full)
│   │   │   │   - [🟠] Optionally call hf::download_model_and_tokenizer(...) (TODO, not implemented)
│   │   │   │   - [🟡] Actually construct all submodules using loaded weights/config (partial: some logic, not full)
│   │   │   │   - [🟠] Use self.tokenizer to encode the prompt (TODO, not implemented)
│   │   │   │   - [🟠] Lookup embeddings for input_ids using self.embedding (TODO, not implemented)
│   │   │   │   - [🟠] Forward through all blocks (attention/feedforward) (TODO, not implemented)
│   │   │   │   - [🟠] Final norm and logits head (TODO, not implemented)
│   │   │   │   - [🟠] Use settings and sampler to produce output tokens (TODO, not implemented)
│   │   │   │   - [🟠] Implement Transformer, TransformerBlock, etc. (TODO, not implemented)
│   │   │   │   - [🟠] Prompt/context management logic (not implemented)
│   │   │   │   - [🟠] Integration of sampler into generation loop (not implemented)
│   │   │   │   - [🟠] Embedding lookup and initial forward pass (not implemented)
│   │   │   │   - [🟠] Tokenizer integration in generation (not implemented)
│   │   │   │   - [🟠] Unit tests for sampler (not implemented)
│   │   │   ├── attention.rs   🟡 Partial, struct and from_weights logic present, core logic (forward, RoPE, etc.) missing
│   │   │   │   - [🟠] Implement multi-head attention, RoPE, etc. (not implemented)
│   │   │   ├── feed_forward.rs🟡 Partial, struct and from_weights logic present, core logic (forward, SwiGLU, etc.) missing
│   │   │   │   - [🟠] Implement SwiGLU, etc. (not implemented)
│   │   │   ├── rms_norm.rs    🟡 Partial, struct and from_weights logic present, wrapper not implemented
│   │   │   │   - [🟠] Implement wrapper for RMSNorm logic (not implemented)
│   │   │   ├── op.rs          🟡 Partial, struct and from_weights logic present, quantization/dispatch not implemented
│   │   │   │   - [🟠] Replace precomputed_lut with actual tensor type (not implemented)
│   │   │   │   - [🟠] Implement quantization and backend dispatch logic (not implemented, use wgpu for GPU)
│   │   │   │   - [🟠] Implement quantization logic (not implemented)
│   │   │   ├── tokenizer.rs   🟡 Partial, wrapper and encode implemented, chat/decode logic stubbed
│   │   │   │   - [🟡] Implement chat template logic (stub exists, not implemented, use tokenizers crate)
│   │   │   │   - [🟡] Implement decoding logic (stub exists, not implemented, use tokenizers crate)
│   │   │   ├── hf.rs          🟡 Partial, download logic implemented, error handling and robustness could be improved
│   │   │   │   - [🟡] Implement download logic using hf_hub (basic logic present)
│   │   │   ├── settings.rs    ✅ InferenceSettings struct and Default implemented
│   │   │   ├── embedding.rs   🟡 Partial, struct and from_weights logic present, forward logic missing
│   │   │   ├── training.rs    🟠 Stub, only TODO present
│   │   │   ├── visualization.rs 🟠 Stub, only TODO present
│   │   │   └── kernels/
│   │   │       ├── mod.rs     🟡 Partial, module aggregation logic present, no backend logic
│   │   │       │   - [🟡] Ensure correct aggregation of all kernel modules (mod.rs present, no logic)
│   │   │       ├── wgpu.rs    🟠 Stub, only TODO/unimplemented
│   │   │       │   - [ ] Implement wgpu kernel logic (no Candle, use WGSL)
│   │   │       ├── cpu.rs     🟠 Stub, only TODO/unimplemented
│   │   │       │   - [ ] Implement runtime SIMD detection and dispatch
│   │   │       ├── cpu_x86.rs 🟠 Stub, only TODO/unimplemented
│   │   │       │   - [ ] Implement AVX2 LUT kernel logic
│   │   │       ├── cpu_arm.rs 🟠 Stub, only TODO/unimplemented
│   │   │       │   - [ ] Implement NEON LUT kernel logic
│   │   │       ├── tests/
│   │   │       │   ├── validation.rs  🟡 Partial, test present for model loading/generation, not full golden tests
│   │   │       │   │   - [🟡] End-to-end golden tests (basic test present, not full coverage)
│   │   │       │   └── kernel_tests.rs🟠 Stub, scalar reference and test stubs present
│   │   │       │   │   - [🟠] Kernel tests (scalar vs SIMD/GPU, stubs present)
│   │   │       │   │   - [🟠] Unit tests for sampler and generation (not implemented)
│   │   │       │   │   - [🟠] Example config and prompt files (not implemented)
│   │   │       │   │   - [🟠] CI integration (clippy, tests) (not implemented)
│   │   │       └── gui/               # Core-level visualization and debugging UI for developers and advanced users
│   │   │           ├── mod.rs             🟡 Partial, entrypoint and module export logic present
│   │   │           ├── dashboard.rs       🟡 Partial, minimal dashboard UI present
│   │   │           ├── weights_viewer.rs  🟡 Partial, minimal weights viewer UI present
│   │   │           ├── kernel_profiler.rs 🟡 Partial, minimal kernel profiler UI present
│   │   │           ├── attention_map.rs   🟡 Partial, minimal attention map UI present
│   │   │           └── README.md          ✅ Core GUI documentation
│   │   ├── bitnet-converter/
│   │   │   ├── Cargo.toml             ✅ Converter crate manifest
│   │   │   ├── src/
│   │   │   │   ├── main.rs            ✅ CLI fully implemented: parses args, runs conversion pipeline, saves streaming-friendly output
│   │   │   │   ├── packer.rs          ✅ Quantize, permute, pack, and record logic fully implemented and tested
│   │   │   │   └── source.rs          ✅ .safetensors (BF16) loading fully implemented and tested
│   │   │   │   └── README.md          ✅ Converter documentation
│   │   │   ├── tests/
│   │   │   │   └── serialization_test.rs  ✅ Unit tests for serialization, streaming, and record types
│   │   │   │   - [✅] Serialization/deserialization tests for custom record types
│   │   │   │   - [✅] Streaming-friendly save/load tests
│   │   │   │   - [ ] End-to-end (E2E) golden test (not implemented)
│   │   │   │   - [ ] Actual model loading/inference test (not implemented)
│   │   │   - [✅] CLI parses args, runs full pipeline, saves per-block files
│   │   │   - [✅] Quantize, permute, pack, and record logic (with tests)
│   │   │   - [✅] .safetensors loading (BF16→f32, with tests)
│   │   │   - [✅] Comprehensive serialization and conversion unit tests (see tests/serialization_test.rs)
│   │   │   - [ ] End-to-end golden test 
│   │   │   - [ ] Actual model loading/inference test 
│   │   │   - [ ] Multi-format/model support 
│   │   ├── bitnet-app/
│   │   │   ├── Cargo.toml             ✅ App crate manifest
│   │   │   └── src/
│   │   │       ├── main.rs            🟠 Stub, only CLI/GUI stub present
│   │   │       │   - [ ] Parse CLI args and launch CLI/GUI
│   │   │       ├── cli.rs             🟠 Stub, only TODO present
│   │   │       │   - [ ] Implement CLI using clap
│   │   │       ├── generation.rs      🟠 Stub, only TODO present
│   │   │       │   - [ ] Implement Generator struct and generation logic
│   │   │       ├── sampler.rs         ✅ LogitsProcessor implemented
│   │   │       └── gui/
│   │   │               ├── mod.rs         🟠 Stub, minimal entrypoint
│   │   │               ├── app.rs         🟠 Stub, minimal UI
│   │   │               │   - [ ] Implement egui::App trait and UI logic
│   │   │               ├── state.rs       🟠 Stub, minimal state struct
│   │   │               │   - [ ] Implement state struct for chat history, settings, etc.
│   │   │               └── backend.rs     🟠 Stub, minimal backend logic
│   │   │               │   - [ ] Implement backend thread and message passing
│   │   │               └── README.md      ✅ App GUI documentation
│   │   │           └── README.md              ✅ App documentation
│   │   └── bitnet-tools/         ✅ Shared tools/utilities crate
│   │       ├── Cargo.toml        ✅ Tools crate manifest
│   │       └── src/
│   │           ├── lib.rs        ✅ Hugging Face download utility (library)
│   │           │   - [✅] download_model_and_tokenizer function
│   │           │   - [✅] Unit test for download logic (mocked)
│   │           ├── error.rs      ✅ Shared BitNetError type (used by all crates)
│   │           │   - [✅] BitNetError definition
│   │           └── main.rs       🟠 (Optional: CLI entry point for tools)
│   │       - [✅] Add hf-hub, tempfile dependencies
│   │       - [✅] Add to workspace
│   │       - [✅] Expose as dependency for other crates
│   │       - [✅] CLI for download (implemented, tested; progress bar functional, minor cosmetic issues possible)
│   │       - [✅] Shared error type for all crates
│   │       └── gui_combiner/
│   │           ├── Cargo.toml    ✅ GUI sub-crate manifest
│   │           └── src/
│   │               └── main.rs   ✅ GUI entry point for file combiner
│   │       ├── combine tool      ✅ File combiner CLI/GUI (cargo alias: combine, integrates both CLI and GUI modes)
│   └── README.md              ✅ App documentation

---

## Future/Optional (Roadmap)

- [ ] Tauri/egui GUI integration
- [ ] ONNX/other backend support

