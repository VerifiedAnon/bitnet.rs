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
- 🛠️ **Model conversion** — HuggingFace to BitNet format
- 🔄 **Quantization** — B1.58 ternary weights
- 🎯 **Optimized** — SIMD, LUT, and GPU kernels
- 🎨 **GUI & CLI** — Interactive and scriptable interfaces
- 🔍 **Visualization** — Attention maps and kernel profiling
- 🌐 **WASM-ready** — (Experimental)
- 🎯 **[Vibe Coding Ready](#vibe-coding)** — AI-assisted development with comprehensive planning i.e Project Plan, Checklist, and Cursor integration

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
git clone https://github.com/ocentra/bitnet-ocentra.git
cd bitnet-ocentra

# Build everything
cargo build --workspace

# Download a model and convert it
cargo run -p bitnet-converter -- --input-dir models/Original/microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir models/Converted/microsoft/bitnet-b1.58-2B-4T-bf16

# Run the app (CLI/GUI)
cargo run -p bitnet-app -- --help
```

---

## Build Instructions

1. **Install Rust (nightly recommended):**
   https://rustup.rs/
2. **Clone the repository:**
   ```sh
   git clone https://github.com/ocentra/bitnet-ocentra.git
   cd bitnet-ocentra
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
- **File Combiner Tools:**
  ```sh
  # Run the GUI version
  cargo run --release -p file_combiner_gui
  
  # Run the CLI version
  cargo run -p bitnet-tools --bin combine_files -- --help
  ```

---

## Contributing

Before contributing, please:

1. Read our [PROJECT_PLAN.md](PROJECT_PLAN.md) for architecture details
2. Review [CHECKLIST.md](CHECKLIST.md) for implementation status
3. Check crate-level READMEs for module-specific guidelines
4. Set up [Cursor](https://cursor.sh/) for optimal vibe coding experience

We use vibe coding practices to maintain high code quality and efficient development. See our [Vibe Coding](#vibe-coding) section for details.

---

## Vibe Coding

This project was developed using a modern AI-assisted development approach we call "vibe coding". Here's our exact workflow and how you can use it in your projects.

### Our Real-World Development Process

#### 1. Project Planning & Structure
```bash
# Initial project structure
PROJECT_PLAN.md        # Detailed architecture & implementation plan
CHECKLIST.md          # Task tracking & validation requirements
crates/               # Modular crate structure
  ├── bitnet-core/    # Core engine with its own README
  ├── bitnet-app/     # Application with its own README
  └── ...
```

#### 2. AI-Assisted Development Workflow

Here's exactly how we built BitNet-rs using AI tools:

1. **Initial Planning with AI**
   ```bash
   # 1. Created PROJECT_PLAN.md with high-level architecture
   # 2. Used Cursor + Claude to refine the plan:
   #    - Validated technical approaches
   #    - Identified potential issues
   #    - Added detailed implementation notes
   ```

2. **Code Organization & Review**
   ```bash
   # Used our file combiner tool to prepare code for AI review:
   cargo combine-files bitnet-core   # Combines core crate files
   cargo combine-files bitnet-app    # Combines app crate files
   
   # Generated files:
   bitnet-core_combined.txt    # Single file for AI review
   bitnet-app_combined.txt     # Single file for AI review
   ```

3. **Iterative Development with AI**
   ```bash
   # 1. Write code with Cursor + Claude
   # 2. Regular check-ins with AI:
   #    - Review code structure
   #    - Validate implementations
   #    - Debug issues
   #    - Optimize performance
   ```

### Practical Tips from Our Experience

1. **Project Structure First**
   - Start with a detailed `PROJECT_PLAN.md`
   - Create a `CHECKLIST.md` for tracking
   - Add READMEs in each major directory
   - This structure helps AI understand context

2. **Code Review Workflow**
   ```bash
   # 1. Combine related files:
   cargo combine-files crate-name
   
   # 2. Ask AI to review in Cursor:
   "Please review this code for..."
   
   # 3. Apply suggestions using Cursor's AI
   ```

3. **Using Multiple AI Models**
   - **Cursor + Claude**: Main development
   - **Gemini Pro**: Architecture review
   - **GitHub Copilot**: Quick suggestions
   - Cross-validate between models

4. **Documentation Strategy**
   ```markdown
   # Each crate has:
   - README.md           # Usage & examples
   - src/lib.rs         # API documentation
   - tests/             # Test documentation
   ```

### Our Tools & Setup

1. **File Combiner**
   ```toml
   # .cargo/config.toml
   [alias]
   combine-files = "run --package bitnet-tools --bin combine_files --"
   ```

2. **Development Environment**
   ```bash
   # Primary: Cursor IDE
   # - AI-assisted coding
   # - Code navigation
   # - Integrated terminal
   ```

3. **AI Integration**
   ```bash
   # 1. Cursor for main development
   # 2. Browser tabs for:
   #    - Gemini Pro (architecture)
   #    - GitHub Copilot (quick fixes)
   ```

### Real Examples from This Project

1. **Planning Phase**
   ```markdown
   # PROJECT_PLAN.md excerpt:
   ## Deep Dive: Critical Components
   1. BitLinear CustomOp
   2. CPU SIMD Implementation
   3. GPU Kernel Architecture
   ```

2. **Implementation Phase**
   ```rust
   // Example of AI-assisted implementation:
   // 1. Wrote high-level structure
   // 2. AI helped optimize SIMD code
   // 3. Validated with test cases
   ```

3. **Review & Optimization**
   ```bash
   # 1. Combined files for review
   # 2. AI analyzed performance
   # 3. Implemented suggestions
   ```

### Getting Started with Vibe Coding

1. **Setup Your Project**
   ```bash
   # 1. Create planning documents
   touch PROJECT_PLAN.md CHECKLIST.md
   
   # 2. Set up file combiner
   cargo add bitnet-tools --path crates/bitnet-tools
   ```

2. **Development Workflow**
   ```bash
   # 1. Plan with AI
   # 2. Implement with Cursor
   # 3. Review with combined files
   # 4. Iterate and optimize
   ```

3. **Best Practices**
   - Keep planning documents updated
   - Use consistent file structure
   - Document AI interactions
   - Cross-validate between models

### Advanced AI Development Patterns

#### Test-Driven Development with AI

We discovered that asking AI to write code directly often produces suboptimal results. Instead, we developed this effective pattern:

1. **Test-First Approach**
   ```rust
   // 1. Ask AI to write tests first
   #[test]
   fn test_bitnet_kernel_correctness() {
       // AI writes detailed test cases
       // with expected inputs/outputs
   }
   
   // 2. Use tests to define exact API
   pub trait BitnetKernel {
       fn execute(&self, input: &[f32]) -> Result<Vec<f32>>;
   }
   
   // 3. Implement based on test requirements
   // 4. Add validation tests
   ```

2. **Development Flow**
   ```bash
   # 1. Write test with AI
   cargo test --test kernel_tests -- --nocapture
   
   # 2. Iterative Implementation
   while test_status != "passed" {
       # Ask AI to analyze failure
       # Implement fixes
       # Run tests again
   }
   ```

#### Using Multiple AI Models Effectively

1. **Project-Wide Context with Gemini**
   ```bash
   # 1. Combine all project files
   cargo combine-files --all-crates
   
   # 2. Feed to Gemini 2.5 Pro (1M context)
   # - Full project structure
   # - All planning documents
   # - Implementation details
   ```

2. **Specialized Problem Solving**
   ```bash
   # Example: Working on bitnet-converter
   
   # 1. Get solution from Gemini
   cargo combine-files bitnet-converter
   # Feed to Gemini with specific task
   
   # 2. Validate with Cursor
   "Analyze this code from Gemini, focusing on:
    - Dependency correctness
    - Error handling
    - Performance implications"
   ```

#### AI-Driven Test Development

Here's our proven pattern for complex implementations:

1. **Goal-Based Testing**
   ```rust
   // Tell Cursor: "Goal: Test WGSL kernel correctness
   // Don't stop until tests pass"
   
   #[test]
   fn test_wgsl_kernel() {
       // 1. AI writes comprehensive test
       // 2. AI runs and debugs
       // 3. AI improves test coverage
   }
   ```

2. **YOLO Auto Mode**
   ```bash
   # Tell Cursor:
   "Enter YOLO auto mode:
    1. Search for similar implementations
    2. Write test cases
    3. Implement solution
    4. Test and fix
    5. Repeat until all tests pass"
   ```

3. **Real Example: Shader Testing**
   ```rust
   // Complex WGSL kernel testing
   #[test]
   fn test_bitnet_shader_computation() {
       let input = generate_test_data();
       let kernel = BitnetShader::new();
       
       // 1. Test scalar path
       let scalar_result = compute_scalar_reference(&input);
       
       // 2. Test GPU path
       let gpu_result = kernel.execute(&input)?;
       
       // 3. Compare results
       assert_results_match(scalar_result, gpu_result);
   }
   ```

#### Tips for Complex Implementations

1. **Shader/Kernel Development**
   ```bash
   # 1. Write scalar version first
   # 2. Ask AI to write exhaustive tests
   # 3. Implement optimized version
   # 4. Validate against scalar
   ```

2. **Using AI for Research**
   ```bash
   # 1. Combine relevant code
   cargo combine-files crates/bitnet-core/src/kernels
   
   # 2. Ask AI to:
   # - Analyze similar implementations
   # - Suggest optimization strategies
   # - Write validation tests
   ```

3. **Iterative Refinement**
   ```rust
   // 1. Start with basic test
   #[test]
   fn test_basic_functionality() {
       // Simple case
   }
   
   // 2. Add edge cases
   #[test]
   fn test_edge_cases() {
       // AI helps identify cases
   }
   
   // 3. Performance testing
   #[test]
   fn test_performance_requirements() {
       // AI helps set benchmarks
   }
   ```

This approach helped us tackle even the most complex parts of the project, like WGSL shaders and SIMD kernels, with confidence and reliability.

### Leveraging Large Context Windows & Multi-AI Workflow

#### Gemini 2.5 Pro: The Project Architect

We discovered a game-changing workflow using Gemini 2.5 Pro's massive 1M token context window in Google AI Studio:

1. **Full Project Context**
   ```bash
   # Combine EVERYTHING into one file:
   cargo combine-files --all-crates --include-docs
   
   # This includes:
   PROJECT_PLAN.md          # Architecture & deep dives
   CHECKLIST.md            # Implementation status
   All README.md files     # From each crate
   All source files        # Entire codebase
   All test files          # Complete test suite
   ```

2. **Real Example: Project-Wide Analysis**
   ```bash
   # 1. Feed the combined file to Gemini 2.5 Pro
   # Example prompt:
   "Analyze this entire BitNet-rs project. Focus on:
    - WGSL shader implementation in bitnet-core
    - Integration with CPU kernels
    - Test coverage gaps
    - Performance bottlenecks"
   
   # 2. Gemini sees EVERYTHING at once:
   # - Can reference code from any crate
   # - Understands full architecture
   # - Spots cross-crate dependencies
   # - Identifies global patterns
   ```

3. **Complex Problem Solving**
   ```bash
   # Real workflow we used for SIMD kernels:
   
   # 1. Combine all relevant files
   cargo combine-files \
     crates/bitnet-core/src/kernels \
     crates/bitnet-core/tests/kernel_tests.rs \
     PROJECT_PLAN.md
   
   # 2. Ask Gemini to architect the solution:
   "Design a SIMD kernel implementation that:
    - Matches the scalar implementation
    - Uses AVX2 intrinsics efficiently
    - Includes comprehensive test cases
    - Follows our project architecture"
   
   # 3. Get complete solution including:
   # - Full implementation
   # - Test suite
   # - Performance considerations
   # - Integration guidelines
   ```

#### Multi-AI Synergy Workflow

We developed a powerful workflow combining multiple AI models' strengths:

1. **Gemini → Claude → Cursor Pipeline**
   ```bash
   # Step 1: Get Initial Design (Gemini 2.5 Pro)
   # Feed full project context
   # Get comprehensive solution
   
   # Step 2: Refinement (Claude)
   "Analyze this code from Gemini. Focus on:
    1. Rust idioms and safety
    2. Error handling patterns
    3. Performance implications
    4. Integration with existing code"
   
   # Step 3: Implementation (Cursor)
   "You're the main coder. Review this design:
    1. Validate dependencies
    2. Check safety assumptions
    3. Implement with proper error handling
    4. Add comprehensive tests"
   ```

2. **Real Example: Shader Development**
   ```rust
   // 1. Gemini: Architecture & Initial Code
   // Given full project context, designed:
   struct BitnetShader {
       // Complete shader architecture
       // Memory layout design
       // Workgroup optimization
   }
   
   // 2. Claude: Code Review & Refinement
   // Analyzed and improved:
   // - Safety considerations
   // - Error handling
   // - Performance optimizations
   
   // 3. Cursor: Implementation & Testing
   // Final implementation with:
   // - Proper Rust idioms
   // - Comprehensive test suite
   // - Performance benchmarks
   ```

3. **Workflow Benefits**
   ```markdown
   # 1. Gemini 2.5 Pro
   - Sees entire project context
   - Understands global architecture
   - Provides complete solutions
   
   # 2. Claude
   - Excellent code review
   - Strong Rust knowledge
   - Safety focus
   
   # 3. Cursor
   - Immediate feedback
   - Code navigation
   - Test execution
   ```

#### Real-World Example: Complex Feature Development

Here's how we actually developed the BitNet quantization pipeline:

1. **Initial Architecture (Gemini)**
   ```bash
   # 1. Combined all files:
   cargo combine-files \
     PROJECT_PLAN.md \
     crates/bitnet-converter/src/*.rs \
     crates/bitnet-core/src/kernels/*.rs
   
   # 2. Asked Gemini:
   "Design a quantization pipeline that:
    - Converts FP32 weights to ternary
    - Implements efficient packing
    - Includes validation tests
    - Matches reference implementation"
   ```

2. **Code Review & Refinement (Claude)**
   ```rust
   // Received from Claude:
   // - Improved error handling
   // - Better type safety
   // - Optimized algorithms
   // - Additional test cases
   
   pub struct QuantizationPipeline {
       // Refined implementation
       // With Claude's improvements
   }
   ```

3. **Final Implementation (Cursor)**
   ```rust
   // Cursor helped:
   // 1. Integrate with existing code
   // 2. Add proper error types
   // 3. Implement all tests
   // 4. Validate performance
   ```

This multi-AI workflow was crucial for handling complex features like:
- WGSL shader implementation
- SIMD kernel optimization
- Quantization pipeline
- Test suite development

The key was leveraging each AI's strengths:
- Gemini's massive context window for architecture
- Claude's code review and safety focus
- Cursor's immediate feedback and testing

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
