# BitNet App (`bitnet-app`)

The main user-facing application for BitNet, providing both a command-line interface (CLI) and a modern graphical user interface (GUI) for model inference, chat, and exploration.

---

## Table of Contents

- [Purpose](#purpose)
- [Features](#features)
- [How to Run](#how-to-run)
- [Directory Structure](#directory-structure)
- [Implementation Notes](#implementation-notes)

---

## Purpose

- Run BitNet model inference from the command line or a desktop GUI
- Provide a simple, extensible interface for end users
- Integrate seamlessly with the core BitNet engine and tokenizer
- Support interactive chat, batch inference, and model exploration

## Features

- **CLI** for batch or scripted inference
- **GUI** for interactive chat and model exploration (built with egui/eframe)
- Configurable sampling, settings, and prompt management
- Extensible architecture for new features and UI panels
- Robust error handling and user feedback

## How to Run

### CLI

```sh
cargo run -p bitnet-app --features <features> -- <cli-args>
```

### GUI

```sh
cargo run -p bitnet-app --features egui --gui
```

## Directory Structure

- `src/main.rs`: Entry point, parses CLI args and launches CLI or GUI
- `src/cli.rs`: CLI argument parsing and logic
- `src/generation.rs`: Core text generation loop
- `src/sampler.rs`: Logits processing and sampling
- `src/gui/`: GUI modules (see gui/README.md)
    - `app.rs`: Main egui App implementation and UI layout
    - `state.rs`: UI state management (chat history, settings, etc.)
    - `backend.rs`: Backend thread and message passing for async inference

## Implementation Notes

- Uses egui/eframe for GUI
- Uses clap for CLI argument parsing
- Designed for extensibility and integration with bitnet-core
- See `src/gui/README.md` for details on extending the GUI

---

**For questions or contributions, see the main project README or open an issue.** 