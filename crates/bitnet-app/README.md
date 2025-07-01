# BitNet App (`bitnet-app`)

> **NOTE:** As of now, the CLI and GUI are stubs/TODOs. Only `sampler.rs` is fully implemented. All other modules (main.rs, cli.rs, generation.rs, gui/app.rs, gui/state.rs, gui/backend.rs) are placeholders and not functional yet.

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

- Run BitNet model inference from the command line or a desktop GUI (planned)
- Provide a simple, extensible interface for end users (planned)
- Integrate seamlessly with the core BitNet engine and tokenizer (planned)
- Support interactive chat, batch inference, and model exploration (planned)

## Features

- **CLI** for batch or scripted inference (**stub**)
- **GUI** for interactive chat and model exploration (built with egui/eframe) (**stub**)
- Configurable sampling, settings, and prompt management (**planned**)
- Extensible architecture for new features and UI panels (**planned**)
- Robust error handling and user feedback (**planned**)

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

- `src/main.rs`: Entry point, parses CLI args and launches CLI or GUI (**stub**)
- `src/cli.rs`: CLI argument parsing and logic (**stub**)
- `src/generation.rs`: Core text generation loop (**stub**)
- `src/sampler.rs`: Logits processing and sampling (**implemented**)
- `src/gui/`: GUI modules (see gui/README.md)
    - `app.rs`: Main egui App implementation and UI layout (**stub**)
    - `state.rs`: UI state management (chat history, settings, etc.) (**stub**)
    - `backend.rs`: Backend thread and message passing for async inference (**stub**)

## Implementation Notes

- Uses egui/eframe for GUI (planned)
- Uses clap for CLI argument parsing (planned)
- Designed for extensibility and integration with bitnet-core (planned)
- See `src/gui/README.md` for details on extending the GUI

---

**For questions or contributions, see the main project README or open an issue.** 