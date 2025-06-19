# BitNet App GUI Module (bitnet-app/src/gui/)

This module provides the user-facing graphical interface for BitNet, built with egui/eframe.

## Purpose
- Enable interactive chat and model inference
- Display chat history, settings, and streaming output
- Provide a responsive, modern UI for end users

## Files/Modules
- `mod.rs`: Module root for the GUI
- `app.rs`: Main egui::App implementation and UI layout
- `state.rs`: Manages the state of the UI (chat history, settings, etc.)
- `backend.rs`: Handles communication between the UI and the inference thread

## How to Extend
- Add new UI panels or widgets in `app.rs`
- Store new state in `state.rs`
- Use `backend.rs` for async model inference or streaming

## Implementation Notes
- Uses egui/eframe for cross-platform desktop GUI
- Designed for extensibility and responsiveness
- See the main app for how to launch the GUI 