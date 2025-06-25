use wasm_bindgen::prelude::*;
use js_sys::Function;
use web_sys::console;

#[wasm_bindgen]
pub struct BitNetWasm {
    // model: Option<bitnet_core::model::Transformer>,
}

#[wasm_bindgen]
impl BitNetWasm {
    /// Creates a new, empty BitNet instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // TODO: Initialize fields
        Self {
            // model: None,
        }
    }

    /// Loads a model and tokenizer from a URL (async).
    pub async fn load_model(&mut self, model_url: String) -> Result<(), JsValue> {
        // TODO: Fetch and deserialize model
        Ok(())
    }

    /// Generates text token-by-token, streaming results to a JS callback (async).
    pub async fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
        on_token_stream: &Function,
    ) -> Result<(), JsValue> {
        // TODO: Generation loop, call JS callback for each token
        Ok(())
    }

    /// Minimal test export for WASM <-> JS integration.
    #[wasm_bindgen]
    pub fn hello() -> String {
        "Hello from Rust!".to_string()
    }
}

#[wasm_bindgen]
pub fn hello() -> String {
    "Hello from Rust!".to_string()
}

#[wasm_bindgen]
pub fn hello_from_rust() {
    console::log_1(&"Hello from Rust!".into());
} 