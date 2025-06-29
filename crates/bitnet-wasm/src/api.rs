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

// Helper trait to convert GpuTestError to JsValue
trait ToJsValueResult<T> {
    fn to_jsvalue(self) -> Result<T, JsValue>;
}
impl<T> ToJsValueResult<T> for Result<T, crate::tests::GpuTestError> {
    fn to_jsvalue(self) -> Result<T, JsValue> {
        self.map_err(|e| JsValue::from_str(&format!("{:?}", e)))
    }
}
impl<T> ToJsValueResult<T> for Result<T, JsValue> {
    fn to_jsvalue(self) -> Result<T, JsValue> { self }
}

// --- Macro-generated test runners and test list ---
macro_rules! export_tests {
    (
        $(
            $(#[$meta:meta])*
            $fn_name:ident
        ),* $(,)?
    ) => {
        $(
            $(#[$meta])*
            #[wasm_bindgen]
            pub async fn $fn_name() -> Result<JsValue, JsValue> {
                crate::tests::$fn_name().await
            }
        )*
        #[wasm_bindgen]
        pub fn get_test_list() -> js_sys::Array {
            let arr = js_sys::Array::new();
            $(
                arr.push(&JsValue::from_str(stringify!($fn_name)));
            )*
            arr
        }
    }
}

export_tests!(
    run_test_kernel_compile_optimal,
    run_test_kernel_compile_wasm,
    run_test_kernel_compile_default,
    run_test_inline_add_scalar_shader_compile,   
    test_ping_gpu_info,
    run_test_minimal_shader_compile,
); 

