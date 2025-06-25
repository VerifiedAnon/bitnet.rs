// main.js - BitNet WASM demo entry point (ESM, top-level await)

const outputArea = document.getElementById('output-area');
outputArea.value = 'BitNet WASM demo loaded.\n';

// Check if the WASM binary exists before importing the glue code
try {
    const wasmResponse = await fetch('./pkg/bitnet_wasm_bg.wasm', { method: 'HEAD' });
    if (!wasmResponse.ok) {
        throw new Error('WASM binary not found at ./pkg/bitnet_wasm_bg.wasm');
    }
} catch (err) {
    outputArea.value += `WASM file check failed: ${err}\n`;
    throw err;
}

try {
    const wasmModule = await import('./pkg/bitnet_wasm.js');
    // If the glue code exports a default init function, call it first:
    if (typeof wasmModule.default === 'function') {
        await wasmModule.default(); // This initializes the WASM instance
    }
    if (typeof wasmModule.hello !== 'function') {
        outputArea.value += 'WASM loaded, but hello() export is missing or not a function.\n';
    } else {
        const helloMsg = wasmModule.hello();
        outputArea.value += `WASM says: ${helloMsg}\n`;
    }
} catch (err) {
    outputArea.value += `Failed to load WASM: ${err}\n`;
    throw err;
}

// TODO: Wire up BitNetWasm API, prompt input, buttons, and streaming output

window.addEventListener('DOMContentLoaded', async () => {
    // Dynamically import the WASM module (pkg/bitnet_wasm.js)
    try {
        const wasm = await import('./pkg/bitnet_wasm.js');
        const helloMsg = wasm.hello();
        outputArea.value += `WASM says: ${helloMsg}\n`;
    } catch (err) {
        outputArea.value += `Failed to load WASM: ${err}\n`;
    }
}); 