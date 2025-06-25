// main.js - BitNet WASM demo entry point (ESM, top-level await, Trunk workflow)

const outputArea = document.getElementById('output-area');
outputArea.value = 'BitNet WASM demo loaded.\n';

try {
    // Trunk generates the WASM JS glue in the output root, so import directly
    const init = (await import('./bitnet_wasm.js')).default;
    const { hello } = await import('./bitnet_wasm.js');
    await init(); // Initialize the WASM module

    if (typeof hello === 'function') {
        const helloMsg = hello();
        outputArea.value += `WASM says: ${helloMsg}\n`;
    } else {
        outputArea.value += 'WASM loaded, but hello() export is missing or not a function.\n';
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