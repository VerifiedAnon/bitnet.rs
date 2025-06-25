// main.js - BitNet WASM demo entry point (ESM, top-level await)

// --- WebSocket Log Bridge ---
let ws = null;
function setupWebSocket() {
    ws = new WebSocket('ws://localhost:8080/ws');
    ws.onopen = () => {
        console.info('[LogBridge] Connected to Rust WebSocket log server');
    };
    ws.onerror = (e) => {
        console.warn('[LogBridge] WebSocket error:', e);
    };
    ws.onclose = () => {
        console.info('[LogBridge] Disconnected from Rust WebSocket log server');
    };
    function sendLog(level, args) {
        const msg = {
            type: 'log',
            level,
            message: Array.from(args).map(String).join(' '),
            ts: new Date().toISOString(),
        };
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(msg));
        }
    }
    const origLog = console.log;
    const origError = console.error;
    console.log = function(...args) {
        sendLog('info', args);
        origLog.apply(console, args);
    };
    console.error = function(...args) {
        sendLog('error', args);
        origError.apply(console, args);
    };
}
setupWebSocket();

const outputArea = document.getElementById('output-area');
outputArea.value = '';

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

import init, { hello, hello_from_rust } from './pkg/bitnet_wasm.js';

async function main() {
    await init();
    document.getElementById('test-ping-btn').onclick = () => {
        const result = hello();
        appendToConsole(`WASM says: ${result}`);
        console.log('Hello from Rust!'); // Send log to server
    };
    document.getElementById('generate-report-btn').onclick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'done', test: 'browser_test' }));
            appendToConsole('[UI] Requested report generation...');
        } else {
            appendToConsole('[UI] WebSocket not connected, cannot generate report.');
        }
    };
}
main();

function appendToConsole(msg) {
    outputArea.value += (outputArea.value ? '\n' : '') + msg;
    outputArea.scrollTop = outputArea.scrollHeight;
}

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

// --- Tab Switching Logic ---
function switchTab(tab) {
    const tabs = ['test', 'chat', 'convert'];
    tabs.forEach(t => {
        document.getElementById('tab-content-' + t).classList.add('hidden');
        document.getElementById('tab-' + t).classList.remove('active');
    });
    document.getElementById('tab-content-' + tab).classList.remove('hidden');
    document.getElementById('tab-' + tab).classList.add('active');
}
document.getElementById('tab-test').onclick = () => switchTab('test');
document.getElementById('tab-chat').onclick = () => switchTab('chat');
document.getElementById('tab-convert').onclick = () => switchTab('convert');

// Default to Test Suits tab
switchTab('test'); 