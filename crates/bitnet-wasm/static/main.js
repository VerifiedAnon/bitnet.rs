// main.js - BitNet WASM demo entry point (ESM, top-level await)

// --- Global Error Handlers for WASM panics and JS errors ---
window.onerror = function(message, source, lineno, colno, error) {
    window.BitNetLogger.error(`[GlobalError] ${message} at ${source}:${lineno}:${colno}`);
    appendToConsole(`[GlobalError] ${message} at ${source}:${lineno}:${colno}`, window.LogLevel.ERROR);
};
window.onunhandledrejection = function(event) {
    window.BitNetLogger.error(`[UnhandledPromiseRejection] ${event.reason}`);
    appendToConsole(`[UnhandledPromiseRejection] ${event.reason}`, window.LogLevel.ERROR);
};

// --- WebSocket Log Bridge ---
let ws = null;

function sendLog(level, message, metadata) {
    const msg = {
        type: 'log',
        level,
        message,
        ts: new Date().toISOString(),
        ...metadata,
    };
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

function setupWebSocket() {
    ws = new WebSocket('ws://localhost:8080/ws');
    ws.onopen = () => {
        window.BitNetLogger.info('[LogBridge] Connected to Rust WebSocket log server');
    };
    ws.onerror = (e) => {
        window.BitNetLogger.error('[LogBridge] WebSocket error: ' + e);
    };
    ws.onclose = () => {
        window.BitNetLogger.info('[LogBridge] Disconnected from Rust WebSocket log server');
    };
}

// --- LogLevel Enum-like Object ---
window.LogLevel = Object.freeze({
    LOG: 'log',
    INFO: 'info',
    WARN: 'warn',
    ERROR: 'error',
    DEBUG: 'debug',
});

// --- Enhanced appendToConsole ---
function appendToConsole(msg, level = window.LogLevel.INFO) {
    // Extract stack trace for method name, file, and line number
    const stack = new Error().stack;
    let callerInfo = { method: 'unknown', file: 'unknown', line: 'unknown' };
    if (stack) {
        const lines = stack.split('\n');
        for (let i = 2; i < lines.length; i++) {
            if (lines[i] && !lines[i].includes('appendToConsole') && !lines[i].includes('BitNetLogger')) {
                const match = lines[i].match(/at\s+(.*?)\s+\((.*?):(\d+):(\d+)\)/) || lines[i].match(/at\s+(.*?):(\d+):(\d+)/);
                if (match) {
                    callerInfo = {
                        method: match[1] || 'anonymous',
                        file: match[2] || 'unknown',
                        line: match[3] || 'unknown',
                    };
                    break;
                }
            }
        }
    }
    const prefix = `[${callerInfo.file}:${callerInfo.line}] [${callerInfo.method}] `;
    const levelPrefix = `[${level.toUpperCase()}] `;
    const fullMsg = levelPrefix + prefix + msg;
    // UI output
    const outputArea = document.getElementById('output-area');
    outputArea.value += (outputArea.value ? '\n' : '') + fullMsg;
    outputArea.scrollTop = outputArea.scrollHeight;
    // Browser console output
    const consoleMethod = {
        [window.LogLevel.ERROR]: console.error,
        [window.LogLevel.WARN]: console.warn,
        [window.LogLevel.DEBUG]: console.debug,
        [window.LogLevel.INFO]: console.log,
        [window.LogLevel.LOG]: console.log,
    }[level] || console.log;
    consoleMethod(fullMsg);
    // WebSocket output
    sendLog(level, msg, callerInfo);
}

// Initialize WebSocket
setupWebSocket();

// --- BitNetLogger: Unified Logging ---
window.BitNetLogger = {
    log: (msg) => appendToConsole(msg, window.LogLevel.LOG),
    info: (msg) => appendToConsole(msg, window.LogLevel.INFO),
    warn: (msg) => appendToConsole(msg, window.LogLevel.WARN),
    error: (msg) => appendToConsole(msg, window.LogLevel.ERROR),
    debug: (msg) => appendToConsole(msg, window.LogLevel.DEBUG),
};
window.bitnet_wasm_log = window.BitNetLogger.info; // For WASM compatibility

// --- Main Logic ---
window.BitNetLogger.info('[BitNet] main.js loaded');

// Check if the WASM binary exists
try {
    window.BitNetLogger.info('[BitNet] Checking for WASM binary at ./pkg/bitnet_wasm_bg.wasm ...');
    const wasmResponse = await fetch('./pkg/bitnet_wasm_bg.wasm', { method: 'HEAD' });
    if (!wasmResponse.ok) {
        throw new Error('WASM binary not found at ./pkg/bitnet_wasm_bg.wasm');
    }
    window.BitNetLogger.info('[BitNet] WASM binary found.');
} catch (err) {
    window.BitNetLogger.error(`[BitNet] WASM file check failed: ${err}`);
    throw err;
}

try {
    window.BitNetLogger.info('[BitNet] Importing WASM module...');
    const wasmModule = await import('./pkg/bitnet_wasm.js');
    window.BitNetLogger.info('[BitNet] WASM module imported:', wasmModule);
    if (typeof wasmModule.default === 'function') {
        await wasmModule.default();
        window.BitNetLogger.info('[BitNet] WASM default() init called.');
    }
    if (typeof wasmModule.hello !== 'function') {
        window.BitNetLogger.error('[BitNet] hello() export missing.');
    } else {
        const helloMsg = wasmModule.hello();
        window.BitNetLogger.info(`[BitNet] hello() returned: ${helloMsg}`);
    }
} catch (err) {
    window.BitNetLogger.error(`[BitNet] Failed to load WASM: ${err}`);
    throw err;
}

// --- Robust Browser Test Runner and Reporter ---
window.bitnetTestResults = [];
window.bitnetTestLogs = [];

// --- Enhanced Global Log Capture ---
window.bitnetTestGlobalLogs = [];
(function() {
    const origLog = console.log;
    const origWarn = console.warn;
    const origError = console.error;
    console.log = function(...args) {
        window.bitnetTestGlobalLogs.push(`[INFO] ${args.join(' ')}`);
        origLog.apply(console, args);
    };
    console.warn = function(...args) {
        window.bitnetTestGlobalLogs.push(`[WARN] ${args.join(' ')}`);
        origWarn.apply(console, args);
    };
    console.error = function(...args) {
        window.bitnetTestGlobalLogs.push(`[ERROR] ${args.join(' ')}`);
        origError.apply(console, args);
    };
})();

async function runSingleTest(wasm, testName) {
    let status = "pass";
    let error = null;
    let logs = [];
    const origAppend = window.appendToConsole;
    window.appendToConsole = (msg, level = window.LogLevel.INFO) => {
        logs.push(`[${level}] ${msg}`);
        origAppend(msg, level);
    };
    const start = performance.now();
    try {
        await wasm[testName]();
    } catch (e) {
        status = "fail";
        error = e && e.message ? e.message : String(e);
        logs.push(`[ERROR] ${error}`);
    }
    const time_ms = performance.now() - start;
    window.appendToConsole = origAppend;
    return { test: testName, status, time_ms, error, logs };
}

async function runAllTestsAndSendReport() {
    const { default: init, get_test_list, ...wasm } = await import('./pkg/bitnet_wasm.js');
    await init();
    const testList = get_test_list();
    window.bitnetTestResults = [];
    window.bitnetTestLogs = [];
    window.BitNetLogger.info(`[TestRunner] Found ${testList.length} tests: ${Array.from(testList).join(', ')}`);
    appendToConsole(`[TestRunner] Starting test run for ${testList.length} tests...`, window.LogLevel.INFO);
    for (let i = 0; i < testList.length; i++) {
        const testName = testList[i];
        window.BitNetLogger.info(`[TestRunner] Running test ${i+1}/${testList.length}: ${testName}`);
        appendToConsole(`[TestRunner] Running test ${i+1}/${testList.length}: ${testName}`, window.LogLevel.INFO);
        const result = await runSingleTest(wasm, testName);
        window.bitnetTestResults.push(result);
        window.bitnetTestLogs.push(...result.logs);
        const statusMsg = result.status === 'pass' ? '✅ PASS' : `❌ FAIL (${result.error || 'unknown error'})`;
        appendToConsole(`[TestRunner] Finished test ${testName}: ${statusMsg} in ${result.time_ms.toFixed(2)} ms`, window.LogLevel.INFO);
    }
    appendToConsole(`[TestRunner] All tests complete. Sending report to server...`, window.LogLevel.INFO);
    if (ws && ws.readyState === WebSocket.OPEN) {
        const report = {
            type: 'test_report',
            results: window.bitnetTestResults,
            logs: window.bitnetTestGlobalLogs.slice(),
            // special_finding: ... (future extensibility)
        };
        ws.send(JSON.stringify(report));
        window.BitNetLogger.info('[UI] Sent structured test report to server.');
        appendToConsole(`[TestRunner] Report sent to server.`, window.LogLevel.INFO);
    } else {
        window.BitNetLogger.error('[UI] WebSocket not connected, cannot send test report.');
        appendToConsole(`[TestRunner] ERROR: WebSocket not connected, cannot send test report.`, window.LogLevel.ERROR);
    }
}

function setupGenerateReportButton() {
    const reportBtn = document.getElementById('generate-report-btn');
    if (reportBtn) {
        reportBtn.onclick = runAllTestsAndSendReport;
        window.BitNetLogger.info('[BitNet] Generate Report button handler attached (robust).');
    } else {
        window.BitNetLogger.warn('[BitNet] Generate Report button not found in DOM.');
    }
}

// --- Test Buttons Setup ---
async function setupTestButtons() {
    window.BitNetLogger.info('[BitNet] Initializing WASM via init()...');
    const { default: init, ...wasm } = await import('./pkg/bitnet_wasm.js');
    await init();
    window.BitNetLogger.info('[BitNet] WASM initialized. Exports:', wasm);
    if (typeof wasm.get_test_list !== 'function') {
        window.BitNetLogger.error('[BitNet] get_test_list() not exported from WASM!');
        return;
    }
    const testList = wasm.get_test_list();
    window.BitNetLogger.info('[BitNet] Test list from WASM:', testList);
    const grid = document.getElementById('test-buttons-grid');
    Array.from(grid.children).forEach(child => {
        if (child.id !== 'generate-report-btn' && child.id !== 'test-ping-btn') grid.removeChild(child);
    });

    for (let i = 0; i < testList.length; i++) {
        const testName = testList[i];
        const btn = document.createElement('button');
        btn.className = 'btn';
        btn.textContent = testName;
        btn.onclick = () => runTestWithReporting(testName, () => wasm[testName]());
        grid.appendChild(btn);
        window.BitNetLogger.info(`[BitNet] Test button generated: ${testName}`);
    }
    
    setupGenerateReportButton();
}

// --- Tab Switching Logic ---
function switchTab(tab) {
    const tabs = ['test', 'chat', 'convert'];
    tabs.forEach(t => {
        document.getElementById('tab-content-' + t).classList.add('hidden');
        document.getElementById('tab-' + t).classList.remove('active');
    });
    document.getElementById('tab-content-' + tab).classList.remove('hidden');
    document.getElementById('tab-' + tab).classList.add('active');
    window.BitNetLogger.info(`[BitNet] Switched to tab: ${tab}`);
}

// Attach tab switching handlers
document.getElementById('tab-test').onclick = () => switchTab('test');
document.getElementById('tab-chat').onclick = () => switchTab('chat');
document.getElementById('tab-convert').onclick = () => switchTab('convert');

// Default to Test Suits tab
switchTab('test');

// Initialize test buttons
if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', setupTestButtons);
} else {
    window.BitNetLogger.info('[BitNet] DOMContentLoaded event fired, calling setupTestButtons...');
    setupTestButtons();
}