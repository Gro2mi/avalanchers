import init, { WasmSimulation } from "./pkg/avalanchers.js";

async function run() {
    // 1. This fetches 'avalanchers_bg.wasm', compiles it, 
    // and populates the internal 'wasm' object.
    const statusEl = document.getElementById("status");
    
    statusEl.textContent = "Loading Engine...";
    await init();
    
    statusEl.textContent = "Engine Ready!";
    
    // 2. Now 'wasm' is defined, and you can call your functions.
    // greet("Avalanchers");
    const sim = await WasmSimulation.create_default("path/to/dem");
}

run().catch(console.error);