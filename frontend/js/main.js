import init, { WasmSimulation } from "./pkg/avalanchers.js";

window.sim = null;
async function run() {
    // 1. This fetches 'avalanchers_bg.wasm', compiles it, 
    // and populates the internal 'wasm' object.
    const statusEl = document.getElementById("status");
    
    statusEl.textContent = "Loading Engine...";
    await init();
    
    statusEl.textContent = "Engine Ready!";
    
    // 2. Now 'wasm' is defined, and you can call your functions.
    // greet("Avalanchers");
    window.sim = await WasmSimulation.create_default("avaMal");
    window.sim.run();
}

run().catch(console.error);