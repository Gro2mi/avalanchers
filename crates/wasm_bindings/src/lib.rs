use wasm_bindgen::prelude::*;
use compute_core::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

// Export a `greet` function from Rust to JavaScript, that alerts a
// hello message.
#[wasm_bindgen]
pub fn greet(name: &str) {
    // The format! macro is safe, but the FFI call to alert is not.
    let message = format!("Hello, {}!", name);
    
    unsafe {
        alert(&message);
    }
}

// 🗝️ Key Architecture Tips for 2026

//     Async is Mandatory: In the browser, requestAdapter() and requestDevice() are asynchronous. You must use wasm-bindgen-futures to await these in Rust or your UI will hang.

//     The Canvas: Since your compute_core is shared, ensure it accepts a wgpu::Surface. On the web, you'll create this surface from an HtmlCanvasElement via web-sys.

//     Performance: Always use Float32Array::view (the unsafe version in Rust) to pass large data to JS. This avoids copying the entire elevation grid across the WASM boundary every frame.

//     Error Handling: Use console_error_panic_hook in your lib.rs init function. Without it, if your Rust code panics, the browser console will only show "unreachable", which is impossible to debug.

// Rust

#[wasm_bindgen(start)]
pub fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    compute_core::init_logging();
}