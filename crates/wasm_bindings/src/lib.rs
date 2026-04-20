use compute_core::{Simulation, TimestepData, settings::Settings};
use js_sys::Float32Array;
use std::sync::OnceLock;
use tracing::info;
use wasm_bindgen::prelude::*;
use web_sys::window;
static BASE_URL: OnceLock<String> = OnceLock::new();

#[wasm_bindgen(start)]
pub fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    tracing_wasm::set_as_global_default();
    compute_core::init_logging();
    let window = window().expect("no global window");
    let location = window.location();
    let origin = location.origin().unwrap_or_default() + "/";
    info!("Base URI: {}", origin);
    BASE_URL.set(origin).ok();
}

// Helper for error conversion to JS strings
fn to_js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

pub fn base_url() -> &'static str {
    BASE_URL.get().map(|s| s.as_str()).unwrap_or("./")
}

#[wasm_bindgen]
pub struct WasmTimestepData {
    inner: TimestepData,
}

#[wasm_bindgen]
impl WasmTimestepData {
    /// Helper to convert nested slices to a flat Float32Array for JS
    fn flatten_to_js<const N: usize>(&self, data: &[[f32; N]]) -> Float32Array {
        let flattened = data.as_flattened();
        unsafe { Float32Array::view(flattened) }
    }

    #[wasm_bindgen(getter)]
    pub fn velocity(&self) -> Float32Array {
        self.flatten_to_js(&self.inner.velocity)
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> Float32Array {
        self.flatten_to_js(&self.inner.position)
    }

    #[wasm_bindgen(getter)]
    pub fn dt(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.dt) }
    }
}

#[wasm_bindgen]
pub struct WasmSettings {
    pub(crate) inner: Settings,
}

#[wasm_bindgen]
impl WasmSettings {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Settings::default(),
        }
    }

    pub fn from_json(json: &str) -> Result<WasmSettings, JsValue> {
        let settings = Settings::from_json(json).map_err(to_js_err)?;
        Ok(WasmSettings { inner: settings })
    }

    #[wasm_bindgen(getter)]
    pub fn dem_path(&self) -> String {
        self.inner.dem_path.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_dem_path(&mut self, path: String) {
        self.inner.dem_path = path;
    }
}

impl Default for WasmSettings {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub struct WasmSimulation {
    inner: Simulation,
}

#[wasm_bindgen]
impl WasmSimulation {
    /// In WASM, we use wasm-bindgen-futures to handle async properly
    /// instead of pollster::block_on (which can freeze the browser thread).
    pub async fn create_default(dem_path: String) -> Result<WasmSimulation, JsValue> {
        let path = base_url().to_owned() + "data/avaframe/" + &dem_path + ".png";
        info!("Creating simulation with DEM path: {}", path);
        let inner = Simulation::create_default(path).await.map_err(to_js_err)?;
        Ok(WasmSimulation { inner })
    }

    pub async fn run(&mut self) -> Result<(), JsValue> {
        self.inner.run().await.map_err(to_js_err)
    }

    #[wasm_bindgen(getter)]
    pub fn cell_size(&self) -> f32 {
        self.inner.dem.cell_size
    }

    /// Returns the DEM data as a flat array.
    /// JS will need to know the width/height to treat it as 2D.
    #[wasm_bindgen(getter)]
    pub fn dem(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.dem.data1d) }
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.inner.dem.width as u32
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.inner.dem.height as u32
    }

    pub async fn get_max_velocity(&mut self) -> Result<Float32Array, JsValue> {
        let data = self.inner.get_max_velocity().await.map_err(to_js_err)?;
        // Note: Creating a view of a temporary Vec is unsafe, so we copy here
        Ok(Float32Array::from(data.as_slice()))
    }

    pub async fn get_timestep_data(&mut self) -> Result<WasmTimestepData, JsValue> {
        let data = self.inner.get_timestep_data().await.map_err(to_js_err)?;
        Ok(WasmTimestepData {
            inner: data.clone(),
        })
    }
}
