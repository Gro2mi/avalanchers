use compute_core::{TimestepData, settings::Settings};
use js_sys::Float32Array;
#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Object, Reflect, Uint8Array};
use serde_wasm_bindgen::from_value;
use simulation::{Simulation, init_logging};
use std::sync::OnceLock;
use tracing::{info, trace};
use wasm_bindgen::prelude::*;
use web_sys::window;
static BASE_URL: OnceLock<String> = OnceLock::new();

#[wasm_bindgen(start)]
pub fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    tracing_wasm::set_as_global_default();
    init_logging();
    let window = window().expect("no global window");
    let location = window.location();
    let origin = location.origin().unwrap_or_default() + "/";
    trace!("Base URI: {}", origin);
    trace!("Full URI: {}", location.href().unwrap_or_default());
    BASE_URL.set(location.href().unwrap_or_default()).ok();
}

// Helper for error conversion to JS strings
fn to_js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[cfg(target_arch = "wasm32")]
/// Decodes a blosc compressed Zarr chunk, including the bitshuffle filter.
#[wasm_bindgen]
pub fn decode_blosc_chunk(bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    data_processor::blosc::decode_blosc(bytes).map_err(to_js_err)
}

#[cfg(target_arch = "wasm32")]
/// Decodes a standalone zstd frame.
#[wasm_bindgen]
pub fn decode_zstd_chunk(bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    data_processor::blosc::decode_zstd_frame(bytes).map_err(to_js_err)
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
    pub fn position(&self) -> Float32Array {
        self.flatten_to_js(&self.inner.position)
    }
    #[wasm_bindgen(getter, js_name = dt)]
    pub fn dt(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.dt) }
    }

    // #[wasm_bindgen(getter, js_name = elevation)]
    // pub fn elevation(&self) -> Float32Array {
    //     unsafe { Float32Array::view(&self.inner.elevation) }
    // }

    #[wasm_bindgen(getter, js_name = velocityMagnitude)]
    pub fn velocity_magnitude(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.velocity_magnitude) }
    }

    #[wasm_bindgen(getter, js_name = time)]
    pub fn time(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.time) }
    }

    #[wasm_bindgen(getter, js_name = stepDistance)]
    pub fn step_distance(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.step_distance2d) }
    }

    #[wasm_bindgen(getter, js_name = travelDistance)]
    pub fn travel_distance(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.travel_distance2d) }
    }

    #[wasm_bindgen(getter, js_name = cfl)]
    pub fn cfl(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.inner.cfl) }
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
        self.inner.dem_path.clone().unwrap_or_else(|| "".into())
    }

    #[wasm_bindgen(setter)]
    pub fn set_dem_path(&mut self, path: String) {
        self.inner.dem_path = Some(path);
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
    pub async fn new() -> Result<WasmSimulation, JsValue> {
        let inner = Simulation::new().await.map_err(to_js_err)?;
        Ok(WasmSimulation { inner })
    }
    pub async fn create_example(&mut self, dem_path: String) -> Result<(), JsValue> {
        let path = base_url().to_owned() + "data/avaframe/" + &dem_path + ".png";
        info!("Creating simulation with DEM path: {}", path);
        self.inner.create_example(&path).await.map_err(to_js_err)?;
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn create(&mut self, val: JsValue) -> Result<(), JsValue> {
        let settings: Settings = from_value(val).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // 2. Run the async creation
        // Browser environment REQUIRES .await here. block_on() will panic.
        self.inner.create(settings).await.map_err(to_js_err)?;
        Ok(())
    }

    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub async fn set_dem(
        &mut self,
        dem_data: &[f32],
        width: u32,
        height: u32,
        cell_size: f32,
        bounds_xmin: f32,
        bounds_xmax: f32,
        bounds_ymin: f32,
        bounds_ymax: f32,
        map_factor: f32,
    ) -> Result<(), JsValue> {
        self.inner
            .set_dem_with_bounds(
                dem_data,
                width as usize,
                height as usize,
                cell_size,
                bounds_xmin,
                bounds_xmax,
                bounds_ymin,
                bounds_ymax,
                map_factor,
            )
            .map_err(to_js_err)?;
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn set_dem_default(
        &mut self,
        dem_data: &[f32],
        width: u32,
        height: u32,
        cell_size: f32,
    ) -> Result<(), JsValue> {
        self.inner
            .set_dem_default(dem_data, width as usize, height as usize, cell_size)
            .map_err(to_js_err)?;
        Ok(())
    }

    /// Loads a DEM from an uploaded file. `ext` selects the parser, e.g. "asc" or "tif".
    #[wasm_bindgen]
    pub async fn load_dem_bytes(
        &mut self,
        bytes: &[u8],
        ext: String,
        source: String,
    ) -> Result<(), JsValue> {
        let dem = data_processor::load_dem_from_bytes(bytes, &ext, &source).map_err(to_js_err)?;
        info!(
            "Loaded DEM '{}' from bytes: {}x{} at {}m",
            source, dem.width, dem.height, dem.cell_size
        );
        self.inner
            .set_dem_with_bounds(
                &dem.data1d,
                dem.width,
                dem.height,
                dem.cell_size,
                dem.bounds.xmin,
                dem.bounds.xmax,
                dem.bounds.ymin,
                dem.bounds.ymax,
                dem.map_factor,
            )
            .map_err(to_js_err)?;
        Ok(())
    }

    /// Loads release areas from an uploaded file. Requires a DEM to be set first.
    #[wasm_bindgen]
    pub async fn load_release_areas_bytes(
        &mut self,
        bytes: &[u8],
        ext: String,
    ) -> Result<(), JsValue> {
        let data = data_processor::load_release_areas_from_bytes(bytes, &ext).map_err(to_js_err)?;
        self.inner.set_release_areas(&data).map_err(to_js_err)?;
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn set_release_areas(&mut self, release_areas: &[f32]) -> Result<(), JsValue> {
        self.inner
            .set_release_areas(release_areas)
            .map_err(to_js_err)?;
        Ok(())
    }

    pub async fn prepare(&mut self) -> Result<(), JsValue> {
        self.inner.prepare().await.map_err(to_js_err)
    }

    #[cfg(target_arch = "wasm32")]
    /// Builds a Zarr store of the current results as `{ path, bytes }` entries.
    ///
    /// The browser cannot write files from Rust, so the caller persists them.
    pub async fn save_results_zarr(&mut self) -> Result<Array, JsValue> {
        let entries = self.inner.export_zarr_entries().await.map_err(to_js_err)?;

        let files = Array::new();
        for entry in entries {
            let file = Object::new();
            Reflect::set(&file, &"path".into(), &JsValue::from_str(&entry.path))?;
            Reflect::set(&file, &"bytes".into(), &Uint8Array::from(&entry.bytes[..]))?;
            files.push(&file);
        }
        Ok(files)
    }

    #[cfg(target_arch = "wasm32")]
    /// Suggested folder name for the exported store.
    #[wasm_bindgen(getter)]
    pub fn result_store_name(&self) -> String {
        format!("{}.zarr", self.inner.site_name())
    }

    pub async fn run(&mut self) -> Result<(), JsValue> {
        self.inner.run().await.map_err(to_js_err)
    }

    #[wasm_bindgen(getter)]
    pub fn bounds(&self) -> Float32Array {
        let b = &self.inner.dem.bounds;
        Float32Array::from(&[b.xmin, b.ymin, b.xmax, b.ymax][..])
    }

    #[wasm_bindgen(getter)]
    pub fn cell_size(&self) -> f32 {
        self.inner.dem.cell_size
    }

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

    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Float32Array {
        Float32Array::from(self.inner.dem.x.as_slice())
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float32Array {
        Float32Array::from(self.inner.dem.y.as_slice())
    }

    #[wasm_bindgen(getter)]
    pub fn dem_trajectory_info(&self) -> Float32Array {
        let vals = [
            self.inner.dem.bounds.xmin,
            self.inner.dem.bounds.ymin,
            self.inner.dem.map_factor,
        ];
        Float32Array::from(&vals[..])
    }

    pub async fn fetch_peak_velocity(&mut self) -> Result<(), JsValue> {
        self.inner.fetch_peak_velocity().await.map_err(to_js_err)?;
        Ok(())
    }

    pub async fn fetch_peak_flow_thickness(&mut self) -> Result<(), JsValue> {
        self.inner
            .fetch_peak_flow_thickness()
            .await
            .map_err(to_js_err)?;
        Ok(())
    }

    /// Reads the release areas into the cache. Requires `prepare` or `run` first.
    pub async fn fetch_release_areas(&mut self) -> Result<(), JsValue> {
        self.inner.fetch_release_areas().await.map_err(to_js_err)?;
        Ok(())
    }

    /// Reads roughness into the cache. Requires `prepare` or `run` first.
    pub async fn fetch_roughness(&mut self) -> Result<(), JsValue> {
        self.inner.fetch_roughness().await.map_err(to_js_err)?;
        Ok(())
    }

    pub async fn fetch_results(&mut self) -> Result<(), JsValue> {
        self.inner.fetch_results().await.map_err(to_js_err)?;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn peak_velocity(&self) -> Float32Array {
        match self.inner.gpu_cache.peak_velocity.as_ref() {
            Some(data) => unsafe { Float32Array::view(data) },
            None => Float32Array::new_with_length(0),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn slope_aspect(&self) -> Float32Array {
        unsafe { Float32Array::view(self.inner.gpu_cache.slope_aspect.as_ref().unwrap()) }
    }

    #[wasm_bindgen(getter)]
    pub fn slope_angle(&self) -> Float32Array {
        unsafe { Float32Array::view(self.inner.gpu_cache.slope_angle.as_ref().unwrap()) }
    }

    #[wasm_bindgen(getter)]
    pub fn roughness(&self) -> Float32Array {
        unsafe { Float32Array::view(self.inner.gpu_cache.roughness.as_ref().unwrap()) }
    }

    #[wasm_bindgen(getter)]
    pub fn release_areas(&self) -> Float32Array {
        unsafe { Float32Array::view(self.inner.gpu_cache.release_areas.as_ref().unwrap()) }
    }

    #[wasm_bindgen(getter)]
    pub fn peak_flow_thickness(&self) -> Float32Array {
        match self.inner.gpu_cache.peak_flow_thickness.as_ref() {
            Some(data) => unsafe { Float32Array::view(data) },
            None => Float32Array::new_with_length(0),
        }
    }

    #[wasm_bindgen]
    pub async fn get_timestep_data(&mut self) -> Result<WasmTimestepData, JsValue> {
        let data = self.inner.fetch_timestep_data().await.map_err(to_js_err)?;
        Ok(WasmTimestepData {
            inner: data.clone(),
        })
    }
}
