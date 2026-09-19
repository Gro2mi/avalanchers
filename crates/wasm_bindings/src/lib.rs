use compute_core::{TimestepData, settings::Settings};
use js_sys::Float32Array;
#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Object, Reflect, Uint8Array};
use serde_wasm_bindgen::from_value;
use simulation::Simulation;
use simulation::init_logging;
use std::sync::OnceLock;
#[allow(unused_imports)]
use tracing::{info, trace};
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::window;

#[cfg(target_arch = "wasm32")]
mod renderer;
#[cfg(target_arch = "wasm32")]
use renderer::RenderView;
#[cfg(target_arch = "wasm32")]
use simulation::SimulationState;

static BASE_URL: OnceLock<String> = OnceLock::new();

#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    tracing_wasm::set_as_global_default();
    init_logging();
    #[cfg(target_arch = "wasm32")]
    {
        let window = window().expect("no global window");
        let location = window.location();
        let origin = location.origin().unwrap_or_default() + "/";
        trace!("Base URI: {}", origin);
        trace!("Full URI: {}", location.href().unwrap_or_default());
        BASE_URL.set(location.href().unwrap_or_default()).ok();
    }
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
        let settings = Settings::loads(json).map_err(to_js_err)?;
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
    #[cfg(target_arch = "wasm32")]
    view: Option<RenderView>,
    /// Camera of the most recent view. Invalidations (DEM changes, re-runs) drop
    /// the view long before the re-attach happens, so the viewpoint is stashed
    /// here to survive that gap.
    #[cfg(target_arch = "wasm32")]
    last_camera: Option<render_core::OrbitCamera>,
}

/// Renderer housekeeping that is not part of the JS API.
#[cfg(target_arch = "wasm32")]
impl WasmSimulation {
    fn view_mut(&mut self) -> Result<&mut RenderView, JsValue> {
        self.view
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer not attached"))
    }

    /// A new DEM invalidates the renderer's terrain and every cloned buffer, so the
    /// canvas must be re-attached after the simulation is prepared again. The
    /// camera is stashed so the viewpoint survives the re-attach.
    fn invalidate_renderer(&mut self) {
        if let Some(view) = self.view.take() {
            self.last_camera = Some(view.renderer.camera);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WasmSimulation {
    fn invalidate_renderer(&mut self) {}
}

#[wasm_bindgen]
impl WasmSimulation {
    pub async fn new() -> Result<WasmSimulation, JsValue> {
        let inner = Simulation::new().await.map_err(to_js_err)?;
        Ok(WasmSimulation {
            inner,
            #[cfg(target_arch = "wasm32")]
            view: None,
            #[cfg(target_arch = "wasm32")]
            last_camera: None,
        })
    }
    pub async fn create_example(&mut self, dem_path: String) -> Result<(), JsValue> {
        let path = base_url().to_owned() + "data/avaframe/" + &dem_path + ".png";
        info!("Creating simulation with DEM path: {}", path);
        self.inner.create_example(&path).await.map_err(to_js_err)?;
        self.invalidate_renderer();
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn create(&mut self, val: JsValue) -> Result<(), JsValue> {
        let settings: Settings = from_value(val).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // 2. Run the async creation
        // Browser environment REQUIRES .await here. block_on() will panic.
        self.inner.create(settings).await.map_err(to_js_err)?;
        self.invalidate_renderer();
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
        self.invalidate_renderer();
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
        self.invalidate_renderer();
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
        self.invalidate_renderer();
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
        self.inner.prepare().await.map_err(to_js_err)?;
        // Re-preparing rebuilds buffers the renderer watches (e.g. new release areas).
        #[cfg(target_arch = "wasm32")]
        if let Some(view) = self.view.as_mut() {
            view.refresh_buffers(self.inner.orchestrator(), self.inner.number_particles());
        }
        Ok(())
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

    pub async fn fetch_slope_angle(&mut self) -> Result<(), JsValue> {
        self.inner.fetch_slope_angle().await.map_err(to_js_err)?;
        Ok(())
    }

    pub async fn fetch_slope_aspect(&mut self) -> Result<(), JsValue> {
        self.inner.fetch_slope_aspect().await.map_err(to_js_err)?;
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSimulation {
    /// Attaches a live renderer to `canvas`, drawing the simulation's own GPU
    /// buffers without copying data back to the CPU. The simulation must be
    /// prepared (`prepare()` or `run()`) so its buffers exist, and the canvas'
    /// `width`/`height` attributes must be set (ideally `clientWidth *
    /// devicePixelRatio`). Re-attaching replaces any previous renderer.
    pub fn attach_renderer(
        &mut self,
        canvas: web_sys::HtmlCanvasElement,
        exaggeration: f32,
    ) -> Result<(), JsValue> {
        // Re-attaches keep the current viewpoint: invalidation stashed the camera
        // of the dropped view, and a live view hands its camera over directly.
        self.invalidate_renderer();
        if self.inner.dem.data1d.is_empty() {
            return Err(JsValue::from_str("set a DEM before attaching the renderer"));
        }

        let mut view = RenderView::new(
            self.inner.orchestrator(),
            canvas,
            &self.inner.dem,
            exaggeration,
            self.inner.number_particles(),
        )
        .map_err(to_js_err)?;
        if let Some(camera) = self.last_camera {
            view.renderer.camera = camera;
            let (width, height) = view.renderer.size();
            view.renderer.camera.set_aspect(width, height);
        }
        self.view = Some(view);
        info!("renderer attached to canvas");
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn renderer_attached(&self) -> bool {
        self.view.is_some()
    }

    /// True once the simulation reaches (or passes) the `Finished` state, so a JS
    /// frame loop can stop stepping and keep rendering for camera inspection.
    #[wasm_bindgen(getter)]
    pub fn is_finished(&self) -> bool {
        self.inner.get_state() >= SimulationState::Finished
    }

    /// Advances the simulation by `steps` timesteps and renders one frame onto the
    /// canvas. Call this from a `requestAnimationFrame` loop; `steps = 0` only
    /// redraws, for example while the camera moves after the run finished.
    pub async fn render_frame(&mut self, steps: u32) -> Result<(), JsValue> {
        if steps > 0 && self.inner.get_state() < SimulationState::Finished {
            let info = self.inner.run_n_steps(steps).await.map_err(to_js_err)?;
            self.view_mut()?
                .renderer
                .particles_mut()
                .set_count(info.number_particles);
        }
        self.view_mut()?.render();
        Ok(())
    }

    /// Rotates the camera; inputs are pixel deltas from a drag.
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) -> Result<(), JsValue> {
        self.view_mut()?.orbit(delta_x, delta_y);
        Ok(())
    }

    /// Moves the camera target in the screen plane; inputs are pixel deltas from a drag.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) -> Result<(), JsValue> {
        self.view_mut()?.pan(delta_x, delta_y);
        Ok(())
    }

    /// Zooms towards or away from the target; positive `delta` zooms in. Pass the
    /// wheel/scroll delta.
    pub fn zoom(&mut self, delta: f32) -> Result<(), JsValue> {
        self.view_mut()?.zoom(delta);
        Ok(())
    }

    /// Resets the camera to the default terrain framing.
    pub fn reset_view(&mut self) -> Result<(), JsValue> {
        self.view_mut()?.reset_view();
        Ok(())
    }

    /// Drops the live view and the remembered camera without stashing either, so
    /// the next attach frames the new terrain afresh. Call before loading a
    /// different DEM; re-runs of the same DEM should keep the view instead.
    pub fn forget_view(&mut self) {
        self.view = None;
        self.last_camera = None;
    }

    /// Selects the scalar field tinting the terrain: "none", "peak_velocity",
    /// "peak_flow_thickness", "grid_mass", "release_areas", "slope_angle",
    /// "slope_aspect" or "roughness". A colour bar legend is drawn while an
    /// overlay is active.
    pub fn set_overlay(&mut self, name: String) -> Result<(), JsValue> {
        let overlay = renderer::Overlay::from_name(&name)
            .ok_or_else(|| JsValue::from_str(&format!("unknown overlay '{name}'")))?;
        let view = self.view_mut()?;
        view.overlay = overlay;
        view.apply_overlay();
        Ok(())
    }

    /// Shows or hides the simulation particles.
    pub fn set_particles_visible(&mut self, visible: bool) -> Result<(), JsValue> {
        let view = self.view_mut()?;
        view.show_particles = visible;
        view.apply_particles();
        Ok(())
    }

    /// Resizes the canvas surface; call after the canvas' `width`/`height`
    /// attributes changed, for example on window resize.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let view = self.view_mut()?;
        view.config.width = width;
        view.config.height = height;
        view.surface.configure(&view.device, &view.config);
        view.renderer.resize(&view.device, width, height);
        Ok(())
    }
}
