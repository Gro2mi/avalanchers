//! High-level, GPU-accelerated snow avalanche simulation.
//!
//! This crate drives the wgpu compute shaders in [`compute_core`] through a
//! single [`Simulation`] type that owns the full pipeline:
//!
//! 1. **Load** settings, a DEM and optionally a region of interest from a
//!    [`Settings`] file (`Simulation::create*`), or set the DEM, release
//!    areas and outline programmatically
//!    ([`Simulation::set_dem_with_bounds`], [`Simulation::set_release_areas`],
//!    [`Simulation::set_roi`]).
//! 2. **Prepare** the run: terrain analysis, release-area determination and
//!    particle seeding ([`Simulation::prepare`]).
//! 3. **Simulate**: step the particle or MPM model
//!    ([`Simulation::run_n_steps`], [`Simulation::run`]).
//! 4. **Consume**: read results back through the cached `fetch_*` methods,
//!    extract the avalanche mask ([`Simulation::post_process`]), evaluate it
//!    against the outline ([`Simulation::evaluate`],
//!    [`Simulation::evaluate_gpu`]) and persist everything to a Zarr store
//!    ([`Simulation::save`] on native, `export_zarr_entries` in the browser).
//!
//! Each stage records its completion in a [`SimulationState`]; methods bail
//! with an error when their prerequisites have not run yet. The same code
//! compiles for native targets and `wasm32`, but file output only exists on
//! native.
//!
//! A minimal native example:
//!
//! ```ignore
//! let mut sim = Simulation::new().await?;
//! sim.create_default_with_release_areas("data/dem.png", "data/release.png")
//!     .await?;
//! sim.run().await?;                       // terrain + release + particles + physics
//! sim.post_process().await?;              // avalanche mask from peak flow thickness
//! let evaluation = sim.evaluate().await?;  // metrics against the outline
//! sim.save().await?;                      // write Zarr results
//! ```

use anyhow::{Result, bail};
use compute_core::{
    ComputeOrchestrator, GpuCache, ParticleState, SimInfo, SimInfoFlags, TextureRgba, TimestepData,
    buffers::{AtomicValues, BufferName, CenterOfMassResult, TextureName},
    dem::{Bounds, Dem, StableHasher},
    evaluation::MassMovementEvaluation,
    post_processing::*,
    settings::{CrownLineMethod, ReleaseFillMethod, Settings, SimModel, SimSettings},
    utils::*,
};
use data_processor::store_schema::{RunRecord, store_path_stem};
#[cfg(target_arch = "wasm32")]
use data_processor::zarr_writer::ZarrEntry;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Once;
use web_time::{Instant, SystemTime, UNIX_EPOCH};

/// Release-area and crown-line estimation from a DEM and avalanche outline.
pub mod release_estimation;
static INIT: Once = Once::new();
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initializes the global tracing subscriber; later calls are no-ops.
///
/// The filter is fixed per build profile: debug builds log
/// `compute_core`/`simulation` at `trace` level, release builds only `info`
/// and above (plus `error` from all other crates).
pub fn init_logging() {
    INIT.call_once(|| {
        #[cfg(debug_assertions)]
        let filter = EnvFilter::new(
            "error,simulation=trace,compute_core=trace,data_processor=debug,cli=debug",
        );
        #[cfg(not(debug_assertions))]
        let filter =
            EnvFilter::new("error,simulation=info,compute_core=info,data_processor=info,cli=info");

        let _ = tracing_subscriber::registry()
            .with(fmt::layer().with_target(false))
            .with(filter)
            .try_init();

        debug!("Avalanchers logging initialized");
    });
}

/// Lifecycle of a [`Simulation`], ordered from creation to evaluation so that
/// `state >= X` means "stage `X` and every stage before it have completed".
///
/// [`Simulation`] methods check this ordering and return an error when their
/// prerequisites have not run yet.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SimulationState {
    /// Freshly constructed; no settings or DEM loaded.
    Uninitialized,
    /// Data applied, but the DEM is empty.
    DemMissing,
    /// A DEM is loaded and ready for terrain analysis.
    DemLoaded,
    /// Slope, aspect, curvature and related terrain metrics computed on the GPU.
    TerrainAnalyzed,
    /// Release areas determined and the particle count fixed.
    ReleaseAreasComputed,
    /// Particles seeded on the GPU; the simulation can be stepped.
    ParticlesInitialized,
    /// At least one compute step has run; more steps can follow.
    Running,
    /// Physics finished: `max_steps` reached or the GPU signalled `SIM_STOPPED`.
    Finished,
    /// Peak grids thresholded and the avalanche mask extracted
    /// ([`Simulation::post_process`]).
    PostProcessed,
    /// Evaluation metrics computed against the region of interest
    /// ([`Simulation::evaluate`]).
    Evaluated,
}

/// Inputs of the release stage besides the [`SimSettings`]; a change between
/// [`Simulation::store_initial_state`] and
/// [`Simulation::restore_initial_state`] re-runs the release areas and the
/// particle initialization.
#[derive(Debug, Clone, PartialEq)]
struct ReleaseConfig {
    release_areas_path: Option<String>,
    release_hash: u64,
    roi_hash: u64,
    release_area_fraction: Option<f32>,
    crown_line_method: CrownLineMethod,
    crown_line_exit_distance: Option<f32>,
    crown_elevation_band: Option<f32>,
    min_particles_per_crown_cell: Option<u32>,
    release_fill_method: ReleaseFillMethod,
    release_slope_weight: Option<f32>,
    release_slope_peak: Option<f32>,
    release_slope_width: Option<f32>,
    release_slope_smoothing_length: Option<f32>,
    enable_particle_relaxation: bool,
}

/// Settings the stored initial particles were prepared with.
struct InitialState {
    settings: SimSettings,
    release: ReleaseConfig,
    /// Release thickness per cell as prepared, rescaled on the GPU together
    /// with the particle masses when the release thickness changes.
    release_areas: Vec<f32>,
    /// Factor the GPU release areas currently carry relative to
    /// `release_areas`.
    release_areas_scale: f32,
}

/// Earliest preparation stage a settings change invalidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PreparationStage {
    /// Only loop parameters changed: restore the stored particles.
    None,
    /// Density or release thickness changed: restore with rescaled masses.
    ParticleMass,
    /// The release cells may change: release areas and particle seeding.
    Release,
    /// The terrain analysis changes: full preparation.
    Terrain,
}

/// Data produced by [`Simulation::load_data`] and consumed by
/// [`Simulation::apply_data`]; splitting the two lets callers load data
/// concurrently with GPU initialization.
pub struct SimulationLoadResult {
    /// Simulation parameters resolved from the settings file and DEM.
    pub settings: SimSettings,
    /// Whether to track the center of mass during the run.
    pub enable_center_of_mass: bool,
    /// Whether freshly initialized particles are relaxed to the hexagonal
    /// packing spacing before the run.
    pub enable_particle_relaxation: bool,
    /// Loaded digital elevation model.
    pub dem: Dem,
    /// Region-of-interest (area that should be considered for release areas, e.g. avalanche outline) cell mask, `width * height` long.
    pub roi: Vec<bool>,
    /// Path the DEM was loaded from; empty when unknown.
    pub dem_path: String,
    /// Path of a release-area texture, if one was configured.
    pub release_areas_path: Option<String>,
    /// Fraction of the outline area to fill with release mass via crown-line
    /// estimation, if configured.
    pub release_area_fraction: Option<f32>,
    /// Method used to estimate the crown line when `release_area_fraction`
    /// is set; `None` falls back to flow routing.
    pub crown_line_method: Option<CrownLineMethod>,
    /// Exit distance threshold (fraction of the cell size) of the
    /// gradient-step crown line method; `None` keeps the default.
    pub crown_line_exit_distance: Option<f32>,
    /// Elevation-band filter share of the crown components; `None` keeps the
    /// default.
    pub crown_elevation_band: Option<f32>,
    /// Entry count threshold of the particle-simulation crown line method;
    /// `None` keeps the default.
    pub min_particles_per_crown_cell: Option<u32>,
    /// How the release cells below the crown line are ordered; `None` falls
    /// back to the geodesic fill.
    pub release_fill_method: Option<ReleaseFillMethod>,
    /// Blend weight of the slope-weighted fill; `None` keeps the default.
    pub release_slope_weight: Option<f32>,
    /// Peak (deg) of the release slope membership; `None` keeps the default.
    pub release_slope_peak: Option<f32>,
    /// Half width (deg) of the release slope membership; `None` keeps the
    /// default.
    pub release_slope_width: Option<f32>,
    /// DEM smoothing width (m) for the release slope; `None` keeps the
    /// default.
    pub release_slope_smoothing_length: Option<f32>,
    /// Number of compute steps dispatched per GPU submission. The OS might reset the GPU or interrupt the submission, if computing a batch takes too long.
    pub batch_compute_steps: Option<u32>,
    pub particle_sort_interval: Option<u32>,
    /// Destination for results written by [`Simulation::save`].
    pub output_path: Option<String>,
}

/// A single avalanche simulation: owns the GPU orchestrator, the loaded DEM,
/// the pipeline [`SimulationState`] and a CPU-side cache of GPU readbacks.
///
/// Typical flow: construct with [`Simulation::new`], load data with one of the
/// `create*` methods (or set a DEM directly with
/// [`Simulation::set_dem_with_bounds`]), then [`Simulation::prepare`] and
/// [`Simulation::run`]/[`Simulation::run_n_steps`]. Results are read back
/// lazily by the `fetch_*` methods and memoized in [`Simulation::gpu_cache`]
/// until the next pipeline stage invalidates them.
pub struct Simulation {
    orchestrator: ComputeOrchestrator,
    /// Physics and numerics parameters in effect for this run.
    pub settings: SimSettings,
    /// Whether the center-of-mass trajectory is tracked on the GPU.
    pub enable_center_of_mass: bool,
    /// Whether freshly initialized particles are relaxed to the hexagonal
    /// packing spacing with a soft sphere repulsion before the simulation
    /// starts. Measurement sims (e.g. crown line detection) keep the plain
    /// random initialization for comparable counts.
    pub enable_particle_relaxation: bool,
    /// Path the DEM was loaded from; empty when set programmatically.
    pub dem_path: String,
    /// The digital elevation model the simulation runs on.
    pub dem: Dem,
    /// Region-of-interest (avalanche outline) cell mask, one entry per cell.
    pub roi: Vec<bool>,
    roi_uploaded: bool,
    /// Destination of [`Simulation::save`] (Zarr store path or prefix).
    pub output_path: String,
    release_areas_path: Option<String>,
    release_area_fraction: Option<f32>,
    crown_line_method: CrownLineMethod,
    crown_line_exit_distance: Option<f32>,
    crown_elevation_band: Option<f32>,
    min_particles_per_crown_cell: Option<u32>,
    release_fill_method: ReleaseFillMethod,
    release_slope_weight: Option<f32>,
    release_slope_peak: Option<f32>,
    release_slope_width: Option<f32>,
    release_slope_smoothing_length: Option<f32>,
    release_areas_array: Option<Vec<f32>>,
    /// Crown line detected by release area estimation; empty unless
    /// `release_area_fraction` was set.
    pub crown_line: Vec<bool>,
    /// Crown candidates before component filtering (size and elevation
    /// band); empty unless `release_area_fraction` was set.
    pub crown_candidates: Vec<bool>,
    /// Potential release areas inside the outline (slope, elevation and
    /// roughness thresholds, computed on the GPU) from which the crown-line
    /// estimation selects the release cells; empty unless
    /// `release_area_fraction` was set.
    pub potential_release_areas: Vec<bool>,
    sim_info: SimInfo,
    number_particles: u32,
    state: SimulationState,
    /// CPU-side memoization of GPU readbacks, invalidated automatically as
    /// the simulation advances.
    pub gpu_cache: GpuCache,
    /// Avalanche deposit mask extracted by [`Simulation::post_process`].
    pub ava_mask: Vec<bool>,
    /// Settings of the particles kept by [`Simulation::store_initial_state`].
    initial_state: Option<InitialState>,
    #[cfg(not(target_arch = "wasm32"))]
    output: Option<data_processor::output::Output>,
}

impl Simulation {
    /// Creates an empty simulation on the default GPU adapter.
    pub async fn new() -> Result<Self> {
        Self::new_with_gpu(None).await
    }
    /// Creates an empty simulation, optionally pinning it to a GPU adapter by
    /// name (see `compute_core::list_devices`); `None` uses the default
    /// adapter.
    pub async fn new_with_gpu(gpu: Option<String>) -> Result<Self> {
        timer_new();
        let orchestrator = ComputeOrchestrator::new_with_gpu(gpu).await?;
        Ok(Self {
            orchestrator,
            settings: SimSettings::default(),
            enable_center_of_mass: true,
            enable_particle_relaxation: true,
            output_path: "avalanchers".to_string(),
            dem_path: String::new(),
            dem: Dem::default(),
            roi: Vec::new(),
            roi_uploaded: false,
            number_particles: 0,
            state: SimulationState::Uninitialized,
            gpu_cache: GpuCache::default(),
            sim_info: SimInfo::default(),
            release_areas_path: None,
            release_area_fraction: None,
            crown_line_method: CrownLineMethod::FlowRouting,
            crown_line_exit_distance: None,
            crown_elevation_band: None,
            min_particles_per_crown_cell: None,
            release_fill_method: ReleaseFillMethod::Geodesic,
            release_slope_weight: None,
            release_slope_peak: None,
            release_slope_width: None,
            release_slope_smoothing_length: None,
            release_areas_array: None,
            crown_line: Vec::new(),
            crown_candidates: Vec::new(),
            potential_release_areas: Vec::new(),
            ava_mask: Vec::new(),
            initial_state: None,
            #[cfg(not(target_arch = "wasm32"))]
            output: None,
        })
    }

    /// Creates an empty simulation and loads the data described by `settings`
    /// in parallel; equivalent to [`Self::new`] + [`Self::apply_data`].
    pub async fn new_with_settings(settings: Settings) -> Result<Self> {
        let (simulation, simulation_data) =
            futures::join!(Simulation::new(), Simulation::load_data(&settings),);

        let mut simulation = simulation?;
        let simulation_data = simulation_data?;

        simulation.apply_data(simulation_data);
        Ok(simulation)
    }

    /// Current position in the [`SimulationState`] lifecycle.
    pub fn get_state(&self) -> SimulationState {
        self.state
    }

    /// Resets simulation progress while preserving the loaded DEM and settings.
    pub fn reset(&mut self) {
        self.gpu_cache.reset_all();
        self.sim_info = SimInfo::default();
        self.crown_line = Vec::new();
        self.crown_candidates = Vec::new();
        self.potential_release_areas = Vec::new();
        self.initial_state = None;
        self.state = if self.dem.data1d.is_empty() {
            SimulationState::DemMissing
        } else {
            SimulationState::DemLoaded
        };
    }

    /// GPU device, queue and buffers backing the simulation. A renderer can bind these
    /// directly to visualise the simulation without copying data back to the CPU.
    pub fn orchestrator(&self) -> &ComputeOrchestrator {
        &self.orchestrator
    }

    /// Number of particles seeded from the release areas; fixed by
    /// [`Self::prepare`] (release cells × particles per cell).
    pub fn number_particles(&self) -> u32 {
        self.number_particles
    }

    /// Stable hash of the release-area array; part of [`Self::scenario_name`]
    /// so different release masks end up in different scenarios.
    pub fn release_hash(&self) -> u64 {
        hash_release_areas(self.release_areas_array.as_deref().unwrap_or_default())
    }

    /// Number of GPU→CPU readbacks issued so far; exposed so tests can verify
    /// that [`Self::gpu_cache`] actually avoids re-reads.
    pub fn get_gpu_cache_read_count(&self) -> usize {
        self.gpu_cache.read_count
    }

    /// Elevation below which particles are considered to have left the DEM
    /// (minimum DEM elevation minus a small margin).
    pub fn elevation_threshold(&self) -> f32 {
        self.sim_info.elevation_threshold
    }

    /// Resolves a [`Settings`] descriptor into [`SimulationLoadResult`]
    /// (settings, DEM, region of interest) without needing a simulation
    /// instance; pair the result with [`Self::apply_data`].
    pub async fn load_data(settings: &Settings) -> Result<SimulationLoadResult> {
        timer_checkpoint("Start create");
        #[cfg(not(target_arch = "wasm32"))]
        let (dem_path, release_areas_path) = data_processor::store_reader::resolve_paths(settings)?;
        #[cfg(target_arch = "wasm32")]
        let (dem_path, release_areas_path) = (
            settings.dem_path.clone(),
            settings.release_areas_path.clone(),
        );

        let (settings_result, dem_result, outline) =
            data_processor::create_sim_settings_and_dem(settings).await?;
        settings_result.validate().map_err(anyhow::Error::msg)?;

        timer_checkpoint("Load settings");

        Ok(SimulationLoadResult {
            settings: settings_result,
            enable_center_of_mass: settings.enable_center_of_mass.unwrap_or(true),
            enable_particle_relaxation: settings.enable_particle_relaxation.unwrap_or(true),
            dem: dem_result,
            roi: outline,
            batch_compute_steps: settings.batch_compute_steps,
            particle_sort_interval: settings.particle_sort_interval,
            dem_path: dem_path.unwrap_or_default(),
            release_areas_path,
            release_area_fraction: settings.release_area_fraction,
            crown_line_method: settings.crown_line_method,
            crown_line_exit_distance: settings.crown_line_exit_distance,
            crown_elevation_band: settings.crown_elevation_band,
            min_particles_per_crown_cell: settings.min_particles_per_crown_cell,
            release_fill_method: settings.release_fill_method,
            release_slope_weight: settings.release_slope_weight,
            release_slope_peak: settings.release_slope_peak,
            release_slope_width: settings.release_slope_width,
            release_slope_smoothing_length: settings.release_slope_smoothing_length,
            output_path: settings.output_path.clone(),
        })
    }

    /// Installs previously loaded data and resets all GPU state; the
    /// simulation ends in [`SimulationState::DemLoaded`], or
    /// [`SimulationState::DemMissing`] when the DEM is empty.
    pub fn apply_data(&mut self, data: SimulationLoadResult) {
        self.settings = data.settings;
        self.enable_center_of_mass = data.enable_center_of_mass;
        self.enable_particle_relaxation = data.enable_particle_relaxation;
        self.orchestrator
            .set_enable_center_of_mass(self.enable_center_of_mass);
        if let Some(batch_steps) = data.batch_compute_steps {
            self.orchestrator.batch_compute_steps = batch_steps;
        }
        if let Some(interval) = data.particle_sort_interval {
            self.orchestrator.particle_sort_interval = interval;
        }

        self.dem = data.dem;
        self.dem_path = data.dem_path;
        self.roi = data.roi;
        self.roi_uploaded = false;
        self.output_path = data
            .output_path
            .unwrap_or_else(|| "avalanchers.zarr".to_string());
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.output = None;
        }
        self.release_areas_path = data.release_areas_path;
        self.release_area_fraction = data.release_area_fraction;
        self.crown_line_method = data
            .crown_line_method
            .unwrap_or(CrownLineMethod::FlowRouting);
        self.crown_line_exit_distance = data.crown_line_exit_distance;
        self.crown_elevation_band = data.crown_elevation_band;
        self.min_particles_per_crown_cell = data.min_particles_per_crown_cell;
        self.release_fill_method = data
            .release_fill_method
            .unwrap_or(ReleaseFillMethod::Geodesic);
        self.release_slope_weight = data.release_slope_weight;
        self.release_slope_peak = data.release_slope_peak;
        self.release_slope_width = data.release_slope_width;
        self.release_slope_smoothing_length = data.release_slope_smoothing_length;
        self.crown_line = Vec::new();
        self.crown_candidates = Vec::new();
        self.potential_release_areas = Vec::new();
        self.initial_state = None;

        self.gpu_cache.reset_all();

        if self.dem.data1d.is_empty() {
            self.state = SimulationState::DemMissing;
        } else {
            self.state = SimulationState::DemLoaded;

            info!(
                "Updated simulation with DEM path: {:?}\nSettings: {:#?}",
                self.dem_path, self.settings
            );
        }

        timer_checkpoint("Simulation updated/created");
    }

    /// Loads and applies the data described by `settings`, replacing anything
    /// currently loaded ([`Self::load_data`] + [`Self::apply_data`]).
    pub async fn create(&mut self, settings: Settings) -> Result<()> {
        let data = Self::load_data(&settings).await?;
        self.apply_data(data);
        Ok(())
    }

    /// Creates the simulation from a DEM path with default settings.
    pub async fn create_default(&mut self, dem_path: &str) -> Result<()> {
        let settings = Settings {
            dem_path: Some(dem_path.to_string()),
            ..Settings::default()
        };
        self.create(settings).await?;
        Ok(())
    }

    /// Creates the simulation from a DEM and a release-area texture, with
    /// default settings otherwise.
    pub async fn create_default_with_release_areas(
        &mut self,
        dem_path: &str,
        release_areas_path: &str,
    ) -> Result<()> {
        let settings = Settings {
            dem_path: Some(dem_path.to_string()),
            release_areas_path: Some(release_areas_path.to_string()),
            ..Settings::default()
        };
        self.create(settings).await?;
        Ok(())
    }

    /// Creates the simulation from a `.png` DEM, deriving the release texture
    /// path from it by the example naming convention
    /// (`avaFoo.png` → `avaFooreleaseTexture.png`).
    pub async fn create_example(&mut self, dem_path: &str) -> Result<()> {
        let release_areas_path = dem_path.to_string().replace(".png", "releaseTexture.png");
        let settings = Settings {
            dem_path: Some(dem_path.to_string()),
            release_areas_path: Some(release_areas_path.to_string()),
            ..Settings::default()
        };
        self.create(settings).await?;
        Ok(())
    }

    /// Sets the DEM from raw cell data with bounds starting at the origin and
    /// unit map factor; identical to [`Self::set_dem`].
    pub fn set_dem_default(
        &mut self,
        dem_data: &[f32],
        width: usize,
        height: usize,
        cell_size: f32,
    ) -> Result<()> {
        self.set_dem_with_bounds(
            dem_data,
            width,
            height,
            cell_size,
            0.0,
            width as f32 * cell_size,
            0.0,
            height as f32 * cell_size,
            1.0,
        )
    }

    /// Sets the DEM from `width * height` row-major elevations with square
    /// `cell_size` cells and bounds starting at the coordinate origin.
    pub fn set_dem(
        &mut self,
        dem_data: &[f32],
        width: usize,
        height: usize,
        cell_size: f32,
    ) -> Result<()> {
        self.set_dem_with_bounds(
            dem_data,
            width,
            height,
            cell_size,
            0.0,
            width as f32 * cell_size,
            0.0,
            height as f32 * cell_size,
            1.0,
        )
    }

    pub fn set_dem_with_origin(
        &mut self,
        dem_data: &[f32],
        width: usize,
        height: usize,
        cell_size: f32,
        origin_x: f32,
        origin_y: f32,
    ) -> Result<()> {
        self.set_dem_with_bounds(
            dem_data,
            width,
            height,
            cell_size,
            origin_x,
            origin_x + width as f32 * cell_size,
            origin_y,
            origin_y + height as f32 * cell_size,
            1.0,
        )
    }

    /// Sets the DEM from raw data with full georeferencing: row-major
    /// `width * height` elevations, square `cell_size`, world-space bounds
    /// and `map_factor` scaling. Also derives minimum elevation and coordinate
    /// axes, and syncs [`Self::settings`] to the new grid.
    ///
    /// Errors if the data length does not match the dimensions, or if cell
    /// size, map factor or bounds are non-finite or degenerate.
    #[allow(clippy::too_many_arguments)]
    pub fn set_dem_with_bounds(
        &mut self,
        dem_data: &[f32],
        width: usize,
        height: usize,
        cell_size: f32,
        bounds_xmin: f32,
        bounds_xmax: f32,
        bounds_ymin: f32,
        bounds_ymax: f32,
        map_factor: f32,
    ) -> Result<()> {
        let expected_len = width
            .checked_mul(height)
            .ok_or_else(|| anyhow::anyhow!("DEM dimensions overflow usize"))?;
        if width == 0 || height == 0 {
            bail!("DEM width and height must be greater than zero");
        }
        if dem_data.len() != expected_len {
            bail!(
                "DEM array length ({}) does not match dimensions ({}x{}={})",
                dem_data.len(),
                width,
                height,
                expected_len
            );
        }
        if !cell_size.is_finite() || cell_size <= 0.0 {
            bail!("cell_size must be finite and greater than zero");
        }
        if !map_factor.is_finite() || map_factor <= 0.0 {
            bail!("map_factor must be finite and greater than zero");
        }
        if !bounds_xmin.is_finite()
            || !bounds_xmax.is_finite()
            || !bounds_ymin.is_finite()
            || !bounds_ymax.is_finite()
        {
            bail!("DEM bounds must be finite");
        }
        if bounds_xmin >= bounds_xmax {
            bail!(
                "xmin ({}) must be less than xmax ({})",
                bounds_xmin,
                bounds_xmax
            );
        }
        if bounds_ymin >= bounds_ymax {
            bail!(
                "ymin ({}) must be less than ymax ({})",
                bounds_ymin,
                bounds_ymax
            );
        }

        self.dem = Dem {
            data: to_2d(dem_data, width, height),
            minimum_elevation: Dem::calculate_minimum_elevation(dem_data),
            data1d: dem_data.to_vec(),
            width,
            height,
            cell_size,
            map_factor,
            bounds: Bounds {
                xmin: bounds_xmin,
                xmax: bounds_xmax,
                ymin: bounds_ymin,
                ymax: bounds_ymax,
            },
            x: cell_centers(bounds_xmin, bounds_xmax, width),
            y: cell_centers(bounds_ymin, bounds_ymax, height),
            source: String::new(),
            projection: String::new(),
        };
        self.settings.set_dem(&self.dem);
        self.initial_state = None;

        self.state = SimulationState::DemLoaded;
        info!(
            "Updated simulation with DEM path: {}\nSettings: {:#?}",
            self.dem_path, self.settings
        );
        Ok(())
    }

    /// Sets the per-cell release thickness directly, overriding any configured
    /// release-area path. The length must match the DEM, so the DEM has to be
    /// set first.
    pub fn set_release_areas(&mut self, release_areas: &[f32]) -> Result<()> {
        if release_areas.len() != self.dem.width * self.dem.height {
            bail!(
                "Release areas array length ({}) does not match DEM dimensions ({}x{}={}). You have to set the DEM first.",
                release_areas.len(),
                self.dem.width,
                self.dem.height,
                self.dem.width * self.dem.height
            );
        }
        self.release_areas_array = Some(release_areas.to_vec());
        self.release_areas_path = None;
        Ok(())
    }

    /// Sets the region of interest (avalanche outline) as a cell mask. Must
    /// match the DEM dimensions; cells outside the mask never become release
    /// area or deposit.
    pub fn set_roi(&mut self, roi: &[bool]) -> Result<()> {
        let expected_len = self
            .dem
            .width
            .checked_mul(self.dem.height)
            .ok_or_else(|| anyhow::anyhow!("DEM dimensions overflow usize"))?;
        if roi.len() != expected_len {
            bail!(
                "ROI length ({}) does not match DEM dimensions ({}x{}={}). You have to set the DEM first.",
                roi.len(),
                self.dem.width,
                self.dem.height,
                expected_len
            );
        }
        self.roi = roi.to_vec();
        self.roi_uploaded = false;
        Ok(())
    }

    /// Uploads the current region-of-interest mask to the GPU. GPU evaluation
    /// requires this method to have completed after the most recent
    /// [`Self::set_roi`].
    pub async fn upload_roi(&mut self) -> Result<()> {
        if self.roi.is_empty() {
            bail!("Cannot upload an empty region of interest");
        }
        let roi_words = Self::pack_roi_mask(&self.roi);
        self.orchestrator
            .write_buffer(BufferName::RegionOfInterest, &roi_words)
            .await?;
        self.roi_uploaded = true;
        Ok(())
    }

    /// Runs the pre-simulation stages in order: terrain analysis, release-area
    /// determination (see `load_release_areas` for the source priority) and
    /// particle initialization.
    pub async fn prepare(&mut self) -> Result<()> {
        self.analyze_terrain().await?;
        if !self.roi.is_empty() {
            self.upload_roi().await?;
        }
        let _ = self.load_release_areas().await?;
        self.initialize_particles().await?;
        Ok(())
    }

    /// Changes settings of a loaded simulation without reloading its data;
    /// unset fields keep their value. The change takes effect with the next
    /// [`Self::restore_initial_state`] or preparation. Input data (DEM,
    /// release area and outline files, output path) can only be changed with
    /// [`Self::create`].
    pub fn update_settings(&mut self, patch: &Settings) -> Result<()> {
        let data_keys = [
            ("dem_path", patch.dem_path.is_some()),
            ("release_areas_path", patch.release_areas_path.is_some()),
            ("outlines_path", patch.outlines_path.is_some()),
            ("outlines_padding", patch.outlines_padding.is_some()),
            ("output_path", patch.output_path.is_some()),
        ];
        if let Some((key, _)) = data_keys.iter().find(|(_, is_set)| *is_set) {
            bail!("`{key}` changes the input data; load it with create instead");
        }
        let mut settings = self.settings;
        settings.apply_patch(patch);
        settings.validate().map_err(anyhow::Error::msg)?;
        self.settings = settings;
        if let Some(steps) = patch.batch_compute_steps {
            self.orchestrator.batch_compute_steps = steps;
        }
        if let Some(interval) = patch.particle_sort_interval {
            self.orchestrator.particle_sort_interval = interval;
        }
        if let Some(enable) = patch.enable_center_of_mass {
            self.enable_center_of_mass = enable;
            self.orchestrator.set_enable_center_of_mass(enable);
        }
        if let Some(enable) = patch.enable_particle_relaxation {
            self.enable_particle_relaxation = enable;
        }
        if let Some(fraction) = patch.release_area_fraction {
            self.release_area_fraction = Some(fraction);
        }
        if let Some(method) = patch.crown_line_method {
            self.crown_line_method = method;
        }
        if let Some(distance) = patch.crown_line_exit_distance {
            self.crown_line_exit_distance = Some(distance);
        }
        if let Some(band) = patch.crown_elevation_band {
            self.crown_elevation_band = Some(band);
        }
        if let Some(count) = patch.min_particles_per_crown_cell {
            self.min_particles_per_crown_cell = Some(count);
        }
        if let Some(method) = patch.release_fill_method {
            self.release_fill_method = method;
        }
        if let Some(weight) = patch.release_slope_weight {
            self.release_slope_weight = Some(weight);
        }
        if let Some(peak) = patch.release_slope_peak {
            self.release_slope_peak = Some(peak);
        }
        if let Some(width) = patch.release_slope_width {
            self.release_slope_width = Some(width);
        }
        if let Some(length) = patch.release_slope_smoothing_length {
            self.release_slope_smoothing_length = Some(length);
        }
        Ok(())
    }

    fn release_config(&self) -> ReleaseConfig {
        let mut roi_hasher = DefaultHasher::new();
        self.roi.hash(&mut roi_hasher);
        ReleaseConfig {
            release_areas_path: self.release_areas_path.clone(),
            release_hash: self.release_hash(),
            roi_hash: roi_hasher.finish(),
            release_area_fraction: self.release_area_fraction,
            crown_line_method: self.crown_line_method,
            crown_line_exit_distance: self.crown_line_exit_distance,
            crown_elevation_band: self.crown_elevation_band,
            min_particles_per_crown_cell: self.min_particles_per_crown_cell,
            release_fill_method: self.release_fill_method,
            release_slope_weight: self.release_slope_weight,
            release_slope_peak: self.release_slope_peak,
            release_slope_width: self.release_slope_width,
            release_slope_smoothing_length: self.release_slope_smoothing_length,
            enable_particle_relaxation: self.enable_particle_relaxation,
        }
    }

    /// Whether the release thickness comes from `slab_thickness_factor`
    /// (computed or estimated release areas) rather than a file or array.
    fn release_thickness_from_settings(&self) -> bool {
        self.release_areas_path.is_none() && self.release_areas_array.is_none()
    }

    fn preparation_stage(&self, initial: &InitialState) -> PreparationStage {
        let (old, new) = (&initial.settings, &self.settings);
        if old.sim_model != new.sim_model
            || old.grid_shape_x != new.grid_shape_x
            || old.grid_shape_y != new.grid_shape_y
            || old.cell_size != new.cell_size
            || old.world_size_x != new.world_size_x
            || old.world_size_y != new.world_size_y
            || old.reference_surface_length != new.reference_surface_length
        {
            return PreparationStage::Terrain;
        }
        if old.min_slope_angle != new.min_slope_angle
            || old.max_slope_angle != new.max_slope_angle
            || old.release_min_elevation != new.release_min_elevation
            || old.release_max_elevation != new.release_max_elevation
            || old.roughness_threshold != new.roughness_threshold
            || old.released_particles_per_cell != new.released_particles_per_cell
            || initial.release != self.release_config()
        {
            return PreparationStage::Release;
        }
        if old.density != new.density || old.slab_thickness_factor != new.slab_thickness_factor {
            return PreparationStage::ParticleMass;
        }
        PreparationStage::None
    }

    /// Prepares the simulation if needed and keeps its initial particles on
    /// the GPU, so that [`Self::restore_initial_state`] can start further
    /// runs (e.g. a back calibration with changed friction parameters)
    /// without repeating terrain analysis, release areas and particle
    /// seeding. Only the particle positions and masses are kept; everything
    /// else a run changes is reset on restore.
    pub async fn store_initial_state(&mut self) -> Result<()> {
        if self.state < SimulationState::ParticlesInitialized {
            self.prepare().await?;
        }
        if self.state != SimulationState::ParticlesInitialized {
            bail!(
                "The initial state must be stored before the simulation runs; call prepare or restore_initial_state first"
            );
        }
        self.orchestrator
            .store_initial_particles(self.settings.sim_model, self.number_particles)
            .await?;
        let release_areas = self
            .orchestrator
            .read_buffer::<f32>(BufferName::ReleaseAreas)
            .await?;
        self.initial_state = Some(InitialState {
            settings: self.settings,
            release: self.release_config(),
            release_areas,
            release_areas_scale: 1.0,
        });
        Ok(())
    }

    /// Resets the simulation to the state kept by
    /// [`Self::store_initial_state`], ready for the next [`Self::run`].
    /// Settings changed since then (see [`Self::update_settings`]) take
    /// effect; only the preparation stages they affect run again:
    ///
    /// - model, grid or reference surface: full preparation,
    /// - release thresholds, particles per cell or release configuration:
    ///   release areas and particle seeding,
    /// - density or release thickness: the particle masses are rescaled,
    /// - anything else: the stored particles are restored as they are.
    pub async fn restore_initial_state(&mut self) -> Result<()> {
        let Some(initial) = &self.initial_state else {
            bail!("No initial state stored; call store_initial_state first");
        };
        let stage = self.preparation_stage(initial);
        match stage {
            PreparationStage::Terrain => {
                info!("Settings change the terrain analysis; preparing again");
                self.state = SimulationState::DemLoaded;
                self.store_initial_state().await?;
            }
            PreparationStage::Release => {
                info!("Settings change the release; seeding the particles again");
                self.state = SimulationState::TerrainAnalyzed;
                // the release shaders count with atomicAdd
                self.orchestrator
                    .write_buffer(BufferName::AtomicValues, &[AtomicValues::default()])
                    .await?;
                self.load_release_areas().await?;
                self.initialize_particles().await?;
                self.store_initial_state().await?;
            }
            PreparationStage::ParticleMass | PreparationStage::None => {}
        }

        let thickness_from_settings = self.release_thickness_from_settings();
        let Some(initial) = &mut self.initial_state else {
            bail!("No initial state stored; call store_initial_state first");
        };
        // the ratios are taken against the stored settings, so repeated
        // restores do not drift
        let thickness_scale = if thickness_from_settings {
            self.settings.slab_thickness_factor / initial.settings.slab_thickness_factor
        } else {
            1.0
        };
        let mass_scale = self.settings.density / initial.settings.density * thickness_scale;
        if thickness_scale != initial.release_areas_scale {
            let release_areas: Vec<f32> = initial
                .release_areas
                .iter()
                .map(|thickness| thickness * thickness_scale)
                .collect();
            self.orchestrator
                .write_buffer(BufferName::ReleaseAreas, &release_areas)
                .await?;
            initial.release_areas_scale = thickness_scale;
        }
        self.orchestrator
            .restore_initial_particles(mass_scale, thickness_scale)
            .await?;

        self.gpu_cache.reset_simulation_result();
        self.gpu_cache.release_areas = None;
        self.sim_info = SimInfo::default();
        self.ava_mask = Vec::new();
        self.state = SimulationState::ParticlesInitialized;
        Ok(())
    }

    /// Whether [`Self::store_initial_state`] holds particles to restore.
    pub fn has_initial_state(&self) -> bool {
        self.initial_state.is_some()
    }

    /// Advances the simulation by up to `steps` steps, automatically running
    /// [`Self::prepare`] and setting up the GPU pipeline for the configured
    /// [`SimModel`] on first call. Returns the updated [`SimInfo`].
    ///
    /// The request is clamped to the remaining budget up to `max_steps`;
    /// reaching it (or the GPU signalling `SIM_STOPPED`) moves the state to
    /// [`SimulationState::Finished`], after which calls become no-ops.
    pub async fn run_n_steps(&mut self, steps: u32) -> Result<SimInfo> {
        if self.state >= SimulationState::Finished {
            return Ok(self.sim_info);
        }

        if self.state < SimulationState::ParticlesInitialized {
            self.prepare().await?;
        }

        if self.state != SimulationState::Running {
            match self.settings.sim_model {
                model
                    if model == SimModel::TerrainFollowing.as_int()
                        || model == SimModel::SlidingBlock.as_int() =>
                {
                    self.orchestrator
                        .prepare_compute_particles(
                            &self.settings,
                            self.number_particles,
                            self.dem.minimum_elevation,
                        )
                        .await?;
                }
                model if model == SimModel::Curvilinear.as_int() => {
                    self.orchestrator
                        .prepare_mpm(
                            &self.settings,
                            self.number_particles,
                            self.dem.minimum_elevation,
                        )
                        .await?;
                }
                model if model == SimModel::MpmDaC.as_int() => {
                    self.orchestrator
                        .prepare_mpmdac(
                            &self.settings,
                            self.number_particles,
                            self.dem.minimum_elevation,
                        )
                        .await?;
                }
                _ => bail!("Unsupported simulation model: {}", self.settings.sim_model),
            }
            self.sim_info = self.fetch_sim_info().await?;
            self.state = SimulationState::Running;
        }

        let remaining_steps = self
            .settings
            .max_steps
            .saturating_sub(self.sim_info.timestep);
        if remaining_steps == 0 {
            self.state = SimulationState::Finished;
            return Ok(self.sim_info);
        }
        let steps = steps.min(remaining_steps);
        if steps == 0 {
            return Ok(self.sim_info);
        }

        self.gpu_cache.reset_simulation_result();
        self.sim_info = match self.settings.sim_model {
            model if model == SimModel::TerrainFollowing.as_int() => {
                self.orchestrator.step_terrain_following(steps).await?
            }
            model if model == SimModel::Curvilinear.as_int() => {
                self.orchestrator.step_curvilinear(steps).await?
            }
            model if model == SimModel::MpmDaC.as_int() => {
                self.orchestrator.step_mpmdac(steps).await?
            }
            model if model == SimModel::SlidingBlock.as_int() => {
                self.orchestrator.step_sliding_block(steps).await?
            }
            _ => bail!("Unsupported simulation model: {}", self.settings.sim_model),
        };

        if self.sim_info.timestep >= self.settings.max_steps
            || self
                .sim_info
                .parsed_flags()
                .contains(SimInfoFlags::SIM_STOPPED)
        {
            self.state = SimulationState::Finished;
        }

        Ok(self.sim_info)
    }

    /// Runs the whole pipeline to completion in one call: terrain analysis,
    /// release areas, particle initialization and the full simulation, ending
    /// in [`SimulationState::Finished`].
    ///
    /// Fails when the release areas contain no cells (nothing to simulate).
    ///
    /// Preparation is skipped when the particles are already initialized, by
    /// [`Self::prepare`] or [`Self::restore_initial_state`].
    pub async fn run(&mut self) -> Result<()> {
        if self.state != SimulationState::ParticlesInitialized {
            self.analyze_terrain().await?;
            timer_checkpoint("Terrain analyzed");
            if !self.roi.is_empty() {
                self.upload_roi().await?;
            }
            let _ = self.load_release_areas().await?;
            timer_checkpoint("Release areas loaded");
            if self.number_particles == 0 {
                bail!("No particles to simulate! Check if release areas are correctly defined.");
            }
            self.initialize_particles().await?;
            timer_checkpoint("Particles initialized");
        }
        self.compute_particles().await?;
        self.sim_info = self.fetch_sim_info().await?;
        self.state = SimulationState::Finished;

        timer_checkpoint("Simulation finished");
        info!("{}", timer_get_summary());
        Ok(())
    }

    /// Extracts the avalanche mask from the results: the peak flow thickness
    /// is thresholded (`peak_flow_thickness_threshold` setting) and reduced to
    /// its biggest connected blob, becoming [`Self::ava_mask`]; peak velocity
    /// is masked to the same region. Requires
    /// [`SimulationState::Finished`].
    pub async fn post_process(&mut self) -> Result<()> {
        if self.state < SimulationState::Finished {
            bail!("Simulation must be finished before post-processing results");
        }
        let threshold = self.settings.peak_flow_thickness_threshold;
        let dem_width = self.dem.width;
        let (peak_flow_thickness, ava_mask) = mask_threshold_and_biggest_blob(
            self.fetch_peak_flow_thickness().await?,
            dem_width,
            threshold,
        );
        self.ava_mask = ava_mask;
        self.gpu_cache.peak_flow_thickness = Some(peak_flow_thickness);
        self.fetch_peak_velocity().await?;
        mask_in_place(
            self.gpu_cache.peak_velocity.as_mut().unwrap(),
            &self.ava_mask,
        );
        self.state = SimulationState::PostProcessed;
        Ok(())
    }

    /// Compares the simulated avalanche against the region of interest and
    /// moves the state to [`SimulationState::Evaluated`]. Requires
    /// [`SimulationState::PostProcessed`].
    ///
    /// Returns the CPU-derived overlap, runout, and peak velocity metrics.
    /// GPU-derived chamfer metrics can be added with [`Self::evaluate_gpu`].
    /// Release cells are left out of the overlap, as in the GPU evaluation.
    pub async fn evaluate(&mut self) -> Result<MassMovementEvaluation> {
        if self.state < SimulationState::PostProcessed {
            bail!("Simulation must be post-processed before evaluation");
        }
        let release_areas = self.fetch_release_areas().await?.to_vec();
        let outside_release = |mask: &[bool]| -> Vec<bool> {
            mask.iter()
                .zip(&release_areas)
                .map(|(&set, &thickness)| {
                    set && thickness <= compute_core::RELEASE_THICKNESS_THRESHOLD
                })
                .collect()
        };
        let overlap = compute_core::evaluation::evaluate_mass_movement_area(
            &outside_release(&self.roi),
            &outside_release(&self.ava_mask),
        )
        .map_err(|e| anyhow::anyhow!("Mass movement area evaluation failed: {e:?}"))?;
        let (horizontal_distance, vertical_drop) = self
            .dem
            .get_elevation_extrema_distance_and_drop(&self.ava_mask)
            .unwrap_or((0.0, 0.0));
        let (horizontal_distance_ref, vertical_drop_ref) = self
            .dem
            .get_elevation_extrema_distance_and_drop(&self.roi)
            .unwrap_or((0.0, 0.0));
        let velocities = self.fetch_peak_velocity().await?;
        let peak_velocity = velocities.iter().copied().reduce(f32::max).unwrap_or(0.0);
        self.state = SimulationState::Evaluated;
        Ok(overlap.with_runout(
            horizontal_distance as f64,
            vertical_drop as f64,
            horizontal_distance_ref as f64,
            vertical_drop_ref as f64,
            peak_velocity as f64,
        ))
    }

    /// Runs the GPU evaluation metrics: overlap components, diagonal-normalized
    /// chamfer distance, and the 3D beeline distance between the highest and
    /// lowest avalanche point.
    /// Packs the per-cell ROI mask into u32 words (least significant bit =
    /// first cell), the layout the evaluation shaders expect.
    fn pack_roi_mask(roi: &[bool]) -> Vec<u32> {
        roi.chunks(32)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |word, (bit, &set)| word | ((set as u32) << bit))
            })
            .collect()
    }

    pub async fn evaluate_gpu(&mut self) -> Result<MassMovementEvaluation> {
        if !self.roi_uploaded {
            bail!("Region of interest must be uploaded before GPU evaluation");
        }
        self.orchestrator.evaluate_gpu(&self.settings).await
    }

    /// Native only: writes the results to the Zarr store at `path`, switching
    /// away from [`Self::output_path`] if it differs.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn save_with_path(&mut self, path: &str) -> Result<()> {
        if self.output_path != path {
            self.output = None;
            self.output_path = path.to_string();
        }
        self.save().await
    }

    /// Name of the site this simulation writes to, derived from the DEM.
    pub fn site_name(&self) -> String {
        let source = if !self.dem_path.is_empty() {
            self.dem_path.as_str()
        } else {
            self.dem.source.as_str()
        };
        // The source of fetched tiles is a multi-line attribution, not a path.
        let stem = store_path_stem(source)
            .or_else(|| std::path::Path::new(source).file_stem()?.to_str())
            .filter(|s| !s.is_empty() && !s.contains('\n'))
            .unwrap_or("default-site")
            .replace("_", "-");
        format!("{}_{:x}", stem, self.dem.calculate_hash())
    }

    /// Name of the scenario, derived from the release areas it was run with.
    pub fn scenario_name(&self, release_areas: &[f32]) -> String {
        let base = match &self.release_areas_path {
            Some(path) => store_path_stem(path)
                .or_else(|| std::path::Path::new(path).file_stem()?.to_str())
                .unwrap_or(path)
                .to_string()
                .replace("_", "-"),
            None => format!(
                "calculated-elev{}-minslope{}-maxslope{}-rough{}",
                self.settings.release_min_elevation,
                self.settings.min_slope_angle,
                self.settings.max_slope_angle,
                self.settings.roughness_threshold,
            ),
        };
        format!("{}_{:x}", base, hash_release_areas(release_areas))
    }

    /// Gathers everything the result stores keep of the finished run.
    async fn collect_run(&mut self) -> Result<RunRecord> {
        if self.state < SimulationState::Finished {
            bail!("Run the simulation before saving results");
        }
        let sim_info = self.fetch_sim_info().await?;
        let release_volume_m3 = self.get_total_volume().await?;
        let peak_flow_velocity = self.fetch_peak_velocity().await?.clone();
        let peak_flow_thickness = self.fetch_peak_flow_thickness().await?.to_vec();

        let origin_x = self.dem.bounds.xmin;
        let origin_y = self.dem.bounds.ymin;
        let center_of_mass = self.fetch_center_of_mass().await?;
        let time = center_of_mass.iter().map(|result| result.time).collect();
        let (center_of_mass_x, center_of_mass_y, travel_length, travel_angle) =
            trajectory_summary(center_of_mass, origin_x, origin_y);

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(f64::NAN, |elapsed| elapsed.as_secs_f64());

        Ok(RunRecord {
            peak_flow_velocity,
            peak_flow_thickness,
            center_of_mass_x,
            center_of_mass_y,
            time,
            release_volume_m3,
            travel_length,
            travel_angle,
            final_timestep: sim_info.timestep,
            simulated_time: sim_info.elapsed_time,
            number_particles: sim_info.number_particles,
            stop_flags: sim_info.flags,
            created_at,
            settings: self.settings,
        })
    }

    /// Builds an in-memory Zarr store of the current results.
    ///
    /// Used by the browser, which cannot write files itself; the caller is
    /// responsible for persisting the returned entries.
    #[cfg(target_arch = "wasm32")]
    pub async fn export_zarr_entries(&mut self) -> Result<Vec<ZarrEntry>> {
        let run = self.collect_run().await?;
        let release_areas = self.fetch_release_areas().await?.to_vec();
        Ok(data_processor::zarr_writer::build_result_store(
            &self.site_name(),
            &self.scenario_name(&release_areas),
            &self.dem,
            &release_areas,
            &run,
        ))
    }

    /// Native only: appends this run to the Zarr store at
    /// [`Self::output_path`], creating the site (derived from the DEM, see
    /// [`Self::site_name`]) and scenario (derived from the release areas, see
    /// [`Self::scenario_name`]) entries as needed.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn save(&mut self) -> Result<()> {
        let run = self.collect_run().await?;
        let release_areas = self.fetch_release_areas().await?.to_vec();
        timer_checkpoint("Peak data fetched");

        let site_name = self.site_name();
        let scenario_name = self.scenario_name(&release_areas);
        if self.output.is_none() {
            self.output = Some(data_processor::output::Output::new(&self.output_path)?);
        }
        let output = self.output.as_ref().unwrap();
        output.add_run(&site_name, &scenario_name, &self.dem, &release_areas, &run)?;
        Ok(())
    }

    /// Native only: writes the grid `layer` (`peak_velocity`,
    /// `peak_flow_thickness` or `release_areas`) to `path` as an ESRI ASCII grid
    /// georeferenced by the DEM.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn write_asc(&mut self, layer: &str, path: &str) -> Result<()> {
        let data = match layer {
            "peak_velocity" => self.fetch_peak_velocity().await?.clone(),
            "peak_flow_thickness" => self.fetch_peak_flow_thickness().await?.to_vec(),
            "release_areas" => self.fetch_release_areas().await?.to_vec(),
            _ => bail!(
                "Unknown layer \"{layer}\", expected peak_velocity, peak_flow_thickness or release_areas"
            ),
        };
        data_processor::EsriGrid::from_dem_layer(&self.dem, &data)?
            .to_file(path)
            .map_err(anyhow::Error::msg)
    }

    /// Reads the run info (timestep, dt, stop flags, ...) from the GPU and
    /// stores it in `self.sim_info`.
    pub async fn fetch_sim_info(&mut self) -> Result<SimInfo> {
        self.sim_info = self
            .orchestrator
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await?
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("SimInfo buffer was empty"))?;
        Ok(self.sim_info)
    }

    /// Reads the GPU atomic counters.
    pub async fn fetch_atomic_values(&mut self) -> Result<AtomicValues> {
        let atomic_values = self
            .orchestrator
            .read_buffer::<AtomicValues>(BufferName::AtomicValues)
            .await?
            .first()
            .cloned()
            .unwrap_or_default();
        Ok(atomic_values)
    }

    /// Runs the terrain analysis shaders (slope, aspect, curvature, terrain
    /// geometry) and invalidates the whole result cache.
    async fn analyze_terrain(&mut self) -> Result<()> {
        if self.state < SimulationState::DemLoaded {
            bail!("DEM and settings must be loaded before running normals shader");
        }
        self.gpu_cache.reset_all();
        self.orchestrator
            .run_analyze_terrain(&self.settings, &self.dem)
            .await?;
        self.state = SimulationState::TerrainAnalyzed;
        Ok(())
    }

    /// Detects the crown line by running a GPU particle simulation on a copy
    /// of the DEM where the outline is masked out with NaN: particles are
    /// released everywhere outside the outline (minus a one-cell ring, whose
    /// spawn jitter could seed particles inside the mask), slide downslope
    /// with the plain particle model (no curvature, particle interaction,
    /// earth pressure or entrainment) and stop with `out_of_dem_data` set
    /// where the bilinear terrain sampling first blends in the masked cells -
    /// at the latest on the boundary itself. Each stopped particle is
    /// attributed to the outline cell its velocity points into; outline cells
    /// entered by at least `config.min_particles_per_crown_cell` particles
    /// become crown candidates, and candidate components are then filtered
    /// exactly like in the flow-routing method.
    ///
    /// Compared to D8 flow routing, the particles add inertia, so terrain that
    /// only reaches the outline with momentum (through a flat approach or a
    /// counter-slope shoulder) is still detected.
    async fn detect_crown_line_by_particle_simulation(
        &self,
        config: &release_estimation::ReleaseEstimationConfig,
        terrain_metrics: &release_estimation::TerrainMetrics<'_>,
    ) -> Result<release_estimation::CrownDetection> {
        /// Particles per release cell for the detection simulation. Four
        /// entries per crown cell on planar inflow keep the entry counts well
        /// above the threshold while keeping the total particle count
        /// manageable on large tiles.
        const DETECTION_PARTICLES_PER_CELL: u32 = 4;

        // Mask the outline out of the DEM and release everything except a
        // one-cell ring around the outline: a particle crossing into the
        // outline samples a NaN normal, is flagged out_of_dem_data and stops
        // on that cell. The ring stays unreleased because particles spawn
        // with +-0.5 cell jitter - ring cells would seed particles inside the
        // masked outline, where they would count as entries without ever
        // flowing in.
        let number_cells = self.dem.width.saturating_mul(self.dem.height);
        let mut masked_dem = self.dem.data1d.clone();
        let mut detection_release = vec![0.0f32; number_cells];
        for idx in 0..number_cells {
            if self.roi[idx] {
                masked_dem[idx] = f32::NAN;
                continue;
            }
            let adjacent_to_outline = release_estimation::any_neighbor_is(
                idx,
                self.dem.width,
                self.dem.height,
                &self.roi,
            );
            if !adjacent_to_outline {
                detection_release[idx] = 1.0;
            }
        }

        let mut settings = Settings::default();
        // independent sliding blocks without grid coupling trace the flow paths
        settings.sim_model = Some(SimModel::SlidingBlock);
        settings.max_steps = Some(self.settings.max_steps);
        settings.released_particles_per_cell = Some(DETECTION_PARTICLES_PER_CELL);
        settings.enable_curvature = Some(false);
        settings.enable_particle_interaction = Some(false);
        settings.enable_earth_pressure_coefficient = Some(false);
        settings.enable_entrainment = Some(false);

        let mut detection_sim = Simulation::new().await?;
        detection_sim.create(settings).await?;
        // keep the plain random particle initialization so the entry counts
        // stay comparable with the tuned crown line thresholds
        detection_sim.enable_particle_relaxation = false;
        detection_sim.set_dem(
            &masked_dem,
            self.dem.width,
            self.dem.height,
            self.dem.cell_size,
        )?;
        detection_sim.set_roi(&vec![true; masked_dem.len()])?;
        detection_sim.set_release_areas(&detection_release)?;
        detection_sim.run().await?;

        let sim_info = detection_sim.fetch_sim_info().await?;
        if !sim_info.parsed_flags().intersects(
            SimInfoFlags::ALL_PARTICLES_STOPPED
                | SimInfoFlags::AT_REST
                | SimInfoFlags::LOW_MOMENTUM,
        ) {
            warn!(
                "crown detection simulation stopped at max_steps; entry counts may be incomplete"
            );
        }

        let positions = detection_sim.fetch_particles_position().await?.clone();
        let velocities = detection_sim.fetch_particles_velocity().await?.clone();
        let states = detection_sim.fetch_particles_state().await?.clone();
        let mut counts = vec![0u32; number_cells];
        let mut entered = 0usize;
        for ((state, position), velocity) in
            states.iter().zip(positions.iter()).zip(velocities.iter())
        {
            if !state.out_of_dem_data {
                continue;
            }
            // the NaN normals reach past the unreleased ring, so particles
            // spawned next to it stop before they move; they would have
            // started downslope
            let mut direction = *velocity;
            if direction == [0.0, 0.0] {
                let cell_x = (position[0] / self.dem.cell_size).floor();
                let cell_y = (position[1] / self.dem.cell_size).floor();
                if cell_x >= 0.0
                    && cell_y >= 0.0
                    && (cell_x as usize) < self.dem.width
                    && (cell_y as usize) < self.dem.height
                {
                    let idx = cell_y as usize * self.dem.width + cell_x as usize;
                    direction = [
                        -terrain_metrics.gradient_x[idx],
                        -terrain_metrics.gradient_y[idx],
                    ];
                }
            }
            if let Some(idx) = release_estimation::entry_cell(
                *position,
                direction,
                &self.roi,
                self.dem.width,
                self.dem.height,
                self.dem.cell_size,
            ) {
                counts[idx] += 1;
                entered += 1;
            }
        }
        if entered == 0 {
            bail!("crown detection simulation: no particles reached the outline");
        }
        info!(
            "crown detection simulation: {entered} of {} particles entered the outline",
            positions.len()
        );

        release_estimation::crown_line_from_particle_counts_with_terrain_metrics(
            &counts,
            &self.dem,
            &self.roi,
            config,
            terrain_metrics,
        )
    }

    /// Determines the release areas and returns the number of release cells.
    ///
    /// Sources are tried in priority order: a release-area file
    /// (`release_areas_path`), a directly set array
    /// ([`Self::set_release_areas`]), crown-line estimation from the outline
    /// (`release_area_fraction`, also fills [`Self::crown_line`]), and finally
    /// automatic computation from roughness and slope thresholds on the GPU.
    /// Fixes [`Self::number_particles`] and moves the state to
    /// [`SimulationState::ReleaseAreasComputed`].
    async fn load_release_areas(&mut self) -> Result<u32> {
        if self.state < SimulationState::TerrainAnalyzed {
            bail!("Terrain must be analyzed before loading release areas");
        }
        self.orchestrator
            .write_buffer(BufferName::SimSettings, self.settings.as_bytes())
            .await?;

        self.gpu_cache.release_areas = None;
        self.gpu_cache.reset_simulation_result();
        let number_release_cells = match &self.release_areas_path {
            Some(path) => {
                info!("Loading release areas from path: {}", path);
                let data = data_processor::load_release_areas(path)
                    .await
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                self.validate_release_areas_len(&data)?;
                self.orchestrator
                    .write_buffer(BufferName::ReleaseAreas, &data)
                    .await?;
                data.iter()
                    .filter(|&&x| x > compute_core::RELEASE_THICKNESS_THRESHOLD)
                    .count() as u32
            }
            None => match &self.release_areas_array {
                Some(data) => {
                    info!("Loading release areas from provided array");
                    self.validate_release_areas_len(data)?;
                    self.orchestrator
                        .write_buffer(BufferName::ReleaseAreas, data)
                        .await?;
                    data.iter()
                        .filter(|&&x| x > compute_core::RELEASE_THICKNESS_THRESHOLD)
                        .count() as u32
                }
                None => match self.release_area_fraction {
                    Some(fraction) => {
                        info!(
                            "Estimating release areas from crown line (fraction {fraction:.2}, method {}, fill {})",
                            self.crown_line_method, self.release_fill_method
                        );
                        let defaults = release_estimation::ReleaseEstimationConfig::default();
                        let config = release_estimation::ReleaseEstimationConfig {
                            fraction,
                            slab_thickness: self.settings.slab_thickness_factor,
                            crown_exit_distance: self
                                .crown_line_exit_distance
                                .unwrap_or(defaults.crown_exit_distance),
                            crown_elevation_band: self
                                .crown_elevation_band
                                .unwrap_or(defaults.crown_elevation_band),
                            min_particles_per_crown_cell: self
                                .min_particles_per_crown_cell
                                .unwrap_or(defaults.min_particles_per_crown_cell),
                            expected_crown_slope_range: (
                                self.settings.min_slope_angle,
                                self.settings.max_slope_angle,
                            ),
                            min_crown_slope_deg: self.settings.min_slope_angle,
                            fill_method: self.release_fill_method,
                            slope_weight: self
                                .release_slope_weight
                                .unwrap_or(defaults.slope_weight),
                            slope_peak_deg: self
                                .release_slope_peak
                                .unwrap_or(defaults.slope_peak_deg),
                            slope_width_deg: self
                                .release_slope_width
                                .unwrap_or(defaults.slope_width_deg),
                            slope_smoothing_length: self
                                .release_slope_smoothing_length
                                .unwrap_or(defaults.slope_smoothing_length),
                            ..defaults
                        };
                        // the threshold classification on the GPU marks every
                        // outline cell that could release; the crown line then
                        // picks the cells that actually did
                        self.orchestrator
                            .run_compute_roughness(&self.settings)
                            .await?;
                        let number_potential = self
                            .orchestrator
                            .run_compute_release_areas(&self.settings, &self.roi)
                            .await?;
                        let potential: Vec<bool> = self
                            .orchestrator
                            .read_buffer::<f32>(BufferName::ReleaseAreas)
                            .await?
                            .iter()
                            .map(|&thickness| thickness > compute_core::RELEASE_THICKNESS_THRESHOLD)
                            .collect();
                        let slope_angles = self.fetch_slope_angle().await?.to_vec();
                        let (gradient_x, gradient_y) = self.terrain_gradient().await?;
                        let terrain_metrics = release_estimation::TerrainMetrics {
                            slope_angles: &slope_angles,
                            gradient_x: &gradient_x,
                            gradient_y: &gradient_y,
                        };
                        info!(
                            "Potential release areas: {number_potential} of {} outline cells",
                            self.roi.iter().filter(|&&inside| inside).count()
                        );
                        let detection = match self.crown_line_method {
                            CrownLineMethod::FlowRouting => {
                                release_estimation::detect_crown_line_with_terrain_metrics(
                                    &self.dem,
                                    &self.roi,
                                    &config,
                                    &terrain_metrics,
                                )?
                            }
                            CrownLineMethod::ParticleSimulation => {
                                // Box::pin breaks the future type recursion:
                                // run() -> load_release_areas() -> here -> run()
                                // (the nested simulation is a separate instance,
                                // so the indirection is purely a type-level fix)
                                Box::pin(self.detect_crown_line_by_particle_simulation(
                                    &config,
                                    &terrain_metrics,
                                ))
                                .await?
                            }
                            CrownLineMethod::GradientStep => {
                                release_estimation::detect_crown_line_by_gradient_step_with_terrain_metrics(
                                    &self.dem,
                                    &self.roi,
                                    &config,
                                    &terrain_metrics,
                                )?
                            }
                        };
                        let estimate = release_estimation::estimate_release_areas_with_crown_and_terrain_metrics(
                            &self.dem,
                            &self.roi,
                            Some(&potential),
                            &detection,
                            &config,
                            &terrain_metrics,
                        )?;
                        self.crown_line = estimate.crown_line;
                        self.crown_candidates = estimate.crown_candidates;
                        self.potential_release_areas = potential;
                        self.orchestrator
                            .write_buffer(BufferName::ReleaseAreas, &estimate.release_areas)
                            .await?;
                        estimate.number_release_cells as u32
                    }
                    None => {
                        info!("Computing release areas from DEM");
                        self.orchestrator
                            .run_compute_roughness(&self.settings)
                            .await?;
                        self.orchestrator
                            .run_compute_release_areas(&self.settings, &self.roi)
                            .await?
                    }
                },
            },
        };
        self.number_particles = checked_particle_count(
            number_release_cells,
            self.settings.released_particles_per_cell,
        )?;
        self.state = SimulationState::ReleaseAreasComputed;
        info!(
            "Number of release cells: {} of {} ({:.1}%)",
            number_release_cells,
            self.dem.width * self.dem.height,
            (number_release_cells as f64 / (self.dem.width * self.dem.height) as f64 * 100.0)
        );
        Ok(number_release_cells)
    }

    fn validate_release_areas_len(&self, release_areas: &[f32]) -> Result<()> {
        let expected_len = self
            .dem
            .width
            .checked_mul(self.dem.height)
            .ok_or_else(|| anyhow::anyhow!("DEM dimensions overflow usize"))?;
        if release_areas.len() != expected_len {
            bail!(
                "Release areas array length ({}) does not match DEM dimensions ({}x{}={})",
                release_areas.len(),
                self.dem.width,
                self.dem.height,
                expected_len
            );
        }
        Ok(())
    }

    /// Seeds the particles inside the release areas on the GPU; requires the
    /// release areas to be computed and at least one release cell.
    async fn initialize_particles(&mut self) -> Result<()> {
        if self.state < SimulationState::ReleaseAreasComputed {
            bail!("Release areas must be computed before initializing particles");
        }
        if self.number_particles == 0 {
            bail!("No particles to initialize! Check if release areas are correctly defined.");
        }
        self.gpu_cache.reset_simulation_result();
        // set parameters that depend on the number of particles
        self.orchestrator
            .run_initialize_particles(
                &self.settings,
                self.number_particles,
                self.enable_particle_relaxation,
            )
            .await?;
        self.state = SimulationState::ParticlesInitialized;
        Ok(())
    }

    /// Advances the particle simulation. Requires [`Simulation::prepare`] first, and unlike
    /// [`Simulation::run`] it leaves the existing GPU buffers in place so anything already
    /// bound to them stays valid.
    pub async fn compute_particles(&mut self) -> Result<()> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Particles must be initialized before running particle simulation");
        }
        self.gpu_cache.reset_simulation_result();
        self.orchestrator
            .run_sim(
                &self.settings,
                self.number_particles,
                self.dem.minimum_elevation,
            )
            .await?;
        self.sim_info = self.fetch_sim_info().await?;
        self.state = SimulationState::Finished;
        info!(
            "Allocated GPU Memory: {:.1} MB",
            self.orchestrator.resources.get_total_allocated_memory_mb()
        );
        Ok(())
    }

    async fn get_texture_data<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<TextureRgba<T>> {
        Ok(TextureRgba::from(
            self.orchestrator.read_texture(name).await?,
        ))
    }

    /// Per-cell terrain roughness from terrain analysis.
    pub async fn fetch_roughness(&mut self) -> Result<&Vec<f32>> {
        if self.state < SimulationState::ReleaseAreasComputed {
            bail!("Release areas must be computed before reading roughness texture");
        }
        if self.gpu_cache.roughness.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.roughness =
                Some(self.orchestrator.read_buffer(BufferName::Roughness).await?);
        }
        Ok(self.gpu_cache.roughness.as_ref().unwrap())
    }

    /// Per-cell maximum flow thickness over the whole run; requires
    /// [`SimulationState::Finished`].
    pub async fn fetch_peak_flow_thickness(&mut self) -> Result<&[f32]> {
        if self.state < SimulationState::Finished {
            bail!("Simulation must be finished before reading peak flow thickness buffer");
        }
        if self.gpu_cache.peak_flow_thickness.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.peak_flow_thickness = Some(
                self.orchestrator
                    .read_buffer(BufferName::GridPeakFlowThickness)
                    .await?,
            );
        }
        Ok(self.gpu_cache.peak_flow_thickness.as_deref().unwrap())
    }

    /// Per-cell slope angle in degrees from terrain analysis.
    pub async fn fetch_slope_angle(&mut self) -> Result<&[f32]> {
        if self.state < SimulationState::TerrainAnalyzed {
            bail!("Terrain metrics must be computed before reading slope texture");
        }
        if self.gpu_cache.slope_angle.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.slope_angle = Some(
                self.orchestrator
                    .read_buffer(BufferName::SlopeAngle)
                    .await?,
            );
        }
        Ok(self.gpu_cache.slope_angle.as_deref().unwrap())
    }

    /// Per-cell slope aspect in degrees from terrain analysis.
    pub async fn fetch_slope_aspect(&mut self) -> Result<&[f32]> {
        if self.state < SimulationState::TerrainAnalyzed {
            bail!("Terrain metrics must be computed before reading slope aspect texture");
        }
        if self.gpu_cache.slope_aspect.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.slope_aspect = Some(
                self.orchestrator
                    .read_buffer(BufferName::SlopeAspect)
                    .await?,
            );
        }
        Ok(self.gpu_cache.slope_aspect.as_deref().unwrap())
    }

    /// Terrain geometry texture: surface normals (curvilinear metric terms
    /// `l_x`, `l_y`, Jacobian) in RGB and the projected gravity y component
    /// in A.
    async fn fetch_terrain_geometry_texture(&mut self) -> Result<&TextureRgba<f32>> {
        if self.state < SimulationState::TerrainAnalyzed {
            bail!("Terrain geometry must be computed before reading terrain geometry texture");
        }
        if self.gpu_cache.terrain_geometry.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.terrain_geometry =
                Some(self.get_texture_data(TextureName::TerrainGeometry).await?);
        }
        Ok(self.gpu_cache.terrain_geometry.as_ref().unwrap())
    }
    /// Curvature texture: `k_xx` in R, `k_yy` in G, `k_xy` in B and the
    /// projected gravity x component in A.
    pub async fn fetch_terrain_curvature(&mut self) -> Result<&TextureRgba<f32>> {
        if self.state < SimulationState::TerrainAnalyzed {
            bail!("Terrain curvature must be computed before reading terrain curvature texture");
        }
        if self.gpu_cache.curvature.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.curvature = Some(self.get_texture_data(TextureName::Curvature).await?);
        }
        Ok(self.gpu_cache.curvature.as_ref().unwrap())
    }

    /// Slope-projected gravity components `(g_x, g_y)` per cell for curvilinear models.
    pub async fn get_slope_gravity(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let g_x = self.fetch_terrain_curvature().await?.a.clone();
        let g_y = self.fetch_terrain_geometry_texture().await?.a.clone();
        Ok((g_x, g_y))
    }

    /// Terrain curvature components `(k_xx, k_yy, k_xy)` per cell.
    pub async fn get_curvature(&mut self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let k_x = self.fetch_terrain_curvature().await?.r.clone();
        let k_y = self.fetch_terrain_curvature().await?.g.clone();
        let k_xy = self.fetch_terrain_curvature().await?.b.clone();
        Ok((k_x, k_y, k_xy))
    }

    /// Elevation gradient `(dz/dx, dz/dy)` per meter, recovered from the
    /// terrain analysis textures.
    async fn terrain_gradient(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        const GRAVITY: f32 = 9.81;
        if self.settings.sim_model == SimModel::Curvilinear.as_int() {
            // metric terms l = sqrt(1 + dz^2) lose the sign; the slope-parallel
            // gravity g = -GRAVITY * dz / l restores it
            let g_x = self.fetch_terrain_curvature().await?.a.clone();
            let geometry = self.fetch_terrain_geometry_texture().await?;
            let gradient_x = geometry
                .r
                .iter()
                .zip(&g_x)
                .map(|(l_x, g_x)| -g_x * l_x / GRAVITY)
                .collect();
            let gradient_y = geometry
                .g
                .iter()
                .zip(&geometry.a)
                .map(|(l_y, g_y)| -g_y * l_y / GRAVITY)
                .collect();
            return Ok((gradient_x, gradient_y));
        }
        // surface normal n = (-dz/dx, -dz/dy, 1) / |...|
        let geometry = self.fetch_terrain_geometry_texture().await?;
        let gradient_x = geometry
            .r
            .iter()
            .zip(&geometry.b)
            .map(|(n_x, n_z)| -n_x / n_z)
            .collect();
        let gradient_y = geometry
            .g
            .iter()
            .zip(&geometry.b)
            .map(|(n_y, n_z)| -n_y / n_z)
            .collect();
        Ok((gradient_x, gradient_y))
    }

    /// First channel of the terrain geometry texture: surface normal x
    /// component (curvilinear metric term `l_x`).
    pub async fn get_terrain_geometry_x(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_terrain_geometry_texture().await?.r.clone())
    }

    /// Second channel of the terrain geometry texture: surface normal y
    /// component (curvilinear metric term `l_y`).
    pub async fn get_terrain_geometry_y(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_terrain_geometry_texture().await?.g.clone())
    }

    /// Third channel of the terrain geometry texture: surface normal z
    /// component (curvilinear Jacobian).
    pub async fn get_terrain_geometry_z(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_terrain_geometry_texture().await?.b.clone())
    }

    /// Per-cell release thickness in effect for this run (loaded from file,
    /// set directly, estimated or computed); requires
    /// [`SimulationState::ReleaseAreasComputed`].
    pub async fn fetch_release_areas(&mut self) -> Result<&[f32]> {
        if self.state < SimulationState::ReleaseAreasComputed {
            bail!("Release areas must be computed before reading release areas texture");
        }
        if self.gpu_cache.release_areas.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.release_areas = Some(
                self.orchestrator
                    .read_buffer(BufferName::ReleaseAreas)
                    .await?,
            );
        }
        Ok(self.gpu_cache.release_areas.as_deref().unwrap())
    }

    /// Per-cell maximum flow velocity over the whole run; requires
    /// [`SimulationState::Finished`].
    pub async fn fetch_peak_velocity(&mut self) -> Result<&Vec<f32>> {
        if self.state < SimulationState::Finished {
            bail!("Simulation must be finished before reading peak velocity");
        }
        if self.gpu_cache.peak_velocity.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.peak_velocity = Some(
                self.orchestrator
                    .read_buffer(BufferName::GridPeakVelocity)
                    .await?,
            );
        }
        Ok(self.gpu_cache.peak_velocity.as_ref().unwrap())
    }

    /// Per-step trajectory record of a randomly tracked particle.
    /// velocity, position, dt, uv, travel distances and CFL numbers, one
    /// entry per completed step. Requires [`SimulationState::Finished`].
    pub async fn fetch_timestep_data(&mut self) -> Result<&TimestepData> {
        if self.state < SimulationState::Finished {
            bail!("Simulation must run and be finished before reading timestep data");
        }
        if self.gpu_cache.timestep_data.is_none() {
            self.gpu_cache.read_count += 1;
            let full_data = self
                .orchestrator
                .read_buffer(BufferName::TimestepData)
                .await?;

            let data_aos: Vec<_> = full_data.into_iter().step_by(3).collect();

            self.gpu_cache.timestep_data = Some(TimestepData::from_aos(
                &data_aos,
                self.settings.cell_size,
                self.sim_info.timestep as usize,
            ));
        }
        Ok(self.gpu_cache.timestep_data.as_ref().unwrap())
    }

    /// Particle xy positions, flattened as `[x0, y0, x1, y1, ...]`.
    pub async fn fetch_particles_position(&mut self) -> Result<&Vec<[f32; 2]>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_position.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.particles_position = Some(
                self.orchestrator
                    .read_buffer(BufferName::ParticlesPosition)
                    .await?,
            );
        }
        Ok(self.gpu_cache.particles_position.as_ref().unwrap())
    }

    /// Particle horizontal velocities, stored as `[vx, vy]` pairs.
    pub async fn fetch_particles_velocity(&mut self) -> Result<&Vec<[f32; 2]>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_velocity.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.particles_velocity = Some(
                self.orchestrator
                    .read_buffer(BufferName::ParticlesVelocity)
                    .await?,
            );
        }
        Ok(self.gpu_cache.particles_velocity.as_ref().unwrap())
    }

    /// Particle vertical velocities. Curvilinear velocities are reconstructed
    /// tangent to the bed; MPMDAC still returns a synthetic zero component.
    pub async fn fetch_particles_velocity_z(&mut self) -> Result<&Vec<f32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_velocity_z.is_none() {
            self.gpu_cache.read_count += 1;
            if self.settings.sim_model == SimModel::MpmDaC.as_int() {
                self.gpu_cache.particles_velocity_z =
                    Some(vec![0.0; self.number_particles as usize]);
            } else {
                self.gpu_cache.particles_velocity_z = Some(
                    self.orchestrator
                        .read_buffer(BufferName::ParticlesVelocityZ)
                        .await?,
                );
            }
        }
        Ok(self.gpu_cache.particles_velocity_z.as_ref().unwrap())
    }

    /// Per-particle mass.
    pub async fn fetch_particles_mass(&mut self) -> Result<&Vec<f32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_mass.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.particles_mass = Some(
                self.orchestrator
                    .read_buffer(BufferName::ParticlesMass)
                    .await?,
            );
        }
        Ok(self.gpu_cache.particles_mass.as_ref().unwrap())
    }

    /// Raw per-cell deposited mass, as quantized by p2g (u32, scaled by
    /// `MASS_FACTOR` in the shader utils). The buffers reflect the p2g deposit
    /// of the most recently completed step, i.e. the mass field the grid
    /// physics pass consumed.
    pub async fn fetch_grid_mass(&mut self) -> Result<&Vec<u32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading grid buffers");
        }
        if self.gpu_cache.grid_mass.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.grid_mass = Some(
                self.orchestrator
                    .read_buffer::<u32>(BufferName::GridMass)
                    .await?,
            );
        }
        Ok(self.gpu_cache.grid_mass.as_ref().unwrap())
    }

    /// Raw per-cell deposited momentum as quantized by p2g (i32 pairs
    /// `u, v` per cell, scaled by `MOMENTUM_FACTOR` in the shader utils).
    pub async fn fetch_grid_momentum(&mut self) -> Result<&Vec<i32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading grid buffers");
        }
        if self.gpu_cache.grid_momentum.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.grid_momentum = Some(
                self.orchestrator
                    .read_buffer::<i32>(BufferName::GridMomentum)
                    .await?,
            );
        }
        Ok(self.gpu_cache.grid_momentum.as_ref().unwrap())
    }

    /// Per-particle elevation values.
    pub async fn fetch_particles_elevation(&mut self) -> Result<&Vec<f32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_elevation.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.particles_elevation = Some(
                self.orchestrator
                    .read_buffer(BufferName::ParticlesElevation)
                    .await?,
            );
        }
        Ok(self.gpu_cache.particles_elevation.as_ref().unwrap())
    }

    /// Decoded per-particle [`ParticleState`] (moving/stopped flags and the
    /// timestep at which each particle stopped).
    pub async fn fetch_particles_state(&mut self) -> Result<&Vec<ParticleState>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_stopped.is_none() {
            self.gpu_cache.read_count += 1;
            let raw_states = self
                .orchestrator
                .read_buffer::<u32>(BufferName::ParticlesState)
                .await?;
            self.gpu_cache.particles_stopped =
                Some(raw_states.into_iter().map(ParticleState::from).collect());
        }
        Ok(self.gpu_cache.particles_stopped.as_ref().unwrap())
    }

    /// Convenience: fetches position, velocity, mass, elevation and state in
    /// one go.
    pub async fn fetch_particles_all(&mut self) -> Result<()> {
        self.fetch_particles_position().await?;
        self.fetch_particles_velocity().await?;
        self.fetch_particles_mass().await?;
        self.fetch_particles_elevation().await?;
        self.fetch_particles_state().await?;
        Ok(())
    }

    /// Raw center-of-mass record per step (`com_x`, `com_y`, elevation, total
    /// mass); requires simulation to be finished.
    pub async fn fetch_center_of_mass(&mut self) -> Result<&Vec<CenterOfMassResult>> {
        if self.state < SimulationState::Finished {
            bail!("Simulation must be finished before reading center of mass");
        }
        if self.gpu_cache.center_of_mass.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.center_of_mass = Some(
                self.orchestrator
                    .read_buffer(BufferName::CenterOfMass)
                    .await?,
            );
        }
        Ok(self.gpu_cache.center_of_mass.as_ref().unwrap())
    }

    /// `(x, y, elevation)` vectors derived from [`Self::fetch_center_of_mass`],
    /// with non-finite records filtered out.
    pub async fn get_center_of_mass(&mut self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let center_of_mass = self.fetch_center_of_mass().await?;
        let mut x: Vec<f32> = Vec::with_capacity(center_of_mass.len());
        let mut y: Vec<f32> = Vec::with_capacity(center_of_mass.len());
        let mut z: Vec<f32> = Vec::with_capacity(center_of_mass.len());
        for com in center_of_mass.iter() {
            let com_x = com.com_x;
            let com_y = com.com_y;
            let elevation = com.elevation;
            if com_x.is_finite() && com_y.is_finite() && elevation.is_finite() {
                x.push(com_x);
                y.push(com_y);
                z.push(elevation);
            }
        }
        Ok((x, y, z))
    }

    /// Sum of all particle masses.
    pub async fn get_total_mass(&mut self) -> Result<f32> {
        let particles_mass = self.fetch_particles_mass().await?;
        let mass_total: f32 = particles_mass.iter().sum();
        Ok(mass_total)
    }

    /// Total released volume, obtained from [`Self::get_total_mass`] via the
    /// configured snow density.
    pub async fn get_total_volume(&mut self) -> Result<f32> {
        Ok(self.get_total_mass().await? / self.settings.density)
    }

    /// Raw debug from debug buffer.
    /// requires [`SimulationState::Finished`].
    pub async fn get_compute_particles_debug(&self) -> Result<Vec<f32>> {
        if self.state < SimulationState::Finished {
            bail!("Simulation must be finished before reading cell count grid");
        }
        self.orchestrator.read_buffer(BufferName::Debug).await
    }

    /// Pre-loads all results into [`Self::gpu_cache`] so that subsequent
    /// `fetch_*` calls are served from memory.
    pub async fn fetch_results(&mut self) -> Result<()> {
        let start = Instant::now();
        self.fetch_peak_flow_thickness().await?;
        self.fetch_peak_velocity().await?;
        self.fetch_particles_all().await?;
        self.fetch_timestep_data().await?;
        self.fetch_roughness().await?;
        self.fetch_slope_angle().await?;
        self.fetch_slope_aspect().await?;
        self.fetch_terrain_geometry_texture().await?;
        self.fetch_terrain_curvature().await?;
        self.fetch_release_areas().await?;
        let end = Instant::now();
        trace!(
            "Time taken to fetch all results from GPU: {:?}",
            end - start
        );
        Ok(())
    }

    /// Prints a `width * height` grid as ASCII art to stdout, box-averaged
    /// down to at most `max_w` × `max_h` characters. Values are clamped to
    /// `[0, 1]` and mapped onto a ` .:-=+*#%@` brightness ramp.
    pub fn print_grid(&self, grid: &[f32], max_w: usize, max_h: usize) -> Result<()> {
        if max_w == 0 || max_h == 0 {
            bail!("Maximum grid width and height must be greater than zero");
        }
        let expected_len = self
            .dem
            .width
            .checked_mul(self.dem.height)
            .ok_or_else(|| anyhow::anyhow!("DEM dimensions overflow usize"))?;
        if grid.len() != expected_len {
            bail!(
                "Grid length ({}) does not match DEM dimensions ({}x{}={})",
                grid.len(),
                self.dem.width,
                self.dem.height,
                expected_len
            );
        }
        // 1. Calculate dynamic strides to fit within max_w and max_h
        let stride_w = self.dem.width.div_ceil(max_w);
        let stride_h = self.dem.height.div_ceil(max_h);

        // We use the same stride for both dimensions to maintain aspect ratio
        let stride = stride_w.max(stride_h);

        let chars = " .:-=+*#%@";
        let n = chars.len() - 1;

        for y in (0..self.dem.height).step_by(stride) {
            for x in (0..self.dem.width).step_by(stride) {
                let mut sum = 0.0;
                let mut count = 0;

                // Average the local box
                for dy in 0..stride {
                    for dx in 0..stride {
                        let cur_y = y + dy;
                        let cur_x = x + dx;

                        if cur_y < self.dem.height && cur_x < self.dem.width {
                            sum += grid[cur_y * self.dem.width + cur_x];
                            count += 1;
                        }
                    }
                }

                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                let index = (avg.clamp(0.0, 1.0) * n as f32).round() as usize;
                print!("{}", chars.chars().nth(index).unwrap());
            }
            println!();
        }
        Ok(())
    }
}

/// Release cells times particles per cell, erroring on `u32` overflow.
fn checked_particle_count(release_cells: u32, particles_per_cell: u32) -> Result<u32> {
    release_cells
        .checked_mul(particles_per_cell)
        .ok_or_else(|| anyhow::anyhow!("Particle count exceeds u32 capacity"))
}

/// Converts center-of-mass records into absolute world coordinates (adding
/// the DEM origin) and derives the total travel length and average travel
/// angle in degrees.
fn hash_release_areas(release_areas: &[f32]) -> u64 {
    let mut hasher = StableHasher::default();
    for value in release_areas {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn trajectory_summary(
    center_of_mass: &[CenterOfMassResult],
    origin_x: f32,
    origin_y: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    // Slots of steps that did not run are NaN and the shader reports (0, 0, 0)
    // for steps without mass, neither of which is a position.
    let is_record = |result: &&CenterOfMassResult| {
        result.total_mass > 0.0
            && result.com_x.is_finite()
            && result.com_y.is_finite()
            && result.elevation.is_finite()
    };
    let positions = |origin: f32, coordinate: fn(&CenterOfMassResult) -> f32| {
        center_of_mass
            .iter()
            .map(|result| {
                if is_record(&result) {
                    origin + coordinate(result)
                } else {
                    f32::NAN
                }
            })
            .collect::<Vec<_>>()
    };
    let center_of_mass_x = positions(origin_x, |result| result.com_x);
    let center_of_mass_y = positions(origin_y, |result| result.com_y);

    let records: Vec<_> = center_of_mass.iter().filter(is_record).collect();
    let (Some(first), Some(last)) = (records.first(), records.last()) else {
        return (center_of_mass_x, center_of_mass_y, f32::NAN, f32::NAN);
    };
    let travel_length = records
        .windows(2)
        .map(|results| {
            let dx = results[1].com_x - results[0].com_x;
            let dy = results[1].com_y - results[0].com_y;
            dx.hypot(dy)
        })
        .sum::<f32>();
    let vertical_drop = first.elevation - last.elevation;
    let travel_angle = if travel_length > 0.0 {
        vertical_drop.atan2(travel_length).to_degrees()
    } else {
        0.0
    };

    (
        center_of_mass_x,
        center_of_mass_y,
        travel_length,
        travel_angle,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::buffers::AtomicValues;
    use pollster::block_on;
    use std::collections::HashSet;
    use std::hash::Hash;
    use std::hash::Hasher;

    const INCLINED_PLANE_PATH: &str = "../../data/avaframe/avaInclinedPlane.png";
    const RELEASE_TEXTURE_PATH: &str = "../../data/avaframe/avaInclinedPlanereleaseTexture.png";
    const GAR_PATH: &str = "../../data/avaframe/avaGar.png";
    const GAR_RELEASE_TEXTURE_PATH: &str = "../../data/avaframe/avaGarreleaseTexture.png";
    const ROI_PATH: &str = "../../data/outline/polygon_10721.shp";

    #[test]
    fn test_init_logging_idempotent() {
        // Call it once
        init_logging();

        // Call it again - it should not panic or error because of .call_once()
        init_logging();
    }

    #[test]
    fn test_trajectory_summary_uses_simulation_data() {
        let center_of_mass = vec![
            CenterOfMassResult {
                com_x: 0.0,
                com_y: 0.0,
                elevation: 10.0,
                total_mass: 1.0,
                ..Default::default()
            },
            CenterOfMassResult {
                com_x: 3.0,
                com_y: 4.0,
                elevation: 5.0,
                total_mass: 1.0,
                ..Default::default()
            },
        ];
        let (x, y, length, angle) = trajectory_summary(&center_of_mass, 100.0, 300.0);

        assert_eq!(x, vec![100.0, 103.0]);
        assert_eq!(y, vec![300.0, 304.0]);
        assert!((length - 5.0).abs() < f32::EPSILON);
        assert!((angle - 45.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trajectory_summary_skips_slots_without_a_record() {
        let record = |com_x, com_y, elevation, total_mass| CenterOfMassResult {
            com_x,
            com_y,
            elevation,
            total_mass,
            ..Default::default()
        };
        let center_of_mass = vec![
            CenterOfMassResult::default(), // slot 0 is never written
            record(0.0, 0.0, 10.0, 1.0),
            record(3.0, 4.0, 5.0, 1.0),
            record(0.0, 0.0, 0.0, 0.0),    // all mass has left
            CenterOfMassResult::default(), // the run ended
        ];
        let (x, y, length, angle) = trajectory_summary(&center_of_mass, 100.0, 300.0);

        assert!(x[0].is_nan() && y[0].is_nan());
        assert_eq!(x[1..3], [100.0, 103.0]);
        assert_eq!(y[1..3], [300.0, 304.0]);
        assert!(x[3].is_nan() && x[4].is_nan());
        assert!((length - 5.0).abs() < f32::EPSILON);
        assert!((angle - 45.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trajectory_summary_without_records_is_nan() {
        let center_of_mass = vec![CenterOfMassResult::default(); 3];
        let (x, _, length, angle) = trajectory_summary(&center_of_mass, 100.0, 300.0);

        assert!(x.iter().all(|value| value.is_nan()));
        assert!(length.is_nan() && angle.is_nan());
    }

    #[test_log::test]
    fn test_set_dem_rejects_invalid_inputs() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");

        assert!(sim.set_dem(&[1.0, 2.0, 3.0], 2, 2, 1.0).is_err());
        assert!(sim.set_dem(&[], 0, 0, 1.0).is_err());
        assert!(sim.set_dem(&[1.0], 1, 1, 0.0).is_err());
        assert!(sim.set_dem(&[1.0], 1, 1, f32::NAN).is_err());
        assert!(
            sim.set_dem_with_bounds(&[1.0], 1, 1, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
                .is_err()
        );
        assert!(
            sim.set_dem_with_bounds(&[1.0], 1, 1, 1.0, f32::NEG_INFINITY, 1.0, 0.0, 1.0, 1.0,)
                .is_err()
        );
    }

    #[test]
    fn test_checked_particle_count_rejects_overflow() {
        assert_eq!(checked_particle_count(12, 8).unwrap(), 96);
        assert!(checked_particle_count(u32::MAX, 2).is_err());
    }

    #[test_log::test]
    fn test_release_area_validation_tracks_current_dem() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        sim.set_dem(&[0.0; 4], 2, 2, 1.0).unwrap();
        sim.set_release_areas(&[1.0; 4]).unwrap();
        sim.set_dem(&[0.0; 9], 3, 3, 1.0).unwrap();

        assert!(
            sim.validate_release_areas_len(sim.release_areas_array.as_ref().unwrap())
                .is_err()
        );
    }

    #[test_log::test]
    fn test_lifecycle_preconditions_return_errors() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");

        assert!(block_on(sim.post_process()).is_err());
        assert!(block_on(sim.evaluate()).is_err());
        assert!(block_on(sim.compute_particles()).is_err());
        assert!(block_on(sim.fetch_peak_velocity()).is_err());
        assert!(block_on(sim.fetch_timestep_data()).is_err());
    }

    #[test_log::test]
    fn test_sim_create_without_path() {
        let settings = Settings::default();
        let mut sim: Simulation =
            block_on(Simulation::new()).expect("Failed to create Simulation without path");
        block_on(sim.create(settings)).expect("Failed to create simulation with default settings");
        assert_eq!(sim.state, SimulationState::DemMissing);
    }

    #[test_log::test]
    fn test_gpu_cache_read_count() {
        let number_cache_elements = 14;
        let number_sim_results_elements = 8;
        let mut sim: Simulation = setup_simple_sim(0.0, 1.0);
        block_on(sim.run()).expect("Failed to run simulation");
        let count_before = sim.get_gpu_cache_read_count();

        // First call: Should trigger a "read" and populate the Option
        block_on(sim.fetch_results()).expect("Failed to get data on first call");
        let first_ref =
            block_on(sim.fetch_particles_state()).expect("Failed to get particles on first call");
        let uncached_state = calculate_hash(&first_ref);
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + number_cache_elements,
            "Expected read_count to increase by {} after first call, but it did not.",
            number_cache_elements
        );

        // Second call: Should return the cached value
        block_on(sim.fetch_results()).expect("Failed to get data on second call");
        let second_ref =
            block_on(sim.fetch_particles_state()).expect("Failed to get particles on second call");
        let cached_state = calculate_hash(&second_ref);
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + number_cache_elements,
            "Expected read_count to NOT increase on second call, but it did."
        );

        // uncached and cached state should be the same
        assert_eq!(
            uncached_state, cached_state,
            "Cache failed: Second call returned different hash"
        );

        sim.gpu_cache.reset_simulation_result();
        assert!(
            sim.gpu_cache.particles_mass.is_none(),
            "Reset failed: GPU cache particles Option was not cleared"
        );

        // Cache the 5 results again after reset, should trigger reads again
        block_on(sim.fetch_results()).expect("Failed to get data on third call");
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + number_cache_elements + number_sim_results_elements,
            "Expected read_count to increase by {} after third call, but it did not",
            number_cache_elements + number_sim_results_elements
        );

        sim.settings.released_particles_per_cell = 7;
        block_on(sim.run()).expect("Failed to run simulation after changing settings");

        block_on(sim.fetch_results()).expect("Failed to get data on second call");
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + 2 * number_cache_elements + number_sim_results_elements,
            "Expected read_count to increase by {} after third call, but it did not",
            number_cache_elements + number_sim_results_elements
        );

        let third_ref =
            block_on(sim.fetch_particles_state()).expect("Failed to get particles on third call");
        let third_state = calculate_hash(&third_ref);
        // hash changed after sim with different settings, confirming cache was reset
        assert_ne!(
            cached_state, third_state,
            "Reset failed: Hash remained the same even after clearing cache"
        );
    }

    #[test_log::test]
    pub fn test_automatic_gpu_cache_reset() {
        let mut sim: Simulation = setup_simple_sim(0.0, 1.0);
        assert!(
            sim.gpu_cache.particles_mass.is_none()
                && sim.gpu_cache.particles_position.is_none()
                && sim.gpu_cache.particles_velocity.is_none()
                && sim.gpu_cache.particles_elevation.is_none()
                && sim.gpu_cache.particles_stopped.is_none()
                && sim.gpu_cache.roughness.is_none()
                && sim.gpu_cache.curvature.is_none()
                && sim.gpu_cache.terrain_geometry.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.terrain_geometry.is_none()
                && sim.gpu_cache.slope_angle.is_none()
                && sim.gpu_cache.slope_aspect.is_none()
                && sim.gpu_cache.peak_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should start empty"
        );
        block_on(sim.run()).expect("Failed to run simulation");

        assert!(
            sim.gpu_cache.particles_mass.is_none()
                && sim.gpu_cache.particles_position.is_none()
                && sim.gpu_cache.particles_velocity.is_none()
                && sim.gpu_cache.particles_elevation.is_none()
                && sim.gpu_cache.particles_stopped.is_none()
                && sim.gpu_cache.roughness.is_none()
                && sim.gpu_cache.curvature.is_none()
                && sim.gpu_cache.terrain_geometry.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.terrain_geometry.is_none()
                && sim.gpu_cache.slope_angle.is_none()
                && sim.gpu_cache.slope_aspect.is_none()
                && sim.gpu_cache.peak_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should stay empty after simulation run (no caching yet)"
        );
        block_on(sim.fetch_results()).expect("Failed to cache results");
        assert!(
            sim.gpu_cache.particles_mass.is_some()
                && sim.gpu_cache.particles_position.is_some()
                && sim.gpu_cache.particles_velocity.is_some()
                && sim.gpu_cache.particles_elevation.is_some()
                && sim.gpu_cache.particles_stopped.is_some()
                && sim.gpu_cache.roughness.is_some()
                && sim.gpu_cache.curvature.is_some()
                && sim.gpu_cache.terrain_geometry.is_some()
                && sim.gpu_cache.release_areas.is_some()
                && sim.gpu_cache.terrain_geometry.is_some()
                && sim.gpu_cache.slope_angle.is_some()
                && sim.gpu_cache.slope_aspect.is_some()
                && sim.gpu_cache.peak_velocity.is_some()
                && sim.gpu_cache.timestep_data.is_some(),
            "GPU cache should be fully populated after caching results"
        );

        block_on(sim.analyze_terrain()).expect("Failed to run normals shader");
        assert!(
            sim.gpu_cache.particles_mass.is_none()
                && sim.gpu_cache.particles_position.is_none()
                && sim.gpu_cache.particles_velocity.is_none()
                && sim.gpu_cache.particles_elevation.is_none()
                && sim.gpu_cache.particles_stopped.is_none()
                && sim.gpu_cache.roughness.is_none()
                && sim.gpu_cache.curvature.is_none()
                && sim.gpu_cache.terrain_geometry.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.terrain_geometry.is_none()
                && sim.gpu_cache.slope_angle.is_none()
                && sim.gpu_cache.slope_aspect.is_none()
                && sim.gpu_cache.peak_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should be empty after loading new DEM and running normals shader"
        );

        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.fetch_results()).expect("Failed to cache results");
        block_on(sim.load_release_areas()).expect("Failed to run release shader");

        assert!(sim.gpu_cache.particles_mass.is_none());
        assert!(sim.gpu_cache.particles_position.is_none());
        assert!(sim.gpu_cache.particles_velocity.is_none());
        assert!(sim.gpu_cache.particles_elevation.is_none());
        assert!(sim.gpu_cache.particles_stopped.is_none());
        assert!(sim.gpu_cache.release_areas.is_none());
        assert!(sim.gpu_cache.curvature.is_some());
        assert!(sim.gpu_cache.terrain_geometry.is_some());
        assert!(sim.gpu_cache.roughness.is_some());
        assert!(sim.gpu_cache.terrain_geometry.is_some());
        assert!(sim.gpu_cache.slope_angle.is_some());
        assert!(sim.gpu_cache.slope_aspect.is_some());
        assert!(sim.gpu_cache.peak_velocity.is_none());
        assert!(sim.gpu_cache.timestep_data.is_none());

        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.fetch_results()).expect("Failed to cache results");
        block_on(sim.initialize_particles()).expect("Failed to run initialize particles shader");

        assert!(sim.gpu_cache.particles_mass.is_none());
        assert!(sim.gpu_cache.particles_position.is_none());
        assert!(sim.gpu_cache.particles_velocity.is_none());
        assert!(sim.gpu_cache.particles_elevation.is_none());
        assert!(sim.gpu_cache.particles_stopped.is_none());
        assert!(sim.gpu_cache.release_areas.is_some());
        assert!(sim.gpu_cache.curvature.is_some());
        assert!(sim.gpu_cache.terrain_geometry.is_some());
        assert!(sim.gpu_cache.roughness.is_some());
        assert!(sim.gpu_cache.terrain_geometry.is_some());
        assert!(sim.gpu_cache.slope_angle.is_some());
        assert!(sim.gpu_cache.slope_aspect.is_some());
        assert!(sim.gpu_cache.peak_velocity.is_none());
        assert!(sim.gpu_cache.timestep_data.is_none());

        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.fetch_results()).expect("Failed to cache results");
        block_on(sim.compute_particles()).expect("Failed to run compute particles shader");

        assert!(sim.gpu_cache.particles_mass.is_none());
        assert!(sim.gpu_cache.particles_position.is_none());
        assert!(sim.gpu_cache.particles_velocity.is_none());
        assert!(sim.gpu_cache.particles_elevation.is_none());
        assert!(sim.gpu_cache.particles_stopped.is_none());
        assert!(sim.gpu_cache.release_areas.is_some());
        assert!(sim.gpu_cache.curvature.is_some());
        assert!(sim.gpu_cache.terrain_geometry.is_some());
        assert!(sim.gpu_cache.roughness.is_some());
        assert!(sim.gpu_cache.terrain_geometry.is_some());
        assert!(sim.gpu_cache.slope_angle.is_some());
        assert!(sim.gpu_cache.slope_aspect.is_some());
        assert!(sim.gpu_cache.peak_velocity.is_none());
        assert!(sim.gpu_cache.timestep_data.is_none());
    }

    pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = std::hash::DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    #[test_log::test]
    fn test_set_dem_initialization() {
        // 1. Setup mock data
        // A 2x3 grid (width=2, height=3)
        let dem_data = vec![
            10.0, 11.0, // Row 0
            20.0, 21.0, // Row 1
            30.0, 31.0, // Row 2
        ];

        // Ensure you have a way to create a 'blank' Simulation
        // If Simulation::new() is too heavy (GPU init), use a mock or Default
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create(Settings::default()))
            .expect("Failed to create simulation with default settings");
        // 2. Execute
        let result = sim.set_dem_with_bounds(
            &dem_data, 2,    // width
            3,    // height
            3.0,  // cell_size
            0.0,  // xmin
            2.0,  // xmax
            10.0, // ymin
            13.0, // ymax
            4.0,  // map_factor
        );

        // 3. Assertions
        assert!(result.is_ok(), "set_dem should return Ok");
        assert_eq!(
            sim.state,
            SimulationState::DemLoaded,
            "State should be Ready after setting DEM"
        );

        // Verify metadata
        assert_eq!(sim.dem.width, 2);
        assert_eq!(sim.dem.height, 3);
        assert_eq!(sim.dem.cell_size, 3.0);

        // Verify 1D data integrity (cloned correctly)
        assert_eq!(sim.dem.data1d, dem_data);

        // Verify 2D data transformation
        // Checking row 1, col 0 (which should be the 3rd element in 1D: 20.0)
        assert_eq!(sim.dem.data[1][0], 20.0);

        // Verify minimum elevation logic
        assert_eq!(sim.dem.minimum_elevation, 10.0);

        // Verify Bounds struct assignment
        assert_eq!(sim.dem.bounds.xmin, 0.0);
        assert_eq!(sim.dem.bounds.ymax, 13.0);

        // Coordinates are the cell centers
        assert_eq!(sim.dem.x, vec![0.5, 1.5]);
        assert_eq!(sim.dem.y, vec![10.5, 11.5, 12.5]);
        assert_eq!(sim.dem.bounds.xmin, 0.0);
        assert_eq!(sim.dem.bounds.ymin, 10.0);
        assert_eq!(sim.dem.bounds.xmax, 2.0);
        assert_eq!(sim.dem.bounds.ymax, 13.0);
        assert_eq!(sim.dem.map_factor, 4.0);
        assert_eq!(sim.dem.minimum_elevation, 10.0);

        assert_eq!(sim.settings.cell_size, 3.0);
        assert_eq!(sim.settings.grid_shape_x, 2);
        assert_eq!(sim.settings.grid_shape_y, 3);

        assert_eq!(sim.settings.world_size_x, 3.0 * 2 as f32);
        assert_eq!(sim.settings.world_size_y, 3.0 * 3 as f32);
        assert_eq!(sim.settings.release_min_elevation, 1500.0);

        block_on(sim.analyze_terrain()).expect("Failed to compute normals after setting DEM");
    }

    #[test_log::test]
    fn test_set_dem_initialization_invalid() {
        // 1. Setup mock data
        // A 2x3 grid (width=2, height=3)
        let dem_data = vec![
            10.0, 11.0, // Row 0
            20.0, 21.0, // Row 1
            30.0, 31.0, // Row 2
        ];

        // Ensure you have a way to create a 'blank' Simulation
        // If Simulation::new() is too heavy (GPU init), use a mock or Default
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.set_dem_with_bounds(
                &dem_data, 2,    // width
                2,    // height
                3.0,  // cell_size
                0.0,  // xmin
                2.0,  // xmax
                10.0, // ymin
                13.0, // ymax
                1.0,  // map_factor
            )
            .unwrap();
        }));
        assert!(
            result.is_err(),
            "set_dem should panic with invalid input for shape"
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.set_dem_with_bounds(
                &dem_data, 2,    // width
                3,    // height
                3.0,  // cell_size
                5.0,  // xmin
                2.0,  // xmax
                10.0, // ymin
                13.0, // ymax
                1.0,  // map_factor
            )
            .unwrap();
        }));
        assert!(
            result.is_err(),
            "set_dem should panic with invalid input for bounds"
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sim.set_dem_with_bounds(
                &dem_data, 2,    // width
                3,    // height
                3.0,  // cell_size
                0.0,  // xmin
                2.0,  // xmax
                10.0, // ymin
                3.0,  // ymax
                1.0,  // map_factor
            )
            .unwrap();
        }));
        assert!(
            result.is_err(),
            "set_dem should panic with invalid input for bounds"
        );
    }

    #[test_log::test]
    fn test_compute_release_areas() {
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create_default(GAR_PATH)).expect("Failed to create simulation");
        block_on(sim.prepare()).expect("Failed to prepare simulation");
    }

    #[test_log::test]
    fn test_roughness_is_model_independent_vector_ruggedness() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        let plane: Vec<f32> = (0..width * height)
            .map(|idx| 2000.0 + (idx / width) as f32 * 3.0)
            .collect();
        // irregular bumps of up to 4 m (a checkerboard would cancel out in the
        // central differences)
        let bumpy: Vec<f32> = plane
            .iter()
            .enumerate()
            .map(|(idx, &z)| z + ((idx % width * 7 + idx / width * 13) % 5) as f32)
            .collect();
        let interior = |idx: usize| {
            let (x, y) = (idx % width, idx / width);
            x > 0 && y > 0 && x + 1 < width && y + 1 < height
        };

        for model in [
            SimModel::TerrainFollowing,
            SimModel::Curvilinear,
            SimModel::MpmDaC,
            SimModel::SlidingBlock,
        ] {
            for (name, dem) in [("plane", &plane), ("bumpy", &bumpy)] {
                let mut sim: Simulation =
                    block_on(Simulation::new()).expect("Failed to create Simulation");
                // thresholds wide open so every cell releases and prepare()
                // succeeds on the rough surface too
                let settings = Settings {
                    sim_model: Some(model),
                    min_slope_angle: Some(0.0),
                    max_slope_angle: Some(90.0),
                    roughness_threshold: Some(1.0),
                    ..Default::default()
                };
                block_on(sim.create(settings)).expect("Failed to create simulation");
                sim.set_dem(dem, width, height, cell_size)
                    .expect("Failed to set DEM");
                sim.set_roi(&vec![true; width * height])
                    .expect("Failed to set ROI");
                block_on(sim.prepare()).expect("Failed to prepare simulation");
                let roughness = block_on(sim.fetch_roughness())
                    .expect("Failed to fetch roughness")
                    .clone();
                for (idx, &r) in roughness.iter().enumerate() {
                    assert!(
                        (0.0..=1.0).contains(&r),
                        "{model:?} {name}: roughness {r} outside [0, 1] at {idx}"
                    );
                    if !interior(idx) {
                        continue;
                    }
                    if name == "plane" {
                        assert!(r < 1e-5, "{model:?} plane: roughness {r} at {idx}");
                    }
                }
                if name == "bumpy" {
                    let interior_values: Vec<f32> = (0..roughness.len())
                        .filter(|&idx| interior(idx))
                        .map(|idx| roughness[idx])
                        .collect();
                    let mean = interior_values.iter().sum::<f32>() / interior_values.len() as f32;
                    assert!(mean > 1e-3, "{model:?} bumpy: mean roughness {mean}");
                }
            }
        }
    }

    #[test_log::test]
    fn test_release_estimation_crown_line_end_to_end() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        // inclined plane draining towards decreasing y
        let dem_data: Vec<f32> = (0..width * height)
            .map(|idx| 2000.0 + (idx / width) as f32 * 5.0)
            .collect();
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create(Settings::default())).expect("Failed to create simulation");
        sim.set_dem(&dem_data, width, height, cell_size)
            .expect("Failed to set DEM");
        // outline rectangle, upstream edge at y = 19
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                (4..20).contains(&x) && (4..20).contains(&y)
            })
            .collect();
        sim.set_roi(&roi).expect("Failed to set ROI");
        sim.release_area_fraction = Some(0.25);

        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.state, SimulationState::Finished);

        // 256 outline cells -> 25% -> 64 release cells on the upstream rows
        assert_eq!(sim.number_particles(), 64 * 8);
        assert_eq!(sim.crown_line.iter().filter(|&&c| c).count(), 16);
        let release_areas =
            block_on(sim.fetch_release_areas()).expect("Failed to fetch release areas");
        assert_eq!(release_areas.iter().filter(|&&t| t > 0.0).count(), 64);
        for (idx, &inside) in roi.iter().enumerate() {
            if !inside {
                assert_eq!(release_areas[idx], 0.0, "release outside outline at {idx}");
            }
        }
    }

    #[test_log::test]
    fn test_release_estimation_gradient_crown_line_end_to_end() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        // inclined plane draining towards decreasing y
        let dem_data: Vec<f32> = (0..width * height)
            .map(|idx| 2000.0 + (idx / width) as f32 * 5.0)
            .collect();
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create(Settings::default())).expect("Failed to create simulation");
        sim.set_dem(&dem_data, width, height, cell_size)
            .expect("Failed to set DEM");
        // outline rectangle, upstream edge at y = 19
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                (4..20).contains(&x) && (4..20).contains(&y)
            })
            .collect();
        sim.set_roi(&roi).expect("Failed to set ROI");
        sim.release_area_fraction = Some(0.25);
        sim.crown_line_method = CrownLineMethod::GradientStep;

        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.state, SimulationState::Finished);

        for (idx, &is_crown) in sim.crown_line.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = y == 19 && (4..20).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
        // 256 outline cells -> 25% -> 64 release cells on the upstream rows
        assert_eq!(sim.number_particles(), 64 * 8);
        let release_areas =
            block_on(sim.fetch_release_areas()).expect("Failed to fetch release areas");
        assert_eq!(release_areas.iter().filter(|&&t| t > 0.0).count(), 64);
    }

    #[test_log::test]
    fn test_release_estimation_slope_weighted_fill_end_to_end() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        // inclined plane draining towards decreasing y
        let dem_data: Vec<f32> = (0..width * height)
            .map(|idx| 2000.0 + (idx / width) as f32 * 5.0)
            .collect();
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create(Settings::default())).expect("Failed to create simulation");
        sim.set_dem(&dem_data, width, height, cell_size)
            .expect("Failed to set DEM");
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                (4..20).contains(&x) && (4..20).contains(&y)
            })
            .collect();
        sim.set_roi(&roi).expect("Failed to set ROI");
        sim.release_area_fraction = Some(0.25);
        sim.release_fill_method = ReleaseFillMethod::SlopeWeighted;

        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.state, SimulationState::Finished);
        // the uniform 45° plane weights every step alike: 256 outline cells
        // -> 25% -> 64 release cells
        assert_eq!(sim.number_particles(), 64 * 8);
        let release_areas =
            block_on(sim.fetch_release_areas()).expect("Failed to fetch release areas");
        assert_eq!(release_areas.iter().filter(|&&t| t > 0.0).count(), 64);
    }

    #[test_log::test]
    fn test_release_estimation_particle_crown_line_end_to_end() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        // inclined plane draining towards decreasing y
        let dem_data: Vec<f32> = (0..width * height)
            .map(|idx| 2000.0 + (idx / width) as f32 * 5.0)
            .collect();
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create(Settings::default())).expect("Failed to create simulation");
        sim.set_dem(&dem_data, width, height, cell_size)
            .expect("Failed to set DEM");
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                (4..20).contains(&x) && (4..20).contains(&y)
            })
            .collect();
        sim.set_roi(&roi).expect("Failed to set ROI");
        sim.release_area_fraction = Some(0.25);
        sim.crown_line_method = CrownLineMethod::ParticleSimulation;

        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.state, SimulationState::Finished);

        // the crown line is the upstream row y = 19
        let mut crown_cells = 0;
        for (idx, &is_crown) in sim.crown_line.iter().enumerate() {
            if is_crown {
                let (x, y) = (idx % width, idx / width);
                assert_eq!(y, 19, "crown cell outside the upstream row at x={x}");
                crown_cells += 1;
            }
        }
        assert!(
            crown_cells >= 12,
            "only {crown_cells} crown cells detected, expected most of row 19"
        );

        // fill below the crown line is unchanged by the detection method
        assert_eq!(sim.number_particles(), 64 * 8);
        let release_areas =
            block_on(sim.fetch_release_areas()).expect("Failed to fetch release areas");
        assert_eq!(release_areas.iter().filter(|&&t| t > 0.0).count(), 64);
        for (idx, &inside) in roi.iter().enumerate() {
            if !inside {
                assert_eq!(release_areas[idx], 0.0, "release outside outline at {idx}");
            }
        }
    }

    fn create_slope(ncols: usize, nrows: usize, cellsize: f32, slope_degrees: f32) -> Vec<f32> {
        let slope_radians = slope_degrees.to_radians();
        let elevation_rise_per_cell = cellsize * slope_radians.tan();

        // Base starting elevation for the westernmost column edge
        let base_elevation = 100.0;

        // 2. Build the flat row-major data layout
        let mut data = Vec::with_capacity(ncols * nrows);

        for _row in 0..nrows {
            for col in 0..ncols {
                // Elevation increases linearly with the column step index
                let cell_elevation = base_elevation + (col as f32 * elevation_rise_per_cell);
                data.push(cell_elevation);
            }
        }
        data
    }

    fn setup_simple_sim(slope_angle: f32, cell_size: f32) -> Simulation {
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        // a 6x6 DEM is smaller than the grid models' 3-cell p2g boundary, so
        // the fixture runs independent sliding blocks; tests override the model
        // the step counts asserted by the fixture's users assume CFL 0.5
        let settings = Settings {
            sim_model: Some(SimModel::SlidingBlock),
            cfl: Some(0.5),
            ..Default::default()
        };
        block_on(sim.create(settings)).expect("Failed to create simulation");
        sim.set_dem(
            &create_slope(6, 6, cell_size, slope_angle), // dem_data
            6,                                           // width
            6,                                           // height
            cell_size,                                   // cell_size
        )
        .expect("Failed to set DEM");
        sim.set_release_areas(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ])
        .expect("Failed to set release areas");
        sim
    }

    #[test_log::test]
    fn test_print_default_simulation_buffer_sizes() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create(Settings::default())).expect("Failed to create simulation");
        sim.set_dem(&[0.0; 36], 6, 6, 3.0)
            .expect("Failed to set DEM");
        sim.set_release_areas(&[
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ])
        .expect("Failed to set release areas");
        block_on(sim.run()).expect("Failed to run default simulation");

        let mut buffers = sim.orchestrator().resources.buffer_sizes();
        buffers.sort_by(|left, right| left.0.cmp(&right.0));
        println!("Default simulation buffers:");
        for (name, size_bytes) in buffers {
            println!(
                "  {name}: {size_bytes} bytes ({:.2} KiB)",
                size_bytes as f64 / 1024.0
            );
        }
    }

    #[test_log::test]
    fn test_creates() {
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        assert_eq!(sim.state, SimulationState::Uninitialized);
        block_on(sim.create_default_with_release_areas(GAR_PATH, GAR_RELEASE_TEXTURE_PATH))
            .expect("Failed to create simulation with default settings and release areas");
        assert_eq!(sim.state, SimulationState::DemLoaded);
        let dem = sim.dem.data1d.clone();
        block_on(sim.create_example(GAR_PATH)).expect("Failed to create example");
        assert!(vecs_are_equal(&dem, &sim.dem.data1d));
    }

    #[test_log::test]
    fn test_evaluate_gpu_chamfer_with_roi() {
        // Regression test: the evaluation shaders read the region of interest
        // from the GPU buffer, which must be uploaded from the CPU-side mask
        // before evaluating. Without the upload every ROI read is zero and
        // the chamfer distance is infinite.
        let mut sim = setup_simple_sim(40.0, 3.0);
        block_on(sim.run()).expect("Failed to run simulation");
        let simulated_cells = [14usize, 15, 20, 21];
        let mut peak_flow_thickness = vec![0.0f32; sim.dem.width * sim.dem.height];
        sim.ava_mask = vec![false; peak_flow_thickness.len()];
        for &cell in &simulated_cells {
            peak_flow_thickness[cell] = 1.0;
            sim.ava_mask[cell] = true;
        }
        block_on(
            sim.orchestrator
                .write_buffer(BufferName::GridPeakFlowThickness, &peak_flow_thickness),
        )
        .expect("Failed to seed simulated peak flow thickness");

        // ROI identical to the simulated mask: every nearest distance is 0
        sim.set_roi(&sim.ava_mask.clone())
            .expect("Failed to set roi");
        block_on(sim.upload_roi()).expect("Failed to upload roi");
        let evaluation = block_on(sim.evaluate_gpu()).expect("GPU evaluation failed");
        assert!(
            (evaluation.chamfer - 0.0).abs() < 1e-9,
            "chamfer was {}",
            evaluation.chamfer
        );
        assert!(
            (evaluation.iou - 1.0).abs() < 1e-9,
            "iou was {}",
            evaluation.iou
        );

        // ROI narrowed to the two release cells only: the simulated cells
        // that ran downslope are no longer covered, so the chamfer distance
        // is positive but finite
        let mut release_only = vec![false; sim.ava_mask.len()];
        release_only[14] = true;
        release_only[15] = true;
        sim.set_roi(&release_only).expect("Failed to set roi");
        block_on(sim.upload_roi()).expect("Failed to upload roi");
        let shifted_evaluation = block_on(sim.evaluate_gpu()).expect("GPU evaluation failed");
        assert!(
            shifted_evaluation.chamfer.is_finite() && shifted_evaluation.chamfer > 0.0,
            "chamfer was {}",
            shifted_evaluation.chamfer
        );
    }

    #[test_log::test]
    fn test_compute_simple() {
        let slope_angle: f32 = 40.0;
        let cell_size: f32 = 3.0;
        let mut sim: Simulation = setup_simple_sim(slope_angle, cell_size);
        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.state, SimulationState::Finished);
        let sim_info = block_on(sim.fetch_sim_info()).expect("Failed to fetch sim info");
        info!("Sim info: {:?}", sim_info);
        assert_eq!(sim.elevation_threshold(), 99.9);
        assert!(sim_info.timestep < 6);
        let atomics = block_on(sim.fetch_atomic_values()).expect("Failed to fetch sim info");
        info!("Atomic values: {:?}", atomics);
        assert_eq!(atomics.number_release_particles, 16);
        assert_eq!(atomics.stopped_particles, 16);
        let state = block_on(sim.fetch_particles_state()).expect("Failed to fetch particles");
        for p in state.iter() {
            info!("{:?}", p);
        }
        println!(
            "Particles stopped at step 0: {}",
            state.iter().filter(|state| state.timestep == 0).count()
        );
        assert_eq!(state.iter().filter(|state| state.timestep > 3).count(), 0);
        assert_eq!(state.iter().filter(|state| state.stopped).count(), 16);
        assert_eq!(state.iter().filter(|state| state.out_of_bounds).count(), 16);
        for p in state.iter() {
            info!("{:?}", p);
        }
        let cell_area = cell_size * cell_size;
        let mass_diff = cell_area * (2 * 200) as f32 / slope_angle.to_radians().cos()
            - block_on(sim.get_total_mass()).expect("Failed to get total mass");
        let volume_diff = cell_area * 2 as f32 / slope_angle.to_radians().cos()
            - block_on(sim.get_total_volume()).expect("Failed to get total volume");
        println!("Volume difference: {}", volume_diff);
        println!("Mass difference: {}", mass_diff);
        assert!(mass_diff.abs() < 0.4);
        assert!(volume_diff.abs() < 1e-2);
        let max_velocity = block_on(sim.fetch_peak_velocity()).expect("Failed to get max velocity");
        info!(
            "Max velocity after simulation: {:.2} m/s",
            max_velocity.max_value().unwrap(),
        );
        assert!(max_velocity.max_value().unwrap() < 12.0);
    }

    #[test_log::test]
    fn test_compute() {
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        let settings = Settings {
            dem_path: Some(INCLINED_PLANE_PATH.to_string()),
            release_areas_path: Some(
                INCLINED_PLANE_PATH
                    .to_string()
                    .replace(".png", "releaseTexture.png"),
            ),
            cfl: Some(0.5),
            max_steps: Some(6000),
            sim_model: Some(SimModel::SlidingBlock),
            // the velocity bounds below were tuned for these Voellmy parameters
            // (the defaults before c121632); the steady speed on the 34 deg plane
            // is sqrt(xi h (sin - mu cos)), ~38 m/s here and ~26 m/s with the
            // current defaults
            friction_coefficient: Some(0.155),
            drag_coefficient: Some(4000.0),
            ..Default::default()
        };
        block_on(sim.create(settings)).expect("Failed to create simulation");
        // block_on(sim.create_example(dem_path))
        block_on(sim.run()).expect("Failed to run simulation");
        let debug_buffer: Vec<f32> = block_on(sim.orchestrator.resources.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::Debug,
        ))
        .expect("Failed to read out_debug_normals_buffer");
        log_debug_buffer(&debug_buffer);
        let peak_velocity =
            block_on(sim.fetch_peak_velocity()).expect("Failed to get max velocity");
        info!("Peak velocity: {:?}", peak_velocity.max_value().unwrap());

        let width = 401usize;
        let x = 900usize;
        let count_above_1 = peak_velocity
            .as_slice()
            .chunks(width)
            .nth(x)
            .map(|row| row.iter().filter(|&&v| v > 1.0).count())
            .unwrap_or(0);
        info!(
            "Count of cells at x={} with peak velocity > 1: {}",
            x, count_above_1
        );

        let sim_info: SimInfo = *block_on(sim.orchestrator.resources.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::SimInfo,
        ))
        .expect("failed to read SimInfo buffer")
        .first()
        .expect("SimInfo buffer was empty");
        info!("Read sim info: {:?}", sim_info);
        // particles dont stop, they fall off the DEM
        let state = block_on(sim.fetch_particles_state()).expect("Failed to read particles buffer");
        info!(
            "Min step particle stopped: {}",
            state.iter().map(|state| state.timestep).min().unwrap()
        );
        info!(
            "Max step particle stopped: {}",
            state.iter().map(|state| state.timestep).max().unwrap()
        );
        // TODO fix this test
        assert_eq!(
            state.iter().filter(|state| state.timestep > 4900).count(),
            0
        );
        println!(
            "Particles stopped at step 0: {}",
            state.iter().filter(|state| state.timestep == 0).count()
        );
        assert!(state.iter().filter(|state| state.timestep == 0).count() < 20);

        let max_velocity = block_on(sim.fetch_peak_velocity()).expect("Failed to get max velocity");

        info!(
            "Max velocity after simulation: {:.2} m/s",
            max_velocity.max_value().unwrap(),
        );
        assert!(max_velocity.max_value().unwrap() > 25.0);
        assert!(max_velocity.max_value().unwrap() < 60.0);

        let max_steps = sim.settings.max_steps as usize;
        let timestep_data =
            block_on(sim.fetch_timestep_data()).expect("Failed to read timestep data buffer");
        let timesteps = sim_info.timestep as usize;
        assert!(
            timestep_data.position.len() <= max_steps,
            "Expected timestep data length to be less than max_steps {}, but got {}",
            max_steps,
            timestep_data.position.len()
        );

        // velocity X should be above 30.0 after step 500
        for i in 500..timesteps {
            let vel_x = timestep_data.velocity[i][0];
            assert!(
                vel_x > 30.0,
                "Velocity X dropped below 30.0 (value: {}) at step {}",
                vel_x,
                i
            );
        }

        // monotonically increasing position X
        for i in 1..timesteps {
            let pos_prev = timestep_data.position[i - 1][0];
            let pos_curr = timestep_data.position[i][0];
            if pos_curr != 0.0 && pos_curr.is_finite() {
                assert!(
                    pos_curr > pos_prev,
                    "Position X did not increase at step {}: {} -> {}",
                    i,
                    pos_prev,
                    pos_curr
                );
            }
        }

        let (com_x, com_y, com_elevation) =
            block_on(sim.get_center_of_mass()).expect("Failed to get center of mass");
        info!(
            "Center of mass: ({:?}, {:?}, {:?})",
            com_x, com_y, com_elevation
        );
    }

    #[test_log::test]
    fn test_compute_particles_finishes_direct_run() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        block_on(sim.prepare()).expect("Failed to prepare simulation");

        block_on(sim.compute_particles()).expect("Failed to run simulation");

        assert_eq!(sim.state, SimulationState::Finished);
        block_on(sim.fetch_peak_velocity()).expect("Failed to fetch finished results");
    }

    #[test_log::test]
    fn test_run_without_particles_returns_error() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        sim.set_dem(&create_slope(6, 6, 3.0, 40.0), 6, 6, 3.0)
            .expect("Failed to set DEM");
        sim.set_release_areas(&[0.0; 36])
            .expect("Failed to set release areas");

        assert!(block_on(sim.run()).is_err());
        assert_ne!(sim.state, SimulationState::Finished);
    }

    fn log_debug_buffer(buffer: &[f32]) {
        info!("Debug buffer length: {}", buffer.len());
        for (i, value) in buffer.iter().enumerate() {
            if *value != 0.0 {
                info!("{}: {}", i, value);
            }
        }
    }

    #[test_log::test]
    fn test_analyze_terrain_curvilinear() {
        if std::env::var("GITHUB_ACTIONS").is_ok()
            && (cfg!(target_os = "macos") || cfg!(target_os = "windows"))
        {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        let dem: &[f32] = &[
            15.0, 9.0, 8.0, 9.0, 12.0, 6.0, 3.0, 2.0, 3.0, 6.0, 4.0, 1.0, 0.0, 1.0, 4.0, 6.0, 3.0,
            2.0, 3.0, 6.0, 12.0, 9.0, 8.0, 9.0, 12.0,
        ];
        let expected_slope_angle: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 65.90516, 63.43495, 65.90516, 0.0, 0.0, 45.0, 0.0, 45.0,
            0.0, 0.0, 65.90516, 63.43495, 65.90516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_slope_aspect: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 206.56505, 180.0, 153.43495, 0.0, 0.0, 270.0, -1.0, 90.0,
            0.0, 0.0, 333.43494, 0.0, 26.565048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_terrain_l_x: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4142135, 1.0, 1.4142135, 0.0, 0.0, 1.4142135, 1.0,
            1.4142135, 0.0, 0.0, 1.4142135, 1.0, 1.4142135, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_terrain_l_y: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2360678, 2.2360678, 2.2360678, 0.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 2.2360678, 2.2360678, 2.2360678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_terrain_jacobian: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.4494896, 2.2360678, 2.4494896, 0.0, 0.0, 1.4142135,
            1.0, 1.4142135, 0.0, 0.0, 2.4494896, 2.2360678, 2.4494896, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let expected_k_xx: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20412414, 0.22360678, 0.20412414, 0.0, 0.0, 0.35355335,
            0.5, 0.35355335, 0.0, 0.0, 0.20412414, 0.22360678, 0.20412414, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let expected_k_yy: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40824828, 0.44721356, 0.40824828, 0.0, 0.0, 0.7071067,
            1.0, 0.7071067, 0.0, 0.0, 0.40824828, 0.44721356, 0.40824828, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let expected_k_xy: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07654655, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_g_x: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.936717, -0.0, -6.936717, 0.0, 0.0, 6.936717, -0.0,
            -6.936717, 0.0, 0.0, 6.936717, -0.0, -6.936717, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_g_y: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.77433, 8.77433, 8.77433, 0.0, 0.0, -0.0, -0.0, -0.0,
            0.0, 0.0, -8.77433, -8.77433, -8.77433, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let release: &[f32] = &[1.0; 25];
        sim.set_dem(dem, 5, 5, 2.0).expect("Failed to set DEM");
        sim.set_release_areas(release)
            .expect("Failed to set release areas");
        sim.settings.sim_model = 1;
        block_on(sim.prepare()).expect("Failed to prepare simulation");
        let slope_angle = block_on(sim.fetch_slope_angle())
            .expect("Failed to fetch slope angle")
            .to_vec();
        let slope_aspect = block_on(sim.fetch_slope_aspect())
            .expect("Failed to fetch slope aspect")
            .to_vec();
        let l_x = block_on(sim.get_terrain_geometry_x()).expect("Failed to get terrain metric l_x");
        let l_y = block_on(sim.get_terrain_geometry_y()).expect("Failed to get terrain metric l_y");
        let jacobian =
            block_on(sim.get_terrain_geometry_z()).expect("Failed to get terrain metric j");
        let (k_xx, k_yy, k_xy) =
            block_on(sim.get_curvature()).expect("Failed to get terrain metric curvature");
        let (g_x, g_y) =
            block_on(sim.get_slope_gravity()).expect("Failed to get terrain metric gravity ");
        info!("slope_angle: {:?}", slope_angle);
        info!("slope_aspect: {:?}", slope_aspect);
        info!("l_x: {:?}", l_x);
        info!("l_y: {:?}", l_y);
        info!("jacobian: {:?}", jacobian);
        info!("g_x: {:?}", g_x);
        info!("g_y: {:?}", g_y);
        info!("k_xx: {:?}", k_xx);
        info!("k_yy: {:?}", k_yy);
        info!("k_xy: {:?}", k_xy);

        for idx in 0..25 {
            assert!((l_x[idx] - expected_terrain_l_x[idx]).abs() < 1e-6);
            assert!((l_y[idx] - expected_terrain_l_y[idx]).abs() < 1e-6);
            assert!((jacobian[idx] - expected_terrain_jacobian[idx]).abs() < 1e-6);

            assert!((k_xx[idx] - expected_k_xx[idx]).abs() < 1e-6);
            assert!((k_yy[idx] - expected_k_yy[idx]).abs() < 1e-6);
            assert!((k_xy[idx] - expected_k_xy[idx]).abs() < 1e-6);

            assert!((slope_angle[idx] - expected_slope_angle[idx]).abs() < 1e-1);
            assert!((slope_aspect[idx] - expected_slope_aspect[idx]).abs() < 1e-1);

            assert!((g_x[idx] - expected_g_x[idx]).abs() < 1e-6);
            assert!((g_y[idx] - expected_g_y[idx]).abs() < 1e-6);
        }
    }

    #[test_log::test]
    fn test_analyze_terrain() {
        if std::env::var("GITHUB_ACTIONS").is_ok()
            && (cfg!(target_os = "macos") || cfg!(target_os = "windows"))
        {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        let dem: &[f32] = &[
            15.0, 9.0, 8.0, 9.0, 12.0, 6.0, 3.0, 2.0, 3.0, 6.0, 4.0, 1.0, 0.0, 1.0, 4.0, 6.0, 3.0,
            2.0, 3.0, 6.0, 12.0, 9.0, 8.0, 9.0, 12.0,
        ];
        let expected_slope_angle: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 65.90516, 63.43495, 65.90516, 0.0, 0.0, 45.0, 0.0, 45.0,
            0.0, 0.0, 65.90516, 63.43495, 65.90516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        // TODO one calculates slope aspect wrong
        let expected_slope_aspect: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 206.56505, 180.0, 153.43495, 0.0, 0.0, 270.0, -1.0, 90.0,
            0.0, 0.0, 333.43494, 0.0, 26.565048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_normals_x: &[f32] = &[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.40824828,
            -0.0,
            -0.40824828,
            0.0,
            0.0,
            0.7071067,
            -0.0,
            -0.7071067,
            0.0,
            0.0,
            0.40824828,
            -0.0,
            -0.40824828,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let expected_normals_y: &[f32] = &[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.81649655,
            0.8944271,
            0.81649655,
            0.0,
            0.0,
            -0.0,
            -0.0,
            -0.0,
            0.0,
            0.0,
            -0.81649655,
            -0.8944271,
            -0.81649655,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let expected_normals_z: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40824828, 0.44721356, 0.40824828, 0.0, 0.0, 0.7071067,
            1.0, 0.7071067, 0.0, 0.0, 0.40824828, 0.44721356, 0.40824828, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let expected_k_xx: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5,
            0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_k_yy: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let expected_k_xy: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let release: &[f32] = &[1.0; 25];
        sim.set_dem(dem, 5, 5, 2.0).expect("Failed to set DEM");
        sim.set_release_areas(release)
            .expect("Failed to set release areas");
        sim.settings.sim_model = 0;
        block_on(sim.prepare()).expect("Failed to prepare simulation");
        let slope_angle = block_on(sim.fetch_slope_angle())
            .expect("Failed to fetch slope angle")
            .to_vec();
        let slope_aspect = block_on(sim.fetch_slope_aspect())
            .expect("Failed to fetch slope aspect")
            .to_vec();
        let normals_x =
            block_on(sim.get_terrain_geometry_x()).expect("Failed to get terrain metric l_x");
        let normals_y =
            block_on(sim.get_terrain_geometry_y()).expect("Failed to get terrain metric l_y");
        let normals_z =
            block_on(sim.get_terrain_geometry_z()).expect("Failed to get terrain metric j");
        let (k_xx, k_yy, k_xy) =
            block_on(sim.get_curvature()).expect("Failed to get terrain metric curvature");
        info!("slope_angle: {:?}", slope_angle);
        info!("slope_aspect: {:?}", slope_aspect);
        info!("normals_x: {:?}", normals_x);
        info!("normals_y: {:?}", normals_y);
        info!("normals_z: {:?}", normals_z);
        info!("k_xx: {:?}", k_xx);
        info!("k_yy: {:?}", k_yy);
        info!("k_xy: {:?}", k_xy);

        for idx in 0..25 {
            assert!((normals_x[idx] - expected_normals_x[idx]).abs() < 1e-6);
            assert!((normals_y[idx] - expected_normals_y[idx]).abs() < 1e-6);
            assert!((normals_z[idx] - expected_normals_z[idx]).abs() < 1e-6);

            assert!((k_xx[idx] - expected_k_xx[idx]).abs() < 1e-6);
            assert!((k_yy[idx] - expected_k_yy[idx]).abs() < 1e-6);
            assert!((k_xy[idx] - expected_k_xy[idx]).abs() < 1e-6);

            assert!((slope_angle[idx] - expected_slope_angle[idx]).abs() < 1e-1);
            assert!((slope_aspect[idx] - expected_slope_aspect[idx]).abs() < 1e-1);
        }
    }
    #[test_log::test]
    fn test_load_release_areas() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create_example(INCLINED_PLANE_PATH)).unwrap();
        block_on(sim.prepare()).expect("Failed to prepare simulation");
        let number_release_cells = block_on(sim.load_release_areas()).unwrap();
        let release_thickness = block_on(sim.fetch_release_areas()).unwrap();
        info!(
            "Read release_texture: len: {} max: {:?} {:?}",
            release_thickness.len(),
            release_thickness.max_value(),
            release_thickness[1020..1040].to_vec(),
        );
        assert_eq!(number_release_cells, 3245);
        assert_eq!(release_thickness.iter().filter(|&&x| x > 0.0).count(), 3245);
        assert!(
            release_thickness
                .iter()
                .all(|&x| x == 0.0 || (x - 1.0).abs() < 1e-6)
        );
        info!("Read number_release_cells: {:?}", number_release_cells);
    }
    #[test_log::test]
    fn test_load_release_areas_gar() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create_example(GAR_PATH)).unwrap();
        block_on(sim.prepare()).expect("Failed to prepare simulation");
        let number_release_cells = block_on(sim.load_release_areas()).unwrap();
        let release_thickness = block_on(
            sim.orchestrator
                .read_buffer::<f32>(BufferName::ReleaseAreas),
        )
        .expect("Failed to get release_areas");
        info!(
            "Read release_texture: len: {} max: {:?} {:?}",
            release_thickness.len(),
            release_thickness.max_value().unwrap(),
            release_thickness[1020..1040].to_vec(),
        );
        assert_eq!(number_release_cells, 1628);
        assert_eq!(release_thickness.iter().filter(|&&x| x > 0.0).count(), 1628);
        assert!(
            release_thickness
                .iter()
                .all(|&x| x == 0.0 || (x - 1.2).abs() < 1e-6)
        );
        info!("Read number_release_cells: {:?}", number_release_cells);
    }

    /// Seeded and relaxed particles must lie in release cells under the
    /// floor(position / cell_size) convention of the simulation kernels.
    /// The L-shaped release has concave and convex edges on all four
    /// sides, and one cell below RELEASE_THICKNESS_THRESHOLD that must
    /// neither be seeded nor counted when sizing the particle buffer.
    #[test_log::test]
    fn test_particles_stay_in_release_cells() {
        const WIDTH: usize = 12;
        const HEIGHT: usize = 10;
        const CELL_SIZE: f32 = 5.0;
        const PARTICLES_PER_CELL: u32 = 8;
        let is_release_cell = |x: usize, y: usize| {
            (2..=8).contains(&x) && (2..=6).contains(&y) && !(x >= 5 && y >= 5)
        };
        let dem: Vec<f32> = (0..WIDTH * HEIGHT)
            .map(|idx| 1000.0 - 3.0 * (idx / WIDTH) as f32 * CELL_SIZE)
            .collect();
        let mut release: Vec<f32> = (0..WIDTH * HEIGHT)
            .map(|idx| f32::from(is_release_cell(idx % WIDTH, idx / WIDTH)))
            .collect();
        release[8 * WIDTH + 10] = 0.5 * compute_core::RELEASE_THICKNESS_THRESHOLD;
        let release_cells = release
            .iter()
            .filter(|&&thickness| thickness == 1.0)
            .count();
        assert_eq!(release_cells, 27);

        for relax in [false, true] {
            let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
            sim.set_dem(&dem, WIDTH, HEIGHT, CELL_SIZE)
                .expect("Failed to set DEM");
            sim.set_release_areas(&release)
                .expect("Failed to set release areas");
            sim.settings.released_particles_per_cell = PARTICLES_PER_CELL;
            sim.enable_particle_relaxation = relax;
            block_on(sim.prepare()).expect("Failed to prepare simulation");
            assert_eq!(
                sim.number_particles as usize,
                release_cells * PARTICLES_PER_CELL as usize
            );

            let positions = block_on(sim.fetch_particles_position())
                .expect("Failed to fetch particle positions")
                .clone();
            assert_eq!(positions.len(), sim.number_particles as usize);
            let mut particles_per_cell = vec![0u32; WIDTH * HEIGHT];
            for &[x, y] in &positions {
                let cell_x = (x / CELL_SIZE).floor();
                let cell_y = (y / CELL_SIZE).floor();
                assert!(
                    cell_x >= 0.0
                        && cell_y >= 0.0
                        && is_release_cell(cell_x as usize, cell_y as usize),
                    "particle ({x}, {y}) outside the release cells (relax: {relax})"
                );
                particles_per_cell[cell_y as usize * WIDTH + cell_x as usize] += 1;
            }
            if !relax {
                for (idx, &count) in particles_per_cell.iter().enumerate() {
                    let expected = if is_release_cell(idx % WIDTH, idx / WIDTH) {
                        PARTICLES_PER_CELL
                    } else {
                        0
                    };
                    assert_eq!(count, expected, "seeded particles in cell {idx}");
                }
            }
        }
    }

    #[test_log::test]
    fn test_initialize_particles() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create_example(INCLINED_PLANE_PATH)).expect("Failed to create example");
        sim.settings.released_particles_per_cell = 10;
        info!("Sim settings: {:?}", sim.settings);
        block_on(sim.analyze_terrain()).expect("Failed to analyze terrain");
        let data = block_on(data_processor::load_release_areas(RELEASE_TEXTURE_PATH))
            .expect("Failed to read release areas");
        let number_release_cells: u32 =
            block_on(sim.load_release_areas()).expect("Failed to load_release_areas");
        let estimated_release_volume = block_on(sim.orchestrator.run_initialize_particles(
            &sim.settings,
            number_release_cells * sim.settings.released_particles_per_cell,
            true,
        ))
        .expect("Failed to run initialize_particles shader");
        info!("Estimated release volume: {}", estimated_release_volume);
        let atomic_values = block_on(
            sim.orchestrator
                .read_buffer::<AtomicValues>(BufferName::AtomicValues),
        )
        .expect("Failed to read atomic values buffer")[0];
        info!("Atomic values: {:?}", atomic_values);
        let number_release_particles = atomic_values.number_release_particles;
        info!("Number release particles: {}", number_release_particles);
        assert_eq!(number_release_particles, 3245 * 10);
        assert_eq!(data.iter().filter(|&&x| x > 0.0).count(), 3245);
        let positions = block_on(
            sim.orchestrator
                .read_buffer::<f32>(BufferName::ParticlesPosition),
        )
        .expect("Failed to read particles buffer");
        info!(
            "Min: {:?} Max: {:?} Hist: {:?}",
            positions.min_value().unwrap(),
            positions.max_value().unwrap(),
            positions.hist_float()
        );
        assert_eq!(number_release_particles as usize * 2, positions.len());
        for chunk in positions.chunks_exact(2) {
            let [x, y] = chunk else {
                continue;
            };
            assert!(*x > 100.0);
            assert!(*x < 400.0);
            assert!(*y < 1150.0);
            assert!(*y > 850.0);
        }

        let start_elevations = block_on(
            sim.orchestrator
                .read_buffer::<f32>(BufferName::ParticlesElevation),
        )
        .expect("Failed to read particles buffer");
        for elevation in start_elevations.iter() {
            assert!(*elevation > 3000.0);
            assert!(*elevation < 3350.0);
        }
        let mass = block_on(
            sim.orchestrator
                .read_buffer::<f32>(BufferName::ParticlesMass),
        )
        .expect("Failed to read particles buffer");
        info!("len mass: {}", mass.len());
        info!(
            "Mass values min: {:?}, max: {:?}",
            mass.min_value().unwrap(),
            mass.max_value().unwrap()
        );
        for m in mass.iter() {
            assert!((*m - 603.1099).abs() < 1e-1);
        }
        let unique_values = positions
            .iter()
            .map(|p| p.to_bits())
            .collect::<HashSet<_>>()
            .len();
        info!(
            "Unique values: {}, {}%",
            unique_values,
            unique_values as f32 / positions.len() as f32 * 100.0
        );
        assert!(
            unique_values as f32 / positions.len() as f32 > 0.98,
            "Duplicate position found in vector"
        );
    }

    #[test_log::test]
    fn test_run_n_steps_prepares_and_resumes() {
        let mut sim = setup_simple_sim(40.0, 3.0);

        let initial = block_on(sim.run_n_steps(0)).expect("Failed to prepare simulation");
        assert_eq!(initial.timestep, 0);
        assert_eq!(sim.state, SimulationState::Running);

        let advanced = block_on(sim.run_n_steps(1)).expect("Failed to advance simulation");
        assert_eq!(advanced.timestep, 1);
        assert_eq!(sim.state, SimulationState::Running);

        // the particles leave the 6x6 DEM during the third step
        let resumed = block_on(sim.run_n_steps(2)).expect("Failed to resume simulation");
        assert_eq!(resumed.timestep, 3);

        let finished = block_on(sim.run_n_steps(1)).expect("Failed to resume simulation");
        assert_eq!(finished.timestep, resumed.timestep);
        assert_eq!(sim.state, SimulationState::Finished);

        let atomics = block_on(sim.fetch_atomic_values()).expect("Failed to fetch atomic values");
        assert_eq!(atomics.stopped_particles, sim.number_particles);
    }

    #[test_log::test]
    fn test_run_n_steps_clamps_to_max_steps() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        // the particles would need three steps to leave the DEM
        sim.settings.max_steps = 1;

        let result = block_on(sim.run_n_steps(u32::MAX)).expect("Failed to advance simulation");

        assert_eq!(result.timestep, 1);
        assert_eq!(sim.state, SimulationState::Finished);
    }

    #[test_log::test]
    fn test_curvilinear_velocity_z_is_tangent_to_bed() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        sim.settings.sim_model = SimModel::Curvilinear.as_int();
        sim.set_dem(&create_slope(16, 16, 3.0, 40.0), 16, 16, 3.0)
            .expect("Failed to set interior test DEM");
        let mut release = vec![0.0; 16 * 16];
        for y in 6..10 {
            for x in 6..10 {
                release[y * 16 + x] = 1.0;
            }
        }
        sim.set_release_areas(&release)
            .expect("Failed to set interior release");
        block_on(sim.prepare()).expect("Failed to prepare MPM simulation");
        let number_particles = sim.number_particles as usize;

        let velocity_z = block_on(sim.fetch_particles_velocity_z())
            .expect("Failed to fetch MPM vertical velocity");

        assert_eq!(velocity_z.len(), number_particles);
        assert!(velocity_z.iter().all(|velocity| *velocity == 0.0));

        block_on(sim.run_n_steps(1)).expect("Failed to advance curvilinear simulation");
        let velocity_z = block_on(sim.fetch_particles_velocity_z())
            .expect("Failed to fetch vertical velocity")
            .clone();
        let velocity_xy =
            block_on(sim.fetch_particles_velocity()).expect("Failed to fetch horizontal velocity");
        assert!(velocity_z.iter().any(|velocity| velocity.abs() > 1e-3));
        for (horizontal, vertical) in velocity_xy.iter().zip(velocity_z) {
            assert!(vertical.is_finite());
            assert!(vertical.abs() <= horizontal[0].hypot(horizontal[1]) * 2.0);
        }
    }

    #[test_log::test]
    fn test_sorted_p2g_matches_direct_p2g() {
        // integer grid sums do not depend on the deposit order, so the
        // tile-sorted P2G must reproduce the direct P2G exactly
        let run = |sort_interval: u32| {
            let mut sim = setup_simple_sim(30.0, 5.0);
            sim.settings.sim_model = SimModel::Curvilinear.as_int();
            sim.enable_particle_relaxation = false;
            sim.orchestrator.particle_sort_interval = sort_interval;
            sim.set_dem(&create_slope(40, 40, 5.0, 30.0), 40, 40, 5.0)
                .expect("Failed to set DEM");
            let mut release = vec![0.0; 40 * 40];
            for y in 12..28 {
                for x in 22..34 {
                    release[y * 40 + x] = 1.0;
                }
            }
            sim.set_release_areas(&release)
                .expect("Failed to set release");
            block_on(sim.run_n_steps(40)).expect("Failed to run");
            assert_eq!(sim.orchestrator.particle_sort_enabled(), sort_interval > 0);
            let mut positions = block_on(sim.fetch_particles_position())
                .expect("Failed to fetch positions")
                .clone();
            positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let grid_mass: Vec<u32> = block_on(sim.orchestrator.read_buffer(BufferName::GridMass))
                .expect("Failed to read grid mass");
            (positions, grid_mass)
        };
        let (direct_positions, direct_mass) = run(0);
        let (sorted_positions, sorted_mass) = run(3);
        assert!(direct_mass.iter().any(|mass| *mass > 0));
        assert_eq!(direct_mass, sorted_mass);
        assert_eq!(direct_positions, sorted_positions);
    }

    #[test_log::test]
    fn test_curvilinear_weak_pressure_gradient() {
        // one step of a flat square release: the weak-form grad h of
        // p2g_weak_pressure follows the centred difference at the release edge
        // and nearly vanishes inside the uniform release
        const N: usize = 40;
        const CELL: f32 = 5.0;
        let run = |weak_pressure: bool| {
            let mut sim = setup_simple_sim(0.0, CELL);
            sim.settings.sim_model = SimModel::Curvilinear.as_int();
            if weak_pressure {
                sim.settings.flags |=
                    compute_core::settings::SimFlags::new(false, false, false, false, false, true)
                        .mask;
            }
            sim.set_dem(&create_slope(N, N, CELL, 0.0), N, N, CELL)
                .expect("Failed to set DEM");
            let mut release = vec![0.0; N * N];
            for y in 12..28 {
                for x in 12..28 {
                    release[y * N + x] = 1.0;
                }
            }
            sim.set_release_areas(&release)
                .expect("Failed to set release");
            block_on(sim.run_n_steps(1)).expect("Failed to run");
            let mass: Vec<u32> = block_on(sim.orchestrator.read_buffer(BufferName::GridMass))
                .expect("Failed to read grid mass");
            let sums: Vec<i32> = block_on(sim.orchestrator.read_buffer(BufferName::GridForces))
                .expect("Failed to read weak pressure sums");
            (mass, sums, sim.settings.density)
        };

        let (_, centred_sums, _) = run(false);
        assert!(centred_sums.iter().all(|sum| *sum == 0));

        let (mass, sums, density) = run(true);
        // dequantize with MASS_FACTOR = 1e1 and PRESSURE_FACTOR = 1e3 (utils.wgsl)
        let node_mass = |x: usize, y: usize| mass[y * N + x] as f32 * 0.1;
        let thickness = |x: usize, y: usize| node_mass(x, y) / (density * CELL * CELL);
        let mut interior = Vec::new();
        let mut edge = Vec::new();
        let mut edge_agrees = 0;
        for y in 4..N - 4 {
            for x in 4..N - 4 {
                if node_mass(x, y) < 1.0 {
                    continue;
                }
                let weak = [
                    sums[(y * N + x) * 2] as f32 * 1e-3 / node_mass(x, y),
                    sums[(y * N + x) * 2 + 1] as f32 * 1e-3 / node_mass(x, y),
                ];
                let centred = [
                    (thickness(x + 1, y) - thickness(x - 1, y)) / (2.0 * CELL),
                    (thickness(x, y + 1) - thickness(x, y - 1)) / (2.0 * CELL),
                ];
                if (16..24).contains(&x) && (16..24).contains(&y) {
                    interior.push(weak[0].hypot(weak[1]));
                } else if centred[0].hypot(centred[1]) > 0.05 {
                    edge.push(weak[0].hypot(weak[1]));
                    if weak[0] * centred[0] + weak[1] * centred[1] > 0.0 {
                        edge_agrees += 1;
                    }
                }
            }
        }
        let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
        info!(
            "weak grad h: interior mean {:.4}, edge mean {:.4}, {}/{} edge nodes agree",
            mean(&interior),
            mean(&edge),
            edge_agrees,
            edge.len()
        );
        assert!(!interior.is_empty() && !edge.is_empty());
        assert!(edge_agrees as f32 >= 0.9 * edge.len() as f32);
        assert!(mean(&interior) < 0.25 * mean(&edge));
    }

    #[test_log::test]
    fn test_momentum_stop_fraction_stops_decaying_flow() {
        // a block spreading on flat ground gains and then loses momentum
        let run = |fraction: f32| {
            let mut sim = setup_simple_sim(0.0, 5.0);
            sim.settings.sim_model = SimModel::Curvilinear.as_int();
            sim.settings.friction_coefficient = 0.05;
            sim.settings.momentum_stop_fraction = fraction;
            sim.settings.max_steps = 2000;
            sim.enable_particle_relaxation = false;
            sim.set_dem(&create_slope(40, 40, 5.0, 0.0), 40, 40, 5.0)
                .expect("Failed to set DEM");
            let mut release = vec![0.0; 40 * 40];
            for y in 16..24 {
                for x in 16..24 {
                    release[y * 40 + x] = 1.0;
                }
            }
            sim.set_release_areas(&release)
                .expect("Failed to set release");
            let mut info = block_on(sim.run_n_steps(50)).expect("Failed to run");
            while !info.parsed_flags().contains(SimInfoFlags::SIM_STOPPED) && info.timestep < 2000 {
                info = block_on(sim.run_n_steps(50)).expect("Failed to run");
            }
            info
        };
        let off = run(0.0);
        let on = run(0.5);
        assert!(!off.parsed_flags().contains(SimInfoFlags::LOW_MOMENTUM));
        assert!(on.parsed_flags().contains(SimInfoFlags::LOW_MOMENTUM));
        assert!(on.timestep < off.timestep);
    }

    /// Slab released on a 30 degree slope, with relaxed particles, for the
    /// initial-state tests.
    fn setup_restore_sim(sim_model: SimModel) -> Simulation {
        let mut sim = setup_simple_sim(30.0, 5.0);
        sim.settings.sim_model = sim_model.as_int();
        sim.settings.max_steps = 150;
        sim.set_dem(&create_slope(40, 40, 5.0, 30.0), 40, 40, 5.0)
            .expect("Failed to set DEM");
        let mut release = vec![0.0; 40 * 40];
        for y in 16..24 {
            for x in 16..24 {
                release[y * 40 + x] = 1.0;
            }
        }
        sim.set_release_areas(&release)
            .expect("Failed to set release");
        sim
    }

    /// Step count, peak flow thickness and peak velocity of a full run.
    fn run_and_collect(sim: &mut Simulation) -> (u32, Vec<f32>, Vec<f32>) {
        block_on(sim.run()).expect("Failed to run");
        let timestep = sim.sim_info.timestep;
        let thickness = block_on(sim.fetch_peak_flow_thickness())
            .expect("Failed to fetch peak flow thickness")
            .to_vec();
        let velocity = block_on(sim.fetch_peak_velocity())
            .expect("Failed to fetch peak velocity")
            .clone();
        (timestep, thickness, velocity)
    }

    /// Integer grid sums are order-independent, so repeated runs from the
    /// same particles match exactly; float atomics may round differently.
    fn assert_same_run(sim: &Simulation, left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        if sim.orchestrator.has_float32_atomic() {
            let max = left.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            for (l, r) in left.iter().zip(right) {
                assert!((l - r).abs() <= 1e-3 * max.max(1.0), "{l} != {r}");
            }
        } else {
            assert_eq!(left, right);
        }
    }

    #[test_log::test]
    fn test_restore_initial_state_repeats_the_run() {
        for model in [SimModel::Curvilinear, SimModel::SlidingBlock] {
            let mut sim = setup_restore_sim(model);
            block_on(sim.store_initial_state()).expect("Failed to store initial state");
            assert!(sim.has_initial_state());
            let (steps, thickness, velocity) = run_and_collect(&mut sim);
            assert!(thickness.iter().any(|h| *h > 0.0));

            block_on(sim.restore_initial_state()).expect("Failed to restore");
            assert_eq!(sim.get_state(), SimulationState::ParticlesInitialized);
            let (steps_again, thickness_again, velocity_again) = run_and_collect(&mut sim);
            assert_eq!(steps, steps_again, "{model}");
            assert_same_run(&sim, &thickness, &thickness_again);
            assert_same_run(&sim, &velocity, &velocity_again);
        }
    }

    #[test_log::test]
    fn test_restore_initial_state_applies_loop_settings() {
        let mut sim = setup_restore_sim(SimModel::Curvilinear);
        block_on(sim.store_initial_state()).expect("Failed to store initial state");
        let (_, thickness, _) = run_and_collect(&mut sim);

        let patch = Settings {
            friction_coefficient: Some(0.5),
            ..Default::default()
        };
        sim.update_settings(&patch)
            .expect("Failed to update settings");
        block_on(sim.restore_initial_state()).expect("Failed to restore");
        let (_, thickness_rough, _) = run_and_collect(&mut sim);
        assert_ne!(thickness, thickness_rough);

        // back to the stored friction: the first run again
        let patch = Settings {
            friction_coefficient: Some(0.2),
            ..Default::default()
        };
        sim.update_settings(&patch)
            .expect("Failed to update settings");
        block_on(sim.restore_initial_state()).expect("Failed to restore");
        let (_, thickness_again, _) = run_and_collect(&mut sim);
        assert_same_run(&sim, &thickness, &thickness_again);
    }

    #[test_log::test]
    fn test_restore_initial_state_rescales_mass_and_reseeds_release() {
        let mut sim = setup_restore_sim(SimModel::Curvilinear);
        block_on(sim.store_initial_state()).expect("Failed to store initial state");
        let positions = block_on(sim.fetch_particles_position())
            .expect("Failed to fetch positions")
            .clone();
        let mass = block_on(sim.fetch_particles_mass())
            .expect("Failed to fetch mass")
            .clone();
        run_and_collect(&mut sim);

        // density only rescales the stored particles
        let patch = Settings {
            density: Some(400.0),
            ..Default::default()
        };
        sim.update_settings(&patch)
            .expect("Failed to update settings");
        block_on(sim.restore_initial_state()).expect("Failed to restore");
        let restored_positions = block_on(sim.fetch_particles_position())
            .expect("Failed to fetch positions")
            .clone();
        let restored_mass = block_on(sim.fetch_particles_mass())
            .expect("Failed to fetch mass")
            .clone();
        assert_eq!(positions, restored_positions);
        for (m, restored) in mass.iter().zip(&restored_mass) {
            assert!((m * 2.0 - restored).abs() <= 1e-6 * restored.abs());
        }

        // particles per cell change the seeding
        let number_particles = sim.number_particles();
        let patch = Settings {
            released_particles_per_cell: Some(4),
            ..Default::default()
        };
        sim.update_settings(&patch)
            .expect("Failed to update settings");
        block_on(sim.restore_initial_state()).expect("Failed to restore");
        assert_eq!(sim.number_particles(), number_particles / 2);
        let (_, thickness, _) = run_and_collect(&mut sim);
        assert!(thickness.iter().any(|h| *h > 0.0));
    }

    #[test_log::test]
    fn test_update_settings_rejects_input_data() {
        let mut sim = setup_restore_sim(SimModel::Curvilinear);
        let patch = Settings {
            dem_path: Some("other.tif".to_string()),
            ..Default::default()
        };
        assert!(sim.update_settings(&patch).is_err());
        assert!(block_on(sim.restore_initial_state()).is_err());
    }

    #[test_log::test]
    fn test_terrain_following_velocity_is_tangent_to_bed() {
        let slope_angle: f32 = 40.0;
        let mut sim = setup_simple_sim(slope_angle, 3.0);
        sim.settings.sim_model = SimModel::TerrainFollowing.as_int();
        sim.set_dem(&create_slope(16, 16, 3.0, slope_angle), 16, 16, 3.0)
            .expect("Failed to set interior test DEM");
        let mut release = vec![0.0; 16 * 16];
        for y in 6..10 {
            for x in 6..10 {
                release[y * 16 + x] = 1.0;
            }
        }
        sim.set_release_areas(&release)
            .expect("Failed to set interior release");
        block_on(sim.prepare()).expect("Failed to prepare terrain-following simulation");
        let start_x: Vec<f32> = block_on(sim.fetch_particles_position())
            .expect("Failed to read positions")
            .iter()
            .map(|p| p[0])
            .collect();

        block_on(sim.run_n_steps(5)).expect("Failed to advance terrain-following simulation");
        let velocity_z = block_on(sim.fetch_particles_velocity_z())
            .expect("Failed to fetch vertical velocity")
            .clone();
        let velocity_xy = block_on(sim.fetch_particles_velocity())
            .expect("Failed to fetch horizontal velocity")
            .clone();
        let positions = block_on(sim.fetch_particles_position())
            .expect("Failed to read positions")
            .clone();

        // the bed rises along +x, so a tangent velocity has v_z = tan(slope) v_x
        let tan_slope = slope_angle.to_radians().tan();
        for (horizontal, vertical) in velocity_xy.iter().zip(&velocity_z) {
            assert!(vertical.is_finite() && horizontal[0].is_finite());
            let speed = horizontal[0].hypot(horizontal[1]).hypot(*vertical);
            assert!(
                (vertical - tan_slope * horizontal[0]).abs() <= 1e-3 * speed + 1e-4,
                "velocity ({}, {}, {vertical}) leaves the bed",
                horizontal[0],
                horizontal[1]
            );
        }
        let mean_vx = velocity_xy.iter().map(|v| v[0]).sum::<f32>() / velocity_xy.len() as f32;
        assert!(
            mean_vx < 0.0,
            "mass must accelerate downslope, mean v_x {mean_vx}"
        );
        let mean_dx = positions
            .iter()
            .zip(&start_x)
            .map(|(p, x0)| p[0] - x0)
            .sum::<f32>()
            / positions.len() as f32;
        assert!(mean_dx < 0.0, "mass must move downslope, mean dx {mean_dx}");
    }

    #[test_log::test]
    fn test_curvilinear_apic_preserves_affine_velocity() {
        let mut sim = setup_simple_sim(0.0, 3.0);
        sim.settings.sim_model = SimModel::Curvilinear.as_int();
        sim.settings.flags = 0;
        sim.settings.velocity_threshold = 0.0;
        sim.settings.friction_model = 0;
        sim.settings.friction_coefficient = 0.0;
        sim.set_dem(&create_slope(24, 24, 3.0, 0.0), 24, 24, 3.0)
            .expect("Failed to set flat DEM");
        let mut release = vec![0.0; 24 * 24];
        for y in 6..18 {
            for x in 6..18 {
                release[y * 24 + x] = 1.0;
            }
        }
        sim.set_release_areas(&release)
            .expect("Failed to set release");
        block_on(sim.prepare()).expect("Failed to prepare APIC test");
        let positions = block_on(sim.fetch_particles_position())
            .expect("Failed to read positions")
            .clone();
        let velocities: Vec<[f32; 2]> = positions
            .iter()
            .map(|p| {
                [
                    0.5 + 0.02 * p[0] / 3.0 - 0.01 * p[1] / 3.0,
                    0.4 + 0.01 * p[0] / 3.0 + 0.03 * p[1] / 3.0,
                ]
            })
            .collect();
        let affine = vec![[0.02f32, 0.01, -0.01, 0.03]; positions.len()];
        block_on(
            sim.orchestrator
                .write_buffer(BufferName::ParticlesVelocity, &velocities),
        )
        .expect("Failed to seed affine velocities");
        block_on(
            sim.orchestrator
                .write_buffer(BufferName::ParticlesAffineMatrix, &affine),
        )
        .expect("Failed to seed affine matrices");
        block_on(sim.run_n_steps(1)).expect("Failed to advance APIC test");
        let actual = block_on(sim.fetch_particles_velocity()).expect("Failed to read velocities");
        for (p, (expected, actual)) in positions.iter().zip(velocities.iter().zip(actual)) {
            if p.iter()
                .all(|coordinate| *coordinate > 27.0 && *coordinate < 45.0)
            {
                for axis in 0..2 {
                    assert!(
                        (actual[axis] - expected[axis]).abs() < 3e-4,
                        "affine velocity: expected {}, got {}",
                        expected[axis],
                        actual[axis]
                    );
                }
            }
        }
        let matrices: Vec<[f32; 4]> = block_on(
            sim.orchestrator
                .read_buffer(BufferName::ParticlesAffineMatrix),
        )
        .expect("Failed to read affine matrices");
        for (p, matrix) in positions.iter().zip(matrices) {
            if p.iter()
                .all(|coordinate| *coordinate > 27.0 && *coordinate < 45.0)
            {
                for (actual, expected) in matrix.iter().zip(affine[0]) {
                    assert!((actual - expected).abs() < 5e-4);
                }
            }
        }
    }

    #[test_log::test]
    fn test_mpmdac_velocity_z_is_synthetic_zero() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        sim.settings.sim_model = SimModel::MpmDaC.as_int();
        block_on(sim.prepare()).expect("Failed to prepare MPMDAC simulation");
        let number_particles = sim.number_particles as usize;

        let velocity_z = block_on(sim.fetch_particles_velocity_z())
            .expect("Failed to fetch MPMDAC vertical velocity");

        assert_eq!(velocity_z.len(), number_particles);
        assert!(velocity_z.iter().all(|velocity| *velocity == 0.0));
    }

    // Diagnostic: steps the MPMDAC model one step at a time and dumps
    // the raw quantized grid buffer maxima to expose overflow or amplification.
    #[test_log::test]
    #[ignore]
    fn test_mpmdac_diagnostic_steps() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        sim.settings.sim_model = SimModel::MpmDaC.as_int();
        sim.settings.max_steps = 15;
        block_on(sim.prepare()).expect("Failed to prepare MPMDAC simulation");

        for _ in 0..15 {
            let info = block_on(sim.run_n_steps(1)).expect("step failed");
            let positions = block_on(sim.fetch_particles_position())
                .expect("positions")
                .clone();
            let velocities = block_on(sim.fetch_particles_velocity())
                .expect("velocities")
                .clone();
            let orchestrator = sim.orchestrator();
            let mass: Vec<u32> =
                block_on(orchestrator.read_buffer(BufferName::GridMass)).expect("mass buffer");
            let momentum: Vec<i32> = block_on(orchestrator.read_buffer(BufferName::GridMomentum))
                .expect("momentum buffer");
            let forces: Vec<i32> =
                block_on(orchestrator.read_buffer(BufferName::GridForces)).expect("forces buffer");
            let grid_velocity: Vec<[f32; 2]> =
                block_on(orchestrator.read_buffer(BufferName::GridVelocity))
                    .expect("velocity buffer");
            let max_mass_kg = mass.iter().copied().max().unwrap_or(0) as f32 * 0.1;
            let max_momentum = momentum.iter().copied().map(|v| v.abs()).max().unwrap_or(0);
            let max_force = forces.iter().copied().map(|v| v.abs()).max().unwrap_or(0);
            let max_grid_speed = grid_velocity
                .iter()
                .map(|velocity| {
                    velocity
                        .iter()
                        .map(|component| component.abs())
                        .fold(0.0f32, f32::max)
                })
                .fold(0.0f32, f32::max);
            let max_particle_speed = velocities
                .iter()
                .map(|velocity| (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt())
                .fold(0.0f32, f32::max);
            let min_x = positions.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
            let max_x = positions.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
            let min_y = positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            let max_y = positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
            println!(
                "t={:>3} dt={:>7.3} sim_max_v={:>9.3} flags={:#010x} | mass_kg={:>10.1} mom_i32={:>8} force_i32={:>8} grid_v={:>8.3} particle_v={:>8.3} bbox_x=({:.1},{:.1}) bbox_y=({:.1},{:.1})",
                info.timestep,
                info.dt,
                info.max_velocity,
                info.flags,
                max_mass_kg,
                max_momentum,
                max_force,
                max_grid_speed,
                max_particle_speed,
                min_x,
                max_x,
                min_y,
                max_y,
            );
        }
    }

    // Diagnostic on the real avaWog case (reproduces the reported fireworks).
    #[test_log::test]
    #[ignore]
    fn test_mpmdac_diagnostic_avawog() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        let settings = Settings::loads(
            r#"{
                "dem_path": "C:/git/avalanchers/data/avaframe/avaWog.png",
                "release_areas_path": "C:/git/avalanchers/data/avaframe/avaWogreleaseTexture.png",
                "sim_model": "mpmdac",
                "friction_model": "voellmy",
                "max_steps": 40,
                "released_particles_per_cell": 4,
                "slab_thickness_factor": 1.0,
                "friction_coefficient": 0.2,
                "drag_coefficient": 2000.0,
                "velocity_threshold": 0.1,
                "enable_center_of_mass": false
            }"#,
        )
        .expect("Failed to parse settings");
        block_on(sim.create(settings)).expect("Failed to load avaWog data");

        for _ in 0..40 {
            let info = block_on(sim.run_n_steps(1)).expect("step failed");
            let positions = block_on(sim.fetch_particles_position())
                .expect("positions")
                .clone();
            let velocities = block_on(sim.fetch_particles_velocity())
                .expect("velocities")
                .clone();
            let orchestrator = sim.orchestrator();
            let mass: Vec<u32> =
                block_on(orchestrator.read_buffer(BufferName::GridMass)).expect("mass buffer");
            let momentum: Vec<i32> = block_on(orchestrator.read_buffer(BufferName::GridMomentum))
                .expect("momentum buffer");
            let forces: Vec<i32> =
                block_on(orchestrator.read_buffer(BufferName::GridForces)).expect("forces buffer");
            let grid_velocity: Vec<[f32; 2]> =
                block_on(orchestrator.read_buffer(BufferName::GridVelocity))
                    .expect("velocity buffer");
            let max_mass_kg = mass.iter().copied().max().unwrap_or(0) as f32 * 0.1;
            let max_momentum = momentum.iter().copied().map(|v| v.abs()).max().unwrap_or(0);
            let max_force = forces.iter().copied().map(|v| v.abs()).max().unwrap_or(0);
            let max_grid_speed = grid_velocity
                .iter()
                .map(|velocity| {
                    velocity
                        .iter()
                        .map(|component| component.abs())
                        .fold(0.0f32, f32::max)
                })
                .fold(0.0f32, f32::max);
            let max_particle_speed = velocities
                .iter()
                .map(|velocity| (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt())
                .fold(0.0f32, f32::max);
            let min_x = positions.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
            let max_x = positions.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
            let min_y = positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            let max_y = positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
            // clumping metric: particles per DEM cell
            let cell_size = sim.settings.cell_size;
            let mut occupancy: std::collections::HashMap<(u32, u32), u32> =
                std::collections::HashMap::new();
            for position in &positions {
                let key = (
                    (position[0] / cell_size) as u32,
                    (position[1] / cell_size) as u32,
                );
                *occupancy.entry(key).or_insert(0) += 1;
            }
            let mut counts: Vec<u32> = occupancy.values().copied().collect();
            counts.sort_unstable();
            let max_occupancy = counts.last().copied().unwrap_or(0);
            let p99_occupancy = counts[counts.len().saturating_sub(1 + counts.len() / 100)]
                .max(counts.first().copied().unwrap_or(0));
            println!(
                "t={:>3} dt={:>7.3} sim_max_v={:>9.3} flags={:#010x} | mass_kg={:>10.1} mom_i32={:>8} force_i32={:>8} grid_v={:>8.3} particle_v={:>8.3} occ_max={:>4} occ_p99={:>4} bbox_x=({:.1},{:.1}) bbox_y=({:.1},{:.1})",
                info.timestep,
                info.dt,
                info.max_velocity,
                info.flags,
                max_mass_kg,
                max_momentum,
                max_force,
                max_grid_speed,
                max_particle_speed,
                max_occupancy,
                p99_occupancy,
                min_x,
                max_x,
                min_y,
                max_y,
            );
        }
    }

    // Ensure set_release_areas returns an error when the provided array length
    // does not match DEM dimensions.
    #[test_log::test]
    fn test_set_release_areas_length_mismatch() {
        let mut sim = block_on(Simulation::new()).expect("failed to create simulation");

        // set a DEM of size 4x3 => expected length 12
        let dem_data = vec![0.0f32; 4 * 3];
        sim.set_dem_default(&dem_data, 4, 3, 1.0)
            .expect("set_dem_default failed");

        // provide a release areas array of incorrect length (e.g., 5)
        let bad_release_areas = vec![1.0f32; 5];
        let res = sim.set_release_areas(&bad_release_areas);
        assert!(
            res.is_err(),
            "expected error for mismatched release areas length"
        );
        let err = res.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("does not match DEM dimensions"));
    }

    #[test]
    fn test_fetch_peak_flow_before_finish_returns_error() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        // Do not run simulation; directly call fetch_peak_flow_thickness and expect assertion
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on(sim.fetch_peak_flow_thickness()).unwrap();
        }));
        assert!(res.is_err());
    }

    #[test]
    fn test_fetch_timestep_data_before_finish_returns_error() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on(sim.fetch_timestep_data()).unwrap();
        }));
        assert!(res.is_err());
    }

    #[test]
    fn test_scenario_name_formatting() {
        let dem_path = "path_with_underscores/dem_file_1.tif";
        let release_path = "release_with_underscores/rel_1.png";
        let settings = SimSettings::new();
        let hash_hex = format!("{:x}", settings.calculate_hash());

        let release_areas_str = Some(release_path.to_string())
            .as_ref()
            .map(|p| p.replace('_', ""))
            .unwrap_or_else(|| "calculated".to_string());
        let scenario_name = format!(
            "{}_{}_{:x}",
            dem_path.replace('_', ""),
            release_areas_str,
            settings.calculate_hash()
        );

        assert_eq!(
            scenario_name,
            format!("pathwithunderscores/demfile1.tif_releasewithunderscores/rel1.png_{hash_hex}")
        );

        let release_areas_none: Option<String> = None;
        let release_none_str = match &release_areas_none {
            Some(path) => path.replace('_', ""),
            None => format!(
                "calculated-elev{}-minslope{}-maxslope{}-rough{}-slab{}",
                settings.release_min_elevation,
                settings.min_slope_angle,
                settings.max_slope_angle,
                settings.roughness_threshold,
                settings.slab_thickness_factor,
            ),
        };
        let scenario_name_none = format!(
            "{}_{}_{:x}",
            dem_path.replace('_', ""),
            release_none_str,
            settings.calculate_hash()
        );

        assert_eq!(
            scenario_name_none,
            format!(
                "pathwithunderscores/demfile1.tif_calculated-elev{}-minslope{}-maxslope{}-rough{}-slab{}_{hash_hex}",
                settings.release_min_elevation,
                settings.min_slope_angle,
                settings.max_slope_angle,
                settings.roughness_threshold,
                settings.slab_thickness_factor,
            )
        );
    }

    #[test_log::test]
    fn test_create_sim_with_gpu() {
        let gpus = block_on(compute_core::list_devices()).expect("Failed to list GPUs");
        let _ = block_on(Simulation::new_with_gpu(gpus.first().cloned()))
            .expect("Failed to create Simulation with GPU");
    }

    #[test_log::test]
    fn test_print_grid() {
        let sim = setup_simple_sim(40.0, 4.0);
        let grid = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            0.0, 1.0, 2.0, 3.0,
        ];
        let width = 6;
        let height = 6;
        sim.print_grid(&grid, width, height)
            .expect("Failed to print grid");
        assert!(sim.print_grid(&grid, 0, height).is_err());
        assert!(
            sim.print_grid(&grid[..grid.len() - 1], width, height)
                .is_err()
        );
    }

    #[test_log::test]
    fn test_get_gpu_data() {
        let mut sim = setup_simple_sim(40.0, 4.0);
        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.get_curvature()).expect("Failed to get normals_x");
        block_on(sim.get_terrain_geometry_x()).expect("Failed to get terrain_geometry_x");
        block_on(sim.get_terrain_geometry_y()).expect("Failed to get terrain_geometry_y");
        block_on(sim.get_terrain_geometry_z()).expect("Failed to get terrain_geometry_z");
        block_on(sim.get_slope_gravity()).expect("Failed to get slope");
        block_on(sim.fetch_slope_angle()).expect("Failed to get slope");
        block_on(sim.fetch_slope_aspect()).expect("Failed to get slope aspect");
        block_on(sim.fetch_particles_all()).expect("Failed to get particles");
        block_on(sim.fetch_peak_flow_thickness()).expect("Failed to get peak flow thickness");
        block_on(sim.fetch_peak_velocity()).expect("Failed to get peak velocity");
        block_on(sim.fetch_release_areas()).expect("Failed to get release areas");
        block_on(sim.fetch_roughness()).expect("Failed to get roughness");
        block_on(sim.fetch_timestep_data()).expect("Failed to get timestep data");
        block_on(sim.fetch_sim_info()).expect("Failed to get sim info");
    }

    #[test_log::test]
    fn test_sim_save() {
        let mut sim = setup_simple_sim(40.0, 4.0);

        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("test_sim_save.zarr");
        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.get_state(), SimulationState::Finished);
        block_on(sim.save_with_path(&file_path.to_string_lossy()))
            .expect("Failed to save simulation");
    }

    /// A run started from a store continues the scenario it came from.
    #[test_log::test]
    fn test_runs_import_their_inputs_from_a_store() {
        let mut sim = setup_simple_sim(40.0, 4.0);
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("runs.zarr").display().to_string();
        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.save_with_path(&path)).expect("Failed to save simulation");
        let release = block_on(sim.fetch_release_areas()).unwrap().to_vec();
        let site = sim.site_name();
        let scenario = sim.scenario_name(&release);

        let settings = Settings {
            dem_path: Some(path.clone()),
            output_path: Some(path.clone()),
            sim_model: Some(SimModel::SlidingBlock),
            ..Default::default()
        };
        let mut imported = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(imported.create(settings)).expect("Failed to create simulation");
        assert_eq!(imported.dem.width, sim.dem.width);
        assert_eq!(imported.dem.data1d, sim.dem.data1d);
        assert_eq!(imported.site_name(), site);

        block_on(imported.run()).expect("Failed to run imported simulation");
        assert_eq!(block_on(imported.fetch_release_areas()).unwrap(), release);
        assert_eq!(imported.scenario_name(&release), scenario);
        block_on(imported.save()).expect("Failed to save imported simulation");

        let metadata = std::fs::read(format!("{path}/{site}/{scenario}/zarr.json")).unwrap();
        let metadata: serde_json::Value = serde_json::from_slice(&metadata).unwrap();
        assert_eq!(metadata["attributes"]["number_of_runs"], 2);
    }

    #[test_log::test]
    fn test_write_asc() {
        let mut sim = setup_simple_sim(40.0, 4.0);
        block_on(sim.run()).expect("Failed to run simulation");

        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("peak_velocity.asc");
        let path = file_path.to_string_lossy();
        block_on(sim.write_asc("peak_velocity", &path)).expect("Failed to write grid");

        let bytes = std::fs::read(&file_path).unwrap();
        let reloaded = data_processor::load_dem_from_bytes(&bytes, "asc", &path).unwrap();
        let peak_velocity = block_on(sim.fetch_peak_velocity()).unwrap();
        assert_eq!(&reloaded.data1d, peak_velocity);

        assert!(block_on(sim.write_asc("no_such_layer", &path)).is_err());
    }

    #[test_log::test]
    fn test_scenario_name_identifies_the_release() {
        let sim = setup_simple_sim(40.0, 4.0);
        let name = sim.scenario_name(&[0.0, 1.0]);

        assert_eq!(name, sim.scenario_name(&[0.0, 1.0]));
        assert_ne!(name, sim.scenario_name(&[0.0, 2.0]));
    }

    #[test_log::test]
    fn test_release_hash() {
        let sim = setup_simple_sim(40.0, 4.0);
        assert_eq!(sim.release_hash(), 13523635338801950485);
    }

    #[test_log::test]
    fn test_internal_force() {
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        let settings = Settings {
            dem_path: Some(INCLINED_PLANE_PATH.to_string()),
            release_areas_path: Some(
                INCLINED_PLANE_PATH
                    .to_string()
                    .replace(".png", "releaseTexture.png"),
            ),
            cfl: Some(0.5),
            max_steps: Some(6000),
            sim_model: Some(SimModel::TerrainFollowing),
            ..Default::default()
        };
        block_on(sim.create(settings)).expect("Failed to create simulation");
        // block_on(sim.create_example(dem_path))
        block_on(sim.run()).expect("Failed to run simulation");
        let debug_buffer: Vec<f32> = block_on(sim.orchestrator.resources.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::Debug,
        ))
        .expect("Failed to read out_debug_normals_buffer");
        log_debug_buffer(&debug_buffer);
        let peak_flow_thickness = block_on(sim.fetch_peak_flow_thickness())
            .expect("Failed to fetch peak flow thickness")
            .to_vec();

        let x_start = 100usize;
        let x = 900usize;
        let width = sim.dem.width;
        let height = sim.dem.height;
        assert!(x < width, "x={} is out of bounds for width={}", x, width);

        let cells_above_threshold_start = (0..height)
            .filter(|&y| peak_flow_thickness[y * width + x_start] > 0.1)
            .count();
        let cells_above_threshold = (0..height)
            .filter(|&y| peak_flow_thickness[y * width + x] > 0.1)
            .count();

        println!(
            "peak_flow_thickness cells > 0.1 at x={} across y: {}\npeak_flow_thickness cells > 0.1 at x={} across y: {}",
            x_start, cells_above_threshold_start, x, cells_above_threshold
        );
        assert!(
            cells_above_threshold_start < cells_above_threshold,
            "Expected less cells above threshold at x={} than at x={}",
            x_start,
            x
        );
    }

    #[tokio::test]
    async fn test_load_roi() {
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        let settings = Settings {
            outlines_path: Some(ROI_PATH.to_string()),
            outlines_padding: Some(50.0),
            ..Default::default()
        };
        sim.create(settings)
            .await
            .expect("Failed to create simulation");
        // block_on(sim.create(settings)).expect("Failed to create simulation");
        assert!(!sim.roi.is_empty(), "ROI should be loaded");
        assert_eq!(
            sim.roi.len(),
            sim.dem.data1d.len(),
            "ROI length should match DEM data length"
        );
    }
}

#[cfg(test)]
mod avawog_diagnostics {
    use super::*;
    use pollster::block_on;

    /// Runs the avaWog avalanche with the curvilinear model and default
    /// Voellmy friction (mu = 0.2, xi = 2000) and verifies that the flow
    /// comes to rest on the track: most particles must stop, the maximum
    /// speed must decay, and neither NaN nor out-of-DEM flags may appear.
    #[test_log::test]
    fn test_curvilinear_diagnostic_avawog() {
        if std::env::var("GITHUB_ACTIONS").is_ok() {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        let settings = Settings::loads(
            r#"{
                "dem_path": "C:/git/avalanchers/data/avaframe/avaWog.png",
                "release_areas_path": "C:/git/avalanchers/data/avaframe/avaWogreleaseTexture.png",
                "sim_model": "curvilinear",
                "friction_model": "voellmy",
                "max_steps": 3000,
                "released_particles_per_cell": 4,
                "slab_thickness_factor": 1.0,
                "friction_coefficient": 0.2,
                "drag_coefficient": 2000.0,
                "velocity_threshold": 0.1,
                "enable_center_of_mass": false
            }"#,
        )
        .expect("Failed to parse settings");
        block_on(sim.create(settings)).expect("Failed to load avaWog data");
        info!(
            "DIAG settings: friction_model={} density={} fc={} drag={} vt={}",
            sim.settings.friction_model,
            sim.settings.density,
            sim.settings.friction_coefficient,
            sim.settings.drag_coefficient,
            sim.settings.velocity_threshold
        );

        let total_particles = sim.number_particles as usize;
        let mut final_max_speed = 0.0f32;
        let mut final_stopped = 0usize;
        for step in 1..=3000u32 {
            let info = block_on(sim.run_n_steps(1)).expect("step failed");
            if step % 250 != 0 && step != 3000 {
                continue;
            }
            let flags = info.parsed_flags();
            assert!(
                !flags.contains(SimInfoFlags::IS_NAN),
                "NaN flag set at step {}",
                info.timestep
            );
            assert!(
                !flags.contains(SimInfoFlags::PARTICLE_OUT_OF_DEM_DATA),
                "out of DEM flag set at step {}",
                info.timestep
            );
            let states = block_on(sim.fetch_particles_state())
                .expect("states")
                .clone();
            let velocities = block_on(sim.fetch_particles_velocity())
                .expect("velocities")
                .clone();
            assert!(
                velocities
                    .iter()
                    .all(|v| v[0].is_finite() && v[1].is_finite()),
                "non-finite particle velocity at step {}",
                info.timestep
            );
            final_stopped = states
                .iter()
                .filter(|state| state.stopped || state.resting)
                .count();
            final_max_speed = velocities
                .iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
                .fold(0.0f32, f32::max);
            info!(
                "DIAG step {} dt{:.4} stopped {}/{} max_speed {:.2}",
                info.timestep,
                info.dt,
                final_stopped,
                states.len(),
                final_max_speed
            );
        }

        // the flow must come to rest on the track instead of racing into the
        // runout zone at 30+ m/s
        assert!(
            final_stopped * 2 > total_particles,
            "only {final_stopped} of {total_particles} particles stopped"
        );
        assert!(
            final_max_speed < 20.0,
            "particles still move at {final_max_speed} m/s after 3000 steps"
        );
    }
}
