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
    dem::{Bounds, Dem},
    evaluation::MassMovementEvaluation,
    post_processing::*,
    settings::{CrownLineMethod, Settings, SimModel, SimSettings},
    utils::*,
};
#[cfg(target_arch = "wasm32")]
use data_processor::zarr_writer::{ResultGrids, ZarrEntry};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Once;
use web_time::Instant;

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
    /// Number of compute steps dispatched per GPU submission. The OS might reset the GPU or interrupt the submission, if computing a batch takes too long.
    pub batch_compute_steps: Option<u32>,
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
    release_areas_array: Option<Vec<f32>>,
    /// Crown line detected by release area estimation; empty unless
    /// `release_area_fraction` was set.
    pub crown_line: Vec<bool>,
    sim_info: SimInfo,
    number_particles: u32,
    state: SimulationState,
    /// CPU-side memoization of GPU readbacks, invalidated automatically as
    /// the simulation advances.
    pub gpu_cache: GpuCache,
    /// Avalanche deposit mask extracted by [`Simulation::post_process`].
    pub ava_mask: Vec<bool>,
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
            release_areas_array: None,
            crown_line: Vec::new(),
            ava_mask: Vec::new(),
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
        let mut s = DefaultHasher::new();
        if let Some(release_areas_array) = &self.release_areas_array {
            for val in release_areas_array.iter() {
                val.to_bits().hash(&mut s);
            }
        }
        s.finish()
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
        let (settings_result, dem_result, outline) =
            data_processor::create_sim_settings_and_dem(settings).await?;

        timer_checkpoint("Load settings");

        Ok(SimulationLoadResult {
            settings: settings_result,
            enable_center_of_mass: settings.enable_center_of_mass.unwrap_or(true),
            enable_particle_relaxation: settings.enable_particle_relaxation.unwrap_or(true),
            dem: dem_result,
            roi: outline,
            batch_compute_steps: settings.batch_compute_steps,
            dem_path: settings.dem_path.clone().unwrap_or_default(),
            release_areas_path: settings.release_areas_path.clone(),
            release_area_fraction: settings.release_area_fraction,
            crown_line_method: settings.crown_line_method,
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
        self.crown_line = Vec::new();

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
            x: linspace(bounds_xmin, bounds_xmax, width),
            y: linspace(bounds_ymin, bounds_ymax, height),
            source: String::new(),
            projection: String::new(),
        };
        self.settings.set_dem(&self.dem);

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
                model if model == SimModel::TerrainFollowing.as_int() => {
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
    pub async fn run(&mut self) -> Result<()> {
        self.analyze_terrain().await?;
        timer_checkpoint("Terrain analyzed");
        if !self.roi.is_empty() {
            self.upload_roi().await?;
        }
        let _ = self.load_release_areas().await?;
        timer_checkpoint("Release areas loaded");
        if self.number_particles == 0 {
            bail!("No particles to simulate! Check if release areas are correctly defined.");
        } else {
            self.initialize_particles().await?;
            timer_checkpoint("Particles initialized");
            self.compute_particles().await?;

            self.sim_info = self.fetch_sim_info().await?;
        }
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
    pub async fn evaluate(&mut self) -> Result<MassMovementEvaluation> {
        if self.state < SimulationState::PostProcessed {
            bail!("Simulation must be post-processed before evaluation");
        }
        let overlap =
            compute_core::evaluation::evaluate_mass_movement_area(&self.ava_mask, &self.roi)
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
        let source = if !self.dem.source.is_empty() {
            self.dem.source.as_str()
        } else {
            self.dem_path.as_str()
        };
        let stem = std::path::Path::new(source)
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("default-site")
            .replace("_", "-");
        format!("{}_{:x}", stem, self.dem.calculate_hash())
    }

    /// Name of the scenario, derived from the release areas.
    pub fn scenario_name(&self) -> String {
        let base = match &self.release_areas_path {
            Some(path) => std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
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
        format!("{}_{:x}", base, self.release_hash())
    }

    /// Builds an in-memory Zarr store of the current results.
    ///
    /// Used by the browser, which cannot write files itself; the caller is
    /// responsible for persisting the returned entries.
    #[cfg(target_arch = "wasm32")]
    pub async fn export_zarr_entries(&mut self) -> Result<Vec<ZarrEntry>> {
        if self.state < SimulationState::Finished {
            bail!("Run the simulation before saving results");
        }

        let site_name = self.site_name();
        let scenario_name = self.scenario_name();

        let release_areas = self.fetch_release_areas().await?.to_vec();
        self.fetch_peak_velocity().await?;
        self.fetch_peak_flow_thickness().await?;

        let peak_velocity = self
            .gpu_cache
            .peak_velocity
            .clone()
            .unwrap_or_else(|| vec![0.0; release_areas.len()]);
        let peak_flow_thickness = self
            .gpu_cache
            .peak_flow_thickness
            .clone()
            .unwrap_or_else(|| vec![0.0; release_areas.len()]);

        let settings = serde_json::to_value(self.settings).unwrap_or(serde_json::Value::Null);

        Ok(data_processor::zarr_writer::build_result_store(
            &site_name,
            &scenario_name,
            &self.dem,
            &ResultGrids {
                release_areas: &release_areas,
                peak_velocity: &peak_velocity,
                peak_flow_thickness: &peak_flow_thickness,
            },
            settings,
        ))
    }

    /// Native only: appends this run to the Zarr store at
    /// [`Self::output_path`], creating the site (derived from the DEM, see
    /// [`Self::site_name`]) and scenario (derived from the release areas, see
    /// [`Self::scenario_name`]) entries as needed. Each run stores peak
    /// velocity, peak flow thickness, release volume and the center-of-mass
    /// trajectory.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn save(&mut self) -> Result<()> {
        let mut site_name = std::path::Path::new(&self.dem_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default_site")
            .to_string()
            .replace("_", "-");
        site_name += &format!("_{:x}", self.dem.calculate_hash());
        let release_areas_str = match &self.release_areas_path {
            Some(path) => std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(path)
                .to_string()
                .replace("_", "-"),
            None => format!(
                "calculated-elev{}-minslope{}-maxslope{}-rough{}",
                // TODO save release relevant parameters to zarr store
                self.settings.release_min_elevation,
                self.settings.min_slope_angle,
                self.settings.max_slope_angle,
                self.settings.roughness_threshold,
            ),
        };

        let scenario_name = format!("{}_{:x}", release_areas_str, self.release_hash());
        if self.output.is_none() {
            self.output = Some(data_processor::output::Output::new(&self.output_path)?);
        }
        let release_areas = self.fetch_release_areas().await?.to_vec();

        self.fetch_peak_velocity().await?;
        self.fetch_peak_flow_thickness().await?;
        let release_volume = self.get_total_volume().await?;
        let origin_x = self.dem.bounds.xmin;
        let origin_y = self.dem.bounds.ymin;
        let center_of_mass = self.fetch_center_of_mass().await?;
        let (center_of_mass_x, center_of_mass_y, travel_length, travel_angle) =
            trajectory_summary(center_of_mass, origin_x, origin_y);
        timer_checkpoint("Peak data fetched");

        let output = self.output.as_mut().unwrap();
        if !output.site_exists(&site_name) {
            output.add_new_site(&site_name, &self.dem)?;
        }

        if output.scenario_exists(&site_name, &scenario_name) {
            output.connect_scenario(&site_name, &scenario_name)?;
        } else {
            output.add_new_scenario(
                &site_name,
                &scenario_name,
                10000, // number runs
                self.settings.max_steps as u64,
                &release_areas,
                f32::NAN, // Release aspect is not currently calculated.
                release_volume,
                self.dem.y.clone(),
                &self.dem.x,
            )?;
        }
        output.add_new_run(
            &site_name,
            &scenario_name,
            self.gpu_cache.peak_velocity.as_ref().unwrap(),
            self.gpu_cache.peak_flow_thickness.as_ref().unwrap(),
            &center_of_mass_x,
            &center_of_mass_y,
            travel_length,
            travel_angle,
            &self.settings,
        )?;
        Ok(())
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
        settings.sim_model = Some(SimModel::TerrainFollowing);
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
        if !sim_info
            .parsed_flags()
            .contains(SimInfoFlags::ALL_PARTICLES_STOPPED)
        {
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
            if let Some(idx) = release_estimation::entry_cell(
                *position,
                *velocity,
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

        release_estimation::crown_line_from_particle_counts(&counts, &self.dem, &self.roi, config)
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
                data.iter().filter(|&&x| x > 1e-3).count() as u32
            }
            None => match &self.release_areas_array {
                Some(data) => {
                    info!("Loading release areas from provided array");
                    self.validate_release_areas_len(data)?;
                    self.orchestrator
                        .write_buffer(BufferName::ReleaseAreas, data)
                        .await?;
                    data.iter().filter(|&&x| x > 1e-3).count() as u32
                }
                None => match self.release_area_fraction {
                    Some(fraction) => {
                        info!(
                            "Estimating release areas from crown line (fraction {fraction:.2}, method {})",
                            self.crown_line_method
                        );
                        let config = release_estimation::ReleaseEstimationConfig {
                            fraction,
                            slab_thickness: self.settings.slab_thickness_factor,
                            expected_crown_slope_range: (
                                self.settings.min_slope_angle,
                                self.settings.max_slope_angle,
                            ),
                            ..Default::default()
                        };
                        let detection = match self.crown_line_method {
                            CrownLineMethod::FlowRouting => release_estimation::detect_crown_line(
                                &self.dem, &self.roi, &config,
                            )?,
                            CrownLineMethod::ParticleSimulation => {
                                // Box::pin breaks the future type recursion:
                                // run() -> load_release_areas() -> here -> run()
                                // (the nested simulation is a separate instance,
                                // so the indirection is purely a type-level fix)
                                Box::pin(self.detect_crown_line_by_particle_simulation(&config))
                                    .await?
                            }
                        };
                        let estimate = release_estimation::estimate_release_areas_with_crown(
                            &self.dem, &self.roi, &detection, &config,
                        )?;
                        self.crown_line = estimate.crown_line;
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

    /// Particle xy velocities, flattened as `[vx0, vy0, vx1, vy1, ...]`.
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

    /// Particle vertical velocities; identically zero for the MPM model,
    /// which does not track a separate vertical component.
    pub async fn fetch_particles_velocity_z(&mut self) -> Result<&Vec<f32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_velocity_z.is_none() {
            self.gpu_cache.read_count += 1;
            if self.settings.sim_model != SimModel::TerrainFollowing.as_int() {
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
fn trajectory_summary(
    center_of_mass: &[CenterOfMassResult],
    origin_x: f32,
    origin_y: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    let center_of_mass_x: Vec<_> = center_of_mass
        .iter()
        .map(|result| origin_x + result.com_x)
        .collect();
    let center_of_mass_y: Vec<_> = center_of_mass
        .iter()
        .map(|result| origin_y + result.com_y)
        .collect();
    let travel_length = center_of_mass
        .windows(2)
        .map(|results| {
            let dx = results[1].com_x - results[0].com_x;
            let dy = results[1].com_y - results[0].com_y;
            dx.hypot(dy)
        })
        .sum::<f32>();
    let vertical_drop = center_of_mass
        .first()
        .zip(center_of_mass.last())
        .map_or(0.0, |(first, last)| first.elevation - last.elevation);
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
            },
            CenterOfMassResult {
                com_x: 3.0,
                com_y: 4.0,
                elevation: 5.0,
                total_mass: 1.0,
            },
        ];
        let (x, y, length, angle) = trajectory_summary(&center_of_mass, 100.0, 300.0);

        assert_eq!(x, vec![100.0, 103.0]);
        assert_eq!(y, vec![300.0, 304.0]);
        assert!((length - 5.0).abs() < f32::EPSILON);
        assert!((angle - 45.0).abs() < f32::EPSILON);
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

        // Verify coordinate generation (linspace)
        // x: 0.0 to 2.0 with width 2 -> [0.0, 2.0]
        assert_eq!(sim.dem.x.len(), 2);
        assert_eq!(sim.dem.x[0], 0.0);
        assert_eq!(sim.dem.x[1], 2.0);

        // y: 10.0 to 13.0 with height 3 -> [10.0, 11.5, 13.0]
        assert_eq!(sim.dem.y.len(), 3);
        assert_eq!(sim.dem.y[0], 10.0);
        assert_eq!(sim.dem.y[2], 13.0);
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
    fn test_release_estimation_crown_line_end_to_end() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        // inclined plane draining towards decreasing y
        let dem_data: Vec<f32> = (0..width * height)
            .map(|idx| 100.0 + (idx / width) as f32 * 5.0)
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
    fn test_release_estimation_particle_crown_line_end_to_end() {
        let width = 24;
        let height = 24;
        let cell_size = 5.0;
        // inclined plane draining towards decreasing y
        let dem_data: Vec<f32> = (0..width * height)
            .map(|idx| 100.0 + (idx / width) as f32 * 5.0)
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
        let settings = Settings::default();
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
        assert_eq!(state.iter().filter(|state| state.timestep > 2).count(), 0);
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

        let finished = block_on(sim.run_n_steps(1)).expect("Failed to resume simulation");
        assert_eq!(finished.timestep, advanced.timestep);
        assert_eq!(sim.state, SimulationState::Finished);

        let atomics = block_on(sim.fetch_atomic_values()).expect("Failed to fetch atomic values");
        assert_eq!(atomics.stopped_particles, sim.number_particles);
    }

    #[test_log::test]
    fn test_run_n_steps_clamps_to_max_steps() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        sim.settings.max_steps = 2;

        let result = block_on(sim.run_n_steps(u32::MAX)).expect("Failed to advance simulation");

        assert_eq!(result.timestep, 1);
        assert_eq!(sim.state, SimulationState::Finished);
    }

    #[test_log::test]
    fn test_mpm_velocity_z_is_synthetic_zero() {
        let mut sim = setup_simple_sim(40.0, 3.0);
        sim.settings.sim_model = SimModel::Curvilinear.as_int();
        block_on(sim.prepare()).expect("Failed to prepare MPM simulation");
        let number_particles = sim.number_particles as usize;

        let velocity_z = block_on(sim.fetch_particles_velocity_z())
            .expect("Failed to fetch MPM vertical velocity");

        assert_eq!(velocity_z.len(), number_particles);
        assert!(velocity_z.iter().all(|velocity| *velocity == 0.0));
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

    #[test_log::test]
    fn test_release_hash() {
        let sim = setup_simple_sim(40.0, 4.0);
        assert_eq!(sim.release_hash(), 8805650549818317918);
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
            final_stopped = states.iter().filter(|state| state.stopped).count();
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
