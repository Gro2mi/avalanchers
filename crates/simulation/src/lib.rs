use anyhow::{Result, bail};
use compute_core::{
    ComputeOrchestrator, GpuCache, SimInfo, TextureRgba, TimestepData,
    buffers::{AtomicValues, BufferName, TextureName},
    dem::{Bounds, Dem},
    post_processing::*,
    settings::{Settings, SimSettings},
    utils::*,
};
#[cfg(target_arch = "wasm32")]
use data_processor::zarr_writer::{ResultGrids, ZarrEntry};
// use data_processor;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Once;
use web_time::Instant;
static INIT: Once = Once::new();
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initializes the global tracing subscriber.
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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SimulationState {
    Uninitialized,
    DemMissing,
    DemLoaded,
    TerrainAnalyzed,
    ReleaseAreasComputed,
    ParticlesInitialized,
    Running,
    Finished,
    PostProcessed,
    Evaluated,
}

pub struct Simulation {
    orchestrator: ComputeOrchestrator,
    pub settings: SimSettings,
    pub dem_path: String,
    pub dem: Dem,
    pub output_path: String,
    release_areas_path: Option<String>,
    release_areas_array: Option<Vec<f32>>,
    sim_info: SimInfo,
    number_particles: u32,
    state: SimulationState,
    pub gpu_cache: GpuCache,
    pub ava_mask: Vec<bool>,
    #[cfg(not(target_arch = "wasm32"))]
    output: Option<data_processor::output::Output>,
}

pub struct SimulationLoadResult {
    pub settings: SimSettings,
    pub dem: Dem,
    pub dem_path: String,
    pub release_areas_path: Option<String>,
    pub batch_compute_steps: Option<u32>,
    pub output_path: Option<String>,
}

impl Simulation {
    pub async fn new() -> Result<Self> {
        Self::new_with_gpu(None).await
    }
    pub async fn new_with_gpu(gpu: Option<String>) -> Result<Self> {
        timer_new();
        let orchestrator = ComputeOrchestrator::new_with_gpu(gpu).await?;
        Ok(Self {
            orchestrator,
            settings: SimSettings::default(),
            output_path: "avalanchers".to_string(),
            dem_path: String::new(),
            dem: Dem::default(),
            number_particles: 0,
            state: SimulationState::Uninitialized,
            gpu_cache: GpuCache::default(),
            sim_info: SimInfo::default(),
            release_areas_path: None,
            release_areas_array: None,
            ava_mask: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            output: None,
        })
    }

    pub async fn new_with_settings(settings: Settings) -> Result<Self> {
        let (simulation, simulation_data) =
            futures::join!(Simulation::new(), Simulation::load_data(&settings),);

        let mut simulation = simulation?;
        let simulation_data = simulation_data?;

        simulation.apply_data(simulation_data);
        Ok(simulation)
    }

    pub fn get_state(&self) -> SimulationState {
        self.state
    }

    /// GPU device, queue and buffers backing the simulation. A renderer can bind these
    /// directly to visualise the simulation without copying data back to the CPU.
    pub fn orchestrator(&self) -> &ComputeOrchestrator {
        &self.orchestrator
    }

    pub fn number_particles(&self) -> u32 {
        self.number_particles
    }

    pub fn release_hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        if let Some(release_areas_array) = &self.release_areas_array {
            for val in release_areas_array.iter() {
                val.to_bits().hash(&mut s);
            }
        }
        s.finish()
    }

    pub fn get_gpu_cache_read_count(&self) -> usize {
        self.gpu_cache.read_count
    }

    pub fn elevation_threshold(&self) -> f32 {
        self.sim_info.elevation_threshold
    }

    pub async fn load_data(settings: &Settings) -> Result<SimulationLoadResult> {
        timer_checkpoint("Start create");
        let (settings_result, dem_result, _outline) =
            data_processor::create_sim_settings_and_dem(settings).await?;

        timer_checkpoint("Load settings");

        Ok(SimulationLoadResult {
            settings: settings_result,
            dem: dem_result,
            batch_compute_steps: settings.batch_compute_steps,
            dem_path: settings.dem_path.clone().unwrap_or_default(),
            release_areas_path: settings.release_areas_path.clone(),
            output_path: settings.output_path.clone(),
        })
    }

    pub fn apply_data(&mut self, data: SimulationLoadResult) {
        self.settings = data.settings;
        if let Some(batch_steps) = data.batch_compute_steps {
            self.orchestrator.batch_compute_steps = batch_steps;
        }

        self.dem = data.dem;
        self.dem_path = data.dem_path;
        self.output_path = data
            .output_path
            .unwrap_or_else(|| "avalanchers.zarr".to_string());
        self.release_areas_path = data.release_areas_path;

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

    pub async fn create(&mut self, settings: Settings) -> Result<()> {
        let data = Self::load_data(&settings).await?;
        self.apply_data(data);
        Ok(())
    }

    pub async fn create_default(&mut self, dem_path: &str) -> Result<()> {
        let settings = Settings {
            dem_path: Some(dem_path.to_string()),
            ..Settings::default()
        };
        self.create(settings).await?;
        Ok(())
    }

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

    pub async fn prepare(&mut self) -> Result<()> {
        self.analyze_terrain().await?;
        let _ = self.load_release_areas().await?;
        self.initialize_particles().await?;
        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        self.analyze_terrain().await?;
        timer_checkpoint("Terrain analyzed");
        let _ = self.load_release_areas().await?;
        timer_checkpoint("Release areas loaded");
        if self.number_particles == 0 {
            warn!("No particles to simulate! Check if release areas are correctly defined.");
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

    pub async fn post_process(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before post-processing results"
        );
        let threshold = 0.01;
        let (peak_flow_thickness, ava_mask) = mask_threshold_and_biggest_blob(
            &self.fetch_peak_flow_thickness().await?,
            self.dem.width,
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

    pub async fn evaluate(&mut self) -> Result<(f32, f32, f32, f32)> {
        assert!(
            self.state >= SimulationState::PostProcessed,
            "Simulation must be post-processed before evaluation"
        );
        let iou = 0.0;
        let (horizontal_distance, vertical_drop) = self
            .dem
            .get_elevation_extrema_distance_and_drop(&self.ava_mask);
        let velocities = self.fetch_peak_velocity().await?;
        let peak_velocity = velocities.iter().copied().reduce(f32::max).unwrap_or(0.0);
        self.state = SimulationState::Evaluated;
        Ok((iou, horizontal_distance, vertical_drop, peak_velocity))
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn save_with_path(&mut self, path: &str) -> Result<()> {
        self.output_path = path.to_string();
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

        let release_areas = self.fetch_release_areas().await?;
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
        let release_areas = self.fetch_release_areas().await?;

        self.fetch_peak_velocity().await?;
        self.fetch_peak_flow_thickness().await?;
        timer_checkpoint("Peak data fetched");
        let travel_length = 1000.0;
        let travel_angle = 25.0;
        let center_of_mass_x = vec![1000.0, 1100.0, 900.0];
        let center_of_mass_y = vec![1000.0, 1100.0, 900.0];

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
                20.0,    //aspect_release_value: f32,
                10000.0, //release_volume: f32,
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

    pub async fn fetch_sim_info(&mut self) -> Result<SimInfo> {
        self.sim_info = self
            .orchestrator
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await?
            .first()
            .cloned()
            .unwrap_or_default();
        Ok(self.sim_info)
    }

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

    async fn analyze_terrain(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::DemLoaded,
            "DEM and settings must be loaded before running normals shader"
        );
        self.gpu_cache.reset_all();
        self.orchestrator
            .run_analyze_terrain(&self.settings, &self.dem)
            .await?;
        self.state = SimulationState::TerrainAnalyzed;
        Ok(())
    }

    async fn load_release_areas(&mut self) -> Result<u32> {
        assert!(
            self.state >= SimulationState::TerrainAnalyzed,
            "Terrain must be analyzed before loading release areas"
        );
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
                    .expect("Failed to read PNG at release areas path");
                self.print_grid(&data, 60, 30);
                self.orchestrator
                    .write_buffer(BufferName::ReleaseAreas, &data)
                    .await?;
                data.iter().filter(|&&x| x > 1e-3).count() as u32
            }
            None => match &self.release_areas_array {
                Some(data) => {
                    info!("Loading release areas from provided array");
                    self.orchestrator
                        .write_buffer(BufferName::ReleaseAreas, data)
                        .await?;
                    data.iter().filter(|&&x| x > 1e-3).count() as u32
                }
                None => {
                    info!("Computing release areas from DEM");
                    self.orchestrator
                        .run_compute_roughness(&self.settings)
                        .await?;
                    self.orchestrator
                        .run_compute_release_areas(&self.settings)
                        .await?
                }
            },
        };
        self.number_particles = number_release_cells * self.settings.released_particles_per_cell;
        self.state = SimulationState::ReleaseAreasComputed;
        info!(
            "Number of release cells: {} of {} ({:.1}%)",
            number_release_cells,
            self.dem.width * self.dem.height,
            (number_release_cells as f64 / (self.dem.width * self.dem.height) as f64 * 100.0)
        );
        Ok(number_release_cells)
    }

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
            .run_initialize_particles(&self.settings, self.number_particles)
            .await?;
        self.state = SimulationState::ParticlesInitialized;
        Ok(())
    }

    /// Advances the particle simulation. Requires [`Simulation::prepare`] first, and unlike
    /// [`Simulation::run`] it leaves the existing GPU buffers in place so anything already
    /// bound to them stays valid.
    pub async fn compute_particles(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::ParticlesInitialized,
            "Particles must be initialized before running particle simulation"
        );
        self.gpu_cache.reset_simulation_result();
        self.orchestrator
            .run_sim(
                &self.settings,
                self.number_particles,
                self.dem.minimum_elevation,
            )
            .await?;
        self.state = SimulationState::Running;
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
            self.orchestrator
                .read_texture(name)
                .await
                .expect("Failed to read texture"),
        ))
    }

    pub async fn fetch_roughness(&mut self) -> Result<&Vec<f32>> {
        assert!(
            self.state >= SimulationState::ReleaseAreasComputed,
            "Release areas must be computed before reading roughness texture"
        );
        if self.gpu_cache.roughness.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.roughness =
                Some(self.orchestrator.read_buffer(BufferName::Roughness).await?);
        }
        Ok(self.gpu_cache.roughness.as_ref().unwrap())
    }

    pub async fn fetch_peak_flow_thickness(&mut self) -> Result<Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading peak flow thickness buffer"
        );
        if self.gpu_cache.peak_flow_thickness.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.peak_flow_thickness = Some(
                self.orchestrator
                    .read_buffer(BufferName::GridPeakFlowThickness)
                    .await?,
            );
        }
        Ok(self.gpu_cache.peak_flow_thickness.clone().unwrap())
    }

    pub async fn fetch_slope_angle(&mut self) -> Result<Vec<f32>> {
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
        Ok(self.gpu_cache.slope_angle.clone().unwrap())
    }

    pub async fn fetch_slope_aspect(&mut self) -> Result<Vec<f32>> {
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
        Ok(self.gpu_cache.slope_aspect.clone().unwrap())
    }

    async fn fetch_terrain_geometry_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::TerrainAnalyzed,
            "Terrain geometry must be computed before reading terrain geometry texture"
        );
        if self.gpu_cache.terrain_geometry.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.terrain_geometry =
                Some(self.get_texture_data(TextureName::TerrainGeometry).await?);
        }
        Ok(self.gpu_cache.terrain_geometry.as_ref().unwrap())
    }
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

    pub async fn get_slope_gravity(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let g_x = self.fetch_terrain_curvature().await?.a.clone();
        let g_y = self.fetch_terrain_geometry_texture().await?.a.clone();
        Ok((g_x, g_y))
    }

    pub async fn get_curvature(&mut self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let k_x = self.fetch_terrain_curvature().await?.r.clone();
        let k_y = self.fetch_terrain_curvature().await?.g.clone();
        let k_xy = self.fetch_terrain_curvature().await?.b.clone();
        Ok((k_x, k_y, k_xy))
    }

    pub async fn get_terrain_geometry_x(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_terrain_geometry_texture().await?.r.clone())
    }

    pub async fn get_terrain_geometry_y(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_terrain_geometry_texture().await?.g.clone())
    }

    pub async fn get_terrain_geometry_z(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_terrain_geometry_texture().await?.b.clone())
    }

    pub async fn fetch_release_areas(&mut self) -> Result<Vec<f32>> {
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
        Ok(self.gpu_cache.release_areas.clone().unwrap())
    }

    pub async fn fetch_peak_velocity(&mut self) -> Result<&Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading peak velocity"
        );
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

    pub async fn fetch_timestep_data(&mut self) -> Result<&TimestepData> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must run and be finished before reading timestep data"
        );
        if self.gpu_cache.timestep_data.is_none() {
            self.gpu_cache.read_count += 1;
            let full_data = self
                .orchestrator
                .read_buffer(BufferName::TimestepData)
                .await?;

            let data_aos: Vec<_> = full_data.into_iter().step_by(3).collect();

            self.gpu_cache.timestep_data =
                Some(TimestepData::from_aos(&data_aos, self.settings.cell_size));
        }
        Ok(self.gpu_cache.timestep_data.as_ref().unwrap())
    }

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

    pub async fn fetch_particles_stopped(&mut self) -> Result<&Vec<u32>> {
        if self.state < SimulationState::ParticlesInitialized {
            bail!("Simulation must be initialized before reading particles");
        }
        if self.gpu_cache.particles_stopped.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.particles_stopped = Some(
                self.orchestrator
                    .read_buffer(BufferName::ParticlesStopped)
                    .await?,
            );
        }
        Ok(self.gpu_cache.particles_stopped.as_ref().unwrap())
    }

    pub async fn fetch_particles_all(&mut self) -> Result<()> {
        self.fetch_particles_position().await?;
        self.fetch_particles_velocity().await?;
        self.fetch_particles_mass().await?;
        self.fetch_particles_elevation().await?;
        self.fetch_particles_stopped().await?;
        Ok(())
    }

    pub async fn get_total_mass(&mut self) -> Result<f32> {
        let particles_mass = self.fetch_particles_mass().await?;
        let mass_total: f32 = particles_mass.iter().sum();
        Ok(mass_total)
    }

    pub async fn get_total_volume(&mut self) -> Result<f32> {
        Ok(self.get_total_mass().await? / self.settings.density)
    }

    pub async fn get_compute_particles_debug(&self) -> Result<Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading cell count grid"
        );
        self.orchestrator.read_buffer(BufferName::Debug).await
    }

    /// This function can be used to pre-load all results into the cache, so that subsequent calls to getters will be fast
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

    pub fn print_grid(&self, grid: &[f32], max_w: usize, max_h: usize) {
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
    }
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

    #[test]
    fn test_init_logging_idempotent() {
        // Call it once
        init_logging();

        // Call it again - it should not panic or error because of .call_once()
        init_logging();
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
            block_on(sim.fetch_particles_stopped()).expect("Failed to get particles on first call");
        let uncached_state = calculate_hash(&first_ref);
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + number_cache_elements,
            "Expected read_count to increase by {} after first call, but it did not.",
            number_cache_elements
        );

        // Second call: Should return the cached value
        block_on(sim.fetch_results()).expect("Failed to get data on second call");
        let second_ref = block_on(sim.fetch_particles_stopped())
            .expect("Failed to get particles on second call");
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
            block_on(sim.fetch_particles_stopped()).expect("Failed to get particles on third call");
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
            &create_slope(4, 4, cell_size, slope_angle), // dem_data
            4,                                           // width
            4,                                           // height
            cell_size,                                   // cell_size
        )
        .expect("Failed to set DEM");
        sim.set_release_areas(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        .expect("Failed to set release areas");
        sim
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
    fn test_compute_simple() {
        let slope_angle: f32 = 40.0;
        let cell_size: f32 = 3.0;
        let mut sim: Simulation = setup_simple_sim(slope_angle, cell_size);
        block_on(sim.run()).expect("Failed to run simulation");
        assert_eq!(sim.state, SimulationState::Finished);
        let sim_info = block_on(sim.fetch_sim_info()).expect("Failed to fetch sim info");
        info!("Sim info: {:?}", sim_info);
        assert_eq!(sim.elevation_threshold(), 99.9);
        assert!(sim_info.timestep < 10);
        let atomics = block_on(sim.fetch_atomic_values()).expect("Failed to fetch sim info");
        info!("Atomic values: {:?}", atomics);
        assert_eq!(atomics.number_release_particles, 16);
        assert_eq!(atomics.stopped_particles, 16);
        let stopped = block_on(sim.fetch_particles_stopped()).expect("Failed to fetch particles");
        assert_eq!(stopped.iter().filter(|&&x| x > 10).count(), 0);
        assert_eq!(stopped.iter().filter(|&&x| x == 0).count(), 0);
        for p in stopped.iter() {
            info!("{:?}", p);
        }
        let cell_area = cell_size * cell_size;
        assert!(
            (cell_area * (2 * 200) as f32 / slope_angle.to_radians().cos()
                - block_on(sim.get_total_mass()).expect("Failed to get total mass"))
            .abs()
                < 1e-2
        );
        assert!(
            (cell_area / slope_angle.to_radians().cos() * 2 as f32
                - block_on(sim.get_total_volume()).expect("Failed to get total volume"))
            .abs()
                < 1e-2
        );
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

        let sim_info: Vec<SimInfo> = block_on(sim.orchestrator.resources.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::SimInfo,
        ))
        .expect("Failed to read sim info buffer");
        info!("Read sim info: {:?}", sim_info);
        // particles dont stop, they fall off the DEM
        let stopped =
            block_on(sim.fetch_particles_stopped()).expect("Failed to read particles buffer");
        info!(
            "Min step particle stopped: {}",
            stopped.min_value().unwrap()
        );
        info!(
            "Max step particle stopped: {}",
            stopped.max_value().unwrap()
        );
        // TODO fix this test
        assert_eq!(stopped.iter().filter(|&&x| x > 4900).count(), 0);
        println!(
            "Particles stopped at step 0: {}",
            stopped.iter().filter(|&&x| x == 0).count()
        );
        assert!(stopped.iter().filter(|&&x| x == 0).count() < 20);

        let max_velocity = block_on(sim.fetch_peak_velocity()).expect("Failed to get max velocity");

        info!(
            "Max velocity after simulation: {:.2} m/s",
            max_velocity.max_value().unwrap(),
        );
        assert!(max_velocity.max_value().unwrap() > 41.0);
        assert!(max_velocity.max_value().unwrap() < 42.0);

        let max_steps = sim.settings.max_steps as usize;
        let timestep_data =
            block_on(sim.fetch_timestep_data()).expect("Failed to read timestep data buffer");
        let timesteps = timestep_data.position.len();
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

            assert!(
                pos_curr > pos_prev,
                "Position X did not increase at step {}: {} -> {}",
                i,
                pos_prev,
                pos_curr
            );
        }
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
        let slope_angle = block_on(sim.fetch_slope_angle()).expect("Failed to fetch slope angle");
        let slope_aspect =
            block_on(sim.fetch_slope_aspect()).expect("Failed to fetch slope aspect");
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
        let slope_angle = block_on(sim.fetch_slope_angle()).expect("Failed to fetch slope angle");
        let slope_aspect =
            block_on(sim.fetch_slope_aspect()).expect("Failed to fetch slope aspect");
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
    fn test_fetch_peak_flow_before_finish_panics() {
        let mut sim = block_on(Simulation::new()).expect("Failed to create Simulation");
        // Do not run simulation; directly call fetch_peak_flow_thickness and expect assertion
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on(sim.fetch_peak_flow_thickness()).unwrap();
        }));
        assert!(res.is_err());
    }

    #[test]
    fn test_fetch_timestep_data_before_finish_panics() {
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
        ];
        let width = 4;
        let height = 4;
        sim.print_grid(&grid, width, height);
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
        let release_hash = sim.release_hash();
        assert_eq!(release_hash, 8163359721371807571);
    }
}
