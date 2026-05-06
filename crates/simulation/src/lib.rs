use anyhow::Result;
use compute_core::{
    ComputeOrchestrator, GpuCache, Particle, SimInfo, TextureRgba, TimestepData,
    buffers::{BufferName, TextureName},
    dem::{Bounds, Dem},
    settings::{Settings, SimSettings},
    utils::*,
};
use std::sync::Once;
use web_time::Instant;
static INIT: Once = Once::new();
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

const MAX_VELOCITY_FACTOR: f32 = 1e7;

/// Initializes the global tracing subscriber.
pub fn init_logging() {
    INIT.call_once(|| {
        #[cfg(debug_assertions)]
        let filter = EnvFilter::new(
            "error,simulation=trace,compute_core=trace,data_processor=debug,cli=debug",
        );
        #[cfg(not(debug_assertions))]
        let filter = EnvFilter::new("error,compute_core=info,data_processor=info,cli=info");

        let _ = tracing_subscriber::registry()
            .with(fmt::layer().with_target(false))
            .with(filter)
            .try_init();

        info!("Simulation logging initialized");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SimulationState {
    Uninitialized,
    DemMissing,
    Ready,
    NormalsComputed,
    ReleaseAreasComputed,
    ParticlesInitialized,
    Running,
    Finished,
}

#[allow(dead_code)]
pub struct Simulation {
    orchestrator: ComputeOrchestrator,
    pub settings: SimSettings,
    pub info: SimInfo,
    pub dem_path: String,
    pub dem: Dem,
    pub normals: Vec<f32>,
    pub slope: Vec<f32>,
    pub roughness: Vec<f32>,
    pub cell_count: Vec<u32>,
    pub max_velocity: Vec<f32>,
    release_areas_path: Option<String>,
    release_areas_array: Option<Vec<f32>>,
    sim_info: SimInfo,
    number_particles: u32,
    particles: Vec<Particle>,
    state: SimulationState,
    pub gpu_cache: GpuCache,
}

impl Simulation {
    pub async fn new() -> Result<Self> {
        let orchestrator = ComputeOrchestrator::new().await?;
        Ok(Self {
            orchestrator,
            settings: SimSettings::default(),
            info: SimInfo::default(),
            dem_path: String::new(),
            dem: Dem::default(),
            normals: Vec::new(),
            slope: Vec::new(),
            roughness: Vec::new(),
            cell_count: Vec::new(),
            max_velocity: Vec::new(),
            number_particles: 0,
            particles: Vec::new(),
            state: SimulationState::Uninitialized,
            gpu_cache: GpuCache::default(),
            sim_info: SimInfo::default(),
            release_areas_path: None,
            release_areas_array: None,
        })
    }
    pub fn get_state(&self) -> SimulationState {
        self.state
    }

    pub fn get_gpu_cache_read_count(&self) -> usize {
        self.gpu_cache.read_count
    }

    pub fn get_sim_info(&self) -> SimInfo {
        self.sim_info
    }

    pub fn elevation_threshold(&self) -> f32 {
        self.sim_info.elevation_threshold
    }

    pub async fn create(&mut self, settings: Settings) -> Result<()> {
        let mut timer = Timer::new("Total Simulation Creation Time");
        let (settings_result, dem_result) =
            data_processor::create_sim_settings_and_dem(&settings).await;
        self.settings = settings_result;
        self.dem = dem_result;
        self.dem_path = settings.dem_path.unwrap_or_default().clone();
        self.release_areas_path = settings.release_areas_path.clone();
        self.gpu_cache.reset_all();
        if self.dem.data1d.is_empty() {
            self.state = SimulationState::DemMissing;
        } else {
            self.state = SimulationState::Ready;
            info!(
                "Updated simulation with DEM path: {}\nSettings: {:#?}",
                self.dem_path, self.settings
            );
        }
        timer.checkpoint("Simulation updated/created");
        debug!("{}", timer.get_summary());

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
        self.set_dem(
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
    #[allow(clippy::too_many_arguments)]
    pub fn set_dem(
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
        };

        assert!(
            self.dem.bounds.xmin < self.dem.bounds.xmax,
            "xmin ({}) must be less than or equal to xmax ({})",
            self.dem.bounds.xmin,
            self.dem.bounds.xmax
        );
        assert!(
            self.dem.bounds.ymin < self.dem.bounds.ymax,
            "ymin ({}) must be less than or equal to ymax ({})",
            self.dem.bounds.ymin,
            self.dem.bounds.ymax
        );
        self.settings.set_dem(&self.dem);

        self.state = SimulationState::Ready;
        info!(
            "Updated simulation with DEM path: {}\nSettings: {:#?}",
            self.dem_path, self.settings
        );
        Ok(())
    }

    pub fn set_release_areas(&mut self, release_areas: &[f32]) -> Result<()> {
        self.release_areas_array = Some(release_areas.to_vec());
        self.release_areas_path = None;
        Ok(())
    }

    pub async fn prepare(&mut self) -> Result<()> {
        self.compute_normals().await?;
        let _ = self.get_release_areas().await?;
        self.initialize_particles().await?;
        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        let mut timer = Timer::new("Total Simulation Run Time");
        self.compute_normals().await?;
        timer.checkpoint("Normals computed");
        let _ = self.get_release_areas().await?;
        timer.checkpoint("Release areas loaded");
        if self.number_particles == 0 {
            warn!("No particles to simulate! Check if release areas are correctly defined.");
        } else {
            self.initialize_particles().await?;
            timer.checkpoint("Particles initialized");
            self.compute_particles().await?;

            self.sim_info = self.fetch_sim_info().await?;
        }
        self.state = SimulationState::Finished;

        timer.checkpoint("Simulation finished");
        info!("{}", timer.get_summary());
        Ok(())
    }

    async fn fetch_sim_info(&mut self) -> Result<SimInfo> {
        self.sim_info = self
            .orchestrator
            .read_buffer::<SimInfo>(BufferName::SimInfo)
            .await?
            .first()
            .cloned()
            .unwrap_or_default();
        Ok(self.sim_info)
    }

    async fn compute_normals(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::Ready,
            "DEM and settings must be loaded before running normals shader"
        );
        self.gpu_cache.reset_all();
        self.orchestrator
            .run_analyze_terrain(&self.settings, &self.dem)
            .await?;
        self.state = SimulationState::NormalsComputed;
        Ok(())
    }

    async fn get_release_areas(&mut self) -> Result<u32> {
        assert!(
            self.state >= SimulationState::NormalsComputed,
            "Normals must be computed before loading release areas"
        );
        self.gpu_cache.release_areas = None;
        self.gpu_cache.reset_simulation_result();
        let number_release_cells = match &self.release_areas_path {
            Some(path) => {
                info!("Loading release areas from path: {}", path);
                let data = data_processor::load_release_areas(path)
                    .await
                    .expect("Failed to read PNG at release areas path");
                self.orchestrator
                    .run_load_release_areas(&data, &self.settings)
                    .await?
            }
            None => {
                match &self.release_areas_array {
                    Some(array) => {
                        info!("Loading release areas from provided array");

                        let mut interleaved_data: Vec<f32> = Vec::with_capacity(array.len() * 4);
                        let mut counter: u32 = 0;
                        for r in array.iter() {
                            if *r > 0.1 {
                                counter += 1;
                            }
                            interleaved_data.push(*r); // R
                            interleaved_data.push(0.0); // G
                            interleaved_data.push(0.0); // B
                            interleaved_data.push(0.0); // A
                        }
                        self.orchestrator
                            .write_texture::<f32>(TextureName::ReleaseAreas, &interleaved_data)
                            .expect("Failed to write release areas texture");
                        counter
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
                }
            }
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
        assert!(
            self.state >= SimulationState::ReleaseAreasComputed,
            "Release areas must be computed before initializing particles"
        );
        assert_ne!(
            self.number_particles, 0,
            "Number of particles must be greater than 0 to initialize particles"
        );
        self.gpu_cache.reset_simulation_result();
        // set parameters that depend on the number of particles
        self.orchestrator
            .run_initialize_particles(&self.settings, self.number_particles)
            .await?;
        self.state = SimulationState::ParticlesInitialized;
        Ok(())
    }

    async fn compute_particles(&mut self) -> Result<()> {
        assert!(
            self.state >= SimulationState::ParticlesInitialized,
            "Particles must be initialized before running particle simulation"
        );
        self.gpu_cache.reset_simulation_result();
        self.orchestrator
            .run_compute_particles(&self.settings, self.number_particles)
            .await?;
        self.state = SimulationState::Running;
        info!(
            "Allocated GPU Memory: {:.1} MB",
            self.orchestrator.buffers.get_total_allocated_memory_mb()
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

    async fn get_texture_data_single_channel<T: bytemuck::Pod + Send + Sync>(
        &self,
        name: TextureName,
    ) -> Result<Vec<T>> {
        self.orchestrator.read_texture_single_channel(name).await
    }

    async fn fetch_roughness_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::ReleaseAreasComputed,
            "Release areas must be computed before reading roughness texture"
        );
        if self.gpu_cache.roughness.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.roughness = Some(self.get_texture_data(TextureName::Roughness).await?);
        }
        Ok(self.gpu_cache.roughness.as_ref().unwrap())
    }

    pub async fn get_roughness_aspect(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_roughness_texture().await?.r.clone())
    }

    async fn fetch_slope_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::NormalsComputed,
            "Normals must be computed before reading normals texture"
        );
        if self.gpu_cache.slope.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.slope = Some(self.get_texture_data(TextureName::Slope).await?);
        }
        Ok(self.gpu_cache.slope.as_ref().unwrap())
    }

    pub async fn get_slope_angle(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_slope_texture().await?.r.clone())
    }

    pub async fn get_slope_aspect(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_slope_texture().await?.g.clone())
    }

    async fn fetch_normals_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::NormalsComputed,
            "Normals must be computed before reading normals texture"
        );
        if self.gpu_cache.normals.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.normals = Some(self.get_texture_data(TextureName::Normals).await?);
        }
        Ok(self.gpu_cache.normals.as_ref().unwrap())
    }

    pub async fn get_normals_x(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_normals_texture().await?.r.clone())
    }

    pub async fn get_normals_y(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_normals_texture().await?.g.clone())
    }

    pub async fn get_normals_z(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_normals_texture().await?.b.clone())
    }

    pub async fn get_curvature(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_normals_texture().await?.a.clone())
    }

    pub async fn get_dem_texture(&self) -> Result<Vec<f32>> {
        self.get_texture_data_single_channel(TextureName::Dem).await
    }

    async fn fetch_release_areas_texture(&mut self) -> Result<&TextureRgba<f32>> {
        assert!(
            self.state >= SimulationState::ReleaseAreasComputed,
            "Release areas must be computed before reading release areas texture"
        );
        if self.gpu_cache.release_areas.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.release_areas =
                Some(self.get_texture_data(TextureName::ReleaseAreas).await?);
        }
        Ok(self.gpu_cache.release_areas.as_ref().unwrap())
    }

    pub async fn fetch_release_areas(&mut self) -> Result<Vec<f32>> {
        Ok(self.fetch_release_areas_texture().await?.r.clone())
    }

    pub async fn fetch_max_velocity(&mut self) -> Result<&Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading max velocity"
        );
        if self.gpu_cache.max_velocity.is_none() {
            self.gpu_cache.read_count += 1;
            let data: Vec<u32> = self
                .orchestrator
                .read_buffer(BufferName::VelocityGrid)
                .await?;
            self.gpu_cache.max_velocity = Some(
                data.into_iter()
                    .map(|x| x as f32 / MAX_VELOCITY_FACTOR)
                    .collect(),
            );
        }
        Ok(self.gpu_cache.max_velocity.as_ref().unwrap())
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

    pub async fn fetch_cell_count(&mut self) -> Result<&Vec<u32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading cell count grid"
        );
        if self.gpu_cache.cell_count.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.cell_count = Some(
                self.orchestrator
                    .read_buffer(BufferName::CellCountGrid)
                    .await?,
            );
        }
        Ok(self.gpu_cache.cell_count.as_ref().unwrap())
    }

    pub async fn fetch_particles(&mut self) -> Result<&Vec<Particle>> {
        assert!(
            self.state >= SimulationState::ParticlesInitialized,
            "Simulation must be finished before reading cell count grid"
        );
        if self.gpu_cache.particles.is_none() {
            self.gpu_cache.read_count += 1;
            self.gpu_cache.particles =
                Some(self.orchestrator.read_buffer(BufferName::Particles).await?);
        }
        Ok(self.gpu_cache.particles.as_ref().unwrap())
    }

    pub async fn get_compute_particles_debug(&self) -> Result<Vec<f32>> {
        assert!(
            self.state >= SimulationState::Finished,
            "Simulation must be finished before reading cell count grid"
        );
        self.orchestrator
            .read_buffer(BufferName::OutDebugNormals)
            .await
    }

    /// This function can be used to pre-load all results into the cache, so that subsequent calls to getters will be fast
    pub async fn fetch_results(&mut self) -> Result<()> {
        let start = Instant::now();
        self.fetch_max_velocity().await?;
        self.fetch_cell_count().await?;
        self.fetch_particles().await?;
        self.fetch_timestep_data().await?;
        self.fetch_roughness_texture().await?;
        self.fetch_slope_texture().await?;
        self.fetch_normals_texture().await?;
        self.fetch_release_areas_texture().await?;
        let end = Instant::now();
        trace!(
            "Time taken to fetch all results from GPU: {:?}",
            end - start
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;
    use std::collections::HashSet;
    use std::hash::Hash;
    use std::hash::Hasher;

    const INCLINED_PLANE_PATH: &str = "../../frontend/data/avaframe/avaInclinedPlane.png";
    const RELEASE_TEXTURE_PATH: &str =
        "../../frontend/data/avaframe/avaInclinedPlanereleaseTexture.png";
    const GAR_PATH: &str = "../../frontend/data/avaframe/avaGar.png";
    const GAR_RELEASE_TEXTURE_PATH: &str = "../../frontend/data/avaframe/avaGarreleaseTexture.png";

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
        if std::env::var("GITHUB_ACTIONS").is_ok()
            && (cfg!(target_os = "macos") || cfg!(target_os = "windows"))
        {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let number_cache_elements = 8;
        let number_sim_results_elements = 4;
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create_example(INCLINED_PLANE_PATH)).expect("Failed to create simulation");
        block_on(sim.run()).expect("Failed to run simulation");
        let count_before = sim.get_gpu_cache_read_count();

        // First call: Should trigger a "read" and populate the Option
        block_on(sim.fetch_results()).expect("Failed to get data on first call");
        let first_ref =
            block_on(sim.fetch_particles()).expect("Failed to get particles on first call");
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
            block_on(sim.fetch_particles()).expect("Failed to get particles on second call");
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
            sim.gpu_cache.particles.is_none(),
            "Reset failed: GPU cache particles Option was not cleared"
        );

        // Cache the 4 results again after reset, should trigger reads again
        block_on(sim.fetch_results()).expect("Failed to get data on third call");
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + number_cache_elements + number_sim_results_elements,
            "Expected read_count to increase by {} after third call, but it did not",
            number_cache_elements + number_sim_results_elements
        );

        sim.settings.friction_coefficient = 0.2;
        block_on(sim.run()).expect("Failed to run simulation after changing settings");

        block_on(sim.fetch_results()).expect("Failed to get data on second call");
        assert_eq!(
            sim.get_gpu_cache_read_count(),
            count_before + 2 * number_cache_elements + number_sim_results_elements,
            "Expected read_count to increase by {} after third call, but it did not",
            number_cache_elements + number_sim_results_elements
        );

        let third_ref =
            block_on(sim.fetch_particles()).expect("Failed to get particles on third call");
        let third_state = calculate_hash(&third_ref);
        // hash changed after sim with different settings, confirming cache was reset
        assert_ne!(
            cached_state, third_state,
            "Reset failed: Hash remained the same even after clearing cache"
        );
    }

    #[test_log::test]
    pub fn test_automatic_gpu_cache_reset() {
        if std::env::var("GITHUB_ACTIONS").is_ok()
            && (cfg!(target_os = "macos") || cfg!(target_os = "windows"))
        {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut sim: Simulation = block_on(Simulation::new()).expect("Failed to create Simulation");
        block_on(sim.create_example(INCLINED_PLANE_PATH)).expect("Failed to create simulation");
        assert!(
            sim.gpu_cache.particles.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.normals.is_none()
                && sim.gpu_cache.slope.is_none()
                && sim.gpu_cache.cell_count.is_none()
                && sim.gpu_cache.max_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should start empty"
        );
        block_on(sim.run()).expect("Failed to run simulation");

        assert!(
            sim.gpu_cache.particles.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.normals.is_none()
                && sim.gpu_cache.slope.is_none()
                && sim.gpu_cache.cell_count.is_none()
                && sim.gpu_cache.max_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should stay empty after simulation run (no caching yet)"
        );
        block_on(sim.fetch_results()).expect("Failed to cache results");
        assert!(
            sim.gpu_cache.particles.is_some()
                && sim.gpu_cache.release_areas.is_some()
                && sim.gpu_cache.normals.is_some()
                && sim.gpu_cache.slope.is_some()
                && sim.gpu_cache.cell_count.is_some()
                && sim.gpu_cache.max_velocity.is_some()
                && sim.gpu_cache.timestep_data.is_some(),
            "GPU cache should be fully populated after caching results"
        );

        block_on(sim.compute_normals()).expect("Failed to run normals shader");
        assert!(
            sim.gpu_cache.particles.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.normals.is_none()
                && sim.gpu_cache.slope.is_none()
                && sim.gpu_cache.cell_count.is_none()
                && sim.gpu_cache.max_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should be empty after loading new DEM and running normals shader"
        );

        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.fetch_results()).expect("Failed to cache results");
        block_on(sim.get_release_areas()).expect("Failed to run release shader");

        assert!(
            sim.gpu_cache.particles.is_none()
                && sim.gpu_cache.release_areas.is_none()
                && sim.gpu_cache.normals.is_some()
                && sim.gpu_cache.slope.is_some()
                && sim.gpu_cache.cell_count.is_none()
                && sim.gpu_cache.max_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should be empty"
        );

        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.fetch_results()).expect("Failed to cache results");
        block_on(sim.initialize_particles()).expect("Failed to run initialize particles shader");

        assert!(
            sim.gpu_cache.particles.is_none()
                && sim.gpu_cache.release_areas.is_some()
                && sim.gpu_cache.normals.is_some()
                && sim.gpu_cache.slope.is_some()
                && sim.gpu_cache.cell_count.is_none()
                && sim.gpu_cache.max_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should be empty except for normals and slope"
        );

        block_on(sim.run()).expect("Failed to run simulation");
        block_on(sim.fetch_results()).expect("Failed to cache results");
        block_on(sim.compute_particles()).expect("Failed to run compute particles shader");

        assert!(
            sim.gpu_cache.particles.is_none()
                && sim.gpu_cache.release_areas.is_some()
                && sim.gpu_cache.normals.is_some()
                && sim.gpu_cache.slope.is_some()
                && sim.gpu_cache.cell_count.is_none()
                && sim.gpu_cache.max_velocity.is_none()
                && sim.gpu_cache.timestep_data.is_none(),
            "GPU cache should be empty except for release areas, normals, and slope"
        );
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
        let result = sim.set_dem(
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
            SimulationState::Ready,
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

        block_on(sim.compute_normals()).expect("Failed to compute normals after setting DEM");
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
            sim.set_dem(
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
            sim.set_dem(
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
            sim.set_dem(
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
    fn test_compute() {
        if std::env::var("GITHUB_ACTIONS").is_ok()
            && (cfg!(target_os = "macos") || cfg!(target_os = "windows"))
        {
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
            cfl: Some(0.3),
            max_steps: Some(5000),
            ..Default::default()
        };
        block_on(sim.create(settings)).expect("Failed to create simulation");
        // block_on(sim.create_example(dem_path))
        block_on(sim.run()).expect("Failed to run simulation");
        let debug_buffer: Vec<f32> = block_on(sim.orchestrator.buffers.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::OutDebugNormals,
        ))
        .expect("Failed to read out_debug_normals_buffer");
        log_debug_buffer(&debug_buffer);
        let cell_count = block_on(sim.fetch_cell_count()).expect("Failed to get cell count");
        info!("Cell count max: {:?}", cell_count.max_value().unwrap());
        let max_velocity = block_on(sim.fetch_max_velocity()).expect("Failed to get max velocity");
        info!("Max velocity: {:?}", max_velocity.max_value().unwrap());

        let sim_info: Vec<SimInfo> = block_on(sim.orchestrator.buffers.read_buffer(
            &sim.orchestrator.device,
            &sim.orchestrator.queue,
            BufferName::SimInfo,
        ))
        .expect("Failed to read sim info buffer");
        info!("Read sim info: {:?}", sim_info);

        // particles dont stop, they fall off the DEM
        let particles = block_on(sim.fetch_particles()).expect("Failed to read particles buffer");
        assert_eq!(particles.iter().filter(|&&x| x.stopped > 4000).count(), 0);
        assert_eq!(particles.iter().filter(|&&x| x.stopped < 2000).count(), 0);

        // max velocity should be above 39 m/s
        let max_velocity = block_on(sim.fetch_max_velocity()).expect("Failed to get max velocity");
        assert!(max_velocity.max_value().unwrap() > 41.0);
        assert!(max_velocity.max_value().unwrap() < 42.0);
        info!(
            "Max velocity after simulation: {:?}",
            max_velocity.max_value().unwrap()
        );

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
    fn test_compute_orchestrator_creation() {
        if std::env::var("GITHUB_ACTIONS").is_ok()
            && (cfg!(target_os = "macos") || cfg!(target_os = "windows"))
        {
            println!("Skipping heavy GPU test on CI (macOS/Windows)");
            return;
        }
        let mut orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let (sim_settings, dem) = block_on(data_processor::create_sim_settings_and_dem_from_path(
            INCLINED_PLANE_PATH,
        ));
        block_on(orchestrator.run_analyze_terrain(&sim_settings, &dem))
            .expect("Failed to run normals shader");

        let (slope_angle, slope_aspect, _, _) =
            block_on(orchestrator.read_texture::<f32>(TextureName::Slope)) // get_texture::<f32>("slope"))
                .expect("Failed to get slope texture");
        let (normal_x, normal_y, normal_z, profile_curvature) =
            block_on(orchestrator.read_texture::<f32>(TextureName::Normals))
                .expect("Failed to get normals texture");
        data_processor::save_grid(&dem, "slope_aspect.bin", slope_aspect.clone())
            .expect("Failed to save slope_aspect");
        data_processor::save_grid(&dem, "slope_angle.bin", slope_angle.clone())
            .expect("Failed to save slope_angle");
        data_processor::save_grid(&dem, "profile_curvature.bin", profile_curvature.clone())
            .expect("Failed to save profile_curvature");
        // println!("{}", slope_texture[5].to_f32());
        let debug_buffer: Vec<f32> = block_on(orchestrator.buffers.read_buffer(
            &orchestrator.device,
            &orchestrator.queue,
            BufferName::OutDebugNormals,
        ))
        .expect("Failed to read out_debug_normals_buffer");
        info!("Read out_debug_normals_buffer: {:?}", debug_buffer);
        assert!(
            slope_angle.iter().all(|&x| (x - 34.00012).abs() < 4e-3),
            "Slope angle values are not as expected. Min: {:?}, Max: {:?}\nHist:\n{:?}",
            slope_angle.min_value(),
            slope_angle.max_value(),
            slope_angle.hist_float()
        );
        assert!(slope_aspect.iter().all(|&x| (x - 90.0).abs() < 1e-6));
        let epsilon = 1e-4;
        assert!(normal_x.iter().all(|&x| (x - 0.55919474).abs() < epsilon));
        assert!(normal_y.iter().all(|&x| (x - 0.0).abs() < epsilon));
        assert!(normal_z.iter().all(|&x| (x - 0.82903636).abs() < epsilon));
        assert!(profile_curvature.iter().all(|&x| (x - 0.0).abs() < epsilon));
    }
    #[test_log::test]
    fn test_load_release_areas() {
        let mut orchestrator: ComputeOrchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let (sim_settings, _dem) = block_on(data_processor::create_sim_settings_and_dem_from_path(
            INCLINED_PLANE_PATH,
        ));
        let data = block_on(data_processor::load_release_areas(RELEASE_TEXTURE_PATH))
            .expect("Failed to read release areas");
        info!("Max: {:?}", data.max_value().unwrap());

        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .expect("Failed to create buffers and texture descriptions");
        let number_release_cells: u32 =
            block_on(orchestrator.run_load_release_areas(&data, &sim_settings))
                .expect("Failed to run load_release_areas shader");
        let (release_thickness, _, _, _) =
            block_on(orchestrator.read_texture::<f32>(TextureName::ReleaseAreas))
                .expect("Failed to get release_areas");
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
        let mut orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let (sim_settings, _dem) = block_on(data_processor::create_sim_settings_and_dem_from_path(
            GAR_PATH,
        ));
        let data = block_on(data_processor::load_release_areas(GAR_RELEASE_TEXTURE_PATH))
            .expect("Failed to read PNG");
        info!("Max: {:?}", data.max_value().unwrap());

        orchestrator
            .create_buffers_and_texture_descriptions(&sim_settings)
            .expect("Failed to create buffers and texture descriptions");
        let number_release_cells: u32 =
            block_on(orchestrator.run_load_release_areas(&data, &sim_settings))
                .expect("Failed to run load_release_areas shader");
        let (release_thickness, _, _, _) =
            block_on(orchestrator.read_texture::<f32>(TextureName::ReleaseAreas))
                .expect("Failed to get release_areas");
        info!(
            "Read release_texture: len: {} max: {:?} {:?}",
            release_thickness.len(),
            release_thickness.max_value(),
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
        let mut orchestrator =
            block_on(ComputeOrchestrator::new()).expect("Failed to create ComputeOrchestrator");
        let (mut sim_settings, dem) = block_on(
            data_processor::create_sim_settings_and_dem_from_path(INCLINED_PLANE_PATH),
        );
        sim_settings.released_particles_per_cell = 10;
        info!("Sim settings: {:?}", sim_settings);
        block_on(orchestrator.run_analyze_terrain(&sim_settings, &dem))
            .expect("Failed to run normals shader");
        let data = block_on(data_processor::load_release_areas(RELEASE_TEXTURE_PATH))
            .expect("Failed to read release areas");
        let number_release_cells: u32 =
            block_on(orchestrator.run_load_release_areas(&data, &sim_settings))
                .expect("Failed to run load_release_areas shader");
        block_on(orchestrator.run_initialize_particles(
            &sim_settings,
            number_release_cells * sim_settings.released_particles_per_cell,
        ))
        .expect("Failed to run initialize_particles shader");
        let number_release_particles =
            block_on(orchestrator.read_buffer::<u32>(BufferName::NumberReleaseParticles))
                .expect("Failed to read particle index buffer")[0];
        info!("Number release particles: {}", number_release_particles);
        let particles = block_on(orchestrator.read_buffer::<Particle>(BufferName::Particles))
            .expect("Failed to read particles buffer");
        let particle = particles.first().expect("No particles found");
        for p in particles.iter() {
            assert!(p.position[0] > 100.0);
            assert!(p.position[0] < 400.0);
            assert!(p.position[1] < 1150.0);
            assert!(p.position[1] > 850.0);
            assert!(p.position[2] > 3000.0);
            assert!(p.position[2] < 3350.0);
        }
        assert_eq!(particle.mass, 500.0);
        let unique_values = particles
            .iter()
            .map(|p| {
                [
                    p.position[0].to_bits(),
                    p.position[0].to_bits(),
                    p.position[0].to_bits(),
                ]
            })
            .collect::<HashSet<_>>()
            .len();
        info!(
            "Unique values: {}, {}%",
            unique_values,
            unique_values as f32 / particles.len() as f32 * 100.0
        );
        assert!(
            unique_values as f32 / particles.len() as f32 > 0.98,
            "Duplicate position found in vector"
        );
        info!(
            "Read particles buffer: len: {}, first particle: {:?}",
            particles.len(),
            particles.last()
        );
        let cell_count_grid = block_on(orchestrator.read_buffer::<u32>(BufferName::CellCountGrid))
            .expect("Failed to read cell count grid");

        info!(
            "Read cell count grid: len: {:?}, max value: {:?}",
            cell_count_grid.hist(),
            cell_count_grid.max_value().unwrap()
        );
        assert_eq!(particles.iter().filter(|&&x| x.mass > 0.0).count(), 32450);
        assert!(cell_count_grid.max_value().unwrap() <= 12); // 10 particles per cell + some edge cases
        assert!(cell_count_grid.hist().get(&0).unwrap() > &398150); // most cells dont have particles
    }
}
