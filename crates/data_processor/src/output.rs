use compute_core::dem::Dem;
use compute_core::settings::SimSettings;
use half::f16;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use zarrs::array::ArraySubset;
use zarrs::array::BytesToBytesCodecTraits;
use zarrs::array::{
    Array, ArrayBuilder, FillValue,
    codec::bytes_to_bytes::blosc::{
        BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    },
};
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::StorageError;
use zarrs_filesystem::FilesystemStore;

use zarrs_data_type::DataType;

const FORMAT_VERSION: &str = "0.1.0";

#[derive(thiserror::Error, Debug)]
pub enum OutputError {
    #[error("Run ID is out of bounds: got {0}, expected max {1}")]
    InvalidRunID(u64, u64),

    #[error("Site already exists: {0}")]
    SiteAlreadyExists(String),

    #[error("Site not found: {0}")]
    SiteNotFound(String),

    #[error("Scenario already exists: {0}")]
    ScenarioAlreadyExists(String),

    #[error("Scenario not found: {0}")]
    ScenarioNotFound(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("TIFF decoding error: {0}")]
    Tiff(#[from] tiff::TiffError),

    #[error("Zarr storage error: {0}")]
    Zarr(#[from] zarrs::array::ArrayError),

    #[error("Zarr creation error: {0}")]
    ZarrCreate(#[from] zarrs::array::ArrayCreateError),

    #[error("Missing or invalid data: {0}")]
    InvalidData(String),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Invalid tile shape: got {0:?}")]
    InvalidShape((usize, usize)),

    #[error(transparent)]
    ZarrDimensionality(#[from] zarrs::array::IncompatibleDimensionalityError),

    #[error(transparent)]
    Store(#[from] StorageError),

    #[error(transparent)]
    GroupCreate(#[from] zarrs::group::GroupCreateError),

    #[error(transparent)]
    PluginCreate(#[from] zarrs_plugin::PluginCreateError),

    #[error(transparent)]
    FilesystemStoreCreate(#[from] zarrs::filesystem::FilesystemStoreCreateError),

    #[error(transparent)]
    ArraySubset(#[from] zarrs::array::ArraySubsetError),
}

#[allow(dead_code)]
pub struct Output {
    path: PathBuf,
    store: Arc<FilesystemStore>,
    blosc_f16: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    blosc_f32: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    pub scenarios: HashMap<String, Scenario>,
}
#[allow(dead_code)]
impl Output {
    pub fn new(path: &str) -> Result<Self, OutputError> {
        let path = PathBuf::from(path);
        let path = if path.extension().is_some_and(|ext| ext == "zarr") {
            path
        } else {
            path.with_extension("zarr")
        };
        let store = Arc::new(FilesystemStore::new(&path)?);
        if path.is_dir() {
            debug!("Zarr store already exists, skipping creation");
        } else {
            let mut root_group = GroupBuilder::new().build(store.clone(), "/")?;
            let global_attrs = json!({
                "title": "Avalanchers Simulation Output",
                "conventions": "CF-1.8",
                "source": "avalanchers",
                "avalanchers_version": env!("CARGO_PKG_VERSION"),
                "avalanchers_repo": "https://github.com/Gro2mi/avalanchers",
                "avalanchers_format": FORMAT_VERSION,

            });
            root_group
                .attributes_mut()
                .extend(global_attrs.as_object().unwrap().clone());
            root_group.store_metadata()?;
        }
        let blosc_level = 1;
        let blosc_f32: Vec<Arc<dyn BytesToBytesCodecTraits>> = vec![Arc::new(BloscCodec::new(
            BloscCompressor::Zstd,
            BloscCompressionLevel::try_from(blosc_level).expect("Invalid compression level"),
            None, // automatic blocksize
            BloscShuffleMode::BitShuffle,
            Some(4), // f32 = 4 bytes
        )?)];

        let blosc_f16: Vec<Arc<dyn BytesToBytesCodecTraits>> = vec![Arc::new(BloscCodec::new(
            BloscCompressor::Zstd,
            BloscCompressionLevel::try_from(blosc_level).expect("Invalid compression level"),
            None, // automatic blocksize
            BloscShuffleMode::BitShuffle,
            Some(2), // f16 = 2 bytes
        )?)];

        Ok(Self {
            path,
            store,
            blosc_f16,
            blosc_f32,
            scenarios: HashMap::new(),
        })
    }

    pub fn site_exists(&self, site_name: &str) -> bool {
        self.path.join(site_name).is_dir()
    }

    pub fn scenario_exists(&self, site_name: &str, scenario_name: &str) -> bool {
        self.path.join(site_name).join(scenario_name).is_dir()
    }

    pub fn add_new_site(&mut self, site_name: &str, dem: &Dem) -> Result<(), OutputError> {
        if self.site_exists(site_name) {
            return Err(OutputError::SiteAlreadyExists(site_name.to_string()));
        }

        let mut y_coords = dem.y.clone();
        let x_coords = &dem.x;
        let dem_data = &dem.data1d;

        y_coords.reverse();
        let ylen = y_coords.len() as u64;
        let xlen = x_coords.len() as u64;
        let _size = xlen * ylen;

        let site_group_path = format!("/{}", site_name);
        let mut site_group = GroupBuilder::new().build(self.store.clone(), &site_group_path)?;
        site_group.attributes_mut().extend(
            json!({
                "dem_width": dem.width,
                "dem_height": dem.height,
                "dem_cell_size": dem.cell_size,
                "dem_map_factor": dem.map_factor,
                "dem_minimum_elevation": dem.minimum_elevation,
                "dem_source": dem.source,
                "dem_projection": dem.projection,
                "dem_hash": format!("{:x}", dem.calculate_hash()),
                "dem_bounds": {
                    "xmin": dem.bounds.xmin,
                    "xmax": dem.bounds.xmax,
                    "ymin": dem.bounds.ymin,
                    "ymax": dem.bounds.ymax
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        site_group.store_metadata()?;

        let mut y = zarrs::array::ArrayBuilder::new(
            vec![ylen],
            vec![ylen],
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["y"].into())
        .build(self.store.clone(), &format!("{}/y", site_group_path))?;
        #[allow(clippy::single_range_in_vec_init)]
        y.store_chunks(&[0..1], y_coords)?;
        y.attributes_mut().extend(
            json!({
                "standard_name": "y",
                "long_name": "Northing",
                "units": "m",
                "axis": "Y"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        y.store_metadata()?;

        let mut x = zarrs::array::ArrayBuilder::new(
            vec![xlen],
            vec![xlen],
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["x"].into())
        .build(self.store.clone(), &format!("{}/x", site_group_path))?;
        x.attributes_mut().extend(
            json!({
                "standard_name": "x",
                "long_name": "Easting",
                "units": "m",
                "axis": "X"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        x.store_metadata()?;
        #[allow(clippy::single_range_in_vec_init)]
        x.store_chunks(&[0..1], x_coords)?;

        let mut dem_array_builder = ArrayBuilder::new(
            vec![ylen, xlen],
            vec![ylen, xlen],
            zarrs::array::data_type::float32(),
            FillValue::from(0.0_f32),
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["y", "x"].into())
        .build(self.store.clone(), &format!("{}/dem", site_group_path))?;
        dem_array_builder.attributes_mut().extend(
            json!({
                "standard_name": "elevation",
                "long_name": "Digital Elevation Model",
                "units": "m",
                "source": dem.source,
                "projection": dem.projection,
                "cell_size": dem.cell_size,
                "map_factor": dem.map_factor,

                "width": dem.width,
                "height": dem.height,
                "hash": format!("{:x}", dem.calculate_hash()),
                "bounds": {
                    "xmin": dem.bounds.xmin,
                    "xmax": dem.bounds.xmax,
                    "ymin": dem.bounds.ymin,
                    "ymax": dem.bounds.ymax
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        dem_array_builder.store_metadata()?;

        let dem_array = Array::open(self.store.clone(), &format!("/{}/dem", site_name))?;
        dem_array.store_chunk(&[0, 0], dem_data)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_new_scenario(
        &mut self,
        site_name: &str,
        scenario_name: &str,
        max_number_runs: u64,
        max_timesteps: u64,
        release_area: &[f32],
        aspect_release_value: f32,
        release_volume: f32,
        mut y_coords: Vec<f32>,
        x_coords: &[f32],
    ) -> Result<(), OutputError> {
        let scenario_key = format!("{site_name}/{scenario_name}");
        if self.scenarios.contains_key(&scenario_key) {
            return Err(OutputError::ScenarioAlreadyExists(
                scenario_name.to_string(),
            ));
        }
        let scenario_group_path = format!("/{}/{}", site_name, scenario_name);

        y_coords.reverse();
        let ylen = y_coords.len() as u64;
        let xlen = x_coords.len() as u64;
        let size = xlen * ylen;

        let mut y = zarrs::array::ArrayBuilder::new(
            vec![ylen],
            vec![ylen],
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["y"].into())
        .build(self.store.clone(), &format!("{}/y", scenario_group_path))?;
        #[allow(clippy::single_range_in_vec_init)]
        y.store_chunks(&[0..1], y_coords)?;
        y.attributes_mut().extend(
            json!({
                "standard_name": "y",
                "long_name": "Northing",
                "units": "m",
                "axis": "Y"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        y.store_metadata()?;

        let mut x = zarrs::array::ArrayBuilder::new(
            vec![xlen],
            vec![xlen],
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["x"].into())
        .build(self.store.clone(), &format!("{}/x", scenario_group_path))?;
        x.attributes_mut().extend(
            json!({
                "standard_name": "x",
                "long_name": "Easting",
                "units": "m",
                "axis": "X"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        x.store_metadata()?;
        #[allow(clippy::single_range_in_vec_init)]
        x.store_chunks(&[0..1], x_coords)?;

        if !release_area.is_empty() && release_area.len() != size as usize {
            return Err(OutputError::InvalidData(format!(
                "Release area has incorrect size: expected {}, got {}",
                size,
                release_area.len()
            )));
        }

        let chunk_timestep = std::cmp::min(200, max_timesteps);
        let mut scenario_group =
            GroupBuilder::new().build(self.store.clone(), &scenario_group_path)?;
        scenario_group.attributes_mut().extend(
            json!({
                "aspect_release_degrees": aspect_release_value,
                "release_volume_m3": release_volume,
                "number_of_runs": 0,
                "max_number_runs": max_number_runs,
                "max_timesteps": max_timesteps,
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        scenario_group.store_metadata()?;

        let mut runs = zarrs::array::ArrayBuilder::new(
            vec![max_number_runs],
            vec![max_number_runs],
            zarrs::array::data_type::uint64(),
            0,
        )
        .dimension_names(["run"].into())
        .build(self.store.clone(), &format!("{}/run", scenario_group_path))?;
        runs.attributes_mut().extend(
            json!({
                "standard_name": "run",
                "long_name": "Run ID",
                "units": "-",
                "axis": ""
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        runs.store_metadata()?;
        let run_ids: Vec<u64> = (0..max_number_runs).collect();
        #[allow(clippy::single_range_in_vec_init)]
        runs.store_chunks(&[0..1], run_ids)?;

        let mut timesteps = zarrs::array::ArrayBuilder::new(
            vec![max_timesteps],
            vec![max_timesteps],
            zarrs::array::data_type::uint64(),
            0,
        )
        .dimension_names(["timestep"].into())
        .build(
            self.store.clone(),
            &format!("{}/timestep", scenario_group_path),
        )?;
        timesteps.attributes_mut().extend(
            json!({
                "standard_name": "timestep",
                "long_name": "Timestep",
                "units": "-",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        timesteps.store_metadata()?;
        let timestep_data: Vec<u64> = (0..max_timesteps).collect();
        #[allow(clippy::single_range_in_vec_init)]
        timesteps.store_chunks(&[0..1], timestep_data)?;

        let grid_shape = vec![max_number_runs, ylen, xlen];
        let grid_chunks = vec![1, ylen, xlen];

        let mut peak_flow_velocity = ArrayBuilder::new(
            grid_shape.clone(),
            grid_chunks.clone(),
            zarrs::array::data_type::float16(),
            FillValue::from(f16::from_f32(0.0)),
        )
        .bytes_to_bytes_codecs(self.blosc_f16.clone())
        .dimension_names(["run", "y", "x"].into())
        .build(
            self.store.clone(),
            &format!("{}/peak_flow_velocity", scenario_group_path),
        )?;
        peak_flow_velocity.attributes_mut().extend(
            json!({
                "standard_name": "pfv",
                "long_name": "Peak Flow Velocity",
                "units": "m/s",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        peak_flow_velocity.store_metadata()?;

        let mut peak_flow_thickness = ArrayBuilder::new(
            grid_shape.clone(),
            grid_chunks.clone(),
            zarrs::array::data_type::float16(),
            FillValue::from(f16::from_f32(0.0)),
        )
        .bytes_to_bytes_codecs(self.blosc_f16.clone())
        .dimension_names(["run", "y", "x"].into())
        .build(
            self.store.clone(),
            &format!("{}/peak_flow_thickness", scenario_group_path),
        )?;
        peak_flow_thickness.attributes_mut().extend(
            json!({
                "standard_name": "pft",
                "long_name": "Peak Flow Thickness",
                "units": "m",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        peak_flow_thickness.store_metadata()?;

        let mut travel_length = zarrs::array::ArrayBuilder::new(
            vec![max_number_runs],
            vec![1],
            zarrs::array::data_type::float32(),
            0,
        )
        .dimension_names(["run"].into())
        .build(
            self.store.clone(),
            &format!("{}/travel_length", scenario_group_path),
        )?;
        travel_length.attributes_mut().extend(
            json!({
                "standard_name": "travel_length",
                "long_name": "Travel length",
                "units": "m",
                "description": "Travel length following the path of the center of mass of the avalanche"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        travel_length.store_metadata()?;

        let mut travel_angle = zarrs::array::ArrayBuilder::new(
            vec![max_number_runs],
            vec![1],
            zarrs::array::data_type::float32(),
            0,
        )
        .dimension_names(["run"].into())
        .build(
            self.store.clone(),
            &format!("{}/travel_angle", scenario_group_path),
        )?;
        travel_angle.attributes_mut().extend(
            json!({
                "standard_name": "travel_angle",
                "long_name": "Travel angle",
                "units": "degrees",
                "description": "Travel angle of the avalanche following the path of the center of mass of the avalanche"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        travel_angle.store_metadata()?;

        let mut mu = zarrs::array::ArrayBuilder::new(
            vec![max_number_runs],
            vec![1],
            zarrs::array::data_type::float32(),
            0,
        )
        .dimension_names(["run"].into())
        .build(self.store.clone(), &format!("{}/mu", scenario_group_path))?;
        mu.attributes_mut().extend(
            json!({
                "standard_name": "mu",
                "long_name": "Coulomb friction coefficient",
                "units": "-",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        mu.store_metadata()?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "mu",
            "Coulomb friction coefficient",
            "-",
            zarrs::array::data_type::float32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "xsi",
            "Turbulent friction coefficient",
            "m/s²",
            zarrs::array::data_type::float32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "sim_model",
            "Simulation model",
            "-",
            zarrs::array::data_type::uint32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "friction_model",
            "Friction model",
            "-",
            zarrs::array::data_type::uint32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "released_particles_per_cell",
            "Released particles per cell",
            "-",
            zarrs::array::data_type::uint32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "density",
            "Density",
            "kg/m³",
            zarrs::array::data_type::float32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "slab_thickness_factor",
            "Slab thickness factor",
            "-",
            zarrs::array::data_type::float32(),
        )?;
        // self.create_run_variable(
        //     &scenario_group_path,
        //     max_number_runs,
        //     "grain_diameter",
        //     "Grain diameter",
        //     "m",
        //     zarrs::array::data_type::float32(),
        // )?;
        // self.create_run_variable(
        //     &scenario_group_path,
        //     max_number_runs,
        //     "internal_friction_angle",
        //     "Internal friction angle",
        //     "degrees",
        //     zarrs::array::data_type::float32(),
        // )?;
        // self.create_run_variable(
        //     &scenario_group_path,
        //     max_number_runs,
        //     "basal_friction_angle",
        //     "Basal friction angle",
        //     "degrees",
        //     zarrs::array::data_type::float32(),
        // )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "cfl",
            "CFL number",
            "-",
            zarrs::array::data_type::float32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "velocity_threshold",
            "Velocity threshold",
            "m/s",
            zarrs::array::data_type::float32(),
        )?;
        self.create_run_variable(
            &scenario_group_path,
            max_number_runs,
            "flags",
            "Flags to switch physics",
            "-",
            zarrs::array::data_type::uint32(),
        )?;

        let mut center_of_mass_x = ArrayBuilder::new(
            vec![max_number_runs, max_timesteps],
            vec![1, chunk_timestep],
            zarrs::array::data_type::float32(),
            FillValue::from(f32::NAN),
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["run", "timestep"].into())
        .build(
            self.store.clone(),
            &format!("{}/center_of_mass_x", scenario_group_path),
        )?;
        center_of_mass_x.attributes_mut().extend(
            json!({
                "standard_name": "center_of_mass_x",
                "long_name": "Center of Mass X",
                "units": "m",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        center_of_mass_x.store_metadata()?;

        let mut center_of_mass_y = ArrayBuilder::new(
            vec![max_number_runs, max_timesteps],
            vec![1, chunk_timestep],
            zarrs::array::data_type::float32(),
            FillValue::from(f32::NAN),
        )
        .bytes_to_bytes_codecs(self.blosc_f32.clone())
        .dimension_names(["run", "timestep"].into())
        .build(
            self.store.clone(),
            &format!("{}/center_of_mass_y", scenario_group_path),
        )?;
        center_of_mass_y.attributes_mut().extend(
            json!({
                "standard_name": "center_of_mass_y",
                "long_name": "Center of Mass Y",
                "units": "m",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        center_of_mass_y.store_metadata()?;

        let mut release_area_array_builder = ArrayBuilder::new(
            vec![ylen, xlen],
            vec![ylen, xlen],
            zarrs::array::data_type::float16(),
            FillValue::from(f16::from_f32(0.0)),
        )
        .bytes_to_bytes_codecs(self.blosc_f16.clone())
        .dimension_names(["y", "x"].into())
        .build(
            self.store.clone(),
            &format!("{}/release_area", scenario_group_path),
        )?;
        release_area_array_builder.attributes_mut().extend(
            json!({
                "standard_name": "release_thickness",
                "long_name": "Release Area Thickness",
                "units": "m",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        release_area_array_builder.store_metadata()?;

        let release_area_array = Array::open(
            self.store.clone(),
            &format!("/{}/{}/release_area", site_name, scenario_name),
        )?;
        let release_area_f16: Vec<f16> = release_area.iter().map(|x| f16::from_f32(*x)).collect();
        release_area_array.store_chunk(&[0, 0], &release_area_f16)?;

        let scenario = Scenario::connect_existing(self.store.clone(), site_name, scenario_name)?;
        self.scenarios.insert(scenario_key, scenario);
        Ok(())
    }

    fn create_run_variable(
        &self,
        scenario_group_path: &str,
        max_number_runs: u64,
        name: &str,
        long_name: &str,
        units: &str,
        data_type: DataType,
    ) -> Result<(), OutputError> {
        let mut var = zarrs::array::ArrayBuilder::new(vec![max_number_runs], vec![1], data_type, 0)
            .dimension_names(["run"].into())
            .build(
                self.store.clone(),
                &format!("{}/{}", scenario_group_path, name),
            )?;
        var.attributes_mut().extend(
            json!({
                "standard_name": name,
                "long_name": long_name,
                "units": units,
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        var.store_metadata()?;
        Ok(())
    }

    fn normalize_store_path(path: &str) -> PathBuf {
        let path = PathBuf::from(path);
        if path.extension().is_some_and(|ext| ext == "zarr") {
            path
        } else {
            path.with_extension("zarr")
        }
    }

    fn list_child_dirs(base_path: &Path) -> Result<Vec<String>, OutputError> {
        let entries = fs::read_dir(base_path).map_err(|e| {
            OutputError::InvalidData(format!(
                "Failed to list directories in '{}': {e}",
                base_path.display()
            ))
        })?;

        let mut dirs = Vec::new();
        for entry_result in entries {
            let entry = entry_result.map_err(|e| {
                OutputError::InvalidData(format!(
                    "Failed to read directory entry in '{}': {e}",
                    base_path.display()
                ))
            })?;
            if entry.path().is_dir() {
                dirs.push(entry.file_name().to_string_lossy().to_string());
            }
        }
        Ok(dirs)
    }

    fn resolve_unique_partial_match(
        base_path: &Path,
        query: &str,
        level_name: &str,
    ) -> Result<String, OutputError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(OutputError::InvalidData(format!(
                "{level_name} query must not be empty"
            )));
        }

        let dirs = Self::list_child_dirs(base_path)?;
        if dirs.is_empty() {
            return Err(OutputError::InvalidData(format!(
                "No {level_name} directories found under '{}'",
                base_path.display()
            )));
        }

        let query_lower = query.to_lowercase();

        let exact: Vec<String> = dirs
            .iter()
            .filter(|name| name.to_lowercase() == query_lower)
            .cloned()
            .collect();
        if exact.len() == 1 {
            return Ok(exact[0].clone());
        }

        let matches: Vec<String> = dirs
            .iter()
            .filter(|name| name.to_lowercase().contains(&query_lower))
            .cloned()
            .collect();

        match matches.len() {
            1 => Ok(matches[0].clone()),
            0 => Err(OutputError::InvalidData(format!(
                "No {level_name} matches for '{query}' under '{}'. Available: {}",
                base_path.display(),
                dirs.join(", ")
            ))),
            _ => Err(OutputError::InvalidData(format!(
                "Ambiguous {level_name} query '{query}' under '{}'. Matches: {}",
                base_path.display(),
                matches.join(", ")
            ))),
        }
    }

    pub fn read_dem_from_store(store_path: &str, site_query: &str) -> Result<Dem, OutputError> {
        let zarr_path = Self::normalize_store_path(store_path);
        let store = Arc::new(FilesystemStore::new(&zarr_path)?);

        let site_name = Self::resolve_unique_partial_match(&zarr_path, site_query, "site")?;
        let site_group_path = format!("/{site_name}");
        let site_group = Group::open(store.clone(), &site_group_path)?;

        let dem_array = Array::open(store.clone(), &format!("/{site_name}/dem"))?;
        let dem_shape = dem_array.shape().to_vec();
        if dem_shape.len() != 2 {
            return Err(OutputError::InvalidData(format!(
                "DEM array in site '{site_name}' must be 2D, got shape {:?}",
                dem_shape
            )));
        }

        let dem_subset = ArraySubset::new_with_start_shape(vec![0, 0], dem_shape.clone())?;
        let dem_data: Vec<f32> = dem_array.retrieve_array_subset::<Vec<f32>>(&dem_subset)?;

        let x_array = Array::open(store.clone(), &format!("/{site_name}/x"))?;
        let x_shape = x_array.shape().to_vec();
        if x_shape.len() != 1 {
            return Err(OutputError::InvalidData(format!(
                "x array in site '{site_name}' must be 1D, got shape {:?}",
                x_shape
            )));
        }
        let x_subset = ArraySubset::new_with_start_shape(vec![0], x_shape.clone())?;
        let x_coords: Vec<f32> = x_array.retrieve_array_subset::<Vec<f32>>(&x_subset)?;

        let y_array = Array::open(store.clone(), &format!("/{site_name}/y"))?;
        let y_shape = y_array.shape().to_vec();
        if y_shape.len() != 1 {
            return Err(OutputError::InvalidData(format!(
                "y array in site '{site_name}' must be 1D, got shape {:?}",
                y_shape
            )));
        }
        let y_subset = ArraySubset::new_with_start_shape(vec![0], y_shape.clone())?;
        let y_coords: Vec<f32> = y_array.retrieve_array_subset::<Vec<f32>>(&y_subset)?;

        let width = dem_shape[1] as usize;
        let height = dem_shape[0] as usize;

        if x_coords.len() != width || y_coords.len() != height {
            return Err(OutputError::InvalidData(format!(
                "DEM/x/y shape mismatch for site '{site_name}': dem={:?}, x={}, y={}",
                dem_shape,
                x_coords.len(),
                y_coords.len()
            )));
        }

        let source = site_group
            .attributes()
            .get("dem_source")
            .and_then(|v| v.as_str())
            .unwrap_or("zarr")
            .to_string();

        let projection = site_group
            .attributes()
            .get("dem_projection")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let cell_size = site_group
            .attributes()
            .get("dem_cell_size")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let map_factor = site_group
            .attributes()
            .get("dem_map_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let (xmin, xmax) = x_coords
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(*v), mx.max(*v))
            });
        let (ymin, ymax) = y_coords
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(*v), mx.max(*v))
            });

        let data = dem_data
            .chunks(width)
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f32>>>();

        Ok(Dem {
            width,
            height,
            bounds: compute_core::dem::Bounds {
                xmin,
                xmax,
                ymin,
                ymax,
            },
            data1d: dem_data.clone(),
            data,
            x: x_coords,
            y: y_coords,
            cell_size,
            map_factor,
            minimum_elevation: Dem::calculate_minimum_elevation(&dem_data),
            source,
            projection,
        })
    }

    pub fn read_release_area_from_store(
        store_path: &str,
        site_query: &str,
        scenario_query: &str,
    ) -> Result<Vec<f32>, OutputError> {
        let zarr_path = Self::normalize_store_path(store_path);
        let store = Arc::new(FilesystemStore::new(&zarr_path)?);

        let site_name = Self::resolve_unique_partial_match(&zarr_path, site_query, "site")?;
        let site_path = zarr_path.join(&site_name);
        let scenario_name =
            Self::resolve_unique_partial_match(&site_path, scenario_query, "scenario")?;

        let release_area = Array::open(
            store.clone(),
            &format!("/{site_name}/{scenario_name}/release_area"),
        )?;
        let shape = release_area.shape().to_vec();
        if shape.len() != 2 {
            return Err(OutputError::InvalidData(format!(
                "release_area in '{site_name}/{scenario_name}' must be 2D, got shape {:?}",
                shape
            )));
        }

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], shape.clone())?;
        let release_area_f16: Vec<f16> = release_area.retrieve_array_subset::<Vec<f16>>(&subset)?;
        Ok(release_area_f16
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<f32>>())
    }

    pub fn write_flow_fields(
        &mut self,
        site_name: &str,
        scenario_name: &str,
        run_id: u64,
        timestep: u64,
        flow_velocity_data: &[f32],
        flow_thickness_data: &[f32],
    ) -> Result<(), OutputError> {
        let scenario_key = format!("{site_name}/{scenario_name}");
        let scenario = self
            .scenarios
            .get_mut(&scenario_key)
            .ok_or_else(|| OutputError::ScenarioNotFound(scenario_key.clone()))?;

        if flow_velocity_data.len() != (scenario.width * scenario.height) as usize
            || flow_thickness_data.len() != (scenario.width * scenario.height) as usize
        {
            return Err(OutputError::InvalidData(format!(
                "Flow field shape mismatch for scenario '{site_name}/{scenario_name}': expected {}, got {} and {}",
                scenario.width * scenario.height,
                flow_velocity_data.len(),
                flow_thickness_data.len()
            )));
        }

        if run_id >= scenario.max_number_runs {
            return Err(OutputError::InvalidRunID(
                run_id,
                scenario.max_number_runs.saturating_sub(1),
            ));
        }
        if timestep >= scenario.max_timesteps {
            return Err(OutputError::InvalidData(format!(
                "Timestep {} out of bounds for scenario '{site_name}/{scenario_name}' (max {})",
                timestep, scenario.max_timesteps
            )));
        }

        if scenario.flow_velocity.is_none() || scenario.flow_thickness.is_none() {
            let scenario_group_path = format!("/{site_name}/{scenario_name}");
            let mut flow_velocity = ArrayBuilder::new(
                vec![
                    scenario.max_number_runs,
                    scenario.width,
                    scenario.height,
                    scenario.max_timesteps,
                ],
                vec![1, scenario.width, scenario.height, 1],
                zarrs::array::data_type::float16(),
                FillValue::from(f16::from_f32(0.0)),
            )
            .bytes_to_bytes_codecs(self.blosc_f16.clone())
            .dimension_names(["run", "x", "y", "timestep"].into())
            .build(
                self.store.clone(),
                &format!("{}/flow_velocity", scenario_group_path),
            )?;
            flow_velocity.attributes_mut().extend(
                json!({
                    "standard_name": "flow_velocity",
                    "long_name": "Flow velocity grid",
                    "units": "m/s",
                })
                .as_object()
                .unwrap()
                .clone(),
            );
            flow_velocity.store_metadata()?;

            let mut flow_thickness = ArrayBuilder::new(
                vec![
                    scenario.max_number_runs,
                    scenario.width,
                    scenario.height,
                    scenario.max_timesteps,
                ],
                vec![1, scenario.width, scenario.height, 1],
                zarrs::array::data_type::float16(),
                FillValue::from(f16::from_f32(0.0)),
            )
            .bytes_to_bytes_codecs(self.blosc_f16.clone())
            .dimension_names(["run", "x", "y", "timestep"].into())
            .build(
                self.store.clone(),
                &format!("{}/flow_thickness", scenario_group_path),
            )?;
            flow_thickness.attributes_mut().extend(
                json!({
                    "standard_name": "flow_thickness",
                    "long_name": "Flow thickness grid",
                    "units": "m",
                })
                .as_object()
                .unwrap()
                .clone(),
            );
            flow_thickness.store_metadata()?;

            scenario.flow_velocity = Some(flow_velocity);
            scenario.flow_thickness = Some(flow_thickness);
        }

        let flow_velocity = scenario.flow_velocity.as_ref().unwrap();
        let flow_thickness = scenario.flow_thickness.as_ref().unwrap();

        let velocity_subset = ArraySubset::new_with_start_shape(
            vec![run_id, 0, 0, timestep],
            vec![1, scenario.width, scenario.height, 1],
        )?;
        let thickness_subset = ArraySubset::new_with_start_shape(
            vec![run_id, 0, 0, timestep],
            vec![1, scenario.width, scenario.height, 1],
        )?;

        let velocity_data_f16: Vec<f16> = flow_velocity_data
            .iter()
            .map(|x| f16::from_f32(*x))
            .collect();
        let thickness_data_f16: Vec<f16> = flow_thickness_data
            .iter()
            .map(|x| f16::from_f32(*x))
            .collect();

        flow_velocity.store_array_subset(&velocity_subset, &velocity_data_f16)?;
        flow_thickness.store_array_subset(&thickness_subset, &thickness_data_f16)?;
        Ok(())
    }

    pub fn write_particle_position(
        &mut self,
        site_name: &str,
        scenario_name: &str,
        run_id: u64,
        timestep: u64,
        particle_position_data: &[f32],
    ) -> Result<(), OutputError> {
        let scenario_key = format!("{site_name}/{scenario_name}");
        let scenario = self
            .scenarios
            .get_mut(&scenario_key)
            .ok_or_else(|| OutputError::ScenarioNotFound(scenario_key.clone()))?;

        if run_id >= scenario.max_number_runs {
            return Err(OutputError::InvalidRunID(
                run_id,
                scenario.max_number_runs.saturating_sub(1),
            ));
        }
        if timestep >= scenario.max_timesteps {
            return Err(OutputError::InvalidData(format!(
                "Timestep {} out of bounds for scenario '{site_name}/{scenario_name}' (max {})",
                timestep, scenario.max_timesteps
            )));
        }
        if particle_position_data.is_empty() {
            return Err(OutputError::InvalidData(format!(
                "Particle position data is empty for scenario '{site_name}/{scenario_name}'"
            )));
        }
        if !particle_position_data.len().is_multiple_of(3) {
            return Err(OutputError::InvalidData(format!(
                "Particle position data length must be divisible by 3 (x,y,z) for scenario '{site_name}/{scenario_name}', got {}",
                particle_position_data.len()
            )));
        }

        if scenario.particle_position.is_none() {
            let scenario_group_path = format!("/{site_name}/{scenario_name}");
            let particle_count = (particle_position_data.len() / 3) as u64;
            let mut particle_position = ArrayBuilder::new(
                vec![
                    scenario.max_number_runs,
                    scenario.max_timesteps,
                    particle_count,
                    3,
                ],
                vec![1, 1, particle_count, 3],
                zarrs::array::data_type::float32(),
                FillValue::from(0.0_f32),
            )
            .bytes_to_bytes_codecs(self.blosc_f32.clone())
            .dimension_names(["run", "timestep", "particle_id", "component"].into())
            .build(
                self.store.clone(),
                &format!("{}/particle_position", scenario_group_path),
            )?;
            particle_position.attributes_mut().extend(
                json!({
                    "standard_name": "particle_position",
                    "long_name": "Particle xyz position values",
                    "units": "m",
                    "components": ["x", "y", "z"],
                })
                .as_object()
                .unwrap()
                .clone(),
            );
            particle_position.store_metadata()?;

            scenario.particle_position = Some(particle_position);
            scenario.particle_count = Some(particle_count);
        }

        let particle_count = scenario.particle_count.ok_or_else(|| {
            OutputError::InvalidData(format!(
                "Missing particle count for scenario '{site_name}/{scenario_name}'"
            ))
        })?;

        if particle_position_data.len() != particle_count as usize * 3 {
            return Err(OutputError::InvalidData(format!(
                "Particle position shape mismatch for scenario '{site_name}/{scenario_name}': expected {} values ({} particles * 3 components), got {}",
                particle_count * 3,
                particle_count,
                particle_position_data.len()
            )));
        }

        let particle_position = scenario.particle_position.as_ref().ok_or_else(|| {
            OutputError::InvalidData(format!(
                "Missing particle_position array for scenario '{site_name}/{scenario_name}'"
            ))
        })?;

        let position_subset = ArraySubset::new_with_start_shape(
            vec![run_id, timestep, 0, 0],
            vec![1, 1, particle_count, 3],
        )?;

        particle_position.store_array_subset(&position_subset, particle_position_data)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_new_run(
        &mut self,
        site_name: &str,
        scenario_name: &str,
        peak_flow_velocity_data: &[f32],
        peak_flow_thickness_data: &[f32],
        center_of_mass_x_data: &[f32],
        center_of_mass_y_data: &[f32],
        travel_length_data: f32,
        travel_angle_data: f32,
        settings: &SimSettings,
    ) -> Result<(), OutputError> {
        let scenario_key = format!("{site_name}/{scenario_name}");
        let scenario = self
            .scenarios
            .get_mut(&scenario_key)
            .ok_or_else(|| OutputError::ScenarioNotFound(scenario_key.clone()))?;
        let run_id = scenario.number_of_runs;

        Self::write_array_f16(
            &scenario.peak_flow_velocity,
            scenario.width,
            scenario.height,
            run_id,
            peak_flow_velocity_data,
        )?;
        Self::write_array_f16(
            &scenario.peak_flow_thickness,
            scenario.width,
            scenario.height,
            run_id,
            peak_flow_thickness_data,
        )?;
        Self::write_scalar(&scenario.travel_length, run_id, travel_length_data)?;
        Self::write_scalar(&scenario.travel_angle, run_id, travel_angle_data)?;
        Self::write_scalar(&scenario.mu, run_id, settings.friction_coefficient)?;
        Self::write_scalar(&scenario.xsi, run_id, settings.drag_coefficient)?;
        Self::write_scalar(&scenario.density, run_id, settings.density)?;
        Self::write_scalar(
            &scenario.slab_thickness_factor,
            run_id,
            settings.slab_thickness_factor,
        )?;
        Self::write_scalar_u32(&scenario.sim_model, run_id, settings.sim_model)?;
        Self::write_scalar_u32(&scenario.friction_model, run_id, settings.friction_model)?;
        Self::write_scalar(&scenario.cfl, run_id, settings.cfl)?;
        Self::write_scalar(
            &scenario.velocity_threshold,
            run_id,
            settings.velocity_threshold,
        )?;
        Self::write_scalar_u32(&scenario.flags, run_id, settings.flags)?;

        Self::write_com(&scenario.center_of_mass_x, run_id, center_of_mass_x_data)?;
        Self::write_com(&scenario.center_of_mass_y, run_id, center_of_mass_y_data)?;
        let attributes = scenario.group.attributes_mut();

        attributes.insert("number_of_runs".to_string(), json!(run_id + 1));

        scenario.group.store_metadata()?;
        scenario.number_of_runs += 1;

        Ok(())
    }

    pub fn connect_scenario(
        &mut self,
        site_name: &str,
        scenario_name: &str,
    ) -> Result<(), OutputError> {
        let scenario_key = format!("{site_name}/{scenario_name}");
        if self.scenarios.contains_key(&scenario_key) {
            return Ok(());
        }
        let scenario = Scenario::connect_existing(self.store.clone(), site_name, scenario_name)?;
        self.scenarios.insert(scenario_key, scenario);
        Ok(())
    }

    fn write_scalar(
        array: &Array<FilesystemStore>,
        run_id: u64,
        data: f32,
    ) -> Result<(), OutputError> {
        let subset = ArraySubset::new_with_start_shape(
            vec![run_id], // Coordinate start [run, y, x]
            vec![1],      // Shape of the chunk slice
        )?;
        array.store_array_subset(&subset, &[data])?;
        Ok(())
    }
    fn write_scalar_u32(
        array: &Array<FilesystemStore>,
        run_id: u64,
        data: u32,
    ) -> Result<(), OutputError> {
        let subset = ArraySubset::new_with_start_shape(
            vec![run_id], // Coordinate start [run, y, x]
            vec![1],      // Shape of the chunk slice
        )?;
        array.store_array_subset(&subset, &[data])?;
        Ok(())
    }

    fn write_array_f16(
        array: &Array<FilesystemStore>,
        width: u64,
        height: u64,
        run_id: u64,
        data: &[f32],
    ) -> Result<(), OutputError> {
        let subset = ArraySubset::new_with_start_shape(
            vec![run_id, 0, 0],     // Coordinate start [run, y, x]
            vec![1, height, width], // Shape of the chunk slice
        )?;
        let pfv_f16: Vec<f16> = data.iter().map(|x| f16::from_f32(*x)).collect();
        array.store_array_subset(&subset, &pfv_f16)?;
        Ok(())
    }

    fn write_com(
        array: &Array<FilesystemStore>,
        run_id: u64,
        data: &[f32],
    ) -> Result<(), OutputError> {
        let subset = ArraySubset::new_with_start_shape(
            vec![run_id, 0],            // Coordinate start [run, y, x]
            vec![1, data.len() as u64], // Shape of the chunk slice
        )?;
        array.store_array_subset(&subset, data)?;
        Ok(())
    }
}

pub struct Scenario {
    pub site_name: String,
    pub scenario_name: String,
    width: u64,
    height: u64,
    pub max_number_runs: u64,
    pub max_timesteps: u64,
    pub number_of_runs: u64,

    pub peak_flow_velocity: Array<FilesystemStore>,
    pub peak_flow_thickness: Array<FilesystemStore>,
    pub flow_velocity: Option<Array<FilesystemStore>>,
    pub flow_thickness: Option<Array<FilesystemStore>>,
    pub particle_position: Option<Array<FilesystemStore>>,
    pub particle_count: Option<u64>,
    pub travel_length: Array<FilesystemStore>,
    pub travel_angle: Array<FilesystemStore>,
    // TODO take the settings struct and store in a single array, provide a python function to parse it. Or does this make data analysis too hard?
    pub mu: Array<FilesystemStore>,
    pub xsi: Array<FilesystemStore>,
    pub released_particles_per_cell: Array<FilesystemStore>,
    pub density: Array<FilesystemStore>,
    pub slab_thickness_factor: Array<FilesystemStore>,
    pub sim_model: Array<FilesystemStore>,
    pub friction_model: Array<FilesystemStore>,
    pub cfl: Array<FilesystemStore>,
    pub velocity_threshold: Array<FilesystemStore>,
    pub flags: Array<FilesystemStore>,

    pub center_of_mass_x: Array<FilesystemStore>,
    pub center_of_mass_y: Array<FilesystemStore>,
    pub group: Group<FilesystemStore>,
}

impl Scenario {
    pub fn connect_existing(
        store: Arc<FilesystemStore>,
        site_name: &str,
        scenario_name: &str,
    ) -> Result<Self, OutputError> {
        let base = format!("/{site_name}/{scenario_name}");
        let peak_flow_velocity = Array::open(store.clone(), &format!("{base}/peak_flow_velocity"))?;
        let shape = peak_flow_velocity.shape();
        let height = shape[1];
        let width = shape[2];
        let avalanche_group = Group::open(store.clone(), &base)?;
        for (key, value) in avalanche_group.attributes() {
            info!("{key}: {value}");
        }
        let number_of_runs = avalanche_group
            .attributes()
            .get("number_of_runs")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                OutputError::InvalidData(format!(
                    "Missing or invalid 'number_of_runs' attribute in scenario '{site_name}/{scenario_name}'"
                ))
            })?;
        let max_number_runs = avalanche_group
            .attributes()
            .get("max_number_runs")
            .and_then(|v| v.as_u64())
            .unwrap_or(number_of_runs);
        let max_timesteps = avalanche_group
            .attributes()
            .get("max_timesteps")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let flow_velocity = Array::open(store.clone(), &format!("{base}/flow_velocity")).ok();
        let flow_thickness = Array::open(store.clone(), &format!("{base}/flow_thickness")).ok();
        let particle_position =
            Array::open(store.clone(), &format!("{base}/particle_position")).ok();
        let particle_count = particle_position.as_ref().map(|array| array.shape()[2]);
        Ok(Self {
            site_name: site_name.to_string(),
            scenario_name: scenario_name.to_string(),
            width,
            height,
            max_number_runs,
            max_timesteps,
            number_of_runs,
            peak_flow_velocity,
            peak_flow_thickness: Array::open(
                store.clone(),
                &format!("{base}/peak_flow_thickness"),
            )?,
            flow_velocity,
            flow_thickness,
            particle_position,
            particle_count,
            travel_length: Array::open(store.clone(), &format!("{base}/travel_length"))?,
            travel_angle: Array::open(store.clone(), &format!("{base}/travel_angle"))?,
            mu: Array::open(store.clone(), &format!("{base}/mu"))?,
            xsi: Array::open(store.clone(), &format!("{base}/xsi"))?,
            sim_model: Array::open(store.clone(), &format!("{base}/sim_model"))?,
            friction_model: Array::open(store.clone(), &format!("{base}/friction_model"))?,
            released_particles_per_cell: Array::open(
                store.clone(),
                &format!("{base}/released_particles_per_cell"),
            )?,
            density: Array::open(store.clone(), &format!("{base}/density"))?,
            slab_thickness_factor: Array::open(
                store.clone(),
                &format!("{base}/slab_thickness_factor"),
            )?,
            // grain_diameter: Array::open(store.clone(), &format!("{base}/grain_diameter"))?,
            // internal_friction_angle: Array::open(
            //     store.clone(),
            //     &format!("{base}/internal_friction_angle"),
            // )?,
            // basal_friction_angle: Array::open(
            //     store.clone(),
            //     &format!("{base}/basal_friction_angle"),
            // )?,
            cfl: Array::open(store.clone(), &format!("{base}/cfl"))?,
            velocity_threshold: Array::open(store.clone(), &format!("{base}/velocity_threshold"))?,
            flags: Array::open(store.clone(), &format!("{base}/flags"))?,
            center_of_mass_x: Array::open(store.clone(), &format!("{base}/center_of_mass_x"))?,
            center_of_mass_y: Array::open(store.clone(), &format!("{base}/center_of_mass_y"))?,
            group: avalanche_group,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::dem::{Bounds, Dem};
    use tempfile::TempDir;

    fn build_dem(x: &[f32], y: &[f32], data1d: &[f32]) -> Dem {
        Dem {
            width: x.len(),
            height: y.len(),
            bounds: Bounds {
                xmin: *x.first().unwrap_or(&0.0),
                xmax: *x.last().unwrap_or(&0.0),
                ymin: *y.first().unwrap_or(&0.0),
                ymax: *y.last().unwrap_or(&0.0),
            },
            data1d: data1d.to_vec(),
            data: Vec::new(),
            x: x.to_vec(),
            y: y.to_vec(),
            cell_size: 1.0,
            map_factor: 1.0,
            minimum_elevation: Dem::calculate_minimum_elevation(data1d),
            source: "test_dem".to_string(),
            projection: "EPSG:2056".to_string(),
        }
    }

    #[test_log::test]
    fn test() {
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("test.zarr");
        let mut output =
            Output::new(zarr_path.to_str().unwrap()).expect("Failed to create Output struct");
        let dem = build_dem(
            &[3.0, 4.0, 5.0],
            &[2.0, 3.0],
            &[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        );
        output
            .add_new_site("site_test", &dem)
            .expect("Failed to add new site");
        output
            .add_new_scenario(
                "site_test",
                "avalanche_test",
                10,
                2000,
                &[0.1, 1.2, 0.10, 0.8, 0.20, 0.20],
                113.0,
                12300786.9,
                vec![2.0, 3.0],
                &[3.0, 4.0, 5.0],
            )
            .expect("Failed to add new avalanche scenario");
        let mock_velocity_data: Vec<f32> = vec![34.0, 35.0, 36.0, 24.0, 25.0, 26.0];
        let mock_thickness_data: Vec<f32> = vec![1.0, 1.1, 1.2, 0.1, 0.2, 0.3];
        let mock_com_x: Vec<f32> = vec![3.0; 610];
        let mock_com_y: Vec<f32> = vec![4.0; 610];
        let settings = SimSettings::default();

        output
            .add_new_run(
                "site_test",
                "avalanche_test",
                &mock_velocity_data,
                &mock_thickness_data,
                &mock_com_x,
                &mock_com_y,
                3400.0,
                25.0,
                &settings,
            )
            .expect("Failed to add new run");
    }
    #[test_log::test]
    fn test_cached_handles_workflow() {
        // Create an isolated temporary directory
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("avalanche_model.zarr");

        // 1. Initialize Output Engine
        let mut output = Output::new(zarr_path.to_str().unwrap()).unwrap();

        // Dummy dimensions and properties
        let y_coords = vec![1000.0, 950.0, 900.0]; // height = 3
        let x_coords = vec![500.0, 550.0, 600.0, 650.0]; // width = 4
        let dem = build_dem(&x_coords, &y_coords, &vec![2100.0; 12]); // 3x4
        let release_area = vec![1.5; 12];

        let max_timesteps = 50;
        let max_number_runs = 5;

        // 2. Register site
        output.add_new_site("Chamonix", &dem).unwrap();

        // 3. Register scenario
        output
            .add_new_scenario(
                "Chamonix",
                "Valley_A",
                max_number_runs,
                max_timesteps,
                &release_area,
                32.5,
                15000.0,
                y_coords,
                &x_coords,
            )
            .unwrap();

        // Assert structural configuration details mapped correctly
        {
            let cached_scen = output.scenarios.get("Chamonix/Valley_A").unwrap();
            assert_eq!(cached_scen.height, 3);
            assert_eq!(cached_scen.width, 4);
            assert_eq!(cached_scen.number_of_runs, 0);
        }

        // 4. Populate simulation runs
        let simulated_cells = 3 * 4; // height * width
        let velocity_mock = vec![12.4f32; simulated_cells];
        let thickness_mock = vec![2.1f32; simulated_cells];
        let com_x_mock = vec![520.0f32; max_timesteps as usize];
        let com_y_mock = vec![940.0f32; max_timesteps as usize];
        let settings = SimSettings::default();

        output
            .add_new_run(
                "Chamonix",
                "Valley_A",
                &velocity_mock,
                &thickness_mock,
                &com_x_mock,
                &com_y_mock,
                1200.5,
                28.2,
                &settings,
            )
            .unwrap();

        assert_eq!(
            output
                .scenarios
                .get("Chamonix/Valley_A")
                .unwrap()
                .number_of_runs,
            1
        );

        // 5. Verify structural targets are populated correctly on disk
        let velocity_chunk_file = zarr_path.join("Chamonix/Valley_A/peak_flow_velocity/c/0/0/0");
        let dem_chunk_file = zarr_path.join("Chamonix/dem/c/0/0");
        assert!(zarr_path.join("Chamonix/zarr.json").exists());
        assert!(zarr_path.join("Chamonix/dem/zarr.json").exists());
        assert!(zarr_path.join("Chamonix/Valley_A/zarr.json").exists());
        assert!(
            zarr_path
                .join("Chamonix/Valley_A/peak_flow_velocity/zarr.json")
                .exists()
        );
        assert!(
            velocity_chunk_file.exists(),
            "Data chunk wasn't written to the expected location."
        );
        assert!(
            dem_chunk_file.exists(),
            "DEM chunk wasn't written to the expected location."
        );
    }

    #[test_log::test]
    fn test_write_flow_fields_lazily_creates_optional_grids() {
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("flow_fields.zarr");
        let mut output = Output::new(zarr_path.to_str().unwrap()).unwrap();

        let y_coords = vec![1000.0, 950.0, 900.0];
        let x_coords = vec![500.0, 550.0, 600.0, 650.0];
        let dem = build_dem(&x_coords, &y_coords, &vec![2100.0; 12]);
        let release_area = vec![1.5; 12];

        output.add_new_site("Chamonix", &dem).unwrap();
        output
            .add_new_scenario(
                "Chamonix",
                "Valley_A",
                2,
                10,
                &release_area,
                32.5,
                15000.0,
                y_coords,
                &x_coords,
            )
            .unwrap();

        let velocity = vec![1.0f32; 12];
        let thickness = vec![0.5f32; 12];

        output
            .write_flow_fields("Chamonix", "Valley_A", 0, 3, &velocity, &thickness)
            .unwrap();

        let scenario = output.scenarios.get("Chamonix/Valley_A").unwrap();
        assert!(scenario.flow_velocity.is_some());
        assert!(scenario.flow_thickness.is_some());
        assert!(
            zarr_path
                .join("Chamonix/Valley_A/flow_velocity/zarr.json")
                .exists()
        );
        assert!(
            zarr_path
                .join("Chamonix/Valley_A/flow_thickness/zarr.json")
                .exists()
        );

        let bad_result =
            output.write_flow_fields("Chamonix", "Valley_A", 0, 3, &[1.0f32; 11], &thickness);
        assert!(bad_result.is_err());
    }

    #[test_log::test]
    fn test_write_particle_position_lazily_creates_and_validates_shape() {
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("particle_position.zarr");
        let mut output = Output::new(zarr_path.to_str().unwrap()).unwrap();

        let y_coords = vec![1000.0, 950.0, 900.0];
        let x_coords = vec![500.0, 550.0, 600.0, 650.0];
        let dem = build_dem(&x_coords, &y_coords, &vec![2100.0; 12]);
        let release_area = vec![1.5; 12];

        output.add_new_site("Chamonix", &dem).unwrap();
        output
            .add_new_scenario(
                "Chamonix",
                "Valley_A",
                2,
                10,
                &release_area,
                32.5,
                15000.0,
                y_coords,
                &x_coords,
            )
            .unwrap();

        let particle_positions = vec![
            10.0f32, 11.0, 12.0, // particle 0: x,y,z
            13.0, 14.0, 15.0, // particle 1: x,y,z
        ];
        output
            .write_particle_position("Chamonix", "Valley_A", 0, 3, &particle_positions)
            .unwrap();

        let scenario = output.scenarios.get("Chamonix/Valley_A").unwrap();
        assert!(scenario.particle_position.is_some());
        assert_eq!(scenario.particle_count, Some(2));
        assert!(
            zarr_path
                .join("Chamonix/Valley_A/particle_position/zarr.json")
                .exists()
        );

        let bad_result =
            output.write_particle_position("Chamonix", "Valley_A", 0, 3, &[10.0f32, 11.0, 12.0]);
        assert!(bad_result.is_err());

        let bad_component_count = output.write_particle_position(
            "Chamonix",
            "Valley_A",
            0,
            4,
            &[10.0f32, 11.0, 12.0, 13.0],
        );
        assert!(bad_component_count.is_err());
    }

    #[test_log::test]
    fn test_read_dem_and_release_area_with_partial_matches() {
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("reader_test.zarr");
        let mut output = Output::new(zarr_path.to_str().unwrap()).unwrap();

        let y_coords = vec![1000.0, 950.0, 900.0];
        let x_coords = vec![500.0, 550.0, 600.0, 650.0];
        let dem_data = vec![2100.0; 12];
        let dem = build_dem(&x_coords, &y_coords, &dem_data);
        let release_area = vec![1.5f32; 12];

        output.add_new_site("Chamonix", &dem).unwrap();
        output
            .add_new_scenario(
                "Chamonix",
                "Valley_A",
                2,
                10,
                &release_area,
                32.5,
                15000.0,
                y_coords,
                &x_coords,
            )
            .unwrap();

        let loaded_dem = Output::read_dem_from_store(zarr_path.to_str().unwrap(), "amoni").unwrap();
        assert_eq!(loaded_dem.width, dem.width);
        assert_eq!(loaded_dem.height, dem.height);
        assert_eq!(loaded_dem.data1d.len(), dem_data.len());

        let loaded_release_area =
            Output::read_release_area_from_store(zarr_path.to_str().unwrap(), "hamon", "ley_a")
                .unwrap();
        assert_eq!(loaded_release_area.len(), release_area.len());
        assert!(
            loaded_release_area
                .iter()
                .zip(release_area.iter())
                .all(|(a, b)| (a - b).abs() < 0.01)
        );
    }

    #[test_log::test]
    fn test_readers_fail_on_ambiguous_partial_match() {
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("reader_ambiguous.zarr");
        let mut output = Output::new(zarr_path.to_str().unwrap()).unwrap();

        let y_coords = vec![1000.0, 950.0, 900.0];
        let x_coords = vec![500.0, 550.0, 600.0, 650.0];
        let dem = build_dem(&x_coords, &y_coords, &vec![2100.0; 12]);

        output.add_new_site("Chamonix", &dem).unwrap();
        output.add_new_site("Chamrousse", &dem).unwrap();

        let ambiguous_site_result =
            Output::read_dem_from_store(zarr_path.to_str().unwrap(), "cham");
        assert!(ambiguous_site_result.is_err());
    }
}
