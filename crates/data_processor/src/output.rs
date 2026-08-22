use compute_core::settings::SimSettings;
use half::f16;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
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

    pub fn add_new_site(
        &mut self,
        site_name: &str,
        mut y_coords: Vec<f32>,
        x_coords: &[f32],
        dem: &[f32],
    ) -> Result<(), OutputError> {
        if self.site_exists(site_name) {
            return Err(OutputError::SiteAlreadyExists(site_name.to_string()));
        }

        y_coords.reverse();
        let ylen = y_coords.len() as u64;
        let xlen = x_coords.len() as u64;
        let size = xlen * ylen;

        if !dem.is_empty() && dem.len() != size as usize {
            return Err(OutputError::InvalidData(format!(
                "DEM has incorrect size: expected {}, got {}",
                size,
                dem.len()
            )));
        }

        let site_group_path = format!("/{}", site_name);
        let site_group = GroupBuilder::new().build(self.store.clone(), &site_group_path)?;
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
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        dem_array_builder.store_metadata()?;

        let dem_array = Array::open(self.store.clone(), &format!("/{}/dem", site_name))?;
        dem_array.store_chunk(&[0, 0], dem)?;

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
            "slab_thickness",
            "Slab thickness",
            "m",
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
        Self::write_scalar(&scenario.slab_thickness, run_id, settings.slab_thickness)?;
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
    pub number_of_runs: u64,

    pub peak_flow_velocity: Array<FilesystemStore>,
    pub peak_flow_thickness: Array<FilesystemStore>,
    pub travel_length: Array<FilesystemStore>,
    pub travel_angle: Array<FilesystemStore>,
    // TODO take the settings struct and store in a single array, provide a python function to parse it. Or does this make data analysis too hard?
    pub mu: Array<FilesystemStore>,
    pub xsi: Array<FilesystemStore>,
    pub released_particles_per_cell: Array<FilesystemStore>,
    pub density: Array<FilesystemStore>,
    pub slab_thickness: Array<FilesystemStore>,
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
        Ok(Self {
            site_name: site_name.to_string(),
            scenario_name: scenario_name.to_string(),
            width,
            height,
            number_of_runs,
            peak_flow_velocity,
            peak_flow_thickness: Array::open(
                store.clone(),
                &format!("{base}/peak_flow_thickness"),
            )?,
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
            slab_thickness: Array::open(store.clone(), &format!("{base}/slab_thickness"))?,
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
    use tempfile::TempDir;

    #[test_log::test]
    fn test() {
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("test.zarr");
        let mut output =
            Output::new(zarr_path.to_str().unwrap()).expect("Failed to create Output struct");
        output
            .add_new_site(
                "site_test",
                vec![2.0, 3.0],
                &[3.0, 4.0, 5.0],
                &[1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
            )
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
        let dem = vec![2100.0; 12]; // 3x4
        let release_area = vec![1.5; 12];

        let max_timesteps = 50;
        let max_number_runs = 5;

        // 2. Register site
        output
            .add_new_site("Chamonix", y_coords.clone(), &x_coords, &dem)
            .unwrap();

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
}
