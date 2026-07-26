use half::f16;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use zarrs::array::ArraySubset;
use zarrs::array::{
    Array, ArrayBuilder, FillValue,
    codec::bytes_to_bytes::blosc::{
        BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    },
};
use zarrs::group::GroupBuilder;
use zarrs::storage::StorageError;
use zarrs_filesystem::FilesystemStore;

#[derive(thiserror::Error, Debug)]
pub enum OutputError {
    #[error("Run ID is out of bounds: got {0}, expected max {1}")]
    InvalidRunID(u64, u64),

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
struct Output {
    store: Arc<FilesystemStore>,
    scenarios: HashMap<String, Scenario>,
}
#[allow(dead_code)]
impl Output {
    pub fn new(path: &str) -> Result<Self, OutputError> {
        let path = if path.ends_with(".zarr") {
            path.to_string()
        } else {
            format!("{path}.zarr")
        };
        let store = Arc::new(FilesystemStore::new(path)?);
        let mut root_group = GroupBuilder::new().build(store.clone(), "/")?;
        let global_attrs = json!({
            "title": "Avalanchers Simulation Output",
            "conventions": "CF-1.8",
            "source": "avalanchers",

        });
        root_group
            .attributes_mut()
            .extend(global_attrs.as_object().unwrap().clone());
        root_group.store_metadata()?;

        Ok(Self {
            store,
            scenarios: HashMap::new(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn add_new_scenario(
        &mut self,
        scenario_name: &str,
        number_runs: u64,
        max_timesteps: u64,
        mut y_coords: Vec<f32>,
        x_coords: Vec<f32>,
        dem: &[f32],
        release_area: &[f32],
        aspect_release_value: f32,
        release_volume: f32,
    ) -> Result<(), OutputError> {
        if self.scenarios.contains_key(scenario_name) {
            return Err(OutputError::ScenarioAlreadyExists(
                scenario_name.to_string(),
            ));
        }

        let chunk_timestep = std::cmp::min(200, max_timesteps);

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

        if !release_area.is_empty() && release_area.len() != size as usize {
            return Err(OutputError::InvalidData(format!(
                "Release area has incorrect size: expected {}, got {}",
                size,
                release_area.len()
            )));
        }

        let avalanche_group_path = format!("/{}", scenario_name);
        let mut avalanche_group =
            GroupBuilder::new().build(self.store.clone(), &avalanche_group_path)?;
        avalanche_group.attributes_mut().extend(
            json!({
                "aspect_release_degrees": aspect_release_value,
                "release_volume_m3": release_volume,
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        avalanche_group.store_metadata()?;

        let blosc = Arc::new(BloscCodec::new(
            BloscCompressor::Zstd,
            BloscCompressionLevel::try_from(9).expect("Invalid compression level"),
            None, // automatic blocksize
            BloscShuffleMode::BitShuffle,
            Some(4), // f32 = 4 bytes
        )?);

        let mut y = zarrs::array::ArrayBuilder::new(
            vec![ylen], // array shape
            vec![ylen], // regular chunk shape
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["y"].into())
        .build(self.store.clone(), &format!("{}/y", avalanche_group_path))?;
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
            vec![xlen], // array shape
            vec![xlen], // regular chunk shape
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["x"].into())
        .build(self.store.clone(), &format!("{}/x", avalanche_group_path))?;
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

        let mut runs = zarrs::array::ArrayBuilder::new(
            vec![number_runs], // array shape
            vec![number_runs], // regular chunk shape
            zarrs::array::data_type::uint64(),
            0,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run"].into())
        .build(self.store.clone(), &format!("{}/run", avalanche_group_path))?;
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
        let run_ids: Vec<u64> = (0..number_runs).collect();
        #[allow(clippy::single_range_in_vec_init)]
        runs.store_chunks(&[0..1], run_ids)?;

        let mut timesteps = zarrs::array::ArrayBuilder::new(
            vec![max_timesteps], // array shape
            vec![max_timesteps], // regular chunk shape
            zarrs::array::data_type::uint64(),
            0,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["timestep"].into())
        .build(
            self.store.clone(),
            &format!("{}/timestep", avalanche_group_path),
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

        let grid_shape = vec![number_runs, ylen, xlen];
        // Chunk Shapes: [1, Y, X] -> Optimizes for appending/writing exactly 1 run at a time
        let grid_chunks = vec![1, ylen, xlen];

        let mut peak_flow_velocity = ArrayBuilder::new(
            grid_shape.clone(),
            grid_chunks.clone(),
            zarrs::array::data_type::float16(),
            FillValue::from(f16::from_f32(0.0)), // Clear unsimulated cells default to 0.0 for max compression
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run", "y", "x"].into())
        .build(
            self.store.clone(),
            &format!("{}/peak_flow_velocity", avalanche_group_path),
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
            FillValue::from(f16::from_f32(0.0)), // Clear unsimulated cells default to 0.0 for max compression
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run", "y", "x"].into())
        .build(
            self.store.clone(),
            &format!("{}/peak_flow_thickness", avalanche_group_path),
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
            vec![number_runs], // array shape
            vec![1],           // regular chunk shape
            zarrs::array::data_type::float32(),
            0,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run"].into())
        .build(
            self.store.clone(),
            &format!("{}/travel_length", avalanche_group_path),
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
            vec![number_runs], // array shape
            vec![1],           // regular chunk shape
            zarrs::array::data_type::float32(),
            0,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run"].into())
        .build(
            self.store.clone(),
            &format!("{}/travel_angle", avalanche_group_path),
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
            vec![number_runs], // array shape
            vec![1],           // regular chunk shape
            zarrs::array::data_type::float32(),
            0,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run"].into())
        .build(self.store.clone(), &format!("{}/mu", avalanche_group_path))?;
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

        let mut xsi = zarrs::array::ArrayBuilder::new(
            vec![number_runs], // array shape
            vec![1],           // regular chunk shape
            zarrs::array::data_type::float32(),
            0,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run"].into())
        .build(self.store.clone(), &format!("{}/xsi", avalanche_group_path))?;
        xsi.attributes_mut().extend(
            json!({
                "standard_name": "xsi",
                "long_name": "Turbulent friction coefficient",
                "units": "-",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        xsi.store_metadata()?;

        let mut center_of_mass_x = ArrayBuilder::new(
            vec![number_runs, max_timesteps],
            vec![1, chunk_timestep],
            zarrs::array::data_type::float32(),
            FillValue::from(f32::NAN),
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run", "timestep"].into())
        .build(
            self.store.clone(),
            &format!("{}/center_of_mass_x", avalanche_group_path),
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
            vec![number_runs, max_timesteps],
            vec![1, chunk_timestep],
            zarrs::array::data_type::float32(),
            FillValue::from(f32::NAN),
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["run", "timestep"].into())
        .build(
            self.store.clone(),
            &format!("{}/center_of_mass_y", avalanche_group_path),
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

        let mut dem_array_builder = ArrayBuilder::new(
            vec![ylen, xlen],
            vec![ylen, xlen],
            zarrs::array::data_type::float32(),
            FillValue::from(0.0_f32), // Clear unsimulated cells default to 0.0 for max compression
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["y", "x"].into())
        .build(self.store.clone(), &format!("{}/dem", avalanche_group_path))?;
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

        let mut release_area_array_builder = ArrayBuilder::new(
            vec![ylen, xlen],
            vec![ylen, xlen],
            zarrs::array::data_type::float16(),
            FillValue::from(f16::from_f32(0.0)),
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["y", "x"].into())
        .build(
            self.store.clone(),
            &format!("{}/release_area", avalanche_group_path),
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

        let dem_array = Array::open(self.store.clone(), &format!("/{}/dem", scenario_name))?;
        let subset = ArraySubset::new_with_start_shape(
            vec![0, 0],       // Coordinate start [y, x]
            vec![ylen, xlen], // Shape of the chunk slice
        )?;
        dem_array.store_array_subset(&subset, dem)?;

        let release_area_array = Array::open(
            self.store.clone(),
            &format!("/{}/release_area", scenario_name),
        )?;
        let subset = ArraySubset::new_with_start_shape(
            vec![0, 0],       // Coordinate start [y, x]
            vec![ylen, xlen], // Shape of the chunk slice
        )?;
        let release_area_f16: Vec<f16> = release_area.iter().map(|x| f16::from_f32(*x)).collect();
        release_area_array.store_array_subset(&subset, &release_area_f16)?;

        let scenario =
            Scenario::connect_existing(self.store.clone(), scenario_name, xlen, ylen, number_runs)?;
        self.scenarios.insert(scenario_name.to_string(), scenario);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_new_run(
        &mut self,
        scenario_name: &str,
        run_id: u64,
        peak_flow_velocity_data: &[f32],
        peak_flow_thickness_data: &[f32],
        center_of_mass_x_data: &[f32],
        center_of_mass_y_data: &[f32],
        travel_length_data: f32,
        travel_angle_data: f32,
        mu: f32,
        xsi: f32,
    ) -> Result<(), OutputError> {
        let scenario = self
            .scenarios
            .get(scenario_name)
            .ok_or_else(|| OutputError::ScenarioNotFound(scenario_name.to_string()))?;

        if run_id >= scenario.number_of_runs {
            return Err(OutputError::InvalidRunID(run_id, scenario.number_of_runs));
        }
        self.write_array_f16(
            &scenario.peak_flow_velocity,
            scenario.width,
            scenario.height,
            run_id,
            peak_flow_velocity_data,
        )?;
        self.write_array_f16(
            &scenario.peak_flow_thickness,
            scenario.width,
            scenario.height,
            run_id,
            peak_flow_thickness_data,
        )?;
        self.write_scalar(&scenario.travel_length, run_id, travel_length_data)?;
        self.write_scalar(&scenario.travel_angle, run_id, travel_angle_data)?;
        self.write_scalar(&scenario.mu, run_id, mu)?;
        self.write_scalar(&scenario.xsi, run_id, xsi)?;
        self.write_com(&scenario.center_of_mass_x, run_id, center_of_mass_x_data)?;
        self.write_com(&scenario.center_of_mass_y, run_id, center_of_mass_y_data)?;

        Ok(())
    }

    fn write_scalar(
        &self,
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

    fn write_array_f16(
        &self,
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
        &self,
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

        // let array = Array::open(self.store.clone(), &format!("/{}/{}", scenario_name, name))?;
        // let total_elements = data.len() as u64;
        // let chunk_size = 200;
        // // Loop through the data in steps of 'chunk_size'
        // for (i, chunk) in data.chunks(chunk_size as usize).enumerate() {
        //     let offset = i as u64 * chunk_size;
        //     let current_chunk_len = chunk.len() as u64;

        //     // Define the subset for this specific piece of the data
        //     let subset = ArraySubset::new_with_start_shape(
        //         vec![run_id, offset],       // Increment the offset
        //         vec![1, current_chunk_len], // Shape of this partial chunk
        //     )?;

        //     // Write this segment to the Zarr store
        //     array.store_array_subset(&subset, chunk.to_vec())?;
        // }
        // Ok(())
    }
}

struct Scenario {
    width: u64,
    height: u64,
    number_of_runs: u64,

    pub peak_flow_velocity: Array<FilesystemStore>,
    pub peak_flow_thickness: Array<FilesystemStore>,
    pub travel_length: Array<FilesystemStore>,
    pub travel_angle: Array<FilesystemStore>,
    pub mu: Array<FilesystemStore>,
    pub xsi: Array<FilesystemStore>,
    pub center_of_mass_x: Array<FilesystemStore>,
    pub center_of_mass_y: Array<FilesystemStore>,
}

impl Scenario {
    pub fn connect_existing(
        store: Arc<FilesystemStore>,
        scenario_name: &str,
        width: u64,
        height: u64,
        number_of_runs: u64,
    ) -> Result<Self, OutputError> {
        let base = format!("/{scenario_name}");
        Ok(Self {
            width,
            height,
            number_of_runs,
            peak_flow_velocity: Array::open(store.clone(), &format!("{base}/peak_flow_velocity"))?,
            peak_flow_thickness: Array::open(
                store.clone(),
                &format!("{base}/peak_flow_thickness"),
            )?,
            travel_length: Array::open(store.clone(), &format!("{base}/travel_length"))?,
            travel_angle: Array::open(store.clone(), &format!("{base}/travel_angle"))?,
            mu: Array::open(store.clone(), &format!("{base}/mu"))?,
            xsi: Array::open(store.clone(), &format!("{base}/xsi"))?,
            center_of_mass_x: Array::open(store.clone(), &format!("{base}/center_of_mass_x"))?,
            center_of_mass_y: Array::open(store.clone(), &format!("{base}/center_of_mass_y"))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    #[test]
    fn test() {
        let mut output = Output::new("test.zarr").expect("Failed to create Output struct");
        output
            .add_new_scenario(
                "avalanche_test",
                10,
                2000,
                vec![2.0, 3.0],
                vec![3.0, 4.0, 5.0],
                &vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
                &vec![0.1, 1.2, 0.10, 0.8, 0.20, 0.20],
                113.0,
                12300786.9,
            )
            .expect("Failed to add new avalanche scenario");
        let mock_velocity_data: Vec<f32> = vec![34.0, 35.0, 36.0, 24.0, 25.0, 26.0];
        let mock_thickness_data: Vec<f32> = vec![1.0, 1.1, 1.2, 0.1, 0.2, 0.3];
        let mock_com_x: Vec<f32> = vec![3.0; 610];
        let mock_com_y: Vec<f32> = vec![4.0; 610];
        output
            .add_new_run(
                "avalanche_test",
                4,
                &mock_velocity_data,
                &mock_thickness_data,
                &mock_com_x,
                &mock_com_y,
                3400.0,
                25.0,
                0.4,
                2000.0,
            )
            .expect("Failed to add new run");
    }
    #[test]
    fn test_cached_handles_workflow() {
        // Create an isolated temporary directory
        let tmp_dir = TempDir::new().unwrap();
        let zarr_path = tmp_dir.path().join("avalanche_model.zarr");
        // let zarr_path = PathBuf::from("avalanche_model.zarr");

        // 1. Initialize Output Engine
        let mut engine = Output::new(zarr_path.to_str().unwrap()).unwrap();

        // Dummy dimensions and properties
        let y_coords = vec![1000.0, 950.0, 900.0]; // height = 3
        let x_coords = vec![500.0, 550.0, 600.0, 650.0]; // width = 4
        let dem = vec![2100.0; 12]; // 3x4
        let release_area = vec![1.5; 12];

        let max_timesteps = 50;
        let number_runs = 5;

        // 2. Register and provision structural groups and descriptors
        engine
            .add_new_scenario(
                "Chamonix_Valley_A",
                number_runs,
                max_timesteps,
                y_coords,
                x_coords,
                &dem,
                &release_area,
                32.5,
                15000.0,
            )
            .unwrap();

        // Assert structural configuration details mapped correctly
        {
            let cached_scen = engine.scenarios.get("Chamonix_Valley_A").unwrap();
            assert_eq!(cached_scen.height, 3);
            assert_eq!(cached_scen.width, 4);
            assert_eq!(cached_scen.number_of_runs, 5);
        }

        // 3. Populate simulation runs (using immutable reference access)
        let simulated_cells = 3 * 4; // height * width
        let velocity_mock = vec![12.4f32; simulated_cells];
        let thickness_mock = vec![2.1f32; simulated_cells];
        let com_x_mock = vec![520.0f32; max_timesteps as usize];
        let com_y_mock = vec![940.0f32; max_timesteps as usize];

        // Call run insertion—notice no disk metadata re-parsing happens inside
        engine
            .add_new_run(
                "Chamonix_Valley_A",
                0, // run_id
                &velocity_mock,
                &thickness_mock,
                &com_x_mock,
                &com_y_mock,
                1200.5,
                28.2,
                0.15,
                0.22,
            )
            .unwrap();

        // 4. Verify structural targets are populated correctly on disk
        let velocity_chunk_file = zarr_path.join("Chamonix_Valley_A/peak_flow_velocity/c/0/0/0");
        assert!(zarr_path.join("Chamonix_Valley_A/zarr.json").exists());
        assert!(
            zarr_path
                .join("Chamonix_Valley_A/peak_flow_velocity/zarr.json")
                .exists()
        );
        assert!(
            velocity_chunk_file.exists(),
            "Data chunk wasn't written to the expected location."
        );
    }
}
