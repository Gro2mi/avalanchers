//! Native writer of the result stores, see [`crate::store_schema`] for the layout.
//!
//! A save is safe against crashes and other writers: every save holds a lock
//! on the store, a group's metadata is written after its arrays, and a run only
//! counts once `number_of_runs` has been replaced atomically. Anything left by
//! an interrupted save is overwritten by the next one.

use crate::store_schema::{
    self, Attributes, FORMAT_VERSION, NO_DATA_U32, RunField, RunRecord, SOFTWARE_VERSION, Scalar,
};
use compute_core::dem::Dem;
use half::f16;
use serde_json::{Value, json};
use std::fs::{self, File};
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

/// The `run` dimension grows in blocks of this many runs, which saves
/// rewriting the metadata of every array on each run.
const RUN_BLOCK: u64 = 1024;
/// Run columns hold one value per run, so a chunk spans many runs.
const COLUMN_CHUNK: u64 = 1024;
const TRACK_CHUNK: [u64; 2] = [64, 512];
const COMPRESSION_LEVEL: u8 = 3;

#[derive(thiserror::Error, Debug)]
pub enum OutputError {
    #[error(
        "Store '{path}' has format {found}, expected {expected}. Older stores are not supported, \
        delete it or choose another output path."
    )]
    Format {
        path: String,
        found: String,
        expected: &'static str,
    },

    #[error("Missing or invalid data: {0}")]
    InvalidData(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("Zarr storage error: {0}")]
    Zarr(#[from] zarrs::array::ArrayError),

    #[error("Zarr creation error: {0}")]
    ZarrCreate(#[from] zarrs::array::ArrayCreateError),

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

pub struct Output {
    path: PathBuf,
    store: Arc<FilesystemStore>,
    run_block: u64,
}

impl Output {
    /// Opens the store at `path`, creating it if needed.
    ///
    /// Errors if the store has another format than [`FORMAT_VERSION`].
    pub fn new(path: &str) -> Result<Self, OutputError> {
        let path = PathBuf::from(path);
        let path = if path.extension().is_some_and(|ext| ext == "zarr") {
            path
        } else {
            path.with_extension("zarr")
        };
        fs::create_dir_all(&path)?;
        let output = Self {
            store: Arc::new(FilesystemStore::new(&path)?),
            path,
            run_block: RUN_BLOCK,
        };

        let _lock = output.lock()?;
        if output.node_exists("") {
            output.check_format()?;
        } else {
            let mut root = GroupBuilder::new().build(output.store.clone(), "/")?;
            root.attributes_mut()
                .extend(store_schema::root_attributes());
            root.store_metadata()?;
        }
        Ok(output)
    }

    /// Appends a run to `<site>/<scenario>`, creating both as needed, and
    /// returns its run id.
    pub fn add_run(
        &self,
        site_name: &str,
        scenario_name: &str,
        dem: &Dem,
        release_area: &[f32],
        run: &RunRecord,
    ) -> Result<u64, OutputError> {
        Self::validate(dem, release_area, run)?;

        let _lock = self.lock()?;
        if !self.node_exists(site_name) {
            self.create_site(site_name, dem)?;
        }
        let scenario = format!("{site_name}/{scenario_name}");
        if !self.node_exists(&scenario) {
            self.create_scenario(&scenario, dem, release_area, run)?;
        }
        self.append_run(&scenario, dem, run)
    }

    fn validate(dem: &Dem, release_area: &[f32], run: &RunRecord) -> Result<(), OutputError> {
        let cells = dem.width * dem.height;
        let grids = run.grids(dem);
        let sizes = [
            ("release_area", release_area.len()),
            (grids[0].name, grids[0].data.len()),
            (grids[1].name, grids[1].data.len()),
        ];
        for (name, size) in sizes {
            if size != cells {
                return Err(OutputError::InvalidData(format!(
                    "{name} has {size} values, expected {cells}"
                )));
            }
        }
        let tracks = run.tracks();
        if tracks.iter().any(|t| t.data.len() != tracks[0].data.len()) {
            return Err(OutputError::InvalidData(
                "Per-timestep series differ in length".to_string(),
            ));
        }
        Ok(())
    }

    /// Exclusive lock on the store, released when the returned file is dropped.
    /// It sits next to the store because readers would trip over it inside.
    fn lock(&self) -> Result<File, OutputError> {
        let mut path = self.path.clone().into_os_string();
        path.push(".lock");
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        file.lock()?;
        Ok(file)
    }

    fn check_format(&self) -> Result<(), OutputError> {
        let root = Group::open(self.store.clone(), "/")?;
        let found = root
            .attributes()
            .get("avalanchers_format")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if found != FORMAT_VERSION {
            return Err(OutputError::Format {
                path: self.path.display().to_string(),
                found: found.to_string(),
                expected: FORMAT_VERSION,
            });
        }
        Ok(())
    }

    fn node_path(&self, node: &str) -> PathBuf {
        self.path.join(node.trim_start_matches('/'))
    }

    /// A group only exists once its metadata is there; a directory without it
    /// is what an interrupted save left behind.
    fn node_exists(&self, node: &str) -> bool {
        self.node_path(node).join("zarr.json").is_file()
    }

    fn blosc(typesize: usize) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>, OutputError> {
        Ok(vec![Arc::new(BloscCodec::new(
            BloscCompressor::Zstd,
            BloscCompressionLevel::try_from(COMPRESSION_LEVEL).expect("Invalid compression level"),
            None, // automatic blocksize
            BloscShuffleMode::Shuffle,
            Some(typesize),
        )?)])
    }

    #[allow(clippy::too_many_arguments)]
    fn create_array(
        &self,
        node: &str,
        shape: Vec<u64>,
        chunks: Vec<u64>,
        data_type: DataType,
        fill_value: FillValue,
        typesize: usize,
        dimension_names: &[&str],
        attributes: Attributes,
    ) -> Result<Array<FilesystemStore>, OutputError> {
        let mut array = ArrayBuilder::new(shape, chunks, data_type, fill_value)
            .bytes_to_bytes_codecs(Self::blosc(typesize)?)
            .dimension_names(Some(
                dimension_names
                    .iter()
                    .map(|d| Some(d.to_string()))
                    .collect::<Vec<_>>(),
            ))
            .build(self.store.clone(), &format!("/{node}"))?;
        array.attributes_mut().extend(attributes);
        array.store_metadata()?;
        Ok(array)
    }

    /// Writes the coordinates shared by the site and scenario groups.
    fn create_coordinates(&self, group: &str, dem: &Dem) -> Result<(), OutputError> {
        let axes = [
            (
                "x",
                store_schema::x_coordinates(dem),
                store_schema::x_attributes(),
            ),
            (
                "y",
                store_schema::y_coordinates(dem),
                store_schema::y_attributes(),
            ),
        ];
        for (axis, coordinates, attributes) in axes {
            let length = coordinates.len() as u64;
            let array = self.create_array(
                &format!("{group}/{axis}"),
                vec![length],
                vec![length],
                zarrs::array::data_type::float64(),
                FillValue::from(f64::NAN),
                8,
                &[axis],
                attributes,
            )?;
            array.store_chunk(&[0], coordinates)?;
        }

        if let Some(attributes) = store_schema::spatial_ref_attributes(dem) {
            let mut spatial_ref = ArrayBuilder::new(
                vec![],
                Vec::<u64>::new(),
                zarrs::array::data_type::int32(),
                FillValue::from(0i32),
            )
            .build(self.store.clone(), &format!("/{group}/spatial_ref"))?;
            spatial_ref.attributes_mut().extend(attributes);
            spatial_ref.store_metadata()?;
        }
        Ok(())
    }

    fn create_group(&self, node: &str, attributes: Attributes) -> Result<(), OutputError> {
        let mut group = GroupBuilder::new().build(self.store.clone(), &format!("/{node}"))?;
        group.attributes_mut().extend(attributes);
        group.store_metadata()?;
        Ok(())
    }

    fn create_site(&self, site: &str, dem: &Dem) -> Result<(), OutputError> {
        let (height, width) = (dem.height as u64, dem.width as u64);
        self.create_coordinates(site, dem)?;
        let array = self.create_array(
            &format!("{site}/dem"),
            vec![height, width],
            vec![height, width],
            zarrs::array::data_type::float32(),
            FillValue::from(f32::NAN),
            4,
            &["y", "x"],
            store_schema::dem_attributes(dem),
        )?;
        array.store_chunk(&[0, 0], dem.data1d.as_slice())?;

        self.create_group(site, store_schema::site_attributes(dem))
    }

    fn create_scenario(
        &self,
        scenario: &str,
        dem: &Dem,
        release_area: &[f32],
        run: &RunRecord,
    ) -> Result<(), OutputError> {
        let (height, width) = (dem.height as u64, dem.width as u64);
        self.create_coordinates(scenario, dem)?;

        let array = self.create_array(
            &format!("{scenario}/release_area"),
            vec![height, width],
            vec![height, width],
            zarrs::array::data_type::float32(),
            FillValue::from(0.0_f32),
            4,
            &["y", "x"],
            store_schema::release_area_attributes(dem),
        )?;
        array.store_chunk(&[0, 0], release_area)?;

        for RunField {
            name, attributes, ..
        } in run.grids(dem)
        {
            self.create_array(
                &format!("{scenario}/{name}"),
                vec![self.run_block, height, width],
                vec![1, height, width],
                zarrs::array::data_type::float16(),
                FillValue::from(f16::from_f32(0.0)),
                2,
                &["run", "y", "x"],
                attributes,
            )?;
        }

        for RunField {
            name,
            data,
            attributes,
        } in run.tracks()
        {
            self.create_array(
                &format!("{scenario}/{name}"),
                vec![self.run_block, track_width(data.len())],
                TRACK_CHUNK.to_vec(),
                zarrs::array::data_type::float32(),
                FillValue::from(f32::NAN),
                4,
                &["run", "timestep"],
                attributes,
            )?;
        }

        self.create_group(scenario, store_schema::scenario_attributes(0, &[]))
    }

    fn append_run(&self, scenario: &str, dem: &Dem, run: &RunRecord) -> Result<u64, OutputError> {
        let (height, width) = (dem.height as u64, dem.width as u64);
        let mut group = Group::open(self.store.clone(), &format!("/{scenario}"))?;
        let run_id = group
            .attributes()
            .get("number_of_runs")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                OutputError::InvalidData(format!("Scenario '{scenario}' has no number_of_runs"))
            })?;
        let software_version = register_software_version(group.attributes_mut())?;

        let capacity = self.grow_runs(scenario, run_id)?;
        self.grow_tracks(scenario, run)?;

        for RunField { name, data, .. } in run.grids(dem) {
            let array = Array::open(self.store.clone(), &format!("/{scenario}/{name}"))?;
            let subset =
                ArraySubset::new_with_start_shape(vec![run_id, 0, 0], vec![1, height, width])?;
            let data: Vec<f16> = data.iter().map(|v| f16::from_f32(*v)).collect();
            array.store_array_subset(&subset, &data)?;
        }

        for RunField { name, data, .. } in run.tracks() {
            let array = Array::open(self.store.clone(), &format!("/{scenario}/{name}"))?;
            // The whole row is written, as an interrupted save may have left
            // a longer series in this slot.
            let track_width = array.shape()[1];
            let mut row = vec![f32::NAN; track_width as usize];
            row[..data.len()].copy_from_slice(data);
            let subset = ArraySubset::new_with_start_shape(vec![run_id, 0], vec![1, track_width])?;
            array.store_array_subset(&subset, &row)?;
        }

        for column in run.columns(software_version) {
            let node = format!("{scenario}/{}", column.name);
            let array = if self.node_exists(&node) {
                Array::open(self.store.clone(), &format!("/{node}"))?
            } else {
                self.create_column(&node, capacity, column.value, column.attributes)?
            };
            let subset = ArraySubset::new_with_start_shape(vec![run_id], vec![1])?;
            match column.value {
                Scalar::U32(value) => array.store_array_subset(&subset, &[value])?,
                Scalar::F32(value) => array.store_array_subset(&subset, &[value])?,
                Scalar::F64(value) => array.store_array_subset(&subset, &[value])?,
            }
        }

        group
            .attributes_mut()
            .insert("number_of_runs".to_string(), json!(run_id + 1));
        self.commit_group(scenario, &group)?;
        Ok(run_id)
    }

    fn create_column(
        &self,
        node: &str,
        capacity: u64,
        value: Scalar,
        attributes: Attributes,
    ) -> Result<Array<FilesystemStore>, OutputError> {
        let (data_type, fill_value, typesize) = match value {
            Scalar::U32(_) => (
                zarrs::array::data_type::uint32(),
                FillValue::from(NO_DATA_U32),
                4,
            ),
            Scalar::F32(_) => (
                zarrs::array::data_type::float32(),
                FillValue::from(f32::NAN),
                4,
            ),
            Scalar::F64(_) => (
                zarrs::array::data_type::float64(),
                FillValue::from(f64::NAN),
                8,
            ),
        };
        self.create_array(
            node,
            vec![capacity],
            vec![COLUMN_CHUNK],
            data_type,
            fill_value,
            typesize,
            &["run"],
            attributes,
        )
    }

    /// Makes room for `run_id` in every array along `run`, which all have to
    /// stay the same length, and returns the capacity.
    fn grow_runs(&self, scenario: &str, run_id: u64) -> Result<u64, OutputError> {
        const CAPACITY_REFERENCE: &str = "peak_flow_velocity";
        let reference = Array::open(
            self.store.clone(),
            &format!("/{scenario}/{CAPACITY_REFERENCE}"),
        )?;
        let capacity = reference.shape()[0];
        if run_id < capacity {
            return Ok(capacity);
        }
        let capacity = (run_id / self.run_block + 1) * self.run_block;

        let mut names = Vec::new();
        for entry in fs::read_dir(self.node_path(scenario))? {
            let name = entry?.file_name().to_string_lossy().to_string();
            if self.node_exists(&format!("{scenario}/{name}")) {
                names.push(name);
            }
        }
        // The reference goes last, so an interrupted save grows them all again.
        names.sort_by_key(|name| name == CAPACITY_REFERENCE);

        for name in names {
            let mut array = Array::open(self.store.clone(), &format!("/{scenario}/{name}"))?;
            let along_run = array
                .dimension_names()
                .as_ref()
                .and_then(|names| names.first().cloned().flatten())
                .is_some_and(|dimension| dimension == "run");
            if along_run {
                let mut shape = array.shape().to_vec();
                shape[0] = capacity;
                array.set_shape(shape)?;
                array.store_metadata()?;
            }
        }
        Ok(capacity)
    }

    /// Widens the series when a run has more steps than any run before it.
    fn grow_tracks(&self, scenario: &str, run: &RunRecord) -> Result<(), OutputError> {
        for RunField { name, data, .. } in run.tracks() {
            let mut array = Array::open(self.store.clone(), &format!("/{scenario}/{name}"))?;
            let mut shape = array.shape().to_vec();
            if (data.len() as u64) > shape[1] {
                shape[1] = track_width(data.len());
                array.set_shape(shape)?;
                array.store_metadata()?;
            }
        }
        Ok(())
    }

    /// Replaces the metadata of `group` atomically; this is what commits a run.
    fn commit_group(&self, node: &str, group: &Group<FilesystemStore>) -> Result<(), OutputError> {
        let target = self.node_path(node).join("zarr.json");
        let temporary = target.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(group.metadata())?)?;
        fs::rename(&temporary, &target)?;
        Ok(())
    }
}

/// Width of the `timestep` dimension for a series of `length` values.
fn track_width(length: usize) -> u64 {
    (length as u64).div_ceil(TRACK_CHUNK[1]).max(1) * TRACK_CHUNK[1]
}

/// Returns the index of this build in `software_versions`, adding it if needed.
fn register_software_version(attributes: &mut Attributes) -> Result<u32, OutputError> {
    let versions = attributes
        .entry("software_versions")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| OutputError::InvalidData("software_versions is no list".to_string()))?;
    let index = versions
        .iter()
        .position(|version| version == SOFTWARE_VERSION)
        .unwrap_or_else(|| {
            versions.push(json!(SOFTWARE_VERSION));
            versions.len() - 1
        });
    Ok(index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zarr_writer::build_result_store;
    use compute_core::dem::Bounds;
    use compute_core::settings::SimSettings;
    use std::collections::BTreeMap;
    use std::path::Path;
    use tempfile::TempDir;

    const WIDTH: usize = 4;
    const HEIGHT: usize = 3;
    const CELLS: usize = WIDTH * HEIGHT;
    const SCENARIO: &str = "Chamonix/Valley_A";

    fn dem() -> Dem {
        let data1d: Vec<f32> = (0..CELLS).map(|i| 2000.0 + i as f32).collect();
        Dem {
            width: WIDTH,
            height: HEIGHT,
            bounds: Bounds {
                xmin: 500.0,
                xmax: 520.0,
                ymin: 900.0,
                ymax: 915.0,
            },
            minimum_elevation: Dem::calculate_minimum_elevation(&data1d),
            data1d,
            data: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            cell_size: 5.0,
            map_factor: 1.0,
            source: "test_dem".to_string(),
            projection: "EPSG:2056".to_string(),
        }
    }

    fn run(value: f32, steps: usize) -> RunRecord {
        RunRecord {
            peak_flow_velocity: vec![value; CELLS],
            peak_flow_thickness: vec![value / 2.0; CELLS],
            center_of_mass_x: vec![value; steps],
            center_of_mass_y: vec![value + 1.0; steps],
            time: vec![0.5; steps],
            release_volume_m3: value * 10.0,
            travel_length: value * 100.0,
            travel_angle: 30.0,
            final_timestep: steps as u32,
            simulated_time: 12.5,
            number_particles: 64,
            stop_flags: 1 << 31,
            created_at: 1_790_000_000.0,
            settings: SimSettings {
                released_particles_per_cell: 4,
                friction_coefficient: value,
                ..SimSettings::default()
            },
        }
    }

    fn store(name: &str) -> (TempDir, PathBuf, Output) {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join(name);
        let output = Output::new(path.to_str().unwrap()).unwrap();
        (tmp_dir, path, output)
    }

    fn add(output: &Output, run: &RunRecord) -> u64 {
        output
            .add_run("Chamonix", "Valley_A", &dem(), &[1.5; CELLS], run)
            .unwrap()
    }

    fn read<T: zarrs::array::FromArrayBytes>(output: &Output, node: &str) -> T {
        let array = Array::open(output.store.clone(), &format!("/{node}")).unwrap();
        let subset = ArraySubset::new_with_shape(array.shape().to_vec());
        array.retrieve_array_subset::<T>(&subset).unwrap()
    }

    fn number_of_runs(output: &Output) -> u64 {
        Group::open(output.store.clone(), &format!("/{SCENARIO}"))
            .unwrap()
            .attributes()["number_of_runs"]
            .as_u64()
            .unwrap()
    }

    fn shape(output: &Output, name: &str) -> Vec<u64> {
        Array::open(output.store.clone(), &format!("/{SCENARIO}/{name}"))
            .unwrap()
            .shape()
            .to_vec()
    }

    #[test_log::test]
    fn test_new_store_records_the_format() {
        let (_tmp_dir, path, output) = store("format.zarr");
        let root = Group::open(output.store.clone(), "/").unwrap();
        assert_eq!(root.attributes()["avalanchers_format"], FORMAT_VERSION);
        assert_eq!(root.attributes()["Conventions"], "CF-1.8");

        // Reopening a store of this format works.
        Output::new(path.to_str().unwrap()).unwrap();
    }

    #[test_log::test]
    fn test_other_formats_are_rejected() {
        let (_tmp_dir, path, output) = store("old.zarr");
        let mut root = Group::open(output.store.clone(), "/").unwrap();
        root.attributes_mut()
            .insert("avalanchers_format".to_string(), json!("0.1.0"));
        root.store_metadata().unwrap();

        let result = Output::new(path.to_str().unwrap());
        assert!(matches!(result, Err(OutputError::Format { found, .. }) if found == "0.1.0"));
    }

    #[test_log::test]
    fn test_runs_are_stored_at_their_id() {
        let (_tmp_dir, path, output) = store("runs.zarr");
        assert_eq!(add(&output, &run(10.0, 5)), 0);
        assert_eq!(add(&output, &run(20.0, 5)), 1);

        assert_eq!(number_of_runs(&output), 2);
        assert!(path.join("Chamonix/dem/c/0/0").exists());
        assert!(
            path.join(SCENARIO)
                .join("peak_flow_velocity/c/1/0/0")
                .exists()
        );

        let velocity: Vec<f16> = read(&output, &format!("{SCENARIO}/peak_flow_velocity"));
        assert_eq!(velocity[0].to_f32(), 10.0);
        assert_eq!(velocity[CELLS].to_f32(), 20.0);

        let friction: Vec<f32> = read(&output, &format!("{SCENARIO}/friction_coefficient"));
        assert_eq!(friction[..2], [10.0, 20.0]);
        assert!(friction[2].is_nan(), "unwritten runs are NaN, not 0");

        let particles: Vec<u32> = read(&output, &format!("{SCENARIO}/released_particles_per_cell"));
        assert_eq!(particles[..3], [4, 4, NO_DATA_U32]);

        let volume: Vec<f32> = read(&output, &format!("{SCENARIO}/release_volume_m3"));
        assert_eq!(volume[..2], [100.0, 200.0], "the volume belongs to the run");
    }

    /// Row 0 of every grid is the south edge, so `y` has to ascend with the rows.
    #[test_log::test]
    fn test_coordinates_are_ascending_cell_centers() {
        let (_tmp_dir, _path, output) = store("coordinates.zarr");
        add(&output, &run(1.0, 1));

        for group in ["Chamonix", SCENARIO] {
            let x: Vec<f64> = read(&output, &format!("{group}/x"));
            let y: Vec<f64> = read(&output, &format!("{group}/y"));
            assert_eq!(x, [502.5, 507.5, 512.5, 517.5]);
            assert_eq!(y, [902.5, 907.5, 912.5]);
        }
        let stored: Vec<f32> = read(&output, "Chamonix/dem");
        assert_eq!(stored, dem().data1d);
    }

    #[test_log::test]
    fn test_runs_grow_every_array_past_the_capacity() {
        let (_tmp_dir, _path, mut output) = store("grow.zarr");
        output.run_block = 2;
        for id in 0..5 {
            assert_eq!(add(&output, &run(id as f32 + 1.0, 3)), id);
        }

        assert_eq!(shape(&output, "peak_flow_velocity")[0], 6);
        assert_eq!(shape(&output, "center_of_mass_x")[0], 6);
        assert_eq!(shape(&output, "friction_coefficient")[0], 6);
        assert_eq!(shape(&output, "x"), [WIDTH as u64], "x is not along run");

        let friction: Vec<f32> = read(&output, &format!("{SCENARIO}/friction_coefficient"));
        assert_eq!(friction[..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test_log::test]
    fn test_longer_series_widen_the_scenario_without_truncation() {
        let (_tmp_dir, _path, output) = store("tracks.zarr");
        add(&output, &run(1.0, 300));
        let width = shape(&output, "time")[1];

        let steps = width as usize + 7;
        add(&output, &run(2.0, steps));
        let wider = shape(&output, "center_of_mass_x")[1];
        assert!(wider >= steps as u64);
        assert_eq!(shape(&output, "time")[1], wider);

        let x: Vec<f32> = read(&output, &format!("{SCENARIO}/center_of_mass_x"));
        let row = wider as usize;
        assert_eq!(x[row + steps - 1], 2.0, "the last step of the run is kept");
        assert!(x[row + steps].is_nan());
        assert_eq!(x[299], 1.0);
        assert!(x[300].is_nan());
    }

    #[test_log::test]
    fn test_settings_added_later_get_their_own_column() {
        let (_tmp_dir, path, output) = store("new_setting.zarr");
        add(&output, &run(1.0, 3));
        fs::remove_dir_all(path.join(SCENARIO).join("cfl")).unwrap();

        add(&output, &run(2.0, 3));
        let cfl: Vec<f32> = read(&output, &format!("{SCENARIO}/cfl"));
        assert!(cfl[0].is_nan());
        assert_eq!(cfl[1], SimSettings::default().cfl);
    }

    #[test_log::test]
    fn test_invalid_runs_leave_the_store_untouched() {
        let (_tmp_dir, path, output) = store("invalid.zarr");
        let mut bad = run(1.0, 3);
        bad.peak_flow_velocity.pop();

        let result = output.add_run("Chamonix", "Valley_A", &dem(), &[1.5; CELLS], &bad);
        assert!(matches!(result, Err(OutputError::InvalidData(_))));
        assert!(!path.join("Chamonix").exists());
    }

    #[test_log::test]
    fn test_interrupted_saves_are_recovered() {
        let (_tmp_dir, path, output) = store("interrupted.zarr");

        // A scenario that never reached its group metadata does not exist.
        add(&output, &run(1.0, 3));
        fs::remove_file(path.join(SCENARIO).join("zarr.json")).unwrap();
        assert_eq!(add(&output, &run(2.0, 3)), 0);

        // Data of a run that was not committed is replaced by the next one.
        let group = Group::open(output.store.clone(), &format!("/{SCENARIO}")).unwrap();
        add(&output, &run(3.0, 9));
        output.commit_group(SCENARIO, &group).unwrap();
        assert_eq!(number_of_runs(&output), 1);

        assert_eq!(add(&output, &run(4.0, 3)), 1);
        let x: Vec<f32> = read(&output, &format!("{SCENARIO}/center_of_mass_x"));
        let row = shape(&output, "center_of_mass_x")[1] as usize;
        assert_eq!(x[row + 2], 4.0);
        assert!(x[row + 3].is_nan(), "the longer abandoned series is gone");
    }

    #[test_log::test]
    fn test_concurrent_writers_do_not_lose_runs() {
        let (_tmp_dir, path, output) = store("concurrent.zarr");
        let path = path.to_str().unwrap().to_string();
        const WRITERS: usize = 4;
        const RUNS: usize = 10;

        std::thread::scope(|scope| {
            for writer in 0..WRITERS {
                let path = path.clone();
                scope.spawn(move || {
                    let output = Output::new(&path).unwrap();
                    for id in 0..RUNS {
                        add(&output, &run((writer * RUNS + id) as f32 + 1.0, 3));
                    }
                });
            }
        });

        assert_eq!(number_of_runs(&output), (WRITERS * RUNS) as u64);
        let friction: Vec<f32> = read(&output, &format!("{SCENARIO}/friction_coefficient"));
        let mut values: Vec<f32> = friction[..WRITERS * RUNS].to_vec();
        values.sort_by(f32::total_cmp);
        let expected: Vec<f32> = (1..=WRITERS * RUNS).map(|v| v as f32).collect();
        assert_eq!(values, expected);
    }

    /// `(data_type, dimension_names)` of every array below `path`.
    fn schema(path: &Path) -> BTreeMap<String, (Value, Value)> {
        let mut nodes = BTreeMap::new();
        let mut pending = vec![path.to_path_buf()];
        while let Some(directory) = pending.pop() {
            let metadata = directory.join("zarr.json");
            if metadata.is_file() {
                let parsed: Value = serde_json::from_slice(&fs::read(metadata).unwrap()).unwrap();
                let name = directory.strip_prefix(path).unwrap().display().to_string();
                nodes.insert(
                    name.replace('\\', "/"),
                    (
                        parsed["data_type"].clone(),
                        parsed["dimension_names"].clone(),
                    ),
                );
            }
            for entry in fs::read_dir(&directory).unwrap() {
                let entry = entry.unwrap();
                if entry.path().is_dir() && entry.file_name() != "c" {
                    pending.push(entry.path());
                }
            }
        }
        nodes
    }

    fn write_browser_store(path: &Path) {
        let entries =
            build_result_store("Chamonix", "Valley_A", &dem(), &[1.5; CELLS], &run(1.0, 5));
        for entry in entries {
            let file = path.join(&entry.path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, entry.bytes).unwrap();
        }
    }

    #[test_log::test]
    fn test_browser_stores_have_the_native_layout() {
        let (tmp_dir, native_path, output) = store("native.zarr");
        add(&output, &run(1.0, 5));

        let browser_path = tmp_dir.path().join("browser.zarr");
        write_browser_store(&browser_path);

        assert_eq!(schema(&browser_path), schema(&native_path));
    }

    #[test_log::test]
    fn test_native_runs_append_to_browser_stores() {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("browser.zarr");
        write_browser_store(&path);

        let output = Output::new(path.to_str().unwrap()).unwrap();
        assert_eq!(add(&output, &run(2.0, 9)), 1);

        let friction: Vec<f32> = read(&output, &format!("{SCENARIO}/friction_coefficient"));
        assert_eq!(friction[..2], [1.0, 2.0]);
        let velocity: Vec<f16> = read(&output, &format!("{SCENARIO}/peak_flow_velocity"));
        assert_eq!(velocity[CELLS].to_f32(), 2.0);
    }
}
