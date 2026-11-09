//! Minimal Zarr v3 writer used to export simulation results from the browser.
//!
//! The canonical writer in [`crate::output`] builds on `zarrs`, which cannot be
//! compiled for `wasm32-unknown-unknown`. This module emits the same layout, see
//! [`crate::store_schema`], as plain in-memory files using the uncompressed
//! `bytes` codec, so the result can be written to disk by the browser and
//! re-opened by every reader of the native stores.

use crate::store_schema::{
    self, Attributes, FORMAT_VERSION, NO_DATA_U32, Outline, RunField, RunRecord, SOFTWARE_VERSION,
    Scalar,
};
use compute_core::dem::Dem;
use half::f16;
use serde_json::{Value, json};

/// A single file of a Zarr store, keyed by its store-relative path.
#[derive(Debug, Clone)]
pub struct ZarrEntry {
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Default)]
pub struct ZarrStoreBuilder {
    entries: Vec<ZarrEntry>,
}

impl ZarrStoreBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_group(&mut self, path: &str, attributes: Attributes) {
        let metadata = json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attributes,
        });
        self.push_metadata(path, &metadata);
    }

    /// Adds an array stored as a single uncompressed chunk. `bytes` is the
    /// little-endian content, `None` leaves the array at its fill value.
    #[allow(clippy::too_many_arguments)]
    fn add_array(
        &mut self,
        path: &str,
        data_type: &str,
        fill_value: Value,
        shape: &[usize],
        dimension_names: &[&str],
        attributes: Attributes,
        bytes: Option<Vec<u8>>,
    ) {
        let mut metadata = json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": data_type,
            "chunk_grid": {
                "name": "regular",
                "configuration": { "chunk_shape": shape },
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": { "separator": "/" },
            },
            "fill_value": fill_value,
            "codecs": [{
                "name": "bytes",
                "configuration": { "endian": "little" },
            }],
            "attributes": attributes,
        });
        if !dimension_names.is_empty() {
            metadata["dimension_names"] = json!(dimension_names);
        }
        self.push_metadata(path, &metadata);

        if let Some(bytes) = bytes {
            // The single chunk always sits at index zero of every dimension.
            let index = vec!["0"; shape.len()].join("/");
            self.entries.push(ZarrEntry {
                path: format!("{path}/c/{index}"),
                bytes,
            });
        }
    }

    fn add_f64_array(&mut self, path: &str, dimension: &str, attributes: Attributes, data: &[f64]) {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_array(
            path,
            "float64",
            json!("NaN"),
            &[data.len()],
            &[dimension],
            attributes,
            Some(bytes),
        );
    }

    fn add_f32_array(
        &mut self,
        path: &str,
        shape: &[usize],
        dimension_names: &[&str],
        fill_value: Value,
        attributes: Attributes,
        data: &[f32],
    ) {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_array(
            path,
            "float32",
            fill_value,
            shape,
            dimension_names,
            attributes,
            Some(bytes),
        );
    }

    fn add_f16_array(
        &mut self,
        path: &str,
        shape: &[usize],
        dimension_names: &[&str],
        attributes: Attributes,
        data: &[f32],
    ) {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        let bytes = data
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        self.add_array(
            path,
            "float16",
            json!(0.0),
            shape,
            dimension_names,
            attributes,
            Some(bytes),
        );
    }

    fn add_u8_array(
        &mut self,
        path: &str,
        shape: &[usize],
        dimension_names: &[&str],
        attributes: Attributes,
        data: Vec<u8>,
    ) {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        self.add_array(
            path,
            "uint8",
            json!(0),
            shape,
            dimension_names,
            attributes,
            Some(data),
        );
    }

    fn add_column(&mut self, path: &str, value: Scalar, attributes: Attributes) {
        let (data_type, fill_value, bytes) = match value {
            Scalar::U32(v) => ("uint32", json!(NO_DATA_U32), v.to_le_bytes().to_vec()),
            Scalar::F32(v) => ("float32", json!("NaN"), v.to_le_bytes().to_vec()),
            Scalar::F64(v) => ("float64", json!("NaN"), v.to_le_bytes().to_vec()),
        };
        self.add_array(
            path,
            data_type,
            fill_value,
            &[1],
            &["run"],
            attributes,
            Some(bytes),
        );
    }

    /// Adds the coordinates shared by the site and scenario groups.
    fn add_coordinates(&mut self, group: &str, dem: &Dem) {
        self.add_f64_array(
            &format!("{group}/x"),
            "x",
            store_schema::x_attributes(),
            &store_schema::x_coordinates(dem),
        );
        self.add_f64_array(
            &format!("{group}/y"),
            "y",
            store_schema::y_attributes(),
            &store_schema::y_coordinates(dem),
        );
        if let Some(attributes) = store_schema::spatial_ref_attributes(dem) {
            self.add_array(
                &format!("{group}/spatial_ref"),
                "int32",
                json!(0),
                &[],
                &[],
                attributes,
                None,
            );
        }
    }

    fn push_metadata(&mut self, path: &str, metadata: &Value) {
        let path = if path.is_empty() {
            "zarr.json".to_string()
        } else {
            format!("{path}/zarr.json")
        };
        self.entries.push(ZarrEntry {
            path,
            bytes: metadata.to_string().into_bytes(),
        });
    }

    pub fn into_entries(self) -> Vec<ZarrEntry> {
        self.entries
    }
}

/// Builds a `<site>/<scenario>` store holding the DEM, the optional outline
/// and a single run.
pub fn build_result_store(
    site_name: &str,
    scenario_name: &str,
    dem: &Dem,
    release_areas: &[f32],
    outline: Option<Outline<'_>>,
    run: &RunRecord,
) -> Vec<ZarrEntry> {
    let mut builder = ZarrStoreBuilder::new();
    let (height, width) = (dem.height, dem.width);
    debug_assert_eq!(
        store_schema::root_attributes()["avalanchers_format"],
        FORMAT_VERSION
    );

    builder.add_group("", store_schema::root_attributes());

    builder.add_group(site_name, store_schema::site_attributes(dem));
    builder.add_coordinates(site_name, dem);
    builder.add_f32_array(
        &format!("{site_name}/dem"),
        &[height, width],
        &["y", "x"],
        json!("NaN"),
        store_schema::dem_attributes(dem),
        &dem.data1d,
    );

    let scenario_path = format!("{site_name}/{scenario_name}");
    let mut scenario_attributes =
        store_schema::scenario_attributes(1, &[SOFTWARE_VERSION], &[&run.label]);
    if let Some(outline) = outline {
        scenario_attributes.insert("outline_hash".to_string(), json!(outline.hash()));
    }
    builder.add_group(&scenario_path, scenario_attributes);
    builder.add_coordinates(&scenario_path, dem);
    builder.add_f32_array(
        &format!("{scenario_path}/release_area"),
        &[height, width],
        &["y", "x"],
        json!(0.0),
        store_schema::release_area_attributes(dem),
        release_areas,
    );
    if let Some(outline) = outline {
        builder.add_u8_array(
            &format!("{scenario_path}/outline"),
            &[height, width],
            &["y", "x"],
            store_schema::outline_attributes(dem, outline.source),
            outline.to_u8(),
        );
    }
    for RunField {
        name,
        data,
        attributes,
    } in run.grids(dem)
    {
        builder.add_f16_array(
            &format!("{scenario_path}/{name}"),
            &[1, height, width],
            &["run", "y", "x"],
            attributes,
            data,
        );
    }
    for RunField {
        name,
        data,
        attributes,
    } in run.tracks()
    {
        builder.add_f32_array(
            &format!("{scenario_path}/{name}"),
            &[1, data.len()],
            &["run", "timestep"],
            json!("NaN"),
            attributes,
            data,
        );
    }
    for column in run.columns(0, 0) {
        builder.add_column(
            &format!("{scenario_path}/{}", column.name),
            column.value,
            column.attributes,
        );
    }

    builder.into_entries()
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::dem::Bounds;
    use compute_core::settings::SimSettings;

    fn test_dem() -> Dem {
        let (width, height) = (4usize, 3usize);
        let data1d: Vec<f32> = (0..width * height).map(|i| 1000.0 + i as f32).collect();
        Dem {
            data: Vec::new(),
            data1d,
            width,
            height,
            cell_size: 5.0,
            map_factor: 1.0,
            minimum_elevation: 1000.0,
            bounds: Bounds {
                xmin: 0.0,
                xmax: 20.0,
                ymin: 0.0,
                ymax: 15.0,
            },
            x: vec![2.5, 7.5, 12.5, 17.5],
            y: vec![2.5, 7.5, 12.5],
            source: "test.asc".to_string(),
            projection: "EPSG:2056".to_string(),
        }
    }

    fn build() -> Vec<ZarrEntry> {
        let dem = test_dem();
        let cells = dem.width * dem.height;
        let release: Vec<f32> = vec![1.0; cells];
        let run = RunRecord {
            peak_flow_velocity: (0..cells).map(|i| i as f32).collect(),
            peak_flow_thickness: vec![0.5; cells],
            center_of_mass_x: vec![f32::NAN, 1.0, 2.0],
            center_of_mass_y: vec![f32::NAN, 3.0, 4.0],
            time: vec![f32::NAN, 0.5, 1.0],
            release_volume_m3: 12.0,
            travel_length: 100.0,
            travel_angle: 30.0,
            final_timestep: 2,
            simulated_time: 1.0,
            number_particles: 8,
            stop_flags: 0,
            created_at: 1.0,
            label: String::new(),
            settings: SimSettings::default(),
        };
        build_result_store("site-a_1234", "scenario-b_5678", &dem, &release, None, &run)
    }

    fn metadata(entries: &[ZarrEntry], path: &str) -> Value {
        let entry = entries.iter().find(|e| e.path == path).expect(path);
        serde_json::from_slice(&entry.bytes).unwrap()
    }

    #[test]
    fn test_store_contains_expected_layout() {
        let paths: Vec<String> = build().into_iter().map(|e| e.path).collect();
        for expected in [
            "zarr.json",
            "site-a_1234/zarr.json",
            "site-a_1234/x/zarr.json",
            "site-a_1234/x/c/0",
            "site-a_1234/spatial_ref/zarr.json",
            "site-a_1234/dem/zarr.json",
            "site-a_1234/dem/c/0/0",
            "site-a_1234/scenario-b_5678/zarr.json",
            "site-a_1234/scenario-b_5678/x/c/0",
            "site-a_1234/scenario-b_5678/release_area/c/0/0",
            "site-a_1234/scenario-b_5678/peak_flow_velocity/c/0/0/0",
            "site-a_1234/scenario-b_5678/peak_flow_thickness/c/0/0/0",
            "site-a_1234/scenario-b_5678/center_of_mass_x/c/0/0",
            "site-a_1234/scenario-b_5678/time/c/0/0",
            "site-a_1234/scenario-b_5678/released_particles_per_cell/c/0",
        ] {
            assert!(paths.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn test_array_metadata_and_chunk_bytes_match_the_dem() {
        let entries = build();
        let dem = test_dem();

        let parsed = metadata(&entries, "site-a_1234/dem/zarr.json");
        assert_eq!(parsed["node_type"], "array");
        assert_eq!(parsed["data_type"], "float32");
        assert_eq!(parsed["shape"], json!([dem.height, dem.width]));
        assert_eq!(parsed["codecs"][0]["name"], "bytes");

        let chunk = entries
            .iter()
            .find(|e| e.path == "site-a_1234/dem/c/0/0")
            .expect("dem chunk");
        assert_eq!(chunk.bytes.len(), dem.width * dem.height * 4);

        let decoded: Vec<f32> = chunk
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, dem.data1d);
    }

    #[test]
    fn test_groups_are_valid_metadata() {
        let entries = build();
        for path in ["zarr.json", "site-a_1234/zarr.json"] {
            let parsed = metadata(&entries, path);
            assert_eq!(parsed["zarr_format"], 3);
            assert_eq!(parsed["node_type"], "group");
        }
        assert_eq!(
            metadata(&entries, "zarr.json")["attributes"]["avalanchers_format"],
            FORMAT_VERSION
        );
        let scenario = metadata(&entries, "site-a_1234/scenario-b_5678/zarr.json");
        assert_eq!(scenario["attributes"]["number_of_runs"], 1);
    }
}
