//! Minimal Zarr v3 writer used to export simulation results from the browser.
//!
//! The canonical writer in [`crate::output`] builds on `zarrs`, which cannot be
//! compiled for `wasm32-unknown-unknown`. This module emits the same store
//! layout as plain in-memory files using the uncompressed `bytes` codec, so the
//! result can be written to disk by the browser and re-opened by the frontend.

use compute_core::dem::Dem;
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

    pub fn add_group(&mut self, path: &str, attributes: Value) {
        let metadata = json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": attributes,
        });
        self.push_metadata(path, &metadata);
    }

    /// Adds an `f32` array stored as a single uncompressed chunk.
    pub fn add_f32_array(
        &mut self,
        path: &str,
        shape: &[usize],
        dimension_names: &[&str],
        data: &[f32],
    ) {
        let expected: usize = shape.iter().product();
        debug_assert_eq!(expected, data.len(), "array data does not match its shape");

        let metadata = json!({
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": { "chunk_shape": shape },
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": { "separator": "/" },
            },
            "fill_value": 0.0,
            "codecs": [{
                "name": "bytes",
                "configuration": { "endian": "little" },
            }],
            "dimension_names": dimension_names,
        });
        self.push_metadata(path, &metadata);

        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        // The single chunk always sits at index zero of every dimension.
        let index = vec!["0"; shape.len()].join("/");
        self.entries.push(ZarrEntry {
            path: format!("{path}/c/{index}"),
            bytes,
        });
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

/// Grids exported for a single simulation run.
pub struct ResultGrids<'a> {
    pub release_areas: &'a [f32],
    pub peak_velocity: &'a [f32],
    pub peak_flow_thickness: &'a [f32],
}

/// Builds a `<site>/<scenario>` store holding the DEM and the run results.
pub fn build_result_store(
    site_name: &str,
    scenario_name: &str,
    dem: &Dem,
    grids: &ResultGrids,
    settings: Value,
) -> Vec<ZarrEntry> {
    let mut builder = ZarrStoreBuilder::new();
    let shape = [dem.height, dem.width];

    builder.add_group(
        "",
        json!({
            "title": "Avalanche simulation results",
            "avalanchers_version": env!("CARGO_PKG_VERSION"),
            "avalanchers_repo": "https://github.com/Gro2mi/avalanchers",
            "avalanchers_format": "site/scenario",
        }),
    );

    builder.add_group(
        site_name,
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
                "ymax": dem.bounds.ymax,
            },
        }),
    );
    builder.add_f32_array(&format!("{site_name}/x"), &[dem.width], &["x"], &dem.x);
    builder.add_f32_array(&format!("{site_name}/y"), &[dem.height], &["y"], &dem.y);
    builder.add_f32_array(
        &format!("{site_name}/dem"),
        &shape,
        &["y", "x"],
        &dem.data1d,
    );

    let scenario_path = format!("{site_name}/{scenario_name}");
    builder.add_group(&scenario_path, json!({ "settings": settings }));
    builder.add_f32_array(
        &format!("{scenario_path}/release_area"),
        &shape,
        &["y", "x"],
        grids.release_areas,
    );
    builder.add_f32_array(
        &format!("{scenario_path}/peak_flow_velocity"),
        &shape,
        &["y", "x"],
        grids.peak_velocity,
    );
    builder.add_f32_array(
        &format!("{scenario_path}/peak_flow_thickness"),
        &shape,
        &["y", "x"],
        grids.peak_flow_thickness,
    );

    builder.into_entries()
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::dem::Bounds;

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
            x: vec![0.0, 5.0, 10.0, 15.0],
            y: vec![0.0, 5.0, 10.0],
            source: "test.asc".to_string(),
            projection: "unknown".to_string(),
        }
    }

    fn build() -> Vec<ZarrEntry> {
        let dem = test_dem();
        let cells = dem.width * dem.height;
        let release: Vec<f32> = vec![1.0; cells];
        let velocity: Vec<f32> = (0..cells).map(|i| i as f32).collect();
        let thickness: Vec<f32> = vec![0.5; cells];
        build_result_store(
            "site-a_1234",
            "scenario-b_5678",
            &dem,
            &ResultGrids {
                release_areas: &release,
                peak_velocity: &velocity,
                peak_flow_thickness: &thickness,
            },
            json!({ "cfl": 0.5 }),
        )
    }

    #[test]
    fn test_store_contains_expected_layout() {
        let paths: Vec<String> = build().into_iter().map(|e| e.path).collect();
        for expected in [
            "zarr.json",
            "site-a_1234/zarr.json",
            "site-a_1234/x/zarr.json",
            "site-a_1234/x/c/0",
            "site-a_1234/dem/zarr.json",
            "site-a_1234/dem/c/0/0",
            "site-a_1234/scenario-b_5678/zarr.json",
            "site-a_1234/scenario-b_5678/release_area/c/0/0",
            "site-a_1234/scenario-b_5678/peak_flow_velocity/c/0/0",
            "site-a_1234/scenario-b_5678/peak_flow_thickness/c/0/0",
        ] {
            assert!(paths.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn test_array_metadata_and_chunk_bytes_match_the_dem() {
        let entries = build();
        let dem = test_dem();

        let metadata = entries
            .iter()
            .find(|e| e.path == "site-a_1234/dem/zarr.json")
            .expect("dem metadata");
        let parsed: Value = serde_json::from_slice(&metadata.bytes).unwrap();
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
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, dem.data1d);
    }

    #[test]
    fn test_root_and_site_groups_are_valid_metadata() {
        let entries = build();
        for path in ["zarr.json", "site-a_1234/zarr.json"] {
            let entry = entries.iter().find(|e| e.path == path).unwrap();
            let parsed: Value = serde_json::from_slice(&entry.bytes).unwrap();
            assert_eq!(parsed["zarr_format"], 3);
            assert_eq!(parsed["node_type"], "group");
        }
    }
}
