//! Reads the inputs of a run back from a result store, see
//! [`crate::store_schema`] for the layout.
//!
//! `dem_path` and `release_areas_path` accept `<store>.zarr[/<site>[/<scenario>]]`.
//! Sites and scenarios match by their full name, the name without the hash, or
//! the hash; a level left out is taken if it is the only one in the store.

use crate::output::{OutputError, check_format};
use crate::store_schema::split_store_path;
use compute_core::dem::{Bounds, Dem};
use compute_core::settings::Settings;
use compute_core::utils::cell_centers;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::sync::Arc;
#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use zarrs::array::{Array, ArraySubset};
use zarrs::group::Group;
use zarrs_filesystem::FilesystemStore;

/// The site and scenario a store path selects.
#[derive(Clone, Debug, PartialEq)]
struct Selection {
    store: String,
    site: String,
    scenario: Option<String>,
}

impl Selection {
    fn parse(path: &str) -> Result<Self, OutputError> {
        let (store, nodes) = split_store_path(path)
            .ok_or_else(|| invalid(format!("'{path}' is not a path into a Zarr store")))?;
        if nodes.len() > 2 {
            return Err(invalid(format!(
                "'{path}' goes below <store>.zarr/<site>/<scenario>"
            )));
        }
        let root = Path::new(store);
        if !root.join("zarr.json").is_file() {
            return Err(invalid(format!("No Zarr store at '{store}'")));
        }
        check_format(&Arc::new(FilesystemStore::new(root)?), root)?;

        let sites = child_groups(root)?;
        let site = match nodes.first() {
            Some(query) => find(&sites, query, "site", store)?,
            None => match sites.as_slice() {
                [site] => site.clone(),
                _ => {
                    return Err(invalid(format!(
                        "Choose a site with <store>.zarr/<site>, '{store}' has: {}",
                        sites.join(", ")
                    )));
                }
            },
        };

        let scenarios = child_groups(&root.join(&site))?;
        let scenario = match nodes.get(1) {
            Some(query) => Some(find(&scenarios, query, "scenario", &site)?),
            None => match scenarios.as_slice() {
                [scenario] => Some(scenario.clone()),
                _ => None,
            },
        };

        Ok(Self {
            store: store.to_string(),
            site,
            scenario,
        })
    }

    fn site_path(&self) -> String {
        format!("{}/{}", self.store, self.site)
    }

    fn scenario_path(&self) -> Option<String> {
        let scenario = self.scenario.as_ref()?;
        Some(format!("{}/{scenario}", self.site_path()))
    }

    fn open(&self) -> Result<Arc<FilesystemStore>, OutputError> {
        Ok(Arc::new(FilesystemStore::new(&self.store)?))
    }
}

fn invalid(message: String) -> OutputError {
    OutputError::InvalidData(message)
}

/// Names of the groups directly below `directory`, which leaves out arrays.
fn child_groups(directory: &Path) -> Result<Vec<String>, OutputError> {
    let mut names = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.path().join("zarr.json");
        if metadata.is_file() {
            let metadata: Value = serde_json::from_slice(&fs::read(metadata)?)?;
            if metadata["node_type"] == "group" {
                names.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

fn find(names: &[String], query: &str, level: &str, parent: &str) -> Result<String, OutputError> {
    if names.iter().any(|name| name == query) {
        return Ok(query.to_string());
    }
    let matches: Vec<&String> = names
        .iter()
        .filter(|name| {
            name.rsplit_once('_')
                .is_some_and(|(stem, hash)| stem == query || hash == query)
        })
        .collect();
    match matches.as_slice() {
        [name] => Ok((*name).clone()),
        [] => Err(invalid(format!(
            "No {level} '{query}' in '{parent}', it has: {}",
            names.join(", ")
        ))),
        _ => Err(invalid(format!(
            "Several {level}s in '{parent}' match '{query}': {}",
            names.join(", ")
        ))),
    }
}

/// Expands the store paths in `settings` to `(dem_path, release_areas_path)`,
/// with the release coming from the store unless `settings` chose another one.
///
/// Errors if the store offers several scenarios and none was chosen. Without
/// any scenario the release areas are left to be calculated.
pub fn resolve_paths(settings: &Settings) -> Result<(Option<String>, Option<String>), OutputError> {
    let mut release_areas_path = match settings.release_areas_path.as_deref() {
        Some(path) if split_store_path(path).is_some() => {
            let selection = Selection::parse(path)?;
            Some(selection.scenario_path().ok_or_else(|| no_scenario(path))?)
        }
        path => path.map(str::to_string),
    };

    let Some(dem_path) = settings.dem_path.as_deref() else {
        return Ok((None, release_areas_path));
    };
    let Some((_, nodes)) = split_store_path(dem_path) else {
        return Ok((Some(dem_path.to_string()), release_areas_path));
    };

    let selection = Selection::parse(dem_path)?;
    let scenario_named = nodes.len() == 2;
    if release_areas_path.is_none() && (scenario_named || settings.release_area_fraction.is_none())
    {
        release_areas_path = selection.scenario_path();
        match &release_areas_path {
            Some(path) => info!("Using the release areas of {path}"),
            None => {
                let scenarios = child_groups(&Path::new(&selection.store).join(&selection.site))?;
                if !scenarios.is_empty() {
                    return Err(invalid(format!(
                        "Site '{}' has several scenarios, choose one with \
                        <store>.zarr/<site>/<scenario> or set release_areas_path: {}",
                        selection.site,
                        scenarios.join(", ")
                    )));
                }
                warn!(
                    "Site '{}' has no scenario, so its release areas will be calculated",
                    selection.site
                );
            }
        }
    }
    Ok((Some(selection.site_path()), release_areas_path))
}

fn no_scenario(path: &str) -> OutputError {
    invalid(format!(
        "'{path}' selects no scenario, use <store>.zarr/<site>/<scenario>"
    ))
}

fn read_array(
    store: &Arc<FilesystemStore>,
    node: &str,
) -> Result<(Vec<f32>, Vec<u64>), OutputError> {
    let array = Array::open(store.clone(), node)?;
    let shape = array.shape().to_vec();
    if shape.len() != 2 {
        return Err(invalid(format!("'{node}' must be 2D, got shape {shape:?}")));
    }
    let subset = ArraySubset::new_with_shape(shape.clone());
    Ok((array.retrieve_array_subset::<Vec<f32>>(&subset)?, shape))
}

/// Reads the DEM of the site `path` selects. The elevation derived fields are
/// left to the caller, see `load_dem`.
pub(crate) fn read_dem(path: &str) -> Result<Dem, OutputError> {
    let selection = Selection::parse(path)?;
    let store = selection.open()?;
    let site = format!("/{}", selection.site);
    let group = Group::open(store.clone(), &site)?;
    let attributes = group.attributes();

    let number = |value: Option<&Value>, name: &str| {
        value
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .ok_or_else(|| invalid(format!("Site '{}' has no {name}", selection.site)))
    };
    let bounds = attributes.get("dem_bounds");
    let bound = |name: &str| number(bounds.and_then(|bounds| bounds.get(name)), name);
    let bounds = Bounds {
        xmin: bound("xmin")?,
        xmax: bound("xmax")?,
        ymin: bound("ymin")?,
        ymax: bound("ymax")?,
    };
    let text = |name: &str| {
        attributes
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };

    let (data1d, shape) = read_array(&store, &format!("{site}/dem"))?;
    let (height, width) = (shape[0] as usize, shape[1] as usize);
    Ok(Dem {
        width,
        height,
        x: cell_centers(bounds.xmin, bounds.xmax, width),
        y: cell_centers(bounds.ymin, bounds.ymax, height),
        bounds,
        data1d,
        data: Vec::new(),
        cell_size: number(attributes.get("dem_cell_size"), "dem_cell_size")?,
        map_factor: number(attributes.get("dem_map_factor"), "dem_map_factor")?,
        minimum_elevation: f32::INFINITY,
        source: text("dem_source"),
        projection: text("dem_projection"),
    })
}

/// Reads the release areas of the scenario `path` selects.
pub(crate) fn read_release_areas(path: &str) -> Result<Vec<f32>, OutputError> {
    let selection = Selection::parse(path)?;
    let scenario = selection
        .scenario
        .as_ref()
        .ok_or_else(|| no_scenario(path))?;
    let node = format!("/{}/{scenario}/release_area", selection.site);
    Ok(read_array(&selection.open()?, &node)?.0)
}

/// Reads the outline of the scenario `path` selects together with the
/// `source` it was stored with, or `None` if the scenario has no outline.
pub fn read_outline(path: &str) -> Result<Option<(Vec<bool>, String)>, OutputError> {
    let selection = Selection::parse(path)?;
    let scenario = selection
        .scenario
        .as_ref()
        .ok_or_else(|| no_scenario(path))?;
    let node = format!("{}/{scenario}/outline", selection.site);
    if !Path::new(&selection.store)
        .join(&node)
        .join("zarr.json")
        .is_file()
    {
        return Ok(None);
    }
    let array = Array::open(selection.open()?, &format!("/{node}"))?;
    let subset = ArraySubset::new_with_shape(array.shape().to_vec());
    let cells: Vec<u8> = array.retrieve_array_subset::<Vec<u8>>(&subset)?;
    let source = array
        .attributes()
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(Some((
        cells.into_iter().map(|cell| cell != 0).collect(),
        source,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::Output;
    use crate::store_schema::RunRecord;
    use crate::{load_dem, load_release_areas};
    use compute_core::settings::SimSettings;
    use pollster::block_on;
    use tempfile::TempDir;

    const CELLS: usize = 6;

    fn dem() -> Dem {
        Dem {
            width: 3,
            height: 2,
            bounds: Bounds {
                xmin: 2_600_000.0,
                xmax: 2_600_015.0,
                ymin: 1_200_000.0,
                ymax: 1_200_010.0,
            },
            x: cell_centers(2_600_000.0, 2_600_015.0, 3),
            y: cell_centers(1_200_000.0, 1_200_010.0, 2),
            data1d: vec![1000.0, 1001.0, f32::NAN, 1003.0, 1004.0, 1005.0],
            data: Vec::new(),
            cell_size: 5.0,
            map_factor: 1.0,
            minimum_elevation: 1000.0,
            source: "dem.asc".to_string(),
            projection: "EPSG:2056".to_string(),
        }
    }

    fn run() -> RunRecord {
        RunRecord {
            peak_flow_velocity: vec![1.0; CELLS],
            peak_flow_thickness: vec![1.0; CELLS],
            center_of_mass_x: vec![1.0],
            center_of_mass_y: vec![1.0],
            time: vec![1.0],
            release_volume_m3: 1.0,
            travel_length: 1.0,
            travel_angle: 1.0,
            final_timestep: 1,
            simulated_time: 1.0,
            number_particles: 1,
            stop_flags: 0,
            created_at: 1.0,
            label: String::new(),
            settings: SimSettings::default(),
        }
    }

    const RELEASE_A: [f32; CELLS] = [0.1, 0.0, 0.0, 1.5, 0.0, 0.0];
    const RELEASE_B: [f32; CELLS] = [0.0, 2.5, 0.0, 0.0, 0.0, 0.0];

    /// A store with `scenarios` scenarios of the site `pass_abc`.
    fn store(scenarios: &[(&str, &[f32])]) -> (TempDir, String) {
        let tmp_dir = TempDir::new().unwrap();
        let path = tmp_dir.path().join("runs.zarr").display().to_string();
        let output = Output::new(&path).unwrap();
        for (name, release) in scenarios {
            output
                .add_run("pass_abc", name, &dem(), release, None, &run())
                .unwrap();
        }
        (tmp_dir, path)
    }

    fn resolve(
        dem_path: &str,
        release_areas_path: Option<&str>,
        release_area_fraction: Option<f32>,
    ) -> (Option<String>, Option<String>) {
        resolve_paths(&Settings {
            dem_path: Some(dem_path.to_string()),
            release_areas_path: release_areas_path.map(str::to_string),
            release_area_fraction,
            ..Default::default()
        })
        .unwrap()
    }

    #[test_log::test]
    fn test_dem_round_trips_through_the_store() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        let loaded = block_on(load_dem(&path)).unwrap();
        let original = dem();

        assert_eq!(loaded.width, original.width);
        assert_eq!(loaded.bounds, original.bounds);
        assert_eq!(loaded.x, original.x);
        assert_eq!(loaded.y, original.y);
        assert_eq!(loaded.source, original.source);
        assert_eq!(loaded.minimum_elevation, 1000.0);
        assert_eq!(loaded.data.len(), original.height);
        assert_eq!(
            loaded.calculate_hash(),
            original.calculate_hash(),
            "the site keeps its identity"
        );
    }

    #[test_log::test]
    fn test_release_round_trips_through_the_store() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        let loaded = block_on(load_release_areas(
            &format!("{path}/pass_abc/north_def"),
            &Default::default(),
            1.0,
        ))
        .unwrap();
        assert_eq!(loaded.thickness, RELEASE_A);
    }

    #[test_log::test]
    fn test_sites_and_scenarios_match_by_name_stem_or_hash() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A), ("south_ghi", &RELEASE_B)]);
        let scenario = format!("{path}/pass_abc/north_def");

        for query in ["pass_abc/north_def", "pass/north", "abc/def"] {
            let (dem_path, release) = resolve(&format!("{path}/{query}"), None, None);
            assert_eq!(dem_path.unwrap(), format!("{path}/pass_abc"));
            assert_eq!(release.unwrap(), scenario);
        }
    }

    #[test_log::test]
    fn test_a_single_scenario_supplies_the_release() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        let (dem_path, release) = resolve(&path, None, None);
        assert_eq!(dem_path.unwrap(), format!("{path}/pass_abc"));
        assert_eq!(release.unwrap(), format!("{path}/pass_abc/north_def"));
    }

    #[test_log::test]
    fn test_several_scenarios_have_to_be_chosen_from() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A), ("south_ghi", &RELEASE_B)]);
        let settings = Settings {
            dem_path: Some(path.clone()),
            ..Default::default()
        };

        let error = resolve_paths(&settings).unwrap_err().to_string();
        assert!(
            error.contains("north_def") && error.contains("south_ghi"),
            "{error}"
        );
    }

    #[test_log::test]
    fn test_several_scenarios_are_fine_with_another_release() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A), ("south_ghi", &RELEASE_B)]);

        let (dem_path, release) = resolve(&path, Some("release.asc"), None);
        assert_eq!(dem_path.unwrap(), format!("{path}/pass_abc"));
        assert_eq!(release.unwrap(), "release.asc");

        let (dem_path, release) = resolve(&path, None, Some(0.5));
        assert_eq!(dem_path.unwrap(), format!("{path}/pass_abc"));
        assert_eq!(release, None);
    }

    #[test_log::test]
    fn test_a_site_without_scenarios_leaves_the_release_to_be_calculated() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        fs::remove_dir_all(Path::new(&path).join("pass_abc/north_def")).unwrap();

        let (dem_path, release) = resolve(&path, None, None);
        assert_eq!(dem_path.unwrap(), format!("{path}/pass_abc"));
        assert_eq!(release, None);
        assert!(block_on(load_dem(&path)).is_ok(), "the DEM still loads");
    }

    #[test_log::test]
    fn test_explicit_release_choices_win() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);

        let (_, release) = resolve(&path, Some("release.asc"), None);
        assert_eq!(release.unwrap(), "release.asc");

        let (_, release) = resolve(&path, None, Some(0.5));
        assert_eq!(release, None, "an estimated release is not overridden");

        let (_, release) = resolve(&format!("{path}/pass/north"), None, Some(0.5));
        assert!(release.is_some(), "a named scenario is a choice as well");
    }

    #[test_log::test]
    fn test_release_paths_can_point_into_a_store_on_their_own() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A), ("south_ghi", &RELEASE_B)]);

        let (_, release) = resolve("dem.asc", Some(&format!("{path}/pass/south")), None);
        assert_eq!(release.unwrap(), format!("{path}/pass_abc/south_ghi"));

        let result = resolve_paths(&Settings {
            release_areas_path: Some(path),
            ..Default::default()
        });
        assert!(matches!(result, Err(OutputError::InvalidData(_))));
    }

    #[test_log::test]
    fn test_unclear_selections_name_the_options() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        Output::new(&path)
            .unwrap()
            .add_run("ridge_jkl", "north_def", &dem(), &RELEASE_A, None, &run())
            .unwrap();

        for query in [path.clone(), format!("{path}/nope")] {
            let error = Selection::parse(&query).unwrap_err().to_string();
            assert!(
                error.contains("pass_abc") && error.contains("ridge_jkl"),
                "{error}"
            );
        }
        assert!(Selection::parse(&format!("{path}/pass/north/dem")).is_err());
    }

    #[test_log::test]
    fn test_arrays_are_not_mistaken_for_scenarios() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        assert!(Selection::parse(&format!("{path}/pass/dem")).is_err());
    }

    #[test_log::test]
    fn test_missing_stores_and_other_formats_are_rejected() {
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        assert!(block_on(load_dem(&format!("{path}.missing.zarr"))).is_err());

        let output = FilesystemStore::new(&path).unwrap();
        let mut root = Group::open(Arc::new(output), "/").unwrap();
        root.attributes_mut()
            .insert("avalanchers_format".to_string(), serde_json::json!("0.1.0"));
        root.store_metadata().unwrap();
        let result = Selection::parse(&path);
        assert!(matches!(result, Err(OutputError::Format { .. })));
    }

    #[test_log::test]
    fn test_outline_is_read_back_with_its_source() {
        use crate::store_schema::Outline;
        let (_tmp_dir, path) = store(&[("north_def", &RELEASE_A)]);
        let scenario = format!("{path}/pass_abc/north_def");
        assert_eq!(read_outline(&scenario).unwrap(), None);

        let cells = [true, false, false, true, true, false];
        Output::new(&path)
            .unwrap()
            .add_run(
                "pass_abc",
                "south_xyz",
                &dem(),
                &RELEASE_A,
                Some(Outline {
                    cells: &cells,
                    source: "polygon_3",
                }),
                &run(),
            )
            .unwrap();
        let (read, source) = read_outline(&format!("{path}/pass_abc/south_xyz"))
            .unwrap()
            .unwrap();
        assert_eq!(read, cells);
        assert_eq!(source, "polygon_3");
    }
}
