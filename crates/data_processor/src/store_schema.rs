//! Layout of the result stores, shared by the native writer in
//! [`crate::output`] and the browser writer in [`crate::zarr_writer`].
//!
//! ```text
//! <store>.zarr/                  attrs: avalanchers_format, Conventions, ...
//!   <site>_<dem hash>/           dem, x, y, spatial_ref
//!     <scenario>_<release hash>/ x, y, spatial_ref, release_area,
//!                                outline (y, x; optional, attrs: outline_hash),
//!                                peak_flow_velocity, peak_flow_thickness (run, y, x),
//!                                center_of_mass_x/y, time (run, timestep),
//!                                one 1-D array per run value (see RunRecord::columns)
//! ```
//!
//! Row 0 of every grid is the south edge and `x`/`y` are cell centers. The `run`
//! dimension can be longer than the number of runs: `number_of_runs` in the
//! scenario attributes is authoritative and is the last thing written per run.
//!
//! Runs are told apart by their settings and by `run_label`, an index into the
//! scenario attribute `run_labels` (e.g. `["", "cma_best", "predicted"]`); the
//! empty label marks a run saved without one.

use compute_core::SimInfoFlags;
use compute_core::dem::Dem;
use compute_core::settings::SimSettings;
use compute_core::utils::cell_centers_f64;
use serde_json::{Map, Value, json};

/// Version of the layout. Stores of any other version are rejected on open.
pub const FORMAT_VERSION: &str = "0.2.0";

/// Identifies the build that wrote a run: `<version>+<short commit>`, with
/// `.dirty` appended when the build sources differed from the commit.
pub const SOFTWARE_VERSION: &str =
    concat!(env!("CARGO_PKG_VERSION"), "+", env!("AVALANCHERS_GIT_HASH"));

/// Full hash of the commit the build was made from (`unknown` without git).
pub const GIT_COMMIT: &str = env!("AVALANCHERS_GIT_COMMIT");

/// Whether the build sources differed from [`GIT_COMMIT`].
pub const GIT_DIRTY: bool = const_str_eq(env!("AVALANCHERS_GIT_DIRTY"), "true");

const fn const_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Fill value of integer run columns, where 0 is usually a valid value.
pub const NO_DATA_U32: u32 = u32::MAX;

pub type Attributes = Map<String, Value>;

fn attributes(value: Value) -> Attributes {
    match value {
        Value::Object(map) => map,
        _ => unreachable!("attributes are always a JSON object"),
    }
}

/// Splits `<store>.zarr/<node>/<node>` into the store and the nodes below it,
/// or returns `None` for a path outside of a store.
pub fn split_store_path(path: &str) -> Option<(&str, Vec<&str>)> {
    let separators = ['/', '\\'];
    let mut end = 0;
    for segment in path.split(separators) {
        end += segment.len();
        if segment.to_lowercase().ends_with(".zarr") {
            let nodes = path[end..]
                .split(separators)
                .filter(|node| !node.is_empty())
                .collect();
            return Some((&path[..end], nodes));
        }
        end += 1;
    }
    None
}

/// Name of the deepest node of a store path without its hash suffix, which is
/// the name its results are saved under again.
pub fn store_path_stem(path: &str) -> Option<&str> {
    let (_, nodes) = split_store_path(path)?;
    let node = nodes.last()?;
    Some(node.rsplit_once('_').map_or(*node, |(stem, _)| stem))
}

/// Cell center eastings of the DEM grid.
pub fn x_coordinates(dem: &Dem) -> Vec<f64> {
    cell_centers_f64(dem.bounds.xmin, dem.bounds.xmax, dem.width)
}

/// Cell center northings of the DEM grid, ascending with the rows.
pub fn y_coordinates(dem: &Dem) -> Vec<f64> {
    cell_centers_f64(dem.bounds.ymin, dem.bounds.ymax, dem.height)
}

pub fn root_attributes() -> Attributes {
    attributes(json!({
        "title": "Avalanchers Simulation Output",
        "Conventions": "CF-1.8",
        "source": "avalanchers",
        "avalanchers_version": env!("CARGO_PKG_VERSION"),
        "avalanchers_software_version": SOFTWARE_VERSION,
        "avalanchers_git_commit": GIT_COMMIT,
        "avalanchers_git_dirty": GIT_DIRTY,
        "avalanchers_repo": "https://github.com/Gro2mi/avalanchers",
        "avalanchers_format": FORMAT_VERSION,
    }))
}

pub fn site_attributes(dem: &Dem) -> Attributes {
    attributes(json!({
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
    }))
}

/// `software_versions` lists the builds that wrote to the scenario and
/// `run_labels` the labels of its runs; the `software_version` and `run_label`
/// run columns index into them.
pub fn scenario_attributes(
    number_of_runs: u64,
    software_versions: &[&str],
    run_labels: &[&str],
) -> Attributes {
    attributes(json!({
        "aspect_release_degrees": null,
        "number_of_runs": number_of_runs,
        "software_versions": software_versions,
        "run_labels": run_labels,
    }))
}

pub fn x_attributes() -> Attributes {
    attributes(json!({
        "standard_name": "projection_x_coordinate",
        "long_name": "Easting of the cell center",
        "units": "m",
        "axis": "X",
    }))
}

pub fn y_attributes() -> Attributes {
    attributes(json!({
        "standard_name": "projection_y_coordinate",
        "long_name": "Northing of the cell center",
        "units": "m",
        "axis": "Y",
    }))
}

/// The DEM projection as an `EPSG:<code>` string, if it is known.
fn epsg_code(dem: &Dem) -> Option<&str> {
    let code = dem.projection.strip_prefix("EPSG:")?;
    let known = code.parse::<u32>().is_ok_and(|code| code > 0);
    known.then_some(dem.projection.as_str())
}

/// Attributes of the scalar `spatial_ref` variable, if the DEM has a known CRS.
pub fn spatial_ref_attributes(dem: &Dem) -> Option<Attributes> {
    epsg_code(dem).map(|code| {
        attributes(json!({
            "long_name": "Coordinate reference system",
            "spatial_ref": code,
        }))
    })
}

fn with_grid_mapping(dem: &Dem, mut attributes: Attributes) -> Attributes {
    if epsg_code(dem).is_some() {
        attributes.insert("grid_mapping".to_string(), json!("spatial_ref"));
    }
    attributes
}

pub fn dem_attributes(dem: &Dem) -> Attributes {
    with_grid_mapping(
        dem,
        attributes(json!({
            "standard_name": "surface_altitude",
            "long_name": "Digital Elevation Model",
            "units": "m",
            "source": dem.source,
        })),
    )
}

pub fn release_area_attributes(dem: &Dem) -> Attributes {
    with_grid_mapping(
        dem,
        attributes(json!({
            "long_name": "Release Area Thickness",
            "units": "m",
        })),
    )
}

/// The outline (region of interest) the runs of a scenario are evaluated
/// against, stored as a `uint8` 0/1 mask next to the release area.
#[derive(Clone, Copy, Debug)]
pub struct Outline<'a> {
    /// One entry per DEM cell, `true` inside the outline.
    pub cells: &'a [bool],
    /// Where the outline came from, e.g. the stem of its shapefile.
    pub source: &'a str,
}

impl Outline<'_> {
    /// The mask as stored, 1 inside and 0 outside.
    pub fn to_u8(&self) -> Vec<u8> {
        self.cells.iter().map(|&inside| inside as u8).collect()
    }

    /// FNV-1a hash of the mask in hex. It is stable across builds, so a
    /// scenario can tell whether a later run used the same outline.
    pub fn hash(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &inside in self.cells {
            hash ^= inside as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash ^= self.cells.len() as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        format!("{hash:016x}")
    }
}

pub fn outline_attributes(dem: &Dem, source: &str) -> Attributes {
    with_grid_mapping(
        dem,
        attributes(json!({
            "long_name": "Avalanche outline",
            "description": "Mapped outline the runs are evaluated against",
            "flag_values": [0, 1],
            "flag_meanings": "outside inside",
            "source": source,
        })),
    )
}

/// A value of one run, stored as one element of a 1-D array along `run`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    U32(u32),
    F32(f32),
    F64(f64),
}

pub struct RunColumn {
    pub name: String,
    pub value: Scalar,
    pub attributes: Attributes,
}

/// A grid or per-timestep series of one run.
pub struct RunField<'a> {
    pub name: &'static str,
    pub data: &'a [f32],
    pub attributes: Attributes,
}

/// Everything stored for one run.
pub struct RunRecord {
    pub peak_flow_velocity: Vec<f32>,
    pub peak_flow_thickness: Vec<f32>,
    /// Indexed by timestep; NaN where the step has no record.
    pub center_of_mass_x: Vec<f32>,
    pub center_of_mass_y: Vec<f32>,
    pub time: Vec<f32>,
    pub release_volume_m3: f32,
    pub travel_length_2d: f32,
    pub travel_length_3d: f32,
    pub travel_vertical_drop: f32,
    pub travel_angle: f32,
    pub final_timestep: u32,
    pub simulated_time: f32,
    pub number_particles: u32,
    /// [`compute_core::SimInfoFlags`] of the finished run.
    pub stop_flags: u32,
    /// Unix time in seconds at which the run was saved.
    pub created_at: f64,
    /// What the run is, e.g. `cma_best` or `predicted`; empty for none.
    pub label: String,
    pub settings: SimSettings,
}

impl RunRecord {
    /// The `(run, y, x)` grids.
    pub fn grids(&self, dem: &Dem) -> [RunField<'_>; 2] {
        [
            RunField {
                name: "peak_flow_velocity",
                data: &self.peak_flow_velocity,
                attributes: with_grid_mapping(
                    dem,
                    attributes(json!({"long_name": "Peak Flow Velocity", "units": "m/s"})),
                ),
            },
            RunField {
                name: "peak_flow_thickness",
                data: &self.peak_flow_thickness,
                attributes: with_grid_mapping(
                    dem,
                    attributes(json!({"long_name": "Peak Flow Thickness", "units": "m"})),
                ),
            },
        ]
    }

    /// The `(run, timestep)` series.
    pub fn tracks(&self) -> [RunField<'_>; 3] {
        [
            RunField {
                name: "center_of_mass_x",
                data: &self.center_of_mass_x,
                attributes: attributes(json!({"long_name": "Center of Mass X", "units": "m"})),
            },
            RunField {
                name: "center_of_mass_y",
                data: &self.center_of_mass_y,
                attributes: attributes(json!({"long_name": "Center of Mass Y", "units": "m"})),
            },
            RunField {
                name: "time",
                data: &self.time,
                attributes: attributes(json!({"long_name": "Simulated time", "units": "s"})),
            },
        ]
    }

    /// The `(run)` values: results, provenance and every field of the settings.
    /// `software_version` and `run_label` are indices into the scenario
    /// attributes `software_versions` and `run_labels`.
    pub fn columns(&self, software_version: u32, run_label: u32) -> Vec<RunColumn> {
        let column = |name: &str, value, attributes| RunColumn {
            name: name.to_string(),
            value,
            attributes,
        };
        let plain = |long_name: &str, units: &str| {
            attributes(json!({"long_name": long_name, "units": units}))
        };

        let mut columns = vec![
            column(
                "release_volume_m3",
                Scalar::F32(self.release_volume_m3),
                plain("Release volume", "m3"),
            ),
            column(
                "travel_length_2d",
                Scalar::F32(self.travel_length_2d),
                attributes(json!({
                    "long_name": "2D travel length",
                    "units": "m",
                    "description": "Horizontal length of the path of the center of mass",
                })),
            ),
            column(
                "travel_length_3d",
                Scalar::F32(self.travel_length_3d),
                attributes(json!({
                    "long_name": "3D travel length",
                    "units": "m",
                    "description": "Three-dimensional length of the path of the center of mass",
                })),
            ),
            column(
                "travel_vertical_drop",
                Scalar::F32(self.travel_vertical_drop),
                attributes(json!({
                    "long_name": "Center-of-mass vertical drop",
                    "units": "m",
                    "description": "First minus last valid center-of-mass elevation; negative for a net rise",
                })),
            ),
            column(
                "travel_angle",
                Scalar::F32(self.travel_angle),
                attributes(json!({
                    "long_name": "Travel angle",
                    "units": "degrees",
                    "description": "Angle of the path of the center of mass",
                })),
            ),
            column(
                "final_timestep",
                Scalar::U32(self.final_timestep),
                plain("Last simulated timestep", "1"),
            ),
            column(
                "simulated_time",
                Scalar::F32(self.simulated_time),
                plain("Simulated duration", "s"),
            ),
            column(
                "number_particles",
                Scalar::U32(self.number_particles),
                plain("Number of particles", "1"),
            ),
            column(
                "stop_flags",
                Scalar::U32(self.stop_flags),
                stop_flags_attributes(),
            ),
            column(
                "created_at",
                Scalar::F64(self.created_at),
                attributes(json!({
                    "long_name": "Time the run was saved",
                    "units": "seconds since 1970-01-01T00:00:00Z",
                })),
            ),
            column(
                "software_version",
                Scalar::U32(software_version),
                attributes(json!({
                    "long_name": "Index into the software_versions attribute of the scenario",
                    "units": "1",
                })),
            ),
            column(
                "run_label",
                Scalar::U32(run_label),
                attributes(json!({
                    "long_name": "Index into the run_labels attribute of the scenario",
                    "units": "1",
                })),
            ),
        ];
        columns.extend(settings_columns(&self.settings));
        columns
    }
}

fn stop_flags_attributes() -> Attributes {
    let (names, masks): (Vec<_>, Vec<_>) = SimInfoFlags::all()
        .iter_names()
        .map(|(name, flag)| (name.to_lowercase(), flag.bits()))
        .unzip();
    attributes(json!({
        "long_name": "Simulation status when the run stopped",
        "flag_masks": masks,
        "flag_meanings": names.join(" "),
    }))
}

/// Derived from the serialized settings, so a new setting is stored without
/// further wiring.
fn settings_columns(settings: &SimSettings) -> Vec<RunColumn> {
    let Ok(Value::Object(fields)) = serde_json::to_value(settings) else {
        return Vec::new();
    };
    fields
        .into_iter()
        .filter(|(name, _)| !name.starts_with('_'))
        .filter_map(|(name, value)| {
            let value = match value {
                Value::Number(number) if number.is_f64() => Scalar::F32(number.as_f64()? as f32),
                Value::Number(number) => Scalar::U32(number.as_u64()? as u32),
                // Non-finite floats serialize to null.
                Value::Null => Scalar::F32(f32::NAN),
                _ => return None,
            };
            let attributes = setting_attributes(&name);
            Some(RunColumn {
                name,
                value,
                attributes,
            })
        })
        .collect()
}

fn setting_attributes(name: &str) -> Attributes {
    let mut result = attributes(json!({"long_name": name.replace('_', " ")}));
    if let Some(units) = setting_units(name) {
        result.insert("units".to_string(), json!(units));
    }
    match name {
        "sim_model" => result.extend(flag_values(
            &[0, 1, 2, 3, 4],
            "terrain_following curvilinear mpm_dac sliding_block three_d_depth_averaged",
        )),
        "friction_model" => result.extend(flag_values(
            &[0, 1, 2, 3, 5],
            "coulomb voellmy voellmy_min_shear samos_at mu_i",
        )),
        "constitutive_model" => result.extend(flag_values(&[0, 1], "drucker_prager mu_i")),
        "flags" => result.extend(attributes(json!({
            "flag_masks": [1, 2, 4, 8, 16, 32],
            "flag_meanings": "curvature particle_interaction earth_pressure_coefficient \
                entrainment center_of_mass_biggest_blob weak_pressure",
        }))),
        _ => {}
    }
    result
}

fn flag_values(values: &[u32], meanings: &str) -> Attributes {
    attributes(json!({"flag_values": values, "flag_meanings": meanings}))
}

fn setting_units(name: &str) -> Option<&'static str> {
    Some(match name {
        "world_size_x"
        | "world_size_y"
        | "cell_size"
        | "grain_diameter"
        | "release_min_elevation"
        | "release_max_elevation"
        | "release_thickness"
        | "peak_flow_thickness_threshold"
        | "reference_surface_length" => "m",
        "density" => "kg/m3",
        "release_swe" => "kg/m2",
        "velocity_threshold" => "m/s",
        "drag_coefficient" => "m/s2",
        "internal_friction_angle"
        | "basal_friction_angle"
        | "min_slope_angle"
        | "max_slope_angle" => "degrees",
        "shear_modulus" | "hardening_modulus" | "bulk_modulus" | "compaction_pressure" => "Pa",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(columns: &[RunColumn], name: &str) -> Scalar {
        columns.iter().find(|c| c.name == name).unwrap().value
    }

    fn record(settings: SimSettings) -> RunRecord {
        RunRecord {
            peak_flow_velocity: Vec::new(),
            peak_flow_thickness: Vec::new(),
            center_of_mass_x: Vec::new(),
            center_of_mass_y: Vec::new(),
            time: Vec::new(),
            release_volume_m3: 1.0,
            travel_length_2d: 2.0,
            travel_length_3d: 2.5,
            travel_vertical_drop: 1.5,
            travel_angle: 3.0,
            final_timestep: 4,
            simulated_time: 5.0,
            number_particles: 6,
            stop_flags: 7,
            created_at: 8.0,
            label: String::new(),
            settings,
        }
    }

    #[test]
    fn test_store_paths_are_split_at_the_store() {
        assert_eq!(
            split_store_path("C:\\data\\runs.zarr\\site_a/sc_b"),
            Some(("C:\\data\\runs.zarr", vec!["site_a", "sc_b"]))
        );
        assert_eq!(split_store_path("runs.ZARR/"), Some(("runs.ZARR", vec![])));
        assert_eq!(split_store_path("data/dem.asc"), None);
        assert_eq!(split_store_path("data/zarr/dem.tif"), None);
    }

    #[test]
    fn test_store_path_stems_drop_the_hash() {
        assert_eq!(store_path_stem("runs.zarr/a-b_123"), Some("a-b"));
        assert_eq!(
            store_path_stem("runs.zarr/site_1/rough0.01_2"),
            Some("rough0.01")
        );
        assert_eq!(store_path_stem("runs.zarr"), None);
        assert_eq!(store_path_stem("dem_1.asc"), None);
    }

    #[test]
    fn test_travel_length_columns_describe_horizontal_and_3d_paths() {
        let columns = record(SimSettings::default()).columns(0, 0);
        assert_eq!(find(&columns, "travel_length_2d"), Scalar::F32(2.0));
        assert_eq!(find(&columns, "travel_length_3d"), Scalar::F32(2.5));
        assert_eq!(find(&columns, "travel_vertical_drop"), Scalar::F32(1.5));
        assert!(columns.iter().all(|column| column.name != "travel_length"));
        for name in [
            "travel_length_2d",
            "travel_length_3d",
            "travel_vertical_drop",
        ] {
            let column = columns.iter().find(|column| column.name == name).unwrap();
            assert_eq!(column.attributes["units"], "m");
        }
    }

    #[test]
    fn test_columns_cover_every_setting() {
        let settings = SimSettings {
            released_particles_per_cell: 4,
            friction_coefficient: 0.2,
            release_max_elevation: f32::INFINITY,
            ..SimSettings::default()
        };
        let columns = record(settings).columns(0, 0);

        assert_eq!(
            find(&columns, "released_particles_per_cell"),
            Scalar::U32(4)
        );
        assert_eq!(find(&columns, "friction_coefficient"), Scalar::F32(0.2));
        assert!(matches!(find(&columns, "release_max_elevation"), Scalar::F32(v) if v.is_nan()));
        assert!(columns.iter().all(|c| !c.name.starts_with('_')));

        let Value::Object(fields) = serde_json::to_value(settings).unwrap() else {
            panic!("settings serialize to an object");
        };
        for name in fields.keys().filter(|name| !name.starts_with('_')) {
            assert!(columns.iter().any(|c| &c.name == name), "missing {name}");
        }
    }

    #[test]
    fn test_coordinates_are_cell_centers() {
        let dem = Dem {
            width: 2,
            height: 3,
            bounds: compute_core::dem::Bounds {
                xmin: 2_600_000.0,
                xmax: 2_600_010.0,
                ymin: 1_200_000.0,
                ymax: 1_200_015.0,
            },
            ..Dem::default()
        };
        assert_eq!(x_coordinates(&dem), [2_600_002.5, 2_600_007.5]);
        assert_eq!(y_coordinates(&dem), [1_200_002.5, 1_200_007.5, 1_200_012.5]);
    }

    #[test]
    fn test_only_known_projections_get_a_grid_mapping() {
        let mut dem = Dem {
            projection: "EPSG:2056".to_string(),
            ..Dem::default()
        };
        assert!(spatial_ref_attributes(&dem).is_some());
        assert_eq!(dem_attributes(&dem)["grid_mapping"], "spatial_ref");

        for projection in ["unknown", "", "EPSG:0"] {
            dem.projection = projection.to_string();
            assert!(spatial_ref_attributes(&dem).is_none());
            assert!(!dem_attributes(&dem).contains_key("grid_mapping"));
        }
    }
}
