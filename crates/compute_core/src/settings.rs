use bytemuck::{Pod, Zeroable}; // Ensure bytemuck has the "derive" feature enabled in Cargo.toml
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Write};
use std::str::FromStr;
use std::{fmt, fs};

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

use crate::dem::Dem;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimFlags {
    pub mask: u32,
}

impl SimFlags {
    pub fn new(
        curvature: bool,
        particle_interaction: bool,
        earth_pressure_coefficient: bool,
        entrainment: bool,
    ) -> Self {
        let mut mask = 0u32;
        if curvature {
            mask |= 1 << 0;
        }
        if particle_interaction {
            mask |= 1 << 1;
        }
        if earth_pressure_coefficient {
            mask |= 1 << 2;
        }
        if entrainment {
            mask |= 1 << 3;
        }

        SimFlags { mask }
    }

    pub fn from_u32(value: u32) -> Self {
        SimFlags { mask: value }
    }

    pub fn is_curvature_enabled(&self) -> bool {
        (self.mask & (1 << 0)) != 0
    }
    pub fn is_particle_interaction_enabled(&self) -> bool {
        (self.mask & (1 << 1)) != 0
    }
    pub fn is_earth_pressure_coefficient_enabled(&self) -> bool {
        (self.mask & (1 << 2)) != 0
    }
    pub fn is_entrainment_enabled(&self) -> bool {
        (self.mask & (1 << 3)) != 0
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
pub struct SimSettings {
    // Integers
    pub max_steps: u32,
    pub sim_model: u32,
    pub friction_model: u32,
    pub released_particles_per_cell: u32,
    pub grid_shape_x: u32,
    pub grid_shape_y: u32,
    // Floats
    pub world_size_x: f32,
    pub world_size_y: f32,
    pub density: f32,
    pub slab_thickness_factor: f32,
    pub friction_coefficient: f32,
    pub drag_coefficient: f32,
    pub n0: f32,
    pub i0: f32,
    pub mu0: f32,
    pub mu2: f32,
    pub grain_diameter: f32,
    pub internal_friction_angle: f32,
    pub basal_friction_angle: f32,
    pub cfl: f32,
    pub cell_size: f32,
    pub min_slope_angle: f32,
    pub max_slope_angle: f32,
    pub release_min_elevation: f32,
    pub velocity_threshold: f32,
    pub roughness_threshold: f32,
    pub flags: u32,
}

impl Hash for SimSettings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Integers
        self.max_steps.hash(state);
        self.sim_model.hash(state);
        self.friction_model.hash(state);
        self.released_particles_per_cell.hash(state);
        self.grid_shape_x.hash(state);
        self.grid_shape_y.hash(state);
        // Floats
        self.world_size_x.to_bits().hash(state);
        self.world_size_y.to_bits().hash(state);
        self.density.to_bits().hash(state);
        self.slab_thickness_factor.to_bits().hash(state);
        self.friction_coefficient.to_bits().hash(state);
        self.drag_coefficient.to_bits().hash(state);
        self.n0.to_bits().hash(state);
        self.i0.to_bits().hash(state);
        self.mu0.to_bits().hash(state);
        self.mu2.to_bits().hash(state);
        self.grain_diameter.to_bits().hash(state);
        self.internal_friction_angle.to_bits().hash(state);
        self.basal_friction_angle.to_bits().hash(state);
        self.cfl.to_bits().hash(state);
        self.cell_size.to_bits().hash(state);
        self.min_slope_angle.to_bits().hash(state);
        self.max_slope_angle.to_bits().hash(state);
        self.release_min_elevation.to_bits().hash(state);
        self.velocity_threshold.to_bits().hash(state);
        self.roughness_threshold.to_bits().hash(state);
        // Flags
        self.flags.hash(state);
    }
}

impl Default for SimSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl SimSettings {
    pub fn calculate_hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }

    pub fn new() -> Self {
        Self {
            max_steps: 6000,
            sim_model: SimModel::ParticleInteraction.as_int(),
            friction_model: FrictionModel::Voellmy.as_int(),
            released_particles_per_cell: 8,
            grid_shape_x: 1,
            grid_shape_y: 1,
            world_size_x: 1.0,
            world_size_y: 1.0,
            density: 200.0,
            slab_thickness_factor: 1.0,

            friction_coefficient: 0.2,
            drag_coefficient: 2000.0,
            n0: 70.0,
            i0: 0.29,
            mu0: 0.38,
            mu2: 0.65,
            grain_diameter: 0.002,
            internal_friction_angle: 40.0,
            basal_friction_angle: 25.0,

            cfl: 0.5,
            cell_size: 1.0,
            velocity_threshold: 0.1,
            roughness_threshold: 0.01,
            flags: SimFlags::new(true, true, true, true).mask,

            min_slope_angle: 28.0,
            max_slope_angle: 60.0,
            release_min_elevation: 1500.0,
        }
    }

    pub fn set_dem(&mut self, dem: &Dem) {
        self.cell_size = dem.cell_size;
        self.grid_shape_x = dem.width as u32;
        self.grid_shape_y = dem.height as u32;
        self.world_size_x = dem.width as f32 * dem.cell_size;
        self.world_size_y = dem.height as f32 * dem.cell_size;
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    pub fn from_settings(patch: &Settings, dem: &Dem) -> Self {
        let mut settings = SimSettings::new();
        if let Some(val) = patch.max_steps {
            settings.max_steps = val;
        }
        if let Some(val) = patch.sim_model {
            settings.sim_model = val.as_int();
        }
        if let Some(val) = patch.friction_model {
            settings.friction_model = val.as_int();
        }
        if let Some(val) = patch.released_particles_per_cell {
            settings.released_particles_per_cell = val;
        }
        if let Some(val) = patch.density {
            settings.density = val;
        }
        if let Some(val) = patch.slab_thickness_factor {
            settings.slab_thickness_factor = val;
        }
        if let Some(val) = patch.friction_coefficient {
            settings.friction_coefficient = val;
        }
        if let Some(val) = patch.drag_coefficient {
            settings.drag_coefficient = val;
        }
        if let Some(val) = patch.n0 {
            settings.n0 = val;
        }
        if let Some(val) = patch.i0 {
            settings.i0 = val;
        }
        if let Some(val) = patch.mu0 {
            settings.mu0 = val;
        }
        if let Some(val) = patch.mu2 {
            settings.mu2 = val;
        }
        if let Some(val) = patch.grain_diameter {
            settings.grain_diameter = val;
        }
        if let Some(val) = patch.internal_friction_angle {
            settings.internal_friction_angle = val;
        }
        if let Some(val) = patch.basal_friction_angle {
            settings.basal_friction_angle = val;
        }
        if let Some(val) = patch.cfl {
            settings.cfl = val;
        }
        if let Some(val) = patch.min_slope_angle {
            settings.min_slope_angle = val;
        }
        if let Some(val) = patch.max_slope_angle {
            settings.max_slope_angle = val;
        }
        if let Some(val) = patch.release_min_elevation {
            settings.release_min_elevation = val;
        }
        if let Some(val) = patch.velocity_threshold {
            settings.velocity_threshold = val;
        }
        if let Some(val) = patch.roughness_threshold {
            settings.roughness_threshold = val;
        }
        if let Some(val) = patch.enable_curvature {
            if val {
                settings.flags |= 1 << 0;
            } else {
                settings.flags &= !(1 << 0);
            }
        }
        if let Some(val) = patch.enable_particle_interaction {
            if val {
                settings.flags |= 1 << 1;
            } else {
                settings.flags &= !(1 << 1);
            }
        }
        if let Some(val) = patch.enable_earth_pressure_coefficient {
            if val {
                settings.flags |= 1 << 2;
            } else {
                settings.flags &= !(1 << 2);
            }
        }
        if let Some(val) = patch.enable_entrainment {
            if val {
                settings.flags |= 1 << 3;
            } else {
                settings.flags &= !(1 << 3);
            }
        }
        settings.set_dem(dem);
        settings
    }

    pub fn to_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrictionModel {
    Coulomb,
    Voellmy,
    VoellmyMinShear,
    SamosAT,
}

impl FrictionModel {
    pub fn from_int(value: u32) -> Option<Self> {
        match value {
            0 => Some(FrictionModel::Coulomb),
            1 => Some(FrictionModel::Voellmy),
            2 => Some(FrictionModel::VoellmyMinShear),
            3 => Some(FrictionModel::SamosAT),
            _ => None,
        }
    }

    pub fn as_int(&self) -> u32 {
        match self {
            FrictionModel::Coulomb => 0,
            FrictionModel::Voellmy => 1,
            FrictionModel::VoellmyMinShear => 2,
            FrictionModel::SamosAT => 3,
        }
    }
}

impl FromStr for FrictionModel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "coulomb" => Ok(Self::Coulomb),
            "voellmy" => Ok(Self::Voellmy),
            "voellmyminshear" | "voellmy-min-shear" => Ok(Self::VoellmyMinShear),
            "samosat" | "samos-at" => Ok(Self::SamosAT),
            _ => Err(format!("unknown friction model: {value}")),
        }
    }
}

impl fmt::Display for FrictionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Coulomb => "coulomb",
            Self::Voellmy => "voellmy",
            Self::VoellmyMinShear => "voellmy-min-shear",
            Self::SamosAT => "samos-at",
        })
    }
}

impl Serialize for FrictionModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for FrictionModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value {
            Int(u32),
            String(String),
        }

        match Value::deserialize(deserializer)? {
            Value::Int(value) => Self::from_int(value)
                .ok_or_else(|| D::Error::custom(format!("invalid friction model: {value}"))),

            Value::String(value) => value.parse::<Self>().map_err(D::Error::custom),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimModel {
    Block,
    ParticleInteraction,
    MPM,
}

impl SimModel {
    pub fn from_int(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Block),
            1 => Some(Self::ParticleInteraction),
            2 => Some(Self::MPM),
            _ => None,
        }
    }

    pub fn as_int(&self) -> u32 {
        match self {
            Self::Block => 0,
            Self::ParticleInteraction => 1,
            Self::MPM => 2,
        }
    }
}

impl FromStr for SimModel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "block" => Ok(Self::Block),
            "particle-interaction" | "particleinteraction" => Ok(Self::ParticleInteraction),
            "mpm" => Ok(Self::MPM),
            _ => Err(format!("unknown simulation model: {value}")),
        }
    }
}

impl fmt::Display for SimModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Block => "block",
            Self::ParticleInteraction => "particle-interaction",
            Self::MPM => "mpm",
        })
    }
}

impl Serialize for SimModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SimModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value {
            Int(u32),
            String(String),
        }

        match Value::deserialize(deserializer)? {
            Value::Int(value) => Self::from_int(value)
                .ok_or_else(|| D::Error::custom(format!("invalid simulation model: {value}"))),

            Value::String(value) => value.parse::<Self>().map_err(D::Error::custom),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub outlines_path: Option<String>,
    pub outlines_padding: Option<f32>,
    pub dem_path: Option<String>,
    pub release_areas_path: Option<String>,
    pub output_path: Option<String>,
    pub max_steps: Option<u32>,
    pub sim_model: Option<SimModel>,
    pub batch_compute_steps: Option<u32>,
    pub friction_model: Option<FrictionModel>,
    pub released_particles_per_cell: Option<u32>,
    pub density: Option<f32>,
    pub slab_thickness_factor: Option<f32>,
    pub friction_coefficient: Option<f32>,
    pub drag_coefficient: Option<f32>,
    pub n0: Option<f32>,
    pub i0: Option<f32>,
    pub mu0: Option<f32>,
    pub mu2: Option<f32>,
    pub grain_diameter: Option<f32>,
    pub internal_friction_angle: Option<f32>,
    pub basal_friction_angle: Option<f32>,

    pub cfl: Option<f32>,
    pub min_slope_angle: Option<f32>,
    pub max_slope_angle: Option<f32>,
    pub release_min_elevation: Option<f32>,
    pub velocity_threshold: Option<f32>,
    pub roughness_threshold: Option<f32>,

    pub enable_curvature: Option<bool>,
    pub enable_particle_interaction: Option<bool>,
    pub enable_earth_pressure_coefficient: Option<bool>,
    pub enable_entrainment: Option<bool>,
}

impl Settings {
    pub fn loads(json_str: &str) -> io::Result<Self> {
        let settings: Settings = serde_json::from_str(json_str).map_err(|e| {
            error!("Failed to deserialize settings: {e}");
            io::Error::new(io::ErrorKind::InvalidData, e)
        })?;
        Ok(settings)
    }

    pub fn dumps(&self) -> io::Result<String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            error!("Failed to serialize settings: {e}");
            io::Error::new(io::ErrorKind::InvalidData, e)
        })?;
        Ok(json)
    }

    pub fn from_json(path: &str) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let settings = Self::loads(&data)?;
        Ok(settings)
    }

    pub fn get_sim_settings(&self) -> SimSettings {
        SimSettings::from_settings(self, &Dem::default())
    }
}

impl fmt::Display for Settings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string_pretty(self) {
            Ok(json) => write!(f, "{json}"),
            Err(e) => write!(f, "Failed to serialize settings: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::NamedTempFile;

    fn create_test_dem() -> Dem {
        let mut dem = Dem::default();
        dem.cell_size = 1.0;
        dem.width = 10;
        dem.height = 20;
        dem
    }

    #[test_log::test]
    fn test_simsettings_new_defaults() {
        let settings = SimSettings::new();
        assert_eq!(settings.max_steps, 6000);
        assert_eq!(settings.sim_model, SimModel::ParticleInteraction.as_int());
        assert_eq!(settings.friction_model, FrictionModel::Voellmy.as_int());
        assert_eq!(settings.released_particles_per_cell, 8);
        assert_eq!(settings.grid_shape_x, 1);
        assert_eq!(settings.grid_shape_y, 1);
        assert_eq!(settings.world_size_x, 1.0);
        assert_eq!(settings.world_size_y, 1.0);
        assert_eq!(settings.density, 200.0);
        assert_eq!(settings.slab_thickness_factor, 1.0);
        assert_eq!(settings.friction_coefficient, 0.2);
        assert_eq!(settings.drag_coefficient, 2000.0);
        assert_eq!(settings.cfl, 0.5);
        assert_eq!(settings.cell_size, 1.0);
        assert_eq!(settings.min_slope_angle, 28.0);
        assert_eq!(settings.max_slope_angle, 60.0);
        assert_eq!(settings.release_min_elevation, 1500.0);
        assert_eq!(settings.velocity_threshold, 0.1);
        assert_eq!(settings.roughness_threshold, 0.01);
    }

    #[test_log::test]
    fn test_simsettings_set_dem() {
        let dem = create_test_dem();

        let mut settings = SimSettings::new();
        settings.set_dem(&dem);
        assert_eq!(settings.cell_size, dem.cell_size);
        assert_eq!(settings.grid_shape_x, dem.width as u32);
        assert_eq!(settings.grid_shape_y, dem.height as u32);
        assert_eq!(settings.world_size_x, dem.cell_size * dem.width as f32);
        assert_eq!(settings.world_size_y, dem.cell_size * dem.height as f32);
    }

    #[test_log::test]
    fn test_simsettings_as_bytes_length() {
        let settings = SimSettings::new();
        let bytes = settings.as_bytes();
        assert_eq!(bytes.len(), std::mem::size_of::<SimSettings>());
    }

    #[test_log::test]
    fn test_simsettings_from_json_patch() {
        let mut patch = Settings {
            outlines_path: Some("path/to/outlines".to_string()),
            outlines_padding: Some(10.0),
            max_steps: Some(42),
            sim_model: Some(SimModel::ParticleInteraction),
            friction_model: Some(FrictionModel::VoellmyMinShear),
            released_particles_per_cell: Some(3),
            density: Some(123.4),
            slab_thickness_factor: Some(5.6),
            friction_coefficient: Some(0.2),
            drag_coefficient: Some(999.0),
            n0: Some(1.0),
            i0: Some(2.0),
            mu0: Some(0.1),
            mu2: Some(0.3),
            grain_diameter: Some(0.5),
            internal_friction_angle: Some(30.0),
            basal_friction_angle: Some(45.0),
            batch_compute_steps: Some(100),
            cfl: Some(0.9),
            min_slope_angle: Some(10.0),
            max_slope_angle: Some(20.0),
            release_min_elevation: Some(100.0),
            velocity_threshold: Some(0.001),
            roughness_threshold: Some(0.002),
            dem_path: Some(String::from("dem.png")),
            release_areas_path: Some(String::from("release_area.png")),
            output_path: Some(String::from("output")),
            enable_curvature: Some(false),
            enable_particle_interaction: Some(false),
            enable_earth_pressure_coefficient: Some(false),
            enable_entrainment: Some(false),
        };
        let dem = create_test_dem();
        let mut sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(sim_settings.max_steps, 42);
        assert_eq!(sim_settings.sim_model, 1);
        assert_eq!(sim_settings.friction_model, 2);
        assert_eq!(sim_settings.released_particles_per_cell, 3);
        assert_eq!(sim_settings.density, 123.4);
        assert_eq!(sim_settings.slab_thickness_factor, 5.6);
        assert_eq!(sim_settings.friction_coefficient, 0.2);
        assert_eq!(sim_settings.drag_coefficient, 999.0);
        assert_eq!(sim_settings.cfl, 0.9);
        assert_eq!(sim_settings.min_slope_angle, 10.0);
        assert_eq!(sim_settings.max_slope_angle, 20.0);
        assert_eq!(sim_settings.release_min_elevation, 100.0);
        assert_eq!(sim_settings.velocity_threshold, 0.001);
        assert_eq!(sim_settings.roughness_threshold, 0.002);
        assert_eq!(sim_settings.grid_shape_x, dem.width as u32);
        assert_eq!(sim_settings.grid_shape_y, dem.height as u32);
        assert_eq!(sim_settings.cell_size, dem.cell_size);
        assert_eq!(sim_settings.world_size_x, dem.cell_size * dem.width as f32);
        assert_eq!(sim_settings.world_size_y, dem.cell_size * dem.height as f32);
        assert_eq!(sim_settings.n0, 1.0);
        assert_eq!(sim_settings.i0, 2.0);
        assert_eq!(sim_settings.mu0, 0.1);
        assert_eq!(sim_settings.mu2, 0.3);
        assert_eq!(sim_settings.grain_diameter, 0.5);
        assert_eq!(sim_settings.internal_friction_angle, 30.0);
        assert_eq!(sim_settings.basal_friction_angle, 45.0);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(false, false, false, false).mask
        );

        // Test enabling flags one by one
        patch.enable_curvature = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, false, false, false).mask
        );

        patch.enable_particle_interaction = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, false, false).mask
        );

        patch.enable_earth_pressure_coefficient = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, true, false).mask
        );

        patch.enable_entrainment = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, true, true).mask
        );
    }

    #[test_log::test]
    fn test_simsettings_to_json_and_read_back() {
        let settings = SimSettings::new();
        let mut file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        settings.to_json(path).unwrap();

        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let deserialized: SimSettings = serde_json::from_str(&contents).unwrap();
        assert_eq!(settings.max_steps, deserialized.max_steps);
        assert_eq!(settings.sim_model, deserialized.sim_model);
        assert_eq!(settings.friction_model, deserialized.friction_model);
        assert_eq!(
            settings.released_particles_per_cell,
            deserialized.released_particles_per_cell
        );
        assert_eq!(settings.grid_shape_x, deserialized.grid_shape_x);
        assert_eq!(settings.grid_shape_y, deserialized.grid_shape_y);
        assert_eq!(settings.world_size_x, deserialized.world_size_x);
        assert_eq!(settings.world_size_y, deserialized.world_size_y);
        assert_eq!(settings.density, deserialized.density);
        assert_eq!(
            settings.slab_thickness_factor,
            deserialized.slab_thickness_factor
        );
        assert_eq!(
            settings.friction_coefficient,
            deserialized.friction_coefficient
        );
        assert_eq!(settings.drag_coefficient, deserialized.drag_coefficient);
        assert_eq!(settings.cfl, deserialized.cfl);
        assert_eq!(settings.cell_size, deserialized.cell_size);
        assert_eq!(settings.min_slope_angle, deserialized.min_slope_angle);
        assert_eq!(settings.max_slope_angle, deserialized.max_slope_angle);
        assert_eq!(
            settings.release_min_elevation,
            deserialized.release_min_elevation
        );
        assert_eq!(settings.velocity_threshold, deserialized.velocity_threshold);
        assert_eq!(
            settings.roughness_threshold,
            deserialized.roughness_threshold
        );
        assert_eq!(settings.flags, deserialized.flags);
    }

    #[test_log::test]
    fn test_friction_model_from_int_and_as_int() {
        for (i, variant) in [
            FrictionModel::Coulomb,
            FrictionModel::Voellmy,
            FrictionModel::VoellmyMinShear,
            FrictionModel::SamosAT,
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(FrictionModel::from_int(i as u32), Some(*variant));
            assert_eq!(variant.as_int(), i as u32);
        }
        assert_eq!(FrictionModel::from_int(99), None);
    }

    #[test_log::test]
    fn test_settings_display() {
        let settings = Settings {
            dem_path: Some(String::from("foo.tif")),
            ..Default::default()
        };
        let display = format!("{}", settings);
        assert!(display.contains("foo.tif"));
        assert!(display.contains("dem_path"));
    }

    #[test]
    fn sim_model_int_roundtrip() {
        let models = [
            SimModel::Block,
            SimModel::ParticleInteraction,
            SimModel::MPM,
        ];

        for model in models {
            let value = model.as_int();
            let converted = SimModel::from_int(value);

            assert_eq!(converted, Some(model));
        }
    }

    #[test]
    fn sim_model_from_int() {
        assert_eq!(SimModel::from_int(0), Some(SimModel::Block));
        assert_eq!(SimModel::from_int(1), Some(SimModel::ParticleInteraction));
        assert_eq!(SimModel::from_int(2), Some(SimModel::MPM));

        assert_eq!(SimModel::from_int(3), None);
        assert_eq!(SimModel::from_int(u32::MAX), None);
    }

    #[test]
    fn sim_model_string_roundtrip() {
        let models = [
            (SimModel::Block, "block"),
            (SimModel::ParticleInteraction, "particle-interaction"),
            (SimModel::MPM, "mpm"),
        ];

        for (model, string) in models {
            assert_eq!(model.to_string(), string);
            assert_eq!(SimModel::from_str(string), Ok(model));
        }
    }

    #[test]
    fn sim_model_string_aliases() {
        assert_eq!(
            SimModel::from_str("particleinteraction"),
            Ok(SimModel::ParticleInteraction)
        );

        assert_eq!(SimModel::from_str("BLOCK"), Ok(SimModel::Block));
        assert_eq!(SimModel::from_str("MPM"), Ok(SimModel::MPM));
    }

    #[test]
    fn sim_model_invalid_string() {
        assert!(SimModel::from_str("invalid").is_err());
        assert!(SimModel::from_str("").is_err());
    }

    #[test]
    fn friction_model_int_roundtrip() {
        let models = [
            FrictionModel::Coulomb,
            FrictionModel::Voellmy,
            FrictionModel::VoellmyMinShear,
            FrictionModel::SamosAT,
        ];

        for model in models {
            let value = model.as_int();
            let converted = FrictionModel::from_int(value);

            assert_eq!(converted, Some(model));
        }
    }

    #[test]
    fn friction_model_from_int() {
        assert_eq!(FrictionModel::from_int(0), Some(FrictionModel::Coulomb));
        assert_eq!(FrictionModel::from_int(1), Some(FrictionModel::Voellmy));
        assert_eq!(
            FrictionModel::from_int(2),
            Some(FrictionModel::VoellmyMinShear)
        );
        assert_eq!(FrictionModel::from_int(3), Some(FrictionModel::SamosAT));

        assert_eq!(FrictionModel::from_int(4), None);
        assert_eq!(FrictionModel::from_int(u32::MAX), None);
    }

    #[test]
    fn friction_model_string_roundtrip() {
        let models = [
            (FrictionModel::Coulomb, "coulomb"),
            (FrictionModel::Voellmy, "voellmy"),
            (FrictionModel::VoellmyMinShear, "voellmy-min-shear"),
            (FrictionModel::SamosAT, "samos-at"),
        ];

        for (model, string) in models {
            assert_eq!(model.to_string(), string);
            assert_eq!(FrictionModel::from_str(string), Ok(model));
        }
    }

    #[test]
    fn friction_model_string_aliases() {
        assert_eq!(
            FrictionModel::from_str("voellmyminshear"),
            Ok(FrictionModel::VoellmyMinShear)
        );

        assert_eq!(
            FrictionModel::from_str("samosat"),
            Ok(FrictionModel::SamosAT)
        );

        assert_eq!(
            FrictionModel::from_str("VOELLMY"),
            Ok(FrictionModel::Voellmy)
        );
    }

    #[test]
    fn friction_model_invalid_string() {
        assert!(FrictionModel::from_str("invalid").is_err());
        assert!(FrictionModel::from_str("").is_err());
    }

    #[test]
    fn test_sim_settings_hash_deterministic() {
        let settings1 = SimSettings::new();
        let settings2 = SimSettings::new();
        assert_eq!(settings1.calculate_hash(), settings2.calculate_hash());
    }

    #[test]
    fn test_sim_settings_hash_changes_on_field_update() {
        let settings1 = SimSettings::new();
        let mut settings2 = settings1;
        settings2.max_steps = 9999;
        assert_ne!(settings1.calculate_hash(), settings2.calculate_hash());

        let mut settings3 = settings1;
        settings3.friction_coefficient = 0.55;
        assert_ne!(settings1.calculate_hash(), settings3.calculate_hash());
    }
}
