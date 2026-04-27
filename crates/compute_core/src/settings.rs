use bytemuck::{Pod, Zeroable}; // Ensure bytemuck has the "derive" feature enabled in Cargo.toml
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};

use crate::dem::Dem;
use std::fmt;

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
    pub slab_thickness: f32,
    pub friction_coefficient: f32,
    pub drag_coefficient: f32,
    pub cfl: f32,
    pub cell_size: f32,
    pub min_slope_angle: f32,
    pub max_slope_angle: f32,
    pub release_min_elevation: f32,
    pub velocity_threshold: f32,
    pub roughness_threshold: f32,
}

impl Default for SimSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl SimSettings {
    pub fn new() -> Self {
        Self {
            max_steps: 3000,
            sim_model: 0,
            friction_model: FrictionModel::VoellmyMinShear.as_int(),
            released_particles_per_cell: 8,
            grid_shape_x: 1,
            grid_shape_y: 1,
            world_size_x: 1.0,
            world_size_y: 1.0,
            density: 200.0,
            slab_thickness: 1.0,
            friction_coefficient: 0.155,
            drag_coefficient: 4000.0,
            cfl: 0.5,
            cell_size: 1.0,
            min_slope_angle: 35.0,
            max_slope_angle: 45.0,
            release_min_elevation: 1500.0,
            velocity_threshold: 1e-6,
            roughness_threshold: 0.01,
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
            settings.sim_model = val;
        }
        if let Some(val) = patch.friction_model {
            settings.friction_model = val;
        }
        if let Some(val) = patch.released_particles_per_cell {
            settings.released_particles_per_cell = val;
        }
        if let Some(val) = patch.density {
            settings.density = val;
        }
        if let Some(val) = patch.slab_thickness {
            settings.slab_thickness = val;
        }
        if let Some(val) = patch.friction_coefficient {
            settings.friction_coefficient = val;
        }
        if let Some(val) = patch.drag_coefficient {
            settings.drag_coefficient = val;
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
        if let Some(val) = patch.min_elevation {
            settings.release_min_elevation = val;
        }
        if let Some(val) = patch.velocity_threshold {
            settings.velocity_threshold = val;
        }
        if let Some(val) = patch.roughness_threshold {
            settings.roughness_threshold = val;
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

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Settings {
    pub dem_path: Option<String>,
    pub release_areas_path: Option<String>,
    pub max_steps: Option<u32>,
    pub sim_model: Option<u32>,
    pub friction_model: Option<u32>,
    pub released_particles_per_cell: Option<u32>,
    pub density: Option<f32>,
    pub slab_thickness: Option<f32>,
    pub friction_coefficient: Option<f32>,
    pub drag_coefficient: Option<f32>,
    pub cfl: Option<f32>,
    pub min_slope_angle: Option<f32>,
    pub max_slope_angle: Option<f32>,
    pub min_elevation: Option<f32>,
    pub velocity_threshold: Option<f32>,
    pub roughness_threshold: Option<f32>,
}

impl Settings {
    pub fn loads(json_str: &str) -> io::Result<Self> {
        let settings: Settings = serde_json::from_str(json_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(settings)
    }

    pub fn dumps(&self) -> io::Result<String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(json)
    }

    pub fn from_json(path: &str) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let settings = Self::loads(&data)?;
        Ok(settings)
    }

    pub fn to_json(&self, path: &str) -> io::Result<()> {
        let json = self.dumps()?;
        let mut file = fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub async fn create_from_json(file_path: &str) -> (SimSettings, Dem) {
        let settings =
            Settings::from_json(file_path).expect("Failed to load settings from JSON file");
        settings.create().await
    }
    pub async fn create_from_path(file_path: &str) -> (SimSettings, Dem) {
        let settings = Settings {
            dem_path: Some(file_path.to_string()),
            ..Default::default()
        };
        settings.create().await
    }
    pub async fn create(&self) -> (SimSettings, Dem) {
        let dem = match &self.dem_path {
            Some(path) => Dem::new(path.as_ref())
                .await
                .expect("Failed to load DEM from path"),
            None => Dem::default(),
        };
        let sim_settings = SimSettings::from_settings(self, &dem);
        (sim_settings, dem)
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
        assert_eq!(settings.max_steps, 3000);
        assert_eq!(settings.sim_model, 0);
        assert_eq!(
            settings.friction_model,
            FrictionModel::VoellmyMinShear.as_int()
        );
        assert_eq!(settings.released_particles_per_cell, 8);
        assert_eq!(settings.grid_shape_x, 1);
        assert_eq!(settings.grid_shape_y, 1);
        assert_eq!(settings.world_size_x, 1.0);
        assert_eq!(settings.world_size_y, 1.0);
        assert_eq!(settings.density, 200.0);
        assert_eq!(settings.slab_thickness, 1.0);
        assert_eq!(settings.friction_coefficient, 0.155);
        assert_eq!(settings.drag_coefficient, 4000.0);
        assert_eq!(settings.cfl, 0.5);
        assert_eq!(settings.cell_size, 1.0);
        assert_eq!(settings.min_slope_angle, 35.0);
        assert_eq!(settings.max_slope_angle, 45.0);
        assert_eq!(settings.release_min_elevation, 1500.0);
        assert_eq!(settings.velocity_threshold, 1e-6);
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
        let patch = Settings {
            max_steps: Some(42),
            sim_model: Some(1),
            friction_model: Some(2),
            released_particles_per_cell: Some(3),
            density: Some(123.4),
            slab_thickness: Some(5.6),
            friction_coefficient: Some(0.2),
            drag_coefficient: Some(999.0),
            cfl: Some(0.9),
            min_slope_angle: Some(10.0),
            max_slope_angle: Some(20.0),
            min_elevation: Some(100.0),
            velocity_threshold: Some(0.001),
            roughness_threshold: Some(0.002),
            dem_path: Some(String::from("dem.png")),
            release_areas_path: Some(String::from("release_area.png")),
        };
        let dem = create_test_dem();
        let settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(settings.max_steps, 42);
        assert_eq!(settings.sim_model, 1);
        assert_eq!(settings.friction_model, 2);
        assert_eq!(settings.released_particles_per_cell, 3);
        assert_eq!(settings.density, 123.4);
        assert_eq!(settings.slab_thickness, 5.6);
        assert_eq!(settings.friction_coefficient, 0.2);
        assert_eq!(settings.drag_coefficient, 999.0);
        assert_eq!(settings.cfl, 0.9);
        assert_eq!(settings.min_slope_angle, 10.0);
        assert_eq!(settings.max_slope_angle, 20.0);
        assert_eq!(settings.release_min_elevation, 100.0);
        assert_eq!(settings.velocity_threshold, 0.001);
        assert_eq!(settings.roughness_threshold, 0.002);
        assert_eq!(settings.grid_shape_x, dem.width as u32);
        assert_eq!(settings.grid_shape_y, dem.height as u32);
        assert_eq!(settings.cell_size, dem.cell_size);
        assert_eq!(settings.world_size_x, dem.cell_size * dem.width as f32);
        assert_eq!(settings.world_size_y, dem.cell_size * dem.height as f32);
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
    fn test_settings_to_json_and_from_json() {
        let settings = Settings {
            dem_path: Some(String::from("dem.png")),
            release_areas_path: Some(String::from("release_areas.png")),
            max_steps: Some(100),
            sim_model: Some(1),
            friction_model: Some(2),
            released_particles_per_cell: Some(3),
            density: Some(4.0),
            slab_thickness: Some(5.0),
            friction_coefficient: Some(6.0),
            drag_coefficient: Some(7.0),
            cfl: Some(8.0),
            min_slope_angle: Some(9.0),
            max_slope_angle: Some(10.0),
            min_elevation: Some(11.0),
            velocity_threshold: Some(12.0),
            roughness_threshold: Some(13.0),
        };
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        settings.to_json(path).unwrap();

        let loaded = Settings::from_json(path).unwrap();
        assert_eq!(loaded.dem_path, Some(String::from("dem.png")));
        assert_eq!(
            loaded.release_areas_path,
            Some(String::from("release_areas.png"))
        );
        assert_eq!(loaded.max_steps, Some(100));
        assert_eq!(loaded.sim_model, Some(1));
        assert_eq!(loaded.friction_model, Some(2));
        assert_eq!(loaded.released_particles_per_cell, Some(3));
        assert_eq!(loaded.density, Some(4.0));
        assert_eq!(loaded.slab_thickness, Some(5.0));
        assert_eq!(loaded.friction_coefficient, Some(6.0));
        assert_eq!(loaded.drag_coefficient, Some(7.0));
        assert_eq!(loaded.cfl, Some(8.0));
        assert_eq!(loaded.min_slope_angle, Some(9.0));
        assert_eq!(loaded.max_slope_angle, Some(10.0));
        assert_eq!(loaded.min_elevation, Some(11.0));
        assert_eq!(loaded.velocity_threshold, Some(12.0));
        assert_eq!(loaded.roughness_threshold, Some(13.0));
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
}
