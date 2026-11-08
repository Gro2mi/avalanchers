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
use crate::{DEFAULT_RELEASE_THICKNESS, RELEASE_THICKNESS_THRESHOLD};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimFlags {
    pub mask: u32,
}

impl SimFlags {
    pub const CURVATURE: u32 = 1 << 0;
    pub const PARTICLE_INTERACTION: u32 = 1 << 1;
    pub const EARTH_PRESSURE_COEFFICIENT: u32 = 1 << 2;
    pub const ENTRAINMENT: u32 = 1 << 3;
    pub const CENTER_OF_MASS_BIGGEST_BLOB: u32 = 1 << 4;
    pub const WEAK_PRESSURE: u32 = 1 << 5;

    pub fn new(
        curvature: bool,
        particle_interaction: bool,
        earth_pressure_coefficient: bool,
        entrainment: bool,
        center_of_mass_biggest_blob: bool,
        weak_pressure: bool,
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
        if center_of_mass_biggest_blob {
            mask |= 1 << 4;
        }
        if weak_pressure {
            mask |= 1 << 5;
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
    pub fn is_center_of_mass_biggest_blob_enabled(&self) -> bool {
        (self.mask & (1 << 4)) != 0
    }
    pub fn is_weak_pressure_enabled(&self) -> bool {
        (self.mask & (1 << 5)) != 0
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
    /// Release thickness (m) replacing that of the release source, NaN if
    /// unset; read it with [`Self::release_thickness_override`].
    #[serde(deserialize_with = "deserialize_unset_as_nan")]
    pub release_thickness: f32,
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
    pub release_max_elevation: f32,
    pub peak_flow_thickness_threshold: f32,
    // MPMDAC constitutive model
    pub constitutive_model: u32,
    pub shear_modulus: f32,
    pub hardening_modulus: f32,
    // MPMDAC compressibility; bulk_modulus 0 = incompressible
    pub bulk_modulus: f32,
    pub compaction_pressure: f32,
    /// Grid flow models: stop once the total momentum falls below this fraction
    /// of its peak; 0 disables the criterion.
    pub momentum_stop_fraction: f32,
    /// The no-new-cells stop window spans round(travel / cfl) steps, at least
    /// 10: the steps the fastest material needs to travel this many cells.
    pub no_new_cells_travel: f32,
    /// Curvilinear, terrain-following and 3D depth averaged: Gaussian width (m)
    /// of the reference surface that defines gully wall inclinations for the
    /// confinement friction; 0 disables it.
    pub reference_surface_length: f32,
    /// Snow water equivalent (kg/m²) of every release cell, set instead of
    /// `release_thickness`, NaN if unset; read it with
    /// [`Self::release_swe_override`].
    #[serde(
        default = "unset_release_swe",
        deserialize_with = "deserialize_unset_as_nan"
    )]
    pub release_swe: f32,
}

/// Hashes only the fields the selected simulation, friction and constitutive
/// model read, so a parameter added for one model, or changed while another
/// model is selected, leaves the hash unchanged. When a model starts reading a
/// field, add it here under that model.
impl Hash for SimSettings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash_f32 = |value: f32, state: &mut H| value.to_bits().hash(state);
        let sim_model = SimModel::from_int(self.sim_model);
        let grid_flow_model = matches!(
            sim_model,
            Some(
                SimModel::TerrainFollowing | SimModel::Curvilinear | SimModel::ThreeDDepthAveraged
            )
        );

        // Shared by all models: run length, grid, release and evaluation
        self.max_steps.hash(state);
        self.sim_model.hash(state);
        self.friction_model.hash(state);
        self.released_particles_per_cell.hash(state);
        self.grid_shape_x.hash(state);
        self.grid_shape_y.hash(state);
        hash_f32(self.world_size_x, state);
        hash_f32(self.world_size_y, state);
        hash_f32(self.cell_size, state);
        hash_f32(self.density, state);
        hash_f32(self.release_thickness, state);
        hash_f32(self.release_swe, state);
        hash_f32(self.min_slope_angle, state);
        hash_f32(self.max_slope_angle, state);
        hash_f32(self.release_min_elevation, state);
        hash_f32(self.release_max_elevation, state);
        hash_f32(self.roughness_threshold, state);
        hash_f32(self.cfl, state);
        hash_f32(self.velocity_threshold, state);
        hash_f32(self.peak_flow_thickness_threshold, state);
        hash_f32(self.no_new_cells_travel, state);

        // Basal friction model
        match FrictionModel::from_int(self.friction_model) {
            Some(FrictionModel::Coulomb | FrictionModel::SamosAT) => {
                hash_f32(self.friction_coefficient, state);
            }
            Some(FrictionModel::Voellmy | FrictionModel::VoellmyMinShear) => {
                hash_f32(self.friction_coefficient, state);
                hash_f32(self.drag_coefficient, state);
            }
            Some(FrictionModel::MuI) | None => {}
        }
        // mu(I) rheology, as friction model or MPMDAC constitutive law
        if self.uses_mu_i() {
            hash_f32(self.i0, state);
            hash_f32(self.mu0, state);
            hash_f32(self.mu2, state);
            hash_f32(self.grain_diameter, state);
        }

        let mut flags_mask = SimFlags::CENTER_OF_MASS_BIGGEST_BLOB;
        match sim_model {
            Some(SimModel::MpmDaC) => {
                flags_mask |= SimFlags::CURVATURE;
                self.constitutive_model.hash(state);
                hash_f32(self.n0, state);
                hash_f32(self.shear_modulus, state);
                hash_f32(self.hardening_modulus, state);
                if self.constitutive_model == ConstitutiveModel::DruckerPrager.as_int() {
                    hash_f32(self.internal_friction_angle, state);
                }
                hash_f32(self.bulk_modulus, state);
                if self.bulk_modulus > 0.0 {
                    hash_f32(self.compaction_pressure, state);
                }
            }
            Some(_) => {
                // depth-averaged hydrostatic pressure with earth pressure
                // coefficients: all other models
                flags_mask |= SimFlags::CURVATURE
                    | SimFlags::PARTICLE_INTERACTION
                    | SimFlags::EARTH_PRESSURE_COEFFICIENT;
                if sim_model == Some(SimModel::Curvilinear) {
                    flags_mask |= SimFlags::WEAK_PRESSURE;
                }
                if SimFlags::from_u32(self.flags).is_earth_pressure_coefficient_enabled() {
                    hash_f32(self.internal_friction_angle, state);
                    hash_f32(self.basal_friction_angle, state);
                }
            }
            None => {}
        }
        if grid_flow_model {
            hash_f32(self.momentum_stop_fraction, state);
            hash_f32(self.reference_surface_length, state);
        }

        (self.flags & flags_mask).hash(state);
    }
}

/// An unset `release_thickness` or `release_swe` is NaN, which JSON holds as
/// null.
fn deserialize_unset_as_nan<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(f32::NAN))
}

/// Settings files written before `release_swe` existed do not hold it.
fn unset_release_swe() -> f32 {
    f32::NAN
}

fn is_release_thickness(thickness: f32) -> bool {
    thickness.is_finite() && thickness > RELEASE_THICKNESS_THRESHOLD
}

impl Default for SimSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl SimSettings {
    /// Length in steps of the no-new-cells rolling window, see utils.wgsl.
    pub fn no_new_cells_window(&self) -> u32 {
        ((self.no_new_cells_travel / self.cfl).round() as u32).max(10)
    }

    /// Whether the mu(I) rheology is in use, either as the basal friction
    /// model or as the constitutive law of the MPMDAC model.
    pub fn uses_mu_i(&self) -> bool {
        self.friction_model == FrictionModel::MuI.as_int()
            || (self.sim_model == SimModel::MpmDaC.as_int()
                && self.constitutive_model == ConstitutiveModel::MuI.as_int())
    }

    /// Sanity checks of the parameters the shaders cannot guard against
    /// without producing NaN or meaningless friction. The mu(I) parameters
    /// are only checked when the mu(I) rheology is in use.
    pub fn validate(&self) -> Result<(), String> {
        match (
            self.release_swe_override(),
            self.release_thickness_override(),
        ) {
            (Some(_), _) if !self.release_thickness.is_nan() => {
                return Err(
                    "`release_thickness` and `release_swe` exclude each other, set only one"
                        .to_string(),
                );
            }
            (Some(swe), _) if !(swe.is_finite() && swe > 0.0) => {
                return Err(format!("`release_swe` must be positive, got {swe}"));
            }
            // thinner cells are not seeded, so nothing would release
            (Some(swe), Some(thickness)) if !is_release_thickness(thickness) => {
                return Err(format!(
                    "`release_swe` of {swe} kg/m² at a `density` of {} kg/m³ is a thickness of \
                     {thickness} m, which must be above {RELEASE_THICKNESS_THRESHOLD} m",
                    self.density
                ));
            }
            (None, Some(thickness)) if !is_release_thickness(thickness) => {
                return Err(format!(
                    "`release_thickness` must be above {RELEASE_THICKNESS_THRESHOLD} m, got {thickness}"
                ));
            }
            _ => {}
        }
        if self.uses_mu_i() {
            let finite = [
                ("i0", self.i0),
                ("mu0", self.mu0),
                ("mu2", self.mu2),
                ("grain_diameter", self.grain_diameter),
            ];
            if let Some((name, value)) = finite.iter().find(|(_, value)| !value.is_finite()) {
                return Err(format!("mu(I): `{name}` must be finite, got {value}"));
            }
            if self.grain_diameter <= 0.0 {
                return Err(format!(
                    "mu(I): `grain_diameter` must be positive, got {}",
                    self.grain_diameter
                ));
            }
            if self.i0 <= 0.0 {
                return Err(format!("mu(I): `i0` must be positive, got {}", self.i0));
            }
            if self.mu0 < 0.0 {
                return Err(format!(
                    "mu(I): `mu0` must not be negative, got {}",
                    self.mu0
                ));
            }
            if self.mu2 < self.mu0 {
                return Err(format!(
                    "mu(I): `mu2` ({}) must not be smaller than `mu0` ({})",
                    self.mu2, self.mu0
                ));
            }
        }
        Ok(())
    }

    /// The `release_swe` setting, in kg/m².
    pub fn release_swe_override(&self) -> Option<f32> {
        (!self.release_swe.is_nan()).then_some(self.release_swe)
    }

    /// The `release_thickness` setting or the thickness `release_swe` gives
    /// at `density`; `None` keeps the thickness of the release source.
    pub fn release_thickness_override(&self) -> Option<f32> {
        (!self.release_thickness.is_nan())
            .then_some(self.release_thickness)
            .or(self.release_swe_override().map(|swe| swe / self.density))
    }

    /// Thickness of release areas that hold none of their own.
    pub fn release_thickness_or_default(&self) -> f32 {
        self.release_thickness_override()
            .unwrap_or(DEFAULT_RELEASE_THICKNESS)
    }

    pub fn calculate_hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish()
    }

    pub fn new() -> Self {
        Self {
            max_steps: 6000,
            sim_model: SimModel::TerrainFollowing.as_int(),
            friction_model: FrictionModel::Voellmy.as_int(),
            released_particles_per_cell: 8,
            grid_shape_x: 1,
            grid_shape_y: 1,
            world_size_x: 1.0,
            world_size_y: 1.0,
            density: 200.0,
            release_thickness: f32::NAN,

            friction_coefficient: 0.2,
            drag_coefficient: 2000.0,
            n0: 70.0,
            i0: 0.29,
            mu0: 0.38,
            mu2: 0.65,
            grain_diameter: 0.002,
            internal_friction_angle: 40.0,
            basal_friction_angle: 25.0,

            cfl: 0.9,
            cell_size: 1.0,
            velocity_threshold: 0.1,
            roughness_threshold: 0.01,
            // earth pressure coefficient off: Voellmy-type parameters are
            // calibrated with hydrostatic lateral pressure (K = 1)
            flags: SimFlags::new(true, true, false, true, false, false).mask,

            min_slope_angle: 28.0,
            max_slope_angle: 60.0,
            release_min_elevation: 1500.0,
            release_max_elevation: 8848.0,

            peak_flow_thickness_threshold: 0.1,

            constitutive_model: ConstitutiveModel::DruckerPrager.as_int(),
            shear_modulus: 5.0e4,
            hardening_modulus: 0.0,

            bulk_modulus: 2.0e4,
            compaction_pressure: 2.0e3,

            momentum_stop_fraction: 0.0,
            no_new_cells_travel: 65.0,
            reference_surface_length: 0.0,
            release_swe: f32::NAN,
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
        settings.apply_patch(patch);
        settings.set_dem(dem);
        settings
    }

    /// Overwrites the fields set in `patch`; unset fields keep their value.
    pub fn apply_patch(&mut self, patch: &Settings) {
        let settings = self;
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
        // a patch cannot unset a field, so setting one of the two clears the
        // other; a patch with both fails validation
        if patch.release_thickness.is_some() || patch.release_swe.is_some() {
            settings.release_thickness = patch.release_thickness.unwrap_or(f32::NAN);
            settings.release_swe = patch.release_swe.unwrap_or(f32::NAN);
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
        if let Some(val) = patch.release_max_elevation {
            settings.release_max_elevation = val;
        }
        if let Some(val) = patch.velocity_threshold {
            settings.velocity_threshold = val;
        }
        if let Some(val) = patch.roughness_threshold {
            settings.roughness_threshold = val;
        }
        if let Some(val) = patch.peak_flow_thickness_threshold {
            settings.peak_flow_thickness_threshold = val;
        }
        if let Some(val) = patch.constitutive_model {
            settings.constitutive_model = val.as_int();
        }
        if let Some(val) = patch.shear_modulus {
            settings.shear_modulus = val;
        }
        if let Some(val) = patch.hardening_modulus {
            settings.hardening_modulus = val;
        }
        if let Some(val) = patch.bulk_modulus {
            settings.bulk_modulus = val;
        }
        if let Some(val) = patch.compaction_pressure {
            settings.compaction_pressure = val;
        }
        if let Some(val) = patch.momentum_stop_fraction {
            settings.momentum_stop_fraction = val;
        }
        if let Some(val) = patch.no_new_cells_travel {
            settings.no_new_cells_travel = val;
        }
        if let Some(val) = patch.reference_surface_length {
            settings.reference_surface_length = val;
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
        if let Some(val) = patch.center_of_mass_biggest_blob {
            if val {
                settings.flags |= 1 << 4;
            } else {
                settings.flags &= !(1 << 4);
            }
        }
        if let Some(val) = patch.enable_weak_pressure {
            if val {
                settings.flags |= 1 << 5;
            } else {
                settings.flags &= !(1 << 5);
            }
        }
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
    /// rate-dependent mu(I) basal friction (model 5; model 4 is a stub)
    MuI,
}

impl FrictionModel {
    pub fn from_int(value: u32) -> Option<Self> {
        match value {
            0 => Some(FrictionModel::Coulomb),
            1 => Some(FrictionModel::Voellmy),
            2 => Some(FrictionModel::VoellmyMinShear),
            3 => Some(FrictionModel::SamosAT),
            5 => Some(FrictionModel::MuI),
            _ => None,
        }
    }

    pub fn as_int(&self) -> u32 {
        match self {
            FrictionModel::Coulomb => 0,
            FrictionModel::Voellmy => 1,
            FrictionModel::VoellmyMinShear => 2,
            FrictionModel::SamosAT => 3,
            FrictionModel::MuI => 5,
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
            "mu-i" | "mui" | "mu(i)" => Ok(Self::MuI),
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
            Self::MuI => "mu-i",
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
pub enum ConstitutiveModel {
    /// pressure-dependent Drucker-Prager / Mohr-Coulomb yield
    DruckerPrager,
    /// rate-dependent mu(I) inertial rheology
    MuI,
}

impl ConstitutiveModel {
    pub fn from_int(value: u32) -> Option<Self> {
        match value {
            0 => Some(ConstitutiveModel::DruckerPrager),
            1 => Some(ConstitutiveModel::MuI),
            _ => None,
        }
    }

    pub fn as_int(&self) -> u32 {
        match self {
            ConstitutiveModel::DruckerPrager => 0,
            ConstitutiveModel::MuI => 1,
        }
    }
}

impl FromStr for ConstitutiveModel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "drucker-prager" | "drucker_prager" | "dp" | "mohr-coulomb" => Ok(Self::DruckerPrager),
            "mu-i" | "mui" | "mu(i)" | "mu-i-rheology" => Ok(Self::MuI),
            _ => Err(format!("unknown constitutive model: {value}")),
        }
    }
}

impl fmt::Display for ConstitutiveModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DruckerPrager => "drucker-prager",
            Self::MuI => "mu-i",
        })
    }
}

impl Serialize for ConstitutiveModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ConstitutiveModel {
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
                .ok_or_else(|| D::Error::custom(format!("invalid constitutive model: {value}"))),

            Value::String(value) => value.parse::<Self>().map_err(D::Error::custom),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimModel {
    TerrainFollowing,
    Curvilinear,
    MpmDaC,
    /// Legacy per-particle sliding-block model (former model 0).
    SlidingBlock,
    /// Depth-averaged terrain-following grid model that carries the full 3D
    /// Cartesian velocity.
    ThreeDDepthAveraged,
}

impl SimModel {
    pub fn from_int(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::TerrainFollowing),
            1 => Some(Self::Curvilinear),
            2 => Some(Self::MpmDaC),
            3 => Some(Self::SlidingBlock),
            4 => Some(Self::ThreeDDepthAveraged),
            _ => None,
        }
    }

    pub fn as_int(&self) -> u32 {
        match self {
            Self::TerrainFollowing => 0,
            Self::Curvilinear => 1,
            Self::MpmDaC => 2,
            Self::SlidingBlock => 3,
            Self::ThreeDDepthAveraged => 4,
        }
    }
}

impl FromStr for SimModel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "terrain-following" | "terrain" | "t" => Ok(Self::TerrainFollowing),
            "curvilinear" | "curvi" | "c" => Ok(Self::Curvilinear),
            "mpmdac" | "mpm-dac" | "mpm_depth_averaged_curvilinear" => Ok(Self::MpmDaC),
            "sliding-block" | "block" | "b" => Ok(Self::SlidingBlock),
            "3d-depth-averaged" | "3d" => Ok(Self::ThreeDDepthAveraged),
            _ => Err(format!("unknown simulation model: {value}")),
        }
    }
}

impl fmt::Display for SimModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TerrainFollowing => "terrain-following",
            Self::Curvilinear => "curvilinear",
            Self::MpmDaC => "mpmdac",
            Self::SlidingBlock => "sliding-block",
            Self::ThreeDDepthAveraged => "3d-depth-averaged",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrownLineMethod {
    FlowRouting,
    ParticleSimulation,
    GradientStep,
}

impl FromStr for CrownLineMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().replace('_', "-").as_str() {
            "flow-routing" | "flow" | "d8" => Ok(Self::FlowRouting),
            "particle-simulation" | "particle" => Ok(Self::ParticleSimulation),
            "gradient-step" | "gradient" => Ok(Self::GradientStep),
            _ => Err(format!("unknown crown line method: {value}")),
        }
    }
}

impl fmt::Display for CrownLineMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::FlowRouting => "flow-routing",
            Self::ParticleSimulation => "particle-simulation",
            Self::GradientStep => "gradient-step",
        })
    }
}

impl Serialize for CrownLineMethod {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for CrownLineMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse::<Self>()
            .map_err(D::Error::custom)
    }
}

/// How the release estimation orders the potential release cells below the
/// crown line: by geodesic distance alone, or by a geodesic distance whose
/// steps through unfavourable release slopes count as longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseFillMethod {
    Geodesic,
    SlopeWeighted,
}

impl FromStr for ReleaseFillMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().replace('_', "-").as_str() {
            "geodesic" | "distance" => Ok(Self::Geodesic),
            "slope-weighted" | "slope" => Ok(Self::SlopeWeighted),
            _ => Err(format!("unknown release fill method: {value}")),
        }
    }
}

impl fmt::Display for ReleaseFillMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::Geodesic => "geodesic",
            Self::SlopeWeighted => "slope-weighted",
        })
    }
}

impl Serialize for ReleaseFillMethod {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ReleaseFillMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse::<Self>()
            .map_err(D::Error::custom)
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
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
    /// Steps between particle sorts for the tile-sorted P2G of the grid flow models (default 25);
    /// 0 disables it.
    pub particle_sort_interval: Option<u32>,
    pub friction_model: Option<FrictionModel>,
    pub released_particles_per_cell: Option<u32>,
    pub density: Option<f32>,
    /// Release thickness (m), measured normal to the slope, of every release
    /// cell. If unset, rasters, arrays, stores and shapefiles with thickness
    /// attributes keep their own thickness and all other release areas get
    /// [`DEFAULT_RELEASE_THICKNESS`]. Excludes `release_swe`.
    pub release_thickness: Option<f32>,
    /// Snow water equivalent (kg/m², the same as mm of water) of every release
    /// cell, set instead of `release_thickness`. It is the mass per horizontal
    /// area, so a cell holds `release_swe * cell_size²` whatever its slope, and
    /// `release_swe / density` must be above [`RELEASE_THICKNESS_THRESHOLD`].
    pub release_swe: Option<f32>,
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
    /// Minimum slope angle (deg) of release terrain. Together with
    /// `max_slope_angle`, `release_min_elevation`, `release_max_elevation` and
    /// `roughness_threshold` it defines the potential release areas, from
    /// which the `release_area_fraction` estimation selects its release cells.
    pub min_slope_angle: Option<f32>,
    pub max_slope_angle: Option<f32>,
    pub release_min_elevation: Option<f32>,
    pub release_max_elevation: Option<f32>,
    /// Target release area as a fraction of the outline area (0.2 - 0.3 is a
    /// common range). When set, release areas are estimated by detecting the
    /// outline's crown line and selecting this share of the outline area from
    /// the potential release areas (the terrain-threshold classification),
    /// nearest to the crown line first.
    pub release_area_fraction: Option<f32>,
    /// How the crown line is detected for `release_area_fraction`: D8 flow
    /// routing on the CPU (default), a GPU particle simulation released
    /// everywhere outside the outline, or a gradient step (outline cells whose
    /// one-cell step uphill leaves the outline).
    pub crown_line_method: Option<CrownLineMethod>,
    /// Exit distance threshold of the `gradient-step` crown line method as a
    /// fraction of the cell size, in [0, 1) (defaults to 0.75). Outline cells
    /// whose one-cell step uphill lands outside the outline and farther than
    /// this from the nearest outline cell centre become crown line; lower
    /// values also pick up spreading flanks (0.5 or less counts every exit).
    pub crown_line_exit_distance: Option<f32>,
    /// Crown candidate components whose mean elevation lies below this share
    /// of the outline's elevation range are discarded as flank gullies, for
    /// all crown line methods (defaults to 0, which keeps every component).
    pub crown_elevation_band: Option<f32>,
    /// Minimum number of entering particles for an outline cell to become a
    /// crown candidate in the `particle-simulation` method (defaults to 2).
    pub min_particles_per_crown_cell: Option<u32>,
    /// How `release_area_fraction` orders the potential release cells below
    /// the crown line: `geodesic` distance (default), or `slope-weighted`,
    /// where steps through slopes outside the typical release range count
    /// as longer, so the fill reaches further into favourable terrain.
    pub release_fill_method: Option<ReleaseFillMethod>,
    /// Blend weight w of the `slope-weighted` fill: a step costs its length
    /// times 1 + w (1 - mu), with mu the release slope membership. 0 equals
    /// the geodesic fill (defaults to 3).
    pub release_slope_weight: Option<f32>,
    /// Slope angle (deg) of maximum release slope membership of the
    /// `slope-weighted` fill (defaults to 43, Veitinger et al. 2016).
    pub release_slope_peak: Option<f32>,
    /// Half width (deg) of the release slope membership of the
    /// `slope-weighted` fill: mu is 0.5 at peak +- width (defaults to 11).
    pub release_slope_width: Option<f32>,
    /// Gaussian width (m) of the DEM smoothing before the slope of the
    /// `slope-weighted` fill is computed, so the slope approaches the
    /// starting-zone scale the release statistics were measured at
    /// (defaults to 10; 0 uses the raw DEM cells).
    pub release_slope_smoothing_length: Option<f32>,
    pub velocity_threshold: Option<f32>,
    pub roughness_threshold: Option<f32>,
    pub peak_flow_thickness_threshold: Option<f32>,

    /// Constitutive law of the MPMDAC model: pressure-dependent
    /// Drucker-Prager (default) or the rate-dependent mu(I) rheology.
    pub constitutive_model: Option<ConstitutiveModel>,
    /// Deviatoric shear modulus (Pa) of the MPMDAC model's elastic
    /// predictor; large values approach the rigid-plastic limit.
    pub shear_modulus: Option<f32>,
    /// Linear isotropic hardening modulus (Pa per unit accumulated plastic
    /// strain) of the MPMDAC model; 0 disables hardening.
    pub hardening_modulus: Option<f32>,
    /// Volumetric bulk modulus (Pa) of the MPMDAC model. Non-zero makes the
    /// flow material compressible: pressure responds elastically to
    /// volumetric strain (0 = incompressible, the default is 2e4).
    pub bulk_modulus: Option<f32>,
    /// Compaction pressure (Pa) above which the MPMDAC material densifies
    /// plastically (irreversibly); the volumetric pressure is capped there.
    pub compaction_pressure: Option<f32>,
    /// Grid flow models: stop once the total grid momentum falls below this
    /// fraction of its peak, e.g. 0.01; 0 or unset disables the criterion.
    pub momentum_stop_fraction: Option<f32>,
    /// The simulation stops when fewer than 4 cells were newly reached within
    /// the steps the fastest material needs to travel this many cells,
    /// round(travel / cfl) and at least 10 (default 65 cells, 72 steps at CFL 0.9).
    pub no_new_cells_travel: Option<f32>,
    /// Curvilinear, terrain-following and 3D depth averaged: Gaussian width (m)
    /// of the smoothed reference surface.
    /// Bed inclined by beta against it counts as a gully wall whose normal
    /// load grows by 1 + K tan^2(beta) from the lateral pressure of the snow
    /// beside it; 0 or unset disables it.
    pub reference_surface_length: Option<f32>,

    pub enable_curvature: Option<bool>,
    pub enable_particle_interaction: Option<bool>,
    pub enable_earth_pressure_coefficient: Option<bool>,
    /// Curvilinear: hydrostatic pressure gradient from the MPM weak form,
    /// -sum_p m_p h_p / 2 grad N_i(x_p) per node, instead of the centred
    /// difference of the nodal thickness. Thin front nodes are then pushed by
    /// the full pressure of the particles behind them (defaults to false).
    pub enable_weak_pressure: Option<bool>,
    pub enable_entrainment: Option<bool>,
    pub enable_center_of_mass: Option<bool>,
    /// Relax freshly initialized particles to the hexagonal packing spacing
    /// with a soft sphere repulsion before the simulation starts (defaults
    /// to true).
    pub enable_particle_relaxation: Option<bool>,
    /// Center the mass of the biggest connected mass blob instead of the
    /// whole grid (defaults to false).
    pub center_of_mass_biggest_blob: Option<bool>,
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

    #[test]
    fn no_new_cells_window_covers_constant_front_travel() {
        let mut settings = SimSettings::new();
        assert_eq!(settings.no_new_cells_window(), 72);
        settings.cfl = 0.05;
        assert_eq!(settings.no_new_cells_window(), 1300);
        settings.no_new_cells_travel = 1.0;
        settings.cfl = 0.9;
        assert_eq!(settings.no_new_cells_window(), 10);
    }

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
        assert_eq!(settings.sim_model, SimModel::TerrainFollowing.as_int());
        assert_eq!(settings.friction_model, FrictionModel::Voellmy.as_int());
        assert_eq!(settings.released_particles_per_cell, 8);
        assert_eq!(settings.grid_shape_x, 1);
        assert_eq!(settings.grid_shape_y, 1);
        assert_eq!(settings.world_size_x, 1.0);
        assert_eq!(settings.world_size_y, 1.0);
        assert_eq!(settings.density, 200.0);
        assert_eq!(settings.release_thickness_override(), None);
        assert_eq!(settings.friction_coefficient, 0.2);
        assert_eq!(settings.drag_coefficient, 2000.0);
        assert_eq!(settings.cfl, 0.9);
        assert_eq!(settings.cell_size, 1.0);
        assert_eq!(settings.min_slope_angle, 28.0);
        assert_eq!(settings.max_slope_angle, 60.0);
        assert_eq!(settings.release_min_elevation, 1500.0);
        assert_eq!(settings.release_max_elevation, 8848.0);
        assert_eq!(settings.velocity_threshold, 0.1);
        assert_eq!(settings.roughness_threshold, 0.01);
        assert_eq!(settings.peak_flow_thickness_threshold, 0.1);
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
            sim_model: Some(SimModel::Curvilinear),
            friction_model: Some(FrictionModel::VoellmyMinShear),
            released_particles_per_cell: Some(3),
            density: Some(123.4),
            release_thickness: Some(5.6),
            release_swe: None,
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
            particle_sort_interval: None,
            momentum_stop_fraction: None,
            no_new_cells_travel: None,
            reference_surface_length: Some(12.5),
            cfl: Some(0.9),
            min_slope_angle: Some(10.0),
            max_slope_angle: Some(20.0),
            release_min_elevation: Some(100.0),
            release_max_elevation: Some(200.0),
            release_area_fraction: Some(0.25),
            crown_line_method: Some(CrownLineMethod::GradientStep),
            crown_line_exit_distance: Some(0.7),
            crown_elevation_band: Some(0.3),
            min_particles_per_crown_cell: Some(3),
            release_fill_method: Some(ReleaseFillMethod::SlopeWeighted),
            release_slope_weight: Some(2.0),
            release_slope_peak: Some(40.0),
            release_slope_width: Some(10.0),
            release_slope_smoothing_length: Some(5.0),
            velocity_threshold: Some(0.001),
            roughness_threshold: Some(0.002),
            peak_flow_thickness_threshold: Some(1.5),
            constitutive_model: Some(ConstitutiveModel::MuI),
            shear_modulus: Some(1234.5),
            hardening_modulus: Some(678.9),
            bulk_modulus: Some(246.8),
            compaction_pressure: Some(135.7),
            dem_path: Some(String::from("dem.png")),
            release_areas_path: Some(String::from("release_area.png")),
            output_path: Some(String::from("output")),
            enable_curvature: Some(false),
            enable_particle_interaction: Some(false),
            enable_particle_relaxation: Some(false),
            enable_earth_pressure_coefficient: Some(false),
            enable_entrainment: Some(false),
            enable_weak_pressure: Some(false),
            enable_center_of_mass: Some(true),
            center_of_mass_biggest_blob: Some(false),
        };
        let dem = create_test_dem();
        let mut sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(sim_settings.max_steps, 42);
        assert_eq!(sim_settings.sim_model, 1);
        assert_eq!(sim_settings.friction_model, 2);
        assert_eq!(sim_settings.released_particles_per_cell, 3);
        assert_eq!(sim_settings.density, 123.4);
        assert_eq!(sim_settings.release_thickness_override(), Some(5.6));
        assert_eq!(sim_settings.friction_coefficient, 0.2);
        assert_eq!(sim_settings.drag_coefficient, 999.0);
        assert_eq!(sim_settings.cfl, 0.9);
        assert_eq!(sim_settings.min_slope_angle, 10.0);
        assert_eq!(sim_settings.max_slope_angle, 20.0);
        assert_eq!(sim_settings.release_min_elevation, 100.0);
        assert_eq!(sim_settings.release_max_elevation, 200.0);
        assert_eq!(sim_settings.velocity_threshold, 0.001);
        assert_eq!(sim_settings.roughness_threshold, 0.002);
        assert_eq!(sim_settings.peak_flow_thickness_threshold, 1.5);
        assert_eq!(sim_settings.constitutive_model, 1);
        assert_eq!(sim_settings.shear_modulus, 1234.5);
        assert_eq!(sim_settings.hardening_modulus, 678.9);
        assert_eq!(sim_settings.bulk_modulus, 246.8);
        assert_eq!(sim_settings.compaction_pressure, 135.7);
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
            SimFlags::new(false, false, false, false, false, false).mask
        );

        // Test enabling flags one by one
        patch.enable_curvature = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, false, false, false, false, false).mask
        );

        patch.enable_particle_interaction = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, false, false, false, false).mask
        );

        patch.enable_earth_pressure_coefficient = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, true, false, false, false).mask
        );

        patch.enable_entrainment = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, true, true, false, false).mask
        );

        patch.center_of_mass_biggest_blob = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, true, true, true, false).mask
        );

        patch.enable_weak_pressure = Some(true);
        sim_settings = SimSettings::from_settings(&patch, &dem);
        assert_eq!(
            sim_settings.flags,
            SimFlags::new(true, true, true, true, true, true).mask
        );
        assert!(SimFlags::from_u32(sim_settings.flags).is_weak_pressure_enabled());
    }

    #[test_log::test]
    fn test_settings_reject_the_former_slab_thickness_factor() {
        // an alias would turn a value rasters ignored into an override
        let error = Settings::loads(r#"{"slab_thickness_factor": 1.0}"#).unwrap_err();
        assert!(error.to_string().contains("slab_thickness_factor"));
    }

    #[test_log::test]
    fn test_release_thickness_must_exceed_the_release_threshold() {
        let mut settings = SimSettings::new();
        assert!(settings.validate().is_ok(), "unset is valid");
        assert_eq!(
            settings.release_thickness_or_default(),
            DEFAULT_RELEASE_THICKNESS
        );

        for invalid in [-1.0, 0.0, RELEASE_THICKNESS_THRESHOLD, f32::INFINITY] {
            settings.release_thickness = invalid;
            assert!(settings.validate().is_err(), "accepted {invalid}");
        }

        settings.release_thickness = 0.5;
        assert!(settings.validate().is_ok());
        assert_eq!(settings.release_thickness_or_default(), 0.5);
    }

    #[test_log::test]
    fn test_release_swe_excludes_release_thickness() {
        let mut settings = SimSettings::new();
        settings.release_swe = 100.0;
        assert!(settings.validate().is_ok());

        settings.release_thickness = 0.5;
        let error = settings.validate().unwrap_err();
        assert!(error.contains("exclude each other"), "{error}");
    }

    #[test_log::test]
    fn test_release_swe_must_be_positive() {
        let mut settings = SimSettings::new();
        for invalid in [-1.0, 0.0, f32::INFINITY] {
            settings.release_swe = invalid;
            let error = settings.validate().unwrap_err();
            assert!(error.contains("must be positive"), "{error}");
        }
    }

    #[test_log::test]
    fn test_release_swe_must_give_a_seedable_thickness() {
        let mut settings = SimSettings::new();
        settings.density = 200.0;
        settings.release_swe = 1.0;
        let error = settings.validate().unwrap_err();
        assert!(error.contains("`density`"), "{error}");

        settings.release_swe = 3.0;
        assert!(settings.validate().is_ok());
    }

    #[test_log::test]
    fn test_release_swe_gives_the_thickness_at_the_density() {
        let mut settings = SimSettings::new();
        assert_eq!(settings.release_swe_override(), None);

        settings.density = 250.0;
        settings.release_swe = 100.0;
        assert_eq!(settings.release_swe_override(), Some(100.0));
        assert_eq!(settings.release_thickness_override(), Some(0.4));
        assert_eq!(settings.release_thickness_or_default(), 0.4);
    }

    #[test_log::test]
    fn test_patch_switches_between_release_thickness_and_swe() {
        let mut settings = SimSettings::new();
        settings.apply_patch(&Settings {
            release_thickness: Some(1.5),
            ..Default::default()
        });
        settings.apply_patch(&Settings {
            release_swe: Some(100.0),
            ..Default::default()
        });
        assert!(settings.release_thickness.is_nan());
        assert_eq!(settings.release_swe_override(), Some(100.0));

        settings.apply_patch(&Settings {
            release_thickness: Some(1.5),
            ..Default::default()
        });
        assert_eq!(settings.release_swe_override(), None);
        assert_eq!(settings.release_thickness_override(), Some(1.5));

        settings.apply_patch(&Settings {
            density: Some(300.0),
            ..Default::default()
        });
        assert_eq!(
            settings.release_thickness_override(),
            Some(1.5),
            "unrelated patch"
        );

        settings.apply_patch(&Settings {
            release_thickness: Some(1.5),
            release_swe: Some(100.0),
            ..Default::default()
        });
        assert!(settings.validate().is_err(), "a patch with both must fail");
    }

    #[test_log::test]
    fn test_settings_json_without_release_swe_reads_as_unset() {
        let mut json = serde_json::to_value(SimSettings::new()).unwrap();
        json.as_object_mut().unwrap().remove("release_swe");
        let settings: SimSettings = serde_json::from_value(json).unwrap();
        assert_eq!(settings.release_swe_override(), None);
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
            settings.release_thickness_override(),
            deserialized.release_thickness_override()
        );
        assert_eq!(
            settings.release_swe_override(),
            deserialized.release_swe_override()
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
        assert_eq!(
            settings.release_max_elevation,
            deserialized.release_max_elevation
        );
        assert_eq!(settings.velocity_threshold, deserialized.velocity_threshold);
        assert_eq!(
            settings.roughness_threshold,
            deserialized.roughness_threshold
        );
        assert_eq!(
            settings.peak_flow_thickness_threshold,
            deserialized.peak_flow_thickness_threshold
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
        assert_eq!(FrictionModel::from_int(5), Some(FrictionModel::MuI));
        assert_eq!(FrictionModel::MuI.as_int(), 5);
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
            SimModel::TerrainFollowing,
            SimModel::Curvilinear,
            SimModel::MpmDaC,
            SimModel::SlidingBlock,
            SimModel::ThreeDDepthAveraged,
        ];

        for model in models {
            let value = model.as_int();
            let converted = SimModel::from_int(value);

            assert_eq!(converted, Some(model));
        }
    }

    #[test]
    fn sim_model_from_int() {
        assert_eq!(SimModel::from_int(0), Some(SimModel::TerrainFollowing));
        assert_eq!(SimModel::from_int(1), Some(SimModel::Curvilinear));
        assert_eq!(SimModel::from_int(2), Some(SimModel::MpmDaC));

        assert_eq!(SimModel::from_int(3), Some(SimModel::SlidingBlock));

        assert_eq!(SimModel::from_int(4), Some(SimModel::ThreeDDepthAveraged));
        assert_eq!(SimModel::from_int(5), None);
        assert_eq!(SimModel::from_int(u32::MAX), None);
    }

    #[test]
    fn sim_model_string_roundtrip() {
        let models = [
            (SimModel::TerrainFollowing, "terrain-following"),
            (SimModel::Curvilinear, "curvilinear"),
            (SimModel::MpmDaC, "mpmdac"),
            (SimModel::SlidingBlock, "sliding-block"),
        ];

        for (model, string) in models {
            assert_eq!(model.to_string(), string);
            assert_eq!(SimModel::from_str(string), Ok(model));
        }
    }

    #[test]
    fn sim_model_string_aliases() {
        for value in ["terrain-following", "terrain", "t", "TERRAIN"] {
            assert_eq!(SimModel::from_str(value), Ok(SimModel::TerrainFollowing));
        }
        for value in ["curvilinear", "curvi", "c", "CURVI"] {
            assert_eq!(SimModel::from_str(value), Ok(SimModel::Curvilinear));
        }
        for value in ["mpmdac", "MPM-DAC", "mpm_depth_averaged_curvilinear"] {
            assert_eq!(SimModel::from_str(value), Ok(SimModel::MpmDaC));
        }
        for value in ["sliding-block", "block", "b", "BLOCK"] {
            assert_eq!(SimModel::from_str(value), Ok(SimModel::SlidingBlock));
        }
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
            FrictionModel::MuI,
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
        assert_eq!(FrictionModel::from_int(5), Some(FrictionModel::MuI));

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
            (FrictionModel::MuI, "mu-i"),
        ];

        for (model, string) in models {
            assert_eq!(model.to_string(), string);
            assert_eq!(FrictionModel::from_str(string), Ok(model));
        }
    }

    #[test]
    fn constitutive_model_roundtrip() {
        assert_eq!(
            ConstitutiveModel::from_int(0),
            Some(ConstitutiveModel::DruckerPrager)
        );
        assert_eq!(ConstitutiveModel::from_int(1), Some(ConstitutiveModel::MuI));
        assert_eq!(ConstitutiveModel::from_int(2), None);

        assert_eq!(
            ConstitutiveModel::DruckerPrager.to_string(),
            "drucker-prager"
        );
        assert_eq!(ConstitutiveModel::MuI.to_string(), "mu-i");
        assert_eq!(
            ConstitutiveModel::from_str("drucker-prager"),
            Ok(ConstitutiveModel::DruckerPrager)
        );
        assert_eq!(
            ConstitutiveModel::from_str("dp"),
            Ok(ConstitutiveModel::DruckerPrager)
        );
        assert_eq!(
            ConstitutiveModel::from_str("mohr-coulomb"),
            Ok(ConstitutiveModel::DruckerPrager)
        );
        assert_eq!(
            ConstitutiveModel::from_str("mui"),
            Ok(ConstitutiveModel::MuI)
        );
        assert!(ConstitutiveModel::from_str("invalid").is_err());
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

        assert_eq!(FrictionModel::from_str("mu(I)"), Ok(FrictionModel::MuI));
    }

    #[test]
    fn mu_i_parameters_validated_only_when_selected() {
        let invalid = SimSettings {
            grain_diameter: 0.0,
            i0: -1.0,
            mu0: 0.5,
            mu2: 0.2,
            ..SimSettings::new()
        };
        // Voellmy basal friction with the Drucker-Prager law ignores them
        assert!(invalid.validate().is_ok());
        // the constitutive mu(I) law only runs in the MPMDAC model
        let curvilinear_mu_i = SimSettings {
            sim_model: SimModel::Curvilinear.as_int(),
            constitutive_model: ConstitutiveModel::MuI.as_int(),
            ..invalid
        };
        assert!(curvilinear_mu_i.validate().is_ok());
        let mpmdac_mu_i = SimSettings {
            sim_model: SimModel::MpmDaC.as_int(),
            ..curvilinear_mu_i
        };
        assert!(mpmdac_mu_i.validate().is_err());

        let basal_mu_i = SimSettings {
            friction_model: FrictionModel::MuI.as_int(),
            ..SimSettings::new()
        };
        assert!(basal_mu_i.validate().is_ok(), "defaults must be valid");
        for broken in [
            SimSettings {
                grain_diameter: 0.0,
                ..basal_mu_i
            },
            SimSettings {
                i0: 0.0,
                ..basal_mu_i
            },
            SimSettings {
                mu0: -0.1,
                ..basal_mu_i
            },
            SimSettings {
                mu2: basal_mu_i.mu0 - 0.1,
                ..basal_mu_i
            },
            SimSettings {
                mu2: f32::NAN,
                ..basal_mu_i
            },
        ] {
            assert!(broken.validate().is_err(), "{broken:?}");
        }
    }

    #[test]
    fn friction_model_invalid_string() {
        assert!(FrictionModel::from_str("invalid").is_err());
        assert!(FrictionModel::from_str("").is_err());
    }

    #[test]
    fn crown_line_method_string_roundtrip() {
        let methods = [
            (CrownLineMethod::FlowRouting, "flow-routing"),
            (CrownLineMethod::ParticleSimulation, "particle-simulation"),
            (CrownLineMethod::GradientStep, "gradient-step"),
        ];

        for (method, string) in methods {
            assert_eq!(method.to_string(), string);
            assert_eq!(CrownLineMethod::from_str(string), Ok(method));
        }
    }

    #[test]
    fn crown_line_method_string_aliases() {
        assert_eq!(
            CrownLineMethod::from_str("flow_routing"),
            Ok(CrownLineMethod::FlowRouting)
        );
        assert_eq!(
            CrownLineMethod::from_str("D8"),
            Ok(CrownLineMethod::FlowRouting)
        );
        assert_eq!(
            CrownLineMethod::from_str("particle"),
            Ok(CrownLineMethod::ParticleSimulation)
        );
        assert_eq!(
            CrownLineMethod::from_str("Gradient_Step"),
            Ok(CrownLineMethod::GradientStep)
        );
        assert_eq!(
            CrownLineMethod::from_str("gradient"),
            Ok(CrownLineMethod::GradientStep)
        );
    }

    #[test]
    fn crown_line_method_invalid_string() {
        assert!(CrownLineMethod::from_str("invalid").is_err());
        assert!(CrownLineMethod::from_str("").is_err());
    }

    #[test]
    fn release_fill_method_string_roundtrip() {
        let methods = [
            (ReleaseFillMethod::Geodesic, "geodesic"),
            (ReleaseFillMethod::SlopeWeighted, "slope-weighted"),
        ];

        for (method, string) in methods {
            assert_eq!(method.to_string(), string);
            assert_eq!(ReleaseFillMethod::from_str(string), Ok(method));
        }
    }

    #[test]
    fn release_fill_method_string_aliases() {
        assert_eq!(
            ReleaseFillMethod::from_str("distance"),
            Ok(ReleaseFillMethod::Geodesic)
        );
        assert_eq!(
            ReleaseFillMethod::from_str("Slope_Weighted"),
            Ok(ReleaseFillMethod::SlopeWeighted)
        );
        assert_eq!(
            ReleaseFillMethod::from_str("slope"),
            Ok(ReleaseFillMethod::SlopeWeighted)
        );
    }

    #[test]
    fn release_fill_method_invalid_string() {
        assert!(ReleaseFillMethod::from_str("invalid").is_err());
        assert!(ReleaseFillMethod::from_str("").is_err());
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

    #[test]
    fn test_sim_settings_hash_ignores_parameters_of_other_models() {
        // terrain following with Voellmy friction
        let settings = SimSettings::new();
        let hash = settings.calculate_hash();
        let unchanged = |change: fn(&mut SimSettings)| {
            let mut changed = settings;
            change(&mut changed);
            changed.calculate_hash() == hash
        };

        // mu(I) friction, MPMDAC and earth pressure (disabled) parameters
        assert!(unchanged(|s| s.i0 = 0.5));
        assert!(unchanged(|s| s.grain_diameter = 0.01));
        assert!(unchanged(|s| s.n0 = 1.0));
        assert!(unchanged(|s| s.shear_modulus = 1.0));
        assert!(unchanged(|s| s.bulk_modulus = 1.0));
        assert!(unchanged(|s| s.constitutive_model = 1));
        assert!(unchanged(|s| s.internal_friction_angle = 30.0));
        assert!(unchanged(|s| s.flags |= SimFlags::WEAK_PRESSURE));

        assert!(!unchanged(|s| s.drag_coefficient = 1000.0));
        assert!(!unchanged(|s| s.reference_surface_length = 20.0));
        assert!(!unchanged(
            |s| s.flags |= SimFlags::EARTH_PRESSURE_COEFFICIENT
        ));
        assert!(!unchanged(|s| {
            s.flags |= SimFlags::EARTH_PRESSURE_COEFFICIENT;
            s.internal_friction_angle = 30.0;
        }));
    }

    #[test]
    fn test_sim_settings_hash_follows_the_selected_models() {
        let mut mu_i = SimSettings::new();
        mu_i.friction_model = FrictionModel::MuI.as_int();
        let mut changed = mu_i;
        changed.drag_coefficient = 1000.0;
        assert_eq!(mu_i.calculate_hash(), changed.calculate_hash());
        changed.mu0 = 0.3;
        assert_ne!(mu_i.calculate_hash(), changed.calculate_hash());

        let mut mpmdac = SimSettings::new();
        mpmdac.sim_model = SimModel::MpmDaC.as_int();
        let mut changed = mpmdac;
        changed.reference_surface_length = 20.0;
        changed.momentum_stop_fraction = 0.01;
        assert_eq!(mpmdac.calculate_hash(), changed.calculate_hash());
        changed.internal_friction_angle = 30.0;
        assert_ne!(mpmdac.calculate_hash(), changed.calculate_hash());
    }
}
