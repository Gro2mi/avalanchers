//! **The calibration harness — most of our work is here.** Prepares cases (rasterise outline → fetch padded DEM → Horn slope/aspect → build release band), evaluates against the observed outline, and drives the search. Subcommands: `run`, `sweep`, `search` (one global vector), `per-event` (per-case Nelder-Mead, logs every evaluation to `.evals.jsonl`), `grid` (2-D μ×ξ identifiability scan), `apply` (each case at its own vector — the only honest way to score a regressor), `dump` (rasters for plotting). `Params` + `BOUNDS`/`DIM_NAMES` define the search space; `--gpu-index` pins a card.
//! Calibration / baseline harness.
//!
//! Scores the runout simulator against observed avalanche outlines
//! (SPOT6 24 Jan 2018, Hafner & Bühler, EnviDat DOI 10.16904/envidat.77)
//! using the indices in `compute_core::evaluation`.
//!
//! One "case" is a single mapped avalanche polygon (EPSG:2056). For each case
//! the harness
//!   1. rasterises the polygon at 5 m and pads it,
//!   2. pulls the matching swissALTI3D DEM through `TileManager`,
//!   3. derives a release area from the upper part of the observed polygon,
//!   4. runs the GPU model,
//!   5. compares the simulated affected area against the rasterised polygon.
//!
//! Everything is deterministic (the particle seed is hard-coded to 42 in the
//! shader), so repeated evaluations of the same parameter vector are identical.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use compute_core::dem::Dem;
use compute_core::evaluation::{
    MassMovementEvaluation, evaluate_distance_weighted_mass_movement_runout,
    evaluate_mass_movement_area,
};
use data_processor::rasterizer::RasterGrid;
use data_processor::shapefile_reader::read_shapefile_nth_polygon;
use data_processor::tile_manager::{BBox, TileManager};
use serde::{Deserialize, Serialize};
use simulation::Simulation;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const CELL: f64 = 5.0;

/// Set in `main` for every subcommand; every single simulation (not just
/// per-case summaries) is appended here as one JSON line, flushed immediately
/// -- so a hard kill loses at most the evaluation in flight. This is a global
/// rather than a threaded-through parameter because `evaluate_case` is called
/// from all six subcommands and from inside the Nelder-Mead objective, and the
/// tokio runtime here has multiple worker threads, so a thread-local would not
/// reliably see writes across `.await` points.
static EVAL_LOG: OnceLock<Mutex<std::io::BufWriter<std::fs::File>>> = OnceLock::new();

/// Which experiment stage and which global candidate this process is
/// evaluating. Stamped onto every row so a database built from many runs can
/// be sliced without reconstructing provenance from file paths.
static PROVENANCE: OnceLock<Provenance> = OnceLock::new();

#[derive(Debug, Clone, Default)]
struct Provenance {
    stage: String,
    candidate: String,
}

/// Per-case evaluation counter. An inner-loop iteration index has to be
/// recoverable from the row itself: file order is not enough once several
/// cases interleave in one log, and it is what distinguishes "the search
/// started here" from "the search converged here".
static ITER: OnceLock<Mutex<std::collections::HashMap<String, u64>>> = OnceLock::new();

fn next_iter(case: &str) -> u64 {
    let m = ITER.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    // A poisoned lock here means another thread panicked mid-log; the counter
    // is still consistent, and losing the whole log to that would be worse.
    let mut g = match m.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let c = g.entry(case.to_string()).or_insert(0);
    *c += 1;
    *c
}

/// The scored quantities of one simulation. Everything a downstream analysis
/// needs to reproduce a decision, and nothing per-cell -- rasters are dumped
/// separately, only for final winning vectors.
#[derive(Serialize)]
struct EvalMetrics {
    omega: f64,
    /// Omega_T of the "a particle visited this cell" footprint. Scored on
    /// every run so candidates can be compared across the particle-interaction
    /// flag, which otherwise silently changes the footprint definition.
    omega_cells: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    hwri_l1: f64,
    hwri_l05: f64,
    release_only_omega: f64,
    reach_err_m: f32,
    sim_reach_m: f32,
    obs_reach_m: f32,
    /// filter diagnostics: what the fixed scoring conventions removed
    sim_reach_raw_m: f32,
    sim_cells_raw: usize,
    omega_raw: f64,
    release_clipped_frac: f64,
    release_clip_severe: bool,
    connect_fallback: bool,
    sim_cells: usize,
    ref_cells: usize,
    release_cells: usize,
    release_volume_m3: f64,
    mean_slab_m: f64,
    steps: u32,
    max_velocity: f32,
    sim_flags: u32,
    clipped_at_edge: bool,
    seconds: f64,
}

impl From<&CaseResult> for EvalMetrics {
    fn from(c: &CaseResult) -> Self {
        EvalMetrics {
            omega: c.area.omega,
            omega_cells: c.omega_cells,
            alpha: c.area.alpha,
            beta: c.area.beta,
            gamma: c.area.gamma,
            hwri_l1: c.hwri_l1.omega,
            hwri_l05: c.hwri_l05.omega,
            release_only_omega: c.release_only_omega,
            reach_err_m: c.reach_err_m,
            sim_reach_m: c.sim_reach_m,
            obs_reach_m: c.obs_reach_m,
            sim_reach_raw_m: c.sim_reach_raw_m,
            sim_cells_raw: c.sim_cells_raw,
            omega_raw: c.omega_raw,
            release_clipped_frac: c.release_clipped_frac,
            release_clip_severe: c.release_clip_severe,
            connect_fallback: c.connect_fallback,
            sim_cells: c.sim_cells,
            ref_cells: c.ref_cells,
            release_cells: c.release_cells,
            release_volume_m3: c.release_volume_m3,
            mean_slab_m: c.mean_slab_m,
            steps: c.steps,
            max_velocity: c.max_velocity,
            sim_flags: c.sim_flags,
            clipped_at_edge: c.clipped_at_edge,
            seconds: c.seconds,
        }
    }
}

fn log_eval(case_name: &str, p: &Params, result: Option<&CaseResult>, err: Option<&str>) {
    let Some(m) = EVAL_LOG.get() else { return };
    let prov = PROVENANCE.get().cloned().unwrap_or_default();
    #[derive(Serialize)]
    struct EvalRec<'a> {
        stage: &'a str,
        candidate: &'a str,
        case: &'a str,
        iter: u64,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        err: Option<&'a str>,
        params: &'a Params,
        #[serde(skip_serializing_if = "Option::is_none")]
        m: Option<EvalMetrics>,
    }
    let rec = EvalRec {
        stage: &prov.stage,
        candidate: &prov.candidate,
        case: case_name,
        iter: next_iter(case_name),
        ok: result.is_some(),
        err,
        params: p,
        m: result.map(EvalMetrics::from),
    };
    let Ok(line) = serde_json::to_string(&rec) else {
        return;
    };
    let mut w = match m.lock() {
        Ok(w) => w,
        Err(e) => e.into_inner(),
    };
    let _ = writeln!(w, "{line}");
    let _ = w.flush();
}

// ---------------------------------------------------------------- case input

#[derive(Debug, Clone, Deserialize)]
struct CaseSpec {
    name: String,
    shp: String,
    #[serde(default, deserialize_with = "null_default")]
    area: f64,
    #[serde(default, deserialize_with = "null_default")]
    sze: i64,
    #[serde(default, deserialize_with = "null_default")]
    start_zone: f64,
    #[serde(default, deserialize_with = "null_default")]
    dpo_alt: f64,
    #[serde(default, deserialize_with = "null_default")]
    aspect: String,
    #[serde(default, deserialize_with = "null_default")]
    aval_shape: i64,
    #[serde(default, deserialize_with = "null_default")]
    split: String,
    /// per-case storm forcing, interpolated to this avalanche's location by
    /// `fit_weather.py`. Falls back to the global `--weather` file.
    #[serde(default)]
    wx: Option<Weather>,
}

/// `#[serde(default)]` covers an absent key but not an explicit JSON `null`,
/// which is what pool extraction writes for dead source columns (all of 1999's
/// start_zone/dpo_alt/aspect). Treat null as the default too.
fn null_default<'de, D, T>(de: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(de)?.unwrap_or_default())
}

/// Everything that only depends on terrain + observation, i.e. is constant
/// across parameter evaluations.
struct PreparedCase {
    spec: CaseSpec,
    dem: Dem,
    /// observed (rasterised) polygon, row-major, row 0 = southern edge
    reference: Vec<bool>,
    /// per-cell slope angle in degrees (Horn)
    slope: Vec<f32>,
    /// per-cell aspect: azimuth of steepest descent, degrees clockwise from north
    aspect: Vec<f32>,
    /// does this cell's D8 descent reach the outline's lower body? Terrain and
    /// observation only, so it is computed once here rather than per evaluation.
    drains: Vec<bool>,
    /// storm forcing at this avalanche's location
    wx: Weather,
    width: usize,
    height: usize,
}

// ------------------------------------------------------------------ forcing
//
// Snowpack forcing for the spatially varying release thickness. Every field is
// derived from IMIS station observations for the storm (see `fit_weather.py`);
// nothing here is invented. `hn_a` / `hn_b` describe the new-snow depth as a
// linear function of elevation, fitted across stations:
//
//     HN(z) [m] = hn_a + hn_b * (z - z_ref) / 100
//
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Weather {
    /// new-snow depth at the reference elevation, metres
    hn_a: f32,
    /// change in new-snow depth per 100 m of elevation, metres
    hn_b: f32,
    /// reference elevation for the fit, metres
    z_ref: f32,
    /// azimuth (deg from north, clockwise) that slopes must face to be
    /// wind-loaded, i.e. the downwind direction during the storm
    lee_azimuth: f32,
    /// new-snow density at release, kg/m3
    density: f32,
}

impl Default for Weather {
    fn default() -> Self {
        Self {
            hn_a: 1.0,
            hn_b: 0.0,
            z_ref: 2000.0,
            lee_azimuth: 135.0,
            density: 200.0,
        }
    }
}

/// Swiss-guideline slab-depth correction for slope angle (Salm, Burkard &
/// Gubler 1990): thinner slabs on steeper terrain, normalised to 1 at 28 deg.
fn slope_thinning(theta_deg: f32) -> f32 {
    if theta_deg <= 28.0 {
        return 1.0;
    }
    let t = theta_deg.to_radians();
    let denom = t.sin() - 0.202 * t.cos();
    if denom <= 1e-3 {
        return 1.0;
    }
    (0.291 / denom).clamp(0.15, 1.0)
}

// ------------------------------------------------------------------- params

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Params {
    friction_coefficient: f32,
    drag_coefficient: f32,
    slab_thickness: f32,
    /// fraction of the polygon's elevation range, measured down from its top,
    /// that is treated as the release zone
    release_band_frac: f32,
    /// peak flow thickness [m] above which a cell counts as "affected"
    flow_threshold: f32,
    /// slope band a cell must fall in to be part of the release area
    release_min_slope: f32,
    release_max_slope: f32,
    friction_model: u32,
    max_steps: u32,
    cfl: f32,
    released_particles_per_cell: u32,
    /// bit 0 curvature, 1 particle interaction, 2 earth pressure, 3 entrainment
    flags: u32,
    /// snow density at release, kg/m3
    density: f32,
    /// 0 = uniform `slab_thickness`; 1 = terrain- and weather-driven field;
    /// 2 = weather depth only, terrain corrections off (ablation)
    slab_mode: u32,
    /// multiplies the weather-driven field (dimensionless; 1.0 = take HN as is)
    slab_amp: f32,
    /// amplitude of the aspect-dependent wind-loading term, 0 = none
    slab_wind_amp: f32,
    /// 0 = constant `density`; 1 = the per-case measured storm-slab density
    density_mode: u32,
    /// Voellmy earth-pressure internal friction angle, degrees (SimSettings default 40.0)
    internal_friction_angle: f32,
    /// VRM roughness gate on release eligibility (SimSettings default 0.01)
    roughness_threshold: f32,
    /// particle stop / no-friction speed cutoff, m/s (SimSettings default 1e-6;
    /// read raw at compute_particles.wgsl:220,256 — the dt site floors it at 1e-3)
    velocity_threshold: f32,

    // ---- scoring conventions, NOT tunable parameters -----------------------
    // These three shape the footprint that gets scored, exactly as
    // `flow_threshold` does. They are carried in `Params` so that every logged
    // evaluation records the convention it was scored under; they must be held
    // fixed across a calibration campaign, never optimised.
    /// Minimum normalised residence for a cell to count as affected.
    /// `grid_cell_count` accumulates one unit per particle per timestep, so the
    /// raw count scales with both `released_particles_per_cell` and 1/`cfl`
    /// (smaller steps mean more of them). Normalising by
    /// `count * cfl / ppc` gives roughly "how many cell-equivalents of material
    /// passed through here", which is comparable across numerics settings --
    /// essential, because `cfl` and `ppc` are themselves being chosen in tier 0
    /// and a convention that moved with them would be circular. 0 disables.
    min_residence: f32,
    /// Drop parts of the footprint not 8-connected to the release area.
    /// Observed outlines are single mapped polygons; a detached blob is a
    /// contour artefact of the threshold, not something the observation could
    /// contain. 0 disables.
    require_release_connected: u32,
    /// Drop release cells whose D8 steepest-descent path never reaches the
    /// observed outline -- the polygon/DEM misalignment that puts release cells
    /// over a ridge, so mass launches into the neighbouring watershed. 0
    /// disables.
    clip_release_to_drainage: u32,
    /// depression-fill tolerance used when tracing release drainage, metres.
    /// Recorded for provenance; set from `--pit-tolerance` in `main`, which is
    /// also what preparation used, so the two cannot disagree.
    pit_fill_tolerance_m: f32,
}

impl Default for Params {
    fn default() -> Self {
        // repo defaults (SimSettings::new) + our evaluation-side defaults
        Self {
            friction_coefficient: 0.155,
            drag_coefficient: 4000.0,
            slab_thickness: 1.0,
            release_band_frac: 0.25,
            flow_threshold: 0.1,
            release_min_slope: 28.0,
            release_max_slope: 60.0,
            friction_model: 1, // Voellmy
            max_steps: 3000,
            cfl: 0.5,
            released_particles_per_cell: 8,
            flags: 0b0111, // curvature + particle interaction + earth pressure, no entrainment
            density: 200.0,
            slab_mode: 0,
            slab_amp: 1.0,
            slab_wind_amp: 0.0,
            density_mode: 0,
            internal_friction_angle: 40.0,
            roughness_threshold: 0.01,
            velocity_threshold: 1e-6,
            // Scoring conventions, on by default. 0.25 is measured, not
            // argued: swept over {0, 0.125, 0.25, 0.5, 1, 2, 4} on all 105
            // cases, it sits on the plateau (paired +0.0029 against no gate,
            // CI [+0.0022,+0.0036], better on 82/105). The 1.0 this replaced
            // was already past the plateau -- no mean benefit over 0 and up to
            // -0.169 on the worst case.
            min_residence: 0.25,
            require_release_connected: 1,
            clip_release_to_drainage: 1,
            pit_fill_tolerance_m: PIT_FILL_TOLERANCE_M,
        }
    }
}

// ------------------------------------------------------------------ results

#[derive(Debug, Clone, Serialize)]
struct Eval {
    alpha: f64,
    beta: f64,
    gamma: f64,
    omega: f64,
}
impl From<MassMovementEvaluation> for Eval {
    fn from(m: MassMovementEvaluation) -> Self {
        Eval {
            alpha: m.alpha,
            beta: m.beta,
            gamma: m.gamma,
            omega: m.omega,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct CaseResult {
    name: String,
    ref_cells: usize,
    sim_cells: usize,
    release_cells: usize,
    release_volume_m3: f64,
    grid_w: usize,
    grid_h: usize,
    steps: u32,
    max_velocity: f32,
    /// SimInfo flag bitmask (bit 0 = a particle left the domain)
    sim_flags: u32,
    /// simulated affected area touches the padded domain edge -> overshoot is clipped
    clipped_at_edge: bool,
    seconds: f64,
    area: Eval,
    /// Omega_T of the "a particle visited this cell" footprint. Identical to
    /// `area.omega` whenever particle interaction is off (that is already the
    /// footprint rule then); scored always so that candidates either side of
    /// the particle-interaction flag can be compared on one common metric.
    omega_cells: f64,
    hwri_l1: Eval,
    hwri_l05: Eval,
    /// score of the trivial "nothing moves past the release" model
    release_only_omega: f64,
    /// mean release thickness over the release cells [m]
    mean_slab_m: f64,
    /// observed / simulated runout length along the path axis [m]
    obs_runout_m: f32,
    sim_runout_m: f32,
    /// observed / simulated 90th-percentile width across the path axis [m]
    obs_width_m: f32,
    sim_width_m: f32,
    /// straight-line distance from the release apex to the furthest observed /
    /// simulated cell [m] -- "how far did it reach", the decision-relevant number
    obs_reach_m: f32,
    sim_reach_m: f32,
    /// simulated minus observed reach [m]; positive = the model ran too far
    reach_err_m: f32,

    // ---- what the scoring-convention filters removed ----------------------
    /// simulated cells before the residence and connectivity filters
    sim_cells_raw: usize,
    /// Omega_T of the unfiltered footprint. Recorded so a single run carries
    /// its own before/after for the tail filter -- the alternative, a paired
    /// run with the filters disabled, doubles the cost and invites the two
    /// halves to drift apart.
    omega_raw: f64,
    /// reach of the unfiltered footprint [m]; the difference against
    /// `sim_reach_m` is what the tail filter removed
    sim_reach_raw_m: f32,
    /// release cells dropped because their descent never reaches the outline
    release_clipped_frac: f64,
    /// the drainage clip removed more than half the release band
    release_clip_severe: bool,
    /// the connectivity filter found no release-connected seed, so it was
    /// skipped and the unfiltered footprint scored
    connect_fallback: bool,
    /// case metadata, carried through so downstream analysis can stratify
    sze: i64,
    aval_shape: i64,
    aspect: String,
    split: String,
}

#[derive(Debug, Clone, Serialize)]
struct RunResult {
    params: Params,
    mean_omega: f64,
    mean_hwri_l1: f64,
    /// mean Omega_T of the trivial "only the release area is affected" model
    mean_release_only: f64,
    cases: Vec<CaseResult>,
    /// cases that could not be evaluated at these parameters, with the reason
    failures: Vec<(String, String)>,
}

// --------------------------------------------------------------------- prep

/// Horn (1981) 3x3 slope and aspect. Row 0 is the southern edge, so +y is north.
/// Aspect is the compass azimuth the slope faces (direction of steepest descent),
/// degrees clockwise from north; flat cells get -1.
fn horn_slope_aspect(dem: &Dem) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (dem.width, dem.height);
    let cs = dem.cell_size;
    let mut slope = vec![0.0f32; w * h];
    let mut aspect = vec![-1.0f32; w * h];
    let at = |x: usize, y: usize| -> f32 { dem.data1d[y * w + x] };
    for y in 0..h {
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            let dzdx = ((at(xp, ym) + 2.0 * at(xp, y) + at(xp, yp))
                - (at(xm, ym) + 2.0 * at(xm, y) + at(xm, yp)))
                / (8.0 * cs);
            let dzdy = ((at(xm, yp) + 2.0 * at(x, yp) + at(xp, yp))
                - (at(xm, ym) + 2.0 * at(x, ym) + at(xp, ym)))
                / (8.0 * cs);
            let g2 = dzdx * dzdx + dzdy * dzdy;
            slope[y * w + x] = g2.sqrt().atan().to_degrees();
            if g2 > 1e-12 {
                // steepest descent points along -grad; azimuth measured clockwise
                // from north (+y), so atan2(east, north) of the descent vector.
                let a = (-dzdx).atan2(-dzdy).to_degrees();
                aspect[y * w + x] = if a < 0.0 { a + 360.0 } else { a };
            }
        }
    }
    (slope, aspect)
}

/// The eight neighbours, as (dy, dx).
const NB8: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// For every cell, does its D8 steepest-descent path reach the observed
/// outline?
///
/// This is the ridge/watershed leak test. The simulation domain is the outline
/// padded by 300 m, and the polygon is mapped from 1.5 m imagery onto a 5 m
/// DEM, so the upper edge of a release band can land on the far side of a
/// ridge crest. Mass released there runs down the wrong valley: it can never
/// intersect the observation, so it is pure overshoot penalty attributable to
/// misalignment rather than to physics.
///
/// Defined against the outline directly rather than against a clustered basin
/// id (as `campaign/analysis/crossing_analysis.py` does) because the clustering has
/// a failure mode this must not inherit: on `aval_6719` it labels 100% of the
/// release as "outside the intended drainage" while the case still scores
/// +0.337, which is a clustering artefact, not a real leak. "Does water from
/// here run into the observed polygon" needs no clustering and no threshold.
///
/// Depends only on terrain and the observation, so it is computed once per case
/// at preparation time and is free at evaluation time.
/// Descent-path length cap, as a multiple of the grid diagonal.
///
/// A cell whose water only reaches the outline after wandering further than the
/// domain is wide is not in the outline's drainage in any useful sense -- it has
/// gone down one valley and come back. Expressed against the case's own grid
/// rather than as an absolute step count because domains vary from roughly 100
/// to 400 cells across, so any fixed number would be lenient on small cases and
/// severe on large ones. A fixed scoring convention, not a tunable.
const DESCENT_STEP_CAP_DIAGONALS: f64 = 1.0;

/// Fraction of the outline's elevation range, measured up from its lowest
/// cell, that counts as the "body" a release must drain into.
///
/// The target cannot be the outline itself. `build_release` only ever draws
/// candidates from inside the observed polygon, so "does this cell drain to any
/// reference cell" is answered yes by the candidate's own membership and the
/// clip can never fire -- which is exactly the no-op this replaced. More
/// fundamentally, treating polygon membership as proof of drainage assumes the
/// polygon is ground truth in DEM space, and the entire premise of this fix is
/// that it is not: a 1.5 m-imagery outline laid on a 5 m DEM can cover cells
/// that sit past a crest.
///
/// So the target is the outline's *lower body*, which a release cell has to
/// reach by descending. Defined on the outline's own elevation range and so
/// independent of `release_band_frac` -- it stays precomputable once per case
/// even when the band is a search dimension. A fixed convention, not a tunable.
const CLIP_TARGET_BODY_FRAC: f32 = 0.5;

/// Depth below which a closed depression is filled before tracing descent,
/// metres. A fixed scoring convention.
///
/// D8 descent is a water droplet: it stops dead in any closed depression. A
/// moving avalanche is not a droplet -- it carries momentum and rides over
/// shallow hollows without noticing them. Measured on `aval_13722`, the case
/// this was added for: the pits terminating its release descent are **0.06 to
/// 0.51 m deep** on a 5 m DEM, which is quantisation and micro-relief rather
/// than terrain. Treating those as drainage boundaries clipped 98% of a release
/// whose cells were travelling only 80-230 m before stopping.
const PIT_FILL_TOLERANCE_M: f32 = 2.0;

/// Fill closed depressions shallower than `tol`, returning a modified elevation
/// field used **only** for descent tracing -- never for physics, which runs on
/// the untouched DEM.
///
/// Priority-flood (Barnes et al. 2014): flood inward from the domain edge in
/// elevation order, so each cell is first reached over its lowest spill route
/// and takes `max(own elevation, spill elevation)`. The depth limit is the
/// deviation from the standard algorithm: a cell that would need raising by
/// more than `tol` keeps its own elevation and stays a genuine sink, so real
/// basins still terminate a descent while micro-relief does not.
fn fill_shallow_pits(z: &[f32], w: usize, h: usize, tol: f32) -> Vec<f32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = w * h;
    if tol <= 0.0 {
        return z.to_vec();
    }
    // Integer millimetre keys: f32 is not Ord, and a new dependency for one
    // comparison is not worth it. Alpine elevations are ~4.5e6 mm, far inside i64.
    let key = |v: f32| -> i64 { (v * 1000.0) as i64 };

    let mut filled = vec![f32::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
    for y in 0..h {
        for x in 0..w {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                let i = y * w + x;
                filled[i] = z[i];
                heap.push(Reverse((key(z[i]), i)));
            }
        }
    }
    while let Some(Reverse((_, c))) = heap.pop() {
        let (cy, cx) = (c / w, c % w);
        for (dy, dx) in NB8 {
            let (ny, nx) = (cy as i32 + dy, cx as i32 + dx);
            if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if filled[j].is_finite() {
                continue;
            }
            // The +EPS is not cosmetic. Filling a depression exactly to its
            // spill elevation creates a flat, and D8 cannot route across a
            // flat -- every cell would find no strictly-lower neighbour and
            // the descent would still die, just at the rim instead of the
            // bottom. The epsilon gradient gives filled cells a downhill exit.
            const EPS: f32 = 1e-3;
            let spill = (filled[c] + EPS).max(z[j]);
            filled[j] = if spill - z[j] <= tol { spill } else { z[j] };
            heap.push(Reverse((key(filled[j]), j)));
        }
    }
    filled
}

/// The lower part of the observed outline: the target a release must drain into.
fn outline_body(dem: &Dem, reference: &[bool]) -> Vec<bool> {
    let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for (i, &r) in reference.iter().enumerate() {
        if r {
            zmin = zmin.min(dem.data1d[i]);
            zmax = zmax.max(dem.data1d[i]);
        }
    }
    if !zmin.is_finite() {
        return vec![false; reference.len()];
    }
    let cut = zmin + CLIP_TARGET_BODY_FRAC * (zmax - zmin);
    reference
        .iter()
        .enumerate()
        .map(|(i, &r)| r && dem.data1d[i] <= cut)
        .collect()
}

/// Steps along the D8 descent path from each cell to the target set;
/// `u32::MAX` where the path never reaches it (pit, cycle on a flat, or a
/// terminus elsewhere).
///
/// Exact and O(n): the descent graph is functional -- every cell has exactly one
/// receiver -- so the distance to the target is well defined and memoises,
/// with each cell resolved once regardless of how long its path is.
fn descent_steps_to_target(z: &[f32], cs: f32, target: &[bool], w: usize, h: usize) -> Vec<u32> {
    let n = w * h;

    // Steepest-descent receiver per cell, by drop per unit distance so that
    // diagonals are not favoured by their longer reach.
    let mut recv = vec![usize::MAX; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let zc = z[i];
            let mut best = 0.0f32;
            for (dy, dx) in NB8 {
                let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                let dist = cs * ((dy * dy + dx * dx) as f32).sqrt();
                let slope = (zc - z[j]) / dist;
                if slope > best {
                    best = slope;
                    recv[i] = j;
                }
            }
        }
    }

    const UNKNOWN: u8 = 0;
    const DONE: u8 = 1;
    const WALKING: u8 = 2;
    const NEVER: u32 = u32::MAX;

    let mut state = vec![UNKNOWN; n];
    let mut steps = vec![NEVER; n];
    // ONLY target cells terminate a walk. A reference cell above the body must
    // descend to the body like any other cell -- auto-passing it is what made
    // the previous version tautological, since every release candidate is a
    // reference cell by construction.
    for i in 0..n {
        if target[i] {
            state[i] = DONE;
            steps[i] = 0;
        }
    }

    let mut path: Vec<usize> = Vec::new();
    for start in 0..n {
        if state[start] != UNKNOWN {
            continue;
        }
        path.clear();
        let mut cur = start;
        let terminus = loop {
            match state[cur] {
                DONE => break steps[cur],
                // A cycle can only form on a flat or in a pit; either way the
                // descent terminates without reaching the outline.
                WALKING => break NEVER,
                _ => {}
            }
            state[cur] = WALKING;
            path.push(cur);
            match recv[cur] {
                usize::MAX => break NEVER, // pit
                nxt => cur = nxt,
            }
        };
        // Unwind: the last cell pushed is one step from the terminus.
        let mut s = terminus;
        for &c in path.iter().rev() {
            s = if s == NEVER { NEVER } else { s + 1 };
            steps[c] = s;
            state[c] = DONE;
        }
    }
    steps
}

/// Does each cell's descent reach the outline's lower body within the step cap?
fn drains_to_body(
    dem: &Dem,
    reference: &[bool],
    w: usize,
    h: usize,
    pit_tol: f32,
) -> (Vec<bool>, u32) {
    // The target is defined on real elevations -- it is a statement about the
    // observed outline. Only the descent runs on the pit-filled field.
    let target = outline_body(dem, reference);
    let z = fill_shallow_pits(&dem.data1d, w, h, pit_tol);
    let steps = descent_steps_to_target(&z, dem.cell_size, &target, w, h);
    let diag = ((w * w + h * h) as f64).sqrt();
    let cap = (diag * DESCENT_STEP_CAP_DIAGONALS).ceil() as u32;
    (steps.iter().map(|&s| s <= cap).collect(), cap)
}

/// Has enough material passed through this cell to count as affected?
///
/// `grid_cell_count` accumulates one unit per particle per timestep, so the raw
/// count scales with `released_particles_per_cell` and with 1/`cfl` (smaller
/// steps mean more of them). Both are chosen in tier 0, so the gate is applied
/// to `count * cfl / ppc` -- roughly the number of cell-equivalents of material
/// that passed through -- which keeps the scoring convention from drifting with
/// the numerics it is meant to be independent of.
fn resident(cell_count: &[u32], p: &Params, i: usize) -> bool {
    if p.min_residence <= 0.0 {
        return true;
    }
    let ppc = p.released_particles_per_cell.max(1) as f32;
    (cell_count[i] as f32) * p.cfl / ppc >= p.min_residence
}

/// Keep only the part of `mask` 8-connected to the release area.
///
/// `campaign/analysis/frag_analysis.py` measured 105/105 cases where the largest
/// component touches the release, and a median of 9 components per case whose
/// detached remainder is 0.37% of the footprint by area. Those fragments are a
/// contour artefact of the 0.1 m threshold -- no fragment is reachable by
/// descent from any release cell -- and an observed outline, being one mapped
/// polygon, cannot contain them.
///
/// Returns `mask` unchanged if no seed exists, rather than returning an empty
/// footprint: an empty simulated area scores Omega_T = -1 and would look like a
/// catastrophic physics failure when it is really a filter with nothing to
/// hold on to.
fn keep_release_connected(
    mask: &[bool],
    release: &[bool],
    w: usize,
    h: usize,
) -> (Vec<bool>, bool) {
    let n = w * h;
    let mut keep = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();

    // Seed from masked cells that are release cells or touch one. The second
    // case matters when a high residence threshold has already removed the
    // release cells themselves from the footprint.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] || keep[i] {
                continue;
            }
            let mut seed = release[i];
            if !seed {
                for (dy, dx) in NB8 {
                    let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                    if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                        continue;
                    }
                    if release[ny as usize * w + nx as usize] {
                        seed = true;
                        break;
                    }
                }
            }
            if seed {
                keep[i] = true;
                stack.push(i);
            }
        }
    }
    if stack.is_empty() {
        return (mask.to_vec(), true);
    }

    while let Some(i) = stack.pop() {
        let (y, x) = (i / w, i % w);
        for (dy, dx) in NB8 {
            let (ny, nx) = (y as i32 + dy, x as i32 + dx);
            if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if mask[j] && !keep[j] {
                keep[j] = true;
                stack.push(j);
            }
        }
    }
    (keep, false)
}

async fn prepare_case(
    tm: &TileManager,
    spec: &CaseSpec,
    padding_m: f64,
    global_wx: &Weather,
    pit_tol: f32,
) -> Result<PreparedCase> {
    let poly = read_shapefile_nth_polygon(&spec.shp, 0)
        .with_context(|| format!("reading {}", spec.shp))?;

    // padded footprint -> DEM request
    let mut padded = RasterGrid::from_polygon(&poly, CELL);
    padded.add_padding(padding_m)?;
    let bbox = BBox {
        min_easting: padded.origin_x as u32,
        max_easting: (padded.origin_x + padded.width as f64 * padded.cell_size) as u32,
        min_northing: padded.origin_y as u32,
        max_northing: (padded.origin_y + padded.height as f64 * padded.cell_size) as u32,
    };
    let dem = tm.get_dem(&bbox).await?;
    if dem.data1d.iter().any(|v| v.is_nan()) {
        bail!(
            "case {}: DEM contains NaN (missing swissALTI3D coverage)",
            spec.name
        );
    }

    // observed polygon rasterised onto exactly the DEM grid
    let outline = RasterGrid::from_polygon(&poly, CELL);
    let ref_u8 = outline.to_padded_tile(
        dem.bounds.xmin as f64,
        dem.bounds.ymin as f64,
        dem.width,
        dem.height,
    );
    let reference: Vec<bool> = ref_u8.iter().map(|&v| v == 1).collect();
    if !reference.iter().any(|&b| b) {
        bail!("case {}: rasterised reference is empty", spec.name);
    }

    let (slope, aspect) = horn_slope_aspect(&dem);
    let (drains, _step_cap) = drains_to_body(&dem, &reference, dem.width, dem.height, pit_tol);
    Ok(PreparedCase {
        wx: spec.wx.unwrap_or(*global_wx),
        spec: spec.clone(),
        width: dem.width,
        height: dem.height,
        dem,
        reference,
        slope,
        aspect,
        drains,
    })
}

/// Cells below this release thickness are dropped. The kernel
/// (`initialize_particles.wgsl`) releases from anything above 0.01 m while
/// `Simulation::set_release_areas` sizes the particle buffer from the same
/// cutoff, so this only has to stay clear of that boundary; 2 cm of snow
/// carries no meaningful mass anyway.
const MIN_SLAB: f32 = 0.02;

/// Release area: the upper `release_band_frac` of the observed polygon's
/// elevation range, restricted to the model's own release slope band.
///
/// In `slab_mode = 0` every release cell gets the same `slab_thickness`.
/// In `slab_mode = 1` the thickness is a field,
///
///   d(x) = slab_amp * HN(z) * f_slope(theta) * (1 + wind_amp * cos(aspect - lee))
///
/// with HN(z) the elevation-dependent new-snow depth measured at the IMIS
/// stations, f_slope the Swiss-guideline slope thinning, and the last factor a
/// crude wind-loading term that thickens lee slopes and thins windward ones.
struct ReleaseBand {
    thickness: Vec<f32>,
    cells: usize,
    apex: (usize, usize),
    /// share of otherwise-eligible release cells removed by the drainage clip
    clipped_frac: f64,
    /// the clip removed more than `CLIP_SEVERE_FRAC` of the release. Not an
    /// error, but it means the misregistration is worse than a single edge
    /// straying over a crest, and the case is worth looking at by hand.
    clip_severe: bool,
    /// the clip removed every eligible cell, or reduced a viable release below
    /// `MIN_RELEASE_CELLS`. The caller must refuse to simulate this rather than
    /// score a stub release.
    clip_refused: bool,
}

/// Above this share of the release removed, the case is flagged. Beyond a
/// ridge-crest sliver this is whole-polygon misregistration. A fixed reporting
/// convention, not a tunable.
const CLIP_SEVERE_FRAC: f64 = 0.5;

/// If the drainage clip takes a release that had at least this many cells down
/// below it, the case is refused outright: a handful of surviving cells (the
/// measured worst case kept 4 of 183) is not a simulation anyone should trust,
/// but it is not literally empty, so the emptiness check alone misses it. The
/// floor deliberately applies only to clipped cases — a release that was
/// naturally this small before the clip is pre-existing behaviour and stays. A
/// fixed convention, not a tunable; it generalises to the 20k harvest where
/// named case exclusions cannot.
const MIN_RELEASE_CELLS: usize = 20;

fn build_release(case: &PreparedCase, p: &Params) -> ReleaseBand {
    let wx = &case.wx;
    let n = case.width * case.height;
    let mut zmin = f32::INFINITY;
    let mut zmax = f32::NEG_INFINITY;
    for i in 0..n {
        if case.reference[i] {
            let z = case.dem.data1d[i];
            zmin = zmin.min(z);
            zmax = zmax.max(z);
        }
    }
    let z_cut = zmax - p.release_band_frac * (zmax - zmin);

    // Candidates first, drainage clip second: the fallback needs to know
    // whether the clip would empty the band before it is applied.
    let mut cands: Vec<(usize, f32)> = Vec::new();
    for i in 0..n {
        if !case.reference[i] {
            continue;
        }
        let z = case.dem.data1d[i];
        let s = case.slope[i];
        if !(z >= z_cut && (p.release_min_slope..=p.release_max_slope).contains(&s)) {
            continue;
        }
        let d = if p.slab_mode == 0 {
            p.slab_thickness
        } else if p.slab_mode == 2 {
            // ablation: the locally measured storm-slab depth, with the slope
            // thinning and wind terms switched off, to separate "local weather"
            // from "terrain correction".
            p.slab_amp * (wx.hn_a + wx.hn_b * (z - wx.z_ref) / 100.0).max(0.0)
        } else {
            let hn = (wx.hn_a + wx.hn_b * (z - wx.z_ref) / 100.0).max(0.0);
            let a = case.aspect[i];
            let wind = if a < 0.0 || p.slab_wind_amp == 0.0 {
                1.0
            } else {
                1.0 + p.slab_wind_amp * ((a - wx.lee_azimuth).to_radians().cos())
            };
            p.slab_amp * hn * slope_thinning(s) * wind.max(0.0)
        };
        if d < MIN_SLAB {
            continue;
        }
        cands.push((i, d));
    }

    let n_cand = cands.len();
    let clip = p.clip_release_to_drainage != 0;
    let kept: Vec<(usize, f32)> = if clip {
        cands.into_iter().filter(|&(i, _)| case.drains[i]).collect()
    } else {
        cands
    };
    let dropped = (n_cand - kept.len()) as f64 / n_cand.max(1) as f64;
    // Silently falling back to the unclipped band would hide exactly the cases
    // most worth seeing, and scoring an empty release would read as a physics
    // failure when it is misregistration. So: flag a severe clip, and refuse
    // outright when nothing survives.
    let clip_refused = clip
        && n_cand > 0
        && (kept.is_empty() || (n_cand >= MIN_RELEASE_CELLS && kept.len() < MIN_RELEASE_CELLS));
    let clip_severe = clip && n_cand > 0 && dropped > CLIP_SEVERE_FRAC;
    if clip_severe && !clip_refused {
        eprintln!(
            "  !! {}: drainage clip removed {:.0}% of the release ({} of {} cells) -- \
             polygon/DEM misregistration beyond a ridge crest",
            case.spec.name,
            dropped * 100.0,
            n_cand - kept.len(),
            n_cand
        );
    }

    let mut thickness = vec![0.0f32; n];
    let mut apex = (0usize, 0usize);
    let mut apex_z = f32::NEG_INFINITY;
    for &(i, d) in &kept {
        thickness[i] = d;
        let z = case.dem.data1d[i];
        if z > apex_z {
            apex_z = z;
            apex = (i / case.width, i % case.width);
        }
    }
    ReleaseBand {
        cells: kept.len(),
        clipped_frac: if n_cand > 0 { dropped } else { 0.0 },
        clip_severe,
        clip_refused,
        thickness,
        apex,
    }
}

/// Runout length and width of a footprint, measured in the case's own path
/// frame: the axis runs from the release apex to the lowest observed cell.
/// Returns (length along axis, 90th-percentile half-width x2), both in metres.
fn path_geometry(
    mask: &[bool],
    w: usize,
    cell: f32,
    apex: (usize, usize),
    axis: (f32, f32),
) -> (f32, f32) {
    let (ar, ac) = (apex.0 as f32, apex.1 as f32);
    let mut len = 0.0f32;
    let mut widths: Vec<f32> = Vec::new();
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            continue;
        }
        let dr = (i / w) as f32 - ar;
        let dc = (i % w) as f32 - ac;
        let along = dr * axis.0 + dc * axis.1;
        let across = (dr * axis.1 - dc * axis.0).abs();
        if along > len {
            len = along;
        }
        if along > 0.0 {
            widths.push(across);
        }
    }
    widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = if widths.is_empty() {
        0.0
    } else {
        widths[((widths.len() as f32 * 0.9) as usize).min(widths.len() - 1)]
    };
    (len * cell, 2.0 * p90 * cell)
}

/// Straight-line distance from the release apex to the furthest cell of a mask,
/// in metres. Unlike `path_geometry` this needs no axis, so it is well defined
/// even where the path bends; it is the number a reader reasons about when
/// asking "how far did the avalanche run?".
fn max_reach(mask: &[bool], w: usize, cell: f32, apex: (usize, usize)) -> f32 {
    let (ar, ac) = (apex.0 as f32, apex.1 as f32);
    let mut best = 0.0f32;
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            continue;
        }
        let dr = (i / w) as f32 - ar;
        let dc = (i % w) as f32 - ac;
        let d = (dr * dr + dc * dc).sqrt();
        if d > best {
            best = d;
        }
    }
    best * cell
}

/// Unit vector from the apex towards the lowest cell of the observed outline,
/// in (row, col) index space.
fn path_axis(case: &PreparedCase, apex: (usize, usize)) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut toe = apex;
    for i in 0..case.width * case.height {
        if case.reference[i] && case.dem.data1d[i] < lo {
            lo = case.dem.data1d[i];
            toe = (i / case.width, i % case.width);
        }
    }
    let dr = toe.0 as f32 - apex.0 as f32;
    let dc = toe.1 as f32 - apex.1 as f32;
    let m = (dr * dr + dc * dc).sqrt();
    if m < 1e-6 {
        (1.0, 0.0)
    } else {
        (dr / m, dc / m)
    }
}

// ------------------------------------------------------------------ scoring

fn to_rows(flat: &[bool], w: usize, h: usize) -> Vec<Vec<bool>> {
    (0..h).map(|y| flat[y * w..(y + 1) * w].to_vec()).collect()
}

/// Evaluate one case and record the result. Every simulation in every
/// subcommand funnels through here, so this is the single place that writes
/// the evaluation log -- the alternative, logging at each call site, is what
/// left `sweep`/`search`/`grid` silently unrecorded before.
async fn evaluate_case(
    sim: &mut Simulation,
    case: &PreparedCase,
    p: &Params,
) -> Result<CaseResult> {
    let r = evaluate_case_inner(sim, case, p).await;
    match &r {
        Ok(c) => log_eval(&case.spec.name, p, Some(c), None),
        Err(e) => log_eval(&case.spec.name, p, None, Some(&e.to_string())),
    }
    r
}

async fn evaluate_case_inner(
    sim: &mut Simulation,
    case: &PreparedCase,
    p: &Params,
) -> Result<CaseResult> {
    let t0 = Instant::now();
    let band = build_release(case, p);
    let (release, release_cells, apex) = (&band.thickness, band.cells, band.apex);
    if band.clip_refused {
        bail!(
            "case {}: the drainage clip left only {} release cells ({:.0}% removed; \
             refusal floor {}) -- the release band does not credibly drain into the \
             observed outline. This is polygon/DEM misregistration, not a parameter \
             that can be fitted; the case must be excluded rather than scored.",
            case.spec.name,
            band.cells,
            band.clipped_frac * 100.0,
            MIN_RELEASE_CELLS
        );
    }
    if release_cells == 0 {
        bail!("case {}: empty release area", case.spec.name);
    }

    sim.set_dem_with_bounds(
        &case.dem.data1d,
        case.width,
        case.height,
        case.dem.cell_size,
        case.dem.bounds.xmin,
        case.dem.bounds.xmax,
        case.dem.bounds.ymin,
        case.dem.bounds.ymax,
        1.0,
    )?;
    sim.settings.friction_model = p.friction_model;
    sim.settings.friction_coefficient = p.friction_coefficient;
    sim.settings.drag_coefficient = p.drag_coefficient;
    // NB: only the PNG release path multiplies the texture by this; when the
    // release areas come from an array (as here) the per-cell values are used
    // verbatim. Set it anyway so dumped settings are self-consistent.
    sim.settings.slab_thickness = p.slab_thickness;
    sim.settings.density = if p.density_mode == 1 {
        case.wx.density
    } else {
        p.density
    };
    sim.settings.max_steps = p.max_steps;
    sim.settings.cfl = p.cfl;
    sim.settings.released_particles_per_cell = p.released_particles_per_cell;
    sim.settings.flags = p.flags;
    sim.settings.internal_friction_angle = p.internal_friction_angle;
    sim.settings.roughness_threshold = p.roughness_threshold;
    sim.settings.velocity_threshold = p.velocity_threshold;
    sim.set_release_areas(release)?;
    sim.run().await?;

    let info = sim.fetch_sim_info().await?;
    let peak_h = sim.fetch_peak_flow_thickness().await?;
    let cell_count = sim.fetch_cell_count().await?.clone();

    // "affected" = peak flow thickness above threshold. peak flow thickness is
    // only produced by the grid pass, which needs particle interaction; without
    // it fall back to "a particle visited this cell".
    let use_h = (p.flags & 0b10) != 0;
    let rel_bool: Vec<bool> = release.iter().map(|&t| t > 0.0).collect();

    // Residence gate, applied to both footprint rules. `grid_cell_count` counts
    // one per particle per timestep, so a cell a single particle rolled through
    // holds a handful of counts while a cell the body flowed over holds
    // thousands. Normalising by cfl/ppc keeps the gate comparable across the
    // numerics settings tier 0 is choosing between.
    // One helper for every footprint source. The particle-interaction flag
    // switches which source is scored, so if the filters were applied to only
    // one of them -- or applied differently -- a stage-1 comparison across that
    // flag would be comparing a filtered objective against an unfiltered one and
    // attributing the difference to physics.
    let apply_filters = |raw: &[bool]| -> (Vec<bool>, bool) {
        let gated: Vec<bool> = raw
            .iter()
            .enumerate()
            .map(|(i, &m)| m && resident(&cell_count, p, i))
            .collect();
        if p.require_release_connected != 0 {
            keep_release_connected(&gated, &rel_bool, case.width, case.height)
        } else {
            (gated, false)
        }
    };

    let raw_h: Vec<bool> = peak_h.iter().map(|&h| h >= p.flow_threshold).collect();
    let raw_c: Vec<bool> = cell_count.iter().map(|&c| c > 0).collect();
    let (foot_h, fb_h) = apply_filters(&raw_h);
    let (foot_c, fb_c) = apply_filters(&raw_c);
    let connect_fallback = if use_h { fb_h } else { fb_c };

    let simulated: Vec<bool> = if use_h {
        foot_h.clone()
    } else {
        foot_c.clone()
    };
    // The unfiltered footprint under the same rule, kept only to measure what
    // the filters removed. This is the before/after evidence for the tail fix,
    // recorded per evaluation rather than reconstructed later from rasters.
    let raw_simulated: Vec<bool> = if use_h { raw_h } else { raw_c };

    // does the simulated body reach the padded domain edge? then the overshoot
    // penalty is clipped by the domain rather than by physics.
    let mut edge_touched = false;
    for y in 0..case.height {
        for x in 0..case.width {
            if simulated[y * case.width + x]
                && (x < 2 || y < 2 || x + 2 >= case.width || y + 2 >= case.height)
            {
                edge_touched = true;
            }
        }
    }

    let refr = to_rows(&case.reference, case.width, case.height);
    let simr = to_rows(&simulated, case.width, case.height);
    let area = evaluate_mass_movement_area(&refr, &simr).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // The same footprint under the threshold-free rule. Turning particle
    // interaction off switches `simulated` from "peak thickness >= threshold"
    // to "a particle visited", so a comparison across that flag is otherwise
    // comparing two different definitions of "affected" as well as two
    // physics configurations. Scoring both costs one extra pass over the grid
    // against a GPU simulation, i.e. nothing.
    // The footprint as it would have been scored without the tail filters.
    let rawr = to_rows(&raw_simulated, case.width, case.height);
    let area_raw =
        evaluate_mass_movement_area(&refr, &rawr).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let visr = to_rows(&foot_c, case.width, case.height);
    let area_cells =
        evaluate_mass_movement_area(&refr, &visr).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let hwri1 = evaluate_distance_weighted_mass_movement_runout(&refr, &simr, apex, 1.0)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let hwri05 = evaluate_distance_weighted_mass_movement_runout(&refr, &simr, apex, 0.5)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // null model: only the release area is affected
    let rel_rows = to_rows(&rel_bool, case.width, case.height);
    let release_only =
        evaluate_mass_movement_area(&refr, &rel_rows).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // path-frame geometry: how much of the overshoot is length, how much width
    let axis = path_axis(case, apex);
    let cs = case.dem.cell_size;
    let (obs_runout_m, obs_width_m) = path_geometry(&case.reference, case.width, cs, apex, axis);
    let (sim_runout_m, sim_width_m) = path_geometry(&simulated, case.width, cs, apex, axis);
    let obs_reach_m = max_reach(&case.reference, case.width, cs, apex);
    let sim_reach_m = max_reach(&simulated, case.width, cs, apex);
    // Reach is a max over the footprint, so a single surviving trace cell sets
    // it. That makes reach far more sensitive to the tail filter than Omega_T,
    // which only sees the handful of cells as area -- and reach is the quantity
    // the "runouts read too long" complaint is actually about.
    let sim_reach_raw_m = max_reach(&raw_simulated, case.width, cs, apex);

    let slab_sum: f64 = release
        .iter()
        .filter(|&&t| t > 0.0)
        .map(|&t| t as f64)
        .sum();

    Ok(CaseResult {
        name: case.spec.name.clone(),
        ref_cells: case.reference.iter().filter(|&&b| b).count(),
        sim_cells: simulated.iter().filter(|&&b| b).count(),
        release_cells,
        release_volume_m3: CELL * CELL * slab_sum,
        grid_w: case.width,
        grid_h: case.height,
        steps: info.timestep,
        max_velocity: info.max_velocity,
        sim_flags: info.flags,
        clipped_at_edge: edge_touched,
        seconds: t0.elapsed().as_secs_f64(),
        area: area.into(),
        omega_cells: area_cells.omega,
        hwri_l1: hwri1.into(),
        hwri_l05: hwri05.into(),
        release_only_omega: release_only.omega,
        mean_slab_m: slab_sum / release_cells as f64,
        obs_runout_m,
        sim_runout_m,
        obs_width_m,
        sim_width_m,
        obs_reach_m,
        sim_reach_m,
        reach_err_m: sim_reach_m - obs_reach_m,
        sim_cells_raw: raw_simulated.iter().filter(|&&b| b).count(),
        omega_raw: area_raw.omega,
        sim_reach_raw_m,
        release_clipped_frac: band.clipped_frac,
        release_clip_severe: band.clip_severe,
        connect_fallback,
        sze: case.spec.sze,
        aval_shape: case.spec.aval_shape,
        aspect: case.spec.aspect.clone(),
        split: case.spec.split.clone(),
    })
}

/// Evaluate every case. A case that fails at these parameters (degenerate
/// release area, solver error) is recorded in `failures` rather than aborting
/// the whole run: with a hundred cases a single bad one must not throw away the
/// other ninety-nine, and a silently dropped case is worse than a logged one.
async fn evaluate_all(
    sim: &mut Simulation,
    cases: &[PreparedCase],
    p: &Params,
) -> Result<RunResult> {
    let mut out = Vec::new();
    let mut failures = Vec::new();
    for c in cases {
        match evaluate_case(sim, c, p).await {
            Ok(r) => out.push(r),
            Err(e) => {
                eprintln!("  !! {} failed: {e}", c.spec.name);
                failures.push((c.spec.name.clone(), format!("{e}")));
            }
        }
    }
    if out.is_empty() {
        bail!("every case failed");
    }
    let mean_omega = out.iter().map(|c| c.area.omega).sum::<f64>() / out.len() as f64;
    let mean_hwri_l1 = out.iter().map(|c| c.hwri_l1.omega).sum::<f64>() / out.len() as f64;
    let mean_release_only =
        out.iter().map(|c| c.release_only_omega).sum::<f64>() / out.len() as f64;
    Ok(RunResult {
        params: *p,
        mean_omega,
        mean_hwri_l1,
        mean_release_only,
        cases: out,
        failures,
    })
}

// ------------------------------------------------------------ search (N-M)

#[derive(Clone, Copy, Debug)]
struct Bound {
    lo: f64,
    hi: f64,
    log: bool,
}

const NDIM: usize = 10;
const DIM_NAMES: [&str; NDIM] = [
    "mu",
    "xi",
    "slab",
    "band",
    "amp",
    "wind",
    "density",
    "ifa",
    "roughness",
    "entrain",
];

/// The knobs the simplex may move, in its own coordinate order. `entrain` is
/// a continuous [0,1] relaxation of the entrainment bit -- >=0.5 sets it --
/// so full-parameter per-event search can explore it with the same
/// Nelder-Mead machinery as everything else.
const BOUNDS: [Bound; NDIM] = [
    Bound {
        lo: 0.05,
        hi: 0.60,
        log: false,
    }, // mu
    Bound {
        lo: 200.0,
        hi: 12000.0,
        log: true,
    }, // xi
    Bound {
        lo: 0.10,
        hi: 2.00,
        log: false,
    }, // slab thickness [m]
    Bound {
        lo: 0.10,
        hi: 0.60,
        log: false,
    }, // release band fraction
    Bound {
        lo: 0.05,
        hi: 2.00,
        log: false,
    }, // slab_amp (weather field)
    Bound {
        lo: 0.00,
        hi: 0.90,
        log: false,
    }, // wind loading amplitude
    Bound {
        lo: 100.0,
        hi: 400.0,
        log: false,
    }, // density [kg/m3]
    Bound {
        lo: 15.0,
        hi: 45.0,
        log: false,
    }, // internal friction angle [deg]
    Bound {
        lo: 0.002,
        hi: 0.05,
        log: true,
    }, // roughness_threshold
    Bound {
        lo: 0.0,
        hi: 1.0,
        log: false,
    }, // entrain (thresholded at 0.5)
];

type Vec6 = [f64; NDIM];

fn to_unit(x: Vec6) -> Vec6 {
    let mut u = [0.0; NDIM];
    for i in 0..NDIM {
        let b = BOUNDS[i];
        u[i] = if b.log {
            (x[i].ln() - b.lo.ln()) / (b.hi.ln() - b.lo.ln())
        } else {
            (x[i] - b.lo) / (b.hi - b.lo)
        };
    }
    u
}

fn from_unit(u: Vec6) -> Vec6 {
    let mut x = [0.0; NDIM];
    for i in 0..NDIM {
        let b = BOUNDS[i];
        let t = u[i].clamp(0.0, 1.0);
        x[i] = if b.log {
            (b.lo.ln() + t * (b.hi.ln() - b.lo.ln())).exp()
        } else {
            b.lo + t * (b.hi - b.lo)
        };
    }
    x
}

fn params_from(base: &Params, x: Vec6) -> Params {
    let mut p = *base;
    p.friction_coefficient = x[0] as f32;
    p.drag_coefficient = x[1] as f32;
    p.slab_thickness = x[2] as f32;
    p.release_band_frac = x[3] as f32;
    p.slab_amp = x[4] as f32;
    p.slab_wind_amp = x[5] as f32;
    p.density = x[6] as f32;
    p.internal_friction_angle = x[7] as f32;
    p.roughness_threshold = x[8] as f32;
    p.flags = if x[9] >= 0.5 {
        p.flags | 0b1000
    } else {
        p.flags & !0b1000
    };
    p
}

fn params_vec(p: &Params) -> Vec6 {
    [
        p.friction_coefficient as f64,
        p.drag_coefficient as f64,
        p.slab_thickness as f64,
        p.release_band_frac as f64,
        p.slab_amp as f64,
        p.slab_wind_amp as f64,
        p.density as f64,
        p.internal_friction_angle as f64,
        p.roughness_threshold as f64,
        if p.flags & 0b1000 != 0 { 1.0 } else { 0.0 },
    ]
}

/// One objective evaluation: mean Omega_T over the calibration cases, negated
/// so that the Nelder-Mead simplex minimises it.
#[allow(clippy::too_many_arguments)]
async fn objective(
    p0: &Params,
    u: Vec6,
    sim: &mut Simulation,
    prepared: &[PreparedCase],
    history: &mut Vec<(Params, f64, f64)>,
    evals: &mut usize,
    quiet: bool,
) -> Result<f64> {
    let p = params_from(p0, from_unit(u));
    // No logging here: `evaluate_case` already records one row per simulation,
    // which for a multi-case objective is strictly more information than the
    // one aggregate row this used to write.
    let r = match evaluate_all(sim, prepared, &p).await {
        Ok(r) => r,
        Err(e) => {
            *evals += 1;
            if !quiet {
                eprintln!("[{:3}] {p:?} -> unusable ({e})", *evals);
            }
            return Ok(2.0); // worse than any attainable -Omega_T
        }
    };
    *evals += 1;
    if !quiet {
        eprintln!(
            "[{:3}] mu={:.4} xi={:7.0} h={:.3} band={:.3} amp={:.3} wind={:.2} -> omega={:+.4} hwri={:+.4} (null {:+.3})",
            *evals,
            p.friction_coefficient,
            p.drag_coefficient,
            p.slab_thickness,
            p.release_band_frac,
            p.slab_amp,
            p.slab_wind_amp,
            r.mean_omega,
            r.mean_hwri_l1,
            r.mean_release_only
        );
    }
    history.push((p, r.mean_omega, r.mean_hwri_l1));
    Ok(-r.mean_omega)
}

/// Bounded Nelder-Mead in the unit cube over the `free` subspace. Returns the
/// evaluation history, best first.
#[allow(clippy::too_many_arguments)]
async fn nelder_mead(
    p0: &Params,
    free: [bool; NDIM],
    budget: usize,
    sim: &mut Simulation,
    prepared: &[PreparedCase],
    quiet: bool,
) -> Result<(Vec<(Params, f64, f64)>, usize, String)> {
    let u0 = to_unit(params_vec(p0));
    let mut history: Vec<(Params, f64, f64)> = Vec::new();
    let mut evals = 0usize;

    let mut simplex: Vec<Vec6> = vec![u0];
    for i in 0..NDIM {
        if !free[i] {
            continue;
        }
        let mut u = u0;
        u[i] = if u[i] < 0.5 { u[i] + 0.25 } else { u[i] - 0.25 };
        simplex.push(u);
    }
    let mut fvals = Vec::new();
    for u in &simplex {
        fvals.push(objective(p0, *u, sim, prepared, &mut history, &mut evals, quiet).await?);
    }

    let mut why = "budget exhausted".to_string();
    while evals < budget {
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        simplex = order.iter().map(|&i| simplex[i]).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        let mut c = [0.0; NDIM];
        for u in &simplex[..simplex.len() - 1] {
            for i in 0..NDIM {
                c[i] += u[i] / (simplex.len() - 1) as f64;
            }
        }
        let worst = *simplex.last().unwrap();
        let fworst = *fvals.last().unwrap();

        let spread: f64 = simplex
            .iter()
            .map(|u| {
                (0..NDIM)
                    .map(|i| (u[i] - c[i]).abs())
                    .fold(0.0f64, f64::max)
            })
            .fold(0.0f64, f64::max);
        if spread < 0.01 {
            why = format!("simplex collapsed (spread {spread:.4})");
            break;
        }

        let mix = |f: f64| -> Vec6 {
            std::array::from_fn(|i| {
                if free[i] {
                    (c[i] + f * (c[i] - worst[i])).clamp(0.0, 1.0)
                } else {
                    u0[i]
                }
            })
        };
        let refl = mix(1.0);
        let frefl = objective(p0, refl, sim, prepared, &mut history, &mut evals, quiet).await?;
        if frefl < fvals[0] {
            let exp = mix(2.0);
            if evals >= budget {
                break;
            }
            let fexp = objective(p0, exp, sim, prepared, &mut history, &mut evals, quiet).await?;
            if fexp < frefl {
                *simplex.last_mut().unwrap() = exp;
                *fvals.last_mut().unwrap() = fexp;
            } else {
                *simplex.last_mut().unwrap() = refl;
                *fvals.last_mut().unwrap() = frefl;
            }
        } else if frefl < fvals[fvals.len() - 2] {
            *simplex.last_mut().unwrap() = refl;
            *fvals.last_mut().unwrap() = frefl;
        } else {
            let con = mix(-0.5);
            if evals >= budget {
                break;
            }
            let fcon = objective(p0, con, sim, prepared, &mut history, &mut evals, quiet).await?;
            if fcon < fworst {
                *simplex.last_mut().unwrap() = con;
                *fvals.last_mut().unwrap() = fcon;
            } else {
                let best = simplex[0];
                for k in 1..simplex.len() {
                    let s: Vec6 = std::array::from_fn(|i| {
                        if free[i] {
                            best[i] + 0.5 * (simplex[k][i] - best[i])
                        } else {
                            u0[i]
                        }
                    });
                    simplex[k] = s;
                    if evals >= budget {
                        break;
                    }
                    fvals[k] =
                        objective(p0, s, sim, prepared, &mut history, &mut evals, quiet).await?;
                }
            }
        }
    }
    history.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok((history, evals, why))
}

fn free_dims(spec: &str) -> [bool; NDIM] {
    std::array::from_fn(|i| spec.split(',').any(|t| t.trim() == DIM_NAMES[i]))
}

// ----------------------------------------------------------------------- CLI

#[derive(Parser, Debug)]
#[command(name = "calibrate")]
struct Args {
    /// JSON list of cases
    #[arg(long, default_value = "cases.json")]
    cases: String,
    /// swissALTI3D zarr cache directory
    #[arg(long, default_value = "dtm_cache.zarr")]
    cache: String,
    /// padding around the observed outline, metres
    #[arg(long, default_value_t = 300.0)]
    padding: f64,
    /// restrict to these case names (comma separated)
    #[arg(long)]
    only: Option<String>,
    /// restrict to one split of the case file ("cal" or "val")
    #[arg(long)]
    split: Option<String>,
    /// JSON with the fitted storm forcing (see Weather); omit for the defaults
    #[arg(long)]
    weather: Option<String>,
    /// output JSON path
    #[arg(long, default_value = "out.json")]
    out: String,
    /// pin to this GPU adapter index (position in wgpu's adapter enumeration)
    /// instead of letting wgpu pick one; lets N concurrent processes each
    /// claim a distinct physical GPU
    #[arg(long)]
    gpu_index: Option<usize>,
    /// experiment stage this process belongs to, stamped onto every logged
    /// evaluation (e.g. "s1_structure")
    #[arg(long, default_value = "")]
    stage: String,
    /// identifier of the global candidate being evaluated, stamped onto every
    /// logged evaluation (e.g. "voellmy_c1p1e1n0")
    #[arg(long, default_value = "")]
    candidate: String,
    /// where to append the per-evaluation log; defaults to `<out>.evals.jsonl`.
    /// Pass "none" to disable logging entirely.
    #[arg(long)]
    eval_log: Option<String>,
    /// scoring convention: fill closed depressions shallower than this (metres)
    /// before tracing release drainage, so a moving flow is not stopped by
    /// micro-relief the way a water droplet would be. 0 disables.
    #[arg(long, default_value_t = PIT_FILL_TOLERANCE_M)]
    pit_tolerance: f32,
    #[command(subcommand)]
    cmd: Cmd,
}

/// Every parameter override, shared by all subcommands so they cannot drift.
#[derive(clap::Args, Debug, Clone, Default)]
struct ParamOpts {
    #[arg(long)]
    mu: Option<f32>,
    #[arg(long)]
    xi: Option<f32>,
    #[arg(long)]
    slab: Option<f32>,
    #[arg(long)]
    band: Option<f32>,
    #[arg(long)]
    thr: Option<f32>,
    #[arg(long)]
    model: Option<u32>,
    #[arg(long)]
    flags: Option<u32>,
    #[arg(long)]
    max_steps: Option<u32>,
    #[arg(long)]
    ppc: Option<u32>,
    #[arg(long)]
    cfl: Option<f32>,
    #[arg(long)]
    slope_min: Option<f32>,
    #[arg(long)]
    slope_max: Option<f32>,
    #[arg(long)]
    density: Option<f32>,
    /// 0 = uniform slab, 1 = weather/terrain-driven release thickness field
    #[arg(long)]
    slab_mode: Option<u32>,
    /// amplitude multiplying the weather-driven thickness field
    #[arg(long)]
    amp: Option<f32>,
    /// wind-loading amplitude for the aspect term
    #[arg(long)]
    wind: Option<f32>,
    /// 0 = use --density, 1 = use each case's measured storm-slab density
    #[arg(long)]
    density_mode: Option<u32>,
    #[arg(long)]
    ifa: Option<f32>,
    #[arg(long)]
    roughness: Option<f32>,
    /// particle stop / no-friction speed cutoff, m/s
    #[arg(long)]
    velocity_threshold: Option<f32>,
    /// scoring convention: minimum normalised residence per cell (0 = off)
    #[arg(long)]
    min_residence: Option<f32>,
    /// scoring convention: drop footprint not connected to the release (0/1)
    #[arg(long)]
    require_connected: Option<u32>,
    /// scoring convention: drop release cells that drain away from the outline (0/1)
    #[arg(long)]
    clip_drainage: Option<u32>,
}

impl ParamOpts {
    fn apply(&self, base: Params) -> Params {
        let mut p = base;
        macro_rules! set {
            ($opt:ident, $field:ident) => {
                if let Some(v) = self.$opt {
                    p.$field = v;
                }
            };
        }
        set!(mu, friction_coefficient);
        set!(xi, drag_coefficient);
        set!(slab, slab_thickness);
        set!(band, release_band_frac);
        set!(thr, flow_threshold);
        set!(model, friction_model);
        set!(flags, flags);
        set!(max_steps, max_steps);
        set!(ppc, released_particles_per_cell);
        set!(cfl, cfl);
        set!(slope_min, release_min_slope);
        set!(slope_max, release_max_slope);
        set!(density, density);
        set!(slab_mode, slab_mode);
        set!(amp, slab_amp);
        set!(wind, slab_wind_amp);
        set!(density_mode, density_mode);
        set!(ifa, internal_friction_angle);
        set!(roughness, roughness_threshold);
        set!(velocity_threshold, velocity_threshold);
        set!(min_residence, min_residence);
        set!(require_connected, require_release_connected);
        set!(clip_drainage, clip_release_to_drainage);
        p
    }
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// evaluate one parameter vector
    Run {
        #[command(flatten)]
        p: ParamOpts,
    },
    /// one-at-a-time sweep of a single parameter
    Sweep {
        #[arg(long)]
        param: String,
        #[arg(long, value_delimiter = ',')]
        values: Vec<f64>,
        #[command(flatten)]
        p: ParamOpts,
    },
    /// Nelder-Mead over one global parameter vector for all cases
    Search {
        #[arg(long, default_value_t = 80)]
        budget: usize,
        /// which of mu,xi,slab,band,amp,wind the simplex may move
        #[arg(long, default_value = "mu,xi,slab")]
        free: String,
        #[command(flatten)]
        p: ParamOpts,
    },
    /// Nelder-Mead independently per case (the per-event calibration the
    /// exposé plans, minus CMA-ES). Writes one record per case.
    PerEvent {
        #[arg(long, default_value_t = 40)]
        budget: usize,
        #[arg(long, default_value = "mu,xi,slab")]
        free: String,
        #[command(flatten)]
        p: ParamOpts,
    },
    /// 2-D identifiability grid: the outer product of --mus and --xis is
    /// evaluated on every case, so the shape of Omega_T around a per-event
    /// optimum can be measured (sharp optimum vs plateau vs diagonal ridge).
    Grid {
        #[arg(long, value_delimiter = ',')]
        mus: Vec<f64>,
        #[arg(long, value_delimiter = ',')]
        xis: Vec<f64>,
        /// JSON object mapping case name -> slab thickness [m]; cases missing
        /// from it keep the shared --slab value
        #[arg(long)]
        slab_from: Option<String>,
        #[command(flatten)]
        p: ParamOpts,
    },
    /// Evaluate each case at its OWN parameter vector, read from a JSON map of
    /// case name -> {mu, xi, slab}. This is what scores a regressor end to end:
    /// predicted parameters are only useful if simulating with them beats a
    /// single global vector, which R^2 on the parameters themselves cannot say.
    Apply {
        /// JSON object: {"aval_123": {"mu": 0.3, "xi": 900, "slab": 0.4}, ...}
        /// Any missing key falls back to the shared value from the flags below.
        #[arg(long)]
        params_from: String,
        #[command(flatten)]
        p: ParamOpts,
    },
    /// dump reference/simulated/release rasters for plotting
    Dump {
        #[arg(long, default_value = "dump")]
        dir: String,
        #[command(flatten)]
        p: ParamOpts,
    },
}

fn print_cases(r: &RunResult) {
    for c in &r.cases {
        eprintln!(
            "{:>14}  omega={:+.3} (a={:.3} b={:.3} g={:.3})  sim/ref={}/{}  rel={} d={:.2}m  L {:.0}/{:.0}  W {:.0}/{:.0}  reach {:+.0}m  v={:.1}  {:.1}s",
            c.name,
            c.area.omega,
            c.area.alpha,
            c.area.beta,
            c.area.gamma,
            c.sim_cells,
            c.ref_cells,
            c.release_cells,
            c.mean_slab_m,
            c.sim_runout_m,
            c.obs_runout_m,
            c.sim_width_m,
            c.obs_width_m,
            c.reach_err_m,
            c.max_velocity,
            c.seconds
        );
    }
    eprintln!(
        "MEAN omega={:+.4}  hwri1={:+.4}  null={:+.4}   ({} cases, {} failed)",
        r.mean_omega,
        r.mean_hwri_l1,
        r.mean_release_only,
        r.cases.len(),
        r.failures.len()
    );
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let specs: Vec<CaseSpec> =
        serde_json::from_str(&std::fs::read_to_string(&args.cases)?).context("parsing cases")?;
    let specs: Vec<CaseSpec> = match &args.only {
        Some(list) => {
            let want: Vec<&str> = list.split(',').collect();
            specs
                .into_iter()
                .filter(|s| want.contains(&s.name.as_str()))
                .collect()
        }
        None => specs,
    };
    let specs: Vec<CaseSpec> = match &args.split {
        Some(sp) => specs.into_iter().filter(|s| &s.split == sp).collect(),
        None => specs,
    };
    eprintln!("cases: {}", specs.len());

    // Provenance and the evaluation log are set up before any simulation runs,
    // for every subcommand -- `sweep`, `search` and `grid` used to throw their
    // individual evaluations away and keep only the summary.
    let _ = PROVENANCE.set(Provenance {
        stage: args.stage.clone(),
        candidate: args.candidate.clone(),
    });
    let log_path = args
        .eval_log
        .clone()
        .unwrap_or_else(|| format!("{}.evals.jsonl", args.out));
    // Truncating, not appending: each job owns its log path exclusively, so a
    // re-run after a partial failure must replace that job's rows rather than
    // leave a half-run's evaluations duplicated in front of the good ones.
    if log_path != "none" {
        match std::fs::File::create(&log_path) {
            Ok(f) => {
                let _ = EVAL_LOG.set(Mutex::new(std::io::BufWriter::new(f)));
                eprintln!("logging every evaluation to {log_path}");
            }
            Err(e) => eprintln!("!! could not open {log_path} for per-eval logging: {e}"),
        }
    }

    let wx: Weather = match &args.weather {
        Some(path) => {
            serde_json::from_str(&std::fs::read_to_string(path)?).context("parsing weather file")?
        }
        None => Weather::default(),
    };

    // Preparation (rasterise + DEM fetch) is where cases die: missing
    // swissALTI3D coverage, degenerate geometry. Record, do not abort.
    let tm = TileManager::new(&args.cache)?;
    let mut prepared = Vec::new();
    let mut prep_failures: Vec<(String, String)> = Vec::new();
    for (i, s) in specs.iter().enumerate() {
        let t = Instant::now();
        match prepare_case(&tm, s, args.padding, &wx, args.pit_tolerance).await {
            Ok(c) => {
                eprintln!(
                    "[{}/{}] prepared {} : grid {}x{} ({} ref cells) in {:.1}s",
                    i + 1,
                    specs.len(),
                    s.name,
                    c.width,
                    c.height,
                    c.reference.iter().filter(|&&b| b).count(),
                    t.elapsed().as_secs_f64()
                );
                prepared.push(c);
            }
            Err(e) => {
                eprintln!(
                    "[{}/{}] !! prepare {} FAILED: {e}",
                    i + 1,
                    specs.len(),
                    s.name
                );
                prep_failures.push((s.name.clone(), format!("{e}")));
            }
        }
    }
    eprintln!(
        "prepared {}/{} cases ({} failed)",
        prepared.len(),
        specs.len(),
        prep_failures.len()
    );
    if prepared.is_empty() {
        bail!("no case could be prepared");
    }
    let _ = std::fs::write(
        format!("{}.prepfail.json", args.out),
        serde_json::to_string_pretty(&prep_failures)?,
    );

    let mut sim = Simulation::new_with_gpu_index(args.gpu_index).await?;
    // Single source of truth: preparation used this value, so this is what gets logged.
    let base = Params {
        pit_fill_tolerance_m: args.pit_tolerance,
        ..Default::default()
    };

    match args.cmd {
        Cmd::Run { p } => {
            let p = p.apply(base);
            let mut r = evaluate_all(&mut sim, &prepared, &p).await?;
            print_cases(&r);
            r.failures.extend(prep_failures);
            std::fs::write(&args.out, serde_json::to_string(&r)?)?;
        }
        Cmd::Sweep { param, values, p } => {
            let p0 = p.apply(base);
            let mut all = Vec::new();
            for v in values {
                let mut p = p0;
                match param.as_str() {
                    "mu" => p.friction_coefficient = v as f32,
                    "xi" => p.drag_coefficient = v as f32,
                    "slab" => p.slab_thickness = v as f32,
                    "band" => p.release_band_frac = v as f32,
                    "thr" => p.flow_threshold = v as f32,
                    "ppc" => p.released_particles_per_cell = v as u32,
                    "cfl" => p.cfl = v as f32,
                    "slope_min" => p.release_min_slope = v as f32,
                    "slope_max" => p.release_max_slope = v as f32,
                    "model" => p.friction_model = v as u32,
                    "flags" => p.flags = v as u32,
                    "density" => p.density = v as f32,
                    "amp" => p.slab_amp = v as f32,
                    "wind" => p.slab_wind_amp = v as f32,
                    "max_steps" => p.max_steps = v as u32,
                    "ifa" => p.internal_friction_angle = v as f32,
                    "roughness" => p.roughness_threshold = v as f32,
                    "velocity_threshold" => p.velocity_threshold = v as f32,
                    "min_residence" => p.min_residence = v as f32,
                    "require_connected" => p.require_release_connected = v as u32,
                    "clip_drainage" => p.clip_release_to_drainage = v as u32,
                    "slab_mode" => p.slab_mode = v as u32,
                    "density_mode" => p.density_mode = v as u32,
                    other => bail!("unknown sweep parameter {other}"),
                }
                let r = evaluate_all(&mut sim, &prepared, &p).await?;
                eprintln!(
                    "{param}={v:<10} mean omega={:+.4}  hwri1={:+.4}  null={:+.4}  ({} ok)",
                    r.mean_omega,
                    r.mean_hwri_l1,
                    r.mean_release_only,
                    r.cases.len()
                );
                all.push(r);
            }
            std::fs::write(&args.out, serde_json::to_string(&all)?)?;
        }
        Cmd::Search { budget, free, p } => {
            let p0 = p.apply(base);
            let free = free_dims(&free);
            eprintln!(
                "free dimensions: {:?}",
                DIM_NAMES
                    .iter()
                    .zip(free)
                    .filter(|(_, f)| *f)
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
            );
            let (history, evals, why) =
                nelder_mead(&p0, free, budget, &mut sim, &prepared, false).await?;
            let best = history[0];
            eprintln!(
                "BEST omega={:+.4} mu={:.4} xi={:.0} h={:.3} band={:.3} amp={:.3} wind={:.2} after {evals} evals ({why})",
                best.1,
                best.0.friction_coefficient,
                best.0.drag_coefficient,
                best.0.slab_thickness,
                best.0.release_band_frac,
                best.0.slab_amp,
                best.0.slab_wind_amp
            );
            #[derive(Serialize)]
            struct Hist {
                params: Params,
                mean_omega: f64,
                mean_hwri_l1: f64,
                evals: usize,
                terminated: String,
                n_cases: usize,
            }
            let hist: Vec<Hist> = history
                .iter()
                .map(|(p, o, h)| Hist {
                    params: *p,
                    mean_omega: *o,
                    mean_hwri_l1: *h,
                    evals,
                    terminated: why.clone(),
                    n_cases: prepared.len(),
                })
                .collect();
            std::fs::write(&args.out, serde_json::to_string(&hist)?)?;
        }
        Cmd::PerEvent { budget, free, p } => {
            let p0 = p.apply(base);
            let freeb = free_dims(&free);
            #[derive(Serialize)]
            struct PerEventRec {
                name: String,
                sze: i64,
                aval_shape: i64,
                aspect: String,
                split: String,
                start_zone: f64,
                dpo_alt: f64,
                area: f64,
                /// score of the shared starting vector on this case
                omega_start: f64,
                /// score after the per-event search
                omega_best: f64,
                best: Params,
                evals: usize,
                terminated: String,
                /// how flat the optimum is: spread of the top-decile evaluations
                top_decile_spread: [f64; NDIM],
                result: CaseResult,
            }
            let mut out: Vec<PerEventRec> = Vec::new();
            let t_all = Instant::now();
            for (i, case) in prepared.iter().enumerate() {
                let one = std::slice::from_ref(case);
                let start = match evaluate_case(&mut sim, case, &p0).await {
                    Ok(r) => r.area.omega,
                    Err(e) => {
                        eprintln!(
                            "[{}/{}] {} start eval failed: {e}",
                            i + 1,
                            prepared.len(),
                            case.spec.name
                        );
                        continue;
                    }
                };
                let (history, evals, why) =
                    match nelder_mead(&p0, freeb, budget, &mut sim, one, true).await {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!(
                                "[{}/{}] {} search failed: {e}",
                                i + 1,
                                prepared.len(),
                                case.spec.name
                            );
                            continue;
                        }
                    };
                let (bp, bo, _) = history[0];
                // spread of the best 10 % of evaluations, per dimension: a wide
                // spread means the optimum is not identified on this case
                let k = (history.len() / 10).max(2).min(history.len());
                let top: Vec<Vec6> = history[..k].iter().map(|(p, _, _)| params_vec(p)).collect();
                let spread: [f64; NDIM] = std::array::from_fn(|d| {
                    let vals: Vec<f64> = top.iter().map(|v| v[d]).collect();
                    vals.iter().cloned().fold(f64::MIN, f64::max)
                        - vals.iter().cloned().fold(f64::MAX, f64::min)
                });
                let res = evaluate_case(&mut sim, case, &bp).await?;
                eprintln!(
                    "[{}/{}] {:>14} {:+.3} -> {:+.3}  mu={:.3} xi={:5.0} h={:.2}  ({evals} evals, {why})",
                    i + 1,
                    prepared.len(),
                    case.spec.name,
                    start,
                    bo,
                    bp.friction_coefficient,
                    bp.drag_coefficient,
                    bp.slab_thickness
                );
                out.push(PerEventRec {
                    name: case.spec.name.clone(),
                    sze: case.spec.sze,
                    aval_shape: case.spec.aval_shape,
                    aspect: case.spec.aspect.clone(),
                    split: case.spec.split.clone(),
                    start_zone: case.spec.start_zone,
                    dpo_alt: case.spec.dpo_alt,
                    area: case.spec.area,
                    omega_start: start,
                    omega_best: bo,
                    best: bp,
                    evals,
                    terminated: why,
                    top_decile_spread: spread,
                    result: res,
                });
                // checkpoint after every case -- a hard kill loses at most the
                // case currently being searched, not the whole run.
                if let Ok(s) = serde_json::to_string(&out) {
                    let _ = std::fs::write(&args.out, s);
                }
            }
            let m0 = out.iter().map(|r| r.omega_start).sum::<f64>() / out.len() as f64;
            let m1 = out.iter().map(|r| r.omega_best).sum::<f64>() / out.len() as f64;
            eprintln!(
                "PER-EVENT: {} cases, mean {:+.4} -> {:+.4} (gap {:+.4}) in {:.0}s",
                out.len(),
                m0,
                m1,
                m1 - m0,
                t_all.elapsed().as_secs_f64()
            );
            std::fs::write(&args.out, serde_json::to_string(&out)?)?;
        }
        Cmd::Grid {
            mus,
            xis,
            slab_from,
            p,
        } => {
            let p0 = p.apply(base);
            let slabs: std::collections::HashMap<String, f32> = match &slab_from {
                Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)
                    .context("parsing slab_from file")?,
                None => Default::default(),
            };
            #[derive(Serialize)]
            struct GridRec {
                name: String,
                sze: i64,
                aval_shape: i64,
                aspect: String,
                area: f64,
                slab: f32,
                mus: Vec<f64>,
                xis: Vec<f64>,
                /// omega[i][j] for mus[i], xis[j]; null where the case failed
                omega: Vec<Vec<Option<f64>>>,
                clipped: Vec<Vec<bool>>,
                reach_err_m: Vec<Vec<f32>>,
                sim_cells: Vec<Vec<usize>>,
            }
            let mut out: Vec<GridRec> = Vec::new();
            let t_all = Instant::now();
            for (i, case) in prepared.iter().enumerate() {
                let name = case.spec.name.clone();
                let slab = *slabs.get(&name).unwrap_or(&p0.slab_thickness);
                let mut omega = Vec::new();
                let mut clipped = Vec::new();
                let mut reach = Vec::new();
                let mut cells = Vec::new();
                let t = Instant::now();
                for &mu in &mus {
                    let (mut ro, mut rc, mut rr, mut rn) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for &xi in &xis {
                        let mut q = p0;
                        q.friction_coefficient = mu as f32;
                        q.drag_coefficient = xi as f32;
                        q.slab_thickness = slab;
                        match evaluate_case(&mut sim, case, &q).await {
                            Ok(r) => {
                                ro.push(Some(r.area.omega));
                                rc.push(r.clipped_at_edge);
                                rr.push(r.reach_err_m);
                                rn.push(r.sim_cells);
                            }
                            Err(e) => {
                                eprintln!("  !! {name} mu={mu} xi={xi}: {e}");
                                ro.push(None);
                                rc.push(false);
                                rr.push(f32::NAN);
                                rn.push(0);
                            }
                        }
                    }
                    omega.push(ro);
                    clipped.push(rc);
                    reach.push(rr);
                    cells.push(rn);
                }
                let best = omega
                    .iter()
                    .flatten()
                    .filter_map(|v| *v)
                    .fold(f64::MIN, f64::max);
                eprintln!(
                    "[{}/{}] {:>14} slab={:.2} grid {}x{} best omega={:+.3} ({:.0}s)",
                    i + 1,
                    prepared.len(),
                    name,
                    slab,
                    mus.len(),
                    xis.len(),
                    best,
                    t.elapsed().as_secs_f64()
                );
                out.push(GridRec {
                    name,
                    sze: case.spec.sze,
                    aval_shape: case.spec.aval_shape,
                    aspect: case.spec.aspect.clone(),
                    area: case.spec.area,
                    slab,
                    mus: mus.clone(),
                    xis: xis.clone(),
                    omega,
                    clipped,
                    reach_err_m: reach,
                    sim_cells: cells,
                });
            }
            eprintln!(
                "GRID: {} cases x {} points in {:.0}s",
                out.len(),
                mus.len() * xis.len(),
                t_all.elapsed().as_secs_f64()
            );
            std::fs::write(&args.out, serde_json::to_string(&out)?)?;
        }
        Cmd::Apply { params_from, p } => {
            let p0 = p.apply(base);
            #[derive(Deserialize)]
            struct PerCase {
                mu: Option<f32>,
                xi: Option<f32>,
                slab: Option<f32>,
            }
            let table: std::collections::HashMap<String, PerCase> =
                serde_json::from_str(&std::fs::read_to_string(&params_from)?)
                    .context("parsing params_from file")?;
            let mut out = Vec::new();
            let mut failures = Vec::new();
            let mut missing = 0usize;
            for case in &prepared {
                let mut q = p0;
                match table.get(&case.spec.name) {
                    Some(pc) => {
                        if let Some(v) = pc.mu {
                            q.friction_coefficient = v;
                        }
                        if let Some(v) = pc.xi {
                            q.drag_coefficient = v;
                        }
                        if let Some(v) = pc.slab {
                            q.slab_thickness = v;
                        }
                    }
                    None => missing += 1,
                }
                match evaluate_case(&mut sim, case, &q).await {
                    Ok(r) => out.push(r),
                    Err(e) => {
                        eprintln!("  !! {} failed: {e}", case.spec.name);
                        failures.push((case.spec.name.clone(), format!("{e}")));
                    }
                }
            }
            if out.is_empty() {
                bail!("every case failed");
            }
            if missing > 0 {
                eprintln!(
                    "note: {missing} case(s) absent from {params_from}, used the shared vector"
                );
            }
            let mean_omega = out.iter().map(|c| c.area.omega).sum::<f64>() / out.len() as f64;
            let mean_hwri_l1 = out.iter().map(|c| c.hwri_l1.omega).sum::<f64>() / out.len() as f64;
            let mean_release_only =
                out.iter().map(|c| c.release_only_omega).sum::<f64>() / out.len() as f64;
            let mut r = RunResult {
                params: p0,
                mean_omega,
                mean_hwri_l1,
                mean_release_only,
                cases: out,
                failures,
            };
            print_cases(&r);
            r.failures.extend(prep_failures);
            std::fs::write(&args.out, serde_json::to_string(&r)?)?;
        }
        Cmd::Dump { dir, p } => {
            let p = p.apply(base);
            std::fs::create_dir_all(&dir)?;
            for case in &prepared {
                let r = match evaluate_case(&mut sim, case, &p).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("!! {} {e}", case.spec.name);
                        continue;
                    }
                };
                let peak_h = sim.fetch_peak_flow_thickness().await?;
                let band = build_release(case, &p);
                let (release, apex) = (&band.thickness, band.apex);
                #[derive(Serialize)]
                struct DumpOut<'a> {
                    name: &'a str,
                    w: usize,
                    h: usize,
                    xmin: f32,
                    ymin: f32,
                    cell: f32,
                    apex: (usize, usize),
                    result: &'a CaseResult,
                    dem: &'a [f32],
                    reference: Vec<u8>,
                    release: Vec<u8>,
                    peak_h: Vec<f32>,
                }
                let d = DumpOut {
                    name: &case.spec.name,
                    w: case.width,
                    h: case.height,
                    xmin: case.dem.bounds.xmin,
                    ymin: case.dem.bounds.ymin,
                    cell: case.dem.cell_size,
                    apex,
                    result: &r,
                    dem: &case.dem.data1d,
                    reference: case.reference.iter().map(|&b| b as u8).collect(),
                    release: release.iter().map(|&t| (t > 0.0) as u8).collect(),
                    peak_h: peak_h
                        .iter()
                        .map(|&v| (v * 1000.0).round() / 1000.0)
                        .collect(),
                };
                std::fs::write(
                    format!("{}/{}.json", dir, case.spec.name),
                    serde_json::to_string(&d)?,
                )?;
                eprintln!("dumped {} omega={:+.3}", case.spec.name, r.area.omega);
            }
        }
    }
    Ok(())
}

// ----------------------------------------------------------------- tests
//
// The two scoring-convention fixes are pure geometry over a DEM and a mask, so
// they are testable without a GPU. These pin the behaviour that matters: that
// the ridge clip actually separates the two sides of a crest, that the
// connectivity filter removes detached blobs while keeping the body, and that
// both fail safe rather than emptying a case.

#[cfg(test)]
mod tests {
    use super::*;

    fn dem_from(rows: &[Vec<f32>], cell: f32) -> Dem {
        let h = rows.len();
        let w = rows[0].len();
        let mut d = Dem::default();
        d.width = w;
        d.height = h;
        d.cell_size = cell;
        d.data1d = rows.iter().flatten().copied().collect();
        d
    }

    /// A north-south ridge down the middle of the grid: elevation falls away to
    /// the west and to the east. The observed outline sits on the west flank.
    /// Cells east of the crest drain into the other valley and must be excluded.
    #[test]
    fn drainage_clip_separates_the_two_sides_of_a_ridge() {
        let (w, h) = (11usize, 5usize);
        let crest = 5i32;
        let rows: Vec<Vec<f32>> = (0..h)
            .map(|_| {
                (0..w)
                    .map(|x| 100.0 - ((x as i32 - crest).abs() as f32) * 10.0)
                    .collect()
            })
            .collect();
        let dem = dem_from(&rows, 5.0);

        // outline: the two westmost columns, i.e. the bottom of the west valley
        let mut reference = vec![false; w * h];
        for y in 0..h {
            for x in 0..2 {
                reference[y * w + x] = true;
            }
        }

        let (drains, _) = drains_to_body(&dem, &reference, w, h, PIT_FILL_TOLERANCE_M);
        for y in 0..h {
            for x in 0..w {
                let got = drains[y * w + x];
                if x <= crest as usize {
                    assert!(got, "west flank cell ({x},{y}) should drain to the outline");
                } else {
                    assert!(!got, "east flank cell ({x},{y}) drains into the far valley");
                }
            }
        }
    }

    /// A cell inside the outline drains to it by definition, even if it is a
    /// local pit with nowhere to descend to.
    #[test]
    fn cells_inside_the_outline_always_drain_to_it() {
        let rows = vec![vec![10.0, 10.0], vec![10.0, 10.0]];
        let dem = dem_from(&rows, 5.0);
        let reference = vec![true, false, false, false];
        let (drains, _) = drains_to_body(&dem, &reference, 2, 2, PIT_FILL_TOLERANCE_M);
        assert!(
            drains[0],
            "a reference cell must count as draining to the outline"
        );
    }

    /// Flat ground cannot descend anywhere, so nothing outside the outline
    /// reaches it -- and crucially the walk must terminate rather than cycle
    /// between two equal-height neighbours.
    #[test]
    fn flat_terrain_terminates_and_reaches_nothing() {
        let rows = vec![vec![7.0; 6]; 6];
        let dem = dem_from(&rows, 5.0);
        let mut reference = vec![false; 36];
        reference[0] = true;
        let (drains, _) = drains_to_body(&dem, &reference, 6, 6, PIT_FILL_TOLERANCE_M);
        assert!(drains[0]);
        assert_eq!(
            drains.iter().filter(|&&b| b).count(),
            1,
            "on flat ground only the outline itself should qualify"
        );
    }

    /// The step cap is derived from the grid, not hard-coded, so it stays
    /// proportionate as domains vary from ~100 to ~400 cells across.
    #[test]
    fn step_cap_scales_with_the_grid_diagonal() {
        let small = dem_from(&vec![vec![1.0; 10]; 10], 5.0);
        let large = dem_from(&vec![vec![1.0; 200]; 100], 5.0);
        let (_, cap_small) = drains_to_body(&small, &vec![true; 100], 10, 10, PIT_FILL_TOLERANCE_M);
        let (_, cap_large) =
            drains_to_body(&large, &vec![true; 20000], 200, 100, PIT_FILL_TOLERANCE_M);
        assert_eq!(
            cap_small,
            ((10f64 * 10.0 + 10.0 * 10.0).sqrt()).ceil() as u32
        );
        assert!(cap_large > cap_small * 10, "cap must grow with the domain");
    }

    /// Exact step counts, not just reachability: a straight descent of k cells
    /// to the outline must report exactly k.
    #[test]
    fn descent_step_count_is_exact() {
        let (w, h) = (6usize, 1usize);
        // rises to the east, so every cell descends one step west toward the
        // outline at column 0
        let rows = vec![
            (0..w)
                .map(|x| 100.0 + x as f32 * 10.0)
                .collect::<Vec<f32>>(),
        ];
        let dem = dem_from(&rows, 5.0);
        let mut reference = vec![false; w];
        reference[0] = true;
        let target = outline_body(&dem, &reference);
        let steps = descent_steps_to_target(&dem.data1d, dem.cell_size, &target, w, h);
        for x in 0..w {
            assert_eq!(
                steps[x], x as u32,
                "cell {x} should be {x} steps from the outline"
            );
        }
    }

    #[test]
    fn connectivity_drops_detached_blobs_and_keeps_the_body() {
        let (w, h) = (10usize, 3usize);
        let mut mask = vec![false; w * h];
        let mut release = vec![false; w * h];
        // body: columns 0..4 of the middle row, released at column 0
        for x in 0..5 {
            mask[1 * w + x] = true;
        }
        release[1 * w + 0] = true;
        // detached blob at columns 7..9, separated by two empty columns
        for x in 7..10 {
            mask[1 * w + x] = true;
        }

        let (keep, fallback) = keep_release_connected(&mask, &release, w, h);
        assert!(!fallback);
        assert_eq!(keep.iter().filter(|&&b| b).count(), 5);
        for x in 0..5 {
            assert!(keep[1 * w + x], "body cell {x} must survive");
        }
        for x in 7..10 {
            assert!(!keep[1 * w + x], "detached cell {x} must be dropped");
        }
    }

    /// A thin trail attached to the body is NOT removed by connectivity -- it is
    /// connected. This is why the residence gate exists as a separate filter,
    /// and the test records that division of labour so nobody later "simplifies"
    /// one of the two away.
    #[test]
    fn connectivity_alone_does_not_remove_an_attached_tail() {
        let (w, h) = (10usize, 3usize);
        let mut mask = vec![false; w * h];
        let mut release = vec![false; w * h];
        for x in 0..9 {
            mask[1 * w + x] = true; // body plus a one-cell-wide tail, contiguous
        }
        release[1 * w + 0] = true;
        let (keep, _) = keep_release_connected(&mask, &release, w, h);
        assert_eq!(
            keep.iter().filter(|&&b| b).count(),
            9,
            "an attached tail survives connectivity; the residence gate is what removes it"
        );
    }

    /// With no release cell touching the mask the filter has nothing to seed
    /// from. It must hand back the input rather than an empty footprint, which
    /// would score Omega_T = -1 and read as a physics failure.
    #[test]
    fn connectivity_falls_back_when_there_is_no_seed() {
        let (w, h) = (6usize, 3usize);
        let mut mask = vec![false; w * h];
        mask[1 * w + 4] = true;
        let release = vec![false; w * h];
        let (keep, fallback) = keep_release_connected(&mask, &release, w, h);
        assert!(fallback, "no seed must be reported, not silently swallowed");
        assert_eq!(keep, mask);
    }

    /// A release band that drains entirely into the neighbouring valley must
    /// be refused, not silently reduced to nothing and scored as a physics
    /// failure — and so must one the clip reduces to an untrustworthy stub
    /// (the measured worst case kept 4 of 183 cells). The caller turns
    /// `clip_refused` into an error naming the cause.
    #[test]
    fn a_leaking_release_is_refused_below_the_floor() {
        let verdict = |n_cand: usize, kept: usize| {
            let dropped = (n_cand - kept) as f64 / n_cand.max(1) as f64;
            let refused = n_cand > 0
                && (kept == 0 || (n_cand >= MIN_RELEASE_CELLS && kept < MIN_RELEASE_CELLS));
            (refused, dropped > CLIP_SEVERE_FRAC)
        };
        assert_eq!(verdict(40, 0), (true, true), "all cells lost -> refuse");
        assert_eq!(
            verdict(183, 4),
            (true, true),
            "the aval_13722 shape -> refuse"
        );
        assert_eq!(
            verdict(40, 19),
            (true, true),
            "just under the floor -> refuse"
        );
        assert_eq!(
            verdict(40, 20),
            (false, false),
            "at the floor, half kept -> ordinary"
        );
        assert_eq!(verdict(40, 10), (true, true), "stub below floor -> refuse");
        assert_eq!(verdict(40, 30), (false, false), "25% lost -> ordinary");
        assert_eq!(
            verdict(10, 3),
            (false, true),
            "naturally tiny release -> floor exempt, flagged"
        );
        assert_eq!(
            verdict(0, 0),
            (false, false),
            "no candidates is a different failure"
        );
    }

    fn prepared_from(rows: &[Vec<f32>], reference: Vec<bool>, cell: f32) -> PreparedCase {
        let dem = dem_from(rows, cell);
        let (w, h) = (dem.width, dem.height);
        let (slope, aspect) = horn_slope_aspect(&dem);
        let (drains, _) = drains_to_body(&dem, &reference, w, h, PIT_FILL_TOLERANCE_M);
        PreparedCase {
            spec: CaseSpec {
                name: "synthetic".into(),
                shp: String::new(),
                area: 0.0,
                sze: 3,
                start_zone: 0.0,
                dpo_alt: 0.0,
                aspect: String::new(),
                aval_shape: 1,
                split: "cal".into(),
                wx: None,
            },
            dem,
            reference,
            slope,
            aspect,
            drains,
            wx: Weather::default(),
            width: w,
            height: h,
        }
    }

    /// THE REGRESSION TEST FOR THE TAUTOLOGY.
    ///
    /// The first version of this clip asked "does the descent reach any
    /// reference cell", seeding every reference cell as already-arrived. Since
    /// `build_release` only ever draws candidates from inside the reference,
    /// the answer was always yes and the clip was a no-op on all 105 real
    /// cases -- while a unit test of the descent function in isolation passed,
    /// because it exercised far-flank cells OUTSIDE the polygon, which is
    /// exactly the population the caller never asks about.
    ///
    /// So this drives the whole path `build_release` actually takes, with a
    /// polygon that spans a ridge: the west flank drains into the body, the
    /// sliver east of the crest drains into the next valley and must be cut.
    #[test]
    fn build_release_clips_the_far_flank_of_a_polygon_spanning_a_ridge() {
        let (w, h) = (21usize, 5usize);
        let crest = 10i32;
        // symmetric ridge: 5 m of fall per 5 m cell on both flanks
        let rows: Vec<Vec<f32>> = (0..h)
            .map(|_| {
                (0..w)
                    .map(|x| 100.0 - ((x as i32 - crest).abs() as f32) * 5.0)
                    .collect()
            })
            .collect();
        // outline covers the west flank and one column past the crest
        let mut reference = vec![false; w * h];
        for y in 0..h {
            for x in 4..=11 {
                reference[y * w + x] = true;
            }
        }
        let case = prepared_from(&rows, reference, 5.0);

        // Sanity on the fixture: the release band must actually contain cells
        // on both sides of the crest, or the test proves nothing.
        let mut unclipped = Params::default();
        unclipped.clip_release_to_drainage = 0;
        let before = build_release(&case, &unclipped);
        let cols = |b: &ReleaseBand| -> Vec<usize> {
            let mut c: Vec<usize> = (0..w)
                .filter(|&x| (0..h).any(|y| b.thickness[y * w + x] > 0.0))
                .collect();
            c.sort();
            c
        };
        assert_eq!(
            cols(&before),
            vec![9, 11],
            "fixture: the band should straddle the crest (9 west, 11 east)"
        );
        assert_eq!(before.clipped_frac, 0.0);

        let after = build_release(&case, &Params::default());
        assert_eq!(
            cols(&after),
            vec![9],
            "only the far-flank column should be removed"
        );
        assert!(
            after.clipped_frac > 0.0,
            "the clip must actually fire on a ridge-spanning polygon"
        );
        assert_eq!(after.clipped_frac, 0.5);
        assert_eq!(after.cells, before.cells / 2);
        assert!(!after.clip_refused && !after.clip_severe);
    }

    /// The complement: a polygon wholly on one flank must not lose anything.
    /// A clip that fires everywhere would be as useless as one that never
    /// fires, just harder to notice.
    #[test]
    fn build_release_leaves_a_well_behaved_polygon_untouched() {
        let (w, h) = (21usize, 5usize);
        let rows: Vec<Vec<f32>> = (0..h)
            .map(|_| {
                (0..w)
                    .map(|x| 100.0 - ((x as i32 - 10).abs() as f32) * 5.0)
                    .collect()
            })
            .collect();
        let mut reference = vec![false; w * h];
        for y in 0..h {
            for x in 4..=9 {
                reference[y * w + x] = true; // west flank only
            }
        }
        let case = prepared_from(&rows, reference, 5.0);
        let band = build_release(&case, &Params::default());
        assert!(band.cells > 0, "a clean polygon must still release");
        assert_eq!(band.clipped_frac, 0.0, "nothing should be clipped here");
    }

    /// A shallow hollow must not stop a descent; a deep basin must.
    ///
    /// D8 treats any closed depression as a terminus, which on a 5 m DEM means
    /// sub-metre micro-relief severs drainage. Measured on `aval_13722`, the
    /// pits ending its release descent were 0.06-0.51 m deep -- terrain a
    /// moving avalanche crosses without noticing.
    #[test]
    fn shallow_pits_are_filled_and_deep_ones_are_not() {
        // A plane tilted west, with one cell dug out to make a closed pit.
        let (w, h) = (11usize, 5usize);
        // Gentle slope: on a steep one, digging a cell out still leaves it
        // above its downhill neighbour, so no closed depression forms and the
        // test would pass for the wrong reason.
        let mk = |depth: f32| -> Vec<Vec<f32>> {
            let mut rows: Vec<Vec<f32>> = (0..h)
                .map(|_| (0..w).map(|x| 100.0 + x as f32 * 0.1).collect())
                .collect();
            rows[2][5] -= depth;
            rows
        };

        // 0.5 m hollow, tolerance 2 m -> filled, so descent flows through it
        let shallow = dem_from(&mk(0.5), 5.0);
        let filled = fill_shallow_pits(&shallow.data1d, w, h, 2.0);
        assert!(
            filled[2 * w + 5] > shallow.data1d[2 * w + 5],
            "a 0.5 m hollow should be filled at a 2 m tolerance"
        );

        // 20 m basin, same tolerance -> untouched, still a genuine sink
        let deep = dem_from(&mk(20.0), 5.0);
        let filled = fill_shallow_pits(&deep.data1d, w, h, 2.0);
        assert_eq!(
            filled[2 * w + 5],
            deep.data1d[2 * w + 5],
            "a 20 m basin must remain a sink"
        );

        // and it changes the answer the clip depends on: with the pit open the
        // descent dies in it, with it filled the descent reaches the outline
        let mut reference = vec![false; w * h];
        for y in 0..h {
            reference[y * w + 0] = true; // outline at the low (west) edge
        }
        let start = 2 * w + 6; // immediately east of the hollow, drains into it
        let hollow = dem_from(&mk(0.5), 5.0);
        let target = outline_body(&hollow, &reference);
        let unfilled = descent_steps_to_target(&hollow.data1d, 5.0, &target, w, h);
        let z = fill_shallow_pits(&hollow.data1d, w, h, 2.0);
        let tolerant = descent_steps_to_target(&z, 5.0, &target, w, h);
        assert_eq!(
            unfilled[start],
            u32::MAX,
            "without filling the descent dies in the hollow"
        );
        assert!(
            tolerant[start] < u32::MAX,
            "with filling it spills over and reaches the outline"
        );
    }

    /// The residence gate must not move when tier 0 changes `cfl` or `ppc`,
    /// or the scoring convention would drift with the numerics being chosen.
    /// Doubling ppc doubles the raw count; halving cfl doubles the step count.
    #[test]
    fn residence_normalisation_is_invariant_to_ppc_and_cfl() {
        let gate = |count: u32, cfl: f32, ppc: u32| (count as f32) * cfl / ppc as f32;
        let base = gate(800, 0.5, 8);
        assert!((gate(1600, 0.5, 16) - base).abs() < 1e-4, "ppc doubled");
        assert!((gate(1600, 0.25, 8) - base).abs() < 1e-4, "cfl halved");
        assert!((gate(3200, 0.25, 16) - base).abs() < 1e-4, "both");
    }

    #[test]
    fn explicit_null_fields_in_case_specs_deserialize_as_defaults() {
        // the 1999 pool writes explicit nulls for its dead source columns
        let json = r#"[{"name":"aval_1","shp":"cases/aval_1.shp","area":12.0,
            "sze":3,"start_zone":null,"dpo_alt":null,"aspect":null,
            "aval_shape":1,"split":"cal"}]"#;
        let cases: Vec<CaseSpec> = serde_json::from_str(json).unwrap();
        assert_eq!(cases[0].start_zone, 0.0);
        assert_eq!(cases[0].dpo_alt, 0.0);
        assert_eq!(cases[0].aspect, "");
        assert_eq!(cases[0].area, 12.0);
        assert_eq!(cases[0].split, "cal");
    }
}
