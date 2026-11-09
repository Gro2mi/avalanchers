//! Release area estimation from an avalanche outline and a DEM.
//!
//! Two stages:
//! 1. Crown line detection: the crown line is taken as the "inflow boundary"
//!    of the outline - the outline cells that receive terrain from outside the
//!    outline. Three detection methods are available: D8 flow routing on a
//!    sink-filled DEM (CPU), a GPU particle simulation released everywhere
//!    outside the outline (see
//!    `Simulation::detect_crown_line_by_particle_simulation`), or a purely
//!    geometric gradient step: outline cells whose one-cell step uphill leaves
//!    the outline by more than a threshold. All depend only on how terrain
//!    drains into the outline, so they also work when the crown line cuts
//!    across the contours.
//! 2. Geodesic fill: chamfer-weighted geodesic distance from the crown line
//!    inside the outline mask; cells are selected by increasing distance until
//!    the target share of the outline area is reached. The fill front advances
//!    parallel to the crown line instead of following the contours. The
//!    slope-weighted variant lengthens steps through slopes outside the
//!    typical release range (see [`release_slope_membership`]), so the front
//!    advances faster through favourable release terrain.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use anyhow::{Result, bail};
use compute_core::dem::{Dem, smooth_gaussian};
use compute_core::settings::ReleaseFillMethod;
use tracing::{info, warn};

/// 8-neighborhood in clockwise order, starting at the top-left diagonal.
const NEIGHBOR_OFFSETS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Index of the offset opposite to `NEIGHBOR_OFFSETS[k]`.
const OPPOSITE: [u8; 8] = [7, 6, 5, 4, 3, 2, 1, 0];

/// True when any in-grid 8-neighbor of the cell satisfies `mask`. Off-grid
/// neighbors count as false.
pub fn any_neighbor_is(idx: usize, width: usize, height: usize, mask: &[bool]) -> bool {
    let (x, y) = (idx % width, idx / width);
    NEIGHBOR_OFFSETS.iter().any(|(dx, dy)| {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            return false;
        }
        mask[ny as usize * width + nx as usize]
    })
}

/// How far (in cells) a stopped particle is traced along its velocity to the
/// outline. The terrain normals use central differences, so they turn NaN
/// one cell before the masked DEM does, and the bilinear sampling blends in
/// another half cell: a particle crossing the boundary stops up to two cells
/// short.
const ENTRY_TRACE_CELLS: f32 = 2.0;

/// Index of the outline cell a stopped particle entered, if any. A particle
/// that stopped inside the outline counts directly. Because the bilinear
/// terrain sampling blends in the masked cells, a particle crossing the
/// boundary usually stops short, in a finite cell just outside the outline;
/// such a particle is attributed to the first outline cell along its
/// velocity. Particles whose path does not reach the outline (for example
/// particles passing along a flank, or particles stopped by NaN normals at
/// DEM edges) are not entries.
pub fn entry_cell(
    position: [f32; 2],
    velocity: [f32; 2],
    roi: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
) -> Option<usize> {
    let cell_x = (position[0] / cell_size).floor();
    let cell_y = (position[1] / cell_size).floor();
    if cell_x < 0.0 || cell_y < 0.0 || cell_x >= width as f32 || cell_y >= height as f32 {
        return None;
    }
    let (x, y) = (cell_x as usize, cell_y as usize);
    let idx = y * width + x;
    if roi[idx] {
        return Some(idx);
    }
    let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();
    if speed <= 0.0 {
        return None;
    }
    // a nearest-neighbour pick would count flank particles that slide past an
    // outline corner; the ray only hits cells the particle actually heads into
    trace_to_outline(position, velocity, speed, roi, width, height, cell_size)
}

/// First outline cell on the ray from `position` along `velocity`, up to
/// `ENTRY_TRACE_CELLS` cells away, sampled in quarter-cell steps.
fn trace_to_outline(
    position: [f32; 2],
    velocity: [f32; 2],
    speed: f32,
    roi: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
) -> Option<usize> {
    let direction = [velocity[0] / speed, velocity[1] / speed];
    let steps = (ENTRY_TRACE_CELLS * 4.0) as usize;
    (1..=steps).find_map(|step| {
        let distance = step as f32 * 0.25 * cell_size;
        let cell_x = ((position[0] + direction[0] * distance) / cell_size).floor();
        let cell_y = ((position[1] + direction[1] * distance) / cell_size).floor();
        if cell_x < 0.0 || cell_y < 0.0 || cell_x >= width as f32 || cell_y >= height as f32 {
            return None;
        }
        let idx = cell_y as usize * width + cell_x as usize;
        roi[idx].then_some(idx)
    })
}

#[derive(Debug, Clone)]
pub struct ReleaseEstimationConfig {
    /// Target release area as a fraction of the outline area (typically 0.2 - 0.3).
    pub fraction: f32,
    /// Snow thickness \[m\] written to the selected release cells.
    pub release_thickness: f32,
    /// Crown components whose mean elevation is below this share of the outline
    /// elevation range are discarded as flank gullies (0.5 keeps the top half,
    /// 0 disables the filter).
    pub crown_elevation_band: f32,
    /// Crown components whose mean slope angle \[deg\] is below this are
    /// discarded: release zones are steep, while inflow from an opposite
    /// slope onto the valley floor or into the runout lands on gentle
    /// terrain. There is deliberately no upper bound: crowns often break
    /// below rock bands steeper than any release terrain. 0 disables the
    /// filter.
    pub min_crown_slope_deg: f32,
    /// Minimum number of cells for a crown component to be kept.
    pub min_crown_component_cells: usize,
    /// Gradient \[m\] per cell applied when filling sinks. Must stay well above
    /// the f32 ulp of the elevations so filled flats still have a gradient.
    pub sink_fill_epsilon: f32,
    /// Minimum number of particles that must have entered an outline cell in
    /// the particle-simulation detection method for it to become crown line.
    /// With 4 particles per release cell, planar inflow yields about 4 entries
    /// per crown cell, so 2 filters stray single entries.
    pub min_particles_per_crown_cell: u32,
    /// Exit distance threshold of the gradient-step detection method, as a
    /// fraction of the cell size: an outline cell is a crown candidate when
    /// its one-cell step uphill lands outside the outline and farther than
    /// this from the nearest outline cell centre. Steps that leave the
    /// outline land between 0.5 and 1.0 away: a step straight out of the
    /// upstream edge lands 1.0 away, while flank cells of a spreading
    /// avalanche step out only slightly and stay below the threshold. Must be
    /// in [0, 1); values up to 0.5 count every exit.
    pub crown_exit_distance: f32,
    /// Plausibility window for the crown slope in degrees; violations are only
    /// reported as warnings.
    pub expected_crown_slope_range: (f32, f32),
    /// How the potential release cells below the crown line are ordered.
    pub fill_method: ReleaseFillMethod,
    /// Blend weight w of the slope-weighted fill: a step costs its length
    /// times 1 + w (1 - mu), so a step through a slope with membership 0 is
    /// 1 + w times as long. 0 reproduces the geodesic fill.
    pub slope_weight: f32,
    /// Slope angle \[deg\] of maximum release slope membership. 43° is the
    /// slope membership of Veitinger et al. (2016) as implemented in
    /// AutoATES v2.0 (Toft et al. 2024); human-triggered slab avalanches
    /// release at a median 38° (IQR 36-40°, Schweizer and Lütschg 2001).
    pub slope_peak_deg: f32,
    /// Half width \[deg\] of the release slope membership: 0.5 at
    /// `slope_peak_deg` +- this (11°, Veitinger et al. 2016).
    pub slope_width_deg: f32,
    /// Gaussian width \[m\] of the DEM smoothing before the slope is
    /// computed. The release slope statistics are starting-zone averages, so
    /// the slope of fine DEMs is evaluated closer to that scale; 0 uses the
    /// raw cells.
    pub slope_smoothing_length: f32,
}

impl Default for ReleaseEstimationConfig {
    fn default() -> Self {
        Self {
            fraction: 0.25,
            release_thickness: compute_core::DEFAULT_RELEASE_THICKNESS,
            crown_elevation_band: 0.0,
            min_crown_slope_deg: 28.0,
            min_crown_component_cells: 3,
            sink_fill_epsilon: 1e-2,
            min_particles_per_crown_cell: 2,
            crown_exit_distance: 0.75,
            expected_crown_slope_range: (28.0, 60.0),
            fill_method: ReleaseFillMethod::Geodesic,
            slope_weight: 3.0,
            slope_peak_deg: 43.0,
            slope_width_deg: 11.0,
            slope_smoothing_length: 10.0,
        }
    }
}

/// Exponent b of the Cauchy release slope membership (Veitinger et al. 2016):
/// higher values flatten the top and steepen the flanks of the bell.
const SLOPE_MEMBERSHIP_STEEPNESS: i32 = 4;

/// Release slope membership in [0, 1]: the generalized bell (Cauchy)
/// function `1 / (1 + |(slope - peak) / width|^(2b))` that Veitinger et al.
/// (2016) use for potential slab release areas. With the defaults (peak 43°,
/// width 11°) it is about 1 between 35° and 50°, 0.5 at 32° and 54° and
/// below 0.1 under 28° and above 58°.
pub fn release_slope_membership(slope_deg: f32, peak_deg: f32, width_deg: f32) -> f32 {
    1.0 / (1.0
        + ((slope_deg - peak_deg) / width_deg)
            .abs()
            .powi(2 * SLOPE_MEMBERSHIP_STEEPNESS))
}

/// Per-cell release slope membership, with the slope taken on the DEM
/// smoothed by `config.slope_smoothing_length`.
fn slope_memberships(dem: &Dem, config: &ReleaseEstimationConfig) -> Vec<f32> {
    let smoothed = smooth_gaussian(
        &dem.data1d,
        dem.width,
        dem.height,
        config.slope_smoothing_length / dem.cell_size,
    );
    (0..smoothed.len())
        .map(|idx| {
            let slope = slope_angle(&smoothed, dem.width, dem.height, dem.cell_size, idx, None);
            release_slope_membership(slope, config.slope_peak_deg, config.slope_width_deg)
        })
        .collect()
}

fn validate_slope_weighting(config: &ReleaseEstimationConfig) -> Result<()> {
    if !(config.slope_weight.is_finite() && config.slope_weight >= 0.0) {
        bail!("release slope weight must be finite and >= 0");
    }
    if !config.slope_peak_deg.is_finite() {
        bail!("release slope peak must be finite");
    }
    if !(config.slope_width_deg.is_finite() && config.slope_width_deg > 0.0) {
        bail!("release slope width must be finite and > 0");
    }
    if !(config.slope_smoothing_length.is_finite() && config.slope_smoothing_length >= 0.0) {
        bail!("release slope smoothing length must be finite and >= 0");
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ReleaseEstimate {
    /// Snow thickness per cell, 0 outside the release area. Same length and
    /// indexing as the DEM (`y * width + x`).
    pub release_areas: Vec<f32>,
    /// Mask marking the detected crown line cells.
    pub crown_line: Vec<bool>,
    /// Mask marking the crown candidates before component filtering.
    pub crown_candidates: Vec<bool>,
    pub number_roi_cells: usize,
    pub number_release_cells: usize,
    pub number_crown_cells: usize,
    pub number_crown_components: usize,
    pub mean_crown_slope_deg: f32,
    pub release_fraction: f32,
}

/// Total order on finite f32 distances so they can key a `BinaryHeap`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cost(f32);

impl Eq for Cost {}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority-flood depression filling (Barnes et al.): every cell is raised to
/// the minimum elevation that keeps a monotonically descending path to the grid
/// border, plus `epsilon` per cell. NaN cells act as barriers.
fn fill_sinks(z: &[f32], width: usize, height: usize, epsilon: f32) -> Vec<f32> {
    let mut filled = z.to_vec();
    let mut visited = vec![false; z.len()];
    let mut heap = BinaryHeap::new();

    for y in 0..height {
        for x in 0..width {
            if x != 0 && y != 0 && x + 1 != width && y + 1 != height {
                continue;
            }
            let idx = y * width + x;
            if filled[idx].is_finite() {
                visited[idx] = true;
                heap.push(Reverse((Cost(filled[idx]), idx)));
            }
        }
    }

    while let Some(Reverse((Cost(elevation), idx))) = heap.pop() {
        let (x, y) = (idx % width, idx / width);
        for (dx, dy) in NEIGHBOR_OFFSETS {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let nidx = ny as usize * width + nx as usize;
            if visited[nidx] || !filled[nidx].is_finite() {
                continue;
            }
            visited[nidx] = true;
            if filled[nidx] < elevation + epsilon {
                filled[nidx] = elevation + epsilon;
            }
            heap.push(Reverse((Cost(filled[nidx]), nidx)));
        }
    }

    filled
}

/// D8 flow direction per cell: index into [`NEIGHBOR_OFFSETS`] of the steepest
/// downslope neighbor. `None` for no-data cells, grid-border sinks and residual
/// flats (rare after sink filling).
fn d8_flow_directions(z: &[f32], width: usize, height: usize, cell_size: f32) -> Vec<Option<u8>> {
    let diagonal_distance = cell_size * std::f32::consts::SQRT_2;
    let mut flow = vec![None; z.len()];

    for idx in 0..z.len() {
        let z0 = z[idx];
        if !z0.is_finite() {
            continue;
        }
        let (x, y) = (idx % width, idx / width);
        let mut best = None;
        let mut best_slope = 0.0f32;
        for (k, (dx, dy)) in NEIGHBOR_OFFSETS.iter().enumerate() {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let zn = z[ny as usize * width + nx as usize];
            if !zn.is_finite() {
                continue;
            }
            let distance = if dx != &0 && dy != &0 {
                diagonal_distance
            } else {
                cell_size
            };
            let slope = (z0 - zn) / distance;
            if slope > best_slope {
                best_slope = slope;
                best = Some(k as u8);
            }
        }
        flow[idx] = best;
    }

    flow
}

/// Result of a crown line detection, shared by both detection methods.
#[derive(Debug, Clone)]
pub struct CrownDetection {
    /// Crown cells after component filtering.
    pub crown: Vec<bool>,
    /// Crown candidates before component filtering; shows what the size,
    /// slope and elevation-band filters removed.
    pub candidates: Vec<bool>,
    /// Number of crown cells after filtering.
    pub number_crown_cells: usize,
    /// Number of connected components kept.
    pub number_components: usize,
    /// Crown candidates before component filtering.
    pub number_candidates: usize,
}

/// Terrain metrics precomputed by the terrain analysis shaders.
pub(crate) struct TerrainMetrics<'a> {
    pub slope_angles: &'a [f32],
    /// Elevation gradient `dz/dx` per meter.
    pub gradient_x: &'a [f32],
    /// Elevation gradient `dz/dy` per meter.
    pub gradient_y: &'a [f32],
}

fn validate_terrain_metrics(metrics: &TerrainMetrics<'_>, number_cells: usize) -> Result<()> {
    if metrics.slope_angles.len() != number_cells
        || metrics.gradient_x.len() != number_cells
        || metrics.gradient_y.len() != number_cells
    {
        bail!(
            "terrain metric lengths ({}, {}, {}) do not match the DEM ({number_cells})",
            metrics.slope_angles.len(),
            metrics.gradient_x.len(),
            metrics.gradient_y.len()
        );
    }
    Ok(())
}

/// Groups crown candidate cells into connected components (8-connectivity) and
/// filters them by size, by mean slope (see
/// [`ReleaseEstimationConfig::min_crown_slope_deg`]) and by mean elevation, so
/// stray cells, inflow onto gentle valley floors or runouts and flank gullies
/// drop out. Falls back to the largest component when nothing survives the
/// filters.
#[allow(clippy::too_many_arguments)]
fn filter_crown_components(
    candidate: Vec<bool>,
    elevations: &[f32],
    roi: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> CrownDetection {
    let number_candidates = candidate.iter().filter(|&&c| c).count();

    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut visited = vec![false; roi.len()];
    for seed in 0..roi.len() {
        if !candidate[seed] || visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![seed];
        visited[seed] = true;
        while let Some(idx) = stack.pop() {
            component.push(idx);
            let (x, y) = (idx % width, idx / width);
            for (dx, dy) in NEIGHBOR_OFFSETS {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if candidate[nidx] && !visited[nidx] {
                    visited[nidx] = true;
                    stack.push(nidx);
                }
            }
        }
        components.push(component);
    }

    let mut roi_min = f32::INFINITY;
    let mut roi_max = f32::NEG_INFINITY;
    for (idx, &inside) in roi.iter().enumerate() {
        if inside {
            roi_min = roi_min.min(elevations[idx]);
            roi_max = roi_max.max(elevations[idx]);
        }
    }
    let threshold = roi_min + config.crown_elevation_band * (roi_max - roi_min);

    let mut number_slope_rejected = 0;
    let mut kept: Vec<&Vec<usize>> = components
        .iter()
        .filter(|component| component.len() >= config.min_crown_component_cells)
        .filter(|component| {
            let mean_slope = component
                .iter()
                .map(|&idx| slope_angle(elevations, width, height, cell_size, idx, terrain_metrics))
                .sum::<f32>()
                / component.len() as f32;
            let keep = mean_slope >= config.min_crown_slope_deg;
            if !keep {
                number_slope_rejected += 1;
            }
            keep
        })
        .filter(|component| {
            let mean_elevation =
                component.iter().map(|&idx| elevations[idx]).sum::<f32>() / component.len() as f32;
            mean_elevation >= threshold
        })
        .collect();
    if number_slope_rejected > 0 {
        info!(
            "{number_slope_rejected} crown component(s) discarded: flatter than {:.0}°",
            config.min_crown_slope_deg
        );
    }
    if kept.is_empty() && !components.is_empty() {
        warn!("no crown component passed the filters, keeping the largest");
        kept = vec![components.iter().max_by_key(|c| c.len()).unwrap()];
    }

    let mut crown = vec![false; roi.len()];
    for component in &kept {
        for &idx in component.iter() {
            crown[idx] = true;
        }
    }
    let number_crown_cells = crown.iter().filter(|&&c| c).count();

    CrownDetection {
        crown,
        candidates: candidate,
        number_crown_cells,
        number_components: kept.len(),
        number_candidates,
    }
}

/// Detects the crown line as the outline's inflow boundary: outline cells that
/// are the D8 receiver of at least one cell outside the outline.
#[allow(clippy::too_many_arguments)]
fn detect_crown_line_from_flow(
    elevations: &[f32],
    flow: &[Option<u8>],
    roi: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> CrownDetection {
    let mut candidate = vec![false; roi.len()];
    for idx in 0..roi.len() {
        if !roi[idx] {
            continue;
        }
        let (x, y) = (idx % width, idx / width);
        for (k, (dx, dy)) in NEIGHBOR_OFFSETS.iter().enumerate() {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let nidx = ny as usize * width + nx as usize;
            if roi[nidx] {
                continue;
            }
            // neighbor outside the outline drains into this cell
            if flow[nidx] == Some(OPPOSITE[k]) {
                candidate[idx] = true;
            }
        }
    }
    filter_crown_components(
        candidate,
        elevations,
        roi,
        width,
        height,
        cell_size,
        config,
        terrain_metrics,
    )
}

/// Elevation gradient per cell (not per meter). Uses central differences,
/// falling back to one-sided differences at the grid border or next to
/// no-data cells. `None` for no-data cells.
fn elevation_gradient(z: &[f32], width: usize, height: usize, idx: usize) -> Option<[f32; 2]> {
    let z0 = z[idx];
    if !z0.is_finite() {
        return None;
    }
    let (x, y) = ((idx % width) as isize, (idx / width) as isize);
    let sample = |nx: isize, ny: isize| -> Option<f32> {
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            return None;
        }
        let zn = z[ny as usize * width + nx as usize];
        zn.is_finite().then_some(zn)
    };
    let derivative = |minus: Option<f32>, plus: Option<f32>| match (minus, plus) {
        (Some(m), Some(p)) => (p - m) / 2.0,
        (None, Some(p)) => p - z0,
        (Some(m), None) => z0 - m,
        (None, None) => 0.0,
    };
    Some([
        derivative(sample(x - 1, y), sample(x + 1, y)),
        derivative(sample(x, y - 1), sample(x, y + 1)),
    ])
}

/// Normalized uphill direction in cell units. `None` for no-data cells and
/// flats.
fn uphill_direction(
    z: &[f32],
    width: usize,
    height: usize,
    idx: usize,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> Option<[f32; 2]> {
    let [gx, gy] = match terrain_metrics {
        Some(metrics) => [metrics.gradient_x[idx], metrics.gradient_y[idx]],
        None => elevation_gradient(z, width, height, idx)?,
    };
    let norm = (gx * gx + gy * gy).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return None;
    }
    Some([gx / norm, gy / norm])
}

/// Slope angle in degrees from terrain analysis, or the elevation gradient
/// when no precomputed terrain metrics are available.
fn slope_angle(
    z: &[f32],
    width: usize,
    height: usize,
    cell_size: f32,
    idx: usize,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> f32 {
    if let Some(metrics) = terrain_metrics {
        return if metrics.slope_angles[idx].is_finite() {
            metrics.slope_angles[idx]
        } else {
            0.0
        };
    }
    elevation_gradient(z, width, height, idx)
        .map(|[gx, gy]| ((gx * gx + gy * gy).sqrt() / cell_size).atan().to_degrees())
        .filter(|angle| angle.is_finite())
        .unwrap_or(0.0)
}

/// Detects the crown line geometrically: every outline cell steps one cell
/// length uphill from its centre, and cells whose stepped point lands outside
/// the outline and farther than `config.crown_exit_distance` (in cells) from
/// the nearest outline cell centre are crown candidates. The threshold keeps flank cells of a
/// spreading avalanche, which step out of the outline only slightly, from
/// becoming crown line.
fn detect_crown_line_from_gradient(
    elevations: &[f32],
    roi: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> CrownDetection {
    let mut candidate = vec![false; roi.len()];
    for idx in 0..roi.len() {
        if !roi[idx] {
            continue;
        }
        let Some(direction) = uphill_direction(elevations, width, height, idx, terrain_metrics)
        else {
            continue;
        };
        let (x, y) = (idx % width, idx / width);
        let point = [x as f32 + 0.5 + direction[0], y as f32 + 0.5 + direction[1]];
        let (px, py) = (point[0].floor() as isize, point[1].floor() as isize);
        let inside_grid = px >= 0 && py >= 0 && px < width as isize && py < height as isize;
        if inside_grid && roi[py as usize * width + px as usize] {
            continue; // the step stays inside the outline
        }
        // the origin cell is 1.0 away, so the nearest outline centre always
        // lies in the 3x3 window around the stepped point's cell
        let mut nearest = f32::INFINITY;
        for (dx, dy) in NEIGHBOR_OFFSETS.iter().copied().chain([(0, 0)]) {
            let (nx, ny) = (px + dx, py + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            if !roi[ny as usize * width + nx as usize] {
                continue;
            }
            let (ex, ey) = (nx as f32 + 0.5 - point[0], ny as f32 + 0.5 - point[1]);
            nearest = nearest.min((ex * ex + ey * ey).sqrt());
        }
        candidate[idx] = nearest > config.crown_exit_distance;
    }
    filter_crown_components(
        candidate,
        elevations,
        roi,
        width,
        height,
        cell_size,
        config,
        terrain_metrics,
    )
}

/// Chamfer-weighted geodesic distance (Dijkstra) from the seed cells, only
/// traversing `traversable` cells. With `cost_factors`, each step is scaled
/// by the mean factor of the two cells it connects. Unreachable cells stay at
/// infinity.
fn geodesic_distance_from(
    seeds: &[bool],
    traversable: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
    cost_factors: Option<&[f32]>,
) -> Vec<f32> {
    let diagonal_distance = cell_size * std::f32::consts::SQRT_2;
    let mut distance = vec![f32::INFINITY; seeds.len()];
    let mut heap = BinaryHeap::new();
    for (idx, &is_seed) in seeds.iter().enumerate() {
        if is_seed && traversable[idx] {
            distance[idx] = 0.0;
            heap.push(Reverse((Cost(0.0), idx)));
        }
    }

    while let Some(Reverse((Cost(d0), idx))) = heap.pop() {
        if d0 > distance[idx] {
            continue; // stale heap entry
        }
        let (x, y) = (idx % width, idx / width);
        for (dx, dy) in NEIGHBOR_OFFSETS {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let nidx = ny as usize * width + nx as usize;
            if !traversable[nidx] {
                continue;
            }
            let mut weight = if dx != 0 && dy != 0 {
                diagonal_distance
            } else {
                cell_size
            };
            if let Some(factors) = cost_factors {
                weight *= 0.5 * (factors[idx] + factors[nidx]);
            }
            let candidate_distance = d0 + weight;
            if candidate_distance < distance[nidx] {
                distance[nidx] = candidate_distance;
                heap.push(Reverse((Cost(candidate_distance), nidx)));
            }
        }
    }

    distance
}

/// Crown slope angle from terrain analysis, or steepest descent on the raw
/// elevations when no precomputed terrain metrics are available.
fn steepest_descent_angle(
    z: &[f32],
    width: usize,
    height: usize,
    cell_size: f32,
    idx: usize,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> f32 {
    if let Some(metrics) = terrain_metrics {
        return if metrics.slope_angles[idx].is_finite() {
            metrics.slope_angles[idx]
        } else {
            0.0
        };
    }

    let z0 = z[idx];
    if !z0.is_finite() {
        return 0.0;
    }
    let (x, y) = (idx % width, idx / width);
    let diagonal_distance = cell_size * std::f32::consts::SQRT_2;
    let mut max_slope = 0.0f32;
    for (dx, dy) in NEIGHBOR_OFFSETS {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            continue;
        }
        let zn = z[ny as usize * width + nx as usize];
        if !zn.is_finite() {
            continue;
        }
        let distance = if dx != 0 && dy != 0 {
            diagonal_distance
        } else {
            cell_size
        };
        max_slope = max_slope.max((z0 - zn) / distance);
    }
    max_slope.atan().to_degrees()
}

/// Validates the outline against the DEM and returns the number of outline
/// cells.
fn validate_roi(dem: &Dem, roi: &[bool]) -> Result<usize> {
    let number_cells = dem.width.saturating_mul(dem.height);
    if roi.len() != number_cells {
        bail!(
            "outline length ({}) does not match DEM dimensions ({}x{})",
            roi.len(),
            dem.width,
            dem.height
        );
    }
    let number_roi_cells = roi.iter().filter(|&&inside| inside).count();
    if number_roi_cells == 0 {
        bail!("outline is empty");
    }
    if number_roi_cells == roi.len() {
        bail!(
            "outline covers the entire DEM; crown line detection requires cells outside the outline"
        );
    }
    Ok(number_roi_cells)
}

/// Detects the crown line of the avalanche outline with D8 flow routing on a
/// sink-filled DEM. Fails if no terrain outside the outline drains into it.
pub fn detect_crown_line(
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
) -> Result<CrownDetection> {
    detect_crown_line_impl(dem, roi, config, None)
}

pub(crate) fn detect_crown_line_with_terrain_metrics(
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
    terrain_metrics: &TerrainMetrics<'_>,
) -> Result<CrownDetection> {
    validate_terrain_metrics(terrain_metrics, dem.width.saturating_mul(dem.height))?;
    detect_crown_line_impl(dem, roi, config, Some(terrain_metrics))
}

fn detect_crown_line_impl(
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> Result<CrownDetection> {
    validate_roi(dem, roi)?;
    info!(
        "Crown line estimation: filling sinks on {}x{} grid",
        dem.width, dem.height
    );
    let filled = fill_sinks(&dem.data1d, dem.width, dem.height, config.sink_fill_epsilon);
    let flow = d8_flow_directions(&filled, dem.width, dem.height, dem.cell_size);
    let detection = detect_crown_line_from_flow(
        &dem.data1d,
        &flow,
        roi,
        dem.width,
        dem.height,
        dem.cell_size,
        config,
        terrain_metrics,
    );
    check_detection(&detection)?;
    Ok(detection)
}

/// Detects the crown line of the avalanche outline with the gradient-step
/// method (see `ReleaseEstimationConfig::crown_exit_distance`). Fails if no
/// outline cell steps out of the outline by more than the threshold.
pub fn detect_crown_line_by_gradient_step(
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
) -> Result<CrownDetection> {
    detect_crown_line_by_gradient_step_impl(dem, roi, config, None)
}

pub(crate) fn detect_crown_line_by_gradient_step_with_terrain_metrics(
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
    terrain_metrics: &TerrainMetrics<'_>,
) -> Result<CrownDetection> {
    validate_terrain_metrics(terrain_metrics, dem.width.saturating_mul(dem.height))?;
    detect_crown_line_by_gradient_step_impl(dem, roi, config, Some(terrain_metrics))
}

fn detect_crown_line_by_gradient_step_impl(
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> Result<CrownDetection> {
    validate_roi(dem, roi)?;
    if !(0.0..1.0).contains(&config.crown_exit_distance) {
        bail!("crown exit distance must be in [0, 1) cells");
    }
    info!(
        "Crown line estimation: gradient step on {}x{} grid (exit distance {:.2} cells)",
        dem.width, dem.height, config.crown_exit_distance
    );
    let detection = detect_crown_line_from_gradient(
        &dem.data1d,
        roi,
        dem.width,
        dem.height,
        dem.cell_size,
        config,
        terrain_metrics,
    );
    check_detection(&detection)?;
    Ok(detection)
}

/// Derives the crown line from per-cell counts of particles that entered the
/// outline in a simulation on a DEM where the outline was masked out (see
/// `Simulation::detect_crown_line_by_particle_simulation` for how entries are
/// counted). Outline cells whose entry count reaches
/// `min_particles_per_crown_cell` are candidates.
pub fn crown_line_from_particle_counts(
    counts: &[u32],
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
) -> Result<CrownDetection> {
    crown_line_from_particle_counts_impl(counts, dem, roi, config, None)
}

pub(crate) fn crown_line_from_particle_counts_with_terrain_metrics(
    counts: &[u32],
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
    terrain_metrics: &TerrainMetrics<'_>,
) -> Result<CrownDetection> {
    validate_terrain_metrics(terrain_metrics, dem.width.saturating_mul(dem.height))?;
    crown_line_from_particle_counts_impl(counts, dem, roi, config, Some(terrain_metrics))
}

fn crown_line_from_particle_counts_impl(
    counts: &[u32],
    dem: &Dem,
    roi: &[bool],
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> Result<CrownDetection> {
    validate_roi(dem, roi)?;
    let number_cells = dem.width.saturating_mul(dem.height);
    if counts.len() != number_cells {
        bail!(
            "particle count length ({}) does not match DEM dimensions ({}x{})",
            counts.len(),
            dem.width,
            dem.height
        );
    }
    let candidate: Vec<bool> = (0..number_cells)
        .map(|idx| roi[idx] && counts[idx] >= config.min_particles_per_crown_cell)
        .collect();
    let detection = filter_crown_components(
        candidate,
        &dem.data1d,
        roi,
        dem.width,
        dem.height,
        dem.cell_size,
        config,
        terrain_metrics,
    );
    check_detection(&detection)?;
    Ok(detection)
}

fn validate_potential(potential: Option<&[bool]>, roi: &[bool]) -> Result<()> {
    if let Some(potential) = potential
        && potential.len() != roi.len()
    {
        bail!(
            "potential release area length ({}) does not match the outline ({})",
            potential.len(),
            roi.len()
        );
    }
    Ok(())
}

fn check_detection(detection: &CrownDetection) -> Result<()> {
    if detection.number_candidates == 0 {
        bail!("no crown line detected: no terrain outside the outline drains into the outline");
    }
    Ok(())
}

/// Fills the target share of the outline area below the given crown line.
/// Distances are measured through the whole outline; with potential release
/// areas only those cells are selected, nearest to the crown first, so terrain
/// that cannot release (too flat, too steep, too rough) is skipped without
/// cutting off the release terrain behind it. With
/// [`ReleaseFillMethod::SlopeWeighted`], steps through unfavourable release
/// slopes count as longer, so the fill reaches further into favourable
/// terrain and less far across benches and rock bands.
///
/// Fails if the outline is empty, covers the whole DEM (crown line detection
/// needs cells outside the outline), if no inflow boundary exists or if no
/// potential release cell is reachable from the crown line.
pub fn estimate_release_areas_with_crown(
    dem: &Dem,
    roi: &[bool],
    potential: Option<&[bool]>,
    crown: &CrownDetection,
    config: &ReleaseEstimationConfig,
) -> Result<ReleaseEstimate> {
    estimate_release_areas_with_crown_impl(dem, roi, potential, crown, config, None)
}

pub(crate) fn estimate_release_areas_with_crown_and_terrain_metrics(
    dem: &Dem,
    roi: &[bool],
    potential: Option<&[bool]>,
    crown: &CrownDetection,
    config: &ReleaseEstimationConfig,
    terrain_metrics: &TerrainMetrics<'_>,
) -> Result<ReleaseEstimate> {
    validate_terrain_metrics(terrain_metrics, dem.width.saturating_mul(dem.height))?;
    estimate_release_areas_with_crown_impl(
        dem,
        roi,
        potential,
        crown,
        config,
        Some(terrain_metrics),
    )
}

fn estimate_release_areas_with_crown_impl(
    dem: &Dem,
    roi: &[bool],
    potential: Option<&[bool]>,
    crown: &CrownDetection,
    config: &ReleaseEstimationConfig,
    terrain_metrics: Option<&TerrainMetrics<'_>>,
) -> Result<ReleaseEstimate> {
    let width = dem.width;
    let height = dem.height;
    let number_roi_cells = validate_roi(dem, roi)?;
    check_detection(crown)?;
    if !(0.0..=1.0).contains(&config.fraction) {
        bail!("release area fraction must be in [0, 1]");
    }

    info!(
        "Crown line detected: {} cells in {} component(s)",
        crown.number_crown_cells, crown.number_components
    );

    let memberships = match config.fill_method {
        ReleaseFillMethod::Geodesic => None,
        ReleaseFillMethod::SlopeWeighted => {
            validate_slope_weighting(config)?;
            Some(slope_memberships(dem, config))
        }
    };
    // a step through a slope with membership mu is 1 + w (1 - mu) times as long
    let cost_factors: Option<Vec<f32>> = memberships.as_ref().map(|memberships| {
        memberships
            .iter()
            .map(|membership| 1.0 + config.slope_weight * (1.0 - membership))
            .collect()
    });
    let distance = geodesic_distance_from(
        &crown.crown,
        roi,
        width,
        height,
        dem.cell_size,
        cost_factors.as_deref(),
    );

    validate_potential(potential, roi)?;
    let selectable = |idx: usize| roi[idx] && potential.is_none_or(|p| p[idx]);
    let number_selectable = (0..roi.len()).filter(|&idx| selectable(idx)).count();

    let mut ordered: Vec<(f32, usize)> = (0..roi.len())
        .filter(|&idx| selectable(idx) && distance[idx].is_finite())
        .map(|idx| (distance[idx], idx))
        .collect();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    if ordered.is_empty() {
        bail!(
            "no potential release cell of the outline is reachable from the crown line; \
             check min_slope_angle, max_slope_angle, release_min_elevation, \
             release_max_elevation and roughness_threshold"
        );
    }

    if ordered.len() < number_selectable {
        warn!(
            "{} of {} {} cells are unreachable from the crown line and can never be selected",
            number_selectable - ordered.len(),
            number_selectable,
            if potential.is_some() {
                "potential release"
            } else {
                "outline"
            }
        );
    }
    let requested = ((config.fraction * number_roi_cells as f32).round() as usize).max(1);
    if requested > ordered.len() {
        warn!(
            "release target of {requested} cells exceeds the {} reachable potential release \
             cells; selecting all of them",
            ordered.len()
        );
    }
    let target = requested.min(ordered.len());

    let mut release_areas = vec![0.0f32; roi.len()];
    for &(_, idx) in &ordered[..target] {
        release_areas[idx] = config.release_thickness;
    }

    let crown_indices: Vec<usize> = crown
        .crown
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c)
        .map(|(idx, _)| idx)
        .collect();
    let mean_crown_slope_deg = if crown_indices.is_empty() {
        0.0
    } else {
        crown_indices
            .iter()
            .map(|&idx| {
                steepest_descent_angle(
                    &dem.data1d,
                    width,
                    height,
                    dem.cell_size,
                    idx,
                    terrain_metrics,
                )
            })
            .sum::<f32>()
            / crown_indices.len() as f32
    };
    if mean_crown_slope_deg < config.expected_crown_slope_range.0
        || mean_crown_slope_deg > config.expected_crown_slope_range.1
    {
        warn!(
            "mean crown slope {:.1}° is outside the expected release window {:.0}°-{:.0}°; \
             the outline may not include the release zone",
            mean_crown_slope_deg,
            config.expected_crown_slope_range.0,
            config.expected_crown_slope_range.1
        );
    }

    info!(
        "Release fill ({}): {} of {} outline cells ({:.0}%), mean crown slope {:.1}°",
        config.fill_method,
        target,
        number_roi_cells,
        target as f32 / number_roi_cells as f32 * 100.0,
        mean_crown_slope_deg
    );
    if let Some(memberships) = &memberships {
        let mean_membership = |cells: &[(f32, usize)]| {
            cells.iter().map(|&(_, idx)| memberships[idx]).sum::<f32>() / cells.len() as f32
        };
        info!(
            "Slope-weighted fill (w {:.1}, peak {:.0}°, width {:.0}°): mean release slope \
             membership {:.2} of the selected cells, {:.2} of all reachable candidates",
            config.slope_weight,
            config.slope_peak_deg,
            config.slope_width_deg,
            mean_membership(&ordered[..target]),
            mean_membership(&ordered)
        );
    }

    Ok(ReleaseEstimate {
        release_areas,
        crown_line: crown.crown.clone(),
        crown_candidates: crown.candidates.clone(),
        number_roi_cells,
        number_release_cells: target,
        number_crown_cells: crown.number_crown_cells,
        number_crown_components: crown.number_components,
        mean_crown_slope_deg,
        release_fraction: target as f32 / number_roi_cells as f32,
    })
}

/// Estimates release areas by detecting the crown line of the avalanche
/// outline with flow routing and filling the target share of the outline area
/// below it.
pub fn estimate_release_areas(
    dem: &Dem,
    roi: &[bool],
    potential: Option<&[bool]>,
    config: &ReleaseEstimationConfig,
) -> Result<ReleaseEstimate> {
    let crown = detect_crown_line(dem, roi, config)?;
    estimate_release_areas_with_crown(dem, roi, potential, &crown, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute_core::dem::Bounds;
    use compute_core::utils::to_2d;

    fn plane_dem(width: usize, height: usize, cell_size: f32) -> Dem {
        let mut dem = Dem::default();
        dem.width = width;
        dem.height = height;
        dem.cell_size = cell_size;
        dem.bounds = Bounds {
            xmin: 0.0,
            xmax: width as f32 * cell_size,
            ymin: 0.0,
            ymax: height as f32 * cell_size,
        };
        // 45° plane rising with y, steep enough for the crown slope filter
        dem.data1d = (0..width * height)
            .map(|idx| (idx / width) as f32 * cell_size)
            .collect();
        dem.data = to_2d(&dem.data1d, width, height);
        dem
    }

    fn rectangle_roi(
        width: usize,
        height: usize,
        x0: usize,
        x1: usize,
        y0: usize,
        y1: usize,
    ) -> Vec<bool> {
        (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                x >= x0 && x < x1 && y >= y0 && y < y1
            })
            .collect()
    }

    #[test]
    fn fill_sinks_raises_pit_to_spill_level() {
        // plane z = y with a pit at the center
        let width = 5;
        let height = 5;
        let mut z: Vec<f32> = (0..width * height)
            .map(|idx| (idx / width) as f32)
            .collect();
        z[2 * width + 2] = -5.0;
        let filled = fill_sinks(&z, width, height, 1e-2);
        let center = 2 * width + 2;
        assert!(filled[center] > -5.0, "pit was not filled");
        assert!(filled[center] >= 1.0, "pit not raised to its surroundings");
    }

    #[test]
    fn d8_on_inclined_plane_points_downslope() {
        let width = 10;
        let height = 10;
        let dem = plane_dem(width, height, 1.0);
        let flow = d8_flow_directions(&dem.data1d, width, height, 1.0);
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                // elevation increases with y, so flow must be (0, -1) = index 1
                assert_eq!(flow[idx], Some(1), "at ({x},{y})");
            }
        }
    }

    #[test]
    fn crown_line_is_upstream_edge_of_rectangle() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig::default();
        let filled = fill_sinks(&dem.data1d, width, height, config.sink_fill_epsilon);
        let flow = d8_flow_directions(&filled, width, height, dem.cell_size);
        let detection = detect_crown_line_from_flow(
            &dem.data1d,
            &flow,
            &roi,
            width,
            height,
            dem.cell_size,
            &config,
            None,
        );

        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            // upstream edge is the row y = 14 (its outside neighbors at y = 15
            // drain towards decreasing y)
            let expected = y == 14 && (5..15).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
        assert_eq!(detection.number_components, 1);
    }

    #[test]
    fn gradient_step_uses_precomputed_terrain_metrics() {
        let width = 20;
        let height = 20;
        let dem = Dem {
            width,
            height,
            cell_size: 10.0,
            data1d: vec![100.0; width * height],
            ..Default::default()
        };
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let slope_angles = vec![45.0; width * height];
        let gradient_x = vec![0.0; width * height];
        let gradient_y = vec![1.0; width * height];
        let metrics = TerrainMetrics {
            slope_angles: &slope_angles,
            gradient_x: &gradient_x,
            gradient_y: &gradient_y,
        };
        let config = ReleaseEstimationConfig::default();

        let detection =
            detect_crown_line_by_gradient_step_with_terrain_metrics(&dem, &roi, &config, &metrics)
                .unwrap();

        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            assert_eq!(is_crown, y == 14 && (5..15).contains(&x), "at ({x},{y})");
        }
    }

    #[test]
    fn slope_filter_drops_inflow_onto_flat_valley_floor() {
        // 45° slope above y = 8, a ~6° valley floor below; inflow onto the
        // floor is not a crown even with the elevation band disabled
        let width = 20;
        let height = 20;
        let mut dem = plane_dem(width, height, 10.0);
        dem.data1d = (0..width * height)
            .map(|idx| {
                let y = (idx / width) as f32;
                if y >= 8.0 { y * 10.0 } else { 80.0 + (y - 8.0) }
            })
            .collect();
        dem.data = to_2d(&dem.data1d, width, height);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let mut counts = vec![0u32; width * height];
        for x in 5..15 {
            counts[14 * width + x] = 4;
        }
        for x in 8..12 {
            counts[5 * width + x] = 4;
        }

        let config = ReleaseEstimationConfig::default();
        let detection = crown_line_from_particle_counts(&counts, &dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = y == 14 && (5..15).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
        assert_eq!(detection.number_candidates, 14);

        // with the filter disabled the valley-floor component survives
        let config = ReleaseEstimationConfig {
            min_crown_slope_deg: 0.0,
            ..Default::default()
        };
        let detection = crown_line_from_particle_counts(&counts, &dem, &roi, &config).unwrap();
        assert_eq!(detection.number_components, 2);
        assert_eq!(detection.number_crown_cells, 14);
    }

    #[test]
    fn crown_line_from_particle_counts_filters_strays_and_flank_gullies() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig {
            crown_elevation_band: 0.5,
            ..Default::default()
        };
        let mut counts = vec![0u32; width * height];
        // upstream row y = 14 fed by ~4 entering particles per cell
        for x in 5..15 {
            counts[14 * width + x] = 4;
        }
        // single stray entry below the particle threshold
        counts[7 * width + 7] = 1;
        // flank gully spike: many particles funnelling in low on the flank
        for y in 6..9 {
            counts[y * width + 14] = 50;
        }

        let detection = crown_line_from_particle_counts(&counts, &dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = y == 14 && (5..15).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
        assert_eq!(detection.number_crown_cells, 10);
        assert_eq!(detection.number_components, 1);
        assert_eq!(detection.number_candidates, 13); // 10 crown + 3 gully cells
        // the dropped gully cells stay visible in the candidate mask
        for y in 6..9 {
            assert!(detection.candidates[y * width + 14]);
            assert!(!detection.crown[y * width + 14]);
        }
        assert_eq!(detection.candidates.iter().filter(|&&c| c).count(), 13);
    }

    #[test]
    fn release_fill_hits_exact_fraction_below_crown() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig {
            fraction: 0.25,
            release_thickness: 1.5,
            ..Default::default()
        };
        let estimate = estimate_release_areas(&dem, &roi, None, &config).unwrap();

        assert_eq!(estimate.number_release_cells, 25); // 25% of 100 cells
        assert_eq!(estimate.number_crown_cells, 10);
        for (idx, &thickness) in estimate.release_areas.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            if !roi[idx] {
                assert_eq!(thickness, 0.0, "release outside outline at ({x},{y})");
                continue;
            }
            // fill front is at 2.5 cells below the crown row y = 14: rows 14
            // and 13 complete, then the first 5 cells of row 12
            let expected = if y >= 13 || (y == 12 && x < 10) {
                1.5
            } else {
                0.0
            };
            assert_eq!(thickness, expected, "at ({x},{y})");
        }
    }

    #[test]
    fn oblique_crown_fill_follows_crown_not_contours() {
        // diagonal band on a plane; the crown line is the x+y=30 diagonal even
        // though it crosses every contour of the plane
        let width = 40;
        let height = 40;
        let dem = plane_dem(width, height, 1.0);
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                let s = x + y;
                (10..=30).contains(&s)
            })
            .collect();
        let config = ReleaseEstimationConfig {
            fraction: 0.25,
            crown_elevation_band: 0.25,
            ..Default::default()
        };
        let estimate = estimate_release_areas(&dem, &roi, None, &config).unwrap();

        for (idx, &is_crown) in estimate.crown_line.iter().enumerate() {
            if is_crown {
                let (x, y) = (idx % width, idx / width);
                assert_eq!(x + y, 30, "crown cell off the upstream diagonal");
            }
        }
        assert_eq!(estimate.number_crown_cells, 31);

        // lowest cell of the outline sits on the crown line and must be
        // released; the highest-elevation cell on the far diagonal must not
        let low_on_crown = 30; // row 0, column 30
        let high_far_from_crown = 10 * width; // row 10, column 0
        assert_eq!(estimate.release_areas[low_on_crown], 1.0);
        assert_eq!(estimate.release_areas[high_far_from_crown], 0.0);
        assert_eq!(estimate.number_release_cells, 110); // 25% of 441 cells
    }

    #[test]
    fn gradient_step_crown_is_upstream_edge_of_rectangle() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig::default();
        let detection = detect_crown_line_by_gradient_step(&dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = y == 14 && (5..15).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
        assert_eq!(detection.number_components, 1);
    }

    #[test]
    fn gradient_step_threshold_separates_flank_from_crown() {
        // plane z = 2x + 3y: the uphill direction (2, 3)/sqrt(13) leaves the
        // right flank of the rectangle ~0.58-0.65 cells from the outline, the
        // upstream edge ~0.94-1.0 cells
        let width = 20;
        let height = 20;
        let mut dem = plane_dem(width, height, 10.0);
        dem.data1d = (0..width * height)
            .map(|idx| (2.0 * (idx % width) as f32 + 3.0 * (idx / width) as f32) * 10.0)
            .collect();
        dem.data = to_2d(&dem.data1d, width, height);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);

        let config = ReleaseEstimationConfig::default();
        let detection = detect_crown_line_by_gradient_step(&dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = y == 14 && (5..15).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }

        // a lower threshold also picks up the right flank
        let config = ReleaseEstimationConfig {
            crown_exit_distance: 0.5,
            ..Default::default()
        };
        let detection = detect_crown_line_by_gradient_step(&dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = roi[idx] && (y == 14 || x == 14);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
    }

    #[test]
    fn gradient_step_drops_isolated_stair_cells_of_spreading_outline() {
        // outline widening downslope by one cell per side every three rows;
        // the stair cells step out of the outline but stay isolated
        let width = 30;
        let height = 30;
        let dem = plane_dem(width, height, 10.0);
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                if !(5..20).contains(&y) {
                    return false;
                }
                let widening = (19 - y) / 3;
                (10 - widening..20 + widening).contains(&x)
            })
            .collect();
        let config = ReleaseEstimationConfig::default();
        let detection = detect_crown_line_by_gradient_step(&dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            let expected = y == 19 && (10..20).contains(&x);
            assert_eq!(is_crown, expected, "at ({x},{y})");
        }
        assert!(detection.number_candidates > detection.number_crown_cells);
    }

    #[test]
    fn gradient_step_crown_follows_oblique_band() {
        let width = 40;
        let height = 40;
        let dem = plane_dem(width, height, 1.0);
        let roi: Vec<bool> = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                (10..=30).contains(&(x + y))
            })
            .collect();
        let config = ReleaseEstimationConfig {
            crown_elevation_band: 0.25,
            ..Default::default()
        };
        let detection = detect_crown_line_by_gradient_step(&dem, &roi, &config).unwrap();
        for (idx, &is_crown) in detection.crown.iter().enumerate() {
            if is_crown {
                let (x, y) = (idx % width, idx / width);
                assert_eq!(x + y, 30, "crown cell off the upstream diagonal");
            }
        }
        assert_eq!(detection.number_crown_cells, 31);
    }

    #[test]
    fn gradient_step_rejects_exit_distance_outside_range() {
        let dem = plane_dem(20, 20, 1.0);
        let roi = rectangle_roi(20, 20, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig {
            crown_exit_distance: 1.5,
            ..Default::default()
        };
        assert!(detect_crown_line_by_gradient_step(&dem, &roi, &config).is_err());
    }

    #[test]
    fn release_fill_skips_cells_outside_potential_release_areas() {
        // a non-release band at row 13 directly below the crown row 14: the
        // fill measures distance through it but selects only potential cells
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let potential: Vec<bool> = (0..width * height)
            .map(|idx| roi[idx] && idx / width != 13)
            .collect();
        let config = ReleaseEstimationConfig::default();
        let estimate = estimate_release_areas(&dem, &roi, Some(&potential), &config).unwrap();

        assert_eq!(estimate.number_release_cells, 25); // 25% of the 100 outline cells
        for (idx, &thickness) in estimate.release_areas.iter().enumerate() {
            let (x, y) = (idx % width, idx / width);
            // rows 14 and 12 complete, then the first 5 cells of row 11
            let expected = roi[idx] && (y == 14 || y == 12 || (y == 11 && x < 10));
            assert_eq!(thickness > 0.0, expected, "at ({x},{y})");
        }
    }

    #[test]
    fn release_fill_is_capped_by_potential_release_areas() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let potential = rectangle_roi(width, height, 5, 15, 14, 15);
        let config = ReleaseEstimationConfig::default();
        let estimate = estimate_release_areas(&dem, &roi, Some(&potential), &config).unwrap();
        assert_eq!(estimate.number_release_cells, 10);
        assert_eq!(
            estimate.release_areas.iter().filter(|&&t| t > 0.0).count(),
            10
        );

        // a mask of the wrong size is rejected
        assert!(estimate_release_areas(&dem, &roi, Some(&potential[1..]), &config).is_err());
    }

    #[test]
    fn slope_membership_matches_veitinger_bell() {
        let membership = |slope| release_slope_membership(slope, 43.0, 11.0);
        assert_eq!(membership(43.0), 1.0);
        assert!((membership(32.0) - 0.5).abs() < 1e-6);
        assert!((membership(54.0) - 0.5).abs() < 1e-6);
        assert!(membership(38.0) > 0.99);
        assert!(membership(50.0) > 0.95);
        assert!(membership(28.0) < 0.1);
        assert!(membership(60.0) < 0.05);
    }

    /// Crown along the upstream row `y` of the rectangle `x0..x1`.
    fn crown_row(width: usize, height: usize, y: usize, x0: usize, x1: usize) -> CrownDetection {
        let crown = rectangle_roi(width, height, x0, x1, y, y + 1);
        CrownDetection {
            candidates: crown.clone(),
            crown,
            number_crown_cells: x1 - x0,
            number_components: 1,
            number_candidates: x1 - x0,
        }
    }

    #[test]
    fn slope_weighted_fill_without_weight_equals_geodesic() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let crown = crown_row(width, height, 14, 5, 15);
        let geodesic = ReleaseEstimationConfig::default();
        let slope_weighted = ReleaseEstimationConfig {
            fill_method: ReleaseFillMethod::SlopeWeighted,
            slope_weight: 0.0,
            ..Default::default()
        };
        let expected =
            estimate_release_areas_with_crown(&dem, &roi, None, &crown, &geodesic).unwrap();
        let estimate =
            estimate_release_areas_with_crown(&dem, &roi, None, &crown, &slope_weighted).unwrap();
        assert_eq!(estimate.release_areas, expected.release_areas);
    }

    #[test]
    fn slope_weighted_fill_extends_into_favourable_slope() {
        // the plane rises at 40° left of x = 10 and at 25° right of it, below
        // a straight crown along the upstream row y = 14
        let width = 20;
        let height = 20;
        let cell_size = 10.0;
        let mut dem = plane_dem(width, height, cell_size);
        dem.data1d = (0..width * height)
            .map(|idx| {
                let (x, y) = (idx % width, idx / width);
                let slope_deg: f32 = if x < 10 { 40.0 } else { 25.0 };
                y as f32 * cell_size * slope_deg.to_radians().tan()
            })
            .collect();
        dem.data = to_2d(&dem.data1d, width, height);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let crown = crown_row(width, height, 14, 5, 15);
        let release_cells = |estimate: &ReleaseEstimate, columns: std::ops::Range<usize>| {
            (0..width * height)
                .filter(|&idx| columns.contains(&(idx % width)))
                .filter(|&idx| estimate.release_areas[idx] > 0.0)
                .count()
        };

        let geodesic = ReleaseEstimationConfig {
            slope_smoothing_length: 0.0,
            ..Default::default()
        };
        let estimate =
            estimate_release_areas_with_crown(&dem, &roi, None, &crown, &geodesic).unwrap();
        // rows 14 and 13 complete, then the first 5 cells of row 12
        assert_eq!(release_cells(&estimate, 5..10), 15);
        assert_eq!(release_cells(&estimate, 10..15), 10);

        let slope_weighted = ReleaseEstimationConfig {
            fill_method: ReleaseFillMethod::SlopeWeighted,
            ..geodesic
        };
        let estimate =
            estimate_release_areas_with_crown(&dem, &roi, None, &crown, &slope_weighted).unwrap();
        assert_eq!(estimate.number_release_cells, 25); // still 25% of 100 cells
        let favourable = release_cells(&estimate, 5..9);
        let unfavourable = release_cells(&estimate, 11..15);
        assert!(
            favourable > 2 * unfavourable,
            "{favourable} cells on the 40° half, {unfavourable} on the 25° half"
        );
        // every column keeps the crown row: the fill stays attached to it
        for x in 5..15 {
            assert!(
                estimate.release_areas[14 * width + x] > 0.0,
                "crown cell {x}"
            );
        }
    }

    #[test]
    fn slope_weighted_fill_rejects_invalid_parameters() {
        let dem = plane_dem(20, 20, 10.0);
        let roi = rectangle_roi(20, 20, 5, 15, 5, 15);
        let crown = crown_row(20, 20, 14, 5, 15);
        let valid = ReleaseEstimationConfig {
            fill_method: ReleaseFillMethod::SlopeWeighted,
            ..Default::default()
        };
        assert!(estimate_release_areas_with_crown(&dem, &roi, None, &crown, &valid).is_ok());
        for invalid in [
            ReleaseEstimationConfig {
                slope_weight: -1.0,
                ..valid.clone()
            },
            ReleaseEstimationConfig {
                slope_width_deg: 0.0,
                ..valid.clone()
            },
            ReleaseEstimationConfig {
                slope_smoothing_length: f32::NAN,
                ..valid.clone()
            },
        ] {
            assert!(estimate_release_areas_with_crown(&dem, &roi, None, &crown, &invalid).is_err());
        }
    }

    #[test]
    fn rejects_full_coverage_outline() {
        let dem = plane_dem(5, 5, 1.0);
        let roi = vec![true; 25];
        let config = ReleaseEstimationConfig::default();
        assert!(estimate_release_areas(&dem, &roi, None, &config).is_err());
    }
}
