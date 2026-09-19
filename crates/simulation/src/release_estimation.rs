//! Release area estimation from an avalanche outline and a DEM.
//!
//! Two stages:
//! 1. Crown line detection: the crown line is taken as the "inflow boundary"
//!    of the outline - the outline cells that receive terrain from outside the
//!    outline. Two detection methods are available: D8 flow routing on a
//!    sink-filled DEM (CPU), or a GPU particle simulation released everywhere
//!    outside the outline (see
//!    `Simulation::detect_crown_line_by_particle_simulation`). Both depend only
//!    on how terrain drains into the outline, so they also work when the crown
//!    line cuts across the contours.
//! 2. Geodesic fill: chamfer-weighted geodesic distance from the crown line
//!    inside the outline mask; cells are selected by increasing distance until
//!    the target share of the outline area is reached. The fill front advances
//!    parallel to the crown line instead of following the contours.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use anyhow::{Result, bail};
use compute_core::dem::Dem;
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

/// Index of the outline cell a stopped particle entered, if any. A particle
/// that stopped inside the outline counts directly. Because the bilinear
/// terrain sampling blends in the masked cells, a particle crossing the
/// boundary usually stops up to one cell short, in a finite cell just outside
/// the outline; such a particle is attributed to the neighbouring outline
/// cell its velocity points into. Particles whose velocity does not point
/// into any outline cell (for example particles passing along a flank, or
/// particles stopped by NaN normals at DEM edges) are not entries.
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
    // (alignment with velocity, negated distance, cell index)
    let mut best: Option<(f32, f32, usize)> = None;
    for (dx, dy) in NEIGHBOR_OFFSETS {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
            continue;
        }
        let nidx = ny as usize * width + nx as usize;
        if !roi[nidx] {
            continue;
        }
        let center = [(nx as f32 + 0.5) * cell_size, (ny as f32 + 0.5) * cell_size];
        let direction = [center[0] - position[0], center[1] - position[1]];
        let distance = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if distance <= f32::EPSILON {
            return Some(nidx);
        }
        let dot = velocity[0] * direction[0] + velocity[1] * direction[1];
        if dot <= 0.0 {
            continue; // not moving into this outline cell
        }
        let alignment = dot / (distance * speed);
        let candidate = (alignment, -distance, nidx);
        if best.is_none_or(|b| candidate > b) {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, nidx)| nidx)
}

#[derive(Debug, Clone)]
pub struct ReleaseEstimationConfig {
    /// Target release area as a fraction of the outline area (typically 0.2 - 0.3).
    pub fraction: f32,
    /// Snow thickness \[m\] written to the selected release cells.
    pub slab_thickness: f32,
    /// Crown components whose mean elevation is below this share of the outline
    /// elevation range are discarded as flank gullies (0.5 keeps the top half).
    pub crown_elevation_band: f32,
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
    /// Plausibility window for the crown slope in degrees; violations are only
    /// reported as warnings.
    pub expected_crown_slope_range: (f32, f32),
}

impl Default for ReleaseEstimationConfig {
    fn default() -> Self {
        Self {
            fraction: 0.25,
            slab_thickness: 1.0,
            crown_elevation_band: 0.5,
            min_crown_component_cells: 3,
            sink_fill_epsilon: 1e-2,
            min_particles_per_crown_cell: 2,
            expected_crown_slope_range: (28.0, 60.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReleaseEstimate {
    /// Snow thickness per cell, 0 outside the release area. Same length and
    /// indexing as the DEM (`y * width + x`).
    pub release_areas: Vec<f32>,
    /// Mask marking the detected crown line cells.
    pub crown_line: Vec<bool>,
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
    /// Number of crown cells after filtering.
    pub number_crown_cells: usize,
    /// Number of connected components kept.
    pub number_components: usize,
    /// Crown candidates before elevation-band filtering.
    pub number_candidates: usize,
}

/// Groups crown candidate cells into connected components (8-connectivity) and
/// filters them by mean elevation so flank gullies drop out. Falls back to the
/// largest component when nothing survives the filter.
fn filter_crown_components(
    candidate: Vec<bool>,
    elevations: &[f32],
    roi: &[bool],
    width: usize,
    height: usize,
    config: &ReleaseEstimationConfig,
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

    let mut kept: Vec<&Vec<usize>> = components
        .iter()
        .filter(|component| component.len() >= config.min_crown_component_cells)
        .filter(|component| {
            let mean_elevation =
                component.iter().map(|&idx| elevations[idx]).sum::<f32>() / component.len() as f32;
            mean_elevation >= threshold
        })
        .collect();
    if kept.is_empty() && !components.is_empty() {
        warn!("no crown component passed the elevation filter, keeping the largest");
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
        number_crown_cells,
        number_components: kept.len(),
        number_candidates,
    }
}

/// Detects the crown line as the outline's inflow boundary: outline cells that
/// are the D8 receiver of at least one cell outside the outline.
fn detect_crown_line_from_flow(
    elevations: &[f32],
    flow: &[Option<u8>],
    roi: &[bool],
    width: usize,
    height: usize,
    config: &ReleaseEstimationConfig,
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
    filter_crown_components(candidate, elevations, roi, width, height, config)
}

/// Chamfer-weighted geodesic distance (Dijkstra) from the seed cells, only
/// traversing `traversable` cells. Unreachable cells stay at infinity.
fn geodesic_distance_from(
    seeds: &[bool],
    traversable: &[bool],
    width: usize,
    height: usize,
    cell_size: f32,
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
            let weight = if dx != 0 && dy != 0 {
                diagonal_distance
            } else {
                cell_size
            };
            let candidate_distance = d0 + weight;
            if candidate_distance < distance[nidx] {
                distance[nidx] = candidate_distance;
                heap.push(Reverse((Cost(candidate_distance), nidx)));
            }
        }
    }

    distance
}

/// Steepest-descent slope angle in degrees on the raw elevations.
fn steepest_descent_angle(
    z: &[f32],
    width: usize,
    height: usize,
    cell_size: f32,
    idx: usize,
) -> f32 {
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
    validate_roi(dem, roi)?;
    info!(
        "Crown line estimation: filling sinks on {}x{} grid",
        dem.width, dem.height
    );
    let filled = fill_sinks(&dem.data1d, dem.width, dem.height, config.sink_fill_epsilon);
    let flow = d8_flow_directions(&filled, dem.width, dem.height, dem.cell_size);
    let detection =
        detect_crown_line_from_flow(&dem.data1d, &flow, roi, dem.width, dem.height, config);
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
    let detection =
        filter_crown_components(candidate, &dem.data1d, roi, dem.width, dem.height, config);
    check_detection(&detection)?;
    Ok(detection)
}

fn check_detection(detection: &CrownDetection) -> Result<()> {
    if detection.number_candidates == 0 {
        bail!("no crown line detected: no terrain outside the outline drains into the outline");
    }
    Ok(())
}

/// Fills the target share of the outline area below the given crown line.
///
/// Fails if the outline is empty, covers the whole DEM (crown line detection
/// needs cells outside the outline) or if no inflow boundary exists.
pub fn estimate_release_areas_with_crown(
    dem: &Dem,
    roi: &[bool],
    crown: &CrownDetection,
    config: &ReleaseEstimationConfig,
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

    let distance = geodesic_distance_from(&crown.crown, roi, width, height, dem.cell_size);

    let mut ordered: Vec<(f32, usize)> = roi
        .iter()
        .enumerate()
        .filter(|&(idx, inside)| *inside && distance[idx].is_finite())
        .map(|(idx, _)| (distance[idx], idx))
        .collect();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    let target =
        ((config.fraction * number_roi_cells as f32).round() as usize).clamp(1, ordered.len());
    if ordered.len() < number_roi_cells {
        warn!(
            "{} of {} outline cells are unreachable from the crown line and can never be selected",
            number_roi_cells - ordered.len(),
            number_roi_cells
        );
    }

    let mut release_areas = vec![0.0f32; roi.len()];
    for &(_, idx) in &ordered[..target] {
        release_areas[idx] = config.slab_thickness;
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
            .map(|&idx| steepest_descent_angle(&dem.data1d, width, height, dem.cell_size, idx))
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
        "Release fill: {} of {} outline cells ({:.0}%), mean crown slope {:.1}°",
        target,
        number_roi_cells,
        target as f32 / number_roi_cells as f32 * 100.0,
        mean_crown_slope_deg
    );

    Ok(ReleaseEstimate {
        release_areas,
        crown_line: crown.crown.clone(),
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
    config: &ReleaseEstimationConfig,
) -> Result<ReleaseEstimate> {
    let crown = detect_crown_line(dem, roi, config)?;
    estimate_release_areas_with_crown(dem, roi, &crown, config)
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
        dem.data1d = (0..width * height)
            .map(|idx| (idx / width) as f32)
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
        let detection =
            detect_crown_line_from_flow(&dem.data1d, &flow, &roi, width, height, &config);

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
    fn crown_line_from_particle_counts_filters_strays_and_flank_gullies() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig::default();
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
    }

    #[test]
    fn release_fill_hits_exact_fraction_below_crown() {
        let width = 20;
        let height = 20;
        let dem = plane_dem(width, height, 10.0);
        let roi = rectangle_roi(width, height, 5, 15, 5, 15);
        let config = ReleaseEstimationConfig {
            fraction: 0.25,
            slab_thickness: 1.5,
            ..Default::default()
        };
        let estimate = estimate_release_areas(&dem, &roi, &config).unwrap();

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
        let estimate = estimate_release_areas(&dem, &roi, &config).unwrap();

        for (idx, &is_crown) in estimate.crown_line.iter().enumerate() {
            if is_crown {
                let (x, y) = (idx % width, idx / width);
                assert_eq!(x + y, 30, "crown cell off the upstream diagonal");
            }
        }
        assert_eq!(estimate.number_crown_cells, 31);

        // lowest cell of the outline sits on the crown line and must be
        // released; the highest-elevation cell on the far diagonal must not
        let low_on_crown = 0 * width + 30;
        let high_far_from_crown = 10 * width + 0;
        assert_eq!(estimate.release_areas[low_on_crown], 1.0);
        assert_eq!(estimate.release_areas[high_far_from_crown], 0.0);
        assert_eq!(estimate.number_release_cells, 110); // 25% of 441 cells
    }

    #[test]
    fn rejects_full_coverage_outline() {
        let dem = plane_dem(5, 5, 1.0);
        let roi = vec![true; 25];
        let config = ReleaseEstimationConfig::default();
        assert!(estimate_release_areas(&dem, &roi, &config).is_err());
    }
}
