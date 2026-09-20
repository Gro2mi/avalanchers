#[derive(thiserror::Error, Debug)]
pub enum RasterizerError {
    #[error("Padding must be positive")]
    PaddingMustBePositive,
    #[error("Padding is too small")]
    PaddingIsTooSmall,
    #[error("Raster grid is empty")]
    RasterGridIsEmpty,
}

/// Bounding box of a raster grid in projected coordinates. Lives here rather than
/// in `tile_manager` because the rasterizer also compiles for wasm.
#[derive(Debug, Clone)]
pub struct BBox {
    pub min_northing: u32,
    pub max_northing: u32,
    pub min_easting: u32,
    pub max_easting: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone)]
pub struct GeoPolygon {
    pub rings: Vec<Vec<Point2D>>, // Outer ring + optional inner holes
}

pub struct RasterGrid {
    pub origin_x: f64,  // Snapped to a multiple of cell_size
    pub origin_y: f64,  // Snapped to a multiple of cell_size
    pub cell_size: f64, // Grid resolution (e.g., 0.5m, 1.0m, 2.0m)
    pub width: usize,
    pub height: usize,
    pub data: Vec<bool>, // Flat 1D vector minimal fit
}

impl RasterGrid {
    /// Creates a RasterGrid directly from a polygon, computing its minimal
    /// bounding box and snapping the origin to a multiple of the cell size.
    pub fn from_polygon(polygon: &GeoPolygon, cell_size: f64) -> Self {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        // 1. Compute the true minimal fit bounding box around all rings
        for ring in &polygon.rings {
            for pt in ring {
                if pt.x < min_x {
                    min_x = pt.x;
                }
                if pt.y < min_y {
                    min_y = pt.y;
                }
                if pt.x > max_x {
                    max_x = pt.x;
                }
                if pt.y > max_y {
                    max_y = pt.y;
                }
            }
        }

        // Fallback for empty or invalid polygons
        if min_x == f64::MAX {
            return Self {
                origin_x: 0.0,
                origin_y: 0.0,
                cell_size,
                width: 0,
                height: 0,
                data: Vec::new(),
            };
        }

        // 2. Snap the origin down to the nearest grid line multiple
        let origin_x = (min_x / cell_size).floor() * cell_size;
        let origin_y = (min_y / cell_size).floor() * cell_size;

        // 3. Compute width and height based on the max bounds relative to the snapped origin
        let width = ((max_x - origin_x) / cell_size).ceil() as usize;
        let height = ((max_y - origin_y) / cell_size).ceil() as usize;

        let data = vec![false; width * height];

        let mut grid = Self {
            origin_x,
            origin_y,
            cell_size,
            width,
            height,
            data,
        };
        grid.rasterize(polygon);
        grid
    }

    #[inline(always)]
    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn get_bbox(&self) -> BBox {
        BBox {
            min_easting: self.origin_x as u32,
            max_easting: (self.origin_x + self.width as f64 * self.cell_size) as u32,
            min_northing: self.origin_y as u32,
            max_northing: (self.origin_y + self.height as f64 * self.cell_size) as u32,
        }
    }

    /// Generates a separate, zero-padded 1D flat vector matching the exact
    /// dimensions and origin of a specific target DTM / swissALTI3D tile.
    pub fn to_padded_tile(
        &self,
        tile_origin_x: f64,
        tile_origin_y: f64,
        tile_width: usize,
        tile_height: usize,
    ) -> Vec<bool> {
        // Allocate the target flat tile filled entirely with zeroes
        let mut tile_data = vec![false; tile_width * tile_height];

        // Calculate the discrete integer cell offset between the tile and this minimal grid.
        // Rounding handles any minuscule floating-point inaccuracies safely.
        let offset_x = ((self.origin_x - tile_origin_x) / self.cell_size).round() as isize;
        let offset_y = ((self.origin_y - tile_origin_y) / self.cell_size).round() as isize;

        // Copy matching blocks over into the tile frame
        for y in 0..self.height {
            let target_y = y as isize + offset_y;

            // Skip rows falling completely outside the target tile space
            if target_y < 0 || target_y >= tile_height as isize {
                continue;
            }

            for x in 0..self.width {
                let target_x = x as isize + offset_x;

                // Skip columns falling outside the target tile space
                if target_x < 0 || target_x >= tile_width as isize {
                    continue;
                }

                // Map local grid data index to target padded tile index
                let local_idx = self.get_index(x, y);
                if self.data[local_idx] {
                    let tile_idx = (target_y as usize) * tile_width + (target_x as usize);
                    tile_data[tile_idx] = true;
                }
            }
        }

        tile_data
    }

    /// Standard world-space scanline fill rasterization
    pub fn rasterize(&mut self, polygon: &GeoPolygon) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        for ring in &polygon.rings {
            if ring.len() < 3 {
                continue;
            }

            for y in 0..self.height {
                let world_y = self.origin_y + (y as f64 + 0.5) * self.cell_size;
                let mut intersections = Vec::new();

                for i in 0..ring.len() {
                    let p1 = ring[i];
                    let p2 = ring[(i + 1) % ring.len()];

                    if (p1.y <= world_y && p2.y > world_y) || (p2.y <= world_y && p1.y > world_y) {
                        let t = (world_y - p1.y) / (p2.y - p1.y);
                        let intersect_x = p1.x + t * (p2.x - p1.x);
                        intersections.push(intersect_x);
                    }
                }

                intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                for pair in intersections.as_chunks::<2>().0 {
                    // 1. Calculate the continuous cell coordinate values for the intersections
                    let start_x_float = (pair[0] - self.origin_x) / self.cell_size;
                    let end_x_float = (pair[1] - self.origin_x) / self.cell_size;

                    // 2. To strictly obey the Cell Center Rule:
                    // A cell center at index `x` is `x + 0.5`.
                    // - For the start, if the center is less than start_x_float, we must move to the next cell.
                    //   (start_x_float - 0.5).ceil() gives us the first cell index whose center >= pair[0]
                    // - For the end, (end_x_float - 0.5).ceil() gives us the first cell index whose center >= pair[1]
                    let start_x = ((start_x_float - 0.5).ceil().max(0.0) as usize).min(self.width);
                    let end_x = ((end_x_float - 0.5).ceil().max(0.0) as usize).min(self.width);

                    // 3. Fill the cells whose centers are strictly inside the intersection pair
                    for x in start_x..end_x {
                        let idx = self.get_index(x, y);
                        self.data[idx] = true;
                    }
                }
            }
        }
    }

    /// Adds a zero-filled padding around the current raster data in meters.
    /// The padding is converted to discrete cells based on the cell size.
    pub fn add_padding(&mut self, padding_meters: f64) -> Result<(), RasterizerError> {
        if padding_meters <= 0.0 {
            return Err(RasterizerError::PaddingMustBePositive);
        }
        if self.width == 0 || self.height == 0 {
            return Err(RasterizerError::RasterGridIsEmpty);
        }
        let padding_cells = (padding_meters / self.cell_size).ceil() as usize;
        if padding_cells == 0 {
            return Err(RasterizerError::PaddingIsTooSmall);
        }
        let new_width = self.width + 2 * padding_cells;
        let new_height = self.height + 2 * padding_cells;
        let mut new_data = vec![false; new_width * new_height];
        for y in 0..self.height {
            let old_start = y * self.width;
            let old_end = old_start + self.width;
            let new_start = (y + padding_cells) * new_width + padding_cells;
            new_data[new_start..new_start + self.width]
                .copy_from_slice(&self.data[old_start..old_end]);
        }

        self.origin_x -= padding_cells as f64 * self.cell_size;
        self.origin_y -= padding_cells as f64 * self.cell_size;
        self.width = new_width;
        self.height = new_height;
        self.data = new_data;
        Ok(())
    }

    pub fn print(&self) {
        println!(
            "Grid: {}x{} cells, {}x{} m, {} m² avalanche area",
            self.width,
            self.height,
            self.width as f64 * self.cell_size,
            self.height as f64 * self.cell_size,
            self.cell_size * self.cell_size * self.data.iter().filter(|&&v| v).count() as f64
        );
        println!("Origin: {}, {}", self.origin_x, self.origin_y);
        println!("Cell Size: {}", self.cell_size);
        print_grid(&self.data, self.width);
    }
}

impl Default for RasterGrid {
    fn default() -> Self {
        Self {
            origin_x: 0.0,
            origin_y: 0.0,
            cell_size: 5.0,
            width: 0,
            height: 0,
            data: Vec::new(),
        }
    }
}

pub fn print_grid(data: &[bool], width: usize) {
    for row in data.chunks(width) {
        let line: String = row
            .iter()
            .map(|&val| if val { "██" } else { "xx" })
            .collect();
        println!("{}", line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a basic square polygon
    fn create_square_poly(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> GeoPolygon {
        GeoPolygon {
            rings: vec![vec![
                Point2D { x: min_x, y: min_y },
                Point2D { x: max_x, y: min_y },
                Point2D { x: max_x, y: max_y },
                Point2D { x: min_x, y: max_y },
            ]],
        }
    }
    /// Helper to create a basic triangle polygon
    fn create_triangle_poly(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> GeoPolygon {
        GeoPolygon {
            rings: vec![vec![
                Point2D { x: min_x, y: min_y },
                Point2D { x: max_x, y: min_y },
                Point2D { x: max_x, y: max_y },
            ]],
        }
    }

    #[test_log::test]
    fn test_basic_triangle_rasterization() {
        let poly = create_triangle_poly(0.0, 0.0, 800.0, 500.0);
        let mut grid = RasterGrid::from_polygon(&poly, 5.0);
        grid.rasterize(&poly);
        // grid.debug_print();
        let raster_area = grid.data.iter().filter(|&&v| v).count() * 25; // Each cell is 5m x 5m = 25 m²
        let polygon_area = 800.0 * 500.0 / 2.0;
        println!(
            "Area filled: {}, Area polygon: {}, Relative Error: {}",
            raster_area,
            polygon_area,
            (polygon_area - raster_area as f64).abs() / polygon_area
        );
        assert!((polygon_area - raster_area as f64).abs() / polygon_area < 0.01);
    }

    #[test_log::test]
    fn test_basic_triangle_equilateral_above_half_rasterization() {
        let l = 3.51;
        let poly = create_triangle_poly(0.0, 0.0, l, l);
        let mut grid = RasterGrid::from_polygon(&poly, 1.0);
        grid.rasterize(&poly);
        // grid.debug_print();
        let raster_area = grid.data.iter().filter(|&&v| v).count();
        let polygon_area = l * l / 2.0;
        grid.print();
        println!(
            "Area filled: {}, Area polygon: {}, Relative Error: {}",
            raster_area,
            polygon_area,
            (polygon_area - raster_area as f64).abs() / polygon_area
        );
        assert_eq!(10, raster_area);
    }

    #[test_log::test]
    fn test_basic_triangle_equilateral_below_half_rasterization() {
        let l = 3.5;
        let poly = create_triangle_poly(0.0, 0.0, l, l);
        let mut grid = RasterGrid::from_polygon(&poly, 1.0);
        grid.rasterize(&poly);
        // grid.debug_print();
        let raster_area = grid.data.iter().filter(|&&v| v).count();
        let polygon_area = l * l / 2.0;
        grid.print();
        println!(
            "Area filled: {}, Area polygon: {}, Relative Error: {}",
            raster_area,
            polygon_area,
            (polygon_area - raster_area as f64).abs() / polygon_area
        );
        assert_eq!(6, raster_area);
    }

    #[test_log::test]
    fn test_basic_square_rasterization() {
        let l = 10.1;
        let poly = create_square_poly(0.0, 0.0, l, l);
        let mut grid = RasterGrid::from_polygon(&poly, 5.0);
        grid.rasterize(&poly);
        grid.print();
        let raster_area = grid.data.iter().filter(|&&v| v).count() * 25; // Each cell is 5m x 5m = 25 m²
        let polygon_area = l * l;
        println!(
            "Area filled: {}, Area polygon: {}, Relative Error: {}",
            raster_area,
            polygon_area,
            (polygon_area - raster_area as f64).abs() / polygon_area
        );
        // assert!((polygon_area - raster_area as f64).abs() / polygon_area < 0.01);
    }

    // =========================================================================
    // 1. ORIGIN SNAPPING & BOUNDING BOX EDGE CASES
    // =========================================================================

    #[test]
    fn test_origin_snapping_swiss_coordinates() {
        // Simulating Swiss LV95 coordinates that don't land perfectly on grid lines
        let poly = create_square_poly(2600001.34, 1200003.89, 2600007.10, 1200009.50);
        let cell_size = 2.0;

        let grid = RasterGrid::from_polygon(&poly, cell_size);

        // Origin must round DOWN to the nearest multiple of 2.0
        assert_eq!(grid.origin_x, 2600000.0);
        assert_eq!(grid.origin_y, 1200002.0);

        // Width logic: (2600007.10 - 2600000.0) / 2.0 = 3.55 -> Ceil = 4 cells
        assert_eq!(grid.width, 4);
        // Height logic: (1200009.50 - 1200002.0) / 2.0 = 3.75 -> Ceil = 4 cells
        assert_eq!(grid.height, 4);
        assert_eq!(grid.data.len(), 16);
    }

    #[test]
    fn test_degenerate_empty_polygon() {
        let empty_poly = GeoPolygon { rings: vec![] };
        let grid = RasterGrid::from_polygon(&empty_poly, 1.0);

        assert_eq!(grid.width, 0);
        assert_eq!(grid.height, 0);
        assert!(grid.data.is_empty());

        // Ensure rasterizing an empty grid doesn't panic
        let mut mut_grid = grid;
        mut_grid.rasterize(&empty_poly);
    }

    #[test]
    fn test_degenerate_line_polygon() {
        // A flat line technically has zero area, should create a valid grid structure but fill nothing
        let line_poly = GeoPolygon {
            rings: vec![vec![
                Point2D { x: 10.0, y: 10.0 },
                Point2D { x: 20.0, y: 10.0 },
            ]],
        };
        let mut grid = RasterGrid::from_polygon(&line_poly, 1.0);
        grid.rasterize(&line_poly);

        // Should be flat, width 10, height 0 (or 1 depending on ceiling rules for flat bounds)
        assert!(grid.width > 0);
        // Ensure data vector contains purely zeros because scanline needs pairs to fill
        assert!(grid.data.iter().all(|&v| !v));
    }

    // =========================================================================
    // 2. RASTERIZATION & SCANLINE EDGE CASES
    // =========================================================================

    #[test]
    fn test_sub_cell_polygon_misses_centers() {
        // A tiny polygon entirely inside a 2x2 cell, but positioned so it misses the pixel center (0.5)
        // Cell boundary: X: [0.0, 2.0], Y: [0.0, 2.0]. Center is at (1.0, 1.0)
        // We place the polygon at the top-right corner of the cell: [1.2, 1.8]
        let poly = create_square_poly(1.2, 1.2, 1.8, 1.8);
        let mut grid = RasterGrid::from_polygon(&poly, 2.0);
        grid.rasterize(&poly);

        // It should build a 1x1 grid
        assert_eq!(grid.width, 1);
        assert_eq!(grid.height, 1);

        // Because the scanline checks Y = 1.0 (the center), and our poly is between Y=1.2 and 1.8,
        // it must NOT rasterize it (evaluates to 0). This is expected behavior for discrete grids.
        assert_eq!(grid.data[0], false);
    }

    #[test]
    fn test_sub_cell_polygon_hits_center() {
        // This tiny polygon straddles the center (1.0, 1.0) of the 2x2 cell
        let poly = create_square_poly(0.8, 0.8, 1.2, 1.2);
        let mut grid = RasterGrid::from_polygon(&poly, 2.0);
        grid.rasterize(&poly);

        // It hits the Y=1.0 scanline, so it should be painted
        assert_eq!(grid.data[0], true);
    }

    // =========================================================================
    // 3. TILE PADDING & ALIGNMENT EDGE CASES
    // =========================================================================

    #[test]
    fn test_padded_tile_perfect_fit() {
        let poly = create_square_poly(2.0, 2.0, 6.0, 6.0); // 4x4 meters
        let mut grid = RasterGrid::from_polygon(&poly, 2.0); // origin: 2.0, 2.0. size: 2x2 cells
        grid.rasterize(&poly);

        // Target tile is exactly identical to the grid's bounding footprint
        let padded = grid.to_padded_tile(2.0, 2.0, 2, 2);

        assert_eq!(padded.len(), 4);
        assert!(padded.iter().all(|&v| v));
    }

    #[test]
    fn test_padded_tile_completely_out_of_bounds() {
        let poly = create_square_poly(0.0, 0.0, 4.0, 4.0);
        let mut grid = RasterGrid::from_polygon(&poly, 2.0);
        grid.rasterize(&poly);

        // Target swissALTI3D tile is completely somewhere else (e.g., thousands of meters away)
        let tile_width = 10;
        let tile_height = 10;
        let padded = grid.to_padded_tile(5000.0, 5000.0, tile_width, tile_height);

        assert_eq!(padded.len(), 100);
        // The function must safely discard the data without crashing, leaving a completely zeroed tile
        assert!(padded.iter().all(|&v| !v));
    }

    #[test]
    fn test_padded_tile_partial_overlap_west_and_south() {
        // Polygon sits around the origin (0.0, 0.0) spanning from -2.0 to 4.0
        let poly = create_square_poly(-2.0, -2.0, 4.0, 4.0);
        let mut grid = RasterGrid::from_polygon(&poly, 2.0); // Origin: -2.0, -2.0. Width: 3, Height: 3
        grid.rasterize(&poly);

        // Target tile starts at (0.0, 0.0) with a size of 2x2 cells.
        // This means the row/col at index -1 relative to the tile should be safely clipped.
        let tile_width = 2;
        let tile_height = 2;
        let padded = grid.to_padded_tile(0.0, 0.0, tile_width, tile_height);

        assert_eq!(padded.len(), 4);
        // The parts overlapping the tile (0.0 to 4.0) should be fully painted
        assert!(padded.iter().all(|&v| v));
    }

    #[test]
    fn test_padded_tile_partial_overlap_east_and_north() {
        // Polygon spills past the edge of our target tile layout
        let poly = create_square_poly(2.0, 2.0, 8.0, 8.0);
        let mut grid = RasterGrid::from_polygon(&poly, 2.0);
        grid.rasterize(&poly);

        // Target tile is only 2x2 cells starting at (0.0, 0.0) [Max X: 4.0, Max Y: 4.0]
        // Grid elements at world positions >= 4.0 must be dropped safely
        let tile_width = 2;
        let tile_height = 2;
        let padded = grid.to_padded_tile(0.0, 0.0, tile_width, tile_height);

        // Index layout for a 2x2 tile:
        // Row 1 (y=1): [x=0, x=1] -> World Y mid-point is 3.0.
        // Row 0 (y=0): [x=0, x=1] -> World Y mid-point is 1.0.

        // At y=0 (world_y=1.0): Polygon starts at Y=2.0, so this row is empty (0, 0)
        assert_eq!(padded[0], false); // (x=0, y=0)
        assert_eq!(padded[1], false); // (x=1, y=0)

        // At y=1 (world_y=3.0): Polygon covers X from 2.0 to 8.0.
        // Tile column 0 (world_x=1.0) is empty. Tile column 1 (world_x=3.0) intersects the polygon!
        assert_eq!(padded[2], false); // (x=0, y=1)
        assert_eq!(padded[3], true); // (x=1, y=1)
    }

    #[test]
    fn test_add_padding() {
        let poly = create_square_poly(2.0, 2.0, 6.0, 6.0); // 4x4 meters
        let mut grid = RasterGrid::from_polygon(&poly, 2.0); // origin: 2.0, 2.0. size: 2x2 cells

        // Grid data is originally:
        // [1, 1,
        //  1, 1]
        assert_eq!(grid.data, vec![true, true, true, true]);

        // Add a padding of 2.0 meters (which is 1 cell)
        grid.add_padding(2.0).expect("Failed to add padding");

        // New dimensions: width + 2*1 = 4, height + 2*1 = 4
        // Padded data should be:
        // [0, 0, 0, 0,
        //  0, 1, 1, 0,
        //  0, 1, 1, 0,
        //  0, 0, 0, 0]
        assert_eq!(
            grid.data,
            vec![
                false, false, false, false, false, true, true, false, false, true, true, false,
                false, false, false, false,
            ]
        );
    }

    #[test]
    fn test_add_padding_zero() {
        let poly = create_square_poly(2.0, 2.0, 6.0, 6.0); // 4x4 meters
        let mut grid = RasterGrid::from_polygon(&poly, 2.0); // origin: 2.0, 2.0. size: 2x2 cells
        grid.rasterize(&poly);

        // Grid data is originally:
        // [1, 1,
        //  1, 1]
        assert_eq!(grid.data, vec![true, true, true, true]);

        // Add a padding of 0.0 should return PaddingMustBePositive error
        let res = grid.add_padding(0.0);
        assert!(matches!(res, Err(RasterizerError::PaddingMustBePositive)));

        assert_eq!(grid.data, vec![true, true, true, true]);
    }

    #[test]
    fn test_get_bbox() {
        let poly = create_square_poly(12.0, 22.0, 16.0, 26.0);
        let mut grid = RasterGrid::from_polygon(&poly, 2.0);
        let mut bbox = grid.get_bbox();
        assert_eq!(bbox.min_easting, 12);
        assert_eq!(bbox.max_easting, 16);
        assert_eq!(bbox.min_northing, 22);
        assert_eq!(bbox.max_northing, 26);
        grid.add_padding(4.0).unwrap();
        bbox = grid.get_bbox();
        assert_eq!(bbox.min_easting, 12 - 4);
        assert_eq!(bbox.max_easting, 16 + 4);
        assert_eq!(bbox.min_northing, 22 - 4);
        assert_eq!(bbox.max_northing, 26 + 4);
    }
}
