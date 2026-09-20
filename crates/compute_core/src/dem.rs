use crate::utils::*;
use core::f32;
use std::hash::{Hash, Hasher};

#[derive(Default)]
struct StableHasher(u64);

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        if self.0 == 0 {
            self.0 = 0xcbf29ce484222325;
        }
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct Bounds {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
}

impl Hash for Bounds {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.xmin.to_bits().hash(state);
        self.xmax.to_bits().hash(state);
        self.ymin.to_bits().hash(state);
        self.ymax.to_bits().hash(state);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dem {
    pub width: usize,
    pub height: usize,
    pub bounds: Bounds,
    pub data1d: Vec<f32>,
    pub data: Vec<Vec<f32>>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub cell_size: f32,
    pub map_factor: f32,
    pub minimum_elevation: f32,
    pub source: String,
    pub projection: String,
}

impl Hash for Dem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.height.hash(state);
        self.bounds.hash(state);
        for val in &self.data1d {
            val.to_bits().hash(state);
        }
        for val in &self.x {
            val.to_bits().hash(state);
        }
        for val in &self.y {
            val.to_bits().hash(state);
        }
        self.cell_size.to_bits().hash(state);
        self.map_factor.to_bits().hash(state);
        self.minimum_elevation.to_bits().hash(state);
        self.projection.hash(state);
    }
}

impl Default for Dem {
    fn default() -> Self {
        Dem {
            width: 0,
            height: 0,
            bounds: Bounds {
                xmin: 0.0,
                xmax: 1.0,
                ymin: 0.0,
                ymax: 1.0,
            },
            data1d: Vec::new(),
            data: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            cell_size: 1.0,
            map_factor: 1.0,
            minimum_elevation: 1.0,
            source: String::new(),
            projection: String::new(),
        }
    }
}

impl Dem {
    pub fn calculate_hash(&self) -> u64 {
        // Content identity excludes `source` and the duplicate row-oriented `data`.
        let mut hasher = StableHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
    pub fn calculate_minimum_elevation(data1d: &[f32]) -> f32 {
        data1d
            .iter()
            .filter(|value| value.is_finite())
            .min_by(|a: &&f32, b: &&f32| a.total_cmp(b))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn get_index(&self, pt: &Point) -> Option<(f32, f32)> {
        let scale = self.cell_size * self.map_factor;
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        Some((
            (pt.x - self.bounds.xmin) / scale,
            (pt.y - self.bounds.ymin) / scale,
        ))
    }

    pub fn interpolate_elevation(&self, pt: &Point) -> Point {
        let z = self
            .get_index(pt)
            .and_then(|(x, y)| bilinear_interpolate(x, y, &self.data));
        Point {
            x: pt.x,
            y: pt.y,
            z,
        }
    }
    pub fn parse_bounds_lines<I: Iterator<Item = String>>(lines: I) -> Option<Bounds> {
        let values = lines
            .map(|line| line.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let [xmin, ymin, xmax, ymax] = values.as_slice() else {
            return None;
        };
        if !values.iter().all(|value| value.is_finite()) || xmin >= xmax || ymin >= ymax {
            return None;
        }
        Some(Bounds {
            xmin: *xmin,
            xmax: *xmax,
            ymin: *ymin,
            ymax: *ymax,
        })
    }

    fn get_elevation_extrema_points(
        &self,
        mask: &[bool],
    ) -> Option<((usize, usize), (usize, usize))> {
        if self.width == 0 || self.data1d.len() != mask.len() {
            return None;
        }

        let mut min_idx = None;
        let mut max_idx = None;

        for (i, (&elevation, &valid)) in self.data1d.iter().zip(mask.iter()).enumerate() {
            if !valid || !elevation.is_finite() {
                continue;
            }

            if min_idx.is_none_or(|idx| elevation < self.data1d[idx]) {
                min_idx = Some(i);
            }

            if max_idx.is_none_or(|idx| elevation > self.data1d[idx]) {
                max_idx = Some(i);
            }
        }

        let min_idx = min_idx?;
        let max_idx = max_idx?;

        let min_point = (min_idx / self.width, min_idx % self.width);
        let max_point = (max_idx / self.width, max_idx % self.width);

        Some((min_point, max_point))
    }

    pub fn get_elevation_extrema(&self, mask: &[bool]) -> Option<(f32, f32)> {
        let (min_point, max_point) = self.get_elevation_extrema_points(mask)?;
        let min_elevation = self.data1d[min_point.0 * self.width + min_point.1];
        let max_elevation = self.data1d[max_point.0 * self.width + max_point.1];
        Some((min_elevation, max_elevation))
    }

    pub fn get_elevation_extrema_distance_and_drop(&self, mask: &[bool]) -> Option<(f32, f32)> {
        let (min_point, max_point) = self.get_elevation_extrema_points(mask)?;
        let min_elevation = self.data1d[min_point.0 * self.width + min_point.1];
        let max_elevation = self.data1d[max_point.0 * self.width + max_point.1];

        let dx = (max_point.1 as isize - min_point.1 as isize) as f32;
        let dy = (max_point.0 as isize - min_point.0 as isize) as f32;
        let distance2d = ((dx * dx + dy * dy).sqrt()) * self.cell_size * self.map_factor;

        let drop = max_elevation - min_elevation;

        Some((distance2d, drop))
    }

    pub fn mask_above_elevation(&self, max_elevation: f32) -> Dem {
        let mut new_dem = self.clone();
        for val in new_dem.data1d.iter_mut() {
            if *val > max_elevation {
                *val = f32::NAN;
            }
        }
        new_dem
    }
}

pub struct GeoMetadata {
    pub width: u32,
    pub height: u32,
    /// ModelPixelScaleTag: [scale_x, scale_y, scale_z]
    /// Defines the size of a pixel in CRS units.
    pub pixel_scale: [f64; 3],
    pub cell_size: f32,
    /// ModelTiepointTag: [i, j, k, x, y, z]
    /// Maps pixel coordinates (i,j) to CRS coordinates (x,y).
    pub tiepoints: Vec<f64>,
    pub bounds: Bounds,
    /// GeoKeyDirectoryTag: The projection/CRS information (e.g., EPSG code)
    pub epsg_code: u32,
    /// NoData Value: Crucial for simulations to ignore empty cells
    pub nodata: Option<f64>,
}

pub struct GeoTiff {
    pub metadata: GeoMetadata,
    /// The actual grid data stored in a flat Vector for performance
    pub data: TiffData,
}

impl GeoTiff {
    /// Calculate the world coordinates of a specific cell (row, col)
    pub fn cell_to_world(&self, col: u32, row: u32) -> Option<(f64, f64)> {
        let tiepoint = self.metadata.tiepoints.get(..6)?;
        let x = tiepoint[3] + (f64::from(col) - tiepoint[0]) * self.metadata.pixel_scale[0];
        let y = tiepoint[4] - (f64::from(row) - tiepoint[1]) * self.metadata.pixel_scale[1];
        if x.is_finite() && y.is_finite() {
            Some((x, y))
        } else {
            None
        }
    }
    pub fn get_f32(&self, col: usize, row: usize) -> Option<f32> {
        if col >= self.metadata.width as usize || row >= self.metadata.height as usize {
            return None; // Out of bounds
        }
        self.data.get_f32(col, row, self.metadata.width as usize)
    }
    pub fn flip_y(&mut self) {
        // Convert current data to F32 variant and take ownership
        let mut d = std::mem::replace(&mut self.data, TiffData::U8(vec![])).as_f32();

        // Perform the flip
        flip_rows_flat_vec(&mut d, self.metadata.width, self.metadata.height);

        // Store it back as the F32 variant
        self.data = TiffData::F32(d);
    }
}

#[derive(Clone)]
pub enum TiffData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl TiffData {
    pub fn as_f32(self) -> Vec<f32> {
        match self {
            Self::U8(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::U16(v) => v.into_iter().map(|x| x as f32).collect(),
            Self::F32(v) => v, // No allocation/copy here!
        }
    }
    pub fn byte_len(&self) -> usize {
        match self {
            TiffData::U8(v) => v.len() * std::mem::size_of::<u8>(),
            TiffData::U16(v) => v.len() * std::mem::size_of::<u16>(),
            TiffData::F32(v) => v.len() * std::mem::size_of::<f32>(),
        }
    }
    pub fn get_f32(&self, col: usize, row: usize, width: usize) -> Option<f32> {
        if width == 0 || col >= width {
            return None;
        }
        let index = row.checked_mul(width)?.checked_add(col)?;

        match self {
            TiffData::U8(v) => v.get(index).map(|&val| val as f32),
            TiffData::U16(v) => v.get(index).map(|&val| val as f32),
            TiffData::F32(v) => v.get(index).copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test_log::test]
    fn test_interpolate_elevation_flat() {
        // Flat DEM: all elevations are 10.0
        let width = 3;
        let height = 3;
        let data1d = vec![10.0; width * height];
        let data = vec![vec![10.0; width]; height];
        let dem = Dem {
            width,
            height,
            bounds: Bounds {
                xmin: 0.0,
                xmax: 2.0,
                ymin: 0.0,
                ymax: 2.0,
            },
            data1d,
            data,
            x: vec![0.0, 1.0, 2.0],
            y: vec![0.0, 1.0, 2.0],
            cell_size: 1.0,
            map_factor: 1.0,
            minimum_elevation: 1.0,
            source: String::new(),
            projection: String::new(),
        };
        let pt = Point {
            x: 1.0,
            y: 1.0,
            z: Some(0.0),
        };
        let interp = dem.interpolate_elevation(&pt);
        assert!((interp.z.unwrap() - 10.0).abs() < 1e-6);
    }

    #[test_log::test]
    fn test_dem_get_extrema_points() {
        let width = 3;
        let height = 3;
        let data1d = vec![
            5.0, 2.0, 3.0, // Row 0
            4.0, 1.0, 6.0, // Row 1
            7.0, 8.0, 9.0, // Row 2
        ];
        let dem = Dem {
            width,
            height,
            bounds: Bounds::default(),
            data1d: data1d.clone(),
            data: vec![vec![0.0; width]; height], // Not used in this test
            x: vec![],
            y: vec![],
            cell_size: 2.0,
            map_factor: 1.0,
            minimum_elevation: 1.0,
            source: String::new(),
            projection: String::new(),
        };

        let mask = vec![
            true, true, true, // Row 0
            true, false, true, // Row 1 (mask out the minimum)
            true, true, true, // Row 2
        ];

        let (min_point, max_point) = dem.get_elevation_extrema_points(&mask).unwrap();
        assert_eq!(min_point, (0, 1)); // Elevation of 1.0 at (row=1,col=1)
        assert_eq!(max_point, (2, 2)); // Elevation of 9.0 at (row=2,col=2)
        let (min_elevation, max_elevation) = dem.get_elevation_extrema(&mask).unwrap();
        println!(
            "Min elevation: {}, Max elevation: {}",
            min_elevation, max_elevation
        );
        assert_eq!(min_elevation, 2.0); // The minimum in the masked area is 2.0
        assert_eq!(max_elevation, 9.0); // The maximum is 9.0
        let (distance2d, drop) = dem.get_elevation_extrema_distance_and_drop(&mask).unwrap();
        println!("Distance: {}, Drop: {}", distance2d, drop);
        assert!((distance2d - 4.472136).abs() < 1e-3); // Distance between (0,1) and (2,2) in pixel space is sqrt(2^2 + 1^2) = sqrt(5) ~ 2.236, but scaled by cell_size=2.0 gives ~4.472
        assert_eq!(drop, 7.0); // Drop from 9.0 to 2.0 is 7.0
    }

    fn create_mock_metadata(width: u32, height: u32) -> GeoMetadata {
        GeoMetadata {
            width,
            height,
            // 1 pixel = 10.0 units in world space
            pixel_scale: [10.0, 10.0, 0.0],
            cell_size: 10.0,
            // Tiepoint maps pixel (0,0) to world (500.0, 1000.0)
            // Format: [i, j, k, x, y, z]
            tiepoints: vec![0.0, 0.0, 0.0, 500.0, 1000.0, 0.0],
            bounds: Bounds::default(), // Assuming Bounds has a default
            epsg_code: 4326,
            nodata: Some(-9999.0),
        }
    }

    #[test]
    fn test_cell_to_world_calculation() {
        let meta = create_mock_metadata(100, 100);
        let geotiff = GeoTiff {
            metadata: meta,
            data: TiffData::U8(vec![0; 10000]),
        };

        // Origin (0,0) should match tiepoint (500, 1000)
        let (x0, y0) = geotiff.cell_to_world(0, 0).unwrap();
        assert_eq!(x0, 500.0);
        assert_eq!(y0, 1000.0);

        // Move 2 pixels right (2 * 10.0) and 3 pixels down (3 * 10.0)
        // Note: Y usually decreases as row index increases in GeoTIFFs
        let (x1, y1) = geotiff.cell_to_world(2, 3).unwrap();
        assert_eq!(x1, 520.0);
        assert_eq!(y1, 970.0);
    }

    #[test]
    fn test_tiff_data_indexing_u8() {
        let width = 2;
        let data = TiffData::U8(vec![
            10, 20, // Row 0
            30, 40, // Row 1
        ]);

        assert_eq!(data.get_f32(0, 0, width), Some(10.0));
        assert_eq!(data.get_f32(1, 0, width), Some(20.0));
        assert_eq!(data.get_f32(0, 1, width), Some(30.0));
        assert_eq!(data.get_f32(5, 5, width), None); // Out of bounds
    }

    #[test]
    fn test_tiff_data_indexing_f32() {
        let width = 3;
        let data = TiffData::F32(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);

        assert_eq!(data.get_f32(1, 1, width), Some(5.5));
    }

    #[test]
    fn test_byte_len() {
        let u8_data = TiffData::U8(vec![0, 0, 0]);
        let u16_data = TiffData::U16(vec![0, 0, 0]);
        let f32_data = TiffData::F32(vec![0.0, 0.0, 0.0]);

        assert_eq!(u8_data.byte_len(), 3);
        assert_eq!(u16_data.byte_len(), 6);
        assert_eq!(f32_data.byte_len(), 12);
    }

    #[test]
    fn test_as_f32_variant_check() {
        let f32_vec = vec![1.0, 2.0];
        let data_f32 = TiffData::F32(f32_vec.clone());
        let data_u8 = TiffData::U8(vec![1, 2]);

        assert_eq!(data_f32.as_f32(), f32_vec);
        assert_eq!(data_u8.as_f32(), vec![1.0, 2.0]);
    }

    #[test_log::test]
    fn test_geotiff_get_f32_integration() {
        let meta = create_mock_metadata(2, 2);
        let geotiff = GeoTiff {
            metadata: meta,
            data: TiffData::U16(vec![100, 200, 300, 400]),
        };

        // Test getting value through the high-level GeoTiff struct
        assert_eq!(geotiff.get_f32(1, 1), Some(400.0));
        assert_eq!(geotiff.get_f32(2, 0), None); // OOB width
    }

    #[test]
    fn minimum_elevation_preserves_finite_low_values() {
        assert_eq!(
            Dem::calculate_minimum_elevation(&[2.0, 0.0, -3.0, f32::NAN]),
            -3.0
        );
        assert_eq!(
            Dem::calculate_minimum_elevation(&[f32::NAN, f32::INFINITY]),
            0.0
        );
    }

    #[test]
    fn extrema_return_none_without_valid_cells() {
        let dem = Dem {
            width: 2,
            height: 1,
            data1d: vec![1.0, 2.0],
            ..Dem::default()
        };
        assert_eq!(dem.get_elevation_extrema_points(&[false, false]), None);
        assert_eq!(dem.get_elevation_extrema_points(&[true]), None);
    }

    #[test]
    fn mask_above_elevation_masks_only_values_above_threshold() {
        let dem = Dem {
            width: 2,
            height: 2,
            data1d: vec![100.0, 200.0, 300.0, f32::NAN],
            ..Dem::default()
        };

        let masked = dem.mask_above_elevation(200.0);

        assert_eq!(masked.data1d[0], 100.0);
        assert_eq!(masked.data1d[1], 200.0);
        assert!(masked.data1d[2].is_nan());
        assert!(masked.data1d[3].is_nan());

        // Original DEM should remain unchanged.
        assert_eq!(dem.data1d[0], 100.0);
        assert_eq!(dem.data1d[1], 200.0);
        assert_eq!(dem.data1d[2], 300.0);
        assert!(dem.data1d[3].is_nan());
    }

    #[test]
    fn invalid_spatial_scale_cannot_produce_an_index() {
        let dem = Dem {
            cell_size: 0.0,
            ..Dem::default()
        };
        let point = Point {
            x: 1.0,
            y: 1.0,
            z: None,
        };
        assert_eq!(dem.get_index(&point), None);
        assert_eq!(dem.interpolate_elevation(&point).z, None);
    }

    #[test]
    fn bounds_parser_rejects_malformed_values() {
        let lines = ["0", "1", "2", "3"].map(str::to_string).into_iter();
        assert_eq!(
            Dem::parse_bounds_lines(lines),
            Some(Bounds {
                xmin: 0.0,
                ymin: 1.0,
                xmax: 2.0,
                ymax: 3.0
            })
        );
        assert!(
            Dem::parse_bounds_lines(["0", "bad", "2", "3"].map(str::to_string).into_iter())
                .is_none()
        );
        assert!(
            Dem::parse_bounds_lines(["2", "1", "0", "3"].map(str::to_string).into_iter()).is_none()
        );
    }

    #[test]
    fn geotiff_access_rejects_malformed_metadata_and_indices() {
        let mut metadata = create_mock_metadata(2, 2);
        metadata.tiepoints.clear();
        let geotiff = GeoTiff {
            metadata,
            data: TiffData::U8(vec![1; 4]),
        };
        assert_eq!(geotiff.cell_to_world(0, 0), None);

        let data = TiffData::U8(vec![1; 4]);
        assert_eq!(data.get_f32(2, 0, 2), None);
        assert_eq!(data.get_f32(0, usize::MAX, 2), None);
    }

    #[test]
    fn test_dem_hash_deterministic() {
        let dem1 = Dem {
            width: 2,
            height: 2,
            bounds: Bounds {
                xmin: 0.0,
                xmax: 10.0,
                ymin: 0.0,
                ymax: 10.0,
            },
            data1d: vec![1.0, 2.0, 3.0, 4.0],
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            x: vec![0.0, 5.0],
            y: vec![0.0, 5.0],
            cell_size: 5.0,
            map_factor: 1.0,
            minimum_elevation: 1.0,
            source: String::new(),
            projection: String::new(),
        };

        let dem2 = dem1.clone();
        assert_eq!(dem1.calculate_hash(), dem2.calculate_hash());
    }

    #[test]
    fn test_dem_hash_ignores_data_field() {
        let dem1 = Dem {
            width: 2,
            height: 2,
            bounds: Bounds {
                xmin: 0.0,
                xmax: 10.0,
                ymin: 0.0,
                ymax: 10.0,
            },
            data1d: vec![1.0, 2.0, 3.0, 4.0],
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            x: vec![0.0, 5.0],
            y: vec![0.0, 5.0],
            cell_size: 5.0,
            map_factor: 1.0,
            minimum_elevation: 1.0,
            source: String::new(),
            projection: String::new(),
        };

        let mut dem2 = dem1.clone();
        // Change the 2D `data` field - hash should remain identical since only data1d is hashed
        dem2.data = vec![vec![999.0, 888.0], vec![777.0, 666.0]];
        assert_eq!(dem1.calculate_hash(), dem2.calculate_hash());
    }

    #[test]
    fn test_dem_hash_changes_on_data1d() {
        let dem1 = Dem {
            width: 2,
            height: 2,
            bounds: Bounds {
                xmin: 0.0,
                xmax: 10.0,
                ymin: 0.0,
                ymax: 10.0,
            },
            data1d: vec![1.0, 2.0, 3.0, 4.0],
            data: vec![],
            x: vec![0.0, 5.0],
            y: vec![0.0, 5.0],
            cell_size: 5.0,
            map_factor: 1.0,
            minimum_elevation: -1.0,
            source: String::new(),
            projection: String::new(),
        };

        let mut dem2 = dem1.clone();
        dem2.data1d = vec![1.0, 2.0, 3.0, 5.0];
        assert_ne!(dem1.calculate_hash(), dem2.calculate_hash());
    }
}
