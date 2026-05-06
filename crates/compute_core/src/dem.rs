use crate::utils::*;

#[derive(Default)]
pub struct Bounds {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
}

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
        }
    }
}

impl Dem {
    pub fn calculate_minimum_elevation(data1d: &[f32]) -> f32 {
        data1d
            .iter()
            .filter(|&&v| v > 0.1)
            .min_by(|a: &&f32, b: &&f32| a.total_cmp(b))
            .copied() // Convert Option<&f32> to Option<f32>
            .unwrap_or(0.0) // Provide a default if no value matches the filter
    }

    pub fn get_index(&self, pt: &Point) -> (f32, f32) {
        let dx = (pt.x - self.bounds.xmin) / (self.cell_size * self.map_factor);
        let dy = (pt.y - self.bounds.ymin) / (self.cell_size * self.map_factor);
        (dx, dy)
    }

    pub fn interpolate_elevation(&self, pt: &Point) -> Point {
        let (x, y) = self.get_index(pt);
        let z = bilinear_interpolate(x, y, &self.data);
        Point {
            x: pt.x,
            y: pt.y,
            z,
        }
    }
    pub fn parse_bounds_lines<I: Iterator<Item = String>>(lines: I) -> Option<Bounds> {
        let vals: Vec<f32> = lines.filter_map(|l| l.trim().parse::<f32>().ok()).collect();
        if vals.len() == 4 {
            Some(Bounds {
                xmin: vals[0],
                xmax: vals[2],
                ymin: vals[1],
                ymax: vals[3],
            })
        } else {
            None
        }
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
    pub fn cell_to_world(&self, col: u32, row: u32) -> (f64, f64) {
        let x = self.metadata.tiepoints[3] + (col as f64 * self.metadata.pixel_scale[0]);
        let y = self.metadata.tiepoints[4] - (row as f64 * self.metadata.pixel_scale[1]);
        (x, y)
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
        let index = row * width + col;

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
        };
        let pt = Point {
            x: 1.0,
            y: 1.0,
            z: Some(0.0),
        };
        let interp = dem.interpolate_elevation(&pt);
        assert!((interp.z.unwrap() - 10.0).abs() < 1e-6);
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
        let (x0, y0) = geotiff.cell_to_world(0, 0);
        assert_eq!(x0, 500.0);
        assert_eq!(y0, 1000.0);

        // Move 2 pixels right (2 * 10.0) and 3 pixels down (3 * 10.0)
        // Note: Y usually decreases as row index increases in GeoTIFFs
        let (x1, y1) = geotiff.cell_to_world(2, 3);
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
}
