use std::{path::PathBuf, vec::Vec};

use crate::utils::*;
use data_processor::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
// use reqwest::Client;

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
}

impl Dem {
    pub fn new() -> Self {
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
        }
    }

    // pub fn load_asc(path: PathBuf) -> Self {
    //     let file = File::open(path).map_err(|e| e.to_string())?;
    //     let reader = BufReader::new(file);
    //     let mut width = 0;
    //     let mut height = 0;
    //     let mut xll = 0.0;
    //     let mut yll = 0.0;
    //     let mut cell_size = 1.0;
    //     let mut nodata_value = -9999.0;
    //     let mut data: Vec<f32> = Vec::new();
    //     let mut header_lines = 0;
    //     for line in reader.lines() {
    //         let line = line.map_err(|e| e.to_string())?;
    //         let line = line.trim();
    //         if line.is_empty() {
    //             continue;
    //         }
    //         if header_lines < 6 {
    //             let parts: Vec<&str> = line.split_whitespace().collect();
    //             match parts[0].to_lowercase().as_str() {
    //                 "ncols" => width = parts[1].parse().map_err(|e| e.to_string())?,
    //                 "nrows" => height = parts[1].parse().map_err(|e| e.to_string())?,
    //                 "xllcenter" => xll = parts[1].parse().map_err(|e| e.to_string())?,
    //                 "yllcenter" => yll = parts[1].parse().map_err(|e| e.to_string())?,
    //                 "cellsize" => cell_size = parts[1].parse().map_err(|e| e.to_string())?,
    //                 "nodata_value" => nodata_value = parts[1].parse().map_err(|e| e.to_string())?,
    //                 _ => return Err(format!("Unknown header: {}", parts[0])),
    //             }
    //             header_lines += 1;
    //         } else {
    //             for v in line.split_whitespace() {
    //                 let val: f32 = v.parse().map_err(|e| e.to_string())?;
    //                 data.push(val);
    //             }
    //         }
    //     }
    //     let mut dem = Dem {
    //         width: width,
    //         height: height,
    //         data1d: data1d,
    //         data: Vec::new(),
    //         x: linspace(bounds.xmin, bounds.xmax, width),
    //         y: linspace(bounds.ymin, bounds.ymax, height),
    //         cell_size: (bounds.xmax - bounds.xmin) / (width - 1) as f32,
    //         bounds: bounds,
    //         map_factor: 1.0,
    //     };
    //     dem.data = to_2d(&dem.data1d, width, height);
    //     dem
    // }

    pub fn load_png_as_float32(path: PathBuf) -> Self {
        let (rgba, width, height) = read_png(path.as_path()).expect("Failed to load PNG");
        let bounds: Bounds = Dem::load_bounds(&path).expect("Failed to load bounds");
        println!("Loaded PNG {:?}: {} x {}", path.as_os_str(), width, height);
        let mut dem = Dem {
            width: width,
            height: height,
            data1d: rgba_bytes_to_f32(&rgba),
            data: Vec::new(),
            x: linspace(bounds.xmin, bounds.xmax, width),
            y: linspace(bounds.ymin, bounds.ymax, height),
            cell_size: (bounds.xmax - bounds.xmin) / (width - 1) as f32,
            bounds: bounds,
            map_factor: 1.0,
        };
        dem.data = to_2d(&dem.data1d, width, height);
        dem
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
    fn parse_bounds_lines<I: Iterator<Item = String>>(lines: I) -> Option<Bounds> {
        let vals: Vec<f32> = lines
            .filter_map(|l| l.trim().parse::<f32>().ok())
            .collect();
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

    fn load_bounds(path: &PathBuf) -> Result<Bounds, String> {
        let mut aabb_path = path.clone();
        aabb_path.set_extension("aabb");
        let file = File::open(&aabb_path).map_err(|e| format!("Failed to open bounds file: {}", e))?;
        let reader = BufReader::new(file);
        let lines = reader.lines().filter_map(|l| l.ok());
        Self::parse_bounds_lines(lines).ok_or_else(|| "Failed to parse bounds from file".to_string())
    }

    // #[cfg(feature = "reqwest")]
    // async fn load_bounds_from_url(url: &str) -> Result<Bounds, String> {
    //     let client = Client::new();
    //     let resp = client.get(url).send().await.map_err(|e| format!("Failed to fetch bounds: {}", e))?;
    //     let text = resp.text().await.map_err(|e| format!("Failed to read response text: {}", e))?;
    //     let lines = text.lines().map(|l| l.to_string());
    //     Self::parse_bounds_lines(lines).ok_or_else(|| "Failed to parse bounds from URL".to_string())
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_dem_new_defaults() {
        let dem = Dem::new();
        assert_eq!(dem.width, 0);
        assert_eq!(dem.height, 0);
        assert_eq!(dem.bounds.xmin, 0.0);
        assert_eq!(dem.bounds.xmax, 1.0);
        assert_eq!(dem.bounds.ymin, 0.0);
        assert_eq!(dem.bounds.ymax, 1.0);
        assert_eq!(dem.cell_size, 1.0);
        assert_eq!(dem.map_factor, 1.0);
        assert!(dem.data1d.is_empty());
        assert!(dem.data.is_empty());
        assert!(dem.x.is_empty());
        assert!(dem.y.is_empty());
    }

    #[test]
    fn test_get_index() {
        let mut dem = Dem::new();
        dem.bounds.xmin = 100.0;
        dem.bounds.xmax = 1000.0;
        dem.bounds.ymin = 300.0;
        dem.bounds.ymax = 2000.0;
        dem.cell_size = 5.0;
        dem.map_factor = (47.0 * PI / 180.0).cos();
        let pt = Point {
            x: 350.0,
            y: 800.0,
            z: Some(0.0),
        };
        let (dx, dy) = dem.get_index(&pt);
        assert_eq!(dx, 73.3139648);
        assert_eq!(dy, 146.627930);
    }

    #[test]
    fn test_load_png_as_float32() {
        let path = PathBuf::from("../avaframe/avaParabola.png");
        let dem = Dem::load_png_as_float32(path);
        assert_eq!(dem.width, 1001);
        assert_eq!(dem.height, 401);
        assert_eq!(dem.bounds.xmin, 1000.0);
        assert_eq!(dem.bounds.xmax, 6000.0);
        assert_eq!(dem.bounds.ymin, -5000.0);
        assert_eq!(dem.bounds.ymax, -3000.0);
        let mut expected: Vec<f32> = vec![
            2200.0,
            2193.260085,
            2186.530510,
            2179.811275,
            2173.102380,
            2166.403825,
            2159.715610,
            2153.037735,
            2146.370200,
            2139.713005,
            2133.066150,
            2126.429636,
        ];
        add(&mut expected, 1.0);
        if dem.data1d[..expected.len()] != expected[..] {
            println!("Expected: {:?}", expected);
            println!("Actual:   {:?}", &dem.data1d[..expected.len()]);
        }
        assert_eq!(dem.data1d[..expected.len()], expected[..]);
    }
    #[test]
    fn test_load_bounds() {
        let path = PathBuf::from("../avaframe/avaInclinedPlane.png");
        let bounds = Dem::load_bounds(&path).expect("Failed to load bounds");
        assert_eq!(bounds.xmin, 1000.0);
        assert_eq!(bounds.xmax, 6000.0);
        assert_eq!(bounds.ymin, -5000.0);
        assert_eq!(bounds.ymax, -3000.0);
    }
    #[test]
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
        };
        let pt = Point {
            x: 1.0,
            y: 1.0,
            z: Some(0.0),
        };
        let interp = dem.interpolate_elevation(&pt);
        assert!((interp.z.unwrap() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_fetch_bounds_returns_default() {
        let path = PathBuf::from("dummy/path");
        let result = std::panic::catch_unwind(|| {
            Dem::load_bounds(&path).unwrap()
        });
        assert!(result.is_err(), "Expected panic when loading bounds from a non-existent file");
    }
}
