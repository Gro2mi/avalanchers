use image::{GenericImageView, ImageReader};
use std::fs::File;
use std::io::Cursor;
use std::io::{self, BufWriter, Write};
use std::io::{BufRead, BufReader};
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::vec::Vec;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use compute_core::dem::{Bounds, Dem, GeoMetadata, GeoTiff, TiffData};
use compute_core::settings::{Settings, SimSettings};
use compute_core::utils::*;

#[cfg(target_arch = "wasm32")]
pub mod blosc;
pub mod caaml_parser;
pub mod rasterizer;
pub mod shapefile_reader;
#[cfg(target_arch = "wasm32")]
pub mod zarr_writer;
use rasterizer::RasterGrid;

#[cfg(not(target_arch = "wasm32"))]
use tile_manager::TileManager;
#[cfg(not(target_arch = "wasm32"))]
pub mod output;
#[cfg(not(target_arch = "wasm32"))]
pub mod tile_manager;

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

#[derive(thiserror::Error, Debug)]
pub enum DataProcessorError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error("Failed to read shapefile: {0}")]
    ShapefileError(#[from] crate::shapefile_reader::ShapeError),
    #[error("Failed to rasterize: {0}")]
    RasterError(#[from] crate::rasterizer::RasterizerError),
    #[error("Failed to read DEM: {0}")]
    DemError(String),
    #[error("Failed to create sim settings: {0}")]
    SimSettingsError(String),
    #[error("No outlines padding provided")]
    NoneOutlinesPadding,
    #[error("No outlines path provided")]
    NoneOutlinesPath,
    #[cfg(not(target_arch = "wasm32"))]
    #[error("Failed to create tile manager")]
    TileManagerError(#[from] crate::tile_manager::TileManagerError),
    #[error("Unsupported DEM format: {0}")]
    UnsupportedDemFormat(String),
    #[error("No DEM provided")]
    NoneDem,
    #[error("No avalanche path provided")]
    NoneAvalanchePath,
    #[error(transparent)]
    EsriGrid(#[from] EsriGridError),
    #[error(transparent)]
    GeoTiff(#[from] GeoTiffError),
}

#[derive(Debug, thiserror::Error)]
pub enum EsriGridError {
    #[error("Error reading header line: {0}")]
    HeaderRead(std::io::Error),

    #[error("Unexpected end of file while parsing header")]
    UnexpectedEof,

    #[error("Invalid header line: '{0}'")]
    InvalidHeaderLine(String),

    #[error("Unknown header key: {0}")]
    UnknownHeaderKey(String),

    #[error("Invalid ncols value")]
    InvalidNcols,

    #[error("Invalid nrows value")]
    InvalidNrows,

    #[error("Invalid xllcorner value")]
    InvalidXllCorner,

    #[error("Invalid yllcorner value")]
    InvalidYllCorner,

    #[error("Invalid xllcenter value")]
    InvalidXllCenter,

    #[error("Invalid yllcenter value")]
    InvalidYllCenter,

    #[error("Invalid cellsize value")]
    InvalidCellsize,

    #[error("Invalid nodata_value")]
    InvalidNodataValue,

    #[error("Missing X origin coordinate (xllcorner or xllcenter)")]
    MissingXOrigin,

    #[error("Missing Y origin coordinate (yllcorner or yllcenter)")]
    MissingYOrigin,

    #[error("Error reading data line: {0}")]
    DataRead(std::io::Error),

    #[error("Failed to parse data token: '{0}'")]
    InvalidDataToken(String),

    #[error("Data size mismatch. Expected {expected} values ({ncols}x{nrows}), but found {found}")]
    DataSizeMismatch {
        expected: usize,
        ncols: usize,
        nrows: usize,
        found: usize,
    },
}

#[derive(Debug, Clone)]
pub struct EsriGridHeader {
    pub ncols: usize,
    pub nrows: usize,
    pub cellsize: f32,
    pub nodata_value: Option<f32>,
    pub xllcorner: Option<f32>,
    pub yllcorner: Option<f32>,
    pub xllcenter: Option<f32>,
    pub yllcenter: Option<f32>,
}

impl EsriGridHeader {
    /// Computes or retrieves the absolute lower-left corner (outer edge) of the grid
    pub fn get_xllcorner(&self) -> f32 {
        match self.xllcorner {
            Some(corner) => corner,
            None => {
                self.xllcenter
                    .expect("Header must have xllcorner or xllcenter")
                    - (self.cellsize / 2.0)
            }
        }
    }

    pub fn get_yllcorner(&self) -> f32 {
        match self.yllcorner {
            Some(corner) => corner,
            None => {
                self.yllcenter
                    .expect("Header must have yllcorner or yllcenter")
                    - (self.cellsize / 2.0)
            }
        }
    }

    /// Computes or retrieves the center coordinate of the bottom-left cell
    pub fn get_xllcenter(&self) -> f32 {
        match self.xllcenter {
            Some(center) => center,
            None => {
                self.xllcorner
                    .expect("Header must have xllcorner or xllcenter")
                    + (self.cellsize / 2.0)
            }
        }
    }

    pub fn get_yllcenter(&self) -> f32 {
        match self.yllcenter {
            Some(center) => center,
            None => {
                self.yllcorner
                    .expect("Header must have yllcorner or yllcenter")
                    + (self.cellsize / 2.0)
            }
        }
    }
}

#[derive(Debug)]
pub struct EsriGrid {
    pub header: EsriGridHeader,
    /// Flat vector representing the 2D grid in row-major order
    pub data: Vec<f32>,
}

impl EsriGrid {
    /// Parses an ESRI ASCII grid file from a given path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, DataProcessorError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self::from_reader(reader)?)
    }

    /// Parses the grid from any type implementing `BufRead`.
    pub fn from_reader<R: BufRead>(mut reader: R) -> Result<Self, EsriGridError> {
        let mut ncols = 0;
        let mut nrows = 0;
        let mut xllcorner = None;
        let mut yllcorner = None;
        let mut xllcenter = None;
        let mut yllcenter = None;
        let mut cellsize = 0.0;
        let mut nodata_value = None;

        let mut lines_iter = reader.by_ref().lines();

        // 1. Parse Header (First 6 metadata lines)
        for _ in 0..6 {
            let line = match lines_iter.next() {
                Some(Ok(l)) => l,
                Some(Err(e)) => return Err(EsriGridError::HeaderRead(e)),
                None => return Err(EsriGridError::UnexpectedEof),
            };

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                return Err(EsriGridError::InvalidHeaderLine(line));
            }
            let key = parts[0].to_lowercase();
            let value = parts[1];

            match key.as_str() {
                "ncols" => ncols = value.parse().map_err(|_| EsriGridError::InvalidNcols)?,
                "nrows" => nrows = value.parse().map_err(|_| EsriGridError::InvalidNrows)?,
                "xllcorner" => {
                    xllcorner = Some(value.parse().map_err(|_| EsriGridError::InvalidXllCorner)?)
                }
                "yllcorner" => {
                    yllcorner = Some(value.parse().map_err(|_| EsriGridError::InvalidYllCorner)?)
                }
                "xllcenter" => {
                    xllcenter = Some(value.parse().map_err(|_| EsriGridError::InvalidXllCenter)?)
                }
                "yllcenter" => {
                    yllcenter = Some(value.parse().map_err(|_| EsriGridError::InvalidYllCenter)?)
                }
                "cellsize" => {
                    cellsize = value.parse().map_err(|_| EsriGridError::InvalidCellsize)?
                }
                "nodata_value" => {
                    nodata_value = Some(
                        value
                            .parse()
                            .map_err(|_| EsriGridError::InvalidNodataValue)?,
                    )
                }
                _ => return Err(EsriGridError::UnknownHeaderKey(parts[0].to_owned())),
            }
        }
        if xllcorner.is_none() && xllcenter.is_none() {
            return Err(EsriGridError::MissingXOrigin);
        }

        if yllcorner.is_none() && yllcenter.is_none() {
            return Err(EsriGridError::MissingYOrigin);
        }

        let header = EsriGridHeader {
            ncols,
            nrows,
            xllcorner,
            yllcorner,
            xllcenter,
            yllcenter,
            cellsize,
            nodata_value,
        };

        // 2. Parse Data Matrix
        // Pre-allocate space to avoid re-allocations during push loops
        let mut data: Vec<f32> = Vec::with_capacity(header.ncols * header.nrows);

        for line_result in lines_iter {
            let line = line_result.map_err(EsriGridError::DataRead)?;

            // Skip purely empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Split by whitespace and parse values
            for token in line.split_whitespace() {
                let val: f32 = token
                    .parse()
                    .map_err(|_| EsriGridError::InvalidDataToken(token.to_owned()))?;
                data.push(val);
            }
        }

        // 3. Validation Check
        let expected_size = header.ncols * header.nrows;
        if data.len() != expected_size {
            return Err(EsriGridError::DataSizeMismatch {
                expected: expected_size,
                ncols: header.ncols,
                nrows: header.nrows,
                found: data.len(),
            });
        }

        Ok(EsriGrid { header, data })
    }

    pub fn flip_y(&mut self) {
        // Perform the flip
        flip_rows_flat_vec(
            &mut self.data,
            self.header.ncols as u32,
            self.header.nrows as u32,
        );
    }

    /// Helper to look up a value at a given 2D index (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.header.nrows && col < self.header.ncols {
            let index = row * self.header.ncols + col;
            Some(self.data[index])
        } else {
            None
        }
    }

    /// Helper to export the grid directly to a file path.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let writer = BufWriter::new(file);
        self.to_writer(writer)
    }

    /// Exports the ESRI ASCII grid layout to any stream implementing `Write`.
    pub fn to_writer<W: Write>(&self, mut writer: W) -> Result<(), String> {
        // 1. Write metadata headers
        writeln!(writer, "ncols         {}", self.header.ncols)
            .map_err(|e| format!("Header write error: {}", e))?;
        writeln!(writer, "nrows         {}", self.header.nrows)
            .map_err(|e| format!("Header write error: {}", e))?;

        // Fallback checks to output whatever raw metadata spatial scheme was provided
        if let Some(xcenter) = self.header.xllcenter {
            writeln!(writer, "xllcenter     {}", xcenter)
                .map_err(|e| format!("Header write error: {}", e))?;
        } else {
            writeln!(writer, "xllcorner     {}", self.header.get_xllcorner())
                .map_err(|e| format!("Header write error: {}", e))?;
        }

        if let Some(ycenter) = self.header.yllcenter {
            writeln!(writer, "yllcenter     {}", ycenter)
                .map_err(|e| format!("Header write error: {}", e))?;
        } else {
            writeln!(writer, "yllcorner     {}", self.header.get_yllcorner())
                .map_err(|e| format!("Header write error: {}", e))?;
        }

        writeln!(writer, "cellsize      {}", self.header.cellsize)
            .map_err(|e| format!("Header write error: {}", e))?;

        // Optional nodata configuration mapping
        if let Some(nodata) = self.header.nodata_value {
            writeln!(writer, "NODATA_value  {}", nodata)
                .map_err(|e| format!("Header write error: {}", e))?;
        }

        // 2. Format and stream matrix data blocks row by row
        let nodata_val = self.header.nodata_value.unwrap_or(-9999.0);

        // Iterate through the vector by splitting it into fixed-size row chunks
        for row in self.data.chunks_exact(self.header.ncols) {
            let mut first = true;
            for &val in row {
                if !first {
                    write!(writer, " ").map_err(|e| format!("Data write error: {}", e))?;
                }
                first = false;

                // Handle NaN filtering
                let target_val = if val.is_nan() { nodata_val } else { val };

                // Clean formatting optimization: strips trailing decimals if the float is a whole integer
                if target_val.fract() == 0.0 {
                    write!(writer, "{}", target_val as i64)
                        .map_err(|e| format!("Data write error: {}", e))?;
                } else {
                    write!(writer, "{}", target_val)
                        .map_err(|e| format!("Data write error: {}", e))?;
                }
            }
            // End the current row with a system newline character
            writeln!(writer).map_err(|e| format!("Data newline error: {}", e))?;
        }

        // Explicitly flush the final byte streams out of buffers
        writer
            .flush()
            .map_err(|e| format!("Buffer flush error: {}", e))?;

        Ok(())
    }
}

pub fn write_png(
    path: &Path,
    data: &[u8],
    width: usize,
    height: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Avoid copying data by using a slice reference instead of to_vec()
    let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width as u32, height as u32, data)
        .ok_or("Failed to create image buffer")?;
    img.save(path.with_extension("png"))?;
    Ok(())
}

pub async fn read_file(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    trace!("Reading file: {}", path);
    if path.starts_with("http") {
        fetch_from_url(path).await
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(std::fs::read(PathBuf::from(path))?)
        }
        #[cfg(target_arch = "wasm32")]
        {
            // In WASM, treat local paths as relative HTTP fetches
            let window = web_sys::window().ok_or("no global window")?;
            let location = window.location();
            let origin = location.origin().map_err(|_| "could not get origin")?;

            let url = format!("{}/{}", origin, path);
            fetch_from_url(&url).await
        }
    }
}

pub async fn read_file_to_string(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = read_file(path).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

pub async fn read_png(path: &str) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error>> {
    let bytes = read_file(PathBuf::from(path).with_extension("png").to_str().unwrap()).await?;
    // Decode from memory bytes
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?;

    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();

    Ok((rgba.into_raw(), width as usize, height as usize))
}

pub fn rgba_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn f32_to_rgba_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub async fn fetch_from_url(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    trace!("Fetching from URL: {}", url);
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

#[derive(Debug, thiserror::Error)]
pub enum GeoTiffError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Tiff(#[from] tiff::TiffError),

    #[error("Missing pixel scale metadata (Tag 33550)")]
    MissingPixelScale,

    #[error("Non-uniform grid detected: X and Y scales must match for square cells")]
    NonUniformGrid,

    #[error("TIFF missing GeoTIFF tie point metadata (Tag 33922)")]
    MissingTiePoints,

    #[error("Unsupported GeoTIFF: Only a single tie point at the origin is supported")]
    UnsupportedTiePoint,

    #[error("Unsupported TIFF data type")]
    UnsupportedDataType,
}
pub fn read_geo_tiff(path: &str) -> Result<GeoTiff, GeoTiffError> {
    let file = File::open(path)?;
    read_geo_tiff_from_reader(BufReader::new(file))
}

/// Parses a GeoTIFF from raw bytes, e.g. a file uploaded in the browser.
pub fn read_geo_tiff_from_bytes(bytes: &[u8]) -> Result<GeoTiff, GeoTiffError> {
    read_geo_tiff_from_reader(Cursor::new(bytes))
}

pub fn read_geo_tiff_from_reader<R: Read + Seek>(reader: R) -> Result<GeoTiff, GeoTiffError> {
    let mut decoder = Decoder::new(reader)?;

    let (width, height) = decoder.dimensions()?;

    // --- Metadata Extraction ---
    // Tag 33550: ModelPixelScaleTag
    let pixel_scales = decoder
        .get_tag_f64_vec(Tag::Unknown(33550))
        .unwrap_or_default();

    if pixel_scales.len() <= 2 {
        return Err(GeoTiffError::MissingPixelScale);
    }
    if pixel_scales[0] != pixel_scales[1] {
        return Err(GeoTiffError::NonUniformGrid);
    }

    // Tag 33922: ModelTiepointTag
    let tie_points = decoder
        .get_tag_f64_vec(Tag::Unknown(33922))
        .map_err(|_| GeoTiffError::MissingTiePoints)?;
    let nodata = decoder
        .get_tag_f64_vec(Tag::Unknown(42113)) // Try to get the tag
        .ok() // If it fails (missing or wrong type), return None
        .and_then(|v| v.first().copied());
    if tie_points[0] != 0.0 || tie_points[1] != 0.0 || tie_points[2] != 0.0 {
        return Err(GeoTiffError::UnsupportedTiePoint);
    }

    let metadata = if tie_points.len() >= 6 {
        if tie_points[0] != 0.0 || tie_points[1] != 0.0 || tie_points[2] != 0.0 {
            return Err(GeoTiffError::UnsupportedTiePoint);
        }
        let origin_x = tie_points[3];
        let origin_y = tie_points[4];
        let bounds = Bounds {
            xmin: origin_x as f32,
            ymax: origin_y as f32,
            xmax: (origin_x + width as f64 * pixel_scales[0]) as f32,
            ymin: (origin_y - height as f64 * pixel_scales[1]) as f32,
        };

        GeoMetadata {
            width,
            height,
            pixel_scale: [
                pixel_scales[0],
                pixel_scales[1],
                pixel_scales.get(2).cloned().unwrap_or(0.0),
            ],
            tiepoints: tie_points,
            bounds,
            cell_size: pixel_scales[0] as f32,
            // Assuming EPSG 4326 or similar; extraction of GeoKeyDirectoryTag is complex
            // and usually requires a dedicated crate like 'geotiff'
            epsg_code: 0,
            nodata,
        }
    } else {
        return Err(GeoTiffError::MissingTiePoints);
    };

    // --- Image Data Extraction ---
    // Note: To keep GeoTiff simple, we often normalize data to f32 for simulations
    let data = match decoder.read_image()? {
        DecodingResult::U8(data) => TiffData::U8(data),
        DecodingResult::U16(data) => TiffData::U16(data),
        DecodingResult::F32(data) => TiffData::F32(data),
        _ => return Err(GeoTiffError::UnsupportedDataType),
    };
    info!(
        "Loaded GeoTIFF: {}x{} cells at {}m resolution",
        width, height, metadata.pixel_scale[0]
    );

    Ok(GeoTiff { metadata, data })
}

async fn load_png_as_dem(path: &str) -> Result<Dem, DataProcessorError> {
    let (rgba, width, height) = read_png(path).await.expect("Failed to load PNG");
    let mut bounds: Bounds = load_bounds(path).await.expect("Failed to load bounds");
    debug!("Loaded PNG {}: {} x {}", path, width, height);
    let cell_size = (bounds.xmax - bounds.xmin) / (width - 1) as f32;
    bounds.xmin -= 0.5 * cell_size;
    bounds.xmax += 0.5 * cell_size;
    bounds.ymin -= 0.5 * cell_size;
    bounds.ymax += 0.5 * cell_size;
    let data1d = rgba_bytes_to_f32(&rgba)
        .into_iter()
        .map(|value| if value == 0.0 { f32::NAN } else { value })
        .collect();
    let mut dem = Dem {
        width,
        height,
        // Legacy float-PNG DEMs use an all-zero pixel as their NoData sentinel.
        data1d,
        data: Vec::new(),
        // TODO bounds were exported wrong. Rework when png files get metadata embedded.
        x: linspace(bounds.xmin, bounds.xmax, width),
        y: linspace(bounds.ymin, bounds.ymax, height),
        cell_size,
        bounds,
        map_factor: 1.0,
        minimum_elevation: f32::INFINITY,
        source: path.to_string(),
        projection: "unknown".to_string(),
    };
    dem.data = to_2d(&dem.data1d, width, height);
    Ok(dem)
}

fn load_asc_as_dem(path: &str) -> Result<Dem, DataProcessorError> {
    let grid = EsriGrid::from_file(path)?;
    Ok(esri_grid_to_dem(grid, path))
}

fn esri_grid_to_dem(mut grid: EsriGrid, source: &str) -> Dem {
    flip_rows_flat_vec(
        &mut grid.data,
        grid.header.ncols as u32,
        grid.header.nrows as u32,
    );
    let bounds = Bounds {
        xmin: grid.header.get_xllcorner(),
        xmax: grid.header.get_xllcorner() + grid.header.ncols as f32 * grid.header.cellsize,
        ymin: grid.header.get_yllcorner(),
        ymax: grid.header.get_yllcorner() + grid.header.nrows as f32 * grid.header.cellsize,
    };
    if let Some(nodata) = grid.header.nodata_value {
        for value in &mut grid.data {
            if *value == nodata {
                *value = f32::NAN;
            }
        }
    }
    let mut dem = Dem {
        width: grid.header.ncols,
        height: grid.header.nrows,
        data1d: grid.data,
        data: Vec::new(),
        x: linspace(bounds.xmin, bounds.xmax, grid.header.ncols),
        y: linspace(bounds.ymin, bounds.ymax, grid.header.nrows),
        cell_size: grid.header.cellsize,
        bounds,
        map_factor: 1.0,
        minimum_elevation: f32::INFINITY,
        source: source.to_string(),
        projection: "unknown".to_string(),
    };
    dem.data = to_2d(&dem.data1d, dem.width, dem.height);
    dem
}

fn load_tiff_as_dem(path: &str) -> Result<Dem, DataProcessorError> {
    let tiff: GeoTiff = read_geo_tiff(path)?;
    Ok(geo_tiff_to_dem(tiff, path))
}

fn geo_tiff_to_dem(mut tiff: GeoTiff, source: &str) -> Dem {
    tiff.flip_y();
    let nodata = tiff.metadata.nodata.map(|value| value as f32);
    let data1d = tiff
        .data
        .as_f32()
        .into_iter()
        .map(|value| {
            if nodata.is_some_and(|nodata| value == nodata || nodata.is_nan() && value.is_nan()) {
                f32::NAN
            } else {
                value
            }
        })
        .collect();
    let mut dem = Dem {
        width: tiff.metadata.width as usize,
        height: tiff.metadata.height as usize,
        data1d,
        data: Vec::new(),
        x: linspace(
            tiff.metadata.bounds.xmin,
            tiff.metadata.bounds.xmax,
            tiff.metadata.width as usize,
        ),
        y: linspace(
            tiff.metadata.bounds.ymin,
            tiff.metadata.bounds.ymax,
            tiff.metadata.height as usize,
        ),
        cell_size: tiff.metadata.cell_size,
        bounds: tiff.metadata.bounds,
        map_factor: 1.0,
        minimum_elevation: f32::INFINITY, // Will be calculated later
        source: source.to_string(),
        projection: format!("EPSG:{}", tiff.metadata.epsg_code),
    };
    dem.data = to_2d(&dem.data1d, dem.width, dem.height);
    dem
}

pub async fn load_release_areas(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let data: Vec<f32> = match ext.to_lowercase().as_str() {
        "asc" => {
            let mut grid = EsriGrid::from_file(path)?;
            grid.flip_y();
            grid.data
        }
        "png" => {
            let (rgba, _, _) = read_png(path).await?;
            rgba.iter()
                .skip(3) // Skip the first 3 items (starts at index 3, which is the 4th item)
                .step_by(4) // Take every 4th item from that point onward
                .map(|&val| (val as f32) / 100.0) // Convert &u8 back to u8
                .collect()
        }
        "tif" | "tiff" => {
            let mut tiff = read_geo_tiff(path)?;
            tiff.flip_y();
            tiff.data.as_f32()
        }
        _ => return Err(format!("Unsupported release format: {}", ext).into()),
    };
    Ok(data)
}

/// Loads release areas from raw bytes. `ext` selects the parser, e.g. "asc" or "tif".
pub fn load_release_areas_from_bytes(
    bytes: &[u8],
    ext: &str,
) -> Result<Vec<f32>, DataProcessorError> {
    let data: Vec<f32> = match ext.to_lowercase().as_str() {
        "asc" => {
            let mut grid = EsriGrid::from_reader(Cursor::new(bytes))?;
            grid.flip_y();
            grid.data
        }
        "tif" | "tiff" => {
            let mut tiff = read_geo_tiff_from_bytes(bytes)?;
            tiff.flip_y();
            tiff.data.as_f32()
        }
        _ => return Err(DataProcessorError::UnsupportedDemFormat(ext.to_string())),
    };
    Ok(data)
}

pub async fn load_dem(path: &str) -> Result<Dem, DataProcessorError> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let dem: Dem = match ext.to_lowercase().as_str() {
        "asc" => load_asc_as_dem(path)?,
        "png" => load_png_as_dem(path).await?,
        "tif" | "tiff" => load_tiff_as_dem(path)?,
        _ => return Err(DataProcessorError::UnsupportedDemFormat(ext.to_string())),
    };

    finalize_dem(dem)
}

/// Loads a DEM from raw bytes. `ext` selects the parser, e.g. "asc" or "tif".
pub fn load_dem_from_bytes(
    bytes: &[u8],
    ext: &str,
    source: &str,
) -> Result<Dem, DataProcessorError> {
    let dem: Dem = match ext.to_lowercase().as_str() {
        "asc" => esri_grid_to_dem(EsriGrid::from_reader(Cursor::new(bytes))?, source),
        "tif" | "tiff" => geo_tiff_to_dem(read_geo_tiff_from_bytes(bytes)?, source),
        _ => return Err(DataProcessorError::UnsupportedDemFormat(ext.to_string())),
    };

    finalize_dem(dem)
}

/// Validates the DEM and synchronizes its flat and row-oriented representations.
fn finalize_dem(mut dem: Dem) -> Result<Dem, DataProcessorError> {
    let expected_len = dem
        .width
        .checked_mul(dem.height)
        .ok_or_else(|| DataProcessorError::DemError("DEM dimensions overflow".to_string()))?;
    if dem.data1d.len() != expected_len {
        return Err(DataProcessorError::DemError(format!(
            "DEM data length {} does not match dimensions {}x{}",
            dem.data1d.len(),
            dem.width,
            dem.height
        )));
    }
    dem.minimum_elevation = Dem::calculate_minimum_elevation(&dem.data1d);
    dem.data = to_2d(&dem.data1d, dem.width, dem.height);

    if dem.bounds.xmin >= dem.bounds.xmax {
        return Err(DataProcessorError::DemError(format!(
            "xmin ({}) must be less than xmax ({})",
            dem.bounds.xmin, dem.bounds.xmax
        )));
    }
    if dem.bounds.ymin >= dem.bounds.ymax {
        return Err(DataProcessorError::DemError(format!(
            "ymin ({}) must be less than ymax ({})",
            dem.bounds.ymin, dem.bounds.ymax
        )));
    }

    Ok(dem)
}

async fn load_bounds(path: &str) -> Result<Bounds, String> {
    let mut aabb_path = PathBuf::from(path);
    aabb_path.set_extension("aabb");
    let bytes = read_file(aabb_path.to_str().expect("Load bounds file failed"))
        .await
        .map_err(|e| e.to_string())?;
    let reader = BufReader::new(&bytes[..]);
    let lines = reader.lines().map_while(Result::ok);
    Dem::parse_bounds_lines(lines).ok_or_else(|| "Failed to parse bounds from file".to_string())
}

fn load_outline(path: &str, padding: f32) -> Result<RasterGrid, DataProcessorError> {
    let geo_poly = shapefile_reader::read_shapefile_nth_polygon(path, 0)?;
    let mut outline = RasterGrid::from_polygon(&geo_poly, 5.0);
    outline.add_padding(padding as f64)?;
    Ok(outline)
}

pub async fn create_sim_settings_and_dem_from_path(
    file_path: &str,
) -> Result<(SimSettings, Dem, Vec<bool>), DataProcessorError> {
    let settings = Settings {
        dem_path: Some(file_path.to_string()),
        ..Default::default()
    };
    create_sim_settings_and_dem(&settings).await
}
pub async fn create_sim_settings_and_dem(
    settings: &Settings,
) -> Result<(SimSettings, Dem, Vec<bool>), DataProcessorError> {
    let (outline, dem) = match (&settings.outlines_path, &settings.dem_path) {
        // 1. Outline only -> build outline and DEM from tile manager
        #[cfg(target_arch = "wasm32")]
        (Some(_outline_path), None) => {
            todo!("Tile manager is not supported in WASM. Please provide a DEM path.");
        }
        #[cfg(not(target_arch = "wasm32"))]
        (Some(outline_path), None) => {
            let padding = settings
                .outlines_padding
                .ok_or(DataProcessorError::NoneOutlinesPadding)?;
            let outline = load_outline(outline_path, padding)?;
            if !outline.data.iter().any(|&v| v) {
                return Err(DataProcessorError::DemError(format!(
                    "case {}: rasterised outline is empty",
                    outline_path
                )));
            }
            let tile_manager = TileManager::new("dtm_cache.zarr")?;
            let dem = tile_manager.get_dem(&outline.get_bbox()).await?;

            if dem.data1d.iter().any(|v: &f32| v.is_nan()) {
                return Err(DataProcessorError::DemError(format!(
                    "case {}: DEM contains NaN (missing swissALTI3D coverage)",
                    outline_path
                )));
            }

            // let (_min_elevation, max_elevation) = dem.get_elevation_extrema(&outline.data).unwrap();

            // let dem = dem.mask_above_elevation(max_elevation + 20.0);

            (outline.data, dem)
        }

        // 2. DEM only -> load DEM and create an all-ones outline
        (None, Some(dem_path)) => {
            let dem = load_dem(dem_path).await?;
            (vec![true; dem.width * dem.height], dem)
        }

        // 3. Both provided -> load outline and DEM from file
        (Some(outline_path), Some(dem_path)) => {
            let padding = settings
                .outlines_padding
                .ok_or(DataProcessorError::NoneOutlinesPadding)?;
            let outline = load_outline(outline_path, padding)?;

            let dem = load_dem(dem_path).await?;

            (outline.data, dem)
        }

        // Neither provided
        (None, None) => (Vec::new(), Dem::default()),
    };

    let sim_settings = SimSettings::from_settings(settings, &dem);

    Ok((sim_settings, dem, outline))
}

pub fn settings_to_json_file(settings: &Settings, path: &str) -> io::Result<()> {
    let json = settings.dumps()?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

pub fn settings_from_json_file(path: &str) -> io::Result<Settings> {
    let data = std::fs::read_to_string(path)?;
    let settings = Settings::loads(&data)?;
    Ok(settings)
}

pub async fn sim_settings_and_dem_from_json_file(file_path: &str) -> (SimSettings, Dem, Vec<bool>) {
    let data = std::fs::read_to_string(file_path).expect("Failed to read json file");
    let settings = Settings::loads(&data).expect("Failed to load settings from JSON file");
    create_sim_settings_and_dem(&settings)
        .await
        .expect("Failed to create sim settings and dem")
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use pollster::block_on;
    use std::f32::consts::PI;
    use std::fs;
    use std::fs::File;
    use std::io::Cursor;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use compute_core::settings::{FrictionModel, SimModel};

    const PARABOLA_PATH: &str = "../../data/avaframe/avaParabola.png";

    #[test_log::test]
    fn test_dem_new_defaults() {
        let dem = Dem::default();
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

    #[test_log::test]
    fn test_get_index() {
        let mut dem: Dem = block_on(load_dem(&PARABOLA_PATH.to_string())).unwrap();
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
        let (dx, dy) = dem.get_index(&pt).expect("DEM scale should be valid");
        assert_eq!(dx, 73.3139648);
        assert_eq!(dy, 146.627930);
    }

    #[test_log::test]
    fn test_load_png_as_float32() {
        let path = "../../data/avaframe/avaParabola.png";
        let dem: Dem = block_on(load_dem(path)).expect("Failed to load PNG as float32");
        assert_eq!(dem.width, 1001);
        assert_eq!(dem.height, 401);
        assert_eq!(dem.bounds.xmin, 997.5);
        assert_eq!(dem.bounds.xmax, 6002.5);
        assert_eq!(dem.bounds.ymin, -5002.5);
        assert_eq!(dem.bounds.ymax, -2997.5);
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
    #[test_log::test]
    fn test_load_bounds() {
        let path = "../../data/avaframe/avaInclinedPlane.png";
        let bounds = block_on(load_bounds(path)).expect("Failed to load bounds");
        assert_eq!(bounds.xmin, 1000.0);
        assert_eq!(bounds.xmax, 6000.0);
        assert_eq!(bounds.ymin, -5000.0);
        assert_eq!(bounds.ymax, -3000.0);
    }

    #[test_log::test]
    fn test_fetch_bounds_returns_default() {
        let path = "dummy/path";
        let result = std::panic::catch_unwind(|| block_on(load_bounds(path)).unwrap());
        assert!(
            result.is_err(),
            "Expected panic when loading bounds from a non-existent file"
        );
    }

    #[test_log::test]
    fn test_tiff() {
        let tiff = read_geo_tiff("../../data/vals/PAR6_Vals_Gries_dtm_10_utm32n_bil_.tif").unwrap();
        assert_eq!(
            tiff.data.byte_len(),
            (tiff.metadata.width * tiff.metadata.height * 4) as usize
        );
        assert_eq!(tiff.get_f32(500, 500).unwrap(), 1370.8146);
        assert_eq!(tiff.metadata.cell_size, 10.0);
        assert_eq!(tiff.metadata.bounds.xmin, 684366.3320);
        assert_eq!(tiff.metadata.bounds.ymin, 5205162.1636);
        assert_eq!(
            tiff.metadata.bounds.xmax,
            684366.3320 + 10.0 * tiff.metadata.width as f32
        );
        assert_eq!(
            tiff.metadata.bounds.ymax,
            5205162.1636 + 10.0 * tiff.metadata.height as f32
        );
    }

    #[tokio::test]
    async fn test_fetch_from_url_valid() {
        let url =
            "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png";
        let result = fetch_from_url(url).await;

        assert!(result.is_ok(), "Should successfully fetch the PNG");
        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "Fetched bytes should not be empty");
        // Verify PNG magic numbers: [0x89, 0x50, 0x4E, 0x47]
        assert_eq!(&bytes[0..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[tokio::test]
    async fn test_fetch_from_url_invalid() {
        let url = "https://this.url.does.not.exist.internal";
        let result = fetch_from_url(url).await;
        assert!(result.is_err(), "Should return error for non-existent URL");
    }

    #[test_log::test]
    fn test_settings_to_json_and_from_json() {
        let settings = Settings {
            output_path: Some(String::from("avalanchers.zarr")),
            outlines_path: Some(String::from("outline.shp")),
            outlines_padding: Some(5.0),
            dem_path: Some(String::from("dem.png")),
            release_areas_path: Some(String::from("release_areas.png")),
            max_steps: Some(100),
            sim_model: Some(SimModel::MPM),
            friction_model: Some(FrictionModel::Voellmy),
            released_particles_per_cell: Some(3),
            density: Some(4.0),
            slab_thickness_factor: Some(5.0),
            friction_coefficient: Some(6.0),
            drag_coefficient: Some(7.0),
            n0: Some(1.0),
            i0: Some(2.0),
            mu0: Some(0.1),
            mu2: Some(0.3),
            grain_diameter: Some(0.5),
            internal_friction_angle: Some(30.0),
            basal_friction_angle: Some(45.0),
            batch_compute_steps: Some(100),
            cfl: Some(8.0),
            min_slope_angle: Some(9.0),
            max_slope_angle: Some(10.0),
            release_min_elevation: Some(11.0),
            release_max_elevation: Some(13.0),
            velocity_threshold: Some(12.0),
            roughness_threshold: Some(13.0),
            peak_flow_thickness_threshold: Some(14.0),
            enable_curvature: Some(true),
            enable_entrainment: Some(true),
            enable_particle_interaction: Some(true),
            enable_earth_pressure_coefficient: Some(true),
        };
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        settings_to_json_file(&settings, path).unwrap();

        let loaded = settings_from_json_file(path).unwrap();
        assert_eq!(loaded.dem_path, Some(String::from("dem.png")));
        assert_eq!(
            loaded.release_areas_path,
            Some(String::from("release_areas.png"))
        );
        assert_eq!(loaded.max_steps, Some(100));
        assert_eq!(loaded.sim_model, Some(SimModel::MPM));
        assert_eq!(loaded.friction_model, Some(FrictionModel::Voellmy));
        assert_eq!(loaded.released_particles_per_cell, Some(3));
        assert_eq!(loaded.density, Some(4.0));
        assert_eq!(loaded.slab_thickness_factor, Some(5.0));
        assert_eq!(loaded.friction_coefficient, Some(6.0));
        assert_eq!(loaded.drag_coefficient, Some(7.0));
        assert_eq!(loaded.cfl, Some(8.0));
        assert_eq!(loaded.min_slope_angle, Some(9.0));
        assert_eq!(loaded.max_slope_angle, Some(10.0));
        assert_eq!(loaded.release_min_elevation, Some(11.0));
        assert_eq!(loaded.release_max_elevation, Some(13.0));
        assert_eq!(loaded.velocity_threshold, Some(12.0));
        assert_eq!(loaded.roughness_threshold, Some(13.0));
        assert_eq!(loaded.peak_flow_thickness_threshold, Some(14.0));
        assert_eq!(loaded.enable_curvature, Some(true));
        assert_eq!(loaded.enable_entrainment, Some(true));
        assert_eq!(loaded.enable_particle_interaction, Some(true));
        assert_eq!(loaded.enable_earth_pressure_coefficient, Some(true));
        assert_eq!(loaded.n0, Some(1.0));
        assert_eq!(loaded.i0, Some(2.0));
        assert_eq!(loaded.mu0, Some(0.1));
        assert_eq!(loaded.mu2, Some(0.3));
        assert_eq!(loaded.grain_diameter, Some(0.5));
        assert_eq!(loaded.internal_friction_angle, Some(30.0));
        assert_eq!(loaded.basal_friction_angle, Some(45.0));
    }

    // Helper to create a valid minimal PNG for testing
    fn create_test_png() -> Vec<u8> {
        let mut img = image::RgbaImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255])); // Red pixel

        let mut bytes = std::io::Cursor::new(Vec::new());
        img.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    #[tokio::test]
    async fn test_read_png_from_file() {
        // Arrange: Create a temporary .png file
        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("test_image.png");
        let png_data = create_test_png();

        let mut file = File::create(&file_path).unwrap();
        file.write_all(&png_data).unwrap();

        // Act: Call read_png (passing path without extension as per your logic)
        let path_str = file_path.with_extension("").to_str().unwrap().to_string();
        let result = block_on(read_png(&path_str));

        // Assert
        assert!(result.is_ok());
        let (data, width, height) = result.unwrap();
        assert_eq!(width, 2);
        assert_eq!(height, 2);
        assert_eq!(data.len(), 2 * 2 * 4); // 4 bytes per pixel (RGBA)
    }

    #[tokio::test]
    async fn test_read_png_from_http() {
        // Arrange: Start a mock server
        let mock_server = MockServer::start().await;
        let png_data = create_test_png();

        Mock::given(method("GET"))
            .and(path("/assets/map.png"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(png_data, "image/png"))
            .mount(&mock_server)
            .await;

        // Act
        let url = format!("{}/assets/map.png", &mock_server.uri());
        let result = read_png(&url).await;

        // Assert
        assert!(result.is_ok());
        let (_, width, height) = result.unwrap();
        assert_eq!(width, 2);
        assert_eq!(height, 2);
    }

    #[test_log::test]
    fn data_type_ranges() {
        println!("Data type ranges:");
        println!("Signed integer:");
        println!("i8 min: {}, max: {}", i8::MIN, i8::MAX);
        println!("i16 min: {}, max: {}", i16::MIN, i16::MAX);
        println!("i32 min: {}, max: {}", i32::MIN, i32::MAX);
        println!("i64 min: {}, max: {}", i64::MIN, i64::MAX);
        println!("i128 min: {}, max: {}", i128::MIN, i128::MAX);
        println!("isize min: {}, max: {}", isize::MIN, isize::MAX);
        println!("Unsigned integer:");
        println!("u8 min: {}, max: {}", u8::MIN, u8::MAX);
        println!("u16 min: {}, max: {}", u16::MIN, u16::MAX);
        println!("u32 min: {}, max: {}", u32::MIN, u32::MAX);
        println!("u64 min: {}, max: {}", u64::MIN, u64::MAX);
        println!("u128 min: {}, max: {}", u128::MIN, u128::MAX);
        println!("usize min: {}, max: {}", usize::MIN, usize::MAX);
        println!("Float integer:");
        println!("f16 min: {:.4e}, max: {:.4e}", f16::MIN_POSITIVE, f16::MAX);
        println!("f32 min: {:.4e}, max: {:.4e}", f32::MIN_POSITIVE, f32::MAX);
        println!("f64 min: {:.4e}, max: {:.4e}", f64::MIN_POSITIVE, f64::MAX);
    }

    #[test_log::test]
    fn test_rgba_bytes_to_f32() {
        let floats = [1.0f32, 2.0, 3.0];
        let mut bytes = Vec::new();
        for f in floats.iter() {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let result = rgba_bytes_to_f32(&bytes);
        assert_eq!(result, floats);
        let f32_bytes = f32_to_rgba_bytes(&result);
        assert_eq!(f32_bytes, bytes)
    }

    #[tokio::test]
    async fn test_read_file_helpers_support_local_and_http_sources() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let local_file_path = tmp_dir.path().join("settings.json");
        fs::write(&local_file_path, "local settings").unwrap();

        let local_bytes = read_file(local_file_path.to_str().unwrap()).await.unwrap();
        assert_eq!(local_bytes, b"local settings");

        let local_string = read_file_to_string(local_file_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(local_string, "local settings");

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/settings.txt"))
            .respond_with(ResponseTemplate::new(200).set_body_string("remote settings"))
            .mount(&mock_server)
            .await;

        let remote_string = read_file_to_string(&format!("{}/settings.txt", mock_server.uri()))
            .await
            .unwrap();
        assert_eq!(remote_string, "remote settings");
    }

    #[tokio::test]
    async fn test_write_png_and_load_release_areas_extract_alpha_channel() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let base_path = tmp_dir.path().join("release_map");
        let rgba = vec![
            0, 0, 0, 0, //
            0, 0, 0, 100, //
            0, 0, 0, 200, //
            0, 0, 0, 255,
        ];

        write_png(&base_path, &rgba, 2, 2).unwrap();

        let release_areas = load_release_areas(base_path.with_extension("png").to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(release_areas, vec![0.0, 1.0, 2.0, 2.55]);
    }

    #[tokio::test]
    async fn test_create_sim_settings_helpers_load_dem_and_apply_overrides() {
        let (sim_settings, dem, outline) = create_sim_settings_and_dem_from_path(PARABOLA_PATH)
            .await
            .expect("Failed to create sim settings and dem");
        assert_eq!(dem.width, 1001);
        assert_eq!(dem.height, 401);
        assert_eq!(sim_settings.grid_shape_x, dem.width as u32);
        assert_eq!(sim_settings.grid_shape_y, dem.height as u32);
        assert_eq!(sim_settings.cell_size, dem.cell_size);
        assert_eq!(outline.len(), dem.data1d.len());

        let settings = Settings {
            dem_path: Some(PARABOLA_PATH.to_string()),
            max_steps: Some(42),
            density: Some(321.0),
            ..Settings::default()
        };
        let settings_file = NamedTempFile::new().unwrap();
        fs::write(settings_file.path(), settings.dumps().unwrap()).unwrap();

        let (json_sim_settings, json_dem, json_outline) =
            sim_settings_and_dem_from_json_file(settings_file.path().to_str().unwrap()).await;

        assert_eq!(json_dem.width, dem.width);
        assert_eq!(json_dem.height, dem.height);
        assert_eq!(json_sim_settings.max_steps, 42);
        assert_eq!(json_sim_settings.density, 321.0);
        assert_eq!(json_sim_settings.grid_shape_x, dem.width as u32);
        assert_eq!(json_sim_settings.grid_shape_y, dem.height as u32);
        assert_eq!(json_outline.len(), dem.data1d.len());
    }

    #[test]
    fn test_valid_esri_grid_parsing() {
        let valid_data = "\
ncols         3
nrows         2
xllcorner     100.0
yllcorner     200.0
cellsize      5.0
NODATA_value  -9999.0
1.0 2.0 3.0
4.0 -9999.0 6.0
";

        let cursor = Cursor::new(valid_data);
        let result = EsriGrid::from_reader(cursor);

        // Verify parsing succeeded
        assert!(result.is_ok(), "Expected valid grid to parse successfully");
        let grid = result.unwrap();

        // Verify header values
        assert_eq!(grid.header.ncols, 3);
        assert_eq!(grid.header.nrows, 2);
        assert_eq!(grid.header.xllcorner, Some(100.0));
        assert_eq!(grid.header.yllcorner, Some(200.0));
        assert_eq!(grid.header.cellsize, 5.0);
        assert_eq!(grid.header.nodata_value, Some(-9999.0));

        // Verify 2D data index extraction
        assert_eq!(grid.get(0, 0), Some(1.0));
        assert_eq!(grid.get(0, 2), Some(3.0));
        assert_eq!(grid.get(1, 1), Some(-9999.0));
        assert_eq!(grid.get(1, 2), Some(6.0));

        // Verify bounds checking
        assert_eq!(grid.get(2, 0), None); // Out of bounds row
        assert_eq!(grid.get(0, 3), None); // Out of bounds col
    }

    #[test]
    fn test_variant_header_center() {
        // Some ESRI grids use xllcenter/yllcenter instead of corner
        let center_data = "\
ncols         2
nrows         2
xllcenter     50.0
yllcenter     50.0
cellsize      2.5
NODATA_value  -1
0.0 1.0
2.0 3.0";

        let cursor = Cursor::new(center_data);
        let grid = EsriGrid::from_reader(cursor).unwrap();

        assert_eq!(grid.header.get_xllcorner(), 48.75);
        assert_eq!(grid.header.get_yllcorner(), 48.75);
    }

    const ASC_SAMPLE: &str = "\
ncols         3
nrows         2
xllcorner     100.0
yllcorner     200.0
cellsize      10.0
NODATA_value  -9999
1.0 2.0 3.0
4.0 5.0 6.0";

    #[test]
    fn test_load_dem_from_bytes_asc_georeference_and_row_order() {
        let dem = load_dem_from_bytes(ASC_SAMPLE.as_bytes(), "asc", "upload.asc").unwrap();

        assert_eq!(dem.width, 3);
        assert_eq!(dem.height, 2);
        assert_eq!(dem.cell_size, 10.0);
        assert_eq!(dem.bounds.xmin, 100.0);
        assert_eq!(dem.bounds.xmax, 130.0);
        assert_eq!(dem.bounds.ymin, 200.0);
        assert_eq!(dem.bounds.ymax, 220.0);
        assert_eq!(dem.source, "upload.asc");

        // ESRI rows run north to south, so loading flips them to south-up.
        assert_eq!(dem.data1d, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_load_dem_preserves_low_elevations_and_masks_explicit_nodata() {
        let source = "\
ncols 2
nrows 2
xllcorner 0
yllcorner 0
cellsize 1
NODATA_value -9999
-3.0 0.0
-9999 2.0";
        let dem = load_dem_from_bytes(source.as_bytes(), "asc", "low.asc").unwrap();

        assert_eq!(dem.minimum_elevation, -3.0);
        assert!(dem.data1d[0].is_nan());
        assert_eq!(&dem.data1d[1..], &[2.0, -3.0, 0.0]);
        assert!(dem.data[0][0].is_nan());
        assert_eq!(dem.data[1], vec![-3.0, 0.0]);
    }

    #[test]
    fn test_load_dem_from_bytes_matches_path_loader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.asc");
        std::fs::write(&path, ASC_SAMPLE).unwrap();

        let from_path = pollster::block_on(load_dem(path.to_str().unwrap())).unwrap();
        let from_bytes = load_dem_from_bytes(ASC_SAMPLE.as_bytes(), "asc", "sample.asc").unwrap();

        assert_eq!(from_path.width, from_bytes.width);
        assert_eq!(from_path.height, from_bytes.height);
        assert_eq!(from_path.cell_size, from_bytes.cell_size);
        assert_eq!(from_path.bounds.xmin, from_bytes.bounds.xmin);
        assert_eq!(from_path.bounds.ymax, from_bytes.bounds.ymax);
        assert_eq!(from_path.data1d, from_bytes.data1d);
    }

    #[test]
    fn test_load_dem_from_bytes_rejects_unsupported_extension() {
        let result = load_dem_from_bytes(b"irrelevant", "zarr", "store.zarr");
        assert!(matches!(
            result,
            Err(DataProcessorError::UnsupportedDemFormat(ref ext)) if ext == "zarr"
        ));
    }

    #[test]
    fn test_load_release_areas_from_bytes_asc_is_flipped() {
        let data = load_release_areas_from_bytes(ASC_SAMPLE.as_bytes(), "asc").unwrap();
        assert_eq!(data, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_load_release_areas_from_bytes_rejects_unsupported_extension() {
        let result = load_release_areas_from_bytes(b"irrelevant", "png");
        assert!(matches!(
            result,
            Err(DataProcessorError::UnsupportedDemFormat(ref ext)) if ext == "png"
        ));
    }

    #[test]
    fn test_data_mismatch_fewer_elements() {
        // Header claims 3x2 (6 elements), but only 5 are provided
        let broken_data = "\
ncols         3
nrows         2
xllcorner     0.0
yllcorner     0.0
cellsize      1.0
NODATA_value  -999
1.0 2.0 3.0
4.0 5.0";

        let cursor = Cursor::new(broken_data);
        let result = EsriGrid::from_reader(cursor);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EsriGridError::DataSizeMismatch {
                expected: 6,
                ncols: 3,
                nrows: 2,
                found: 5
            }
        ));
    }

    #[test]
    fn test_invalid_header_key() {
        let bad_header = "\
ncols         3
invalid_key   2
xllcorner     0.0
yllcorner     0.0
cellsize      1.0
NODATA_value  -999
1.0 2.0 3.0";

        let cursor = Cursor::new(bad_header);
        let result = EsriGrid::from_reader(cursor);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EsriGridError::UnknownHeaderKey(_)
        ));
    }

    #[test]
    fn test_unparsed_token() {
        let bad_token = "\
ncols         2
nrows         1
xllcorner     0.0
yllcorner     0.0
cellsize      1.0
NODATA_value  -999
1.0 corrupt_data";

        let cursor = Cursor::new(bad_token);
        let result = EsriGrid::from_reader(cursor);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EsriGridError::InvalidDataToken(e) if e.contains("corrupt_data")
        ));
    }
    #[test_log::test]
    fn test_dem_png_asc_equality() {
        let png_path = "../../data/avaframe/avaGar.png";
        let asc_path = "../../data/avaframe/avaGar_remeshedDEM5.00.asc";

        let png_dem = block_on(load_dem(png_path)).expect("Failed to load PNG DEM");
        let asc_dem = block_on(load_dem(asc_path)).expect("Failed to load ASC DEM");

        assert_eq!(png_dem.width, asc_dem.width);
        assert_eq!(png_dem.height, asc_dem.height);
        assert_eq!(png_dem.cell_size, asc_dem.cell_size);
        assert_eq!(png_dem.bounds, asc_dem.bounds);
        let mismatch = png_dem
            .data1d
            .iter()
            .zip(&asc_dem.data1d)
            .enumerate()
            .find(|(_, (png, asc))| png != asc && !(png.is_nan() && asc.is_nan()));
        assert!(mismatch.is_none(), "first DEM mismatch: {mismatch:?}");
    }

    #[test]
    fn test_variant_header_center_preserved() {
        let center_data = "\
ncols         2
nrows         2
xllcenter     50.0
yllcenter     50.0
cellsize      10.0
NODATA_value  -1
0.0 1.0
2.0 3.0";

        let cursor = Cursor::new(center_data);
        let grid = EsriGrid::from_reader(cursor).unwrap();

        // Assert the raw values are captured exactly as written
        assert_eq!(grid.header.xllcenter, Some(50.0));
        assert_eq!(grid.header.yllcenter, Some(50.0));
        assert_eq!(grid.header.xllcorner, None);
        assert_eq!(grid.header.yllcorner, None);
        assert_eq!(grid.header.get_xllcenter(), 50.0);
        assert_eq!(grid.header.get_yllcenter(), 50.0);

        // Assert the computed helpers handle the half-cell offset calculation dynamically
        assert_eq!(grid.header.get_xllcorner(), 45.0);
        assert_eq!(grid.header.get_yllcorner(), 45.0);
    }
    #[test]
    fn test_variant_header_corner_preserved() {
        let center_data = "\
ncols         2
nrows         2
xllcorner     50.0
yllcorner     50.0
cellsize      10.0
NODATA_value  -1
0.0 1.0
2.0 3.0";

        let cursor = Cursor::new(center_data);
        let grid = EsriGrid::from_reader(cursor).unwrap();

        // Assert the raw values are captured exactly as written
        assert_eq!(grid.header.xllcorner, Some(50.0));
        assert_eq!(grid.header.yllcorner, Some(50.0));
        assert_eq!(grid.header.xllcenter, None);
        assert_eq!(grid.header.yllcenter, None);
        assert_eq!(grid.header.get_xllcorner(), 50.0);
        assert_eq!(grid.header.get_yllcorner(), 50.0);

        // Assert the computed helpers handle the half-cell offset calculation dynamically
        assert_eq!(grid.header.get_xllcenter(), 55.0);
        assert_eq!(grid.header.get_yllcenter(), 55.0);
    }
    #[test]
    fn test_round_trip_export() {
        use std::io::Cursor;

        // Create a model layout containing standard values, a raw center coordinate, and a NaN element
        let original_header = EsriGridHeader {
            ncols: 3,
            nrows: 2,
            cellsize: 10.0,
            nodata_value: Some(-999.0),
            xllcorner: None,
            yllcorner: None,
            xllcenter: Some(100.0),
            yllcenter: Some(200.0),
        };

        let original_grid = EsriGrid {
            header: original_header,
            data: vec![1.5, f32::NAN as f32, 3.0, 4.25, 5.0, 6.1],
        };

        // 1. Export the struct to a memory vector (Simulating writing a file)
        let mut buffer = Vec::new();
        assert!(original_grid.to_writer(&mut buffer).is_ok());

        // Convert output buffer to a string so you can inspect the format if needed
        let output_string = String::from_utf8(buffer.clone()).unwrap();
        println!("Generated File Output:\n{}", output_string);

        // 2. Parse that generated string right back into a new EsriGrid struct
        let cursor = Cursor::new(buffer);
        let parsed_grid = EsriGrid::from_reader(cursor).unwrap();

        // 3. Verify parity transformations across bounds and NaN replacement rules
        assert_eq!(parsed_grid.header.ncols, original_grid.header.ncols);
        assert_eq!(parsed_grid.header.xllcenter, Some(100.0));

        // Check cell indexing structure
        assert_eq!(parsed_grid.get(0, 0), Some(1.5));
        assert_eq!(parsed_grid.get(0, 1), Some(-999.0)); // The NaN became the specified nodata value
        assert_eq!(parsed_grid.get(0, 2), Some(3.0));
        assert_eq!(parsed_grid.get(1, 1), Some(5.0));
    }

    #[test_log::test]
    fn test_esri_header_missing_origin_errors() {
        // Header missing both xllcorner and xllcenter should error during parsing
        let bad_header = "\
ncols         2
nrows         2
cellsize      1.0
NODATA_value  -999
1.0 2.0
3.0 4.0
";

        let cursor = Cursor::new(bad_header);
        let res = EsriGrid::from_reader(cursor);
        assert!(res.is_err());
    }

    #[test]
    fn test_esri_header_helpers_expect_panic_when_no_coords() {
        // Construct header without corner or center and expect helper methods to panic
        let header = EsriGridHeader {
            ncols: 1,
            nrows: 1,
            cellsize: 1.0,
            nodata_value: None,
            xllcorner: None,
            yllcorner: None,
            xllcenter: None,
            yllcenter: None,
        };

        let res = std::panic::catch_unwind(|| header.get_xllcorner());
        assert!(res.is_err());

        let res = std::panic::catch_unwind(|| header.get_xllcenter());
        assert!(res.is_err());
    }

    #[test_log::test]
    fn test_load_tiff_as_dem() {
        let tiff_path = "../../data/vals/PAR6_Vals_Gries_dtm_10_utm32n_bil_.tif";
        let dem = load_tiff_as_dem(tiff_path).expect("Failed to load TIFF as DEM");

        // Verify DEM structure
        assert_eq!(dem.width, 1701, "Width should be 1701");
        assert_eq!(dem.height, 1145, "Height should be 1145");
        assert_eq!(
            dem.data1d.len(),
            dem.width * dem.height,
            "Data1D length should equal width * height"
        );
        assert_eq!(
            dem.data.len(),
            dem.height,
            "2D data should have height rows"
        );
        assert_eq!(
            dem.data[0].len(),
            dem.width,
            "Each row should have width columns"
        );

        // Verify coordinates
        assert_eq!(
            dem.x.len(),
            dem.width,
            "X coordinates should have width elements"
        );
        assert_eq!(
            dem.y.len(),
            dem.height,
            "Y coordinates should have height elements"
        );
        assert!(
            dem.x[0] < dem.x[dem.x.len() - 1],
            "X coordinates should be increasing"
        );
        assert!(
            dem.y[0] < dem.y[dem.y.len() - 1],
            "Y coordinates should be increasing"
        );

        // Verify bounds
        assert_eq!(dem.bounds.xmin, dem.x[0]);
        assert_eq!(dem.bounds.xmax, dem.x[dem.x.len() - 1]);
        assert_eq!(dem.bounds.ymin, dem.y[0]);
        assert_eq!(dem.bounds.ymax, dem.y[dem.y.len() - 1]);

        // Verify cell size and map factor
        assert_eq!(dem.cell_size, 10.0, "Cell size should be 10.0");
        assert_eq!(dem.map_factor, 1.0, "Map factor should be 1.0");

        // Verify minimum elevation is initialized
        assert_eq!(
            dem.minimum_elevation,
            f32::INFINITY,
            "Minimum elevation should be INFINITY initially"
        );

        // Verify that the 1D and 2D data are consistent
        for (row_idx, row) in dem.data.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                let idx_1d = row_idx * dem.width + col_idx;
                assert_eq!(
                    val, dem.data1d[idx_1d],
                    "2D data at ({}, {}) should match 1D data at index {}",
                    row_idx, col_idx, idx_1d
                );
            }
        }
        assert_eq!(
            dem.data1d[0], -9999.0,
            "First elevation value should be -9999.0"
        );
        assert_eq!(
            dem.data1d[1701 * 500 + 500],
            1686.6079,
            "Elevation value in the middle of the texture should be 1686.6m"
        );
    }

    #[test_log::test]
    fn test_load_tiff_as_dem_invalid_path() {
        let invalid_path = "nonexistent/path/to/file.tif";
        let result = load_tiff_as_dem(invalid_path);

        assert!(
            result.is_err(),
            "Loading DEM from non-existent path should return error"
        );
    }
}
