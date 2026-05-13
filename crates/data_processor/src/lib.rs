use std::fs::File;
use std::io::Cursor;
use std::io::{self, BufWriter, Write};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::vec::Vec;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use bincode::{Decode, Encode, config};
use std::io::Read;

use compute_core::dem::{Bounds, Dem, GeoMetadata, GeoTiff, TiffData};
use compute_core::settings::{Settings, SimSettings};
use compute_core::utils::*;

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
#[cfg(not(target_arch = "wasm32"))]
use xz2::{read::XzDecoder, write::XzEncoder};

#[cfg(not(target_arch = "wasm32"))]
use zstd::stream::{decode_all, encode_all};

use image::{GenericImageView, ImageReader};

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum DataType {
    F16,
    F32,
    F64,
}

impl DataType {
    pub fn from_int(value: u8) -> Option<Self> {
        match value {
            16 => Some(DataType::F16),
            32 => Some(DataType::F32),
            64 => Some(DataType::F64),
            _ => None,
        }
    }

    pub fn as_int(&self) -> u8 {
        match self {
            DataType::F16 => 16,
            DataType::F32 => 32,
            DataType::F64 => 64,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::F16 => "f16",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum Unit {
    MetersPerSecond,
    Degree,
    Kilogram,
    Dimensionless,
}

impl Unit {
    pub fn from_int(value: u8) -> Option<Self> {
        match value {
            0 => Some(Unit::MetersPerSecond),
            1 => Some(Unit::Degree),
            2 => Some(Unit::Kilogram),
            _ => Some(Unit::Dimensionless),
        }
    }
    pub fn as_int(&self) -> u8 {
        match self {
            Unit::MetersPerSecond => 0,
            Unit::Degree => 1,
            Unit::Kilogram => 2,
            Unit::Dimensionless => 255,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Unit::MetersPerSecond => "m/s",
            Unit::Degree => "°",
            Unit::Kilogram => "kg",
            Unit::Dimensionless => "-",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum Variable {
    Velocity,
    SlopeAngle,
    Curvature,
    SlopeAspect,
    NormalX,
    NormalY,
    NormalZ,
    Mass,
    Undefined,
}
impl Variable {
    pub fn from_int(value: u8) -> Option<Self> {
        match value {
            0 => Some(Variable::Velocity),
            1 => Some(Variable::SlopeAngle),
            2 => Some(Variable::Curvature),
            3 => Some(Variable::SlopeAspect),
            4 => Some(Variable::NormalX),
            5 => Some(Variable::NormalY),
            6 => Some(Variable::NormalZ),
            7 => Some(Variable::Mass),
            _ => Some(Variable::Undefined),
        }
    }
    pub fn as_int(&self) -> u8 {
        match self {
            Variable::Velocity => 0,
            Variable::SlopeAngle => 1,
            Variable::Curvature => 2,
            Variable::SlopeAspect => 3,
            Variable::NormalX => 4,
            Variable::NormalY => 5,
            Variable::NormalZ => 6,
            Variable::Mass => 7,
            Variable::Undefined => 255,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Variable::Velocity => "velocity",
            Variable::SlopeAngle => "slope_angle",
            Variable::Curvature => "curvature",
            Variable::SlopeAspect => "slope_aspect",
            Variable::NormalX => "normal_x",
            Variable::NormalY => "normal_y",
            Variable::NormalZ => "normal_z",
            Variable::Mass => "mass",
            Variable::Undefined => "undefined",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Binary,
    Lz4,
    Png,
}
impl FileFormat {
    pub fn from_fileformat_str(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "binary" => Some(FileFormat::Binary),
            "compressedbinary" => Some(FileFormat::Lz4),
            "png" => Some(FileFormat::Png),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FileFormat::Binary => "binary",
            FileFormat::Lz4 => "compressedbinary",
            FileFormat::Png => "png",
        }
    }
    pub fn as_extension(&self) -> &'static str {
        match self {
            FileFormat::Binary => "bin",
            FileFormat::Lz4 => "lz4",
            FileFormat::Png => "png",
        }
    }
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "bin" => Some(FileFormat::Binary),
            "lz4" => Some(FileFormat::Lz4),
            "png" => Some(FileFormat::Png),
            _ => None,
        }
    }
}

pub struct MetaGridParams {
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub map_factor: f32,
    pub epsg_code: u16,
    pub top: f32,
    pub left: f32,
    pub data_type: DataType,
    pub variable: Variable,
    pub unit: Unit,
}

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct MetaGrid {
    magic_bytes: u32,
    pub version: u8,
    pub width: u32,
    pub height: u32,
    pub cell_size: f32,
    pub map_factor: f32,
    pub epsg_code: u16,
    pub top: f32,
    pub left: f32,
    pub data_type: DataType,
    pub variable: Variable,
    pub unit: Unit,
}

impl MetaGrid {
    /// Creates a new MetaGrid with the given parameters.
    pub fn new(params: &MetaGridParams) -> Self {
        MetaGrid {
            magic_bytes: u32::from_le_bytes(*b"AVAG"),
            version: 1,
            width: params.width,
            height: params.height,
            cell_size: params.cell_size,
            map_factor: params.map_factor,
            epsg_code: params.epsg_code,
            top: params.top,
            left: params.left,
            data_type: params.data_type,
            variable: params.variable,
            unit: params.unit,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(magic_bytes='{}', version={}, width={}, height={}, data_type={})",
            std::str::from_utf8(&self.magic_bytes.to_le_bytes()).unwrap_or("????"),
            self.version,
            self.width,
            self.height,
            self.data_type.as_str()
        )
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
    fn __eq__(&self, other: &MetaGrid) -> bool {
        self.magic_bytes == other.magic_bytes
            && self.version == other.version
            && self.width == other.width
            && self.height == other.height
            && self.data_type == other.data_type
    }
}

#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct MetaParticle {
    magic_bytes: u32,
    pub version: u8,
    pub length: u32,
    pub data_type: u8,
}

impl MetaParticle {
    /// Creates a new MetaParticle with the given length and data type.
    pub fn new(length: u32, data_type: DataType) -> Self {
        MetaParticle {
            magic_bytes: u32::from_le_bytes(*b"AVAP"),
            version: 1,
            length,
            data_type: data_type.as_int(),
        }
    }

    // A simple representation for Python's `repr()`
    fn __repr__(&self) -> String {
        format!(
            "MetadataParticle(magic_bytes='{}', version={}, length={}, data_type={})",
            std::str::from_utf8(&self.magic_bytes.to_le_bytes()).unwrap_or("????"),
            self.version,
            self.length,
            DataType::from_int(self.data_type).unwrap().as_str()
        )
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
    fn __eq__(&self, other: &MetaParticle) -> bool {
        self.magic_bytes == other.magic_bytes
            && self.version == other.version
            && self.length == other.length
            && self.data_type == other.data_type
    }
}

#[derive(Encode, Decode, PartialEq, Debug)]
pub struct F32Data {
    pub metadata: MetaGrid,
    pub data: Vec<f32>,
}

impl F32Data {
    /// Creates a new F32Data instance with the given metadata and data.
    pub fn new(metadata: &MetaGrid, data: Vec<f32>) -> Self {
        assert_eq!(
            metadata.width * metadata.height,
            data.len() as u32,
            "Data width does not match metadata dimensions"
        );
        F32Data {
            metadata: metadata.clone(),
            data,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "F32Data(metadata={}, data_len={})",
            self.metadata.__repr__(),
            self.data.len()
        )
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new(path);
        let encoded_bytes = bincode::encode_to_vec(self, config::standard())
            .map_err(|e| format!("Bincode serialization failed: {}", e))?;
        write_bin(path, &encoded_bytes);
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer = read_bin(&PathBuf::from(path))?;
        let (data, _): (F32Data, _) = bincode::decode_from_slice(&buffer, config::standard())
            .map_err(|e| format!("Bincode deserialization failed: {}", e))?;
        assert_eq!(
            data.metadata.magic_bytes,
            u32::from_le_bytes(*b"AVAG"),
            "Invalid magic bytes"
        );
        Ok(data)
    }
}
pub fn read_bin(path: &PathBuf) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

pub fn write_bin(path: &Path, buffer: &[u8]) {
    let file = File::create(path.with_extension("bin")).expect("Failed to create file");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16 MB buffer
    writer.write_all(buffer).expect("Failed to write data");
}

pub fn write_lz4_bin(path: &Path, buffer: &[u8]) {
    let file = File::create(path.with_extension("lz4")).expect("Failed to create file");
    let compressed_data = compress_prepend_size(buffer);
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16 MB buffer
    writer
        .write_all(&compressed_data)
        .expect("Failed to write data");
}

pub fn read_lz4(path: &Path) -> io::Result<Vec<u8>> {
    read_bin(&path.with_extension("lz4")).and_then(|buffer| {
        decompress_size_prepended(&buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    })
}

#[allow(unused_variables)]
pub fn write_zstd(path: &Path, buffer: &Vec<u8>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let file = File::create(path.with_extension("zst")).expect("Failed to create file");
        let compressed_data = encode_all(Cursor::new(buffer), 22).expect("Failed to compress data");
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16 MB buffer
        writer
            .write_all(&compressed_data)
            .expect("Failed to write data");
    }

    #[cfg(target_arch = "wasm32")]
    {
        panic!("Zstd compression is not supported on this platform");
    }
}

#[allow(unused_variables)]
pub fn read_zstd_bin(path: &Path) -> io::Result<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        read_bin(&path.with_extension("zst"))
            .and_then(|buffer| decode_all(Cursor::new(&buffer[..])))
    }

    #[cfg(target_arch = "wasm32")]
    {
        panic!("Zstd compression is not supported on this platform");
    }
}

#[allow(unused_variables)]
pub fn write_xz(path: &Path, buffer: &Vec<u8>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut encoder = XzEncoder::new(
            BufWriter::new(File::create(path.with_extension("xz")).expect("Failed to create file")),
            6,
        ); // level 0-9

        std::io::copy(&mut BufReader::new(Cursor::new(buffer)), &mut encoder)
            .expect("Failed to write data");
        encoder.finish().expect("Failed to finish encoding");
    }
    #[cfg(target_arch = "wasm32")]
    {
        panic!("XZ compression is not supported on this platform");
    }
}

#[allow(unused_variables)]
pub fn read_xz(path: &Path) -> io::Result<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let compressed = File::open(path.with_extension("xz")).expect("Failed to open file");
        let mut decoder = XzDecoder::new(BufReader::new(compressed));
        let mut decompressed = Vec::new();

        std::io::copy(&mut decoder, &mut decompressed)?;
        Ok(decompressed)
    }
    #[cfg(target_arch = "wasm32")]
    {
        panic!("XZ compression is not supported on this platform");
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
    data.chunks_exact(4)
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

pub fn read_geo_tiff(path: &str) -> Result<GeoTiff, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?;

    let (width, height) = decoder.dimensions()?;

    // --- Metadata Extraction ---
    // Tag 33550: ModelPixelScaleTag
    let pixel_scales = decoder
        .get_tag_f64_vec(Tag::Unknown(33550))
        .unwrap_or_default();

    if pixel_scales.len() >= 2 {
        // Ensure the grid is uniform for your simulation
        assert_eq!(
            pixel_scales[0], pixel_scales[1],
            "Non-uniform grid detected: X and Y scales must match for square cells."
        );
    } else {
        return Err("Missing pixel scale metadata (Tag 33550)".into());
    }

    // Tag 33922: ModelTiepointTag
    let tie_points = decoder.get_tag_f64_vec(Tag::Unknown(33922))?;
    let nodata = decoder
        .get_tag_f64_vec(Tag::Unknown(42113)) // Try to get the tag
        .ok() // If it fails (missing or wrong type), return None
        .and_then(|v| v.first().copied());

    let metadata = if tie_points.len() >= 6 {
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
        return Err("TIFF missing GeoTIFF tie point metadata (Tag 33922).".into());
    };

    // --- Image Data Extraction ---
    // Note: To keep GeoTiff simple, we often normalize data to f32 for simulations
    let data = match decoder.read_image()? {
        DecodingResult::U8(data) => TiffData::U8(data),
        DecodingResult::U16(data) => TiffData::U16(data),
        DecodingResult::F32(data) => TiffData::F32(data),
        _ => return Err("Unsupported TIFF data type".into()),
    };
    info!(
        "Loaded GeoTIFF: {}x{} cells at {}m resolution",
        width, height, metadata.pixel_scale[0]
    );

    Ok(GeoTiff { metadata, data })
}

async fn load_png_as_float32(path: &str) -> Result<Dem, Box<dyn std::error::Error>> {
    let (rgba, width, height) = read_png(path).await.expect("Failed to load PNG");
    let bounds: Bounds = load_bounds(path).await.expect("Failed to load bounds");
    debug!("Loaded PNG {}: {} x {}", path, width, height);
    let mut dem = Dem {
        width,
        height,
        data1d: rgba_bytes_to_f32(&rgba),
        data: Vec::new(),
        x: linspace(bounds.xmin, bounds.xmax, width),
        y: linspace(bounds.ymin, bounds.ymax, height),
        cell_size: (bounds.xmax - bounds.xmin) / (width - 1) as f32,
        bounds,
        map_factor: 1.0,
        minimum_elevation: f32::INFINITY,
    };
    dem.data = to_2d(&dem.data1d, width, height);
    Ok(dem)
}

fn load_tiff_as_dem(path: &str) -> Result<Dem, Box<dyn std::error::Error>> {
    let mut tiff: GeoTiff = read_geo_tiff(path)?;
    tiff.flip_y();
    let mut dem = Dem {
        width: tiff.metadata.width as usize,
        height: tiff.metadata.height as usize,
        data1d: tiff.data.as_f32(),
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
    };
    dem.data = to_2d(&dem.data1d, dem.width, dem.height);
    Ok(dem)
}

pub fn save_grid(dem: &Dem, path: &str, data: Vec<f32>) -> Result<(), std::io::Error> {
    let params = MetaGridParams {
        width: dem.width as u32,
        height: dem.height as u32,
        cell_size: dem.cell_size,
        map_factor: dem.map_factor,
        epsg_code: 0,
        top: 0.0,
        left: 0.0,
        data_type: DataType::F32,
        variable: Variable::Undefined,
        unit: Unit::Dimensionless,
    };
    F32Data::new(&MetaGrid::new(&params), data)
        .save(path.as_ref())
        .unwrap_or_else(|_| panic!("Failed to save grid {}", path));
    Ok(())
}

pub async fn load_release_areas(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let data: Vec<f32> = match ext.to_lowercase().as_str() {
        "asc" => return Err("ASC format not supported yet".into()),
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

pub async fn load_dem(path: &str) -> Result<Dem, Box<dyn std::error::Error>> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let mut dem: Dem = match ext.to_lowercase().as_str() {
        "asc" => return Err("ASC format not supported yet".into()),
        "png" => load_png_as_float32(path).await?,
        "tif" | "tiff" => load_tiff_as_dem(path)?,
        _ => return Err(format!("Unsupported DEM format: {}", ext).into()),
    };

    dem.minimum_elevation = Dem::calculate_minimum_elevation(&dem.data1d);

    dem.data1d = dem
        .data1d
        .into_iter()
        .map(|v| {
            if v >= dem.minimum_elevation {
                v
            } else {
                f32::NAN
            }
        })
        .collect();
    assert!(
        dem.bounds.xmin < dem.bounds.xmax,
        "xmin ({}) must be less than or equal to xmax ({})",
        dem.bounds.xmin,
        dem.bounds.xmax
    );
    assert!(
        dem.bounds.ymin < dem.bounds.ymax,
        "ymin ({}) must be less than or equal to ymax ({})",
        dem.bounds.ymin,
        dem.bounds.ymax
    );

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

pub async fn create_sim_settings_and_dem_from_path(file_path: &str) -> (SimSettings, Dem) {
    let settings = Settings {
        dem_path: Some(file_path.to_string()),
        ..Default::default()
    };
    create_sim_settings_and_dem(&settings).await
}
pub async fn create_sim_settings_and_dem(settings: &Settings) -> (SimSettings, Dem) {
    let dem: Dem = match &settings.dem_path {
        Some(path) => load_dem(path.as_ref())
            .await
            .expect("Failed to load DEM from path"),
        None => Dem::default(),
    };
    let sim_settings = SimSettings::from_settings(settings, &dem);
    (sim_settings, dem)
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

pub async fn sim_settings_and_dem_from_json_file(file_path: &str) -> (SimSettings, Dem) {
    let data = std::fs::read_to_string(file_path).expect("Failed to read json file");
    let settings = Settings::loads(&data).expect("Failed to load settings from JSON file");
    create_sim_settings_and_dem(&settings).await
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

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use pollster::block_on;
    use std::env;
    use std::f32::consts::PI;
    use std::fs;
    use std::fs::File;
    use std::io::Write;
    use std::time::Instant;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const PARABOLA_PATH: &str = "../../frontend/data/avaframe/avaParabola.png";

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
        let (dx, dy) = dem.get_index(&pt);
        assert_eq!(dx, 73.3139648);
        assert_eq!(dy, 146.627930);
    }

    #[test_log::test]
    fn test_load_png_as_float32() {
        let path = "../../frontend/data/avaframe/avaParabola.png";
        let dem: Dem = block_on(load_dem(path)).expect("Failed to load PNG as float32");
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
    #[test_log::test]
    fn test_load_bounds() {
        let path = "../../frontend/data/avaframe/avaInclinedPlane.png";
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
            release_min_elevation: Some(11.0),
            velocity_threshold: Some(12.0),
            roughness_threshold: Some(13.0),
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
        assert_eq!(loaded.release_min_elevation, Some(11.0));
        assert_eq!(loaded.velocity_threshold, Some(12.0));
        assert_eq!(loaded.roughness_threshold, Some(13.0));
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
    fn test_metadata_new_grid() {
        let params = MetaGridParams {
            width: 128,
            height: 256,
            cell_size: 5.0,
            map_factor: 1.0,
            epsg_code: 4326,
            top: 0.0,
            left: 0.0,
            data_type: DataType::F32,
            variable: Variable::Undefined,
            unit: Unit::Dimensionless,
        };
        let metadata = MetaGrid::new(&params);
        assert_eq!(metadata.width, params.width);
        assert_eq!(metadata.height, params.height);
        assert_eq!(metadata.data_type, DataType::F32);
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.magic_bytes, u32::from_le_bytes(*b"AVAG"));
    }

    #[test_log::test]
    fn test_data_type_from_and_as_int() {
        assert_eq!(DataType::from_int(16), Some(DataType::F16));
        assert_eq!(DataType::from_int(32), Some(DataType::F32));
        assert_eq!(DataType::from_int(64), Some(DataType::F64));
        assert_eq!(DataType::from_int(8), None);

        assert_eq!(DataType::F16.as_int(), 16);
        assert_eq!(DataType::F32.as_int(), 32);
        assert_eq!(DataType::F64.as_int(), 64);
    }

    #[test_log::test]
    fn test_data_type_as_str() {
        assert_eq!(DataType::F16.as_str(), "f16");
        assert_eq!(DataType::F32.as_str(), "f32");
        assert_eq!(DataType::F64.as_str(), "f64");
    }

    #[test_log::test]
    fn test_f32data_new_and_repr() {
        let params = MetaGridParams {
            width: 1,
            height: 3,
            cell_size: 5.0,
            map_factor: 1.0,
            epsg_code: 4326,
            top: 0.0,
            left: 0.0,
            data_type: DataType::F32,
            variable: Variable::Undefined,
            unit: Unit::Dimensionless,
        };
        let metadata = MetaGrid::new(&params);

        let data = vec![1.0f32, 2.0, 3.0];
        let f32data = F32Data::new(&metadata, data.clone());
        assert_eq!(f32data.metadata, metadata);
        assert_eq!(f32data.data, data);
        let repr = f32data.__repr__();
        assert!(repr.contains("F32Data(metadata="));
        assert!(repr.contains("data_len=3"));
    }

    #[test_log::test]
    fn test_metadata_repr() {
        let params = MetaGridParams {
            width: 5,
            height: 6,
            cell_size: 5.0,
            map_factor: 1.0,
            epsg_code: 4326,
            top: 0.0,
            left: 0.0,
            data_type: DataType::F32,
            variable: Variable::Undefined,
            unit: Unit::Dimensionless,
        };
        let metadata = MetaGrid::new(&params);
        let repr = metadata.__repr__();
        assert!(repr.contains("Metadata(magic_bytes='AVAG'"));
        assert!(repr.contains("version=1"));
        assert!(repr.contains("width=5"));
        assert!(repr.contains("height=6"));
        assert!(repr.contains("data_type=f32"));
    }

    #[test_log::test]
    fn test_write_and_read_bin() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_write_and_read_bin");
        let data = vec![1u8, 2, 3, 4, 5];
        write_bin(&file_path, &data);
        let read = read_bin(&file_path.with_extension("bin")).unwrap();
        assert_eq!(read, data);
        let _ = fs::remove_file(file_path.with_extension("bin"));
    }

    #[test_log::test]
    fn test_f32data_save_and_load() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_f32data_save_and_load.bin");

        let params = MetaGridParams {
            width: 2,
            height: 2,
            cell_size: 5.0,
            map_factor: 1.0,
            epsg_code: 4326,
            top: 0.0,
            left: 0.0,
            data_type: DataType::F32,
            variable: Variable::Undefined,
            unit: Unit::Dimensionless,
        };
        let metadata = MetaGrid::new(&params);
        let data = vec![0.1, 0.2, 0.3, 0.4];
        let f32data = F32Data::new(&metadata, data.clone());
        f32data.save(file_path.to_str().unwrap()).unwrap();

        let loaded = F32Data::load(file_path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.metadata, metadata);
        assert_eq!(loaded.data, data);

        let _ = fs::remove_file(file_path);
    }

    #[test_log::test]
    fn test_dimension_mismatch() {
        let params = MetaGridParams {
            width: 2,
            height: 2,
            cell_size: 5.0,
            map_factor: 1.0,
            epsg_code: 4326,
            top: 0.0,
            left: 0.0,
            data_type: DataType::F32,
            variable: Variable::Undefined,
            unit: Unit::Dimensionless,
        };
        let metadata = MetaGrid::new(&params);
        let data = vec![0.1, 0.2]; // Incorrect length
        let result = std::panic::catch_unwind(|| F32Data::new(&metadata, data));
        assert!(result.is_err(), "Expected panic due to dimension mismatch");
    }
    #[test_log::test]
    fn test_file_format_from_str_and_as_str() {
        assert_eq!(
            FileFormat::from_fileformat_str("binary"),
            Some(FileFormat::Binary)
        );
        assert_eq!(
            FileFormat::from_fileformat_str("compressedbinary"),
            Some(FileFormat::Lz4)
        );
        assert_eq!(
            FileFormat::from_fileformat_str("png"),
            Some(FileFormat::Png)
        );
        assert_eq!(FileFormat::from_fileformat_str("unknown"), None);

        assert_eq!(FileFormat::Binary.as_str(), "binary");
        assert_eq!(FileFormat::Lz4.as_str(), "compressedbinary");
        assert_eq!(FileFormat::Png.as_str(), "png");
    }

    #[test_log::test]
    fn test_file_format_from_and_as_extension() {
        assert_eq!(FileFormat::from_extension("bin"), Some(FileFormat::Binary));
        assert_eq!(FileFormat::from_extension("lz4"), Some(FileFormat::Lz4));
        assert_eq!(FileFormat::from_extension("png"), Some(FileFormat::Png));
        assert_eq!(FileFormat::from_extension("txt"), None);

        assert_eq!(FileFormat::Binary.as_extension(), "bin");
        assert_eq!(FileFormat::Lz4.as_extension(), "lz4");
        assert_eq!(FileFormat::Png.as_extension(), "png");
    }
    #[test_log::test]
    fn test_write_and_read_lz4_bin() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_write_and_read_lz4_bin");
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        write_lz4_bin(&file_path, &data);
        let decompressed = read_lz4(&file_path.with_extension("lz4")).unwrap();
        assert_eq!(decompressed, data);
        let _ = fs::remove_file(file_path.with_extension("lz4"));
    }

    #[test_log::test]
    fn test_read_compressed_bin_invalid_data() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_invalid_compressed.lz4");
        write_bin(&file_path, &vec![1, 2, 3, 4]); // Not actually compressed
        let result = read_lz4(&file_path.with_extension("bin"));
        assert!(result.is_err());
        let _ = fs::remove_file(file_path.with_extension("bin"));
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

    #[test_log::test]
    fn test_unit_and_variable_conversions_cover_fallbacks() {
        assert_eq!(Unit::from_int(0), Some(Unit::MetersPerSecond));
        assert_eq!(Unit::from_int(1), Some(Unit::Degree));
        assert_eq!(Unit::from_int(2), Some(Unit::Kilogram));
        assert_eq!(Unit::from_int(99), Some(Unit::Dimensionless));
        assert_eq!(Unit::Dimensionless.as_int(), 255);
        assert_eq!(Unit::Degree.as_str(), "°");

        assert_eq!(Variable::from_int(0), Some(Variable::Velocity));
        assert_eq!(Variable::from_int(3), Some(Variable::SlopeAspect));
        assert_eq!(Variable::from_int(7), Some(Variable::Mass));
        assert_eq!(Variable::from_int(99), Some(Variable::Undefined));
        assert_eq!(Variable::NormalZ.as_int(), 6);
        assert_eq!(Variable::Undefined.as_str(), "undefined");
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
        let (sim_settings, dem) = create_sim_settings_and_dem_from_path(PARABOLA_PATH).await;
        assert_eq!(dem.width, 1001);
        assert_eq!(dem.height, 401);
        assert_eq!(sim_settings.grid_shape_x, dem.width as u32);
        assert_eq!(sim_settings.grid_shape_y, dem.height as u32);
        assert_eq!(sim_settings.cell_size, dem.cell_size);

        let settings = Settings {
            dem_path: Some(PARABOLA_PATH.to_string()),
            max_steps: Some(42),
            density: Some(321.0),
            ..Settings::default()
        };
        let settings_file = NamedTempFile::new().unwrap();
        fs::write(settings_file.path(), settings.dumps().unwrap()).unwrap();

        let (json_sim_settings, json_dem) =
            sim_settings_and_dem_from_json_file(settings_file.path().to_str().unwrap()).await;

        assert_eq!(json_dem.width, dem.width);
        assert_eq!(json_dem.height, dem.height);
        assert_eq!(json_sim_settings.max_steps, 42);
        assert_eq!(json_sim_settings.density, 321.0);
        assert_eq!(json_sim_settings.grid_shape_x, dem.width as u32);
        assert_eq!(json_sim_settings.grid_shape_y, dem.height as u32);
    }

    #[test_log::test]
    fn test_save_grid_writes_round_trippable_metadata_and_data() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let file_path = tmp_dir.path().join("saved_grid");
        let dem = Dem {
            width: 2,
            height: 2,
            bounds: Bounds {
                xmin: 10.0,
                xmax: 20.0,
                ymin: 30.0,
                ymax: 40.0,
            },
            data1d: vec![1.0, 2.0, 3.0, 4.0],
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            x: vec![10.0, 20.0],
            y: vec![30.0, 40.0],
            cell_size: 5.0,
            map_factor: 0.75,
            minimum_elevation: 1.0,
        };
        let values = vec![9.0, 8.0, 7.0, 6.0];

        save_grid(&dem, file_path.to_str().unwrap(), values.clone()).unwrap();

        let stored = F32Data::load(file_path.to_str().unwrap()).unwrap();
        assert_eq!(stored.metadata.width, 2);
        assert_eq!(stored.metadata.height, 2);
        assert_eq!(stored.metadata.cell_size, 5.0);
        assert_eq!(stored.metadata.map_factor, 0.75);
        assert_eq!(stored.data, values);
    }

    #[tokio::test]
    // #[ignore]
    async fn test_write_and_read_file_size_bin() {
        // For avaMal.png
        // Start PNG:      744017 bytes
        // Format  WriteTime   FileSize         ReadTime
        // LZ4:    18.3895ms    1127904 bytes    12.0114ms
        // ZST:   2.8399438s     902141 bytes    26.5853ms
        // BIN:     1.4439ms    1779504 bytes    10.7315ms
        // PNG:   237.1214ms     868965 bytes     97.231ms
        // XZ :   566.5664ms     621664 bytes    87.8251ms

        // For avaArzlerUni.png
        // Start PNG:     2463733 bytes
        // Format  WriteTime   FileSize           ReadTime
        // LZ4:        9.1ms    3775084 bytes    14.7216ms
        // ZST:   2.3181056s    2994253 bytes    71.6658ms
        // BIN:     3.0754ms    3763260 bytes    10.6038ms
        // PNG:   550.2348ms    2860167 bytes   214.7707ms
        // XZ :   1.9335498s    2158308 bytes   243.1962ms
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_write_file");
        let png_path = "../../frontend/data/avaframe/avaArzlerUni.png";
        print!("Start PNG: ");
        print_file_size(Path::new(png_path));
        println!();
        let (data, width, height) = read_png(png_path).await.expect("Failed to load PNG");
        println!("Format  WriteTime   FileSize           ReadTime");

        let mut start = Instant::now();
        write_lz4_bin(&file_path, &data);
        let mut duration = start.elapsed();
        print!("LZ4: {:>12?}", duration);
        print_file_size(&file_path.with_extension("lz4"));

        start = Instant::now();
        let decompressed = read_lz4(&file_path.with_extension("lz4")).unwrap();
        duration = start.elapsed();
        println!(" {:>12?}", duration);
        assert_eq!(decompressed, data);

        start = Instant::now();
        write_zstd(&file_path, &data);
        duration = start.elapsed();
        print!("ZST: {:>12?}", duration);
        print_file_size(&file_path.with_extension("zst"));

        start = Instant::now();
        let decompressed_zstd = read_zstd_bin(&file_path.with_extension("zst")).unwrap();
        duration = start.elapsed();
        println!(" {:>12?}", duration);
        assert_eq!(decompressed_zstd, data);

        start = Instant::now();
        write_bin(&file_path, &data);
        duration = start.elapsed();
        print!("BIN: {:>12?}", duration);
        print_file_size(&file_path.with_extension("bin"));

        start = Instant::now();
        let decompressed_bin = read_bin(&file_path.with_extension("bin")).unwrap();
        duration = start.elapsed();
        println!(" {:>12?}", duration);
        assert_eq!(decompressed_bin, data);

        start = Instant::now();
        let _ = write_png(&file_path, &data, width, height);
        duration = start.elapsed();
        print!("PNG: {:>12?}", duration);
        print_file_size(&file_path.with_extension("png"));

        start = Instant::now();
        let decompressed_png = read_png(file_path.with_extension("png").to_str().unwrap())
            .await
            .unwrap()
            .0;
        duration = start.elapsed();
        println!(" {:>12?}", duration);
        assert_eq!(decompressed_png, data);

        start = Instant::now();
        let _ = write_xz(&file_path, &data);
        duration = start.elapsed();
        print!("XZ : {:>12?}", duration);
        print_file_size(&file_path.with_extension("xz"));

        start = Instant::now();
        let decompressed_xz = read_xz(&file_path.with_extension("xz")).unwrap();
        duration = start.elapsed();
        println!(" {:>12?}", duration);
        assert_eq!(decompressed_xz, data);
        // let _ = fs::remove_file(file_path.with_extension("lz4"));
    }

    // fn file_format_performance(fwrite: impl Fn(&Path, &Vec<u8>), fread: impl Fn(&Path) -> Vec<u8>, format: &str, data: &Vec<u8>) {
    //     let mut start = Instant::now();
    //     fwrite(&file_path, &data);
    //     let mut duration = start.elapsed();
    //     println!("File format performance took: {:?}", duration);
    //     print_file_size(&file_path.with_extension(format), format!("{}  File", format));

    //     start = Instant::now();
    //     let decompressed_bin = fread(&file_path.with_extension(format)).unwrap();
    //     duration = start.elapsed();
    //     println!("Decompression took: {:?}\n", duration);
    //     assert_eq!(decompressed_bin, data);
    // }

    fn print_file_size(path: &Path) {
        let file_size = std::fs::metadata(path)
            .expect("Failed to get file metadata")
            .len();
        print!(" {:>10} bytes", file_size);
    }
}
