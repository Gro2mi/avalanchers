use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use bincode::{Decode, Encode, config};
use std::io::BufReader;
use std::io::Cursor;
use std::io::Read;

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
#[cfg(not(target_arch = "wasm32"))]
use xz2::{read::XzDecoder, write::XzEncoder};
use zstd::stream::{decode_all, encode_all};

use image::{GenericImageView, ImageReader};

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

pub fn write_zstd(path: &Path, buffer: &Vec<u8>) {
    let file = File::create(path.with_extension("zst")).expect("Failed to create file");
    let compressed_data = encode_all(Cursor::new(buffer), 22).expect("Failed to compress data");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16 MB buffer
    writer
        .write_all(&compressed_data)
        .expect("Failed to write data");
}

pub fn read_zstd_bin(path: &Path) -> io::Result<Vec<u8>> {
    read_bin(&path.with_extension("zst")).and_then(|buffer| decode_all(Cursor::new(&buffer[..])))
}

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
        panic!(panic!("XZ compression is not supported on this platform");)
    }
}

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
        panic!(panic!("XZ compression is not supported on this platform");)
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

pub fn read_png(path: &Path) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error>> {
    let img = ImageReader::open(path.with_extension("png"))?.decode()?;
    let rgba = img.to_rgba8();
    let (width, height) = img.dimensions();
    Ok((rgba.into_raw().to_vec(), width as usize, height as usize))
}

pub fn rgba_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn f32_to_rgba_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::env;
    use std::fs;
    use std::time::Instant;

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
    // #[ignore]
    fn test_write_and_read_file_size_bin() {
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
        let png_path = PathBuf::from("../../data/avaframe/avaArzlerUni.png");
        print!("Start PNG: ");
        print_file_size(&png_path);
        println!();
        let (data, width, height) = read_png(&png_path).expect("Failed to load PNG");
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
        let decompressed_png = read_png(&file_path.with_extension("png")).unwrap().0;
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
