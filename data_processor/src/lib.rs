use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

// src/lib.rs
use bincode::{Decode, Encode, config};
use half::f16;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use pyo3::exceptions::PyValueError;
use pyo3::{buffer, prelude::*};
use std::io::Read;

use image::{GenericImageView, ImageBuffer, ImageReader};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Binary,
    Lz4,
    Png,
}
impl FileFormat {
    pub fn from_str(value: &str) -> Option<Self> {
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

#[pyclass]
#[derive(Encode, Decode, PartialEq, Debug, Clone)]
pub struct Metadata {
    #[pyo3(get)]
    magic_bytes: u32,
    #[pyo3(get)]
    pub version: u8,
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
    pub data_type: u8, // 2 for f32
}

// Implement methods for MyMetadata for Python
#[pymethods]
impl Metadata {
    #[new] // This makes it callable as `MyMetadata(name, timestamp, version)` in Python
    fn new(magic_bytes: u32, width: u32, height: u32, data_type: DataType) -> Self {
        Metadata {
            // u32::from_le_bytes(*b"AVAG")
            magic_bytes: magic_bytes,
            version: 1,
            width: width,
            height: height,
            data_type: data_type.as_int(),
        }
    }

    // A simple representation for Python's `repr()`
    fn __repr__(&self) -> String {
        format!(
            "Metadata(magic_bytes='{}', version={}, width={}, height={}, data_type={})",
            std::str::from_utf8(&self.magic_bytes.to_le_bytes()).unwrap_or("????"),
            self.version,
            self.width,
            self.height,
            DataType::from_int(self.data_type).unwrap().as_str()
        )
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
    fn __eq__(&self, other: &Metadata) -> bool {
        self.magic_bytes == other.magic_bytes
            && self.version == other.version
            && self.width == other.width
            && self.height == other.height
            && self.data_type == other.data_type
    }
}

pub struct MetaGrid{
    pub metadata: Metadata,
}
impl MetaGrid {
    pub fn new(width: u32, height: u32, data_type: DataType) -> Self {
        MetaGrid {
            metadata: Metadata::new(u32::from_le_bytes(*b"AVAG"), width, height, data_type),
        }
    }
}

#[pyclass]
#[derive(Encode, Decode, PartialEq, Debug)]
pub struct F32Data {
    #[pyo3(get, set)]
    pub metadata: Metadata,
    #[pyo3(get, set)]
    pub data: Vec<f32>,
}

#[pymethods]
impl F32Data {
    #[new]
    fn new(metadata: &Metadata, data: Vec<f32>) -> Self {
        assert!(
            metadata.width * metadata.height == data.len() as u32,
            "Data length does not match metadata dimensions"
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

    fn save(&self, path: &str) -> PyResult<()> {
        let path = Path::new(path);
        // Serialize the F32Data object to a binary format
        let encoded_bytes = bincode::encode_to_vec(&self, config::standard())
            .map_err(|e| PyValueError::new_err(format!("Bincode serialization failed: {}", e)))?;
        write_bin(path, &encoded_bytes);
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let buffer = read_bin(Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("Failed to read file: {}", e)))?;
        let (data, _): (F32Data, _) = bincode::decode_from_slice(&buffer, config::standard())
            .map_err(|e| PyValueError::new_err(format!("Bincode deserialization failed: {}", e)))?;
        assert!(
            data.metadata.magic_bytes == u32::from_le_bytes(*b"AVAG"),
            "Invalid magic bytes"
        );
        assert!(
            data.metadata.data_type == 32,
            "Wrong data type: {} instead of f32",
            data.metadata.data_type
        );
        Ok(data)
    }
}

pub fn read_bin(path: &Path) -> PyResult<Vec<u8>> {
    let path = Path::new(path);
    let mut file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| PyValueError::new_err(format!("Failed to read file: {}", e)))?;
    Ok(buffer)
}

pub fn write_bin(path: &Path, buffer: &Vec<u8>) {
    let file = File::create(path.with_extension("bin")).expect("Failed to create file");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16 MB buffer
    writer.write_all(&buffer).expect("Failed to write data");
}

pub fn write_compressed_bin(path: &Path, buffer: &Vec<u8>) {
    let file = File::create(path.with_extension("lz4")).expect("Failed to create file");
    let compressed_data = compress_prepend_size(buffer);
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16 MB buffer
    writer
        .write_all(&compressed_data)
        .expect("Failed to write data");
}

pub fn read_compressed_bin(path: &Path) -> PyResult<Vec<u8>> {
    read_bin(path).and_then(|buffer| {
        decompress_size_prepended(&buffer)
            .map_err(|e| PyValueError::new_err(format!("Failed to decompress data: {}", e)))
    })
}

pub fn save_png(
    path: &Path,
    data: &[u8],
    width: usize,
    height: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    // Avoid copying data by using a slice reference instead of to_vec()
    let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width as u32, height as u32, data)
        .ok_or("Failed to create image buffer")?;
    img.save(path)?;
    let duration = start.elapsed();
    println!("Image creation and saving took: {:?}", duration);
    Ok(())
}

pub fn load_png(path: &Path) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error>> {
    let img = ImageReader::open(path)?.decode()?;
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
    data.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

#[pymodule]
fn data_processor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the Rust structs as Python classes
    m.add_class::<Metadata>()?;
    m.add_class::<F32Data>()?;
    m.add_class::<DataType>()?;
    m.add_class::<FileFormat>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[test]
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
    #[test]
    fn test_metadata_new_grid() {
        let width = 128;
        let height = 256;
        let data_type = DataType::F32;
        let metadata = Metadata::new_grid(width, height, data_type);
        assert_eq!(metadata.width, width);
        assert_eq!(metadata.height, height);
        assert_eq!(metadata.data_type, 32);
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.magic_bytes, u32::from_le_bytes(*b"AVAG"));
    }

    #[test]
    fn test_data_type_from_and_as_int() {
        assert_eq!(DataType::from_int(16), Some(DataType::F16));
        assert_eq!(DataType::from_int(32), Some(DataType::F32));
        assert_eq!(DataType::from_int(64), Some(DataType::F64));
        assert_eq!(DataType::from_int(8), None);

        assert_eq!(DataType::F16.as_int(), 16);
        assert_eq!(DataType::F32.as_int(), 32);
        assert_eq!(DataType::F64.as_int(), 64);
    }

    #[test]
    fn test_data_type_as_str() {
        assert_eq!(DataType::F16.as_str(), "f16");
        assert_eq!(DataType::F32.as_str(), "f32");
        assert_eq!(DataType::F64.as_str(), "f64");
    }

    #[test]
    fn test_f32data_new_and_repr() {
        let metadata = Metadata::new_grid(1, 3, DataType::F32);
        let data = vec![1.0f32, 2.0, 3.0];
        let f32data = F32Data::new(&metadata, data.clone());
        assert_eq!(f32data.metadata, metadata);
        assert_eq!(f32data.data, data);
        let repr = f32data.__repr__();
        assert!(repr.contains("F32Data(metadata="));
        assert!(repr.contains("data_len=3"));
    }

    #[test]
    fn test_metadata_repr() {
        let metadata = Metadata::new_grid(5, 6, DataType::F32);
        let repr = metadata.__repr__();
        assert!(repr.contains("Metadata(magic_bytes="));
        assert!(repr.contains("version=1"));
        assert!(repr.contains("width=5"));
        assert!(repr.contains("height=6"));
        assert!(repr.contains("data_type=32"));
    }

    #[test]
    fn test_write_and_read_bin() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_write_and_read_bin");
        let data = vec![1u8, 2, 3, 4, 5];
        write_bin(&file_path, &data);
        let read = read_bin(&file_path.with_extension("bin")).unwrap();
        assert_eq!(read, data);
        let _ = fs::remove_file(file_path.with_extension("bin"));
    }

    #[test]
    fn test_f32data_save_and_load() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_f32data_save_and_load.bin");
        let metadata = Metadata::new_grid(2, 2, DataType::F32);
        let data = vec![0.1, 0.2, 0.3, 0.4];
        let f32data = F32Data::new(&metadata, data.clone());
        f32data.save(file_path.to_str().unwrap()).unwrap();

        let loaded = F32Data::load(file_path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.metadata, metadata);
        assert_eq!(loaded.data, data);

        let _ = fs::remove_file(file_path);
    }

    #[test]
    fn test_dimension_mismatch() {
        let metadata = Metadata::new_grid(2, 2, DataType::F32);
        let data = vec![0.1, 0.2]; // Incorrect length
        let result = std::panic::catch_unwind(|| F32Data::new(&metadata, data));
        assert!(result.is_err(), "Expected panic due to dimension mismatch");
    }
    #[test]
    fn test_file_format_from_str_and_as_str() {
        assert_eq!(FileFormat::from_str("binary"), Some(FileFormat::Binary));
        assert_eq!(
            FileFormat::from_str("compressedbinary"),
            Some(FileFormat::Lz4)
        );
        assert_eq!(FileFormat::from_str("png"), Some(FileFormat::Png));
        assert_eq!(FileFormat::from_str("unknown"), None);

        assert_eq!(FileFormat::Binary.as_str(), "binary");
        assert_eq!(FileFormat::Lz4.as_str(), "compressedbinary");
        assert_eq!(FileFormat::Png.as_str(), "png");
    }

    #[test]
    fn test_file_format_from_and_as_extension() {
        assert_eq!(FileFormat::from_extension("bin"), Some(FileFormat::Binary));
        assert_eq!(FileFormat::from_extension("lz4"), Some(FileFormat::Lz4));
        assert_eq!(FileFormat::from_extension("png"), Some(FileFormat::Png));
        assert_eq!(FileFormat::from_extension("txt"), None);

        assert_eq!(FileFormat::Binary.as_extension(), "bin");
        assert_eq!(FileFormat::Lz4.as_extension(), "lz4");
        assert_eq!(FileFormat::Png.as_extension(), "png");
    }
    #[test]
    fn test_write_and_read_compressed_bin() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_write_and_read_compressed_bin");
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        write_compressed_bin(&file_path, &data);
        let decompressed = read_compressed_bin(&file_path.with_extension("lz4")).unwrap();
        assert_eq!(decompressed, data);
        let _ = fs::remove_file(file_path.with_extension("lz4"));
    }

    #[test]
    fn test_read_compressed_bin_invalid_data() {
        let tmp_dir = env::temp_dir();
        let file_path = tmp_dir.join("test_invalid_compressed.lz4");
        write_bin(&file_path, &vec![1, 2, 3, 4]); // Not actually compressed
        let result = read_compressed_bin(&file_path.with_extension("bin"));
        assert!(result.is_err());
        let _ = fs::remove_file(file_path.with_extension("bin"));
    }

    #[test]
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
}
