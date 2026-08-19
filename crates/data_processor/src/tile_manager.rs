//! swissALTI3D fetch and cache. Tries 8 acquisition years per tile (the shipped code hardcoded 2019), falls back to `decode_tiled_lzw_f32` for tiles whose LZW stream lacks an end code, resamples 2 m → 5 m by area weighting, stores into a Zarr array. `get_dem(bbox)` is the entry point.
use ndarray::{Array1, Array2};
use reqwest::Client;
use serde_json::json;
use std::io::Cursor;
use std::sync::Arc;
use tiff::decoder::{Decoder, DecodingResult};

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};
use zarrs::array::ArraySubset;
use zarrs::array::{
    Array, ArrayBuilder, FillValue,
    codec::bytes_to_bytes::blosc::{
        BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    },
};
use zarrs::group::GroupBuilder;
use zarrs::storage::StorageError;
use zarrs_filesystem::FilesystemStore;

use compute_core::dem::{Bounds, Dem};
use compute_core::utils::{linspace, to_2d};

#[derive(thiserror::Error, Debug)]
pub enum TileManagerError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("TIFF decoding error: {0}")]
    Tiff(#[from] tiff::TiffError),

    #[error("Zarr storage error: {0}")]
    Zarr(#[from] zarrs::array::ArrayError),

    #[error("Zarr creation error: {0}")]
    ZarrCreate(#[from] zarrs::array::ArrayCreateError),

    #[error("Missing or invalid data")]
    InvalidData,

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Invalid tile shape: got {0:?}")]
    InvalidShape((usize, usize)),

    #[error(transparent)]
    ZarrDimensionality(#[from] zarrs::array::IncompatibleDimensionalityError),

    #[error(transparent)]
    Store(#[from] StorageError),

    #[error(transparent)]
    GroupCreate(#[from] zarrs::group::GroupCreateError),

    #[error(transparent)]
    PluginCreate(#[from] zarrs_plugin::PluginCreateError),

    #[error(transparent)]
    FilesystemStoreCreate(#[from] zarrs::filesystem::FilesystemStoreCreateError),

    #[error("Fallback TIFF decode failed: {0}")]
    TiffFallback(String),
}

/// Minimal reader for the exact flavour of GeoTIFF that swisstopo serves for
/// swissALTI3D: tiled, LZW-compressed, single-band 32-bit float, no predictor.
///
/// Some of those tiles end their LZW stream without an end-of-information code.
/// The `tiff` crate treats that as a hard error ("no lzw end code found") and
/// refuses the whole image, so we decode the tiles ourselves and simply accept
/// a stream that ends early. Returns `(width, height, row-major f32 data)` with
/// row 0 at the *top* (native TIFF order), matching `Decoder::read_image`.
pub fn decode_tiled_lzw_f32(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), TileManagerError> {
    use weezl::{BitOrder, decode::Decoder as LzwDecoder};

    let err = |m: &str| TileManagerError::TiffFallback(m.to_string());
    if bytes.len() < 8 {
        return Err(err("file too short"));
    }
    let little = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Err(err("bad byte order mark")),
    };
    let u16at = |o: usize| -> u32 {
        let b = [bytes[o], bytes[o + 1]];
        if little {
            u16::from_le_bytes(b) as u32
        } else {
            u16::from_be_bytes(b) as u32
        }
    };
    let u32at = |o: usize| -> u32 {
        let b = [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
        if little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        }
    };
    if u16at(2) != 42 {
        return Err(err("not a classic TIFF"));
    }

    let ifd = u32at(4) as usize;
    if ifd + 2 > bytes.len() {
        return Err(err("IFD out of range"));
    }
    let n_entries = u16at(ifd) as usize;
    let mut tags: std::collections::HashMap<u32, (u32, u32, usize)> = Default::default();
    for i in 0..n_entries {
        let e = ifd + 2 + i * 12;
        if e + 12 > bytes.len() {
            return Err(err("IFD entry out of range"));
        }
        tags.insert(u16at(e), (u16at(e + 2), u32at(e + 4), e + 8));
    }

    // Reads a tag as a list of u32 (handling SHORT/LONG, inline or out-of-line).
    let values = |tag: u32| -> Option<Vec<u32>> {
        let &(ty, count, voff) = tags.get(&tag)?;
        let size = match ty {
            3 => 2usize,
            4 => 4usize,
            _ => return None,
        };
        let total = size * count as usize;
        let base = if total <= 4 {
            voff
        } else {
            u32at(voff) as usize
        };
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let o = base + i * size;
            if o + size > bytes.len() {
                return None;
            }
            out.push(if size == 2 { u16at(o) } else { u32at(o) });
        }
        Some(out)
    };
    let scalar = |tag: u32, default: u32| -> u32 {
        values(tag)
            .and_then(|v| v.first().copied())
            .unwrap_or(default)
    };

    let width = scalar(256, 0);
    let height = scalar(257, 0);
    if width == 0 || height == 0 {
        return Err(err("missing image dimensions"));
    }
    if scalar(258, 0) != 32 || scalar(339, 1) != 3 || scalar(277, 1) != 1 {
        return Err(err("expected single-band 32-bit float samples"));
    }
    if scalar(259, 1) != 5 {
        return Err(err("expected LZW compression"));
    }
    if scalar(317, 1) != 1 {
        return Err(err("predictor not supported"));
    }
    let tile_w = scalar(322, 0) as usize;
    let tile_h = scalar(323, 0) as usize;
    if tile_w == 0 || tile_h == 0 {
        return Err(err("expected a tiled TIFF"));
    }
    let offsets = values(324).ok_or_else(|| err("missing tile offsets"))?;
    let counts = values(325).ok_or_else(|| err("missing tile byte counts"))?;
    if offsets.len() != counts.len() {
        return Err(err("tile offset/bytecount mismatch"));
    }

    let (w, h) = (width as usize, height as usize);
    let tiles_across = w.div_ceil(tile_w);
    let mut out = vec![f32::NAN; w * h];
    let tile_bytes = tile_w * tile_h * 4;
    let mut raw = vec![0u8; tile_bytes];

    for (i, (&off, &len)) in offsets.iter().zip(counts.iter()).enumerate() {
        let (off, len) = (off as usize, len as usize);
        if off + len > bytes.len() {
            return Err(err("tile data out of range"));
        }
        raw.iter_mut().for_each(|b| *b = 0);
        let mut dec = LzwDecoder::with_tiff_size_switch(BitOrder::Msb, 8);
        // weezl decodes incrementally; keep feeding it until the stream is done,
        // the output tile is full, or it stops making progress. A stream that
        // ends without an EOI code is accepted with whatever it produced.
        let mut in_pos = 0usize;
        let mut produced = 0usize;
        loop {
            let res = dec.decode_bytes(&bytes[off + in_pos..off + len], &mut raw[produced..]);
            in_pos += res.consumed_in;
            produced += res.consumed_out;
            let done = match res.status {
                Ok(weezl::LzwStatus::Done) | Ok(weezl::LzwStatus::NoProgress) => true,
                Ok(weezl::LzwStatus::Ok) => res.consumed_in == 0 && res.consumed_out == 0,
                Err(_) => true,
            };
            if done || in_pos >= len || produced >= tile_bytes {
                break;
            }
        }
        if produced == 0 {
            return Err(err("tile decoded to zero bytes"));
        }

        let tx = (i % tiles_across) * tile_w;
        let ty = (i / tiles_across) * tile_h;
        for row in 0..tile_h {
            let y = ty + row;
            if y >= h {
                break;
            }
            for col in 0..tile_w {
                let x = tx + col;
                if x >= w {
                    break;
                }
                let src = (row * tile_w + col) * 4;
                if src + 4 > produced {
                    continue;
                }
                let b = [raw[src], raw[src + 1], raw[src + 2], raw[src + 3]];
                let v = if little {
                    f32::from_le_bytes(b)
                } else {
                    f32::from_be_bytes(b)
                };
                out[y * w + x] = v;
            }
        }
    }

    Ok((width, height, out))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    pub northing: u32,
    pub easting: u32,
}

#[derive(Debug, Clone)]
pub struct BBox {
    pub min_northing: u32,
    pub max_northing: u32,
    pub min_easting: u32,
    pub max_easting: u32,
}

pub struct TileManager {
    client: Client,
    // zarr_storage: Arc<FilesystemStore>,
    dtm: Array<FilesystemStore>,
    min_easting_km: u64,
    min_northing_km: u64,
}

impl TileManager {
    /// Initializes the Tile Manager with a local Zarr filesystem store
    pub fn new(cache_dir: &str) -> Result<Self, TileManagerError> {
        let store = Arc::new(FilesystemStore::new(cache_dir)?);

        let array_path = "/dtm";
        // for switzerland
        let min_easting_km: u64 = 2480;
        let max_easting: u64 = 2840;
        let min_northing_km: u64 = 1060;
        let max_northing: u64 = 1300;
        let tile_size: u64 = 200;
        let cell_size = 5.0;

        let overall_shape = [
            (max_northing - min_northing_km + 1) * tile_size,
            (max_easting - min_easting_km + 1) * tile_size,
        ];
        let chunk_shape = [tile_size, tile_size];

        let mut root_group = GroupBuilder::new().build(store.clone(), "/")?;
        let global_attrs = json!({
            "title": "SwissALTI3D DTM Resampled to 5m",
            "comment": "Swiss terrain model resampled from original 2m product",
            "conventions": "CF-1.8",
            "source": "Federal Office of Topography swisstopo",
            "crs": "EPSG:2056",

        });
        root_group
            .attributes_mut()
            .extend(global_attrs.as_object().unwrap().clone());
        root_group.store_metadata()?;

        let blosc = Arc::new(BloscCodec::new(
            BloscCompressor::Zstd,
            BloscCompressionLevel::try_from(9).expect("Invalid compression level"),
            None, // automatic blocksize
            BloscShuffleMode::BitShuffle,
            Some(4), // f32 = 4 bytes
        )?);
        let mut dtm = zarrs::array::ArrayBuilder::new(
            overall_shape, // array shape
            chunk_shape,   // regular chunk shape
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["y", "x"].into())
        .build(store.clone(), array_path)?;

        let x_data = Array1::from_iter(
            (0..overall_shape[1])
                .map(|i| cell_size / 2.0 + i as f32 * cell_size + min_easting_km as f32 * 1000.0),
        );

        let mut x = zarrs::array::ArrayBuilder::new(
            vec![overall_shape[1]], // array shape
            vec![overall_shape[1]], // regular chunk shape
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["x"].into())
        .build(store.clone(), "/x")?;
        x.attributes_mut().extend(
            json!({
                "standard_name": "x",
                "long_name": "Easting",
                "units": "m",
                "axis": "X"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        x.store_metadata()?;
        #[allow(clippy::single_range_in_vec_init)]
        x.store_chunks(&[0..1], x_data)?;

        let y_data = Array1::from_iter(
            (0..overall_shape[0])
                .map(|i| cell_size / 2.0 + i as f32 * cell_size + min_northing_km as f32 * 1000.0),
        );
        let mut y = zarrs::array::ArrayBuilder::new(
            vec![overall_shape[0]], // array shape
            vec![overall_shape[0]], // regular chunk shape
            zarrs::array::data_type::float32(),
            f32::NAN,
        )
        .bytes_to_bytes_codecs(vec![blosc.clone()])
        .dimension_names(["y"].into())
        .build(store.clone(), "/y")?;
        #[allow(clippy::single_range_in_vec_init)]
        y.store_chunks(&[0..1], y_data)?;
        y.attributes_mut().extend(
            json!({
                "standard_name": "y",
                "long_name": "Northing",
                "units": "m",
                "axis": "Y"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        y.store_metadata()?;

        dtm.attributes_mut().extend(
            json!({
                    "title": "SwissALTI3D Digital Terrain Model (5 m) for avalanche simulatio",
                "coordinates": "x y",
                "standard_name": "elevation",
                "long_name": "Elevation",
                "units": "m",
                "grid_mapping": "spatial_ref",
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        dtm.store_metadata()?;
        let mut spatial_ref = ArrayBuilder::new(
            vec![],                           // Shape: 0D
            Vec::<u64>::new(),                // Chunk Grid for 0D
            zarrs::array::data_type::int32(), // Data Type
            FillValue::from(0i32),            // Fill Value
        )
        .build(store.clone(), "/spatial_ref")?;

        spatial_ref.attributes_mut().extend(
            json!({
                "_ARRAY_DIMENSIONS": [],
                "grid_mapping_name": "hotine_oblique_mercator", // LV95 uses a modified oblique mercator
                "crs_string": "EPSG:2056",
                "spatial_ref": "PROJCS[\"CH1903+ / LV95\",
    GEOGCS[\"CH1903+\",
        DATUM[\"CH1903+\",
            SPHEROID[\"Bessel 1841\",6377397.155,299.1528128],
            TOWGS84[674.374,15.056,405.346,0,0,0,0]],
        PRIMEM[\"Greenwich\",0,
            AUTHORITY[\"EPSG\",\"8901\"]],
        UNIT[\"degree\",0.0174532925199433,
            AUTHORITY[\"EPSG\",\"9122\"]],
        AUTHORITY[\"EPSG\",\"4150\"]],
    PROJECTION[\"Hotine_Oblique_Mercator_Azimuth_Center\"],
    PARAMETER[\"latitude_of_center\",46.9524055555556],
    PARAMETER[\"longitude_of_center\",7.43958333333333],
    PARAMETER[\"azimuth\",90],
    PARAMETER[\"rectified_grid_angle\",90],
    PARAMETER[\"scale_factor\",1],
    PARAMETER[\"false_easting\",2600000],
    PARAMETER[\"false_northing\",1200000],
    UNIT[\"metre\",1,
        AUTHORITY[\"EPSG\",\"9001\"]],
    AXIS[\"Easting\",EAST],
    AXIS[\"Northing\",NORTH],
    AUTHORITY[\"EPSG\",\"2056\"]]"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        spatial_ref.store_metadata()?;
        Ok(Self {
            client: Client::new(),
            // zarr_storage: store,
            dtm,
            min_easting_km,
            min_northing_km,
            // tile_size,
        })
    }

    /// Fetches the TIFF from geo.admin.ch and decodes it into a 500x500 f32 array
    async fn fetch_and_parse_tif(&self, id: TileId) -> Result<Array2<f32>, TileManagerError> {
        // swissALTI3D is flown region by region, so the acquisition year in the
        // asset name is not the same everywhere. Try the most common year first
        // and fall through the others rather than 404-ing on whole regions.
        const YEARS: [u32; 8] = [2019, 2020, 2021, 2022, 2023, 2024, 2018, 2017];
        let mut response = None;
        let mut url = String::new();
        let mut last_status = reqwest::StatusCode::NOT_FOUND;
        for year in YEARS {
            let candidate = format!(
                "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_{year}_{}-{}/swissalti3d_{year}_{}-{}_2_2056_5728.tif",
                id.easting, id.northing, id.easting, id.northing,
            );
            let r = self.client.get(&candidate).send().await?;
            last_status = r.status();
            if r.status().is_success() {
                url = candidate;
                response = Some(r);
                break;
            }
        }
        let response = match response {
            Some(r) => r,
            None => {
                return Err(TileManagerError::Http(format!(
                    "tile {:?}: no swissALTI3D asset found (last status {})",
                    id, last_status
                )));
            }
        };

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("<unknown>")
            .to_string();

        if !status.is_success() {
            return Err(TileManagerError::Http(format!(
                "tile {:?}: HTTP {} ({})",
                id, status, url
            )));
        }

        let bytes = response.bytes().await?;

        if bytes.len() < 8 {
            return Err(TileManagerError::InvalidData);
        }

        // Primary path: the `tiff` crate. Some swissALTI3D tiles have an LZW
        // stream without a terminating end-of-information code, which it
        // rejects outright; those fall through to our own tolerant decoder.
        let decoded: Option<(u32, u32, Vec<f32>)> = match Decoder::new(Cursor::new(bytes.clone())) {
            Ok(mut decoder) => match (decoder.dimensions(), decoder.read_image()) {
                (Ok((width, height)), Ok(DecodingResult::F32(vec))) => Some((width, height, vec)),
                (Ok(_), Ok(other)) => {
                    warn!(
                        "Unexpected TIFF sample type for {:?}: {:?}",
                        id,
                        std::mem::discriminant(&other)
                    );
                    None
                }
                (_, Err(e)) | (Err(e), _) => {
                    warn!(
                        "tiff crate could not decode {:?} (content-type={}, bytes={}): {:?}; \
                         retrying with the tolerant LZW decoder",
                        id,
                        content_type,
                        bytes.len(),
                        e
                    );
                    None
                }
            },
            Err(e) => {
                warn!(
                    "tiff crate could not open {:?} (content-type={}, bytes={}): {:?}; \
                     retrying with the tolerant LZW decoder",
                    id,
                    content_type,
                    bytes.len(),
                    e
                );
                None
            }
        };

        let (width, height, mut vec) = match decoded {
            Some(v) => v,
            None => decode_tiled_lzw_f32(&bytes)?,
        };

        if vec.len() != (width * height) as usize {
            return Err(TileManagerError::InvalidData);
        }

        compute_core::utils::flip_rows_flat_vec(&mut vec, width, height);

        Array2::from_shape_vec((height as usize, width as usize), vec)
            .map_err(|_| TileManagerError::InvalidData)
    }

    fn resample_2m_to_5m(input: &Array2<f32>) -> Array2<f32> {
        let (in_h, in_w) = input.dim();

        let out_h = 200;
        let out_w = 200;

        let scale_y = in_h as f32 / out_h as f32; // should be 2.5
        let scale_x = in_w as f32 / out_w as f32;

        let mut output = Array2::<f32>::zeros((out_h, out_w));

        for y in 0..out_h {
            let y0 = y as f32 * scale_y;
            let y1 = (y + 1) as f32 * scale_y;

            let iy0 = y0.floor() as usize;
            let iy1 = y1.ceil() as usize;

            for x in 0..out_w {
                let x0 = x as f32 * scale_x;
                let x1 = (x + 1) as f32 * scale_x;

                let ix0 = x0.floor() as usize;
                let ix1 = x1.ceil() as usize;

                let mut sum = 0.0_f32;
                let mut weight_sum = 0.0_f32;

                for iy in iy0..iy1.min(in_h) {
                    // source pixel vertical bounds
                    let py0 = iy as f32;
                    let py1 = (iy + 1) as f32;

                    let dy = (py1.min(y1) - py0.max(y0)).max(0.0);
                    if dy == 0.0 {
                        continue;
                    }

                    for ix in ix0..ix1.min(in_w) {
                        let px0 = ix as f32;
                        let px1 = (ix + 1) as f32;

                        let dx = (px1.min(x1) - px0.max(x0)).max(0.0);
                        if dx == 0.0 {
                            continue;
                        }

                        let w = dx * dy;

                        sum += input[[iy, ix]] * w;
                        weight_sum += w;
                    }
                }

                output[[y, x]] = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    0.0
                };
            }
        }

        output
    }

    pub async fn get_dem(&self, bbox: &BBox) -> Result<Dem, TileManagerError> {
        // snap to 5 m grid
        let min_e = (bbox.min_easting / 5) * 5;
        let max_e = bbox.max_easting.div_ceil(5) * 5;

        let min_n = (bbox.min_northing / 5) * 5;
        let max_n = bbox.max_northing.div_ceil(5) * 5;

        let col0 = ((min_e as u64) - self.min_easting_km * 1000) / 5;
        let col1 = ((max_e as u64) - self.min_easting_km * 1000) / 5;

        let row0 = ((min_n as u64) - self.min_northing_km * 1000) / 5;
        let row1 = ((max_n as u64) - self.min_northing_km * 1000) / 5;

        let subset = ArraySubset::new_with_ranges(&[row0..row1, col0..col1]);

        //
        // ensure chunks exist
        //
        let chunks = self
            .dtm
            .chunks_in_array_subset(&subset)?
            .ok_or(TileManagerError::InvalidData)?;

        for chunk_indices in chunks.indices() {
            if self
                .dtm
                .retrieve_encoded_chunk(&chunk_indices)
                .map_err(|e| TileManagerError::Http(e.to_string()))?
                .is_none()
            {
                let tile = TileId {
                    easting: self.min_easting_km as u32 + chunk_indices[1] as u32,
                    northing: self.min_northing_km as u32 + chunk_indices[0] as u32,
                };

                let tif = self.fetch_and_parse_tif(tile).await?;
                let resampled = Self::resample_2m_to_5m(&tif);

                self.dtm
                    .store_chunk(&chunk_indices, resampled.as_slice().unwrap())?;
                info!("Cache miss")
            }
            info!("Cache hit")
        }

        //
        // read requested subset
        //
        let data: Vec<f32> = self.dtm.retrieve_array_subset::<Vec<f32>>(&subset)?;

        let rows = (row1 - row0) as usize;
        let cols = (col1 - col0) as usize;

        // `data` is retrieved row-major over the subset [northing_rows, easting_cols],
        // i.e. index = row * cols + col. A `Dem` is indexed data1d[y * width + x] with
        // x = easting and y = northing, so width must be the number of easting columns
        // and height the number of northing rows.
        let mut dem = Dem {
            width: cols,
            height: rows,
            data1d: data,
            data: Vec::new(),
            x: linspace(min_e as f32, max_e as f32, cols),
            y: linspace(min_n as f32, max_n as f32, rows),
            cell_size: 5.0,
            bounds: Bounds {
                xmin: min_e as f32,
                xmax: max_e as f32,
                ymin: min_n as f32,
                ymax: max_n as f32,
            },
            map_factor: 1.0,
            minimum_elevation: f32::INFINITY, // Will be calculated later
        };
        dem.data = to_2d(&dem.data1d, dem.width, dem.height);
        // `load_dem` does this for file-backed DEMs; without it the elevation
        // threshold handed to the particle shader stays at +inf.
        dem.minimum_elevation = Dem::calculate_minimum_elevation(&dem.data1d);

        Ok(dem)
    }
}
use std::fs::File;
use std::io::{BufWriter, Write};
pub fn export_array2_to_csv(
    array: &Array2<f32>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for row in array.rows() {
        let mut iter = row.iter();

        // Write the very first item in the row (if it exists)
        if let Some(&first_val) = iter.next() {
            write!(writer, "{}", first_val)?;
        }

        // Write all subsequent items preceded by a comma delimiter
        for &value in iter {
            write!(writer, ",{}", value)?;
        }

        // Append a line breaks to mark the completion of the row context
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, s};
    use tempfile::tempdir;

    #[tokio::test]
    async fn fetch_tiles() {
        let manager = TileManager::new("./dtm_cache").expect("msg");

        // Define bounding box (using your example coordinates)
        // let bbox = BBox {
        //     min_northing: 1159,
        //     max_northing: 1160,
        //     min_easting: 2666,
        //     max_easting: 2670,
        // };
        // let bbox = BBox {
        //     min_northing: 1159,
        //     max_northing: 1159,
        //     min_easting: 2666,
        //     max_easting: 2666,
        // };
        let bbox = BBox {
            min_northing: 1159502,
            max_northing: 1159549,
            min_easting: 2666100,
            max_easting: 2666150,
        };

        println!("Fetching, caching, and stitching tiles...");
        let dem = manager.get_dem(&bbox).await.expect("msg");
        assert_eq!(10, dem.width);
        assert_eq!(10, dem.height);
        println!(
            "width {}, height {}, data: {:?}",
            dem.width, dem.height, dem.data1d
        );
        let expected_data: Vec<f32> = vec![
            2193.765, 2194.0151, 2193.748, 2193.6294, 2193.561, 2193.7139, 2194.2805, 2194.536,
            2194.9507, 2195.626, 2191.5327, 2191.4226, 2191.3772, 2191.008, 2191.0254, 2191.234,
            2191.4453, 2191.792, 2192.5737, 2193.5688, 2189.028, 2188.6758, 2188.5618, 2188.2568,
            2188.1125, 2188.3752, 2188.4995, 2189.215, 2190.123, 2191.464, 2186.6091, 2186.1787,
            2185.8325, 2185.6245, 2185.6572, 2186.0525, 2186.2058, 2186.8484, 2188.0393, 2189.6365,
            2183.744, 2183.0432, 2182.8381, 2182.6968, 2182.8113, 2183.2524, 2183.5571, 2184.311,
            2186.238, 2187.9111, 2181.347, 2180.929, 2180.2715, 2180.165, 2180.377, 2180.5796,
            2181.1045, 2181.9316, 2183.7107, 2185.0256, 2178.81, 2178.4128, 2177.768, 2177.4077,
            2177.5413, 2177.7524, 2178.5132, 2179.2224, 2179.8389, 2181.1472, 2176.7634, 2176.1433,
            2175.3271, 2174.9033, 2174.7737, 2175.0557, 2175.6042, 2176.2063, 2176.9907, 2178.47,
            2174.459, 2173.519, 2172.7375, 2172.352, 2171.9524, 2172.0044, 2172.642, 2173.3938,
            2174.9087, 2176.5313, 2171.9233, 2171.021, 2170.148, 2169.5645, 2169.3586, 2169.3037,
            2170.0542, 2170.9807, 2172.953, 2174.6018,
        ];
        assert_eq!(dem.data1d, expected_data);
    }

    /// 1. Test the Zarr Cache Initialization
    /// Ensures that the directory is created and the root Zarr group is written.
    #[test]
    fn test_tile_manager_initialization() {
        // Create a temporary directory that will be automatically deleted
        let dir = tempdir().expect("Failed to create temp dir");
        let cache_path = dir.path().to_str().unwrap();

        let manager = TileManager::new(cache_path);

        assert!(
            manager.is_ok(),
            "TileManager should initialize successfully"
        );

        // FIX: Zarr V3 uses 'zarr.json' at the root instead of V2's '.zgroup'
        let zgroup_path = dir.path().join("zarr.json");
        assert!(
            zgroup_path.exists(),
            "Root zarr.json file should be created"
        );
    }

    /// 2. Test Resampling on Uniform Terrain
    /// A perfectly flat 2m terrain should result in a perfectly flat 5m terrain.
    #[test]
    fn test_resample_uniform_terrain() {
        // Create a 500x500 array where every point is exactly 1200.5 meters high
        let input_dtm = Array2::from_elem((500, 500), 1200.5f32);

        let output_dtm = TileManager::resample_2m_to_5m(&input_dtm);

        // Shape must be exactly 200x200
        assert_eq!(output_dtm.shape(), &[200, 200]);

        // Every pixel must remain strictly 1200.5
        for &elevation in output_dtm.iter() {
            assert_eq!(elevation, 1200.5);
        }
    }

    /// 3. Test Resampling on Step Terrain (Block Mean accuracy)
    /// Tests that the fractional bounding box logic correctly averages pixels.
    #[test]
    fn test_resample_step_terrain() {
        let mut input_dtm = Array2::<f32>::zeros((500, 500));

        // Fill the left half with 100m and the right half with 50m
        input_dtm.slice_mut(s![.., 0..250]).fill(100.0);
        input_dtm.slice_mut(s![.., 250..500]).fill(50.0);

        let output_dtm = TileManager::resample_2m_to_5m(&input_dtm);

        // Check the far left (should be purely 100)
        assert_eq!(output_dtm[[100, 10]], 100.0);

        // Check the far right (should be purely 50)
        assert_eq!(output_dtm[[100, 190]], 50.0);

        // Check the boundary (x = 100 in the 5m grid corresponds to x = 250 in the 2m grid)
        // A 5m pixel on the boundary might capture a mix of 100s and 50s depending on the footprint overlap.
        let boundary_val = output_dtm[[100, 99]];
        assert!(
            boundary_val > 50.0 && boundary_val <= 100.0,
            "Boundary value {} should be an average",
            boundary_val
        );
    }

    /// 4. Test Y-Axis Flip and Coordinate Math
    /// Validates the stitching grid logic without making network calls.
    #[test]
    fn test_stitching_coordinate_math() {
        let bbox = BBox {
            min_northing: 1000,
            max_northing: 1002, // 3 tiles high
            min_easting: 2000,
            max_easting: 2001, // 2 tiles wide
        };

        let tile_size = 200;
        let mut max_y_start = 0;
        let mut min_y_start = 9999;

        // Simulate the loop inside load_and_stitch
        for n in bbox.min_northing..=bbox.max_northing {
            for e in bbox.min_easting..=bbox.max_easting {
                // The critical Y-axis inversion logic
                let grid_y = (bbox.max_northing - n) as usize;
                let grid_x = (e - bbox.min_easting) as usize;

                let y_start = grid_y * tile_size;
                let x_start = grid_x * tile_size;

                if y_start > max_y_start {
                    max_y_start = y_start;
                }
                if y_start < min_y_start {
                    min_y_start = y_start;
                }

                // Assert X coordinate goes left to right
                assert!(x_start <= (2001 - 2000) * tile_size);
            }
        }

        // max_northing (1002) should be placed at y_start = 0
        // min_northing (1000) should be placed at y_start = 400
        assert_eq!(
            min_y_start, 0,
            "Highest northing should map to top of image (Y=0)"
        );
        assert_eq!(
            max_y_start, 400,
            "Lowest northing should map to bottom of image (Y=400)"
        );
    }

    /// Cross-check of the tolerant LZW fallback against the `tiff` crate.
    /// Needs two locally downloaded swissALTI3D tiles; skipped otherwise.
    #[test]
    fn test_fallback_decoder_matches_tiff_crate() {
        let base = match std::env::var("SWISSALTI_TEST_TILES") {
            Ok(v) => v,
            Err(_) => return,
        };
        for name in ["t2.tif", "t1.tif"] {
            let path = format!("{base}/{name}");
            let bytes = std::fs::read(&path).expect("tile");
            let (w, h, mine) = decode_tiled_lzw_f32(&bytes).expect("fallback decode");
            println!(
                "{name}: {w}x{h} min={:?} max={:?} mean={}",
                mine.iter().cloned().fold(f32::INFINITY, f32::min),
                mine.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                mine.iter().sum::<f32>() / mine.len() as f32
            );
            let mut dec = match Decoder::new(Cursor::new(bytes.clone())) {
                Ok(d) => d,
                Err(_) => continue,
            };
            if let Ok(DecodingResult::F32(theirs)) = dec.read_image() {
                assert_eq!(theirs.len(), mine.len());
                let diff = theirs
                    .iter()
                    .zip(mine.iter())
                    .filter(|(a, b)| (*a - *b).abs() > 1e-6)
                    .count();
                println!("{name}: {diff} differing samples vs tiff crate");
                assert_eq!(diff, 0);
            }
        }
    }

    /// 5. Test Exporting Array2 to CSV
    #[test]
    fn test_export_array2_to_csv() {
        let dir = tempdir().expect("Failed to create temp dir");
        let file_path = dir.path().join("test_export.csv");
        let path_str = file_path.to_str().unwrap();

        let mut arr = Array2::<f32>::zeros((3, 3));
        arr[[0, 0]] = 1.0;
        arr[[0, 1]] = 2.5;
        arr[[0, 2]] = 3.0;
        arr[[1, 0]] = 4.0;
        arr[[1, 1]] = 5.123;
        arr[[1, 2]] = 6.0;
        arr[[2, 0]] = 7.0;
        arr[[2, 1]] = 8.0;
        arr[[2, 2]] = 9.99;

        let res = export_array2_to_csv(&arr, path_str);
        assert!(res.is_ok());

        let contents = std::fs::read_to_string(&file_path).expect("Failed to read exported CSV");
        let parsed_lines: Vec<Vec<f32>> = contents
            .lines()
            .map(|line| {
                line.split(',')
                    .map(|val| val.parse::<f32>().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(parsed_lines.len(), 3);
        assert_eq!(parsed_lines[0], vec![1.0, 2.5, 3.0]);
        assert_eq!(parsed_lines[1], vec![4.0, 5.123, 6.0]);
        assert_eq!(parsed_lines[2], vec![7.0, 8.0, 9.99]);
    }

    /// 6. Test TileError Formatting
    #[test]
    fn test_tile_error_formatting() {
        let err_invalid = TileManagerError::InvalidData;
        assert_eq!(err_invalid.to_string(), "Missing or invalid data");

        let err_http = TileManagerError::Http("404 Not Found".to_string());
        assert_eq!(err_http.to_string(), "HTTP error: 404 Not Found");

        let err_shape = TileManagerError::InvalidShape((100, 100));
        assert_eq!(err_shape.to_string(), "Invalid tile shape: got (100, 100)");
    }

    /// 7. Test BBox and TileId clones and properties
    #[test]
    fn test_bbox_and_tile_id() {
        let bbox = BBox {
            min_northing: 1000,
            max_northing: 2000,
            min_easting: 3000,
            max_easting: 4000,
        };
        let bbox_clone = bbox.clone();
        assert_eq!(bbox_clone.min_northing, bbox.min_northing);
        assert_eq!(bbox_clone.max_northing, bbox.max_northing);
        assert_eq!(bbox_clone.min_easting, bbox.min_easting);
        assert_eq!(bbox_clone.max_easting, bbox.max_easting);

        let id = TileId {
            northing: 1100,
            easting: 2700,
        };
        let id_clone = id.clone();
        assert_eq!(id_clone, id);

        let debug_str = format!("{:?}", id);
        assert!(debug_str.contains("1100"));
        assert!(debug_str.contains("2700"));
    }

    /// 8. Test Resampling with Non-Standard Input Size
    #[test]
    fn test_resample_non_standard_input_dimensions() {
        let input_dtm = Array2::from_elem((10, 10), 42.0f32);
        let output_dtm = TileManager::resample_2m_to_5m(&input_dtm);
        assert_eq!(output_dtm.shape(), &[200, 200]);
        for &elevation in output_dtm.iter() {
            assert_eq!(elevation, 42.0);
        }
    }
}
