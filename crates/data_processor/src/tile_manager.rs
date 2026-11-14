// Austria links: https://data.bev.gv.at/download/ALS/DTM/20250915/ALS_DTM_CRS3035RES50000mN2550000E4650000.tif
// Overview: https://data.bev.gv.at/download/ALS/ALS_Kacheluebersicht.pdf
// All info: https://data.bev.gv.at/geonetwork/srv/api/records/9a4e3e4f-2cd5-4f73-af51-384beb9331d0
// 50km x 50 km tiles, 55 pieces
// doi: https://doi.org/10.48677/9a4e3e4f-2cd5-4f73-af51-384beb9331d0
//! swissALTI3D fetch and cache. Each tile's acquisition year (part of its URL) comes from
//! `data/swissalti3d_years.bin.xz`, compressed from swisstopo's link export; falls back to `decode_tiled_lzw_f32` for
//! tiles whose LZW stream lacks an end code, resamples 2 m → 5 m by area weighting, stores into a
//! Zarr array. `get_dem(bbox)` is the entry point.
use ndarray::{Array1, Array2};
use reqwest::Client;
use serde_json::json;
use std::io::Cursor;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
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
use compute_core::utils::{cell_centers, to_2d};

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

    #[error("tile_missing: tile {0:?} is not in the cache and AVALANCHERS_TILES_OFFLINE is set")]
    TileMissing(TileId),
}

/// When set to `1`, a tile missing from the cache is an error instead of a
/// download, so that many concurrent jobs only ever read a prefilled cache.
pub const OFFLINE_ENV: &str = "AVALANCHERS_TILES_OFFLINE";
/// Pause in milliseconds after each downloaded tile (default 0). Downloads
/// are sequential, so this caps the request rate of a bulk prefetch.
pub const DELAY_ENV: &str = "AVALANCHERS_TILE_DELAY_MS";
/// Retries of a request the server answered with 429 or 503.
const MAX_THROTTLE_RETRIES: u32 = 5;
/// Wait before such a retry when the server sends no `Retry-After`.
const DEFAULT_THROTTLE_WAIT: Duration = Duration::from_secs(30);

fn tiles_offline() -> bool {
    std::env::var(OFFLINE_ENV)
        .is_ok_and(|value| matches!(value.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
}

fn download_delay() -> Duration {
    std::env::var(DELAY_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(Duration::ZERO, Duration::from_millis)
}

/// Wait before retrying a throttled request: the `Retry-After` seconds if the
/// server sent them, otherwise [`DEFAULT_THROTTLE_WAIT`].
fn throttle_wait(retry_after: Option<&str>) -> Duration {
    retry_after
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(DEFAULT_THROTTLE_WAIT, Duration::from_secs)
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

/// reqwest needs a Tokio reactor, but `get_dem` is driven by whatever executor
/// the caller has: the CLI and the simulation tests use `pollster`, the Python
/// bindings Tokio. Every request therefore runs on this dedicated runtime.
fn http_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("swissalti3d-http")
            .enable_all()
            .build()
            .expect("failed to start the HTTP runtime for swissALTI3D downloads")
    })
}

/// Runs `future` on [`http_runtime`] and awaits its result through a channel,
/// which works from any executor, inside a Tokio runtime or not.
async fn on_http_runtime<F, T>(future: F) -> Result<T, TileManagerError>
where
    F: std::future::Future<Output = Result<T, reqwest::Error>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = futures::channel::oneshot::channel();
    http_runtime().spawn(async move {
        let _ = tx.send(future.await);
    });
    rx.await
        .map_err(|_| TileManagerError::Http("HTTP task was cancelled".into()))?
        .map_err(TileManagerError::from)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId {
    pub northing: u32,
    pub easting: u32,
}

pub use crate::rasterizer::BBox;

/// Acquisition year of every swissALTI3D tile (2 m, LV95 GeoTIFF) as an xz-compressed grid with
/// one byte per 1 km tile. Generated from swisstopo's link export by
/// `data/make_swissalti3d_years.py`, which also documents the layout; to refresh it, export the
/// links again from <https://www.swisstopo.admin.ch/en/height-model-swissalti3d>
/// (whole Switzerland, 2 m, GeoTIFF) and rerun the script.
const SWISSALTI3D_YEARS: &[u8] = include_bytes!("../data/swissalti3d_years.bin.xz");

/// Oldest acquisition year probed when a listed URL has gone stale.
const OLDEST_YEAR: u32 = 2017;

struct TileIndex {
    min_easting: u32,
    min_northing: u32,
    width: u32,
    height: u32,
    /// Row-major by northing, `year - 2000`, 0 = no tile.
    grid: Vec<u8>,
    newest_year: u32,
}

impl TileIndex {
    fn decode(packed: &[u8]) -> Result<Self, String> {
        use std::io::Read;
        let mut raw = Vec::new();
        xz2::read::XzDecoder::new(packed)
            .read_to_end(&mut raw)
            .map_err(|e| format!("xz: {e}"))?;
        if raw.len() < 12 || &raw[..4] != b"SA3D" {
            return Err("bad header".into());
        }
        let u16at = |o: usize| u16::from_le_bytes([raw[o], raw[o + 1]]) as u32;
        let (min_easting, min_northing, width, height) = (u16at(4), u16at(6), u16at(8), u16at(10));
        let grid = raw[12..].to_vec();
        if grid.len() != (width * height) as usize {
            return Err(format!(
                "grid has {} bytes, expected {width}x{height}",
                grid.len()
            ));
        }
        let newest_year = grid
            .iter()
            .map(|&y| 2000 + y as u32)
            .max()
            .unwrap_or(OLDEST_YEAR);
        Ok(Self {
            min_easting,
            min_northing,
            width,
            height,
            grid,
            newest_year,
        })
    }

    fn year(&self, id: TileId) -> Option<u32> {
        let x = id.easting.checked_sub(self.min_easting)?;
        let y = id.northing.checked_sub(self.min_northing)?;
        if x >= self.width || y >= self.height {
            return None;
        }
        match self.grid[(y * self.width + x) as usize] {
            0 => None,
            offset => Some(2000 + offset as u32),
        }
    }
}

fn tile_index() -> &'static TileIndex {
    static INDEX: OnceLock<TileIndex> = OnceLock::new();
    INDEX.get_or_init(|| {
        TileIndex::decode(SWISSALTI3D_YEARS).expect("embedded swissALTI3D year table is corrupt")
    })
}

fn tile_url(id: TileId, year: u32) -> String {
    format!(
        "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_{year}_{e}-{n}/swissalti3d_{year}_{e}-{n}_2_2056_5728.tif",
        e = id.easting,
        n = id.northing,
    )
}

pub struct TileManager {
    client: Client,
    // zarr_storage: Arc<FilesystemStore>,
    dtm: Array<FilesystemStore>,
    min_easting_km: u64,
    min_northing_km: u64,
    cell_size: f32,
    /// See [`OFFLINE_ENV`].
    offline: bool,
    /// See [`DELAY_ENV`].
    download_delay: Duration,
}

impl TileManager {
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Overrides [`OFFLINE_ENV`] for this manager.
    pub fn set_offline(&mut self, offline: bool) {
        self.offline = offline;
    }
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
            "Conventions": "CF-1.8",
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
            cell_size,
            offline: tiles_offline(),
            download_delay: download_delay(),
            // tile_size,
        })
    }

    /// GET request that waits and retries while the server throttles (HTTP 429
    /// or 503), up to [`MAX_THROTTLE_RETRIES`] times. Blocking sleep is fine:
    /// tiles are downloaded one after another and never concurrently.
    async fn get(&self, url: &str) -> Result<reqwest::Response, TileManagerError> {
        let mut retries = 0;
        loop {
            let response = on_http_runtime(self.client.get(url).send()).await?;
            let status = response.status();
            let throttled = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status == reqwest::StatusCode::SERVICE_UNAVAILABLE;
            if !throttled || retries == MAX_THROTTLE_RETRIES {
                return Ok(response);
            }
            retries += 1;
            let wait = throttle_wait(
                response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok()),
            );
            info!(
                "The tile server answered HTTP {status} for {url}; retry {retries} of \
                 {MAX_THROTTLE_RETRIES} in {:.1} s",
                wait.as_secs_f64()
            );
            std::thread::sleep(wait);
        }
    }

    /// Fetches the TIFF from geo.admin.ch and decodes it into a 500x500 f32 array
    async fn fetch_and_parse_tif(&self, id: TileId) -> Result<Array2<f32>, TileManagerError> {
        // swissALTI3D is flown region by region, so the acquisition year in the
        // asset name differs between tiles. The embedded year table knows it.
        let index = tile_index();
        let Some(known_year) = index.year(id) else {
            return Err(TileManagerError::Http(format!(
                "tile {:?} is outside the swissALTI3D coverage",
                id
            )));
        };

        let known_url = tile_url(id, known_year);
        let mut url = known_url.clone();
        let mut response = self.get(&url).await?;

        // A tile that has been re-flown since the export was taken is published
        // under a new year and the old asset can disappear. Only then probe the
        // other years, newest first, and ask for the export to be refreshed.
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            warn!(
                "Tile {:?} is missing at {} (HTTP 404); data/swissalti3d_years.bin.xz is \
                 probably outdated. Trying the other survey years",
                id, url
            );
            let mut found = false;
            for year in (OLDEST_YEAR..=index.newest_year + 1)
                .rev()
                .filter(|&y| y != known_year)
            {
                let candidate = tile_url(id, year);
                let r = self.get(&candidate).await?;
                if r.status().is_success() {
                    url = candidate;
                    response = r;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(TileManagerError::Http(format!(
                    "tile {:?}: no swissALTI3D asset found ({} is 404, no other year matched)",
                    id, known_url
                )));
            }
        }

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

        let bytes = on_http_runtime(response.bytes()).await?;

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
                    debug!(
                        "Tile {:?} has the TIFF sample type {:?}; retrying with the LZW decoder",
                        id,
                        std::mem::discriminant(&other)
                    );
                    None
                }
                (_, Err(e)) | (Err(e), _) => {
                    debug!(
                        "Could not decode tile {:?} (content type {}, {} bytes): {:?}; retrying \
                         with the tolerant LZW decoder",
                        id,
                        content_type,
                        bytes.len(),
                        e
                    );
                    None
                }
            },
            Err(e) => {
                debug!(
                    "Could not open tile {:?} (content type {}, {} bytes): {:?}; retrying with \
                     the tolerant LZW decoder",
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

                if self.offline {
                    return Err(TileManagerError::TileMissing(tile));
                }
                // One tile at a time, never concurrently (swisstopo fair use).
                let tif = self.fetch_and_parse_tif(tile).await?;
                let resampled = Self::resample_2m_to_5m(&tif);

                self.dtm
                    .store_chunk(&chunk_indices, resampled.as_slice().unwrap())?;
                trace!("Tile cache miss: {:?}", chunk_indices);
                if !self.download_delay.is_zero() {
                    std::thread::sleep(self.download_delay);
                }
            }
            trace!("Tile cache hit: {:?}", chunk_indices)
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
            x: cell_centers(min_e as f32, max_e as f32, cols),
            y: cell_centers(min_n as f32, max_n as f32, rows),
            cell_size: 5.0,
            bounds: Bounds {
                xmin: min_e as f32,
                xmax: max_e as f32,
                ymin: min_n as f32,
                ymax: max_n as f32,
            },
            map_factor: 1.0,
            minimum_elevation: f32::INFINITY, // Will be calculated later
            source: "Swiss map data © swisstopo\nSource: Federal Office of Topography swisstopo"
                .to_string(),
            projection: "EPSG:2056".to_string(),
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

    #[test]
    fn throttle_wait_uses_retry_after_seconds() {
        assert_eq!(throttle_wait(Some("7")), Duration::from_secs(7));
        assert_eq!(throttle_wait(Some(" 0 ")), Duration::ZERO);
        // HTTP dates and garbage fall back to the default.
        assert_eq!(
            throttle_wait(Some("Wed, 21 Oct 2015 07:28:00 GMT")),
            DEFAULT_THROTTLE_WAIT
        );
        assert_eq!(throttle_wait(None), DEFAULT_THROTTLE_WAIT);
    }

    #[test]
    fn offline_cache_miss_is_an_error() {
        let dir = tempdir().unwrap();
        let mut manager = TileManager::new(dir.path().to_str().unwrap()).unwrap();
        manager.set_offline(true);
        let bbox = BBox {
            min_northing: 1159502,
            max_northing: 1159549,
            min_easting: 2666100,
            max_easting: 2666150,
        };
        let error = pollster::block_on(manager.get_dem(&bbox)).unwrap_err();
        assert!(matches!(error, TileManagerError::TileMissing(_)));
        assert!(error.to_string().starts_with("tile_missing"));
    }

    /// The CLI and the simulation drive `get_dem` with `pollster`, i.e. outside
    /// any Tokio runtime; reqwest used to panic there with "no reactor running".
    #[test]
    fn fetch_tiles_without_tokio_runtime() {
        let dir = tempdir().expect("temp dir");
        let manager = TileManager::new(dir.path().to_str().unwrap()).expect("tile manager");
        let bbox = BBox {
            min_northing: 1159502,
            max_northing: 1159549,
            min_easting: 2666100,
            max_easting: 2666150,
        };
        let dem = pollster::block_on(manager.get_dem(&bbox)).expect("download outside Tokio");
        assert_eq!((dem.width, dem.height), (10, 10));
        assert!(dem.data1d.iter().all(|v| v.is_finite()));
    }

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
        // swissalti3d_2025_2666-1159 (the acquisition listed in the link export)
        let expected_data: Vec<f32> = vec![
            2193.9417, 2194.0571, 2193.8872, 2193.7124, 2193.6594, 2193.8335, 2194.358, 2194.6235,
            2195.0251, 2195.7737, 2191.6135, 2191.386, 2191.4685, 2191.1738, 2191.0537, 2191.278,
            2191.662, 2192.0251, 2192.648, 2193.5308, 2188.9172, 2188.7844, 2188.7593, 2188.3552,
            2188.325, 2188.556, 2188.7834, 2189.2366, 2190.1335, 2191.6125, 2186.4668, 2186.1812,
            2186.1272, 2185.8232, 2185.757, 2186.131, 2186.3052, 2186.8694, 2188.0347, 2189.9502,
            2183.8948, 2183.3433, 2183.0554, 2182.8442, 2182.8606, 2183.248, 2183.6804, 2184.5244,
            2186.2798, 2188.2834, 2181.4624, 2181.0366, 2180.444, 2180.35, 2180.3245, 2180.8633,
            2181.247, 2181.9211, 2183.9072, 2184.915, 2179.0076, 2178.318, 2177.8428, 2177.5286,
            2177.5442, 2178.143, 2178.6748, 2179.3853, 2180.0159, 2181.3088, 2176.6655, 2176.0354,
            2175.3628, 2175.1985, 2175.6504, 2176.1597, 2176.8206, 2177.5706, 2178.0662, 2178.9204,
            2174.541, 2173.509, 2173.1753, 2173.7898, 2174.2112, 2174.5476, 2175.4053, 2176.0073,
            2176.6626, 2177.45, 2171.9114, 2171.3594, 2172.1326, 2172.8123, 2173.0383, 2173.3618,
            2174.1052, 2174.6538, 2175.4678, 2176.0383,
        ];
        assert_eq!(dem.data1d, expected_data);
    }

    /// 1. Test the Zarr Cache Initialization
    ///    Ensures that the directory is created and the root Zarr group is written.
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
    ///    A perfectly flat 2m terrain should result in a perfectly flat 5m terrain.
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
    ///    Tests that the fractional bounding box logic correctly averages pixels.
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
        let id_clone = id;
        assert_eq!(id_clone, id);

        let debug_str = format!("{:?}", id);
        assert!(debug_str.contains("1100"));
        assert!(debug_str.contains("2700"));
    }

    #[test]
    fn test_tile_url() {
        let id = TileId {
            easting: 2774,
            northing: 1264,
        };
        assert_eq!(
            tile_url(id, 2026),
            "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2026_2774-1264/swissalti3d_2026_2774-1264_2_2056_5728.tif"
        );
    }

    #[test]
    fn test_embedded_tile_index() {
        let index = tile_index();
        // The link export this table was generated from had 43 650 tiles.
        assert_eq!(index.grid.iter().filter(|&&y| y != 0).count(), 43_650);
        assert_eq!(index.newest_year, 2026);
        let year = |easting, northing| index.year(TileId { easting, northing });
        assert_eq!(year(2501, 1120), Some(2019));
        assert_eq!(year(2666, 1159), Some(2025));
        assert_eq!(year(2774, 1266), Some(2026));
        // Inside the grid's bounding box but not a tile, and outside it entirely.
        assert_eq!(year(2485, 1064), None);
        assert_eq!(year(2480, 1060), None);
        assert_eq!(year(9999, 9999), None);
    }

    #[test]
    fn test_corrupt_tile_index_is_rejected() {
        assert!(TileIndex::decode(b"not xz").is_err());
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
