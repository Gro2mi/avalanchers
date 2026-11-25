//! Orthophotos draped over a DEM: SWISSIMAGE (swisstopo) for Switzerland and the
//! basemap.at orthofoto for Austria.
//!
//! Both providers serve Web Mercator XYZ tiles, so a DEM in any of the supported
//! national grids ([`DemCrs`]) is mapped to WGS84 with `proj4rs`, then to tile
//! pixels, and the tiles are resampled into an RGBA image aligned with the DEM
//! bounds. Tiles are cached on disk and honour the same offline / throttling
//! environment as the swissALTI3D [`crate::tile_manager`].

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::time::Duration;

use futures::StreamExt;
use proj4rs::proj::Proj;

use compute_core::dem::Bounds;

use crate::tile_manager::{
    MAX_THROTTLE_RETRIES, download_delay, http_runtime, throttle_wait, tiles_offline,
};

#[allow(unused_imports)]
use tracing::{debug, info, trace, warn};

/// Web Mercator tile size in pixels.
const TILE_SIZE: u32 = 256;
/// Downloads in flight at once.
const CONCURRENT_DOWNLOADS: usize = 8;
/// Coarsest zoom that still makes sense for a DEM-sized area.
const MIN_ZOOM: u32 = 8;
/// Upper bound on the tiles of one orthophoto; a coarser zoom is used beyond.
const MAX_TILES: usize = 1024;
/// Output pixels between exact `proj4rs` transforms; the pixel positions in
/// between are interpolated bilinearly (far below a pixel of error).
const PROJ_GRID_STEP: u32 = 16;
/// Default directory of the tile cache, relative to the working directory.
pub const DEFAULT_CACHE_DIR: &str = "ortho_cache";

/// Longitude/latitude windows used to confirm a guessed CRS: Switzerland (with
/// Liechtenstein), Austria, and both together.
const CH_LON: (f64, f64) = (5.9, 10.5);
const CH_LAT: (f64, f64) = (45.8, 47.85);
const AT_LON: (f64, f64) = (9.5, 17.2);
const AT_LAT: (f64, f64) = (46.35, 49.05);
const ALPS_LON: (f64, f64) = (5.8, 17.3);
const ALPS_LAT: (f64, f64) = (45.7, 49.1);

#[derive(thiserror::Error, Debug)]
pub enum OrthoError {
    #[error("projection error: {0}")]
    Proj(#[from] proj4rs::errors::Error),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no orthophoto tiles available for this area ({0})")]
    NoTiles(String),
    #[error("the DEM bounds are empty")]
    EmptyBounds,
}

/// Coordinate reference systems of DEMs in Switzerland and Austria.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DemCrs {
    /// CH1903+ / LV95, EPSG:2056.
    Lv95,
    /// CH1903 / LV03, EPSG:21781.
    Lv03,
    /// MGI / Austria GK West, EPSG:31254.
    MgiGkWest,
    /// MGI / Austria GK Central, EPSG:31255.
    MgiGkCentral,
    /// MGI / Austria GK East, EPSG:31256.
    MgiGkEast,
    /// MGI / Austria Lambert, EPSG:31287.
    MgiLambert,
    /// ETRS89 / LAEA Europe, EPSG:3035 (BEV ALS DTM).
    EtrsLaea,
    /// WGS 84 / UTM zone 32N, EPSG:32632 (also ETRS89 / UTM 32N, EPSG:25832).
    Utm32N,
    /// WGS 84 / UTM zone 33N, EPSG:32633 (also ETRS89 / UTM 33N, EPSG:25833).
    Utm33N,
}

const MGI_TOWGS84: &str = "+towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232";
const CH1903_TOWGS84: &str = "+towgs84=674.374,15.056,405.346,0,0,0,0";

impl DemCrs {
    pub const ALL: [DemCrs; 9] = [
        DemCrs::Lv95,
        DemCrs::Lv03,
        DemCrs::MgiGkWest,
        DemCrs::MgiGkCentral,
        DemCrs::MgiGkEast,
        DemCrs::MgiLambert,
        DemCrs::EtrsLaea,
        DemCrs::Utm32N,
        DemCrs::Utm33N,
    ];

    pub fn epsg(self) -> u32 {
        match self {
            DemCrs::Lv95 => 2056,
            DemCrs::Lv03 => 21781,
            DemCrs::MgiGkWest => 31254,
            DemCrs::MgiGkCentral => 31255,
            DemCrs::MgiGkEast => 31256,
            DemCrs::MgiLambert => 31287,
            DemCrs::EtrsLaea => 3035,
            DemCrs::Utm32N => 32632,
            DemCrs::Utm33N => 32633,
        }
    }

    pub fn from_epsg(code: u32) -> Option<Self> {
        match code {
            25832 => Some(DemCrs::Utm32N),
            25833 => Some(DemCrs::Utm33N),
            _ => Self::ALL.into_iter().find(|crs| crs.epsg() == code),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DemCrs::Lv95 => "CH1903+ / LV95 (EPSG:2056)",
            DemCrs::Lv03 => "CH1903 / LV03 (EPSG:21781)",
            DemCrs::MgiGkWest => "MGI / Austria GK West (EPSG:31254)",
            DemCrs::MgiGkCentral => "MGI / Austria GK Central (EPSG:31255)",
            DemCrs::MgiGkEast => "MGI / Austria GK East (EPSG:31256)",
            DemCrs::MgiLambert => "MGI / Austria Lambert (EPSG:31287)",
            DemCrs::EtrsLaea => "ETRS89 / LAEA Europe (EPSG:3035)",
            DemCrs::Utm32N => "UTM zone 32N (EPSG:32632)",
            DemCrs::Utm33N => "UTM zone 33N (EPSG:32633)",
        }
    }

    pub fn proj4_string(self) -> String {
        let gk = |lon_0: &str| {
            format!(
                "+proj=tmerc +lat_0=0 +lon_0={lon_0} +k=1 +x_0=0 +y_0=-5000000 +ellps=bessel \
                 {MGI_TOWGS84} +units=m +no_defs"
            )
        };
        match self {
            DemCrs::Lv95 => format!(
                "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 \
                 +x_0=2600000 +y_0=1200000 +ellps=bessel {CH1903_TOWGS84} +units=m +no_defs"
            ),
            DemCrs::Lv03 => format!(
                "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 \
                 +x_0=600000 +y_0=200000 +ellps=bessel {CH1903_TOWGS84} +units=m +no_defs"
            ),
            DemCrs::MgiGkWest => gk("10.3333333333333"),
            DemCrs::MgiGkCentral => gk("13.3333333333333"),
            DemCrs::MgiGkEast => gk("16.3333333333333"),
            DemCrs::MgiLambert => format!(
                "+proj=lcc +lat_0=47.5 +lon_0=13.3333333333333 +lat_1=49 +lat_2=46 \
                 +x_0=400000 +y_0=400000 +ellps=bessel {MGI_TOWGS84} +units=m +no_defs"
            ),
            DemCrs::EtrsLaea => "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 \
                 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
                .to_string(),
            DemCrs::Utm32N => "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs".to_string(),
            DemCrs::Utm33N => "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs".to_string(),
        }
    }

    /// Longitude/latitude window the CRS is used in.
    fn region(self) -> ((f64, f64), (f64, f64)) {
        match self {
            DemCrs::Lv95 | DemCrs::Lv03 => (CH_LON, CH_LAT),
            DemCrs::MgiGkWest | DemCrs::MgiGkCentral | DemCrs::MgiGkEast | DemCrs::MgiLambert => {
                (AT_LON, AT_LAT)
            }
            DemCrs::EtrsLaea | DemCrs::Utm32N | DemCrs::Utm33N => (ALPS_LON, ALPS_LAT),
        }
    }

    /// The orthophoto service of the country the CRS belongs to; `None` for
    /// the European grids.
    pub fn country_source(self) -> Option<OrthoSource> {
        match self {
            DemCrs::Lv95 | DemCrs::Lv03 => Some(OrthoSource::SwissImage),
            DemCrs::MgiGkWest | DemCrs::MgiGkCentral | DemCrs::MgiGkEast | DemCrs::MgiLambert => {
                Some(OrthoSource::BasemapAt)
            }
            DemCrs::EtrsLaea | DemCrs::Utm32N | DemCrs::Utm33N => None,
        }
    }

    /// Plausible `(x, y)` window of the CRS's coordinates over Switzerland and
    /// Austria; only a pre-filter, [`detect_crs`] confirms the location.
    fn window(self) -> ((f64, f64), (f64, f64)) {
        match self {
            DemCrs::Lv95 => ((2_400_000.0, 2_900_000.0), (1_000_000.0, 1_350_000.0)),
            DemCrs::Lv03 => ((400_000.0, 900_000.0), (0.0, 350_000.0)),
            DemCrs::MgiGkWest | DemCrs::MgiGkCentral | DemCrs::MgiGkEast => {
                ((-200_000.0, 200_000.0), (100_000.0, 500_000.0))
            }
            DemCrs::MgiLambert => ((100_000.0, 700_000.0), (250_000.0, 600_000.0)),
            DemCrs::EtrsLaea => ((3_900_000.0, 5_000_000.0), (2_400_000.0, 3_000_000.0)),
            DemCrs::Utm32N | DemCrs::Utm33N => ((0.0, 1_000_000.0), (5_000_000.0, 5_500_000.0)),
        }
    }
}

/// Order in which [`detect_crs`] tries the CRSs: unambiguous windows first, then
/// the Austrian grids in the order of how often they hold Alpine DEMs.
const DETECTION_ORDER: [DemCrs; 9] = [
    DemCrs::Lv95,
    DemCrs::EtrsLaea,
    DemCrs::Utm32N,
    DemCrs::Utm33N,
    DemCrs::Lv03,
    DemCrs::MgiLambert,
    DemCrs::MgiGkWest,
    DemCrs::MgiGkCentral,
    DemCrs::MgiGkEast,
];

/// Converts map coordinates in one CRS to WGS84 longitude/latitude in degrees.
pub struct ToWgs84 {
    src: Proj,
    dst: Proj,
}

impl ToWgs84 {
    pub fn new(crs: DemCrs) -> Result<Self, OrthoError> {
        Ok(Self {
            src: Proj::from_proj_string(&crs.proj4_string())?,
            dst: Proj::from_proj_string("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")?,
        })
    }

    /// `(lon, lat)` in degrees.
    pub fn lon_lat(&self, x: f64, y: f64) -> Result<(f64, f64), OrthoError> {
        let mut point = (x, y, 0.0);
        proj4rs::transform::transform(&self.src, &self.dst, &mut point)?;
        Ok((point.0.to_degrees(), point.1.to_degrees()))
    }
}

fn inside(value: f64, (lo, hi): (f64, f64)) -> bool {
    (lo..=hi).contains(&value)
}

fn bounds_center(bounds: &Bounds) -> (f64, f64) {
    (
        0.5 * (bounds.xmin as f64 + bounds.xmax as f64),
        0.5 * (bounds.ymin as f64 + bounds.ymax as f64),
    )
}

/// The CRS of a DEM: its `projection` if that is a supported `EPSG:<code>`,
/// otherwise the first CRS whose coordinate window holds the bounds and that
/// places the centre in its own country (Switzerland or Austria). `None` for synthetic or unsupported
/// DEMs.
pub fn detect_crs(bounds: &Bounds, projection: &str) -> Option<DemCrs> {
    if let Some(crs) = projection
        .trim()
        .strip_prefix("EPSG:")
        .and_then(|code| code.parse().ok())
        .and_then(DemCrs::from_epsg)
    {
        return Some(crs);
    }
    let (x, y) = bounds_center(bounds);
    DETECTION_ORDER.into_iter().find(|crs| {
        let (wx, wy) = crs.window();
        let (region_lon, region_lat) = crs.region();
        inside(x, wx)
            && inside(y, wy)
            && ToWgs84::new(*crs)
                .and_then(|t| t.lon_lat(x, y))
                .is_ok_and(|(lon, lat)| inside(lon, region_lon) && inside(lat, region_lat))
    })
}

/// WGS84 longitude/latitude of the DEM centre, in degrees.
pub fn center_lon_lat(bounds: &Bounds, crs: DemCrs) -> Result<(f64, f64), OrthoError> {
    let (x, y) = bounds_center(bounds);
    ToWgs84::new(crs)?.lon_lat(x, y)
}

/// Orthophoto tile services.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrthoSource {
    /// swisstopo SWISSIMAGE, Switzerland.
    SwissImage,
    /// basemap.at orthofoto, Austria.
    BasemapAt,
}

impl OrthoSource {
    pub const ALL: [OrthoSource; 2] = [OrthoSource::SwissImage, OrthoSource::BasemapAt];

    pub fn url(self, z: u32, x: u32, y: u32) -> String {
        match self {
            OrthoSource::SwissImage => format!(
                "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg"
            ),
            // basemap.at orders the tile row before the column
            OrthoSource::BasemapAt => format!(
                "https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg"
            ),
        }
    }

    pub fn max_zoom(self) -> u32 {
        19
    }

    /// Finest useful ground resolution in metres per pixel.
    pub fn native_resolution(self) -> f64 {
        match self {
            OrthoSource::SwissImage => 0.25,
            OrthoSource::BasemapAt => 0.3,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            OrthoSource::SwissImage => "SWISSIMAGE (CH)",
            OrthoSource::BasemapAt => "basemap.at (AT)",
        }
    }

    pub fn attribution(self) -> &'static str {
        match self {
            OrthoSource::SwissImage => "© swisstopo",
            OrthoSource::BasemapAt => "© basemap.at",
        }
    }

    fn cache_name(self) -> &'static str {
        match self {
            OrthoSource::SwissImage => "swissimage",
            OrthoSource::BasemapAt => "basemap_at",
        }
    }

    /// The service of the CRS's country; for the European grids SWISSIMAGE
    /// inside Switzerland, basemap.at elsewhere.
    pub fn default_for(crs: DemCrs, lon: f64, lat: f64) -> Self {
        if let Some(source) = crs.country_source() {
            return source;
        }
        // Vorarlberg and Tyrol reach into the Swiss window
        let austria = lon > 9.53 && lat > 46.85;
        if inside(lon, CH_LON) && inside(lat, CH_LAT) && !austria {
            OrthoSource::SwissImage
        } else {
            OrthoSource::BasemapAt
        }
    }
}

/// An RGBA8 (sRGB) image covering the DEM bounds edge to edge. Row 0 is the
/// south edge, matching the DEM row order; fully transparent where no tile
/// covers the area.
#[derive(Clone, Debug)]
pub struct Orthophoto {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Global Web Mercator pixel position of a WGS84 point at `zoom`.
fn mercator_pixel(lon: f64, lat: f64, zoom: u32) -> (f64, f64) {
    let size = (TILE_SIZE as f64) * (1u64 << zoom) as f64;
    let lat = lat.clamp(-85.051_128, 85.051_128).to_radians();
    let x = (lon + 180.0) / 360.0 * size;
    let y = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / PI) / 2.0 * size;
    (x, y)
}

/// Ground metres per Web Mercator pixel at `zoom` and latitude `lat`.
fn ground_resolution(lat: f64, zoom: u32) -> f64 {
    2.0 * PI * 6_378_137.0 * lat.to_radians().cos() / (TILE_SIZE as f64 * (1u64 << zoom) as f64)
}

/// Fetches the orthophoto of `bounds` (DEM cell edges in `crs`) from `source`,
/// at most `max_dim` pixels along either side. Tiles are cached in `cache_dir`;
/// with [`crate::tile_manager::OFFLINE_ENV`] set only cached tiles are used.
pub async fn fetch_orthophoto(
    bounds: &Bounds,
    crs: DemCrs,
    source: OrthoSource,
    max_dim: u32,
    cache_dir: &Path,
) -> Result<Orthophoto, OrthoError> {
    fetch_orthophoto_with(bounds, crs, source, max_dim, cache_dir, tiles_offline()).await
}

async fn fetch_orthophoto_with(
    bounds: &Bounds,
    crs: DemCrs,
    source: OrthoSource,
    max_dim: u32,
    cache_dir: &Path,
    offline: bool,
) -> Result<Orthophoto, OrthoError> {
    let (xmin, xmax) = (bounds.xmin as f64, bounds.xmax as f64);
    let (ymin, ymax) = (bounds.ymin as f64, bounds.ymax as f64);
    let (extent_x, extent_y) = (xmax - xmin, ymax - ymin);
    if !(extent_x > 0.0 && extent_y > 0.0) || max_dim == 0 {
        return Err(OrthoError::EmptyBounds);
    }
    let to_wgs84 = ToWgs84::new(crs)?;
    let (_, center_lat) = to_wgs84.lon_lat(0.5 * (xmin + xmax), 0.5 * (ymin + ymax))?;

    // output resolution: the source's native one, coarser if the image would be too big
    let target_res = source
        .native_resolution()
        .max(extent_x.max(extent_y) / max_dim as f64);
    let mut zoom = (ground_resolution(center_lat, 0) / target_res)
        .log2()
        .ceil()
        .clamp(MIN_ZOOM as f64, source.max_zoom() as f64) as u32;
    let out_res = target_res.max(ground_resolution(center_lat, zoom));
    let width = ((extent_x / out_res).ceil() as u32).clamp(1, max_dim);
    let height = ((extent_y / out_res).ceil() as u32).clamp(1, max_dim);

    // WGS84 position of a coarse grid of output pixel centres; the rest is interpolated
    let grid_w = width.div_ceil(PROJ_GRID_STEP) + 1;
    let grid_h = height.div_ceil(PROJ_GRID_STEP) + 1;
    let mut lon_lat = Vec::with_capacity((grid_w * grid_h) as usize);
    for gy in 0..grid_h {
        let row = (gy * PROJ_GRID_STEP).min(height - 1);
        let y = ymin + (row as f64 + 0.5) * extent_y / height as f64;
        for gx in 0..grid_w {
            let col = (gx * PROJ_GRID_STEP).min(width - 1);
            let x = xmin + (col as f64 + 0.5) * extent_x / width as f64;
            lon_lat.push(to_wgs84.lon_lat(x, y)?);
        }
    }

    // finest zoom whose tile range stays below MAX_TILES
    let tile_range = |zoom: u32| {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(lon, lat) in &lon_lat {
            let (px, py) = mercator_pixel(lon, lat, zoom);
            (x0, y0, x1, y1) = (x0.min(px), y0.min(py), x1.max(px), y1.max(py));
        }
        let tile = |p: f64| (p / TILE_SIZE as f64).floor().max(0.0) as u32;
        (tile(x0), tile(y0), tile(x1), tile(y1))
    };
    let (mut tx0, mut ty0, mut tx1, mut ty1) = tile_range(zoom);
    while zoom > MIN_ZOOM && ((tx1 - tx0 + 1) * (ty1 - ty0 + 1)) as usize > MAX_TILES {
        zoom -= 1;
        (tx0, ty0, tx1, ty1) = tile_range(zoom);
    }
    let (tiles_x, tiles_y) = (tx1 - tx0 + 1, ty1 - ty0 + 1);
    info!(
        "Loading the {} orthophoto: {} tiles at zoom {zoom} for {width} x {height} pixels",
        source.label(),
        tiles_x * tiles_y
    );

    let started = std::time::Instant::now();
    let tiles = download_tiles(source, zoom, (tx0, ty0, tx1, ty1), cache_dir, offline).await?;
    let tiles_loaded = started.elapsed();
    if tiles.iter().all(Option::is_none) {
        return Err(OrthoError::NoTiles(format!(
            "{} has no tiles here{}",
            source.label(),
            if offline {
                " or they are not cached"
            } else {
                ""
            }
        )));
    }

    // mosaic sampler: bilinear over the decoded tiles, transparent where missing
    let texel = |px: i64, py: i64| -> [f32; 4] {
        let (tx, ty) = (
            px.div_euclid(TILE_SIZE as i64),
            py.div_euclid(TILE_SIZE as i64),
        );
        let (ix, iy) = (tx - tx0 as i64, ty - ty0 as i64);
        if ix < 0 || iy < 0 || ix >= tiles_x as i64 || iy >= tiles_y as i64 {
            return [0.0; 4];
        }
        match &tiles[(iy * tiles_x as i64 + ix) as usize] {
            Some(tile) => {
                let p = tile.get_pixel(
                    px.rem_euclid(TILE_SIZE as i64) as u32,
                    py.rem_euclid(TILE_SIZE as i64) as u32,
                );
                p.0.map(|c| c as f32)
            }
            None => [0.0; 4],
        }
    };
    let sample = |px: f64, py: f64| -> [u8; 4] {
        let (fx, fy) = (px - 0.5, py - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (wx, wy) = ((fx - x0) as f32, (fy - y0) as f32);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let [a, b, c, d] = [
            texel(x0, y0),
            texel(x0 + 1, y0),
            texel(x0, y0 + 1),
            texel(x0 + 1, y0 + 1),
        ];
        std::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * wx;
            let bottom = c[i] + (d[i] - c[i]) * wx;
            (top + (bottom - top) * wy).round().clamp(0.0, 255.0) as u8
        })
    };

    let pixel_grid: Vec<(f64, f64)> = lon_lat
        .iter()
        .map(|&(lon, lat)| mercator_pixel(lon, lat, zoom))
        .collect();
    let fill_row = |row: u32, out: &mut [u8]| {
        let gy = (row / PROJ_GRID_STEP).min(grid_h - 2);
        let row0 = (gy * PROJ_GRID_STEP).min(height - 1);
        let row1 = ((gy + 1) * PROJ_GRID_STEP).min(height - 1);
        let fy = if row1 > row0 {
            (row - row0) as f64 / (row1 - row0) as f64
        } else {
            0.0
        };
        for col in 0..width {
            let gx = (col / PROJ_GRID_STEP).min(grid_w - 2);
            let col0 = (gx * PROJ_GRID_STEP).min(width - 1);
            let col1 = ((gx + 1) * PROJ_GRID_STEP).min(width - 1);
            let fx = if col1 > col0 {
                (col - col0) as f64 / (col1 - col0) as f64
            } else {
                0.0
            };
            let at = |gx: u32, gy: u32| pixel_grid[(gy * grid_w + gx) as usize];
            let lerp = |a: (f64, f64), b: (f64, f64), t: f64| {
                (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
            };
            let top = lerp(at(gx, gy), at(gx + 1, gy), fx);
            let bottom = lerp(at(gx, gy + 1), at(gx + 1, gy + 1), fx);
            let (px, py) = lerp(top, bottom, fy);
            let offset = (col * 4) as usize;
            out[offset..offset + 4].copy_from_slice(&sample(px, py));
        }
    };
    // rows in parallel, one contiguous band per core
    let row_bytes = (width * 4) as usize;
    let band_rows = (height as usize).div_ceil(worker_threads()).max(1);
    let mut rgba = vec![0u8; row_bytes * height as usize];
    let fill_row = &fill_row;
    std::thread::scope(|scope| {
        for (band, rows) in rgba.chunks_mut(band_rows * row_bytes).enumerate() {
            scope.spawn(move || {
                for (i, out) in rows.chunks_mut(row_bytes).enumerate() {
                    fill_row((band * band_rows + i) as u32, out);
                }
            });
        }
    });

    debug!(
        "Orthophoto ready: tiles loaded in {:.2} s, resampled in {:.2} s",
        tiles_loaded.as_secs_f64(),
        (started.elapsed() - tiles_loaded).as_secs_f64()
    );
    Ok(Orthophoto {
        width,
        height,
        rgba,
    })
}

/// Decoded tiles of the range, row-major; `None` where the service has none.
async fn download_tiles(
    source: OrthoSource,
    zoom: u32,
    (tx0, ty0, tx1, ty1): (u32, u32, u32, u32),
    cache_dir: &Path,
    offline: bool,
) -> Result<Vec<Option<image::RgbaImage>>, OrthoError> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("avalanchers/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| OrthoError::Http(e.to_string()))?;
    let delay = download_delay();
    let jobs = (ty0..=ty1)
        .flat_map(|y| (tx0..=tx1).map(move |x| (x, y)))
        .map(|(x, y)| {
            let client = client.clone();
            let path = cache_dir
                .join(source.cache_name())
                .join(zoom.to_string())
                .join(x.to_string())
                .join(format!("{y}.jpeg"));
            async move { load_tile(client, source, zoom, x, y, path, offline, delay).await }
        });
    let results: Vec<Result<Option<EncodedTile>, OrthoError>> = futures::stream::iter(jobs)
        .buffered(CONCURRENT_DOWNLOADS)
        .collect()
        .await;
    let encoded = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    // decoding dominates once the tiles are cached; spread it over the cores
    Ok(parallel_map(&encoded, |tile| {
        tile.as_ref().and_then(decode_tile)
    }))
}

/// A tile's cache path and its encoded (JPEG) bytes.
type EncodedTile = (PathBuf, Vec<u8>);

/// The encoded tile and its cache path, from the cache, else from the service;
/// `None` if the service has no tile there (or it is not cached while offline).
#[allow(clippy::too_many_arguments)]
async fn load_tile(
    client: reqwest::Client,
    source: OrthoSource,
    zoom: u32,
    x: u32,
    y: u32,
    path: PathBuf,
    offline: bool,
    delay: Duration,
) -> Result<Option<EncodedTile>, OrthoError> {
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(_) if offline => return Ok(None),
        Err(_) => {
            let url = source.url(zoom, x, y);
            let Some(bytes) = download(client, url.clone()).await? else {
                return Ok(None);
            };
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&path, &bytes)?;
            if !delay.is_zero() {
                std::thread::sleep(delay);
            }
            bytes
        }
    };
    Ok(Some((path, bytes)))
}

/// A decoded tile; undecodable tiles count as missing.
fn decode_tile((path, bytes): &EncodedTile) -> Option<image::RgbaImage> {
    match image::load_from_memory(bytes) {
        Ok(image) => Some(image.into_rgba8()),
        Err(e) => {
            warn!(
                "Could not decode the orthophoto tile {}: {e}",
                path.display()
            );
            None
        }
    }
}

fn worker_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// `f` over `items` on all cores, in order.
fn parallel_map<T: Sync, U: Send>(items: &[T], f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    let chunk = items.len().div_ceil(worker_threads()).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|chunk| scope.spawn(move || chunk.iter().map(f).collect::<Vec<U>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("orthophoto worker panicked"))
            .collect()
    })
}

/// GET on the shared HTTP runtime, retrying while the server throttles.
/// `None` for a 404 / 204 (no imagery there).
async fn download(client: reqwest::Client, url: String) -> Result<Option<Vec<u8>>, OrthoError> {
    let (tx, rx) = futures::channel::oneshot::channel();
    http_runtime().spawn(async move {
        let result = async {
            let mut retries = 0;
            loop {
                trace!("Downloading {url}");
                let response = client.get(&url).send().await.map_err(|e| e.to_string())?;
                let status = response.status();
                let throttled = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status == reqwest::StatusCode::SERVICE_UNAVAILABLE;
                if throttled && retries < MAX_THROTTLE_RETRIES {
                    retries += 1;
                    let wait = throttle_wait(
                        response
                            .headers()
                            .get(reqwest::header::RETRY_AFTER)
                            .and_then(|value| value.to_str().ok()),
                    );
                    info!(
                        "The orthophoto server answered HTTP {status} for {url}; retry \
                         {retries} of {MAX_THROTTLE_RETRIES} in {:.1} s",
                        wait.as_secs_f64()
                    );
                    tokio::time::sleep(wait).await;
                    continue;
                }
                if status == reqwest::StatusCode::NOT_FOUND
                    || status == reqwest::StatusCode::NO_CONTENT
                {
                    return Ok(None);
                }
                if !status.is_success() {
                    return Err(format!("HTTP {status} for {url}"));
                }
                let bytes = response.bytes().await.map_err(|e| e.to_string())?;
                return Ok(Some(bytes.to_vec()));
            }
        }
        .await;
        let _ = tx.send(result);
    });
    rx.await
        .map_err(|_| OrthoError::Http("HTTP task was cancelled".into()))?
        .map_err(OrthoError::Http)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Bounds {
        Bounds {
            xmin,
            xmax,
            ymin,
            ymax,
        }
    }

    /// Distance in metres between two WGS84 points (equirectangular, fine for metres).
    fn distance_m((lon1, lat1): (f64, f64), (lon2, lat2): (f64, f64)) -> f64 {
        let r = 6_371_000.0;
        let dx = (lon2 - lon1).to_radians() * r * lat1.to_radians().cos();
        let dy = (lat2 - lat1).to_radians() * r;
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn lv95_reference_point_converts_to_wgs84() {
        // swisstopo reference: Bern old observatory, origin of LV95
        let lon_lat = ToWgs84::new(DemCrs::Lv95)
            .unwrap()
            .lon_lat(2_600_000.0, 1_200_000.0)
            .unwrap();
        // the datum shift moves the origin by a few metres from the Bessel value
        let expected = (7.438_632_5, 46.951_082_8);
        assert!(distance_m(lon_lat, expected) < 5.0, "{lon_lat:?}");
    }

    #[test]
    fn lv03_and_lv95_agree() {
        let lv95 = ToWgs84::new(DemCrs::Lv95)
            .unwrap()
            .lon_lat(2_700_000.0, 1_100_000.0)
            .unwrap();
        let lv03 = ToWgs84::new(DemCrs::Lv03)
            .unwrap()
            .lon_lat(700_000.0, 100_000.0)
            .unwrap();
        assert!(distance_m(lv95, lv03) < 0.01, "{lv95:?} {lv03:?}");
    }

    #[test]
    fn austrian_grids_place_innsbruck() {
        // a point in western Innsbruck, about 11.386 E 47.270 N; only a coarse
        // check that the grid parameters are right, a wrong meridian or datum
        // is off by many kilometres or hundreds of metres
        let expected = (11.3858, 47.2698);
        let gk = ToWgs84::new(DemCrs::MgiGkWest)
            .unwrap()
            .lon_lat(79_670.0, 237_310.0)
            .unwrap();
        assert!(distance_m(gk, expected) < 50.0, "{gk:?}");
        // the same point in Lambert must agree with GK West to well below a pixel
        let mut lambert = (0.0, 0.0);
        let lcc = Proj::from_proj_string(&DemCrs::MgiLambert.proj4_string()).unwrap();
        let gkw = Proj::from_proj_string(&DemCrs::MgiGkWest.proj4_string()).unwrap();
        let mut point = (79_670.0, 237_310.0, 0.0);
        proj4rs::transform::transform(&gkw, &lcc, &mut point).unwrap();
        lambert.0 = point.0;
        lambert.1 = point.1;
        let back = ToWgs84::new(DemCrs::MgiLambert)
            .unwrap()
            .lon_lat(lambert.0, lambert.1)
            .unwrap();
        assert!(distance_m(gk, back) < 0.01, "{gk:?} {back:?}");
    }

    #[test]
    fn detects_the_crs_from_the_coordinates() {
        // swissALTI3D (LV95) around Davos
        let davos = bounds(2_780_000.0, 1_180_000.0, 2_785_000.0, 1_185_000.0);
        assert_eq!(detect_crs(&davos, "unknown"), Some(DemCrs::Lv95));
        assert_eq!(detect_crs(&davos, "EPSG:2056"), Some(DemCrs::Lv95));
        // AvaFrame avaArzlerUni (GK West, Innsbruck)
        let arzl = bounds(79_200.0, 237_860.0, 83_740.0, 243_030.0);
        assert_eq!(detect_crs(&arzl, "unknown"), Some(DemCrs::MgiGkWest));
        // AvaFrame avaGar (Lambert, Tyrol)
        let gar = bounds(247_323.2, 350_705.9, 249_913.2, 352_630.9);
        assert_eq!(detect_crs(&gar, ""), Some(DemCrs::MgiLambert));
        // BEV ALS DTM tile (LAEA)
        let laea = bounds(4_400_000.0, 2_650_000.0, 4_405_000.0, 2_655_000.0);
        assert_eq!(detect_crs(&laea, "EPSG:0"), Some(DemCrs::EtrsLaea));
        // synthetic test DEMs have no location
        let synthetic = bounds(0.0, 0.0, 1_000.0, 1_000.0);
        assert_eq!(detect_crs(&synthetic, "unknown"), None);
        // an explicit projection wins over the window guess
        assert_eq!(detect_crs(&arzl, "EPSG:31255"), Some(DemCrs::MgiGkCentral));
    }

    #[test]
    fn picks_the_source_by_country() {
        // Davos and Innsbruck in UTM 32N
        let utm = DemCrs::Utm32N;
        assert_eq!(
            OrthoSource::default_for(utm, 9.83, 46.8),
            OrthoSource::SwissImage
        );
        assert_eq!(
            OrthoSource::default_for(utm, 11.39, 47.27),
            OrthoSource::BasemapAt
        );
        assert_eq!(
            OrthoSource::default_for(utm, 7.45, 46.95),
            OrthoSource::SwissImage
        );
        // the national grids pick their own service
        assert_eq!(
            OrthoSource::default_for(DemCrs::Lv95, 0.0, 0.0),
            OrthoSource::SwissImage
        );
        assert_eq!(
            OrthoSource::default_for(DemCrs::MgiLambert, 0.0, 0.0),
            OrthoSource::BasemapAt
        );
    }

    #[test]
    fn tile_urls_follow_each_service_layout() {
        assert_eq!(
            OrthoSource::SwissImage.url(17, 68_000, 46_000),
            "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/17/68000/46000.jpeg"
        );
        assert_eq!(
            OrthoSource::BasemapAt.url(17, 69_682, 45_770),
            "https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/17/45770/69682.jpeg"
        );
    }

    #[test]
    fn mercator_pixel_matches_the_tile_scheme() {
        // Innsbruck lies in the OSM slippy map tile x 8710, y 5744 at zoom 14
        let (px, py) = mercator_pixel(11.3936, 47.2665, 14);
        assert_eq!(((px / 256.0) as u32, (py / 256.0) as u32), (8710, 5744));
    }

    #[tokio::test]
    async fn offline_without_cache_reports_no_tiles() {
        let dir = tempfile::tempdir().unwrap();
        let arzl = bounds(79_200.0, 237_860.0, 80_000.0, 238_500.0);
        let source = OrthoSource::BasemapAt;
        let result =
            fetch_orthophoto_with(&arzl, DemCrs::MgiGkWest, source, 256, dir.path(), true).await;
        assert!(matches!(result, Err(OrthoError::NoTiles(_))), "{result:?}");
    }

    #[tokio::test]
    async fn cached_tiles_are_resampled_onto_the_dem() {
        let dir = tempfile::tempdir().unwrap();
        let arzl = bounds(79_200.0, 237_860.0, 79_400.0, 238_000.0);
        let source = OrthoSource::BasemapAt;
        // fill the whole tile range of the area with solid red tiles; the zoom is
        // the one fetch_orthophoto picks for a 64 px image of 200 m (3.1 m/px)
        let (lon, lat) = center_lon_lat(&arzl, DemCrs::MgiGkWest).unwrap();
        let zoom = (ground_resolution(lat, 0) / (200.0 / 64.0)).log2().ceil() as u32;
        let (px, py) = mercator_pixel(lon, lat, zoom);
        let (cx, cy) = ((px / 256.0) as u32, (py / 256.0) as u32);
        let red = image::RgbImage::from_pixel(256, 256, image::Rgb([255, 0, 0]));
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                let path = dir
                    .path()
                    .join(source.cache_name())
                    .join(zoom.to_string())
                    .join(x.to_string());
                std::fs::create_dir_all(&path).unwrap();
                red.save_with_format(path.join(format!("{y}.jpeg")), image::ImageFormat::Jpeg)
                    .unwrap();
            }
        }
        let ortho = fetch_orthophoto_with(&arzl, DemCrs::MgiGkWest, source, 64, dir.path(), true)
            .await
            .unwrap();
        assert_eq!((ortho.width, ortho.height), (64, 45));
        let center = ((ortho.height / 2 * ortho.width + ortho.width / 2) * 4) as usize;
        let pixel = &ortho.rgba[center..center + 4];
        assert!(
            pixel[0] > 240 && pixel[1] < 20 && pixel[3] == 255,
            "{pixel:?}"
        );
    }
}
