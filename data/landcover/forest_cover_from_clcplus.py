"""Canopy cover raster for the forest model from CLCplus Backbone land cover.

CLCplus Backbone (10 m, EPSG:3035) classifies every pixel; the classes 2-4
(woody needle leaved, broadleaved deciduous and broadleaved evergreen trees)
are trees. The cover of a simulation cell is the fraction of its area covered
by tree pixels: the binary tree mask is warped onto the target grid with
average resampling, optionally smoothed with a Gaussian of `--smoothing` m so
the cover is a stand-scale fraction rather than the 0/1 of single 10 m pixels.

The target grid is that of a reference DEM (.tif or .asc, with `--epsg` if the
file carries no CRS) or is given with `--bounds`, `--cell-size` and `--epsg`.
The result is a float32 GeoTIFF (or .asc) with the cover fraction 0..1, ready
for the `forest_cover_path` setting.

The download archive (CLCPLUS_RAS_S2023_R10m_03035.zip) holds one zip per
100 km tile, named after the tile's lower left corner in km / 100 (e.g.
E47N26); the script reads the tiles it needs directly from the archive.

Usage:
    python data/landcover/forest_cover_from_clcplus.py --reference dem.tif \
        --output forest_cover.tif
    python data/landcover/forest_cover_from_clcplus.py --epsg 2056 \
        --bounds 2780000 1180000 2786000 1186000 --cell-size 5 --output cover.tif
"""

import argparse
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

TREE_CLASSES = (2, 3, 4)
# no data, outside area and coastal seawater buffer
INVALID_CLASSES = (0, 253, 254, 255)
CLCPLUS_CRS = CRS.from_epsg(3035)
TILE_SIZE = 100_000.0
DEFAULT_ARCHIVE = Path(__file__).with_name("CLCPLUS_RAS_S2023_R10m_03035.zip")
TILE_PATTERN = re.compile(r"_E(\d+)N(\d+)_")


def target_grid(args):
    """CRS, transform, width and height of the output grid."""
    if args.reference:
        with rasterio.open(args.reference) as ref:
            crs = ref.crs or (CRS.from_epsg(args.epsg) if args.epsg else None)
            if crs is None:
                sys.exit(f"{args.reference} has no CRS; pass --epsg")
            return crs, ref.transform, ref.width, ref.height
    if not (args.bounds and args.cell_size and args.epsg):
        sys.exit("pass --reference, or --bounds, --cell-size and --epsg")
    xmin, ymin, xmax, ymax = args.bounds
    width = int(round((xmax - xmin) / args.cell_size))
    height = int(round((ymax - ymin) / args.cell_size))
    transform = from_origin(xmin, ymax, args.cell_size, args.cell_size)
    return CRS.from_epsg(args.epsg), transform, width, height


def tiles_for_bounds(archive, bounds_3035):
    """GDAL paths of the tile GeoTIFFs in the archive covering the bounds."""
    xmin, ymin, xmax, ymax = bounds_3035
    wanted = {
        (e, n)
        for e in range(int(xmin // TILE_SIZE), int(math.floor(xmax / TILE_SIZE)) + 1)
        for n in range(int(ymin // TILE_SIZE), int(math.floor(ymax / TILE_SIZE)) + 1)
    }
    paths = []
    with zipfile.ZipFile(archive) as outer:
        for name in outer.namelist():
            match = TILE_PATTERN.search(name)
            if not match or (int(match[1]), int(match[2])) not in wanted:
                continue
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(outer.open(name)) as inner:
                    tifs = [n for n in inner.namelist() if n.lower().endswith(".tif")]
                for tif in tifs:
                    paths.append(f"/vsizip/{{/vsizip/{archive.resolve().as_posix()}/{name}}}/{tif}")
            elif name.lower().endswith(".tif"):
                paths.append(f"/vsizip/{archive.resolve().as_posix()}/{name}")
    return paths


def tree_cover(args):
    crs, transform, width, height = target_grid(args)
    left, top = transform.c, transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    bounds_3035 = transform_bounds(crs, CLCPLUS_CRS, left, bottom, right, top, densify_pts=21)
    tiles = tiles_for_bounds(Path(args.archive), bounds_3035)
    if not tiles:
        sys.exit(f"no CLCplus tile of {args.archive} covers the target grid")

    cover = np.full((height, width), np.nan, dtype=np.float32)
    margin = 50.0
    for tile in tiles:
        with rasterio.open(tile) as src:
            window = window_from_bounds(
                bounds_3035[0] - margin, bounds_3035[1] - margin,
                bounds_3035[2] + margin, bounds_3035[3] + margin,
                transform=src.transform,
            ).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            classes = src.read(1, window=window)
            mask = np.isin(classes, TREE_CLASSES).astype(np.float32)
            mask[np.isin(classes, INVALID_CLASSES)] = np.nan
            print(f"{tile.rsplit('/', 1)[-1]}: {np.nanmean(mask):.1%} trees in the window")
            reproject(
                source=mask,
                destination=cover,
                src_transform=src.window_transform(window),
                src_crs=src.crs or CLCPLUS_CRS,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                init_dest_nodata=False,
                resampling=Resampling.average,
            )

    if args.smoothing > 0:
        from scipy.ndimage import gaussian_filter

        sigma = args.smoothing / abs(transform.a)
        valid = np.isfinite(cover)
        weights = gaussian_filter(valid.astype(np.float32), sigma)
        smoothed = gaussian_filter(np.where(valid, cover, 0.0), sigma)
        cover = np.where(valid, smoothed / np.maximum(weights, 1e-6), np.nan).astype(np.float32)

    missing = ~np.isfinite(cover)
    if missing.any():
        print(f"{missing.mean():.1%} of the cells have no land cover data; their cover is 0")
    cover[missing] = 0.0
    return np.clip(cover, 0.0, 1.0), crs, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE), help="CLCplus download zip")
    parser.add_argument("--reference", help="DEM whose grid the cover gets (.tif/.asc)")
    parser.add_argument("--epsg", type=int, help="CRS of the target grid")
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--cell-size", type=float)
    parser.add_argument("--smoothing", type=float, default=10.0,
                        help="Gaussian smoothing width (m) of the cover; 0 disables")
    parser.add_argument("--output", required=True, help="cover raster (.tif or .asc)")
    args = parser.parse_args()

    cover, crs, transform = tree_cover(args)
    driver = "AAIGrid" if args.output.lower().endswith(".asc") else "GTiff"
    profile = dict(driver=driver, width=cover.shape[1], height=cover.shape[0], count=1,
                   dtype="float32", crs=crs, transform=transform)
    if driver == "GTiff":
        profile["compress"] = "lzw"
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(cover, 1)
    print(f"wrote {args.output}: {cover.shape[1]}x{cover.shape[0]} cells, "
          f"mean cover {cover.mean():.2f}, forested (>0.5) {np.mean(cover > 0.5):.1%}")


if __name__ == "__main__":
    main()
