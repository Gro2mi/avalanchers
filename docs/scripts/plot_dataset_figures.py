"""Thesis figures of the datasets chapter.

1. ``dataset_swe_change``: change of the snow water equivalent of the OSHD
   snow dataset (Oberrauch et al. 2025, 250 m, daily snapshots at 07:00) over
   the snow window Jan 20 07:00 -> Jan 23 07:00, 2018, with the mapped
   avalanche outlines of January 2018 on top (``--no-outlines``: without them,
   saved as ``dataset_swe_change_no_outlines``), the estimated mapped area,
   the national and canton borders and the largest cities.

Needs ``data/OSHD_DATA_2018-01.nc``, ``data/aval_outlines2018`` and
``data/naturalearth`` (``ne_10m_admin_1_states_provinces`` from
naturalearthdata.com) of the avalanchers_evaluation checkout.

Usage:
    python plot_dataset_figures.py
    python plot_dataset_figures.py --no-outlines
    python plot_dataset_figures.py --evaluation-dir <dir> --out-dir <dir>
"""

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# shared helpers (thesis_style, cli_args) live in verification/
sys.path.insert(0, str(REPO_ROOT / "verification"))
OUTPUT_DIR = REPO_ROOT / "docs" / "figures"
EVALUATION_DIR = REPO_ROOT.parent / "avalanchers_evaluation"
OSHD_FILE = pathlib.Path("data") / "OSHD_DATA_2018-01.nc"
OUTLINES_FILE = pathlib.Path("data") / "aval_outlines2018" / "outlines2018.shp"
# Natural Earth 1:10m admin-1 boundaries (public domain): the cantons, merged
# into the national border
BOUNDARIES_FILE = (pathlib.Path("data") / "naturalearth"
                   / "ne_10m_admin_1_states_provinces.shp")
# cities for orientation: name -> (latitude, longitude, label alignment), the
# alignment puts the label to the side with the fewest neighbours; English
# names, since the thesis font cmr10 has no accented glyphs
CITIES = {
    "Basel": (47.5596, 7.5886, "left"),
    "Zurich": (47.3769, 8.5417, "left"),
    "St. Gallen": (47.4245, 9.3767, "left"),
    "Bern": (46.9481, 7.4474, "left"),
    "Lucerne": (47.0502, 8.3093, "left"),
    "Lausanne": (46.5197, 6.6323, "left"),
    "Geneva": (46.2044, 6.1432, "left"),
    "Sion": (46.2331, 7.3606, "left"),
    "Chur": (46.8508, 9.5320, "left"),
    "Lugano": (46.0037, 8.9511, "left"),
}

FIGURES = ("swe",)

# snow window of the calibration (avalanchers_evaluation/calibration/config.py)
SNOW_START = "2018-01-20T07:00:00"
SNOW_END = "2018-01-23T07:00:00"
# colour scale of the SWE change (kg m-2): asymmetric around zero, since
# losses are small and gains reach several hundred kg m-2
SWE_RANGE = (-50.0, 250.0)
# mapped area: the outline vertices on the OSHD grid, dilated by
# MAPPED_AREA_CLOSE_KM + MAPPED_AREA_MARGIN_KM and eroded by MAPPED_AREA_CLOSE_KM,
# so gaps up to about twice the closing radius between outlines are filled and
# the area extends MAPPED_AREA_MARGIN_KM beyond the outermost outlines
MAPPED_AREA_CLOSE_KM = 2.0
MAPPED_AREA_MARGIN_KM = 1.0


def mapped_area(outlines, easting, northing):
    """Mask on the (northing, easting) grid of the area covered by the outlines.

    The extent of the satellite scenes is not part of the outline dataset, so
    the area is estimated as a morphological closing of the outline vertices.
    """
    import shapely
    from scipy import ndimage

    points = shapely.get_coordinates(outlines.geometry.values)
    cols = np.rint((points[:, 0] - easting[0]) / (easting[1] - easting[0])).astype(int)
    rows = np.rint((points[:, 1] - northing[0]) / (northing[1] - northing[0])).astype(int)
    inside = (cols >= 0) & (cols < len(easting)) & (rows >= 0) & (rows < len(northing))
    mask = np.zeros((len(northing), len(easting)), bool)
    mask[rows[inside], cols[inside]] = True
    cell = abs(easting[1] - easting[0])

    def disk(radius_km):
        r = round(radius_km / cell)
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return x * x + y * y <= r * r

    mask = ndimage.binary_dilation(mask, disk(MAPPED_AREA_CLOSE_KM + MAPPED_AREA_MARGIN_KM))
    mask = ndimage.binary_erosion(mask, disk(MAPPED_AREA_CLOSE_KM), border_value=0)
    return ndimage.binary_fill_holes(mask)


def draw_orientation(ax, evaluation_dir):
    """Swiss border, canton borders and the CITIES, in km LV95."""
    import geopandas as gpd
    import matplotlib.patheffects as pe
    from pyproj import Transformer

    cantons = gpd.read_file(evaluation_dir / BOUNDARIES_FILE, where="adm0_a3 = 'CHE'")
    cantons = cantons.to_crs(2056)
    cantons = cantons.set_geometry(cantons.geometry.scale(1e-3, 1e-3, origin=(0, 0)))
    cantons.boundary.plot(ax=ax, color="0.45", linewidth=0.35, zorder=2)
    gpd.GeoSeries([cantons.union_all()]).boundary.plot(ax=ax, color="0.15", linewidth=0.8,
                                                       zorder=2)

    to_lv95 = Transformer.from_crs(4326, 2056, always_xy=True)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for name, (lat, lon, ha) in CITIES.items():
        x, y = to_lv95.transform(lon, lat)
        x, y = x / 1000.0, y / 1000.0
        ax.plot(x, y, "o", color="0.15", markersize=2.5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        dx = 2.5 if ha == "left" else -2.5
        ax.text(x + dx, y, name, ha=ha, va="center", fontsize=7, color="0.1",
                path_effects=halo, zorder=5)


def figure_swe(out_dir, evaluation_dir, outlines=True):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import thesis_style as ts
    import xarray as xr
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.lines import Line2D

    with xr.open_dataset(evaluation_dir / OSHD_FILE) as dataset:
        swe = dataset.swe.sel(time=[np.datetime64(SNOW_START), np.datetime64(SNOW_END)])
        swe = swe.transpose("time", "northing", "easting")
        change = (swe[1] - swe[0]).values
        easting = dataset.easting.values / 1000.0
        northing = dataset.northing.values / 1000.0
    half = 0.5 * abs(easting[1] - easting[0])
    extent = (easting.min() - half, easting.max() + half,
              northing.min() - half, northing.max() + half)
    origin = "upper" if northing[0] > northing[-1] else "lower"
    print(f"SWE change {SNOW_START} -> {SNOW_END}: "
          f"median {np.nanmedian(change):.1f}, 99th percentile "
          f"{np.nanpercentile(change, 99):.1f}, max {np.nanmax(change):.1f} kg m-2")
    draw_outlines = outlines
    outlines = gpd.read_file(evaluation_dir / OUTLINES_FILE).to_crs(2056)
    outlines = outlines.set_geometry(outlines.geometry.scale(1e-3, 1e-3, origin=(0, 0)))
    area = mapped_area(outlines, easting, northing)
    print(f"{len(outlines)} outlines, mapped area {area.sum() * (2 * half) ** 2:.0f} km2")

    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(1.0, 0.7), layout="constrained")
        ax.set_aspect("equal")
        ax.grid(False)
        # linear scale with white at zero: the part of the diverging map that
        # SWE_RANGE covers when its ends are symmetric about zero
        low, high = SWE_RANGE
        span = max(-low, high)
        cmap = ListedColormap(plt.get_cmap("RdBu")(
            np.linspace(0.5 + 0.5 * low / span, 0.5 + 0.5 * high / span, 256)))
        image = ax.imshow(change, extent=extent, origin=origin, cmap=cmap,
                          norm=Normalize(low, high), interpolation="nearest", zorder=1)
        draw_orientation(ax, evaluation_dir)
        handles = [Line2D([], [], color="0.15", linewidth=0.8, label="national border"),
                   Line2D([], [], color="0.45", linewidth=0.35, label="canton borders")]
        if draw_outlines:
            outlines.plot(ax=ax, facecolor="none", edgecolor=ts.BLACK, linewidth=0.3,
                          zorder=2)
            handles.append(Line2D([], [], color=ts.BLACK, linewidth=0.6,
                                  label="avalanche outlines"))
        ax.contour(easting, northing, area.astype(float), levels=[0.5], colors=ts.BLACK,
                   linewidths=0.9, linestyles=[(0, (4, 2))], zorder=3)
        handles.append(Line2D([], [], color=ts.BLACK, linewidth=0.9, linestyle=(0, (4, 2)),
                              label="mapped area"))
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
                  edgecolor="none")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("easting (km, LV95)")
        ax.set_ylabel("northing (km, LV95)")
        bar = fig.colorbar(image, ax=ax, location="right", shrink=0.85, pad=0.01,
                           aspect=25, extend="both")
        bar.set_label(r"$\Delta \mathrm{SWE}$ (kg m$^{-2}$)")
        name = "dataset_swe_change" + ("" if draw_outlines else "_no_outlines")
        ts.save(fig, out_dir, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--evaluation-dir", type=pathlib.Path, default=EVALUATION_DIR,
                        help="avalanchers_evaluation checkout with the OSHD dataset and "
                             "the outlines")
    parser.add_argument("--no-outlines", action="store_true",
                        help="SWE change map without the avalanche outlines, saved as "
                             "dataset_swe_change_no_outlines")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out_dir = args.out_dir.resolve()

    if "swe" in args.figures:
        figure_swe(out_dir, args.evaluation_dir.resolve(), outlines=not args.no_outlines)


if __name__ == "__main__":
    main()
