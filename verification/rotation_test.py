"""Rotation test analysis and plots for avalanchers, adapted from AvaFrame.

Port of the AvaFrame rotation test (``avaframe/ana1Tests/rotationTest.py`` +
``runScripts/runRotationTest.py``): analyze the effect of the grid
orientation on the simulation results by running the SAME avalanche on
rotated DEM + release rasters and comparing everything in the reference
frame. Adapted to run on in-memory avalanchers results (see the companion
script ``run_rotation_test.py``) - no output files, no avaframe imports.

Where AvaFrame rotates com1DFA peak rasters with ``geoTrans.rotateRaster``
and compares along an AIMEC path transform, this port:

- rotates the DEM + release rasters with ``scipy.ndimage.rotate`` (+angle =
  counterclockwise in map coordinates, rotated about the domain center,
  domain/bounds kept identical) and hands them to the simulation with
  ``set_dem_with_bounds`` / ``set_release_areas``
- rotates the resulting peak fields back with -angle and compares them to
  the reference run along cross sections perpendicular to the reference
  center-of-mass path (a lightweight stand-in for the AIMEC path transform)

Plots (same spirit as the AvaFrame rotation test report): the energy line
figure per rotation (see ``energy_line_test.py``), runout metrics vs
rotation angle, max/mean values along the path for every rotation, and the
TP/FP/FN area comparison maps against the reference run.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import rotate as ndimage_rotate

FS = 12


# ---------------------------------------------------------------------------
# raster rotation
# ---------------------------------------------------------------------------

def rotate_raster(raster, angle_deg, cval=0.0):
    """Rotate raster content by `angle_deg` (CCW in map coordinates, i.e.
    row 0 = south, origin lower) about the domain center; shape preserved."""
    return ndimage_rotate(np.asarray(raster, dtype=np.float64), angle_deg,
                          reshape=False, order=1, mode="constant",
                          cval=cval, prefilter=False)


def clean_rotated_release(rotated, thickness):
    """Re-binarize a bilinearly rotated release raster to the original
    thickness (cells keep `thickness` where the rotation left >= 50 %)."""
    return np.where(rotated >= 0.5 * thickness, thickness, 0.0)


# ---------------------------------------------------------------------------
# cross sections along a path (lightweight AIMEC path transform)
# ---------------------------------------------------------------------------

def resample_path(path_x, path_y, ds):
    """Resample a polyline to uniform arc-length spacing `ds`."""
    x = np.asarray(path_x, dtype=float)
    y = np.asarray(path_y, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s_new = np.arange(0.0, s[-1], ds)
    return np.interp(s_new, s, x), np.interp(s_new, s, y), s_new


def make_cross_sections(path_x, path_y, step, half_width, ds):
    """Cross sections perpendicular to the resampled path.

    Returns (s, offsets, points_x, points_y, valid): `s` the along-path
    coordinate of every section, `offsets` the signed across-path offsets,
    `points_x/y` the sampled points (len(s) x len(offsets)) and `valid` the
    in-domain mask of the sampled points.
    """
    px, py, s = resample_path(path_x, path_y, ds)
    # tangents from the resampled points
    tx = np.gradient(px)
    ty = np.gradient(py)
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    tx, ty = tx / norm, ty / norm
    # normal points to the left of the flow direction
    nx, ny = -ty, tx
    offsets = np.arange(-half_width, half_width + step / 2.0, step)
    points_x = px[:, None] + offsets[None, :] * nx[:, None]
    points_y = py[:, None] + offsets[None, :] * ny[:, None]
    return s, offsets, points_x, points_y


def sample_field(field, header, points_x, points_y):
    """Bilinear sampling of a row-0-south raster at map coordinates
    (NaN outside the domain or on NoData)."""
    cs = header["cellsize"]
    nrows, ncols = field.shape
    col = (points_x - header["xmin"]) / cs - 0.5
    row = (points_y - header["ymin"]) / cs - 0.5
    ok = (col >= 0) & (col < ncols - 1) & (row >= 0) & (row < nrows - 1)
    col0 = np.clip(np.floor(col).astype(int), 0, ncols - 2)
    row0 = np.clip(np.floor(row).astype(int), 0, nrows - 2)
    tx = np.clip(col - col0, 0.0, 1.0)
    ty = np.clip(row - row0, 0.0, 1.0)
    v = ((1 - ty) * ((1 - tx) * field[row0, col0] + tx * field[row0, col0 + 1])
         + ty * ((1 - tx) * field[row0 + 1, col0] + tx * field[row0 + 1, col0 + 1]))
    return np.where(ok, v, np.nan)


def cross_section_stats(field, header, sections):
    """Max and mean of `field` along every cross section (NaN ignored)."""
    _, _, points_x, points_y = sections
    values = sample_field(field, header, points_x, points_y)
    with np.errstate(invalid="ignore"):
        vmax = np.nanmax(values, axis=1)
        vmean = np.nanmean(values, axis=1)
    return vmax, vmean


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_runout_comparison(results, ref_angle, cfg, sim_name, out_dir):
    """Runout metrics vs rotation angle with the reference values dashed
    (visualization of the rotation test energy line result table)."""
    angles = [r["angle"] for r in results]
    ref = next(r for r in results if r["angle"] == ref_angle)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [("sEnd", "runout s [m]"), ("zEnd", "runout z [m]"),
              ("runoutAngle", "runout angle [deg]")]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(angles, [r[key] for r in results], "b-o",
                label="avalanchers")
        ax.axhline(ref[key], color="k", linestyle="--",
                   label="reference (%g deg)" % ref_angle)
        ax.set_xlabel("grid rotation angle [deg]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    diffs = [(r["angle"], r["sEnd"] - ref["sEnd"], r["zEnd"] - ref["zEnd"],
              r["runoutAngle"] - ref["runoutAngle"]) for r in results
             if r["angle"] != ref_angle]
    text = "diff to reference:\n" + "\n".join(
        "%5.1f deg: ds %+7.2f m, dz %+6.2f m, dAngle %+6.3f deg" % d
        for d in diffs)
    axes[1].text(0.98, 0.98, text, transform=axes[1].transAxes, ha="right",
                 va="top", fontsize=FS - 2,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    fig.suptitle("Rotation test - runout comparison (%s)" % sim_name)
    fig.tight_layout()
    return _save_fig(fig, out_dir, sim_name + "_runout_comparison")


def plot_sl_comparison(sections, stats_per_angle, cfg, sim_name, out_dir):
    """Max and mean values along the path for every rotation (lightweight
    port of the AIMEC 'comparison of mean and max values along path' plot).

    `stats_per_angle`: {angle: {"pft": (vmax, vmean), "pfv": (vmax, vmean)}}.
    """
    s = sections[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = [("pft", "max", 0), ("pft", "mean", 1),
              ("pfv", "max", 0), ("pfv", "mean", 1)]
    for (res, stat, row) in panels:
        ax = axes[0 if res == "pft" else 1][row]
        for angle, stats in sorted(stats_per_angle.items()):
            vmax, vmean = stats[res]
            values = vmax if stat == "max" else vmean
            ax.plot(s, values, label="%g deg" % angle)
        ax.set_ylabel("%s %s along path" % (
            "flow thickness [m]" if res == "pft" else "peak flow velocity [m/s]",
            stat))
        ax.grid(alpha=0.3)
    axes[1][0].set_xlabel("path coordinate s [m]")
    axes[1][1].set_xlabel("path coordinate s [m]")
    axes[0][0].legend(title="grid rotation", fontsize=FS - 2)
    fig.suptitle("Rotation test - comparison of max/mean values along the "
                 "reference path (%s)" % sim_name)
    fig.tight_layout()
    return _save_fig(fig, out_dir, sim_name + "_sl_comparison")


def plot_areas_comparison(header, ref_mask, masks, cfg, sim_name, out_dir):
    """TP/FP/FN maps of every rotation against the reference run
    (lightweight port of the AIMEC area analysis plot). Colors follow the
    avalanchers plot convention: red = reference only (FN), magenta = both
    (TP), blue = simulation only (FP)."""
    cs = header["cellsize"]
    cell_area = cs * cs
    angles = sorted(masks)
    n = len(angles)
    n_cols = min(n, 3)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                             squeeze=False)
    extent = [header["xmin"], header["xmax"], header["ymin"], header["ymax"]]
    cmap = ListedColormap(["w", "#d62728", "#e377c2", "#1f77b4"])
    cmap.set_bad(alpha=0)
    # value v must map to color v: boundaries centered on the integers
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#e377c2", label="TP (both)"),
        plt.Rectangle((0, 0), 1, 1, fc="#d62728", label="FN (reference only)"),
        plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", label="FP (simulation only)"),
    ]
    # fit the map window to the reference corridor (+ buffer)
    flowed = np.where(ref_mask)
    y_buffer = x_buffer = 25.0
    x_lo = header["xmin"] + max(flowed[1].min() * cs - x_buffer, 0.0)
    x_hi = header["xmin"] + min((flowed[1].max() + 1) * cs + x_buffer,
                                header["xmax"] - header["xmin"])
    y_lo = header["ymin"] + max(flowed[0].min() * cs - y_buffer, 0.0)
    y_hi = header["ymin"] + min((flowed[0].max() + 1) * cs + y_buffer,
                                header["ymax"] - header["ymin"])
    for k, angle in enumerate(angles):
        ax = axes[k // n_cols][k % n_cols]
        comparison = np.zeros(ref_mask.shape, dtype=int)
        comparison[ref_mask & ~masks[angle]] = 1   # reference only -> FN
        comparison[ref_mask & masks[angle]] = 2    # both          -> TP
        comparison[~ref_mask & masks[angle]] = 3   # simulation only -> FP
        tp = float((comparison == 2).sum()) * cell_area
        fp = float((comparison == 3).sum()) * cell_area
        fn = float((comparison == 1).sum()) * cell_area
        union = tp + fp + fn
        dice = 2 * tp / (2 * tp + fp + fn) if union > 0 else 1.0
        ax.imshow(np.ma.masked_where(comparison == 0, comparison), origin="lower",
                  extent=extent, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("%g deg - Dice %.3f (TP %.0f m$^2$, FP %.0f, FN %.0f)"
                     % (angle, dice, tp, fp, fn), fontsize=FS - 1)
        ax.set_xlabel("x [m]")
        if k % n_cols == 0:
            ax.set_ylabel("y [m]")
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.suptitle("Rotation test - area comparison vs reference "
                 "(peak flow thickness > %g m)" % cfg["peak_flow_thickness_threshold"])
    # shared legend BELOW the panels so it cannot occlude the deposit ribbon
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=FS - 1, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save_fig(fig, out_dir, sim_name + "_areas_comparison")


def print_results_table(results, ref_angle):
    """Console version of the rotation test energy line result table."""
    ref = next(r for r in results if r["angle"] == ref_angle)
    print()
    print(f"{'angle':>7} {'sEnd [m]':>10} {'zEnd [m]':>10} {'angle [deg]':>12} "
          f"{'ds [m]':>9} {'dz [m]':>9} {'dAngle':>9} {'Dice':>7}")
    for r in results:
        ds = r["sEnd"] - ref["sEnd"]
        dz = r["zEnd"] - ref["zEnd"]
        da = r["runoutAngle"] - ref["runoutAngle"]
        dice = "" if r["dice"] is None else "%7.3f" % r["dice"]
        print(f"{r['angle']:7g} {r['sEnd']:10.2f} {r['zEnd']:10.2f} "
              f"{r['runoutAngle']:12.3f} {ds:+9.2f} {dz:+9.2f} {da:+9.3f} {dice:>7}")


def _ensure_dir(out_dir):
    import pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_fig(fig, out_dir, name):
    out_dir = _ensure_dir(out_dir)
    path = out_dir / (name + ".png")
    fig.savefig(path, dpi=150)
    print("plot saved to %s" % path)
    plt.close(fig)
    return path
