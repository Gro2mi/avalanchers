"""Thesis figures of the methods chapter (flow model and release areas).

Four figures:

1. ``method_gully_confinement``: cross-section of a gully filled with snow.
   Real bed, reference surface z_ref (the bed smoothed with a Gaussian
   filter, computed analytically), bed normal n and reference normal n_ref
   with the wall inclination beta, the depth zeta below the flat free
   surface, and a wall element loaded by the stresses rho g_n zeta and
   K_y rho g_n zeta (Eq. eq:gully).
2. ``method_bed_geometry``: side view of a flow on a curved bed. Normal n,
   slope angle theta, the velocity u tangent to the bed with its horizontal
   part u_h and vertical part u_z, the thickness h measured normal to the bed
   and the vertical column h / cos(theta), and the osculating circles of a
   convex (kappa < 0) and a concave (kappa > 0) part.
3. ``method_arc_step``: one advection step of a particle on a concave bed
   (Eqs. eq:arc-step and eq:rotate). The straight step l t leaves the bed,
   the arc step ends close to it, the elevation is snapped to the bed and the
   velocity is rotated into the new tangent plane at constant speed. The
   step is exaggerated so the terms are visible.
4. ``method_release_area_<id>``: release area estimation of one mapped
   avalanche with the avalanchers engine: (a) outline, potential release
   cells and the one-cell uphill steps of the outline cells that leave the
   outline, split into crown line, candidates removed by the component
   filter and flank cells below the exit distance threshold; (b) geodesic
   distance to the crown line inside the outline and the selected release
   area. The script recomputes the crown candidates and the geodesic fill in
   Python and prints how many of the engine's cells it reproduces.

5. ``method_mpm_step``: one step of the material point method in plan view:
   (a) particle to grid with the B-spline weights of the 3 x 3 stencil,
   (b) grid update of the nodal velocities, (c) grid to particle, (d)
   advection to the new position with its new stencil.
6. ``method_loss_<id>``: loss components of two simulations of the avalanche
   (mu, xi of LOSS_RUNS): release area, overlap, simulated-only and
   observed-only cells, nearest-cell distances of the Chamfer distance and
   the length s between the highest and the lowest cell. The losses are
   recomputed in Python and compared with the engine's evaluation.
7. ``method_center_of_mass_<id>``: path of the center of mass of the first
   run in plan view with the length s of the footprint, and its elevation
   along the horizontal path with the travel angle.
8. ``method_friction_laws``: friction deceleration of the four friction laws
   against speed, and the steady speed on a plane against the slope angle.

Figures 1-3, 5 and 8 are schematic or analytical; 4, 6 and 7 run the engine.
Figures 1-3 are schematic and need no data. Figure 4 needs the outline
shapefiles and the swissALTI3D tile cache of avalanchers_evaluation (the
engine opens ./dtm_cache.zarr, so the script runs the engine from there).

Usage:
    python plot_method_figures.py                       # all four figures
    python plot_method_figures.py --figures gully arc   # a subset
    python plot_method_figures.py --avalanche-id 8078 --out-dir <dir>
"""

import argparse
import contextlib
import heapq
import math
import os
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "verification" / "outputs" / "method_figures"
EVALUATION_DIR = REPO_ROOT.parent / "avalanchers_evaluation"
OUTLINES_SUBDIR = pathlib.Path("data") / "single_avalanche_outlines_switzerland_2018"

FIGURES = ("gully", "bed", "arc", "release", "mpm", "loss", "com", "friction")

G = 9.81
# footprint threshold h_th of Eq. eq:footprint: the engine default
# peak_flow_thickness_threshold, which the calibration does not change
FOOTPRINT_THRESHOLD = 0.1
# (mu, xi) of the runs of the loss and center of mass figures: a run close to
# the outline and a strongly damped one
LOSS_RUNS = ((0.35, 2500.0), (0.7, 500.0))
# friction parameters of Table tab:flow-parameters
FRICTION_PARAMETERS = {"mu": 0.2, "xi": 2000.0, "mu_s": 0.38, "mu_2": 0.65, "I_0": 0.29,
                       "d_g": 0.002}

# release area estimation as in Table tab:release-parameters of the thesis
RELEASE_SETTINGS = {
    "sim_model": "3d-depth-averaged",
    "outlines_padding": 100,
    "density": 200.0,
    "release_swe": 100.0,  # only scales the thickness, not the release cells
    "min_slope_angle": 28.0,
    "max_slope_angle": 60.0,
    "roughness_threshold": 1.0,  # r <= 1 always holds: the filter is off
    "released_particles_per_cell": 4,
    "release_area_fraction": 0.215,
    "crown_line_method": "gradient",
    "crown_line_exit_distance": 0.75,
}
RELEASE_MIN_ELEVATION_PERCENTILE = 40.0
DEFAULT_AVALANCHE_ID = 8078
# center of mass figure: an avalanche whose path descends evenly (the slope
# falls from about 43 to 17 degrees without a plateau)
DEFAULT_COM_AVALANCHE_ID = 17281

# schematic line styles
BED_LW = 1.4
VECTOR_LW = 1.0
ARROW = "-|>,head_length=0.45,head_width=0.22"


# ---------------------------------------------------------------------------
# drawing helpers


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def vector(ax, start, end, color, lw=VECTOR_LW, style="-", zorder=5, shrink=0.0):
    """Arrow from start to end in data coordinates."""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        tuple(start), tuple(end), arrowstyle=ARROW, mutation_scale=10,
        color=color, linewidth=lw, linestyle=style, zorder=zorder,
        shrinkA=shrink, shrinkB=shrink))


def dimension(ax, start, end, color="0.25", lw=0.6, offset=None, zorder=4):
    """Two-headed dimension line from start to end."""
    from matplotlib.patches import FancyArrowPatch
    start, end = np.asarray(start, float), np.asarray(end, float)
    if offset is not None:
        start, end = start + offset, end + offset
    ax.add_patch(FancyArrowPatch(
        tuple(start), tuple(end), arrowstyle="<|-|>,head_length=0.35,head_width=0.15",
        mutation_scale=10, color=color, linewidth=lw, zorder=zorder,
        shrinkA=0.0, shrinkB=0.0))


def angle_arc(ax, center, v_from, v_to, radius, color="0.2", lw=0.7, zorder=4):
    """Arc of the (smaller) angle between the directions v_from and v_to;
    returns the point on the arc bisector just outside it for the label."""
    a0 = math.atan2(v_from[1], v_from[0])
    a1 = math.atan2(v_to[1], v_to[0])
    delta = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    angles = np.linspace(a0, a0 + delta, 40)
    ax.plot(center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles),
            color=color, linewidth=lw, zorder=zorder, solid_capstyle="butt")
    mid = a0 + delta / 2
    return np.array([math.cos(mid), math.sin(mid)])


def label(ax, xy, text, color="black", **kwargs):
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    ax.text(xy[0], xy[1], text, color=color, zorder=8, **kwargs)


def schematic_axes(fig_size):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# figure 1: gully confinement


def figure_gully(out_dir):
    import matplotlib.pyplot as plt
    import thesis_style as ts

    # bed across the gully: Gaussian gully of depth D and width w in a gently
    # concave cross slope; the Gaussian filter of standard deviation L maps
    # a exp(-x^2/w^2) to a w / W exp(-x^2/W^2) with W^2 = w^2 + 2 L^2 and
    # c x^2 to c (x^2 + L^2), so the reference surface is exact
    depth, width, bowl, smoothing = 1.25, 1.15, 0.07, 1.0
    wide = math.sqrt(width ** 2 + 2 * smoothing ** 2)

    def bed(x):
        return bowl * x ** 2 - depth * np.exp(-(x / width) ** 2)

    def bed_slope(x):
        return 2 * bowl * x + 2 * depth * x / width ** 2 * np.exp(-(x / width) ** 2)

    def reference(x):
        return bowl * (x ** 2 + smoothing ** 2) - depth * width / wide * np.exp(-(x / wide) ** 2)

    def reference_slope(x):
        return (2 * bowl * x
                + 2 * depth * width / wide * x / wide ** 2 * np.exp(-(x / wide) ** 2))

    def normal(slope):
        return unit((-slope, 1.0))

    x = np.linspace(-3.3, 3.3, 600)
    surface = 0.05  # flat free surface of the filled gully
    wet = bed(x) <= surface
    colors = {"bed": ts.BLACK, "ref": ts.GREEN, "flow": ts.SKY, "n": ts.BLUE,
              "stress": ts.VERMILION}

    with ts.style():
        fig, ax = schematic_axes(ts.size(1.0, 0.5))
        ax.fill_between(x, bed(x) - 0.35, bed(x), color="0.88", linewidth=0, zorder=0)
        ax.fill_between(x[wet], bed(x[wet]), surface, color=colors["flow"], alpha=0.35,
                        linewidth=0, zorder=1)
        ax.plot(x[wet], np.full(wet.sum(), surface), color=colors["flow"], linewidth=0.9,
                zorder=2)
        ax.plot(x, bed(x), color=colors["bed"], linewidth=BED_LW, zorder=3)
        ax.plot(x, reference(x), color=colors["ref"], linewidth=1.1, linestyle="--", zorder=3)

        # wall inclination beta on the left wall
        xp = -width / math.sqrt(2)
        p = np.array([xp, bed(xp)])
        n, n_ref = normal(bed_slope(xp)), normal(reference_slope(xp))
        length = 1.05
        vector(ax, p, p + length * n, colors["n"])
        vector(ax, p, p + length * n_ref, colors["ref"])
        label(ax, p + (length + 0.14) * n, r"$\mathbf{n}$", colors["n"])
        label(ax, p + (length + 0.16) * n_ref, r"$\mathbf{n}_{\mathrm{ref}}$", colors["ref"])
        bisector = angle_arc(ax, p, n_ref, n, 0.5)
        label(ax, p + 0.66 * bisector, r"$\beta$")
        ax.plot(*p, "o", color=ts.BLACK, markersize=2.5, zorder=6)

        # wall element on the right wall with the two stress components and
        # its depth zeta below the free surface
        xq = width / math.sqrt(2)
        q = np.array([xq, bed(xq)])
        dimension(ax, (xq + 0.42, q[1]), (xq + 0.42, surface))
        ax.plot([xq, xq + 0.5], [q[1], q[1]], color="0.45", linewidth=0.5, linestyle=":",
                zorder=4)
        label(ax, (xq + 0.55, 0.5 * (q[1] + surface)), r"$\zeta$", ha="left")
        tangent = unit((1.0, bed_slope(xq)))
        half = 0.2
        ax.plot(*np.transpose([q - half * tangent, q + half * tangent]),
                color=colors["stress"], linewidth=3.0, solid_capstyle="butt", zorder=4)
        arrow = 0.62
        vector(ax, q + (0, arrow), q + (0, 0.04), colors["stress"])
        vector(ax, q - (arrow, 0), q - (0.04, 0), colors["stress"])
        label(ax, q + (0.04, arrow + 0.12), r"$\rho\, g_n \zeta$", colors["stress"], ha="left")
        label(ax, q - (arrow + 0.02, -0.16), r"$K_y\, \rho\, g_n \zeta$", colors["stress"],
              ha="left", va="bottom")

        # labels of the lines
        label(ax, (2.95, bed(2.95) - 0.18), "bed $z$", ha="right", va="top")
        label(ax, (-3.2, reference(-3.2) + 0.12), r"reference surface $z_{\mathrm{ref}}$",
              colors["ref"], ha="left", va="bottom")
        label(ax, (1.35, surface + 0.08), "free surface", colors["flow"], va="bottom")
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-1.55, 1.0)
        fig.tight_layout(pad=0.1)
        ts.save(fig, out_dir, "method_gully_confinement")


# ---------------------------------------------------------------------------
# figure 2: bed geometry and flow variables


def figure_bed(out_dir):
    import thesis_style as ts

    # convex at the top, steep in the middle, concave at the bottom
    scale = 1.6

    def bed(x):
        return -scale * np.tanh(x / scale)

    def slope(x):
        return -1.0 / np.cosh(x / scale) ** 2

    def second(x):
        return 2.0 / scale * np.tanh(x / scale) / np.cosh(x / scale) ** 2

    def normal(xv):
        return unit((-slope(xv), 1.0))

    def tangent(xv):  # downhill, in the flow direction
        return unit((1.0, slope(xv)))

    def curvature(xv):
        return second(xv) / (1.0 + slope(xv) ** 2) ** 1.5

    x = np.linspace(-4.2, 4.2, 800)
    # flow thickness normal to the bed, tapered at the ends
    h0 = 0.34
    thickness = h0 * np.clip(1.0 - ((x - 0.3) / 3.3) ** 8, 0.0, None)
    normals = np.stack([normal(v) for v in x])
    top = np.stack([x, bed(x)], axis=1) + thickness[:, None] * normals
    colors = {"n": ts.BLUE, "u": ts.ORANGE, "flow": ts.SKY, "kappa": ts.GREEN}

    with ts.style():
        fig, ax = schematic_axes(ts.size(1.0, 0.62))
        ax.fill_between(x, bed(x) - 0.45, bed(x), color="0.88", linewidth=0, zorder=0)
        flow = np.concatenate([np.stack([x, bed(x)], axis=1), top[::-1]])
        ax.fill(flow[:, 0], flow[:, 1], color=colors["flow"], alpha=0.35, linewidth=0,
                zorder=1)
        ax.plot(top[:, 0], top[:, 1], color=colors["flow"], linewidth=0.8, zorder=2)
        ax.plot(x, bed(x), color=ts.BLACK, linewidth=BED_LW, zorder=3)

        # point B: normal, velocity and its parts, slope angle; the velocity
        # is drawn at mid-depth so it does not hide under the bed line
        xb = -0.75
        b = np.array([xb, bed(xb)])
        n, t = normal(xb), tangent(xb)
        hb = float(np.interp(xb, x, thickness))
        m = b + 0.5 * hb * n
        speed = 1.45
        u = speed * t
        vector(ax, b, b + 1.15 * n, colors["n"])
        label(ax, b + 1.29 * n, r"$\mathbf{n}$", colors["n"])
        vector(ax, m, m + u, colors["u"], lw=1.3)
        label(ax, m + u + 0.12 * n + 0.12 * t, r"$\mathbf{u}$", colors["u"])
        vector(ax, m, m + (u[0], 0.0), colors["u"], style=(0, (3, 2)))
        vector(ax, m + (u[0], 0.0), m + u, colors["u"], style=(0, (3, 2)))
        label(ax, m + (0.55 * u[0], 0.15), r"$\mathbf{u}_h$", colors["u"])
        label(ax, m + (u[0] + 0.2, 0.5 * u[1]), r"$u_z$", colors["u"])
        bisector = angle_arc(ax, m, (1.0, 0.0), t, 0.55)
        label(ax, m + 0.72 * bisector, r"$\theta$")
        ax.plot(*m, "o", color=colors["u"], markersize=2.5, zorder=6)

        # point A: thickness normal to the bed and the vertical column
        xa = 1.3
        a = np.array([xa, bed(xa)])
        na, ta = normal(xa), tangent(xa)
        ha = float(np.interp(xa, x, thickness))
        dimension(ax, a, a + ha * na)
        label(ax, a + 0.5 * ha * na - 0.2 * ta, r"$h$")
        column_x = xa + 0.55
        column_bed = float(bed(column_x))
        column_top = float(np.interp(column_x, top[:, 0], top[:, 1]))
        dimension(ax, (column_x, column_bed), (column_x, column_top))
        label(ax, (column_x + 0.1, 0.5 * (column_bed + column_top) + 0.02),
              r"$h / \cos\theta$", ha="left")

        # osculating circles of a convex and a concave part
        for xc, text, offset, ha in ((-2.3, r"convex, $\kappa < 0$", (0.0, -0.25), "center"),
                                     (2.5, r"concave, $\kappa > 0$", (0.45, 0.62), "left")):
            k = curvature(xc)
            radius = 1.0 / abs(k)
            p = np.array([xc, bed(xc)])
            center = p + normal(xc) / k
            phi0 = math.atan2(*(p - center)[::-1])
            phi = np.linspace(phi0 - 0.5, phi0 + 0.5, 60)
            ax.plot(center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi),
                    color=colors["kappa"], linewidth=0.9, linestyle="--", zorder=2)
            label(ax, p + offset, text, colors["kappa"], ha=ha)
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-2.25, 2.3)
        fig.tight_layout(pad=0.1)
        ts.save(fig, out_dir, "method_bed_geometry")


# ---------------------------------------------------------------------------
# figure 3: arc step and velocity rotation


def figure_arc(out_dir):
    import thesis_style as ts

    # concave bed: a circle of radius 4 whose curvature grows slightly
    # downhill, so the arc of the start curvature ends a little below the
    # bed; the step is exaggerated so the terms are visible
    radius, extra, x0, step = 4.0, 0.045, -2.3, 2.6

    def bed(x):
        return radius - np.sqrt(radius ** 2 - x ** 2) + extra * (x - x0) ** 2

    def slope(x):
        return x / np.sqrt(radius ** 2 - x ** 2) + 2 * extra * (x - x0)

    def second(x):
        return radius ** 2 / (radius ** 2 - x ** 2) ** 1.5 + 2 * extra

    def normal(xv):
        return unit((-slope(xv), 1.0))

    def tangent(xv):
        return unit((1.0, slope(xv)))

    def curvature(xv):
        return second(xv) / (1.0 + slope(xv) ** 2) ** 1.5

    r0 = np.array([x0, bed(x0)])
    t, n, k = tangent(x0), normal(x0), curvature(x0)
    along = step - k ** 2 * step ** 3 / 6
    across = k * step ** 2 / 2
    straight = r0 + step * t
    corner = r0 + along * t
    arc_end = corner + across * n
    r1 = np.array([arc_end[0], bed(arc_end[0])])  # snapped to the DEM
    t1 = tangent(r1[0])
    speed = 1.15
    colors = {"step": ts.BLUE, "u": ts.ORANGE, "kappa": ts.GREEN}
    print(f"arc step: kappa {k:.3f}, normal offset {across:.3f}, tangential correction "
          f"{step - along:.3f}, snap {r1[1] - arc_end[1]:.3f}")

    x = np.linspace(-3.1, 1.7, 400)
    with ts.style():
        fig, ax = schematic_axes(ts.size(1.0, 0.66))
        ax.fill_between(x, bed(x) - 2.0, bed(x), color="0.9", linewidth=0, zorder=0)
        ax.plot(x, bed(x), color=ts.BLACK, linewidth=BED_LW, zorder=3)

        # circular arc of curvature kappa_p from r^n
        center = r0 + n / k
        phi0 = math.atan2(*(r0 - center)[::-1])
        phi = np.linspace(phi0, phi0 + step * k, 60)
        ax.plot(center[0] + np.cos(phi) / k, center[1] + np.sin(phi) / k,
                color=colors["kappa"], linewidth=1.0, linestyle="--", zorder=4)
        mid = center + (r0 - center) @ np.array(
            [[math.cos(0.3 * step * k), math.sin(0.3 * step * k)],
             [-math.sin(0.3 * step * k), math.cos(0.3 * step * k)]])
        label(ax, mid + 0.1 * normal(mid[0]) + (0.05, 0.0),
              r"arc of curvature $\kappa_p$", colors["kappa"], ha="left", va="bottom")

        # straight step along the tangent, which leaves the bed
        ax.plot(*np.transpose([r0, straight]), color="0.45", linewidth=0.8,
                linestyle=(0, (3, 2)), zorder=3)
        ax.plot(*straight, "o", color="0.45", markersize=2.5, zorder=5)
        label(ax, straight + 0.12 * t, r"$\mathbf{r}_p^{n} + \ell_p \mathbf{t}$", "0.35",
              ha="left", va="top")
        # dimension lines of the two tangential lengths, below the tangent
        dimension(ax, r0, straight, offset=-0.3 * n)
        label(ax, 0.5 * (r0 + straight) - 0.42 * n, r"$\ell_p$", ha="right", va="top")
        dimension(ax, r0, corner, offset=-0.62 * n, color=colors["step"])
        label(ax, 0.5 * (r0 + corner) - 0.74 * n,
              r"$\ell_p - \kappa_p^2 \ell_p^3 / 6$", colors["step"], ha="right", va="top")
        # normal part of the arc step
        vector(ax, corner, arc_end, colors["step"], lw=1.2)
        label(ax, corner + 0.55 * across * n + 0.12 * t,
              r"$\frac{1}{2} \kappa_p \ell_p^2\, \mathbf{n}_p$", colors["step"], ha="left")
        # snap of the elevation to the DEM
        vector(ax, arc_end, r1, ts.BLACK, lw=0.7)
        label(ax, 0.5 * (arc_end + r1) + (0.1, 0.0), "snap to DEM", "0.25", ha="left",
              fontsize=8)

        # velocity before and after the rotation into the new tangent plane
        vector(ax, r1, r1 + speed * t, colors["u"], lw=1.0, style=(0, (3, 2)), zorder=6)
        vector(ax, r1, r1 + speed * t1, colors["u"], lw=1.3, zorder=6)
        label(ax, r1 + speed * t + 0.1 * t, r"$\mathbf{u}_p^{n+1}$ gathered", colors["u"],
              ha="left", va="top", fontsize=8)
        label(ax, r1 + speed * t1 + 0.1 * t1, "rotated", colors["u"],
              ha="left", va="bottom", fontsize=8)

        # normal at r^n and the two positions
        vector(ax, r0, r0 + 0.6 * n, ts.BLACK, lw=0.8)
        label(ax, r0 + 0.74 * n, r"$\mathbf{n}_p$")
        ax.plot(*r0, "o", color=ts.BLACK, markersize=3, zorder=7)
        label(ax, r0 + 0.32 * t + 0.2 * n, r"$\mathbf{r}_p^{n}$", ha="left")
        ax.plot(*r1, "o", color=ts.BLACK, markersize=3, zorder=7)
        label(ax, r1 + 0.08 * normal(r1[0]) - 0.08 * t1, r"$\mathbf{r}_p^{n+1}$",
              ha="right", va="bottom")
        ax.set_xlim(-3.1, 1.7)
        ax.set_ylim(-1.5, 1.75)
        fig.tight_layout(pad=0.1)
        ts.save(fig, out_dir, "method_arc_step")


# ---------------------------------------------------------------------------
# figure 4: release area estimation


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_release_estimation(evaluation_dir, avalanche_id):
    """DEM, outline and the release estimation layers of one avalanche."""
    import avalanchers

    outline = OUTLINES_SUBDIR / f"polygon_{avalanche_id}.shp"
    settings = dict(RELEASE_SETTINGS, outlines_path=str(outline))
    with working_directory(evaluation_dir):
        if not outline.exists():
            raise FileNotFoundError(evaluation_dir / outline)
        # first pass only for the DEM inside the outline -> z_min
        sim = avalanchers.PySimulation.new()
        sim.create(settings)
        dem, roi = np.asarray(sim.dem, float), np.asarray(sim.roi, bool)
        z_min = float(np.percentile(dem[roi], RELEASE_MIN_ELEVATION_PERCENTILE))
        sim = avalanchers.PySimulation.new()
        sim.create(dict(settings, release_min_elevation=z_min))
        sim.prepare()
        layers = {
            "dem": np.asarray(sim.dem, float),
            "roi": np.asarray(sim.roi, bool),
            "potential": np.asarray(sim.potential_release_areas, bool),
            "candidates": np.asarray(sim.crown_candidates, bool),
            "crown": np.asarray(sim.crown_line, bool),
            "release": np.asarray(sim.release_areas, float) > 0,
            "cell_size": float(sim.cell_size),
            "bounds": np.asarray(sim.dem_bounds, float),
            "z_min": z_min,
        }
    return layers


def uphill_steps(dem, roi):
    """Stepped points p_c (in cell units, cell centres at index + 0.5) and the
    exit distance d_out of every outline cell whose one-cell uphill step leaves
    the outline (Eqs. eq:gradient-step and eq:exit-distance); rows are y."""
    gy, gx = np.gradient(dem)
    exits = []
    height, width = dem.shape
    for row, col in zip(*np.nonzero(roi)):
        g = np.array([gx[row, col], gy[row, col]])
        norm = np.linalg.norm(g)
        if not np.isfinite(norm) or norm == 0:
            continue
        start = np.array([col + 0.5, row + 0.5])
        point = start + g / norm
        pc, pr = int(math.floor(point[0])), int(math.floor(point[1]))
        if 0 <= pr < height and 0 <= pc < width and roi[pr, pc]:
            continue
        nearest = math.inf
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = pr + dr, pc + dc
                if 0 <= r < height and 0 <= c < width and roi[r, c]:
                    nearest = min(nearest, math.hypot(c + 0.5 - point[0], r + 0.5 - point[1]))
        exits.append((row, col, start, point, nearest))
    return exits


def geodesic_distance(seeds, traversable, cell_size):
    """Chamfer geodesic distance (Dijkstra, 8 neighbours) from the seeds,
    through traversable cells only; inf where unreachable."""
    height, width = seeds.shape
    distance = np.full(seeds.shape, np.inf)
    heap = []
    for r, c in zip(*np.nonzero(seeds & traversable)):
        distance[r, c] = 0.0
        heap.append((0.0, r, c))
    heapq.heapify(heap)
    steps = [(dr, dc, cell_size * (math.sqrt(2) if dr and dc else 1.0))
             for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    while heap:
        d0, r, c = heapq.heappop(heap)
        if d0 > distance[r, c]:
            continue
        for dr, dc, w in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and traversable[nr, nc]:
                if d0 + w < distance[nr, nc]:
                    distance[nr, nc] = d0 + w
                    heapq.heappush(heap, (d0 + w, nr, nc))
    return distance


def outline_edges(mask):
    """Line segments (in cell units) along the boundary of a raster mask."""
    segments = []
    padded = np.pad(mask, 1)
    for r, c in zip(*np.nonzero(mask)):
        pr, pc = r + 1, c + 1
        if not padded[pr, pc - 1]:
            segments.append([(c, r), (c, r + 1)])
        if not padded[pr, pc + 1]:
            segments.append([(c + 1, r), (c + 1, r + 1)])
        if not padded[pr - 1, pc]:
            segments.append([(c, r), (c + 1, r)])
        if not padded[pr + 1, pc]:
            segments.append([(c, r + 1), (c + 1, r + 1)])
    return segments


def figure_release(out_dir, evaluation_dir, avalanche_id):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch
    import thesis_style as ts

    layers = run_release_estimation(evaluation_dir, avalanche_id)
    dem, roi = layers["dem"], layers["roi"]
    crown, candidates = layers["crown"], layers["candidates"]
    potential, release = layers["potential"], layers["release"]
    cell = layers["cell_size"]
    threshold = RELEASE_SETTINGS["crown_line_exit_distance"]

    exits = uphill_steps(dem, roi)
    distance = geodesic_distance(crown, roi, cell)
    # check: the release area holds the potential cells nearest to the crown
    target = min(round(RELEASE_SETTINGS["release_area_fraction"] * roi.sum()),
                 int(potential.sum()))
    order = np.argsort(np.where(potential, distance, np.inf), axis=None, kind="stable")
    ours = np.zeros(roi.size, bool)
    ours[order[:target]] = True
    ours = ours.reshape(roi.shape) & np.isfinite(distance)
    agree = (ours & release).sum() / max(release.sum(), 1)
    ours_candidates = {(r, c) for r, c, _, _, d in exits if d > threshold}
    engine_candidates = set(zip(*np.nonzero(candidates)))
    print(f"avalanche {avalanche_id}: {roi.sum()} outline cells, {potential.sum()} potential, "
          f"{candidates.sum()} crown candidates ({len(ours_candidates & engine_candidates)} "
          f"reproduced), {crown.sum()} crown cells, {release.sum()} release cells "
          f"({100 * agree:.1f} % reproduced by the geodesic fill), z_min {layers['z_min']:.0f} m")

    # crop to the outline with a margin
    rows, cols = np.nonzero(roi)
    margin = 3
    r_lo, r_hi = max(rows.min() - margin, 0), min(rows.max() + margin + 1, roi.shape[0])
    c_lo, c_hi = max(cols.min() - margin, 0), min(cols.max() + margin + 1, roi.shape[1])
    aspect = (r_hi - r_lo) / (c_hi - c_lo)
    full = (0, dem.shape[1], 0, dem.shape[0])

    colors = {"potential": ts.SKY, "release": ts.BLUE, "crown": ts.VERMILION,
              "removed": ts.ORANGE, "flank": "0.3", "outline": ts.BLACK}
    kinds = (("crown", "crown line $C$"),
             ("removed", "candidate removed by\nthe component filter"),
             ("flank", r"flank exit, $d_{\mathrm{out}} \leq \delta\,\Delta$"))

    def cells(ax, mask, color, alpha=1.0, zorder=3):
        ax.imshow(np.ma.masked_where(~mask, mask.astype(float)), cmap=ListedColormap([color]),
                  alpha=alpha, extent=full, origin="lower", interpolation="nearest",
                  zorder=zorder)

    def outline(ax, mask, lw=0.9, color=ts.BLACK, zorder=6):
        ax.add_collection(LineCollection(outline_edges(mask), colors=color, linewidths=lw,
                                         zorder=zorder, capstyle="projecting"))

    def scale_bar(ax, length_m):
        length = length_m / cell
        x0, y0 = c_hi - length - 1.5, r_lo + 1.5
        ax.plot([x0, x0 + length], [y0, y0], color=ts.BLACK, linewidth=1.5,
                solid_capstyle="butt", zorder=9)
        ax.text(x0 + length / 2, y0 + 0.6, f"{length_m:g} m", ha="center", va="bottom",
                fontsize=8, zorder=9)

    with ts.style():
        fig, axes = plt.subplots(2, 1, figsize=ts.size(1.0, 1.3 * aspect + 0.06),
                                 layout="constrained")
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ts.hillshade(ax, dem, full, 1.0)
            ax.set_xlim(c_lo, c_hi)
            ax.set_ylim(r_lo, r_hi)
            scale_bar(ax, 50)

        # (a) potential release cells, crown detection with the uphill steps
        ax = axes[0]
        cells(ax, potential, colors["potential"], 0.6)
        outline(ax, roi)
        for row, col, start, point, _ in exits:
            kind = ("crown" if crown[row, col] else
                    "removed" if candidates[row, col] else "flank")
            ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=colors[kind], alpha=0.45,
                                       edgecolor="none", zorder=4))
            ax.add_patch(FancyArrowPatch(
                tuple(start), tuple(point), arrowstyle="-|>,head_length=2.6,head_width=1.3",
                mutation_scale=1.0, color=colors[kind], linewidth=0.7, zorder=7,
                shrinkA=0.0, shrinkB=0.0))
        handles = [Line2D([], [], color=colors["outline"], linewidth=0.9,
                          label=r"outline $\mathcal{M}$"),
                   Patch(facecolor=colors["potential"], alpha=0.6,
                         label="potential release\ncells $P$")]
        handles += [Patch(facecolor=colors[k], alpha=0.6, label=text) for k, text in kinds]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0, fontsize=7, handlelength=1.4)
        ts.panel_label(ax, "a")

        # (b) geodesic distance to the crown line and the release area
        ax = axes[1]
        shown = np.ma.masked_invalid(np.where(roi & np.isfinite(distance), distance, np.nan))
        image = ax.imshow(shown, cmap="viridis", extent=full, origin="lower",
                          interpolation="nearest", zorder=3)
        outline(ax, roi, lw=0.6, color="0.3")
        outline(ax, release, lw=1.4, color=ts.BLACK, zorder=7)
        cells(ax, crown, colors["crown"], zorder=6)
        bar = fig.colorbar(image, ax=ax, location="right", shrink=0.9, pad=0.01, aspect=15)
        bar.set_label(r"geodesic distance $d$ (m)")
        ax.legend(handles=[Patch(facecolor=colors["crown"], label="crown line $C$"),
                           Line2D([], [], color=ts.BLACK, linewidth=1.4,
                                  label=r"release area $R$")],
                  loc="lower left", borderaxespad=0.4, fontsize=7, handlelength=1.4,
                  frameon=True, framealpha=0.9, edgecolor="none")
        ts.panel_label(ax, "b")

        ts.save(fig, out_dir, f"method_release_area_{avalanche_id}")


# ---------------------------------------------------------------------------
# figure 6: one step of the material point method


def bspline(eta):
    """Quadratic B-spline S(eta) of Eq. eq:bspline."""
    a = np.abs(eta)
    return np.where(a < 0.5, 0.75 - a ** 2, np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))


def figure_mpm(out_dir):
    import matplotlib.pyplot as plt
    import thesis_style as ts

    nodes = np.array([(i, j) for i in range(-2, 3) for j in range(-2, 3)], dtype=float)
    xp = np.array([0.3, -0.2])                # particle at step n
    xp_new = np.array([1.05, 0.25])           # particle after advection
    up, up_new = np.array([0.75, 0.3]), np.array([0.8, 0.5])
    others = np.array([[-1.3, 0.8], [-0.6, -1.2], [0.9, 1.1], [1.6, -0.9], [-1.5, -0.4],
                       [0.2, 0.9], [-0.4, 0.3], [1.25, -0.3]])

    def weights(x):
        eta = nodes - x
        return bspline(eta[:, 0]) * bspline(eta[:, 1])

    w, w_new = weights(xp), weights(xp_new)
    # nodal velocities: before (projected) and after the grid update
    rng = np.random.default_rng(3)
    node_mass = np.array([sum(weights(o)[k] for o in np.vstack([others, xp]))
                          for k in range(len(nodes))])
    has_mass = node_mass > 0.02
    u_before = 0.45 * np.array([1.0, 0.3]) + 0.12 * rng.standard_normal((len(nodes), 2))
    u_after = u_before + np.array([0.3, 0.15])  # gravity and pressure, less friction
    colors = {"p": ts.VERMILION, "node": ts.BLUE, "transfer": ts.GREEN, "u": ts.PURPLE}

    with ts.style():
        fig, axes = plt.subplots(2, 2, figsize=ts.size(1.0, 0.98), layout="constrained")
        for ax, title in zip(axes.flat, ("particle to grid", "grid update",
                                         "grid to particle", "advection")):
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            for v in np.arange(-2.5, 3.0, 1.0):
                ax.plot([v, v], [-2.5, 2.5], color="0.8", linewidth=0.5, zorder=0)
                ax.plot([-2.5, 2.5], [v, v], color="0.8", linewidth=0.5, zorder=0)
            ax.scatter(*nodes.T, marker="s", s=6, color="0.6", zorder=1)
            ax.set_title(title, fontsize=9)

        def stencil(ax, weights_, center, toward_particle):
            for node, wk in zip(nodes, weights_):
                if wk <= 0:
                    continue
                start, end = (node, center) if toward_particle else (center, node)
                vector(ax, start, end, colors["transfer"], lw=0.4 + 2.2 * wk, zorder=3,
                       shrink=3.0)
                ax.scatter(*node, marker="s", s=10 + 160 * wk, color=colors["node"],
                           alpha=0.85, zorder=4, linewidths=0)

        # (a) particle to grid
        ax = axes[0, 0]
        ax.scatter(*others.T, s=10, color="0.55", zorder=5)
        stencil(ax, w, xp, toward_particle=False)
        ax.scatter(*xp, s=26, color=colors["p"], zorder=6)
        vector(ax, xp, xp + up, colors["u"], lw=1.5, zorder=7)
        label(ax, xp + up + (0.1, 0.12), r"$\mathbf{u}_p, \mathbf{C}_p$", colors["u"],
              ha="left")
        label(ax, (0.0, -2.42), r"$m_i = \sum_p w_{ip}\, m_p$", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ts.panel_label(ax, "a", loc=(0.0, 1.06))

        # (b) grid update on the nodes with mass
        ax = axes[0, 1]
        for node, a, b in zip(nodes[has_mass], u_before[has_mass], u_after[has_mass]):
            vector(ax, node, node + 0.6 * a, "0.55", lw=0.8, zorder=3)
            vector(ax, node, node + 0.6 * b, colors["node"], lw=1.1, zorder=4)
        label(ax, (-0.1, -2.42), r"$\tilde{\mathbf{u}}_i$", "0.45", ha="right", va="bottom",
              fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        label(ax, (0.1, -2.42), r"$\rightarrow\ \mathbf{u}^{n+1}_i$", colors["node"],
              ha="left", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ts.panel_label(ax, "b", loc=(0.0, 1.06))

        # (c) grid to particle
        ax = axes[1, 0]
        stencil(ax, w, xp, toward_particle=True)
        ax.scatter(*xp, s=26, color=colors["p"], zorder=6)
        vector(ax, xp, xp + up_new, colors["u"], lw=1.5, zorder=7)
        label(ax, xp + up_new + (0.1, 0.12), r"$\mathbf{u}^{n+1}_p, \mathbf{C}^{n+1}_p$",
              colors["u"], ha="left")
        label(ax, (0.0, -2.42), r"$\mathbf{u}^{n+1}_p = \sum_i w_{ip}\, \mathbf{u}^{n+1}_i$",
              va="bottom", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ts.panel_label(ax, "c", loc=(0.0, 1.06))

        # (d) advection along the bed with the new stencil
        ax = axes[1, 1]
        for node, wk in zip(nodes, w):
            if wk > 0:
                ax.scatter(*node, marker="s", s=10 + 160 * wk, facecolor="none",
                           edgecolor="0.55", linewidths=0.6, zorder=3)
        for node, wk in zip(nodes, w_new):
            if wk > 0:
                ax.scatter(*node, marker="s", s=10 + 160 * wk, color=colors["node"],
                           alpha=0.85, zorder=4, linewidths=0)
        path = np.array([xp + s * (xp_new - xp) + (0, 0.12 * math.sin(math.pi * s))
                         for s in np.linspace(0, 1, 30)])
        ax.plot(*path.T, color=colors["p"], linewidth=1.0, linestyle="--", zorder=5)
        ax.scatter(*xp, s=26, facecolor="white", edgecolor=colors["p"], linewidths=1.0,
                   zorder=6)
        ax.scatter(*xp_new, s=26, color=colors["p"], zorder=6)
        label(ax, xp + (-0.1, -0.3), r"$\mathbf{x}_p^{n}$", colors["p"], ha="right")
        label(ax, xp_new + (0.12, -0.3), r"$\mathbf{x}_p^{n+1}$", colors["p"], ha="left")
        label(ax, (0.0, -2.42), "arc step and rotation", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ts.panel_label(ax, "d", loc=(0.0, 1.06))
        ts.save(fig, out_dir, "method_mpm_step")


# ---------------------------------------------------------------------------
# figures 7 and 8: loss components and center of mass of a simulated avalanche


def run_simulation(evaluation_dir, avalanche_id, mu, xi):
    """Runs one avalanche with the calibration settings of
    avalanchers_evaluation (calibration.config.BASE_SETTINGS) at mu, xi."""
    import sys

    import avalanchers
    import pandas as pd

    if str(evaluation_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_dir))
    from calibration.config import BASE_SETTINGS

    with working_directory(evaluation_dir):
        table = pd.read_csv("avalanche_outline_selection.csv").set_index("id")
        release_swe = float(table.loc[avalanche_id, "swe_diff"])
        outline = OUTLINES_SUBDIR / f"polygon_{avalanche_id}.shp"
        settings = dict(BASE_SETTINGS, outlines_path=str(outline), release_swe=release_swe)
        sim = avalanchers.PySimulation.new()
        sim.create(settings)
        dem, roi = np.asarray(sim.dem, float), np.asarray(sim.roi, bool)
        z_min = float(np.percentile(dem[roi], RELEASE_MIN_ELEVATION_PERCENTILE))
        sim = avalanchers.PySimulation.new()
        sim.create(dict(settings, release_min_elevation=z_min, friction_coefficient=mu,
                        drag_coefficient=xi))
        sim.prepare()
        sim.run()
        sim.post_process()
        evaluation = sim.evaluate()
        result = {
            "dem": np.asarray(sim.dem, float),
            "roi": np.asarray(sim.roi, bool),
            "release": np.asarray(sim.release_areas, float) > 0,
            "peak": np.asarray(sim.peak_flow_thickness, float),
            "com": np.stack([np.asarray(sim.center_of_mass_x, float),
                             np.asarray(sim.center_of_mass_y, float),
                             np.asarray(sim.center_of_mass_z, float)], axis=1),
            "cell_size": float(sim.cell_size),
            "summary": dict(sim.run_summary()),
            "engine": {key: float(evaluation[key]) for key in
                       ("iou", "chamfer", "beeline_3d", "beeline_3d_ref", "peak_velocity")},
            "mu": mu, "xi": xi,
        }
    return result


def footprint_metrics(run):
    """Footprints, losses (Eqs. eq:chamfer, eq:area-loss, eq:runout-loss) and
    the nearest-cell links of a run, recomputed from the result rasters."""
    from scipy.ndimage import distance_transform_edt

    cell = run["cell_size"]
    sim = run["peak"] >= FOOTPRINT_THRESHOLD
    obs, release = run["roi"], run["release"]
    sim_out, obs_out = sim & ~release, obs & ~release
    iou = (sim_out & obs_out).sum() / max((sim_out | obs_out).sum(), 1)
    d_obs, near_obs = distance_transform_edt(~obs, sampling=cell, return_indices=True)
    d_sim, near_sim = distance_transform_edt(~sim, sampling=cell, return_indices=True)
    term_sim = d_obs[sim_out].mean() if sim_out.any() else 0.0
    term_obs = d_sim[obs_out].mean() if obs_out.any() else 0.0

    def beeline(mask):
        z = np.where(mask, run["dem"], np.nan)
        hi = np.unravel_index(np.nanargmax(z), z.shape)
        lo = np.unravel_index(np.nanargmin(z), z.shape)
        horizontal = cell * math.hypot(hi[0] - lo[0], hi[1] - lo[1])
        return math.hypot(horizontal, z[hi] - z[lo]), hi, lo

    s_sim, hi_sim, lo_sim = beeline(sim)
    s_obs, hi_obs, lo_obs = beeline(obs)
    return {
        "sim": sim, "obs": obs, "release": release,
        "hit": sim_out & obs_out, "over": sim_out & ~obs, "miss": obs_out & ~sim,
        "d_obs": d_obs, "near_obs": near_obs, "d_sim": d_sim, "near_sim": near_sim,
        # L_dist relative to the observed length (Eq. eq:chamfer, as in
        # calibration/objective.py); the engine returns the chamfer in metres
        "L_area": 1.0 - iou, "L_dist": (term_sim + term_obs) / s_obs,
        "chamfer_m": term_sim + term_obs,
        "L_runout": abs(s_sim - s_obs) / s_obs, "iou": iou,
        "s_sim": s_sim, "s_obs": s_obs, "ends_sim": (hi_sim, lo_sim), "ends_obs": (hi_obs, lo_obs),
    }


def spread_cells(mask, distance, count, spacing):
    """Up to `count` cells of the mask with the largest distance, at least
    `spacing` cells apart."""
    rows, cols = np.nonzero(mask)
    order = np.argsort(-distance[rows, cols])
    picked = []
    for k in order:
        cell = (rows[k], cols[k])
        if all(math.hypot(cell[0] - r, cell[1] - c) >= spacing for r, c in picked):
            picked.append(cell)
        if len(picked) == count:
            break
    return picked


def crop_to(mask, margin):
    rows, cols = np.nonzero(mask)
    return (max(cols.min() - margin, 0), min(cols.max() + margin + 1, mask.shape[1]),
            max(rows.min() - margin, 0), min(rows.max() + margin + 1, mask.shape[0]))


def map_axes(ax, dem, view, cell, bar_m=50, contour_levels=12):
    import thesis_style as ts
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ts.hillshade(ax, dem, (0, dem.shape[1], 0, dem.shape[0]), 1.0,
                 contour_levels=contour_levels)
    c_lo, c_hi, r_lo, r_hi = view
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_lo, r_hi)
    length = bar_m / cell
    x0, y0 = c_hi - length - 1.5, r_lo + 1.5
    ax.plot([x0, x0 + length], [y0, y0], color="black", linewidth=1.5,
            solid_capstyle="butt", zorder=9)
    ax.text(x0 + length / 2, y0 + 0.6, f"{bar_m:g} m", ha="center", va="bottom",
            fontsize=8, zorder=9)


def mask_image(ax, mask, color, alpha=1.0, zorder=3):
    from matplotlib.colors import ListedColormap
    ax.imshow(np.ma.masked_where(~mask, mask.astype(float)), cmap=ListedColormap([color]),
              alpha=alpha, extent=(0, mask.shape[1], 0, mask.shape[0]), origin="lower",
              interpolation="nearest", zorder=zorder)


def figure_loss(out_dir, evaluation_dir, avalanche_id):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch
    import thesis_style as ts

    runs = [run_simulation(evaluation_dir, avalanche_id, mu, xi) for mu, xi in LOSS_RUNS]
    metrics = [footprint_metrics(run) for run in runs]
    for run, m in zip(runs, metrics):
        print(f"loss run mu {run['mu']}, xi {run['xi']}: IoU {m['iou']:.3f} "
              f"(engine {run['engine']['iou']:.3f}), chamfer "
              f"{m['chamfer_m']:.2f} m (engine {run['engine']['chamfer']:.2f} m), "
              f"L_dist {m['L_dist']:.4f}, s_sim {m['s_sim']:.0f} m "
              f"(engine {run['engine']['beeline_3d']:.0f}), s_obs {m['s_obs']:.0f} m "
              f"(engine {run['engine']['beeline_3d_ref']:.0f})")

    dem, cell = runs[0]["dem"], runs[0]["cell_size"]
    view = crop_to(metrics[0]["obs"] | metrics[0]["sim"] | metrics[1]["sim"], 3)
    aspect = (view[3] - view[2]) / (view[1] - view[0])
    colors = {"hit": ts.GREEN, "over": ts.VERMILION, "miss": ts.BLUE, "release": "0.35",
              "sim": ts.VERMILION, "obs": ts.BLUE}

    with ts.style():
        fig, axes = plt.subplots(2, 1, figsize=ts.size(1.0, 1.3 * aspect + 0.06),
                                 layout="constrained")
        for ax, run, m, letter in zip(axes, runs, metrics, "ab"):
            map_axes(ax, dem, view, cell)
            mask_image(ax, m["hit"], colors["hit"], 0.55)
            mask_image(ax, m["over"], colors["over"], 0.55)
            mask_image(ax, m["miss"], colors["miss"], 0.45)
            mask_image(ax, m["release"], colors["release"], 0.55)
            ax.add_collection(LineCollection(outline_edges(m["obs"]), colors=colors["obs"],
                                             linewidths=0.8, zorder=6))
            ax.add_collection(LineCollection(outline_edges(m["sim"]), colors=colors["sim"],
                                             linewidths=0.8, zorder=6))
            # nearest-cell distances of the Chamfer distance
            for key, distance, nearest, color in (("over", "d_obs", "near_obs", colors["sim"]),
                                                  ("miss", "d_sim", "near_sim", colors["obs"])):
                for r, c in spread_cells(m[key], m[distance], 5, 8):
                    target = (m[nearest][1][r, c] + 0.5, m[nearest][0][r, c] + 0.5)
                    ax.add_patch(FancyArrowPatch(
                        (c + 0.5, r + 0.5), target, arrowstyle="-|>,head_length=3,head_width=1.6",
                        mutation_scale=1.0, color=color, linewidth=0.9, zorder=8))
            # avalanche length s between the highest and the lowest cell
            for key, color in (("ends_obs", colors["obs"]), ("ends_sim", colors["sim"])):
                (hr, hc), (lr, lc) = m[key]
                ax.plot([hc + 0.5, lc + 0.5], [hr + 0.5, lr + 0.5], color=color,
                        linewidth=1.0, linestyle=(0, (4, 2)), zorder=7)
                ax.plot(hc + 0.5, hr + 0.5, marker="^", color=color, markersize=4, zorder=8)
                ax.plot(lc + 0.5, lr + 0.5, marker="v", color=color, markersize=4, zorder=8)
            ax.text(0.015, 0.03,
                    rf"$\mu = {run['mu']:g}$, $\xi = {run['xi']:g}$ m s$^{{-2}}$" "\n"
                    rf"$L_{{\mathrm{{area}}}} = {m['L_area']:.2f}$, "
                    rf"$L_{{\mathrm{{dist}}}} = {m['L_dist']:.3f}$, "
                    rf"$L_{{\mathrm{{runout}}}} = {m['L_runout']:.2f}$",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7, zorder=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))
            ts.panel_label(ax, letter)
        handles = [
            Patch(facecolor=colors["release"], alpha=0.55, label="release area $R$"),
            Patch(facecolor=colors["hit"], alpha=0.55,
                  label=r"$A'_{\mathrm{sim}} \cap A'_{\mathrm{obs}}$"),
            Patch(facecolor=colors["over"], alpha=0.55, label="simulated only"),
            Patch(facecolor=colors["miss"], alpha=0.45, label="observed only"),
            Line2D([], [], color=colors["obs"], linewidth=0.8, label=r"outline $A_{\mathrm{obs}}$"),
            Line2D([], [], color=colors["sim"], linewidth=0.8, label=r"footprint $A_{\mathrm{sim}}$"),
            Line2D([], [], color="0.3", linewidth=0.9, marker=">", markersize=3,
                   label="distance to the\nother footprint"),
            Line2D([], [], color="0.3", linewidth=1.0, linestyle=(0, (4, 2)),
                   label="length $s$, highest\nto lowest cell"),
        ]
        axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       borderaxespad=0.0, fontsize=7, handlelength=1.4)
        ts.save(fig, out_dir, f"method_loss_{avalanche_id}")


def figure_com(out_dir, evaluation_dir, avalanche_id):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import thesis_style as ts

    mu, xi = LOSS_RUNS[0]
    run = run_simulation(evaluation_dir, avalanche_id, mu, xi)
    m = footprint_metrics(run)
    cell = run["cell_size"]
    com = run["com"]
    com = com[np.all(np.isfinite(com), axis=1)]
    xy = com[:, :2] / cell                      # metres from the DEM corner -> cells
    steps = np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1)
    s2d = np.concatenate([[0.0], np.cumsum(steps)])
    s3d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(com, axis=0), axis=1))])
    drop = com[0, 2] - com[-1, 2]
    alpha = math.degrees(math.atan2(drop, s2d[-1]))
    summary = run["summary"]
    print(f"COM: s_2D {s2d[-1]:.1f} m (engine {summary['travel_length_2d']:.1f}), s_3D "
          f"{s3d[-1]:.1f} m (engine {summary['travel_length_3d']:.1f}), drop {drop:.1f} m "
          f"(engine {summary['travel_vertical_drop']:.1f}), alpha {alpha:.1f} deg "
          f"(engine {summary['travel_angle']:.1f}), s of the footprint {m['s_sim']:.0f} m")

    from matplotlib.ticker import MaxNLocator

    view = crop_to(m["obs"] | m["sim"], 3)
    width_cells, height_cells = view[1] - view[0], view[3] - view[2]
    colors = {"sim": ts.SKY, "com": ts.BLACK, "s": ts.VERMILION, "alpha": ts.BLUE,
              "release": ts.PURPLE}
    # (b) in equal axes: its limits set its height; its elevation ticks are
    # the contour levels of (a)
    x_lim = (-0.08 * s2d[-1], 1.22 * s2d[-1])
    z_lim = (com[:, 2].min() - 0.2 * drop, com[:, 2].max() + 0.1 * drop)
    z_ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(*z_lim)
    z_ticks = z_ticks[(z_ticks >= z_lim[0]) & (z_ticks <= z_lim[1])]
    map_ratio = height_cells / width_cells
    profile_ratio = (z_lim[1] - z_lim[0]) / (x_lim[1] - x_lim[0])

    with ts.style():
        # the legend right of (a) leaves about 0.68 of the width to the axes
        fig = plt.figure(figsize=ts.size(1.0, 0.68 * (map_ratio + profile_ratio) + 0.14),
                         layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=[map_ratio, profile_ratio])
        ax = fig.add_subplot(grid[0])
        map_axes(ax, run["dem"], view, cell, contour_levels=None)
        # contour only the view, so clabel places the labels inside it; the
        # levels continue the tick spacing of (b) over the whole view
        c_lo, c_hi, r_lo, r_hi = view
        dem_view = run["dem"][r_lo:r_hi, c_lo:c_hi]
        step = z_ticks[1] - z_ticks[0]
        levels = z_ticks[0] + step * np.arange(
            math.floor((np.nanmin(dem_view) - z_ticks[0]) / step),
            math.ceil((np.nanmax(dem_view) - z_ticks[0]) / step) + 1)
        contours = ax.contour(np.arange(c_lo, c_hi) + 0.5, np.arange(r_lo, r_hi) + 0.5,
                              dem_view, levels=levels, colors="0.35", linewidths=0.45,
                              zorder=4)
        for text in ax.clabel(contours, fmt="%.0f", fontsize=6, inline=True,
                              inline_spacing=2):
            text.set_zorder(5)
        mask_image(ax, m["sim"], colors["sim"], 0.6)
        ax.add_collection(LineCollection(outline_edges(m["obs"]), colors="0.2",
                                         linewidths=0.7, zorder=6))
        ax.add_collection(LineCollection(outline_edges(m["release"]), colors=colors["release"],
                                         linewidths=1.0, zorder=6))
        ax.plot(*xy.T, color=colors["com"], linewidth=1.2, zorder=8)
        ax.plot(*xy[0], "o", color=colors["com"], markersize=4, zorder=9)
        ax.plot(*xy[-1], "s", color=colors["com"], markersize=4, zorder=9)
        (hr, hc), (lr, lc) = m["ends_sim"]
        ax.plot([hc + 0.5, lc + 0.5], [hr + 0.5, lr + 0.5], color=colors["s"], linewidth=1.0,
                linestyle=(0, (4, 2)), zorder=7)
        ax.plot(hc + 0.5, hr + 0.5, marker="^", color=colors["s"], markersize=4, zorder=8)
        ax.plot(lc + 0.5, lr + 0.5, marker="v", color=colors["s"], markersize=4, zorder=8)
        ax.legend(handles=[
            Patch(facecolor=colors["sim"], alpha=0.6, label=r"footprint $A_{\mathrm{sim}}$"),
            Line2D([], [], color="0.2", linewidth=0.7, label="outline"),
            Line2D([], [], color=colors["release"], linewidth=1.0, label="release area $R$"),
            Line2D([], [], color="0.35", linewidth=0.45, label="contour lines (m)"),
            Line2D([], [], color=colors["com"], linewidth=1.2, marker="o", markersize=3,
                   label=f"COM path, $s_{{\\mathrm{{3D}}}} = {s3d[-1]:.0f}$ m"),
            Line2D([], [], color=colors["s"], linewidth=1.0, linestyle=(0, (4, 2)),
                   label=f"length $s = {m['s_sim']:.0f}$ m"),
        ], loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=7,
            handlelength=1.6)
        ts.panel_label(ax, "a")

        # (b) elevation of the COM along its horizontal path
        ax = fig.add_subplot(grid[1])
        ax.plot(s2d, com[:, 2], color=colors["com"], linewidth=1.2)
        ax.plot(s2d[0], com[0, 2], "o", color=colors["com"], markersize=4)
        ax.plot(s2d[-1], com[-1, 2], "s", color=colors["com"], markersize=4)
        ax.plot([0, s2d[-1]], [com[0, 2], com[-1, 2]], color=colors["alpha"], linewidth=0.9,
                linestyle=(0, (4, 2)))
        ax.plot([0, s2d[-1]], [com[-1, 2]] * 2, color="0.5", linewidth=0.5, linestyle=":")
        ax.annotate("", xy=(s2d[-1] * 1.02, com[-1, 2]), xytext=(s2d[-1] * 1.02, com[0, 2]),
                    arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.15",
                                    color="0.25", linewidth=0.6, shrinkA=0, shrinkB=0))
        ax.text(s2d[-1] * 1.035, 0.5 * (com[0, 2] + com[-1, 2]),
                rf"$\Delta Z = {drop:.0f}$ m", ha="left", va="center", fontsize=8)
        ax.text(0.5 * s2d[-1], com[-1, 2] - 0.05 * drop,
                rf"$s_{{\mathrm{{2D}}}} = {s2d[-1]:.0f}$ m", ha="center", va="top", fontsize=8)
        radius = 0.16 * s2d[-1]
        ax.text(radius * 1.1, com[0, 2] - 0.06 * drop, rf"$\alpha = {alpha:.1f}^\circ$",
                color=colors["alpha"], ha="left", va="top", fontsize=8)
        ax.set_xlabel(r"horizontal travel distance of the COM (m)")
        ax.set_ylabel(r"elevation $Z$ (m)")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*z_lim)
        ax.set_yticks(z_ticks)
        ax.set_aspect("equal")
        ts.panel_label(ax, "b")
        ts.save(fig, out_dir, f"method_center_of_mass_{avalanche_id}")


# ---------------------------------------------------------------------------
# figure 9: friction laws


def friction_deceleration(law, speed, h, g_n, p=None):
    """Basal friction deceleration a_f (m/s^2) of the friction laws, as in
    basal_friction_acceleration of grid_physics_terrain_following.wgsl
    (chi = 1)."""
    p = dict(FRICTION_PARAMETERS, **(p or {}))
    u = np.asarray(speed, float)
    if law == "Coulomb":
        return np.full_like(u, p["mu"] * g_n)
    if law == "Voellmy":
        return p["mu"] * g_n + G * u ** 2 / (p["xi"] * h)
    if law == "samosAT":
        rs = u ** 2 / (g_n * h)
        drag = (math.log(max(h / 0.05, 1.0)) / 0.43 + 4.13) ** 2
        return p["mu"] * g_n * (1.0 + 0.222 / (0.222 + rs)) + u ** 2 / (h * drag)
    if law == r"$\mu(I)$":
        inertial = 2.5 * p["d_g"] * u / (h * math.sqrt(g_n * h))
        mu_i = p["mu_s"] + (p["mu_2"] - p["mu_s"]) / (p["I_0"] / np.maximum(inertial, 1e-12) + 1)
        return mu_i * g_n
    raise ValueError(law)


def steady_speed(law, theta_deg, h, u_max=120.0):
    """Speed at which friction balances gravity on a plane: 0 if the flow
    stops, inf if it accelerates without bound below u_max."""
    theta = math.radians(theta_deg)
    g_n, drive = G * math.cos(theta), G * math.sin(theta)

    def excess(u):
        return float(friction_deceleration(law, np.array([u]), h, g_n)[0]) - drive

    if excess(1e-6) >= 0:
        return 0.0
    if excess(u_max) < 0:
        return math.inf
    lo, hi = 1e-6, u_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def figure_friction(out_dir):
    import matplotlib.pyplot as plt
    import thesis_style as ts

    laws = ("Coulomb", "Voellmy", "samosAT", r"$\mu(I)$")
    colors = dict(zip(laws, (ts.BLACK, ts.BLUE, ts.GREEN, ts.ORANGE)))
    theta, depths = 35.0, (1.0, 3.0)
    g_n = G * math.cos(math.radians(theta))
    speed = np.linspace(0.0, 60.0, 400)
    angles = np.linspace(10.0, 50.0, 321)

    with ts.style():
        fig, axes = plt.subplots(1, 2, figsize=ts.size(1.0, 0.42), layout="constrained")
        ax = axes[0]
        for law in laws:
            for h, style in zip(depths, ("-", "--")):
                if law == "Coulomb" and h != depths[0]:
                    continue
                ax.plot(speed, friction_deceleration(law, speed, h, g_n), color=colors[law],
                        linestyle=style, label=law if h == depths[0] else None)
        ax.axhline(G * math.sin(math.radians(theta)), color="0.45", linewidth=0.8,
                   linestyle=":")
        ax.text(1.0, G * math.sin(math.radians(theta)) + 0.15, r"$g \sin\theta$",
                color="0.35", fontsize=8, va="bottom")
        ax.set_xlabel(r"speed $\Vert\mathbf{u}\Vert$ (m s$^{-1}$)")
        ax.set_ylabel(r"friction deceleration $a_f$ (m s$^{-2}$)")
        ax.set_xlim(0, speed[-1])
        ax.set_ylim(0, 14)
        ax.set_title(rf"$\theta = {theta:g}^\circ$")
        ts.panel_label(ax, "a")

        ax = axes[1]
        for law in laws[1:]:
            for h, style in zip(depths, ("-", "--")):
                u = np.array([steady_speed(law, a, h) for a in angles])
                ax.plot(angles, np.where(np.isfinite(u), u, np.nan), color=colors[law],
                        linestyle=style)
        mu = FRICTION_PARAMETERS["mu"]
        # onset of motion: tan(theta) = mu for Coulomb and Voellmy, 2 mu for
        # samosAT at rest, mu_s for mu(I)
        for value, text, y, side in ((mu, r"$\arctan\mu$", 44, -1),
                                     (FRICTION_PARAMETERS["mu_s"], r"$\arctan\mu_s$", 59, -1),
                                     (2 * mu, r"$\arctan 2\mu$", 59, 1)):
            angle = math.degrees(math.atan(value))
            ax.axvline(angle, color="0.45", linewidth=0.6, linestyle=":")
            ax.text(angle + 0.45 * side, y, text, fontsize=7, color="0.35", va="top",
                    ha="left" if side > 0 else "right", rotation=90)
        ax.set_xlabel(r"slope angle $\theta$ ($^\circ$)")
        ax.set_ylabel(r"steady speed (m s$^{-1}$)")
        ax.set_xlim(angles[0], angles[-1])
        ax.set_ylim(0, 60)
        ax.set_title("steady flow on a plane")
        ts.panel_label(ax, "b")

        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=colors[law], label=law) for law in laws]
        handles += [Line2D([], [], color=ts.NEUTRAL, linestyle=style, label=f"$h = {h:g}$ m")
                    for h, style in zip(depths, ("-", "--"))]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   handlelength=1.8, columnspacing=1.2)
        ts.save(fig, out_dir, "method_friction_laws")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--avalanche-id", type=int, default=DEFAULT_AVALANCHE_ID,
                        help="avalanche of the release and loss figures "
                             f"(default {DEFAULT_AVALANCHE_ID})")
    parser.add_argument("--com-avalanche-id", type=int, default=DEFAULT_COM_AVALANCHE_ID,
                        help="avalanche of the center of mass figure "
                             f"(default {DEFAULT_COM_AVALANCHE_ID})")
    parser.add_argument("--evaluation-dir", type=pathlib.Path, default=EVALUATION_DIR,
                        help="avalanchers_evaluation checkout with the outlines and "
                             "dtm_cache.zarr")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out_dir = args.out_dir.resolve()

    if "gully" in args.figures:
        figure_gully(out_dir)
    if "bed" in args.figures:
        figure_bed(out_dir)
    if "arc" in args.figures:
        figure_arc(out_dir)
    if "release" in args.figures:
        figure_release(out_dir, args.evaluation_dir.resolve(), args.avalanche_id)
    if "mpm" in args.figures:
        figure_mpm(out_dir)
    if "loss" in args.figures:
        figure_loss(out_dir, args.evaluation_dir.resolve(), args.avalanche_id)
    if "com" in args.figures:
        figure_com(out_dir, args.evaluation_dir.resolve(), args.com_avalanche_id)
    if "friction" in args.figures:
        figure_friction(out_dir)


if __name__ == "__main__":
    main()
