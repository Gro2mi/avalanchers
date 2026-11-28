"""Granular column collapse validation (Lajeunesse et al. 2004).

The validation case of Kammholz et al. (2026, Sec. 3.2): a cylindrical column
of glass beads with radius R_i = 70.5 mm and height H_i = 56.4 mm (aspect
ratio a = H_i / R_i = 0.8) collapses and spreads on a rough horizontal plane.
Coulomb friction, mu = 0.61 (calibrated by Kammholz et al.), 1 mm cells, an
exact cylinder (cells whose center lies inside R_i) without edge smoothing.

Froude scaling. With Coulomb friction and hydrostatic pressure the
depth-averaged equations have no length scale of their own: lengths scale with
R_i, times with sqrt(R_i / g), speeds with sqrt(g R_i) and the deposit is the
same at any size (Mangeney-Castelnau et al. 2005, who ran this case up to
1e4 R_i). The engine has absolute thresholds (1 mm of snow for the CFL nodes,
a 150 m/s speed limit, a 50 m quantization reference column, 1e-2 kg grid
decode floor) that do not suit a 1 mm grid holding 1e-5 kg per cell. The
experiment is therefore run SCALE times larger (default 1000: 1 m cells,
R_i = 70.5 m) and all results are scaled back to laboratory units. The
resting speed threshold is scaled with sqrt(SCALE). ``--scale 1`` runs the
laboratory size directly.

References used (Mangeney-Castelnau et al. 2005, JGR 110, B09103, Sec. 2 and
Sec. 4, summarizing Lajeunesse et al. 2004):
- for a > 0.74 the experimental final height saturates at H_f / R_i ~ 0.74
  (the deposit is a cone, the undisturbed top is consumed);
- the front stops in the laboratory at t / sqrt(R_i / g) ~ 2.7 (~230 ms);
- the final radius R_f is measured where the deposit is thicker than two
  bead diameters (2 x 0.35 mm).
The measured deposit profile is not bundled. Digitize it (e.g. from
Kammholz et al. 2026, Fig. 6a, or Lajeunesse et al. 2004) into a CSV with the
columns ``r_mm,h_mm`` and pass it with ``--experiment-csv`` to draw it in the
cross-section and get the profile error.

Outputs (verification/outputs/granular_collapse_test/):
- granular_collapse_profile: (a) mirrored radial cross-section of the
  deposit at several times and at rest, (b) top view of the final deposit
  with the initial column and the mean front radius;
- granular_collapse_front: front radius and center height over time;
- granular_collapse_orientation: front radius against direction, the grid
  orientation check of Kammholz et al. (circular deposit);
- summary.json with all scalar results.

Usage:
    python run_granular_collapse_test.py
    python run_granular_collapse_test.py --mu 0.61 --particles-per-cell 4
    python run_granular_collapse_test.py --experiment-csv lajeunesse_a08.csv
    python run_granular_collapse_test.py --sim-model curvilinear
"""

import argparse
import json
import math
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "verification" / "outputs" / "granular_collapse_test"
SIM_INFO_STOPPED = 1 << 31
GRAVITY = 9.81

SETTINGS = {
    "enable_entrainment": False,
    "sim_model": "dampm-global-cartesian",
    "friction_model": "coulomb",
    "friction_coefficient": 0.61,
    # pure Coulomb: no turbulent drag (ignored by the Coulomb law anyway)
    "drag_coefficient": 1.0e12,
    # density cancels with Coulomb friction and hydrostatic pressure
    "density": 200.0,
    # One particle alone makes a column of H_i / (N_ppc dx). Above about
    # 3 cell sizes its own central-difference pressure gradient beats
    # Coulomb friction and it never rests (lone-particle self-propulsion),
    # so stray particles creep to the border and the run never stops. With
    # H_i / dx = 56 that needs N_ppc >= 19; 4 (the paper's value) gives 14.
    "released_particles_per_cell": 25,
    # isotropic stress (K = 1), as in Kammholz et al. and Mangeney et al.
    "enable_earth_pressure_coefficient": False,
    # flat bed: no reference surface (detachment defaults to none)
    "reference_surface_length": 0.0,
    "detachment": "none",
    "particle_placement": "uniform-relaxed",
    "max_steps": 200000,
}

COLLAPSE_CFG = {
    # laboratory geometry (Lajeunesse et al. 2004, case a = 0.8)
    "radius": 0.0705,          # R_i [m]
    "height": 0.0564,          # H_i [m]
    "cell_size": 0.001,        # [m], as in Kammholz et al.
    "domain_half": 0.22,       # half width of the square domain [m] (~3 R_i)
    "bead_diameter": 0.35e-3,  # [m]; front cutoff = 2 bead diameters
    # run control, laboratory units
    "t_end": 0.6,              # [s]; the lab front stops at ~0.23 s
    "save_interval": 0.01,     # [s]
    "velocity_threshold": 1.0e-3,  # resting speed [m/s]
    # Froude scale factor of the simulated column (see module docstring)
    "scale": 1000.0,
    # references (Mangeney-Castelnau et al. 2005, Sec. 2 and 4)
    "ref_height_ratio": 0.74,  # H_f / R_i for a > 0.74
    "ref_arrest_time": 2.7,    # t / sqrt(R_i / g) when the front stops
    # cross-section snapshots, laboratory seconds
    "profile_times": (0.03, 0.06, 0.1, 0.15),
    "sector_width_deg": 5.0,
}

BATCH_STEPS = 20


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def build_case(cfg):
    """Flat DEM and cylindrical release in SIMULATION units (scaled).

    Returns dem, release (thickness per cell), bounds, the cell-center
    radius r (lab m), the direction angle (rad) and the scaled cell size.
    """
    s = cfg["scale"]
    cs_lab = cfg["cell_size"]
    n = 2 * int(round(cfg["domain_half"] / cs_lab))
    half = n * cs_lab / 2.0
    centers = (np.arange(n) + 0.5) * cs_lab - half
    x, y = np.meshgrid(centers, centers)  # row 0 = south
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    dem = np.zeros((n, n))
    release = np.where(r <= cfg["radius"], cfg["height"] * s, 0.0)
    bounds = (-half * s, half * s, -half * s, half * s)
    return dem, release, bounds, r, theta, cs_lab * s


def outermost_cell_radius(h, r, cutoff):
    """Largest radius of any cell thicker than `cutoff` (lab m), 0 if none.
    Picks up single stray particles: one particle alone already makes a bump
    of about H_i / N_ppc."""
    wet = h >= cutoff
    return float(r[wet].max()) if wet.any() else 0.0


def profile_front(h, r, cutoff, bin_width, mask=None):
    """Front radius of the azimuthally averaged deposit: the outer edge of the
    first radial bin, from the center outward, whose mean thickness falls
    below `cutoff`. Isolated particles beyond the continuous deposit do not
    move it."""
    if mask is not None:
        h, r = h[mask], r[mask]
    rb, hb = radial_profile(h, r, bin_width, float(r.max()))
    # empty bins (narrow sectors near the center hold no cell center) say
    # nothing about the deposit: only bins with cells can end it
    below = np.nonzero(np.isfinite(hb) & (hb < cutoff))[0]
    if len(below) == 0:
        return float(rb[-1])
    return float(rb[below[0]] - 0.5 * bin_width)


def front_radius(h, r, cutoff, bin_width):
    return profile_front(h, r, cutoff, bin_width)


def sector_front_radii(h, r, theta, cutoff, width_deg, bin_width):
    """Profile front radius per direction sector: (sector centers deg, radii)."""
    edges = np.radians(np.arange(-180.0, 180.0 + 1e-9, width_deg))
    centers, radii = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (theta >= lo) & (theta < hi)
        centers.append(math.degrees(0.5 * (lo + hi)))
        radii.append(profile_front(h, r, cutoff, bin_width, mask=sel))
    return np.asarray(centers), np.asarray(radii)


def radial_profile(h, r, bin_width, r_max):
    """Mean thickness in radial bins: (bin centers, mean h), lab units."""
    edges = np.arange(0.0, r_max + bin_width, bin_width)
    idx = np.digitize(r.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[valid], weights=h.ravel()[valid], minlength=len(edges) - 1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), mean


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------

def thickness_lab(sim, shape, cs_sim, density, scale):
    """Flow thickness (lab m) from the grid mass; flat bed, so cos(theta) = 1."""
    mass = np.asarray(sim.grid_mass, dtype=np.float64).reshape(shape)
    return mass / (density * cs_sim * cs_sim) / scale


def run(settings, cfg):
    """Run the collapse and sample the grid thickness every save_interval."""
    import avalanchers

    s = cfg["scale"]
    t_scale = math.sqrt(s)  # Froude: t_sim = sqrt(scale) t_lab
    dem, release, bounds, r, theta, cs_sim = build_case(cfg)
    settings = dict(settings)
    settings["velocity_threshold"] = cfg["velocity_threshold"] * t_scale

    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.set_dem_with_bounds(np.ascontiguousarray(dem.astype(np.float32)),
                            cs_sim, *bounds, 1.0)
    sim.set_release_areas(np.ascontiguousarray(release.astype(np.float32)))
    sim.prepare()

    density = settings["density"]
    cutoff = 2.0 * cfg["bead_diameter"]
    n_release = int((release > 0).sum())
    r_eff = math.sqrt(n_release * cfg["cell_size"] ** 2 / math.pi)
    print(f"=== Granular column collapse (Lajeunesse et al. 2004), "
          f"{settings['sim_model']}, mu = {settings['friction_coefficient']:.3f} ===")
    print(f"R_i = {1e3 * cfg['radius']:.1f} mm (pixelated: {1e3 * r_eff:.2f} mm), "
          f"H_i = {1e3 * cfg['height']:.1f} mm, a = {cfg['height'] / cfg['radius']:.3f}, "
          f"{dem.shape[1]} x {dem.shape[0]} cells of {1e3 * cfg['cell_size']:g} mm, "
          f"scale {s:g}, {settings['released_particles_per_cell']} particles per cell")

    series = {"t": [], "front": [], "h_center": [], "max_speed": []}
    snapshots = {}
    wanted = sorted(cfg["profile_times"])
    next_save = 0.0
    last_step = -1
    info = sim.run_n_steps(1)
    print(f"\n{'t [ms]':>7} {'steps':>7} {'R_f [mm]':>9} {'h(0) [mm]':>10} "
          f"{'u_max [m/s]':>12}")
    while True:
        t_lab = info.grid_elapsed_time / t_scale
        if t_lab >= next_save - 1e-9:
            h = thickness_lab(sim, dem.shape, cs_sim, density, s)
            rf = front_radius(h, r, cutoff, 2.0 * cfg["cell_size"])
            hc = float(h[r <= 2.0 * cfg["cell_size"]].mean())
            umax = info.max_velocity / t_scale
            series["t"].append(t_lab)
            series["front"].append(rf)
            series["h_center"].append(hc)
            series["max_speed"].append(umax)
            for tw in wanted:
                if tw not in snapshots and t_lab >= tw - 1e-9:
                    snapshots[tw] = (t_lab, h.copy())
            print(f"{1e3 * t_lab:7.1f} {info.timestep:7d} {1e3 * rf:9.2f} "
                  f"{1e3 * hc:10.2f} {umax:12.4f}")
            while next_save <= t_lab + 1e-9:
                next_save += cfg["save_interval"]
        if (info.flags & SIM_INFO_STOPPED) or sim.state == "Finished":
            break
        if t_lab >= cfg["t_end"]:
            break
        if info.timestep == last_step:
            print("simulation made no progress - stopping")
            break
        last_step = info.timestep
        info = sim.run_n_steps(BATCH_STEPS)

    h_final = thickness_lab(sim, dem.shape, cs_sim, density, s)
    # particle positions are relative to the lower-left grid corner
    particles_xy = (np.asarray(sim.particles_position_xy, dtype=np.float64)
                    + [bounds[0], bounds[2]]) / s
    t_final = info.grid_elapsed_time / t_scale
    total = float(np.asarray(sim.grid_mass, dtype=np.float64).sum())
    released = float(release.sum()) * cs_sim ** 2 * density
    print(f"\nstopped after {info.timestep} steps at t = {1e3 * t_final:.1f} ms "
          f"(stopped flag: {bool(info.flags & SIM_INFO_STOPPED)}); grid mass "
          f"{100.0 * total / released:.2f} % of the released mass")
    for k in series:
        series[k] = np.asarray(series[k])
    return {"h_final": h_final, "t_final": t_final, "series": series,
            "snapshots": snapshots, "r": r, "theta": theta, "r_eff": r_eff,
            "mass_fraction": total / released, "release": release / s,
            "particles_xy": particles_xy, "cell_size": cfg["cell_size"],
            "stopped": bool(info.flags & SIM_INFO_STOPPED)}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def arrest_time(series, cfg):
    """Time at which the front stops advancing: the first sample at which the
    front radius reaches its maximum. Afterwards the thin edge only thins
    below the cutoff (the front retreats by a few bins), which a 'last
    change' criterion would mistake for motion."""
    t, front = series["t"], series["front"]
    return float(t[int(np.argmax(front))])


def load_experiment(path):
    """Digitized profile CSV (columns r_mm, h_mm) -> (r m, h m), mirrored."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    r_exp = np.asarray(data["r_mm"], dtype=float) * 1e-3
    h_exp = np.asarray(data["h_mm"], dtype=float) * 1e-3
    order = np.argsort(r_exp)
    return r_exp[order], h_exp[order]


def analyze(result, cfg, experiment=None):
    cutoff = 2.0 * cfg["bead_diameter"]
    h, r, theta = result["h_final"], result["r"], result["theta"]
    ri = cfg["radius"]
    bin_width = 2.0 * cfg["cell_size"]
    rf = front_radius(h, r, cutoff, bin_width)
    r_cell = outermost_cell_radius(h, r, cutoff)
    centers, radii = sector_front_radii(h, r, theta, cutoff, cfg["sector_width_deg"],
                                        bin_width)
    outside = float(h[r > rf].sum() / h.sum()) if h.sum() > 0 else 0.0
    # sectors around the grid axes (0, 90, 180, 270 deg) vs the diagonals
    ang = np.mod(centers, 90.0)
    axis = (ang < 10.0) | (ang > 80.0)
    diag = np.abs(ang - 45.0) < 10.0
    hc = float(h[r <= 2.0 * cfg["cell_size"]].mean())
    t_arrest = arrest_time(result["series"], cfg)
    t_unit = math.sqrt(ri / GRAVITY)
    summary = {
        "aspect_ratio": cfg["height"] / ri,
        "front_cutoff_mm": 1e3 * cutoff,
        "final_radius_mm": 1e3 * rf,
        "runout_ratio_Rf_Ri": rf / ri,
        "outermost_cell_radius_mm": 1e3 * r_cell,
        "mass_fraction_beyond_front": outside,
        "sector_radius_mean_mm": 1e3 * float(np.nanmean(radii)),
        "sector_radius_std_mm": 1e3 * float(np.nanstd(radii)),
        "sector_radius_min_mm": 1e3 * float(np.nanmin(radii)),
        "sector_radius_max_mm": 1e3 * float(np.nanmax(radii)),
        "axis_minus_diagonal_mm": 1e3 * float(np.nanmean(radii[axis]) - np.nanmean(radii[diag])),
        "final_height_mm": 1e3 * hc,
        "height_ratio_Hf_Ri": hc / ri,
        "height_ratio_reference": cfg["ref_height_ratio"],
        "arrest_time_ms": 1e3 * t_arrest,
        "arrest_time_scaled": t_arrest / t_unit,
        "arrest_time_scaled_reference": cfg["ref_arrest_time"],
        "end_time_ms": 1e3 * result["t_final"],
        "stopped_by_engine": result["stopped"],
        "grid_mass_fraction": result["mass_fraction"],
    }
    if experiment is not None:
        r_exp, h_exp = experiment
        rb, hb = radial_profile(h, r, cfg["cell_size"], r.max())
        h_sim = np.interp(r_exp, rb[np.isfinite(hb)], hb[np.isfinite(hb)])
        summary["profile_rmse_mm"] = 1e3 * float(np.sqrt(np.mean((h_sim - h_exp) ** 2)))
        summary["profile_l2_rel"] = float(np.linalg.norm(h_sim - h_exp) / np.linalg.norm(h_exp))
    return summary, (centers, radii)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def interpolate_cell_centers(field, domain_half, cell_size, xy):
    """Bilinear interpolation of a cell-centered field (row 0 = south) at xy."""
    n_y, n_x = field.shape
    fx = np.clip((xy[:, 0] + domain_half) / cell_size - 0.5, 0.0, n_x - 1.0)
    fy = np.clip((xy[:, 1] + domain_half) / cell_size - 0.5, 0.0, n_y - 1.0)
    i0 = np.minimum(fx.astype(int), n_x - 2)
    j0 = np.minimum(fy.astype(int), n_y - 2)
    wx, wy = fx - i0, fy - j0
    return ((1 - wx) * (1 - wy) * field[j0, i0] + wx * (1 - wy) * field[j0, i0 + 1]
            + (1 - wx) * wy * field[j0 + 1, i0] + wx * wy * field[j0 + 1, i0 + 1])


def plot_profile(result, cfg, summary, experiment, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import thesis_style as ts

    ri, hi = cfg["radius"], cfg["height"]
    r = result["r"]
    r_max = 0.95 * cfg["domain_half"]
    snaps = [result["snapshots"][k] for k in sorted(result["snapshots"])]
    colors = ts.time_ramp(ts.VARIABLE_COLORS["h"], len(snaps) + 1)

    # cut both panels to the deposit (largest sector front) with a small
    # margin; the sub-cutoff halo and the few strays beyond it are clipped
    xy = result["particles_xy"]
    h_p = interpolate_cell_centers(result["h_final"], cfg["domain_half"],
                                   cfg["cell_size"], xy)
    lim = summary["sector_radius_max_mm"] + 5.0
    r_max = min(r_max, 1e-3 * lim)
    h_top = 1.1e3 * hi
    with ts.style():
        # Manual layout: both panels have the same width and the same scale
        # (mm per inch) on both axes, so the cross-section and the top view
        # are drawn 1:1 and share their x axis exactly.
        width = ts.size(1.0)[0]
        left, right, bottom, gap, top = 0.55, 0.1, 0.4, 0.08, 0.5
        axis_width = width - left - right
        inch_per_mm = axis_width / (2.0 * lim)
        h_profile = h_top * inch_per_mm
        h_map = 2.0 * lim * inch_per_mm
        height = bottom + h_map + gap + h_profile + top
        fig = plt.figure(figsize=(width, height))
        ax1 = fig.add_axes((left / width, bottom / height, axis_width / width, h_map / height))
        ax0 = fig.add_axes((left / width, (bottom + h_map + gap) / height,
                            axis_width / width, h_profile / height), sharex=ax1)
        # (a) mirrored cross-section
        ax0.plot(1e3 * np.array([-ri, -ri, ri, ri]), 1e3 * np.array([0, hi, hi, 0]),
                 color=ts.NEUTRAL, linestyle=":", lw=0.9, label="initial column")
        for (t, h), c in zip(snaps, colors[:-1]):
            rb, hb = radial_profile(h, r, cfg["cell_size"], r_max)
            ax0.plot(1e3 * np.r_[-rb[::-1], rb], 1e3 * np.r_[hb[::-1], hb], color=c,
                     lw=0.9, label=f"$t = {1e3 * t:.0f}$ ms")
        rb, hb = radial_profile(result["h_final"], r, cfg["cell_size"], r_max)
        ax0.plot(1e3 * np.r_[-rb[::-1], rb], 1e3 * np.r_[hb[::-1], hb], color=colors[-1],
                 lw=1.4, label=f"deposit ($t = {summary['end_time_ms']:.0f}$ ms)")
        if experiment is not None:
            r_exp, h_exp = experiment
            ax0.plot(1e3 * np.r_[-r_exp[::-1], r_exp], 1e3 * np.r_[h_exp[::-1], h_exp],
                     color=ts.BLACK, linestyle="--", lw=1.0, label="experiment")
        ax0.plot(0.0, 1e3 * cfg["ref_height_ratio"] * ri, marker="_", markersize=12,
                 color=ts.VERMILION, mew=1.5, linestyle="none",
                 label=r"$H_f = 0.74\,R_i$ (exp.)")
        ax0.set_xlim(-lim, lim)
        ax0.set_ylim(0.0, h_top)
        ax0.set_aspect("equal")
        ax0.tick_params(labelbottom=False)
        ax0.set_ylabel("$h$ (mm)")
        ax0.legend(fontsize=7, frameon=False, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0, columnspacing=1.2)
        ts.panel_label(ax0, "a")

        # (b) top view of the deposit: every particle, colored by the flow
        # thickness interpolated to its position
        order = np.argsort(h_p)
        im = ax1.scatter(1e3 * xy[order, 0], 1e3 * xy[order, 1], c=1e3 * h_p[order],
                         s=0.15, linewidths=0, cmap="viridis", vmin=0.0,
                         vmax=1e3 * float(h_p.max()), rasterized=True)
        ax1.add_patch(Circle((0, 0), 1e3 * ri, fill=False, color=ts.NEUTRAL,
                             linestyle=":", lw=0.9))
        ax1.add_patch(Circle((0, 0), summary["sector_radius_mean_mm"], fill=False,
                             color=ts.VERMILION, lw=0.9))
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_aspect("equal")
        ax1.set_xlabel("$x$, $r$ (mm)")
        ax1.set_ylabel("$y$ (mm)")
        # inset in the empty upper right corner, so the panel keeps the width
        # of the cross-section above it
        cax = ax1.inset_axes((0.80, 0.93, 0.17, 0.025))
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("$h$ (mm)", fontsize=7, labelpad=1)
        cbar.ax.tick_params(labelsize=6, length=2, pad=1)
        cax.xaxis.set_label_position("top")
        ts.panel_label(ax1, "b")
        return ts.save(fig, out_dir, "granular_collapse_profile")


def plot_front(result, cfg, summary, out_dir):
    import matplotlib.pyplot as plt
    import thesis_style as ts

    s = result["series"]
    t_unit = math.sqrt(cfg["radius"] / GRAVITY)
    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(0.7, 0.6), constrained_layout=True)
        ax.plot(s["t"] / t_unit, s["front"] / cfg["radius"], color=ts.VARIABLE_COLORS["h"],
                label="front radius $R_f / R_i$")
        ax.plot(s["t"] / t_unit, s["h_center"] / cfg["radius"], color=ts.VARIABLE_COLORS["speed"],
                label="center height $h(0) / R_i$")
        ax.axhline(cfg["ref_height_ratio"], color=ts.VERMILION, linestyle="--", lw=0.8,
                   label=r"$H_f / R_i = 0.74$ (exp.)")
        ax.axvline(cfg["ref_arrest_time"], color=ts.NEUTRAL, linestyle=":", lw=0.8,
                   label="front stops (exp.)")
        ax.axvline(summary["arrest_time_scaled"], color=ts.VARIABLE_COLORS["h"],
                   linestyle=":", lw=0.8, label="front stops (sim.)")
        ax.set_xlabel(r"$t / \sqrt{R_i / g}$")
        ax.set_ylabel("normalized length")
        ax.legend(fontsize=7, frameon=False)
        return ts.save(fig, out_dir, "granular_collapse_front")


def plot_orientation(sectors, summary, out_dir):
    import matplotlib.pyplot as plt
    import thesis_style as ts

    centers, radii = sectors
    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(0.7, 0.5), constrained_layout=True)
        ax.plot(centers, 1e3 * radii, marker="o", markersize=2.5, lw=0.8,
                color=ts.VARIABLE_COLORS["h"], label="front radius per sector")
        ax.axhline(summary["sector_radius_mean_mm"], color=ts.NEUTRAL, linestyle="--",
                   lw=0.8, label="mean")
        for a in (-90, 0, 90):
            ax.axvline(a, color="0.85", lw=0.6, zorder=0)
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_xlabel("direction (deg, 0 = +x grid axis)")
        ax.set_ylabel("$R_f$ (mm)")
        ax.legend(fontsize=7, frameon=False)
        return ts.save(fig, out_dir, "granular_collapse_orientation")


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="granular column collapse validation")
    parser.add_argument("--sim-model", default=None)
    parser.add_argument("--mu", type=float, default=None,
                        help="Coulomb friction coefficient (default 0.61)")
    parser.add_argument("--cell-size", type=float, default=None,
                        help="laboratory cell size in m (default 0.001)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Froude scale factor of the simulated column (default 1000)")
    parser.add_argument("--t-end", type=float, default=None, help="laboratory seconds")
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--weak-pressure", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="weak-form pressure gradient (curvilinear and global Cartesian model)")
    parser.add_argument("--experiment-csv", default=None,
                        help="digitized deposit profile, columns r_mm,h_mm")
    parser.add_argument("--max-steps", type=int, default=None)
    from cli_args import add_particles_per_cell_argument
    add_particles_per_cell_argument(parser)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    settings = dict(SETTINGS)
    cfg = dict(COLLAPSE_CFG)
    if args.sim_model:
        settings["sim_model"] = args.sim_model
    if args.mu is not None:
        settings["friction_coefficient"] = args.mu
    if args.cell_size is not None:
        cfg["cell_size"] = args.cell_size
    if args.scale is not None:
        cfg["scale"] = args.scale
    if args.t_end is not None:
        cfg["t_end"] = args.t_end
    if args.cfl is not None:
        settings["cfl"] = args.cfl
    if args.weak_pressure is not None:
        settings["enable_weak_pressure"] = args.weak_pressure
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.particles_per_cell is not None:
        settings["released_particles_per_cell"] = args.particles_per_cell

    experiment = load_experiment(args.experiment_csv) if args.experiment_csv else None
    result = run(settings, cfg)
    summary, sectors = analyze(result, cfg, experiment)
    summary["settings"] = {k: v for k, v in settings.items()}
    summary["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}

    out_dir = pathlib.Path(args.out_dir)
    plot_profile(result, cfg, summary, experiment, out_dir)
    plot_front(result, cfg, summary, out_dir)
    plot_orientation(sectors, summary, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"final radius R_f      : {summary['final_radius_mm']:.1f} mm "
          f"(R_f / R_i = {summary['runout_ratio_Rf_Ri']:.3f}, cutoff "
          f"{summary['front_cutoff_mm']:.2f} mm, azimuthal mean profile)")
    print(f"beyond the front      : {100 * summary['mass_fraction_beyond_front']:.2f} % of the "
          f"mass, outermost cell at {summary['outermost_cell_radius_mm']:.1f} mm")
    print(f"front radius by sector: mean {summary['sector_radius_mean_mm']:.1f} mm, "
          f"std {summary['sector_radius_std_mm']:.2f} mm, range "
          f"[{summary['sector_radius_min_mm']:.1f}, {summary['sector_radius_max_mm']:.1f}] mm, "
          f"axes - diagonals {summary['axis_minus_diagonal_mm']:+.2f} mm")
    print(f"final center height   : {summary['final_height_mm']:.1f} mm "
          f"(H_f / R_i = {summary['height_ratio_Hf_Ri']:.3f}, experiment ~0.74)")
    print(f"front arrest          : {summary['arrest_time_ms']:.0f} ms "
          f"(t / sqrt(R_i/g) = {summary['arrest_time_scaled']:.2f}, experiment ~2.7)")
    if experiment is not None:
        print(f"profile vs experiment : RMSE {summary['profile_rmse_mm']:.2f} mm, "
              f"relative L2 {summary['profile_l2_rel']:.3f}")
    return summary


if __name__ == "__main__":
    main()
