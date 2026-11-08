"""Gully confinement tests for the reference-surface split of the grid models.

Runs avalanchers in-script on synthetic DEMs (no input files) once without
and once with ``reference_surface_length`` (the Gaussian width of the surface
the flow follows; narrower terrain becomes confining basal topography).

1. Wedge friction. A V-notch channel cut into an inclined plane, Coulomb
   friction. Snow filling a V-groove whose walls stand at beta' against the
   plane presses on them with its lateral stress; a rigid wedge needs
   tan(theta) > mu / cos(beta') to slide (K = 1). The mean along-slope
   acceleration a gives the effective friction factor
   f = (sin - a / g) / (mu cos), 1 on an open slope and 1 / cos(beta') for
   the ideal wedge. The unsplit model loads each wall only with the column
   above it (f < 1); the split model must raise f above 1. It stays below
   the ideal wedge: the depth-averaged equations carry no cross-slope shear
   stress, so the gully core slides past stalled walls, and the V kink sits
   between two nodes that see half the wall slope.
2. Pool at rest. A closed, horizontal gully filled to a flat free surface
   with frictionless snow. Any motion is spurious; the peak mass-weighted rms
   speed is reported for both runs (information only: frictionless pools
   already slosh at about 0.5 m/s without the split).

Plots go to verification/outputs/gully_test/. ``--sim-model`` selects the
curvilinear (default), terrain-following or 3d-depth-averaged model.
"""

import argparse
import math
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "verification" / "outputs" / "gully_test"
SIM_INFO_STOPPED = 1 << 31

SETTINGS = {
    "sim_model": "curvilinear",
    "friction_model": "coulomb",
    "friction_coefficient": 0.4,
    # pure Coulomb: no turbulent drag
    "drag_coefficient": 1.0e12,
    "density": 200.0,
    "released_particles_per_cell": 4,
    "cfl": 0.2,
    "velocity_threshold": 0.01,
    "enable_particle_interaction": True,
    # the wedge reference assumes hydrostatic lateral stress (K = 1)
    "enable_earth_pressure_coefficient": False,
    "enable_particle_relaxation": False,
    # window of 3000 steps: a stalling flow must not end the run early
    "no_new_cells_travel": 600.0,
    "max_steps": 20000,
}

GULLY_TEST_CFG = {
    # frictionless pools turn unstable on 0.5 m cells with or without the split
    "cell_size": 1.0,
    "reference_lengths": (0.0, 16.0),
    # wedge friction
    "wall_angle": 45.0,          # V-notch walls against the horizontal (deg)
    "notch_half_width": 12.0,    # m
    "fill_depth": 5.0,           # vertical depth at the thalweg (m)
    "release_x": (40.0, 160.0),  # m along the channel
    "domain": (500.0, 100.0),    # x (downslope) and y extent (m)
    "stall_slope": 24.0,         # deg, mu < tan < mu / cos(beta'): a wedge stalls
    "slide_slope": 34.0,         # deg, beyond the wedge limit
    "t_end": 10.0,
    "fit_window": (2.0, 8.0),
    # pool at rest
    "pool_depth": 8.0,           # gully depth (m)
    "pool_width": 5.0,           # Gaussian half width of the gully (m)
    "pool_length": 60.0,         # half length of the closed gully (m)
    "pool_fill": 3.0,            # free surface above the thalweg (m)
    "pool_domain": (240.0, 80.0),
    "pool_t_end": 10.0,
}


def grid(cfg, domain):
    """Cell-centre coordinates; rows run south to north (row 0 = ymin)."""
    cs = cfg["cell_size"]
    nx, ny = round(domain[0] / cs), round(domain[1] / cs)
    x = (np.arange(nx) + 0.5) * cs
    y = (np.arange(ny) + 0.5) * cs - domain[1] / 2.0
    bounds = (0.0, nx * cs, -domain[1] / 2.0, domain[1] / 2.0)
    return (*np.meshgrid(x, y), bounds)


def bed_normal(depth, dem, cs):
    """Vertical depth -> bed-normal release thickness (mass = rho h dx^2 / cos)."""
    gy, gx = np.gradient(dem, cs)
    return depth / np.sqrt(1.0 + gx**2 + gy**2)


def v_channel(cfg, slope_deg):
    """Inclined plane (downhill = +x) with a V-notch along the x axis."""
    x, y, bounds = grid(cfg, cfg["domain"])
    tan_beta = math.tan(math.radians(cfg["wall_angle"]))
    half_width = cfg["notch_half_width"]
    notch = tan_beta * (np.minimum(np.abs(y), half_width) - half_width)
    dem = (cfg["domain"][0] - x) * math.tan(math.radians(slope_deg)) + notch
    depth = np.clip(cfg["fill_depth"] - tan_beta * np.abs(y), 0.0, None)
    x0, x1 = cfg["release_x"]
    depth[(x < x0) | (x > x1)] = 0.0
    return dem, bed_normal(depth, dem, cfg["cell_size"]), bounds


def closed_gully(cfg):
    """Horizontal plain with a closed, elongated Gaussian gully."""
    x, y, bounds = grid(cfg, cfg["pool_domain"])
    x_c = cfg["pool_domain"][0] / 2.0
    taper = np.exp(-(((x - x_c) / cfg["pool_length"]) ** 8))
    dem = -cfg["pool_depth"] * np.exp(-(y**2) / (2.0 * cfg["pool_width"] ** 2)) * taper
    level = -cfg["pool_depth"] + cfg["pool_fill"]
    depth = np.clip(level - dem, 0.0, None)
    return dem, bed_normal(depth, dem, cfg["cell_size"]), bounds


def run(dem, release, bounds, cfg, settings, t_end, batch=10):
    """Run to t_end (or a stop) sampling mass-weighted particle velocities."""
    import avalanchers

    sim = avalanchers.PySimulation.new()
    sim.create(dict(settings))
    sim.set_dem_with_bounds(np.ascontiguousarray(dem.astype(np.float32)),
                            cfg["cell_size"], *bounds, 1.0)
    sim.set_release_areas(np.ascontiguousarray(release.astype(np.float32)))
    sim.prepare()
    series = {"t": [], "mean_u": [], "rms_speed": []}
    while True:
        info = sim.run_n_steps(batch)
        vel = np.asarray(sim.particles_velocity_xy, dtype=float)
        mass = np.asarray(sim.particles_mass, dtype=float)
        n = min(len(vel), len(mass))
        vel, mass = vel[:n], mass[:n]
        ok = np.isfinite(vel).all(axis=1) & (mass > 0)
        vel, mass = vel[ok], mass[ok]
        series["t"].append(info.elapsed_time)
        series["mean_u"].append(float((mass * vel[:, 0]).sum() / mass.sum()))
        speed2 = vel[:, 0] ** 2 + vel[:, 1] ** 2
        series["rms_speed"].append(float(np.sqrt((mass * speed2).sum() / mass.sum())))
        stopped = (info.flags & SIM_INFO_STOPPED) or sim.state == "Finished"
        if info.elapsed_time >= t_end or stopped:
            break
    return {k: np.asarray(v) for k, v in series.items()}


def wedge_limits(cfg, slope_deg, mu, g=9.81):
    """Along-slope accelerations of the open slope and the K = 1 wedge."""
    theta = math.radians(slope_deg)
    # wall inclination against the inclined plane: tan b' = tan b cos theta
    beta_eff = math.atan(math.tan(math.radians(cfg["wall_angle"])) * math.cos(theta))
    a_open = g * (math.sin(theta) - mu * math.cos(theta))
    a_wedge = g * (math.sin(theta) - mu * math.cos(theta) / math.cos(beta_eff))
    return a_open, a_wedge, math.degrees(beta_eff)


def along_slope(series, slope_deg, surface_velocity=False):
    """Mean x velocity -> mean along-slope speed. A horizontal dx/dt needs the
    slope metric, a surface velocity (terrain-following) already measures it."""
    if surface_velocity:
        return series["mean_u"]
    return series["mean_u"] / math.cos(math.radians(slope_deg))


def fit_acceleration(series, slope_deg, window, surface_velocity=False):
    v = along_slope(series, slope_deg, surface_velocity)
    mask = (series["t"] >= window[0]) & (series["t"] <= window[1])
    if mask.sum() < 3:
        return float("nan")
    return float(np.polyfit(series["t"][mask], v[mask], 1)[0])


def plot(results, cfg, mu, out_dir, show, sim_model="curvilinear"):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colors = {length: c for length, c in zip(cfg["reference_lengths"], ("#6c6155", "#0d6876"))}
    for ax, key in zip(axes[:2], ("stall", "slide")):
        slope = cfg[f"{key}_slope"]
        a_open, a_wedge, beta_eff = wedge_limits(cfg, slope, mu)
        for length in cfg["reference_lengths"]:
            s = results[key][length]
            ax.plot(s["t"], along_slope(s, slope, cfg["surface_velocity"]),
                    color=colors[length],
                    label=f"L = {length:g} m")
        t = np.linspace(0, cfg["t_end"], 2)
        ax.plot(t, np.maximum(a_open * t, 0), ":", color="0.4", label="open slope")
        ax.plot(t, np.maximum(a_wedge * t, 0), "--", color="#b4500d", label="wedge, K = 1")
        ax.set_title(f"V-channel {slope:g} deg, mu = {mu:g}, beta' = {beta_eff:.1f} deg")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mean along-slope speed (m/s)")
        ax.legend(frameon=False)
    ax = axes[2]
    for length in cfg["reference_lengths"]:
        s = results["pool"][length]
        ax.plot(s["t"], s["rms_speed"], color=colors[length], label=f"L = {length:g} m")
    ax.set_title("closed gully at rest, frictionless")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mass-weighted rms speed (m/s)")
    ax.legend(frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gully_test_{sim_model}.png"
    fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-length", type=float,
                        default=GULLY_TEST_CFG["reference_lengths"][1],
                        help="reference_surface_length of the split run (m)")
    parser.add_argument("--mu", type=float, default=SETTINGS["friction_coefficient"])
    parser.add_argument("--sim-model", default=SETTINGS["sim_model"],
                        help="curvilinear | terrain-following | 3d-depth-averaged")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    from similarity_solution_test import SURFACE_VELOCITY_MODELS

    cfg = dict(GULLY_TEST_CFG)
    cfg["surface_velocity"] = args.sim_model.lower() in SURFACE_VELOCITY_MODELS
    cfg["reference_lengths"] = (0.0, args.reference_length)
    mu = args.mu
    results = {"stall": {}, "slide": {}, "pool": {}}
    for length in cfg["reference_lengths"]:
        settings = dict(SETTINGS, sim_model=args.sim_model, friction_coefficient=mu,
                        reference_surface_length=length)
        for key in ("stall", "slide"):
            dem, release, bounds = v_channel(cfg, cfg[f"{key}_slope"])
            results[key][length] = run(dem, release, bounds, cfg, settings, cfg["t_end"])
        dem, release, bounds = closed_gully(cfg)
        pool_settings = dict(settings, friction_coefficient=0.0)
        results["pool"][length] = run(dem, release, bounds, cfg, pool_settings,
                                      cfg["pool_t_end"])

    split = cfg["reference_lengths"][1]
    print(f"\n=== Gully test ({args.sim_model}, Coulomb mu = {mu:g}, split L = {split:g} m) ===")
    print("| case | L (m) | final speed (m/s) | acceleration (m/s^2) | friction factor f "
          "| open slope a, f | wedge a, f |")
    print("|---|---|---|---|---|---|---|")
    checks = []
    for key in ("stall", "slide"):
        slope = cfg[f"{key}_slope"]
        theta = math.radians(slope)
        a_open, a_wedge, beta_eff = wedge_limits(cfg, slope, mu)
        f_wedge = 1.0 / math.cos(math.radians(beta_eff))
        factor = {}
        for length in cfg["reference_lengths"]:
            s = results[key][length]
            v_end = float(along_slope(s, slope, cfg["surface_velocity"])[-1])
            acc = fit_acceleration(s, slope, cfg["fit_window"], cfg["surface_velocity"])
            factor[length] = (math.sin(theta) - acc / 9.81) / (mu * math.cos(theta))
            print(f"| {key} {slope:g} deg | {length:g} | {v_end:.2f} | {acc:.2f} "
                  f"| {factor[length]:.2f} | {a_open:.2f}, 1.00 | {a_wedge:.2f}, {f_wedge:.2f} |")
        checks.append((f"{key}: unsplit walls carry less than open-slope friction (f < 1)",
                       factor[0.0] < 1.0))
        checks.append((f"{key}: split raises the friction factor above 1 and the unsplit run",
                       factor[split] > max(1.0, factor[0.0])))
        checks.append((f"{key}: split friction factor does not exceed the ideal wedge",
                       factor[split] <= f_wedge + 0.05))
    pool_rms = {length: float(results["pool"][length]["rms_speed"].max())
                for length in cfg["reference_lengths"]}
    print(f"\npool at rest (frictionless): peak rms speed {pool_rms[0.0]:.2f} m/s at L = 0, "
          f"{pool_rms[split]:.2f} m/s at L = {split:g} m (information only)")

    path = plot(results, cfg, mu, args.out_dir, args.show, args.sim_model)
    print(f"\nplot: {path}")
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
