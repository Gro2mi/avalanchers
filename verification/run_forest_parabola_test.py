"""Forest model on a parabolic slope (NAKSIN / MoT-Voellmy type forest).

Snow released across most of a parabolic slope (uniform across the slope)
runs into the lower slope and the transition into the flat runout:

    z(x) = BOTTOM + c (VERTEX_X - x)^2   for x < VERTEX_X, BOTTOM beyond

A forest band (canopy cover COVER between FOREST_X) covers only the northern
half of the slope (FOREST_Y), so the southern half shows the same avalanche
without forest side by side. Runouts are measured per half. The forest is
described by the settings
forest_trunk_diameter D and forest_crown_diameter d_c: stem density
n = cover / (pi/4 d_c^2). The stems decelerate the flow by
a = 1/2 C_D n D min(h, H)/h u^2 and break once the bending moment
1/2 rho C_D D h_eff^2 u^2 / 2 exceeds sigma pi D^3 / 32, i.e. once

    u h_eff >= sqrt(sigma pi D^2 / (8 rho C_D)),

after which the standing cover decays at the rate c_f / D (forest.wgsl). The
test stand is young (D = 0.15 m) so that the avalanche breaks part of it.

Runs (dampm-global-cartesian, Voellmy):
  a) no forest
  b) forest that cannot break (modulus of rupture 1e12 Pa)
  c) forest with breaking
  d) forest enabled with zero cover everywhere

Checks:
  - in the forested half runout(b) < runout(c) < runout(a), and in c) the
    forested half stops short of the open half: the forest shortens the runout and
    its destruction weakens that effect
  - d) reproduces a): the forest code is inert without trees (up to the
    run-to-run scatter of the GPU, a few cm in single peak cells)
  - in c) stems only break where the peak values allow the breaking load:
    peak(u) min(peak(h), H) must reach the threshold above (necessary
    condition, the peaks need not coincide in time; the breaking load uses
    the same node velocity and thickness the peak maps record), and cells
    far below it keep their full cover
  - b) never breaks a stem

Outputs (verification/outputs/forest_parabola_test/):
  - forest_parabola_profiles: centreline peak flow thickness, peak velocity
    and standing cover fraction of all runs over the bed profile
  - forest_parabola_destruction: top view of the destroyed cover fraction
  - summary.json

With --export the case is also written in the avaframe format for the live
viewer (data/avaframe/avaParabolaForest{,releaseTexture,forestCover}.png,
.aabb) together with verification/forest_parabola_settings.json:

    cargo run --release -p render_core --example live_sim -- verification/forest_parabola_settings.json

Key 9 then shows the standing forest cover (again: the initial cover).

Usage:
    python verification/run_forest_parabola_test.py
    python verification/run_forest_parabola_test.py --export
"""

import argparse
import json
import math
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "verification" / "outputs" / "forest_parabola_test"
CASE = "avaParabolaForest"
SIM_INFO_STOPPED = 1 << 31

CELL_SIZE = 5.0  # m
LENGTH = 1800.0  # m along the fall line (x)
WIDTH = 400.0  # m across the slope (y)
BOTTOM = 1500.0  # m, elevation of the parabola vertex; 0 is the PNG no-data value
VERTEX_X = 1000.0  # m
DROP = 500.0  # m, height of the parabola at x = 0 above its vertex

RELEASE_X = (60.0, 200.0)  # m
RELEASE_Y = (40.0, 360.0)  # m, across both halves
RELEASE_THICKNESS = 1.5  # m

# the forest covers the northern half only, so the southern half shows the
# same avalanche without forest side by side
FOREST_X = (600.0, 1150.0)  # m
FOREST_Y = (WIDTH / 2.0, WIDTH)  # m
COVER = 0.8
# profiles compared: the middle of the forested and of the open half
FOREST_ROW_Y = 0.5 * (RELEASE_Y[1] + FOREST_Y[0])
OPEN_ROW_Y = 0.5 * (RELEASE_Y[0] + FOREST_Y[0])

SETTINGS = {
    "enable_entrainment": False,
    "sim_model": "dampm-global-cartesian",
    "friction_model": "voellmy",
    "friction_coefficient": 0.2,
    "drag_coefficient": 2000.0,
    "density": 200.0,
    "enable_curvature": True,
    "enable_particle_interaction": True,
    "released_particles_per_cell": 8,
    "particle_placement": "uniform-relaxed",
    "cfl": 0.5,
    "max_steps": 20000,
    "velocity_threshold": 0.1,
    "momentum_stop_fraction": 0.01,
    # forest of the test: a young stand that the avalanche can break
    "forest_drag_coefficient": 1.0,
    "forest_trunk_diameter": 0.15,
    "forest_crown_diameter": 4.0,
    "forest_tree_height": 15.0,
    "forest_modulus_of_rupture": 50.0e6,
    "forest_decay_coefficient": 0.15,
}
UNBREAKABLE = 1.0e12  # Pa
BATCH_STEPS = 200


def parabola(x):
    x = np.asarray(x, dtype=float)
    return np.where(x < VERTEX_X, BOTTOM + DROP / VERTEX_X**2 * (VERTEX_X - x) ** 2, BOTTOM)


def build_case():
    """DEM, release thickness and forest cover; row 0 = south (y = 0)."""
    nx = int(round(LENGTH / CELL_SIZE))
    ny = int(round(WIDTH / CELL_SIZE))
    x = (np.arange(nx) + 0.5) * CELL_SIZE
    y = (np.arange(ny) + 0.5) * CELL_SIZE
    xx, yy = np.meshgrid(x, y)
    dem = parabola(xx).astype(np.float32)
    release = np.where(
        (xx >= RELEASE_X[0]) & (xx < RELEASE_X[1]) & (yy >= RELEASE_Y[0]) & (yy < RELEASE_Y[1]),
        RELEASE_THICKNESS, 0.0).astype(np.float32)
    in_forest = ((xx >= FOREST_X[0]) & (xx < FOREST_X[1])
                 & (yy >= FOREST_Y[0]) & (yy < FOREST_Y[1]))
    cover = np.where(in_forest, COVER, 0.0).astype(np.float32)
    return x, y, dem, release, cover


def breaking_product(settings):
    """u h_eff (m^2/s) at which the bending moment reaches the rupture moment."""
    d = settings["forest_trunk_diameter"]
    return math.sqrt(settings["forest_modulus_of_rupture"] * math.pi * d * d
                     / (8.0 * settings["density"] * settings["forest_drag_coefficient"]))


def run(name, settings, cover):
    import avalanchers

    x, y, dem, release, _ = build_case()
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.set_dem_with_bounds(np.ascontiguousarray(dem), CELL_SIZE,
                            0.0, LENGTH, 0.0, WIDTH, 1.0)
    sim.set_release_areas(np.ascontiguousarray(release))
    if cover is not None:
        sim.set_forest_cover(np.ascontiguousarray(cover))
    sim.prepare()

    info = sim.run_n_steps(1)
    last_step = -1
    while not (info.flags & SIM_INFO_STOPPED) and sim.state != "Finished":
        if info.timestep == last_step:
            break
        last_step = info.timestep
        info = sim.run_n_steps(BATCH_STEPS)

    peak_h = np.asarray(sim.peak_flow_thickness, dtype=np.float64)
    peak_u = np.asarray(sim.peak_velocity, dtype=np.float64)
    standing = sim.forest_standing
    standing = None if standing is None else np.asarray(standing, dtype=np.float64)
    forest_half = y >= FOREST_Y[0]
    runout = runout_x(x, peak_h[forest_half])
    runout_open = runout_x(x, peak_h[~forest_half])
    print(f"{name:<24} {info.timestep:6d} steps, t = {info.grid_elapsed_time:6.1f} s, runout x "
          f"{runout:7.1f} m (forest half) {runout_open:7.1f} m (open half)")
    return {"peak_h": peak_h, "peak_u": peak_u, "standing": standing, "runout": runout,
            "runout_open": runout_open, "steps": int(info.timestep),
            "time": float(info.grid_elapsed_time), "stopped": bool(info.flags & SIM_INFO_STOPPED)}


def runout_x(x, peak_h, threshold=0.1):
    """Largest x of a cell whose peak flow thickness exceeds the threshold."""
    columns = np.where((peak_h > threshold).any(axis=0))[0]
    return float(x[columns.max()]) if columns.size else float("nan")


def check(results, settings, cover):
    checks = {}
    a, b, c, d = (results[k] for k in ("no_forest", "unbreakable", "breakable", "zero_cover"))
    checks["forest shortens the runout (b < a)"] = b["runout"] < a["runout"]
    checks["forested half stops short of the open half (c)"] = c["runout"] < c["runout_open"]
    checks["destruction weakens the forest (b < c)"] = b["runout"] < c["runout"]
    checks["destroyed forest still brakes (c < a)"] = c["runout"] < a["runout"]
    # repeated identical runs scatter on the GPU (atomic summation order, up
    # to about 2 cm in single peak cells, below 1 mm in 99.9 % of them), so
    # the comparison allows for that but not for a drag (metres, m/s)
    dh = np.abs(a["peak_h"] - d["peak_h"])
    du = np.abs(a["peak_u"] - d["peak_u"])
    checks["zero cover reproduces no forest"] = (
        a["runout"] == d["runout"]
        and dh.max() < 0.1 and np.percentile(dh, 99.9) < 5e-3
        and du.max() < 0.1 and np.percentile(du, 99.9) < 1e-2)
    forested = cover > 0
    checks["unbreakable forest stays standing"] = bool(
        np.allclose(b["standing"][forested], cover[forested]))

    product = c["peak_u"] * np.minimum(c["peak_h"], settings["forest_tree_height"])
    threshold = breaking_product(settings)
    destroyed = np.zeros_like(cover, dtype=np.float64)
    destroyed[forested] = 1.0 - c["standing"][forested] / cover[forested]
    broken = destroyed > 0.05
    checks["the breakable forest breaks somewhere"] = bool(broken.any())
    checks["stems break only where u h_eff reaches the breaking load"] = bool(
        np.all(product[destroyed > 0.0] >= 0.999 * threshold))
    far_below = forested & (product < 0.5 * threshold)
    checks["stems far below the breaking load keep their cover"] = bool(
        np.all(destroyed[far_below] == 0.0))
    stats = {
        "breaking_u_h_eff_m2_s": threshold,
        "forested_cells": int(forested.sum()),
        "broken_cells": int(broken.sum()),
        "mean_destroyed_fraction_in_forest": float(destroyed[forested].mean()),
        "max_u_h_eff_in_forest": float(product[forested].max()),
    }
    return checks, destroyed, stats


def plot(x, y, dem, cover, results, destroyed, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = int(np.argmin(np.abs(y - FOREST_ROW_Y)))
    open_row = int(np.argmin(np.abs(y - OPEN_ROW_Y)))
    labels = {
        "no_forest": "a) no forest",
        "unbreakable": "b) unbreakable forest",
        "breakable": "c) breakable forest",
    }
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(x, dem[row], color="0.3")
    axes[0].fill_between(x, dem[row], BOTTOM - 20, where=cover[row] > 0, color="tab:green",
                         alpha=0.3, label=f"forest, cover {COVER}")
    axes[0].set_ylabel("elevation [m]")
    axes[0].legend(loc="upper right")
    colors = {"no_forest": "tab:blue", "unbreakable": "tab:orange", "breakable": "tab:green"}
    for key, label in labels.items():
        r = results[key]
        axes[1].plot(x, r["peak_h"][row], label=label, color=colors[key])
        axes[2].plot(x, r["peak_u"][row], label=label, color=colors[key])
        axes[1].axvline(r["runout"], lw=0.8, ls="--", color=colors[key])
    # the same run beside the forest: the open half of c)
    axes[1].plot(x, results["breakable"]["peak_h"][open_row], color="0.4", lw=0.9, ls="-.",
                 label="c) open half")
    axes[2].plot(x, results["breakable"]["peak_u"][open_row], color="0.4", lw=0.9, ls="-.",
                 label="c) open half")
    axes[1].set_ylabel("peak flow thickness [m]")
    axes[2].set_ylabel("peak velocity [m/s]")
    for key in ("unbreakable", "breakable"):
        axes[3].plot(x, results[key]["standing"][row], label=labels[key], color=colors[key])
    axes[3].plot(x, cover[row], "k--", lw=0.8, label="initial cover")
    axes[3].set_ylabel("standing cover [-]")
    axes[3].set_xlabel("x [m]")
    for ax in axes[1:]:
        ax.legend(loc="upper right")
        for xf in FOREST_X:
            ax.axvline(xf, color="tab:green", lw=0.8, ls=":")
    fig.suptitle(f"Forest model on a parabolic slope (forest half y = {y[row]:.0f} m, open "
                 f"half y = {y[open_row]:.0f} m, "
                 f"D = {settings['forest_trunk_diameter']} m, d_c = {settings['forest_crown_diameter']} m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forest_parabola_profiles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    extent = (0.0, LENGTH, 0.0, WIDTH)
    image = ax.imshow(np.where(cover > 0, destroyed, np.nan), origin="lower", extent=extent,
                      cmap="YlOrBr", vmin=0.0, vmax=1.0, aspect="equal")
    ax.contour(x, y, results["breakable"]["peak_h"], levels=[0.1], colors="k", linewidths=0.6)
    fig.colorbar(image, ax=ax, label="destroyed cover fraction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("c) forest destruction, flow outline (peak thickness 0.1 m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forest_parabola_destruction.png", dpi=150)
    plt.close(fig)


def export_testcase():
    """Writes the case in the avaframe PNG format for the live viewer."""
    from PIL import Image

    out = REPO_ROOT / "data" / "avaframe"
    _, _, dem, release, cover = build_case()
    ny, nx = dem.shape
    # the PNG loader reads row 0 as the first image row, like the other cases
    rgba = np.ascontiguousarray(dem.astype("<f4")).view(np.uint8).reshape(ny, nx, 4)
    Image.fromarray(rgba, mode="RGBA").save(out / f"{CASE}.png")
    half = CELL_SIZE / 2.0
    (out / f"{CASE}.aabb").write_text(
        f"{half:.1f}\n{half:.1f}\n{LENGTH - half:.1f}\n{WIDTH - half:.1f}\n")

    def alpha_png(values, path):
        image = np.zeros((ny, nx, 4), dtype=np.uint8)
        image[..., 0] = 255
        image[..., 3] = np.clip(np.round(values * 100.0), 0, 255).astype(np.uint8)
        Image.fromarray(image, mode="RGBA").save(path)

    alpha_png(release, out / f"{CASE}releaseTexture.png")
    alpha_png(cover, out / f"{CASE}forestCover.png")  # cover in percent
    settings = {
        "dem_path": f"data/avaframe/{CASE}.png",
        "release_areas_path": f"data/avaframe/{CASE}releaseTexture.png",
        "forest_cover_path": f"data/avaframe/{CASE}forestCover.png",
        **SETTINGS,
    }
    path = REPO_ROOT / "verification" / "forest_parabola_settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"wrote data/avaframe/{CASE}{{.png,.aabb,releaseTexture.png,forestCover.png}} and {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", action="store_true",
                        help="also write the live viewer test case")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.export:
        export_testcase()

    x, y, dem, _, cover = build_case()
    settings = dict(SETTINGS)
    print(f"=== Forest on a parabolic slope: {len(x)} x {len(y)} cells of {CELL_SIZE} m, "
          f"forest x = {FOREST_X[0]:.0f}..{FOREST_X[1]:.0f} m, cover {COVER}, "
          f"breaking at u h_eff = {breaking_product(settings):.1f} m^2/s ===")
    results = {
        "no_forest": run("a) no forest", settings, None),
        "unbreakable": run("b) unbreakable forest",
                           {**settings, "forest_modulus_of_rupture": UNBREAKABLE}, cover),
        "breakable": run("c) breakable forest", settings, cover),
        "zero_cover": run("d) zero cover", settings, np.zeros_like(cover)),
    }
    checks, destroyed, stats = check(results, settings, cover)
    plot(x, y, dem, cover, results, destroyed, settings)

    print()
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    for name, value in stats.items():
        print(f"  {name}: {value}")
    summary = {
        "settings": settings,
        "forest_x": FOREST_X,
        "forest_y": FOREST_Y,
        "cover": COVER,
        "runs": {k: {"runout_x_forest_half": r["runout"], "runout_x_open_half": r["runout_open"],
                     "steps": r["steps"], "time_s": r["time"],
                     "stopped": r["stopped"], "max_peak_velocity": float(r["peak_u"].max())}
                 for k, r in results.items()},
        "checks": {k: bool(v) for k, v in checks.items()},
        "stats": stats,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\noutputs in {OUT_DIR}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
