"""Double-bump test of detachment (flying particles) and landing.

Snow released on a 32 degree incline (create_double_bump_testcase.py, after
the double-bump example of Matter, github.com/larsblatny/matter) runs over
two sech-shaped bumps. At the crest of the first bump the bed-normal load
g n_z + kappa |u|^2 turns negative and the particles detach; they fly
ballistically and land on the upslope face of the second bump, flow over it
(partly taking off again) and run out on the flat valley floor.

With Coulomb friction (mu = 0.4) the run is compared with a point slider
(double_bump_reference.py): released at rest at the initial center of mass,
it follows the closed-form velocity height of a Coulomb slider with the
centripetal load, takes off where the normal load turns negative, flies
ballistically and lands with the same friction impulse as the particles.

Only the DAMPM global model with the centripetal load
(enable_curvature) has detachment. The run is sampled every step: take-off
and landing positions come from the change of the flying flag between two
samples. The figures show the particles in a side view (height above the
inclined plane) while most of them land on the second bump, with the point
slider path, its take-offs and landings, plus the take-off and landing
positions along the slope and the number of flying particles over time.

Usage:
    python verification/run_double_bump_test.py
    python verification/run_double_bump_test.py --out-dir verification/outputs/double_bump_test
"""

import argparse
import pathlib

import numpy as np

import create_double_bump_testcase as case
import double_bump_reference

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "verification" / "outputs" / "double_bump_test"
DATA_DIR = REPO_ROOT / "data" / "avaframe"

_DEM_PATH, _RELEASE_PATH = case.raster_paths(DATA_DIR)
SETTINGS = dict(case.MODEL_SETTINGS,
                dem_path=str(_DEM_PATH),
                release_areas_path=str(_RELEASE_PATH))

THREE_D_MODELS = ("dampm-global-cartesian", "global", "3d-depth-averaged", "3d")
DEFAULT_DENSITY = 200.0  # kg/m3, engine default (SimSettings::new)
SIM_INFO_STOPPED = 1 << 31

CREST_1, CREST_2 = case.BUMP_CENTRES
WIDTH = case.BUMP_WIDTH
# take-off from the first bump: the convex part around its crest
TAKEOFF_WINDOW = (CREST_1 - 2.0 * WIDTH, CREST_1 + WIDTH)
# second bump: from the trough between the bumps to one decay length past
# its crest
LANDING_WINDOW = (0.5 * (CREST_1 + CREST_2), CREST_2 + WIDTH)
# x range of the figures: take-off/landing histogram and the side view (equal axes)
VIEW_X = (CREST_1 - 4.0 * WIDTH, CREST_2 + 8.0 * WIDTH)
SIDE_VIEW_X = (CREST_1 - 1.75 * WIDTH, CREST_2 + 3.75 * WIDTH)


def ensure_testcase():
    """Writes the double-bump DEM and release texture if they are missing."""
    if not (_DEM_PATH.is_file() and _RELEASE_PATH.is_file()):
        case.write_testcase(DATA_DIR)


def run_simulation(settings, save_steps=1):
    """Runs the double-bump flow to the end, sampling every save_steps steps.

    Returns a dict with the sample times, per-sample particle x, height above
    the bed, flying flags and speeds, and the take-off / landing events
    (sample index, particle, x)."""
    import avalanchers

    if settings["sim_model"].lower() not in THREE_D_MODELS:
        raise ValueError("Detachment is only modeled in the DAMPM global model.")
    if not settings.get("enable_curvature", False):
        raise ValueError("Detachment needs the centripetal load (enable_curvature).")
    ensure_testcase()
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.prepare()
    n = int(np.asarray(sim.particles_mass).size)
    print(f"=== Double-bump test: {n} particles, {settings['sim_model']}, "
          f"{settings['friction_model']} mu={settings['friction_coefficient']} ===")

    times, xs, ys, heights, flying, speeds = [], [], [], [], [], []
    # particle positions start at the grid edge, half a cell before the first
    # DEM cell centre (x = 0, y = 0 of the test case)
    bounds = np.asarray(sim.dem_bounds, dtype=float)
    x_edge, y_edge = float(bounds[0]), float(bounds[2])

    def sample(t):
        position = np.asarray(sim.particles_position, dtype=float)
        velocity = np.asarray(sim.particles_velocity, dtype=float)
        x = position[:, 0] + x_edge
        times.append(t)
        xs.append(x.astype(np.float32))
        ys.append((position[:, 1] + y_edge).astype(np.float32))
        heights.append((position[:, 2] - case.profile(x)).astype(np.float32))
        flying.append(np.asarray(sim.particles_flying, dtype=bool))
        speeds.append(np.linalg.norm(velocity, axis=1).astype(np.float32))

    sample(0.0)
    last_timestep = -1
    while True:
        info = sim.run_n_steps(save_steps)
        sample(float(info.elapsed_time))
        if (info.flags & SIM_INFO_STOPPED) or sim.state == "Finished":
            break
        if info.timestep == last_timestep:
            print("simulation made no progress - stopping")
            break
        last_timestep = info.timestep
    print(f"stopped after {info.timestep} steps, t = {info.elapsed_time:.2f} s "
          f"({len(times)} samples)")

    run = {
        "t": np.asarray(times),
        "x": np.stack(xs),
        "y": np.stack(ys),
        "mass": np.asarray(sim.particles_mass, dtype=float),
        "cell_size": float(sim.cell_size),
        "density": float(settings.get("density", DEFAULT_DENSITY)),
        "height": np.stack(heights),
        "flying": np.stack(flying),
        "speed": np.stack(speeds),
    }
    # take-off: grounded at the previous sample, flying now (last bed position)
    # landing: flying at the previous sample, grounded now (landing position)
    previous, current = run["flying"][:-1], run["flying"][1:]
    k, p = np.nonzero(~previous & current)
    run["takeoff"] = (k + 1, p, run["x"][k, p])
    k, p = np.nonzero(previous & ~current)
    run["landing"] = (k + 1, p, run["x"][k + 1, p])
    return run


def first_event(events, n):
    """x of the first event of every particle (NaN without one)."""
    sample, particle, x = events
    first = np.full(n, np.nan)
    order = np.argsort(sample, kind="stable")
    seen = np.zeros(n, dtype=bool)
    for i in order:
        if not seen[particle[i]]:
            seen[particle[i]] = True
            first[particle[i]] = x[i]
    return first


def inside(x, window):
    return (x >= window[0]) & (x < window[1])


def reference_path(run, mu, terminal_velocity):
    """Coulomb point slider released at rest at the initial center of mass,
    with the flight drag of the run."""
    return double_bump_reference.point_slider(
        case.profile, case.profile_slope, case.profile_curvature, mu,
        float(run["x"][0].mean()), case.LENGTH,
        terminal_velocity=terminal_velocity)


def analyse(run, mu, coulomb=True, terminal_velocity=0.0):
    """Metrics of the detachment, the comparison with the Coulomb point
    slider of friction coefficient mu (only with Coulomb friction) and the
    snapshot samples of the figures."""
    n = run["x"].shape[1]
    n_flying = run["flying"].sum(axis=1)
    first_takeoff = first_event(run["takeoff"], n)
    first_landing = first_event(run["landing"], n)
    took_off = np.isfinite(first_takeoff)
    landed = np.isfinite(first_landing)
    flight_height = np.where(run["flying"], run["height"], -np.inf).max()

    # snapshot: most landings on the second bump (5-sample running sum)
    sample, particle, x = run["landing"]
    on_face = inside(x, LANDING_WINDOW)
    landings = np.bincount(sample[on_face], minlength=len(run["t"])).astype(float)
    landings = np.convolve(landings, np.ones(5), mode="same")
    i_land = int(np.argmax(landings))
    i_peak = int(np.argmax(n_flying))

    result = {
        "particles": n,
        "max_flying": int(n_flying.max()),
        "t_max_flying": float(run["t"][i_peak]),
        "takeoff_fraction": float(took_off.mean()),
        "takeoff_at_bump_1": float(inside(first_takeoff[took_off], TAKEOFF_WINDOW).mean())
        if took_off.any() else 0.0,
        "landing_on_bump_2": float(inside(first_landing[landed], LANDING_WINDOW).mean())
        if landed.any() else 0.0,
        "max_flight_height": float(flight_height) if np.isfinite(flight_height) else 0.0,
        "flying_at_end": int(run["flying"][-1].sum()),
        "runout_x": float(np.percentile(run["x"][-1], 95)),
        "snapshot": i_land,
    }
    result.update({
        "reference": None,
        "takeoff_x": float(np.median(first_takeoff[took_off])) if took_off.any() else float("nan"),
        "landing_x": float(np.median(first_landing[landed])) if landed.any() else float("nan"),
        "stop_x": float(run["x"][-1].mean()),
    })
    print(f"simulation: median first take-off x {result['takeoff_x']:.1f} m, median first "
          f"landing x {result['landing_x']:.1f} m, final center of mass x {result['stop_x']:.1f} m")
    if coulomb:
        reference = reference_path(run, mu, terminal_velocity)
        result.update({
            "reference": reference,
            "reference_takeoff_x": reference.event_x("takeoff"),
            "reference_landing_x": reference.event_x("landing"),
            "reference_stop_x": reference.event_x("stop"),
        })
        print(f"point slider (mu = {mu}): take-off x {result['reference_takeoff_x']:.1f} m, "
              f"landing x {result['reference_landing_x']:.1f} m, "
              f"stop x {result['reference_stop_x']:.1f} m")
    else:
        print("no point-slider reference: it needs Coulomb friction")
    print(f"particles that took off: {100 * result['takeoff_fraction']:.1f} %, "
          f"first take-off at bump 1 {TAKEOFF_WINDOW}: {100 * result['takeoff_at_bump_1']:.1f} %")
    print(f"first landing on bump 2 {LANDING_WINDOW}: {100 * result['landing_on_bump_2']:.1f} %")
    print(f"most particles in flight: {result['max_flying']} at t = {result['t_max_flying']:.2f} s, "
          f"highest flight {result['max_flight_height']:.2f} m above the bed")
    print(f"still flying at the end: {result['flying_at_end']}, "
          f"95 % runout x = {result['runout_x']:.1f} m")
    return result


def _bed_axes(ax, ts):
    """Bumps above the plane as the bed line of a side view."""
    x = np.linspace(*VIEW_X, 600)
    bed = case.bump(x)
    ax.fill_between(x, bed, -5.0, color="0.85", linewidth=0)
    ax.plot(x, bed, color=ts.BLACK, linewidth=0.8)
    ax.set_xlim(*VIEW_X)
    ax.set_ylim(-2.0, 1.6 * case.BUMP_HEIGHT)


def flow_depth(run, i):
    """Vertical flow depth (m) of the cell under each particle at sample i:
    the mass of the particles on the bed in the DEM cell over rho dx^2."""
    dx = run["cell_size"]
    x, y = run["x"][i], run["y"][i]
    grounded = ~run["flying"][i]
    # cell centers lie on multiples of dx (x = 0 is the first one)
    col = np.floor(x / dx + 0.5).astype(int)
    row = np.floor(y / dx + 0.5).astype(int)
    col -= col.min()
    row -= row.min()
    mass = np.zeros((row.max() + 1, col.max() + 1))
    np.add.at(mass, (row[grounded], col[grounded]), run["mass"][grounded])
    return mass[row, col] / (run["density"] * dx * dx)


def plot_snapshots(run, result, out_dir):
    """Side view of the particles while they land on the second bump: height
    above the inclined plane over x. Particles on the bed are drawn at a
    random height within the flow depth of their cell, flying particles at
    their position. With the point slider path, its take-offs and landings."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import thesis_style as ts

    i = result["snapshot"]
    reference = result["reference"]
    rng = np.random.default_rng(0)
    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(1.0, 0.36), layout="constrained")
        _bed_axes(ax, ts)
        # true proportions, cropped to the jumps
        ax.set_xlim(*SIDE_VIEW_X)
        ax.set_ylim(-1.0, 1.3 * case.BUMP_HEIGHT)
        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([0.0, 10.0, 20.0])
        x = run["x"][i]
        fly = run["flying"][i]
        z = np.where(fly, run["height"][i] + case.bump(x),
                     case.bump(x) + rng.uniform(0.0, 1.0, len(x)) * flow_depth(run, i))
        ax.scatter(x[~fly], z[~fly], s=1.0, color=ts.BLUE, linewidths=0, rasterized=True)
        ax.scatter(x[fly], z[fly], s=1.0, color=ts.VERMILION, linewidths=0, rasterized=True)
        handles = [Line2D([], [], color=ts.BLUE, linestyle="none", marker="o", markersize=3,
                          label="on the bed"),
                   Line2D([], [], color=ts.VERMILION, linestyle="none", marker="o",
                          markersize=3, label="flying")]
        if reference is not None:
            ax.plot(reference.x, reference.z - case.plane(reference.x), color=ts.GREEN,
                    linewidth=1.0)
            events = {"takeoff": ("^", ts.YELLOW, "slider take-off"),
                      "landing": ("v", ts.SKY, "slider landing")}
            for kind, (marker, fill, label) in events.items():
                xs = np.array([e["x"] for e in reference.events if e["kind"] == kind])
                style = dict(linestyle="none", marker=marker, markersize=6,
                             markerfacecolor=fill, markeredgecolor=ts.BLACK,
                             markeredgewidth=0.8)
                ax.plot(xs, case.bump(xs), zorder=5, **style)
                handles.append(Line2D([], [], label=label, **style))
            handles.insert(2, Line2D([], [], color=ts.GREEN, linewidth=1.0,
                                     label="point slider"))
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$z - z_\mathrm{plane}$ (m)")
        ax.text(0.99, 0.95, rf"$t = {run['t'][i]:.1f}$ s, {int(fly.sum())} flying",
                transform=ax.transAxes, ha="right", va="top")
        handles.append(Line2D([], [], color=ts.BLACK, linewidth=0.8, label="bed"))
        fig.legend(handles=handles, loc="outside lower center", ncol=3)
        return ts.save(fig, pathlib.Path(out_dir) / "thesis", "doublebump_snapshots")


def plot_flight(run, result, out_dir):
    """Take-off and landing positions along the slope with the bed (a) and the
    number of flying particles over time with the time of the side view (b)."""
    import matplotlib.pyplot as plt
    import thesis_style as ts

    n = result["particles"]
    with ts.style():
        fig, axes = plt.subplots(2, 1, figsize=ts.size(1.0, 0.66), layout="constrained")
        ax = axes[0]
        bins = np.arange(VIEW_X[0], VIEW_X[1] + 5.0, 5.0)
        ax.hist(run["takeoff"][2], bins=bins, color=ts.ORANGE, alpha=0.8, label="take-off")
        ax.hist(run["landing"][2], bins=bins, color=ts.BLUE, alpha=0.6, label="landing")
        ax.set_xlim(*VIEW_X)
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel("particles per 5 m")
        bed_ax = ax.twinx()
        x = np.linspace(*VIEW_X, 600)
        bed_ax.plot(x, case.bump(x), color=ts.BLACK, linewidth=0.8, label="bed")
        bed_ax.set_ylim(0.0, 2.5 * case.BUMP_HEIGHT)
        bed_ax.set_ylabel(r"$z - z_\mathrm{plane}$ (m)")
        bed_ax.grid(False)
        for window, color in ((TAKEOFF_WINDOW, ts.ORANGE), (LANDING_WINDOW, ts.BLUE)):
            ax.axvspan(*window, color=color, alpha=0.08, linewidth=0)
        handles = ax.get_legend_handles_labels()[0] + bed_ax.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="upper right")
        ts.panel_label(ax, "a")

        ax = axes[1]
        n_flying = run["flying"].sum(axis=1)
        ax.plot(run["t"], 100.0 * n_flying / n, color=ts.VERMILION)
        # up to 10 s after the last flight
        flights = np.nonzero(n_flying)[0]
        if len(flights):
            ax.set_xlim(0.0, min(run["t"][-1], run["t"][flights[-1]] + 10.0))
        t_snapshot = run["t"][result["snapshot"]]
        ax.axvline(t_snapshot, color=ts.NEUTRAL, linestyle=":", linewidth=0.9)
        ax.text(t_snapshot, 1.0, " side view", transform=ax.get_xaxis_transform(),
                va="top", ha="left")
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel("flying particles (%)")
        ax.set_ylim(bottom=0.0)
        ts.panel_label(ax, "b", loc=(0.015, 0.92))
        return ts.save(fig, pathlib.Path(out_dir) / "thesis", "doublebump_flight")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--detachment", choices=("free-fall", "curvature"),
                        default=SETTINGS["detachment"])
    from cli_args import add_particles_per_cell_argument
    add_particles_per_cell_argument(parser)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    if args.save_steps < 1:
        parser.error("--save-steps must be positive")
    settings = dict(SETTINGS, detachment=args.detachment)
    if args.particles_per_cell is not None:
        settings["released_particles_per_cell"] = args.particles_per_cell
    run = run_simulation(settings, args.save_steps)
    result = analyse(run, settings["friction_coefficient"],
                     coulomb=settings["friction_model"] == "coulomb",
                     terminal_velocity=settings["flight_terminal_velocity"])
    pics = pathlib.Path(args.out_dir) / "pics"
    plot_snapshots(run, result, pics)
    plot_flight(run, result, pics)
    return result


if __name__ == "__main__":
    main()
