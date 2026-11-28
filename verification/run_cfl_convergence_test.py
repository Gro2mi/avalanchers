"""Convergence test for the CFL number (avaWog).

Runs the AvaFrame avaWog case (data/avaframe/avaWog.png with its release
texture) with the default flow parameters of the thesis (3D depth-averaged
DAMPM model, Voellmy mu = 0.2, xi = 2000 m/s^2, N_ppc = 4) for a decreasing
CFL number, CFL = 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05. The time step scales
with the CFL number, so this is a time-step convergence test. There is no
analytical reference: the smallest CFL is run 10 times (--reference-runs) and
the mean of these runs (peak fields and scalars) is the reference. Every other
run is compared with it by

- the relative L2 error ||f - f_ref||_2 / ||f_ref||_2 of the peak flow
  thickness and the peak velocity fields,
- the IoU of the affected area (peak flow thickness > 0.1 m), release cells
  excluded,
- the center-of-mass travel angle and 3D travel length (along the bed),
- the avalanche length s (straight horizontal distance between the highest
  and the lowest cell of the affected area, release included) and the
  footprint angle arctan(drop / s).

Checks (PASS/FAIL, exit code):
- every run ends by its stopping criterion before max_steps (stable time
  stepping at every CFL),
- the L2 errors of the largest CFL are larger than those of the second
  smallest CFL (the fields converge towards the reference),
- the default CFL = 0.9 reaches IoU >= 0.9 with the reference.

The GPU reductions are not bit-reproducible: the spread of the reference
runs (each against the mean of the others) gives the noise floor. The
simulation time t_sim is the time stepping alone (no GPU setup, input loading
or result read-back), the median of 3 runs per CFL (of the reference runs for
the smallest CFL); a warm-up run before the sweep is not timed. max_steps is
scaled with 1 / CFL so that small CFL numbers are not cut off.

Outputs go to verification/outputs/cfl_convergence_test/: the table as
markdown (console), CSV and booktabs TeX, and the thesis figures
(pics/thesis/cfl_convergence_errors, cfl_convergence_runtime and
cfl_convergence_footprints).

Usage:
    python verification/run_cfl_convergence_test.py
    python verification/run_cfl_convergence_test.py --cfl 0.9 0.3 0.1 --out-dir /tmp/cfl
    python verification/run_cfl_convergence_test.py --quick
"""

import argparse
import csv
import math
import pathlib
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "verification" / "outputs" / "cfl_convergence_test"
DATA_DIR = REPO_ROOT / "data" / "avaframe"
SIM_INFO_STOPPED = 1 << 31

# default flow parameters of the thesis (Table tab:flow-parameters)
SETTINGS = {
    "enable_entrainment": False,
    "dem_path": str(DATA_DIR / "avaWog.png"),
    "release_areas_path": str(DATA_DIR / "avaWogreleaseTexture.png"),
    "sim_model": "dampm-global-cartesian",
    "friction_model": "voellmy",
    "friction_coefficient": 0.2,
    "drag_coefficient": 2000.0,
    "density": 200.0,
    "released_particles_per_cell": 4,
    "velocity_threshold": 0.1,
    "enable_particle_interaction": True,
    "particle_placement": "uniform-relaxed",
}

CFL_CFG = {
    "cfl": (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9),
    "quick_cfl": (0.1, 0.3, 0.9),
    "affected_threshold": 0.1,    # m, peak flow thickness of the affected area
    "default_cfl": 0.9,           # thesis default (Table tab:flow-parameters)
    "min_iou_default": 0.9,
    "footprint_cfl": (0.9, 0.4, 0.1),  # outlines drawn next to the reference
    "max_steps_at_default": 20000,     # scaled with default_cfl / CFL
    "batch_steps": 500,
    "reference_runs": 10,         # runs of the smallest CFL, averaged
    "timing_runs": 3,             # runs per CFL for the median run time
}


def settings_for(base, cfl):
    max_steps = math.ceil(CFL_CFG["max_steps_at_default"] * CFL_CFG["default_cfl"] / cfl)
    return dict(base, cfl=float(cfl), max_steps=int(max_steps))


def run(settings):
    """Runs one simulation to its end; returns the final metrics and fields."""
    import avalanchers

    t0 = time.perf_counter()
    sim = avalanchers.PySimulation.new()
    sim.create(dict(settings))
    sim.prepare()
    # simulation time: only the time stepping (no GPU setup, no input
    # loading, no read-back of the result fields); run_n_steps returns after
    # the GPU finished the batch
    t1 = time.perf_counter()
    last = -1
    info = None
    while sim.state != "Finished":
        info = sim.run_n_steps(CFL_CFG["batch_steps"])
        if (info.flags & SIM_INFO_STOPPED) or info.timestep == last:
            break
        last = info.timestep
    t2 = time.perf_counter()
    return {
        "cfl": float(settings["cfl"]),
        "particles": int(info.number_particles),
        "steps": int(info.timestep),
        "max_steps": int(settings["max_steps"]),
        "t_end": float(info.elapsed_time),
        "t_setup": t1 - t0,
        "t_sim": t2 - t1,
        "travel_angle": float(info.travel_angle),
        "travel_length_3d": float(info.travel_length_3d),
        "pft": np.nan_to_num(np.asarray(sim.peak_flow_thickness, dtype=float)),
        "pfv": np.nan_to_num(np.asarray(sim.peak_velocity, dtype=float)),
        "release": np.asarray(sim.release_areas, dtype=float) > 0,
        "dem": np.asarray(sim.dem, dtype=float),
        "bounds": tuple(float(v) for v in sim.dem_bounds),
        "cell_size": float(sim.cell_size),
    }


def rel_l2(f, ref):
    return float(np.linalg.norm(f - ref) / np.linalg.norm(ref))


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def cell_centres(r):
    xmin, _, ymin, _ = r["bounds"]
    cs = r["cell_size"]
    ny, nx = r["dem"].shape
    x = xmin + (np.arange(nx) + 0.5) * cs
    y = ymin + (np.arange(ny) + 0.5) * cs
    return np.meshgrid(x, y)


def avalanche_length(r, threshold):
    """Straight horizontal distance and angle between the highest and the
    lowest cell of the affected area (release included)."""
    mask = (r["pft"] > threshold) | r["release"]
    x, y = cell_centres(r)
    z = np.where(mask, r["dem"], np.nan)
    top, low = np.nanargmax(z), np.nanargmin(z)
    s = float(math.hypot(x.flat[low] - x.flat[top], y.flat[low] - y.flat[top]))
    drop = float(z.flat[top] - z.flat[low])
    return s, math.degrees(math.atan2(drop, s))


TIMES = ("t_setup", "t_sim")
SCALARS = ("particles", "steps", "t_end", "t_setup", "t_sim", "travel_angle",
           "travel_length_3d")


def mean_run(runs):
    """Reference from several runs of the same CFL: mean peak fields, mean
    scalars, and the standard deviation of the scalars (key + "_std")."""
    ref = dict(runs[0])
    ref["pft"] = np.mean([r["pft"] for r in runs], axis=0)
    ref["pfv"] = np.mean([r["pfv"] for r in runs], axis=0)
    for key in SCALARS:
        values = np.array([r[key] for r in runs], dtype=float)
        # run times: median, robust against other load on the GPU
        ref[key] = float(np.median(values) if key in TIMES else values.mean())
        ref[key + "_std"] = float(values.std(ddof=1)) if len(runs) > 1 else 0.0
    ref["runs"] = len(runs)
    ref["max_steps_reached"] = any(r["steps"] >= r["max_steps"] for r in runs)
    return ref


def compare(r, ref, cfg):
    thr = cfg["affected_threshold"]
    return {
        "l2_pft": rel_l2(r["pft"], ref["pft"]),
        "l2_pfv": rel_l2(r["pfv"], ref["pfv"]),
        "iou": iou((r["pft"] > thr) & ~r["release"], (ref["pft"] > thr) & ~ref["release"]),
    }


def analyse(results, ref, cfg):
    """Metrics of every run against the reference; the reference is the last
    row (its errors are left empty). Rows are sorted by decreasing CFL."""
    rows = []
    for r in results + [ref]:
        s, angle = avalanche_length(r, cfg["affected_threshold"])
        errors = compare(r, ref, cfg) if r is not ref else \
            {"l2_pft": float("nan"), "l2_pfv": float("nan"), "iou": float("nan")}
        row = {key: r[key] for key in SCALARS}
        row.update({"cfl": r["cfl"], **errors, "length": s, "length_angle": angle,
                    "runs": r.get("runs", 1),
                    "max_steps_reached": bool(r.get("max_steps_reached",
                                                    r["steps"] >= r["max_steps"]))})
        for key in SCALARS:
            row[key + "_std"] = r.get(key + "_std", 0.0)
        rows.append(row)
    return rows


def noise_floor(runs, cfg):
    """Leave-one-out spread of the reference runs: every run against the mean
    of the others. The GPU reductions are not bit-reproducible, so this is the
    smallest difference the comparison can resolve."""
    diffs = [compare(r, mean_run(runs[:i] + runs[i + 1:]), cfg) for i, r in enumerate(runs)]
    return {key: float(np.mean([d[key] for d in diffs])) for key in diffs[0]}


def convergence_order(rows, key):
    """Slope of log(error) over log(CFL), the reference left out (1 for a
    first-order time integration)."""
    c = np.array([row["cfl"] for row in rows[:-1]], dtype=float)
    e = np.array([row[key] for row in rows[:-1]], dtype=float)
    ok = e > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(c[ok]), np.log(e[ok]), 1)[0])


def print_table(rows):
    print("| CFL | runs | steps | t_end (s) | setup (s) | simulation (s) "
          "| L2 h_peak | L2 v_peak | IoU | travel angle (deg) | s_3D (m) | s (m) |")
    print("|---" * 12 + "|")
    for r in rows:
        flag = " (max_steps)" if r["max_steps_reached"] else ""
        print(f"| {r['cfl']:g} | {r['runs']} | {r['steps']:.0f}{flag} "
              f"| {r['t_end']:.1f} | {r['t_setup']:.2f} "
              f"| {r['t_sim']:.2f} +- {r['t_sim_std']:.2f} | {r['l2_pft']:.3f} "
              f"| {r['l2_pfv']:.3f} | {r['iou']:.3f} "
              f"| {r['travel_angle']:.3f} +- {r['travel_angle_std']:.3f} "
              f"| {r['travel_length_3d']:.1f} +- {r['travel_length_3d_std']:.1f} "
              f"| {r['length']:.1f} |")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows, path):
    """Booktabs table (siunitx S columns), reference row without errors."""
    lines = [
        r"\begin{tabular}{S[table-format=1.2] S[table-format=5.0]"
        r" S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]"
        r" S[table-format=2.2] S[table-format=4.1] S[table-format=3.2]}",
        r"\toprule",
        r"{CFL} & {Steps} & {$L_2$, $h_{\max}$} & {$L_2$, $v_{\max}$}"
        r" & {IoU} & {$\alpha$ (\si{\degree})} & {$s_{\mathrm{3D}}$ (m)}"
        r" & {$t_{\mathrm{sim}}$ (s)} \\",
        r"\midrule",
    ]
    for r in rows[:-1]:
        lines.append(f"{r['cfl']:.2f} & {r['steps']:.0f} & {r['l2_pft']:.3f} & {r['l2_pfv']:.3f} "
                     f"& {r['iou']:.3f} & {r['travel_angle']:.2f} "
                     f"& {r['travel_length_3d']:.1f} & {r['t_sim']:.2f} \\\\")
    r = rows[-1]
    lines.append(f"{r['cfl']:.2f} & {r['steps']:.0f} & {{--}} & {{--}} & {{--}} "
                 f"& {r['travel_angle']:.2f} & {r['travel_length_3d']:.1f} "
                 f"& {r['t_sim']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _cfl_axis(ax, c):
    """log CFL axis with every sweep value as a tick."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(c))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in c]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("CFL")


def plot_errors(rows, out_dir, noise=None):
    """(a) differences to the reference over CFL, (b) center-of-mass travel
    angle and 3D travel length over CFL."""
    import matplotlib.pyplot as plt
    import thesis_style as ts

    c = np.array([r["cfl"] for r in rows[:-1]], dtype=float)
    c_all = np.array([r["cfl"] for r in rows], dtype=float)
    with ts.style():
        fig, axes = plt.subplots(1, 2, figsize=ts.size(1.0, 0.42), layout="constrained")
        ax = axes[0]
        ax.plot(c, [r["l2_pft"] for r in rows[:-1]], "o-", color=ts.VARIABLE_COLORS["h"],
                markersize=3, label=r"$L_2$, $h_{\max}$")
        ax.plot(c, [r["l2_pfv"] for r in rows[:-1]], "s-", color=ts.VARIABLE_COLORS["speed"],
                markersize=3, label=r"$L_2$, $v_{\max}$")
        ax.plot(c, [1.0 - r["iou"] for r in rows[:-1]], "^-", color=ts.GREEN,
                markersize=3, label=r"$1 - \mathrm{IoU}$")
        # first-order slope of the time integration, anchored at the largest
        # CFL (first row)
        guide = rows[0]["l2_pft"] * c / c[0]
        ax.plot(c, guide, ":", color=ts.NEUTRAL, label=r"$\propto \mathrm{CFL}$")
        ax.set_yscale("log")
        _cfl_axis(ax, c_all)
        ax.set_ylabel("difference to the reference")
        ax.grid(which="minor", alpha=0.15)
        if noise is not None:
            floor = max(noise["l2_pft"], noise["l2_pfv"], 1.0 - noise["iou"])
            if floor > 0:
                ax.axhline(floor, color=ts.NEUTRAL, linestyle="--", linewidth=0.8)
                ax.set_ylim(bottom=floor / 2.0)
                ax.text(c.max(), floor * 0.93, "reference spread ", ha="right", va="top",
                        fontsize=7, color=ts.NEUTRAL)
        ax.legend(loc="upper left")
        ts.panel_label(ax, "a", loc=(0.88, 0.1))

        ax = axes[1]
        ax.errorbar(c_all, [r["travel_angle"] for r in rows],
                    yerr=[r["travel_angle_std"] for r in rows], fmt="o-",
                    color=ts.VERMILION, markersize=3, capsize=2, label=r"$\alpha$")
        _cfl_axis(ax, c_all)
        ax.set_ylabel(r"travel angle $\alpha$ ($^\circ$)", color=ts.VERMILION)
        ax.axvline(CFL_CFG["default_cfl"], color=ts.NEUTRAL, linestyle="--", linewidth=0.8)
        ax.text(CFL_CFG["default_cfl"], 0.97, "default ", transform=ax.get_xaxis_transform(),
                ha="right", va="top")
        ax2 = ax.twinx()
        ax2.errorbar(c_all, [r["travel_length_3d"] for r in rows],
                     yerr=[r["travel_length_3d_std"] for r in rows], fmt="s-",
                     color=ts.PURPLE, markersize=3, capsize=2, label=r"$s_{\mathrm{3D}}$")
        ax2.set_ylabel(r"travel length $s_{\mathrm{3D}}$ (m)", color=ts.PURPLE)
        ax2.grid(False)
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="center left")
        ts.panel_label(ax, "b", loc=(0.45, 0.97))
        return ts.save(fig, out_dir, "cfl_convergence_errors")


def plot_runtime(rows, out_dir):
    """Simulation time (time stepping only) and setup time over CFL, with the
    1 / CFL scaling of the number of time steps."""
    import matplotlib.pyplot as plt
    import thesis_style as ts

    c = np.array([r["cfl"] for r in rows], dtype=float)
    t_sim = np.array([r["t_sim"] for r in rows])
    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(0.7, 0.6), layout="constrained")
        ax.errorbar(c, t_sim, yerr=[r["t_sim_std"] for r in rows], fmt="o-",
                    color=ts.BLUE, markersize=3, capsize=2, label="simulation")
        ax.plot(c, [r["t_setup"] for r in rows], "s--", color=ts.NEUTRAL, markersize=3,
                label="setup")
        # number of time steps ~ 1 / CFL, anchored at the reference
        ax.plot(c, t_sim[-1] * c[-1] / c, ":", color=ts.NEUTRAL,
                label=r"$\propto \mathrm{CFL}^{-1}$")
        ax.set_yscale("log")
        _cfl_axis(ax, np.sort(c))
        ax.set_ylabel("time (s)")
        ax.grid(which="minor", alpha=0.15)
        ax.legend(loc="upper right")
        return ts.save(fig, out_dir, "cfl_convergence_runtime")


def plot_footprints(results, out_dir, threshold, which):
    """Hillshade with the affected-area outlines of a few runs and the reference."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import thesis_style as ts

    ref = results[-1]
    xmin, xmax, ymin, ymax = ref["bounds"]
    extent = (xmin, xmax, ymin, ymax)
    x, y = cell_centres(ref)
    chosen = [r for r in results[:-1]
              if any(math.isclose(r["cfl"], w) for w in which)] + [ref]
    colors = list(ts.SERIES_COLORS[:len(chosen) - 1]) + [ts.BLACK]
    # window around the union of the footprints
    union = np.zeros_like(ref["release"])
    for r in chosen:
        union |= r["pft"] > threshold
    rows_, cols_ = np.nonzero(union | ref["release"])
    pad = 15
    r0, r1 = max(rows_.min() - pad, 0), min(rows_.max() + pad, union.shape[0] - 1)
    c0, c1 = max(cols_.min() - pad, 0), min(cols_.max() + pad, union.shape[1] - 1)
    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(1.0, 0.75), layout="constrained")
        ts.hillshade(ax, ref["dem"], extent, ref["cell_size"],
                     contour_levels=np.arange(np.floor(np.nanmin(ref["dem"]) / 100) * 100,
                                               np.nanmax(ref["dem"]), 100.0))
        handles = []
        for r, color in zip(chosen, colors):
            lw = 1.3 if r is ref else 0.9
            ax.contour(x, y, (r["pft"] > threshold).astype(float), levels=[0.5],
                       colors=[color], linewidths=lw)
            label = rf"CFL $= {r['cfl']:g}$" + (" (reference)" if r is ref else "")
            handles.append(Line2D([], [], color=color, linewidth=lw, label=label))
        ax.contour(x, y, ref["release"].astype(float), levels=[0.5], colors=["0.25"],
                   linewidths=0.9, linestyles="--")
        handles.append(Line2D([], [], color="0.25", linestyle="--", linewidth=0.9,
                              label="release area"))
        ax.set_xlim(x[0, c0], x[0, c1])
        ax.set_ylim(y[r0, 0], y[r1, 0])
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel(r"$x$ (m)")
        ax.set_ylabel(r"$y$ (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        fig.legend(handles=handles, loc="outside right center")
        return ts.save(fig, out_dir, "cfl_convergence_footprints")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cfl", type=float, nargs="+", default=None,
                        help="CFL numbers; the smallest is the reference")
    parser.add_argument("--quick", action="store_true",
                        help=f"only CFL = {CFL_CFG['quick_cfl']}")
    parser.add_argument("--reference-runs", type=int, default=CFL_CFG["reference_runs"],
                        help="runs of the smallest CFL averaged into the reference")
    parser.add_argument("--particles-per-cell", type=int,
                        default=SETTINGS["released_particles_per_cell"])
    parser.add_argument("--sim-model", default=SETTINGS["sim_model"])
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    # largest CFL first, the reference (smallest CFL) last
    cfl = sorted(set(args.cfl or (CFL_CFG["quick_cfl"] if args.quick else CFL_CFG["cfl"])),
                 reverse=True)
    if len(cfl) < 2 or min(cfl) <= 0:
        parser.error("need at least two positive CFL values")
    settings = dict(SETTINGS, sim_model=args.sim_model,
                    released_particles_per_cell=args.particles_per_cell)
    print(f"=== CFL convergence: avaWog, {args.sim_model}, "
          f"Voellmy mu = {settings['friction_coefficient']:g}, "
          f"xi = {settings['drag_coefficient']:g}, "
          f"N_ppc = {settings['released_particles_per_cell']}, CFL = {cfl} ===")
    # warm-up run (GPU clocks, driver caches), not used: the first timed run
    # would otherwise be slower than the following ones
    run(settings_for(settings, cfl[0]))

    def report(r):
        print(f"CFL = {r['cfl']:5.3g}: {r['particles']:8d} particles, {r['steps']} steps, "
              f"t = {r['t_end']:.1f} s, setup {r['t_setup']:.2f} s, "
              f"simulation {r['t_sim']:.2f} s")

    results = []
    for c in cfl[:-1]:
        # the first run gives the results; the run time is the median of
        # CFL_CFG["timing_runs"] runs (robust against other load on the GPU)
        runs = []
        for _ in range(max(1, CFL_CFG["timing_runs"])):
            runs.append(run(settings_for(settings, c)))
            report(runs[-1])
        r = runs[0]
        for key in TIMES:
            values = np.array([x[key] for x in runs])
            r[key] = float(np.median(values))
            r[key + "_std"] = float(values.std(ddof=1)) if len(runs) > 1 else 0.0
        results.append(r)
    ref_runs = []
    for _ in range(max(1, args.reference_runs)):
        ref_runs.append(run(settings_for(settings, cfl[-1])))
        report(ref_runs[-1])
    ref = mean_run(ref_runs)

    noise = noise_floor(ref_runs, CFL_CFG) if len(ref_runs) > 1 else None
    rows = analyse(results, ref, CFL_CFG)
    print()
    print_table(rows)
    if noise is not None:
        print(f"\nspread of the {len(ref_runs)} reference runs (CFL = {cfl[-1]:g}, each "
              f"against the mean of the others): L2 h_peak {noise['l2_pft']:.4f}, "
              f"L2 v_peak {noise['l2_pfv']:.4f}, IoU {noise['iou']:.4f}; std travel angle "
              f"{ref['travel_angle_std']:.4f} deg, std s_3D {ref['travel_length_3d_std']:.2f} m")
    order_h = convergence_order(rows, "l2_pft")
    order_v = convergence_order(rows, "l2_pfv")
    print(f"\nobserved order (log-log slope over CFL): h_peak {order_h:.2f}, "
          f"v_peak {order_v:.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "cfl_convergence.csv")
    write_tex(rows, args.out_dir / "cfl_convergence.tex")
    thesis = args.out_dir / "pics" / "thesis"
    plot_errors(rows, thesis, noise)
    plot_runtime(rows, thesis)
    plot_footprints(results + [ref], thesis, CFL_CFG["affected_threshold"],
                    CFL_CFG["footprint_cfl"])

    checks = [("every run stops before max_steps",
               not any(r["max_steps_reached"] for r in rows)),
              ("L2 errors of the largest CFL exceed those of the second smallest CFL",
               len(rows) < 3 or (rows[0]["l2_pft"] > rows[-2]["l2_pft"]
                                 and rows[0]["l2_pfv"] > rows[-2]["l2_pfv"]))]
    default = [r for r in rows[:-1] if math.isclose(r["cfl"], CFL_CFG["default_cfl"])]
    if default:
        checks.append((f"CFL = {CFL_CFG['default_cfl']:g} reaches IoU >= "
                       f"{CFL_CFG['min_iou_default']} with the reference",
                       default[0]["iou"] >= CFL_CFG["min_iou_default"]))
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
