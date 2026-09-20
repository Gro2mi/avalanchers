"""Run the dam break test with your own avalanchers simulation.

Adapted from AvaFrame's ``runScripts/runDamBreak.py`` +
``ana1Tests/damBreak.py``: instead of writing com1DFA output files and
reading them back, the avalanchers simulation is created, run and sampled
directly in this script - no output files are needed before the analysis.

The numerical flow fields are decoded DIRECTLY from the simulation's
quantized ``grid_mass`` / ``grid_momentum`` buffers (the exact fields the
grid physics pass consumed) every ``DAM_BREAK_CFG["save_interval"]`` seconds
of simulated time, and compared to the analytic dam break solution of
Faccanoni & Mangeney (2012, Test 2, case 1.2, dry bed, Coulomb friction).
The same plots as the AvaFrame dam break module are produced (see
``dam_break_test.py``): analytic solution figures, error vs time, and the
2x6 summary figure whose bird's-eye map shows the PEAK flow thickness up to
the highlighted time (``t_save``, default 15 s) - a running per-cell maximum
of the decoded grid fields, so it is accessible at every sampled time
without needing the Finished state - on a continuous 0-1.0 m colorscale.
All plots use the fixed x window ``DAM_BREAK_CFG["x_lim"]`` ([-200, 400] m).

With ``--plot-sequence`` (ini: plotSequence) the optional result is added:
the comparison cross-cut figure (flow thickness / velocity / hv
cross-sections, initial vs numerical vs analytical) for every saved time
step, stitched into a looping animated GIF, like the animation in the
AvaFrame dam break documentation.

The dam geometry (front/back position, release thickness h0) and the slope
angle are measured automatically from the release raster and the DEM, so
any plane DEM with a rectangular release works. The analytic solution is
exact for pure Coulomb friction, hence ``"friction_model": "coulomb"``.
Following avalanchers_ana1_tests.py, the bed friction angle default is
delta = 12 deg (10 deg excess over the 22 deg slope) instead of AvaFrame's
21 deg: a near-critical slope never starts to slide under avalanchers'
friction solver.

Usage:
    python run_dam_break_test.py                    # run with the SETTINGS below
    python run_dam_break_test.py --show             # also open the plot windows
    python run_dam_break_test.py --t-end 30 --save-interval 0.5 --delta 12
"""

import argparse
import math
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "verification" / "outputs" / "dam_break_test"

# ---------------------------------------------------------------------------
# Simulation settings - edit here or override on the command line.
# Any key of the avalanchers `Settings` schema is accepted (see README /
# crates/compute_core/src/settings.rs).
# ---------------------------------------------------------------------------
SETTINGS = {
    "dem_path": str(REPO_ROOT / "data" / "avaframe" / "avaDamBreak.png"),
    "release_areas_path": str(REPO_ROOT / "data" / "avaframe" / "avaDamBreakreleaseTexture.png"),
    "sim_model": "curvilinear",
    # Coulomb friction: the analytic solution requires it
    "friction_model": "coulomb",
    # tan(delta), set from DAM_BREAK_CFG["delta"] in main()
    "friction_coefficient": math.tan(math.radians(21)),
    "density": 200.0,
    "released_particles_per_cell": 4,
    "cfl": 0.05,
    "velocity_threshold": 0.01,
    "enable_particle_interaction": True,
    "enable_particle_relaxation": True,
    "max_steps": 20000,
}

# ---------------------------------------------------------------------------
# Dam break test configuration - values from AvaFrame's
# damBreak_com1DFACfg.ini [DAMBREAK] (documented deviations noted)
# ---------------------------------------------------------------------------
DAM_BREAK_CFG = {
    "grav_acc": 9.81,
    "t_end": 20.0,        # simulated end time [s] (ini: tEnd = 30)
    "save_interval": 1.0, # field sampling interval [s] (ini: dt = 0.1)
    "phi": None,          # slope angle [deg]; None = measure from the DEM
    "delta": 12.0,        # bed friction angle [deg] (ini: 21, see docstring)
    # comparison domain, horizontal meters relative to the dam front
    "x_start": -200.0,
    "x_end": 220.0,
    "y_start": -50.0,
    "y_end": 50.0,
    "t_save": 15.0,       # time highlighted in the plots [s]
    "relative_error": True,  # ini: relativError = True
    "scale_coef": 1.05,      # ini: scaleCoef = 1.05
    "x_lim": (-200.0, 400.0),  # fixed x window [m] for all dam break plots
    "plot_sequence": False,  # ini: plotSequence - cross-cut figures for all
                             # saved time steps + animated GIF
}

BATCH_STEPS = 20

# SimInfo flag bit (must match crates/compute_core/src/shaders/utils.wgsl)
SIM_INFO_STOPPED = 1 << 31


def build_dem_dict(sim):
    """Wrap the simulation's DEM in a dict (rasterData + header, row 0 = south)."""
    dem_array = np.asarray(sim.dem, dtype=float)
    xmin, xmax, ymin, ymax = (float(v) for v in sim.dem_bounds)
    return {
        "rasterData": dem_array,
        "header": {
            "cellsize": float(sim.cell_size),
            "ncols": dem_array.shape[1],
            "nrows": dem_array.shape[0],
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        },
    }


def build_geometry(sim, header, frame):
    """Measure the dam geometry from the release raster.

    Returns a dict with h_l (release thickness), x_back_rel (dam back,
    along-slope relative to the front), f_back/f_front (horizontal flow
    coordinates of the dam) and p_center (cross-flow dam center). Also pins
    the profile row (through the dam center) and the analytic table axis
    into `frame`.
    """
    import dam_break_test

    release = np.asarray(sim.release_areas, dtype=float)
    release_cells = release > 0
    if not release_cells.any():
        raise RuntimeError("no release cells found - check release_areas_path")
    f, p = dam_break_test.flow_coordinates(header, frame)
    h_l = float(np.median(release[release_cells]))
    f_back = float(f[release_cells].min())
    f_front = float(f[release_cells].max())
    p_center = float(p[release_cells].mean())
    # profile row through the dam center (AvaFrame: nx_loc = ny // 2)
    frame["f"], frame["p"] = f, p
    frame["row_index"] = int(np.argmin(np.abs(p[:, 0] - p_center)))
    frame["f_profile"] = f[frame["row_index"], :]
    frame["s_rel_profile"] = (frame["f_profile"] - f_front) / frame["cos_phi"]
    return {
        "h_l": h_l,
        "x_back_rel": (f_back - f_front) / frame["cos_phi"],
        "f_back": f_back,
        "f_front": f_front,
        "p_center": p_center,
    }


def sample_fields(sim, dem, header, frame, density):
    """One field sample decoded from the grid mass/momentum buffers."""
    import dam_break_test

    jac = dam_break_test.jacobian(dem["rasterData"], header["cellsize"])
    h, u, v, mass = dam_break_test.fields_from_grid(
        sim, dem["rasterData"].shape, header["cellsize"], density, jac, frame=frame)
    return {"h": h, "u": u, "v": v}, mass


def error_domain(frame, dam, cfg, x_mid):
    """Mask of the comparison domain (port of getDamExtend + analyzeResults):
    cross-flow band around the dam center, from the analytic material
    mid-point (x_mid) to x_end, all along-slope relative to the dam front."""
    s_rel = (frame["f"] - dam["f_front"]) / frame["cos_phi"]
    half_width = (cfg["y_end"] - cfg["y_start"]) / 2.0
    x_end_rel = cfg["x_end"] / frame["cos_phi"]
    return ((np.abs(frame["p"] - dam["p_center"]) <= half_width)
            & (s_rel >= x_mid) & (s_rel <= x_end_rel))


def compute_limits(sample_list, scale_coef):
    """Plot limits (port of outAna1Plots.getPlotLimits): scale_coef * the max
    over all samples of flow thickness, velocity and momentum."""
    max_ft = max(float(np.nanmax(s["h"])) for s in sample_list)
    max_fv = max(float(np.nanmax(np.hypot(s["u"], s["v"]))) for s in sample_list)
    max_fm = max(float(np.nanmax(s["h"] * np.hypot(s["u"], s["v"]))) for s in sample_list)
    return {"max_ft": scale_coef * max_ft,
            "max_fv": scale_coef * max_fv,
            "max_fm": scale_coef * max_fm}


def run_simulation(settings, cfg):
    """Create, run and sample the simulation.

    Returns (sim, dem, header, frame, dam, samples, errors, sim_name):
    `samples` holds the decoded fields per save time, `errors` the error
    arrays (absolute and relative) of flow thickness and momentum.
    """
    import avalanchers
    import dam_break_test

    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.prepare()

    dem = build_dem_dict(sim)
    header = dem["header"]
    frame = dam_break_test.flow_frame(dem["rasterData"], header)
    if cfg["phi"] is not None:
        if abs(cfg["phi"] - frame["phi_deg"]) > 0.5:
            print(f"WARNING: DEM slope is {frame['phi_deg']:.2f} deg, config says "
                  f"{cfg['phi']:.2f} deg - using the config angle")
        frame["phi_deg"] = cfg["phi"]
        frame["cos_phi"] = math.cos(math.radians(cfg["phi"]))
    dam = build_geometry(sim, header, frame)

    mu = settings["friction_coefficient"]
    sim_name = "%s_%s_delta%g" % (pathlib.Path(settings["dem_path"]).stem,
                                  settings["friction_model"],
                                  math.degrees(math.atan(mu)))
    print(f"=== Dam break test (Faccanoni & Mangeney 2012, dry bed) - {sim_name} ===")
    print(f"DEM: {settings['dem_path']}")
    print(f"plane {frame['phi_deg']:.2f} deg, bed friction "
          f"{math.degrees(math.atan(mu)):.2f} deg (mu={mu:.6f}), "
          f"h0={dam['h_l']:.2f} m, dam f-range "
          f"[{dam['f_back']:.1f}, {dam['f_front']:.1f}] m, "
          f"sampling every {cfg['save_interval']} s until t = {cfg['t_end']} s")

    density = settings["density"]
    cs = header["cellsize"]
    cos_phi = frame["cos_phi"]
    s_rel_grid = (frame["f"] - dam["f_front"]) / cos_phi
    samples = {"t": [], "h": [], "u": [], "v": [], "peak_h": []}
    errors = {k: [] for k in ("h_l2", "h_l2rel", "h_lmax", "h_lmaxrel",
                              "vh_l2", "vh_l2rel", "vh_lmax", "vh_lmaxrel")}
    running_peak = None

    def record(info, check_consistency=False):
        """Sample the fields, compare to the analytic solution, append."""
        nonlocal running_peak
        t = info.elapsed_time
        sample, mass = sample_fields(sim, dem, header, frame, density)
        if check_consistency:
            dam_break_test.check_grid_mass_consistency(
                sim, mass, np.asarray(sim.particles_mass, dtype=np.float64))
        # peak flow thickness accessible at every sampled time: running
        # per-cell maximum of the decoded fields (1 x save_interval resolution)
        if running_peak is None:
            running_peak = sample["h"].copy()
        else:
            np.maximum(running_peak, sample["h"], out=running_peak)
        samples["t"].append(t)
        samples["h"].append(sample["h"])
        samples["u"].append(sample["u"])
        samples["v"].append(sample["v"])
        samples["peak_h"].append(running_peak.copy())
        h_ana, u_ana, x_mid = dam_break_test.dam_break_solution(
            frame["phi_deg"], cfg["delta"], dam["h_l"], [t], s_rel_grid,
            cfg["grav_acc"], x_back_rel=dam["x_back_rel"])
        h_ana, u_ana = h_ana[0], u_ana[0]
        domain = error_domain(frame, dam, cfg, x_mid[0])
        speed = np.hypot(sample["u"], sample["v"])
        h_l2, h_l2rel, h_lmax, h_lmaxrel = dam_break_test.norm_l2_scal(
            h_ana[domain], sample["h"][domain], cs, cos_phi)
        vh_l2, vh_l2rel, vh_lmax, vh_lmaxrel = dam_break_test.norm_l2_scal(
            (h_ana * u_ana)[domain], (sample["h"] * speed)[domain], cs, cos_phi)
        errors["h_l2"].append(h_l2)
        errors["h_l2rel"].append(h_l2rel)
        errors["h_lmax"].append(h_lmax)
        errors["h_lmaxrel"].append(h_lmaxrel)
        errors["vh_l2"].append(vh_l2)
        errors["vh_l2rel"].append(vh_l2rel)
        errors["vh_lmax"].append(vh_lmax)
        errors["vh_lmaxrel"].append(vh_lmaxrel)
        return h_l2rel, h_lmaxrel, vh_l2rel, vh_lmaxrel

    # initial sample: the first grid pass holds the release column
    info = sim.run_n_steps(1)
    next_save = cfg["save_interval"]
    last_timestep = -1

    print()
    print(f"{'t [s]':>6} {'steps':>6} {'hL2rel':>10} {'hLMaxrel':>10} "
          f"{'huL2rel':>10} {'huLMaxrel':>10}")

    def record_and_print(info, check_consistency=False):
        h_l2r, h_lmaxr, vh_l2r, vh_lmaxr = record(info, check_consistency)
        print(f"{info.elapsed_time:6.2f} {info.timestep:6d} {h_l2r:10.4f} "
              f"{h_lmaxr:10.4f} {vh_l2r:10.4f} {vh_lmaxr:10.4f}")

    record_and_print(info, check_consistency=True)
    while True:
        if (info.flags & SIM_INFO_STOPPED) or sim.state == "Finished":
            break
        if info.elapsed_time >= cfg["t_end"]:
            break
        if info.timestep == last_timestep:
            print("simulation made no progress - stopping")
            break
        last_timestep = info.timestep
        info = sim.run_n_steps(BATCH_STEPS)
        if info.elapsed_time >= next_save - 1e-6:
            record_and_print(info)
            while info.elapsed_time >= next_save - 1e-6:
                next_save += cfg["save_interval"]

    samples["t"] = np.asarray(samples["t"])
    for key in ("h", "u", "v", "peak_h"):
        samples[key] = np.asarray(samples[key])
    for key in errors:
        errors[key] = np.asarray(errors[key])
    print(f"\nstopped after {info.timestep} steps, t = {info.elapsed_time:.2f} s "
          f"({len(samples['t'])} field samples)")
    return sim, dem, header, frame, dam, samples, errors, sim_name


def main():
    parser = argparse.ArgumentParser(description="avalanchers dam break test")
    parser.add_argument("--t-end", type=float, default=None,
                        help="simulated end time in seconds (default: cfg t_end)")
    parser.add_argument("--save-interval", type=float, default=None,
                        help="field sampling interval in seconds")
    parser.add_argument("--delta", type=float, default=None,
                        help="bed friction angle in degrees")
    parser.add_argument("--phi", type=float, default=None,
                        help="slope angle in degrees (default: measure from DEM)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--gpu", default=None, help="GPU name substring to select")
    parser.add_argument("--plot-sequence", action="store_true",
                        help="save the comparison cross-cut figures for all "
                             "saved time steps and stitch them into an "
                             "animated GIF")
    parser.add_argument("--show", action="store_true", help="open the plot windows")
    args = parser.parse_args()

    # headless backend unless an interactive window was requested
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    settings = dict(SETTINGS)
    cfg = dict(DAM_BREAK_CFG)
    if args.t_end is not None:
        cfg["t_end"] = args.t_end
    if args.save_interval is not None:
        cfg["save_interval"] = args.save_interval
    if args.delta is not None:
        cfg["delta"] = args.delta
    if args.phi is not None:
        cfg["phi"] = args.phi
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    settings["friction_coefficient"] = math.tan(math.radians(cfg["delta"]))
    cfg["plot_sequence"] = cfg["plot_sequence"] or args.plot_sequence

    import dam_break_test

    sim, dem, header, frame, dam, samples, errors, sim_name = run_simulation(
        settings, cfg)

    # plot limits from all samples (port of getPlotLimits)
    frame["limits"] = compute_limits(
        [{k: samples[k][i] for k in ("h", "u", "v")} for i in range(len(samples["t"]))],
        cfg["scale_coef"])

    # analytic solution table for the analytic-only figures
    times = np.arange(0.0, samples["t"][-1] + 1e-6, 0.1)
    s_axis = np.arange(cfg["x_start"] / frame["cos_phi"],
                       cfg["x_end"] / frame["cos_phi"], 0.5)
    h_ana, u_ana, _ = dam_break_test.dam_break_solution(
        frame["phi_deg"], cfg["delta"], dam["h_l"], times, s_axis,
        cfg["grav_acc"], x_back_rel=dam["x_back_rel"])
    sol = {"t": times, "s": s_axis, "h": h_ana, "u": u_ana}

    # fixed axis limits for the animation: numeric and analytic maxima over
    # ALL times, so no panel rescales between frames
    frame["ylim"] = {
        "ft": max(frame["limits"]["max_ft"], cfg["scale_coef"] * float(np.nanmax(h_ana))),
        "fv": max(frame["limits"]["max_fv"], cfg["scale_coef"] * float(np.nanmax(u_ana))),
        "fm": max(frame["limits"]["max_fm"],
                  cfg["scale_coef"] * float(np.nanmax(h_ana * u_ana))),
    }

    out_dir = pathlib.Path(args.out_dir) / "pics"
    dam_break_test.plot_dam_ana_results(sol, cfg["t_save"], out_dir,
                                        x_lim=cfg["x_lim"])
    if cfg["plot_sequence"]:
        # optional result: comparison cross-cut figure for every saved time
        # step (AvaFrame's plotSequence), stitched into an animated GIF
        pngs = []
        for i in range(len(samples["t"])):
            sample0 = {k: samples[k][0] for k in ("h", "u", "v")}
            sample_t = {k: samples[k][i] for k in ("h", "u", "v")}
            pngs.append(dam_break_test.plot_comparison_dam(
                sample0, sample_t, frame, dam, samples["t"][i], cfg, sim_name,
                out_dir))
        dam_break_test.assemble_gif(pngs, out_dir / (sim_name + "_compare.gif"))
    relative = cfg["relative_error"]
    err_h = errors["h_l2rel" if relative else "h_l2"]
    err_h_max = errors["h_lmaxrel" if relative else "h_lmax"]
    err_vh = errors["vh_l2rel" if relative else "vh_l2"]
    err_vh_max = errors["vh_lmaxrel" if relative else "vh_lmax"]
    dam_break_test.plot_error_time(samples["t"], err_h, err_h_max, err_vh,
                                   err_vh_max, relative, cfg["t_save"], sim_name,
                                   out_dir)
    dam_break_test.plot_dam_break_summary(dem["rasterData"], header, frame, dam,
                                          samples,
                                          {"h_l2": err_h, "h_lmax": err_h_max,
                                           "vh_l2": err_vh, "vh_lmax": err_vh_max},
                                          cfg["t_save"], cfg, sim_name, settings,
                                          out_dir)

    i_save = int(min(np.searchsorted(samples["t"], cfg["t_save"]),
                     len(samples["t"]) - 1))
    print()
    print(f"errors at t = {samples['t'][i_save]:.2f} s "
          f"({'relative' if relative else 'absolute'}):")
    print(f"h  L2 : {err_h[i_save]:.4f}   h  LMax : {err_h_max[i_save]:.4f}")
    print(f"hv L2 : {err_vh[i_save]:.4f}   hv LMax : {err_vh_max[i_save]:.4f}")
    return samples, errors


if __name__ == "__main__":
    main()
