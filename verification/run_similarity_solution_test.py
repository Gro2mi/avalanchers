"""Run the similarity solution test with your own avalanchers simulation.

Adapted from AvaFrame's ``runScripts/runSimilaritySol.py`` +
``ana1Tests/simiSolTest.py``: the similarity solution of Hutter, Siegel &
Savage (1993, Acta Mechanica) for a granular avalanche spreading down an
inclined plane is compared to the avalanchers simulation - everything runs
in this script, no output files are generated first.

The parabolic release heap (thickness relTh at the center, 0 at the edges
of the ellipse with main axes L_x, L_y) is built in memory and handed to
the simulation with ``set_release_areas``; the flow fields are decoded
DIRECTLY from the simulation's quantized ``grid_mass`` /
``grid_momentum`` buffers at every save time and compared to the analytic
solution (relative L2/LMax errors of flow thickness and momentum). The
same plots as the AvaFrame simiSol module are produced (see
``similarity_solution_test.py``): the 2x6 summary figure (twin-axis
profiles along/across the flow, thickness contours, error curves,
parameters), the error-vs-time figure and - with ``--plot-sequence`` -
profile and contour comparisons at every save time, each stitched into a
looping animated GIF (the heap sliding and spreading).

The test needs pure Coulomb friction with delta < zeta (AvaFrame values:
35 deg plane, 25 deg bed friction = 10 deg excess, matching the similarity
solution's assumption of a constant downslope acceleration).

Usage:
    python run_similarity_solution_test.py              # run with the SETTINGS below
    python run_similarity_solution_test.py --t-end 10   # shorter run
    python run_similarity_solution_test.py --plot-sequence --show
"""

import argparse
import math
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "verification" / "outputs" / "similarity_solution_test"

# ---------------------------------------------------------------------------
# Simulation settings - edit here or override on the command line.
# Any key of the avalanchers `Settings` schema is accepted (see README /
# crates/compute_core/src/settings.rs). The release heap is set in memory;
# only the DEM comes from a file.
# ---------------------------------------------------------------------------
SETTINGS = {
    "dem_path": str(REPO_ROOT / "data" / "avaframe" / "avaSimilaritySol.png"),
    "sim_model": "curvilinear",
    "friction_model": "coulomb",
    # tan(delta); set from SIMI_SOL_CFG["delta"] in main()
    "friction_coefficient": 0.46630765815499864,
    "density": 200.0,
    "released_particles_per_cell": 4,
    # drag_coefficient is set huge so any turbulent drag term vanishes and
    # pure Coulomb friction remains (same as avalanchers_ana1_tests.py)
    "drag_coefficient": 1.0e12,
    "cfl": 0.05,
    "velocity_threshold": 0.01,
    "enable_particle_interaction": True,
    # hexagonal-packing relaxation of freshly initialized particles
    # (pinned so the validation runs are independent of the default)
    "enable_particle_relaxation": True,
    "max_steps": 20000,
}

# ---------------------------------------------------------------------------
# Similarity solution configuration - values from AvaFrame's
# simiSol_com1DFACfg.ini [SIMISOL]
# ---------------------------------------------------------------------------
SIMI_SOL_CFG = {
    "grav_acc": 9.81,
    "l_x": 80.0,          # heap main axis along the slope [m]
    "l_y": 80.0,          # heap main axis across the slope [m]
    "rel_th": 4.0,        # release thickness at the heap center [m]
    "zeta": 35.0,         # plane inclination [deg] (None = measure from DEM)
    "delta": 25.0,        # bed friction angle [deg] (10 deg excess over zeta)
    "phi": 25.0,          # internal friction angle [deg]
    "flag_earth": False,  # earth pressure coefficients (ini: False)
    "t_end": 20.0,        # simulated end time [s]
    "save_interval": 1.0, # field sampling interval [s]
    "t_save": 5.0,        # time highlighted in the plots [s]
    "relative_error": True,  # ini: relativError = True
    "scale_coef": 1.02,      # ini: scaleCoef = 1.02
    "plot_sequence": False,  # ini: plotSequence - per-time-step figures
}

BATCH_STEPS = 20


def build_release(header, cfg):
    """Parabolic release heap of the similarity solution (port of
    getReleaseThickness): relTh * (1 - x1^2/Lx^2 - y^2/Ly^2) with
    x1 the along-slope coordinate, centered on the heap center
    (map position stored in cfg["center"])."""
    cs = header["cellsize"]
    xs = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * cs
    ys = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * cs
    x_map, y_map = np.meshgrid(xs, ys)
    xc, yc = cfg["center"]
    cos_z = math.cos(math.radians(cfg["zeta"]))
    x1 = (x_map - xc) / cos_z
    y1 = y_map - yc
    thickness = cfg["rel_th"] * (1.0 - x1 * x1 / cfg["l_x"] ** 2
                                 - y1 * y1 / cfg["l_y"] ** 2)
    # zero out sliver thicknesses: the loader counts cells above 1e-3 m while
    # the particle initializer skips cells at or below 0.01 m (see
    # avalanchers_ana1_tests.py)
    thickness = np.where(thickness <= 0.02, 0.0, thickness)
    return thickness, x1, y1


def measured_plane_angle_deg(dem, header):
    """Slope angle from the DEM (central differences, as the sim does)."""
    cs = header["cellsize"]
    zx = (dem[:, 2:] - dem[:, :-2]) / (2.0 * cs)
    zy = (dem[2:, :] - dem[:-2, :]) / (2.0 * cs)
    slope = np.median(np.concatenate([np.abs(zx).ravel(), np.abs(zy).ravel()]))
    return math.degrees(math.atan(slope))


def run_simulation(settings, cfg):
    """Create, run and sample the simulation; compare to the similarity
    solution at every save time.

    Returns (sim, dem, header, samples, errors, sim_name) where `samples`
    holds the decoded fields per save time.
    """
    import avalanchers
    import similarity_solution_test as sst

    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    dem = np.asarray(sim.dem, dtype=float)
    xmin, xmax, ymin, ymax = (float(v) for v in sim.dem_bounds)
    header = {"cellsize": float(sim.cell_size), "ncols": dem.shape[1],
              "nrows": dem.shape[0], "xmin": xmin, "xmax": xmax,
              "ymin": ymin, "ymax": ymax}

    # heap centered at the map origin (the similarity solution's frame)
    cfg["center"] = (0.0, 0.0)
    measured = measured_plane_angle_deg(dem, header)
    if abs(measured - cfg["zeta"]) > 0.5:
        print(f"WARNING: DEM slope is {measured:.2f} deg, config says "
              f"{cfg['zeta']:.2f} deg - using the DEM angle")
        cfg["zeta"] = measured
    settings["friction_coefficient"] = math.tan(math.radians(cfg["delta"]))
    mu = settings["friction_coefficient"]

    release, _, _ = build_release(header, cfg)
    sim.set_release_areas(np.ascontiguousarray(release.astype(np.float32)))

    sim_name = "%s_coulomb_delta%g" % (pathlib.Path(settings["dem_path"]).stem,
                                       cfg["delta"])
    print(f"=== Similarity solution test (Hutter et al. 1993) - {sim_name} ===")
    print(f"DEM: {settings['dem_path']}")
    print(f"plane {cfg['zeta']:.2f} deg, bed friction {cfg['delta']:.2f} deg "
          f"(mu={mu:.6f}), L_x={cfg['l_x']} m, L_y={cfg['l_y']} m, "
          f"relTh={cfg['rel_th']} m, t_end={cfg['t_end']} s")

    # orientation sanity check: the heap is centered on map (0, 0)
    sim.prepare()
    pos = np.asarray(sim.particles_position_xy, dtype=np.float64)
    pos = pos[np.isfinite(pos).all(axis=1)]
    print(f"particles: {len(pos)}, heap centroid on map: "
          f"({pos[:, 0].mean():+.1f}, {pos[:, 1].mean():+.1f}) m - expected (+0.0, +0.0)")

    sol = sst.similarity_solution(cfg["zeta"], cfg["delta"], cfg["phi"],
                                  cfg["flag_earth"], cfg["l_x"], cfg["l_y"],
                                  cfg["rel_th"], cfg["grav_acc"], cfg["t_end"])

    density = settings["density"]
    cs = header["cellsize"]
    cos_z = math.cos(math.radians(cfg["zeta"]))
    xs = xmin + (np.arange(header["ncols"]) + 0.5) * cs
    ys = ymin + (np.arange(header["nrows"]) + 0.5) * cs
    x_map, y_map = np.meshgrid(xs, ys)
    x1_grid = (x_map - cfg["center"][0]) / cos_z
    y1_grid = y_map - cfg["center"][1]
    jac = sst.jacobian(dem, cs)

    samples = {"t": [], "h": [], "u": [], "v": []}
    errors = {k: [] for k in ("h_l2", "h_l2rel", "h_lmax", "h_lmaxrel",
                              "vh_l2", "vh_l2rel", "vh_lmax", "vh_lmaxrel")}
    save_times = sorted({min(t, cfg["t_end"]) for t in np.arange(
        cfg["save_interval"], cfg["t_end"] + 1e-6, cfg["save_interval"])})

    def record(info):
        """Sample the grid fields, compare to the analytic solution."""
        t = info.elapsed_time
        h, u, v, mass = sst.fields_from_grid(sim, dem.shape, cs, density, jac)
        samples["t"].append(t)
        samples["h"].append(h)
        samples["u"].append(u)
        samples["v"].append(v)
        h_ana = sst.similarity_fields(sol, t, x1_grid, y1_grid, cfg["l_x"],
                                      cfg["l_y"], cfg["rel_th"], cfg["zeta"],
                                      cfg["delta"], cfg["grav_acc"])
        h_l2, h_l2rel, h_lmax, h_lmaxrel = sst.norm_l2_scal(
            h_ana["h"], h, cs, cos_z)
        vh_l2, vh_l2rel, vh_lmax, vh_lmaxrel = sst.norm_l2_vect(
            (h_ana["h"] * h_ana["u"], h_ana["h"] * h_ana["v"]),
            (h * u, h * v), cs, cos_z)
        errors["h_l2"].append(h_l2)
        errors["h_l2rel"].append(h_l2rel)
        errors["h_lmax"].append(h_lmax)
        errors["h_lmaxrel"].append(h_lmaxrel)
        errors["vh_l2"].append(vh_l2)
        errors["vh_l2rel"].append(vh_l2rel)
        errors["vh_lmax"].append(vh_lmax)
        errors["vh_lmaxrel"].append(vh_lmaxrel)
        total_h = float(np.nansum(h))
        x_com_sim = float(np.nansum(h * x_map) / total_h) if total_h > 0 else float("nan")
        return t, h_l2rel, h_lmaxrel, vh_l2rel, vh_lmaxrel, x_com_sim, h_ana["x_center"]

    info = sim.run_n_steps(1)
    first_h, _, _, first_mass = sst.fields_from_grid(sim, dem.shape, cs,
                                                     density, jac)
    sst.check_grid_mass_consistency(
        sim, first_h, np.asarray(sim.particles_mass, dtype=np.float64))
    last_step = -1
    idx_time = 0

    print()
    print(f"{'t [s]':>6} {'steps':>6} {'hL2rel':>10} {'hLMaxrel':>10} "
          f"{'huL2rel':>10} {'huLMaxrel':>10} {'xCom sim/ana [m]':>20}")
    while idx_time < len(save_times) and info.timestep != last_step:
        last_step = info.timestep
        while idx_time < len(save_times) and info.elapsed_time >= save_times[idx_time] - 1e-6:
            t, h_l2r, h_lmaxr, vh_l2r, vh_lmaxr, x_com_sim, x_com_ana = record(info)
            print(f"{t:6.2f} {info.timestep:6d} {h_l2r:10.4f} {h_lmaxr:10.4f} "
                  f"{vh_l2r:10.4f} {vh_lmaxr:10.4f} {x_com_sim:9.1f}/{x_com_ana:<9.1f}")
            idx_time += 1
        if idx_time >= len(save_times):
            break
        info = sim.run_n_steps(BATCH_STEPS)

    if not samples["t"]:
        print(f"WARNING: no samples were taken - the simulation stopped at "
              f"t={info.elapsed_time:.2f} s before reaching the first save "
              f"time of {save_times[0]:.2f} s")
    samples["t"] = np.asarray(samples["t"])
    for key in ("h", "u", "v"):
        samples[key] = np.asarray(samples[key])
    for key in errors:
        errors[key] = np.asarray(errors[key])
    print(f"\nstopped after {info.timestep} steps, t = {info.elapsed_time:.2f} s "
          f"({len(samples['t'])} field samples)")
    return sim, dem, header, samples, errors, sim_name


def main():
    parser = argparse.ArgumentParser(description="avalanchers similarity solution test")
    parser.add_argument("--t-end", type=float, default=None,
                        help="simulated end time in seconds (default: cfg t_end)")
    parser.add_argument("--save-interval", type=float, default=None,
                        help="field sampling interval in seconds")
    parser.add_argument("--delta", type=float, default=None,
                        help="bed friction angle in degrees")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--gpu", default=None, help="GPU name substring to select")
    parser.add_argument("--plot-sequence", action="store_true",
                        help="save profile + contour comparison figures at "
                             "every save time")
    parser.add_argument("--show", action="store_true", help="open the plot windows")
    args = parser.parse_args()

    # headless backend unless an interactive window was requested
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    settings = dict(SETTINGS)
    cfg = dict(SIMI_SOL_CFG)
    if args.t_end is not None:
        cfg["t_end"] = args.t_end
    if args.save_interval is not None:
        cfg["save_interval"] = args.save_interval
    if args.delta is not None:
        cfg["delta"] = args.delta
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    settings["friction_coefficient"] = math.tan(math.radians(cfg["delta"]))
    cfg["plot_sequence"] = cfg["plot_sequence"] or args.plot_sequence

    import similarity_solution_test as sst

    sim, dem, header, samples, errors, sim_name = run_simulation(settings, cfg)
    if len(samples["t"]) == 0:
        raise RuntimeError("no field samples - nothing to plot")

    relative = cfg["relative_error"]
    err_h = errors["h_l2rel" if relative else "h_l2"]
    err_h_max = errors["h_lmaxrel" if relative else "h_lmax"]
    err_vh = errors["vh_l2rel" if relative else "vh_l2"]
    err_vh_max = errors["vh_lmaxrel" if relative else "vh_lmax"]

    # analytic solution + limits at the highlighted time
    cos_z = math.cos(math.radians(cfg["zeta"]))
    cs = header["cellsize"]
    xs = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * cs
    ys = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * cs
    x_map, y_map = np.meshgrid(xs, ys)
    x1_grid = (x_map - cfg["center"][0]) / cos_z
    y1_grid = y_map - cfg["center"][1]

    out_dir = pathlib.Path(args.out_dir) / "pics"
    sol = sst.similarity_solution(cfg["zeta"], cfg["delta"], cfg["phi"],
                                  cfg["flag_earth"], cfg["l_x"], cfg["l_y"],
                                  cfg["rel_th"], cfg["grav_acc"], cfg["t_end"])
    i_save = int(min(np.searchsorted(samples["t"], cfg["t_save"]),
                     len(samples["t"]) - 1))
    t_plot = samples["t"][i_save]
    field = {k: samples[k][i_save] for k in ("h", "u", "v")}
    h_ana = sst.similarity_fields(sol, t_plot, x1_grid, y1_grid, cfg["l_x"],
                                  cfg["l_y"], cfg["rel_th"], cfg["zeta"],
                                  cfg["delta"], cfg["grav_acc"])
    speed = np.hypot(field["u"], field["v"])
    limits = sst.get_plot_limits(field["h"], speed, cfg["center"][0],
                                 cfg["center"][1], header, cfg["scale_coef"])

    if cfg["plot_sequence"]:
        contour_pngs = []
        profile_pngs = []
        for i, t in enumerate(samples["t"]):
            field_i = {k: samples[k][i] for k in ("h", "u", "v")}
            h_ana_i = sst.similarity_fields(sol, t, x1_grid, y1_grid, cfg["l_x"],
                                            cfg["l_y"], cfg["rel_th"],
                                            cfg["zeta"], cfg["delta"],
                                            cfg["grav_acc"])
            speed_i = np.hypot(field_i["u"], field_i["v"])
            limits_i = sst.get_plot_limits(field_i["h"], speed_i,
                                           cfg["center"][0], cfg["center"][1],
                                           header, cfg["scale_coef"])
            profile_pngs.append(sst.plot_profile_comparison(
                header, field_i, h_ana_i, limits_i, t, sim_name, out_dir))
            contour_pngs.append(sst.plot_contour_comparison(
                header, field_i["h"], h_ana_i, limits_i, t, sim_name, out_dir))
        # animated sequences: the analytic heap slides and spreads
        sst.assemble_gif(contour_pngs, out_dir / (sim_name + "_contours.gif"))
        sst.assemble_gif(profile_pngs, out_dir / (sim_name + "_profiles.gif"))

    sst.plot_error_time(samples["t"], err_h, err_h_max, err_vh, err_vh_max,
                        relative, cfg["t_save"], sim_name, out_dir)
    sst.plot_simi_sol_summary(header, field, h_ana, limits,
                              {"h_l2": err_h, "h_lmax": err_h_max,
                               "vh_l2": err_vh, "vh_lmax": err_vh_max},
                              samples["t"], t_plot, cfg, sim_name, settings,
                              out_dir)

    print()
    print(f"errors at t = {t_plot:.2f} s "
          f"({'relative' if relative else 'absolute'}):")
    print(f"h  L2 : {err_h[i_save]:.4f}   h  LMax : {err_h_max[i_save]:.4f}")
    print(f"hu L2 : {err_vh[i_save]:.4f}   hu LMax : {err_vh_max[i_save]:.4f}")
    return samples, errors


if __name__ == "__main__":
    main()
