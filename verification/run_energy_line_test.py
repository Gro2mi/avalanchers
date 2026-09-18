"""Run the energy line test with your own avalanchers simulation.

Adapted from AvaFrame's ``runScripts/runEnergyLineTest.py``: instead of
writing com1DFA output files and reading them back, the avalanchers
simulation is created, run and sampled directly in this script - no output
files are needed before the analysis.

While the simulation advances, the mass-averaged center-of-mass path
(position, altitude and squared velocity, weighted by particle mass) is
sampled live from the particle buffers every ``SAVE_STEPS`` steps, exactly
like AvaFrame's ``pathFromPart=True`` energy line test. Afterwards the same
analysis and plots as the AvaFrame ``energyLineTest`` module are produced
(see ``energyLineTest.py``).

The analytic alpha line is only exact for pure Coulomb friction, where the
energy line has slope mu = friction_coefficient - run this test with
``"friction_model": "coulomb"``.

Usage:
    python runEnergyLineTest.py                     # run with the SETTINGS below
    python runEnergyLineTest.py --show              # also open the plot window
    python runEnergyLineTest.py --dem data/avaframe/avaBowl.png \\
        --release data/avaframe/avaBowlreleaseTexture.png --mu 0.3
    python runEnergyLineTest.py --max-steps 2000 --save-steps 10 --t-end 60
"""

import argparse
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "verification" / "outputs" / "energy_line_test"

# ---------------------------------------------------------------------------
# Simulation settings - edit here or override on the command line.
# Any key of the avalanchers `Settings` schema is accepted (see README /
# crates/compute_core/src/settings.rs).
# ---------------------------------------------------------------------------
SETTINGS = {
    "dem_path": str(REPO_ROOT / "data" / "avaframe" / "avaParabola.png"),
    "release_areas_path": str(REPO_ROOT / "data" / "avaframe" / "avaParabolareleaseTexture.png"),
    "sim_model": "curvilinear",
    # Coulomb friction: the energy line then has slope mu = friction_coefficient
    "friction_model": "coulomb",
    "friction_coefficient": 0.4,
    "density": 200.0,
    "released_particles_per_cell": 4,
    "max_steps": 6000,
    "velocity_threshold": 0.1,
}

# sample the center of mass path every N simulation steps
SAVE_STEPS = 25

# energy line test configuration (energyLineTestCfg.ini values)
ENERGY_LINE_TEST_CFG = {
    # regression length (in cells) for the profile extrapolation at the bottom
    "nCellsExtrapolation": 4,
    # also draw the alpha line shifted to the final runout point
    "shiftedAlphaLine": False,
    # AvaFrame's corrected-s option needs per-particle trajectory lengths,
    # which avalanchers does not record; leave False
    "plotScor": False,
}

# SimInfo flag bit (must match crates/compute_core/src/shaders/utils.wgsl)
SIM_INFO_STOPPED = 1 << 31


def sample_dem(dem, x, y):
    """Bilinear DEM elevation (z) at map coordinates; NaN outside or on NoData."""
    header = dem["header"]
    cs = header["cellsize"]
    raster = dem["rasterData"]
    nrows, ncols = raster.shape
    # cell centers sit at (xmin + (i + 0.5) * cs, ymin + (j + 0.5) * cs)
    col = np.asarray(x, dtype=float) - header["xmin"]
    row = np.asarray(y, dtype=float) - header["ymin"]
    col = col / cs - 0.5
    row = row / cs - 0.5
    ok = (col >= 0) & (col < ncols - 1) & (row >= 0) & (row < nrows - 1)
    col0 = np.clip(np.floor(col).astype(int), 0, ncols - 2)
    row0 = np.clip(np.floor(row).astype(int), 0, nrows - 2)
    tx = np.clip(col - col0, 0.0, 1.0)
    ty = np.clip(row - row0, 0.0, 1.0)
    z = ((1 - ty) * ((1 - tx) * raster[row0, col0] + tx * raster[row0, col0 + 1])
         + ty * ((1 - tx) * raster[row0 + 1, col0] + tx * raster[row0 + 1, col0 + 1]))
    z = np.where(ok, z, np.nan)
    z = np.where(np.isfinite(z), z, np.nan)
    return z


def sample_center_of_mass(sim, dem):
    """One mass-averaged path point from the live particle buffers.

    Mirrors AvaFrame's ``getMassAvgPathFromPart``: every particle contributes
    weighted by its mass. Positions are relative to the DEM origin corner,
    the altitude is the DEM elevation under the particle, and ``u2`` is the
    mass-averaged squared velocity (3D norm, like AvaFrame's com1DFA).
    """
    pos = np.asarray(sim.particles_position_xy, dtype=np.float64)
    vel = np.asarray(sim.particles_velocity, dtype=np.float64)
    mass = np.asarray(sim.particles_mass, dtype=np.float64)

    x_min = dem["header"]["xmin"]
    y_min = dem["header"]["ymin"]
    x_map = pos[:, 0] + x_min
    y_map = pos[:, 1] + y_min
    z = sample_dem(dem, x_map, y_map)

    finite = (np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
              & np.isfinite(mass) & (mass > 0) & np.isfinite(z))
    total_mass = float(mass[finite].sum())
    if total_mass <= 0:
        return None
    return {
        "x": float(np.average(x_map[finite], weights=mass[finite])),
        "y": float(np.average(y_map[finite], weights=mass[finite])),
        "z": float(np.average(z[finite], weights=mass[finite])),
        "u2": float(np.average((vel[finite] ** 2).sum(axis=1), weights=mass[finite])),
    }


def run_simulation(settings, save_steps, t_end=None):
    """Create, run and sample the simulation; returns (sim, samples, sim_name).

    Runs ``save_steps`` steps per batch and samples the mass-averaged center
    of mass after the first step (t = 0) and after every batch, until the
    simulation stops (all particles stopped / max_steps) or ``t_end`` of
    simulated time is reached.
    """
    import avalanchers

    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    dem = build_dem_dict(sim)
    mu = settings["friction_coefficient"]
    sim_name = "%s_%s_mu%s" % (pathlib.Path(settings["dem_path"]).stem,
                               settings["friction_model"], mu)

    print(f"=== Energy line test - {sim_name} ===")
    print(f"sampling every {save_steps} steps, max_steps={settings['max_steps']}")

    samples = []
    sim.prepare()
    sample = sample_center_of_mass(sim, dem)
    if sample is not None:
        sample.update({"t": 0.0, "timestep": 0})
        samples.append(sample)

    info = sim.run_n_steps(save_steps)
    last_timestep = -1
    while True:
        sample = sample_center_of_mass(sim, dem)
        if sample is not None:
            sample.update({"t": info.elapsed_time, "timestep": info.timestep})
            samples.append(sample)
        if (info.flags & SIM_INFO_STOPPED) or sim.state == "Finished":
            break
        if t_end is not None and info.elapsed_time >= t_end:
            break
        if info.timestep == last_timestep:
            print("simulation made no progress - stopping")
            break
        last_timestep = info.timestep
        info = sim.run_n_steps(save_steps)

    print(f"stopped after {info.timestep} steps, t = {info.elapsed_time:.2f} s "
          f"({len(samples)} path samples)")
    return sim, dem, samples, sim_name


def build_dem_dict(sim):
    """Wrap the simulation's DEM in the dict expected by energyLineTest."""
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


def build_profile_mass(samples):
    """Assemble the mass-averaged path dict (s = cumulative COM path length)."""
    x = np.array([s["x"] for s in samples])
    y = np.array([s["y"] for s in samples])
    z = np.array([s["z"] for s in samples])
    u2 = np.array([s["u2"] for s in samples])
    t = np.array([s["t"] for s in samples])
    s = np.zeros_like(x)
    s[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
    return {"x": x, "y": y, "z": z, "u2": u2, "s": s, "t": t}


def main():
    parser = argparse.ArgumentParser(description="avalanchers energy line test")
    parser.add_argument("--dem", default=None, help="DEM path (PNG/GeoTIFF/.asc)")
    parser.add_argument("--release", default=None, help="release thickness raster path")
    parser.add_argument("--mu", type=float, default=None,
                        help="Coulomb friction coefficient")
    parser.add_argument("--sim-model", default=None,
                        help="terrain-following | curvilinear | mpmdac")
    parser.add_argument("--friction-model", default=None,
                        help="coulomb | voellmy | voellmy-min-shear | samosat")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None,
                        help="sample the center of mass path every N steps")
    parser.add_argument("--t-end", type=float, default=None,
                        help="stop sampling after this much simulated time (s)")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--gpu", default=None, help="GPU name substring to select")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    args = parser.parse_args()

    # headless backend unless an interactive window was requested
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    settings = dict(SETTINGS)
    if args.dem is not None:
        settings["dem_path"] = args.dem
    if args.release is not None:
        settings["release_areas_path"] = args.release
    if args.mu is not None:
        settings["friction_coefficient"] = args.mu
    if args.sim_model is not None:
        settings["sim_model"] = args.sim_model
    if args.friction_model is not None:
        settings["friction_model"] = args.friction_model
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    save_steps = args.save_steps if args.save_steps is not None else SAVE_STEPS

    import energy_line_test

    sim, dem, samples, sim_name = run_simulation(settings, save_steps, t_end=args.t_end)
    if len(samples) < 2:
        raise RuntimeError("fewer than 2 center of mass samples - the avalanche "
                           "stopped immediately, check the release raster/settings")
    ava_profile_mass = build_profile_mass(samples)

    pfv = np.asarray(sim.peak_velocity, dtype=float)
    result, save_path = energy_line_test.generate_energy_plot(
        ava_profile_mass, dem, {"pfv": pfv}, ENERGY_LINE_TEST_CFG,
        g=9.81, mu=settings["friction_coefficient"], sim_name=sim_name,
        out_dir=args.out_dir, show=args.show)

    print()
    print("Runout s      : %.2f m" % result["sEnd"])
    print("Runout z      : %.2f m" % result["zEnd"])
    print("Runout angle  : %.4f deg (alpha = %.4f deg, diff %.4e)"
          % (result["runoutAngle"], np.rad2deg(np.arctan(settings["friction_coefficient"])),
             result["runOutAngleError"]))
    print("Runout s diff : %.4e m" % result["runOutSError"])
    print("Runout z diff : %.4e m" % result["runOutZError"])
    print("Velocity height rmse : %.4e m" % result["rmseVelocityElevation"])
    return result, save_path


if __name__ == "__main__":
    main()
