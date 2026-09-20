"""Run the rotation test with your own avalanchers simulation.

Adapted from AvaFrame's ``runScripts/runRotationTest.py`` +
``ana1Tests/rotationTest.py``: analyze the effect of the GRID ORIENTATION
on the simulation results. Instead of com1DFA output files (and pre-rotated
input shapefiles), the base DEM and release raster are read once, rotated
in memory by every angle in ``ROTATIONS`` (counterclockwise, about the
domain center, bounds kept identical) and handed to fresh simulations with
``set_dem_with_bounds`` / ``set_release_areas`` - nothing is written before
the analysis.

Every rotation runs to completion with the same live center-of-mass
sampling as the energy line test, gets the full energy line analysis and
figure, and its peak fields (peak flow velocity / peak flow thickness) are
rotated BACK into the reference frame. Comparison plots in the reference
frame (see ``rotation_test.py``): runout metrics vs rotation angle,
max/mean flow thickness and velocity along cross sections perpendicular to
the reference center-of-mass path (lightweight AIMEC), and the TP/FP/FN
area maps against the reference run.

Usage:
    python run_rotation_test.py                     # run with the SETTINGS below
    python run_rotation_test.py --angles 0 45       # choose the rotations
    python run_rotation_test.py --dem data/avaframe/avaBowl.png \\
        --release data/avaframe/avaBowlreleaseTexture.png --show
"""

import argparse
import math
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "verification" / "outputs" / "rotation_test"

# base case: the DEM and release raster that get rotated
BASE_DEM = REPO_ROOT / "data" / "avaframe" / "avaTripleBowl.png"
BASE_RELEASE = REPO_ROOT / "data" / "avaframe" / "avaTripleBowlreleaseTexture.png"

# grid alignment angles in degrees (counterclockwise); the FIRST angle is
# the reference all rotations are compared against
ROTATIONS = [0.0, 30.0, 60.0]

# ---------------------------------------------------------------------------
# Simulation settings - edit here or override on the command line.
# Coulomb friction keeps the per-rotation energy line analysis meaningful.
# NOTE: no dem_path/release_areas_path here - the rotated rasters are set
# in memory per rotation (see module docstring).
# ---------------------------------------------------------------------------
SETTINGS = {
    "sim_model": "curvilinear",
    "friction_model": "coulomb",
    "friction_coefficient": 0.3,
    "density": 200.0,
    "released_particles_per_cell": 4,
    "max_steps": 6000,
    "cfl": 0.1,
    "velocity_threshold": 0.1,
    "enable_particle_interaction": True,
    "enable_particle_relaxation": True,
}

SAVE_STEPS = 25

# rotation test configuration
ROTATION_TEST_CFG = {
    "g": 9.81,
    # energy line test configuration per rotation
    "nCellsExtrapolation": 4,
    "shiftedAlphaLine": False,
    "plotScor": False,
    # cross sections perpendicular to the reference center-of-mass path
    "resample_distance_cells": 3.0,   # path resampling [cells]
    "cross_section_step_cells": 1.0,  # sampling step across the path [cells]
    "cross_section_half_width_cells": 40.0,  # half width of the sections [cells]
    "peak_flow_thickness_threshold": 0.1,
}

# SimInfo flag bit (must match crates/compute_core/src/shaders/utils.wgsl)
SIM_INFO_STOPPED = 1 << 31


def load_base_rasters():
    """Read the unrotated DEM + release raster through a helper simulation."""
    import avalanchers
    import run_energy_line_test as relt

    settings = dict(SETTINGS)
    settings["dem_path"] = str(BASE_DEM)
    settings["release_areas_path"] = str(BASE_RELEASE)
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.prepare()
    dem = relt.build_dem_dict(sim)
    release = np.asarray(sim.release_areas, dtype=float)
    return dem, release


def run_rotation(sim, dem, settings, save_steps):
    """Run one simulation with energy-line-style live sampling.

    Returns (info, profile) where `profile` is the mass-averaged
    center-of-mass path dict (x, y, z, u2, s, t).
    """
    import run_energy_line_test as relt

    samples = []
    sim.prepare()
    sample = relt.sample_center_of_mass(sim, dem)
    if sample is not None:
        sample.update({"t": 0.0, "timestep": 0})
        samples.append(sample)

    info = sim.run_n_steps(save_steps)
    last_timestep = -1
    while True:
        sample = relt.sample_center_of_mass(sim, dem)
        if sample is not None:
            sample.update({"t": info.elapsed_time, "timestep": info.timestep})
            samples.append(sample)
        if (info.flags & relt.SIM_INFO_STOPPED) or sim.state == "Finished":
            break
        if info.timestep == last_timestep:
            break
        last_timestep = info.timestep
        info = sim.run_n_steps(save_steps)
    return info, relt.build_profile_mass(samples)


def finish_simulation(sim, save_steps):
    """Run to the Finished state so the peak field buffers can be read.

    Returns the final SimInfo (None if the sim was already finished)."""
    last_timestep = -1
    info = None
    while sim.state != "Finished":
        info = sim.run_n_steps(save_steps)
        if info.timestep == last_timestep:
            break
        last_timestep = info.timestep
    return info


def main():
    parser = argparse.ArgumentParser(description="avalanchers rotation test")
    parser.add_argument("--angles", type=float, nargs="+", default=None,
                        help="grid rotation angles in degrees; the first is "
                             "the reference (default: %(default)s)")
    parser.add_argument("--dem", default=str(BASE_DEM), help="base DEM path")
    parser.add_argument("--release", default=str(BASE_RELEASE),
                        help="base release thickness raster path")
    parser.add_argument("--mu", type=float, default=None,
                        help="Coulomb friction coefficient")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--gpu", default=None, help="GPU name substring to select")
    parser.add_argument("--show", action="store_true", help="open the plot windows")
    args = parser.parse_args()

    # headless backend unless an interactive window was requested
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    settings = dict(SETTINGS)
    if args.mu is not None:
        settings["friction_coefficient"] = args.mu
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    save_steps = args.save_steps if args.save_steps is not None else SAVE_STEPS
    rotations = args.angles if args.angles is not None else list(ROTATIONS)
    mu = settings["friction_coefficient"]

    import avalanchers
    import energy_line_test
    import rotation_test

    dem, release = load_base_rasters()
    header = dem["header"]
    cs = header["cellsize"]
    h0 = float(np.median(release[release > 0]))
    out_dir = pathlib.Path(args.out_dir)
    pics = out_dir / "pics"
    sim_name = "%s_coulomb_mu%g" % (pathlib.Path(args.dem).stem, mu)

    print(f"=== Rotation test - {sim_name} ===")
    print(f"base DEM: {args.dem}")
    print(f"rotations {rotations} deg (first = reference), "
          f"mu={mu}, h0={h0:.2f} m, cell size {cs} m")

    # half width / resampling of the cross sections along the reference path
    half_width = min(ROTATION_TEST_CFG["cross_section_half_width_cells"] * cs,
                     min(header["nrows"], header["ncols"]) * cs / 2.0)
    ds = ROTATION_TEST_CFG["resample_distance_cells"] * cs
    step = ROTATION_TEST_CFG["cross_section_step_cells"] * cs

    results = []
    fields = {}       # per angle: native-frame peak fields {"pfv", "pft"}
    back = {}         # per angle: peak fields rotated into the reference frame
    ref_path = None
    for angle in rotations:
        print(f"\n--- rotation {angle:g} deg ---")
        dem_rot = dem["rasterData"] if angle == 0 else \
            rotation_test.rotate_raster(dem["rasterData"], angle, cval=np.nan)
        rel_rot = release if angle == 0 else \
            rotation_test.clean_rotated_release(
                rotation_test.rotate_raster(release, angle), h0)

        sim = avalanchers.PySimulation.new()
        sim.create({k: v for k, v in settings.items()})
        sim.set_dem_with_bounds(
            np.ascontiguousarray(dem_rot.astype(np.float32)),
            cs, header["xmin"], header["xmax"], header["ymin"], header["ymax"], 1.0)
        sim.set_release_areas(np.ascontiguousarray(rel_rot.astype(np.float32)))
        dem_dict = {
            "rasterData": np.asarray(sim.dem, dtype=float),
            "header": dict(header),
        }
        info, profile = run_rotation(sim, dem_dict, settings, save_steps)
        final_info = finish_simulation(sim, save_steps)
        if final_info is not None:
            info = final_info
        print(f"finished at step {info.timestep}, t = {info.elapsed_time:.2f} s "
              f"({len(profile['s'])} path samples)")

        # energy line analysis + figure in the sim's own frame
        result, _plot = energy_line_test.generate_energy_plot(
            profile, dem_dict, {"pfv": np.asarray(sim.peak_velocity, dtype=float)},
            ROTATION_TEST_CFG, g=ROTATION_TEST_CFG["g"], mu=mu,
            sim_name="rot%g_%s" % (angle, sim_name), out_dir=pics, show=args.show)

        fields[angle] = {
            "pfv": np.asarray(sim.peak_velocity, dtype=float),
            "pft": np.asarray(sim.peak_flow_thickness, dtype=float),
        }
        # rotate the peak fields back into the reference frame
        back[angle] = {
            "pfv": fields[angle]["pfv"] if angle == 0 else
            rotation_test.rotate_raster(fields[angle]["pfv"], -angle, cval=np.nan),
            "pft": fields[angle]["pft"] if angle == 0 else
            rotation_test.rotate_raster(fields[angle]["pft"], -angle, cval=np.nan),
        }
        results.append({"angle": angle, "simName": "rot%g" % angle,
                        **result, "dice": None})
        if angle == rotations[0]:
            ref_path = (profile["x"], profile["y"])

    # cross sections perpendicular to the reference center-of-mass path
    sections = rotation_test.make_cross_sections(ref_path[0], ref_path[1],
                                                 step, half_width, ds)
    stats_per_angle = {}
    masks = {}
    ref_mask = None
    for angle in rotations:
        stats_per_angle[angle] = {
            "pfv": rotation_test.cross_section_stats(back[angle]["pfv"], header, sections),
            "pft": rotation_test.cross_section_stats(back[angle]["pft"], header, sections),
        }
        masks[angle] = back[angle]["pft"] > ROTATION_TEST_CFG["peak_flow_thickness_threshold"]
        if angle == rotations[0]:
            ref_mask = masks[angle]
    for row in results:
        if row["angle"] != rotations[0]:
            mask = masks[row["angle"]]
            inter = float((ref_mask & mask).sum())
            row["dice"] = 2.0 * inter / (ref_mask.sum() + mask.sum())

    rotation_test.print_results_table(results, rotations[0])
    rotation_test.plot_runout_comparison(results, rotations[0],
                                         ROTATION_TEST_CFG, sim_name, pics)
    rotation_test.plot_sl_comparison(sections, stats_per_angle,
                                     ROTATION_TEST_CFG, sim_name, pics)
    rotation_test.plot_areas_comparison(header, ref_mask,
                                        {a: masks[a] for a in rotations
                                         if a != rotations[0]},
                                        ROTATION_TEST_CFG, sim_name, pics)
    return results


if __name__ == "__main__":
    main()
