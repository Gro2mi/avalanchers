"""Verify curvature-dependent normal load on the energy-line DEM and release.

Run ``python verification\\run_curvature_test.py`` or select ``curvature`` in
run_all_tests.py. Both curvature-on and curvature-off cases use the original
finite release and interaction settings. The point-slider reference omits
internal pressure; finite-release deviations are measured, not assumed zero.
"""

import argparse
import pathlib

import numpy as np

import curvature_test
import run_energy_line_test as energy_line

SETTINGS = dict(energy_line.SETTINGS, enable_curvature=True)
# models whose curvature load is compared against the analytical solution
ANALYTIC_MODELS = ("curvilinear", "curvi", "c", "terrain-following", "terrain", "t",
                   "3d-depth-averaged", "3d")
OUTPUT_DIR = energy_line.REPO_ROOT / "verification" / "outputs" / "curvature_test"


def run_simulation(settings, save_steps=25):
    if settings["sim_model"] not in ANALYTIC_MODELS or settings["friction_model"] != "coulomb":
        raise ValueError(
            "Curvature verification requires the curvilinear or terrain-following model and Coulomb friction."
        )
    mu = settings["friction_coefficient"]
    cases = {}
    for enabled in (False, True):
        sim, dem, samples, name = energy_line.run_simulation(
            dict(settings, enable_curvature=enabled), save_steps
        )
        profile = energy_line.build_profile_mass(samples)
        bed, residual = curvature_test.fit_energy_line_bed(dem)
        result = curvature_test.compare_profile(bed, profile, mu, enabled)
        final_velocity = np.asarray(sim.particles_velocity, dtype=float)
        masses = np.asarray(sim.particles_mass, dtype=float)
        result["final_rms_speed"] = float(np.sqrt(np.average(
            np.sum(final_velocity**2, axis=1), weights=masses)))
        cases[enabled] = {"profile": profile, "result": result, "bed": bed, "fit_residual": residual}
        print(f"curvature={enabled}: analytical runout {result['horizontal_runout']:.3f} m "
              f"(surface {result['surface_runout']:.3f} m), simulated {result['simulated_runout']:.3f} m, "
              f"error {result['runout_error']:+.3f} m, energy RMSE {result['energy_rmse']:.3f} m, "
              f"final RMS speed {result['final_rms_speed']:.4f} m/s, DEM fit residual {residual:.6f} m")
    enabled = cases[True]
    disabled = cases[False]
    wrong_reference = curvature_test.compare_profile(enabled["bed"], enabled["profile"], mu, False)
    result = {
        "cases": cases,
        "runout_reduction": disabled["result"]["simulated_runout"]-enabled["result"]["simulated_runout"],
        "analytical_reduction": disabled["result"]["horizontal_runout"]-enabled["result"]["horizontal_runout"],
        "uncorrected_energy_rmse": wrong_reference["energy_rmse"],
    }
    print(f"Curvature runout reduction: simulated {result['runout_reduction']:.3f} m, "
          f"analytical {result['analytical_reduction']:.3f} m")
    print(f"Curvature-on energy RMSE against the incorrect alpha line: {wrong_reference['energy_rmse']:.3f} m")
    return result


def plot_results(result, out_dir, mu):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for enabled, case in result["cases"].items():
        profile, values = case["profile"], case["result"]
        distance = profile["x"]-profile["x"][0]
        label = "curvature on" if enabled else "curvature off"
        axes[0].plot(distance, values["simulated_energy"], ".", label=f"{label}: simulation")
        axes[0].plot(distance, values["reference_energy"], "-", label=f"{label}: analytical")
        axes[0].axvline(values["horizontal_runout"], linestyle=":", alpha=0.6)
        axes[1].plot(distance, values["simulated_energy"]-values["reference_energy"], ".-", label=label)
    start_z = float(result["cases"][True]["bed"].elevation(result["cases"][True]["profile"]["x"][0]))
    axes[0].plot(distance, start_z-mu*distance, "--", label="uncorrected alpha line")
    axes[0].set_ylabel("Energy height B + v^2/(2g) [m]")
    axes[1].set_ylabel("Energy-height residual [m]")
    axes[1].set_xlabel("Horizontal distance from release COM [m]")
    for ax in axes:
        ax.grid()
        ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "CurvatureTest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {path}")
    return path


def plot_thesis(result, out_dir, mu, g=9.81):
    """Thesis figure (see thesis_style): velocity height u^2/2g along the
    horizontal distance for curvature off/on - simulation markers, analytical
    point slider lines, analytical stopping points dotted (a) - and the
    energy-height residual simulation - analytical (b)."""
    import matplotlib.pyplot as plt
    import thesis_style as ts

    colors = {False: ts.BLUE, True: ts.VERMILION}
    labels = {False: "curvature off", True: "curvature on"}
    with ts.style():
        fig, axes = plt.subplots(2, 1, figsize=ts.size(1.0, 0.66), sharex=True,
                                 height_ratios=(2.0, 1.0), layout="constrained")
        for enabled in (False, True):
            case = result["cases"][enabled]
            profile, values, bed = case["profile"], case["result"], case["bed"]
            distance = profile["x"] - profile["x"][0]
            start_x = float(profile["x"][0])
            # analytical velocity height on a fine axis up to the stopping point
            x_fine = np.linspace(start_x, values["stop_x"], 400)
            w = np.maximum(curvature_test.kinetic_height(bed, start_x, x_fine, mu, enabled), 0.0)
            color = colors[enabled]
            axes[0].plot(x_fine - start_x, w, color=color, linestyle="--")
            axes[0].plot(distance, profile["u2"] / (2.0 * g), linestyle="none", marker="o",
                         markersize=3, markerfacecolor="none", color=color)
            axes[0].axvline(values["horizontal_runout"], color=color, linestyle=":",
                            linewidth=0.9)
            axes[1].plot(distance, values["simulated_energy"] - values["reference_energy"],
                         color=color, marker="o", markersize=2.5)
        axes[0].set_ylabel(r"$\Vert\mathbf{u}\Vert^2/2g$ (m)")
        axes[0].set_ylim(bottom=0.0)
        axes[1].axhline(0.0, color="grey", linewidth=0.5)
        axes[1].set_ylabel(r"$\Delta E$ (m)")
        axes[1].set_xlabel(r"horizontal distance from the release center of mass (m)")
        ts.panel_label(axes[0], "a", loc=(0.015, 0.95))
        ts.panel_label(axes[1], "b", loc=(0.015, 0.92))
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=colors[e], label=labels[e]) for e in (False, True)]
        handles += [Line2D([], [], color=ts.BLACK, linestyle="none", marker="o",
                           markersize=3, markerfacecolor="none", label="simulation"),
                    Line2D([], [], color=ts.BLACK, linestyle="--", label="analytical slider"),
                    Line2D([], [], color=ts.BLACK, linestyle=":", label="analytical stop")]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        return ts.save(fig, pathlib.Path(out_dir) / "thesis", "curvature_velocity_height")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    from cli_args import add_particles_per_cell_argument
    add_particles_per_cell_argument(parser)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    settings = dict(SETTINGS)
    if args.mu is not None:
        settings["friction_coefficient"] = args.mu
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.particles_per_cell is not None:
        settings["released_particles_per_cell"] = args.particles_per_cell
    if args.save_steps < 1:
        parser.error("--save-steps must be positive")
    result = run_simulation(settings, args.save_steps)
    plot_results(result, pathlib.Path(args.out_dir) / "pics", settings["friction_coefficient"])
    plot_thesis(result, pathlib.Path(args.out_dir) / "pics", settings["friction_coefficient"])
    return result


if __name__ == "__main__":
    main()
