"""Run avalanchers verification tests and write a markdown report.

Runs the energy line, curvature, dam break, rotation, similarity solution and
(3D model only) double bump tests
(see the sibling ``run_<test>.py`` scripts) IN-PROCESS with a switchable
simulation model, collects each test's console tables, summary plots and
animated GIFs, and writes everything into one markdown document:

    verification/outputs/all_tests/<model>_<timestamp>/report.md

Model switching: ``--sim-model`` accepts any avalanchers model string
(``curvilinear`` | ``terrain-following`` | ``dampm-global-cartesian`` | ``mpmdac``
| ``sliding-block``) and is injected into every test's settings before it
runs. The analytic reference solutions (alpha line, Faccanoni-Mangeney,
Hutter 1993) apply to the depth-averaged models (curvilinear,
terrain-following, dampm-global-cartesian) with Coulomb friction - for other
models the tests still run and produce plots/tables, but treat them as
model-comparison runs, not analytic validation. The curvature-specific
benchmark is explicitly skipped for other models or friction laws.
``--friction-model`` / ``--mu`` optionally override the friction (defaults:
Coulomb with each test's documented angle).

DAMPM global model: it carries the full Cartesian velocity, so the
similarity solution test compares the momentum in x, y AND z (h*u_z from
the vertical grid momentum) and reports the 3D vector error plus a separate
h*u_z error. It is also the only model in which particles detach from the bed
at sharp convex breaks: the double bump test (skipped for the other models)
checks take-off at the first bump and landing on the second.

Thesis output: every test writes thesis figures (vector PDF + 300 dpi PNG,
15 cm text width, Computer Modern; see ``thesis_style.py``) to
``<test>/pics/thesis`` - the similarity solution and dam break tests also an
error CSV. The report lists them first, with captions, ahead of the
AvaFrame-style quick-look figures, and the run directory gets booktabs
tables: ``summary.tex``, ``similarity_errors.tex``, ``dam_break_errors.tex``,
``curvature.tex``, ``rotation.tex``, ``double_bump.tex`` and
``double_bump_reference.tex`` (``\\usepackage{booktabs,tabularx}``).
All thesis PNG, PDF and TEX files are also copied into one flat folder,
``<run>/thesis_all``, for easy copying into the thesis.

Each test runs in its own subfolder and a failure of one test does not stop
the others - failures are reported in the markdown. Curvilinear/Coulomb runs
also apply numerical regression guardrails; other models are labeled comparison
runs rather than numerical passes.

Usage:
    python run_all_tests.py                          # all tests, dampm-global-cartesian, 64 particles per cell
    python run_all_tests.py --sim-model curvilinear  # switch the model (double bump skipped)
    python run_all_tests.py --tests energy_line dam_break --quick
    python run_all_tests.py --particles-per-cell 8          # particles per release cell
"""

import argparse
import contextlib
import io
import math
import pathlib
import re
import shutil
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # the runner is always headless

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ALL_TESTS_DIR = REPO_ROOT / "verification" / "outputs" / "all_tests"

DEFAULT_SIM_MODEL = "dampm-global-cartesian"
DEFAULT_PARTICLES_PER_CELL = 64

TEST_NAMES = ["energy_line", "curvature", "similarity_solution", "dam_break", "rotation",
              "double_bump"]
# models checked against the analytical references (with Coulomb friction)
ANALYTIC_MODELS = ("dampm-curvilinear", "curvilinear", "curvi", "c",
                   "dampm-terrain-following", "terrain-following", "terrain", "t",
                   "dampm-global-cartesian", "global", "3d-depth-averaged", "3d")

TEST_TITLES = {
    "energy_line": "Energy Line",
    "curvature": "Curvature",
    "similarity_solution": "Similarity Solution",
    "dam_break": "Dam Break",
    "rotation": "Rotation",
    "double_bump": "Double Bump",
}

THREE_D_MODELS = ("dampm-global-cartesian", "global", "3d-depth-averaged", "3d")

# what each test verifies (report text, reusable in the thesis methods)
TEST_DESCRIPTIONS = {
    "energy_line": (
        "AvaFrame energy line test: with pure Coulomb friction the mass-averaged "
        "center of mass must stop where the energy line of slope mu meets its path "
        "(runout angle alpha = arctan mu), and its velocity height u^2/2g must follow "
        "the vertical distance between that energy line and the path."),
    "curvature": (
        "Curvature-dependent normal load on the energy-line DEM and release: runs with "
        "the curvature load on and off are compared to a point-slider reference with "
        "the curvature load (no internal pressure); the curvature load must shorten "
        "the runout on the concave bed and match the corrected energy line better "
        "than the uncorrected alpha line."),
    "similarity_solution": (
        "Parabolic heap spreading on a 35 deg plane with 25 deg Coulomb bed friction, "
        "compared to the similarity solution of Hutter, Siegel & Savage (1993). Flow "
        "thickness h and momentum h*u are compared on the grid at every second "
        "(relative L2 and Lmax norms, AvaFrame simiSol)."),
    "dam_break": (
        "Dry-bed dam break on an inclined plane with Coulomb friction, compared to the "
        "analytic solution of Faccanoni & Mangeney (2012, test 2, case 1.2). The 5 m "
        "DEM is refined 4x (1.25 m cells) so that the initial step is resolved."),
    "rotation": (
        "AvaFrame rotation test: DEM and release are rotated by several angles, every "
        "run gets the energy line analysis and its peak fields are rotated back; "
        "runout and affected area (IoU) must not depend on the grid orientation."),
    "double_bump": (
        "Detachment and landing (3D model only): snow released on a 32 deg incline runs "
        "over two 20 m sech-shaped bumps (after the double-bump example of Matter). Where "
        "the bed-normal load g n_z + kappa |u|^2 falls below -0.2 g (lift-off threshold) "
        "at the first crest, the particles fly ballistically (no air drag); they must land "
        "on the second bump, flow over it and all be back on the bed when the run ends. "
        "Coulomb friction mu = 0.4, curvature load and lift-off on. Reference: a Coulomb point slider released at rest at the initial center "
        "of mass, with the same take-off condition and threshold, a ballistic flight and the same "
        "inelastic landing with friction impulse mu u_n (closed-form velocity height "
        "with the integrating factor exp(2 mu atan B'), evaluated by quadrature)."),
}

# plot files embedded in the report, in order, per test: (glob relative to
# <test>/pics, caption)
REPORT_PLOTS = {
    "energy_line": [
        ("thesis/energyline_profile.png",
         "Center-of-mass altitude along the path with the energy heights z + u^2/2g, "
         "the simulated energy line and the alpha line (a; the inset zooms on the "
         "runout point: dot = simulation, square = alpha line), and the velocity height "
         "against the distance between alpha line and path (b)."),
        ("thesis/energyline_map.png",
         "Peak flow velocity on the DEM with the center-of-mass path."),
        ("*EnergyTest*.png", "AvaFrame-style energy line summary."),
    ],
    "curvature": [
        ("thesis/curvature_velocity_height.png",
         "Velocity height u^2/2g of the center of mass with the curvature load off and "
         "on (a; markers = simulation, dashed = analytical point slider, dotted = "
         "analytical stop) and the energy-height residual simulation - analytical (b)."),
        ("*CurvatureTest*.png", "Energy height and residual (quick-look figure)."),
    ],
    "similarity_solution": [
        ("thesis/simisol_profiles_*.png",
         "Flow thickness h (a, b) and momentum components (c, d) along the flow "
         "(y = 0) and across it through the analytic heap center; simulation solid, "
         "analytical dashed. h*u_z is shown for the 3D model only."),
        ("thesis/simisol_momentum_maps_*.png",
         "Momentum components h*u_x, h*u_y (and h*u_z for the 3D model): "
         "simulation, analytical solution and difference."),
        ("thesis/simisol_centerline_evolution.png",
         "Center-line profiles (y = 0) of h (a) and h|u| (b) at several times; "
         "simulation solid, analytical dashed."),
        ("thesis/simisol_contours_evolution.png",
         "Flow thickness contours at 10 % (thick) and 60 % (thin) of the analytic "
         "peak thickness at several times; + marks the analytic heap center."),
        ("thesis/simisol_errors.png",
         "Relative L2 and Lmax errors of flow thickness (a) and momentum (b) over "
         "time; for the 3D model the momentum error is the 3D vector norm, markers "
         "show the vertical component h*u_z."),
        ("*SimiSolTest.png", "AvaFrame-style summary (simiSol) at the highlighted time."),
        ("*_profiles.gif", "Animated profile comparison (AvaFrame twin-axis panels)."),
        ("*_contours.gif", "Animated flow thickness contour comparison."),
    ],
    "dam_break": [
        ("thesis/dambreak_profiles.png",
         "Flow thickness (a), speed (b) and momentum (c) along the slope through the dam "
         "center at several times; simulation solid, analytical dashed. The analytical "
         "curves start at the material mid-point (ticks in a), the upstream end of the "
         "error domain; upstream the semi-infinite-dam solution does not apply."),
        ("thesis/dambreak_map_*.png",
         "Flow thickness at the highlighted time in the slope frame of the profiles "
         "(along-slope distance s from the dam front, cross-slope distance from the dam "
         "axis) with the error domain, the profile line and the analytical front."),
        ("thesis/dambreak_errors.png",
         "Relative L2 and Lmax errors of flow thickness (a) and momentum |h u| (b) "
         "within the error domain over time."),
        ("*DamBreakTest.png", "AvaFrame-style dam break summary at the highlighted time."),
        ("*_compare.gif", "Animated dam break comparison."),
    ],
    "rotation": [
        ("thesis/rotation_paths.png",
         "Center-of-mass paths and affected-area outlines (peak flow thickness above "
         "the threshold) of all grid rotations, rotated back into the reference frame."),
        ("thesis/rotation_areas.png",
         "Affected-area agreement of each rotated run with the reference run."),
        ("thesis/rotation_along_path.png",
         "Maximum (solid) and mean (dashed) peak flow thickness (a) and peak speed (b) "
         "on cross sections along the reference center-of-mass path."),
        ("*_runout_comparison.png", "Runout per grid rotation (quick-look figure)."),
        ("*_sl_comparison.png", "Max/mean values along the path (quick-look figure)."),
        ("*_areas_comparison.png", "Affected area vs the reference (quick-look figure)."),
    ],
    "double_bump": [
        ("thesis/doublebump_snapshots.png",
         "Side view of the particles while most of them land on the second bump, height "
         "above the inclined plane over x: particles on the bed (blue, drawn at a random "
         "height within the flow depth of their cell) and flying particles (orange), with "
         "the path of the Coulomb point slider (green), its take-offs (yellow) and landings "
         "(light blue)."),
        ("thesis/doublebump_flight.png",
         "Take-off and landing positions along the slope with the bed profile; shaded the "
         "take-off window of the first bump and the landing window of the second (a). "
         "Share of flying particles over time with the time of the side view (b)."),
    ],
}

# acceptance of the double bump test: share of the first take-offs at the
# first bump, share of the first landings on the second bump, distance of
# the median first take-off and landing from the point slider (two 5 m
# cells) and of the final center of mass from its stop (share of the travel)
DOUBLE_BUMP_MIN_TAKEOFF = 0.9
DOUBLE_BUMP_MIN_LANDING = 0.5
DOUBLE_BUMP_MAX_EVENT_ERROR = 10.0
DOUBLE_BUMP_MAX_STOP_ERROR = 0.02

# times of the similarity solution error table
SIMILARITY_TABLE_TIMES = (5.0, 10.0, 15.0, 20.0)

# ANSI escape codes (tracing output)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TRACING_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def _capture(module, args_list):
    """Run module.main() with patched argv; capture stdout.

    Returns (return_value, cleaned_output, error_string_or_None).
    """
    buf = io.StringIO()
    old_argv = sys.argv
    sys.argv = [pathlib.Path(module.__file__).name] + args_list
    error = None
    ret = None
    try:
        with contextlib.redirect_stdout(buf):
            ret = module.main()
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.argv = old_argv
    # strip ANSI + tracing noise, drop empty lines
    lines = []
    for line in buf.getvalue().splitlines():
        line = _ANSI_RE.sub("", line)
        if _TRACING_RE.match(line):
            continue
        if line.strip():
            lines.append(line)
    return ret, "\n".join(lines), error


def _patch_common(settings, args):
    """Inject the runner's model/friction choices into a test's SETTINGS."""
    settings["sim_model"] = args.sim_model
    if args.friction_model:
        settings["friction_model"] = args.friction_model
    if args.mu:
        settings["friction_coefficient"] = args.mu
    if args.weak_pressure is not None:
        settings["enable_weak_pressure"] = args.weak_pressure
    if args.particles_per_cell is not None:
        settings["released_particles_per_cell"] = args.particles_per_cell
    if args.integer_atomics:
        settings["integer_atomics"] = True


def run_energy_line_test(test_dir, args):
    import run_energy_line_test as mod

    _patch_common(mod.SETTINGS, args)
    extra = []
    if args.quick:
        extra += ["--max-steps", "3000"]
    return _capture(mod, ["--out-dir", str(test_dir)] + extra)


def run_curvature_test(test_dir, args):
    if args.sim_model not in ANALYTIC_MODELS or args.friction_model not in (None, "coulomb"):
        return None, "Curvature benchmark skipped: requires curvilinear or terrain-following with Coulomb.", None
    import run_curvature_test as mod

    _patch_common(mod.SETTINGS, args)
    return _capture(mod, ["--out-dir", str(test_dir)])


def run_dam_break_test(test_dir, args):
    import run_dam_break_test as mod

    _patch_common(mod.SETTINGS, args)
    if args.mu:
        mod.DAM_BREAK_CFG["delta"] = math.degrees(math.atan(args.mu))
    extra = []
    if not args.no_sequences:
        extra += ["--plot-sequence"]
    if args.quick:
        extra += ["--t-end", "10"]
    return _capture(mod, ["--out-dir", str(test_dir)] + extra)


def run_rotation_test(test_dir, args):
    import run_rotation_test as mod

    _patch_common(mod.SETTINGS, args)
    extra = []
    if args.quick:
        extra += ["--angles", "0", "45"]
    return _capture(mod, ["--out-dir", str(test_dir)] + extra)


def run_similarity_solution_test(test_dir, args):
    import run_similarity_solution_test as mod

    _patch_common(mod.SETTINGS, args)
    if args.mu:
        mod.SIMI_SOL_CFG["delta"] = math.degrees(math.atan(args.mu))
    extra = []
    if not args.no_sequences:
        extra += ["--plot-sequence"]
    if args.quick:
        extra += ["--t-end", "10"]
    return _capture(mod, ["--out-dir", str(test_dir)] + extra)


def run_double_bump_test(test_dir, args):
    if not _is_three_d(args):
        return None, "Double bump test skipped: detachment exists only in the DAMPM global model.", None
    import run_double_bump_test as mod

    _patch_common(mod.SETTINGS, args)
    return _capture(mod, ["--out-dir", str(test_dir)])


RUNNERS = {
    "energy_line": run_energy_line_test,
    "curvature": run_curvature_test,
    "dam_break": run_dam_break_test,
    "rotation": run_rotation_test,
    "similarity_solution": run_similarity_solution_test,
    "double_bump": run_double_bump_test,
}


def _double_bump_assessment(ret):
    if ret is None:
        return "skipped", ["Detachment exists only in the DAMPM global model."]
    failures = []
    if ret["max_flying"] == 0:
        failures.append("No particle detached.")
    if ret["takeoff_at_bump_1"] < DOUBLE_BUMP_MIN_TAKEOFF:
        failures.append(f"First take-offs at the first bump must be >= "
                        f"{100 * DOUBLE_BUMP_MIN_TAKEOFF:.0f} %.")
    if ret["landing_on_bump_2"] < DOUBLE_BUMP_MIN_LANDING:
        failures.append(f"First landings on the second bump must be >= "
                        f"{100 * DOUBLE_BUMP_MIN_LANDING:.0f} %.")
    if ret["flying_at_end"] > 0:
        failures.append("All particles must have landed when the run ends.")
    reference = ret.get("reference")
    if reference is None:
        return ("failed" if failures else "passed"), failures + [
            "No point-slider comparison: the reference needs Coulomb friction."]
    for key, label in (("takeoff_x", "first take-off"), ("landing_x", "first landing")):
        error = ret[key] - ret[f"reference_{key}"]
        if not np.isfinite(error) or abs(error) > DOUBLE_BUMP_MAX_EVENT_ERROR:
            failures.append(f"Median {label} x must lie within {DOUBLE_BUMP_MAX_EVENT_ERROR:g} m "
                            "of the point slider.")
    travel = reference.x[-1] - reference.x[0]
    stop_error = ret["stop_x"] - ret["reference_stop_x"]
    if not np.isfinite(stop_error) or abs(stop_error) > DOUBLE_BUMP_MAX_STOP_ERROR * travel:
        failures.append(f"Final center of mass must lie within {100 * DOUBLE_BUMP_MAX_STOP_ERROR:g} % "
                        "of the travel from the point slider stop.")
    return ("failed" if failures else "passed"), failures


def _assessment(name, ret, args):
    """Regression guardrails, not a claim of physical model certification."""
    if name == "double_bump":
        return _double_bump_assessment(ret)
    if args.sim_model not in ANALYTIC_MODELS or args.friction_model not in (None, "coulomb"):
        return "comparison", ["Numerical acceptance is only evaluated for curvilinear/terrain-following with Coulomb."]
    failures = []
    if name == "curvature":
        for enabled, case in ret["cases"].items():
            result = case["result"]
            for key, limit in (
                ("runout_error", 0.02 * result["horizontal_runout"]),
                ("energy_rmse", 3.0),
                ("final_rms_speed", 0.1),
            ):
                if not np.isfinite(result[key]) or abs(result[key]) > limit:
                    failures.append(f"Curvature={enabled}: {key} must be finite and within {limit:.3g}.")
        if not np.isfinite(ret["runout_reduction"]) or ret["runout_reduction"] <= 0:
            failures.append("Curvature must reduce runout on this concave bed.")
        corrected = ret["cases"][True]["result"]["energy_rmse"]
        incorrect = ret["uncorrected_energy_rmse"]
        if not np.isfinite(incorrect) or corrected >= 0.5 * incorrect:
            failures.append("Curvature reference must halve the energy error against the uncorrected alpha line.")
    elif name in ("dam_break", "similarity_solution"):
        samples, errors = ret
        times = np.asarray(samples["t"])
        required_time = 10.0 if args.quick else 20.0
        if not len(times) or not np.isfinite(times).all() or times[-1] < required_time:
            failures.append(f"Did not reach the required grid-snapshot time of {required_time:g} s.")
        if len(times) > 1 and not (np.diff(times) > 0).all():
            failures.append("Grid-snapshot times must be strictly increasing.")
        for key, limit in (("h_l2rel", 0.15), ("vh_l2rel", 0.25)):
            values = np.asarray(errors[key])
            if not len(values) or not np.isfinite(values).all() or values[-1] > limit:
                failures.append(f"Final {key} must be finite and <= {limit:g}.")
    elif name == "energy_line":
        result = ret[0]
        for key, limit in (
            ("runOutSError", 0.02 * abs(result["sEnd"])),
            ("rmseVelocityElevation", 3.0),
        ):
            if not np.isfinite(result[key]) or abs(result[key]) > limit:
                failures.append(f"{key} must be finite and within {limit:.3g}.")
    elif name == "rotation":
        if len(ret) < 2:
            failures.append("At least two grid orientations are required.")
        else:
            reference = ret[0]["sEnd"]
            for result in ret:
                if not np.isfinite(result["sEnd"]) or abs(result["sEnd"] - reference) > 0.02 * abs(reference):
                    failures.append(f"Runout at {result['angle']:g} degrees differs by more than 2%.")
                iou = result["iou"]
                if iou is not None and (not np.isfinite(iou) or iou < 0.74):
                    failures.append(f"IoU at {result['angle']:g} degrees must be >= 0.74.")
    return ("failed" if failures else "passed"), failures


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------

def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        # a literal pipe (e.g. |ds|) would split the cell
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(out)


def _details(summary, body):
    return f"<details>\n<summary>{summary}</summary>\n\n```\n{body}\n```\n\n</details>"


def _is_three_d(args):
    return args.sim_model.lower() in THREE_D_MODELS


def _relative_images(test_dir, plots):
    """Markdown image + caption lines for test_dir/pics, in the plot order.

    Each figure gets a numbered caption; thesis figures also link their PDF.
    """
    pics = test_dir / "pics"
    if not pics.is_dir():
        return []
    lines = []
    seen = set()
    for pattern, caption in plots:
        for p in sorted(pics.glob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            rel = p.relative_to(test_dir.parent).as_posix()
            pdf = p.with_suffix(".pdf")
            link = (f" [PDF]({pdf.relative_to(test_dir.parent).as_posix()})"
                    if pdf.is_file() else "")
            lines += [f"![{p.stem}]({rel})", "",
                      f"*{p.name}: {caption}*{link}", ""]
    return lines


def _with_z(name, args):
    """Tests reporting the vertical momentum error separately (3D model)."""
    return name == "similarity_solution" and _is_three_d(args)


def _similarity_rows(ret, args, with_z=None):
    """Error table rows at SIMILARITY_TABLE_TIMES: (t, hL2, hLmax, huL2, huLmax[, huzL2, huzLmax]).

    Used for the similarity solution and the dam break test (same error keys)."""
    samples, errors = ret
    times = np.asarray(samples["t"])
    if not len(times):
        return []
    keys = ["h_l2rel", "h_lmaxrel", "vh_l2rel", "vh_lmaxrel"]
    if (_is_three_d(args) if with_z is None else with_z):
        keys += ["vhz_l2rel", "vhz_lmaxrel"]
    rows = []
    for wanted in SIMILARITY_TABLE_TIMES:
        if wanted > times[-1] + 0.5:
            continue
        i = int(np.argmin(np.abs(times - wanted)))
        rows.append([times[i]] + [float(errors[k][i]) for k in keys])
    return rows


def _similarity_header(args, with_z=None):
    header = ["t [s]", "h L2", "h Lmax", "hu L2", "hu Lmax"]
    if (_is_three_d(args) if with_z is None else with_z):
        header += ["hu_z L2", "hu_z Lmax"]
    return header


def _headline(test, ret, args=None):
    """One-line headline metric per test from its main() return value."""
    try:
        if test == "energy_line":
            result = ret[0]
            return (f"runout s {result['sEnd']:.1f} m, angle {result['runoutAngle']:.2f} deg, "
                    f"ds diff {result['runOutSError']:+.2f} m, "
                    f"velocity-height rmse {result['rmseVelocityElevation']:.2f} m")
        if test == "curvature":
            if ret is None:
                return "skipped: curvature reference requires a depth-averaged model with Coulomb"
            result = ret["cases"][True]["result"]
            return (f"runout sim/analytic {result['simulated_runout']:.1f}/{result['horizontal_runout']:.1f} m, "
                    f"energy RMSE {result['energy_rmse']:.3f} m, "
                    f"runout reduction {ret['runout_reduction']:.1f} m")
        if test in ("dam_break", "similarity_solution"):
            samples, errors = ret
            times = samples["t"]
            i_ref = int(np.argmin(np.abs(times - 5.0)))
            text = (f"rel h L2 {errors['h_l2rel'][i_ref]:.3f} @ t={times[i_ref]:.1f} s, "
                    f"{errors['h_l2rel'][-1]:.3f} @ t={times[-1]:.1f} s; "
                    f"rel hu L2 {errors['vh_l2rel'][-1]:.3f}")
            if test == "similarity_solution" and args is not None and _is_three_d(args):
                text += f" (3D), rel hu_z L2 {errors['vhz_l2rel'][-1]:.3f}"
            return text + f" @ t={times[-1]:.1f} s"
        if test == "rotation":
            results = ret
            ref = results[0]
            max_ds = max(abs(r["sEnd"] - ref["sEnd"]) for r in results)
            ious = [r["iou"] for r in results if r["iou"] is not None]
            iou = f", IoU >= {min(ious):.3f}" if ious else ""
            return (f"runout s {ref['sEnd']:.1f} m at {ref['angle']:g} deg, "
                    f"max |ds| {max_ds:.2f} m over {len(results)} rotations{iou}")
        if test == "double_bump":
            if ret is None:
                return "skipped: detachment requires the DAMPM global model"
            text = (f"{100 * ret['takeoff_fraction']:.0f} % took off, first take-off at bump 1 "
                    f"{100 * ret['takeoff_at_bump_1']:.0f} %, first landing on bump 2 "
                    f"{100 * ret['landing_on_bump_2']:.0f} %, {ret['flying_at_end']} flying at "
                    "the end")
            if ret.get("reference") is not None:
                text += (f"; sim/slider take-off x {ret['takeoff_x']:.0f}/"
                         f"{ret['reference_takeoff_x']:.0f} m, landing x {ret['landing_x']:.0f}/"
                         f"{ret['reference_landing_x']:.0f} m, stop x {ret['stop_x']:.0f}/"
                         f"{ret['reference_stop_x']:.0f} m")
            return text
    except Exception:
        return ""
    return ""


def _test_section(name, test_dir, ret, output, error, args, assessment):
    lines = [f"## {TEST_TITLES[name]}", "", TEST_DESCRIPTIONS[name], ""]
    if error is not None:
        lines += ["**FAILED**", "", "```", error.strip(), "```", ""]
    else:
        lines += [f"status: **{assessment[0]}**", ""]
        lines += [f"- {reason}" for reason in assessment[1]]
        lines.append("")
    headline = _headline(name, ret, args) if error is None else ""
    if headline:
        lines += [headline, ""]

    # rotation test: full result table from the return value
    if name == "rotation" and error is None and ret:
        ref = ret[0]
        rows = []
        for r in ret:
            rows.append([f"{r['angle']:g}", f"{r['sEnd']:.2f}", f"{r['zEnd']:.2f}",
                         f"{r['runoutAngle']:.3f}",
                         f"{r['sEnd'] - ref['sEnd']:+.2f}",
                         f"{r['zEnd'] - ref['zEnd']:+.2f}",
                         f"{r['runoutAngle'] - ref['runoutAngle']:+.3f}",
                         "-" if r["iou"] is None else f"{r['iou']:.3f}"])
        lines += [_md_table(["angle [deg]", "sEnd [m]", "zEnd [m]", "runout angle [deg]",
                             "ds [m]", "dz [m]", "dAngle [deg]", "IoU"], rows), ""]

    # double bump: detachment metrics
    if name == "double_bump" and error is None and ret:
        lines += [_md_table(["metric", "value"], [
            ["particles", ret["particles"]],
            ["particles that took off", f"{100 * ret['takeoff_fraction']:.1f} %"],
            ["first take-off at the first bump", f"{100 * ret['takeoff_at_bump_1']:.1f} %"],
            ["first landing on the second bump", f"{100 * ret['landing_on_bump_2']:.1f} %"],
            ["most particles in flight", f"{ret['max_flying']} at t = {ret['t_max_flying']:.1f} s"],
            ["highest flight above the bed", f"{ret['max_flight_height']:.1f} m"],
            ["flying at the end", ret["flying_at_end"]],
            ["95 % runout x", f"{ret['runout_x']:.0f} m"],
        ]), ""]
        if ret.get("reference") is not None:
            lines += ["Comparison with the Coulomb point slider:", "",
                      _md_table(["event", "simulation x [m]", "point slider x [m]", "difference [m]"], [
                          [label, f"{ret[key]:.1f}", f"{ret['reference_' + key]:.1f}",
                           f"{ret[key] - ret['reference_' + key]:+.1f}"]
                          for key, label in (("takeoff_x", "first take-off (median)"),
                                             ("landing_x", "first landing (median)"),
                                             ("stop_x", "stop (center of mass)"))]), ""]

    # similarity solution / dam break: relative error table at selected times
    if name in ("similarity_solution", "dam_break") and error is None and ret:
        with_z = _with_z(name, args)
        rows = _similarity_rows(ret, args, with_z)
        if rows:
            if name == "dam_break":
                norm = "|h u| within the error domain"
            elif with_z:
                norm = "3D vector norm of (h u_x, h u_y, h u_z)"
            else:
                norm = "in-plane norm of (h u, h v)"
            lines += [f"Relative errors (momentum: {norm}):", "",
                      _md_table(_similarity_header(args, with_z),
                                [[f"{row[0]:.1f}"] + [f"{v:.4f}" for v in row[1:]]
                                 for row in rows]), ""]

    thesis = test_dir / "pics" / "thesis"
    if error is None and thesis.is_dir():
        csvs = sorted(p.name for p in thesis.glob("*.csv"))
        extra = (" and " + ", ".join(f"`{c}`" for c in csvs)) if csvs else ""
        lines += [f"Thesis figures (PDF + PNG){extra}: "
                  f"`{thesis.relative_to(test_dir.parent).as_posix()}/`", ""]

    lines += _relative_images(test_dir, REPORT_PLOTS[name])
    if output:
        lines += ["", _details("console output (tables)", output)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LaTeX tables (booktabs) for the thesis
# ---------------------------------------------------------------------------

_LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\~{}",
                  "^": r"\^{}", "|": r"$|$"}


def _tex(text):
    text = "".join(_LATEX_ESCAPES.get(c, c) for c in str(text))
    return text.replace(">=", r"$\geq$").replace("<=", r"$\leq$")


def _latex_table(caption, label, colspec, header, rows, packages="booktabs",
                 environment="tabular"):
    """booktabs table environment as text; `rows` are lists of cell strings."""
    width = "{\\textwidth}" if environment == "tabularx" else ""
    return "\n".join([
        f"% generated by verification/run_all_tests.py - needs \\usepackage{{{packages}}}",
        "\\begin{table}[htbp]",
        "  \\centering\\small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{{environment}}}{width}{{{colspec}}}",
        "    \\toprule",
        "    " + " & ".join(header) + r" \\",
        "    \\midrule",
        *("    " + " & ".join(row) + r" \\" for row in rows),
        "    \\bottomrule",
        f"  \\end{{{environment}}}",
        "\\end{table}",
        ""])


def _tex_number(value, digits=2):
    """Number with `digits` significant digits; tiny values as a x 10^b."""
    if not np.isfinite(value):
        return "--"
    if value != 0 and abs(value) < 1e-2:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{value:.{digits}g}"


def _error_table_rows(rows):
    return [[f"{row[0]:.1f}"] + [f"{v:.3f}" for v in row[1:]] for row in rows]


def _error_table_header(with_z, momentum=r"h\mathbf{u}"):
    head = [r"$t$ (s)", r"$h$, $L_2$", r"$h$, $L_{\max}$",
            rf"${momentum}$, $L_2$", rf"${momentum}$, $L_{{\max}}$"]
    if with_z:
        head += [r"$hu_z$, $L_2$", r"$hu_z$, $L_{\max}$"]
    return head


def write_latex_tables(run_dir, entries, args):
    """booktabs tables for the thesis, written to the run directory:
    summary.tex (all tests) plus, for every test that ran without error,
    similarity_errors.tex, dam_break_errors.tex, curvature.tex and
    rotation.tex. Returns the written paths."""
    model = _tex(args.sim_model)
    tag = re.sub("[^a-z0-9]", "", args.sim_model.lower())
    tables = {}

    rows = []
    for name in args.tests:
        entry = entries[name]
        status = "error" if entry["error"] else entry["assessment"][0]
        headline = "" if entry["error"] else _headline(name, entry["ret"], args)
        rows.append([_tex(TEST_TITLES[name]), _tex(status), _tex(headline)])
    tables["summary.tex"] = _latex_table(
        f"Verification tests of the {model} model.", f"tab:verification_{tag}", "llX",
        ["test", "status", "result"], rows, "booktabs,tabularx", "tabularx")

    def ok(name):
        entry = entries.get(name)
        return entry is not None and not entry["error"] and entry["ret"]

    if ok("similarity_solution"):
        with_z = _with_z("similarity_solution", args)
        rows = _similarity_rows(entries["similarity_solution"]["ret"], args, with_z)
        if rows:
            norm = ("the 3D vector $(hu_x, hu_y, hu_z)$" if with_z
                    else "the in-plane vector $(hu_x, hu_y)$")
            tables["similarity_errors.tex"] = _latex_table(
                f"Relative errors of the {model} model against the similarity solution of "
                f"Hutter et al.~(1993); the momentum error uses {norm}.",
                f"tab:simisol_{tag}", "r" * len(rows[0]), _error_table_header(with_z),
                _error_table_rows(rows))

    if ok("dam_break"):
        rows = _similarity_rows(entries["dam_break"]["ret"], args, False)
        if rows:
            tables["dam_break_errors.tex"] = _latex_table(
                f"Relative errors of the {model} model against the dam break solution of "
                "Faccanoni and Mangeney~(2012) within the error domain; the momentum error "
                r"uses $h\lVert\mathbf{u}\rVert$.",
                f"tab:dambreak_{tag}", "r" * len(rows[0]),
                _error_table_header(False, r"h\lVert\mathbf{u}\rVert"),
                _error_table_rows(rows))

    if ok("curvature"):
        cases = entries["curvature"]["ret"]["cases"]
        rows = []
        for enabled in (False, True):
            r = cases[enabled]["result"]
            rows.append(["on" if enabled else "off",
                         f"{r['horizontal_runout']:.1f}", f"{r['simulated_runout']:.1f}",
                         f"{r['runout_error']:+.1f}", f"{r['energy_rmse']:.2f}",
                         _tex_number(r['final_rms_speed'])])
        tables["curvature.tex"] = _latex_table(
            f"Curvature test of the {model} model: horizontal runout of the analytical "
            "point slider and of the simulated center of mass, energy-height RMSE and the "
            "final RMS particle speed.",
            f"tab:curvature_{tag}", "lrrrrr",
            ["curvature", "analytical (m)", "simulated (m)", r"$\Delta$ (m)",
             "RMSE (m)", r"$u_\mathrm{rms}$ (m s$^{-1}$)"], rows)

    if ok("rotation"):
        results = entries["rotation"]["ret"]
        ref = results[0]
        rows = [[f"{r['angle']:g}", f"{r['sEnd']:.2f}", f"{r['sEnd'] - ref['sEnd']:+.2f}",
                 f"{r['runoutAngle']:.3f}", f"{r['runoutAngle'] - ref['runoutAngle']:+.3f}",
                 "--" if r["iou"] is None else f"{r['iou']:.3f}"] for r in results]
        tables["rotation.tex"] = _latex_table(
            f"Rotation test of the {model} model: center-of-mass runout length and runout "
            f"angle per grid rotation, differences to the reference ({ref['angle']:g}"
            r"$^\circ$) and intersection over union (IoU) of the affected area.",
            f"tab:rotation_{tag}", "rrrrrr",
            [r"rotation ($^\circ$)", r"$s_\mathrm{end}$ (m)", r"$\Delta s$ (m)",
             r"runout angle ($^\circ$)", r"$\Delta$ ($^\circ$)", "IoU"], rows)

    if ok("double_bump"):
        r = entries["double_bump"]["ret"]
        rows = [[f"{r['particles']}", f"{100 * r['takeoff_fraction']:.1f}",
                 f"{100 * r['takeoff_at_bump_1']:.1f}", f"{100 * r['landing_on_bump_2']:.1f}",
                 f"{r['max_flying']}", f"{r['max_flight_height']:.1f}", f"{r['flying_at_end']}"]]
        tables["double_bump.tex"] = _latex_table(
            f"Double-bump test of the {model} model: share of the particles that took off, "
            "share of the first take-offs at the first bump and of the first landings on the "
            "second bump, most particles in flight at once, highest flight above the bed and "
            "particles still flying at the end.",
            f"tab:doublebump_{tag}", "rrrrrrr",
            ["particles", r"took off (\%)", r"bump 1 (\%)", r"bump 2 (\%)",
             "max. flying", "max. height (m)", "flying at end"], rows)
        if r.get("reference") is not None:
            rows = [[label, f"{r[key]:.1f}", f"{r['reference_' + key]:.1f}",
                     f"{r[key] - r['reference_' + key]:+.1f}"]
                    for key, label in (("takeoff_x", "first take-off (median)"),
                                       ("landing_x", "first landing (median)"),
                                       ("stop_x", "stop (center of mass)"))]
            tables["double_bump_reference.tex"] = _latex_table(
                f"Double-bump test of the {model} model against the Coulomb point slider: "
                r"horizontal position $x$ of the first take-off and landing and of the stop.",
                f"tab:doublebump_reference_{tag}", "lrrr",
                ["event", r"simulation (m)", r"point slider (m)", r"$\Delta x$ (m)"], rows)

    paths = []
    for name, text in tables.items():
        path = run_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def collect_thesis_files(run_dir, test_dirs, tex_paths):
    """Copy all thesis figures (PNG + PDF) and LaTeX tables into
    <run_dir>/thesis_all so they can be copied to the thesis in one go.
    Returns the folder."""
    target = run_dir / "thesis_all"
    target.mkdir(parents=True, exist_ok=True)
    sources = [p for d in test_dirs.values() if (d / "pics" / "thesis").is_dir()
               for p in sorted((d / "pics" / "thesis").iterdir())
               if p.suffix.lower() in (".png", ".pdf", ".tex")]
    for path in list(sources) + list(tex_paths):
        shutil.copy2(path, target / f"verification_{path.name}")
    return target


def _build_info():
    """avalanchers software version (with commit / dirty flag) of this run."""
    try:
        import avalanchers
        return str(avalanchers.__software_version__)
    except Exception:
        return "unknown"


def write_report(run_dir, test_dirs, entries, args, elapsed, timestamp):
    summary_rows = []
    for name in args.tests:
        entry = entries[name]
        status = "error" if entry["error"] else entry["assessment"][0]
        headline = _headline(name, entry["ret"], args) if not entry["error"] else ""
        summary_rows.append([TEST_TITLES[name], status, headline])
    tex_paths = write_latex_tables(run_dir, entries, args)
    collect_thesis_files(run_dir, test_dirs, tex_paths)
    momentum = ("3D (x, y, z) - the model carries the vertical momentum"
                if _is_three_d(args) else "in-plane (x, y)")
    doc = [
        "# avalanchers verification report",
        "",
        _md_table(["run", "value"], [
            ["sim model", f"**{args.sim_model}**"
             + (f", friction model {args.friction_model}" if args.friction_model else "")
             + (f", mu {args.mu}" if args.mu else "")],
            ["momentum compared", momentum],
            ["particles per cell", str(args.particles_per_cell)
             if args.particles_per_cell is not None else "test defaults"],
            ["grid atomics", "integer (forced with --integer-atomics)"
             if args.integer_atomics else "float where the GPU supports them"],
            ["avalanchers build", f"`{_build_info()}`"],
            ["tests", ", ".join(TEST_TITLES[t] for t in args.tests)
             + (" (quick mode)" if args.quick else "")],
            ["timestamp", timestamp],
            ["runtime", f"{elapsed / 60:.1f} min"],
            ["LaTeX tables", ", ".join(f"`{p.name}`" for p in tex_paths)],
        ]),
        "",
        "The analytic references apply to the depth-averaged models (curvilinear, "
        "terrain-following, dampm-global-cartesian) with Coulomb friction; other models are "
        "comparison runs.",
        "",
        "## Summary",
        "",
        _md_table(["test", "status", "headline"], summary_rows),
        "",
        "Regression guardrails (not a physical model certification): final flow thickness "
        "relative L2 <= 0.15 and momentum relative L2 <= 0.25 at t >= 20 s (10 s in quick "
        "mode); energy line runout error <= 2 %, velocity-height RMSE <= 3 m; rotation "
        "runout difference <= 2 % and IoU >= 0.74; curvature: each slider runout error "
        "<= 2 %, energy-height RMSE <= 3 m, final RMS speed <= 0.1 m/s, the curvature load "
        "must shorten the runout and halve the energy error of the uncorrected alpha line; "
        f"double bump (3D only): >= {100 * DOUBLE_BUMP_MIN_TAKEOFF:.0f} % of the first take-offs "
        f"at the first bump, >= {100 * DOUBLE_BUMP_MIN_LANDING:.0f} % of the first landings on "
        "the second bump, no particle flying at the end, median first take-off and landing "
        f"within {DOUBLE_BUMP_MAX_EVENT_ERROR:g} m and the final center of mass within "
        f"{100 * DOUBLE_BUMP_MAX_STOP_ERROR:g} % of the travel of the Coulomb point slider.",
        "",
    ]
    for name in args.tests:
        entry = entries[name]
        doc.append(_test_section(name, test_dirs[name], entry["ret"],
                                 entry["output"], entry["error"], args, entry["assessment"]))
    report = run_dir / "report.md"
    report.write_text("\n".join(doc), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="run all avalanchers verification tests and write a markdown report")
    parser.add_argument("--sim-model", default=DEFAULT_SIM_MODEL,
                        help="dampm-global-cartesian | terrain-following | curvilinear | mpmdac | "
                             f"sliding-block (default: {DEFAULT_SIM_MODEL})")
    parser.add_argument("--friction-model", default=None,
                        help="override the friction model (default: coulomb)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Coulomb friction coefficient override")
    parser.add_argument("--weak-pressure", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="weak-form pressure gradient of the curvilinear model "
                             "(default: each test's SETTINGS)")
    from cli_args import add_particles_per_cell_argument
    add_particles_per_cell_argument(parser)
    parser.set_defaults(particles_per_cell=DEFAULT_PARTICLES_PER_CELL)
    parser.add_argument("--tests", nargs="+", default=TEST_NAMES, choices=TEST_NAMES)
    parser.add_argument("--out-dir", default=None,
                        help="run directory (default: all_tests/<model>_<timestamp>)")
    parser.add_argument("--quick", action="store_true",
                        help="reduced run lengths (rotation 2 angles, t_end 10 s)")
    parser.add_argument("--no-sequences", action="store_true",
                        help="skip the per-time-step figure sequences and GIFs")
    parser.add_argument("--integer-atomics", action="store_true",
                        help="accumulate the grid with integer atomics also on GPUs "
                             "with float atomics, as in the browser")
    args = parser.parse_args()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    atomics = "_int" if args.integer_atomics else ""
    tag = f"{args.sim_model}{atomics}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = pathlib.Path(args.out_dir) if args.out_dir else \
        ALL_TESTS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== running {len(args.tests)} tests: "
          f"{', '.join(TEST_TITLES[t] for t in args.tests)} "
          f"(sim model: {args.sim_model}) ===")
    entries = {}
    test_dirs = {}
    start = time.time()
    for name in args.tests:
        test_dir = run_dir / name
        test_dir.mkdir(parents=True, exist_ok=True)
        test_dirs[name] = test_dir
        print(f"\n--- {TEST_TITLES[name]} ---", flush=True)
        ret, output, error = RUNNERS[name](test_dir, args)
        assessment = ("error", []) if error else _assessment(name, ret, args)
        entries[name] = {"ret": ret, "output": output, "error": error, "assessment": assessment}
        status = assessment[0].upper()
        print(f"--- {TEST_TITLES[name]}: {status} ---", flush=True)
    elapsed = time.time() - start

    report = write_report(run_dir, test_dirs, entries, args, elapsed, timestamp)
    print(f"\nreport written to {report}")
    print(f"LaTeX tables in {run_dir}")
    print(f"all thesis PNG/PDF/TEX files in {run_dir / 'thesis_all'}")
    for name in args.tests:
        if entries[name]["assessment"][0] in ("error", "failed"):
            print(f"WARNING: {name} FAILED - see the report")
    if any(entry["assessment"][0] in ("error", "failed") for entry in entries.values()):
        raise RuntimeError(f"Verification failed; results were saved to {report}")
    return report


if __name__ == "__main__":
    main()
