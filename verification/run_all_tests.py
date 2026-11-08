"""Run avalanchers verification tests and write a markdown report.

Runs the energy line, curvature, dam break, rotation and similarity solution tests
(see the sibling ``run_<test>.py`` scripts) IN-PROCESS with a switchable
simulation model, collects each test's console tables, summary plots and
animated GIFs, and writes everything into one markdown document:

    verification/outputs/all_tests/<model>_<timestamp>/report.md

Model switching: ``--sim-model`` accepts any avalanchers model string
(``terrain-following`` | ``curvilinear`` | ``mpmdac`` | ``sliding-block``) and is injected into
every test's settings before it runs. The analytic reference solutions
(alpha line, Faccanoni-Mangeney, Hutter 1993) are exact only for
``curvilinear`` + Coulomb friction - for other models the tests still run
and produce plots/tables, but treat them as model-comparison runs, not
analytic validation. The curvature-specific benchmark is explicitly skipped
for other models or friction laws. ``--friction-model`` / ``--mu`` optionally override
the friction (defaults: Coulomb with each test's documented angle).

Each test runs in its own subfolder and a failure of one test does not stop
the others - failures are reported in the markdown. Curvilinear/Coulomb runs
also apply numerical regression guardrails; other models are labeled comparison
runs rather than numerical passes.

Usage:
    python run_all_tests.py                          # all 5 tests, curvilinear
    python run_all_tests.py --sim-model mpmdac       # switch the model
    python run_all_tests.py --tests energy_line dam_break --quick
"""

import argparse
import contextlib
import io
import math
import pathlib
import re
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # the runner is always headless

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ALL_TESTS_DIR = REPO_ROOT / "verification" / "outputs" / "all_tests"

TEST_NAMES = ["energy_line", "curvature", "similarity_solution", "dam_break", "rotation"]
# models checked against the analytical references (with Coulomb friction)
ANALYTIC_MODELS = ("curvilinear", "curvi", "c", "terrain-following", "terrain", "t",
                   "3d-depth-averaged", "3d")

TEST_TITLES = {
    "energy_line": "Energy Line",
    "curvature": "Curvature",
    "similarity_solution": "Similarity Solution",
    "dam_break": "Dam Break",
    "rotation": "Rotation",
}

# plot files embedded in the report, in order, per test (globs relative to
# <test>/pics)
REPORT_PLOTS = {
    "energy_line": ["*EnergyTest*.png"],
    "curvature": ["*CurvatureTest*.png"],
    "similarity_solution": ["*SimiSolTest.png", "*_profiles.gif", "*_contours.gif"],
    "dam_break": ["*DamBreakTest.png", "*_compare.gif"],
    "rotation": ["*_runout_comparison.png", "*_sl_comparison.png",
                 "*_areas_comparison.png", "*EnergyTest*.png"],
}

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


RUNNERS = {
    "energy_line": run_energy_line_test,
    "curvature": run_curvature_test,
    "dam_break": run_dam_break_test,
    "rotation": run_rotation_test,
    "similarity_solution": run_similarity_solution_test,
}


def _assessment(name, ret, args):
    """Regression guardrails, not a claim of physical model certification."""
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
                dice = result["dice"]
                if dice is not None and (not np.isfinite(dice) or dice < 0.85):
                    failures.append(f"Dice at {result['angle']:g} degrees must be >= 0.85.")
    return ("failed" if failures else "passed"), failures


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------

def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def _details(summary, body):
    return f"<details>\n<summary>{summary}</summary>\n\n```\n{body}\n```\n\n</details>"


def _relative_images(test_dir, patterns):
    """Markdown image lines for test_dir/pics following the given glob order."""
    pics = test_dir / "pics"
    if not pics.is_dir():
        return []
    lines = []
    seen = set()
    for pattern in patterns:
        for p in sorted(pics.glob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            rel = p.relative_to(test_dir.parent).as_posix()
            lines.append(f"![{p.name}]({rel})")
    return lines


def _headline(test, ret):
    """One-line headline metric per test from its main() return value."""
    try:
        if test == "energy_line":
            result = ret[0]
            return (f"runout s {result['sEnd']:.1f} m, angle {result['runoutAngle']:.2f} deg, "
                    f"ds diff {result['runOutSError']:+.2f} m, "
                    f"velocity-height rmse {result['rmseVelocityElevation']:.2f} m")
        if test == "curvature":
            if ret is None:
                return "skipped: curvature reference requires curvilinear or terrain-following with Coulomb"
            result = ret["cases"][True]["result"]
            return (f"runout sim/analytic {result['simulated_runout']:.1f}/{result['horizontal_runout']:.1f} m, "
                    f"energy RMSE {result['energy_rmse']:.3f} m, "
                    f"runout reduction {ret['runout_reduction']:.1f} m")
        if test in ("dam_break", "similarity_solution"):
            samples, errors = ret
            times = samples["t"]
            i_ref = int(np.argmin(np.abs(times - 5.0)))
            return (f"rel h L2 {errors['h_l2rel'][i_ref]:.3f} @ t={times[i_ref]:.1f} s, "
                    f"{errors['h_l2rel'][-1]:.3f} @ t={times[-1]:.1f} s")
        if test == "rotation":
            results = ret
            ref = results[0]
            max_ds = max(abs(r["sEnd"] - ref["sEnd"]) for r in results)
            dices = [r["dice"] for r in results if r["dice"] is not None]
            dice = f", Dice >= {min(dices):.3f}" if dices else ""
            return (f"runout s {ref['sEnd']:.1f} m at {ref['angle']:g} deg, "
                    f"max |ds| {max_ds:.2f} m over {len(results)} rotations{dice}")
    except Exception:
        return ""
    return ""


def _test_section(name, test_dir, ret, output, error, args, assessment):
    lines = [f"## {TEST_TITLES[name]}", ""]
    if error is not None:
        lines += ["**FAILED**", "", "```", error.strip(), "```", ""]
    else:
        lines += [f"status: **{assessment[0]}**", ""]
        lines += [f"- {reason}" for reason in assessment[1]]
        lines.append("")
    headline = _headline(name, ret) if error is None else ""
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
                         "-" if r["dice"] is None else f"{r['dice']:.3f}"])
        lines += ["| angle [deg] | sEnd [m] | zEnd [m] | runout angle [deg] | ds [m] | dz [m] | dAngle [deg] | Dice |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
        lines.append("")

    lines += _relative_images(test_dir, REPORT_PLOTS[name])
    if output:
        lines += ["", _details("console output (tables)", output)]
    return "\n".join(lines) + "\n"


def write_report(run_dir, test_dirs, entries, args, elapsed, timestamp):
    summary_rows = []
    for name in args.tests:
        entry = entries[name]
        status = "error" if entry["error"] else entry["assessment"][0]
        headline = _headline(name, entry["ret"]) if not entry["error"] else ""
        summary_rows.append([TEST_TITLES[name], status, headline])
    doc = [
        "# avalanchers verification report",
        "",
        f"- tests: {', '.join(TEST_TITLES[t] for t in args.tests)}",
        f"- sim model: **{args.sim_model}**"
        + (f", friction model: {args.friction_model}" if args.friction_model else "")
        + (f", mu: {args.mu}" if args.mu else "")
        + " (analytic references apply to curvilinear and terrain-following with Coulomb only)",
        f"- timestamp: {timestamp}",
        f"- runtime: {elapsed / 60:.1f} min",
        "- numerical guardrails: final thickness relative L2 <= 0.15, momentum relative L2 <= 0.25;",
        "  required field time 20 s (10 s in quick mode); energy runout error <= 2%,",
        "  velocity-height RMSE <= 3 m; rotation runout difference <= 2% and Dice >= 0.85.",
        "- These are regression guardrails, not physical model certification.",
        "- curvature: each slider runout error <= 2%, energy-height RMSE <= 3 m,",
        "  final RMS speed <= 0.1 m/s; curvature must shorten runout and halve",
        "  energy error relative to the uncorrected alpha line.",
        "",
        "## Summary",
        "",
        _md_table(["test", "status", "headline"], summary_rows),
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
    parser.add_argument("--sim-model", default="curvilinear",
                        help="terrain-following | 3d-depth-averaged | curvilinear | mpmdac | sliding-block")
    parser.add_argument("--friction-model", default=None,
                        help="override the friction model (default: coulomb)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Coulomb friction coefficient override")
    parser.add_argument("--weak-pressure", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="weak-form pressure gradient of the curvilinear model "
                             "(default: each test's SETTINGS)")
    parser.add_argument("--tests", nargs="+", default=TEST_NAMES, choices=TEST_NAMES)
    parser.add_argument("--out-dir", default=None,
                        help="run directory (default: all_tests/<model>_<timestamp>)")
    parser.add_argument("--quick", action="store_true",
                        help="reduced run lengths (rotation 2 angles, t_end 10 s)")
    parser.add_argument("--no-sequences", action="store_true",
                        help="skip the per-time-step figure sequences and GIFs")
    args = parser.parse_args()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = f"{args.sim_model}_{time.strftime('%Y%m%d_%H%M%S')}"
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
    for name in args.tests:
        if entries[name]["assessment"][0] in ("error", "failed"):
            print(f"WARNING: {name} FAILED - see the report")
    if any(entry["assessment"][0] in ("error", "failed") for entry in entries.values()):
        raise RuntimeError(f"Verification failed; results were saved to {report}")
    return report


if __name__ == "__main__":
    main()
