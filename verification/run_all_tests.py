"""Run all four avalanchers verification tests and write a markdown report.

Runs the energy line, dam break, rotation and similarity solution tests
(see the sibling ``run_<test>.py`` scripts) IN-PROCESS with a switchable
simulation model, collects each test's console tables, summary plots and
animated GIFs, and writes everything into one markdown document:

    verification/outputs/all_tests/<model>_<timestamp>/report.md

Model switching: ``--sim-model`` accepts any avalanchers model string
(``terrain-following`` | ``curvilinear`` | ``mpmdac``) and is injected into
every test's settings before it runs. The analytic reference solutions
(alpha line, Faccanoni-Mangeney, Hutter 1993) are exact only for
``curvilinear`` + Coulomb friction - for other models the tests still run
and produce all plots/tables, but treat them as model-comparison runs, not
analytic validation. ``--friction-model`` / ``--mu`` optionally override
the friction (defaults: Coulomb with each test's documented angle).

Each test runs in its own subfolder and a failure of one test does not stop
the others - failures are reported in the markdown.

Usage:
    python run_all_tests.py                          # all 4 tests, curvilinear
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

TEST_NAMES = ["energy_line", "similarity_solution", "dam_break", "rotation"]

TEST_TITLES = {
    "energy_line": "Energy Line",
    "similarity_solution": "Similarity Solution",
    "dam_break": "Dam Break",
    "rotation": "Rotation",
}

# plot files embedded in the report, in order, per test (globs relative to
# <test>/pics)
REPORT_PLOTS = {
    "energy_line": ["*EnergyTest*.png"],
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


def run_energy_line_test(test_dir, args):
    import run_energy_line_test as mod

    _patch_common(mod.SETTINGS, args)
    extra = []
    if args.quick:
        extra += ["--max-steps", "3000"]
    return _capture(mod, ["--out-dir", str(test_dir)] + extra)


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
    "dam_break": run_dam_break_test,
    "rotation": run_rotation_test,
    "similarity_solution": run_similarity_solution_test,
}


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


def _test_section(name, test_dir, ret, output, error, args):
    lines = [f"## {TEST_TITLES[name]}", ""]
    if error is not None:
        lines += ["**FAILED**", "", "```", error.strip(), "```", ""]
    else:
        lines += ["status: **ok**", ""]
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
        status = "failed" if entry["error"] else "ok"
        headline = _headline(name, entry["ret"]) if not entry["error"] else ""
        summary_rows.append([TEST_TITLES[name], status, headline])
    doc = [
        "# avalanchers verification report",
        "",
        f"- tests: {', '.join(TEST_TITLES[t] for t in args.tests)}",
        f"- sim model: **{args.sim_model}**"
        + (f", friction model: {args.friction_model}" if args.friction_model else "")
        + (f", mu: {args.mu}" if args.mu else "")
        + " (analytic references are exact for curvilinear + Coulomb only)",
        f"- timestamp: {timestamp}",
        f"- runtime: {elapsed / 60:.1f} min",
        "",
        "## Summary",
        "",
        _md_table(["test", "status", "headline"], summary_rows),
        "",
    ]
    for name in args.tests:
        entry = entries[name]
        doc.append(_test_section(name, test_dirs[name], entry["ret"],
                                 entry["output"], entry["error"], args))
    report = run_dir / "report.md"
    report.write_text("\n".join(doc), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="run all avalanchers verification tests and write a markdown report")
    parser.add_argument("--sim-model", default="curvilinear",
                        help="terrain-following | curvilinear | mpmdac")
    parser.add_argument("--friction-model", default=None,
                        help="override the friction model (default: coulomb)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Coulomb friction coefficient override")
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
        entries[name] = {"ret": ret, "output": output, "error": error}
        status = "FAILED" if error else "ok"
        print(f"--- {TEST_TITLES[name]}: {status} ---", flush=True)
    elapsed = time.time() - start

    report = write_report(run_dir, test_dirs, entries, args, elapsed, timestamp)
    print(f"\nreport written to {report}")
    for name in args.tests:
        if entries[name]["error"]:
            print(f"WARNING: {name} FAILED - see the report")
    return report


if __name__ == "__main__":
    main()
