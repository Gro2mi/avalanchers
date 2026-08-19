#!/usr/bin/env python3
# @atlas: On-GPU before/after evidence for the two scoring-convention fixes (single-particle tails, ridge/watershed release leak). Run this on the box before any tuning stage; writes fix_validation.json for the final report.
"""Measure what the two scoring-convention fixes actually do, on real cases.

These fixes change the objective, so nothing may be tuned against them until
their effect is measured. This produces that measurement and the artifacts the
final report needs. It must be run on a machine with a GPU and the case data;
it cannot be run on a laptop.

What it measures, and how:

  TAIL FILTER (residence + release-connectivity). Every evaluation already
  records both the filtered and the unfiltered footprint -- `omega` vs
  `omega_raw`, `sim_reach_m` vs `sim_reach_raw_m` -- so one run carries its own
  before/after and no paired run is needed. Reach is the number to watch: it is
  a max over the footprint, so a single surviving trace cell sets it, which is
  exactly the "runouts read too long" complaint.

  RIDGE CLIP. This one changes the release area, so everything downstream moves
  with it and it cannot be captured within a single run. Two runs, identical
  but for `--clip-drainage 0/1`, paired per case.

  RESIDENCE SENSITIVITY. `min_residence` is the one convention chosen by
  argument rather than measurement. The sweep shows how sharply the score
  depends on it; a value on a plateau is defensible, a value on a cliff is not.

    python3 python_scripts/validate_fixes.py \\
        --calibrate-bin ~/avalanchers/target/release/calibrate \\
        --cases ~/data/cases100.json --cache ~/data/dtm_cache.zarr \\
        --cwd ~/data --out-dir ~/data/fixval --gpus 8 --per-gpu 8
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# Cases whose release the D8 clip actually cuts, adjudicated one by one from
# the traced descent paths -- the clip's own evidence, not a borrowed list.
#
# WITHDRAWN: this was previously the 12 worst cases by
# `release_frac_outside_drainage` in crossing_report.json, with the criterion
# "the clip has to visibly change these or it is not doing its job". That was
# the wrong yardstick and it failed on the box, correctly: 10 of those 12 show a
# 0.0-0.2% cut. crossing_analysis.py measures DOWNSTREAM mass crossing, which is
# friction-driven and mid-path (notes_for_markus.md item 15); this clip fixes
# RELEASE-cell drainage. The list is also built on the basin clustering whose
# false positives this design refused to inherit -- aval_6719 was on it, a case
# that method calls 100% leaking while it scores +0.337.
#
# Adjudicated verdicts, GPU run:
LEAK_CASES = ["aval_8124", "aval_8025", "aval_11149", "aval_13404"]   # genuine:
#   removed cells travel 335-735 m into foreign drainage or deep pits
FALSE_POSITIVES_FIXED = ["aval_13722"]  # cells travelled 80-230 m into
#   0.06-0.51 m hollows; fixed by the pit-fill tolerance rather than by excluding
#   the case

# Cases with the most detached footprint area in campaign/analysis/frag_report.json.
TAIL_CASES = ["aval_6595", "aval_11100", "aval_6631", "aval_12616", "aval_10321",
              "aval_7945", "aval_8124", "aval_10721", "aval_18649", "aval_8276"]

# The operating point to measure at: the incumbent structure and the xi the
# +0.2864 fixed-xi per-event run used. Deliberately not the repo defaults, which
# never arrest and would make every footprint edge-clipped.
BASE = ["--model", "1", "--flags", "7", "--xi", "754", "--mu", "0.36", "--slab", "0.56"]


def run_calibrate(args, tag, extra, cases=None):
    out = Path(args.out_dir) / f"{tag}.json"
    if out.exists() and not args.force:
        print(f"  ({tag}: already done)")
        return json.loads(out.read_text())
    cmd = [args.calibrate_bin, "--cases", args.cases, "--cache", args.cache,
           "--padding", str(args.padding), "--stage", "fixval", "--candidate", tag,
           "--out", str(out), "--eval-log", str(out) + ".evals.jsonl"]
    if cases:
        cmd += ["--only", ",".join(cases)]
    if args.gpu_index is not None:
        cmd += ["--gpu-index", str(args.gpu_index)]
    cmd += ["run", *BASE, *extra]
    print(f"  $ {' '.join(cmd)}")
    log = Path(args.out_dir) / f"{tag}.log"
    with open(log, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=args.cwd)
    if r.returncode != 0 or not out.exists():
        sys.exit(f"{tag} failed, see {log}")
    return json.loads(out.read_text())


def by_case(res):
    return {c["name"]: c for c in res["cases"]}


def refused_cases(res):
    """Cases the drainage clip refused outright. They are absent from `cases`
    and present in `failures`, so a paired comparison would silently drop them
    -- which is the opposite of what the guard is for."""
    return sorted(n for n, why in res.get("failures", [])
                  if "drainage clip" in why)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate-bin", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cwd", default=".")
    p.add_argument("--padding", type=float, default=300.0)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--per-gpu", type=int, default=8)
    p.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    p.add_argument("--skip-sweep", action="store_true")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    report = {}

    # ---- 1. both fixes on: the run that carries its own tail before/after ----
    print("\n== 1. filters ON (carries raw vs filtered internally) ==")
    on_raw = run_calibrate(args, "filters_on", [])
    on = by_case(on_raw)
    refused = refused_cases(on_raw)
    report["refused_cases"] = refused
    if refused:
        print(f"  {len(refused)} case(s) REFUSED by the drainage clip "
              f"(no release cell drains to the outline): {', '.join(refused)}")

    tail = []
    for n, c in on.items():
        tail.append({
            "case": n,
            "omega_raw": c["omega_raw"], "omega": c["area"]["omega"],
            "d_omega": c["area"]["omega"] - c["omega_raw"],
            "reach_raw_m": c["sim_reach_raw_m"], "reach_m": c["sim_reach_m"],
            "d_reach_m": c["sim_reach_m"] - c["sim_reach_raw_m"],
            "obs_reach_m": c["obs_reach_m"],
            "cells_raw": c["sim_cells_raw"], "cells": c["sim_cells"],
            "reach_err_raw_m": c["sim_reach_raw_m"] - c["obs_reach_m"],
            "reach_err_m": c["reach_err_m"],
            "connect_fallback": c["connect_fallback"],
        })
    report["tail_filter"] = tail
    do = [t["d_omega"] for t in tail]
    dr = [t["d_reach_m"] for t in tail]
    era = [abs(t["reach_err_raw_m"]) for t in tail]
    erf = [abs(t["reach_err_m"]) for t in tail]
    print(f"\n  TAIL FILTER over {len(tail)} cases")
    print(f"    delta omega : mean {sum(do)/len(do):+.4f}  median {med(do):+.4f}  "
          f"max {max(do):+.4f}")
    print(f"    delta reach : mean {sum(dr)/len(dr):+.1f} m  median {med(dr):+.1f} m  "
          f"min {min(dr):+.1f} m")
    print(f"    |reach error| median {med(era):.1f} m -> {med(erf):.1f} m")
    print(f"    cells removed: median "
          f"{med([1 - t['cells']/max(t['cells_raw'],1) for t in tail])*100:.2f}%")
    print(f"    connectivity fallbacks (no seed): "
          f"{sum(1 for t in tail if t['connect_fallback'])}")
    print("\n    worst long-tail cases:")
    for t in sorted(tail, key=lambda t: t["d_reach_m"])[:10]:
        flag = " *" if t["case"] in TAIL_CASES else ""
        print(f"      {t['case']:>12s} reach {t['reach_raw_m']:7.0f} -> {t['reach_m']:7.0f} m "
              f"(obs {t['obs_reach_m']:6.0f})  omega {t['omega_raw']:+.3f} -> "
              f"{t['omega']:+.3f}{flag}")

    # ---- 2. ridge clip: needs a paired run, it changes the release ----------
    print("\n== 2. ridge clip OFF, for the paired comparison ==")
    off = by_case(run_calibrate(args, "clip_off", ["--clip-drainage", "0"]))

    ridge = []
    for n in sorted(set(on) & set(off)):
        a, b = off[n], on[n]
        ridge.append({
            "case": n,
            "omega_unclipped": a["area"]["omega"], "omega_clipped": b["area"]["omega"],
            "d_omega": b["area"]["omega"] - a["area"]["omega"],
            "release_cells_unclipped": a["release_cells"],
            "release_cells_clipped": b["release_cells"],
            "release_clipped_frac": b["release_clipped_frac"],
            "clip_severe": b["release_clip_severe"],
            "reach_err_unclipped_m": a["reach_err_m"], "reach_err_clipped_m": b["reach_err_m"],
            "known_leak": n in LEAK_CASES,
        })
    report["ridge_clip"] = ridge
    touched = [r for r in ridge if r["release_clipped_frac"] > 0]
    known = [r for r in ridge if r["known_leak"]]
    dall = [r["d_omega"] for r in ridge]
    dt = [r["d_omega"] for r in touched]
    print(f"\n  RIDGE CLIP over {len(ridge)} cases")
    print(f"    cases with any release clipped: {len(touched)}/{len(ridge)}")
    severe = [r for r in ridge if r["clip_severe"]]
    print(f"    SEVERE (>50% of the release clipped): {len(severe)}/{len(ridge)}")
    if severe:
        print("      " + ", ".join(f"{r['case']} ({r['release_clipped_frac']*100:.0f}%)"
                                   for r in sorted(severe,
                                                   key=lambda r: -r["release_clipped_frac"])))
    print(f"    cases REFUSED (clip emptied the release): {len(refused)}")
    if refused:
        print("      " + ", ".join(sorted(refused)))
    print(f"    delta omega, all cases : mean {sum(dall)/len(dall):+.4f}  "
          f"median {med(dall):+.4f}")
    if dt:
        print(f"    delta omega, clipped   : mean {sum(dt)/len(dt):+.4f}  "
              f"median {med(dt):+.4f}  max {max(dt):+.4f}")
    print("\n    adjudicated leak cases (D8-traced, not borrowed from crossing_analysis):")
    for r in sorted(known, key=lambda r: -r["release_clipped_frac"]):
        fb = " SEVERE" if r["clip_severe"] else ""
        print(f"      {r['case']:>12s} release {r['release_cells_unclipped']:5d} -> "
              f"{r['release_cells_clipped']:5d} ({r['release_clipped_frac']*100:5.1f}% cut)  "
              f"omega {r['omega_unclipped']:+.3f} -> {r['omega_clipped']:+.3f}{fb}")

    # ---- 3. residence sensitivity ------------------------------------------
    if not args.skip_sweep:
        print("\n== 3. residence sensitivity ==")
        sweep = []
        for v in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            r = run_calibrate(args, f"resid_{v}", ["--min-residence", str(v)])
            sweep.append({"min_residence": v, "mean_omega": r["mean_omega"],
                          "mean_hwri_l1": r["mean_hwri_l1"],
                          "median_reach_err_m": med([abs(c["reach_err_m"])
                                                     for c in r["cases"]])})
        report["residence_sweep"] = sweep
        print(f"\n    {'min_residence':>14} {'mean omega':>11} {'|reach err| med':>16}")
        for s in sweep:
            print(f"    {s['min_residence']:>14.2f} {s['mean_omega']:>+11.4f} "
                  f"{s['median_reach_err_m']:>16.1f}")

    dest = Path(args.out_dir) / "fix_validation.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
