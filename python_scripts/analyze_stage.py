#!/usr/bin/env python3
# @atlas: Turns one ingested stage into a freeze decision: per-candidate panel means, paired bootstrap CIs and Wilcoxon against the incumbent, cal/val consistency, and the explicit accept/keep verdict from the plan's criterion.
"""Score one stage of the global-constants calibration and apply the freeze rule.

For each global candidate this takes the best inner-loop score per case
(the per-case (mu, slab) optimum at that candidate), forms the paired
difference against the incumbent over the shared panel, and reports the
bootstrap CI and Wilcoxon p. Same machinery as `campaign/analysis/e2e_stats.py`,
which is what the earlier end-to-end comparisons used.

The freeze rule, stated in campaign/GLOBAL_CALIBRATION_PLAN.md: a challenger
replaces the incumbent only if the paired mean gain is at least DELTA_MIN
(0.02, the repo's own near-optimal band) *and* the 95% bootstrap CI excludes
zero. Anything else keeps the incumbent, which is the cheaper and simpler
configuration by construction. Ties on score break toward lower cost.

    python3 python_scripts/analyze_stage.py --db ~/data/calibration.sqlite \\
        --stage s1_structure --incumbent m1_voellmy_c1p1e1n0
"""
import argparse
import json
import sqlite3
import sys

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # the box may not have scipy; the bootstrap still works
    sps = None
    # Loudly, because a NaN in a p column reads like "not significant" rather
    # than "not computed", and that misreading survives into a report.
    print("NOTE: scipy not installed -- Wilcoxon p-values will be NaN (NOT "
          "'not significant'). The freeze rule uses the bootstrap CI only, so "
          "decisions are unaffected; install scipy for the reported p column.",
          file=sys.stderr)

# The decision band. 0.02 Omega_T is the width the repo already uses to call
# two parameter vectors indistinguishable (the identifiability grid work), so
# a global knob that moves the panel mean by less than that is not worth
# freezing to a non-default value.
DELTA_MIN = 0.02
N_BOOT = 20000


def mean_cost(con, stage):
    """Measured seconds per simulation per candidate. Stage 0 trades accuracy
    against cost, and cost is recorded rather than declared -- `ppc` scales
    work linearly in principle, but the arresting behaviour of a run at a given
    cfl is not something to predict from a formula."""
    return {cid: s for cid, s in con.execute(
        "SELECT candidate_id, AVG(seconds) FROM evals WHERE stage = ? AND ok = 1 "
        "GROUP BY candidate_id", (stage,)).fetchall()}


def best_per_case(con, stage, metric):
    """Best inner-loop score per (candidate, case). For a per-event stage this
    is the per-case optimum at that candidate; for a one-shot stage (`run`,
    `apply`) there is a single evaluation and the max is that evaluation."""
    rows = con.execute(
        f"SELECT candidate_id, case_name, MAX({metric}) FROM evals "
        f"WHERE stage = ? AND ok = 1 AND {metric} IS NOT NULL "
        f"GROUP BY candidate_id, case_name", (stage,)).fetchall()
    out = {}
    for cid, case, v in rows:
        out.setdefault(cid, {})[case] = v
    return out


def paired(x, base, rng):
    d = x - base
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float("nan")
    if sps is not None and np.any(d != 0):
        p = float(sps.wilcoxon(x, base).pvalue)
    return float(d.mean()), float(lo), float(hi), p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--incumbent", required=True, help="candidate id to compare against")
    ap.add_argument("--metric", default="omega",
                    choices=["omega", "omega_cells", "hwri_l1", "hwri_l05"],
                    help="omega_cells is the one comparable across the "
                         "particle-interaction flag")
    ap.add_argument("--splits", default=None,
                    help="JSON {case: 'cal'|'val'} for the consistency check")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    ap.add_argument("--equivalence", type=float, default=None, metavar="TOL",
                    help="stage-0 mode: instead of asking which candidate is better, "
                         "ask which are indistinguishable from the reference "
                         "(|delta| < TOL, CI inside +/-2*TOL) and report the cheapest. "
                         "The plan uses TOL=0.005 for numerics.")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    scores = best_per_case(con, args.stage, args.metric)
    if not scores:
        raise SystemExit(f"no evaluations for stage {args.stage} in {args.db}")
    if args.incumbent not in scores:
        raise SystemExit(f"incumbent {args.incumbent} not among "
                         f"{len(scores)} candidates: {sorted(scores)[:5]}...")

    # The panel is the intersection: a candidate that failed on a case cannot
    # be credited with the cases it did finish, or a configuration that crashes
    # on the hard cases would look best.
    panel = set.intersection(*(set(v) for v in scores.values()))
    everywhere = set.union(*(set(v) for v in scores.values()))
    dropped = everywhere - panel
    if dropped:
        # Which candidates are short, not just how many cases went missing:
        # one crashed configuration silently shrinking the panel for all 48
        # comparisons is worth naming rather than counting.
        short = sorted(((len(everywhere - set(v)), c) for c, v in scores.items()
                        if set(v) != everywhere), reverse=True)[:5]
        print(f"note: analysing {len(panel)} of {len(everywhere)} cases -- "
              f"{len(dropped)} absent from at least one candidate and so excluded "
              f"from every comparison.", file=sys.stderr)
        print(f"      missing: {sorted(dropped)[:8]}", file=sys.stderr)
        print("      worst candidates: "
              + ", ".join(f"{c} (-{n})" for n, c in short), file=sys.stderr)
    names = sorted(panel)
    if len(names) < 10:
        raise SystemExit(f"only {len(names)} cases shared by all candidates; "
                         "check for a candidate that failed wholesale")

    splits = json.loads(open(args.splits).read()) if args.splits else {}
    cal = [n for n in names if splits.get(n) == "cal"]
    val = [n for n in names if splits.get(n) == "val"]

    base = np.array([scores[args.incumbent][n] for n in names])
    costs = mean_cost(con, args.stage)
    rng = np.random.default_rng(0)
    table = []
    for cid, per_case in scores.items():
        x = np.array([per_case[n] for n in names])
        delta, lo, hi, p = paired(x, base, rng)
        row = {"candidate": cid, "mean": float(x.mean()), "delta": delta,
               "lo": lo, "hi": hi, "p": p, "n": len(names)}
        if cal and val:
            bc = np.array([scores[args.incumbent][n] for n in cal])
            bv = np.array([scores[args.incumbent][n] for n in val])
            row["delta_cal"] = float(np.array([per_case[n] for n in cal]).mean() - bc.mean())
            row["delta_val"] = float(np.array([per_case[n] for n in val]).mean() - bv.mean())
        # The rule, applied rather than left to the reader.
        if args.equivalence is not None:
            tol = args.equivalence
            row["accept"] = bool(abs(delta) < tol and lo > -2 * tol and hi < 2 * tol)
            row["seconds"] = costs.get(cid)
        else:
            row["accept"] = bool(delta >= DELTA_MIN and lo > 0)
        table.append(row)

    if args.equivalence is not None:
        table.sort(key=lambda r: (r["seconds"] or 0.0))
    else:
        table.sort(key=lambda r: -r["mean"])
    inc = next(r for r in table if r["candidate"] == args.incumbent)

    print(f"\nstage {args.stage}, metric {args.metric}, {len(names)} cases, "
          f"{len(table)} candidates")
    print(f"incumbent {args.incumbent}: mean {inc['mean']:+.4f}\n")
    hdr = f"{'candidate':>28} {'mean':>8} {'delta':>8} {'95% CI':>20} {'p':>9} {'accept':>7}"
    if args.equivalence is not None:
        hdr += f" {'s/sim':>7}"
    if cal and val:
        hdr += f" {'cal':>8} {'val':>8}"
    print(hdr)
    for r in table[:args.top]:
        line = (f"{r['candidate']:>28} {r['mean']:>+8.4f} {r['delta']:>+8.4f} "
                f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['p']:>9.2g} "
                f"{'YES' if r['accept'] else '-':>7}")
        if args.equivalence is not None:
            line += f" {r['seconds'] or 0:>7.3f}"
        if cal and val:
            line += f" {r['delta_cal']:>+8.4f} {r['delta_val']:>+8.4f}"
        print(line)

    winners = [r for r in table if r["accept"]]
    print()
    if args.equivalence is not None:
        if not winners:
            print(f"NO EQUIVALENT SETTING: nothing is within {args.equivalence} of "
                  f"{args.incumbent}. Keep the reference and re-check the tolerance.")
        else:
            w = winners[0]
            ref = costs.get(args.incumbent) or 0.0
            speedup = ref / w["seconds"] if w.get("seconds") else float("nan")
            print(f"FREEZE: adopt {w['candidate']} -- indistinguishable from the "
                  f"reference (delta {w['delta']:+.4f}, CI [{w['lo']:+.4f},{w['hi']:+.4f}]) "
                  f"and the cheapest of {len(winners)} that are, at "
                  f"{w['seconds']:.3f} s/sim vs {ref:.3f} ({speedup:.1f}x).")
        if args.out:
            json.dump(table, open(args.out, "w"), indent=1)
            print(f"\nwrote {args.out}")
        return
    if not winners:
        print(f"FREEZE: keep the incumbent {args.incumbent}. No candidate clears "
              f"delta >= {DELTA_MIN} with a CI excluding zero.")
    else:
        w = winners[0]
        print(f"FREEZE: adopt {w['candidate']} (delta {w['delta']:+.4f}, "
              f"CI [{w['lo']:+.4f},{w['hi']:+.4f}], p {w['p']:.2g}); "
              f"{len(winners)} candidate(s) cleared the bar.")
        if cal and val and w.get("delta_cal", 0) * w.get("delta_val", 0) < 0:
            print("  !! cal and val disagree in sign -- the margin is carried by one "
                  "subgroup. Do not freeze on this alone.")

    if args.out:
        json.dump(table, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
