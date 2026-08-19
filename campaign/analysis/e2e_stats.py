#!/usr/bin/env python3
# @atlas: Paired bootstrap + Wilcoxon over the end-to-end variants.
"""Paired statistics for the end-to-end variants. Writes e2e_stats.json."""
import json, os
import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["global_tuned", "global_meanopt", "pred_deploy_ridge", "pred_deploy_rf",
            "pred_all_rf", "pred_all_ridge", "oracle_mu_slab", "oracle_fixedxi",
            "oracle_full"]

om = {}
for v in VARIANTS:
    p = f"{D}/e2e_{v}.json"
    if not os.path.exists(p):
        print(f"  (skipping {v}: not run)")
        continue
    om[v] = {c["name"]: c["area"]["omega"] for c in json.load(open(p))["cases"]}

names = sorted(om["global_tuned"])
base = np.array([om["global_tuned"][n] for n in names])
rng = np.random.default_rng(0)
out = {}

# the do-nothing null is carried in every run record
out["null"] = dict(mean=float(json.load(open(f"{D}/e2e_global_tuned.json"))["mean_release_only"]),
                   delta=0.0, lo=0.0, hi=0.0, p=1.0)

for v in VARIANTS:
    if v not in om:
        continue
    x = np.array([om[v][n] for n in names])
    d = x - base
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 1.0 if v == "global_tuned" else float(stats.wilcoxon(x, base).pvalue)
    out[v] = dict(mean=float(x.mean()), delta=float(d.mean()), lo=float(lo),
                  hi=float(hi), p=p)

# the contrast that matters: outline-outcome features vs deployable-only
a = np.array([om["pred_all_ridge"][n] for n in names])
b = np.array([om["pred_deploy_rf"][n] for n in names])
d = a - b
bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(20000)])
out["contrast"] = dict(delta=float(d.mean()),
                       lo=float(np.percentile(bs, 2.5)),
                       hi=float(np.percentile(bs, 97.5)),
                       p=float(stats.wilcoxon(a, b).pvalue))

json.dump(out, open(f"{D}/e2e_stats.json", "w"), indent=1)
print(f"{'variant':>20} {'mean':>9} {'delta':>9} {'95% CI':>22} {'p':>10}")
for k, r in out.items():
    print(f"{k:>20} {r['mean'] if 'mean' in r else float('nan'):>+9.4f} "
          f"{r['delta']:>+9.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['p']:>10.2g}")
