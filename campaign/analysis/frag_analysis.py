#!/usr/bin/env python3
# @atlas: Connected components of the simulated footprint vs the flow threshold.
"""Prevalence of disconnected simulated footprints across all 105 default-parameter
cases (the "ridge splitting" question raised by aval_10721).

A simulated footprint is the set of cells whose peak flow thickness exceeds the
harness threshold (0.1 m). Physically one release should produce one connected
deposit; components detached from the main body mean particles left the intended
drainage. Mass is proxied by sum(peak_h) * cell area -- peak thickness, not final
deposit, but it is what the harness itself scores on.

Reads only scratchpad dumps; writes frag_report.json.
"""
import json, os, sys
import numpy as np
from scipy import ndimage

D = os.path.dirname(os.path.abspath(__file__))
THR = 0.1
STRUCT8 = np.ones((3, 3), bool)
STRUCT4 = ndimage.generate_binary_structure(2, 1)

props = json.load(open(f"{D}/event_props_merged.json"))
out = []
for fn in sorted(os.listdir(f"{D}/dump100_all")):
    if not fn.endswith(".json"):
        continue
    d = json.load(open(f"{D}/dump100_all/{fn}"))
    w, h, cell = d["w"], d["h"], d["cell"]
    ph = np.array(d["peak_h"], dtype=np.float32).reshape(h, w)
    ref = np.array(d["reference"], dtype=np.uint8).reshape(h, w).astype(bool)
    rel = np.array(d["release"], dtype=np.uint8).reshape(h, w).astype(bool)
    mask = ph > THR
    if not mask.any():
        continue
    ca = cell * cell

    rec = {"name": d["name"], "cell": cell, "w": w, "h": h,
           "sim_cells": int(mask.sum()), "ref_cells": int(ref.sum()),
           "release_cells": int(rel.sum()),
           "omega": d["result"]["area"]["omega"],
           "clipped": d["result"]["clipped_at_edge"]}

    for tag, st in (("c8", STRUCT8), ("c4", STRUCT4)):
        lab, n = ndimage.label(mask, structure=st)
        if n == 0:
            continue
        # mass per component
        mass = ndimage.sum(ph, lab, index=np.arange(1, n + 1)) * ca
        area = np.bincount(lab.ravel())[1:] * ca
        big = int(np.argmax(mass)) + 1
        tot = mass.sum()
        # does a component touch the release area?
        rel_lab = set(np.unique(lab[rel & mask])) - {0}
        # fragments that overlap the observed outline at all
        ref_lab = set(np.unique(lab[ref & mask])) - {0}
        rec[tag] = {
            "n_components": int(n),
            "mass_frac_outside_largest": float(1.0 - mass[big - 1] / tot) if tot > 0 else 0.0,
            "area_frac_outside_largest": float(1.0 - area[big - 1] / area.sum()),
            "largest_mass_m4": float(mass[big - 1]),
            "n_components_ge_10_cells": int((area / ca >= 10).sum()),
            "n_components_ge_1pct_mass": int((mass / tot >= 0.01).sum()) if tot > 0 else 0,
            "largest_touches_release": bool(big in rel_lab),
            "n_frag_outside_ref": int(sum(1 for i in range(1, n + 1)
                                          if i != big and i not in ref_lab)),
            "mass_frac_frags_outside_ref": float(
                sum(mass[i - 1] for i in range(1, n + 1)
                    if i != big and i not in ref_lab) / tot) if tot > 0 else 0.0,
            "frag_area_quartiles_cells": [float(x) for x in np.percentile(
                np.delete(area, big - 1) / ca, [50, 90, 100])] if n > 1 else [0, 0, 0],
        }
    p = props.get(d["name"], {})
    for k in ("sze", "aval_shape", "area", "drop", "bbox_elong", "start"):
        rec[k] = p.get(k)
    out.append(rec)

json.dump(out, open(f"{D}/frag_report.json", "w"), indent=1)

# ------------------------------------------------------------------ summary
n = len(out)
c8 = [r["c8"] for r in out]
print(f"cases: {n}")
for lbl, key in (("any 2nd component", "n_components"),):
    print(f"  {lbl:34s} {sum(1 for c in c8 if c[key] > 1):3d}/{n}")
print(f"  {'>=2 components of >=10 cells':34s} {sum(1 for c in c8 if c['n_components_ge_10_cells'] > 1):3d}/{n}")
print(f"  {'>=2 components with >=1% of mass':34s} {sum(1 for c in c8 if c['n_components_ge_1pct_mass'] > 1):3d}/{n}")
mf = np.array([c["mass_frac_outside_largest"] for c in c8])
print(f"  mass outside largest: median {np.median(mf)*100:.2f}%  p90 {np.percentile(mf,90)*100:.2f}%  max {mf.max()*100:.2f}%")
for t in (0.001, 0.01, 0.05, 0.10):
    print(f"    > {t*100:5.1f}% of mass detached: {int((mf > t).sum()):3d}/{n}")
nc = np.array([c["n_components"] for c in c8])
print(f"  n components: median {np.median(nc):.0f}  p90 {np.percentile(nc,90):.0f}  max {nc.max():.0f}")
print("\n  worst 12 by detached mass fraction:")
for r in sorted(out, key=lambda r: -r["c8"]["mass_frac_outside_largest"])[:12]:
    c = r["c8"]
    print(f"    {r['name']:>12s} {c['mass_frac_outside_largest']*100:6.2f}%  ncomp={c['n_components']:4d}"
          f"  >=10cell={c['n_components_ge_10_cells']:3d}  omega={r['omega']:+.3f}"
          f"  sze={r['sze']} clip={int(r['clipped'])}")
# correlation with omega
om = np.array([r["omega"] for r in out])
print(f"\n  corr(detached mass frac, omega) = {np.corrcoef(mf, om)[0,1]:+.3f}")
