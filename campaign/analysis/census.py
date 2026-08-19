#!/usr/bin/env python3
# @atlas: Composition of the 18,737-polygon mapping and the filter funnel down to the candidate pool.
"""Census of the full SPOT6 24-Jan-2018 mapping (18,737 polygons) and the funnel
that reduces it to the 105-avalanche study sample.

Answers "why not size 1, 2 or 5?" with numbers rather than assertion: what each
filter costs, and whether the excluded size classes are resolvable at all
against a 5 m DEM (the simulation grid) and 1.5 m SPOT6 imagery (the mapping
source). Writes census.json.
"""
import json, math, os
from collections import Counter
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
CELL = 5.0          # simulation grid, m
SPOT6 = 1.5         # source imagery ground sample distance, m

poly = json.load(open(f"{D}/poly_index.json"))
import shapefile
r = shapefile.Reader(f"{D}/outlines/outlines2018.shp")
for p, rec in zip(poly, r.iterRecords()):
    p['aspect'] = rec['aspect']
    p['trg'] = rec['trg_typ']
    p['typ_hum'] = rec['typ_hum']
    p['typ_fractu'] = rec['typ_fractu']

# --- neighbour distance, same 500 m grid method as select_sample.py -----------
CELLW = 500.0
grid = {}
for p in poly:
    for gx in range(int(p['xmin'] // CELLW) - 1, int(p['xmax'] // CELLW) + 2):
        for gy in range(int(p['ymin'] // CELLW) - 1, int(p['ymax'] // CELLW) + 2):
            grid.setdefault((gx, gy), []).append(p['idx'])
def gap(a, b):
    dx = max(b['xmin'] - a['xmax'], a['xmin'] - b['xmax'], 0.0)
    dy = max(b['ymin'] - a['ymax'], a['ymin'] - b['ymax'], 0.0)
    return math.hypot(dx, dy)
by = {p['idx']: p for p in poly}
for p in poly:
    best, seen = 1e9, set()
    for gx in range(int(p['xmin'] // CELLW) - 1, int(p['xmax'] // CELLW) + 2):
        for gy in range(int(p['ymin'] // CELLW) - 1, int(p['ymax'] // CELLW) + 2):
            for j in grid.get((gx, gy), ()):
                if j == p['idx'] or j in seen:
                    continue
                seen.add(j)
                best = min(best, gap(p, by[j]))
    p['nbr_dist'] = round(min(best, 1e9), 1)

for p in poly:
    p['drop'] = p['start'] - p['dpo']
    p['bboxmax'] = max(p['xmax'] - p['xmin'], p['ymax'] - p['ymin'])
    p['cells5'] = p['area'] / (CELL * CELL)

out = {"n_total": len(poly)}

# ------------------------------------------------------------- breakdowns ---
def tab(key, order=None):
    c = Counter(p[key] for p in poly)
    keys = order or sorted(c, key=lambda k: (isinstance(k, str), k))
    return [{"value": str(k), "n": c[k], "pct": 100 * c[k] / len(poly)} for k in keys if k in c]

out["by_size"] = tab("sze")
out["by_type"] = tab("typ")
out["by_trigger"] = tab("trg")
out["by_quality"] = tab("aval_shape")
out["by_human"] = tab("typ_hum")

# size x quality cross-tab
sizes = sorted({p['sze'] for p in poly})
quals = sorted({p['aval_shape'] for p in poly})
out["size_by_quality"] = {
    str(s): {str(q): sum(1 for p in poly if p['sze'] == s and p['aval_shape'] == q)
             for q in quals} for s in sizes}

# ----------------------------------------------- per size class resolution ---
def stats(sel):
    if not sel:
        return None
    a = np.array([p['area'] for p in sel])
    n = np.array([p['npts'] for p in sel])
    d = np.array([p['drop'] for p in sel])
    c = a / (CELL * CELL)
    # equivalent square side, and how many 5 m cells across that is
    side = np.sqrt(a)
    return {
        "n": len(sel),
        "area_ha": [float(np.percentile(a, q) / 1e4) for q in (10, 50, 90)],
        "cells_5m": [float(np.percentile(c, q)) for q in (10, 50, 90)],
        "side_m": [float(np.percentile(side, q)) for q in (10, 50, 90)],
        "cells_across": [float(np.percentile(side / CELL, q)) for q in (10, 50, 90)],
        "npts": [float(np.percentile(n, q)) for q in (10, 50, 90)],
        "vertex_spacing_m": float(np.median(4 * side / n)),
        "drop_m": [float(np.percentile(d, q)) for q in (10, 50, 90)],
        "frac_under_100_cells": float((c < 100).mean()),
        "frac_under_20_across": float((side / CELL < 20).mean()),
        "frac_drop_under_150": float((d < 150).mean()),
    }

out["size_resolution"] = {str(s): stats([p for p in poly if p['sze'] == s]) for s in sizes}

# ------------------------------------------------------------------ funnel ---
MIN_AREA, MAX_AREA, MAX_BBOX, MIN_DROP, MIN_START, MIN_NBR = \
    20000.0, 600000.0, 2000.0, 150.0, 1550.0, 25.0
STAGES = [
    ("all mapped polygons", lambda p: True),
    ("typ = SLAB", lambda p: p['typ'] == 'SLAB'),
    ("trg_typ = NATURAL", lambda p: p['trg'] == 'NATURAL'),
    ("single ring (nparts = 1)", lambda p: p['nparts'] == 1),
    ("size class 2-5", lambda p: 2 <= p['sze'] <= 5),
    ("area 2-60 ha", lambda p: MIN_AREA <= p['area'] <= MAX_AREA),
    ("bbox <= 2000 m", lambda p: p['bboxmax'] <= MAX_BBOX),
    ("drop >= 150 m", lambda p: p['drop'] >= MIN_DROP),
    ("start zone >= 1550 m", lambda p: p['start'] >= MIN_START),
    ("no neighbour within 25 m", lambda p: p['nbr_dist'] >= MIN_NBR),
]
funnel, cur = [], list(poly)
for label, fn in STAGES:
    before = len(cur)
    cur = [p for p in cur if fn(p)]
    # marginal cost: how many would this filter alone remove from the full set?
    alone = sum(1 for p in poly if not fn(p))
    funnel.append({"stage": label, "remaining": len(cur),
                   "removed": before - len(cur),
                   "removed_pct_of_prev": 100 * (before - len(cur)) / before if before else 0,
                   "would_remove_alone": alone})
out["funnel"] = funnel
pool = cur
out["n_pool"] = len(pool)
out["pool_by_size"] = [{"value": str(s), "n": sum(1 for p in pool if p['sze'] == s)} for s in sizes]
out["pool_by_quality"] = [{"value": str(q), "n": sum(1 for p in pool if p['aval_shape'] == q)} for q in quals]

# --- leave-one-out: relax each size bound and see what the pool would gain ----
def pool_with(szmin, szmax, minarea=MIN_AREA, mindrop=MIN_DROP):
    return sum(1 for p in poly if p['typ'] == 'SLAB' and p['trg'] == 'NATURAL'
               and p['nparts'] == 1 and szmin <= p['sze'] <= szmax
               and minarea <= p['area'] <= MAX_AREA and p['bboxmax'] <= MAX_BBOX
               and p['drop'] >= mindrop and p['start'] >= MIN_START
               and p['nbr_dist'] >= MIN_NBR)
out["counterfactual"] = {
    "pool as built (size 2-5)": pool_with(2, 5),
    "include size 1": pool_with(1, 5),
    "include size 1, drop min area to 0.5 ha": pool_with(1, 5, 5000.0),
    "include size 1, no area or drop floor": pool_with(1, 5, 0.0, 0.0),
    "size 3-4 only": pool_with(3, 4),
    "size 5 only": pool_with(5, 5),
}

# what the size-1/2 population looks like BEFORE the area floor
for s in sizes:
    sel = [p for p in poly if p['sze'] == s and p['typ'] == 'SLAB' and p['trg'] == 'NATURAL']
    if sel:
        out.setdefault("natural_slab_by_size", {})[str(s)] = {
            "n": len(sel),
            "median_area_ha": float(np.median([p['area'] for p in sel]) / 1e4),
            "pass_area_floor": sum(1 for p in sel if p['area'] >= MIN_AREA),
            "pass_drop_floor": sum(1 for p in sel if p['drop'] >= MIN_DROP),
            "pass_both": sum(1 for p in sel if p['area'] >= MIN_AREA and p['drop'] >= MIN_DROP),
        }

# realised sample
cases = json.load(open(f"{D}/cases100_wx.json"))
out["n_sample"] = len(cases)
out["sample_by_size"] = dict(Counter(str(c['sze']) for c in cases))
out["sample_by_quality"] = dict(Counter(str(c['aval_shape']) for c in cases))

json.dump(out, open(f"{D}/census.json", "w"), indent=1)

# --------------------------------------------------------------- printout ---
print(f"total polygons {out['n_total']}")
print("\nby size class:")
for r_ in out["by_size"]:
    s = out["size_resolution"][r_["value"]]
    print(f"  sze {r_['value']}: {r_['n']:6d} ({r_['pct']:5.2f}%)  median area {s['area_ha'][1]*1e4:8.0f} m2"
          f" = {s['cells_5m'][1]:7.0f} cells  {s['cells_across'][1]:5.1f} cells across"
          f"  npts med {s['npts'][1]:5.0f}  vertex spacing {s['vertex_spacing_m']:.1f} m"
          f"  <100 cells {s['frac_under_100_cells']*100:4.0f}%")
print("\nby type:      ", {r_['value']: r_['n'] for r_ in out['by_type']})
print("by trigger:   ", {r_['value']: r_['n'] for r_ in out['by_trigger']})
print("by quality:   ", {r_['value']: r_['n'] for r_ in out['by_quality']})
print("\nfunnel:")
for f in out["funnel"]:
    print(f"  {f['stage']:28s} -> {f['remaining']:6d}  (-{f['removed']:5d}, "
          f"{f['removed_pct_of_prev']:5.1f}% of previous; alone would cut {f['would_remove_alone']})")
print("\npool by size:", {r_['value']: r_['n'] for r_ in out['pool_by_size']})
print("pool by quality:", {r_['value']: r_['n'] for r_ in out['pool_by_quality']})
print("\ncounterfactual pool sizes:")
for k, v in out["counterfactual"].items():
    print(f"  {k:44s} {v}")
print("\nnatural slab by size, floors:")
for k, v in out["natural_slab_by_size"].items():
    print(f"  sze {k}: n={v['n']:6d} med {v['median_area_ha']:5.2f} ha  pass area {v['pass_area_floor']:5d}"
          f"  pass drop {v['pass_drop_floor']:5d}  pass both {v['pass_both']:5d}")
print("\nsample by size:", out["sample_by_size"], " by quality:", out["sample_by_quality"])
