#!/usr/bin/env python3
# @atlas: Whether the constructed release area straddles a drainage divide (D8 outlets, 150 m clustering).
"""Does the constructed release area straddle a drainage divide?

The release is built from the upper 20 % of the observed outline's elevation
range (release_band_frac = 0.25 in these runs, intersected with the 28-60 deg
slope band). Nothing in that construction knows about crests, so a release that
sits on a ridge top hands particles to two different valleys -- which is the
proposed upstream cause of the disconnected fragments measured in
frag_analysis.py.

Test: D8 steepest descent from every release cell; a cell's "outlet" is where
its path leaves the domain or hits a pit. Outlets are clustered spatially
(single-link, 150 m); release cells landing in different clusters descend into
different drainages.

Writes divide_report.json.
"""
import json, os
import numpy as np
from scipy import ndimage

D = os.path.dirname(os.path.abspath(__file__))
LINK = 150.0   # m: outlets closer than this are the same drainage
MINFRAC = 0.05  # a drainage counts if it takes >=5 % of the release cells

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def d8_outlets(dem, seeds, cell):
    """Follow steepest descent from each seed; return the outlet cell index."""
    h, w = dem.shape
    # precompute D8 receiver for every cell (-1 = pit / edge)
    recv = np.full(h * w, -1, np.int64)
    best = np.zeros((h, w), np.float32)
    for dy, dx in NB:
        sh = np.full((h, w), np.inf, np.float32)
        ys = slice(max(0, -dy), h - max(0, dy))
        xs = slice(max(0, -dx), w - max(0, dx))
        yd = slice(max(0, dy), h - max(0, -dy))
        xd = slice(max(0, dx), w - max(0, -dx))
        sh[ys, xs] = dem[yd, xd]
        drop = (dem - sh) / (cell * np.hypot(dy, dx))
        m = drop > best
        best[m] = drop[m]
        yy, xx = np.nonzero(m)
        recv[yy * w + xx] = (yy + dy) * w + (xx + dx)

    out = []
    for s in seeds:
        c, n = int(s), 0
        while n < 20000:
            r = recv[c]
            if r < 0:
                break
            c = r
            n += 1
            y, x = divmod(c, w)
            if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                break
        out.append(c)
    return np.array(out)


def cluster(pts, cell, link):
    """single-link clustering of outlet cells; pts = (y,x) array"""
    n = len(pts)
    lab = -np.ones(n, int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], k
        while stack:
            j = stack.pop()
            d = np.hypot(pts[:, 0] - pts[j, 0], pts[:, 1] - pts[j, 1]) * cell
            for t in np.nonzero((d <= link) & (lab < 0))[0]:
                lab[t] = k
                stack.append(int(t))
        k += 1
    return lab, k


frag = {r["name"]: r for r in json.load(open(f"{D}/frag_report.json"))}
out = []
for fn in sorted(os.listdir(f"{D}/dump100_all")):
    if not fn.endswith(".json"):
        continue
    d = json.load(open(f"{D}/dump100_all/{fn}"))
    w, h, cell = d["w"], d["h"], d["cell"]
    dem = np.array(d["dem"], np.float32).reshape(h, w)
    rel = np.array(d["release"], np.uint8).reshape(h, w).astype(bool)
    if rel.sum() == 0:
        continue
    seeds = np.nonzero(rel.ravel())[0]
    outl = d8_outlets(dem, seeds, cell)
    pts = np.stack(np.divmod(outl, w), 1).astype(float)
    lab, k = cluster(pts, cell, LINK)
    cnt = np.bincount(lab, minlength=k)
    frac = cnt / cnt.sum()
    major = int((frac >= MINFRAC).sum())
    # how far apart are the two biggest drainages?
    order = np.argsort(-cnt)
    sep = 0.0
    if k > 1:
        a = pts[lab == order[0]].mean(0)
        b = pts[lab == order[1]].mean(0)
        sep = float(np.hypot(*(a - b)) * cell)
    r = {"name": d["name"], "release_cells": int(rel.sum()),
         "n_drainages": int(k), "n_major_drainages": major,
         "largest_drainage_frac": float(frac.max()),
         "second_drainage_frac": float(np.sort(frac)[-2]) if k > 1 else 0.0,
         "outlet_separation_m": sep}
    f = frag.get(d["name"], {})
    r["mass_frac_outside_largest"] = f.get("c8", {}).get("mass_frac_outside_largest")
    r["area_frac_outside_largest"] = f.get("c8", {}).get("area_frac_outside_largest")
    r["n_components"] = f.get("c8", {}).get("n_components")
    r["omega"] = f.get("omega")
    r["sze"] = f.get("sze")
    out.append(r)

json.dump(out, open(f"{D}/divide_report.json", "w"), indent=1)

n = len(out)
maj = [r for r in out if r["n_major_drainages"] > 1]
print(f"cases {n}")
print(f"  release spans >1 major drainage (>=5% each): {len(maj)}/{n}")
sec = np.array([r["second_drainage_frac"] for r in out])
print(f"  second-drainage share: median {np.median(sec)*100:.1f}%  p90 {np.percentile(sec,90)*100:.1f}%  max {sec.max()*100:.1f}%")
a = np.array([r["area_frac_outside_largest"] for r in out])
print(f"  corr(second-drainage share, detached area frac) = {np.corrcoef(sec,a)[0,1]:+.3f}")
sp = np.array([r["n_major_drainages"] > 1 for r in out])
print(f"  detached area frac: split releases {a[sp].mean()*100:.2f}%  single {a[~sp].mean()*100:.2f}%")
om = np.array([r["omega"] for r in out])
print(f"  mean omega: split {om[sp].mean():+.3f}  single {om[~sp].mean():+.3f}")
print("\n  most-split releases:")
for r in sorted(out, key=lambda r: -r["second_drainage_frac"])[:12]:
    print(f"    {r['name']:>12s} 2nd={r['second_drainage_frac']*100:5.1f}%  ndrain={r['n_drainages']:3d}"
          f"  sep={r['outlet_separation_m']:6.0f}m  detach_area={r['area_frac_outside_largest']*100:5.2f}%"
          f"  ncomp={r['n_components']:4d}  omega={r['omega']:+.3f}")
print("\n  aval_10721:", json.dumps({k: v for k, v in
      next(r for r in out if r['name'] == 'aval_10721').items()}, default=str))
