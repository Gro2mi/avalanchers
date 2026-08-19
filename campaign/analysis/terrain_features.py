#!/usr/bin/env python3
# @atlas: Derives 16 DEM/release features and their raw correlations with the fitted optima.
"""Terrain features that DO exist at prediction time.

Every feature that predicted anything in the pilot was measured off the mapped
outline of an avalanche that had already run. This derives an alternative set
from the DEM and the release area alone -- which, under the settled
"runout given release" scope, is exactly what you have.

The important one is `path_drop_potential`: the elevation a descent path from
the release can still lose before leaving the domain. It is the terrain-side
analogue of the observed `drop`, which was one of the strongest outcome
features -- so if drop was carrying real signal rather than circularity, this
should inherit it.

Writes terrain_feats.json.
"""
import json, os
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def slope_aspect(dem, cell):
    dy, dx = np.gradient(dem, cell)
    return np.degrees(np.arctan(np.hypot(dx, dy))), dx, dy


def curvature(dem, cell):
    """Total (Laplacian) curvature; negative = convex ridge, positive = concave."""
    return (np.gradient(np.gradient(dem, cell, axis=0), cell, axis=0)
            + np.gradient(np.gradient(dem, cell, axis=1), cell, axis=1))


def receivers(dem, cell):
    h, w = dem.shape
    recv = np.full(h * w, -1, np.int64)
    best = np.zeros((h, w), np.float32)
    for dy, dx in NB:
        sh = np.full((h, w), np.inf, np.float32)
        ys = slice(max(0, -dy), h - max(0, dy)); xs = slice(max(0, -dx), w - max(0, dx))
        yd = slice(max(0, dy), h - max(0, -dy)); xd = slice(max(0, dx), w - max(0, -dx))
        sh[ys, xs] = dem[yd, xd]
        drop = (dem - sh) / (cell * np.hypot(dy, dx))
        m = drop > best
        best[m] = drop[m]
        yy, xx = np.nonzero(m)
        recv[yy * w + xx] = (yy + dy) * w + (xx + dx)
    return recv


out = {}
for fn in sorted(os.listdir(f"{D}/dump100_all")):
    if not fn.endswith(".json"):
        continue
    d = json.load(open(f"{D}/dump100_all/{fn}"))
    w, h, cell = d["w"], d["h"], d["cell"]
    dem = np.array(d["dem"], np.float32).reshape(h, w)
    rel = np.array(d["release"], np.uint8).reshape(h, w).astype(bool)
    if rel.sum() == 0:
        continue
    ar, ac = d["apex"]
    slope, dx, dy = slope_aspect(dem, cell)
    curv = curvature(dem, cell)

    # --- release-zone terrain -------------------------------------------------
    f = {
        "rel_area_m2": float(rel.sum() * cell * cell),
        "rel_slope_mean": float(slope[rel].mean()),
        "rel_slope_std": float(slope[rel].std()),
        "rel_curv_mean": float(curv[rel].mean()),
        "rel_curv_std": float(curv[rel].std()),
        "rel_elev_range": float(dem[rel].max() - dem[rel].min()),
        "apex_elev": float(dem[ar, ac]),
    }

    # --- the descent path from the release apex ------------------------------
    recv = receivers(dem, cell)
    c, path = ar * w + ac, []
    for _ in range(20000):
        r = recv[c]
        if r < 0:
            break
        c = r
        path.append(c)
        y, x = divmod(c, w)
        if y in (0, h - 1) or x in (0, w - 1):
            break
    if path:
        py, px = np.divmod(np.array(path), w)
        pz = dem[py, px]
        f["path_drop_potential"] = float(dem[ar, ac] - pz.min())
        f["path_len_m"] = float(len(path) * cell)
        f["path_slope_mean"] = float(slope[py, px].mean())
        # confinement: cross-path curvature. Positive (concave) = channelised.
        f["path_curv_mean"] = float(curv[py, px].mean())
        f["path_curv_p25"] = float(np.percentile(curv[py, px], 25))
        # straight-line vs along-path distance: how sinuous the descent is
        d0 = np.hypot(py[-1] - ar, px[-1] - ac) * cell
        f["path_sinuosity"] = float(len(path) * cell / max(d0, 1.0))
        # mean gradient of the whole descent, a terrain-only alpha-angle proxy
        f["path_alpha_deg"] = float(np.degrees(np.arctan2(
            dem[ar, ac] - pz.min(), max(d0, 1.0))))
    else:
        for k in ("path_drop_potential", "path_len_m", "path_slope_mean",
                  "path_curv_mean", "path_curv_p25", "path_sinuosity",
                  "path_alpha_deg"):
            f[k] = 0.0

    # --- channelisation: contributing area along the path --------------------
    # cheap flow accumulation by descending order of elevation
    order = np.argsort(-dem.ravel())
    acc = np.ones(h * w, np.float64)
    for c0 in order:
        r = recv[c0]
        if r >= 0:
            acc[r] += acc[c0]
    if path:
        f["path_logacc_mean"] = float(np.log10(acc[np.array(path)]).mean())
    else:
        f["path_logacc_mean"] = 0.0
    f["rel_logacc_mean"] = float(np.log10(acc[rel.ravel()]).mean())

    out[d["name"]] = f

json.dump(out, open(f"{D}/terrain_feats.json", "w"), indent=1)
keys = sorted(next(iter(out.values())))
print(f"{len(out)} cases, {len(keys)} terrain features\n")
pe = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}
names = [n for n in sorted(out) if n in pe]
mu = np.array([pe[n]["best"]["friction_coefficient"] for n in names])
sl = np.array([pe[n]["best"]["slab_thickness"] for n in names])
print(f"{'feature':>22} {'r vs mu*':>10} {'r vs slab*':>11}")
rows = []
for k in keys:
    v = np.array([out[n][k] for n in names], float)
    if v.std() == 0:
        continue
    rm = np.corrcoef(v, mu)[0, 1]
    rs = np.corrcoef(v, sl)[0, 1]
    rows.append((max(abs(rm), abs(rs)), k, rm, rs))
for _, k, rm, rs in sorted(rows, reverse=True):
    star = "  *" if max(abs(rm), abs(rs)) > 0.19 else ""
    print(f"{k:>22} {rm:>+10.3f} {rs:>+11.3f}{star}")
print("\n* = |r| > 0.19, significant at the 5% level for n = 105")
