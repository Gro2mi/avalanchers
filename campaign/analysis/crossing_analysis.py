#!/usr/bin/env python3
# @atlas: Share of simulated mass leaving the drainage the observed avalanche used.
"""How much simulated mass leaves the drainage the observed avalanche ran down?

Fragment counting (frag_analysis.py) turned out to measure the 0.1 m contour,
not the flow. This measures the physical question directly: label every cell of
the domain by the outlet its D8 descent path reaches, cluster outlets at 150 m,
call the cluster holding most of the OBSERVED outline the "intended" drainage,
and report the share of simulated mass outside it.

Writes crossing_report.json.
"""
import json, os
import numpy as np
from scipy import ndimage

D = os.path.dirname(os.path.abspath(__file__))
LINK = 150.0
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def outlet_of_all(recv, h, w):
    """Iterative pointer-doubling: outlet[c] = terminal cell of c's descent."""
    out = recv.copy()
    term = out < 0
    out[term] = np.nonzero(term)[0]
    for _ in range(24):                       # 2^24 steps, ample
        nxt = out[out]
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out


def cluster(pts, cell, link):
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


out = []
for fn in sorted(os.listdir(f"{D}/dump100_all")):
    if not fn.endswith(".json"):
        continue
    d = json.load(open(f"{D}/dump100_all/{fn}"))
    w, h, cell = d["w"], d["h"], d["cell"]
    dem = np.array(d["dem"], np.float32).reshape(h, w)
    ph = np.array(d["peak_h"], np.float32).reshape(h, w)
    ref = np.array(d["reference"], np.uint8).reshape(h, w).astype(bool)
    rel = np.array(d["release"], np.uint8).reshape(h, w).astype(bool)

    outl = outlet_of_all(receivers(dem, cell), h, w)
    uniq, inv = np.unique(outl, return_inverse=True)
    pts = np.stack(np.divmod(uniq, w), 1).astype(float)
    cl, k = cluster(pts, cell, LINK)
    basin = cl[inv].reshape(h, w)             # drainage id per cell

    # the drainage the OBSERVED avalanche used
    rb = np.bincount(basin[ref].ravel(), minlength=k)
    intended = int(np.argmax(rb))
    ref_purity = float(rb.max() / rb.sum())

    sim = ph > 0.1
    m_tot = float(ph[sim].sum())
    m_out = float(ph[sim & (basin != intended)].sum())
    a_out = float((sim & (basin != intended)).sum() / max(sim.sum(), 1))
    # release cells sitting in a different drainage than the observed one
    rel_out = float((rel & (basin != intended)).sum() / max(rel.sum(), 1))

    out.append(dict(
        name=d["name"], n_drainages_in_domain=int(k),
        ref_purity=ref_purity, intended=intended,
        mass_frac_outside_drainage=m_out / m_tot if m_tot else 0.0,
        area_frac_outside_drainage=a_out,
        release_frac_outside_drainage=rel_out,
        omega=d["result"]["area"]["omega"],
        clipped=d["result"]["clipped_at_edge"],
    ))

json.dump(out, open(f"{D}/crossing_report.json", "w"), indent=1)

n = len(out)
m = np.array([r["mass_frac_outside_drainage"] for r in out])
a = np.array([r["area_frac_outside_drainage"] for r in out])
rl = np.array([r["release_frac_outside_drainage"] for r in out])
p = np.array([r["ref_purity"] for r in out])
om = np.array([r["omega"] for r in out])
print(f"cases {n}")
print(f"  observed outline confined to one drainage: median purity {np.median(p)*100:.1f}%, "
      f"{int((p>0.95).sum())}/{n} above 95%")
print(f"  simulated MASS outside the observed drainage: median {np.median(m)*100:.2f}%  "
      f"p90 {np.percentile(m,90)*100:.2f}%  max {m.max()*100:.1f}%")
print(f"  simulated AREA outside it:                   median {np.median(a)*100:.2f}%  "
      f"p90 {np.percentile(a,90)*100:.2f}%  max {a.max()*100:.1f}%")
print(f"  RELEASE cells outside it:                    median {np.median(rl)*100:.2f}%  "
      f"p90 {np.percentile(rl,90)*100:.2f}%  max {rl.max()*100:.1f}%")
for t in (0.01, 0.05, 0.10, 0.20):
    print(f"    > {t*100:4.0f}% of mass in the wrong drainage: {int((m>t).sum()):3d}/{n}")
print(f"  corr(mass outside drainage, omega) = {np.corrcoef(m, om)[0,1]:+.3f}")
print("\n  worst 12:")
for r in sorted(out, key=lambda r: -r["mass_frac_outside_drainage"])[:12]:
    print(f"    {r['name']:>12s} mass {r['mass_frac_outside_drainage']*100:5.1f}%  "
          f"area {r['area_frac_outside_drainage']*100:5.1f}%  "
          f"release {r['release_frac_outside_drainage']*100:5.1f}%  "
          f"ref_purity {r['ref_purity']*100:5.1f}%  omega {r['omega']:+.3f}")
r = next(x for x in out if x["name"] == "aval_10721")
print("\n  aval_10721:", json.dumps(r))
