#!/usr/bin/env python3
# @atlas: Shape of Ω_T over the (μ, ξ) grid — near-optimal area, aspect ratio, orientation, transferability, covariate correlations.
"""Is "the optimal parameter vector for this avalanche" a well-posed target?

Maps Omega_T over a 12 x 12 grid in (mu, log xi) for 12 events at each event's
per-event-fitted slab thickness, and asks of each surface: sharp optimum, broad
plateau, or extended ridge? Then asks of the sample: do the optima cluster,
scatter along a shared ridge, or scatter in structured fashion?

Coordinates are normalised so the two axes are comparable: mu on [0.05, 0.60]
-> [0,1], log10(xi) on [log10 200, log10 12000] -> [0,1]. An "aspect ratio" of
the near-optimal region is then dimensionless and an "orientation" of 0 deg
means elongated along mu (xi irrelevant), 90 deg along xi (mu irrelevant), and
+-45 deg is the classic mu/xi trade-off ridge.

Writes ident_report.json.
"""
import json, os
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
TOL = 0.02          # "equally good" band in Omega_T
grids = json.load(open(f"{D}/grid12.json"))
props = json.load(open(f"{D}/event_props_merged.json"))
pe = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}

MU = np.array(grids[0]["mus"])
XI = np.array(grids[0]["xis"])
un = lambda m: (m - 0.05) / (0.60 - 0.05)
ux = lambda x: (np.log10(x) - np.log10(200.0)) / (np.log10(12000.0) - np.log10(200.0))
UM, UX = un(MU), ux(XI)
GU, GV = np.meshgrid(UM, UX, indexing="ij")     # [i=mu, j=xi]

recs = []
for g in grids:
    om = np.array([[v if v is not None else np.nan for v in row] for row in g["omega"]])
    clip = np.array(g["clipped"])
    best = np.nanmax(om)
    bi, bj = np.unravel_index(np.nanargmax(om), om.shape)
    good = np.isfinite(om) & (om >= best - TOL)
    frac = good.mean()

    # shape of the near-optimal region in normalised coords
    pts = np.stack([GU[good], GV[good]], 1)
    if len(pts) >= 3:
        c = pts - pts.mean(0)
        cov = c.T @ c / len(pts)
        w, v = np.linalg.eigh(cov)
        lam = np.sqrt(np.maximum(w, 1e-12))
        ar = float(lam[1] / lam[0]) if lam[0] > 1e-6 else float("inf")
        ang = float(np.degrees(np.arctan2(v[1, 1], v[0, 1])))
        ang = ((ang + 90) % 180) - 90
    else:
        ar, ang = 1.0, 0.0

    # how far can each axis move on its own while staying within TOL of best?
    mu_span = float(UM[good.any(1)].max() - UM[good.any(1)].min()) if good.any() else 0.0
    xi_span = float(UX[good.any(0)].max() - UX[good.any(0)].min()) if good.any() else 0.0
    # sensitivity: Omega range along each axis through the optimum
    mu_range = float(np.nanmax(om[:, bj]) - np.nanmin(om[:, bj]))
    xi_range = float(np.nanmax(om[bi, :]) - np.nanmin(om[bi, :]))

    if frac >= 0.25:
        kind = "broad plateau"
    elif ar >= 3.0:
        kind = "extended ridge"
    elif frac <= 0.05:
        kind = "sharp optimum"
    else:
        kind = "compact optimum"

    p = props[g["name"]]
    recs.append(dict(
        name=g["name"], slab=g["slab"], best=float(best),
        best_mu=float(MU[bi]), best_xi=float(XI[bj]),
        best_u=float(UM[bi]), best_v=float(UX[bj]),
        frac_within_tol=float(frac), n_within_tol=int(good.sum()),
        aspect_ratio=ar, orientation_deg=ang, kind=kind,
        mu_span_norm=mu_span, xi_span_norm=xi_span,
        mu_ratio=float(MU[good.any(1)].max() / MU[good.any(1)].min()) if good.any() else 1.0,
        xi_ratio=float(XI[good.any(0)].max() / XI[good.any(0)].min()) if good.any() else 1.0,
        omega_range_along_mu=mu_range, omega_range_along_xi=xi_range,
        clipped_frac=float(clip.mean()), clipped_at_best=bool(clip[bi, bj]),
        sze=p["sze"], aval_shape=p["aval_shape"], area=p["area"], drop=p["drop"],
        elong=p["bbox_elong"], start=p["start"],
        nm_mu=pe[g["name"]]["best"]["friction_coefficient"],
        nm_xi=pe[g["name"]]["best"]["drag_coefficient"],
        nm_omega=pe[g["name"]]["omega_best"],
    ))

# ---------------------------------------------------- transferability test ---
# The decisive question: does a SINGLE (mu, xi) come close to the per-event
# bests? If yes, per-event optima are not carrying event-specific information.
OM = np.array([[[v if v is not None else np.nan for v in row] for row in g["omega"]]
               for g in grids])                      # [case, mu, xi]
mean_surface = np.nanmean(OM, axis=0)
gi, gj = np.unravel_index(np.nanargmax(mean_surface), mean_surface.shape)
per_event_best = np.nanmax(OM, axis=(1, 2))
shared_best = OM[:, gi, gj]
loss = per_event_best - shared_best

# leave-one-out: fit the shared point on 11, score the 12th
loo = []
for k in range(len(grids)):
    m = np.nanmean(np.delete(OM, k, axis=0), axis=0)
    a, b = np.unravel_index(np.nanargmax(m), m.shape)
    loo.append(float(per_event_best[k] - OM[k, a, b]))

# how well does another event's optimum transfer?
cross = np.zeros((len(grids), len(grids)))
for a, r in enumerate(recs):
    ia = int(np.argmin(np.abs(MU - r["best_mu"])))
    ja = int(np.argmin(np.abs(XI - r["best_xi"])))
    for b in range(len(grids)):
        cross[b, a] = OM[b, ia, ja]
transfer_loss = per_event_best[:, None] - cross      # [case, donor]
off = ~np.eye(len(grids), dtype=bool)

report = dict(
    tol=TOL, n_events=len(recs), events=recs,
    shared_best_mu=float(MU[gi]), shared_best_xi=float(XI[gj]),
    mean_per_event_best=float(per_event_best.mean()),
    mean_at_shared_best=float(shared_best.mean()),
    gap_per_event_minus_shared=float(loss.mean()),
    loss_quantiles=[float(np.percentile(loss, q)) for q in (50, 90, 100)],
    loo_mean_loss=float(np.mean(loo)),
    mean_transfer_loss=float(transfer_loss[off].mean()),
    median_transfer_loss=float(np.median(transfer_loss[off])),
)

# ------------------------------------------ do the 105 NM optima cluster? ----
allmu = np.array([r["best"]["friction_coefficient"] for r in pe.values()])
allxi = np.array([r["best"]["drag_coefficient"] for r in pe.values()])
u, v = un(allmu), ux(allxi)
report["nm105"] = dict(
    mu=[float(np.percentile(allmu, q)) for q in (10, 50, 90)],
    xi=[float(np.percentile(allxi, q)) for q in (10, 50, 90)],
    # spread relative to a uniform draw over the box (std of U[0,1] = 0.2887)
    u_std=float(u.std()), v_std=float(v.std()),
    corr_u_v=float(np.corrcoef(u, v)[0, 1]),
    frac_at_box_edge=float(np.mean((allmu <= 0.0501) | (allmu >= 0.5999) |
                                   (allxi <= 200.1) | (allxi >= 11999))),
)

# correlation of per-event optima with candidate covariates (all 105)
cov = {}
for key in ("sze", "aval_shape", "area", "drop", "bbox_elong", "start", "omega_best"):
    x = np.array([props[n][key] for n in pe])
    for lbl, y in (("mu", allmu), ("log_xi", np.log10(allxi)),
                   ("slab", np.array([r["best"]["slab_thickness"] for r in pe.values()]))):
        ok = np.isfinite(x) & np.isfinite(y)
        cov[f"{key} vs {lbl}"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
report["covariate_corr"] = cov

# --------------------------------- shape of the good region vs tolerance ----
# At 0.02 the near-optimal set is small; the question is whether it opens into a
# SHARED ridge as the tolerance loosens (degeneracy) or just grows isotropically.
multi = {}
for tol in (0.02, 0.05, 0.10, 0.20):
    ars, angs, fracs = [], [], []
    for g in grids:
        om = np.array([[v if v is not None else np.nan for v in row] for row in g["omega"]])
        good = np.isfinite(om) & (om >= np.nanmax(om) - tol)
        fracs.append(float(good.mean()))
        pts = np.stack([GU[good], GV[good]], 1)
        if len(pts) < 3:
            ars.append(1.0)
            angs.append(float("nan"))
            continue
        c = pts - pts.mean(0)
        w, v = np.linalg.eigh(c.T @ c / len(pts))
        lam = np.sqrt(np.maximum(w, 1e-12))
        ars.append(float(lam[1] / max(lam[0], 1e-6)))
        a = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        angs.append(float(((a + 90) % 180) - 90))
    aa = np.array([a for a in angs if np.isfinite(a) and abs(a) < 89.5])
    r2 = np.radians(2 * aa)
    mean_ang = float(np.degrees(0.5 * np.arctan2(np.sin(r2).mean(), np.cos(r2).mean())))
    conc = float(np.hypot(np.sin(r2).mean(), np.cos(r2).mean()))
    multi[str(tol)] = dict(
        median_frac=float(np.median(fracs)), median_ar=float(np.median(ars)),
        mean_orientation_deg=mean_ang, concentration=conc,
        per_event_ar=ars, per_event_orientation=angs, per_event_frac=fracs)
report["shape_vs_tolerance"] = multi

json.dump(report, open(f"{D}/ident_report.json", "w"), indent=1)

print("\n--- shape of the near-optimal region vs tolerance ---")
print(f"{'tol':>6} {'% of box':>9} {'aspect':>7} {'orientation':>12} {'alignment R':>12}")
for t, m in multi.items():
    print(f"{t:>6} {m['median_frac']*100:8.1f}% {m['median_ar']:7.1f} "
          f"{m['mean_orientation_deg']:+11.0f}° {m['concentration']:12.2f}")
print("  (0° = elongated along mu i.e. xi irrelevant is 90°; +45° = classic mu/xi trade-off)")

# --------------------------------------------------------------- printout ---
print(f"tolerance band: {TOL} Omega_T;  grid 12x12 over mu[0.05,0.60] x xi[200,12000] log\n")
print(f"{'event':>12s} {'slab':>5s} {'best':>6s} {'mu*':>5s} {'xi*':>6s} "
      f"{'%within':>7s} {'AR':>5s} {'orient':>7s} {'clip%':>5s}  kind")
for r in sorted(recs, key=lambda r: -r["frac_within_tol"]):
    print(f"{r['name']:>12s} {r['slab']:5.2f} {r['best']:+6.3f} {r['best_mu']:5.2f} "
          f"{r['best_xi']:6.0f} {r['frac_within_tol']*100:6.1f}% {r['aspect_ratio']:5.1f} "
          f"{r['orientation_deg']:+6.0f}d {r['clipped_frac']*100:4.0f}%  {r['kind']}")
f = np.array([r["frac_within_tol"] for r in recs])
a = np.array([r["aspect_ratio"] for r in recs])
o = np.array([r["orientation_deg"] for r in recs])
print(f"\nwithin-{TOL} area: median {np.median(f)*100:.1f}% of the (mu,xi) box, "
      f"range {f.min()*100:.1f}-{f.max()*100:.1f}%")
print(f"aspect ratio: median {np.median(a):.1f}   orientation: median {np.median(o):+.0f} deg, "
      f"spread {o.std():.0f} deg")
print(f"sensitivity: median Omega range along mu {np.median([r['omega_range_along_mu'] for r in recs]):.3f}, "
      f"along xi {np.median([r['omega_range_along_xi'] for r in recs]):.3f}")

print(f"\n--- transferability ---")
print(f"single best shared point on these 12: mu={MU[gi]:.2f} xi={XI[gj]:.0f}")
print(f"  mean per-event best        {per_event_best.mean():+.4f}")
print(f"  mean at shared best point  {shared_best.mean():+.4f}")
print(f"  gap                        {loss.mean():+.4f}  (median {np.median(loss):.3f}, "
      f"p90 {np.percentile(loss,90):.3f}, max {loss.max():.3f})")
print(f"  leave-one-out gap          {np.mean(loo):+.4f}")
print(f"  mean loss using ANOTHER event's optimum {transfer_loss[off].mean():+.4f} "
      f"(median {np.median(transfer_loss[off]):+.4f})")

print(f"\n--- do the 105 Nelder-Mead optima cluster? ---")
n = report["nm105"]
print(f"  mu  p10/50/90 {n['mu'][0]:.3f} / {n['mu'][1]:.3f} / {n['mu'][2]:.3f}")
print(f"  xi  p10/50/90 {n['xi'][0]:.0f} / {n['xi'][1]:.0f} / {n['xi'][2]:.0f}")
print(f"  normalised std: mu {n['u_std']:.3f}, log-xi {n['v_std']:.3f}  "
      f"(uniform over the box would be 0.289)")
print(f"  corr(mu*, log xi*) = {n['corr_u_v']:+.3f}   at box edge: {n['frac_at_box_edge']*100:.0f}%")
print(f"\n--- covariate correlations (n=105) ---")
for k, v_ in sorted(cov.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k:28s} r = {v_:+.3f}")
