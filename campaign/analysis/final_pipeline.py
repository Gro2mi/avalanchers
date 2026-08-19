#!/usr/bin/env python3
# @atlas: The corrected architecture: ξ fixed → μ/slab calibrated at that ξ → regressed on deployable features → re-simulated.
"""The architecture as it should actually be run, end to end.

Everything learned so far, applied at once:
  1. fix xi globally (it is not identified), and calibrate mu + slab CONDITIONAL
     on that fixed xi -- not reuse a mu that was co-fitted with xi free;
  2. regress those targets on features that exist at prediction time, derived
     from the DEM and the release area rather than from the mapped outline;
  3. score by re-simulating with the out-of-fold predicted parameters.

Strictly deployable feature set. Path features are truncated at a fixed 200/300 m
from the release apex so they cannot depend on the domain, which is itself sized
from the observed outline -- the contamination that inflated the first pass.

Writes final_pipeline.json + params_final_*.json.
"""
import json, os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

D = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(23)
props = json.load(open(f"{D}/event_props_merged.json"))
terr = json.load(open(f"{D}/terrain_feats.json"))
trunc = json.load(open(f"{D}/terrain_trunc.json"))
fx = {r["name"]: r for r in json.load(open(f"{D}/perevent_fixedxi.json"))}
fr = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}
names = sorted(n for n in fx if n in terr and n in trunc)
XI_FIXED = float(json.load(open(f"{D}/search100_uniform.json"))[0]["params"]["drag_coefficient"])
ASPECT_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135,
              "S": 180, "SW": 225, "W": 270, "NW": 315}

# Deployable = computable from the DEM plus a release polygon. Path descriptors
# are capped at 200/300 m of descent so no feature can see how far the domain --
# and therefore the observed avalanche -- extends.
DEPLOY = ["alpha200", "slope200", "alpha300", "slope300",
          "rel_slope_mean", "rel_slope_std", "rel_curv_mean", "rel_curv_std",
          "rel_elev_range", "rel_area_m2", "rel_logacc_mean", "apex_elev",
          "start", "frac_wdh", "asp_sin", "asp_cos"]
OUTCOME = ["sze", "area", "drop", "bbox_elong", "w", "h"]


def row(n):
    p, t, u = props[n], terr[n], trunc[n]
    a = np.radians(ASPECT_DEG.get(p["aspect"], 0))
    d = {k: u[k] for k in ("alpha200", "slope200", "alpha300", "slope300")}
    d.update({k: t[k] for k in ("rel_slope_mean", "rel_slope_std", "rel_curv_mean",
                                "rel_curv_std", "rel_elev_range", "rel_area_m2",
                                "rel_logacc_mean", "apex_elev")})
    d.update({"start": p["start"], "frac_wdh": p["frac_wdh"],
              "asp_sin": np.sin(a), "asp_cos": np.cos(a),
              "sze": p["sze"], "area": p["area"], "drop": p["drop"],
              "bbox_elong": p["bbox_elong"], "w": p["w"], "h": p["h"]})
    return d


rows = [row(n) for n in names]
TARGETS = {
    "mu (xi fixed)": np.array([fx[n]["best"]["friction_coefficient"] for n in names]),
    "slab (xi fixed)": np.array([fx[n]["best"]["slab_thickness"] for n in names]),
    "mu (xi free)": np.array([fr[n]["best"]["friction_coefficient"] for n in names]),
    "slab (xi free)": np.array([fr[n]["best"]["slab_thickness"] for n in names]),
}
SETS = {"deployable (terrain + release)": DEPLOY,
        "outline outcomes": OUTCOME,
        "alpha200 + slope200 only": ["alpha200", "slope200"]}


def mk(k):
    return (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))
            if k == "ridge" else
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  max_features=0.5, random_state=0))


def oof(X, y, kind, seed):
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = mk(kind)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

rep = {}
print(f"{'target':>16} {'feature set':>32} {'model':>7} {'OOF R2 (95%)':>26} {'perm p':>8}")
for tname, y in TARGETS.items():
    for sname, feats in SETS.items():
        X = np.array([[r[f] for f in feats] for r in rows], float)
        for kind in ("ridge", "rf"):
            obs = np.array([r2(y, oof(X, y, kind, s)) for s in range(150)])
            null = np.array([r2(yp, oof(X, yp, kind, s)) for s in range(300)
                             for yp in [RNG.permutation(y)]])
            lo, hi = np.percentile(obs, [2.5, 97.5])
            p = float((null >= np.median(obs)).mean())
            rep[f"{tname}|{sname}|{kind}"] = dict(median=float(np.median(obs)),
                                                  lo=float(lo), hi=float(hi), p=p)
            print(f"{tname:>16} {sname:>32} {kind:>7} {np.median(obs):>+9.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] {p:>8.3f}")
    print()

json.dump(rep, open(f"{D}/final_pipeline.json", "w"), indent=1)

# ---- parameter files for the end-to-end run --------------------------------
for kind in ("ridge", "rf"):
    X = np.array([[r[f] for f in DEPLOY] for r in rows], float)
    mu = np.clip(oof(X, TARGETS["mu (xi fixed)"], kind, 0), 0.05, 0.60)
    sl = np.clip(oof(X, TARGETS["slab (xi fixed)"], kind, 0), 0.10, 2.00)
    json.dump({n: {"mu": float(mu[i]), "xi": XI_FIXED, "slab": float(sl[i])}
               for i, n in enumerate(names)},
              open(f"{D}/params_final_deploy_{kind}.json", "w"), indent=1)
    print(f"wrote params_final_deploy_{kind}.json  mu {mu.mean():.3f}±{mu.std():.3f}  "
          f"slab {sl.mean():.3f}±{sl.std():.3f}")

# the global constant that the fixed-xi calibration implies, as the baseline
json.dump({n: {"mu": float(TARGETS["mu (xi fixed)"].mean()), "xi": XI_FIXED,
               "slab": float(TARGETS["slab (xi fixed)"].mean())} for n in names},
          open(f"{D}/params_final_constant.json", "w"), indent=1)
print("wrote params_final_constant.json")
