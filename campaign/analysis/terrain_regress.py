#!/usr/bin/env python3
# @atlas: Cross-validated skill of terrain features vs outline features.
"""Do terrain features -- the ones that exist at prediction time -- work?

The pilot's conclusion was "these features don't work", where "these" meant
geometric descriptors of the mapped outline. This tests the alternative:
features derived from the DEM and the release area alone.

Same protocol as regress_pilot.py so the numbers are comparable: out-of-fold
R^2 against the global mean, 200 repeats of 5-fold CV for the interval, 400
target shuffles for the permutation p. Then the end-to-end test: predicted
parameters simulated on every event.

Writes terrain_regress.json and params_pred_terrain_*.json.
"""
import json, os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

D = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(11)
props = json.load(open(f"{D}/event_props_merged.json"))
terr = json.load(open(f"{D}/terrain_feats.json"))
pe = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}
names = sorted(n for n in pe if n in terr)
tuned = json.load(open(f"{D}/search100_uniform.json"))[0]["params"]
XI_FIXED = float(tuned["drag_coefficient"])

ASPECT_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135,
              "S": 180, "SW": 225, "W": 270, "NW": 315}
TERRAIN = ["path_alpha_deg", "path_slope_mean", "path_drop_potential", "path_len_m",
           "path_curv_mean", "path_curv_p25", "path_sinuosity", "path_logacc_mean",
           "rel_slope_mean", "rel_slope_std", "rel_curv_mean", "rel_curv_std",
           "rel_elev_range", "rel_area_m2", "rel_logacc_mean", "apex_elev"]
RELEASE = ["start", "frac_wdh", "asp_sin", "asp_cos"]
OUTCOME = ["sze", "area", "drop", "bbox_elong", "w", "h"]


def row(n):
    p, t = props[n], terr[n]
    a = np.radians(ASPECT_DEG.get(p["aspect"], 0))
    d = {"start": p["start"], "frac_wdh": p["frac_wdh"],
         "asp_sin": np.sin(a), "asp_cos": np.cos(a), "sze": p["sze"],
         "area": p["area"], "drop": p["drop"], "bbox_elong": p["bbox_elong"],
         "w": p["w"], "h": p["h"]}
    d.update({k: t[k] for k in TERRAIN})
    return d


rows = [row(n) for n in names]
Y = {"mu": np.array([pe[n]["best"]["friction_coefficient"] for n in names]),
     "slab": np.array([pe[n]["best"]["slab_thickness"] for n in names]),
     "log_xi": np.array([np.log10(pe[n]["best"]["drag_coefficient"]) for n in names])}

SETS = {
    "outline only (the pilot's set)": RELEASE + OUTCOME,
    "terrain + release (deployable)": TERRAIN + RELEASE,
    "path_alpha_deg alone": ["path_alpha_deg"],
    "everything": TERRAIN + RELEASE + OUTCOME,
}


def mk(kind):
    return (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))
            if kind == "ridge" else
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  max_features=0.5, random_state=0))


def oof(X, y, kind, seed):
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = mk(kind)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


report = {}
print(f"{'target':>8} {'feature set':>32} {'model':>7} {'OOF R2 (95%)':>26} {'perm p':>8}")
for tgt in ("mu", "slab", "log_xi"):
    for sname, feats in SETS.items():
        X = np.array([[r[f] for f in feats] for r in rows], float)
        for kind in ("ridge", "rf"):
            obs = np.array([r2(Y[tgt], oof(X, Y[tgt], kind, s)) for s in range(200)])
            null = np.array([r2(yp, oof(X, yp, kind, s))
                             for s in range(400)
                             for yp in [RNG.permutation(Y[tgt])]])
            lo, hi = np.percentile(obs, [2.5, 97.5])
            p = float((null >= np.median(obs)).mean())
            report[f"{tgt}|{sname}|{kind}"] = dict(
                median=float(np.median(obs)), lo=float(lo), hi=float(hi), p=p)
            print(f"{tgt:>8} {sname:>32} {kind:>7} {np.median(obs):>+9.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] {p:>8.3f}")
    print()

json.dump(report, open(f"{D}/terrain_regress.json", "w"), indent=1)

# ---- params for the end-to-end test, deployable features only --------------
clip_mu = lambda v: np.clip(v, 0.05, 0.60)
clip_sl = lambda v: np.clip(v, 0.10, 2.00)
for kind in ("ridge", "rf"):
    feats = TERRAIN + RELEASE
    X = np.array([[r[f] for f in feats] for r in rows], float)
    mu = clip_mu(oof(X, Y["mu"], kind, 0))
    sl = clip_sl(oof(X, Y["slab"], kind, 0))
    tbl = {n: {"mu": float(mu[i]), "xi": XI_FIXED, "slab": float(sl[i])}
           for i, n in enumerate(names)}
    json.dump(tbl, open(f"{D}/params_pred_terrain_{kind}.json", "w"), indent=1)
    print(f"wrote params_pred_terrain_{kind}.json  "
          f"mu {mu.mean():.3f}±{mu.std():.3f}  slab {sl.mean():.3f}±{sl.std():.3f}")
