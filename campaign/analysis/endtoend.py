#!/usr/bin/env python3
# @atlas: Builds per-case parameter files from out-of-fold predictions, for `calibrate apply`.
"""End-to-end test of the architecture: do PREDICTED parameters actually
simulate better than a single global vector?

R^2 on the parameters cannot answer this. The Omega_T surface is flat in xi and
fairly flat in mu near the optimum, so a large parameter error can cost almost
no score -- and conversely. The only test that matters is to take out-of-fold
predicted parameters, simulate every event with its own predicted vector, and
compare mean Omega_T against the alternatives.

Writes the params files; the simulation is run by `calibrate apply`.
"""
import json, os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

D = os.path.dirname(os.path.abspath(__file__))
props = json.load(open(f"{D}/event_props_merged.json"))
pe = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}
names = sorted(pe)
tuned = json.load(open(f"{D}/search100_uniform.json"))[0]["params"]
XI_FIXED = float(tuned["drag_coefficient"])

ASPECT_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135,
              "S": 180, "SW": 225, "W": 270, "NW": 315}
DEPLOYABLE = ["start", "frac_wdh", "asp_sin", "asp_cos"]
OUTCOME = ["sze", "area", "drop", "bbox_elong", "w", "h"]


def featrow(n):
    p = props[n]
    a = np.radians(ASPECT_DEG.get(p["aspect"], 0))
    return {"start": p["start"], "frac_wdh": p["frac_wdh"],
            "asp_sin": np.sin(a), "asp_cos": np.cos(a), "sze": p["sze"],
            "area": p["area"], "drop": p["drop"], "bbox_elong": p["bbox_elong"],
            "w": p["w"], "h": p["h"]}


rows = [featrow(n) for n in names]
y_mu = np.array([pe[n]["best"]["friction_coefficient"] for n in names])
y_sl = np.array([pe[n]["best"]["slab_thickness"] for n in names])


def oof_pred(feats, y, kind="ridge", k=5, seed=0):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    pred = np.empty_like(y)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        m = (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))
             if kind == "ridge" else
             RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                   max_features=0.5, random_state=0))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


# Predictions are clipped to the search box -- a regressor is allowed to be
# wrong but not to propose a parameter the calibration could never have chosen.
clip_mu = lambda v: np.clip(v, 0.05, 0.60)
clip_sl = lambda v: np.clip(v, 0.10, 2.00)

variants = {}
for tag, feats in (("pred_all", DEPLOYABLE + OUTCOME),
                   ("pred_deploy", DEPLOYABLE)):
    for kind in ("ridge", "rf"):
        mu = clip_mu(oof_pred(feats, y_mu, kind))
        sl = clip_sl(oof_pred(feats, y_sl, kind))
        variants[f"{tag}_{kind}"] = {n: {"mu": float(mu[i]), "xi": XI_FIXED,
                                         "slab": float(sl[i])}
                                     for i, n in enumerate(names)}

# --- the alternatives it has to beat -------------------------------------
variants["global_tuned"] = {n: {"mu": float(tuned["friction_coefficient"]),
                               "xi": XI_FIXED,
                               "slab": float(tuned["slab_thickness"])}
                            for n in names}
# the mean of the per-event optima, applied globally: isolates whether any gain
# from a regressor is really just "the average optimum is a better constant"
variants["global_meanopt"] = {n: {"mu": float(y_mu.mean()), "xi": XI_FIXED,
                                 "slab": float(y_sl.mean())} for n in names}
# the oracle: each event at its own fitted optimum, xi pinned to the global
# value so the comparison isolates what mu and slab can buy
variants["oracle_mu_slab"] = {n: {"mu": float(y_mu[i]), "xi": XI_FIXED,
                                  "slab": float(y_sl[i])}
                              for i, n in enumerate(names)}
# the full oracle, xi free -- the per-event ceiling
variants["oracle_full"] = {n: {"mu": float(pe[n]["best"]["friction_coefficient"]),
                               "xi": float(pe[n]["best"]["drag_coefficient"]),
                               "slab": float(pe[n]["best"]["slab_thickness"])}
                           for n in names}

for tag, tbl in variants.items():
    json.dump(tbl, open(f"{D}/params_{tag}.json", "w"), indent=1)
    mus = [v["mu"] for v in tbl.values()]
    sls = [v["slab"] for v in tbl.values()]
    print(f"{tag:>18}  mu {np.mean(mus):.3f}±{np.std(mus):.3f}   "
          f"slab {np.mean(sls):.3f}±{np.std(sls):.3f}")
print(f"\nxi pinned at {XI_FIXED:.1f} for every variant except oracle_full")
print(f"wrote {len(variants)} params files")
