#!/usr/bin/env python3
# @atlas: OOF R² from n = 40 to 105 — separates "underpowered" from "no signal".
"""Underpowered, or no signal? A learning curve tells them apart.

If CV R^2 climbs with sample size, more events would help and "underpowered" is
the right verdict. If it sits flat at ~0 from n=40 to n=105, more of the same
events will not rescue it and the features are the problem.

Also fits an in-sample R^2 curve: the gap between in-sample and out-of-fold is
the overfitting the model is doing, which at n=105 is most of what it does.
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
ASPECT_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135,
              "S": 180, "SW": 225, "W": 270, "NW": 315}
FEATS = ["start", "frac_wdh", "asp_sin", "asp_cos",
         "sze", "area", "drop", "bbox_elong", "w", "h"]


def row(n):
    p = props[n]
    a = np.radians(ASPECT_DEG.get(p["aspect"], 0))
    return [p["start"], p["frac_wdh"], np.sin(a), np.cos(a), p["sze"],
            p["area"], p["drop"], p["bbox_elong"], p["w"], p["h"]]


X = np.array([row(n) for n in names], float)
Y = {"mu": np.array([pe[n]["best"]["friction_coefficient"] for n in names]),
     "slab": np.array([pe[n]["best"]["slab_thickness"] for n in names]),
     "log_xi": np.array([np.log10(pe[n]["best"]["drag_coefficient"]) for n in names])}


def mk(kind):
    return (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))
            if kind == "ridge" else
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  max_features=0.5, random_state=0))


def oof_r2(Xs, ys, kind, seed):
    pred = np.empty_like(ys)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        m = mk(kind)
        m.fit(Xs[tr], ys[tr])
        pred[te] = m.predict(Xs[te])
    return 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()


def insample_r2(Xs, ys, kind):
    m = mk(kind)
    m.fit(Xs, ys)
    p = m.predict(Xs)
    return 1 - ((ys - p) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()


SIZES = [40, 55, 70, 85, 105]
N_REP = 120
rng = np.random.default_rng(7)
out = {}
print(f"{'target':>8} {'model':>7} {'n':>5} {'OOF R2 (95%)':>26} {'in-sample R2':>13}")
for tgt in ("mu", "slab", "log_xi"):
    for kind in ("ridge", "rf"):
        curve = []
        for n in SIZES:
            oo, ins = [], []
            for r in range(N_REP):
                idx = rng.choice(len(names), n, replace=False) if n < len(names) \
                    else np.arange(len(names))
                oo.append(oof_r2(X[idx], Y[tgt][idx], kind, r))
                ins.append(insample_r2(X[idx], Y[tgt][idx], kind))
                if n == len(names) and r > 0 and kind == "ridge":
                    pass
            lo, hi = np.percentile(oo, [2.5, 97.5])
            curve.append(dict(n=n, median=float(np.median(oo)), lo=float(lo),
                              hi=float(hi), insample=float(np.median(ins))))
            print(f"{tgt:>8} {kind:>7} {n:>5} {np.median(oo):>+9.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] {np.median(ins):>+13.3f}")
        out[f"{tgt}|{kind}"] = curve
    print()

json.dump({"sizes": SIZES, "curves": out}, open(f"{D}/learning_curve.json", "w"), indent=1)
print("wrote learning_curve.json")
