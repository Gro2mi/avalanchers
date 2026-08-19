#!/usr/bin/env python3
# @atlas: Out-of-fold R² for μ/slab/log ξ across feature sets, with permutation tests and importances.
"""Pilot of the full architecture: can the per-event optima be predicted from
observable features?

Per-event calibration gave 105 (event -> optimal mu, xi, slab) records. This
tests the second half of the pipeline -- the regressor -- on exactly those,
using the outline properties as features.

Metric is out-of-fold R^2 (Q^2): predictions are collected across CV folds and
scored against the GLOBAL mean, so "predict the mean" is exactly 0.0 by
construction and the baseline comparison the brief asks for is built in. A
per-fold R^2 would use each fold's own mean and be far noisier at n=105.

Uncertainty comes from repeating the whole CV with different random splits;
significance from a permutation test that shuffles the target.

Writes regress_report.json.
"""
import json, os, sys, time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import xgboost as xgb

D = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20180124)

props = json.load(open(f"{D}/event_props_merged.json"))
pe = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}
names = sorted(pe)

ASPECT_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135,
              "S": 180, "SW": 225, "W": 270, "NW": 315}

# ---------------------------------------------------------------- features --
# Classified by whether they exist at PREDICTION time under the settled scope
# ("runout given release"). A release area + DEM gives you the start zone, its
# aspect and the fracture width. Everything else here is measured off the
# mapped outline of an avalanche that has already happened -- its final area,
# its drop to the deposit, its bounding box -- and so is an OUTCOME, not an
# input, for any avalanche you are trying to predict.
DEPLOYABLE = ["start", "frac_wdh", "asp_sin", "asp_cos"]
OUTCOME = ["sze", "area", "drop", "bbox_elong", "w", "h"]
LEAK = ["aval_shape"]

rows, Y = [], {"mu": [], "log_xi": [], "slab": []}
for n in names:
    p = props[n]
    a = np.radians(ASPECT_DEG.get(p["aspect"], 0))
    rows.append({
        "start": p["start"], "frac_wdh": p["frac_wdh"],
        "asp_sin": np.sin(a), "asp_cos": np.cos(a),
        "sze": p["sze"], "area": p["area"], "drop": p["drop"],
        "bbox_elong": p["bbox_elong"], "w": p["w"], "h": p["h"],
        "aval_shape": p["aval_shape"],
    })
    b = pe[n]["best"]
    Y["mu"].append(b["friction_coefficient"])
    Y["log_xi"].append(np.log10(b["drag_coefficient"]))
    Y["slab"].append(b["slab_thickness"])

FEATS_ALL = DEPLOYABLE + OUTCOME + LEAK
X_all = np.array([[r[f] for f in FEATS_ALL] for r in rows], float)
Y = {k: np.array(v, float) for k, v in Y.items()}
quality = np.array([props[n]["aval_shape"] for n in names])
# optima pinned to a search-box bound are censored, not stationary points
pinned = np.array([
    pe[n]["best"]["friction_coefficient"] <= 0.0501
    or pe[n]["best"]["friction_coefficient"] >= 0.5999
    or pe[n]["best"]["drag_coefficient"] <= 200.1
    or pe[n]["best"]["drag_coefficient"] >= 11999
    or pe[n]["best"]["slab_thickness"] <= 0.1001
    or pe[n]["best"]["slab_thickness"] >= 1.9999 for n in names])

SETS = {
    "all features": FEATS_ALL,
    "no aval_shape": DEPLOYABLE + OUTCOME,
    "deployable only": DEPLOYABLE,
}


def model(kind):
    if kind == "mean baseline":
        return DummyRegressor(strategy="mean")
    if kind == "ridge":
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 4, 40)))
    if kind == "random forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                     max_features=0.5, random_state=0, n_jobs=1)
    if kind == "xgboost":
        return xgb.XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                min_child_weight=3, random_state=0, n_jobs=1,
                                verbosity=0)
    raise ValueError(kind)


def oof_r2(X, y, kind, seed, k=5):
    """Out-of-fold R^2 against the global mean. Zero == predict-the-mean."""
    pred = np.empty_like(y)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        m = model(kind)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def repeated(X, y, kind, n_rep, k=5):
    return np.array([oof_r2(X, y, kind, s, k) for s in range(n_rep)])


# ------------------------------------------------- stage 1: model selection --
N_SEL = 40
report = {"n": len(names), "n_pinned": int(pinned.sum()),
          "features": {"deployable": DEPLOYABLE, "outcome": OUTCOME, "leak": LEAK}}
sel = {}
t0 = time.time()
print(f"n = {len(names)} events; {int(pinned.sum())} have an optimum pinned to a search bound\n")
print(f"{'target':>8} {'feature set':>16} {'model':>15} {'median OOF R2':>14} {'95% interval':>20}")
for tgt in ("mu", "slab", "log_xi"):
    for sname, feats in SETS.items():
        idx = [FEATS_ALL.index(f) for f in feats]
        X = X_all[:, idx]
        for kind in ("mean baseline", "ridge", "random forest", "xgboost"):
            r = repeated(X, Y[tgt], kind, N_SEL)
            sel[(tgt, sname, kind)] = r
            lo, hi = np.percentile(r, [2.5, 97.5])
            print(f"{tgt:>8} {sname:>16} {kind:>15} {np.median(r):>+14.3f} "
                  f"  [{lo:+.3f}, {hi:+.3f}]")
    print()
print(f"(model selection took {time.time()-t0:.0f}s)\n")

report["selection"] = {f"{t}|{s}|{m}": dict(median=float(np.median(v)),
                                            lo=float(np.percentile(v, 2.5)),
                                            hi=float(np.percentile(v, 97.5)))
                       for (t, s, m), v in sel.items()}

# ---------------------------------------- stage 2: headline + permutation ---
N_REP, N_PERM = 200, 400
best_model = {}
for tgt in ("mu", "slab", "log_xi"):
    cands = [(np.median(sel[(tgt, "no aval_shape", k)]), k)
             for k in ("ridge", "random forest", "xgboost")]
    best_model[tgt] = max(cands)[1]
print("model chosen per target (on the no-aval_shape set):", best_model, "\n")

head = {}
print(f"{'target':>8} {'feature set':>16} {'model':>15} {'OOF R2 (95%)':>26} {'perm p':>8}")
for tgt in ("mu", "slab", "log_xi"):
    kind = best_model[tgt]
    for sname, feats in SETS.items():
        idx = [FEATS_ALL.index(f) for f in feats]
        X = X_all[:, idx]
        obs = repeated(X, Y[tgt], kind, N_REP)
        null = np.array([oof_r2(X, RNG.permutation(Y[tgt]), kind, s)
                         for s in range(N_PERM)])
        p = float((null >= np.median(obs)).mean())
        lo, hi = np.percentile(obs, [2.5, 97.5])
        head[f"{tgt}|{sname}"] = dict(model=kind, median=float(np.median(obs)),
                                      lo=float(lo), hi=float(hi), p=p,
                                      null_median=float(np.median(null)),
                                      null_p95=float(np.percentile(null, 95)))
        print(f"{tgt:>8} {sname:>16} {kind:>15} "
              f"{np.median(obs):>+9.3f} [{lo:+.3f}, {hi:+.3f}] {p:>8.3f}")
    print()
report["headline"] = head

# ------------------------------------------------- quality-1 only subset ----
q1 = quality == 1
print(f"quality-1 outlines only: n = {int(q1.sum())}")
q1res = {}
for tgt in ("mu", "slab", "log_xi"):
    idx = [FEATS_ALL.index(f) for f in SETS["no aval_shape"]]
    obs = repeated(X_all[np.ix_(q1, idx)], Y[tgt][q1], best_model[tgt], N_REP)
    null = np.array([oof_r2(X_all[np.ix_(q1, idx)], RNG.permutation(Y[tgt][q1]),
                            best_model[tgt], s) for s in range(N_PERM)])
    lo, hi = np.percentile(obs, [2.5, 97.5])
    p = float((null >= np.median(obs)).mean())
    q1res[tgt] = dict(n=int(q1.sum()), median=float(np.median(obs)),
                      lo=float(lo), hi=float(hi), p=p)
    print(f"  {tgt:>8} {np.median(obs):>+9.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
report["quality1"] = q1res

# ------------------------------------------- unpinned-only sensitivity ------
print(f"\nexcluding the {int(pinned.sum())} boundary-pinned optima (n = {int((~pinned).sum())}):")
pin = {}
for tgt in ("mu", "slab", "log_xi"):
    idx = [FEATS_ALL.index(f) for f in SETS["no aval_shape"]]
    obs = repeated(X_all[np.ix_(~pinned, idx)], Y[tgt][~pinned], best_model[tgt], N_REP)
    lo, hi = np.percentile(obs, [2.5, 97.5])
    pin[tgt] = dict(n=int((~pinned).sum()), median=float(np.median(obs)),
                    lo=float(lo), hi=float(hi))
    print(f"  {tgt:>8} {np.median(obs):>+9.3f} [{lo:+.3f}, {hi:+.3f}]")
report["unpinned"] = pin

# -------------------------------------------------- permutation importance --
# Importance measured on out-of-fold predictions: shuffle one column of the
# TEST fold and see how much OOF R^2 drops. Averaged over repeats.
def perm_importance(X, y, kind, feats, n_rep=30, n_shuf=6):
    base = np.zeros(n_rep)
    drop = np.zeros((n_rep, X.shape[1]))
    for r in range(n_rep):
        pred = np.empty_like(y)
        shuf = np.zeros((X.shape[1], n_shuf, len(y)))
        rs = np.random.default_rng(1000 + r)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            m = model(kind)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
            for j in range(X.shape[1]):
                for s in range(n_shuf):
                    Xs = X[te].copy()
                    Xs[:, j] = rs.permutation(Xs[:, j])
                    shuf[j, s, te] = m.predict(Xs)
        sst = ((y - y.mean()) ** 2).sum()
        base[r] = 1 - ((y - pred) ** 2).sum() / sst
        for j in range(X.shape[1]):
            rj = [1 - ((y - shuf[j, s]) ** 2).sum() / sst for s in range(n_shuf)]
            drop[r, j] = base[r] - np.mean(rj)
    return base, drop


print("\npermutation importance (drop in OOF R^2 when a feature is shuffled)")
imp = {}
for tgt in ("mu", "slab"):
    feats = SETS["no aval_shape"]
    idx = [FEATS_ALL.index(f) for f in feats]
    base, drop = perm_importance(X_all[:, idx], Y[tgt], best_model[tgt], feats)
    order = np.argsort(-drop.mean(0))
    imp[tgt] = {feats[j]: dict(mean=float(drop[:, j].mean()),
                               lo=float(np.percentile(drop[:, j], 2.5)),
                               hi=float(np.percentile(drop[:, j], 97.5)),
                               kind=("deployable" if feats[j] in DEPLOYABLE else "outcome"))
                for j in range(len(feats))}
    print(f"  --- {tgt} (base OOF R2 {base.mean():+.3f}) ---")
    for j in order:
        tag = "deployable" if feats[j] in DEPLOYABLE else "OUTCOME-ONLY"
        print(f"    {feats[j]:>12} {drop[:, j].mean():>+7.3f}  "
              f"[{np.percentile(drop[:, j], 2.5):+.3f}, {np.percentile(drop[:, j], 97.5):+.3f}]  {tag}")
report["importance"] = imp

json.dump(report, open(f"{D}/regress_report.json", "w"), indent=1)
print("\nwrote regress_report.json")
