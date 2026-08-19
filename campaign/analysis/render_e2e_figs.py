#!/usr/bin/env python3
# @atlas: Figures for the settled results.
"""Figures for the settled results: the end-to-end ladder, the xi
order-of-operations effect, and the terrain-feature contamination check.

None of these depend on the jobs still running. Palette matches the rest of the
report (validated trio BLUE/ORANGE/AQUA, see dataviz references/palette.md).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, LINE, SURF = "#0b0b0b", "#52514e", "#e3e1dc", "#fcfcfb"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

e2e = json.load(open(f"{D}/e2e_stats.json"))
fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.8))

# ---------------- panel A: the end-to-end ladder --------------------------
ROWS = [
    ("null", "do-nothing null", MUTED),
    ("global_meanopt", "mean of per-event optima, applied globally", MUTED),
    ("global_tuned", "best single tuned global vector", INK),
    ("pred_deploy_ridge", "predicted — deployable features (ridge)", ORANGE),
    ("pred_deploy_rf", "predicted — deployable features (RF)", ORANGE),
    ("pred_all_rf", "predicted — incl. outline outcomes (RF)", BLUE),
    ("pred_all_ridge", "predicted — incl. outline outcomes (ridge)", BLUE),
    ("oracle_mu_slab", "oracle μ,slab — ξ pinned $after$ fitting", YELLOW),
    ("oracle_fixedxi", "oracle μ,slab — ξ fixed $before$ fitting", AQUA),
    ("oracle_full", "oracle — full per-event optimum", AQUA),
]
rows = [(k, l, c) for k, l, c in ROWS if k in e2e]
y = np.arange(len(rows))[::-1]
ax = axs[0]
for yi, (k, lab, col) in zip(y, rows):
    v = e2e[k]["mean"]
    ax.barh(yi, v, 0.66, color=col, zorder=3)
    inside = v < -0.05
    ax.text(v + 0.012, yi, f"{v:+.3f}", va="center", ha="left", fontsize=7.5,
            color="white" if inside else INK,
            fontweight="bold" if inside else "normal")
ax.axvline(0, color=INK, lw=1.1)
ax.axvline(e2e["global_tuned"]["mean"], color=INK, lw=1.2, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([l for _, l, _ in rows], fontsize=7.5)
ax.set_xlim(-0.80, 0.44)
ax.set_xlabel("mean Ω$_T$ over 105 events (each re-simulated at its own vector)")
ax.set_title("End-to-end: predicted parameters, actually simulated\n"
             "dashed = the single global vector a regressor must beat", fontsize=9.5)
ax.grid(axis="x", color=LINE, lw=0.7, zorder=0)
ax.set_axisbelow(True)

# ---------------- panel B: xi order of operations --------------------------
ax = axs[1]
labels = ["ξ free\nall three fitted",
          "ξ fixed $before$\nμ, slab fitted at it",
          "ξ pinned $after$\nμ, slab from free-ξ fit"]
vals = [e2e["oracle_full"]["mean"],
        e2e.get("oracle_fixedxi", {}).get("mean", np.nan),
        e2e["oracle_mu_slab"]["mean"]]
cols = [AQUA, AQUA, ORANGE]
x = np.arange(3)
ax.bar(x, vals, 0.58, color=cols, zorder=3)
for xi_, v in zip(x, vals):
    if np.isfinite(v):
        ax.text(xi_, v + 0.008, f"{v:+.4f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi_, v / 2, f"{100*v/vals[0]:.0f}%\nof the gain", ha="center",
                va="center", fontsize=8, color="white", fontweight="bold")
ax.axhline(e2e["global_tuned"]["mean"], color=INK, lw=1.2, ls="--")
ax.text(2.42, e2e["global_tuned"]["mean"] + 0.006, "best global vector",
        ha="right", fontsize=7.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.6)
ax.set_ylim(0, 0.34)
ax.set_ylabel("mean Ω$_T$")
ax.set_title("ξ costs 3% if you fix it first, 52% if you pin it after\n"
             "the ridge couples μ to ξ, so the order of operations matters",
             fontsize=9.5)
ax.grid(axis="y", color=LINE, lw=0.7, zorder=0)
ax.set_axisbelow(True)

# ---------------- panel C: terrain features, raw vs partial ---------------
ax = axs[2]
terr = json.load(open(f"{D}/terrain_feats.json"))
trunc = json.load(open(f"{D}/terrain_trunc.json"))
props = json.load(open(f"{D}/event_props_merged.json"))
pe = {r["name"]: r for r in json.load(open(f"{D}/perevent100.json"))}
names = sorted(n for n in pe if n in terr and n in trunc)
mu = np.array([pe[n]["best"]["friction_coefficient"] for n in names])
o = lambda k: np.array([props[n][k] for n in names], float)
C = np.column_stack([o("area"), o("drop"), o("w"), o("h"), np.ones(len(names))])
res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
rm = res(mu)

FEATS = [("alpha200", trunc, True), ("slope200", trunc, True),
         ("path_slope_mean", terr, True), ("rel_slope_mean", terr, True),
         ("path_alpha_deg", terr, True),
         ("path_drop_potential", terr, False), ("path_len_m", terr, False)]
labs, raw, part, ok = [], [], [], []
for k, src, clean in FEATS:
    v = np.array([src[n][k] for n in names], float)
    labs.append(k)
    raw.append(np.corrcoef(v, mu)[0, 1])
    part.append(np.corrcoef(res(v), rm)[0, 1])
    ok.append(clean)
yy = np.arange(len(labs))[::-1]
ax.barh(yy + 0.19, raw, 0.36, color=[BLUE if c else ORANGE for c in ok],
        zorder=3, label="raw r with μ*")
ax.barh(yy - 0.19, part, 0.36, color=[BLUE if c else ORANGE for c in ok],
        alpha=0.42, zorder=3, label="partial r, controlling for outline size")
ax.axvline(0, color=INK, lw=1.1)
for t in (0.19, -0.19):
    ax.axvline(t, color=MUTED, lw=0.9, ls=":")
ax.set_yticks(yy)
ax.set_yticklabels([f"{l}" + ("" if c else "  ⚠") for l, c in zip(labs, ok)],
                   fontsize=7.5)
ax.set_xlabel("correlation with fitted μ*   (dotted: |r| = 0.19, p = 0.05)")
ax.set_xlim(-0.2, 0.62)
ax.set_title("Terrain features: two of these are a leak\n"
             "⚠ = computed to the domain edge, so it encodes avalanche size",
             fontsize=9.5)
ax.legend(handles=[Patch(facecolor=BLUE, label="domain-independent"),
                   Patch(facecolor=ORANGE, label="contaminated by domain size")],
          fontsize=7.5, loc="lower right", facecolor="white", framealpha=0.92,
          edgecolor=LINE)
ax.grid(axis="x", color=LINE, lw=0.7, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(f"{D}/e2e_summary.png", dpi=150, bbox_inches="tight")
print("wrote e2e_summary.png", os.path.getsize(f"{D}/e2e_summary.png") // 1024, "kB")
