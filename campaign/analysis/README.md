# Analysis scripts and results

Durable copies of the analysis behind the identifiability, ridge-splitting,
census and regression-pilot sections of the campaign's working record and
`notes_for_markus.md`. Everything here reads from the session scratchpad
(raster dumps, per-event calibration output) which is **not** in the repo —
these are kept so the method and the numbers survive, not so they re-run
unmodified. Each script's docstring states what it measures and how.

Set `D` at the top of a script to a directory containing the scratchpad inputs
to re-run one.

## Scripts

| file | what it measures |
|---|---|
| `ident_analysis.py` | shape of Ω_T over the (μ, ξ) grid: near-optimal area, aspect ratio, orientation, transferability, covariate correlations |
| `frag_analysis.py` | connected components of the simulated footprint vs the 0.1 m flow threshold |
| `divide_analysis.py` | whether the constructed release area straddles a drainage divide (D8 outlets, 150 m clustering) |
| `crossing_analysis.py` | share of simulated mass leaving the drainage the observed avalanche used |
| `census.py` | composition of the 18 737-polygon mapping and the filter funnel down to 602 candidates |
| `regress_pilot.py` | out-of-fold R² for μ / slab / log ξ across feature sets, with permutation tests |
| `endtoend.py` | builds per-case parameter files from out-of-fold predictions |
| `e2e_stats.py` | paired bootstrap + Wilcoxon on the end-to-end variants |
| `learning_curve.py` | OOF R² from n = 40 to 105 — separates "underpowered" from "no signal" |
| `terrain_features.py` | 16 DEM/release-derived features, and their raw correlations with the optima |
| `terrain_regress.py` | cross-validated skill of terrain features vs outline features |
| `final_pipeline.py` | the corrected architecture: ξ fixed → μ/slab calibrated at that ξ → regressed on deployable features → re-simulated |

## Results committed here

| file | contents |
|---|---|
| `ident_report.json` | per-event surface geometry, transferability, shape-vs-tolerance, covariate correlations |
| `census.json` | size/type/trigger/quality breakdowns, funnel, per-class resolution stats |
| `crossing_report{,_tuned}.json` | divide-crossing mass fractions at default and tuned parameters |
| `frag_report.json` | per-case component counts and detached mass/area |
| `terrain_feats.json` | 16 terrain features per case (⚠ `path_len_m`, `path_drop_potential` are domain-contaminated — see item 12d) |
| `terrain_trunc.json` | descent features truncated at 200/300/500 m — the domain-independent versions |
| `e2e_stats.json` | mean Ω_T, paired delta, bootstrap CI and Wilcoxon p for every end-to-end variant |
| `perevent_fixedxi.json` | per-event calibration with ξ fixed at 754, μ and slab free — the +0.2864 run |

## Two things to know before reusing any of this

1. **`calibrate apply --params-from`** evaluates each case at its own parameter
   vector. Scoring a candidate regressor by R² on the parameters is misleading —
   Ω_T is flat in ξ and fairly flat in μ near the optimum, so parameter error
   and score error are only loosely coupled. Always re-simulate.
2. **The domain is the observed outline padded by 300 m.** Any feature computed
   over the whole domain leaks avalanche size. Use the truncated forms.

## Known gaps at the time of writing

`regress_report.json`, `learning_curve.json`, `terrain_regress.json` and
`final_pipeline.json` were still computing when this was committed; the scripts
that produce them are here, the outputs are not. The learning curve is the one
that matters most — it is what separates "underpowered at n = 105" from "no
signal in these features".
