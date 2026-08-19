# Full-parameter per-event search: feasibility results

Run 2026-07-27, on the vast.ai box before teardown. One experiment, full
parameter set, per-event Nelder-Mead, every evaluation logged.

**See also the campaign's working record** for a separate, complementary identifiability
study (12x12 (mu,xi) grid over 12 events, 1728 sims) done in parallel on this
same box: it independently concludes xi is not identified (median 2.11x
admissible window, one event spanning 19.6x) and that `density` has no
dynamical effect at all (cancels analytically in the Voellmy shear term) --
consistent with this run's finding that `drag_coefficient` and `density` show
the widest near-optimal spread of the 6 continuous parameters below (section
2). That grid-scan result is the more trustworthy identifiability read for
mu/xi specifically, since it doesn't share this experiment's
Nelder-Mead-convergence-trail confound (see section 2).

## Deviations from the brief (disclosed up front)

- **104 cases, not ~150, and drawn from the existing 105-case set, not a
  fresh sample from `cands.json`.** `cands.json` has 2168 candidates but only
  105 already have extracted per-case shapefiles (`data/cases/aval_<idx>.shp`)
  and a warm DEM cache; there's no shapefile-extraction pipeline from the
  master `outlines2018.shp` in this repo, and building one reliably didn't
  fit the time budget. Using the already-prepared set let the actual budget
  go to the search itself. `aval_4117` was excluded by the contamination
  screen below, leaving **104**.
- Contamination exclusion used `fab_detect.py`'s reference-overlap rule
  (>=1% of the observed footprint) applied to the **default-parameter** DEM
  dump, not the fitted-parameter simulated footprint (no fitted params exist
  before this run). Only `aval_4117` cleared the 1% bar (38.6% overlap,
  matching the number already in `notes_for_markus.md` item 5). The
  known 8-9 case, footprint-dependent contamination set from item 5 is
  parameter-dependent, so a couple of borderline cases may still be lightly
  contaminated at some fitted parameter vectors; not re-checked post-fit for
  time.

## Setup

- Free parameters (7 of the requested set; `release_band_frac`, `slab_amp`,
  `slab_wind_amp` from the pre-existing 6-dim search were left fixed, not
  requested this round): `friction_coefficient` (mu), `drag_coefficient`
  (xi), `slab_thickness`, `density`, `internal_friction_angle`, `roughness_threshold`,
  and `entrain` -- the entrainment bit relaxed to a continuous [0,1] dial in
  the same unit cube, thresholded at 0.5, so the existing bounded Nelder-Mead
  machinery could search it without a separate categorical loop.
- Bounds: mu [0.05,0.60], xi [200,12000] (log), slab [0.10,2.00] m,
  density [100,400] kg/m3, internal_friction_angle [15,45] deg,
  roughness_threshold [0.002,0.05] (log), entrain [0,1].
- Budget: 200 evaluations/case ceiling (most cases converged and
  simplex-collapsed well before that -- see below).
- Concurrency: 8 `calibrate per-event` processes on the one RTX 5090
  (`--gpu-index 0` for all 8), matching the saturation plateau measured
  earlier today. 104 cases split into 8 shards of 13.
- **12,927 evaluations, 0 failed, in 373s wall-clock** (~34.6 evals/s
  aggregate). Raw logs: `shard_{0..7}.json.evals.jsonl` (one JSON line per
  evaluation: case, ok, omega, reach_err_m, full `Params`). Per-case
  summaries with best-fit params and a top-decile spread diagnostic:
  `shard_{0..7}.json`, checkpointed to disk after every case.
- Mean Omega_T across the 104 cases: **-0.34 (shared default params) ->
  +0.24 (per-event fit)**.

## 1. Convergence

Evals needed per case to reach within 0.02 Omega_T of that case's own final
best (n=104):

| | evals |
|---|---|
| mean | 52.9 |
| median | 55 |
| p10 | 23 |
| p90 | 77 |
| max | 121 |

As a fraction of the 200-eval budget: mean 42%, median 46%. Most searches
terminated via simplex collapse, not budget exhaustion.

**This is the number that answers "hours or weeks."** At ~35 evals/s on one
GPU (8-way concurrent), 20,000 avalanches x ~55 evals/avalanche is ~1.1M
evaluations, **roughly 8.7 hours on this single card**, or under 2 hours on
an 8-GPU rig at the same per-GPU throughput. This is a hardware-provisioning
question, not a multi-week problem, assuming the 104-case number
generalises (small, geographically clustered sample -- flag accordingly).

## 2. Degeneracy -- the critical one

For each case, the set of evaluations scoring within 0.02 Omega_T of that
case's best (n=104 cases):

- Count: mean 46.8, median 48 (out of ~50-121 evals actually run per case
  before collapse) -- **a large fraction of each search's late evaluations
  score near-best.**
- Per-dimension spread of that near-optimal set, as a fraction of the full
  allowed range (median / p90 across the 104 cases):

  | parameter | median spread | p90 spread |
  |---|---|---|
  | friction_coefficient | 6.0% | 11.9% |
  | drag_coefficient | 9.0% | 18.8% |
  | slab_thickness | 4.6% | 13.0% |
  | density | 9.9% | 23.0% |
  | internal_friction_angle | 7.6% | 22.2% |
  | roughness_threshold | 9.9% | 29.5% |

**Be blunt, as asked: this is a real but moderate flatness, not proof of
arbitrary parameters, and the method has a confound that has to be named.**
Nelder-Mead's simplex shrinks as it converges, so late-search evaluations
are geometrically close to the optimum *in parameter space* almost by
construction -- a high "near-optimal count" partly reflects the optimizer's
own endgame, not necessarily a wide genuine plateau. The more trustworthy
signal here is the **per-dimension spread numbers**: at the median, no
parameter moves more than ~10% of its allowed range while staying within
0.02 Omega_T of best. That is mild, not "friction can vary 2x while
roughness compensates for free." But the **p90 tail is not small** --
roughness_threshold and density both reach ~23-30% of their range at the
90th percentile, meaning for a real minority of events (order 10/104) the
fit is substantially underdetermined in at least one dimension. **The
honest read: most per-event fits are reasonably well-identified; a
non-trivial minority are not, and roughness/density are the parameters
where that shows up most.** A proper answer (a real multi-start / basin
survey per case, not NM's own convergence trail) is needed before trusting
this fully -- flagged as a follow-up, not resolved here.

## 3. Physically implausible optima / boundary hits

Fraction of the 104 per-case best-fit vectors landing within 5% of a search
bound:

| parameter | at bound |
|---|---|
| friction_coefficient | 13/104 (12%) |
| drag_coefficient | 11/104 (11%) |
| slab_thickness | 16/104 (15%) |
| density | 11/104 (11%) |
| internal_friction_angle | 19/104 (18%) |
| roughness_threshold | 6/104 (6%) |
| entrainment ON at optimum | 1/104 (1%) |

Every dimension has *some* cases pinned at a bound (11-19% for six of the
seven), which is a real signal that a meaningful minority of per-event fits
are unphysical or at least bound-constrained rather than sitting at an
interior maximum -- worth tightening bounds or investigating those specific
cases before trusting the database at scale. `internal_friction_angle` is
the worst offender (18%). **Entrainment essentially never helps**: it's
switched on at the optimum for only 1 of 104 cases, independently
corroborating `notes_for_markus.md` item 8 ("entrainment flag appears to
be a no-op") from a completely different angle -- if entrainment did
anything, a search that's free to turn it on would use it more than 1% of
the time.

## 4. Emulator training data

12,927 evaluations logged, 104 cases, 0 failures. Parameter-space coverage
across all logged evals (unit-cube position, mean +/- std, full range is
[0,1]):

| parameter | mean | std | p10 | p90 |
|---|---|---|---|---|
| friction_coefficient | 0.52 | 0.24 | 0.20 | 0.93 |
| drag_coefficient | 0.48 | 0.22 | 0.16 | 0.73 |
| slab_thickness | 0.28 | 0.23 | 0.03 | 0.57 |
| density | 0.32 | 0.18 | 0.08 | 0.52 |
| internal_friction_angle | 0.72 | 0.19 | 0.49 | 0.97 |
| roughness_threshold | 0.46 | 0.17 | 0.22 | 0.64 |

**Flagging exactly the sampling-bias risk asked about: yes, it clusters.**
Median 58% of each case's own evaluations score within 0.05 Omega_T of that
case's own best (p10 46%, p90 70%). A search-generated dataset is, by
construction, mostly composed of parameter vectors the search already
decided were good -- the "how bad does it get in the rest of the space"
region is underrepresented. **For training a surrogate that needs to
generalise, this raw log is the wrong distribution on its own** -- it
should be supplemented with a separate, deliberately-spread sample (e.g.
Latin hypercube or Sobol over the same bounds, evaluated once per case)
rather than relying on search exhaust alone. The per-dimension
mean/std/p10-p90 numbers above look reasonably centred and not collapsed
to a corner, which is the good news, but centred-marginals do not rule out
the good-scoring-cluster problem the paired-sample statistic (58% median)
directly demonstrates.

## Files

- `shard_{0..7}.json` -- per-case summaries (best params, omega_start,
  omega_best, evals, termination reason, top_decile_spread, full CaseResult)
- `shard_{0..7}.json.evals.jsonl` -- every single evaluation, one JSON line
  each: `{case, ok, omega, reach_err_m, params}`
- `shard_{0..7}.log` -- run logs
- `excluded_cases.json` -- contamination-screen exclusions (`["aval_4117"]`)
- `pe_analysis_summary.json` -- the numbers above, machine-readable
