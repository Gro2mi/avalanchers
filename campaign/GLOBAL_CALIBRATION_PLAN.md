# Freezing the global constants

A staged experiment that fixes every "general" knob of the simulator, so that
the subsequent 20,000-event per-event harvest produces training targets that
are not contaminated by a badly-chosen global configuration.

Companion documents: the campaign's working record (why per-event calibration is the
architecture),
`campaign/analysis/README.md` (the statistical machinery reused here).

---

## Why this experiment exists, and what it is not

The settled result in the campaign's working record is that **ξ must be fixed first, and
μ and slab calibrated at that ξ**: done in that order ξ costs 3% of the
achievable per-event gain, done backwards it costs 52%. That result generalises
past ξ. Every global knob has the same property — the optimal μ for an event
depends on the configuration it was fitted under — so a harvest run at a
carelessly-chosen global configuration produces per-event optima that are
partly compensating for that configuration, and a regressor trained on them
learns the compensation as if it were physics.

So the global knobs get frozen first, deliberately, with the per-case
parameters treated as **nuisance parameters that are re-calibrated inside every
comparison**. That nested structure is not an optimisation; it is the only way
a comparison between two global candidates is honest.

**Explicitly out of scope.** The 20,000-event harvest itself, the deliberately
random parameter draws that must supplement the search exhaust (see the
sampling-bias finding in `campaign/per_event_search_2026-07-27/README.md`),
any regression on the resulting targets, terrain-feature work, and the
case-extraction pipeline that would take the panel past 105 cases. This
experiment ends when a parameter vector is frozen and written down.

---

## Three things found while reading the code, before designing anything

These change what is worth testing, so they come first.

### 1. `roughness_threshold` is a no-op in this harness

It is read by exactly two shaders, `compute_roughness.wgsl` and
`compute_release_areas.wgsl`. `Simulation::get_release_areas` runs neither of
them when release areas arrive as an array — and `calibrate` always supplies an
array, because `build_release` constructs the release band on the CPU from the
observed outline. The parameter is uploaded in the settings uniform and never
read by any code path this experiment exercises.

It is therefore demoted from a Tier-2 constant to a **two-line null check**
(stage `s3_roughness`) whose expected result is scores that agree to within the
solver's own noise floor. Note: **the solver is not bit-reproducible.** Two
identical runs of the same 105 cases differed on two of them by exactly one cell
(3536→3537, 5241→5242), worth ~2×10⁻⁴ in Ω_T — `grid_cell_count` and the mass
grid are built with `atomicAdd`, whose summation order varies between runs. So
the check is |ΔΩ_T| < 10⁻³, not equality, and the file docstring's claim that
"everything is deterministic (the particle seed is hard-coded to 42)" is true of
the seed but not of the reduction. The noise floor is two orders of magnitude
below the 0.02 decision band, so it changes nothing else.

This also retroactively explains the 7-parameter search result in
`campaign/per_event_search_2026-07-27/README.md`, where
`roughness_threshold` showed one of the widest near-optimal spreads (9.9%
median, 29.5% p90). A dimension the objective cannot see is a dimension the
simplex wanders along freely.

### 2. Turning off particle interaction silently changes the scoring rule

In `evaluate_case`, the simulated footprint is `peak_h >= flow_threshold` when
the particle-interaction bit is set, and `cell_count > 0` otherwise — because
peak flow thickness is only produced by the grid pass, which particle
interaction gates. A Tier-1 comparison across that flag is therefore comparing
two definitions of "affected" as well as two physics configurations.

Fixed rather than worked around: `evaluate_case` now scores **both** footprints
on every evaluation and records `omega_cells` alongside `omega`. The cell-count
buffer was already being fetched unconditionally, so this costs one extra pass
over the grid against a GPU simulation — nothing. Stage 1 is decided on
`omega_cells`, the metric that is comparable across the flag, with `omega`
reported alongside.

### 3. Two knobs are conditional, not global

- **ξ (`drag_coefficient`) is read only by Voellmy (1) and VoellmyMinShear
  (2).** Coulomb returns zero drag; samosAT uses a fixed log-profile with
  hard-coded constants and no ξ. The ξ stage only runs if the structure winner
  is a Voellmy variant.
- **`internal_friction_angle` only bites when particle interaction *and* earth
  pressure are both on.** `grid_physics` early-returns without particle
  interaction, and inside it the angle enters only through
  `earth_pressure_coefficient`, which the earth-pressure flag gates. (The one
  other use, a `lateral_factor` yield criterion at `grid_physics.wgsl:104`, is
  computed and then never used — the force written on the next line ignores
  it.) So the ifa stage is conditional on the winner's flags.

A related note for interpreting stage 1: **`density` cancels analytically for
Coulomb, Voellmy and VoellmyMinShear but not for samosAT**, because the samosAT
normal-friction term computes its `rs` ratio against a hard-coded
`const density = 200.0` while `mass_per_area` uses the real value. The
"density is dead" finding in the campaign's working record holds for the models it was
measured on; do not extend it to samosAT without re-measuring.

---

## Stage −1: two objective fixes, which gate everything

Two defects make the simulated footprint systematically unlike anything an
observer could have mapped. Both are fixed **before** any tuning stage runs,
because tuning against a broken objective is precisely the waste the staging
exists to prevent. Their parameters join `flow_threshold` as **fixed scoring
conventions: documented, recorded on every evaluation, never optimised.**

### Fix 1 — single-particle runout tails

An observed outline is a mapped polygon. It never records a lone particle
rolling 50 m past the deposit as a thin filament, but the simulated footprint
does, so simulated runout reads systematically long. Two filters, because they
remove different things and neither subsumes the other:

- **`min_residence`** (default **0.25**, measured). `grid_cell_count`
  accumulates one unit per particle per timestep, so a cell a single particle
  rolled through holds a handful of counts while a cell the body flowed over
  holds thousands. The gate is on `count · cfl / ppc`, roughly "how many
  cell-equivalents of material passed through here". **The normalisation is not
  cosmetic**: the raw count scales with `ppc` and with 1/`cfl`, and both are
  being chosen in tier 0, so an un-normalised threshold would make the scoring
  convention move with the numerics it is meant to be independent of. A unit
  test pins the invariance.
- **`require_release_connected`** (default 1). Drop footprint not 8-connected to
  the release. `frag_analysis.py` measured a median of 9 components per case
  with the largest touching the release in **105/105**, and established the
  fragments are a contour artefact of the 0.1 m threshold — no fragment is
  reachable by descent from any release cell.

**They do different jobs and both are needed.** Connectivity removes *detached*
blobs; it cannot remove an *attached* thin tail, which is by definition
connected. The residence gate removes the tail. A unit test asserts exactly this
so that nobody later deletes one as redundant.

**Measured on all 105 cases** (μ=0.36, ξ=754, filters vs unfiltered within the
same run):

| | tuned (μ=0.36) | defaults (μ=0.155) |
|---|---|---|
| cases overshooting | 48/105 | 105/105 |
| median reach error, **overshooting cases** | 224 → **179 m** | 356 → **345 m** |
| median reach error, undershooting cases | −154 → −166 m | — |
| median reach error, all cases | 173 → 171 m | 356 → 345 m |
| mean ΔΩ_T | +0.0003 | **+0.0229** |
| median cells removed | 2.4% | 8.0% |

**Read honestly: the filter cuts overshoot where overshoot exists, and cuts
undershoot too, and at tuned parameters those roughly cancel.** On the 48
overshooting cases it takes 46 m off the median reach error, which is the
complaint being addressed; on the undershooting half it makes the error slightly
worse. Median reach error across all cases therefore barely moves (173 → 171 m).
At default parameters, where every case overshoots, the picture is unambiguous
and ΔΩ_T of +0.023 clears the decision band.

So the justification for this filter is **correctness, not score**: an observed
outline cannot contain a one-cell filament, so scoring against one is wrong
regardless of which direction the error happens to run.

### Fix 2 — ridge/watershed release leak

The domain is the outline padded by 300 m and the polygon is mapped from 1.5 m
imagery onto a 5 m DEM, so the upper edge of the constructed release band can
land on the far side of a ridge crest. Mass released there runs down the wrong
valley: it can never intersect the observation, so it is pure overshoot
penalty attributable to misalignment rather than physics.

**`clip_release_to_drainage`** (default 1) keeps a release cell only if its D8
steepest-descent path reaches the observed outline **within a step cap of one
grid diagonal**. Computed once per case at preparation time — it depends only on
terrain and the observation — so it is free per evaluation.

The cap is derived from the case's own grid, not a magic constant: domains run
from roughly 100 to 400 cells across, so any fixed step count would be lenient
on small cases and severe on large ones. A cell whose water only reaches the
outline after wandering further than the domain is wide has gone down one valley
and come back, and is not in the outline's drainage in any useful sense. The
implementation computes the **exact** step count per cell rather than a
reachability bit — the descent graph is functional, so distance-to-outline is
well defined and memoises in O(n) — which means the cap is a comparison, not an
early-terminated walk, and the counts themselves are available for diagnosis.

Defined against the outline directly rather than against a clustered basin id,
which is how `crossing_analysis.py` measured the problem. That method has a
failure mode this must not inherit: on `aval_6719` it labels **100%** of the
release as outside the intended drainage while the case still scores **+0.337**,
which is a clustering artefact, not a real leak. "Does water from here run into
the observed polygon" needs neither clustering nor a distance threshold.

### Pit tolerance: D8 is a droplet, an avalanche is not

D8 descent stops dead in any closed depression. A moving avalanche carries
momentum and rides over shallow hollows. On `aval_13722` that difference clipped
98% of a release: its removed cells travelled only 80–230 m before terminating,
and **the pits that stopped them are 0.06–0.51 m deep** — quantisation and
micro-relief on a 5 m DEM, not terrain.

So before tracing descent, closed depressions shallower than
`PIT_FILL_TOLERANCE_M` are filled by depth-limited priority-flood, on a copy
used **only** for drainage — the physics runs on the untouched DEM. Depressions
deeper than the tolerance keep their elevation and still terminate a descent, so
genuine basins are unaffected.

One implementation detail that is load-bearing: the fill adds a 1 mm gradient
rather than levelling to the exact spill elevation. Filling flat would replace a
pit with a plateau, and D8 cannot route across a plateau either — every cell
would find no strictly-lower neighbour and the descent would still die, just at
the rim instead of the bottom. A unit test covers both the shallow and deep
cases and the routing.

**Swept on all 105 cases:**

| tolerance | no-ops | any clip | refused | 4 known leaks still clipped | `aval_13722` |
|---|---|---|---|---|---|
| 0 m | 84 | 19 | 2 | 4/4 | refused |
| 0.5 m | 86 | 17 | 2 | 4/4 | refused |
| **1–100 m** | **86** | **18** | **1** | **4/4** | **78% clipped, 41 cells** |

All four acceptance criteria are met at ≥1 m: `aval_13722` survives, the four
genuine leaks stay clipped, no case stops being a no-op, and two additional
false positives disappear (`aval_10321` 7.6%→0, `aval_6716` 0.2%→0). A third,
**`aval_8075`, drops 27.8%→0.2%** — a substantial false positive that the visual
pass had not flagged. Like-for-like over the 103 cases present at both settings,
the paired ΔΩ_T is +0.0004, CI [−0.0002, +0.0015].

⚠ **Be clear about what this measurement does and does not pin down.** Results
are *identical* from 1 m to 100 m — the parameter is inert above 1 m on this
panel, because these 300 m-padded alpine domains drain off the domain edge
rather than into closed basins. So the data fixes the lower bound and says
nothing about the upper one. **2 m is therefore a physical choice, not a
measured optimum**: comfortably above the measured micro-relief (0.06–0.51 m)
and far below any basin that would genuinely arrest an avalanche. A larger value
would be indistinguishable here but carries more risk at 20k-harvest scale,
where terrain not represented in this panel — glacial cirques, dammed valleys —
could contain real basins worth respecting. This is a place where the harvest
may behave differently from the panel, and it should be re-checked there.

### Why the `crossing_analysis` 12 is not the acceptance list

An earlier version of the stage −1 criterion said the clip "must visibly cut the
release on the known leak cases", meaning the 12 cases with the highest
`release_frac_outside_drainage` in `crossing_report.json`. **That criterion was
wrong and has been withdrawn.** On the box, 10 of those 12 show a 0.0–0.2% cut,
and that is the *correct* outcome.

Two distinct phenomena were being conflated:

- **`crossing_analysis.py` measures downstream mass crossing** — how much
  simulated mass ends up outside the observed drainage, which
  `notes_for_markus.md` item 15 shows is friction-driven, happens mid-path,
  and largely disappears at tuned parameters (>10% of mass in the wrong drainage
  falls from 15/70 cases to 2/70). That is item 9 in disguise, not a release
  problem.
- **This clip fixes release-cell drainage** — cells inside the mapped polygon
  that sit past a crest in DEM space. A different question about a different
  part of the domain.

Worse, the list is built on the basin-clustering method whose false positives
this design explicitly refused to inherit — and `aval_6719`, which that method
calls 100% leaking while the case scores +0.337, **is on it**. The criterion
therefore demanded the clip fire on a case already documented as a false
positive of the method it came from. Two paragraphs apart in this document.

**The lesson worth keeping: a validation criterion has to name evidence produced
by the thing being validated.** Borrowing a list from a neighbouring analysis
imports that analysis's failure modes as acceptance requirements.

The replacement criterion uses the clip's own diagnostics:

1. **Distribution shape** — `release_clipped_frac` mostly zero with a short tail.
   Measured on GPU: 18/104 with any clip, the rest exactly zero.
2. **Per-case D8 adjudication of the tail** — each SEVERE case traced and judged
   individually. That pass confirmed 4 genuine leaks (removed cells travelling
   335–735 m into foreign drainage or pits) and caught 1 false positive
   (`aval_13722`, cells travelling 80–230 m into 0.06–0.51 m hollows), which
   produced the pit-tolerance fix.
3. **Short, adjudicated SEVERE/REFUSED lists** — GPU: SEVERE = {`aval_8025` 82%,
   `aval_13722` 78%, `aval_11149` 66%, `aval_13404` 56%}, REFUSED = {`aval_8124`}.

### Stage −1, as measured on the box

Cross-checked against the Metal numbers in this document; all agree within the
cross-backend noise band (which is itself larger than the ~2×10⁻⁴ atomics floor,
since the two backends order reductions differently).

| | GPU (Box B, n=104) | Metal (n=105) |
|---|---|---|
| tail filter mean ΔΩ_T | +0.0033 | +0.0003 |
| tail filter median ΔΩ_T | +0.0035 | +0.0028 |
| \|reach error\| median | 171.8 → 167.3 m | 173.1 → 171.0 m |
| connectivity fallbacks | 0 | 0 |
| cases with any clip | 18/104 | 18/105 |
| residence plateau | 0 → 1.0, peak near 0.25 (+0.0046) | 0.125–0.5, peak 0.25 (+0.0029) |
| residence falloff | −0.0129 at 2.0, −0.0569 at 4.0 | −0.0148 at 2.0 |

The residence result reproduces independently on two backends, which is the
reassurance that matters for a convention chosen by measurement.

### The re-established incumbent

Measured on the box under the new objective, which is the number every stage-1
margin is read against:

**mean Ω_T = +0.0046** (HWRI λ=1 −0.2349, do-nothing null −0.7056, n = 104 —
`aval_8124` refused by the drainage clip, as expected).

Cross-check: the Metal run of the same vector at the same conventions gave
+0.0061, so the two backends agree to 0.0015 — inside the cross-backend band and
two orders below the 0.02 decision band.

⚠ **Do not read this against the +0.006 "tuned global" row in
the campaign's working record.** They differ in objective *and* in parameter vector (that
row is μ = 0.218; this is μ = 0.36), so their near-equality is a coincidence of
two changes, not evidence that the fixes left the baseline untouched. The only
thing this number is for is being the baseline the campaign's own margins are
measured against.

### ⚠ The tautology this replaced — and why the unit tests missed it

The first implementation asked "does the descent reach **any reference cell**",
seeding every reference cell as already-arrived. `build_release` only ever draws
candidates from *inside* the observed polygon, so every candidate was a
reference cell, every candidate answered yes, and **the clip was a no-op on all
105 cases** — `release_clipped_frac` = 0.0 across the board, release counts
byte-identical with the clip on and off.

The unit tests passed throughout. They exercised the descent function in
isolation on far-flank cells *outside* the polygon — which is precisely the
population `build_release` never asks about. **A test can be correct about a
function and still prove nothing about the system**, and the fix is to test
through the caller: `build_release_clips_the_far_flank_of_a_polygon_spanning_a_ridge`
drives the real path with a polygon straddling a crest, and it fails against the
old wiring (verified by reverting the seeding and watching it go red).

The deeper error was conceptual: treating polygon membership as proof of
drainage assumes the polygon is ground truth in DEM space, when the whole
premise of the fix is that a 1.5 m-imagery outline laid on a 5 m DEM can cover
cells that sit past a crest. Hence the target is now the outline's lower *body*,
which a release cell must descend to reach.

**Measured on all 105 cases, after a first version of this clip turned out to
be a tautological no-op** (see below):

| clipped share of the release | cases |
|---|---|
| zero — genuine no-op | **84/105** |
| any | 21/105 |
| >10% | 9/105 |
| >50% (SEVERE) | 5/105 |
| 100% (REFUSED) | 0/105 |

Mean clipped fraction 4.8%, median 0%, max 97.8%. Paired ΔΩ_T over all 105 is
**+0.0016, CI [−0.0091, +0.0110]** — indistinguishable from zero; on the 21
affected cases +0.0082, CI [−0.0467, +0.0532], 15 improved and 8 degraded.

**This matches the prior expectation exactly**, which is the main reason to
believe the rewire is right: `notes_for_markus.md` item 15 found only 2 of 70
clean cases with a release cell in a different drainage, and concluded the real
divide-crossing problem is downstream and mostly disappears at tuned friction. A
clip firing on 21/105 with a median of zero is the "small tail of modest clips"
that predicts. A clip firing on most cases would have meant the target set was
too aggressive.

**So this fix does not improve the score, and should not be sold as if it does.**
Its justification is correctness: mass released on the far side of a crest
descends into a valley the observation does not cover, so it can only ever be
overshoot penalty attributable to georeferencing. Removing it makes the objective
mean what it says.

⚠ **The five SEVERE cases deserve a decision before the campaign runs.**
`aval_13722` loses 97.8% of its release (183 → 4 cells) and its score collapses
from −0.589 to −0.978; `aval_8124` loses 81.6% and drops −0.713 → −0.854. Four
surviving release cells is not a simulation anybody should trust, but it is not
literally zero so the refusal guard does not fire. **Recommendation: add an
absolute floor — refuse the case when fewer than ~20 release cells survive — or
exclude these five from the panel by name.** I have not made that change
unilaterally because it moves the panel, which is a decision above my level.
The five: `aval_8025`, `aval_8124`, `aval_11149`, `aval_13404`, `aval_13722`.

**Decision (team lead, 2026-07-28): the absolute floor, `MIN_RELEASE_CELLS =
20`.** Named exclusions do not generalise to the 20k harvest; a floor does. It
is scoped to clipped cases only — a release that had fewer than 20 candidate
cells *before* the clip is pre-existing behaviour and passes through (the
floor may not change the panel for cases the clip never touched). Whichever of
the five fall below the floor land in the REFUSED list at stage −1; the rest
run flagged SEVERE. Implemented as `clip_refused` with a unit test covering
the 183 → 4 shape, the just-under/at-floor boundary, and the naturally-tiny
exemption.

**The empty-release guard, in two tiers.** A clip that removes most or all of a
release band is not the ridge condition any more; it is whole-polygon
misregistration, and it must be visible rather than absorbed:

- **More than 50% removed** (`CLIP_SEVERE_FRAC`) — the case is flagged on stderr
  and `release_clip_severe` is recorded on every evaluation, so the list is
  recoverable from the database and appears in the validation report. The case
  still runs.
- **All removed** — the case is **refused**. `evaluate_case` returns an error
  naming the cause, so the case lands in `failures` rather than being simulated
  with an empty release and scored. Silently falling back to the unclipped band
  would hide exactly the cases most worth seeing; scoring an empty release would
  produce Ω_T = −1 and read as a catastrophic physics failure. Neither is
  acceptable, so the case is excluded and named.

Both thresholds are fixed conventions, not tunables. The connectivity filter
keeps a genuine fallback — if no release-connected seed exists it returns the
footprint unchanged and sets `connect_fallback` — because an absent seed means
the filter has nothing to do, which is different from a release that drains
away.

### Footprint-source consistency

The particle-interaction flag switches which footprint is scored: peak thickness
above `flow_threshold` when it is on, cell-visited when it is off. **Both filters
are applied to both sources through one shared code path**, so a stage-1
comparison across that flag cannot end up comparing a filtered objective against
an unfiltered one and attributing the difference to physics. This is structural
rather than incidental — there is a single `apply_filters` closure used for
each source, not two parallel branches that could drift.

### The discontinuity this creates, and what must be re-measured

**Both fixes change the objective, so every Ω_T in the campaign's working record is
measured against the old one and is not comparable to anything this campaign
produces.** That includes the +0.2864 fixed-ξ run, the +0.295 free-ξ oracle, the
+0.006 tuned global baseline, and the whole end-to-end ladder. None of them are
wrong; they answer a different question.

Two consequences that must not be skipped:

1. **The incumbent baseline is re-established under the new objective** before
   stage 1's margins mean anything. One `apply` pass over the panel.
2. **Stage 0's per-case operating points come from
   `perevent_fixedxi.json`**, fitted under the old objective. They are still a
   reasonable operating point for a numerics comparison — the numerics question
   is local — but they are no longer *the* optima. Stated here so the mild
   circularity is visible rather than buried.

### Validating it

`python_scripts/validate_fixes.py` produces the before/after evidence and
`fix_validation.json` for the report. It needs a GPU and the case data, so it
runs on the box as the very first job. It measures the tail filter from a single
run (every evaluation records `omega_raw` and `sim_reach_raw_m` alongside the
filtered values), the ridge clip from a paired run with `--clip-drainage 0/1`
(the clip changes the release, so it cannot be captured within one run), and a
`min_residence` sweep over {0, 0.125, 0.25, 0.5, 1, 2, 4}.

**`min_residence` began as the one number in this document chosen by argument
rather than measurement — 1.0, on the reasoning that it meant "one
cell-equivalent of material passed through". The sweep withdrew it.** 1.0 sits
past the plateau: no mean benefit over no gate at all, and up to −0.169 on the
worst case. The measured plateau is 0.125–0.5 and the default is now **0.25**
(table above).

**The normalisation is portable across numerics — measured, not assumed.** The
gate is normalised by `cfl / ppc` precisely so the convention does not move with
the numerics tier 0 is choosing, which is an argument until it is tested. So the
whole sweep was repeated at a second operating point, cfl 0.35 / ppc 16 — a 2×
change in `ppc` and 1.4× in `cfl`:

| `min_residence` | ΔΩ_T at cfl 0.5 / ppc 8 | ΔΩ_T at cfl 0.35 / ppc 16 |
|---|---|---|
| 0.125 | +0.0025 | +0.0027 |
| **0.25** | **+0.0029** | **+0.0029** |
| 0.5 | +0.0029 | +0.0028 |
| 1.0 | −0.0004 | −0.0003 |
| 2.0 | −0.0148 | −0.0170 |

Agreement to ~0.0002 at every point, peak at 0.25 in both. The convention
travels. **This matters well beyond the numerics freeze**: the per-dataset
extensions re-sweep `min_residence` on panels with different flow regimes, and
this establishes that a shift seen there is a property of the *regime* rather
than of whatever numerics that dataset happened to run at — which is exactly the
confound the extension section warns about elsewhere.

---

## The panel

**102 cases: the existing 105-case set minus `aval_4117`, `aval_7743` and
`aval_847`.**

This is a forced choice, not a preference. Blocker 0 in the campaign's working record is
still open — there is no pipeline extracting cases from the master
`outlines2018.shp`, so only these 105 have per-case shapefiles and a warm DEM
cache. The 602-candidate isolated pool and the 2,168 in `cands.json` are not
reachable for this experiment. The three exclusions are the fabricated-terrain
cases whose contamination overlaps a footprint by ≥1%.

Composition (from the fixed-ξ run — note these are the **105-case** figures,
before the three exclusions; the 102-case composition shifts by up to three
counts and should be recomputed at s0 if anyone needs it exactly): size class
3/4/5 = 58/46/1, outline quality 1/2/3 = 52/40/13, split cal/val = 66/39.

**The same panel is used for every candidate at every stage**, and every
comparison is paired case-by-case. `analyze_stage.py` intersects the case sets
across candidates and refuses to credit a candidate for the cases it completed
if it failed others — a configuration that crashes on the hard cases would
otherwise look best. It names any candidate that shrinks the panel.

**cal/val is a consistency diagnostic, not a holdout.** Selection uses all 102
cases, because the paired standard error at n=102 is what makes a 0.02 decision
band detectable at all. The cal and val margins are reported separately and a
sign disagreement between them blocks a freeze — that catches a winner carried
by one subgroup without paying the power cost of a formal holdout.

### Contamination is parameter-dependent

the campaign's working record is explicit that the fabricated-terrain overlap depends on
where the simulated footprint lands, so it must be a per-run pre-flight check
rather than a one-time filter. The three-case exclusion above is the
default-parameter screen. Before freezing, re-run the overlap check against the
*winning* configuration's footprints and confirm the exclusion set has not
grown; if it has, drop the new cases and re-run `analyze_stage.py`, which is
seconds of work because the database already holds every evaluation.

---

## The inner loop

Every global candidate is scored by **re-calibrating (μ, slab) per case at that
candidate** and taking each case's own best Ω_T. This is the "fix the global
knob first, then calibrate at it" ordering, applied to whichever knob the stage
is choosing.

- **Free dimensions**: `mu,slab`. Nothing else. ξ is a stage-2 decision, not an
  inner-loop parameter, precisely because co-fitting it is what costs 52%.
- **Budget**: 40 evaluations. Measured on exactly this configuration in
  `campaign/analysis/perevent_fixedxi.json` — mean 34.2 evaluations, median 35,
  **81 of 105 searches terminating by simplex collapse rather than budget
  exhaustion**. Raising it buys the 24 stragglers a little; the confirmation
  stage handles them with multi-start instead.
- **Start point**: μ = 0.36, slab = 0.56, the panel medians of the fixed-ξ
  optima. One shared start for every candidate keeps the design paired — every
  candidate gets the identical initial simplex. Warm-starting each case from
  its own fixed-ξ optimum would be cheaper but would advantage configurations
  near the incumbent, which is the one bias this experiment cannot afford.

**Known limitation.** A single-start Nelder-Mead is not a global optimiser, and
a candidate could in principle lose because its basin is awkward from this
start rather than because it is worse. Stage 4 re-runs the finalists from three
different starts and pools the best per case; if the ranking is start-sensitive
that is where it shows up. This is a check, not a proof.

---

## Stages

Greedy, not factorial: each stage freezes one tier and passes its winner
forward. The greedy risk — that stage 1 picks a structure which is only best at
the ξ it was screened at — is bought off explicitly by stage 4, which re-runs
the top four structures at the frozen ξ.

### Stage 0 — numerics (`s0_numerics`)

Cost against accuracy, settled first because every later score depends on it.

**Candidates**: 75 = cfl ∈ {0.9, 0.7, 0.5, 0.35, 0.25} × ppc ∈ {2, 4, 8, 16,
32} × max_steps ∈ {1500, 3000, 6000}. cfl and max_steps are varied *jointly*
rather than one at a time because they interact: halving cfl halves dt and so
roughly doubles the steps needed to reach the same physical time, which a fixed
max_steps then truncates.

**No inner loop.** Each candidate is evaluated once per case at that case's own
fixed-ξ optimum, via `calibrate apply --params-from`. The question being asked
is "does this numerics setting change the score at the operating point the
harvest will actually run at", not "does it move the optimum" — and a per-case
fitted vector is a far more realistic operating point than one shared vector.

**⚠ Original criterion, SUPERSEDED — kept because the near-miss is the lesson:**

> adopt the *cheapest* setting whose paired mean |ΔΩ_T| against the reference
> (cfl 0.25, ppc 32, max_steps 6000) is below 0.005 with a 95% CI inside ±0.01.

That rule ran, produced a recommendation, and the recommendation was adopted and
launched before the result was checked against the data behind it. It was then
reversed. **Its premise is false: the finest setting is not the most accurate,
so there is no accuracy anchor for "indistinguishable from" to point at.** The
measurement, the corrected criterion and the sequence are below.

**Criterion, as applied**: adopt the **cheapest setting above the resolution
floor**, where the floor is established by measuring where the score collapses,
and where "no cheaper setting is demonstrably less accurate" replaces "closest
to the finest setting". Cost is measured, not assumed — `evals.seconds` is
recorded per row.

Two things worth knowing going in. `released_particles_per_cell` is a **pure
convergence knob**: `initialize_particles.wgsl` divides particle mass by it and
`mass_per_area` multiplies it back, so it changes sampling density and cost but
not the physics. And max_steps is **already not binding at tuned parameters** —
in the fixed-ξ run the median case used 707 steps and only 1% reached 3000 — so
the interesting question is whether 1500 is safe, not whether 6000 helps.

**Mild circularity, disclosed**: the per-case operating points come from a run
at cfl 0.5 / ppc 8 / 3000. If the chosen numerics differ sharply, those optima
would shift. Re-running stage 0 after stage 1 costs two minutes and would catch
it. (In the event the freeze landed on exactly those numerics, so the
circularity closed on itself and cost nothing.)

### Stage 0, as measured: Ω_T is not monotone in resolution

Mean Ω_T at max_steps 3000, per-case fitted μ/slab, n = 104:

| cfl \ ppc | 2 | 8 | 32 |
|---|---|---|---|
| 0.25 | +0.2458 | +0.2738 | +0.2639 |
| **0.50** | +0.2541 | **+0.2828** | +0.2660 |
| 0.90 | +0.1920 | +0.2049 | +0.1938 |

**`ppc` has an interior optimum at 8.** Thirty-two particles per cell score
*worse* than eight; two score worse still. A converged quantity is monotone in
resolution and approaches an asymptote — an interior optimum means the finest
tested setting is not the most accurate, it is simply a setting. Meanwhile
cfl 0.9 is genuinely too coarse (−0.0740 against the reference,
CI [−0.1158, −0.0386]), so there **is** a resolution floor; this is not "coarser
is always better".

Paired contrasts (n = 104, 20k bootstrap):

| contrast | ΔΩ_T | 95% CI |
|---|---|---|
| default (cfl 0.5, ppc 8) − reference (cfl 0.25, ppc 32, ms 6000) | **+0.0167** | [+0.0092, +0.0258] |
| default − the rule's pick (cfl 0.35, ppc 16) | **+0.0118** | [+0.0054, +0.0197] |
| the rule's pick − reference | +0.0049 | [+0.0015, +0.0088] |

Serial cost per simulation: reference 0.564 s, rule's pick 0.261 s, **default
0.140 s**, coarsest-tested 0.070 s. So the default is 1.9× cheaper than the
rule's pick and 4× cheaper than the reference, while scoring above both.

**The immediate neighbourhood of the freeze**, filled in afterwards because the
3×3 grid left `ppc 4` untested and a cheaper adequate setting would have halved
the eventual harvest (paired against the freeze, n = 104):

| setting | ΔΩ_T vs freeze | 95% CI | s/sim |
|---|---|---|---|
| ppc 2 | −0.0287 | [−0.0433, −0.0151] | 0.098 |
| **ppc 4** | **−0.0063** | **[−0.0136, +0.0024]** | 0.102 |
| *ppc 8 (frozen)* | — | — | 0.150 |
| ppc 16 | −0.0108 | [−0.0151, −0.0069] | 0.205 |
| ppc 32 | −0.0167 | [−0.0256, −0.0100] | 0.306 |
| max_steps 1500 | −0.0013 | [−0.0052, +0.0028] | 0.131 |
| cfl 0.7 | −0.0644 | [−0.1099, −0.0279] | 0.114 |

**`ppc 4` was considered and rejected, and the reasoning matters more than the
verdict.** Its CI includes zero, so by a literal reading of the corrected
criterion — "no cheaper setting is demonstrably less accurate" — it qualifies,
at 1.5× less cost. It was rejected anyway because the surrounding structure
contradicts the isolated test: the sequence 8 → 4 → 2 declines monotonically
(0, −0.006, −0.029), which is what genuine under-resolution looks like caught
before it becomes statistically resolvable. Adopting it would repeat the stage-0
error in a new costume — trusting a single interval against a decision rule
while ignoring the response surface the rule sits on. `ppc 4` is on the
shoulder, not the plateau.

`max_steps 1500` is genuinely indistinguishable and 1.15× cheaper, but is *not*
adopted either, for a reason specific to what comes next — see the harvest
caveat below.

### Why the equivalence anchor failed

The rule asks "who is indistinguishable from the most accurate setting", and
answers it by comparing every candidate against the finest one. With an interior
optimum in `ppc`, that reference is not the most accurate setting, so the
question has no referent. The rule then **rejected the cheapest and
highest-scoring candidate for being too *different* from that reference**
(|Δ| = 0.0167 > 0.005) while accepting a costlier one whose Δ happened to fall
under the bound.

Note this is not a multiplicity problem, and correcting for multiplicity would
have made it worse. Equivalence testing is conservative under multiple
comparisons in the direction that matters — noise widens CIs and inflates |Δ|,
both pushing *away* from declaring equivalence — so a Bonferroni-style widening
would only have pushed harder toward the expensive reference. Selection among
the equivalent set is on **cost**, which carries negligible noise and is
orthogonal to score, so there is no winner's curse either. The rule was
mechanically sound and pointed at the wrong thing.

### The freeze, and explicitly why

**Frozen: cfl 0.5, ppc 8, max_steps 3000** — the repo default.

**Not because it scores best.** Choosing numerics to maximise Ω_T is tuning a
knob this document declares non-tunable, and if ppc 8 wins by compensating for a
model bias then adopting it *for that reason* bakes the compensation into every
downstream result. The reasons are:

1. **Cheapest above the resolution floor.** ppc 2 and cfl 0.9 are measurably
   degraded; 8 and 0.5 are not.
2. **No demonstrated accuracy ordering.** Nothing in the measurement establishes
   any tested setting as more accurate than another, so cost and continuity
   decide.
3. **Continuity with every prior measurement.** The +0.2864 fixed-ξ run, the
   identifiability grid and the whole end-to-end ladder were all measured here.
   Freezing elsewhere would stack a second discontinuity on top of the objective
   change, for no measured benefit.
4. **The scoring conventions were already measured here.** The `min_residence`
   plateau was swept at exactly cfl 0.5 / ppc 8, so the convention and the
   numerics agree by construction rather than by assumption.

On point 4, the assumption was checked rather than asserted. Re-sweeping
`min_residence` at cfl 0.35 / ppc 16 reproduces the plateau to ~0.0002 at every
point, peak still at 0.25 (0.125: +0.0027, **0.25: +0.0029**, 0.5: +0.0028,
1.0: −0.0003, 2.0: −0.0170, against +0.0025 / +0.0029 / +0.0029 / −0.0004 /
−0.0148 at cfl 0.5 / ppc 8). **The `count · cfl / ppc` normalisation does what
it was designed to do across a 2× `ppc` and 1.4× `cfl` change** — which is worth
recording as a positive result, since it means the convention is portable across
numerics even though the empirical plateau had only ever been measured at one
setting.

### A dead end, recorded: the footprint-shrinkage hypothesis

The first explanation offered for "coarser scores better" was that coarser
numerics shrink the simulated footprint, which flatters a model that
systematically overshoots. It is wrong, and the measurement says so clearly:

| | rule's pick vs reference |
|---|---|
| filtered footprint cells | median **−0.00%**, mean +0.33%, smaller on 52/104 |
| raw footprint cells | median +0.33%, mean +0.45%, smaller on 44/104 |
| ΔΩ_T on overshooting cases (50/104) | +0.0051 |
| ΔΩ_T on undershooting cases | +0.0048 |
| corr(Ω gain, footprint shrinkage) | **−0.04** |

Footprint size is unchanged to within a coin flip, and the gain is the same size
on overshooting and undershooting cases — under the shrinkage mechanism it would
be concentrated on the former and *negative* on the latter. **The mechanism
behind the interior optimum is still unknown.** It is recorded as unexplained
rather than papered over, because the freeze does not depend on knowing it: the
rationale above rests on cost, floor and continuity, none of which require a
mechanism.

### Sequence of events, kept deliberately

The rule's pick was adopted and `s1_structure` was launched on three boxes
before the stage-0 result had been checked against the surface behind it. The
launch was reversed roughly ten minutes in, partial results were quarantined out
of the ingest path, and the numerics were re-frozen at the default.

The full sequence, since the message ordering is itself part of the lesson:
the rule's recommendation was issued; adoption and launch followed before the
recommendation had been checked against its own surface; the local replication
returned a no-go with the non-monotonicity table; the launch was aborted and
quarantined; a subsequent "do not pause" — sent once the only *invalidating*
concern had been measured away — crossed with the abort already being complete
and was moot on arrival. Two of the three exchanges crossed in flight.

Worth keeping for three reasons. First, **a rule producing an output
mechanically is not the same as a validated result** — the rule ran correctly
and still pointed the wrong way, and nothing in its output signalled that its
premise had failed. Second, **"this is invalid" and "this should be stopped" are
different claims and need separating explicitly**; conflating them is what made
the crossed messages confusing, since the objection was about which numerics to
freeze while the run itself was internally valid either way. Third, the check
that caught it was cheap: replicating stage 0 locally on the real panel took
about three minutes, against three boxes running a full-resolution stage at
roughly twice the necessary cost.

**The generalisable rule: reproduce a stage's recommendation against its own
response surface before freezing it, rather than reading it off the decision
table.** The corollary, from `ppc 4` above: when the surface and a single
confidence interval disagree, believe the surface.

### The one thing still open before the harvest

The freeze needs no revisit *for cost* — `ppc 8` is already the cheap end, with
everything cheaper either degraded or on the shoulder. But stage 0 measured
numerics **at per-case fitted optima**, and the harvest is a *search* that
spends most of its evaluations away from the optimum, where runs are longer and
`max_steps` starts to bind. "Adequate at the optimum" does not establish
"adequate along a search trajectory", and that is the regime the harvest
actually lives in. It is also why `max_steps 1500` is not adopted despite
testing clean here: the median case uses 707 steps *at its optimum*, and says
nothing about the poorly-fitted vectors a search visits first.

The check is cheap and specific: sample off-optimum parameter vectors, compare
the frozen numerics against a finer setting on those, and confirm the ordering
does not change. It is a different question from the one stage 0 answered, not a
re-litigation of the freeze.

### Stage 1 — structure (`s1_structure`)

The discrete choice: which friction model, and which optional components. This
is the phase the exposé describes as selecting the friction model on a reduced
subset before large-scale calibration.

**Candidates**: 48. Four friction models × curvature × particle interaction ×
earth pressure × entrainment, with the earth-pressure flag pruned where
particle interaction is off (identical simulation — see finding 3), which
removes 16 of the naive 64.

Run at ξ = 754, **not** the repo default of 4000. 754 is the value the +0.2864
fixed-ξ per-event run used; 4000 sits in what the campaign's working record calls the least
dissipative corner of the RAMMS range, and screening structures there would
reward whichever structure best compensates for a bad ξ.

**Two metrics, used for two different questions** (finding 2). `omega_cells` is
the only metric comparable across the particle-interaction flag, so it decides
whether any particle-interaction-off candidate is competitive at all. But the
cell-visited footprint is not the production scoring convention — `flow_threshold`
= 0.1 m is — so once particle interaction is settled, the choice *among*
particle-interaction-on candidates is made on `omega`. Run the analysis both
ways; they are one flag apart.

Expectations worth writing down in advance, so that confirming them is not
mistaken for discovery: entrainment was on at the optimum in **1 of 104** cases
in the free 7-parameter search, and is independently suspected of being a
no-op. If the entrainment arms are indistinguishable, that is corroboration,
not a finding.

### Stage 1, as measured: samosAT sweeps, and ξ leaves the model

**Winner: `m3_samosat_c1p1e1n0`** — samosAT, curvature + particle interaction +
earth pressure on, entrainment off. Panel n = 101 (`aval_8124` refused by the
drainage clip across all 48 candidates, so the comparison is clean). All eight
bar-clearing candidates are samosAT variants; the top ten are all
particle-interaction-on.

**The structural claim, under four defensible readings.** `analyze_stage` takes
`MAX(metric)` per case, but the inner loop optimises `omega` — so `MAX(omega_cells)`
selects an evaluation the search passed through rather than its converged
optimum. The choice of convention is worth stating because it moves the absolute
number by 0.03; it does **not** move the conclusion:

| selection | samosAT | Voellmy | gap |
|---|---|---|---|
| `max(omega)` — the inner loop's own optimum | +0.3211 | +0.2852 | **+0.0359** |
| `max(omega_cells)` — what stage 1 ranked on | +0.3053 | +0.2721 | **+0.0332** |
| `omega` at the `omega_cells`-best | +0.2955 | +0.2165 | **+0.0790** |
| `omega_cells` at the `omega`-best | +0.2853 | +0.2249 | **+0.0604** |

**samosAT wins under every reading, by +0.033 to +0.079** (headline contrast
+0.0333, CI [+0.0174, +0.0533], p = 5.8×10⁻⁶). That robustness, not any single
figure, is what the claim rests on. **Prefer `max(omega)` when quoting the
frozen configuration** — it is the value at the point the inner loop actually
converged to — and state the convention beside any number.

**It is not winning by undershooting**, which is the obvious way a structural
result could be an artefact of an overshooting model meeting a tail filter. It
is better-conditioned on every diagnostic:

| | samosAT | Voellmy |
|---|---|---|
| fitted μ median | 0.347 | 0.343 |
| μ at a search bound | **11/101** | 14/101 |
| slab at a search bound | **12/101** | 25/101 |
| median reach error | **−8 m** | −14 m |
| overshooting | 36/101 | 40/101 |

Half the slab boundary hits and better-centred reach. A fit sitting on a bound
is the optimiser running out of room rather than finding a stationary point, so
halving that count is a real improvement in how well the model can express these
events.

### What follows from samosAT

**ξ exits the model, and this is a strict simplification.** samosAT reads no
drag coefficient — its drag term is a fixed log-profile with hard-coded
constants. Stage 2 is therefore skipped by the conditional design in this
document. The parameter that consumed the identifiability work, that costs 3% of
the achievable gain handled correctly and 52% handled backwards, and that the
harvest was going to have to fix before anything else, is simply absent. **The
harvest inner loop is (μ, slab) with no non-identified direction to wander
along.**

**Entrainment is inert *in this configuration*.** The raw footprint is
**bit-identical on 101/101 cases** with the flag on and off; the filtered
footprint differs on 2/101 by at most 2.7×10⁻³, which is the atomics noise floor
documented above. Coulomb is bit-identical on 101/101 even after filtering.

Say *in this configuration*, not *inert*. The likely mechanism is that
entrainment needs an erodible-depth field this harness never supplies, so the
bit is set with nothing to entrain — a different claim from "entrainment physics
is unimportant". **It matters for the 1999-wet track**, where full-depth
avalanches are precisely the case entrainment would have a job, and where the
structure stage re-runs from scratch anyway.

### ⚠ `density` is retracted as dead, and frozen rather than fitted

the campaign's working record records density as having no dynamical effect — 6×10⁻⁶ over
120→450 kg/m³. **That holds for the Voellmy family and not for the frozen
model.** Measured on the panel at the frozen numerics:

| density (kg/m³) | samosAT | Voellmy |
|---|---|---|
| 120 | +0.0105 | +0.0063 |
| 200 | +0.0087 | +0.0062 |
| 300 | +0.0049 | +0.0055 |
| 450 | −0.0019 | +0.0054 |

samosAT swings **0.0124 over the physical range, monotone, 104/104 cases
changed, up to 0.44 on a single case**. Voellmy swings 0.0009 — the
cancellation, holding.

**Frozen at `density = 200`. It must not enter the inner loop**, for two
independent reasons:

1. **The dependence is a shader inconsistency, not physics.**
   `compute_particles.wgsl:40` declares `const density = 200.0`, and the samosAT
   normal-friction term uses that constant for its `rs` ratio while
   `mass_per_area` uses the real `snow_density`. The sensitivity is the gap
   between two values that were meant to be the same one. Fitting it fits a bug.
   Raised as an upstream issue (`notes_for_markus.md` item 25).
2. **It is degenerate with μ.** samosAT's shear is
   `normal_stress · μ · (1 + rs0/(rs0+rs))` with `rs ∝ 1/ρ`, so density is a
   friction multiplier — it does what μ already does. Fitting both per-event
   would recreate exactly the non-identified direction that losing ξ just
   removed.

Freezing at 200 also makes the two uses of density agree, restoring the intended
samosAT formulation. `--density 200` belongs in the base-args of every stage
from here **and of the harvest**; it is load-bearing now in a way it never was
under Voellmy.

### ⚠ μ means something different now

Under samosAT the effective friction runs from μ (fast) to 2μ (slow), so μ is
roughly "half the low-speed friction coefficient" rather than the Coulomb
friction it denotes in the Voellmy family. The fitted values happen to sit in a
similar range (median 0.347 against 0.343), which makes this easy to miss.

**Every μ-based result in the campaign's working record was measured on Voellmy fits and
does not transfer**: the terrain-descent correlation (r = +0.42), the
target-noise ceiling (R² ≈ 0.95 for μ), the `aval_shape` leak (+0.25 to +0.31),
and the per-event vs global gap decomposition. The conditions model will train
on samosAT-μ targets. **Those feature-side conclusions need re-measuring against
the new targets before any of them is relied on** — they are not wrong, they are
about a different quantity.

### Reading the absolute number against the ceiling

The winner's `max(omega)` of +0.3211 sits inside the +0.28 to +0.36 inter-mapper
agreement band in `notes_for_markus.md` item 22, which supports that item's
reading: at 1.5 m imagery we are at the noise floor of the ground truth, and
further Ω_T gains here are largely fitting digitisation style rather than
physics.

**What may be claimed**: samosAT beats Voellmy by a paired, significant margin on
the same panel under the same objective, robust across every selection
convention; and the absolute value lands in the ground-truth agreement band.

**What may not**: that this beats the earlier +0.295 per-event oracle. That
number predates both stage −1 fixes, which changed the footprint *and* the
release, was measured on 105 cases rather than 101, and used a different
selection convention. Nor that it "exceeds the ceiling" — the band estimates how
well independent mappers agree with *a* reference, whereas Ω_T here measures
agreement with *one specific* mapping, and the objective has since been modified
in a direction that mechanically raises the score. **Report the absolute against
the band as context; rest every claim on the within-campaign paired contrasts,
which are unaffected by all of this.**

### Stage 2 — the ξ profile (`s2_xi`)

**Candidates**: 9 log-spaced values — 250, 400, 650, 1000, 1600, 2600, 4000,
6500, 10000 — spanning the search bounds in `calibrate.rs`. Each gets a **full
per-case (μ, slab) recalibration**, which makes this a profile: it is exactly
the procedure the campaign's working record measured at 97% of the free-ξ oracle, run at
nine values instead of one.

Runs only if the stage-1 winner is Voellmy or VoellmyMinShear.

**Criterion, and it is deliberately not the argmax.** ξ is the parameter the
identifiability study found *not* identified — median admissible window 2.11×,
one event spanning 19.6×. The profile will be flat, and its argmax will be
noise. Take the region within 0.01 of the maximum and choose the **geometric
centre of that plateau**, which is the choice most robust to per-case variation.
Record the plateau's width: it is the honest statement of how well ξ is
determined at the panel level.

### Stage 3 — remaining constants (`s3_ifa`, `s3_roughness`)

`s3_ifa`: internal friction angle over {15, 20, 25, 30, 35, 40, 45}, full inner
loop, **only if the stage-1 winner has particle interaction and earth pressure
both on**. Note that it interacts with `basal_friction_angle` (fixed at 25.0 and
not exposed by `Params`), so what is being frozen is the angle *at that basal
value*.

`s3_roughness`: the null check from finding 1. Three values, one evaluation per
case, expected bit-identical. Verify with equality, not statistics:

```sql
SELECT case_name, COUNT(DISTINCT omega) FROM evals
WHERE stage='s3_roughness' GROUP BY case_name HAVING COUNT(DISTINCT omega) > 1;
```

An empty result confirms the reading. Any row means stop and re-read the code.

### Stage 3, as measured: ifa is a validity edge, not a peak

**Incumbent kept: `ifa = 40`.** No candidate cleared ΔΩ_T ≥ 0.02 with a CI
excluding zero. But the profile is **not flat**, which this document predicted it
would be, and the reason is worth more than the decision.

Measured (n = 101, per-case inner loop): ifa 15/20/25/30 significantly worse
(−0.0162 to −0.0063, CIs excluding zero, p 0.0006–0.03); ifa 35/45
indistinguishable from 40; total spread 0.0163.

**The mechanism.** `grid_physics.wgsl:151` computes

```
inside = 1.0 − cos²(ifa) / cos²(basal_friction_angle)
root   = sqrt(max(inside, 0.0))
```

`inside > 0` requires **ifa > basal_friction_angle**, fixed at **25°**. Below
that the `max(inside, 0.0)` safety clamp fires, `root` collapses to zero, and
the earth-pressure coefficient loses its dependence on the flow state entirely.
The arithmetic matches the measurement exactly: ifa 25 gives `inside` = 0
precisely, ifa 30 gives root 0.295, ifa 40 gives 0.535, ifa 45 gives 0.625 —
so 15/20/25 are clamped, 30 sits just above the boundary with a small root, and
35–45 is the healthy plateau.

This is the Rankine / Savage-Hutter condition that internal friction must exceed
basal friction for active and passive states to be defined. **So the shape is a
degeneracy boundary with a plateau above it, not a peak.** The correct claim for
the frozen vector is therefore *not* "40 is optimal" but **"40 sits comfortably
inside the valid region, and anything in 35–45 is equivalent"** — a weaker
statement about ifa and a stronger one about the freeze's robustness.

The "expected flat" prior was wrong for an instructive reason: ifa *does* enter
only through earth pressure, as predicted, but earth pressure has a validity
edge nobody had looked for.

**Density-compensation check — pre-registered, run, null.** Because ifa and the
now-frozen density are both resistance-related, the risk was that ifa's optimum
was compensating for density = 200. Measured at the frozen vector:

| density | ifa 20 | ifa 30 | ifa 40 | ifa 45 | argmax |
|---|---|---|---|---|---|
| 150 | −0.0139 | +0.0033 | **+0.0096** | +0.0061 | **40** |
| 200 | −0.0165 | +0.0014 | **+0.0087** | +0.0043 | **40** |
| 300 | −0.0238 | −0.0043 | **+0.0049** | +0.0019 | **40** |

The optimum is 40 at every density and the profile shape is stable (the ifa 20
penalty relative to each density's own ifa 40 is −0.0235 / −0.0252 / −0.0288).
Only the low-end penalty deepens slightly with density, which does not move the
optimum. **No compensation.** Consistent with the mechanism: density scales
*basal* friction while ifa scales *lateral* earth pressure, so unlike μ and
density these two are not competing for the same job.

### ⚠ `basal_friction_angle` is a global constant this experiment never froze

Surfaced by the mechanism above. `basal_friction_angle` is fixed at **25.0** in
`SimSettings::new()` and is **not exposed in `Params` or `ParamOpts`**, so it was
never a candidate in any tier and no stage swept it. It co-determines the
earth-pressure coefficient *and* the valid range of ifa.

It sat at its default throughout, so nothing measured here is invalidated and it
did not block the freeze. But the frozen vector must record it explicitly as
**fixed at its default and never swept**, rather than leaving a reader to assume
it was considered and chosen. Two consequences:

- It belongs in the tier-2 candidate set for the per-dataset extensions.
- **It is a prime suspect for the 1999-wet track**, where basal conditions are
  exactly what differs between a dry slab and a wet full-depth avalanche — and
  where, per the extension design, the structure stage re-runs from scratch
  anyway.

More generally: **a constant that is not in `Params` is invisible to this entire
design**, since every tier is defined over `Params` fields. So the check was run
rather than assumed — every `SimSettings` field absent from `Params`, against
what the shaders actually read:

| constant | default | read by the frozen model? |
|---|---|---|
| `basal_friction_angle` | 25.0 | **yes** — `grid_physics.wgsl:53`, with `ifa`, into the earth-pressure coefficient |
| `velocity_threshold` | 1e-6 | **yes, three sites**, but see below |
| `n0` | 70.0 | no — its only use is commented out (`compute_particles.wgsl:277`) |
| `i0`, `mu0`, `mu2`, `grain_diameter` | 0.29 / 0.38 / 0.65 / 0.002 | no — read only inside the μ(I) branch (model 5), which we do not use |

`velocity_threshold` needed a qualifier, and then a measurement. It is read at
three live sites: at `compute_particles.wgsl:142` it enters as
`max(1e-3, velocity_threshold)`, so the 1e-6 default is masked by the floor
there, but at lines 220 and 256 it is read raw as the "particle has stopped" and
"no friction below this speed" cutoffs. At 1e-6 those are effectively zero, so
the *argument* was that it must be inert — the same shape of argument that was
wrong about `roughness_threshold` and wrong about ifa being flat.

**Measured instead: inert.** Full 105-case panel at the frozen vector, 1e-6
against 1e-3 — mean Ω_T identical to four decimals, **94/104 cases
bit-identical**, the remaining 10 differing by ≤0.0053, inside the documented
atomics noise and two orders below the decision band.

**Final ledger for the six hidden constants**: one live and never swept
(`basal_friction_angle`, recorded as such in the frozen vector), one **verified
inert by measurement** (`velocity_threshold`), and four genuinely unreachable by
the frozen model.

### Stage 4 — confirmation (`s4_confirm`)

The top four structures from stage 1, re-run at the frozen ξ and ifa, with
**three inner-loop starts** pooled per case. This does three jobs at once: it
catches the greedy-ordering risk (a structure that was only best at ξ = 754),
it quantifies inner-loop start sensitivity, and it produces the final margin.

Multi-start is implemented by running the stage three times under different
`--run-tag` values with different `--mu`/`--slab` in the manifest. The job files
differ, the candidate identity does not, and because the analysis takes the
best inner-loop score per (candidate, case) the starts pool automatically.

---

### Stage 4, as measured: v1 frozen

Four candidates — the frozen structure, two samosAT alternates, and the
pre-campaign Voellmy incumbent as the baseline — each with three pooled
inner-loop starts, n = 101.

| candidate | mean Ω_T | Δ vs incumbent | 95% CI | p | Δ cal | Δ val |
|---|---|---|---|---|---|---|
| **m3_samosat_c1p1e1n0** | **+0.3313** | **+0.0207** | [+0.0107, +0.0313] | 6.6×10⁻⁵ | +0.0271 | +0.0104 |
| m3_samosat_c0p1e1n0 | +0.3289 | +0.0183 | [+0.0081, +0.0290] | 0.0019 | +0.0251 | +0.0074 |
| m1_voellmy_c1p1e1n0 *(incumbent)* | +0.3106 | — | — | — | — | — |
| m3_samosat_c1p1e0n0 | +0.3093 | −0.0013 | [−0.0155, +0.0128] | 0.92 | +0.0064 | −0.0136 |

Criteria 1–3 pass: rank preserved, the margin clears 0.02 with a CI excluding
zero, and cal/val agree in sign. **Frozen as v1.**

A detail worth keeping: the earth-pressure-off arm (`c1p1e0n0`) is the one that
shows a **cal/val sign flip**. That is independent corroboration of the ifa
validity-edge finding — ifa only bites through earth pressure, and earth
pressure turns out to be doing real work, so the arm that removes it is also the
arm that destabilises across subgroups.

### ⚠ Criterion 4 was operationalised on the wrong quantity

As written, criterion 4 asked whether the **winner's** multi-start lift was under
0.02. Measured lifts (s4 three-start pooled against s1 single-start, same
candidates, same panel, identical numerics):

| candidate | mean lift | median lift | max single case |
|---|---|---|---|
| m1_voellmy_c1p1e1n0 *(incumbent)* | **+0.0254** | +0.0037 | +0.3848 |
| m3_samosat_c0p1e1n0 | +0.0121 | +0.0024 | +0.1563 |
| m3_samosat_c1p1e1n0 *(winner)* | +0.0101 | +0.0022 | +0.1929 |
| m3_samosat_c1p1e0n0 | +0.0063 | +0.0011 | +0.1133 |

The winner passes; the **incumbent** exceeds the threshold. Taken literally the
criterion fires on a number that threatens nothing, because **the winner's own
lift is nearly irrelevant**: if a candidate lifts under multi-start and still
leads on an equalised budget, nothing is wrong. The s4 delta above is computed
with three starts on *both* sides, so it already reflects the winner beating the
best version of the incumbent that can be produced.

What criterion 4 was actually protecting against is different: **s1 selected
among 48 candidates at single start**, so a non-finalist with a large lift could
have belonged in the top four and never been carried into s4. The correct form
is therefore

> **no non-finalist sits within the largest observed multi-start lift of the
> winner's confirmed s4 score.**

Under the written form the stage fails on an irrelevant number; under the
correct form it passes. Recorded because the per-dataset extensions reuse these
criteria, and because it is the same failure as stage 0 — a rule that ran
correctly and pointed at the wrong quantity.

**The corrected criterion, evaluated.** All 48 s1 candidates ranked on
`max(omega)`, against the lift each would need to beat the winner's confirmed
+0.3313:

| candidate | s1 `max(omega)` | lift needed | |
|---|---|---|---|
| m3_samosat_c1p1e1n0 | +0.3211 | +0.0102 | finalist |
| **m3_samosat_c1p1e1n1** | +0.3210 | +0.0103 | entrainment twin |
| m3_samosat_c0p1e1n0 | +0.3167 | +0.0146 | finalist |
| **m3_samosat_c0p1e1n1** | +0.3167 | +0.0146 | entrainment twin |
| m3_samosat_c1p1e0n0/n1 | +0.3030 | +0.0283 | finalist / twin |
| m2_voellmyminshear_c1p1e1n0/n1 | +0.3007 | +0.0306 | |

**Exactly two non-finalists need less lift than the largest ever observed
(+0.0254), and both are the entrainment-ON twins of finalists already in
stage 4** — provably the same simulation, bit-identical raw footprints on
101/101 cases. They cannot leapfrog because they are not distinct candidates.

The nearest genuinely distinct non-finalist needs **+0.0306**, exceeding the
largest measured lift by 20%, and the mechanism argues against reaching it: the
incumbent's +0.0254 was a mean dragged by a handful of trapped cases (median
+0.0037), not a uniform shift, so a family-wide lift that large would require
more trapped cases than Voellmy itself had. **Risk bounded and small, not zero
— recorded as a limitation rather than an open question.**

### The rule this campaign kept re-learning

Every stage of this experiment produced at least one confident, plausible,
well-reasoned claim that a cheap measurement then contradicted:

| the argument | what measuring it showed |
|---|---|
| "the finest numerics is the most accurate reference" | `ppc` has an interior optimum at 8; the reference is beaten by the default |
| "`min_residence` = 1.0 means one cell-equivalent, so it's principled" | 1.0 is past the plateau; 0.25 is on it |
| "the drainage clip is correct — the unit tests pass" | tautological no-op on 105/105 real cases |
| "shallow-peaked ifa might be compensating for frozen density" | optimum is 40 at every density; null |
| "the other hidden constants are unreachable" | `velocity_threshold` is read at three live sites |
| "`velocity_threshold` must be inert at 1e-6" | inert — but now measured, 94/104 bit-identical |
| "a non-finalist might leapfrog under multi-start" | the only two close enough are provably identical twins |

None of those arguments were careless. Several were mine, and the reasoning was
sound given what was known. **The pattern is that plausible mechanical reasoning
about this solver fails often enough that it cannot be trusted as evidence when
a measurement is available** — and on this panel a measurement is almost always
available for a few minutes of GPU time, against decisions that shape a campaign
and a harvest.

The operational form, for whoever runs the per-dataset extensions:
**reproduce a stage's recommendation against its own response surface before
freezing it, and when a surface and a single confidence interval disagree,
believe the surface.**

## The freeze criterion

A challenger replaces the incumbent only if **both** hold:

1. paired mean ΔΩ_T ≥ **0.02** over the panel, and
2. the 95% paired bootstrap CI **excludes zero**.

Otherwise the incumbent stands. Ties within the band break toward **lower cost,
then fewer enabled components** — a flag that buys nothing measurable should be
off, and the incumbent is the simpler configuration by construction.

0.02 is not arbitrary: it is the band the identifiability work already uses to
call two parameter vectors indistinguishable. A global knob that moves the panel
mean by less than the width at which the repo declines to distinguish two
parameter vectors is not worth freezing to a non-default value.

**Power.** The paired difference standard deviation between two nearby
candidates is expected around 0.10 — smaller than the 0.146 implied by the
end-to-end contrast CI in the campaign's working record, because the inner loop absorbs
much of the difference. At n = 102 that is a standard error near 0.0099, so
0.02 is roughly a 2σ effect. **Measure it rather than trusting it**: stage 0
produces exactly such paired comparisons, so compute the realised paired sd from
the stage-0 rows before reading stage 1. If it is much above 0.10, the panel
cannot resolve 0.02 and the band must widen — which is a finding about the
experiment, not a reason to lower the bar quietly.

`analyze_stage.py` applies this rule and prints the verdict, rather than
printing a table and leaving the reader to apply it.

---

## Budget

Panel 102 cases, inner budget 40. Simulation counts are upper bounds: the median
inner search collapses at ~34 evaluations, so realised counts run about 15%
lower.

| stage | candidates | sims (cap) | sims (realised ≈) |
|---|---|---|---|
| `s-1_fixval` | 8 runs | 816 | 816 |
| `s0_rebaseline` | 1 | 102 | 102 |
| `s0_numerics` | 75 | 11,900\* | 11,900 |
| `s1_structure` | 48 | 195,840 | 166,500 |
| `s2_xi` | 9 | 36,720 | 31,200 |
| `s3_ifa` | 7 | 28,560 | 24,300 |
| `s3_roughness` | 3 | 306 | 306 |
| `s4_confirm` | 4 × 3 starts | 48,960 | 41,600 |
| **total** | | **323,200** | **276,700** |

Stage −1 is 8 whole-panel `run` passes (filters on, clip off, and six residence
values) at one simulation per case — under a minute at any plausible `T`, and it
gates everything.

\* 7,650 raw simulations, weighted ×1.55 for the mean `ppc` cost relative to the
ppc = 8 baseline.

**Wall clock is `N / T`,** where `T` is the box's aggregate throughput in
simulations per second. `T` is unknown until the 8× RTX 4070 Super box is
benchmarked — the 23.4 cases/s/GPU and ~34.6 evals/s/GPU figures in
the benchmark notes record RTX 5090 numbers and a 4070 Super is slower. Two
bracketing scenarios:

| stage | T = 90 sims/s | T = 180 sims/s |
|---|---|---|
| `s0_numerics` | 2.2 min | 1.1 min |
| `s1_structure` | 36.3 min | 18.1 min |
| `s2_xi` | 6.8 min | 3.4 min |
| **stages 0–2 subtotal** | **45.3 min** | **22.6 min** |
| `s3_ifa` + `s3_roughness` | 5.4 min | 2.7 min |
| `s4_confirm` | 9.1 min | 4.5 min |
| **total** | **59.7 min** | **29.9 min** |

Stages 0–2 stay inside the hour in both scenarios. `T` is a config value
(`--throughput` on `run_stage.py`, `DEFAULT_THROUGHPUT` at the top of that
file), used for the ETA only and never for scheduling — the driver's actual
concurrency is `--gpus × --per-gpu`.

**Measure `T` before committing to the plan.** The dry run prints the estimate;
the first completed stage prints the realised sims/s. If `T` comes in far below
90, cut stage 1 first — screening the 48 structures on a stratified third of the
panel and confirming the top 8 on the full panel costs about a third as much,
at the price of raising stage 1's detectable effect from 0.02 to roughly 0.04.

---

## The database

One SQLite file, built by `ingest_evals.py` from the per-evaluation JSONL that
`calibrate` now writes for **every** subcommand — previously only `per-event`
logged individual evaluations, so `sweep`, `search`, `grid`, `run` and `apply`
threw theirs away and kept only summaries.

### Schema

| table | grain | purpose |
|---|---|---|
| `sources` | one job log | provenance: path, stage, candidate, git hash, host, row count, mtime, ingest time |
| `candidates` | one global candidate | the full `Params` JSON once, plus the twelve knobs lifted into columns for querying |
| `evals` | **one simulation** | the scores, the nuisance parameters, and the diagnostics |

`evals` columns: `source_id`, `stage`, `candidate_id`, `case_name`, `iter`
(inner-loop index within the job), `ok`, `err`, `mu`, `slab`, then `omega`,
`omega_cells`, `hwri_l1`, `hwri_l05`, `release_only_omega`, `alpha`, `beta`,
`gamma`, then `reach_err_m`, `sim_cells`, `ref_cells`, `release_cells`, `steps`,
`max_velocity`, `clipped_at_edge`, `sim_flags`, `seconds`. Indexed on
`(stage, candidate_id)`, `(stage, case_name)` and `source_id`.

The full parameter vector lives once per candidate rather than on every row,
which is most of the 3.9× compression against the raw JSONL.

**Ingest is idempotent** — re-ingesting a source file deletes and reloads its
rows — so it can run repeatedly while a stage is still going. It reads only a
`<job>.evals.jsonl` that has a completed sibling `<job>.json`, so the partial
log of a killed job is never mistaken for data. WAL mode keeps the database
readable while a stage writes into it.

### Size, measured on 87,540 synthetic rows through the real ingest path

| | JSONL | SQLite |
|---|---|---|
| per evaluation | 1,055 B | **270 B** |
| this experiment (322k) | 340 MB | **87 MB** |
| 20k-event harvest (1.1M) | 1.16 GB | **297 MB** |

Comfortable against the box's 1.47 TB. The JSONL is the bulkier artefact and is
worth `rsync`-ing off before teardown regardless, since it is the raw record.

**Rasters are not stored per evaluation.** Footprints come from `calibrate dump`
for the final winning vectors only — roughly 1 MB per case at the median grid
size of 43k cells, so ~100 MB for one pass over the panel.

### Parquet export

`export_parquet.py` writes columnar copies with the candidate knobs joined on:
`evals_<stage>.parquet`, `case_best_<stage>.parquet` (the per-(candidate, case)
best inner-loop score — the table every comparison actually operates on), plus
`candidates.parquet` and `sources.parquet`. SQLite stays the write path because
it takes concurrent appends and survives a hard kill; Parquet is the read path
for the report and for anything using pandas/polars/duckdb. Measured on the
synthetic campaign: **zstd Parquet is ~60 B/eval against SQLite's ~270 B**, so
the whole campaign exports to roughly 20 MB.

### Getting it off the box

Checkpoint the SQLite database before copying it off, or the `-wal` file holds
committed rows the `.sqlite` file does not yet have:

```bash
ssh <host> "sqlite3 /root/data/calibration.sqlite 'PRAGMA wal_checkpoint(TRUNCATE);'"
```

**The tile cache is worth archiving.** A cold swissALTI3D fetch is 5-10 s/case
sequentially, which dominates a short campaign, while the cache itself is only a
few hundred MB. Carrying `dtm_cache.zarr` between hosts means the next box
starts warm.

### What the final report needs from the database

The HTML report is assembled separately, but everything it needs is captured
here: per-stage candidate definitions with full params (`candidates`), every
simulation with scores and timings (`evals`), job-level provenance including git
hash and host (`sources`), the fix before/after artifacts
(`omega_raw`/`sim_reach_raw_m`/`release_clipped_frac` on every row, plus
`fix_validation.json`), the per-stage decision tables written by
`analyze_stage.py --out`, and the frozen constants. The one thing not in the
database is rasters: the report's before/after footprint figures come from
`calibrate dump` run at the frozen vector with the filters on and off.

---

## How to run it

Assumes the box is provisioned, Vulkan verified, code and data `rsync`-ed, tile
cache warm, and everything running inside `tmux` on the compute host.
Paths below match that runbook's layout.

```bash
ssh <host> -t 'tmux attach -t main || tmux new -s main'
cd ~/avalanchers && cargo build --release --bin calibrate

# ⚠ RUN EVERY STAGE COMMAND FROM THE DATA DIRECTORY. `calibrate` resolves each
# case's `shp` path relative to the process working directory (they are stored
# relative, e.g. "cases/aval_52.shp"), so from anywhere else every case fails to
# prepare and the run dies having read nothing. This cost the first rebaseline
# attempt on the box. The Python drivers take `--cwd` for exactly this reason;
# the bare `calibrate` invocations below rely on the `cd`, and the script
# paths below are absolute so they still resolve from here.
cd ~/data

# 0. Benchmark T before trusting any estimate below.
python3 ~/avalanchers/python_scripts/run_stage.py --manifest ~/data/stages/s0_numerics.json \
  --calibrate-bin ~/avalanchers/target/release/calibrate \
  --cases ~/data/cases100.json --cache ~/data/dtm_cache.zarr \
  --cwd ~/data --out-dir ~/data/results --gpus 8 --per-gpu 8 --dry-run

# 1. Generate the manifests (writes the stage-0 operating points too).
python3 ~/avalanchers/python_scripts/make_stages.py \
  --cases ~/data/cases100.json --out-dir ~/data/stages \
  --perevent-json ~/avalanchers/campaign/analysis/perevent_fixedxi.json

# 2. The cal/val map the analysis uses for its consistency check.
python3 -c "import json; cs=json.load(open('$HOME/data/cases100.json')); \
json.dump({c['name']: c.get('split','') for c in cs}, open('$HOME/data/splits.json','w'))"
```

Then, per stage — run, ingest, decide. `RS` and `AS` are shorthand for the two
long invocations:

```bash
RS="python3 ~/avalanchers/python_scripts/run_stage.py \
  --calibrate-bin ~/avalanchers/target/release/calibrate \
  --cases ~/data/cases100.json --cache ~/data/dtm_cache.zarr \
  --cwd ~/data --out-dir ~/data/results --gpus 8 --per-gpu 8"
ING="python3 ~/avalanchers/python_scripts/ingest_evals.py --results-dir ~/data/results \
  --db ~/data/calibration.sqlite --repo ~/avalanchers"
AS="python3 ~/avalanchers/python_scripts/analyze_stage.py --db ~/data/calibration.sqlite \
  --splits ~/data/splits.json"

# --- Stage -1: validate the two objective fixes. NOTHING runs before this. ---
python3 ~/avalanchers/python_scripts/validate_fixes.py \
  --calibrate-bin ~/avalanchers/target/release/calibrate \
  --cases ~/data/cases100.json --cache ~/data/dtm_cache.zarr \
  --cwd ~/data --out-dir ~/data/fixval
#   Read four things before continuing:
#     - the tail filter must cut |reach error| on the OVERSHOOTING cases, not
#       just nudge mean omega (it also cuts undershooting cases, and across the
#       whole panel at tuned parameters those largely cancel);
#     - SUPERSEDED: "the ridge clip must visibly cut the release on the known
#       leak cases". Wrong yardstick -- see "Why the crossing_analysis 12 is not
#       the acceptance list" above. Instead: the clipped_frac distribution must
#       be mostly-zero with a short tail, and the D8 verdicts on the tail must
#       adjudicate case by case;
#     - the SEVERE (>50% clipped) and REFUSED lists must be short, and every
#       case on them inspected by hand -- they are misregistration, and a long
#       list means the panel itself needs revisiting before any tuning;
#     - the residence sweep must show a plateau covering 0.125-0.5 with the
#       default on it, and a falloff above; a default sitting on the falloff is
#       not defensible. (This line previously said "a plateau around 1.0" --
#       measurement moved the default to 0.25 and put 1.0 on the falloff.)

# --- Stage 0a: re-establish the incumbent under the NEW objective ------------
#   Every Omega_T in the campaign's working record predates the fixes and is not comparable.
cd ~/data && ~/avalanchers/target/release/calibrate --cases ~/data/cases100.json \
  --cache ~/data/dtm_cache.zarr --stage s0_rebaseline --candidate incumbent \
  --out ~/data/rebaseline.json run --model 1 --flags 7 --xi 754 --mu 0.36 --slab 0.56

# --- Stage 0: numerics -------------------------------------------------------
$RS --manifest ~/data/stages/s0_numerics.json
$ING
$AS --stage s0_numerics --incumbent cfl0.25_ppc32_ms6000 --equivalence 0.005
#   -> equivalence mode ranks by measured s/sim against the FINEST setting.
#   ⚠ Its answer is advisory only. Read the response surface before freezing:
#      the finest setting is not the most accurate (ppc has an interior optimum
#      at 8), so the equivalence anchor has no referent. See "Stage 0, as
#      measured" above. Freeze the cheapest setting above the resolution floor.
#      FROZEN 2026-07-28: cfl 0.5 / ppc 8 / max_steps 3000 (the repo default).
python3 -c "
import sqlite3,collections
c=sqlite3.connect('$HOME/data/calibration.sqlite')
q=c.execute(\"SELECT cand_cfl,cand_ppc,cand_max_steps,AVG(omega),AVG(seconds),COUNT(*) \
FROM (SELECT e.*,c.cfl cand_cfl,c.released_particles_per_cell cand_ppc,c.max_steps cand_max_steps \
FROM evals e JOIN candidates c ON c.candidate_id=e.candidate_id AND c.stage=e.stage \
WHERE e.stage='s0_numerics' AND e.ok=1) GROUP BY 1,2,3 ORDER BY 4 DESC LIMIT 10\")
print(f'{\"cfl\":>6}{\"ppc\":>6}{\"steps\":>8}{\"mean omega\":>12}{\"s/sim\":>9}')
for cfl,ppc,ms,om,sec,n in q: print(f'{cfl:>6}{ppc:>6}{ms:>8}{om:>+12.4f}{sec:>9.3f}')
"   # the surface itself, top 10 -- read this, not just the rule's verdict
#   Also compute the realised paired sd here, which is what says whether the
#   panel can resolve the 0.02 band at all:
python3 -c "import sqlite3,numpy as np,itertools; c=sqlite3.connect('$HOME/data/calibration.sqlite'); \
d={}; [d.setdefault(a,{}).__setitem__(b,v) for a,b,v in c.execute( \
\"SELECT candidate_id,case_name,MAX(omega) FROM evals WHERE stage='s0_numerics' GROUP BY 1,2\")]; \
ks=sorted(d); print('paired sd:', np.median([np.std([d[a][n]-d[b][n] for n in d[a] if n in d[b]],ddof=1) \
for a,b in itertools.islice(itertools.combinations(ks,2),200)]))"

# --- Stage 1: structure ------------------------------------------------------
#   --base-args is what bakes the frozen numerics into every candidate. Without
#   it every structure would silently run at Params::default() numerics instead;
#   make_stages.py ignored --base-args for s1 until 6510099, so check the DB
#   after launch (query below) rather than trusting the flag was honoured.
python3 ~/avalanchers/python_scripts/make_stages.py --cases ~/data/cases100.json \
  --out-dir ~/data/stages --stage s1_structure \
  --base-args "--cfl 0.5 --ppc 8 --max-steps 3000" \
  --perevent-json ~/avalanchers/campaign/analysis/perevent_fixedxi.json
#   Verify the frozen numerics actually reached the candidates:
sqlite3 ~/data/calibration.sqlite "SELECT DISTINCT cfl, released_particles_per_cell, \
  max_steps FROM candidates WHERE stage='s1_structure';"   # expect 0.5 | 8 | 3000
$RS --manifest ~/data/stages/s1_structure.json
$ING
$AS --stage s1_structure --incumbent m1_voellmy_c1p1e1n0 --metric omega_cells

# --- Stage 2: the xi profile (only if the winner is model 1 or 2) ------------
python3 ~/avalanchers/python_scripts/make_stages.py --cases ~/data/cases100.json \
  --out-dir ~/data/stages --stage s2_xi --base-args "--model 1 --flags 7"
$RS --manifest ~/data/stages/s2_xi.json
$ING
$AS --stage s2_xi --incumbent xi650
#   -> centre of the plateau within 0.01 of the max, not the argmax.

# --- Stage 3: ifa, and the roughness null check ------------------------------
python3 ~/avalanchers/python_scripts/make_stages.py --cases ~/data/cases100.json \
  --out-dir ~/data/stages --stage s3_ifa --base-args "--model 1 --flags 7 --xi 650"
$RS --manifest ~/data/stages/s3_ifa.json
$RS --manifest ~/data/stages/s3_roughness.json
$ING
$AS --stage s3_ifa --incumbent ifa40
sqlite3 ~/data/calibration.sqlite "SELECT case_name, COUNT(DISTINCT omega) \
  FROM evals WHERE stage='s3_roughness' GROUP BY case_name \
  HAVING COUNT(DISTINCT omega) > 1;"    # must return nothing

# --- Stage 4: confirmation, three inner-loop starts --------------------------
#   Finalists are the top four ids printed by the stage-1 analysis.
python3 ~/avalanchers/python_scripts/make_stages.py --cases ~/data/cases100.json \
  --out-dir ~/data/stages --stage s4_confirm --base-args "--xi 650 --ifa 40" \
  --finalists "m1_voellmy_c1p1e1n0,m1_voellmy_c0p1e1n0,m1_voellmy_c1p1e0n0,m2_voellmyminshear_c1p1e1n0"
$RS --manifest ~/data/stages/s4_confirm.json --run-tag start1 --inner-start 0.36,0.56
$RS --manifest ~/data/stages/s4_confirm.json --run-tag start2 --inner-start 0.20,0.30
$RS --manifest ~/data/stages/s4_confirm.json --run-tag start3 --inner-start 0.50,1.00
$ING
$AS --stage s4_confirm --incumbent m1_voellmy_c1p1e1n0 --out ~/data/s4_table.json

# --- Freeze: dump footprints for the winner, then get everything off the box --
#   Substitute the frozen vector for the placeholder below; it is not known
#   until stage 4 reports. The example shows the incumbent's own values.
cd ~/data && ~/avalanchers/target/release/calibrate --cases ~/data/cases100.json \
  --cache ~/data/dtm_cache.zarr --out ~/data/frozen.json --eval-log none \
  dump --dir ~/data/dump_frozen \
  --model 1 --flags 7 --xi 650 --ifa 40 --cfl 0.5 --ppc 8 --max-steps 3000
#   and the same vector with the filters off, for the report's before/after
#   footprint figures:
cd ~/data && ~/avalanchers/target/release/calibrate --cases ~/data/cases100.json \
  --cache ~/data/dtm_cache.zarr --out ~/data/frozen_raw.json --eval-log none \
  dump --dir ~/data/dump_unfiltered \
  --model 1 --flags 7 --xi 650 --ifa 40 --min-residence 0 \
  --require-connected 0 --clip-drainage 0

# --- Export, then get everything off the box --------------------------------
python3 ~/avalanchers/python_scripts/export_parquet.py --db ~/data/calibration.sqlite \
  --out-dir ~/data/parquet
sqlite3 ~/data/calibration.sqlite 'PRAGMA wal_checkpoint(TRUNCATE);'
```

Then copy the checkpointed database, the Parquet exports and the tile cache
off the host before it is destroyed.

Every stage is resumable: re-run the identical command and only jobs still
missing a result file are submitted. Check `nvidia-smi` directly rather than
trusting progress output — an idle box has cost this project twenty minutes
before.

---

## Planned extensions: per-dataset freezes

Everything above freezes constants against **one** dataset — SPOT6, 24 Jan 2018,
a single dry-slab storm. That makes "general" an untested word. Two further
mappings exist, and once the extraction pipeline produces panels from them
(schema-compatible with `cases100.json`) the same staged freeze runs on each:

| dataset | polygons | why it is different |
|---|---|---|
| 2018 (done) | 18,737 → 105 panel | dry slab, one storm |
| 2019 | 6,041 | dry slab, **different storm** |
| 1999 | 11,120 | **~40% wet / full-depth** — a different flow regime |

Plus one **pooled** freeze across all three. The cross-dataset comparison of the
frozen constants is then a first-class result: agreement earns the word
"general"; disagreement is a discovered condition-dependence that the conditions
model has to absorb rather than a failure. ξ under wet snow is the prime
suspect.

**A first pass ran on 2026-07-28**, before the full per-dataset freezes: not a
re-freeze on each dataset, but a *transfer test* of the 2018-frozen structure
against panels drawn from all three pools. Results below; the design reasoning
that follows was written before them and is unchanged by them.

### Transfer test, as measured

Four panels of ~100 cases: the campaign panel (in-sample reference), a
**2018-holdout** drawn disjointly from the same pool, 2019, and 1999. Each panel
screened first, then the four stage-4 finalists run with a per-case (μ, slab)
inner loop, paired against the pre-campaign Voellmy incumbent.

| panel | Δ vs incumbent | 95% CI | role |
|---|---|---|---|
| campaign (n=101) | +0.0359 | [+0.0213, +0.0525] | in-sample reference |
| **2018-holdout (n=99)** | **+0.0275** | [+0.0130, +0.0439] | **overfit control** |
| 2019 (n=100) | **+0.0346** | [+0.0191, +0.0514] | storm / population |
| 1999 (n=99) | +0.0229 | [+0.0080, +0.0390] | four-way confounded |
| **pooled (n=399)** | **+0.0303** | [+0.0227, +0.0383] | p = 3.3×10⁻¹⁴ |

**The frozen structure's advantage transfers to all four panels with confidence
intervals excluding zero**, including the overfit control — so the freeze is not
an artifact of its 101 selection cases — and most strongly on 2019, where the
margin exceeds in-sample.

**On no panel is the frozen variant significantly beaten.** On 1999 the
earth-pressure-off arm has a higher *point estimate*, which needs the direct
paired contrast rather than a comparison of two deltas-against-a-third:

| panel | winner − EP-off | 95% CI | winner better on |
|---|---|---|---|
| campaign | +0.0181 | [+0.0072, +0.0291] | 72/101 |
| 2019 | +0.0150 | [+0.0041, +0.0260] | 59/100 |
| 2018-holdout | +0.0046 | [−0.0119, +0.0205] | 50/99 |
| 1999 | **−0.0035** | [−0.0168, +0.0094] | 53/99 |

53 of 99 is a coin flip and the interval spans zero. **And the holdout control
supplies the scale**: campaign and 2018-holdout are the *same dataset, different
draw*, and this contrast differs between them by **0.0135** — while 1999 sits
only 0.0081 from the holdout. The 1999 ordering shift is smaller than the
sampling variation between two draws of one dataset. It needs no dataset-level
explanation and does not support one.

That control was added to test overfitting. It turned out to also calibrate the
noise floor for every cross-panel contrast, which is what makes the rest
readable — worth carrying into the full per-dataset freezes for that reason
alone.

### The validity edge travels — the strongest result of the campaign

The ifa profile was measured on all four panels (fixed μ, no inner loop), asking
whether the earth-pressure validity edge at ifa = `basal_friction_angle` = 25°
found in stage 3 is a property of the formulation or of the 2018 panel:

| panel | ifa 25 | ifa 30 | ifa 40 | ifa 45 | step 25→30 | plateau spread |
|---|---|---|---|---|---|---|
| campaign | −0.0148 | +0.0008 | +0.0072 | +0.0047 | **+0.0157** | 0.0063 |
| 2018-holdout | +0.0616 | +0.0709 | +0.0727 | +0.0674 | **+0.0094** | 0.0054 |
| 2019 | +0.1131 | +0.1190 | +0.1171 | +0.1127 | **+0.0059** | 0.0064 |
| 1999 | +0.0993 | +0.1077 | +0.1066 | +0.1016 | **+0.0085** | 0.0062 |

**ifa 25 is the worst point on every panel**; the step off it is 1.5–2.6× the
entire 30/40/45 spread; and the plateau's internal width is nearly constant
(0.0054–0.0064) across four independent datasets with different storms, mapping
standards and eras.

This is a mechanism **predicted from the shader source** — the
`max(inside, 0.0)` clamp at `grid_physics.wgsl:151` collapsing the
earth-pressure coefficient when internal friction fails to exceed basal friction
— then confirmed on the panel it was found on, then reproduced on three
independent datasets. **A fitted artifact does not reproduce like that.** It is
the strongest evidence in the campaign that the ifa result is physics-of-the-code
rather than curve-fitting, and it independently confirms that the boundary is set
by the hard-coded `basal_friction_angle`, identically everywhere.

It also **argues against** the 1999 earth-pressure observation: the
earth-pressure formulation behaves the same on 1999 as everywhere else, with the
edge in the same place and the plateau the same width. There is no measured sense
in which earth pressure is different on that dataset.

### The scoring conventions transfer too

The stage −1 conventions were developed entirely on 2018. Their clip-fraction
distributions across the four panels: 1999 scores a mean of **0.0183** against
0.0362–0.0488 elsewhere, with **zero** scored cases past the 50% severity line
where the others have one to four; the median is 0.0 on every panel.

So a dataset with 91% partially-estimated outlines, from a different era and a
different filter chain, registers against its DEMs **at least as well** as the
panel the conventions were built on. Together with the `min_residence` plateau
reproducing to ~0.0002 across a 2× `ppc` change, that is two independent
demonstrations that these are properties of the instrument rather than artifacts
tuned to 2018.

*(One reporting caveat: the 2018-holdout panel's single refusal is the
`"empty release area"` bail — no cell passed the elevation/slope/thickness
filters, so there was nothing to clip and the case would have failed identically
with the clip disabled. Recording it at `clipped_frac = 1.0` inflates that
panel's clip statistics with a case carrying no registration information.)*

### ⚠ 1999's absolute scores are the highest, and four explanations were tested and rejected

1999 scores highest in absolute terms (0.343–0.369 against ~0.28–0.35
elsewhere) *despite* 91% quality-2 outlines. Tested:

| hypothesis | verdict |
|---|---|
| larger avalanches score higher | **rejected** — wrong sign; across panels corr(median `ref_cells`, mean Ω) = −0.272, and the panel with the largest avalanches scores lowest |
| coarser digitisation is an easier target | **rejected** — median vertex spacing is 0.9–1.0 m on *all four* panels; across-panel correlation −0.009 |
| outline-quality class drives it | **rejected** — direction flips: quality-2 beats quality-1 on 1999 and 2018-holdout, loses on 2019 and campaign |
| outlines drawn conservatively (smaller footprints) | **rejected** — 1999's size-3 median is 1486 cells / 3.71 ha against the campaign panel's 1477 / 3.71 |

**Report it as unexplained, with the rejected list attached.** The list is worth
more than a fifth conjecture: it tells the next person which four not to spend
time on.

The structural point is what protects the reading. **Absolute Ω_T is not
comparable across panels** — each has its own achievable ceiling set by
ground-truth quality, size mix, terrain and mapping standard, and four attempts
failed to decompose it. This is exactly why the transfer design rests on
*within-panel paired contrasts*: it is immune to the anomaly by construction, and
1999's absolute being highest changes nothing in the reading above.

### What 1999 can and cannot support

Any 1999 claim carries its confound list. The panel differs from 2018 in **flow
regime, storm, mapping standard and trigger population simultaneously** — its
extraction never applied the natural-release filter, so human-triggered
avalanches are present where the other panels have none. And critically, **its
extraction carries no humidity attribute at all**, so the wet cases cannot be
identified and the within-1999 wet/dry contrast that would isolate regime is not
executable.

So: a 1999 result is an **applicability measurement, not a physics measurement**.
It establishes whether constants frozen on 2018 carry to the 1999 mapping as
extracted. It cannot attribute a difference to wet snow, and the phrase "wet snow
needs different constants" is not supportable by anything measured here.

**2019 is the interpretive lever** — dry, natural-filtered like 2018, different
storm and population. It transferred with the largest margin of any panel, which
localises any future 1999 difference to whatever separates 1999 specifically.

A separate limitation, from the same audit: **1999 has zero coverage on every
deployable feature** (start-zone elevation, aspect, fracture width — all
extracted as explicit nulls). It can supply calibration targets but nothing to
predict them from, so it cannot test the conditions model's central bet at all.
2019 is the only new dataset that can.

### The comparison cannot be paired, so it has to be a transfer test

Every comparison in this document is paired case-by-case, which is what makes a
0.02 band detectable. **Across datasets there are no shared cases, so that
machinery does not apply** — comparing 2018's mean Ω_T against 1999's says
almost nothing, because the panels differ in terrain, size distribution and
mapping quality as well as in constants.

The honest form is **cross-application**, which restores pairing:

> Freeze on dataset A. Evaluate A's frozen constants on **B's panel**, with B's
> per-case (μ, slab) inner loop. Compare, paired on B's cases, against B's own
> frozen constants evaluated the same way.

If A's constants score within the 0.02 band of B's own on B's panel, the
constants transfer and "general" is earned. If they lose materially, the gap
*is* the condition-dependence, measured in the units that matter. Run it both
directions — transfer can be asymmetric, and a dry-tuned vector failing on wet
events while the wet-tuned vector works on dry ones would itself be informative.
`calibrate apply` already does exactly this; no new machinery is needed.

### ⚠ Dataset is confounded with regime — separate them or the result is unreadable

**This is the trap in the whole extension.** 1999 differs from 2018 in at least
four ways at once: flow regime (wet vs dry), storm, terrain sampled, and mapping
methodology (different imagery, different era, different mappers). A disagreement
between 1999 and 2018 constants therefore does **not** demonstrate wet-snow
physics — it is equally consistent with 1999's outlines being mapped to a
different standard.

The design that separates them is to stratify *within* 1999 and compare like
with like:

1. **1999-dry vs 2018-dry.** Same regime, everything else different. Any gap
   here is dataset/mapping/storm, not physics — and it calibrates how much
   apparent disagreement to expect from those nuisances alone.
2. **1999-wet vs 1999-dry.** Same dataset, same mappers, same imagery, regime
   varying. This is the clean regime contrast.
3. Only if (1) is small and (2) is large is "wet snow needs different constants"
   supported. If (1) is already large, (2) is confounded and cannot be read.

Run 2019 before 1999 for the same reason: it is dry-slab like 2018, so it
isolates "different storm and event population, same regime" as a single factor
and gives a second read on the nuisance floor from step 1.

### The wet track re-runs structure, not just the constants

A wet full-depth avalanche is not a dry slab with different numbers — entrainment,
deposition and basal conditions all differ. Freezing ξ for the wet track inside a
structure chosen on dry events would inherit a decision made in the wrong regime.
**The 1999-wet track therefore starts at stage 1, not stage 2.** Its structure
winner is allowed to differ, and if it does, that is a result rather than a
nuisance. Entrainment in particular is worth watching: it is switched on at the
optimum in 1 of 104 dry cases, and full-depth avalanches are the case where it
has an obvious physical job.

### The scoring conventions are not automatically portable

`min_residence`, the pit tolerance, the descent step cap and `MIN_RELEASE_CELLS`
were all measured or chosen on 2018. Two of them have known regime sensitivity:

- **`min_residence` does not normalise for flow duration.** It corrects for `ppc`
  and `cfl` but not for how long the simulation runs, which is why a
  quickly-arresting case loses more of its footprint (measured: `aval_13385`,
  183 steps, lost 21%). Wet avalanches are slower and longer-running, so they
  accumulate more residence and the gate will fire *less* on them. Re-sweep per
  dataset; do not assume 0.25 carries over.
- **The pit tolerance is a physical choice, not a measured one** — 1 m to 100 m
  is indistinguishable on the 2018 panel because these domains drain off their
  edges. A dataset sampling different terrain may contain genuine closed basins,
  which is exactly where the choice starts to matter.

Re-validating the conventions per dataset is stage −1 repeated, and it is cheap:
a handful of whole-panel `run` passes. It must happen before each dataset's
stage 1, for the same reason it happened before 2018's.

### What the pooled freeze is, and is not

A pooled freeze over all three panels answers "what single configuration is least
bad everywhere", which is the right input to the harvest **if** the transfer
tests show the constants agree. If they disagree, a pooled freeze averages over a
real condition-dependence and its constants describe no actual regime. In that
case the pooled vector should be reported as a baseline to beat, not shipped —
and the per-dataset vectors become the training signal for the conditions model,
which is the more interesting outcome of the two.

---

## What has to be written down at the end

The frozen vector, and for each stage: the winner, its margin with CI, the
runner-up, and the width of the indifference region. A knob frozen because
nothing beat it by 0.02 is a different claim from a knob frozen because it was
clearly best, and the harvest's interpretation depends on which it was.
