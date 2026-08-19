# Notes for Markus

> **\* Machine-assembled, and I have not read it end to end.** These were
> accumulated by agents while working through the codebase during the
> calibration campaign — bugs found, measurements that contradict a default,
> places the code and the exposé disagree, and untested suggestions. I asked
> for them to be collected and I am passing them on as-is rather than sitting
> on them. Nothing here carries my sign-off, the numbers have not been
> re-checked by me, and where an item disagrees with the code, assume the code
> is right until someone measures otherwise.
>
> Items marked as hypotheses are hypotheses. Items claiming a measurement give
> the file and line; those are worth checking first.
>
> — Cole

---

## Confirmed bugs

These were hit while getting the simulator to run against real Swiss terrain.
All are reproducible; fixes marked where applied locally on
`baseline-calibration`.

1. **swissALTI3D tile path is unusable as shipped.** The download year is
   hardcoded to 2019, and LZW-compressed tiles are rejected by the `tiff`
   crate — which makes entire regions silently unreachable rather than
   erroring usefully. `crates/data_processor/src/tile_manager.rs`.
   *Fixed locally* (commit `0740c2c`, which tries eight candidate years).

   **Quantified since:** swissALTI3D is flown region by region and published
   per acquisition year, so the year in the asset name is not constant. Of 40
   tiles sampled across a real case footprint, **8 (20%) return 404 at 2019**
   and succeed at 2020 / 2022 / 2024 — e.g. tile `2672-1148`. The failure is
   identical at 2 m and 0.5 m, so it's the year, not the product. For an
   alpine study area this silently removes about a fifth of the terrain.
   Resolving the year via the STAC API would be cleaner than our year-list
   fallback. This is probably the single highest-value fix for anyone else
   picking the repo up.

2. **Non-square DEMs are transposed.** Any case whose domain isn't square
   comes out rotated. Hit immediately on real avalanche outlines, which are
   almost never square.

3. **Minimum elevation handling missing** in the case-preparation path.

4. **`p2g.wgsl` quantisation truncates toward zero.** Mass and momentum are
   scattered into integer atomics via `u32(mass * weight * MASS_FACTOR)` with
   a 0.1 kg quantum, and truncation biases the result low — so the forward
   model leaks mass and momentum on every step. Measured three ways against a
   100× finer quantum and against round-instead-of-truncate; the effect on
   Ω_T was under 0.005 at the parameter vectors tested, so it is **not**
   materially affecting results today. Worth fixing on principle.
   *Fixed locally* (commit `05fae17`, rounds instead of truncating).

5. **Data-quality checks miss the failure mode that actually occurs.** The
   harness bails on `NaN`, but swissALTI3D coverage gaps don't produce NaN —
   the provider replicates the last valid elevation column outward, producing
   perfectly valid floats. Also found: literal `-9999` nodata sentinels, which
   are likewise valid floats. Screening all 105 cases found 9 affected (8.6%),
   4 with contamination overlapping a simulated or observed footprint. Worst
   case (`aval_4117`) has **47% of its domain fabricated**, overlapping 38.6%
   of the reference footprint and 54.1% of the simulated one.
   Detector written locally; suggested rule is to drop a case when the
   fabricated mask overlaps ≥1% of either footprint.
   **Note this is parameter-dependent** — five of the nine are harmless at
   current parameters but become contaminated if flow runs further, so it
   needs to be a per-run pre-flight check, not a one-time case filter.

   **Addendum — the NaN bail *does* fire, just not for the failure mode
   above.** `crates/cli/src/bin/calibrate.rs:309-311` bails when the padded
   DEM request contains a real `NaN`, which happens when the request walks
   off the *edge of the swissALTI3D dataset's spatial extent* (e.g. near the
   national border), as opposed to an internal gap within it (which fabricates
   valid floats instead, per above). Concretely: of 8 cases run at padding
   300/600/1000 m (`--gpu-index` saturation/padding testing, see below),
   `aval_52`/`120`/`482`/`1608`/`2861`/`3091`/`3296` prepare fine at all three
   paddings; `aval_3376` prepares at 300 m but bails with `case aval_3376: DEM
   contains NaN (missing swissALTI3D coverage)` at both 600 m and 1000 m. This
   matters for item 9 below: raising the default padding to stop the 102/105
   overshoot cases will silently drop border-adjacent cases entirely rather
   than fixing them, unless padding is capped to the available extent per
   case. n = 8 cases, 1 affected — enough to know the failure mode exists, not
   enough to say how common it is across all 105.

## Dead or misleading code

6. **`lateral_factor`** (`grid_physics.wgsl:103`) is computed and never used.

7. **The modulo in `uv_to_cell_index`** (`utils.wgsl`) is a no-op — `uv_to_cell`
   clamps to `[0, grid_shape-1]` immediately before it, so `% grid_shape`
   can never do anything. Harmless, but it reads like wraparound handling and
   cost us time investigating whether particles were teleporting across the
   domain. *(We initially believed this was a live bug and reported it as
   such internally — it is not. Retained here because the confusion is the
   point: the code implies a behaviour it doesn't have.)*

8. **Entrainment flag appears to be a no-op** — the bit is defined and
   plumbed but we could not find it changing behaviour. ~~Unverified; worth
   asking rather than asserting.~~

   **Verified 2026-07-28, and it is stronger than "no effect we could find".**
   In the structure stage of the freeze campaign every friction model was run
   twice, entrainment off and on, over 101 cases at a per-case-calibrated
   (μ, slab): the **raw simulated footprint is bit-identical on 101/101 cases**
   (`sim_cells_raw`, max difference 0). For Coulomb even the scored Ω_T is
   bit-identical on 101/101. For samosAT and Voellmy the *filtered* Ω_T differs
   on 2/101 by at most 2.7×10⁻³, which is the atomic-summation noise floor
   (see item 24's note that this solver is not bit-reproducible), not an effect.

   **State it as inert *in this configuration*, not inert.** The most likely
   reason is that entrainment needs an erodible snow-depth field along the path
   which this harness never supplies, so the bit is set with nothing to
   entrain — a different claim from "entrainment physics is unimportant". The
   question for Markus is therefore not "is the flag broken" but **"what is the
   flag expecting to be given, and should it fail loudly when it isn't?"** A
   switch that silently does nothing when its input is absent is the failure
   mode that cost us the check.

8b. **`snow_density` has no dynamical effect — it cancels exactly** *(for the
   Voellmy family; **see item 25 — this does NOT hold for samosAT**, where a
   hard-coded constant breaks the cancellation and density swings Ω_T by
   0.0124 over the physical range)*. This one
   is worth raising because it looks like a physical knob and is exposed as
   one. In `compute_particles.wgsl` the Voellmy shear stress is
   `ρ·v²·g/ξ` (line 308), and it is divided by `mass_per_area`, which is
   `particle.mass/cell²·ppc` with `particle.mass = cell_volume·ρ`
   (`initialize_particles.wgsl:31`). Density appears once in the numerator and
   once in the denominator, so the acceleration is independent of it. The same
   cancellation makes the recovered flow thickness density-free
   (`compute_particles.wgsl:117` divides grid mass by `snow_density`).

   Confirmed empirically: sweeping density 120 → 450 kg/m³ over all 105 cases
   moves mean Ω_T from +0.007490 to +0.007484 — six parts in a million, which
   is one cell flipping on one case, i.e. atomic-quantisation noise rather than
   physics. Substituting per-case *measured* storm-slab densities for the
   uniform 200 kg/m³ changes mean Ω_T by 7×10⁻⁵.

   Consequences worth stating plainly: density cannot be calibrated against
   footprint data — there is no gradient to descend. Anything that fits it
   per-event (including a per-event parameter database feeding a regressor) is
   fitting rounding error. If density is meant to matter it has to enter
   somewhere it currently doesn't — an entrainment term, or a
   density-dependent friction closure. Worth asking whether the cancellation
   is intended.

---

## Measurements that contradict the defaults

Based on 105 stratified avalanches from the SPOT6 24 Jan 2018 dataset
(Hafner & Bühler, EnviDat DOI 10.16904/envidat.77), scored with the repo's own
`evaluate_mass_movement_area`. Note Ω_T ≡ 2·IoU − 1, which is worth stating in
the thesis — it makes the metric immediately legible to anyone who knows IoU.

9. **The shipped defaults never arrest.** `friction_coefficient = 0.155` with
   `drag_coefficient = 4000` sits at the least-dissipative corner of the RAMMS
   published range (μ 0.14–0.47, ξ 900–4000). Consequence: **102 of 105 runs
   reach the edge of the 300 m padded domain**, running a median **+357 m**
   past the observed toe. Doubling padding to 600 m does not contain them —
   the figure grows to **+632 m**. They don't overshoot; they fail to stop.
   β ≈ 0.01, so they reach nearly everything that was actually hit.
   Calibrated runs behave properly (30/105 touching the edge, stopping well
   inside). This looks like defaults tuned for the synthetic AvaFrame cases
   that were never revisited for real terrain.

10. **Default slab thickness and density don't match the event.** From IMIS
    station data for the 20–23 Jan 2018 storm (96 stations with complete
    series): storm-slab depth at release **0.48 m** (range 0.06–1.17), density
    **307 kg/m³**. The repo defaults are **1.0 m** and **200 kg/m³** — the
    thickness sits near the 90th percentile of observed, the density well
    below. LAWIS incident records (n = 4378) corroborate: slab height is
    generally 30–100 cm with the modal bin 25–50 cm.

11. **Elevation is a poor predictor of storm-snow depth**, at least for this
    event. R² = 0.023 on storm-layer depth across 96 IMIS stations
    (+1.3 cm/100 m, t = 1.5); between-station scatter of ±0.19 m swamps the
    gradient. The intuitive "more snow higher up" field would have been close
    to worthless. Detrended inverse-distance weighting over the six nearest
    stations worked considerably better.

12. **Per-event calibration substantially outperforms any global vector.**
    Best global fit reaches Ω_T = +0.005; calibrating each event individually
    reaches **+0.295**. That gap is direct quantitative support for the
    exposé's per-event CMA-ES → mixture-density-network design — it says
    roughly 0.29 of Ω_T is unreachable by any single parameter set.
    ~~*Caveat we have not yet resolved: whether per-event optima are
    identifiable, or whether there is a degenerate μ–ξ ridge making the
    "optimum" partly arbitrary. Under investigation; this determines whether
    the MDN has a well-posed target.*~~ **Now measured — see item 12b.**

12b. **μ is identifiable per event; ξ is not.** This is the answer to the
    caveat above, and it matters directly for the MDN target. Method: Ω_T
    mapped over a 12×12 grid in (μ, ξ) — μ ∈ [0.05, 0.60] linear, ξ ∈ [200,
    12000] log — for 12 events spanning the size and geometry range, each at
    its own per-event-fitted slab thickness. 1 728 simulations, `calibrate
    grid` subcommand.

    - **μ is pinned**: the band scoring within 0.02 Ω_T of that event's best
      admits a median **1.21×** range in μ, worst case 2.7×.
    - **ξ is not**: median **2.11×**, and 4 of 12 events tolerate a >3× change
      — up to **19.6×**, which is the entire admissible range — for under 0.02
      Ω_T. Concretely, on aval_6716 the grid optimum is ξ = 12 000 while an
      independent Nelder-Mead run returned ξ = 2 605, for a Ω_T difference of
      **0.004**; on aval_13093, 12 000 vs 3 383 for 0.004. Those are different
      MDN targets and identical physics.
    - **The per-event gain is nonetheless real, not overfitting.** A single
      shared (μ, ξ) tuned on these 12 reaches mean Ω_T +0.073 against +0.260
      for per-event optima; the gap (+0.187) *widens* to +0.207 under
      leave-one-out, and borrowing another event's optimum costs +0.384 on
      average. So item 12's headline stands — something event-specific is
      genuinely being fitted, it is just carried by μ and slab, not ξ.
    - **A shared degeneracy ridge exists, but only at coarser tolerance.** At
      ±0.02 the near-optimal sets are small (median 2.1% of the box) and their
      orientations scatter. At ±0.05–0.20 they elongate with a strikingly
      consistent orientation near **+65°** in normalised (μ, log ξ) coordinates
      (alignment R up to 0.95). +90° would mean "ξ irrelevant" and +45° the
      classic μ–ξ trade-off, so this is predominantly ξ-insensitivity with a
      secondary compensation, not symmetric non-identification.

    *Recommendation for the MDN / regressor: predict μ and slab thickness;
    do not predict ξ.* Fix it, or treat it as a nuisance parameter marginalised
    at prediction time. Correlations of the 105 per-event optima with candidate
    features support the same split — size class → slab r = +0.36, area → slab
    +0.28, start elevation → μ −0.24, while ξ correlates with nothing except
    area (+0.23).

    **One caution that applies to the whole per-event programme:** fitted μ
    correlates with `aval_shape` (outline quality, item 19) at **r = +0.31** —
    stronger than any genuine terrain feature except size class. Part of what
    per-event calibration fits is how carefully the polygon was digitised, and
    that feature does not exist at prediction time. Worth stratifying the
    training set by outline quality, or restricting it to quality-1 outlines.

    *Extrapolation to the full parameter set:* this was measured in 2-D at
    fixed slab, so it is a lower bound on the degeneracy of a 6–8 parameter
    per-event fit. The within-tolerance set in a higher-dimensional space
    contains the 2-D slice, so opening more parameters can add flat directions
    but cannot remove the ones already found. `density` is provably one such
    flat direction (item 8b); roughness is another (item 17).

12c. **A per-event μ is only meaningful jointly with the ξ it was fitted
    against — so fix ξ *before* calibrating, not after.** This is the item with
    the most direct bearing on the per-event CMA-ES → MDN design, and it is a
    methodological point rather than a criticism of anything in the code.

    Item 12b established that the equally-good region is a diagonal ridge at
    ~+65° in normalised (μ, log ξ). A diagonal ridge means the two are coupled:
    **the optimal μ is a function of ξ.** The practical consequence is sharp.
    Taking each event's fitted μ and slab from a free-ξ calibration and then
    pinning ξ to a global constant lands you *off* the ridge — worse than either
    end of it:

    | per-event scheme | mean Ω_T | share of the free-ξ gain |
    |---|---|---|
    | ξ free; μ, slab, ξ all fitted per event | +0.2945 | 100 % |
    | **ξ fixed at 754 first; μ, slab fitted at that ξ** | **+0.2864** | **97 %** |
    | μ, slab from the free-ξ fit; ξ *then* pinned to 754 | +0.1414 | 48 % |

    Provenance: 105 cases, `per-event --budget 40`, `--free mu,slab --xi 753.63`
    vs `--free mu,xi,slab`, `--band 0.25`; each row re-scored independently
    through `calibrate apply` with per-case vectors.

    Why this is good news rather than bad: **ξ costs only 3 % of the achievable
    per-event gain if you fix it up front**, so the MDN can drop it from the
    output layer entirely. And doing so has three side benefits we measured:

    - *The search actually converges.* With ξ free, 11/105 per-event searches
      terminated by simplex collapse and 94 hit the evaluation cap. With ξ
      fixed, **81/105 converge** on the same budget. The optimiser had been
      spending its budget wandering along the non-identified direction.
    - *The targets become more predictable.* Correlations with the fitted slab
      rise across the board: size class +0.36 → +0.44, area +0.28 → +0.37,
      `frac_wdh` +0.30 → +0.34.
    - *The outline-quality leak shrinks*, `aval_shape` vs μ* +0.31 → +0.25.

    Suggestion, offered as a hypothesis: if a per-event ξ is wanted for physical
    reasons rather than for score, it would be better estimated from something
    other than footprint overlap — ξ controls velocity, and the footprint is
    nearly blind to it. A velocity- or impact-pressure-sensitive observable
    would identify it; area overlap will not, at any sample size.

12d. **Watch out for a leak when deriving "terrain" features.** Recorded because
    it cost us a false positive and would cost anyone else the same. The
    calibration domain is the observed outline padded by 300 m, so *any* terrain
    feature computed over the whole domain silently encodes how big the
    avalanche was. Our first descent-path features looked strongly predictive
    until we checked: `path_drop_potential` correlates +0.78 with the observed
    drop and `path_len_m` +0.66. Truncating the descent at a fixed distance from
    the release apex (we used 200 m) removes the dependency; the correlation
    with fitted μ survives at r ≈ +0.42, so the signal is real, but the
    untruncated versions are not usable as predictors.

13. **Measured snowpack forcing helped less than expected on footprint
    overlap.** Replacing the uniform 1.0 m slab with measured storm-slab depth
    and density moved Ω_T from +0.005 to +0.008 — essentially nothing on the
    area index — though it appears to help more on runout distance, which
    suggests footprint IoU is simply insensitive to how far past the toe the
    flow runs. Using measured values *without* refitting friction was worse
    than defaults. Recorded honestly rather than optimistically; the
    interesting question for Markus is which metric he considers primary.

---

## Discrepancy between exposé and code

14. The exposé describes weighting cells **within the observed outline**
    according to their distance **from the release area**. The implementation
    in `evaluate_distance_weighted_mass_movement_runout` weights **every union
    cell** by distance from a **single apex point**. These are different
    estimators. Probably immaterial for hazard framing, but the text and the
    code should agree before either goes into the thesis.

---

## Hypotheses and suggestions (untested unless noted)

15. **Release areas spanning a drainage divide split the avalanche.**
    Observed in `aval_10721`: ~3% of mass ends up in 115 disconnected
    fragments, thin threads crossing a ridge north of the release apex.
    Proposed mechanism — on a crest the terrain normal is near-degenerate, so
    trivial numerical differences decide which side a particle descends, and
    one avalanche descends two catchments.

    The release construction takes the upper 20% of outline elevation
    (Korzeniowska et al. 2017), which can cross a crest. Two possible fixes:

    - *Downstream:* constrain flow to within some angle of a mean descent
      direction. Simple, but would also suppress genuine lateral spreading in
      open bowls.
    - *Upstream, and we think better:* prevent release areas from spanning a
      divide at all — connected-component the release mask by drainage, then
      simulate components separately or drop those draining away from the
      observed footprint. Removes the ambiguity rather than masking its
      consequences.

    ~~Prevalence across all 105 cases is being measured; it may be a footnote
    rather than a systematic issue.~~

    **RETRACTED — the mechanism is wrong, and the symptom that motivated it is
    a plotting threshold.** Measured across all 105 at defaults:

    - *The 115 fragments are not a flow phenomenon.* They are the 0.1 m
      `flow_threshold` contour cutting a thin, tapering flow margin. Sweeping
      the threshold: at 0.1 m, 95/105 cases have ≥2 components (median 9, max
      116); at 0.01 m **all 105 cases are a single connected component**. No
      case anywhere has a second component holding ≥1% of the mass (median
      detached mass 0.06%, max 2.14%). aval_10721's "~3%" is an *area*
      fraction; its mass fraction is 0.76%.
    - *They are not across a divide.* D8 steepest-descent tracing from all 222
      of aval_10721's release cells: every one descends into the main body, and
      **not one fragment is reachable by descent from any release cell**. The
      fragments sit 420–910 m downslope of the apex — they are the distal and
      lateral margins of the same flow.
    - *The upstream fix addresses a near-empty set.* Taking the 70/105 cases
      whose observed outline is ≥95% within one D8 drainage, only **2** have
      any release cell in a different drainage from the observed outline.
      Across all 105, 19 cases have a release spanning >1 major drainage
      (≥5% of release cells each), but that variable is uncorrelated with
      fragmentation (r = +0.00) and those cases score no worse.

    **There is a real divide-crossing problem, but it is downstream and it is
    mostly item 9 in disguise.** In those same 70 cases, 15 send >10% of
    simulated mass into a drainage other than the observed one — with the
    release entirely in the correct drainage in every one of the worst cases,
    so the flow crosses mid-path. Re-running all 105 at the tuned global vector
    (μ = 0.218, ξ = 754) collapses it: **>10% of mass in the wrong drainage
    falls from 15/70 to 2/70**, and median component count at 0.1 m falls from
    9 to 3. Flow that overshoots by +357 m has the surplus energy to climb out
    of its valley; fix the friction and most of this goes with it.

    *Suggestion, revised:* do neither fix yet. Report connectivity-cleaned
    footprints (drop components below ~0.5% of mass) so the count stops
    alarming readers, and re-measure divide crossing once the friction question
    is settled. Provenance: `frag_analysis.py`, `divide_analysis.py`,
    `crossing_analysis.py`; 105 cases at defaults and 105 at tuned, both
    `--band 0.25`.

16. **The roughness field is measuring the wrong scale.** `compute_roughness.wgsl`
    applies a 3×3 VRM stencil to the 5 m DEM — which measures 15 m *macro*
    terrain, not the sub-grid roughness that `roughness_threshold` is
    conceptually gating on. The shader's own TODO says the kernel should scale
    with resolution and snow height, so this looks known rather than
    overlooked.

    Measured on tile `2633-1131` by recomputing the same stencil at each source
    resolution and aggregating back to 5 m:

    | source | mean VRM | p95 |
    |---|---|---|
    | 5 m (current) | 0.01455 | 0.05865 |
    | 2 m | 0.01165 | 0.04259 |
    | 0.5 m | 0.00476 | 0.01502 |

    Correlation between the 0.5 m-derived field and the current one is **0.414** —
    these are substantially different fields, not a rescaling. At the default
    `roughness_threshold` of 0.01, the fraction of cells gated out of release
    moves from **36.7% to 9.3%**: a 4× change in how much terrain is eligible
    to release.

    Not drop-in better — the fine-derived mean is 0.33× the current one, so the
    threshold would need recalibrating; the two aren't on a common scale.
    Deriving roughness from 0.5 m data at preprocessing and delivering a 5 m
    field is cheap (fine tiles stream and discard, no retention) and looks like
    the most defensible use of higher-resolution terrain. *Single tile —
    generalisation is inference, not measurement.*

17. **The roughness gate does not discriminate release terrain — at any source
    resolution.** Following on from item 16, we tested whether roughness
    actually separates terrain that released from terrain that didn't, using
    the observed SPOT6 outlines as ground truth (release cells = inside the
    outline, elevation in the upper 20% of the outline's range; median slopes
    37–50°, so the labelling is sound). Slope-matched AUC, pooled over 11 tiles
    with bootstrap 95% CIs:

    | field | AUC |
    |---|---|
    | current 5 m-derived roughness | 0.526 [0.515, 0.535] |
    | 0.5 m-derived roughness | 0.514 [0.504, 0.524] |
    | slope (positive control, unmatched) | 0.652 [0.648, 0.656] |

    **Both roughness fields sit at chance.** The slope control confirms the
    test has power under identical limitations, so this is a real null rather
    than weak data. Higher-resolution source data does not rescue it — fine is
    marginally *worse*, and the per-tile difference (+0.013, sd 0.230, t = 0.18)
    is noise.

    More pointed: **unmatched, both fields fall below chance** (0.455 fine,
    0.443 coarse) — observed release areas appear slightly *rougher* than
    surrounding terrain, which is the opposite of the gate's premise. That's
    confounded with slope and we would not claim it as a result, but combined
    with the at-chance matched AUC it suggests `roughness_threshold` may not be
    doing what it is assumed to do.

    Caveat stated honestly: absence of an outline isn't proof terrain couldn't
    release — controls may simply not have been triggered in 2018. That caps
    the power of any terrain predictor here. But slope cleared 0.65 under the
    same limitation.

    *Consequence for us: deriving roughness from 0.5 m terrain is cheap and
    technically viable, but not worth building. We are not pursuing it.*

18. **Raising the padding is not a free knob.** Item 9 notes that the defaults
    blow through a 300 m padded domain and that 600 m doesn't contain them. But
    increasing padding can push a case's bounding box off the edge of
    swissALTI3D's actual spatial extent near the national border, where it hits
    the NaN bail-out (`calibrate.rs:309-311`) and **hard-fails rather than
    clipping**. Confirmed on `aval_3376`: prepares fine at 300 m, fails at both
    600 m and 1000 m. 1 of 8 cases tested — flagged as a caveat rather than
    claimed as widespread. Any padding increase wants a fallback: clamp to
    available coverage, or detect and flag border cases.

    Cost of padding, for reference (8 cases, cold cache): 300 m → 0.090 s/case
    and 39 tiles; 600 m → 0.113 s and 68 tiles; 1000 m → 0.144 s and 116 tiles.
    Compute growth is modest; the dominant cost at any padding is the one-time
    cold tile fetch (~5–10 s/case), not GPU time (~0.1–0.15 s/case).

19. **`aval_shape` is undocumented but appears to encode outline quality.**
    The field's value distribution (32.6 / 58.0 / 9.3%) reproduces exactly the
    shares Bühler et al. 2019 report for exact / partially estimated /
    expert-created outlines. Worth documenting — anyone using this dataset
    should know they can filter on ground-truth quality. *Reassuringly, at
    n = 105 the score barely differs between quality bands.*

20. **The observation can never contain a single-particle trace — so the
    objective was penalising the simulator for being visible at higher
    resolution than the mapping.** *(Implemented locally as fixed scoring
    conventions, commit `5be6531`; on-GPU sweep pending.)* A lone particle
    rolling ~50 m past the deposit registers a one-cell-wide filament in the
    simulated footprint; a SPOT6-mapped polygon (1.5 m imagery, hand-outlined)
    cannot and does not record such features, so every one of them is scored
    as overshoot regardless of the physics. Two filters, and the division of
    labour between them is the finding worth relaying:

    - *Release-connectivity* removes detached fragments — item 15 already
      established these are contour artefacts of the 0.1 m threshold
      (105/105 largest components touch the release; median detached area
      0.37%). Predicted effect from the committed frag data alone: mean
      ΔΩ_T **+0.0044**, median +0.0020, max +0.0315 — small on the *area*
      metric.
    - *A residence gate* (per-cell particle-steps, normalised by cfl/ppc so
      the convention is invariant to numerics choices) is the only thing
      that removes the **attached** tail — connectivity keeps it by
      definition. Unit test pins this division so neither filter gets
      deleted as "redundant". The tail's real bite should be on **reach**
      (a max statistic set by a single surviving cell), which bears directly
      on the exposé's distance-weighted extension of Heiser: filament
      inflation biases exactly the metric that extension emphasises. Ties to
      the open question in item 13 — which metric is primary.

    The gate threshold (1.0) is currently *argued, not measured*; a sweep over
    {0, 0.25, 0.5, 1, 2, 4} is the first job on the GPU box
    (`validate_fixes.py`), and every evaluation now carries its raw/filtered
    scores side by side, so one run is its own before/after.

21. **Release cells that drain away from the observed outline are now
    clipped at prep time — a conservative resolution of item 15's upstream
    question.** *(Implemented locally, commit `5be6531`; prevalence
    re-measurement pending.)* Item 15's retraction stands: wrong-drainage
    *mass* is mostly a downstream, friction-driven effect, and release-side
    divide-spanning was rare in the clean-outline subset (2/70). But the
    campaign's inner loop always runs calibrated friction — which item 15
    showed collapses the downstream crossing (15/70 → 2/70) — leaving the
    genuine residue: band construction on a 5 m DEM from a 1.5 m-imagery
    polygon can still place upper-band cells past a crest. The clip keeps a
    release cell only if its D8 steepest-descent path reaches the observed
    outline. Deliberately **not** the basin-clustering test
    (`crossing_analysis.py`), which mislabels `aval_6719` as 100%
    wrong-drainage while the case scores +0.337 — a clustering artefact this
    must not inherit; note its headline count (22/105 cases with any
    release cell in a "different drainage") is therefore not comparable to
    item 15's 2/70 and should not be quoted as prevalence. On unaffected
    cases the clip is a no-op (`release_clipped_frac = 0` is recorded
    per case); it cannot empty a release (fallback flag instead), and a
    >50% clip is flagged as suspected polygon/DEM misregistration rather
    than silently accepted. A 20-case before/after visual sheet of the worst
    offenders is being prepared for eyeball approval; that, not the
    clustering count, is the prevalence evidence worth showing.

    **Addendum (2026-07-28, hours later): the clip as first wired was a
    tautological no-op — caught by the visual sheet, not the unit tests.**
    Running the real binary (commit `5be6531`) over all 105 cases with the
    clip on and off produced byte-identical release rasters and
    `release_clipped_frac` = 0.0 on 105/105. Cause: `build_release` draws
    candidates only from inside the observed polygon, while the descent
    memoisation seeded *every* polygon cell as "trivially drains" — so every
    cell the filter could ever be asked about passed by definition. The unit
    tests were correct about the descent function in isolation but probed
    far-flank cells outside the polygon, exactly the set `build_release`
    never reaches. The conceptual error is treating polygon membership as
    ground truth when the fix exists precisely because it isn't. Being
    rewired to require descent into the outline's lower body; this entry
    gets real prevalence numbers once the corrected clip runs. Worth
    relaying to Markus as a methods cautionary tale: the 20-image eyeball
    pass caught in minutes what ten passing unit tests missed.

22. **Ω_T has a reproducibility ceiling of roughly +0.28 to +0.36 on 1.5 m
    imagery, and per-event calibration is already at it.** This is probably the
    single most useful thing to put in the thesis, because it changes how every
    Ω_T in it should be read.

    His own group measured it. **Hafner et al. (2023), NHESS 23, 2895–2914**,
    data at EnviDat DOI 10.16904/envidat.423, asked independent people to map
    *the same* avalanches and reported the overlap:

    | comparison | IoU | Ω_T = 2·IoU − 1 |
    |---|---|---|
    | 10 mappers, oblique photos, pairwise | 0.52 (0.32–0.69) | **+0.04** |
    | 5 mappers, orthophoto @ 2 m, pairwise | 0.46 | **−0.08** |
    | 5 mappers, orthophoto @ 2 m, vs reference | 0.64 | **+0.28** |
    | same @ 25 cm, pairwise | 0.68 | +0.36 |
    | same @ 25 cm, vs reference | 0.80 | +0.60 |

    SPOT6 is 1.5 m, so the 2 m row is the right read-across. **Our per-event
    calibration reaches +0.295** — indistinguishable from how well a human
    mapper agrees with the reference at that resolution, and far above how well
    two humans agree with each other.

    Consequences worth discussing:

    - **A model scoring ~+0.3 against a 1.5 m-mapped outline is at the noise
      floor of the ground truth.** Further Ω_T gains at this resolution are
      largely fitting digitisation style, not physics. Any thesis result near
      that value should be reported against this bound rather than against 1.0.
    - **It gives the per-event vs global gap a proper denominator.** The
      reachable band is roughly +0.006 (best global) to +0.295 (per-event
      oracle ≈ ceiling), so ~0.29 is the whole prize. Our deployable-feature
      regressor captures none of it and the outcome-feature one about 21% —
      those percentages are only meaningful because the top of the range is now
      known.
    - **It independently corroborates item 12b's `aval_shape` caution.** If
      mappers disagree at IoU 0.46–0.64, then part of what a per-event fit
      "learns" is the mapping, which is exactly the r = +0.31 (later +0.25)
      correlation between fitted μ and outline quality.
    - **It argues for resolution-stratified reporting.** Agreement rises
      sharply with imagery resolution (0.46 → 0.68 pairwise from 2 m to 25 cm)
      and with wetness (0.90 wet vs 0.66 dry at 25 cm). Aggregating events
      mapped under different conditions into one mean Ω_T mixes different noise
      floors.

    *Caveat, stated so it isn't overclaimed:* the study used oblique photographs
    and orthophotos, not SPOT6 itself, and the SPOT6 outlines were produced by
    SLF experts under a consistent protocol, so within-dataset consistency is
    plausibly better than a cross-participant experiment suggests. Treat +0.28
    as a well-grounded estimate of the ceiling, not a measured bound for this
    specific dataset. Verified by fetching the paper directly, 2026-07-28.

23. **Three open storm dates exist, and the paths that ran in more than one are
    the closest thing to a controlled experiment available.** Worth raising
    because it bears directly on the exposé's regressor/MDN design.

    All calibration to date uses a single storm — 24 Jan 2018 — so across the
    entire training set the weather and snowpack are *constant*. A regressor
    meant to map live conditions onto parameters therefore has no condition
    variance to learn from, and "deployable features don't work" cannot
    currently be distinguished from "we never gave it a testable hypothesis".

    Two further SLF datasets, same lineage, open, fix this:

    | storm | polygons | DOI |
    |---|---|---|
    | 25 Feb – 1 Mar 1999 | 11,120 | 10.16904/envidat.579 |
    | 24 Jan 2018 | 18,737 | 10.16904/envidat.77 |
    | 16 Jan 2019 | 6,041 | 10.16904/envidat.235 |

    **The specific idea worth pursuing: isolate the avalanche paths that appear
    in more than one storm** — polygons from different dates that overlap
    spatially, i.e. the same slope running in 1999 *and* 2018, or 2018 *and*
    2019. For those, terrain is held essentially constant while snowpack and
    weather differ. Any change in fitted (μ, slab) between the two events on the
    same path is then attributable to conditions rather than to terrain, which
    is exactly the quantity the regressor is supposed to predict and which a
    single-storm dataset can never expose. It also gives a direct estimate of
    how much of the per-event parameter spread is condition-driven versus
    terrain-driven — currently unknown, and load-bearing for whether an MDN over
    conditions is well-posed at all.

    Caveats to check before claiming anything: 1999 was mapped from panchromatic
    aerial imagery rather than SPOT6, so outline quality and attribute schema may
    not match (item 22's resolution-dependent agreement applies — different
    sensor, different noise floor). A spatial overlap also does not guarantee the
    same release; two storms can run overlapping but differently-shaped paths.
    Treat the overlap set as a candidate list to inspect, not an automatic
    matched pair. Counting how many such coincidences exist is cheap and has not
    been done.

24. **Ω_T is not monotone in resolution — particle count has an interior
    optimum at the shipped default, and "finer is more accurate" has no
    anchor in this model.** Measured 2026-07-28 on the 105-case panel
    (n = 104 scored; per-case fitted μ/slab at each setting, `apply`
    design, paired bootstrap CIs; replicated independently on Metal and
    Vulkan backends):

    | ppc (at cfl 0.5, ms 3000) | mean Ω_T |
    |---|---|
    | 2 | +0.2541 |
    | **8 (shipped default)** | **+0.2828** |
    | 16 | +0.2710 (−0.0108, CI [−0.0151, −0.0069]) |
    | 32 | +0.2660 (−0.0167, CI [−0.0256, −0.0100]) |

    More particles cost 1.4–2× and score *worse*; fewer (4, 2) decline
    monotonically the other way. cfl has a real floor (0.9 degrades by
    −0.074; 0.7 by −0.064) but 0.5 vs 0.25 is a wash. So the shipped
    defaults (`ppc 8, cfl 0.5`) are not merely adequate — on this panel
    they *dominate* every finer setting on both cost and score.

    Why this is worth raising rather than just celebrating: `ppc` is a
    convergence knob (`initialize_particles.wgsl` divides mass by it), so
    a numerics-convergence argument of the usual "refine until stable"
    form would have picked ppc 32 and quietly lost 0.017 of Ω_T at double
    the cost. **The mechanism for the interior optimum is unknown.** One
    candidate explanation — that coarse sampling flatters an overshooting
    model by shrinking footprints — was tested and ruled out (footprint
    size unchanged, median −0.00%; gain equal on overshooting +0.0051 and
    undershooting +0.0048 cases; corr(gain, shrinkage) = −0.04). Worth a
    conversation about what in the p2g/thickness-recovery path could make
    8 particles per cell a sweet spot rather than a compromise; whatever
    it is, it is a property of the discretisation, not of the metric.

    Practical consequence for the thesis: numerics-sensitivity checks
    against this model need a measured response surface, not a
    finest-setting reference — "indistinguishable from the finest run"
    selects against the best available setting here.

25. **`compute_particles.wgsl:40` hard-codes `const density = 200.0`, and
    samosAT uses it alongside the real `snow_density` — so density has a
    dynamical effect under samosAT that it does not have under Voellmy, via
    an inconsistency rather than physics.** This is a genuine bug and it
    became load-bearing for us the moment samosAT won our structure
    selection.

    The mechanism. samosAT's normal-friction term is

        shear = normal_stress · μ · (1 + rs0/(rs0 + rs)),
        rs = density · v² / (normal_stress + 0.001)

    where `density` is the file-scope constant 200.0, while `normal_stress`
    is `|a_n| · mass_per_area` and `mass_per_area ∝ sim_settings.snow_density`
    (`initialize_particles.wgsl:32`). So `rs ∝ 200/ρ_settings`: raising the
    configured density lowers `rs`, raises the bracket toward its ceiling of
    2, and increases friction. The Voellmy drag term by contrast uses
    `sim_settings.snow_density` in numerator and denominator alike and
    cancels exactly, which is item 8b.

    Measured over all 105 cases at μ = 0.36, slab = 0.56, flags = 7,
    cfl 0.5 / ppc 8 / 3000 — **the same sweep on both models, which is what
    makes it a demonstration rather than an assertion**:

    | density (kg/m³) | samosAT mean Ω_T | Voellmy mean Ω_T |
    |---|---|---|
    | 120 | +0.0105 | +0.0063 |
    | 200 | +0.0087 | +0.0062 |
    | 300 | +0.0049 | +0.0055 |
    | 450 | −0.0019 | +0.0054 |

    samosAT swings **0.0124 over the physical range, monotone, with all
    104/104 scored cases changing and up to 0.44 on a single case**. Voellmy
    swings 0.0009, i.e. the documented cancellation still holding. The
    cancellation applying to one friction family and not the other is the
    cleanest evidence that this is the constant, not the physics.

    Two consequences worth raising:

    - **Anything that calibrates density against footprints under samosAT is
      fitting the gap between two numbers that were meant to be the same
      one.** We froze density at 200 rather than fit it, which also makes the
      two uses agree and restores the intended formulation.
    - **Density is degenerate with μ under samosAT** — `rs ∝ 1/ρ` makes it a
      friction multiplier, doing what μ already does. A per-event search free
      in both would wander a non-identified direction, which is the same
      failure mode as ξ in item 12c. Relevant to the exposé's CMA-ES design
      if samosAT is the selected friction model.

    Fix is presumably one line: use `sim_settings.snow_density` in the `rs`
    ratio. Worth checking whether the AvaFrame samosAT reference intends the
    ratio to use flow density at all, since that decides whether the fix is
    "use the setting" or "the ratio is dimensionally different from what we
    assumed".

    **Addendum 2026-07-29 — reference checked, fix applied locally.** The
    AvaFrame com1DFA theory doc gives R_s = ρ·ū²/σ_b with ONE density
    symbol used consistently; since σ_b ∝ ρ, R_s is density-free by
    construction (a Froude-type number), and with τ₀ = 0 (which this
    shader uses) density cancels out of samosAT dynamics entirely under a
    consistent implementation. So there is no reading in which the
    hard-coded numerator is intentional, and R_s0 = 0.222 needs no
    re-fitting (it gates a dimensionless quantity). We applied the
    one-word fix on `baseline-calibration` (settings density at the
    mismatch site; the now-unused `const density` removed). At
    density = 200 the fix is arithmetically identical to the old code, so
    every calibration result stands; at any other density the knob is now
    inert instead of a spurious friction multiplier. Supporting evidence
    of an unfinished refactor: the drag-term function already computed a
    settings-density `rs` and then never used it. Remaining for Markus:
    confirm upstream, and note his `snow_density` mass/thickness sites
    were always consistent — only the friction-bracket numerator was off.

26. **Which friction model you choose changes what the calibrated μ *means*,
    and therefore what a regression onto μ is regressing onto.** This bears
    directly on the exposé's phase-1-selects-the-model, phase-2-calibrates,
    phase-3-fits-an-MDN sequence: the phases are not independent, because
    phase 1 silently redefines phase 3's targets.

    Under Voellmy, μ is the Coulomb friction coefficient: shear =
    μ·normal_stress. Under samosAT it enters as
    `μ · (1 + rs0/(rs0 + rs))`, a bracket running from 2 at rest to 1 at
    speed — so μ is roughly *half the low-speed friction coefficient*, and
    the effective friction is velocity-dependent in a way it is not for
    Voellmy.

    What makes this a trap rather than a note: **the fitted values land in
    almost the same numerical range.** Across our 101-case panel the fitted
    μ median was 0.347 under samosAT against 0.343 under Voellmy. Nothing in
    the numbers signals that they denote different quantities, so a
    regression, a feature-importance table or a correlation carried across a
    model change looks perfectly healthy and is measuring something else.

    For us this invalidated, pending re-measurement, every μ-based
    feature-side result we had: the terrain-descent correlation (r = +0.42),
    the target-noise ceiling (R² ≈ 0.95 for μ), and the `aval_shape`
    digitisation leak. They are not wrong; they are about Voellmy-μ.

    Suggestion: if the thesis reports calibrated parameter distributions or
    any regression onto them, state the friction model in the same breath,
    and re-derive rather than carry forward across a model change.

27. **The earth-pressure term degenerates silently whenever
    `internal_friction_angle ≤ basal_friction_angle`, and one of those two
    is a hidden constant.** `grid_physics.wgsl:151` computes
    `inside = 1 − cos²(ifa)/cos²(basal_friction_angle)` and then
    `sqrt(max(inside, 0.0))`. For ifa ≤ 25° (the hard-coded
    `basal_friction_angle`) the clamp fires and the earth-pressure
    coefficient loses its flow-state dependence entirely — the classical
    Rankine / Savage–Hutter validity condition, enforced by a silent
    `max` rather than a warning or a validated setting.

    We hit this as a *measurement* before finding the mechanism: profiling
    ifa over 15–45° (24,129 evaluations, 101-case paired panel, samosAT at
    the otherwise-frozen vector) gave a step at exactly 25° — ifa 15/20/25
    significantly worse (ΔΩ_T −0.0162 to −0.0122, CIs excluding zero),
    30° marginal, 35–45° a statistical plateau. The arithmetic matches:
    25° gives `inside` = 0 exactly, 30° a root of 0.295, 40° of 0.535.
    A density-compensation check (ifa profile re-run at density 150/200/300)
    was null — the optimum is 40° at every density — so this is the
    earth-pressure edge, not a resistance trade-off.

    Two things worth raising:

    - `basal_friction_angle` is not exposed anywhere as a parameter, so no
      calibration over the exposed set can ever notice it. We froze ifa 40
      *given* basal 25 and recorded basal as fixed-at-default-never-swept.
      For wet or full-depth avalanches (1999 data) basal conditions are
      exactly what changes, so a fixed 25° is a real modelling assumption
      there, not a technicality.
    - Suggestion (untested): validate `ifa > basal_friction_angle` at
      settings load, or at least document that below it the earth-pressure
      flag is effectively a different (state-independent) model. A user
      sweeping ifa low would currently get plausible-looking results from
      degenerate physics with no signal that anything happened.

---

## Provisional calibrated values

Offered as a starting point, not a result. Fitted on Voellmy against
observed outlines; provenance noted because sample size matters here.

| parameter | repo default | fitted | note |
|---|---|---|---|
| `friction_coefficient` (μ) | 0.155 | ~0.28 | 20% release band |
| `drag_coefficient` (ξ) | 4000 | ~530 | |
| `slab_thickness` | 1.0 m | ~0.36 m | vs 0.48 m measured |
| `density` | 200 | — | 307 kg/m³ measured |

Fitted values come from a small pilot and have been superseded by a larger
stratified run; treat the direction as reliable and the precise numbers as
provisional. The consistent signal across every fit is **much more friction
and much less turbulent drag than the defaults**.

---

## Open questions to ask him

- Was the entrainment flag ever wired up, or is it staged for later?
- What is the `predictor` channel in `compute_release_areas.wgsl` for? It's a
  reserved slot in the release texture, hardcoded to `0f`. It looks like the
  hook for a learned or data-driven release term — is that the intent?
- Why is `wind_shelter_index` computed by `computeWindShelter.wgsl`, read in
  `compute_release_areas.wgsl`, and then not used in the release decision?
  Abandoned, or staged?
- Is `slab_thickness` intended to stay a global scalar? The release texture's
  R channel is already per-cell, so a spatially varying slab would be a small
  change with a potentially large effect.
- Which evaluation metric does he consider primary — area overlap or runout
  distance? They disagree about whether measured snowpack forcing helps.
- Is the silent `max(inside, 0.0)` clamp in the earth-pressure coefficient
  (`grid_physics.wgsl:151`) intended as a physical regime switch, or just
  numerical protection? See item 27 — it changes what low
  `internal_friction_angle` values mean.
- Is `velocity_threshold = 1e-6` deliberate? It is read raw as the
  particle-stop and no-friction cutoffs (`compute_particles.wgsl:220,256`)
  where 1e-6 m/s is effectively zero, but floored to 1e-3 at the dt site
  (line 142). If the floor is the real intent, the other two sites may be
  meant to share it. (Measured 2026-07-28: 1e-6 vs 1e-3 on the 105-case
  panel at the frozen vector is inert — 94/104 bit-identical, the rest
  within atomics noise — so this is a question about intent, not about a
  live effect at these magnitudes.)
