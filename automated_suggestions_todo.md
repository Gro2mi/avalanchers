# Automated suggestions and open debts

> **\* Machine-generated, and I have not read it.** I asked an agent to pull
> the reusable parts out of my private working notes; it also went looking and
> added some items of its own that I never wrote down. I have not reviewed the
> result, so nothing here carries my sign-off — not the physics, not the method
> claims, not the priorities. It is a starting list, not a set of assertions.
> Confirm anything before acting on it, and treat disagreements with the code
> as the code being right.
>
> — Cole

Notes for whoever picks this up next. Everything here either fixes something in
the shared codebase or is needed to re-run the calibration campaign and get
comparable numbers. Nothing here is a demand; take what is useful and ignore
the rest.

Cross-references: per-dataset provenance is in `data/pools/<year>/SOURCE.md`.
The campaign method document and the results report are not yet in this
repository.

---

## 1. Read this before quoting any Ω_T number

The ground-truth reproducibility ceiling at this imagery resolution is
**≈ +0.28** — two expert mappers given the same SPOT6 scene agree with each
other only that well (Hafner et al. 2023). Per-event calibration reaches
**+0.295**, which is *at* the noise floor of the mapping, not near it.

So Ω_T should always be reported against ~0.28, never against 1.0. A model
scoring 0.3 is not "30% good"; it is reproducing the observations about as
well as the observations reproduce themselves. This single framing changes
how every result in the report should be read, and it is the easiest thing
in the whole project to get wrong.

---

## 2. Code debts

**No CPU-side shader validation.** This is the highest-value item on the list.
WGSL is only validated when a pipeline is created, so a typo in a `.wgsl` file
survives `cargo test` on any machine without a working GPU — and quietly ships.
A `compute_core` unit test that naga-parses and validates every file in
`src/shaders/` would close it, and runs anywhere. Skip
`computeTrajectories.wgsl`, which has its own divergent `SimSettings` layout.

**`computeTrajectories.wgsl` is orphaned.** It is never referenced and carries
a `SimSettings` layout that no longer matches `settings.rs`. Either wire it back
up or delete it — as it stands it is a trap for anyone who edits the shared
settings struct and assumes all shaders agree.

**`test_compute` is pinned to Voellmy, and a samosAT equivalent is missing.**
Every numeric bound in that test is a Voellmy measurement — the 6000-step
budget, peak velocity 42–90 m/s, and the tracked-particle `vel_x > 30 after
step 500`. samosAT is simply the slower model on that inclined-plane example:
peak velocity 30.2 m/s rather than 47.3, and the last particle arrests at step
7354 rather than 4240. Run out to 20000 steps everything stops, so nothing is
unstable — the old 6000-step cap was just truncating the flow mid-run. The test
therefore pins the model rather than inheriting the default, because retuning
its bounds to samosAT would move the numbers without preserving what they test.
A samosAT regression case with its own measured bounds is worth adding.

**Related, and worth checking before the next campaign:** the numerics were
frozen (tier 0) under the Voellmy incumbent, and the structure stage then
replaced it with samosAT, which needs ~74% more steps to finish the same flow.
Whether `max_steps 3000` binds more often under the frozen model than it did
under the model the budget was measured with is a question the off-optimum
numerics gate should answer explicitly.

**~~Four pre-existing test failures.~~ Resolved.** The two `data_processor`
ESRI parsing failures are fixed by the current parser; the two GPU tests pass
on a real Vulkan adapter and were failing only on a Metal host. Full suite is
180 tests, 0 failures. Original note follows.

**Four pre-existing test failures.** Two in `data_processor` (ESRI ASCII grid
parsing) and two GPU tests in `simulation` that fail on Metal. The GPU pair may
well pass on a Vulkan box; worth checking, then fixing or documenting as
known-Metal-failures so a red suite stops being normal.

**`tracing_subscriber::fmt::init()` is missing in `calibrate`.** The adapter
enumeration logs (including which physical GPU a `--gpu-index` shard actually
claimed) are emitted but never printed.

---

## 3. Environment and dependency management

Right now the build environment is prose in the README plus a full Vulkan
package list that exists only inside `.github/workflows/build.yaml`. Anyone
setting up a local box has to reverse-engineer CI to find it. Concretely:

- **Track `Cargo.lock`.** The workspace ships a binary; without the lockfile
  two people can silently get different `wgpu` patch versions. One-line fix.
- **Fold the two loose `requirements.txt`** (`python_scripts/`, `data/lawis/`)
  into the project definition rather than leaving them unpinned and orphaned.
- **Consider pixi.** It reads its configuration from `pyproject.toml` as
  `[tool.pixi.*]` tables, so it does not add a competing config file: one
  `pixi.lock` can pin the Python packages, the geo stack (GDAL/GEOS/pyogrio via
  conda-forge, which is where pip hurts most), and the Rust toolchain, with
  `[tool.pixi.tasks]` replacing the README's install prose. That also makes a
  separate `rust-toolchain.toml` unnecessary.
- **The GPU driver stays a host prerequisite.** No package manager installs a
  Vulkan ICD. But the package list currently buried in CI should be lifted into
  the README as an explicit prerequisites section.

---

## 4. Things that will bite you when re-running the campaign

**`ingest_evals.py` silently ingests 0 rows for one-shot jobs.** Their eval logs
are named `<name>.json.evals.jsonl` (embedded `.json`), which misses the
`<job>.evals.jsonl` sibling convention the ingester expects — and nothing warns.
It silently produced empty ingests for a screening stage and sixteen runs before
it was noticed. Fix: make the one-shot naming match, or make the ingester warn
on any eval log it did not match.

**One-shot runs all share `candidate_id = "frozen"`.** When several one-shot runs
are ingested together, `candidates` keeps only the last parameter row and the
per-case best-of tables collapse the whole profile into a single best-of-N — so a
question like "how does the score vary along this axis?" silently returns the
wrong shape of answer. Fix: tag one-shot runs with a value-encoding candidate id.

**1999 has four dead attribute columns**, and one of them is dangerous:
`start_zone` is all zeros, so any elevation-dependent logic silently treats the
entire 1999 dataset as being at sea level. `aspect`, `dpo_alt` are also fully
null and `trg_typ` is 99.97% null — which is why the 1999 filter funnel skips
the `trg_typ = NATURAL`, drop and start-zone stages that 2018 and 2019 apply.
**The 1999 pool is therefore filtered less strictly than the other two**, which
matters whenever scores are compared across datasets. See
`data/pools/1999/SOURCE.md`.

**~~Regenerate `data/pools/1999` cleanly.~~ Done.** Its extraction wrote explicit
JSON nulls for the dead columns; the committed manifests now omit them, matching
what was actually run. The harness is null-tolerant either way.

---

## 5. Method — how to get different results deliberately

**The pool funnel is strict on purpose.** Current yields: 2018 → 602 of 18,737;
2019 → 443 of 6,041; 1999 → 457 of 11,120. The isolation filter alone (no
neighbouring avalanche within 25 m, so an outline can be attributed
unambiguously to one release) removes about 85% of everything that survives the
other stages. If more volume is wanted, the dials are the isolation radius, the
area bounds (2–60 ha) and `--keep-multipart-largest`. Relaxing any of them
trades attribution cleanliness for sample size — a deliberate decision, not a
default. Note that "~20k events" in older prose is a polygon count, not a clean
case count; ~1.5k clean cases across three storms is the current truth.

**Pre-harvest off-optimum numerics gate.** The numerics were validated at
per-case optima, but a search spends most of its evaluations off-optimum, where
runs lengthen and `max_steps` starts to bind. Before a large harvest, sample
off-optimum vectors and confirm the frozen-vs-finer ordering still holds. This
is the same reasoning that rejected `max_steps 1500` and `ppc 4` despite clean
confidence intervals.

**`min_residence` does not normalize for flow duration.** A fast-arresting case
lost 21% of its footprint at threshold 1.0; 0.25 was chosen partly for this. A
duration-aware residence criterion is a plausible refinement.

**Pit tolerance (2 m) has no measured upper bound** outside the 2018 panel.
Worth re-examining on terrain with real closed basins.

**Density is resolved and should stay resolved.** AvaFrame samosAT's `R_s` is
density-free by construction — ρ cancels against σ_b — so the hard-coded 200 in
the friction bracket was a bug. The fix makes `R_s` use the settings density, so
density now cancels by construction and exits the samosAT parameter set the way
ξ did. Arithmetic is identical at the frozen value, so prior results stand.
Still worth confirming on a real Vulkan box: a pipeline-creation smoke test of
the edited shader, and a panel spot-check at density 100/200/300 expecting
near-identical footprints.

---

## 6. Worth doing to 1999 specifically

1999 is currently an applicability datapoint rather than a full second dataset,
and that may be fixable rather than inherent. Check whether the **source**
shapefile carries `start_zone`, deposit altitude and aspect — the published
mapping schema lists them, so they may have been dropped during extraction
rather than never recorded. Likewise `trg_typ`: recovering it would let 1999 be
NATURAL-filtered like the other two, removing one of the confounds in any
transfer comparison.

The published 1999 distribution also ships two auxiliary layers —
`area_images_1999.shp` (image coverage) and `clouds_1999.shp` (cloud outlines).
Those give absence-of-outline semantics: without them, "no avalanche mapped
here" and "nothing could be seen here" are indistinguishable.

If a humidity or wetness attribute can be recovered, the wet/dry contrast
*within* 1999 becomes possible — which is the cleanest available regime
experiment, because terrain and mapping era are held fixed.

---

## 7. Suggested next experiments

**Per-dataset freezes and transfer tests.** Freeze a vector on dataset A, apply
it unchanged to dataset B's panel, and compare against B's own freeze. "General"
is then something earned rather than assumed.

Confound discipline matters more than speed here. 2019 first, because it shares
the acquisition and mapping regime with 2018 and so isolates "different storm"
as a single factor. Then 1999-dry against 2018-dry, which establishes the
terrain-and-mapping-era nuisance floor. Only then 1999-wet against 1999-dry,
which is the clean physics contrast. A pooled global vector should ship only if
transfer actually holds; if it does not, the per-dataset vectors are the more
interesting result and should not be averaged away.

**Out-of-storm generalization is the real test** for any conditions model:
train on 2018, test on 2019. Both an MDN and gradient boosting are reasonable
candidates; the choice matters less than the split.

---

## 8. Data provenance

All three outline pools now carry a `SOURCE.md` recording the source dataset,
authors, DOI, licence, distribution filename, exact byte count and sha256, the
filter funnel that produced the pool, and the dataset's role in the campaign.

Two licences are in play and they are **not** the same:

| pool | source | DOI | licence |
|---|---|---|---|
| 2018 | SPOT6 avalanche outlines, 24 Jan 2018 | 10.16904/envidat.77 | ODbL |
| 2019 | SPOT6 avalanche outlines, 16 Jan 2019 | 10.16904/envidat.235 | ODbL |
| 1999 | Avalanche outlines Feb–Mar 1999, aerial imagery | 10.16904/envidat.579 | CC-BY-SA 4.0 |

Attribution is required for all three. Do not carry the ODbL wording over to
1999 or vice versa.

Only the filtered case pools are committed (1,502 cases, ~49 MB). The master
distributions are ~350 MB of zip and stay out of the repo; every sidecar names
the DOI and sha256 needed to fetch and verify the original.
