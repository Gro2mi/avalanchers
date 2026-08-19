# Code atlas

One line per source file, so a newcomer — or an agent — can find the right file
without grepping. Code only: what each file is, not what the project has found.

This file is generated. To change an entry, edit the source file's summary line
— in Rust the first `//!` doc comment, elsewhere an `@atlas: ...` comment — and
run `python3 python_scripts/generate_code_atlas.py`; the pre-commit hook
(`.githooks/pre-commit`) does this automatically on every commit.

## crates/compute_core — GPU orchestration and physics

| file | what it is |
|---|---|
| `src/buffers.rs` | `GpuResources`: named buffer/texture registry (`BufferName`, `TextureName`), allocation, upload, and readback with row-alignment handling. |
| `src/dem.rs` | `Dem`, `Bounds`, `GeoTiff`/`GeoMetadata`/`TiffData` — the elevation grid type and GeoTIFF value accessors. |
| `src/evaluation.rs` | The two scoring indices: `evaluate_mass_movement_area` (Ω_T = α−β−γ over the union) and `evaluate_distance_weighted_mass_movement_runout` (HWRI, apex-distance weighted, asymmetric via λ). |
| `src/lib.rs` | `ComputeOrchestrator`: wgpu adapter/device setup (`new_with_gpu_index` pins an adapter), buffer+texture creation, and the per-step shader dispatch loop (`reset_grid → p2g → grid_physics → compute_particles → update_sim_info`, batched 200 steps per submit). Defines `Particle`, `SimInfo`, `TimestepData`, `SimInfoFlags`. |
| `src/settings.rs` | `SimSettings` (the `#[repr(C)]` struct uploaded verbatim as a shader uniform) and `Settings` (the optional-field JSON patch). Repo defaults live in `SimSettings::new()`. `FrictionModel` enum. |
| `src/shaders.rs` | Shader loading and pipeline construction: `include_str!`s each WGSL file, splices `utils.wgsl`/`random.wgsl` in at `// import` lines, builds bind group layouts per `ShaderName`. |
| `src/utils.rs` | Odds and ends: `Point`, `linspace`, `to_2d`, `bilinear_interpolate`, `flip_rows_flat_vec`, the `MaxValue` trait, and the checkpoint timer. |

## crates/compute_core/src/shaders — WGSL

`utils.wgsl` is textually inlined into every other shader, so its `SimSettings`
struct must stay byte-compatible with `settings.rs`.

| file | what it does |
|---|---|
| `analyze_terrain.wgsl` | DEM → normals, profile curvature, slope angle, aspect. Also copies the wind-shelter index into the slope texture's B channel. |
| `computeTrajectories.wgsl` | Standalone single-point trajectory tracer: integrates up to 3 point trajectories from an input point, recording per-step velocity/acceleration/position and scattering trajectory + velocity textures. Defines its own private `SimSettings` layout and is not referenced by `shaders.rs` — orphaned. |
| `computeWindShelter.wgsl` | 75th-percentile horizon angle over 21 rays upwind. Computed, stored, and never read by the release decision. |
| `compute_particles.wgsl` | The main step: tangent-plane projection, gravity + curvature + G2P lateral force, then the friction closures (`acceleration_by_normal_friction` / `..._drag_friction` — Coulomb / Voellmy / VoellmyMinShear / samosAT / µ(I)). Position update, out-of-bounds and NaN handling. |
| `compute_release_areas.wgsl` | Gates each cell on slope band, minimum elevation, roughness threshold and forest; writes `slab_thickness` into the release texture. Holds the unused `predictor` slot. |
| `compute_roughness.wgsl` | 3×3 vector-ruggedness (VRM) stencil on the normals, plus forest flag passthrough. |
| `grid_physics.wgsl` | Grid solve: recovers flow thickness, earth-pressure coefficient from velocity divergence, lateral pressure force ∝ ∇(h²). Tracks newly-conquered cells for the stop criterion. |
| `initialize_particles.wgsl` | Seeds `released_particles_per_cell` particles per release cell with jittered positions and mass = cell volume × density. |
| `load_release_areas.wgsl` | Alternative release path: reads a supplied texture and scales it by `slab_thickness` (the >0.01 cutoff here must match `initialize_particles`). |
| `p2g.wgsl` | Particle-to-grid scatter of mass and momentum into integer atomics (rounds rather than truncates — local fix). |
| `random.wgsl` | PCG hash + `rand1..4`, used to jitter particle positions within a cell. |
| `reset_grid.wgsl` | Zeroes the grid mass/momentum atomics each step. |
| `test.wgsl` | Trivial shader used by the shader-report test. |
| `update_sim_info.wgsl` | Single-thread pass: advances the timestep, recomputes `dt` from peak velocity, sets the STOPPED / NO_NEW_CELLS termination flags. |
| `utils.wgsl` | Shared prelude: constants, `Particle`/`SimInfo`/`SimSettings`/`AtomicValues` structs, quantisation factors, cell↔uv↔index helpers, MPM quadratic weights. |

## crates/simulation

| file | what it is |
|---|---|
| `benches/sim_benchmark.rs` | Criterion benchmark over the AvaFrame `avaMal` example. |
| `src/lib.rs` | `Simulation`: the stateful façade over the orchestrator. `SimulationState` ordering enforces normals → release areas → particles → run; `set_dem_with_bounds` / `set_release_areas` are the array-based entry points the calibration harness uses; `fetch_*` methods read back through `GpuCache`. `init_logging()` lives here. |

## crates/data_processor

| file | what it is |
|---|---|
| `src/caaml_parser.rs` | CAAML avalanche-bulletin parsing: `AvalancheBulletinCollection` / `Bulletin`, `DangerRating` (with numeric conversion) and `AvalancheProblem`, plus human-readable summaries. |
| `src/lib.rs` | File-format I/O: PNG/GeoTIFF/ESRI-ASCII DEM loading, release-texture loading, settings JSON helpers, `create_sim_settings_and_dem`. |
| `src/output.rs` | Writes simulation results out as Zarr arrays with CF-style metadata. |
| `src/rasterizer.rs` | `RasterGrid`: polygon → binary grid by scanline fill with a cell-centre rule, `add_padding`, and `to_padded_tile` for reprojecting onto a DEM's exact frame. |
| `src/shapefile_reader.rs` | Shapefile → `GeoPolygon`; `read_shapefile_nth_polygon` is what the harness calls per case. |
| `src/tile_manager.rs` | swissALTI3D fetch and cache. Tries 8 acquisition years per tile (the shipped code hardcoded 2019), falls back to `decode_tiled_lzw_f32` for tiles whose LZW stream lacks an end code, resamples 2 m → 5 m by area weighting, stores into a Zarr array. `get_dem(bbox)` is the entry point. |
| `tests/caaml_parser_tests.rs` | Tests for the CAAML bulletin parser, against `tests/fixtures/caaml_sample.json`. |

## crates/cli

| file | what it is |
|---|---|
| `src/bin/calibrate.rs` | **The calibration harness — most of our work is here.** Prepares cases (rasterise outline → fetch padded DEM → Horn slope/aspect → build release band), evaluates against the observed outline, and drives the search. Subcommands: `run`, `sweep`, `search` (one global vector), `per-event` (per-case Nelder-Mead, logs every evaluation to `.evals.jsonl`), `grid` (2-D μ×ξ identifiability scan), `apply` (each case at its own vector — the only honest way to score a regressor), `dump` (rasters for plotting). `Params` + `BOUNDS`/`DIM_NAMES` define the search space; `--gpu-index` pins a card. |
| `src/main.rs` | The upstream CLI: run one simulation from a `settings.json`. |

## Bindings and frontend

| file | what it is |
|---|---|
| `avalanchers_example.py` | Minimal usage example against an AvaFrame case. |
| `crates/python_bindings/src/lib.rs` | PyO3 wrapper: `PySimulation`, numpy in/out. |
| `crates/wasm_bindings/src/lib.rs` | `WasmSimulation` for the browser build. |
| `frontend/dev_server.py` | HTTPS dev server with self-signed certs (WebGPU needs a secure context). |
| `frontend/js/main.js` | Browser app wiring: DEM/GPX loading, settings UI, drives `WasmSimulation`. |
| `frontend/js/plot.js` | Plotly 2-D/3-D result rendering, the dark theme, and percentile-clamped colour limits for velocity and flow thickness. |
| `frontend/js/plotly_dark.js` | Plotly 2-D/3-D result rendering and the dark theme. |
| `frontend/js/tile_utils.js` | Web-Mercator tile maths and small array helpers. |
| `frontend/js/utils.js` | Web-Mercator tile maths and small array helpers. |
| `python_module/avalanchers/__init__.py` | Python-side helpers: `create_mesh`, `plot2d`, `plot3d` (the `[viz]` extra). |
| `python_module/tests/test_run.py` | Pytest smoke tests for the PyO3 bindings: end-to-end runs from settings JSON and the packaged example, plus numpy DEM / release-area roundtrips. |
| `python_module/tests/test_verification.py` | Empty placeholder — no verification tests yet. |

## python_scripts

| file | what it is |
|---|---|
| `analyze_stage.py` | Turns one ingested stage into a freeze decision: per-candidate panel means, paired bootstrap CIs and Wilcoxon against the incumbent, cal/val consistency, and the explicit accept/keep verdict from the plan's criterion. |
| `export_parquet.py` | Exports the calibration SQLite database to per-stage Parquet, plus the joined candidate metadata. Columnar copies for the final report and for anything that wants to read the campaign without SQLite. |
| `extract_cases.py` | Blocker 0 -- master avalanche-outline shapefile (2018/2019/1999) -> per-case shapefiles + JSON panel, reusing campaign/analysis/census.py's filter funnel. |
| `generate_code_atlas.py` | Scans the repo for per-file summary lines (Rust `//!` module docs, `@atlas:` comments elsewhere) and regenerates code_atlas.md — the auto-generated successor to the hand-written file atlas, modeled on Hexagons/bundle.py. |
| `ingest_evals.py` | Loads calibrate's per-evaluation JSONL into one SQLite database with stage/candidate/case/iteration provenance. Idempotent: re-ingesting a file replaces its rows. Rasters are never stored -- footprints only, via `calibrate dump`, for final winning vectors. |
| `make_stages.py` | Generates the stage manifests for the global-constants calibration (campaign/GLOBAL_CALIBRATION_PLAN.md). One JSON per stage: panel, candidate list, inner-loop args. Consumed by run_stage.py. |
| `run_stage.py` | Resumable driver for one stage of the global-constants calibration. Expands a manifest from make_stages.py into (candidate x case-block) jobs, runs them across (GPUs x per-GPU) workers, atomic per-job writes so an interrupt loses at most one block. |
| `shard_calibrate.py` | Resumable batch runner: splits a case list across (GPUs × per-GPU) workers, one `calibrate run` subprocess per case, atomic per-case writes so an interrupt loses at most one case. `--per-gpu 8` is the measured saturation point. |
| `validate_fixes.py` | On-GPU before/after evidence for the two scoring-convention fixes (single-particle tails, ridge/watershed release leak). Run this on the box before any tuning stage; writes fix_validation.json for the final report. |

## campaign/analysis — one-shot analysis scripts

These read scratchpad inputs (raster dumps, per-event calibration output) that
are **not** in the repo; they are kept so the method survives, not to re-run
unmodified. Set `D` at the top of a script to a directory holding the inputs.

| file | what it measures |
|---|---|
| `census.py` | Composition of the 18,737-polygon mapping and the filter funnel down to the candidate pool. |
| `crossing_analysis.py` | Share of simulated mass leaving the drainage the observed avalanche used. |
| `divide_analysis.py` | Whether the constructed release area straddles a drainage divide (D8 outlets, 150 m clustering). |
| `e2e_stats.py` | Paired bootstrap + Wilcoxon over the end-to-end variants. |
| `endtoend.py` | Builds per-case parameter files from out-of-fold predictions, for `calibrate apply`. |
| `final_pipeline.py` | The corrected architecture: ξ fixed → μ/slab calibrated at that ξ → regressed on deployable features → re-simulated. |
| `frag_analysis.py` | Connected components of the simulated footprint vs the flow threshold. |
| `ident_analysis.py` | Shape of Ω_T over the (μ, ξ) grid — near-optimal area, aspect ratio, orientation, transferability, covariate correlations. |
| `learning_curve.py` | OOF R² from n = 40 to 105 — separates "underpowered" from "no signal". |
| `regress_pilot.py` | Out-of-fold R² for μ/slab/log ξ across feature sets, with permutation tests and importances. |
| `render_e2e_figs.py` | Figures for the settled results. |
| `terrain_features.py` | Derives 16 DEM/release features and their raw correlations with the fitted optima. |
| `terrain_regress.py` | Cross-validated skill of terrain features vs outline features. |
