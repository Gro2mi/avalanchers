# PROJECT CONTEXT — avalanchers

> Persistent context document for AI coding sessions. Generated 2026-09-16 from the working tree of
> branch `new_models` (includes uncommitted work: the MPMDAC model, `relax_*.wgsl` shaders,
> `docs/mpmdac.md`, `avalanchers_ana1_tests.py`, `avalanchers_evaluation_example.py`).
> All paths are relative to the repository root. All identifiers are exact.

## 1. Project Overview

**avalanchers** is a GPU-accelerated dense-flow snow-avalanche simulation built on
WebGPU (`wgpu` 30) compute shaders. It is a Material Point Method (MPM) solver that runs a
depth-averaged granular flow over a digital elevation model (DEM). It descends from the
weBIGeo project. License GPL-3.0-only, author Markus Rampp, version 0.2.0.

The Rust core is exposed through four frontends:

1. **CLI** (`crates/cli`) — native binary that loads a `settings.json` and runs a simulation.
2. **Python package** `avalanchers` (PyPI) — pyo3/maturin bindings (`crates/python_bindings`
   → `avalanchers._avalanchers`) wrapped by `python_module/avalanchers/` with matplotlib/pyvista
   plotting.
3. **WASM/JS web frontend** (`crates/wasm_bindings` compiled by wasm-pack into
   `frontend/js/pkg/`, consumed by the static site in `frontend/`) — runs the same simulation in
   the browser with live 3D rendering.
4. **Native GUI example** (`crates/render_core/examples/live_sim.rs`) — winit + egui 3D viewer
   that renders the simulation's GPU buffers with zero CPU copies.

### Simulation models (`sim_model` setting)

| id | enum variant (`compute_core::settings::SimModel`) | strings accepted | pipeline shaders (p2g / grid physics / particle update) | description |
|----|---|---|---|---|
| 0 | `TerrainFollowing` | `"terrain-following"`, `"terrain"`, `"t"` | `p2g_terrain_following` / `grid_physics_terrain_following` / `g2p_terrain_following` | depth-averaged MPM in 3D Cartesian coordinates (bed normals from `analyze_terrain`), 3D APIC transfer (z parts in `GridMomentumZ`, `GridVelocityZ`, `ParticlesAffineZ`), grid friction; particles take a curvature-corrected arc step (`cartesian_geometry.wgsl`), snap to the DEM and are rotated speed-preserving into the new tangent plane |
| 1 | `Curvilinear` | `"curvilinear"`, `"curvi"`, `"c"` | `p2g` / `grid_physics_curvilinear` / `g2p` | depth-averaged MPM in curvilinear terrain coordinates (metric terms `l_x`, `l_y`, `J`), PIC transfer, algebraic friction on the grid |
| 3 | `SlidingBlock` | `"sliding-block"`, `"block"`, `"b"` | `p2g` / `grid_physics` / `compute_particles` | legacy per-particle 3D sliding block (former model 0) with optional grid-based lateral pressure; used by the particle-simulation crown-line detection |
| 2 | `MpmDaC` (**MPMDAC**, new on this branch) | `"mpmdac"`, `"mpm-dac"`, `"mpm_depth_averaged_curvilinear"` | `p2g_mpmdac` / `grid_physics_mpmdac` / `g2p_mpmdac` | compressible depth-averaged MPM in terrain-following horizontal coordinates; per-particle elastic predictor–corrector stress (Drucker–Prager or µ(I) yield), volumetric elasticity + plastic compaction, APIC transfer |

The MPMDAC model is fully documented in `docs/mpmdac.md` (untracked, 424 lines) — read it before
touching model 2 physics.

### Design highlights (non-obvious)

- **All physics runs in WGSL compute shaders**; Rust only orchestrates dispatches, buffers, and
  readbacks. There is no CPU physics fallback.
- **Shared atomics without `atomic<f32>`**: grid mass/momentum/forces are atomically accumulated
  as `u32`/`i32` using fixed-point quantization factors defined in `shaders/utils.wgsl`
  (`MASS_FACTOR = 1e1`, `MOMENTUM_FACTOR = 1e2`, `FORCE_FACTOR = 1e-3`, `MAX_VELOCITY_FACTOR = 1e7`,
  `H_FACTOR = 1e6`). When the adapter has the `SHADER_FLOAT32_ATOMIC` feature, a shader-source
  preprocessing step switches these to real `atomic<f32>` (see `crates/compute_core/src/shaders.rs`
  `load_shader_source_string`).
- **Shader "import" system**: `// import name.wgsl;` comment directives are resolved by regex at
  pipeline-creation time; in native **debug** builds the expanded source is written back to disk,
  so the committed `.wgsl` files contain the hand-written source at the top followed by stale
  inlined `// BEGIN x.wgsl … // END x.wgsl` blocks (stripped before compile). **Edit only the top
  section of a shader file.**
- **PyO3 futures**: `python_bindings` blocks on the async `Simulation` methods via tokio
  current-thread runtimes / `block_on` — the Python API is synchronous.
- The **renderer borrows the simulation's device/queue** (native `live_sim` and the WASM
  `RenderView` both clone wgpu handles), so overlays read buffers the sim keeps writing without
  CPU copies.

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Language | Rust, edition 2024 (workspace version 0.2.0). No `rust-toolchain.toml` — uses the installed default toolchain |
| GPU | `wgpu` 30.0 (WebGPU), WGSL compute shaders (39 files), `bytemuck` POD transfers |
| Native GUI | `winit` 0.30, `egui` 0.36 (+egui-wgpu/-winit) |
| CLI | `clap` 4.6 (derive) |
| Python bindings | `pyo3` 0.28 (`auto-initialize`, `abi3-py37`), `numpy` 0.28, `pythonize`, `pyo3-log`, built with **maturin** (`pyproject.toml`, module name `avalanchers._avalanchers`) |
| WASM | `wasm-bindgen` 0.2, `js-sys`, `web-sys`, `serde-wasm-bindgen`, `tracing-wasm`, `console_error_panic_hook`; built with **wasm-pack** (`--target web`) |
| Data I/O | `tiff` 0.11 (GeoTIFF), `shapefile` 0.8, custom ESRI ASCII reader, `image` 0.25 (PNG DEMs), `zarrs` 0.23 (+filesystem/plugin/datatype) for Zarr v3 stores, `lz4_flex`, `bincode`, `half` (float16), pure-Rust blosc/zstd decode (`ruzstd`, `weezl`) for wasm32 |
| Networking | `reqwest` 0.13 (swissALTI3D / BEV tile download), `wiremock` (tests) |
| Logging | `tracing` + `tracing-subscriber` (env-filter), `test-log` |
| Testing | `cargo test`, criterion 0.5 benches, `wasm-bindgen-test`, pytest (`python_module/tests`), ruff lint |
| CI | GitHub Actions: `.github/workflows/{build,release,gh-pages-deploy,sftp-deploy}.yaml` (software Vulkan/Mesa runners; Tarpaulin coverage ≥80%) |
| Frontend | Vanilla JS modules + Plotly (dark theme bundled as `plotly_dark.js`), Bootstrap-styled `index.html` |
| Python analysis deps | numpy>=2.0, zarr>=3.0, xarray (runtime); matplotlib/pyvista/vtk (`viz` extra); scipy **optional** (analytical tests fall back to RK4) |

`.cargo/config.toml` is required for the wasm build:
`rustflags = ['--cfg', 'getrandom_backend="wasm_js"']` for `wasm32-unknown-unknown`, and
`linker = "rust-lld.exe"` for `x86_64-pc-windows-msvc`.

## 3. Directory Structure

```
avalanchers/
├── Cargo.toml                     # workspace: 7 crates, shared deps
├── Cargo.lock
├── pyproject.toml                 # maturin config for the `avalanchers` Python package
├── README.md / CHANGELOG.md / LICENSE / NOTICE
├── .cargo/config.toml             # wasm getrandom flag; rust-lld linker
├── .github/workflows/             # CI: build, release, gh-pages-deploy, sftp-deploy
├── settings.json                  # default run config (curvilinear on avaWog)
├── settings_outline.json          # (untracked) outline-driven evaluation config
├── settings_vals_example.json     # Vals GeoTIFF config (uses legacy key `slab_thickness` — stale)
├── settings_vals_all_pra.json     # Vals GeoTIFF config (same legacy key — stale)
├── avalanchers_ana1_tests.py      # analytical validation tests (AvaFrame ana1Tests port)
├── verification/                  # AvaFrame-port verification scripts (run their sim in-script)
│   ├── energy_line_test.py        # energy line analysis + plots, AvaFrame port, numpy/matplotlib only
│   ├── run_energy_line_test.py    # runs an avalanchers sim in-script and feeds energy_line_test
│   ├── dam_break_test.py          # dam break analytic solution + grid-field decoding + plots
│   ├── run_dam_break_test.py      # runs an avalanchers sim in-script, compares to Faccanoni-Mangeney
│   ├── rotation_test.py           # rotation test analysis: raster rotation, path cross-sections, plots
│   ├── run_rotation_test.py       # runs rotated DEM/release sims in-script, compares in reference frame
│   ├── similarity_solution_test.py  # Hutter 1993 similarity solution + grid-field decoding + plots
│   ├── run_similarity_solution_test.py  # runs the parabolic-heap sim in-script, compares to simiSol
│   ├── run_all_tests.py           # runs all 4 tests in-process with switchable sim_model + markdown report
│   └── outputs/                   # (untracked) PNG plots written by the run scripts
├── avalanchers_example.py         # minimal end-user example
├── avalanchers_evaluation_example.py  # (untracked) outline → release estimation → evaluate
├── avalanchers_example.ipynb      # notebook version of the example
├── docs/
│   ├── mpmdac.md                  # (untracked) full MPMDAC design doc — READ BEFORE model-2 work
│   └── release_area_estimation_section.tex  # (untracked) LaTeX write-up
├── crates/
│   ├── compute_core/              # GPU engine: orchestrator, buffers, settings, shaders
│   │   └── src/shaders/*.wgsl     # 39 WGSL files (35 tracked + 3 relax_* untracked + constitutive)
│   ├── data_processor/            # DEM/outline/Zarr/tile I/O
│   ├── simulation/                # high-level Simulation state machine + release estimation
│   ├── render_core/               # windowing-agnostic 3D renderer + 4 examples
│   ├── wasm_bindings/             # wasm-bindgen API (crate name `avalanchers`)
│   ├── python_bindings/           # pyo3 API (lib name `_avalanchers`)
│   └── cli/                       # native binary (crate `cli`, command `avalanchers`)
├── python_module/                 # Python source shipped on PyPI
│   ├── avalanchers/__init__.py    # plotting helpers re-exported next to the pyo3 module
│   ├── avalanchers/output.py      # Zarr result-store reader (AvalanchersOutput)
│   └── tests/                     # pytest suite (test_bindings, test_io, test_run, test_verification)
├── frontend/                      # static web app (no framework)
│   ├── index.html, dev_server.py  # HTTPS dev server (WebGPU needs a secure context)
│   ├── js/{main,viewer,plot,sources,tile_utils,utils,plotly_dark}.js
│   └── js/pkg/                    # wasm-pack build output (gitignored, built locally)
├── data/
│   ├── avaframe/                  # AvaFrame test cases: <case>.png DEM + <case>releaseTexture.png
│   ├── vals/                      # Vals PAR6 GeoTIFF DTM + release.tif (Zenodo)
│   ├── outline/                   # mapped avalanche outlines polygon_10720/10721.shp
│   ├── landcover/, lawis/         # research data + notebooks
│   └── (avalanche_outlines_switzerland_2018/, snow/, wind/ — gitignored research data)
├── python_scripts/raster.ipynb
└── outputs/ana1_tests/            # (untracked) PNG plots written by avalanchers_ana1_tests.py
```

## 4. Architecture

### 4.1 Crate layering

```
                       ┌──────────────────────────────┐
  frontends            │  cli   python_bindings   wasm_bindings   render_core examples
                       └───────┬──────────┬───────────────┬──────────────────┬──────────┘
                               │          │               │                  │
  orchestration                │      simulation  ←────────┼──────────────────┘
                               │          │    (Simulation state machine, │
                               │          │     release_estimation)       │
  GPU engine                   │       compute_core  ←─────┘ (ComputeOrchestrator,
                               │          │     SimSettings, GpuResources, ShaderName)
  data I/O                     └──→ data_processor  (create_sim_settings_and_dem, Output/Zarr,
                                            tiles, GeoTIFF/ASC/shapefile, blosc)

  compute_core ← no internal deps (foundation)
  data_processor → compute_core
  simulation → compute_core + data_processor
  render_core → (lib) no internal deps; examples dev-depend on simulation/compute_core/data_processor
  wasm_bindings → compute_core + simulation + data_processor + render_core
  python_bindings → compute_core + simulation + data_processor
  cli → simulation + compute_core + data_processor
```

### 4.2 GPU pipeline (per simulation step)

One `wgpu::CommandEncoder`, one compute pass per step, pipelines pre-created per model in
`ComputeOrchestrator::prepare_simulation` (`crates/compute_core/src/lib.rs:1478`):

```
(model 2 only) copy GridMass → GridMassPrevious        [lib.rs:1673-1706]
reset_grid          (2D dispatch, ceil(grid/16))       zero grid atomics + peak atomics
p2g | p2g_mpmdac    (1D dispatch, particles/64-or-256) particle → grid mass/momentum(/forces)
grid_physics | grid_physics_curvilinear | grid_physics_mpmdac
                    (2D dispatch)                      gravity, friction, pressure → grid velocity
compute_particles | g2p | g2p_mpmdac
                    (1D dispatch)                      particle update + advection + stopping
update_sim_info     (1,1,1)                            adaptive dt, stop flags
[enable_center_of_mass] center_of_mass_seed → center_of_mass_propagate (biggest-blob mode)
                    → compute_center_of_mass (1 workgroup, 256 threads)
```

Steps are batched: `batch_compute_steps` steps per `queue.submit` (the OS can reset the GPU if a
single submission runs too long). The outer loop stops early on `SimInfoFlags::SIM_STOPPED`.

### 4.3 Simulation lifecycle state machine (`crates/simulation/src/lib.rs:91`)

```
Uninitialized → DemMissing → DemLoaded → TerrainAnalyzed → ReleaseAreasComputed
  → ParticlesInitialized → Running → Finished → PostProcessed → Evaluated
```

`Simulation` methods enforce this ordering and bail with `anyhow` errors if prerequisites are
missing. `run_n_steps` auto-runs `prepare()` when needed.

### 4.4 Shader source management

- `ShaderName` enum (38 variants) generated by `define_shaders!` in `crates/compute_core/src/shaders.rs:74`;
  each variant maps to `src/shaders/<snake_case>.wgsl` and the **entry point name equals the file stem**.
- Native debug builds read shader sources from disk (hot reload) and write the import-expanded
  source back; wasm32/release builds `include_str!` the file as-is.
- `load_shader_source_string` (shaders.rs:127): strips `// BEGIN x.wgsl…// END x.wgsl` blocks →
  recursively inlines `// import x.wgsl;` → applies the `// atomic_float` / `// no_atomic_float`
  line switching → `Box::leak`s the result into a `'static str`.
- Binding index **0 is always the `SimSettings` uniform** in every shader (`utils.wgsl:105`).
- Per-shader bindings are declared in `create_shader_configs` (shaders.rs:357) as ordered
  `(name, BindingType)` lists; names resolve against `BufferName`/`TextureName` strings.

### 4.5 Buffer model

`GpuResources` (`crates/compute_core/src/buffers.rs:334`) owns `HashMap<BufferName, Buffer>` +
`HashMap<TextureName, Texture/TextureView>` + samplers. Key buffers (enum `BufferName`,
buffers.rs:106): grid atomics `GridMass(u32)`, `GridMomentum(i32 ×2/node)`,
`GridForces(i32 ×2/node)`, `GridMassPrevious` (MPMDAC snapshot), `GridVelocity(vec2)`,
`GridPeakVelocity`, `GridPeakFlowThickness`; particles `ParticlesPosition`, `ParticlesMass`,
`ParticlesVelocity`, `ParticlesVelocityZ` (model 0), `ParticlesAffineMatrix` (models 1/2),
`ParticlesStress` (model 2, `mat2x2` = τxx,τxy,τyy + cum. plastic strain),
`ParticlesVolumetricStrain` (model 2, `vec2` = elastic + plastic strain), `ParticlesState`,
`ParticlesElevation`; analysis `SlopeAngle`, `SlopeAspect`, `Roughness`, `ReleaseAreas`,
`RegionOfInterest` (bitpacked); results `TimestepData` (3 trajectory slots/timestep),
`CenterOfMass`, `EvaluationResult`, `AtomicValues`, `NewCellsRollingWindow` (40×u32); scratch
`ChamferNearestRoi/Sim`, `ChamferParams`, `RelaxParams`, `RelaxGridHead`, `ParticleNext`, `Debug`.
Textures: `Dem`, `TerrainGeometry`, `Curvature` (rgba32float).

Model-specific buffers are allocated in `ComputeOrchestrator::run_initialize_particles`
(`crates/compute_core/src/lib.rs:1230`): model 0 → `ParticlesVelocityZ` + `GridVelocity`;
model 1 → `ParticlesAffineMatrix`; model 2 → affine + `ParticlesStress` +
`ParticlesVolumetricStrain`.

## 5. Entry Points

| Entry point | How to run | What it does |
|---|---|---|
| **CLI binary** `crates/cli/src/main.rs::main` | `cargo run -p cli -- [settings.json] [--flag value …]` | loads settings JSON, applies clap overrides, `Simulation::new_with_settings` → `run()` → prints diagnostics → `save()` |
| **Library** `simulation::Simulation` | `cargo test`/custom binaries | `new_with_gpu` / `create*` / `prepare` / `run` / `run_n_steps` / `post_process` / `evaluate[_gpu]` / `save` |
| **Native viewer** `crates/render_core/examples/live_sim.rs` | `cargo run -p render_core --example live_sim -- [settings.json] [exaggeration]` | interactive winit/egui viewer; env `LIVE_SIM_SNAPSHOT=<path>` = headless run → PNG; `LIVE_SIM_NO_PARTICLES` |
| **Other render examples** | `cargo run -p render_core --example {dem_viewer,dem_snapshot,video_export}` | static DEM viewers; `video_export` pipes frames into ffmpeg |
| **Python** | `python avalanchers_example.py`; `maturin develop` first | `avalanchers.PySimulation` |
| **WASM** | `wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --release`, then `python frontend/dev_server.py`, open `https://localhost/` | `WasmSimulation` class, `init()` is `#[wasm_bindgen(start)]` |
| **Analytical tests** | `python avalanchers_ana1_tests.py simisol\|dambreak\|all [--t-end S] [--no-plots]` | needs a local AvaFrame checkout (`AVAFRAME_HOME`, default `C:/git/AvaFrame/avaframe`) |
| **Energy line test** | `python verification/run_energy_line_test.py [--dem …] [--release …] [--mu F] [--max-steps N] [--show]` | runs the sim in-script, samples the COM path live, saves the AvaFrame-style energy line plot to `verification/outputs/energy_line_test/` |
| **Dam break test** | `python verification/run_dam_break_test.py [--t-end S] [--save-interval S] [--delta Deg] [--plot-sequence] [--show]` | runs the sim in-script, decodes fields from grid_mass/grid_momentum, saves the AvaFrame-style dam break plots to `verification/outputs/dam_break_test/pics/` |
| **Rotation test** | `python verification/run_rotation_test.py [--angles 0 30 60] [--dem …] [--release …] [--mu F] [--show]` | runs rotated DEM/release sims in-memory, compares runout/fields in the reference frame, plots to `verification/outputs/rotation_test/pics/` |
| **Similarity solution** | `python verification/run_similarity_solution_test.py [--t-end S] [--delta Deg] [--plot-sequence] [--show]` | parabolic heap on the 35° plane vs Hutter 1993, fields from grid_mass/grid_momentum, plots to `verification/outputs/similarity_solution_test/pics/` |
| **All tests + report** | `python verification/run_all_tests.py [--sim-model mpmdac] [--tests …] [--quick] [--mu F]` | runs the 4 tests in-process (model switchable), writes `verification/outputs/all_tests/<model>_<ts>/report.md` with tables, summary plots and GIFs |
| **Benchmarks** | `cargo bench -p simulation` | criterion, `crates/simulation/benches/sim_benchmark.rs` |

## 6. Execution / Data Flows

### 6.1 Main simulation flow (all frontends)

```
Input (settings JSON / dict, DEM, release areas or outline)
  ↓
data_processor::create_sim_settings_and_dem  (crates/data_processor/src/lib.rs:854)
    Settings patch + DEM source (PNG/GeoTIFF/.asc/tile-download) + optional outline
    → (SimSettings, Dem, roi: Vec<bool>)
  ↓
Simulation::load_data → apply_data  (crates/simulation/src/lib.rs:300/325)
    SimSettings.set_dem(dem); upload DEM texture
  ↓
Simulation::prepare  (lib.rs:634)
    analyze_terrain (GPU: AnalyzeTerrain or AnalyzeTerrainCurvilinear)
  + upload_roi (bitpacked RegionOfInterest buffer)
  + load_release_areas  (lib.rs:1186: preloaded raster | crown-line estimation
    [simulation/src/release_estimation.rs] | GPU thresholds via
    ComputeRoughness → ComputeReleaseAreas)
  + initialize_particles → run_initialize_particles (GPU; model-specific buffers)
    [+ relax_particles: 30 iterations of RelaxClearGrid → RelaxBuildGrid → RelaxParticles,
     then RelaxSampleElevation once]
  ↓
prepare_<model>  → per-step compute pass loop (section 4.2), batched by batch_compute_steps
  ↓ stop when SIM_STOPPED / max_steps
Post-processing
    post_process(): threshold peak_flow_thickness → biggest connected blob → ava_mask (CPU)
  ↓
evaluate() [CPU IoU + runout] / evaluate_gpu() [GPU: EvaluateMassMovement →
    EvaluateMassMovementPoints → ChamferPrepare → ChamferFlood×log n → ChamferReduce
    → ComputeBeelineDistance]
  ↓
Output
    save() → Zarr v3 store (native); export_zarr_entries() → Vec<ZarrEntry> (wasm);
    fetch_* getters (numpy arrays in Python, Float32Array views in JS)
```

### 6.2 Adaptive timestep and stopping (`update_sim_info.wgsl`)

One invocation: if all particles stopped → flag `ALL_PARTICLES_STOPPED|STOPPED`; if the 40-step
rolling window of "cells touched for the first time" sums `< 4` → `NO_NEW_CELLS|STOPPED`; else
`timestep += 1`, `dt = max(0.01, cfl·cell_size / expected_max_velocity)` where
`expected_max_velocity` is the lazily-initialized CFL signal `V + sqrt(g·h)` (+ `sqrt(K/ρ)` in
model 2).

### 6.3 Python validation loop (`avalanchers_ana1_tests.py`)

```
read AvaFrame .asc DEM (AVAFRAME_HOME checkout)
  ↓ PySimulation.new() → sim.create(settings_dict) → sim.set_release_areas(thickness_m_array)
  ↓ loop: sim.run_n_steps(n) → SimInfo(.timestep/.elapsed_time/.dt/.flags)
  ↓ fields_from_grid(): decode sim.grid_mass (÷MASS_FACTOR=1e1) and sim.grid_momentum
    (÷MOMENTUM_FACTOR=1e2) into thickness/velocity — MUST match shaders/utils.wgsl
  ↓ compare against analytical similarity solution (ODE) / dam-break solution
  ↓ L2/LMax error tables to stdout, PNGs to outputs/ana1_tests/
```

### 6.4 Zarr exchange format

Written by `data_processor::output::Output` (`crates/data_processor/src/output.rs`), read by
`python_module/avalanchers/output.py::AvalanchersOutput` and the frontend (WASM decodes blosc
chunks via `wasm_bindings::decode_blosc_chunk`):

```
<store>.zarr/
  <site>_<16-hex-dem-hash>/        group; arrays dem, x, y
  └── <scenario>_<16-hex-hash>/    group; arrays per run: peak_flow_velocity,
        peak_flow_thickness, release_area, center_of_mass_x/y, …
        attrs: mu, xsi, release_volume_m3, number_of_runs, avalanchers_format="0.1.0"
```

## 7. Configuration and Environment

### 7.1 Settings schema

Two levels (defined in `crates/compute_core/src/settings.rs`):

- **`Settings`** (line 646) — user-facing, `#[serde(deny_unknown_fields)]`, every field
  `Option<_>`, serde JSON. Keys: `outlines_path`, `outlines_padding`, `dem_path`,
  `release_areas_path`, `output_path`, `max_steps`, `sim_model` (enum, accepts string or int),
  `batch_compute_steps`, `friction_model`, `released_particles_per_cell`, `density`,
  `slab_thickness_factor`, `friction_coefficient`, `drag_coefficient`, `n0`, `i0`, `mu0`, `mu2`,
  `grain_diameter`, `internal_friction_angle`, `basal_friction_angle`, `cfl`, `min_slope_angle`,
  `max_slope_angle`, `release_min_elevation`, `release_max_elevation`, `release_area_fraction`,
  `crown_line_method`, `crown_line_exit_distance`, `crown_elevation_band`, `min_particles_per_crown_cell`, `velocity_threshold`, `roughness_threshold`,
  `peak_flow_thickness_threshold`, `constitutive_model`, `shear_modulus`, `hardening_modulus`,
  `bulk_modulus`, `compaction_pressure`, `enable_curvature`, `enable_particle_interaction`,
  `enable_earth_pressure_coefficient`, `enable_entrainment`, `enable_center_of_mass`,
  `enable_particle_relaxation`, `center_of_mass_biggest_blob`.
  ⚠ Adding a key with a different name than an existing one, or renaming, breaks every JSON —
  `deny_unknown_fields` makes typos hard errors.
- **`SimSettings`** (line 70) — `#[repr(C)]` POD **uniform uploaded to every shader**; field
  order must match `struct SimSettings` in `shaders/utils.wgsl:52` exactly. Defaults
  (settings.rs:167): `max_steps 6000`, `sim_model TerrainFollowing(0)`, `friction_model
  Voellmy(1)`, `released_particles_per_cell 8`, `density 200`, `slab_thickness_factor 1.0`,
  `friction_coefficient 0.2`, `drag_coefficient 2000`, `n0 70`, `i0 0.29`, `mu0 0.38`, `mu2 0.65`,
  `grain_diameter 0.002`, `internal_friction_angle 40`, `basal_friction_angle 25`, `cfl 0.5`,
  `velocity_threshold 0.1`, `roughness_threshold 0.01`, `min_slope_angle 28`, `max_slope_angle 60`,
  `release_min_elevation 1500`, `release_max_elevation 8848`, `peak_flow_thickness_threshold 0.1`,
  `constitutive_model DruckerPrager(0)`, `shear_modulus 5e4`, `hardening_modulus 0`,
  `bulk_modulus 2e4`, `compaction_pressure 2e3`.

Enums: `FrictionModel { Coulomb=0, Voellmy=1, VoellmyMinShear=2, SamosAT=3, MuI=5 }`;
`ConstitutiveModel { DruckerPrager=0, MuI=1 }` (strings "dp", "mohr-coulomb", "mu(i)", …);
`SimModel` (see section 1); `CrownLineMethod { FlowRouting("flow-routing"/"d8"),
ParticleSimulation("particle-simulation"), GradientStep("gradient-step"/"gradient") }`.

Feature flags packed into `SimSettings.flags` via `SimFlags::new` (bit 0 curvature, 1
particle_interaction, 2 earth_pressure_coefficient, 3 entrainment, 4 center_of_mass_biggest_blob).

### 7.2 Config files present

| File | Purpose |
|---|---|
| `settings.json` | default for CLI/live_sim: curvilinear on `data/avaframe/avaWog.png` |
| `settings_outline.json` (untracked) | outline evaluation: `outlines_path: data/outline/polygon_10721.shp`, `outlines_padding: 4000` |
| `settings_vals_example.json`, `settings_vals_all_pra.json` | Vals GeoTIFF runs; ⚠ contain legacy key `slab_thickness` which `Settings` rejects (`deny_unknown_fields`) — stale |
| `crates/cli/settings.json`, `crates/compute_core/settings.json` | crate-local test configs with relative DEM paths |
| `pyproject.toml` | maturin: `manifest-path = "crates/python_bindings/Cargo.toml"`, `module-name = "avalanchers._avalanchers"`, `python-source = "python_module"`; pytest testpaths |
| `.qlty/qlty.toml` | linters: clippy, ruff, bandit, trufflehog, … |
| `.vscode/launch.json` | LLDB debug, Chrome launch, WebGPU inspector; python interpreter `.venv/Scripts/python.exe` |

### 7.3 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `AVAFRAME_HOME` | `avalanchers_ana1_tests.py` (default `C:/git/AvaFrame/avaframe`) | local AvaFrame checkout for analytical-test DEMs |
| `LIVE_SIM_SNAPSHOT` | `render_core/examples/live_sim.rs` | headless run → write PNG to this path |
| `LIVE_SIM_NO_PARTICLES` | live_sim.rs | suppress particles in snapshot |
| `SWISSALTI_TEST_TILES` | `data_processor/src/tile_manager.rs` (test) | local tile fixture dir |
| `GITHUB_ACTIONS` | simulation/wasm tests | skip GPU-heavy asserts on CI |
| `SKIP_FOR_TARPAULIN` | CI | skip coverage-incompatible tests |
| `RUST_LOG` (via `EnvFilter`) | `simulation::init_logging`, live_sim | tracing filter |
| CI-only: `WGPU_BACKEND`, `VK_ICD_FILENAMES` | workflows | software GPU backends |
| SFTP secrets in `sftp-deploy.yaml` | CI | **[REDACTED]** (`SFTP_USERNAME/PASSWORD/HOST/PORT/REMOTE_DIR`) |

### 7.4 Hardware / runtime requirements

- A WebGPU adapter (Vulkan/Metal/DX12/GL). CI installs Mesa `llvmpipe` (`VK_ICD_FILENAMES`) so
  tests run on software rasterizers. The code probes `SHADER_FLOAT32_ATOMIC` and
  `FLOAT32_FILTERABLE` features and falls back to quantized-integer atomics when absent.
- `max_particles` is derived from `max_storage_buffer_binding_size` / compute limits at startup.
- Browser: WebGPU requires a secure context → the frontend must be served over HTTPS
  (`frontend/dev_server.py`, port 443, self-signed cert auto-generated with openssl).
- Native Windows links with `rust-lld.exe` (see `.cargo/config.toml`).

## 8. File Catalog

Format: full template for behavior-critical files; compact entries (`PURPOSE / API / NOTES`) for
small or boilerplate files. "Dependents" lists only in-repo consumers that matter.

---

### 8.1 Root files

```
FILE: Cargo.toml
TYPE: TOML (Cargo workspace manifest)
PURPOSE: Defines the 7-crate workspace, shared dependency versions, and profiles.
RESPONSIBILITIES:
  - members: cli, data_processor, wasm_bindings, compute_core, python_bindings, render_core, simulation; resolver = "3"
  - [workspace.package]: version 0.2.0, license GPL-3.0-only, edition 2024
  - [workspace.dependencies]: wgpu 30, winit 0.30, pyo3 0.28 (auto-initialize, abi3-py37),
    wasm-bindgen 0.2, zarrs 0.23, tiff 0.11, shapefile 0.8, clap 4.6, egui 0.36, criterion 0.5
  - [profile.dev] debug = 1, split-debuginfo = "unpacked"
IMPORTANT DETAILS: Internal crates referenced as path deps (`simulation`, `compute_core`,
  `render_core`, `data_processor`). Adding a crate requires adding it here AND as a dependency of
  the consuming crate.
SOURCE: (see section 2 for the version table; file is the source of truth)
```

```
FILE: .cargo/config.toml
TYPE: TOML (Cargo config)
PURPOSE: Target-specific build flags.
IMPORTANT DETAILS: `[target.wasm32-unknown-unknown] rustflags = ['--cfg', 'getrandom_backend="wasm_js"']`
  (REQUIRED for rand/getrandom on wasm); `[target.x86_64-pc-windows-msvc] linker = "rust-lld.exe"`.
  No rust-toolchain.toml exists anywhere.
```

```
FILE: pyproject.toml
TYPE: TOML (Python packaging / maturin)
PURPOSE: Builds the PyPI package `avalanchers` from the Rust pyo3 crate.
KEYS: requires-python >=3.10; deps numpy>=2.0, zarr>=3.0, xarray; extras `viz`
  (matplotlib/pyvista/vtk), `jupyter`. [tool.maturin] manifest-path =
  "crates/python_bindings/Cargo.toml", module-name = "avalanchers._avalanchers",
  python-source = "python_module". pytest testpaths = python_module/tests.
```

```
FILE: README.md
TYPE: Markdown
PURPOSE: Project intro, install (`pip install avalanchers[viz]`), dev setup (Rust, wasm-pack,
  maturin), build commands, frontend run instructions, Zarr store layout, code guidelines, data
  source credits (AvaFrame EUPL-1.2, Vals/Zenodo, swisstopo, basemap.at).
IMPORTANT DETAILS: Canonical build commands:
  cargo build -p cli --release | maturin develop |
  wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --release
  Checks: cargo fmt; cargo clippy -- -D warnings;
  cargo test -p compute_core -p data_processor -p simulation -p cli -p render_core;
  ruff check ./python_module; pytest python_module; wasm-pack test --headless --chrome
```

```
FILE: CHANGELOG.md — 0.2.0 (2026-05-20): particle interactions w/ grid-based flow height +
  pressure force, criterion benchmark, SamosAT friction, ESRI ASCII support. 0.1.0 (2026-05-04):
  block model, Python bindings, WASM build + web frontend, CLI.
FILE: LICENSE — GPL-3.0. FILE: NOTICE — swisstopo/basemap.at/Hafner-Bühler(EnviDat) attribution.
FILE: .gitignore — ignores web_output/, dtm_cache.zarr, *.pem, *.bin, avalanchers.zarr/,
  outputs scratch notebooks, node_modules/, target/, prompts.md, questions.md, dd_dev.py, etc.
FILE: .gitattributes — marks *.ipynb/*.asc/*.ags/*.rp/*.aabb as linguist-vendored.
FILE: avalanchers.code-workspace — multi-root VS Code workspace (sibling repos avalanchers_evaluation,
  ISeeSnow, AvaFrame).
FILE: .vscode/launch.json — LLDB "Debug Rust", Chrome index.html?debug=vscode, WebGPU inspector attach.
FILE: .qlty/qlty.toml — qlty linter config (clippy, ruff, bandit, ripgrep, trufflehog, zizmor).
FILE: avalanchers_example.ipynb — notebook twin of avalanchers_example.py with plots (large, vendored).
```

```
FILE: verification/energy_line_test.py  (untracked)
TYPE: Python module (verification/ folder, snake_case)
PURPOSE: Energy line test analysis + plotting, ported from
  avaframe/ana1Tests/energyLineTest.py with NO avaframe imports (numpy +
  matplotlib only). Consumes in-memory avalanchers results.
API: get_runout_angle(profile), get_alpha_profile_intersection(cfg, profile,
  mu, csz), get_energy_info(profile, g, mu, …), generate_energy_plot(
  profile, dem, fields, cfg, g, mu, sim_name, out_dir, show) → the 2×2
  AvaFrame figure (pfv bird's-eye on hillshade / energy-vs-alpha profile /
  zoomed runout panel with error text box). Conventions: DEM row 0 = south
  (imshow origin="lower"), dem dict = {rasterData, header{cellsize, ncols,
  nrows, xmin, ymin, xmax, ymax}}, fields = {"pfv": (nrows, ncols)}.
NOTES: pfv colormap = AvaFrame colorsS/speedColorLevels (batlow reversed,
  discrete); the pfv image needs cmap.set_bad(alpha=0) so the hillshade
  stays visible under masked cells.
```

```
FILE: verification/run_energy_line_test.py  (untracked)
TYPE: Python script (verification/ folder)
PURPOSE: Main entry for the energy line test WITHOUT AvaFrame output files:
  creates a PySimulation from the editable SETTINGS dict (default
  data/avaframe/avaParabola + release texture, curvilinear, Coulomb
  mu=0.4), calls prepare(), then loops run_n_steps(SAVE_STEPS=25),
  sampling the mass-averaged center of mass (x, y, z via bilinear DEM
  sample, u2 = mass-avg |v|²) from particles_position_xy /
  particles_velocity / particles_mass each batch; stops on
  SIM_STOPPED/Finished/--t-end. Then builds the profile dict
  (s = cumulative COM path length), the dem dict and sim.peak_velocity
  and calls energy_line_test.generate_energy_plot.
CLI: --dem --release --mu --sim-model --friction-model --max-steps
  --save-steps --t-end --out-dir --gpu --show.
NOTES: use friction_model "coulomb" so the analytic α line (slope
  friction_coefficient) is exact; reference run avaParabola μ=0.4 stops at
  844 steps with runout-angle error 0.08°, runout-s error −7.5 m,
  velocity-height RMSE 1.6 m. Plot → verification/outputs/energy_line_test/.
```

```
FILE: verification/dam_break_test.py  (untracked)
TYPE: Python module (verification/ folder, snake_case)
PURPOSE: Dam break test analysis + plots, ported from
  avaframe/ana1Tests/damBreak.py (Faccanoni & Mangeney 2012, Test 2, case
  1.2 dry bed), analysisTools.py (L2/LMax norms) and the dam-break plots of
  out3Plot/outAna1Plots.py; no avaframe imports.
API: fields_from_grid(sim, shape, cell_size, density, jac, frame) — decodes
  the sim's grid_mass / grid_momentum GETTERS into thickness + flow/cross
  velocity (MASS_FACTOR 1e1 / MOMENTUM_FACTOR 1e2 must match utils.wgsl;
  nodes < 1e-2 mass are empty like the shader); jacobian, flow_frame (plane
  fit → phi + descent dir), flow_coordinates, check_grid_mass_consistency,
  norm_l2_scal, dam_break_solution(phi, delta, h_l, times, s_rel, g,
  x_back_rel) → (h, u, x_mid), dam_break_rear_edge (physical rear edge of
  the material); plots: plot_dam_ana_results, plot_comparison_dam,
  add_error_time, plot_error_time, plot_dam_break_summary (2×6: profiles /
  FT map on DEM slope with error rectangle / error curves / params),
  assemble_gif (stitches the per-time-step comparison figures into a
  looping GIF).
NOTES: analytic coords are along-slope relative to the dam front; DEM row 0
  = south; use "Greys_r" (NOT "Greys", which runs white→black) for the DEM
  slope background; FT colormap = AvaFrame colorsT/thicknessColorLevels
  (lajolla, discrete) with set_bad(alpha=0).
```

```
FILE: verification/run_dam_break_test.py  (untracked)
TYPE: Python script (verification/ folder)
PURPOSE: Main entry for the dam break test WITHOUT AvaFrame output files:
  creates a PySimulation from SETTINGS (default data/avaframe/avaDamBreak +
  release texture, curvilinear, Coulomb), prepare(), then measures the dam
  geometry (h0, front/back, center row) from sim.release_areas and the DEM
  slope from a plane fit; loops run_n_steps(BATCH_STEPS=20) sampling the
  grid fields every save_interval (1 s) and comparing to the analytic
  solution over the AvaFrame domain (x_mid→xEnd, |cross-flow|≤50 m);
  stops on SIM_STOPPED/Finished/t_end. Saves the AvaFrame-style plots to
  verification/outputs/dam_break_test/pics/.
CLI: --t-end --save-interval --delta --phi --max-steps --out-dir --gpu
  --plot-sequence --show.
NOTES: delta default 12 deg (not AvaFrame's 21 — near-critical slopes never
  slide under avalanchers' friction solver, see avalanchers_ana1_tests.py);
  reference run: 1501 steps / 20 s, relative h L2 0.33 at t_save=15 s; error
  domain legitimately empties once the analytic mid passes x_end (zeros at
  t≥19 s). u_num = speed (ini projectVelocity=False).
  Summary bird's-eye map shows the PEAK flow thickness up to the highlighted
  time (t_save, default 15 s): samples["peak_h"] = running per-cell maximum
  of the decoded grid fields, recorded per save interval (1 s resolution) —
  accessible at every sampled time WITHOUT the Finished state (the sim is
  no longer run past t_end). Continuous 0–1.0 m colorscale (FT_COLORS
  gradient, LinearSegmentedColormap + Normalize(0, 1), set_bad alpha=0).
  All plots use the fixed x window DAM_BREAK_CFG["x_lim"] = [-200, 400] m;
  map y window is fitted to the flowed cells (equal aspect via
  adjustable="box").
  --plot-sequence = the optional result: comparison cross-cut figure (FT /
  FV / hv, initial vs numerical vs analytical) for EVERY saved time step +
  assembled animated GIF (<sim>_compare.gif, 500 ms/frame, frames
  downscaled to 1500 px). Animation uses FIXED axis limits (frame["ylim"]
  from numeric+analytic maxima over all times) and CLIPS the analytic
  curves at dam_break_rear_edge — upstream of it the case-1.2 solution
  formally extends h=h_l over the vacated dam (AvaFrame draws this
  artifact); frame times in file names are zero-padded for GIF ordering.
```

```
FILE: verification/rotation_test.py  (untracked)
TYPE: Python module (verification/ folder, snake_case)
PURPOSE: Rotation test analysis, ported from avaframe/ana1Tests/
  rotationTest.py (+ runScripts/runRotationTest.py) without avaframe
  imports: run the same avalanche on rotated DEM + release rasters and
  compare everything in the reference (first-angle) frame.
API: rotate_raster(raster, angle, cval) — scipy ndimage.rotate,
  reshape=False, +angle = CCW in map coords (row 0 = south) about the
  domain center; clean_rotated_release (re-binarize at 50 %); resample_path
  / make_cross_sections (perpendicular sections along the reference COM
  path — lightweight AIMEC path transform); sample_field (bilinear);
  cross_section_stats (max/mean per section); plots: plot_runout_comparison
  (sEnd/zEnd/angle vs angle + reference hlines + diff box),
  plot_sl_comparison (2×2 max/mean of pft/pfv along path per rotation),
  plot_areas_comparison (TP/FP/FN maps vs reference, Dice in titles;
  BoundaryNorm boundaries MUST be [-0.5, 0.5, 1.5, 2.5, 3.5] so value v →
  color v; shared fig-level legend BELOW the panels); print_results_table.
```

```
FILE: verification/run_rotation_test.py  (untracked)
TYPE: Python script (verification/ folder)
PURPOSE: Main entry for the rotation test WITHOUT output files: loads the
  base DEM/release once through a helper sim (default
  data/avaframe/avaTripleBowl + release texture), then per rotation angle
  (ROTATIONS = [0, 30, 60]; first = reference) rotates DEM (+angle, NaN
  fill) and release (re-binarized) in memory, creates a fresh PySimulation
  (settings WITHOUT dem_path → state DemMissing is fine) and feeds the
  rasters via set_dem_with_bounds / set_release_areas; runs with the
  energy-line COM sampling (reuses run_energy_line_test helpers) and does
  the per-rotation energy line analysis + figure; runs to Finished
  (finish_simulation) for the peak field getters; rotates peak fields back
  (-angle) and builds the comparison plots in outputs/rotation_test/pics/.
CLI: --angles --dem --release --mu --max-steps --save-steps --out-dir
  --gpu --show.
NOTES: reference run avaTripleBowl coulomb µ=0.3 (SETTINGS), ~1430 steps /
  21 s per rotation, 13 s wall for 3 rotations; results nearly
  rotation-invariant: ds ≤ 4.6 m over sEnd 456 m, |dAngle| ≤ 0.47°,
  Dice 0.88. sEnd is PATH length (serpentine bowls) — corridor x extent is
  only [-400, 155]; the areas map window is auto-fitted to the reference
  corridor (+25 m buffer), NOT a hardcoded x_lim.
```

```
FILE: verification/similarity_solution_test.py  (untracked)
TYPE: Python module (verification/ folder, snake_case)
PURPOSE: Similarity solution test analysis, ported from avaframe/ana1Tests/
  simiSolTest.py (Hutter, Siegel & Savage 1993), analysisTools.py and the
  simiSol plots of out3Plot/outAna1Plots.py; no avaframe imports.
API: similarity_solution(zeta, delta, phi, flag_earth, lx, ly, rel_th, g,
  t_end) — ODE solve (scipy dopri5, RK4 fallback); similarity_fields(sol,
  t, x1, y1, …) → {h, u, v, x_center}; fields_from_grid (grid_mass/
  grid_momentum getters, utils.wgsl quantization factors); norm_l2_scal /
  norm_l2_vect; get_plot_limits (widths in METERS — AvaFrame
  extentOption=True; a naive index-based width was 5× too small and
  cropped the contour window); blur3x3 (NaN-aware 3×3 mean, used ONLY for
  contour panels — the raw quantized grid field is per-cell noisy unlike
  com1DFA's SPH fields); plots: _plot_variable_panel (twin-axis FT +
  |h·u|/h·ux/h·uy, in-plane only — no h·vz), add_contour_2plot (solid sim /
  dashed analytic nested contours, window centered on the ANALYTIC center
  x_center at the plotted time), add_error_time, plot_error_time,
  plot_profile_comparison, plot_contour_comparison, plot_simi_sol_summary
  (2×6: profiles / contours / error curves / params).
```

```
FILE: verification/run_similarity_solution_test.py  (untracked)
TYPE: Python script (verification/ folder)
PURPOSE: Main entry for the similarity solution test WITHOUT output files:
  reads data/avaframe/avaSimilaritySol.png (35° plane, 5 m cells, domain
  x [-202, 1802]), builds the parabolic release heap in memory
  (relTh·(1 − x1²/Lx² − y²/Ly²), centered at map (0,0), slivers ≤ 0.02 m
  zeroed) and hands it to set_release_areas; runs with 1 s grid-field
  sampling (t_end 20 s, cfl 0.05, coulomb δ=25° = 10° excess, drag 1e12
  like avalanchers_ana1_tests.py), comparing relative L2/LMax errors per
  save time; saves summary + error-time figures and optional per-time
  profile/contour figures to outputs/similarity_solution_test/pics/.
CLI: --t-end --save-interval --delta --max-steps --out-dir --gpu
  --plot-sequence --show.
NOTES: reference run: relative h L2 0.068 at t=5 s (matches
  avalanchers_ana1_tests.py simisol); sim COM lags analytic ~1-2 m.
  --plot-sequence reproduces AvaFrame's plotSequence (saveSimiSolProfile +
  makeContourSimiPlot per save time) and stitches both sequences into
  <sim>_contours.gif / <sim>_profiles.gif.
  ALL four run scripts embed the sim_model in figure file names
  (<case>_<sim_model>_<friction>_…) so different-model runs don't collide.
```

```
FILE: verification/run_all_tests.py  (untracked)
TYPE: Python script (verification/ folder, snake_case)
PURPOSE: Runs all four verification tests IN-PROCESS with a switchable
  sim model and writes one markdown report. Per test: patches the module's
  SETTINGS/cfg dicts (--sim-model/--friction-model/--mu), calls main()
  with patched sys.argv and --out-dir <run>/<test>, captures stdout
  (ANSI/tracing lines filtered), tolerates failures per test.
  Report (outputs/all_tests/<model>_<ts>/report.md): summary table
  (status + headline metric per test), per-test sections with summary
  figures + GIFs embedded (relative links), rotation result table from
  the return value, console tables in <details> blocks.
  Headlines: energy line runout/rmse; dam break + similarity relative
  h L2 at t≈5 s AND last sample; rotation max |ds| + min Dice.
CLI: --sim-model --friction-model --mu --tests --out-dir --quick
  --no-sequences.
NOTES: analytic references are exact ONLY for curvilinear + Coulomb —
  other models are model-comparison runs. A near-critical bed friction
  angle (delta close to the slope angle) stalls the dam break: mpmdac
  stops at t=0 with DAM_BREAK_CFG["delta"]=21 (curvilinear still creeps
  through with degraded errors 0.34@20 s vs 0.14@5 s at delta=12);
  run_dam_break_test raises a descriptive RuntimeError in that case so
  the report shows FAILED + explanation. Reference curvilinear run:
  energy 1212 m/21.88°, dam break 0.136@5 s, rotation |ds|≤4.6 m
  Dice≥0.878, similarity 0.068@5 s; ~61 s wall for all 4 with sequences.
```

```
FILE: settings.json
TYPE: JSON (Settings schema)
PURPOSE: Default CLI/live_sim run: dem_path data/avaframe/avaWog.png,
  release_areas_path data/avaframe/avaWogreleaseTexture.png, sim_model "curvilinear",
  friction_model "voellmy", max_steps 6000, released_particles_per_cell 4,
  slab_thickness_factor 1.0, friction_coefficient 0.2, drag_coefficient 2000.0,
  velocity_threshold 0.1, enable_center_of_mass false.
```

```
FILE: settings_outline.json (untracked) — outlines_path data/outline/polygon_10721.shp,
  outlines_padding 4000, curvilinear/voellmy. Used for the release-estimation/evaluation path.
FILE: settings_vals_example.json / settings_vals_all_pra.json — Vals GeoTIFF configs.
  UNCERTAIN/STALE: use key `slab_thickness` which current Settings (deny_unknown_fields) rejects;
  must be renamed to `slab_thickness_factor` before use.
```

---

### 8.2 crates/compute_core — GPU compute engine

```
FILE: crates/compute_core/Cargo.toml
TYPE: TOML
NOTES: deps wgpu, anyhow, bytemuck(derive), tempfile, serde(+json), rand, regex,
  futures-intrusive, tracing, test-log, futures, web-time, bitflags; dev-dep pollster.
  No internal deps — foundation crate.
```

```
FILE: crates/compute_core/src/lib.rs  (3242 lines)
TYPE: Rust source
PURPOSE: Central GPU orchestrator + core POD data types shared with the shaders + a large
  #[cfg(test)] GPU test module (lines 2046-3242).

RESPONSIBILITIES:
  - POD types mirrored in WGSL: SimInfo (lines 187-198: timestep, dt, elapsed_time,
    number_particles, elevation_threshold, max_velocity, max_flow_thickness, flags),
    TimestepDataAoS (96-byte, lines 221-257) / TimestepData SoA + from_aos (259-273),
    TextureRgba<T>, GpuCache (137-158: CPU memoization of every GPU readback; methods
    reset_simulation_result / reset_all), SimInfoFlags bitflags (34-45: OUT_OF_BOUNDS,
    CFL_EXCEEDED, IS_NAN, PARTICLE_OUT_OF_DEM_DATA, NO_NEW_CELLS(1<<29),
    ALL_PARTICLES_STOPPED(1<<30), SIM_STOPPED(1<<31)), ParticleState decode (86-118) with
    PARTICLE_FLYING/OUT_OF_BOUNDS/IS_NAN/OUT_OF_DEM_DATA/STOPPED bits (79-83).
  - ComputeOrchestrator (470-495): owns wgpu Instance/Adapter/Device/Queue, GpuResources,
    shader_configs HashMap, dispatch sizes, prepared_max_steps/completed_steps/prepared_model,
    SimulationPipelines/BindGroups, has_float32_atomic/filterable, enable_center_of_mass.
  - new_with_gpu(target_gpu: Option<String>) (501-764): adapter selection by name substring or
    PCI device id ("|" splits a display string), fallback HighPerformance→LowPerformance→
    Software; requests TIMESTAMP_QUERY + SHADER_FLOAT32_ATOMIC + FLOAT32_FILTERABLE when
    supported; derives max_particles and max_compute_invocations_per_workgroup.
  - list_devices() (336): enumerate adapters, group by name.
  - run_shader(name, x, y, z) (793): ad-hoc single-pipeline dispatch (bind group from config).
  - create_buffers_and_texture_descriptions(&SimSettings) (842); run_analyze_terrain (861;
    model 0/2 → AnalyzeTerrain, model 1 → AnalyzeTerrainCurvilinear);
    run_compute_roughness (946); run_compute_release_areas(settings, roi) (965);
    run_compute_center_of_mass (1010); dispatch_chamfer_distance (1065);
    evaluate_gpu(settings) -> MassMovementEvaluation (1170).
  - run_initialize_particles(settings, number_release_particles, relax) (1230): allocates
    model-specific particle buffers (see section 4.5), dispatches InitializeParticles,
    optionally relax_particles (1405; 30 iterations, RELAX_MOVE_FACTOR 0.3,
    RELAX_MAX_STEP_FRACTION 0.5, constants at 440-446).
  - prepare_simulation (1478, private): validates steps/particles, allocates CenterOfMass
    (max_steps+1 slots) and TimestepData (3 slots × max_steps) buffers, seeds SimInfo.dt =
    sqrt(2·cfl·cell_size/(9.81·0.866)), selects the per-model shader triple
    (1536-1559: 0 → P2G/GridPhysics/ComputeParticles, 1 → P2G/GridPhysicsCurvilinear/G2P,
    2 → P2GMPMDAC/GridPhysicsMPMDAC/G2PMPMDAC, 3.. → error), creates pipelines + bind groups.
  - step_simulation(steps, SimModel) (1641): enforces prepared model match + remaining-step
    budget; model 2 copies GridMass→GridMassPrevious before the pass (1673-1706); dispatch
    order reset_grid → p2g → grid_physics → particle_update → update_sim_info →
    [center_of_mass seed/propagate/center] (1713-1768); single queue.submit.
  - Public per-model wrappers: prepare_compute_particles/step_terrain_following (1784),
    prepare_mpm/step_curvilinear/run_mpm (1887-1937), prepare_mpmdac/step_mpmdac/run_mpmdac
    (1939-1989); run_sim dispatches by sim_model (1823-1885) and warns when max_steps was
    reached without ALL_PARTICLES_STOPPED.
  - get_sim_info (1776); read_buffer/write_buffer/read_texture/read_texture_single_channel/
    write_texture/add_buffer/add_buffer_with_data (1991-2043).

DATA FLOW: CPU (SimSettings bytes, DEM texture, release raster) → GPU buffers → compute passes
  → SimInfo/AtomicValues/TimestepData/grids read back via staged map.
IMPORTANT DETAILS:
  - completed_steps is decremented by 1 at step_simulation entry (1653) — an off-by-one
    correction relative to the seeding step; do not "fix" without reading tests.
  - GPU test module requires an adapter; CI sets software Vulkan.

FILE: crates/compute_core/src/settings.rs  (1237 lines)
TYPE: Rust source
PURPOSE: All configuration types; the Rust side of the settings contract (see section 7.1).
CONTENTS: SimFlags (14-67), SimSettings POD (68-109) + manual Hash over to_bits (111-152) +
  defaults (167-210) + set_dem/from_settings/as_bytes/to_json (212-355); FrictionModel (358-440),
  ConstitutiveModel (442-516), SimModel (518-595), CrownLineMethod (597-642) — each with
  from_int/as_int/FromStr/Display/serde untagged int-or-string; Settings (644-716,
  deny_unknown_fields) + loads/dumps/from_json/get_sim_settings (718-753). Test module 755-1237.
DEPENDENTS: everything (uniform written into BufferName::SimSettings; WGSL mirror in
  shaders/utils.wgsl:52-87 — field order must match).

FILE: crates/compute_core/src/shaders.rs  (2876 lines)
TYPE: Rust source
PURPOSE: Shader name registry, source preprocessor, pipeline/bind-group construction.
CONTENTS:
  - define_shaders! macro (16-72) → ShaderName enum with 38 variants (74-113). Variant list:
    AnalyzeTerrainCurvilinear, AnalyzeTerrain, ResetGrid, ComputeRoughness, ComputeReleaseAreas,
    ComputeCenterOfMass, CenterOfMassSeed, CenterOfMassPropagate, ChamferPrepare, ChamferFlood,
    ChamferReduce, ComputeBeelineDistance, EvaluateMassMovement, EvaluateMassMovementPoints,
    InitializeParticles, RelaxParticles, RelaxClearGrid, RelaxBuildGrid, RelaxSampleElevation,
    ComputeParticles, G2P,
    G2PMPMDAC("g2p_mpmdac"), P2G, P2GMPMDAC("p2g_mpmdac"), GridPhysicsCurvilinear,
    GridPhysicsMPMDAC("grid_physics_mpmdac"), GridPhysics, Friction, Constitutive, Utils, Random,
    UpdateSimInfo, TestTransfer, TransferG2P, TransferG2PAffine, TransferP2G, TestTransforms,
    TestSampling, TestUtils.
  - read_source (42-69): disk in native debug (hot reload), include_str! otherwise.
  - load_shader_source_string (127-183): strip BEGIN/END → recursive `// import x.wgsl;`
    inlining → `// atomic_float` line switching → Box::leak. load_shader_source (186).
  - ComputeShaderConfig (208-215) + new_with_constants (227; entry point = shader name) +
    create_bind_group (287) + create_resources (307).
  - create_shader_configs (357-~2700): per-shader binding declarations (index 0 = SimSettings
    uniform) and pipeline constants (e.g. WG_SIZE_1D override). generate_shader_report (2708).
DEPENDENTS: compute_core/lib.rs (only consumer of configs).

FILE: crates/compute_core/src/buffers.rs  (1174 lines)
TYPE: Rust source
PURPOSE: GPU buffer/texture registry and POD mirrors of shader-side structs.
CONTENTS: AtomicValues (21), CenterOfMassResult (35), ChamferParams (56), RelaxParams (64),
  EvaluationResult (80-103); BufferName enum (106-181, ~45 variants — see section 4.5; to_str/
  FromStr give the snake_case names used in shader binding lists); TextureName {Dem,
  TerrainGeometry, Curvature} (293-299); GpuResources (334) with add_buffer[_with_data],
  prepare_buffer_contents (f32↔encoded-u32 for non-float-atomic grids), get/write/read helpers;
  create_buffers_and_texture_descriptions (807); DEBUG_BUFFER_SIZE = 400 bytes (806).

FILE: crates/compute_core/src/dem.rs  (683 lines)
TYPE: Rust source
PURPOSE: DEM container + GeoTIFF parsing.
CONTENTS: Bounds (41), Dem (58-70: width, height, bounds, data1d + data 2D, x/y axes, cell_size,
  map_factor, minimum_elevation, source, projection; Hash by fields; interpolate_elevation,
  get_elevation_extrema(&mask), get_elevation_extrema_distance_and_drop, mask_above_elevation),
  GeoMetadata/GeoTiff/TiffData parsing helpers. Dem::default() = 1×1 zero DEM (used by
  Settings::get_sim_settings and tests).

FILE: crates/compute_core/src/evaluation.rs  (610 lines)
TYPE: Rust source
PURPOSE: CPU evaluation metrics + GPU result decoders.
CONTENTS: MassMovementEvaluation (3: intersection/undershoot/overshoot/iou + runout fields +
  ChamferDistance; with_runout/with_chamfer builders), evaluate_mass_movement_area(reference,
  simulated) (82: CPU IoU), ChamferDistance (118) + chamfer_from_sums/evaluation_from_counts
  (normalize GPU chamfer sums by grid diagonal), evaluate_distance_weighted_mass_movement_runout (206).

FILE: crates/compute_core/src/post_processing.rs  (418 lines)
TYPE: Rust source
PURPOSE: CPU connected-component/mask utilities.
CONTENTS: detect_blobs_1d(grid, diagonal_connectivity) (35), get_biggest_blob, threshold_to_bool,
  mask_with_blob, mask_threshold_and_biggest_blob (152; used by Simulation::post_process),
  mask_in_place. FlattenedGrid/Blob types.

FILE: crates/compute_core/src/utils.rs  (970 lines)
TYPE: Rust source
PURPOSE: Small helpers: Point, create_random_rgba_data (PCG), traits To2D/To1D/MaxValue/
  MinValue/MeanValue/Hist/HistFloat, linspace, bilinear_interpolate, split_channels,
  Checkpoint/Timer/timer_checkpoint/timer_new/timer_get_summary (317-380), flip_rows_flat_vec,
  percentile(_with_threshold).

FILE: crates/compute_core/settings.json — test config, dem_path "../avaframe/avaParabola.png".
```

---

### 8.3 crates/compute_core/src/shaders — WGSL compute shaders

Conventions for every file: entry point name = file stem; `@group(0) @binding(0)` is always
`var<uniform> sim_settings: SimSettings`; files contain a hand-written top section plus stale
inlined `// BEGIN/END` copies (created by debug-build write-back) — **edit only the top section**.
Line numbers below refer to the top section.

#### Shared modules (imported, no entry point)

```
FILE: crates/compute_core/src/shaders/utils.wgsl  (255 lines)
PURPOSE: Single source of truth for constants, structs, index math, B-spline weights.
IMPORTANT CONTENTS (exact):
  const WG_SIZE_2D: u32 = 16u;  const g: f32 = 9.81;
  const MAX_VELOCITY_FACTOR: f32 = 1e7;   // u32 limit 430 m/s
  const MASS_FACTOR: f32 = 1e1;           // u32 limit 4.3 t thickness
  const H_FACTOR: f32 = 1e6;
  const MOMENTUM_FACTOR: f32 = 1e2;       // i32 budget ~2.4e8 per node
  const FORCE_FACTOR: f32 = 1e-3;         // i32 limit 2.1e6 N per node
  (+ INV_* inverses)
  struct SimInfo { timestep: u32, dt: f32, elapsed_time: f32, number_particles: u32,
      elevation_threshold: f32, max_velocity: f32, max_flow_thickness: f32, flags: u32 }
  struct SimSettings { num_steps: u32, model_type: u32, friction_model: u32,
      released_particles_per_cell: u32, grid_shape: vec2u, world_size: vec2f,
      snow_density: f32, slab_thickness: f32, friction_coefficient: f32, drag_coefficient: f32,
      n0, i0, mu0, mu2, grain_diameter, internal_friction_angle, basal_friction_angle, cfl,
      cell_size, min_slope_angle, max_slope_angle, min_elevation, velocity_threshold,
      roughness_threshold: f32, flags: u32, release_max_elevation,
      peak_flow_thickness_threshold: f32, constitutive_model: u32, shear_modulus: f32,
      hardening_modulus: f32, bulk_modulus: f32, compaction_pressure: f32 }
      // WGSL names differ from Rust (num_steps/model_type/snow_density/slab_thickness/
      // min_elevation) but ORDER matches the Rust #[repr(C)] struct exactly.
  struct AtomicValues { peak_velocity, peak_flow_thickness, expected_max_velocity,
      travel_length, release_volume, number_release_cells, number_release_particles,
      stopped_particles: atomic<u32> }   // floats bit-cast
  struct G2PUpdate { velocity: vec2f, affine_matrix: mat2x2<f32> }
  SimInfo flags: SIM_INFO_OUT_OF_BOUNDS=1<<0, CFL_EXCEEDED=1<<1, IS_NAN=1<<2,
      PARTICLE_OUT_OF_DEM_DATA=1<<3, SIM_INFO_STOPPED=1<<31, ALL_PARTICLES_STOPPED=1<<30,
      NO_NEW_CELLS=1<<29
  Particle flags: PARTICLE_FLYING=1<<27, PARTICLE_OUT_OF_BOUNDS=1<<28, PARTICLE_IS_NAN=1<<29,
      PARTICLE_OUT_OF_DEM_DATA=1<<30, PARTICLE_STOPPED=1<<31
  helpers: is_nan/is_inf (bitcast), cell_to_uv, position_to_cell, uv_to_idx (row-major
      y*grid_shape.x + x), quadratic_weight / quadratic_weight_gradient (quadratic B-spline,
      gradient /cell_size), get_base_node = floor(grid_pos - 0.5), determinant_2x2,
      identity_2x2, compute_centroid.
DEPENDENTS: imported (directly or transitively) by every compute shader.
CRITICAL: MASS_FACTOR/MOMENTUM_FACTOR are duplicated in avalanchers_ana1_tests.py (lines
  111-114) with a runtime consistency cross-check; changing them requires updating both.

FILE: crates/compute_core/src/shaders/random.wgsl (28) — pcg_hash/next_rand/rand1..rand4.
FILE: crates/compute_core/src/shaders/transfer_p2g.wgsl (40) — quantize_momentum_component +
  transfer_p2g (3×3 quadratic B-spline scatter: atomicAdd mass u32, momentum i32).
FILE: crates/compute_core/src/shaders/transfer_g2p.wgsl (18) — PIC velocity gather.
FILE: crates/compute_core/src/shaders/transfer_g2p_affine.wgsl (23) — APIC gather returning
  G2PUpdate { velocity, affine_matrix } (affine += 4·w·vel·distanceᵀ).
FILE: crates/compute_core/src/shaders/friction.wgsl (53) — basal_friction_acceleration(
  g_perp, proposed_speed, h) -> deceleration m/s². friction_model ids: 0 coulomb, 1 voellmy
  (+turbulent ρV²g/ξ), 2 voellmy minshear (+const 70 Pa), 3 samosAT (rs0=0.222),
  4 voellmy-with-cohesion (commented stub), 5 mu(I) (I = 2.5·d·v/(h·√(g⊥h)), h clamped ≥ 1e-3, I floored at 1e-9). Returns 0 when
  proposed_speed < velocity_threshold or model ≥ 6. Imported by grid_physics_mpmdac.wgsl.
FILE: crates/compute_core/src/shaders/constitutive.wgsl (34) — strain_rate_tensor,
  mat2_deviatoric, frobenius_norm, equivalent_shear_stress (‖τ‖_F/√2),
  equivalent_shear_rate (√2‖ε̇_dev‖_F), mu_inertial(I = γ̇·d/√(P/ρ)). Imported by p2g_mpmdac.wgsl.
```

#### Terrain preprocessing (run once per DEM)

```
FILE: crates/compute_core/src/shaders/analyze_terrain.wgsl (355; entry analyze_terrain, 16×16×1)
PURPOSE: DEM gradients/Hessian → terrain normal, curvatures, slope angle/aspect (models 0+2).
BINDINGS: 1 dem_texture (texture_2d<f32>), 2 terrain_geometry_texture (storage rgba32float,
  writes vec4(normal.xyz, profile_curvature)), 3 curvature_texture (writes vec4(dxx,dxy,dyy,
  plan_curvature)), 4 slope_angle_buffer (degrees(acos(n.z))), 5 slope_aspect_buffer
  (clockwise-from-north), 6 debug.
CALLERS: ComputeOrchestrator::run_analyze_terrain for sim_model 0 and 2.

FILE: crates/compute_core/src/shaders/analyze_terrain_curvilinear.wgsl (376; 16×16×1)
PURPOSE: Curvilinear metric terms for model 1. TerrainGeometry = (l_x, l_y, J, g_y) with
  l_x=√(1+Bx²), l_y=√(1+By²), J=√(1+Bx²+By²), gravity g_x=-g·Bx/l_x, g_y=-g·By/l_y;
  Curvature = (K_xx, K_yy, K_xy, g_x) Hessians. Same bindings as analyze_terrain.

FILE: crates/compute_core/src/shaders/compute_roughness.wgsl (306; 16×16×1)
PURPOSE: Roughness per doi:10.5194/nhess-16-2211-2016 — 1-|Σr|/9 from 3×3 normals rebuilt from
  metric terms; border cells forced to 1.0. Bindings: 1 terrain_metrics, 2 roughness_buffer.
CALLERS: run_compute_roughness (release-area GPU path).

FILE: crates/compute_core/src/shaders/compute_release_areas.wgsl (307; 16×16×1)
PURPOSE: A cell releases iff ROI bit set AND min_slope_angle ≤ slope ≤ max_slope_angle AND
  elevation ≥ min_elevation (the WGSL name for release_min_elevation) AND roughness ≤
  roughness_threshold; writes release_areas[idx] = slab_thickness and counts
  number_release_cells into AtomicValues. Bindings: 1 dem, 2 slope_angles, 3 roughness,
  4 release_areas, 5 debug, 6 atomic_values, 7 roi_bit_packed.
```

#### Particle initialization

```
FILE: crates/compute_core/src/shaders/initialize_particles.wgsl (337; 16×16×1)
PURPOSE: Per release cell: cell_volume = thickness·cell²/cos(slope), cell_mass =
  volume·snow_density; atomically appends released_particles_per_cell particles (PCG jitter,
  seed 42), equal mass split, records start elevation; accumulates release_volume and
  number_release_particles in AtomicValues. Cell i is seeded over [i, i+1)·cell_size (the
  floor(pos/cell_size) convention of all kernels), 1e-3 cell margin against f32 rounding.
  Release cell = thickness > RELEASE_THICKNESS_THRESHOLD (0.01, compute_core lib.rs); the
  same constant is used by relax, evaluation and the CPU release-cell counts.
BINDINGS: 1 sim_info, 2 dem_texture, 3 slope_angle_buffer, 4 release_areas, 5 tex_sampler,
  6 position, 7 mass, 8 start_elevation, 9 atomic_values, 10 debug.

FILE: crates/compute_core/src/shaders/relax_particles.wgsl (126; untracked/new; 4 entries)
PURPOSE: Soft-sphere relaxation of freshly initialized particles to hexagonal packing spacing
  h = cell_size·sqrt(2/(√3·N)) for N particles/cell. Entries: relax_clear_grid (16×16×1, sets
  grid_head = LINKED_LIST_END 0xFFFFFFFF), relax_build_grid (WG_SIZE_1D override, lock-free
  linked-list insert particle_next[i] = atomicExchange(&grid_head[cell], i)), relax_particles
  (WG_SIZE_1D, 3×3-cell neighbor walk; displacement move_factor·overlap clamped to
  max_step_fraction·h; a step leaving the release cells falls back to its larger, then smaller
  axis component, so edge particles slide along the edge; dropped if neither stays inside),
  relax_sample_elevation (run once after the loop; re-samples start_elevation from the DEM
  at the relaxed position, compute_particles reads it as z at step 0).
BINDINGS (relax_particles): 1 relax_params (RelaxParams {num_particles, move_factor,
  max_step_fraction, pad}), 2 particles_position, 3 position snapshot, 4 particles_velocity,
  5 release_areas, 6 grid_head, 7 particle_next, 8 dem_texture, 9 tex_sampler,
  10 start_elevation (ParticlesElevation).
FILES: relax_clear_grid.wgsl (652) and relax_build_grid.wgsl (392) are alias modules —
  one comment line + `// import relax_particles.wgsl;` at top, rest is stale inlined content.
RUST SIDE: constants RELAXATION_ITERATIONS=30, RELAX_MOVE_FACTOR=0.3, RELAX_MAX_STEP_FRACTION=0.5
  (compute_core/lib.rs:440-446); loop in relax_particles() (lib.rs:1405).
```

#### Per-step pipeline (model pipelines; see section 4.2 for dispatch order)

```
FILE: crates/compute_core/src/shaders/reset_grid.wgsl (279; entry reset_grid, 16×16×1)
PURPOSE: Zero GridMass (u32), GridMomentum + GridForces (i32); invocation (0,0) resets
  AtomicValues.peak_velocity/peak_flow_thickness. Comment: bit pattern 0 identical for
  u32/i32/f32. Bindings: 1 grid_mass_atomic, 2 atomic_values, 3 grid_momentum_atomic,
  4 grid_forces_atomic.

FILE: crates/compute_core/src/shaders/p2g.wgsl (333; entry p2g, WG_SIZE_1D override)
PURPOSE: Model 0/1 P2G. Skips when model_type==0 && !particle_interaction (line 20) and for
  particles within 3 cells of the border. Scatter via transfer_p2g (PIC momentum only;
  affine path commented out). Bindings: 1 sim_info, 2 particles_position, 3 particles_velocity,
  4 particles_mass, 5 grid_mass_atomic (atomic<u32>), 6 grid_momentum_atomic (atomic<i32>).

FILE: crates/compute_core/src/shaders/p2g_mpmdac.wgsl (517; entry p2g_mpmdac, WG_SIZE_1D)
PURPOSE: Model 2 P2G: mass, APIC momentum, internal forces from per-particle stress;
  ALSO updates the per-particle constitutive state (this is where the plastic law lives).
ALGORITHM (transfer_p2g_mpmdac, lines 48-219):
  - particle depth h_p gathered from GridMassPrevious with the same B-spline weights
    (clump-aware): h_p = Σw·mass / (snow_density·cell²·J)
  - g_perp = g·n_z; flag curvature: g_perp += speed²·kappa (convex κ<0 lifts), floor 0.1
  - volumetric: eps_v_e += dt·div(APIC) (|Δ|≤0.5); p_vol = -bulk_modulus·eps_v_e; one-sided
    plastic compaction above compaction_pressure (eps_v_p accumulates, density_ratio =
    max(1+eps_v_p, 0.5) shrinks particle volume); tension clamp eps_v_e ≤ 0.5
  - constitutive (constitutive.wgsl): τ_trial = τ + 2G·dt·ε̇_dev; yield on τ_eq = ‖τ‖_F/√2,
    y = n0 + [tan(internal_friction_angle)·P̄ (DP) | mu_inertial(γ̇ = √2‖ε̇_dev‖_F)·P̄ (µ(I))]
    + hardening_modulus·γ_p; radial return τ = τ_trial·y/τ_eq_trial, Δγ_p = (τ_eq_trial-y)/(G+H)
  - pressure P̄ = max(0.5·ρ·g_perp·h_p + p_vol, 0); σ = τ − P̄·I
  - scatter: APIC momentum quantized i32; internal force f_i -= V_p·(h_p·σ)·grad_w_ip
    quantized (FORCE_FACTOR) into grid_forces_atomic
BINDINGS: 1 sim_info, 2-4 particles (position/velocity/mass), 5 grid_mass_atomic,
  6 grid_momentum_atomic, 7 particles_affine_matrix, 8 grid_forces_atomic (atomic<i32>),
  9 terrain_geometry_texture, 10 curvature_texture, 11 grid_mass_previous (array<u32>),
  12 particles_stress (array<mat2x2<f32>>), 13 particles_volumetric_strain (array<vec2<f32>>).

FILE: crates/compute_core/src/shaders/grid_physics.wgsl (461; 16×16×1)
PURPOSE: Model 0 grid support (no time integration): decodes mass/velocity, h = mass/(ρ·cell²)·n.z,
  peak flow thickness + 40-step NewCellsRollingWindow tracking, earth-pressure coefficients
  Ka/Kp = 2(1∓sqrt(1-cos²φ/cos²δ))/cos²φ − 1, lateral pressure grad_h2·0.5/h slope-corrected,
  gated by |grad| > tan(internal_friction_angle) → grad_h_buffer (consumed by compute_particles).
BINDINGS: 1 sim_info, 2 normals_texture, 3 curvature_texture, 4 grid_mass_atomic, 5
  grid_momentum_atomic, 6 grad_h_buffer, 7 peak_flow_thickness, 8 atomic_values,
  9 new_cells_rolling_window, 10 velocity_buffer.

FILE: crates/compute_core/src/shaders/grid_physics_curvilinear.wgsl (478; 16×16×1)
PURPOSE: Model 1 grid physics — the ONLY variant that integrates velocity on the grid.
  g_normal = -g/J; h = mass/(ρ·cell²·J); metric-scaled central-difference ∇h;
  hydrostatic_pressure_acceleration = -g_normal·K·∇h; proposed_u = u + dt·(g - hydrostatic);
  then friction applied as scaling 1 - a·dt/V (zeroing if it would reverse); writes
  grid_velocity[idx]; CFL signal V + sqrt(g·h); empty-node guard mass < 1e-2.
BINDINGS: 1 mass, 2 terrain_geometry_texture, 3 curvature_texture, 4 peak_flow_thickness,
  5 atomic_values, 6 momentum, 7 new_cells_rolling_window, 8 sim_info, 9 grid_velocity,
  10 grid_peak_velocity, 11 dem_texture, 12 tex_sampler, 13 particles_elevation.

FILE: crates/compute_core/src/shaders/grid_physics_mpmdac.wgsl (425; 16×16×1)
PURPOSE: Model 2 grid update: explicit u += dt·(f_int/m + g_t), internal force capped at 5g;
  terrain-parallel gravity g·n_z·normal.xy; basal friction via friction.wgsl applied against
  the proposed direction (no reversal); compressible CFL signal V + sqrt(g·h) + sqrt(K/ρ);
  writes grid_velocity + grid_peak_velocity; tracks peak_flow_thickness/new_cells.
BINDINGS: 1 mass, 2 terrain_geometry_texture, 3 peak_flow_thickness, 4 atomic_values, 5 momentum,
  6 new_cells_rolling_window, 7 sim_info, 8 grid_velocity, 9 grid_peak_velocity,
  10 grid_forces_atomic.

FILE: crates/compute_core/src/shaders/g2p.wgsl (553; entry g2p, WG_SIZE_1D)
PURPOSE: Model 1 G2P + advection. PIC gather (transfer_g2p) over 3×3 nodes from grid_velocity;
  curvilinear advection position += (v.x/l_x, v.y/l_y)·dt; updates particles_elevation from DEM;
  records the middle particle's trajectory into out_timestep_data (3 TimestepData slots);
  stops particles below velocity_threshold, on NaN, or within 2 cells of the world boundary
  (flag bits + stopped_particles counter). Contains a large commented-out friction block
  (lines 192-271) documenting Coulomb/Voellmy/samosAT/µ(I) formulas — reference material.
BINDINGS: 1 sim_info, 2 terrain_metrics, 3 tex_sampler, 4 particles_position, 5 particles_velocity,
  6 particles_state, 7 grid_velocity, 8 atomic_values, 9 out_timestep_data, 10 out_debug,
  11 particles_affine_matrix (unused), 12 dem_texture, 13 particles_elevation.

FILE: crates/compute_core/src/shaders/g2p_mpmdac.wgsl (454; entry g2p_mpmdac, WG_SIZE_1D)
PURPOSE: Model 2 G2P: APIC gather (transfer_g2p_affine) → G2PUpdate{velocity, affine_matrix};
  plain Cartesian advection position += velocity·dt; stores particles_affine_matrix;
  elevation update; same stopping/NaN/boundary logic; TimestepData recording.
BINDINGS: as g2p but 2 = terrain_geometry_texture.

FILE: crates/compute_core/src/shaders/compute_particles.wgsl (719; WG_SIZE_1D)
PURPOSE: Model 0 particle integrator (replaces g2p). Per particle: project 3D velocity to
  tangent plane; acceleration = tangential gravity (g + g·n_z·n) + optional lateral pressure
  from interpolated grad_h/h (enable_particle_interaction, gated by
  min(g·n_z, effective_acceleration_normal)); optional centrifugal term t^T·K·t (bed curvature);
  fixed damping velocity *= 0.99; Coulomb/samosAT/µ(I) normal friction (acceleration_by_normal_friction,
  334-374) + Voellmy/Voellmy-min-shear/samosAT drag (acceleration_by_drag_friction, 376-413);
  friction stop when deceleration·dt > speed; curved-bed position update via arc length
  s = v·dt with s − κ²s³/6, κs²/2 quadric terms; z randomized within flow depth for
  visualization; writes grid_peak_velocity_buffer + TimestepData for particle n/2.
BINDINGS: 1 sim_info, 2 dem_texture, 3 normals_texture, 4 tex_sampler, 5-10 particles
  (position/elevation/velocity/velocity_z/mass/state), 11 atomic_values, 12
  grid_peak_velocity_buffer, 13 out_timestep_data, 14 curvature_texture, 15 out_debug,
  16 grid_mass_atomic, 17 grad_h, 18 grid_velocity_buffer.

FILE: crates/compute_core/src/shaders/update_sim_info.wgsl (301; @workgroup_size(1,1,1))
PURPOSE: Adaptive dt + stop flags (section 6.2). Bindings: 1 sim_info, 2 atomic_values,
  3 new_cells_rolling_window.
```

#### Center of mass (3-shader pipeline, optional per step)

```
FILE: crates/compute_core/src/shaders/center_of_mass_seed.wgsl (292; 16×16×1)
  blob_labels[idx] = idx if mass > 0 else NO_BLOB (0xFFFFFFFF); early-out unless
  center_of_mass_biggest_blob flag. Bindings: 1 mass_buffer, 2 blob_labels, 3 sim_info.
FILE: crates/compute_core/src/shaders/center_of_mass_propagate.wgsl (325; 16×16×1)
  16 rounds of 8-neighbor min-label hooking + pointer compression (labels only decrease).
  Bindings: 1 blob_labels, 2 sim_info.
FILE: crates/compute_core/src/shaders/compute_center_of_mass.wgsl (479; 256×1×1, ONE workgroup)
  workgroupBarrier tree reduction; biggest-blob mode restricts accumulation to the largest
  label (per-blob mass atomics + tree reduction); writes center_of_mass[timestep] =
  { com = (x, y, dem_elevation), total_mass }. Bindings: 1 mass_buffer, 2 center_of_mass,
  3 sim_info, 4 dem, 5 tex_sampler, 6 atomic_values, 7 blob_labels, 8 blob_mass.
```

#### Evaluation / metrics (run once after Finished)

```
FILE: crates/compute_core/src/shaders/evaluate_mass_movement.wgsl (293; 16×16×1)
  Counts intersection/undershoot/overshoot between simulated (peak_flow_thickness >
  peak_flow_thickness_threshold) and ROI bitmask into EvaluationCounts atomics (overlay the
  first 32 bytes of EvaluationResult buffer).
FILE: crates/compute_core/src/shaders/evaluate_mass_movement_points.wgsl (304; 16×16×1)
  Extends counts with min/max elevation + min/max cell index via ordered-float trick
  (float_to_ordered). UNCERTAIN: the min/max elevation atomics are seeded to u32::MAX by
  evaluate_gpu but the shader never atomic-min/maxes the elevation value itself — index
  tracking looks incomplete (work-in-progress); verify before relying on *_ref outputs.
FILE: crates/compute_core/src/shaders/chamfer_prepare.wgsl (294; 16×16×1)
  Seeds chamfer_nearest_roi (ROI bitmask) and chamfer_nearest_sim (thresholded thickness)
  with own coords or NO_SEED.
FILE: crates/compute_core/src/shaders/chamfer_flood.wgsl (336; 16×16×1)
  One Jump-Flooding-Algorthm pass (Rong & Tan) with ChamferParams.step; 8 neighbors at ±step;
  keeps squared-distance-nearest seeds for both fields. Dispatched log2(n) times with
  step n/2…1 (see dispatch_chamfer_distance).
FILE: crates/compute_core/src/shaders/chamfer_reduce.wgsl (347; 256×1×1, one workgroup)
  Tree-reduces sum_sim_to_roi/count_sim/sum_roi_to_sim/count_roi into the chamfer section of
  EvaluationResult; CPU normalizes (chamfer_from_sums in evaluation.rs). Release cells
  (ReleaseAreas > RELEASE_THICKNESS_THRESHOLD) are skipped in the sums but stay seeds; roi_seeded/sim_seeded flag
  non-empty masks so an empty mask (not a release-only one) gives an infinite distance.
  evaluate_mass_movement.wgsl and CPU evaluate() likewise skip release cells, so
  IoU/chamfer do not grow with the release area fraction.
FILE: crates/compute_core/src/shaders/compute_beeline_distance.wgsl (359; 256×1×1)
  Finds highest/lowest DEM elevation among avalanche cells (paired index+elevation through the
  reduction) and writes beeline distance = |Δ(x,y,z)| between cell centers into the last
  section of EvaluationResult.
```

#### GPU test shaders (dispatched only from Rust tests)

```
FILE: test_transfer.wgsl (426) — p2g/g2p round trip in one invocation (entry @workgroup_size(1)).
FILE: test_transforms.wgsl (341) — uv/xy/idx transform round trips (16×16×1).
FILE: test_sampling.wgsl (321) — textureSampleLevel vs textureLoad of the DEM (64×1×1).
FILE: test_utils.wgsl (311) — utils + AtomicValues checks (1×1×1).
(Files are large on disk because of inlined import copies; top section is the source of truth.)
```

---

### 8.4 crates/simulation — high-level orchestration

```
FILE: crates/simulation/Cargo.toml
NOTES: deps data_processor, compute_core, wgpu, anyhow, bytemuck, tempfile, serde(+json), rand,
  regex, futures-intrusive, tracing, test-log, futures, web-time; dev pollster, criterion, tokio.
  [[bench]] sim_benchmark, harness = false.

FILE: crates/simulation/src/lib.rs  (3616 lines)
TYPE: Rust source
PURPOSE: The user-facing Simulation facade: lifecycle state machine, data loading, model
  dispatch, GPU readback caching, post-processing, evaluation, Zarr output. Crate doc (lines
  1-36) describes Load → Prepare → Simulate → Consume; wasm32-compatible (file output native-only).

RESPONSIBILITIES / PUBLIC API:
  - init_logging() (66; Once-guarded tracing_subscriber + EnvFilter from env).
  - SimulationState (91-114): Uninitialized < DemMissing < DemLoaded < TerrainAnalyzed <
    ReleaseAreasComputed < ParticlesInitialized < Running < Finished < PostProcessed < Evaluated.
  - SimulationLoadResult (119-145): settings, enable_center_of_mass, enable_particle_relaxation,
    dem, roi, dem_path, release_areas_path, release_area_fraction, crown_line_method,
    crown_line_exit_distance, batch_compute_steps, output_path. Splitting load_data/apply_data lets callers load
    concurrently with GPU init.
  - Simulation (156-193): owns orchestrator: ComputeOrchestrator; pub fields settings, dem, roi,
    gpu_cache: GpuCache, ava_mask, crown_line, output_path, enable_center_of_mass,
    enable_particle_relaxation; private state/sim_info/number_particles/roi_uploaded/release_*;
    #[cfg(not(wasm32))] output: Option<data_processor::output::Output>.
  - Constructors: new/new_with_gpu (197-229), new_with_settings (233; new + load_data +
    apply_data), create(settings) (371), create_default(dem_path) (378),
    create_default_with_release_areas (389), create_example(dem_path) (406).
  - DEM setters: set_dem_default (419), set_dem (441), set_dem_with_bounds (491),
    set_release_areas(&[f32]) (578; values = snow thickness in METERS per cell),
    set_roi(&[bool]) (596), upload_roi (619).
  - load_data(&Settings) (300) → data_processor::create_sim_settings_and_dem; apply_data (325).
  - prepare() (634): analyze_terrain → upload_roi → load_release_areas → initialize_particles.
  - run_n_steps(steps) (651-732): auto-prepare; first call picks prepare_compute_particles /
    prepare_mpm / prepare_mpmdac by sim_model; clamps to remaining budget; resets
    gpu_cache; steps via step_terrain_following / step_curvilinear / step_mpmdac;
    Finished on timestep ≥ max_steps or SIM_STOPPED. Returns SimInfo.
  - run() (739): full pipeline to Finished in one call; bails when no particles.
  - compute_particles() (1323): run_sim without re-creating buffers (used by run() and by
    renderers that must keep bindings alive).
  - post_process() (768): threshold peak_flow_thickness → biggest blob → ava_mask; masks
    peak_velocity to the same blob; requires Finished.
  - evaluate() (796): CPU IoU + runout vs roi; requires PostProcessed. evaluate_gpu() (839)
    adds chamfer/beeline metrics.
  - save()/save_with_path (849/943; native Zarr), export_zarr_entries() (898; wasm),
    site_name()/scenario_name() (858/874; dem/release file stem + 16-hex hash).
  - Cached fetchers (1354-1705): fetch_roughness / peak_flow_thickness / slope_angle /
    slope_aspect / terrain_geometry_texture / terrain_curvature / release_areas /
    peak_velocity / timestep_data / particles_position / particles_velocity /
    particles_velocity_z / particles_mass / grid_mass / grid_momentum / elevation / state /
    center_of_mass; get_center_of_mass / get_total_mass / get_total_volume (1705-1732);
    fetch_results (1747); print_grid (1770); fetch_sim_info/fetch_atomic_values (1020/1032).
  - Private: analyze_terrain (1045), detect_crown_line_by_particle_simulation (1073),
    load_release_areas (1186; three sources: uploaded array, release_areas_path texture,
    thresholds via GPU, optional release_area_fraction crown-line estimation),
    initialize_particles (1300).
  - Big #[cfg(test)] module (1891-3600): lifecycle, cache reset, end-to-end release estimation.
DATA FLOW: Settings/DEM → GPU buffers → per-step dispatch → lazily-cached CPU readbacks.
IMPORTANT DETAILS:
  - gpu_cache memoizes every fetch_*; run_n_steps and pipeline stages call
    gpu_cache.reset_simulation_result()/reset_all() — never write into gpu_cache directly.
  - `orchestrator()` getter (263) exposes &ComputeOrchestrator so renderers can bind buffers.

FILE: crates/simulation/src/release_estimation.rs  (866 lines)
TYPE: Rust source
PURPOSE: Release-area estimation from an avalanche outline + DEM (module doc lines 1-13):
  detect the outline's crown line, then fill a target fraction of the outline area below it
  with chamfer-weighted geodesic distance weighting.
PUBLIC API: ReleaseEstimationConfig (114; defaults fraction 0.25, slab_thickness 1.0,
  crown_elevation_band 0.0, min_crown_slope_deg 28.0 (Simulation passes min_slope_angle), min_crown_component_cells 3, sink_fill_epsilon 1e-2,
  min_particles_per_crown_cell 2, crown_exit_distance 0.75, expected_crown_slope_range (28.0, 60.0)); ReleaseEstimate
  (152; release_areas: Vec<f32>, crown_line: Vec<bool>, counts, mean_crown_slope_deg,
  release_fraction); CrownDetection (270); detect_crown_line(dem, roi, config) (504; D8 flow
  routing on sink-filled DEM); crown_line_from_particle_counts(counts, dem, roi, config) (527;
  GPU-particle alternative); detect_crown_line_by_gradient_step(dem, roi, config) (one-cell
  uphill step per outline cell, candidate if it exits the outline > crown_exit_distance cells);
  estimate_release_areas_with_crown(dem, roi, potential, crown, config) (fills only potential
  release cells, nearest to the crown first; Simulation passes the GPU threshold classification);
  estimate_release_areas(dem, roi, potential, config); helpers any_neighbor_is (42), entry_cell (61).
CALLERS: Simulation::load_release_areas when release_area_fraction is set.
DEPENDENCIES: compute_core::dem only (pure CPU).

FILE: crates/simulation/benches/sim_benchmark.rs  (87 lines)
NOTES: criterion group GPU_Benchmarks over avaMal (with/without particle interaction) and Vals
  DEMs; run with `cargo bench -p simulation`. Requires a GPU adapter.
```

### 8.5 crates/data_processor — input/output

```
FILE: crates/data_processor/Cargo.toml
NOTES: crate-type ["cdylib","rlib"]; deps compute_core, bincode, half, image, tracing, lz4_flex,
  reqwest, tiff, shapefile, serde(+json), thiserror, weezl; native-only xz2/zstd/zarrs*/ndarray/
  bytes/futures; wasm32-only ruzstd (pure-Rust zstd so blosc chunks decode on wasm), web-sys.
  dev-deps criterion, wiremock, tokio, tempfile.

FILE: crates/data_processor/src/lib.rs  (1816 lines)
TYPE: Rust source
PURPOSE: DEM / release-area / settings I/O and the central settings+DEM loader.
PUBLIC API:
  - DataProcessorError (36), EsriGridError (67), EsriGridHeader (126), EsriGrid (186;
    from_file/from_reader/flip_y/get/to_file) — ESRI ASCII grids.
  - write_png (407), rgba_bytes_to_f32 (460), f32_to_rgba_bytes (468) — the custom PNG DEM
    encoding used by data/avaframe/*.png (little-endian f32 RGBA channels).
  - GeoTiffError (480), read_geo_tiff / _from_bytes / _from_reader (502-512).
  - load_release_areas(path) (712), load_release_areas_from_bytes (743) — returns Vec<f32>
    thickness per cell.
  - load_dem(path) (763), load_dem_from_bytes(bytes, ext, source) (780) — dispatch by
    extension (.asc/.ags → EsriGrid, .tif/.tiff → GeoTiff, .png → rgba f32).
  - settings_to_json_file / settings_from_json_file (919/926).
  - create_sim_settings_and_dem(settings) (854, async): outline-only configs fetch swissALTI3D
    tiles via TileManager; rasterizes GPX/shapefile outlines (rasterizer) into the roi mask;
    returns (SimSettings, Dem, Vec<bool> roi). create_sim_settings_and_dem_from_path (845).
DATA FLOW: file/bytes → Dem/Vec<bool> → Simulation. DEPENDENTS: simulation::load_data,
  wasm_bindings (bytes variants for browser uploads).

FILE: crates/data_processor/src/output.rs  (1835 lines)
PURPOSE: Native Zarr v3 result store (zarrs). PUBLIC API: OutputError (28); Output (85):
  new(path), site_exists, add_new_site (154), add_new_scenario (283), add_new_run (1236),
  write_flow_fields (1001), write_particle_position (1130), read_dem_from_store (851),
  read_release_area_from_store (968); Scenario (1370). Layout documented in section 6.4.
  Native-only (zarrs doesn't build for wasm32 — Simulation::save is #[cfg(not(wasm32))]).

FILE: crates/data_processor/src/zarr_writer.rs  (285 lines)
PURPOSE: Minimal Zarr v3 writer used to export results FROM the browser;
  pub struct ZarrEntry { path, bytes } consumed by wasm export_zarr_entries()/JS download.

FILE: crates/data_processor/src/blosc.rs  (333 lines)
PURPOSE: Pure-Rust blosc1 decoder (zstd + bitshuffle/byte-shuffle) + decode_zstd_frame;
  used on wasm32 (via exported decode_blosc_chunk) to read Zarr chunks; also native reads.

FILE: crates/data_processor/src/tile_manager.rs  (1023 lines)
PURPOSE: swissALTI3D tile fetch/cache (TileManager::new("dtm_cache.zarr"), get_dem(bbox)),
  BEV Austria DTM links, LZW TIFF decode, 2 m→5 m resample. Env SWISSALTI_TEST_TILES for tests.
  Network dependency; caches into dtm_cache.zarr (gitignored).

FILE: crates/data_processor/src/rasterizer.rs  (610 lines)
PURPOSE: Polygon → grid-mask rasterization (RasterizerError, GeoPolygon, Point2D, bbox,
  scanline fill); used for shapefile outlines.

FILE: crates/data_processor/src/shapefile_reader.rs  (282 lines)
PURPOSE: Read polygon shapefiles (ShapeError) → GeoPolygon list; feeds rasterizer.

FILE: crates/data_processor/src/caaml_parser.rs  (559 lines)
PURPOSE: CAAMLv6 avalanche bulletin parsing (AvalancheBulletinCollection, Bulletin, MetaData).
  Tested in tests/caaml_parser_tests.rs with tests/fixtures/caaml_sample.json.
  UNCERTAIN: no production caller found in the workspace — appears to be for future data work.

FILE: crates/data_processor/tests/caaml_parser_tests.rs (726) — caaml_parser coverage;
  fixture: crates/data_processor/tests/fixtures/caaml_sample.json.
```

### 8.6 crates/render_core — renderer

```
FILE: crates/render_core/Cargo.toml
NOTES: lib deps only wgpu, bytemuck, anyhow, tracing (windowing-agnostic). Examples/tests
  dev-depend on winit, egui*, pollster, image, data_processor, simulation, compute_core.
  Declared [[example]]s: dem_viewer, dem_snapshot, live_sim, video_export.

FILE: crates/render_core/src/lib.rs  (238 lines)
PURPOSE: Renderer facade. pub struct GpuContext { instance, adapter, device, queue } with
  new(instance, compatible_surface) / headless(); pub struct Renderer { camera: OrbitCamera,
  terrain: TerrainRenderer, particles: ParticleRenderer, colorbar: ColorbarRenderer, … } with
  new(device, queue, color_format, width, height, terrain), set_grid_overlay(device,
  Option<&Buffer>, OverlayRange), set_particles, resize, render(device, queue, target) —
  one "Scene Pass": terrain → particles → colorbar. DEPTH_FORMAT = Depth32Float. Re-exports
  OrbitCamera, ParticleBuffers, ParticleRenderer, OverlayRange, TerrainData, TerrainRenderer.

FILE: crates/render_core/src/camera.rs  (353 lines) — OrbitCamera: framing(terrain, aspect),
  view_projection(), billboard_axes(), orbit/pan/zoom, set_aspect.
FILE: crates/render_core/src/terrain.rs  (541 lines) — TerrainData::new(width, height,
  cell_size, heights) + with_vertical_exaggeration + fit_samples() mesh builder;
  OverlayRange::new(min,max)/with_threshold/with_label/with_unit; TerrainRenderer::new(device,
  queue, color_format, depth_format, terrain) (264), set_overlay (447; scalar grid storage
  buffer), set_overlay_range, update_camera, draw. Shader:
  crates/render_core/src/shaders/terrain.wgsl (162 lines), include_str! at terrain.rs:385.
FILE: crates/render_core/src/particles.rs  (235 lines) — ParticleBuffers<'a> (5 storage buffers:
  position/velocity/velocity_z/state/elevation), ParticleRenderer::new/set_buffers/set_count/
  set_radius/set_max_velocity/update_camera/draw. Shader:
  crates/render_core/src/shaders/particles.wgsl (93 lines, billboarded points colored by speed).
FILE: crates/render_core/src/colorbar.rs  (877 lines) — private ColorbarRenderer legend;
  shader crates/render_core/src/shaders/colorbar.wgsl (31 lines).
FILE: crates/render_core/src/math.rs  (170 lines) — minimal Vec3, Mat4, perspective_rh,
  look_at_rh, mul, transform_point.
FILE: crates/render_core/src/capture.rs  (96 lines, native only) — render_to_rgba8 offscreen
  capture (used by dem_snapshot, live_sim snapshots, video_export).
FILE: crates/render_core/tests/headless_render.rs — offscreen pixel assertions; skips without GPU.

FILE: crates/render_core/examples/live_sim.rs  (1733 lines)
TYPE: Rust example (the de-facto desktop GUI entry point)
PURPOSE: Interactive 3D viewer that advances the sim STEPS_PER_FRAME = 1 per redraw (~30 ms
  render gate) and renders the Simulation's OWN storage buffers — no CPU copies.
KEY CONTENTS:
  - CLI: [settings.json] [exaggeration]; env LIVE_SIM_SNAPSHOT=<path> (run_to_completion in
    256-step batches → 1024×640 PNG via render_to_rgba8), LIVE_SIM_NO_PARTICLES.
  - apply_panel_defaults(): density 200, slab 1.0, mu 0.2, xi 2000, cfl 0.5, slope angles
    28/60, release elevations 1500-8848, roughness 0.01, internal friction 40°, 8 particles/cell,
    6000 steps, SimModel::Curvilinear, FrictionModel::Voellmy, feature flags on.
  - SimulationView holds the Simulation + cloned device/queue/adapter/instance + cloned
    GRID_BUFFERS (GridPeakVelocity, GridPeakFlowThickness, GridMass, ReleaseAreas, SlopeAngle,
    SlopeAspect, Roughness); apply_settings reloads the sim in place via Simulation::create.
  - egui settings panel: testcase combo (discovers `<case>.png`/`<case>releaseTexture.png`
    pairs next to the DEM), model combo (terrain-following/curvilinear/MPMDAC), friction combo,
    parameter sliders; keyboard: 0-7 overlays, P particles, R restart, V reset view, ± speed
    0.25-256×.
  - Unit tests at the bottom: testcase discovery, in-place reload across models, egui sliders.
FILE: crates/render_core/examples/dem_viewer.rs (261) — winit orbit DEM viewer.
FILE: crates/render_core/examples/dem_snapshot.rs (115) — headless DEM → PNG.
FILE: crates/render_core/examples/video_export.rs (402) — headless run, RGBA frames piped to
  ffmpeg (H.264 MP4).
```

### 8.7 crates/wasm_bindings — browser API

```
FILE: crates/wasm_bindings/Cargo.toml
NOTES: crate-type ["cdylib","rlib"], lib name `avalanchers` (JS import name). deps
  console_error_panic_hook, compute_core, simulation, data_processor, render_core, wgpu,
  bytemuck, wasm-bindgen(-futures), js-sys, web-sys (HtmlCanvasElement), anyhow, thiserror,
  getrandom 0.3 + 0.4 (wasm_js), tracing-wasm, serde-wasm-bindgen.

FILE: crates/wasm_bindings/src/lib.rs  (611 lines)
PURPOSE: The JS-facing API.
PUBLIC API (#[wasm_bindgen]):
  - init() (25; #[wasm_bindgen(start)]): panic hook, tracing_wasm, records page URL → BASE_URL.
  - decode_blosc_chunk(bytes) / decode_zstd_chunk(bytes) (48/55): exposed so JS can decode
    Zarr chunks (window.wasmDecoders in the frontend).
  - WasmTimestepData (64): getters position/dt/velocityMagnitude/time/stepDistance/
    travelDistance/cfl as zero-copy Float32Array views.
  - WasmSettings (112): new(), from_json(json), dem_path getter/setter.
  - WasmSimulation (148): wraps Simulation + optional RenderView. Async: new(),
    create_example(dem_path) (loads data/avaframe/<name>.png), create(val) (settings object via
    serde-wasm-bindgen), set_dem(dem_data, width, height, cell_size, bounds…, map_factor) (216),
    set_dem_default (246), load_dem_bytes(bytes, ext, source) (262),
    load_release_areas_bytes (292), set_release_areas (303), prepare (310),
    save_results_zarr() (324; wasm only → JS Array of {path, bytes}), run (344),
    fetch_peak_velocity / peak_flow_thickness / release_areas / roughness / slope_angle /
    slope_aspect / results (394-432), get_timestep_data (471).
    Zero-copy getters: bounds, cell_size, dem, width, height, x, y, dem_trajectory_info,
    peak_velocity, slope_aspect, slope_angle, roughness, release_areas, peak_flow_thickness,
    result_store_name.
    Renderer (wasm only): attach_renderer(canvas, exaggeration) (487), renderer_attached,
    is_finished, render_frame(steps) (532; step + draw, call from requestAnimationFrame),
    orbit/pan/zoom/reset_view (545-567), forget_view (572), set_overlay(name) (581),
    set_particles_visible (591), resize (600).
FILE: crates/wasm_bindings/src/renderer.rs  (320 lines, wasm only)
PURPOSE: RenderView — the wasm counterpart of live_sim. Overlay enum (32: None,
  PeakFlowVelocity, PeakFlowThickness, GridMass, ReleaseAreas, SlopeAngle, SlopeAspect,
  Roughness; from_name, per-overlay range()/label/unit, buffer_name()); GRID_BUFFERS
  [BufferName; 7] (21); clone_grid_buffers (117); clone_particle_buffers (133 — zero-fills a
  ParticlesVelocityZ stand-in for MPM models); RenderView::new(orchestrator, canvas, dem,
  exaggeration, particle_count) (179), refresh_buffers/apply_overlay/apply_particles/render,
  orbit/pan/zoom/reset_view. Works because the renderer shares the simulation's wgpu Device —
  it observes the same buffers run_n_steps keeps writing.
FILE: crates/wasm_bindings/tests/bindings.rs (node; WasmSettings) and
  crates/wasm_bindings/tests/bindings_gpu.rs (run_in_browser; skips heavy GPU test on CI).
BUILD: wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg
  (README uses --no-opt for dev / --release). Output dir is gitignored (`*` in
  frontend/js/pkg/.gitignore) — the deploy workflows rebuild and copy it.
```

### 8.8 crates/python_bindings — Python API

```
FILE: crates/python_bindings/Cargo.toml
NOTES: lib name `_avalanchers` (cdylib). deps pyo3 0.28, pythonize, numpy, ndarray, pyo3-log,
  tracing, pollster, tokio, serde_json, simulation, compute_core, data_processor.

FILE: crates/python_bindings/src/lib.rs  (1212 lines)
PURPOSE: The `avalanchers._avalanchers` module. All methods synchronous (block_on / tokio
  current-thread runtime inside create()). Grids returned as (height, width) numpy arrays.
PUBLIC API:
  - list_available_gpus() -> list[str]
  - PySimInfo (57): timestep, dt, elapsed_time, number_particles, elevation_threshold,
    max_velocity, max_flow_thickness, flags; __getitem__, __repr__.
  - PyChamferDistance (149): sim_to_roi, roi_to_sim, chamfer_distance.
  - PyEvaluation (196): intersection, undershoot, overshoot, iou, sim_to_roi, roi_to_sim,
    chamfer, chamfer_distance, horizontal_distance, vertical_drop, beeline_3d, *_ref,
    peak_velocity.
  - PyTimestepData (361): velocity/position as PyArray2, dt as PyArray1.
  - PySettings (406): new, from_json, to_json, get_dem_path, set_dem_path.
  - PySimulation (461): new(gpu=None) [staticmethod], create(settings dict) (488 —
    depythonize → Settings::loads), set_max_timesteps, create_example(dem_path),
    set_dem(array2, cell_size) (532 — requires C-contiguous), set_dem_with_bounds(…,
    bounds_xmin/xmax/ymin/ymax, map_factor) (556), set_dem_default (594),
    set_release_areas(array2) (617), run() (624), run_n_steps(steps) -> PySimInfo,
    post_process, evaluate -> PyEvaluation, prepare, save(path); getters: state, cell_size,
    released_particles_per_cell, dem, roi, crown_line, crown_candidates, potential_release_areas, dem_bounds, get_peak_velocity,
    get_terrain_geometry_x/y/z, get_gravity_x/y, get_release_areas, get_peak_flow_thickness,
    get_timestep_data, get_elevation_threshold, get_particles_position(_xy),
    get_particles_velocity(_xy), get_particles_elevation, grid_mass, grid_momentum,
    center_of_mass_x/y/z, stopped.
  - #[pymodule] avalanchers (1198) registers all classes + list_available_gpus.
DEPENDENTS: python_module/avalanchers/__init__.py (`from ._avalanchers import *`).
NOTE: getters raise ValueError when the underlying data is unavailable (e.g. roi/crown_line
  before an outline-based run).
```

### 8.9 crates/cli — native binary

```
FILE: crates/cli/Cargo.toml — deps simulation, compute_core, data_processor, clap(derive),
  anyhow, tracing, pollster. settings.json beside it (demo DEM path).
FILE: crates/cli/src/main.rs  (476 lines)
PURPOSE: `avalanchers` command. STRUCTURE:
  - parse_bool/parse_sim_model/parse_friction_model/parse_crown_line_method/
    parse_constitutive_model (14-42).
  - Args (44-145): positional file_path (settings.json), --about, --list_devices, plus long
    options mirroring every Settings key (--sim_model, --friction_model, --bulk_modulus,
    --compaction_pressure, --enable_*, …).
  - apply_overrides (148-274): only Some() values overwrite the file settings.
  - main (276): init_logging → --about attribution text / --list_devices
    (compute_core::list_devices) → Settings::from_json (default "settings.json") → overrides →
    Simulation::new_with_settings → run() → fetch peak velocity/flow thickness, particle
    diagnostics, total mass/volume → print_grid → save().
  - Test module (406-475): cli_args_override_settings_file.
```

### 8.10 python_module — shipped Python package

```
FILE: python_module/avalanchers/__init__.py  (369 lines)
PURPOSE: `avalanchers` package: re-exports the pyo3 extension + pure-Python viz.
KEY CONTENTS: from ._avalanchers import *; from .output import AvalanchersOutput,
  AvalanchersScenario; __version__ = version("avalanchers").
  create_mesh(sim) (12; dem<nan mask + meshgrid); blur_nan_grid(data, passes) (23);
  plot3d(sim, parameter, particles=False, threshold_value=1e-3, blur_passes=0) (43; pyvista
  StructuredGrid, scalar clip, ×1.3 vertical exaggeration, trame backend in Jupyter, raises
  ImportError pointing at avalanchers[viz]); plot_dem(sim, ax) (109; matplotlib hillshade);
  plot2d(sim, parameter, …) (+ax2d); plot_overview(sim); calculate_dice(model_a, model_b);
  plot_comparison_binary / plot_comparison(sim, parameter, reference_array);
  is_jupyter()/setup_jupyter_3d()/import_plt().
  Consumed sim attributes: sim.dem, sim.cell_size, sim.release_areas, sim.roi,
  sim.elevation_threshold, sim.particles_position/velocity, and any result grid via
  getattr(sim, parameter) (peak_velocity, peak_flow_thickness, …).

FILE: python_module/avalanchers/output.py  (188 lines)
PURPOSE: Read-side of the Zarr store (zarr + xarray only, no Rust).
API: AvalanchersOutput(path) → attrs (title, conventions, source, avalanchers_version,
  avalanchers_repo, avalanchers_format), sites, list_all_scenarios(), get_site(name_or_hash);
  AvalancherSite → .dem (xr.open_dataset(engine="zarr", group=site, consolidated=False)),
  scenario_names, get_scenario; AvalanchersScenario → attrs aspect_release_degrees,
  release_volume_m3, number_of_runs, .dataset (isel(run=slice(0, number_of_runs))).

FILE: python_module/tests/test_bindings.py (204) — GPU binding smoke tests (avaParabola):
  list_available_gpus, PySettings round trip, sim.state/cell_size/dem/dem_bounds/peak_velocity/
  terrain_geometry_x/y/z/gravity_x,y/release_areas/peak_flow_thickness/timestep_data/
  particles_*/stopped/particles_position_xy; run_n_steps → state "Running"; setters and the
  ValueError cases for roi/crown_line; center_of_mass_x/y/z.
FILE: python_module/tests/test_run.py (40) — settings-dict run on avaPfa; create_example;
  numpy DEM/release round trips.
FILE: python_module/tests/test_io.py (112) — save() Zarr store: avalanchers_format "0.1.0",
  avalanchers_repo "https://github.com/Gro2mi/avalanchers", site <name>_<16-hex> (avaPfa →
  d786eb15c92acdd8), scenario hash d1fba762150c532c, release_volume_m3 ≈ 7283.78,
  number_of_runs 2, dims ("y","x"), vars peak_flow_velocity/peak_flow_thickness/release_area.
FILE: python_module/tests/test_verification.py — EMPTY (0 lines; placeholder).
```

### 8.11 frontend — web app

```
FILE: frontend/index.html  (985 lines)
PURPOSE: Bootstrap UI, workflow cards: #shortcutCard (AvaFrame example dropdown #demDropdown +
  #runShortcut), #demCard (drop zones: .gpx/.tif/.tiff/.asc files, webkitdirectory Zarr-store
  picker, #zarrSiteDropdown, zoom slider 12-18), #releaseCard (sliders doubling as settings:
  density, slab thickness, min/max slope angle, release min/max elevation, roughness threshold;
  #calculateReleaseButton, #zarrScenarioDropdown), #settingsCard (#simModelDropdown,
  #frictionModelDropdown, feature checkboxes enable_curvature / enable_particle_interaction /
  enable_particle_relaxation / enable_earth_pressure_coefficient, step/CFL/friction sliders),
  loading screen, #viewerCanvas 3D card. Loads Plotly CDN + js modules; main.js as ES module.

FILE: frontend/js/main.js  (1168 lines)
PURPOSE: Workflow controller. Line 1: import init, { WasmSimulation, decode_blosc_chunk,
  decode_zstd_chunk } from "./pkg/avalanchers.js"; window.sim = await withTimeout(
  WasmSimulation.new(), 5000, …) (line ~1135).
KEY FUNCTIONS: getSettings() (assembles the settings object — keys must match Settings schema),
  applySources/applyDem/applyReleaseAreas, loadDemFromGpx (tile download), loadDemFromRaster,
  loadDemFromZarr/selectZarrSite/selectZarrScenario, deriveGeoreference, calculateReleaseAreas,
  runSimulation, saveResults/writeStoreFile (Zarr results via save_results_zarr),
  prepareSimulation, loadExampleCase, renderResults, updateHistogramForOverlay
  (HISTOGRAM_VARIABLE_BY_OVERLAY), SHORTCUT_OVERLAYS, withTimeout.

FILE: frontend/js/viewer.js  (332 lines) — export class LiveViewer: owns the canvas +
  requestAnimationFrame loop calling sim.render_frame(steps); engine calls wrapped in trySim;
  camera drag/wheel → sim.orbit/pan/zoom; DPR cap 2; pinch zoom.
FILE: frontend/js/sources.js  (416 lines) — file ingestion: drag&drop, pickers, Zarr v3 store
  reading <store>/<site>/<scenario> (ZARR_DEM_ARRAY='dem' at site level,
  ZARR_RELEASE_ARRAY='release_area' per scenario); uses window.wasmDecoders for blosc/zstd.
FILE: frontend/js/plot.js  (279 lines) — Plotly result plots; fetchers
  sim.fetch_release_areas() / fetch_slope_angle() / fetch_slope_aspect() / fetch_roughness() /
  fetch_peak_velocity() / fetch_peak_flow_thickness().
FILE: frontend/js/tile_utils.js  (258 lines) — latLonToWebMercator, latLonToTile, IndexedDB
  tile cache (dbPromise).
FILE: frontend/js/utils.js (151) — helpers. FILE: frontend/js/plotly_dark.js (552) — Plotly theme.
FILE: frontend/dev_server.py  (70 lines) — HTTPS server on port 443 (WebGPU needs a secure
  context); registers application/wasm MIME; copies ../data/avaframe → frontend/data/avaframe;
  auto-generates self-signed server-cert.pem/server-key.pem with openssl; Cache-Control: no-store.
FILE: frontend/js/pkg/ — wasm-pack build output (avalanchers.js/.d.ts/.wasm, package.json);
  gitignored; mirror may exist at crates/wasm_bindings/wasm_dist/ after CI builds.
FILE: frontend/data/avaframe/ — copy of the AvaFrame DEM/release PNGs (+ .rp/.aabb) created by
  dev_server.py or manually; frontend/data/gpx/ sample outlines; icons/ (incl.
  android-chrome-512x512.png used as live_sim window icon); site.webmanifest.
```

### 8.12 CI, docs, data

```
FILE: .github/workflows/build.yaml
  Matrix (ubuntu always; macos+windows on v* tags). Installs Mesa llvmpipe / Metal / DX12
  (WGPU_BACKEND, VK_ICD_FILENAMES). Steps: cargo fmt --check; cargo clippy -q -- -D warnings;
  cargo build -p cli --release; cargo test -p compute_core -p data_processor -p simulation
  --release; Tarpaulin coverage --fail-under 80 (env SKIP_FOR_TARPAULIN=true); pip install .;
  ruff check ./python_module; pytest ./python_module/tests; maturin wheel build;
  wasm-pack build … --out-dir wasm_dist; wasm-pack test --node; node --check on all 7 frontend
  JS files; artifact upload. wasm-pack line 207.
FILE: .github/workflows/release.yaml — on successful Build of a v* tag: GitHub Release + PyPI
  publish (environment pypi).
FILE: .github/workflows/gh-pages-deploy.yaml — on successful Build of main: rsync frontend/ +
  data/avaframe/ (excluding *.pem *.py *.rp *.gpx), copy built .wasm/.js into deploy_dir/js/pkg/,
  push to gh-pages (gro2mi.github.io/avalanchers).
FILE: .github/workflows/sftp-deploy.yaml — push to main: lftp mirror of frontend/ to a server
  (secrets SFTP_USERNAME/PASSWORD/HOST/PORT/REMOTE_DIR — [REDACTED]).
FILE: .github/workflows/copilot-review-instructions.md — instructions for the Copilot review
  workflow.

FILE: docs/mpmdac.md  (424 lines, untracked)
PURPOSE: Design doc for sim_model 2 (MPMDAC). Sections: model table; physical formulation
  (J = 1/n_z, h = m/(ρΔ²J), g_t = g·n_z·(n_x,n_y)); momentum balance; constitutive model
  (elastic predictor–corrector, Drucker-Prager y = c + P̄·tanφ vs µ(I) y = c + μ(I)·P̄,
  radial return, hardening H·ε_p); volumetric response (bulk_modulus, compaction_pressure,
  density_ratio feedback); friction laws; settings table; implementation map (p2g_mpmdac.wgsl,
  grid_physics_mpmdac.wgsl, g2p_mpmdac.wgsl, constitutive.wgsl, friction.wgsl, simulation/lib.rs,
  live_sim.rs); validation counts (143 compute_core tests, 48 simulation tests, #[ignore]
  test_mpmdac_diagnostic_avawog); known limitations. READ BEFORE touching model 2.
FILE: docs/release_area_estimation_section.tex (untracked) — LaTeX write-up of
  release_estimation.rs (thesis section).

FILE: data/avaframe/ — AvaFrame test cases as custom-encoded PNG DEMs (little-endian f32 in
  RGBA channels; decoded by data_processor::rgba_bytes_to_f32): <case>.png (DEM),
  <case>releaseTexture.png (release), <case>.rp (release params), <case>.aabb (bounds), plus
  avaGar_remeshedDEM5.00.asc and LICENSE.txt (EUPL-1.2). Cases: avaAlr, avaArzlerUni, avaBowl,
  avaChannel, avaDamBreak, avaFPtest, avaFlatPlane, avaFlowPy (no release texture), avaGar,
  avaHelix, avaHelixChannel, avaHit, avaHockeyChannel, avaHockeySmall, avaHof,
  avaInclinedPlane, avaKot, avaMal, avaParabola, avaPfa, avaPyramid(45), avaSimilaritySol,
  avaSlide, avaTripleBowl, avaWog.
FILE: data/vals/ — PAR6_Vals_Gries_dtm_10_utm32n_bil_.tif (GeoTIFF DTM) + release.tif
  (Zenodo Flow_py dataset, doi 10.5281/zenodo.5154787).
FILE: data/outline/ — polygon_10720.shp / polygon_10721.shp (+dbf/prj/shx/cpg): mapped
  avalanche outlines used by the outlines_path/evaluation path.
FILE: data/landcover/ — README.md (Copernicus CLC tiling recipe, category table),
  create_landcover_tiles.ps1.
FILE: data/lawis/ — lawis_analysis.ipynb + requirements.txt (research; data/ gitignored).
FILE: python_scripts/raster.ipynb + requirements.txt — raster preprocessing notebook.
FILE: avalanchers_evaluation_example.py (77 lines, untracked) — evaluation-path example:
  outlines_path data/outline/polygon_10721.shp, outlines_padding 1000,
  release_area_fraction 0.25, sim_model "curvilinear" → create/run → post_process → evaluate
  → print(evaluation.chamfer).
```

## 9. Important Classes and Functions

### Rust

**`compute_core::ComputeOrchestrator`** (`crates/compute_core/src/lib.rs:470`) — owns all wgpu
state; the only type that dispatches compute work.
- `new_with_gpu(Option<String>) -> Self` — adapter pinning by name/PCI id; fallback chain.
- `run_analyze_terrain(&SimSettings, &Dem)` — picks AnalyzeTerrain (models 0/2) vs
  AnalyzeTerrainCurvilinear (model 1).
- `run_compute_release_areas(&SimSettings, roi: &[bool]) -> Result<u32>` — returns particle count.
- `run_initialize_particles(...)` — allocates model-specific buffers; dispatches
  InitializeParticles (+ optional relax).
- `prepare_compute_particles` / `prepare_mpm` / `prepare_mpmdac` — all delegate to private
  `prepare_simulation`, which selects the shader triple by `sim_model`.
- `step_terrain_following` / `step_curvilinear` / `step_mpmdac(steps) -> SimInfo` — all delegate
  to private `step_simulation` (dispatch order in section 4.2; model-2 GridMass snapshot).
- `run_compute_particles` / `run_mpm` / `run_mpmdac` — batch loops (`batch_compute_steps` per
  submit, early exit on `SIM_STOPPED`). `run_sim` dispatches by `sim_model` (0/1/2, else error).
- `evaluate_gpu(&SimSettings) -> MassMovementEvaluation` — Evaluate* + chamfer JFA + beeline.
- `read_buffer::<T>(BufferName)` / `read_texture` / `write_buffer` / `add_buffer[_with_data]`.
- Callers: `simulation::Simulation` (everything), `wasm_bindings::renderer` (buffers + device),
  `render_core/examples/live_sim.rs`.

**`compute_core::settings::{SimSettings, Settings, SimModel, FrictionModel, ConstitutiveModel,
CrownLineMethod, SimFlags}`** — see sections 4/7. `SimSettings::from_settings(&Settings, &Dem)`
is the patch→uniform merge; `as_bytes()` is what gets uploaded — layout contract with
`utils.wgsl` is load-bearing.

**`simulation::Simulation`** (`crates/simulation/src/lib.rs:156`) — see section 8.4 for the
method list. State-machine invariants are enforced with `bail!`; `gpu_cache` invalidation is
automatic but only at stage boundaries.

**`simulation::release_estimation::{detect_crown_line, crown_line_from_particle_counts,
detect_crown_line_by_gradient_step,
estimate_release_areas(_with_crown), ReleaseEstimationConfig, ReleaseEstimate}`** — pure-CPU
crown-line detection + chamfer-weighted fill. Called from `Simulation::load_release_areas`.

**`compute_core::buffers::GpuResources`** — buffer/texture registry; `prepare_buffer_contents`
handles the f32↔quantized-u32 encoding depending on float32-atomic support.

**`compute_core::shaders::{ShaderName, load_shader_source, ComputeShaderConfig,
create_shader_configs}`** — shader registry/preprocessor. Adding a shader = add `.wgsl` file +
variant in `define_shaders!` + binding list in `create_shader_configs` (+ entry into
`SimulationPipelines` if it joins the per-step loop).

**`data_processor::create_sim_settings_and_dem(&Settings) -> (SimSettings, Dem, Vec<bool>)`** —
the single funnel from user settings to simulation inputs (async; network only for outline-only
configs). Callers: `Simulation::load_data`, `wasm_bindings` (bytes variants).

**`data_processor::output::Output`** — Zarr writer (`add_new_site`, `add_new_scenario`,
`add_new_run`, `write_flow_fields`, `write_particle_position`). Callers: `Simulation::save`.

**`render_core::{GpuContext, Renderer, TerrainRenderer, ParticleRenderer, OrbitCamera,
OverlayRange}`** — windowing-agnostic scene renderer; used by all four examples and the wasm
RenderView.

### Python / JS

**`avalanchers.PySimulation`** (pyo3) — see section 8.8. The analytical tests additionally use
`sim.grid_mass` / `sim.grid_momentum` (raw quantized buffers!) and
`sim.particles_position_xy` / `particles_velocity_xy` / `particles_mass`.

**`avalanchers.plot2d / plot3d / plot_overview / plot_dem / calculate_dice /
AvalanchersOutput`** — pure-Python layers; see sections 8.10.

**`WasmSimulation`** (JS) — see section 8.7; `render_frame(steps)` is the rAF entry the
`LiveViewer` drives.

### Key WGSL functions (physics cores)

- `transfer_p2g` (transfer_p2g.wgsl) — quantized B-spline scatter; used by p2g.wgsl.
- `transfer_p2g_mpmdac` (p2g_mpmdac.wgsl:48-219) — model-2 full constitutive update + scatter.
- `basal_friction_acceleration` (friction.wgsl) — model-2 basal friction dispatch (ids 0-5).
- `mu_inertial`, `strain_rate_tensor`, `mat2_deviatoric`, `frobenius_norm` (constitutive.wgsl).
- `acceleration_by_normal_friction` / `acceleration_by_drag_friction` (compute_particles.wgsl
  334-413) — model-0 friction.
- `calculate_weight` / `calculate_weight_gradient` (utils.wgsl) — quadratic B-spline + physical
  gradient used by every P2G/G2P.

## 10. Dependency Relationships

```
avalanchers_example.py / avalanchers_ana1_tests.py / .zcode/diag_*.py
      │ python
      ▼
python_module.avalanchers ──► avalanchers._avalanchers (crates/python_bindings)
      │                              │
      │                              ▼
      │                        simulation::Simulation ──► data_processor (load/save)
      │                              │                        │
      │                              ▼                        ▼
      │                        compute_core ◄──────── compute_core::dem
      │                              │
      ▼                              ▼
frontend/js/*.js ──► frontend/js/pkg/avalanchers.js (crates/wasm_bindings)
      │                              │
      ▼                              ├──► simulation, data_processor, compute_core
LiveViewer (viewer.js) ──► WasmSimulation.render_frame ──► wasm_bindings::renderer::RenderView
                                 │
                                 ▼
                           render_core::Renderer (terrain/particles/colorbar)
```

Shader include graph (`// import x.wgsl;`):

```
utils.wgsl ◄── (almost everything)
random.wgsl ◄── initialize_particles.wgsl, compute_particles.wgsl
transfer_p2g.wgsl ◄── p2g.wgsl, test_transfer.wgsl
transfer_g2p.wgsl ◄── g2p.wgsl, test_transfer.wgsl
transfer_g2p_affine.wgsl ◄── g2p_mpmdac.wgsl
friction.wgsl ◄── grid_physics_mpmdac.wgsl
constitutive.wgsl ◄── p2g_mpmdac.wgsl
relax_particles.wgsl ◄── relax_clear_grid.wgsl, relax_build_grid.wgsl (alias modules)
```

Settings flag ↔ shader coupling: `SimSettings.flags` bits are produced by `SimFlags::new`
(Rust) and consumed as raw bits in WGSL; `enable_*` keys in `Settings` map to bits in
`Simulation::apply_data`. `sim_model` is read in shaders as `sim_settings.model_type`
(e.g. p2g.wgsl line 20 skips P2G for model 0 without particle_interaction).

## 11. Tests

| Suite | Location | Run | Covers |
|---|---|---|---|
| Rust unit + GPU | `#[cfg(test)]` in compute_core/src/lib.rs (2046-3242), settings.rs, dem.rs, etc. | `cargo test -p compute_core` (needs an adapter; CI uses software Vulkan) | shader transfer/transforms/sampling/utils round trips, evaluate GPU/CPU parity, center-of-mass (incl. serpentine timing), chamfer/beeline, settings parsing |
| Simulation | inline tests in simulation/src/lib.rs (1891-3600) | `cargo test -p simulation` | lifecycle ordering, gpu_cache reset (`test_automatic_gpu_cache_reset`), release estimation end-to-end, MPMDAC runs; `GITHUB_ACTIONS` skips GPU-heavy asserts; `#[ignore]` diagnostics (`test_mpmdac_diagnostic_avawog`) |
| data_processor | tests/caaml_parser_tests.rs + inline | `cargo test -p data_processor` | CAAML parsing, EsriGrid, tile manager (`SWISSALTI_TEST_TILES`), wiremock-based |
| render_core | tests/headless_render.rs | `cargo test -p render_core` | offscreen render pixels (skips w/o GPU); live_sim.rs has inline unit tests (testcase discovery, in-place reload, egui sliders) |
| cli | inline tests main.rs (406-475) | `cargo test -p cli` | `cli_args_override_settings_file` |
| wasm | tests/bindings.rs (node), bindings_gpu.rs (browser) | `wasm-pack test --headless --chrome` (in crates/wasm_bindings) | WasmSettings, GPU smoke (skipped on CI) |
| Python | python_module/tests/{test_bindings,test_io,test_run}.py | `pytest python_module` (after `maturin develop`) | binding smoke, run, Zarr layout/values |
| Analytical | avalanchers_ana1_tests.py | `python avalanchers_ana1_tests.py all` (needs `AVAFRAME_HOME`) | similarity solution + dam break vs model 1 (curvilinear/Coulomb) |
| Verification plots | verification/run_energy_line_test.py, run_dam_break_test.py, run_rotation_test.py, run_similarity_solution_test.py | `python verification/run_<test>.py` (repo data only, no AvaFrame checkout) | energy line (COM path vs α line, avaParabola), dam break (grid fields vs Faccanoni-Mangeney, avaDamBreak), rotation test (rotated inputs vs reference, avaTripleBowl), similarity solution (parabolic heap vs Hutter 1993, avaSimilaritySol) |
| Bench | simulation/benches/sim_benchmark.rs | `cargo bench -p simulation` | full-run throughput |

Notable gaps: `python_module/tests/test_verification.py` is empty; the analytical tests are not
wired into CI; no test pins `utils.wgsl` quantization factors to the Python constants (only the
runtime warning inside `check_grid_mass_consistency`).

## 12. External Integrations

- **AvaFrame** (test data + analytical reference): `data/avaframe/*` PNG DEMs (EUPL-1.2);
  `avalanchers_ana1_tests.py` reads `.asc` DEMs from a **local AvaFrame checkout**
  (`AVAFRAME_HOME`, default `C:/git/AvaFrame/avaframe`) — an out-of-repo data dependency.
- **swissALTI3D / swisstopo** tile service: fetched by `data_processor::tile_manager` when an
  outline (no DEM) is given; cached in `dtm_cache.zarr`. **BEV Austria** DTM links also present.
- **Shapefiles**: ESRI polygon `.shp` outlines (`outlines_path` + `outlines_padding`) →
  rasterized ROI (data/outline/polygon_10720|10721.shp).
- **CAAMLv6** bulletins: parser exists (data_processor::caaml_parser); no production caller.
- **Zarr v3** stores: written natively (zarrs), written from the browser via the minimal
  `zarr_writer`, read back by Python (zarr+xarray) and by the JS frontend (blosc/zstd decoded
  in WASM).
- **PyPI**: package `avalanchers` (release.yaml publishes wheels on `v*` tags).
- **GitHub Pages** (gro2mi.github.io/avalanchers) and an **SFTP mirror** (secrets [REDACTED]).
- **ffmpeg** (optional, external): `video_export.rs` pipes raw frames into an ffmpeg process.
- **GPU drivers / WebGPU**: wgpu backends Vulkan/Metal/DX12/GL; software rendering via Mesa
  llvmpipe in CI.

## 13. MODIFICATION NOTES

1. **The Rust↔WGSL struct layout contract is the #1 hazard.** `SimSettings` (settings.rs:70-109),
   `SimInfo` (lib.rs:187), `TimestepDataAoS` (lib.rs:221), `AtomicValues`, `EvaluationResult`,
   `CenterOfMassResult`, `RelaxParams`, `ChamferParams` are `#[repr(C)]` PODs whose field order
   and size must match their WGSL twins in `shaders/utils.wgsl` / the individual shaders. Change
   one side → change both, and remember the WGSL names differ (`num_steps`/`model_type`/
   `snow_density`/`slab_thickness`/`min_elevation`/`grid_shape` vec2/`world_size` vec2).
2. **Quantization factors live in three places**: `shaders/utils.wgsl` (source of truth),
   `data_processor::buffers::prepare_buffer_contents` (CPU-side encode/decode), and
   `avalanchers_ana1_tests.py` (`MASS_FACTOR`/`MOMENTUM_FACTOR`, with a >1% consistency
   warning). Changing them breaks old cached stores and the analytical tests.
3. **Edit only the top section of a `.wgsl` file.** Everything after the first
   `// BEGIN x.wgsl` block is machine-written stale import expansion (regenerated on the next
   native debug run). Files `relax_clear_grid.wgsl`/`relax_build_grid.wgsl` are alias stubs —
   edit `relax_particles.wgsl` instead.
4. **Shader hot-reload only in native debug.** Release/wasm builds `include_str!` at compile
   time — a shader edit requires a rebuild. Also, `load_shader_source` leaks expanded sources
   (`Box::leak`): they are process-lifetime constants, keyed by (name, float32-atomic) — fine,
   but don't call it dynamically per-frame.
5. **Dual atomics paths**: shaders contain paired lines marked `// atomic_float` and
   `// no_atomic_float`; the preprocessor picks one at pipeline creation based on
   `has_float32_atomic`. New grid accumulation code must provide BOTH variants or it will
   silently use the quantized path.
6. **Model-specific buffer allocation**: `run_initialize_particles` allocates different particle
   buffers per `sim_model`; `step_simulation` errors if the prepared model doesn't match. A new
   model needs: shader triple in `prepare_simulation` (lib.rs:1536), a `prepare_*`/`step_*`/
   `run_*` wrapper trio, a branch in `Simulation::run_n_steps` + `run`/`compute_particles`, and
   buffer allocation in `run_initialize_particles`.
7. **MPMDAC invariant**: every step begins with `copy_buffer_to_buffer(GridMass →
   GridMassPrevious)` because `p2g_mpmdac` derives particle depth from the previous grid mass
   (lib.rs:1678-1706). Don't reorder the dispatch or skip the copy.
8. **`deny_unknown_fields`** on `Settings`: adding a JSON key is a breaking change for every
   stored settings file unless it's an `Option` with a serde default — unknown keys hard-error.
   The `settings_vals_*.json` files still carry the removed `slab_thickness` key and fail today.
9. **gpu_cache is memoization only**: `Simulation` invalidates it at stage boundaries
   (`reset_simulation_result`/`reset_all`). If you add a mutation path, invalidate or you'll
   serve stale readbacks. Never mutate `gpu_cache` fields directly from new code.
10. **Renderers share the sim's device/queue/buffers** (live_sim clones handles; RenderView
    clones buffers). Re-creating buffers mid-run (e.g. re-`prepare`) breaks those bindings —
    that's why `Simulation::compute_particles` exists (steps without reallocating).
11. **Async surface**: all Simulation/Orchestrator methods are `async` (wgpu callbacks +
    readback polling). Python wraps them with `block_on` — never call Python-bound methods from
    within another async context. `create()` builds its own tokio current-thread runtime.
12. **Performance-sensitive dispatch sizing**: `dispatch_number_workgroups_*` and
    `max_particles` are derived once in `new_with_gpu` from device limits; `batch_compute_steps`
    exists because oversized submissions can trigger OS GPU resets (see SimulationLoadResult doc).
13. **Buffer size math**: `CenterOfMass` = (max_steps + 1) slots; `TimestepData` = 3 × max_steps
    AoS slots (96 B each). Raising `max_steps` linearly raises GPU memory; `prepare_simulation`
    checks overflow.
14. **`Settings` serde enums accept string OR int** (untagged) — keep both `from_int` and
    `FromStr` in sync when adding variants, plus the CLI parsers in cli/main.rs, the egui combo
    in live_sim.rs, the `#simModelDropdown` in frontend/index.html + main.js, and
    `Overlay::from_name` in wasm_bindings/renderer.rs where applicable.
15. **wgsl entry point = file stem**; `ComputeShaderConfig` resolves `binding_names` strings
    against `BufferName::FromStr`/`TextureName::FromStr` — a typo in a binding list panics at
    `create_resources`, not at first dispatch.
16. **Cross-platform guards**: `Output`/`capture.rs` are `#[cfg(not(target_arch = "wasm32"))]`;
    `wasm_bindings/renderer.rs` is wasm-only. `ruzstd` exists because zstd-sys can't cross-compile
    to wasm32 — don't "simplify" it to the zstd crate.
17. **Tests require a GPU adapter**; CI provides software Vulkan. If your change makes tests
    device-capability-dependent, guard like the existing `has_float32_atomic` tests do.
18. **`avalanchers.code-workspace` references sibling repos** (`../avalanchers_evaluation`,
    `../ISeeSnow`, `../AvaFrame`) — `avalanchers_ana1_tests.py` assumes the AvaFrame checkout
    exists at `C:/git/AvaFrame/avaframe` (override with `AVAFRAME_HOME`).
19. **Generated artifacts not to hand-edit**: `frontend/js/pkg/*` (wasm-pack output, gitignored),
    `Cargo.lock`, shader `// BEGIN/END` regions, `crates/compute_core/shader_report*.html`.
20. **Python numpy getters require C-contiguous arrays** (`set_dem` etc. call `as_slice()` and
    error otherwise) — the documented remedy is `.copy()`.

## 14. Known Uncertainties

- `UNCERTAIN: evaluate_mass_movement_points.wgsl` — min/max elevation atomics are seeded by
  `evaluate_gpu` but the shader never updates the elevation values (only indices, and only when
  elevation equals the seeded sentinel); min/max-elevation outputs look work-in-progress.
- `UNCERTAIN: data_processor::caaml_parser` — no production caller found in the workspace; seems
  prepared for future bulletin-driven features.
- `UNCERTAIN: friction_model ids 4 (voellmy with cohesion) and 5 (µ(I)) in friction.wgsl` —
  id 4 is a commented stub; `FrictionModel` enum only exposes 0-3, so 4/5 are currently
  unreachable from settings (reachable in tests/shaders only).
- `UNCERTAIN: settings_vals_all_pra.json / settings_vals_example.json` — use the legacy key
  `slab_thickness` which `Settings` (`deny_unknown_fields`) rejects; stale examples.
- `UNCERTAIN: frontend/js/pkg/.gitignore` — the artifacts are present on disk but gitignored
  (`*`), so deployments rely on the CI build step; a stale local pkg can mask wasm-binding
  drift after Rust API changes (README explicitly warns to re-run wasm-pack).
- `UNCERTAIN: g2p.wgsl binding 11 (particles_affine_matrix)` is declared but unused (future APIC
  work); large commented friction block kept as reference.
- `UNCERTAIN: p2g.wgsl affine momentum path` is commented out — model 0/1 use PIC-only transfer.
- `UNCERTAIN: data/snow, data/wind, avalanche_outlines_switzerland_2018` directories exist
  locally (mostly gitignored) with research notebooks; their scripts are not part of CI.
- The exact behavior of `Simulation::load_release_areas` when both `release_areas_path` and
  `release_area_fraction` are set follows the priority encoded there (uploaded array → texture →
  GPU thresholds → crown-line estimation); re-check `crates/simulation/src/lib.rs:1186` before
  changing that precedence.

## 15. AI Working Guide

**When adding/changing a simulation setting:**
1. `crates/compute_core/src/settings.rs` — add the field to `Settings` (Option) AND
   `SimSettings` (POD, correct position) + default in `new()` + hash in the manual `Hash` impl +
   merge in `from_settings`.
2. `crates/compute_core/src/shaders/utils.wgsl` — add the field to `struct SimSettings` at the
   SAME position (WGSL name may differ).
3. Use it in the relevant `.wgsl` (top section only).
4. If CLI-relevant: `crates/cli/src/main.rs` (Args + apply_overrides). Frontend:
   `frontend/index.html` + `frontend/js/main.js getSettings()`. egui: live_sim.rs panel.
5. Run: `cargo test -p compute_core -p simulation` and `cargo clippy -- -D warnings`.

**When changing physics of a model:**
- Model 0: `compute_particles.wgsl` (friction/integration) + `grid_physics.wgsl` (pressure).
- Model 1: `grid_physics_curvilinear.wgsl` (friction/integration on grid) + `g2p.wgsl`
  (advection); terrain metrics from `analyze_terrain_curvilinear.wgsl`.
- Model 2: FIRST read `docs/mpmdac.md`; then `p2g_mpmdac.wgsl` (constitutive + force scatter),
  `grid_physics_mpmdac.wgsl` (gravity/friction/CFL), `g2p_mpmdac.wgsl` (APIC + advection),
  helpers in `constitutive.wgsl` / `friction.wgsl`.
- Validate physics with `python avalanchers_ana1_tests.py all` (model 1) and the in-crate GPU
  tests; keep `utils.wgsl` quantization budgets in mind (comments explain i32/u32 limits).

**When adding a new shader:**
1. Create `crates/compute_core/src/shaders/<name>.wgsl` (top = source of truth; start with
   `// import utils.wgsl;` if you need settings/index helpers).
2. Add the variant to `define_shaders!` in `crates/compute_core/src/shaders.rs` (entry point =
   file stem).
3. Declare bindings in `create_shader_configs` (binding 0 = SimSettings uniform, names must
   parse as `BufferName`/`TextureName`).
4. Dispatch via `ComputeOrchestrator::run_shader(name, x, y, z)` or join `SimulationPipelines`.

**When adding a GPU buffer:**
1. `BufferName` variant in `crates/compute_core/src/buffers.rs` (+ snake_case name).
2. Size/creation in `create_buffers_and_texture_descriptions` (or the stage that owns it, like
   `prepare_simulation` / `run_initialize_particles`).
3. Bind in the shader config; add the `// atomic_float` + `// no_atomic_float` dual lines for
   float accumulations.

**When changing the Python API:**
1. `crates/python_bindings/src/lib.rs` (pyclass/pymethods) → `maturin develop`.
2. Wrappers/plotting in `python_module/avalanchers/__init__.py`.
3. Tests: `pytest python_module` (test_bindings asserts the getter surface).

**When changing the JS API:**
1. `crates/wasm_bindings/src/lib.rs` (+ renderer.rs) → re-run
   `wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --release`.
2. Consumers in `frontend/js/*.js` (`main.js`, `viewer.js`, `sources.js`, `plot.js`).
3. `node --check` runs in CI; wasm tests via `wasm-pack test --headless --chrome`.

**When touching output formats:**
- Writer: `crates/data_processor/src/output.rs` (native) and `zarr_writer.rs` (wasm path via
  `Simulation::export_zarr_entries`).
- Readers: `python_module/avalanchers/output.py`, `frontend/js/sources.js` (expects `dem`/`x`/
  `y` at site level, `release_area` per scenario), `python_module/tests/test_io.py` pins the
  format (`avalanchers_format "0.1.0"`, site/scenario hashing in `Simulation::site_name`/
  `scenario_name`).

**How to run things quickly:**
- Native sim: `cargo run -p cli -- settings.json`
- GUI: `cargo run -p render_core --example live_sim`
- Headless snapshot: `LIVE_SIM_SNAPSHOT=out.png cargo run -p render_core --example live_sim -- settings.json`
- Python: `maturin develop && python avalanchers_example.py`
- Web: `wasm-pack build … --release && python frontend/dev_server.py` → https://localhost/
- Full checks: `cargo fmt && cargo clippy -- -D warnings && cargo test -p compute_core -p
  data_processor -p simulation -p cli -p render_core && ruff check ./python_module && pytest python_module`



