# Changelog

All notable changes to this project might be documented in this file.

## [Unreleased]

### Changed
- **Breaking:** `slab_thickness_factor` is now `release_thickness` in metres
  (`--release-thickness` in the CLI) and the old key is rejected. It is unset by default: rasters,
  arrays, stores and shapefile attributes keep their thickness, all other release areas get 1 m.
  If set, it replaces the thickness of every release cell, with a warning for sources that have
  their own; rasters and arrays used to ignore it. It must be above 0.01 m. Result stores keep it
  as `release_thickness`, NaN if unset.
- `terrain-following` (model 0) is now a depth-averaged grid MPM like the curvilinear
  model, formulated in 3D Cartesian coordinates: 3D APIC transfer, gravity, hydrostatic
  pressure and basal friction on the grid with bed-normal projection; particles keep the
  curvature-corrected arc step and snap onto the DEM.
- The particle arc step no longer adds the displacement twice per step.
- Curvilinear: basal friction now also acts below `velocity_threshold`, so Coulomb friction
  holds material at rest while the driving force is below yield, independent of the timestep.
- Curvilinear: slow particles no longer freeze permanently. They rest after 20 consecutive
  steps below `velocity_threshold` and are reactivated once the grid speed exceeds twice the
  threshold. Resting particles keep depositing mass but do not move.
- Curvilinear: the simulation also stops (`AT_REST` flag) once 99% of the particles rest or
  are stopped. The 40-step no-new-cells criterion is unchanged.
- Curvilinear: the CFL timestep follows the current maximum of `v + sqrt(g h)` instead of the
  run's all-time maximum, so dt grows again once the flow slows down.
- Curvilinear: the CFL wave speed uses the bed-normal gravity `g cos(theta)` instead of `g`.
- Curvilinear: grid nodes whose 3x3 neighbourhood holds no mass skip the grid update.
- P2G skips atomic adds of zero momentum, which removes most atomics of resting particles.
- Default `batch_compute_steps` is 50 (was 200).
- Default `cfl` is 0.9 (was 0.5).
- The no-new-cells stop window spans the steps the fastest material needs to travel
  `no_new_cells_travel` cells (CLI `--no-new-cells-travel`, default 65), i.e. round(65 / cfl)
  and at least 10 steps: 72 at CFL 0.9 instead of a fixed 40.
- Zarr store format 0.2.0, older stores are rejected on open. The native and browser writers
  now share one layout (`data_processor::store_schema`), so browser exports open everywhere.
  - `y` ascends with the rows. It was reversed against the data, so label based selection and
    plots in xarray were mirrored north-south. Row 0 of every grid is the south edge.
  - `x`/`y` (and `Dem::x`/`y`) are cell centers in float64. They spanned edge to edge, up to
    half a cell off.
  - Every setting is stored per run, named as in the settings (`mu` is now
    `friction_coefficient`, `xsi` is `drag_coefficient`), plus `release_volume_m3` (was a
    scenario attribute), `final_timestep`, `simulated_time`, `stop_flags`, `number_particles`,
    `created_at`, `software_version` and the per-step `time`.
  - Unwritten float values are NaN and integers 4294967295 instead of 0.
  - Scenarios have no fixed run or timestep capacity. Longer runs were silently cut off.
  - Run values share chunks of 1024 runs instead of one file each.
  - `run` and `timestep` have no coordinate arrays, `release_area` is float32, blosc uses
    zstd level 3 with byte shuffle.
  - CF: `Conventions`, valid `standard_name`s, `flag_values`/`flag_masks`, and a `spatial_ref`
    variable with `grid_mapping` when the CRS is known.
  - Saves are locked per store and survive crashes: arrays are written first and a run only
    counts once `number_of_runs` is replaced atomically.
- `travel_length` and `travel_angle` skip steps without a center of mass record. They were NaN
  and 0 for every run since slot 0 is unused, and earlier inflated by a jump to the DEM origin.
  They are NaN when the center of mass is disabled.
- Scenario names hash the release areas that were actually run. The hash covered only releases
  passed as arrays, so every release from a file got the same one. It uses the stable DEM
  hasher, so it no longer depends on the Rust version. Site hashes change with the coordinates.
- The center of mass records carry the simulated time, stored as `time` in the result stores.
- Holes in outline polygons are no longer part of the region of interest. They were filled.

### Added
- `release_areas_path` accepts polygon shapefiles (`.shp`, native and Python). All features are
  rasterized onto the DEM grid; holes stay free. As in AvaFrame, the thickness of each feature comes
  from its `thickness` or `d0` attribute, either all features or none must have one, and features
  must not overlap. Without these attributes the thickness is 1 m.
- `dem_path` and `release_areas_path` accept paths into a result store, such as
  `avalanchers.zarr/avaWog/avaWogreleaseTexture`, so the DEM and, if present, the release areas
  of an earlier run can be reused. Saving into the same store continues the scenario.
- `sim.write_asc(layer, path)` writes `peak_velocity`, `peak_flow_thickness` or `release_areas` as an
  ESRI ASCII grid, north row first and georeferenced by the DEM.
- Reusable initial state for back calibration (Rust `Simulation` and Python `PySimulation`):
  `store_initial_state()` prepares once and keeps the seeded, relaxed particle positions and
  masses on the GPU; `update_settings(dict)` changes settings without reloading the data;
  `restore_initial_state()` resets particle state and the peak fields for the next `run()`.
  Changed settings re-run only the preparation stage they affect: model or reference surface
  the full preparation, release thresholds, particles per cell or crown settings the release
  and seeding, density or a changed uniform `release_thickness` a mass rescale.
- `run()` no longer repeats terrain analysis, release areas and particle seeding when the
  particles are already initialized (after `prepare()` or `restore_initial_state()`).
- `sliding-block` (model 3): the previous per-particle terrain-following model.
- Optional particle elevation jitter in the renderer (live viewer key `J`).
- `particle_sort_interval` (CLI `--particle-sort-interval`, default 25): curvilinear P2G over
  particles sorted by 8x8-cell tiles, accumulating each workgroup in shared memory before the
  global atomics; 0 disables it. Results are identical to the direct P2G on the integer-atomic
  grid; on the float-atomic grid the shared sums keep the integer quantization.
- `momentum_stop_fraction` (CLI `--momentum-stop-fraction`, default 0 = off): stops a curvilinear
  run once the total grid momentum falls below this fraction of its peak (flag `LOW_MOMENTUM`).
- `reference_surface_length` (CLI `--reference-surface-length`, default 0 = off): curvilinear
  gully confinement. Bed inclined by beta against a Gaussian-smoothed reference surface of this
  width counts as a gully wall: the snow beside it presses it with its lateral stress, so the
  normal load of the basal friction grows by `1 + K tan^2 beta` (K = 1, or the cross-slope
  earth pressure coefficient when enabled). All other terms stay on the real bed. (Moving the
  metric, gravity and pressure onto the reference surface, as in Gray et al. 1999, made deposits
  creep on gorge floors below steep walls and dropped the centrifugal load of narrow bends.)
  The live viewer exposes the setting under "terrain & release" (0 = off).
  `verification/run_gully_test.py` checks the V-channel friction factor against the wedge limit.
- `enable_earth_pressure_coefficient` is off by default in all models (engine, live viewer, web
  frontend, Python defaults): Voellmy-type parameters are calibrated with hydrostatic pressure.
- Curvilinear: `enable_earth_pressure_coefficient` now applies the Savage-Hutter
  earth pressure coefficients of Hutter et al. (1993) to the hydrostatic pressure gradient:
  `k_x` along the fall line of the (reference) surface and `k_y` across it, each active or
  passive by the sign of the strain rate in its direction, from `internal_friction_angle` and
  `basal_friction_angle`. Previously the flag had no effect in this model. The strain rates use
  each node's own mass (the old `div_u` divided neighbour momentum by the centre mass) and turn
  one-sided at the flow edge. `run_similarity_solution_test.py --earth-pressure` compares it
  with the analytic solution; the dam break, similarity and gully tests pin the flag to match
  their references.
- Sliding block: `div_u` for the earth pressure coefficient uses each node's own mass (it divided
  neighbour momentum by the centre mass), turns one-sided at the flow edge and reads float-atomic
  momentum correctly.

## [0.2.0] - 2026-05-20

### Added
- Particle interactions with a grid based flow height and resulting pressure force
- Criterion benchmark
- Internal friction angle as threshold for pressure force
- samosAT friction model is useful now with particle interaction enabled
- ESRI ASCII support

## [0.1.0] - 2026-05-04

### Added
- **Core:** Block avalanche model supporting:
  - Curvature effects.
  - Friction models: Coulomb, Voellmy, and VoellmyMinShear.
- **Bindings:** Python bindings for integration into data science workflows.
- **Web:** WASM build and a browser-based frontend.
- **CLI:** Command-line interface for batch processing simulation files.
