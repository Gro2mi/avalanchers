# Changelog

All notable changes to this project might be documented in this file.

## [Unreleased]

### Fixed
- The flow front no longer stalls behind a steep wall on GPUs with float atomics when particles
  are light (many particles per cell, thin release margins). The tile-sorted P2G still rounded
  every contribution to 0.1 kg and dropped those that rounded to zero, and the grid treated nodes
  below 10 g as empty, so particles at the margin gathered zero velocity from their outer stencil
  nodes and lost a few percent of their speed every step. Float-atomic devices now use the direct
  P2G and treat only massless nodes as empty. The similarity solution test of the 3D model at 64
  particles per cell drops from 15.2 % to 5.1 % thickness error at 20 s, and centre-of-mass
  runouts no longer depend on the particle count.
- The Linux wheel imports on systems with an older glibc than the CI runner, e.g. Ubuntu 24.04. It
  is built in the manylinux_2_28 container and runs from glibc 2.28 on.
- Errors of the swissALTI3D tile manager now say what went wrong instead of only "Failed to create
  tile manager", also when the tiles were fetched for an existing manager.
- The simulation runs in Chrome again. The center of mass shader failed WGSL uniformity validation
  there, which Firefox does not enforce.
- The biggest-blob center of mass no longer risks reading stale blob labels or masses. Its phases
  only synchronized workgroup memory, not the storage buffers they hand over.
- GPUs with float atomics no longer produce NaN velocities on empty grid nodes in the curvilinear,
  MPMDAC and sliding block models, and show the particle height jitter again.
- The curvilinear model no longer produces NaN when the flow reaches the outermost grid nodes on
  GPUs that evaluate 0/0 strictly.
- Repeated GPU evaluations no longer grow GPU memory until the driver reports `Out of Memory`.
  `evaluate` records its passes into one command encoder instead of about 30 submissions, and the
  chamfer buffers are allocated once per grid instead of on every call. The particle relaxation
  records its 30 iterations the same way and reuses its buffers while the grid and particle count
  stay the same.
- Creating a simulation or a headless render context no longer leaves 16 MB of GPU memory behind
  after it is dropped. Only the primary backends (Vulkan, Metal, DX12, WebGPU) are used; the unused
  OpenGL backend caused this.

### Changed
- The integer-atomic grid (WebGPU, or `integer_atomics`) sizes its fixed point units per run from
  the cell size and density instead of fixed 0.1 kg and 0.01 kg m/s: the u32 mass and i32 momentum
  ranges hold a 30 m deep flow over a cell at 100 m/s with a safety factor of 2. A particle then
  holds the same number of units at every cell size (about 1.2 million mass units for a 1 m release
  at 64 particles per cell); 5 m cells resolve 7.5e-5 kg and 0.015 kg m/s, and large cells no
  longer overflow the momentum range of fast, deep flows. Nodes lighter than one momentum unit per
  1 m/s count as empty, and the run logs the units a release particle holds. On the float GPU of
  this machine the integer grid now matches the float grid in all verification tests (similarity
  solution 3.5 % thickness error at 20 s on both).
- **Breaking:** `enable_particle_relaxation` is replaced by `particle_placement` (CLI
  `--particle-placement`): `random`, `relaxed`, `uniform-random` or `uniform-relaxed`. The uniform
  placements split every release cell into one equal-area stratum per particle and draw each
  particle in its own stratum, which removes most of the sampling noise of random seeding in the
  grid thickness; the relaxed ones run the relaxation afterwards. The default is now
  `uniform-relaxed` (before: random seeding with relaxation, now `relaxed`), so results change
  slightly. `enable_particle_relaxation: true` becomes `relaxed`, `false` becomes `random`. At 64
  particles per cell the thickness error of the similarity solution (3D) drops from 5.1 % to 3.5 %
  at 20 s, that of the dam break from 2.5 % to 2.0 %, and their profiles no longer wiggle; the
  energy line, curvature and rotation tests are unchanged. Crown line detection keeps `random`, its
  entry counts are tuned to it.
- **Breaking:** `slope_aspect` is now the direction the slope faces (downhill), as in GIS and
  avalanche bulletins: a north-facing slope is 0 degrees. It used to be the uphill bearing, 180
  degrees off. Flat cells stay -1.
- **Breaking:** `grid_mass` and `grid_momentum` now return f32 in kg and kg m/s on every GPU instead
  of raw quantized integers, so callers no longer divide by `MASS_FACTOR` and `MOMENTUM_FACTOR`.
- **Breaking:** `slab_thickness_factor` is now `release_thickness` in metres
  (`--release-thickness` in the CLI) and the old key is rejected. It is unset by default: rasters,
  arrays, stores and shapefile attributes keep their thickness, all other release areas get 1 m.
  If set, it replaces the thickness of every release cell, with a warning for sources that have
  their own; rasters and arrays used to ignore it. It must be above 0.01 m. Result stores keep it
  as `release_thickness`, NaN if unset.
- **Breaking:** `terrain-following` (model 0) is now a depth-averaged grid MPM like the
  curvilinear model, with gravity, hydrostatic pressure and basal friction on the grid and the
  curvature-corrected arc step on the particles, which snap onto the DEM. Velocities stay in the
  surface plane and are transferred as 2D vectors: the tangent velocity rotated into the horizontal
  plane, so their magnitude is the true speed but their components are not `dx/dt`. There is no z
  velocity: `fetch_particles_velocity_z` returns an error and Python `particles_velocity` is
  `(n, 2)` for this model. It runs about as fast as the curvilinear model.
- The particle arc step no longer adds the displacement twice per step.
- Curvilinear, terrain-following, 3D: basal friction now also acts below `velocity_threshold`, so
  Coulomb friction holds material at rest while the driving force is below yield, independent of
  the timestep.
- Curvilinear, terrain-following, 3D: slow particles no longer freeze permanently. They rest after
  20 consecutive steps below `velocity_threshold` and are reactivated once the grid speed exceeds
  twice the threshold. Resting particles keep depositing mass but do not move.
- Curvilinear, terrain-following, 3D: the simulation also stops (`AT_REST` flag) once 99% of the
  particles rest or are stopped. The 40-step no-new-cells criterion is unchanged.
- Curvilinear, terrain-following, 3D: the CFL timestep follows the current maximum of
  `v + sqrt(g h)` instead of the run's all-time maximum, so dt grows again once the flow slows down.
- Curvilinear, terrain-following, 3D: the CFL wave speed uses the bed-normal gravity `g cos(theta)`
  instead of `g`.
- Curvilinear, terrain-following, 3D: grid nodes whose 3x3 neighbourhood holds no mass skip the
  grid update.
- Terrain-following, 3D: particles stopped for good no longer deposit their last velocity.
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
- `release_max_elevation` now limits the release areas. It had no effect before, cells above it
  were released.
- MPMDAC: the plastic compaction strain had the wrong sign, so compaction enlarged the particle
  volume in the internal force instead of shrinking it. Peak speeds drop by about 10 % at the
  default `bulk_modulus`.
- Sliding block: the timestep follows the CFL condition. It stayed at the value of the first step,
  so particles could move several cells per step. Runs now take about three times as many steps,
  and the `particle-simulation` crown line detection is affected as well.
- Sliding block: the fixed velocity damping of 1 % per step is gone. It depended on the timestep and
  shortened the runout: the energy line benchmark is now 5 m from its reference, not 307 m.

### Added
- `integer_atomics` (CLI `--integer-atomics`, verification `run_all_tests.py --integer-atomics`)
  accumulates the grid with integer atomics also on GPUs with float atomics, to reproduce browser
  results or test the WebGPU path natively. Switching it on a prepared simulation rebuilds the
  shader pipelines and prepares the simulation again.
- The dam break verification test runs on a refined grid: `--refine N` (default 4, cell size
  1.25 m instead of the DEM's 5 m) interpolates the plane DEM and splits every release cell. At 64
  particles per cell the thickness error at 15 s drops from 7.4 % to 2.5 % and the front lag at
  h = 1 mm from 25 m to 7 m; the 5 m grid smeared the initial step over 3-4 cells.
- Every build names its commit: the version string written to result stores and shown by
  `cli --version` is `<version>+<short commit>`, with `.dirty` when the build sources (`crates/`,
  `Cargo.toml`, `Cargo.lock`) differed from the commit. New stores also get the full hash in the
  root attributes `avalanchers_git_commit`, `avalanchers_git_dirty` and
  `avalanchers_software_version`. Python: `avalanchers.build_info()`, `__git_commit__`,
  `__software_version__`, `__git_dirty__`. Builds without a git checkout take the commit from
  the environment variables `AVALANCHERS_GIT_COMMIT` and `AVALANCHERS_GIT_DIRTY`. Release builds
  track the dirty flag exactly; debug builds only update it on commits, checkouts and staging.
- Runs can carry a label that tells them apart within a scenario, e.g. a calibrated optimum and a
  predicted run: `Simulation::save_labeled` / `save_with_path_labeled`, and
  `PySimulation.save(path, label=...)` in Python. The scenario attribute `run_labels` lists the
  labels and the new `run_label` column indexes into it, like `software_version`; the empty label
  marks a run saved without one. `AvalanchersScenario.run_labels`, `.labels` and
  `.runs_labeled(label)` read them back. The store format stays 0.2.0: older stores have no
  `run_label` column, and their runs count as unlabeled.
- Result stores keep the outline: a scenario run from an outline gets an `outline` array (`uint8`,
  1 inside) and its `outline_hash`. A run with another outline is not appended to that scenario,
  and a scenario loaded from a store brings its outline back as the region of interest. DEM-only
  runs store none. The store format stays 0.2.0.
- A simulation created from an outline with a DEM fetched from swissALTI3D tiles names its site
  after the outline file (`polygon-10720_<hash>`) instead of `default-site_<hash>`.
- `save` returns the `store`, `site`, `scenario` and `run` it wrote to. Python also gets the
  `sim_info` and `adapter_info` getters and `run_summary()`, which gives the release volume and the
  center-of-mass travel length and angle without the grid readbacks of a save.
- `AVALANCHERS_TILES_OFFLINE=1` turns a missing swissALTI3D tile into a `tile_missing` error
  instead of a download, so many concurrent jobs only read a prefilled cache.
  `AVALANCHERS_TILE_DELAY_MS` pauses after each downloaded tile. Tiles are still downloaded one
  after another, and a request the server throttles (HTTP 429 or 503) is retried after its
  `Retry-After`, up to five times.
- `Simulation::release_area_terrain` and the Python getters `release_mean_slope_angle`,
  `release_max_slope_angle`, `release_mean_aspect` and `release_mean_elevation` give the mean
  terrain and the largest slope angle of the release cells.
  The aspect is a circular mean.
- `release_swe` (`--release-swe` in the CLI) sets the release by snow water equivalent in kg/m²
  instead of `release_thickness`; setting both is an error. It is the mass per horizontal area, so
  a release cell holds `release_swe * cell_size²` whatever its slope. It must be positive, and
  `release_swe / density` must be above 0.01 m. Result stores keep it as `release_swe`, NaN if unset.
- CLI flags `--release-max-elevation` and `--enable-particle-relaxation`.
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
- `3d-depth-averaged` (model 4, alias `3d`): the terrain-following model with a full 3D Cartesian velocity through P2G and G2P.
  Slower than `terrain-following` but keeps the z velocity and affine state on the particles.
- Optional particle elevation jitter in the renderer (live viewer key `J`).
- `particle_sort_interval` (CLI `--particle-sort-interval`, default 25): curvilinear,
  terrain-following and 3D P2G over particles sorted by 8x8-cell tiles, accumulating each workgroup in shared memory before the
  global atomics; 0 disables it. Results are identical to the direct P2G on the integer-atomic
  grid. Devices with float atomics do not sort and always use the direct P2G, which does not
  quantize.
- `momentum_stop_fraction` (CLI `--momentum-stop-fraction`, default 0 = off): stops a curvilinear,
  terrain-following or 3D run once the total grid momentum falls below this fraction of its peak (flag `LOW_MOMENTUM`).
- `reference_surface_length` (CLI `--reference-surface-length`, default 0 = off): gully
  confinement in the curvilinear, terrain-following and 3d-depth-averaged models. Bed inclined by beta against a Gaussian-smoothed reference surface of this
  width counts as a gully wall: the snow beside it presses it with its lateral stress, so the
  normal load of the basal friction grows by `1 + K tan^2 beta` (K = 1, or the cross-slope
  earth pressure coefficient when enabled). All other terms stay on the real bed. (Moving the
  metric, gravity and pressure onto the reference surface, as in Gray et al. 1999, made deposits
  creep on gorge floors below steep walls and dropped the centrifugal load of narrow bends.)
  The live viewer exposes the setting under "terrain & release" (0 = off).
  `verification/run_gully_test.py` checks the V-channel friction factor against the wedge limit
  (`--sim-model` selects the model). All grid models load the walls with the cross-slope earth
  pressure coefficient when it is enabled, otherwise with K = 1.
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
- Terrain-following and 3d-depth-averaged: the optional earth pressure coefficients (off by
  default) act on the pressure gradient and the gully wall load like in the curvilinear model.
  The helpers moved into the shared `earth_pressure.wgsl`. On the similarity solution with earth
  pressure the relative thickness error at 20 s is 0.075 (0.066 with K = 1).
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
