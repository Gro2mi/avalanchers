# Changelog

All notable changes to this project might be documented in this file.

## [Unreleased]

### Changed
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

### Added
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
- Curvilinear: `enable_earth_pressure_coefficient` (on by default) now applies the Savage-Hutter
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
