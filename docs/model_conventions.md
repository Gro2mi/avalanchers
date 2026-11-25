# Model Conventions

## DAMPM curvilinear velocity convention

The DAMPM curvilinear solver (`curvilinear`, `curvi`) stores horizontal velocities `(dx/dt, dy/dt)`. Gravity,
pressure and friction use the full surface metric, including its cross term.
APIC transfers preserve affine velocity variations between particles and grid.
Geometric transport along curved terrain is always included.
`enable_curvature` controls the additional centripetal basal normal load;
the alpha-line energy benchmark disables that load correction to match its
reference assumption.
`enable_weak_pressure` (default off) takes the hydrostatic pressure gradient
from the MPM weak form, `m_i grad h = -sum_p m_p h_p / 2 grad N_i(x_p)`, in an
extra particle pass after P2G instead of the centred difference of the nodal
thickness. The small mass of a thin front node then receives the full push of
the particles behind it; on the dam break benchmark the front moves about 5 m
closer to the analytic solution at 5 m cells.
Particle xyz velocities reconstruct the vertical component from the terrain
slope; xy velocities and grid momentum are horizontal components. Analytical
comparisons convert these to surface-distance velocities where needed.

## DAMPM terrain-following velocity convention

The DAMPM terrain-following model (`terrain-following`, `terrain`) stores velocities in the surface plane: the 3D
tangent velocity rotated into the horizontal plane by the rotation that takes the
bed normal to z. The norm of `particles_velocity` is therefore the true speed, but
unlike curvilinear velocities the components are not `(dx/dt, dy/dt)`. A particle
reaching a differently inclined bed loses the velocity component normal to it, so
flows are deflected by side walls instead of running up them. There is no z velocity:
`particles_velocity` is `(n, 2)` and `fetch_particles_velocity_z` returns an
error. The DAMPM global model (`global`) carries the full Cartesian vector instead, at a higher cost; see [the DAMPM global model](dampm_global.md).

## Simulation timing

`SimInfo.elapsed_time` is particle time after the executed step; `SimInfo.dt`
is the next timestep. `SimInfo.grid_elapsed_time` timestamps the deposited
grid mass/momentum snapshot before that step's advection. Verification uses
the snapshot time and applies numerical regression guardrails instead of
treating successful execution as numerical acceptance.

## Curvature benchmark

The curvature benchmark reuses the energy-line parabola and release, compares
curvature-on/off runout against an analytical integrating-factor solution, and
plots the curved energy line. Run it with
`.venv\Scripts\python.exe verification\run_curvature_test.py`, or use
`verification\run_all_tests.py --tests curvature`. It is included in the full
curvilinear/Coulomb verification suite; other comparison configurations explicitly
skip this benchmark.

## Center-of-mass travel metrics

The center-of-mass shader accumulates horizontal `travel_length_2d`, `travel_length_3d`,
`travel_vertical_drop` and travel angle during each simulation step. These values live in the existing `SimInfo`
buffer and arrive with its normal batch readback: no post-simulation shader or additional metrics buffer is needed.
Both lengths sum the distances between consecutive valid center-of-mass records, in metres; the 3D length includes
elevation changes, not just the net vertical drop. Invalid slots remain NaN and are skipped across gaps.
Without valid records all metrics are NaN; a stationary trajectory has zero lengths and angle. The travel angle
still uses horizontal length and the first-to-last valid elevation drop (negative for a net rise).
`run_summary` (including Python) uses the already-read simulation info, without a trajectory readback.
Saves read the center-of-mass records only for coordinates and timestamps and add the DEM origin on the CPU.
All four metrics are saved in native and browser Zarr stores; appending
to an older native store adds missing length columns with NaN for prior runs; existing `travel_length` columns
are not renamed automatically. New summaries and saves use `travel_length_2d` only for horizontal length.
GPU parity tests require exact coordinates
and timestamps, each length within `1e-4 + 1e-4 * abs(CPU length)` and angle within `0.001` degrees.

## Benchmarks

`cargo bench -p simulation`

Computation time of the DAMPM terrain-following, global and curvilinear models on the
avaWog test case, a full run until the simulation stops by itself:

`cargo bench -p simulation --bench model_benchmark`

Append `-- dampm-terrain-following` (or `dampm-global-cartesian`, `dampm-curvilinear`) to run one model only.
Criterion reports the change against the previous run and writes HTML reports to
`target/criterion`.
