# Curvilinear avalanche model review

Date: 2026-10-31

## Summary

The late-time spreading slowdown and grid-orientation dependence have different
causes. The pressure sign is correct, but its geometry is inconsistent. PIC
transfer dissipates most of the spreading motion at late times.

This review used the current shader sources, the September verification report,
and controlled GPU runs through the workspace's Python bindings. No solver
changes were made during the review.

## 1. PIC damping explains the late-time spreading slowdown

[G2P replaces particle velocity with grid-interpolated velocity every
step](../crates/compute_core/src/shaders/g2p.wgsl#L93), without FLIP or APIC
preservation. [P2G deposits only particle translational
momentum](../crates/compute_core/src/shaders/transfer_p2g.wgsl), so the round trip
smooths velocity variations repeatedly.

At approximately 20 s in the similarity test, a reconstruction of the transfer
and grid update measured the following mass-normalized downslope spreading-energy
rates:

| Contribution | Rate [m2/s3] |
| --- | ---: |
| PIC loss | -0.134 |
| Pressure work | +0.129 |

These contributions almost cancel. The reconstructed next-step particle
velocities differed from the actual GPU result by at most approximately
0.0024 m/s, supporting the operator-level diagnosis without claiming an exact
CPU/GPU match.

Increasing CFL from 0.05 to 0.1, reducing transfer frequency, lowered the final
relative thickness L2 error from 0.614 to 0.393. This is diagnostic evidence,
not a recommended fix: timestep changes also affect other numerical errors.

## 2. Pressure uses inconsistent thickness and distance scaling

[Local normal thickness includes the surface
Jacobian](../crates/compute_core/src/shaders/grid_physics_curvilinear.wgsl#L74):

```text
h = mass / (density * cell_size^2 * J)
```

However, [get_h omits
J](../crates/compute_core/src/shaders/grid_physics_curvilinear.wgsl#L159).
Consequently, neighboring values used in the pressure gradient represent
projected-area thickness `J*h`, rather than normal thickness `h`.

The [gradient denominator is also inverted with respect to physical
distance](../crates/compute_core/src/shaders/grid_physics_curvilinear.wgsl#L95):
dividing by `cell_size / l_x` multiplies the horizontal derivative by `l_x`.
For an axis-aligned planar slope, differentiation along the surface should
instead divide the horizontal derivative by `l_x`.

For the 35-degree axis-aligned similarity-test plane, `l_x = J = sec(35 deg)`
and `l_y = 1`. Relative to normal-thickness surface derivatives, the combined
scaling errors amplify pressure acceleration by approximately:

| Direction | Amplification |
| --- | ---: |
| Downslope | `J * l_x^2`, approximately 1.82 |
| Across-slope | `J`, approximately 1.22 |

On varying terrain, differentiating `J*h` can also introduce a gradient caused
by geometry rather than thickness. Correct each neighbor's thickness using its
own geometry.

The pressure sign itself is consistent: positive normal-load magnitude
multiplies the thickness gradient, and that quantity is subtracted from the
velocity update. Simply strengthening pressure would therefore not correct the
equations or resolve PIC damping.

## 3. Non-orthogonal velocity components are treated as Cartesian

[Terrain analysis computes gravity independently along normalized coordinate
tangents](../crates/compute_core/src/shaders/analyze_terrain_curvilinear.wgsl#L54):

```text
g_x = -g * B_x / l_x
g_y = -g * B_y / l_y
```

[Particle motion interprets the stored velocity as scaled contravariant
components](../crates/compute_core/src/shaders/g2p.wgsl#L95):

```text
horizontal_velocity = (u / l_x, v / l_y)
```

For terrain `z = B(x, y)`, the surface metric is:

```text
G = [[1 + B_x^2, B_x*B_y],
     [B_x*B_y,   1 + B_y^2]]
```

When both terrain derivatives are nonzero, the coordinate tangents are not
orthogonal. Independent component scaling and Euclidean speed calculations omit
the cross term. Projections of gravity onto normalized tangents are not generally
the coefficients needed to reconstruct acceleration in that basis.

This affects gravity, pressure, friction-speed evaluation, and velocity-dependent
diagnostics. A fix needs one consistent velocity representation and the complete
metric, not just a changed pressure stencil.

### Controlled GPU reproduction

An identical 35-degree planar slope was tested at two grid orientations, using
zero friction, disabled curvature, and particles in the interior of a uniform
release:

| Grid orientation | First-step physical speed [m/s] |
| --- | ---: |
| 0 degrees | 1.365 |
| 45 degrees | 1.634 |

Physical speed was reconstructed from horizontal motion and the terrain slope,
not from the shader's Euclidean component norm. The approximately 20% difference
appears in the first step and isolates a solver error from raster-rotation
interpolation effects.

## 4. Low-speed particles are permanently frozen

[The low-speed branch sets
PARTICLE_STOPPED](../crates/compute_core/src/shaders/g2p.wgsl#L141).
[Subsequent G2P updates skip those particles
permanently](../crates/compute_core/src/shaders/g2p.wgsl#L55), rather than
allowing pressure or gravity to reactivate them.

P2G does not check this state, so frozen particles still deposit mass and their
stored momentum. This is not equivalent to temporarily resting material.

### Controlled GPU reproduction

A frictionless flat parabolic heap was tested with different velocity thresholds:

- At threshold 0.1 m/s, spreading effectively stalled. The simulation stopped
  at step 40 with the no-new-cells flag.
- At threshold 0, the heap continued spreading through step 100.

Changing the threshold from 0.01 to 0 in the default similarity test did not
change its late-time errors materially. Permanent freezing is a real issue, but
is not the principal cause of that test's slowdown.

## 5. Occupancy-based stopping can terminate active flow

[The global update stops the simulation if fewer than four newly occupied cells
were counted over 40
steps](../crates/compute_core/src/shaders/update_sim_info.wgsl#L13).
This does not establish that velocity, pressure forces, or redistribution within
already occupied cells have ceased.

Additionally, [grid invocations increment the shared new-cell counter without an
atomic operation](../crates/compute_core/src/shaders/grid_physics_curvilinear.wgsl#L79).
Concurrent writes can lose increments, making the stopping signal unreliable.

## 6. Verification results and reporting issues

The current code reproduced the late-time errors in the
[September verification report](../verification/outputs/all_tests/curvilinear_20260922_101330/report.md):

| Test | Relative thickness L2 near 5 s | Relative thickness L2 near 20 s |
| --- | ---: | ---: |
| Similarity solution | 0.068 | 0.614 |
| Dam break | 0.136 | 0.342 |

In the similarity test at approximately 20.1 s:

| Quantity | Simulation | Analytical reference |
| --- | ---: | ---: |
| RMS horizontal downslope width [m] | 40.10 | 62.78 |
| RMS across-slope width [m] | 40.25 | 45.25 |
| Thickness-weighted mean absolute across-slope velocity [m/s] | 0.328 | 0.667 |

### Report status does not indicate numerical acceptance

[The runner labels a test "ok" when it does not raise an
exception](../verification/run_all_tests.py#L267). It does not apply quantitative
acceptance thresholds to the reported errors.

### Similarity mass warning is misleading

[The similarity consistency check receives first_h instead of
first_mass](../verification/run_similarity_solution_test.py#L229).
It compares summed thickness against total particle mass. The resulting 99.98%
warning is not evidence that the quantization factors are incorrect.

### Time and field sampling are misaligned

[The simulation dispatches grid physics and G2P before updating
SimInfo](../crates/compute_core/src/lib.rs#L1717). In
[update_sim_info](../crates/compute_core/src/shaders/update_sim_info.wgsl#L34),
the next timestep is calculated and assigned before advancing elapsed time.
Elapsed time therefore advances by the newly calculated timestep, not the
timestep just executed.

For the controlled planar test, the first update used approximately 0.2426 s,
while the reported timestep was approximately 0.0486 s.

Grid mass and momentum snapshots are also deposited before the particle update.
They should not be interpreted as post-update fields at the reported particle
time without accounting for this staging.

## Recommended correction order

1. Establish a consistent velocity representation using the full surface metric.
   Wire gravity, pressure, friction, advection, curvature, and diagnostics to
   that convention.
2. Correct pressure thickness decoding and physical-gradient geometry.
3. Implement genuine APIC or FLIP on both transfer directions. Merely mixing
   old particle velocity with PIC velocity is not a true FLIP update.
4. Replace irreversible low-speed freezing with an explicit resting/reactivation
   policy, and replace occupancy-only global stopping with a physically
   meaningful criterion. Make shared counters atomic.
5. Correct timestep accounting and field sampling, repair the mass check, and
   add numerical acceptance thresholds.
6. Re-run dam break, similarity, energy-line, and rotation verification.
   Include controlled rotated-plane, flat-heap, and transfer-preservation
   regression tests.

A pressure-only patch will not resolve the combined problems. Correcting the
currently excessive pressure while retaining PIC damping may initially make the
spreading slowdown worse, so geometry and transfer behavior must be validated
together.

## Staged implementation results

### Step 1: full-metric velocity formulation

The solver now stores horizontal contravariant velocities. Gravity and pressure
use the inverse surface metric; friction and peak-speed diagnostics use physical
surface speed. Curvature includes the horizontal geometric acceleration as well
as the normal-load correction. Particle xyz output reconstructs the tangent
vertical velocity. Analytical readers convert horizontal components to
surface-distance components.

The pressure-distance scaling change is part of this metric correction; neighbor
normal-thickness decoding remains for step 2.

Validation:

- 14 rotated-plane GPU cases passed (seven orientations, with and without
  Coulomb friction).
- Rust curvilinear vertical-velocity readback test passed.
- First-step planar physical speeds: 1.365041 m/s at 0 degrees and
  1.365089 m/s at 45 degrees, approximately 0.0035% apart, versus approximately
  20% before the correction.
- Similarity relative thickness L2: 0.0702 near 5 s and 0.7340 near 20 s.
  Correcting excessive pressure without fixing PIC damping worsens spreading,
  as anticipated above.
- Dam break relative thickness L2: 0.1406 near 5 s. The run then terminated
  early at approximately 5.36 s; no valid 20 s result exists for this stage.
  Resting and global stopping are still pending.

These are intermediate results, not acceptance of the complete solver.

### Step 2: normal-thickness pressure

Each pressure-stencil neighbor now divides deposited mass by its own surface
Jacobian. The inverse-metric gradient acts on normal thickness, and disabling
particle interaction disables the pressure contribution.

- All 17 GPU regressions passed, including three comparisons against the
  complete discrete pressure operator on rotated planes.
- Similarity relative thickness L2: 0.0815 near 5 s and 0.8070 near 20 s.
- Dam break relative thickness L2: 0.1406 near 5 s and 0.3528 near 20 s.
  This stage reached 20 s.

The pressure geometry is now consistent, but reduced pressure exposes the
remaining PIC damping. Late-time accuracy is not yet acceptable.

### Step 3: APIC transfer in both directions

Curvilinear P2G now scatters affine momentum, and G2P gathers and stores the
affine matrix. The matrix represents the horizontal velocity gradient in cell
units. The existing PIC entry point and MPMDAC transfer convention are unchanged.

- All 17 metric and pressure GPU regressions passed.
- The Rust force-free APIC regression preserves an affine velocity field and
  its matrix through the actual GPU scatter/gather.
- Similarity relative thickness L2: 0.0605 near 5 s and 0.0947 near 20 s,
  versus 0.614 before any corrections (approximately 85% lower late-time error).
- Dam break relative thickness L2: 0.0912 near 5 s and 0.0948 near 20 s,
  versus 0.342 before corrections (approximately 72% lower late-time error).

This is the major spreading improvement. Resting, stopping, and verification
time accounting still require separate corrections.

### Step 4: resting and stopping deferred

At the user's request, the current resting and stopping behavior is retained.
No changes are made to permanent low-speed particle freezing, the 40-step
new-cell stopping criterion, or the shared non-atomic new-cell counter.

These remain known limitations to revisit. The proposed alternative is to allow
resting particles to reactivate, apply friction consistently at low speeds, and
stop only at motionless equilibrium or the configured step limit. That proposal
has not been implemented or accepted. In particular, very slow spreading runs
can still freeze or terminate early under the retained policy.

### Step 5: verification time accounting and guardrails

Elapsed time now advances by the timestep actually executed, before the next
timestep is selected. `SimInfo.grid_elapsed_time` explicitly timestamps the
deposited mass/momentum snapshot before advection. Analytical field comparisons
use that timestamp. Already-stopped batches preserve their last grid snapshot
instead of clearing it during unused queued steps.

The similarity consistency check now receives mass rather than thickness; the
false quantization warning is gone.

The aggregate runner distinguishes execution errors, numerical failures,
numerical passes, and other-model comparison runs. Initial regression guardrails
require:

- Grid-snapshot coverage through 20 s, or 10 s in quick mode.
- Final relative thickness L2 <= 0.15 and momentum L2 <= 0.25.
- Energy-line runout error <= 2% and velocity-height RMSE <= 3 m.
- Rotation runout difference <= 2% and Dice >= 0.85.

The report retains measured results on numerical failure, and the CLI exits
unsuccessfully. These guardrails are not physical model certification.

At corrected snapshot times:

| Test | Thickness L2 near 5 s | Thickness L2 near 20 s | Momentum L2 near 20 s |
| --- | ---: | ---: | ---: |
| Similarity | 0.0782 | 0.0776 | 0.0780 |
| Dam break | 0.0895 | 0.0903 | 0.1202 |

The existing stopping policy remains unchanged. The timestep floor and CFL
signal accumulation are also unchanged; they remain considerations for a later
timestep-control review.

Step-5 validation passed: 23 focused Python regressions, all 145 Rust core
tests, and 61 Rust simulation tests (two pre-existing ignored tests). The shared
shader `SimInfo` layout and its generated import copies are synchronized.

### Step 6: complete benchmark validation

The first full benchmark run passed similarity, dam break, and rotation, but
failed the energy-line guardrails. Investigation identified an analytical
assumption mismatch: the alpha-line reference assumes basal normal load
`m*g*cos(theta)`, whereas curvature-enabled Coulomb friction also includes
`m*v^2*kappa`. That additional load dissipates energy beyond the alpha-line
prediction.

The energy-line benchmark now explicitly disables the additional curvature
normal load. Geometric transport remains active: it is necessary to keep
horizontal velocities consistent with the curved surface, independent of the
normal-load flag. Two GPU regressions verify the geometric acceleration with
the load correction both enabled and disabled.

With the extra load enabled, the corrected model produces approximately
1151.3 m runout and fails this particular alpha-line reference. With the
reference-compatible load setting, runout is approximately 1214.2 m and all
four benchmark suites pass their guardrails. The failed comparison was not
resolved by relaxing thresholds.

| Metric | Original review baseline | Final |
| --- | ---: | ---: |
| Similarity thickness relative L2 near 20 s | 0.614 | 0.0776 |
| Dam-break thickness relative L2 near 20 s | 0.342 | 0.0903 |
| Controlled 0/45-degree plane speed discrepancy | approximately 20% | approximately 0.0035% |
| Minimum rotation Dice | 0.878 | 0.983 |
| Maximum rotation runout difference [m] | 4.61 | 6.44 |
| Energy-line runout error [m] | -7.46 | -1.83 |
| Energy-line velocity-height RMSE [m] | 1.60 | 1.00 |

Similarity and dam-break late-time thickness errors decreased by approximately
87% and 74%, respectively. Rotation footprint overlap improved substantially,
but the maximum runout difference did not improve; it remains approximately
1.55% of the reference runout, within the 2% guardrail.

The original energy-line run used a different curvature-load setting, so its
before/after numbers are not a matched-configuration accuracy comparison.

The [final report with plots and full tables](../verification/outputs/all_tests/curvilinear_20261031_metric_apic_final/report.md)
was generated with:

```text
.venv\Scripts\python.exe verification\run_all_tests.py --sim-model curvilinear --no-sequences --out-dir verification\outputs\all_tests\curvilinear_20261031_metric_apic_final
```

The report and plots are generated, untracked verification outputs; this summary
and all solver/test changes are committed. Resting/freezing and occupancy-based
stopping remain explicitly deferred at the user's request. Flat-heap regressions
exercise both continued spreading at zero threshold and the retained freezing
behavior at the standard threshold.

Final validation: all four numerical benchmark suites passed, all 40 Python
package tests passed (27 focused solver/report regressions), and
`cargo check --workspace` and Rust formatting checks passed. The earlier
step-5 full Rust run passed 145 core and 61 simulation tests.

### Follow-up: rotation without curvature load and animated results

The rotation benchmark now also sets `enable_curvature=False`, as requested.
Geometric transport is still included. All four benchmarks pass on the rerun;
rotation reference runout is 457.8 m, maximum runout difference is 7.84 m
(approximately 1.71%), and minimum Dice is 0.979.

The [updated report](../verification/outputs/all_tests/curvilinear_20261031_no_curvature_rotation_animations/report.md)
embeds the dam-break comparison GIF (21 frames) and similarity profile and
contour GIFs (20 frames each). Every animation frame was decoded successfully.
The existing aggregate runner generates these sequences by default; omit
`--no-sequences` when running it. Standalone runners generate them with
`--plot-sequence`.

## Curvature verification: analytical reference

The new curvature benchmark uses the DEM, release, Coulomb coefficient
`mu = 0.4`, and sampling setup of the energy-line test. It runs both values of
`enable_curvature`. Internal interaction and all other settings are kept the
same in those two runs.

### Derivation

Consider a point mass sliding downhill in increasing horizontal `x`, on
`z = B(x)`, from rest at `x0`. Define:

```text
p = B'(x)
J = sqrt(1 + p^2)
ds = J dx                         (surface arc length)
kappa = B'' / J^3                 (positive for the concave bed)
v = ds/dt                         (physical surface speed)
N/m = g/J + kappa*v^2             (basal normal load)
w = v^2/(2g)                     (kinetic height)
H = B + w                         (mechanical energy height)
```

Tangential gravity and Coulomb friction give:

```text
dv/dt = -g*p/J - mu*(g/J + kappa*v^2)
dH/ds = -mu*(1/J + 2*kappa*w)
dH/dx = -mu - 2*mu*B''/(1+p^2)*w
dw/dx + 2*mu*B''/(1+p^2)*w = -p-mu.
```

This is a linear equation for `w`, despite the velocity-squared normal load.
Its integrating factor, normalized to one at release, is:

```text
I(x) = exp(2*mu*(atan(p(x)) - atan(p(x0))))

w(x) = [w(x0) + integral(x0..x, I(q)*(-B'(q)-mu) dq)] / I(x)
H(x) = B(x) + w(x).
```

This is an analytical definite-integral solution, not an elementary straight
energy line and not a time-stepped reference simulation. The implementation
evaluates its smooth integral with 64-point Gauss-Legendre quadrature. An
independent high-accuracy ODE integration checks that evaluation.

For the upstream parabola:

```text
B(x) = z_vertex + (b/2)*(x-x_vertex)^2
p(x) = b*(x-x_vertex)
b = B'' > 0
```

The bed parameters are recovered from the actual DEM and verified against
every DEM cell, including its horizontal continuation. The maximum permitted
fit residual is 0.001 m.

For release from rest, motion starts only if `p(x0) < -mu`. The first
nontrivial stopping point `x_stop` solves:

```text
integral(x0..x_stop, I(q)*(-B'(q)-mu) dq) = 0.
```

The integrand changes sign at `p = -mu`; the zero after this point is bracketed
and solved. If the particle reaches the vertex with positive kinetic height,
the flat continuation has `w' = -mu`, so:

```text
x_stop = x_vertex + w(x_vertex)/mu.
horizontal runout = x_stop-x0
surface runout = integral(x0..x_stop, sqrt(1+p^2) dx).
```

The parabolic arc-length primitive is
`[p*sqrt(1+p^2) + asinh(p)]/(2*b)`; the flat contribution is added separately.
The reference therefore reports both lengths and does not confuse horizontal
distance with surface distance.

With the normal-load correction disabled, `I = 1` and:

```text
w(x) = B(x0)-B(x)-mu*(x-x0)
H(x) = B(x0)-mu*(x-x0).
```

This recovers the ordinary alpha line exactly. With curvature enabled and
`w > 0`, the additional dissipation makes the energy line bend below the
alpha line and shortens runout.

### Assumptions and acceptance

The derivation is exact for a point slider in contact with this concave bed,
with pure Coulomb friction, no entrainment, and no internal pressure.
The existing energy-line setup has a finite particle release and particle
interaction. Its mass-averaged path is consequently only approximated by a
slider released at the initial center of mass; finite-width, pressure, transfer,
and stopping effects are not hidden by fitting an energy offset.

A pressure-disabled finite-release probe exhibited unreliable stopping and
residual particle velocities under the retained policy. The benchmark therefore
keeps the original energy-line interaction setting rather than changing the
user-deferred stopping policy.

The new checks require both runout errors <= 2%, both energy-height RMSEs <= 3 m,
and final mass-weighted RMS speeds <= 0.1 m/s. Curvature must shorten simulated
runout, and the curvature-aware energy reference must at least halve the error
against the incorrect alpha line. Formula tests cover the alpha-line limit,
stopping on both curved and flat sections, invalid non-sliding releases, and
agreement with the independent ODE.

Run the standalone benchmark with:

```text
.venv\Scripts\python.exe verification\run_curvature_test.py
```

It also runs by default in the aggregate suite, or can be selected using
`verification\run_all_tests.py --tests curvature`. The report includes a plot
of both energy lines and residuals, and the console lists analytical horizontal
and surface runout alongside simulated runout.

### Measured verification results

At `mu = 0.4`, the fitted bed has `B'' = 0.00041360158 m^-1` and a maximum
DEM residual of `0.000163 m`. The initial center of mass is at approximately
`x0 = 2688.074 m`.

| Normal-load curvature | Analytical horizontal runout | Simulated horizontal runout | Runout error | Energy-height RMSE |
|---|---:|---:|---:|---:|
| Off | 1212.892 m | 1214.173 m | +1.281 m | 1.294 m |
| On | 1150.657 m | 1151.300 m | +0.643 m | 1.110 m |

Analytical surface runouts are 1316.477 m and 1253.430 m, respectively.
Curvature shortens horizontal runout by 62.234 m analytically and 62.872 m
in the simulation. The curvature-on energy error against the incorrect alpha
line is 9.858 m: using the derived reference reduces that error by about 89%.
Final mass-weighted RMS speeds are approximately 0.063 m/s in both cases.
Small run-to-run variations are expected from GPU transfers.

All five aggregate benchmarks and all 48 Python tests pass with the new test
included. The tests also
explicitly reject an energy reference that cannot distinguish the curvature
correction from the alpha line. Other simulation models or friction laws retain
comparison-mode behavior, with this curvature-specific benchmark explicitly
skipped rather than treated as a numerical failure.

The generated report and plot are in
`verification/outputs/all_tests/curvature_20261031/report.md`; the complete
five-benchmark report is in
`verification/outputs/all_tests/curvilinear_20261031_with_curvature_test/report.md`.
