# Dam-break analysis: front lag and rear thickness spike

This note analyses two symptoms in the curvilinear dam-break benchmark
(Faccanoni & Mangeney 2012, Test 2, case 1.2: dry inclined bed, Coulomb
friction, 22° slope, 21° bed friction angle, h0 = 1 m, 5 m cells):

- the numerical front lags the analytical solution in flow thickness, velocity
  and `hu`;
- the flow thickness shows an unphysical jump at the rear of the release.

There are **two distinct solver problems**, plus limitations in the analytical
comparison. The rear spike is largely caused by irreversible particle stopping;
the front lag is primarily a wet–dry resolution/transfer problem.

No solver changes were made during this analysis.

## 1. Rear thickness spike: particles freeze permanently

In [G2P](../crates/compute_core/src/shaders/g2p.wgsl), a particle whose
interpolated speed is below `velocity_threshold` is marked `PARTICLE_STOPPED`.
Subsequent steps skip its motion, but
[P2G](../crates/compute_core/src/shaders/p2g.wgsl) (`p2g_2d`) still
deposits its mass and stored momentum every step.

Controlled runs (default settings, ≈20 s):

| Setting | Mass stationary from the first step onward | Rear thickness peak |
|---|---:|---:|
| Default threshold (0.01 m/s) | **4.7%** (145 of 3072 particles) | ≈0.63 m |
| Threshold disabled | **0%** | ≈0.37 m |

All immobilized particles start in the upstream 7 m of the release
(x ≈ −120 … −112.6 m relative to the dam front). There, the pressure force
initially opposes downhill gravity, and Coulomb friction can legitimately bring
the material temporarily to rest. **Turning temporary rest into permanent
immobilization is the bug.** As neighbouring material leaves, the frozen mass
remains behind as a sharp remnant.

**Proposed fix:** distinguish *resting* from *permanently stopped*. Resting
particles must continue to evaluate forces and reactivate when the driving force
exceeds static friction. A velocity threshold alone must not remove particles
from the dynamics.

This revisits the stopping policy previously deferred in
[the curvilinear model review](curvilinear_model_review.md); it has not been
changed.

## 2. Front lag: the thin rarefaction tail is truncated

At ≈20 s in the existing benchmark:

| Quantity | Position |
|---|---:|
| Analytical dry front | **146.4 m** |
| Analytical `h = 0.001 m` contour | **141.1 m** |
| Numerical `h = 0.001 m` contour | **105 m** |

This is therefore **not** merely a thickness cutoff compared against a
mathematical zero-depth front.

Profile near the front (≈20 s):

| Position | Numerical h | Analytical h | Numerical u | Analytical u |
|---:|---:|---:|---:|---:|
| 50 m | 0.286 m | 0.328 m | 6.11 m/s | 6.26 m/s |
| 75 m | 0.192 m | 0.180 m | 7.00 m/s | 7.15 m/s |
| 100 m | 0.0836 m | 0.0760 m | 8.51 m/s | 8.05 m/s |
| 105 m | 0.0139 m | 0.0605 m | 8.96 m/s | 8.23 m/s |
| 110 m | **0** | **0.0468 m** | 0 | 8.41 m/s |
| 120 m | **0** | **0.0246 m** | 0 | 8.76 m/s |

The model represents mass with particles scattered through a quadratic B-spline
kernel. Near vacuum, insufficient particle support truncates the tail. APIC
preserves affine velocity fields in well-supported regions but **does not
provide an exact wet–dry front treatment**.

The velocity is not uniformly too small: at 105 m it is **8.96 m/s versus
8.23 m/s** analytically. The apparent velocity lag is mainly the prematurely
terminated support, with some local overshoot.

### Resolution evidence

Matched synthetic plane and release, permanent freezing disabled, CFL 0.1:

| Grid spacing | Front-contour lag at ≈20 s |
|---:|---:|
| 5 m | **38.3 m** |
| 2.5 m | **22.1 m** |
| 1.25 m | **10.0 m** |

Increasing particles per cell from 4 to 16 at 5 m spacing moved the
`h = 0.001 m` contour only from 105 m to 110 m. **Grid resolution matters much
more than particle count alone.**

**Proposed fixes:**

- Establish front-position convergence tests before tuning pressure.
- Improve particle coverage of thin regions using conservative
  redistribution/resampling.
- Preserve total mass and momentum; do not manufacture a thin layer in dry
  cells.
- Report multiple thickness contours and tail mass, not only bulk L2 errors.

## 3. Centered pressure and unlimited APIC poorly control sharp gradients

The [grid pressure](../crates/compute_core/src/shaders/grid_physics_curvilinear.wgsl)
uses a centered thickness gradient. The
[APIC gather](../crates/compute_core/src/shaders/transfer_g2p_affine.wgsl)
reconstructs an affine field without a limiter.

Consequences:

- A centered gradient cannot detect a perfectly alternating nodal thickness
  perturbation: neighbouring values cancel.
- Particle-density noise can therefore persist as cell-scale thickness
  oscillations.
- Unlimited affine reconstruction can overshoot near poorly supported edges.
- There is no explicit positivity-preserving, entropy-aware wet–dry Riemann
  treatment.

**Qualification:** on this uniform plane, the centered pressure force can be
written as a conservative pairwise flux. The issue is **not simply that
pressure is nonconservative**; it is the absence of robust
discontinuity/wet–dry handling and compatible transport.

This dry-bed dam break primarily tests a **rarefaction into vacuum**, not a
hydraulic shock. It cannot by itself establish whether shocks are handled
correctly.

**Proposed fix:** introduce a compatible conservative pressure/transport
treatment with positivity and bounded reconstruction. A finite-volume HLL/HLLE
type scheme is a strong option for robust shallow-water behaviour. Simply adding
numerical viscosity to the current gradient may suppress oscillations while
worsening front lag. Any hybrid must avoid advecting mass twice (on the grid and
again through particles).

## 4. Other problems that obscure diagnosis

- **Global stopping:** the 40-step "no new cells" rule in
  [update_sim_info.wgsl](../crates/compute_core/src/shaders/update_sim_info.wgsl)
  stopped a halved-CFL run at **1.55 s** (and a run without particle
  relaxation at 2.49 s) despite ongoing motion. A fixed step-count occupancy
  window is not a physical stopping criterion and makes results depend on CFL.
- **Initialization alignment:**
  [particle initialization](../crates/compute_core/src/shaders/initialize_particles.wgsl)
  places particles at `(cell + r − 0.5) * cell_size`, i.e. half a cell upstream
  of the cell convention used elsewhere, while the benchmark measures the
  release edges from cell centers rather than faces. Fix both together; this
  explains small alignment errors, not the ≈40 m lag.
- **Rear analytical curve:** the
  [analytical helper](../verification/dam_break_test.py) implements the
  downstream fan and extends uniform thickness upstream. Clipping at a nominal
  rear edge does **not** supply the missing rear rarefaction, so the red curve
  is not a valid full finite-column rear reference. The finite column also
  interacts with the front fan after roughly `L / (2 c) ≈ 20.6 s`.
- **Velocity plots:** the comparison plots use speed magnitude rather than
  signed downhill velocity, hiding upstream motion at the rear.
- **Quantization:** the initial grid mass error was only ≈0.00035%, and the
  mass quantization floor corresponds to ≈2e-5 m thickness, far below the
  missing analytical depths. Quantization is not the main explanation.

## Recommended order

1. **Correct diagnostics and initialization:** signed `u` and `hu`, a valid
   reference domain including the rear rarefaction, cell-face release
   geometry, front and tail-mass metrics.
2. **Replace irreversible resting and occupancy-based global stopping.**
3. **Add resolution and particle-layout convergence tests** for front position.
4. **Improve compatible pressure/transfer treatment:** checkerboard
   suppression, bounded reconstruction and wet–dry handling.
5. **Add a wet-bed dam break or hydraulic-jump test** to verify genuine shock
   propagation and conservation.
