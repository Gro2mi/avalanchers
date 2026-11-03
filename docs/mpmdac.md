# The MPMDAC Model (MPM Depth-Averaged Curvilinear, `sim_model = 2`)

This document describes the third simulation model of *avalanchers*, **MPMDAC**
(MPM Depth-Averaged Curvilinear): a **2D terrain-following, depth-averaged,
compressible Material Point Method (MPM)** for snow avalanches with an
**elastic predictor–corrector constitutive model** (Drucker–Prager or µ(I)
yield), elastic volumetric response and plastic compaction. It runs entirely
on the GPU via WebGPU compute shaders, like the two existing models.

| model id | name | short | one-line description |
|---|---|---|---|
| 0 | `TerrainFollowing` | `"terrain-following"`, `"t"` | per-particle 3D sliding block with optional grid-based lateral pressure |
| 1 | `Curvilinear` | `"curvilinear"`, `"c"` | depth-averaged MPM in curvilinear terrain coordinates (metric terms `l_x`, `l_y`, `J`), plain PIC transfer, algebraic Voellmy friction |
| **2** | **`MpmDaC`** | **`"mpmdac"`, `"mpm-dac"`** | **compressible depth-averaged MPM in terrain-following horizontal coordinates with a per-particle elastic predictor–corrector stress law (Drucker–Prager or µ(I) yield), volumetric elasticity with plastic compaction, and APIC transfer** |

Selected via `settings.json` (`"sim_model": "mpmdac"`), the CLI
(`--sim_model mpmdac`), the Python API (the same settings key), or the model
combo box in `live_sim`.

---

## 1. Physical formulation

### 1.1 Geometry and kinematics

The snow is treated as a shallow layer of density `ρ` (`density`) and depth
`h(x, y)` moving over the bed `z = b(x, y)` given by the DEM. The flow state is
the depth-averaged **horizontal** velocity `u = (u, v)`; no vertical velocity
component exists (unlike model 0), and advection uses plain Cartesian
coordinates `x += u·dt` (unlike model 1, which divides by the metric
coefficients).

All bed-parallel quantities are derived from the DEM surface normal `n`
(precomputed once per run by the shared terrain-analysis shader
`analyze_terrain.wgsl`, the same pass model 0 uses):

- surface area factor `J = 1 / n_z` (bed surface area per horizontal area, ≥ 1),
- bed depth used everywhere: `h = m / (ρ · Δ² · J)` for a cell column of mass
  `m` and cell size `Δ`,
- gravity projected onto the tangent plane, horizontal components
  `g_t = g · n_z · (n_x, n_y)` with magnitude `g·sinθ·cosθ`,
- effective bed-normal acceleration `g_perp = g · n_z = g·cosθ`, optionally
  corrected for centrifugal release on convex beds (section 1.4).

### 1.2 Momentum balance (as discretized)

Each grid node receives three acceleration contributions per timestep:

```
u_node ← u_node + dt · ( g_t + f_int / m_node )     (explicit Euler)
```

then basal friction is applied against the proposed flow direction (section
1.4). The contributions are:

1. **Gravitational driving** `g_t` from the terrain slope (section 1.1).
2. **Internal stress divergence** `f_int` — the MPM internal force accumulated
   from per-particle stresses (section 1.3). Where the flow depth varies, the
   hydrostatic part of `σ` produces the familiar `−ρ g h ∇h` pressure driving;
   where the flow shears or compresses, the plastic part resists it.
3. **Basal friction** — drag between flow and bed, selecting the same friction
   laws as model 0 (section 1.5).

Because internal stresses come from a *yield-limited* plastic law, the
active/passive lateral pressure behaviour that model 0 approximates with an
explicit earth-pressure coefficient `K(div u)` **emerges naturally** here; the
`enable_earth_pressure_coefficient` flag has no effect in this model.

### 1.3 Constitutive model — elastic predictor–corrector with Drucker–Prager or µ(I) yield

Stresses are evaluated **per particle** in `p2g_mpmdac.wgsl` from the APIC velocity
gradient, stored in the `ParticlesStress` buffer and evolved by an elastic
predictor–plastic corrector (radial return):

- the affine matrix `B` gathered in `g2p_mpmdac.wgsl` is the velocity gradient in
  *cell units*; the physical gradient is `∇u = B / Δ`,
- strain rate `ε̇ = ½ (∇u + ∇uᵀ)`, deviatoric part `ε̇_dev`,
- **predictor**: `τ_trial = τ_old + 2 G · dt · ε̇_dev` with the deviatoric shear
  modulus `G = shear_modulus` (Pa, default 5·10⁴ — large values approach the
  rigid-plastic limit),
- **yield** is checked on the equivalent shear stress `τ_eq = √(½ τ:τ)` with the
  shear rate `γ̇ = √(2 ε̇_dev:ε̇_dev)` (Jop et al. 2006); both reduce to `τ_xy` and
  `γ̇` in simple shear, so literature friction angles and µ(I) parameters apply
  directly. In these measures the elastic response is `τ_eq = G·γ`,
- **yield stress** `y`, selected by `constitutive_model`:
  - `0` / `"drucker-prager"` (default): pressure-dependent Mohr–Coulomb /
    Drucker–Prager `y = c + P̄·tan φ`,
  - `1` / `"mu-i"`: rate-dependent inertial rheology `y = c + μ(I)·P̄` with
    `μ(I) = μ0 + (μ2 − μ0) / (i0/I + 1)` and the inertial number
    `I = γ̇ · d / √(P̄/ρ)` (`grain_diameter = d`),
  - both optionally hardened: `y += H · γ_p_cum` with the linear isotropic
    hardening modulus `H = hardening_modulus` (Pa, 0 = off),
- **corrector** (radial return): if `τ_eq,trial > y`, scale
  `τ = τ_trial · y/τ_eq,trial` and accumulate the plastic shear strain
  `γ_p += (τ_eq,trial − y)/(G + H)`; otherwise the step was elastic and `τ`
  keeps its trial value (elastic unloading is possible),
- total stress `σ = τ − P̄·I` with lithostatic `P̄ = ½ ρ g_perp h_p`.

The stored state per particle is the symmetric deviatoric stress
`(τ_xx, τ_xy, τ_yy)` plus the cumulative plastic shear strain `γ_p_cum`, packed into
one `mat2x2` (`ParticlesStress`, zero-initialized). Because the stress is
carried, near-stopping material relaxes elastically instead of dropping
stresslessly to zero, and deposited snow retains a yield-limited residual
stress.

The Cauchy stress is depth-integrated to `S_p = h_p · σ_p` (N/m) and scattered
to the grid nodes as the MPM internal force

```
f_i = − Σ_p (m_p / ρ) · S_p · ∇w_ip          (m_p/ρ is the particle "volume")
```

using the analytic gradient of the quadratic B-spline weights
(`calculate_weight_gradient` in `utils.wgsl`, in 1/m). Note that the product
`A_p · h_p = m_p / ρ` is exactly the particle's column volume, so the force
needs no current-area bookkeeping.

Because the deviatoric stress saturates at a pressure-dependent yield stress,
the active/passive lateral pressure behaviour that model 0 approximates with an
explicit earth-pressure coefficient `K(div u)` **emerges naturally** here; the
`enable_earth_pressure_coefficient` flag has no effect in this model.

Implementation: `constitutive.wgsl` (`strain_rate_tensor`, `mat2_deviatoric`,
`frobenius_norm`, `equivalent_shear_stress`, `equivalent_shear_rate`, `mu_inertial`).

### 1.3.1 Volumetric response — compressibility and compaction

The material is compressible through a per-particle volumetric strain,
stored in `ParticlesVolumetricStrain` as `(ε_v,e, ε_v,p)` (elastic and
plastic compaction parts) and evolved in `p2g_mpmdac.wgsl`:

- **Strain increment** from the APIC divergence,
  `Δε_v = dt · tr(∇u)`, rate-limited to |Δε_v| ≤ 0.5 per step;
- **elastic volumetric pressure** (compression positive)
  `p_vol = −K · ε_v,e` with bulk modulus `K = bulk_modulus` (Pa);
- **one-sided plastic compaction**: when `p_vol` exceeds the compaction
  pressure `p_c = compaction_pressure`, the material densifies
  irreversibly — the elastic strain relaxes onto the plateau
  `ε_v,e = −p_c/K` and `ε_v,p` accumulates the difference. Dilation stays
  elastic, clamped at `ε_v,e ≤ 0.5`. `bulk_modulus = 0` switches the
  volumetric law off entirely (incompressible material, purely lithostatic
  pressure);
- **total mean pressure**: `P̄ = max(½ ρ g_perp h_p + p_vol, 0)` — this is
  the pressure that feeds both the stress tensor and the yield stress, so
  compressed flow pushes and yields harder, expanded flow softer;
- **density feedback**: compaction raises the particle density by the ratio
  `1 + ε_v,p`, which shrinks the particle volume used in the internal force
  scatter (`m_p / (ρ · density_ratio)`), keeping the depth-integrated force
  consistent with the densified column.

The CFL signal speed gains the volumetric wave speed
`√(K/ρ)` in `grid_physics_mpmdac.wgsl`, so a stiffer material automatically
reduces the adaptive timestep.

Compressibility makes the flow behave measurably differently from an
incompressible one: converging flow absorbs part of the compression by
densifying (thicker, denser, slightly slower piles instead of immediate
pressure-driven spreading).

### 1.4 Centrifugal curvature correction (optional, `enable_curvature`)

With the curvature flag set (on by default), the normal acceleration gains a
flow-direction curvature term as in model 0:

```
g_perp = max( g·n_z + |u|² · κ , 0.1 ) ,   κ = t̂ᵀ H t̂
```

where `t̂ = u/|u|` and `H` is the bed second-derivative (curvature) tensor
`(dxx, dxy, dyy)` from the shared `Curvature` texture. Convex beds (`κ < 0`)
lift the flow and reduce normal stress and friction; concave beds press it
down. The corrected `g_perp` enters the lithostatic pressure `P̄` only; basal
friction uses the plain `g·n_z`.

### 1.5 Basal friction

Basal friction is applied on the grid (in `grid_physics_mpmdac.wgsl`), not per
particle: if the friction deceleration `a_f · dt` would reverse the proposed
velocity, the velocity is zeroed, otherwise scaled by
`1 − a_f·dt/|u_proposed|`. The deceleration comes from the shared
`friction.wgsl` import and honors the full `friction_model` selection with
mass-per-area `ρ·h`:

| `friction_model` | law | deceleration |
|---|---|---|
| 0 | Coulomb | `μ · g_perp` |
| 1 | Voellmy | `μ · g_perp + g·|u|² / (ξ·h)` |
| 2 | Voellmy min shear | Voellmy `+ 70 Pa / (ρ h)` |
| 3 | SamosAT | velocity- and pressure-dependent saltation term |
| 5 | µ(I) | `µ(I) · g_perp` with `I = 5/2 · d·|u| / (h √(g_perp h))` (Bagnold profile; parameters `i0, mu0, mu2, grain_diameter`, checked on load when selected) |

`μ = friction_coefficient`, `ξ = drag_coefficient`.

---

## 2. The MPM timestep

One timestep = one iteration of the loop in `step_simulation`
(`compute_core/src/lib.rs`), all inside a single submitted command batch:

| stage | shader | dispatch | purpose |
|---|---|---|---|
| 0 | — (GPU copy) | — | snapshot `GridMass` → `GridMassPrevious` (MPMDAC model only) |
| 1 | `reset_grid` | 2D, full grid | zero mass, momentum, forces, peak trackers |
| 2 | `p2g_da` | 1D, per particle | scatter mass, APIC momentum, internal forces |
| 3 | `grid_physics_da` | 2D, per cell | gravity + internal force + basal friction → `grid_velocity`; peak/thickness bookkeeping |
| 4 | `g2p_da` | 1D, per particle | APIC velocity gather, `F` update, advection, stop conditions |
| 5 | `update_sim_info` | 1×1 | adaptive `dt`, stop flags, rolling new-cell window |

optional center-of-mass stages are appended and are model-agnostic.

### 2.1 Particle → grid (`p2g_da`)

Each particle scatters to the 3×3 node stencil around
`grid_pos = x/Δ − 0.5` (nodes sit at cell centres) with quadratic B-spline
weights `w`:

- **mass**: `m_p · w`, quantized to `u32` atomics (`MASS_FACTOR = 10`),
- **momentum**: APIC form `m_p · (v_p + B_p · d) · w` with node offset `d` in
  cell units, quantized to `i32` atomics (`MOMENTUM_FACTOR = 1e-2`),
- **internal force**: `−(m_p/ρ) · S_p · ∇w_ip`, quantized to `i32` atomics
  (`FORCE_FACTOR = 1e-3`).

Every atomic write exists twice in the source: a quantized integer variant
tagged `// no_atomic_float` and a native `atomic<f32>` variant tagged
`// atomic_float`; the shader loader activates the right one based on the
device's `SHADER_FLOAT32_ATOMIC` capability.

**Particle depth.** `h_p` — and with it the pressure `P̄` — is *not* derived
from the particle's own deformation state. It is gathered from the
**previous step's grid mass** with the same B-spline weights:

```
h_p = ( Σ_stencil w · m_node,prev ) / ( ρ · Δ² · J )
```

The mass snapshot (`GridMassPrevious`, copied by the GPU at the start of each
step) is required because the mass buffer is being concurrently accumulated in
this very dispatch. See section 4.2 for why the per-particle alternative
`h_p = m_p / (ρ·A0·J·det F)` was rejected.

### 2.2 Grid → particle (`g2p_da`) and advection

The particle gathers the APIC pair `(v_p, B_p)` from the node velocities
(`transfer_g2p_affine.wgsl` — note the gather function is named `transfer_g2p`
and returns a `G2PUpdate`). Then:

- **advection** `x ← x + v_p·dt`, elevation resampled from the DEM,
- **stopping**: `|v_p|² < velocity_threshold²` sets `PARTICLE_STOPPED` with the
  timestep in the lower bits; NaN and the 2-cell border produce the usual
  `OUT_OF_DEM_DATA` / `OUT_OF_BOUNDS` flags and feed `SimInfo`, which the batch
  loop uses for early termination (`SIM_STOPPED`).

The per-particle deviatoric stress and plastic strain (`ParticlesStress`) are
not touched here; they are read, updated and written back by the constitutive
model in `p2g_da` (section 1.3).

### 2.3 Adaptive timestep

`update_sim_info.wgsl` (shared with model 1) sets
`dt = max(0.01, cfl · Δ / (|v| + √(g·h)))` using the
`expected_max_velocity` accumulated in `grid_physics_da`. The
shallow-water wave speed `√(g·h)` is also the dominant plastic pressure-wave
speed of this model, so no additional CFL term is required.

---

## 3. What each piece of state means

| buffer | per | content in model 2 |
|---|---|---|
| `ParticlesPosition` | particle | horizontal position (m) |
| `ParticlesVelocity` | particle | horizontal velocity (m/s) |
| `ParticlesAffineMatrix` | particle | APIC velocity gradient `B`, cell units (rebuilt every step) |
| `ParticlesStress` | particle | deviatoric stress `(τ_xx, τ_xy, τ_yy)` + cumulative plastic strain `ε_p_cum` (evolved by the constitutive model in `p2g_da`) |
| `ParticlesState` | particle | bitflags + stop timestep (identical semantics to models 0/1) |
| `GridMass`, `GridMomentum` | node | quantized atomic accumulation, reset each step |
| `GridMassPrevious` | node | previous step's mass snapshot (depth source) |
| `GridForces` | node | accumulated internal force (N), zeroed by `reset_grid` for **all** models |
| `GridVelocity` | node | flow velocity after friction; sampled by `g2p_da` and rendering |

Outputs consumed by post-processing, evaluation, Zarr export and rendering
(`GridPeakFlowThickness`, `GridPeakVelocity`, center-of-mass slots,
`SimInfo` flags) are produced identically to the other models — the downstream
pipeline needs no changes per model.

---

## 4. Numerical safeguards and design decisions

### 4.1 Limits in the shaders

| safeguard | where | value | protects against |
|---|---|---|---|
| force acceleration cap | `grid_physics_da` | `|f_int/m| ≤ 5 g` | any residual stress/quantization pathology launching the flow |
| quantization clamps | `p2g_da` | clamp to ±i32 range before round | atomic overflow |
| `n_z` floor | `p2g_da`, `grid_physics_da` | `n_z ≥ 0.2` (θ ≤ ~78°) | near-vertical DEM cells |
| `g_perp` floor | `p2g_da` | `≥ 0.1 m/s²` | tension from the curvature correction |

### 4.2 Why depth does *not* come from `det(F)`

An earlier implementation computed `h_p = m_p / (ρ·A0·J·det F)`. It failed in
two distinct ways on real terrain (avaWog), both verified with per-step buffer
diagnostics (`test_mpmdac_diagnostic_avawog`):

1. **Explosion.** The APIC gradient only resolves cell-scale velocity
   variation. dt is CFL-tuned for the shallow-water wave, so on steep terrain
   `dt·∇u` exceeded 1, the explicit F-update overturned, det F collapsed — and
   the `max(det F, 0.01)` floor turned each collapsed particle into a ~100×
   pressure bomb (forces of ~10⁸ N, velocities of 400 m/s within 6 steps).
2. **Clumping, then bursts.** Sub-cell convergence is invisible to `det F`:
   particles bunched up with no restoring pressure, until accumulated
   compression hit the clamp band and released at once.

Gathering the depth from the previous step's grid mass fixes both: the mass
field *by construction* sees clumping (a clump raises the local column mass,
so its particles sit on higher depth and receive the spreading pressure
gradient), and it is bounded by mass conservation. Diagnostics show the
busiest node's mass oscillating (25–30 t, spread by pressure) instead of
growing monotonically to 79 t (clumped).

### 4.3 Quantization factors

`utils.wgsl` constants used by this model: `MASS_FACTOR = 10`
(u32 mass atomics), `MOMENTUM_FACTOR = 1e-2` (i32 momentum atomics),
`FORCE_FACTOR = 1e-3` (i32 force atomics, decode ×10³ → ±2.1·10¹² N/node
range; observed working values are ≤ ~5·10⁵ N). Without
`SHADER_FLOAT32_ATOMIC` (e.g. integrated GPUs) the sim automatically uses the
quantized paths.

---

## 5. Settings interface

All physics parameters live in the flat settings namespace shared by the other
models; nothing new was added to the GPU-side `SimSettings` layout.

| setting | used by model 2 as |
|---|---|
| `sim_model: "mpmdac"` (or `"mpm-dac"`, `2`) | model selection |
| `constitutive_model` (`"drucker-prager"`/`"dp"`, `"mu-i"`/`"mui"`) | deviatoric yield law (default Drucker–Prager) |
| `shear_modulus` (Pa) | elastic predictor stiffness `G` (default 5·10⁴; → rigid-plastic as it grows) |
| `hardening_modulus` (Pa) | linear isotropic hardening on accumulated plastic strain (default 0 = off) |
| `bulk_modulus` (Pa) | volumetric elasticity `K`; compressibility on by default at 2·10⁴ (0 = incompressible) |
| `compaction_pressure` (Pa) | plastic densification plateau `p_c` (default 2·10³) |
| `internal_friction_angle` (°) | Drucker–Prager yield `φ` (default 40°) |
| `n0` (Pa) | cohesion `c` (default 70) |
| `friction_model` + `friction_coefficient`, `drag_coefficient` | basal friction law (section 1.5) |
| `density`, `cell_size`, `released_particles_per_cell` | depth/mass conversion; particle footprint `A0 = Δ²/pps` |
| `cfl` | adaptive timestep |
| `velocity_threshold` | stopping threshold (grid and particle) |
| `enable_curvature` | centrifugal correction (default **on**) |
| `max_steps`, `batch_compute_steps` | run length |

Not used by this model: `enable_earth_pressure_coefficient` (emergent from the
plastic law), `enable_particle_interaction` (interaction *is* the method),
`enable_entrainment` (not implemented in any model yet).

---

## 6. Implementation map

| file | role |
|---|---|
| `crates/compute_core/src/shaders/p2g_mpmdac.wgsl` | particle→grid: mass, APIC momentum, Drucker–Prager forces |
| `crates/compute_core/src/shaders/grid_physics_mpmdac.wgsl` | grid update: gravity, stress divergence, basal friction, peaks |
| `crates/compute_core/src/shaders/g2p_mpmdac.wgsl` | grid→particle: APIC gather, advection, stopping |
| `crates/compute_core/src/shaders/constitutive.wgsl` | strain-rate, deviatoric and µ(I) rheology helpers (shared import) |
| `crates/compute_core/src/shaders/friction.wgsl` | basal friction laws (shared import) |
| `crates/compute_core/src/shaders/utils.wgsl` | weight gradients, `FORCE_FACTOR`, mat helpers |
| `crates/compute_core/src/shaders/reset_grid.wgsl` | additionally zeroes `GridForces` (all models) |
| `crates/compute_core/src/settings.rs` | `SimModel::DepthAveraged = 2` + parsing |
| `crates/compute_core/src/shaders.rs` | shader registry + bind group layouts for the new shaders |
| `crates/compute_core/src/buffers.rs` | `GridMassPrevious`, `ParticlesDeformationGradient` names; `GridForces` base allocation |
| `crates/compute_core/src/lib.rs` | dispatch arms (terrain analysis `0|2`, init buffers `2`, shader trio `2`), `prepare/step/run_mpmdac`, per-step mass snapshot copy |
| `crates/simulation/src/lib.rs` | `run_n_steps` arms, zero-filled `particles_velocity_z`, diagnostics tests |
| `crates/render_core/examples/live_sim.rs` | model combo box entry |

Frontends need no code changes: the CLI parses the new enum alias, and Python
passes `"sim_model": "mpmdac"` through serde. The web `frontend/`
dropdown was already out of sync with the backend model aliases before this
change and was left alone.

---

## 7. Validation

- `cargo test -p compute_core` (143 tests) creates every pipeline, which
  validates all WGSL through naga, and covers the shared transfer/evaluation
  shaders; `cargo test -p simulation` (48 tests) runs end-to-end sims of
  models 0 and 1 to guard against regressions from the shared-loop changes.
- `test_mpmdac_velocity_z_is_synthetic_zero` covers the model-2
  prepare path and the render buffer contract.
- Two `#[ignore]`d diagnostics print per-step buffer maxima:
  - `test_mpmdac_diagnostic_steps` — 6×6 slope, unit-style check;
  - `test_mpmdac_diagnostic_avawog` — the real avaWog DEM + release
    texture (489×554), printing `dt`, velocity, momentum/force quantons,
    peak node mass, per-cell particle occupancy and the particle bounding box
    each step. Run with
    `cargo test -p simulation --lib -- test_mpmdac_diagnostic_avawog --ignored --nocapture`.

Reference behaviour on avaWog (Voellmy μ = 0.2, ξ = 2000, compressible with
K = 2·10⁴ Pa, p_c = 2·10³ Pa): velocity rises to a ~18 m/s plateau, dt settles
near 0.08 s (shorter than the incompressible case because of the volumetric
wave speed), internal forces stay ≤ ~1·10⁵ N per node, particle occupancy is
bounded (compression packs particles tighter, as intended), no NaN/stop flags.
Setting `bulk_modulus: 0` recovers the incompressible variant with its
previously validated ~20–28 m/s plateau.

---

## 8. Known limitations and follow-ups

- **Entrainment** is not implemented (no model has it).
- **Depth is one step delayed** (previous-step mass). During very fast
  transients the pressure lags by one timestep; in steady flow this is
  irrelevant.
- **The new settings are not yet exported to the Zarr run metadata**
  (`constitutive_model`, `shear_modulus`, `hardening_modulus`), following the
  precedent of `internal_friction_angle` and friends which are also not
  exported.
- **The 0.01 s dt floor** (`update_sim_info`) bounds the simulated avalanche
  duration at `max_steps`; very long runouts on flat runout zones may hit the
  cap before all particles stop.
- **Basal friction is grid-side only**: a particle in a cell whose velocity was
  zeroed by friction stops on the grid but keeps its last particle velocity
  below the threshold (as in model 1).
