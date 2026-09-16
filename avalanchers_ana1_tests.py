"""Analytical validation tests for avalanchers, adapted from AvaFrame ana1Tests.

Runs two analytical test cases against the avalanchers GPU simulation and
compares the results directly in this script (no result files are written;
the simulation is stepped with ``run_n_steps``, and the flow fields are
decoded from the simulation's own grid mass/momentum buffers - the exact
fields the grid physics pass consumed):

* ``simisol``  - Similarity solution of a granular avalanche spreading down an
  inclined plane (Hutter, Siegel, Savage; Acta Mechanica 1993). Adapted from
  ``avaframe/ana1Tests/simiSolTest.py``.

* ``dambreak`` - Dam break of a granular column over a dry inclined bed with
  Coulomb friction (Faccanoni & Mangeney 2012, Test 2, case 1.2). Adapted from
  ``avaframe/ana1Tests/damBreak.py``.

Error measures are ports of ``avaframe/ana1Tests/analysisTools.py``
(L2 / LMax norms, absolute and relative).

The test DEMs are read from a local AvaFrame checkout (default
``C:/git/AvaFrame/avaframe``, override with the ``AVAFRAME_HOME`` environment
variable):
    data/avaSimilaritySol/Inputs/DEM_IP_Topo.asc   (35 degree plane)
    data/avaDamBreak/Inputs/DEM_IP_Topo.asc        (22 degree plane)

Release thickness is generated here (parabolic heap / rectangular dam) and
handed to the simulation with ``set_release_areas``; avalanchers interprets
the release raster as per-cell snow thickness in meters.

Deviations from the AvaFrame reference configurations (documented where they
occur):
* dam break bed friction angle: AvaFrame uses delta = 21 deg (1 deg excess over
  the 22 deg slope). avalanchers' friction solver stops material whenever the
  friction deceleration over one time step exceeds the velocity, so a
  near-critical slope never starts to slide. We use delta = 12 deg (10 deg
  excess, the same excess as in the similarity solution test) so the case
  remains a valid Faccanoni-Mangeney solution while being resolvable by the
  time integration.

Usage:
    python avalanchers_ana1_tests.py simisol
    python avalanchers_ana1_tests.py dambreak
    python avalanchers_ana1_tests.py all
"""

import argparse
import logging
import math
import os
import pathlib
import sys

import numpy as np

log = logging.getLogger("ana1_tests")

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_AVAFRAME_HOME = pathlib.Path("C:/git/AvaFrame/avaframe")
OUTPUT_DIR = REPO_ROOT / "outputs" / "ana1_tests"


# ---------------------------------------------------------------------------
# ESRI ASCII raster IO
# ---------------------------------------------------------------------------

def read_asc(path):
    """Read an ESRI ASCII grid; returns (header dict, data array, north-first)."""
    header = {}
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                try:
                    float(parts[1])
                except ValueError:
                    header[parts[0].lower()] = parts[1]
                    continue
                header[parts[0].lower()] = float(parts[1])
            elif parts:
                rows.append([float(v) for v in parts])
    data = np.array(rows, dtype=np.float64)
    return header, data


def measured_plane_angle_deg(dem, cell_size):
    """Slope angle from the DEM (central differences along x, as the sim does)."""
    zx = (dem[:, 2:] - dem[:, :-2]) / (2.0 * cell_size)
    zy = (dem[2:, :] - dem[:-2, :]) / (2.0 * cell_size)
    slope = np.median(np.concatenate([np.abs(zx).ravel(), np.abs(zy).ravel()]))
    return math.degrees(math.atan(slope))


def jacobian(dem, cell_size):
    """Surface area Jacobian per cell (matches analyze_terrain_curvilinear)."""
    zx = np.zeros_like(dem)
    zy = np.zeros_like(dem)
    zx[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) / (2.0 * cell_size)
    zx[:, 0] = zx[:, 1]
    zx[:, -1] = zx[:, -2]
    zy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / (2.0 * cell_size)
    zy[0, :] = zy[1]
    zy[-1, :] = zy[-2]
    return np.sqrt(1.0 + zx * zx + zy * zy)


# ---------------------------------------------------------------------------
# Grid fields, read directly from the simulation's grid mass/momentum buffers
# ---------------------------------------------------------------------------

# p2g quantization constants - MUST match crates/compute_core/src/shaders/utils.wgsl.
# fields_from_grid cross-checks them against the particle total mass.
MASS_FACTOR = 1e1       # p2g stores round(mass * MASS_FACTOR) as u32
MOMENTUM_FACTOR = 1e2   # p2g stores round(mass * velocity * MOMENTUM_FACTOR) as i32


def fields_from_grid(sim, shape, cell_size, density, jac):
    """Decode the sim's grid mass/momentum buffers into thickness + velocity.

    These are exactly the fields the grid physics pass consumed this step
    (including p2g quantization and its boundary guards), not a python
    re-scattering of the particles. Returns flow thickness (normal to the
    surface, m) and depth averaged velocity (along-slope u, cross-slope v)
    as (nrows, ncols) arrays, row 0 = south, like the sim's buffers.
    """
    nrows, ncols = shape
    mass = np.asarray(sim.grid_mass, dtype=np.float64) * (1.0 / MASS_FACTOR)
    mom = np.asarray(sim.grid_momentum, dtype=np.float64) * (1.0 / MOMENTUM_FACTOR)
    mom = mom.reshape((nrows * ncols, 2))

    # mirror the shader's decode: nodes below the mass quantization floor are
    # empty (their momentum can round to nonzero while mass rounds to zero)
    has_mass = mass >= 1e-2
    vel_u = np.where(has_mass, mom[:, 0] / np.maximum(mass, 1e-6), 0.0)
    vel_v = np.where(has_mass, mom[:, 1] / np.maximum(mass, 1e-6), 0.0)
    mass = mass.reshape(shape)
    vel_u = vel_u.reshape(shape)
    vel_v = vel_v.reshape(shape)
    thickness = mass / (density * cell_size * cell_size * jac)
    return thickness, vel_u, vel_v, mass


def check_grid_mass_consistency(sim, grid_mass, particle_mass):
    """The dequantized grid mass must match the particle total mass."""
    grid_total = float(grid_mass.sum())
    particle_total = float(np.nansum(particle_mass))
    if particle_total <= 0:
        return
    rel = abs(grid_total - particle_total) / particle_total
    if rel > 0.01:
        log.warning(
            "grid mass (%.1f kg) differs from particle mass (%.1f kg) by %.2f%% - "
            "MASS_FACTOR/MOMENTUM_FACTOR in this script no longer match "
            "crates/compute_core/src/shaders/utils.wgsl",
            grid_total, particle_total, 100.0 * rel)


# ---------------------------------------------------------------------------
# Error norms - ported from avaframe/ana1Tests/analysisTools.py
# ---------------------------------------------------------------------------

def l2_norm(norm2_array, cell_size, cos_angle):
    return math.sqrt(cell_size * cell_size / cos_angle * np.nansum(norm2_array))


def _error_and_norm(local_error2, analytic2, cell_size, cos_angle):
    error_l2 = l2_norm(local_error2, cell_size, cos_angle)
    error_max = math.sqrt(np.nanmax(np.append(local_error2, 0.0)))
    analytic_l2 = l2_norm(analytic2, cell_size, cos_angle)
    analytic_max = math.sqrt(np.nanmax(np.append(analytic2, 0.0)))
    rel_l2 = error_l2 / analytic_l2 if analytic_l2 > 0 else error_l2
    rel_max = error_max / analytic_max if analytic_max > 0 else error_max
    return error_l2, rel_l2, error_max, rel_max


def norm_l2_scal(analytic, numerical, cell_size, cos_angle):
    local = (analytic - numerical) ** 2
    return _error_and_norm(local, analytic * analytic, cell_size, cos_angle)


def norm_l2_vect(analytic, numerical, cell_size, cos_angle):
    """L2/LMax of a vector field given as (fx, fy) arrays (in-plane components)."""
    local = (analytic[0] - numerical[0]) ** 2 + (analytic[1] - numerical[1]) ** 2
    ref = analytic[0] ** 2 + analytic[1] ** 2
    return _error_and_norm(local, ref, cell_size, cos_angle)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def make_settings(dem_path, mu):
    # sim_model 1 = curvilinear, friction_model 0 = coulomb.
    # drag_coefficient is set huge so the Voellmy/turbulent drag term of the
    # curvilinear grid physics vanishes and pure Coulomb friction remains.
    # max_steps must be large enough for run_n_steps to reach t_end at the
    # CFL limited dt (max(0.01, cfl * cell_size / (v + sqrt(g h)))).
    return {
        "dem_path": str(dem_path),
        "max_steps": 20000,
        "sim_model": 1,
        "released_particles_per_cell": 4,
        "friction_model": 0,
        "density": 200.0,
        "friction_coefficient": mu,
        "drag_coefficient": 1.0e12,
        "cfl": 0.05,
        "velocity_threshold": 0.01,
        "enable_particle_interaction": True,
    }


# SimInfo flag bits (must match utils.wgsl) for stop diagnostics
SIM_INFO_FLAGS = [
    ("OUT_OF_BOUNDS", 1 << 0),
    ("CFL_EXCEEDED", 1 << 1),
    ("IS_NAN", 1 << 2),
    ("NO_NEW_CELLS", 1 << 29),
    ("ALL_PARTICLES_STOPPED", 1 << 30),
    ("STOPPED", 1 << 31),
]


def describe_flags(flags):
    names = [name for name, bit in SIM_INFO_FLAGS if flags & bit]
    return "|".join(names) if names else f"0x{flags:x}"


def warn_if_no_samples(results, info, save_times, test_name):
    """Make an empty table impossible to miss: report why the sim stopped."""
    if results:
        return
    first = save_times[0] if save_times else float("nan")
    print(f"WARNING [{test_name}]: no samples were taken. The simulation stopped at "
          f"t={info.elapsed_time:.2f} s (timestep {info.timestep}, "
          f"flags: {describe_flags(info.flags)}) before reaching the first "
          f"save time of {first:.2f} s.")


def fetch_layers(sim):
    pos = np.asarray(sim.particles_position_xy, dtype=np.float64)
    vel = np.asarray(sim.particles_velocity_xy, dtype=np.float64)
    mass = np.asarray(sim.particles_mass, dtype=np.float64)
    return pos, vel, mass


# ---------------------------------------------------------------------------
# Similarity solution test - ported from avaframe/ana1Tests/simiSolTest.py
# ---------------------------------------------------------------------------

def define_earth_press_coeff(phi, delta):
    """Earth pressure coefficients (Hutter 1993); port of simiSolTest."""
    cos2phi = np.cos(phi) ** 2
    cos2delta = np.cos(delta) ** 2
    tan2delta = np.tan(delta) ** 2
    root1 = np.sqrt(1.0 - cos2phi / cos2delta)
    k = np.zeros(6)
    k[0] = 2 / cos2phi * (1.0 - root1) - 1.0
    k[1] = 2 / cos2phi * (1.0 + root1) - 1.0
    kx = k[0]
    root2 = np.sqrt((1.0 - kx) * (1.0 - kx) + 4.0 * tan2delta)
    k[2] = 0.5 * (kx + 1.0 - root2)
    k[3] = 0.5 * (kx + 1.0 + root2)
    kx = k[1]
    root2 = np.sqrt((1.0 - kx) * (1.0 - kx) + 4.0 * tan2delta)
    k[4] = 0.5 * (kx + 1.0 - root2)
    k[5] = 0.5 * (kx + 1.0 + root2)
    return k


def compute_earth_press_coeff(x, k):
    """K_x, K_y depending on the sign of the strain rates (port)."""
    g_p, f_p = x[1], x[3]
    if g_p >= 0:
        k_x = k[0]
        k_y = k[2] if f_p >= 0 else k[3]
    else:
        k_x = k[1]
        k_y = k[4] if f_p >= 0 else k[5]
    return k_x, k_y


def compute_f_coeff(k_x, k_y, zeta, delta, eps_x, eps_y):
    """Coefficients of eq 3.2 in Hutter 1993 (port)."""
    a = np.sin(zeta)
    b = eps_x * np.cos(zeta) * k_x
    c = np.cos(zeta) * np.tan(delta)
    d = eps_y * eps_y / eps_x * np.cos(zeta) * k_y
    if a == 0:
        e, c = 1.0, 0.0
    else:
        e = (a - c) / a
        c = np.cos(zeta) * np.tan(delta)
    return a, b, c, d, e


def calc_early_sol(t, k, x0, zeta, delta, eps_x, eps_y):
    """Early time solution 0 < t < t1 to avoid the ODE singularity (port)."""
    assert x0[3] == 0, "f'(t=0) must be 0"
    k_x, k_y = compute_earth_press_coeff(x0, k)
    _, b, c, d, e = compute_f_coeff(k_x, k_y, zeta, delta, eps_x, eps_y)
    g0, g_p0, f0, f_p0 = x0
    g = g0 + g_p0 * t + b / (f0 * g0 ** 2) * t ** 2
    g_p = g_p0 + 2 * b / (f0 * g0 ** 2) * t
    f = f0 + f_p0 * t + d * e / (g0 * f0 ** 2) * t ** 2
    f_p = f_p0 + 2 * d * e / (g0 * f0 ** 2) * t
    return {"timeAdim": t, "g_sol": g, "g_p_sol": g_p, "f_sol": f, "f_p_sol": f_p}


def _ffunction(t, x, k, zeta, delta, eps_x, eps_y):
    """RHS of the Hutter 1993 ODE system (port of Ffunction)."""
    k_x, k_y = compute_earth_press_coeff(x, k)
    _, b, c, d, _ = compute_f_coeff(k_x, k_y, zeta, delta, eps_x, eps_y)
    u_c = (np.sin(zeta) - c) * t
    g, g_p, f, f_p = x
    dx0 = g_p
    dx1 = 2 * b / (g ** 2 * f)
    dx2 = f_p
    if c == 0:
        dx3 = 2 * d / (g * f ** 2)
    else:
        dx3 = 2 * d / (g * f ** 2) - c * f_p / u_c
    return [dx0, dx1, dx2, dx3]


def similarity_solution(zeta_deg, delta_deg, phi_deg, flag_earth, lx, ly, rel_th, g, t_end):
    """Compute the similarity solution; port of mainSimilaritySol."""
    zeta = math.radians(zeta_deg)
    delta = math.radians(delta_deg)
    phi = math.radians(phi_deg)

    t_scale = math.sqrt(lx / g)
    eps_x = rel_th / lx
    eps_y = rel_th / ly

    t_end_adim = (t_end + 1.0) / t_scale  # +1 s buffer like AvaFrame
    dt_adim = 0.01 / t_scale

    if flag_earth:
        k = define_earth_press_coeff(phi, delta)
    else:
        k = np.ones(6)

    x0 = [1.0, 0.0, 1.0, 0.0]  # circular start
    t1 = 0.1
    t_early = np.arange(0.0, t1, dt_adim)
    t_early = np.append(t_early, t1)
    sol = calc_early_sol(t_early, k, x0, zeta, delta, eps_x, eps_y)

    def fun(t, x):
        return _ffunction(t, x, k, zeta, delta, eps_x, eps_y)

    try:
        from scipy.integrate import ode

        x_init = [sol["g_sol"][-1], sol["g_p_sol"][-1], sol["f_sol"][-1], sol["f_p_sol"][-1]]
        solver = ode(fun)
        solver.set_integrator("dopri5")
        solver.set_initial_value(x_init, t1)
        while solver.successful() and solver.t < t_end_adim:
            solver.integrate(solver.t + dt_adim, step=True)
            sol["timeAdim"] = np.append(sol["timeAdim"], solver.t)
            sol["g_sol"] = np.append(sol["g_sol"], solver.y[0])
            sol["g_p_sol"] = np.append(sol["g_p_sol"], solver.y[1])
            sol["f_sol"] = np.append(sol["f_sol"], solver.y[2])
            sol["f_p_sol"] = np.append(sol["f_p_sol"], solver.y[3])
    except ImportError:
        log.warning("scipy not available, falling back to fixed step RK4")
        x = np.array([sol["g_sol"][-1], sol["g_p_sol"][-1], sol["f_sol"][-1], sol["f_p_sol"][-1]])
        t = t1
        while t < t_end_adim:
            k1 = np.array(fun(t, x))
            k2 = np.array(fun(t + dt_adim / 2, x + dt_adim / 2 * k1))
            k3 = np.array(fun(t + dt_adim / 2, x + dt_adim / 2 * k2))
            k4 = np.array(fun(t + dt_adim, x + dt_adim * k3))
            x = x + dt_adim / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt_adim
            sol["timeAdim"] = np.append(sol["timeAdim"], t)
            for key, value in zip(["g_sol", "g_p_sol", "f_sol", "f_p_sol"], x):
                sol[key] = np.append(sol[key], value)

    sol["time"] = sol["timeAdim"] * t_scale
    return sol


def simi_fields_at(sol, t, x1, y, lx, ly, rel_th, zeta_deg, delta_deg, g):
    """Analytic h, u (along slope), v (cross slope) and center at time t.

    Port of getSimiSolParameters / computeH / computeU / computeV / computeXC.
    x1 is the along-slope coordinate, y the cross-slope coordinate.
    """
    zeta = math.radians(zeta_deg)
    delta = math.radians(delta_deg)
    idx = min(np.searchsorted(sol["time"], t), len(sol["time"]) - 1)
    tau = sol["timeAdim"][idx]
    g_sol, g_p = sol["g_sol"][idx], sol["g_p_sol"][idx]
    f_sol, f_p = sol["f_sol"][idx], sol["f_p_sol"][idx]

    a = np.sin(zeta)
    c = np.cos(zeta) * np.tan(delta)
    a_minus_c = a - c
    u_scale = math.sqrt(g * lx)
    v_scale = math.sqrt(g * ly)

    xi = x1 / lx - a_minus_c / 2.0 * tau ** 2
    eta2 = (y / ly) ** 2
    h = rel_th * (1.0 - (xi / g_sol) ** 2 - eta2 / f_sol ** 2) / (f_sol * g_sol)
    h = np.where(h <= 0, 0.0, h)
    u = u_scale * (a_minus_c * tau + (x1 / lx - a_minus_c / 2.0 * tau ** 2) * g_p / g_sol)
    v = v_scale * y / ly * f_p / f_sol
    u = np.where(h <= 0, 0.0, u)
    v = np.where(h <= 0, 0.0, v)
    x_center = lx * a_minus_c / 2.0 * tau ** 2 * math.cos(zeta)
    return h, u, v, x_center


def clean_release(thickness):
    """Zero out sliver thicknesses.

    The loader counts release cells above 1e-3 m while the particle
    initializer skips cells at or below 0.01 m; cells in between would spawn
    zero-mass placeholder particles that go NaN in the first g2p step.
    """
    return np.where(thickness <= 0.02, 0.0, thickness)


def simi_release_thickness(dem_header, ncols, nrows, zeta_deg, lx, ly, rel_th):
    """Parabolic release heap h = H (1 - X1^2/Lx^2 - Y^2/Ly^2), south-first."""
    cs = dem_header["cellsize"]
    xllc = dem_header["xllcenter"]
    yllc = dem_header["yllcenter"]
    cos_z = math.cos(math.radians(zeta_deg))
    cols = xllc + np.arange(ncols) * cs
    rows = yllc + np.arange(nrows) * cs  # row 0 = south
    x_map, y_map = np.meshgrid(cols, rows)
    x1 = x_map / cos_z
    thickness = rel_th * (1.0 - (x1 / lx) ** 2 - (y_map / ly) ** 2)
    return clean_release(np.where(thickness < 0, 0.0, thickness))


def run_simisol_test(avaframe_home, t_end=5.0, make_plots=True):
    import avalanchers

    dem_path = avaframe_home / "data" / "avaSimilaritySol" / "Inputs" / "DEM_IP_Topo.asc"
    header, dem = read_asc(dem_path)
    cs = header["cellsize"]
    nrows, ncols = dem.shape
    angle_measured = measured_plane_angle_deg(dem, cs)

    # AvaFrame simiSol_com1DFACfg.ini
    cfg = {
        "planeinclinationAngle": 35.0,
        "bedFrictionAngle": 25.0,
        "internalFrictionAngle": 25.0,
        "L_x": 80.0,
        "L_y": 80.0,
        "relTh": 4.0,
        "flagEarth": False,
        "gravAcc": 9.81,
    }
    zeta = cfg["planeinclinationAngle"]
    delta = cfg["bedFrictionAngle"]
    mu = math.tan(math.radians(delta))  # AvaFrame mucoulomb = tan(25 deg)
    if abs(angle_measured - zeta) > 0.5:
        log.warning("DEM slope is %.2f deg, config says %.2f deg - using the DEM angle",
                    angle_measured, zeta)
        zeta = angle_measured

    print("=== Similarity solution test (Hutter et al. 1993) ===")
    print(f"DEM: {dem_path}")
    print(f"plane {zeta:.2f} deg, bed friction {delta:.2f} deg (mu={mu:.6f}), "
          f"Lx={cfg['L_x']} m, Ly={cfg['L_y']} m, relTh={cfg['relTh']} m, t_end={t_end} s")

    release = simi_release_thickness(header, ncols, nrows, zeta,
                                     cfg["L_x"], cfg["L_y"], cfg["relTh"])
    jac = jacobian(dem, cs)

    settings = make_settings(dem_path, mu)
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.set_release_areas(np.ascontiguousarray(release, dtype=np.float32))

    # orientation sanity check: the heap is centered on map (0, 0)
    sim.run_n_steps(1)
    pos, vel, mass = fetch_layers(sim)
    xoff = header["xllcenter"] - 0.5 * cs
    yoff = header["yllcenter"] - 0.5 * cs
    pos = pos[np.isfinite(pos).all(axis=1)]
    centroid_map = (pos[:, 0].mean() + xoff, pos[:, 1].mean() + yoff)
    print(f"particles: {len(pos)}, heap centroid on map: "
          f"({centroid_map[0]:+.1f}, {centroid_map[1]:+.1f}) m - expected (+0.0, +0.0)")

    sol = similarity_solution(zeta, delta, cfg["internalFrictionAngle"], cfg["flagEarth"],
                              cfg["L_x"], cfg["L_y"], cfg["relTh"], cfg["gravAcc"], t_end)

    cos_z = math.cos(math.radians(zeta))
    cols = xoff + (np.arange(ncols) + 0.5) * cs  # cell centers, map coords
    rows = yoff + (np.arange(nrows) + 0.5) * cs
    x_map, y_map = np.meshgrid(cols, rows)
    x1_grid = x_map / cos_z
    x_cell_centers = cols[None, :].repeat(nrows, axis=0)

    density = settings["density"]
    results = []
    save_times = sorted({min(t, t_end) for t in np.arange(1.0, t_end + 0.01, 1.0)})
    idx_time = 0
    info = sim.run_n_steps(1)
    last_step = -1
    print()
    print(f"{'t [s]':>6} {'steps':>6} {'hL2rel':>10} {'hLMaxrel':>10} "
          f"{'huL2rel':>10} {'huLMaxrel':>10} {'xCom sim/ana [m]':>20}")
    while idx_time < len(save_times) and info.timestep != last_step:
        last_step = info.timestep
        while idx_time < len(save_times) and info.elapsed_time >= save_times[idx_time] - 1e-6:
            t = info.elapsed_time
            h_sim, u_sim, v_sim, mass_grid = fields_from_grid(sim, (nrows, ncols), cs, density, jac)
            if idx_time == 0:
                _, _, particle_mass = fetch_layers(sim)
                check_grid_mass_consistency(sim, mass_grid, particle_mass)

            h_ana, u_ana, v_ana, x_com_ana = simi_fields_at(
                sol, t, x1_grid, y_map, cfg["L_x"], cfg["L_y"], cfg["relTh"], zeta, delta, cfg["gravAcc"])

            h_l2, h_l2r, h_max, h_maxr = norm_l2_scal(h_ana, h_sim, cs, cos_z)
            vh_l2, vh_l2r, vh_max, vh_maxr = norm_l2_vect(
                (h_ana * u_ana, h_ana * v_ana), (h_sim * u_sim, h_sim * v_sim), cs, cos_z)
            # mass-weighted center of mass from the grid mass buffer
            total_mass = mass_grid.sum()
            x_com_sim = ((mass_grid * x_cell_centers).sum() / total_mass) if total_mass > 0 else float("nan")
            results.append({"t": t, "hL2": h_l2, "hL2rel": h_l2r, "hMaxrel": h_maxr,
                            "vhL2rel": vh_l2r, "vhMaxrel": vh_maxr,
                            "xComSim": x_com_sim, "xComAna": x_com_ana,
                            "fields": (h_sim, u_sim, v_sim, h_ana, u_ana, v_ana, x_com_ana)})
            print(f"{t:6.2f} {info.timestep:6d} {h_l2r:10.4f} {h_maxr:10.4f} "
                  f"{vh_l2r:10.4f} {vh_maxr:10.4f} {x_com_sim:9.1f}/{x_com_ana:<9.1f}")
            if make_plots:
                _plot_simisol_snapshot(OUTPUT_DIR / "pics", t, save_times[idx_time],
                                       h_sim, u_sim, h_ana, u_ana, x_com_ana, cols, rows, cfg)
            idx_time += 1
        if idx_time >= len(save_times):
            break
        info = sim.run_n_steps(20)

    warn_if_no_samples(results, info, save_times, "simisol")
    if results and make_plots:
        _plot_simisol(OUTPUT_DIR / "pics", results, cols, rows, cfg, zeta)
    return results


def _snapshot_figure():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


BAND_HALF_WIDTH = 2  # cells: profiles average 2*2+1 = 5 rows/columns


def _band_profile(field, center, half_width, axis):
    """Average `field` over a band of grid lines around `center`.

    axis=0 averages over rows (result: profile along the columns), axis=1
    averages over columns (profile along the rows). The band is clipped to
    [2, n-3]: the sim leaves the outer guard cells empty, and averaging them
    in would put artificial zero shoulders on the profile ends.
    """
    n = field.shape[0] if axis == 0 else field.shape[1]
    lo = max(center - half_width, 2)
    hi = min(center + half_width + 1, n - 2)
    if axis == 0:
        return field[lo:hi, :].mean(axis=0)
    return field[:, lo:hi].mean(axis=1)


def _plot_simisol_snapshot(out_dir, t, save_time, h_sim, u_sim, h_ana, u_ana,
                           x_com, cols, rows, cfg):
    """Profile-comparison snapshot saved once per second of simulated time.

    Profiles are averaged over a band of BAND_HALF_WIDTH rows/columns around
    the center line (simulation and analytic identically) instead of sampling
    a single line: this smooths the particle-sampling noise while staying
    clear of the domain edges.
    """
    plt = _snapshot_figure()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    row_mid = int(np.argmin(np.abs(rows - 0.0)))
    col_mid = int(np.argmin(np.abs(cols - x_com)))

    h_sim_x = _band_profile(h_sim, row_mid, BAND_HALF_WIDTH, axis=0)
    h_ana_x = _band_profile(h_ana, row_mid, BAND_HALF_WIDTH, axis=0)
    h_sim_y = _band_profile(h_sim, col_mid, BAND_HALF_WIDTH, axis=1)
    h_ana_y = _band_profile(h_ana, col_mid, BAND_HALF_WIDTH, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(cols, h_ana_x, "k--", label="analytic")
    ax.plot(cols, h_sim_x, "b", label="avalanchers")
    ax.axvline(x_com, color="grey", ls=":")
    ax.set_xlim(x_com - 3.0 * cfg["L_x"], x_com + 4.0 * cfg["L_x"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.set_title(f"along slope, averaged over y = +/-{BAND_HALF_WIDTH * 5:.0f} m")
    ax.legend()

    ax = axes[1]
    ax.plot(rows, h_ana_y, "k--", label="analytic")
    ax.plot(rows, h_sim_y, "b", label="avalanchers")
    ax.set_xlim(-2.5 * cfg["L_y"], 2.5 * cfg["L_y"])
    ax.set_xlabel("y [m]")
    ax.set_ylabel("h [m]")
    ax.set_title(f"across slope at x = {cols[col_mid]:.0f} m, averaged over x = +/-{BAND_HALF_WIDTH * 5:.0f} m")
    ax.legend()

    fig.suptitle(f"Similarity solution test - t = {t:.2f} s")
    fig.tight_layout()
    fig.savefig(out_dir / f"simisol_t{int(round(save_time)):d}.png", dpi=110)
    plt.close(fig)


def _plot_dambreak_snapshot(out_dir, t, save_time, h_sim, u_sim, v_sim,
                            h_ana, u_ana, cols, rows):
    """Profile-comparison snapshot saved once per second of simulated time.

    Profiles are averaged over a band of BAND_HALF_WIDTH rows around y = 0
    (simulation and analytic identically) instead of sampling a single line.
    """
    plt = _snapshot_figure()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    row_mid = int(np.argmin(np.abs(rows - 0.0)))

    h_sim_row = _band_profile(h_sim, row_mid, BAND_HALF_WIDTH, axis=0)
    h_ana_row = _band_profile(h_ana, row_mid, BAND_HALF_WIDTH, axis=0)
    hu_sim_row = _band_profile(h_sim * u_sim, row_mid, BAND_HALF_WIDTH, axis=0)
    hu_ana_row = _band_profile(h_ana * u_ana, row_mid, BAND_HALF_WIDTH, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(cols, h_ana_row, "k--", label="analytic")
    ax.plot(cols, h_sim_row, "b", label="avalanchers")
    ax.set_xlim(-220.0, 540.0)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.set_title(f"flow thickness profile (y = 0, averaged over y = +/-{BAND_HALF_WIDTH * 5:.0f} m)")
    ax.legend()

    ax = axes[1]
    ax.plot(cols, hu_ana_row, "k--", label="analytic")
    ax.plot(cols, hu_sim_row, "b", label="avalanchers")
    ax.set_xlim(-220.0, 540.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$hu$ [m$^2$/s]")
    ax.set_title(r"momentum $h\bar{u}$ profile (y = 0)")
    ax.legend()

    fig.suptitle(f"Dam break test - t = {t:.2f} s")
    fig.tight_layout()
    fig.savefig(out_dir / f"dambreak_t{int(round(save_time)):d}.png", dpi=110)
    plt.close(fig)

def _plot_simisol(out_dir, results, cols, rows, cfg, zeta_deg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available - skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    h_sim, u_sim, v_sim, h_ana, u_ana, v_ana, x_com = results[-1]["fields"]
    t = results[-1]["t"]
    row_mid = int(np.argmin(np.abs(rows - 0.0)))
    col_mid = int(np.argmin(np.abs(cols - x_com)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0][0]
    ax.plot(cols, h_ana[row_mid, :], "k--", label="analytic")
    ax.plot(cols, h_sim[row_mid, :], "b", label="avalanchers")
    ax.axvline(x_com, color="grey", ls=":")
    ax.set_title(f"flow thickness along slope (y=0), t={t:.1f}s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.set_xlim(-3.0 * cfg["L_x"], 3.0 * cfg["L_x"])
    ax.legend()

    ax = axes[0][1]
    ax.plot(rows, h_ana[:, col_mid], "k--", label="analytic")
    ax.plot(rows, h_sim[:, col_mid], "b", label="avalanchers")
    ax.set_title(f"flow thickness across slope (x={cols[col_mid]:.0f} m)")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("h [m]")
    ax.legend()

    ax = axes[1][0]
    ax.plot(cols, (h_ana * u_ana)[row_mid, :], "k--", label="analytic")
    ax.plot(cols, (h_sim * u_sim)[row_mid, :], "g", label="avalanchers")
    ax.set_title(r"momentum $h\bar{u}_x$ along slope")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$h u$ [m$^2$/s]")
    ax.set_xlim(-3.0 * cfg["L_x"], 3.0 * cfg["L_x"])
    ax.legend()

    ax = axes[1][1]
    times = [r["t"] for r in results]
    ax.plot(times, [r["hL2rel"] for r in results], "k-o", label="h L2 rel")
    ax.plot(times, [r["vhL2rel"] for r in results], "g-o", label=r"$h\bar{u}$ L2 rel")
    ax.set_yscale("log")
    ax.set_title("relative L2 error vs time")
    ax.set_xlabel("t [s]")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Similarity solution test - avalanchers vs Hutter 1993 "
                 f"(slope {zeta_deg:.0f} deg, delta {cfg['bedFrictionAngle']:.0f} deg)")
    fig.tight_layout()
    fig.savefig(out_dir / "simisol_summary.png", dpi=130)
    plt.close(fig)
    print(f"plot saved to {out_dir / 'simisol_summary.png'}")


# ---------------------------------------------------------------------------
# Dam break test - ported from avaframe/ana1Tests/damBreak.py
# ---------------------------------------------------------------------------

def dam_break_solution(phi_deg, delta_deg, h_l, times, s_grid, g=9.81):
    """Analytic dam break over dry bed on an inclined plane.

    Port of damBreakSol (Faccanoni & Mangeney 2012, Test 2, case 1.2), in
    along-slope coordinates s (dam front at s = 0, body extends upstream).
    ``s_grid`` may be 1-D or 2-D; returns h, u with shape
    (len(times),) + np.shape(s_grid).
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    gz = g * math.cos(phi)
    m0 = gz * (math.tan(phi) - math.tan(delta))
    c_l = math.sqrt(gz * h_l)

    h = np.zeros((len(times),) + np.shape(s_grid))
    u = np.zeros((len(times),) + np.shape(s_grid))
    for mi, t in enumerate(times):
        if t <= 0:
            continue
        cond1 = (m0 * t / 2.0 - c_l) * t
        cond2 = (2.0 * c_l + m0 * t / 2.0) * t
        uu = np.where(cond2 >= s_grid, (2.0 / 3.0) * (c_l + s_grid / t + m0 * t), 0.0)
        hh = np.where(cond2 >= s_grid, (2.0 * c_l - s_grid / t + m0 * t / 2.0) ** 2 / (9.0 * gz), 0.0)
        uu = np.where(cond1 >= s_grid, m0 * t, uu)
        hh = np.where(cond1 >= s_grid, h_l, hh)
        h[mi] = hh
        u[mi] = uu
    return h, u


def run_dambreak_test(avaframe_home, t_end=20.0, make_plots=True):
    import avalanchers

    dem_path = avaframe_home / "data" / "avaDamBreak" / "Inputs" / "DEM_IP_Topo.asc"
    header, dem = read_asc(dem_path)
    cs = header["cellsize"]
    nrows, ncols = dem.shape
    angle_measured = measured_plane_angle_deg(dem, cs)

    # AvaFrame damBreak_com1DFACfg.ini [DAMBREAK], with the documented
    # deviation: delta = 12 deg instead of 21 deg (see module docstring).
    cfg = {
        "phi": 22.0,
        "delta": 12.0,
        "relTh": 1.0,
        "xBack": -120.0,   # dam back, along-slope [m]; front at s = 0
        "damWidth": 200.0, # total dam width across slope [m]
    }
    phi = cfg["phi"]
    delta = cfg["delta"]
    mu = math.tan(math.radians(delta))
    if abs(angle_measured - phi) > 0.5:
        log.warning("DEM slope is %.2f deg, config says %.2f deg - using the DEM angle",
                    angle_measured, phi)
        phi = angle_measured

    print("=== Dam break test (Faccanoni & Mangeney 2012, dry bed) ===")
    print(f"DEM: {dem_path}")
    print(f"plane {phi:.2f} deg, bed friction {delta:.2f} deg (mu={mu:.6f}), "
          f"h0={cfg['relTh']} m, dam back {cfg['xBack']} m (along slope), t_end={t_end} s")

    # release: rectangular dam of uniform thickness, along-slope s in
    # [xBack, 0], i.e. map x in [xBack*cos(phi), 0]
    csz = cs
    xllc = header["xllcenter"]
    yllc = header["yllcenter"]
    cols = xllc + np.arange(ncols) * csz
    rows = yllc + np.arange(nrows) * csz  # row 0 = south
    x_map, y_map = np.meshgrid(cols, rows)
    s_map = x_map / math.cos(math.radians(phi))
    release = clean_release(np.where((s_map >= cfg["xBack"]) & (s_map <= 0.0)
                                     & (np.abs(y_map) <= cfg["damWidth"] / 2.0), cfg["relTh"], 0.0))
    jac = jacobian(dem, csz)

    settings = make_settings(dem_path, mu)
    # settings["enable_particle_relaxation"] = False
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.set_release_areas(np.ascontiguousarray(release, dtype=np.float32))

    sim.run_n_steps(1)
    pos, vel, mass = fetch_layers(sim)
    xoff = xllc - 0.5 * csz
    yoff = yllc - 0.5 * csz
    pos = pos[np.isfinite(pos).all(axis=1)]
    centroid = ((pos[:, 0].mean() + xoff) / math.cos(math.radians(phi)),
                pos[:, 1].mean() + yoff)
    print(f"particles: {len(pos)}, dam centroid s/y: ({centroid[0]:+.1f}, {centroid[1]:+.1f}) m "
          f"- expected ({(cfg['xBack']) / 2:+.1f}, +0.0)")

    density = settings["density"]
    cos_p = math.cos(math.radians(phi))
    cols_c = xoff + (np.arange(ncols) + 0.5) * csz
    rows_c = yoff + (np.arange(nrows) + 0.5) * csz
    xc_map, yc_map = np.meshgrid(cols_c, rows_c)
    s_grid_cells = xc_map / cos_p  # along-slope coordinate of cell centers

    save_times = sorted({min(t, t_end) for t in np.arange(1.0, t_end + 0.01, 1.0)})
    results = []
    info = sim.run_n_steps(1)
    last_step = -1
    idx_time = 0
    print()
    print(f"{'t [s]':>6} {'steps':>6} {'hL2rel':>10} {'hLMaxrel':>10} "
          f"{'huL2rel':>10} {'huLMaxrel':>10} {'front s sim/ana [m]':>22}")
    gz = 9.81 * cos_p
    m0 = gz * (math.tan(math.radians(phi)) - math.tan(math.radians(delta)))
    c_l = math.sqrt(gz * cfg["relTh"])

    while idx_time < len(save_times) and info.timestep != last_step:
        last_step = info.timestep
        while idx_time < len(save_times) and info.elapsed_time >= save_times[idx_time] - 1e-6:
            t = info.elapsed_time
            h_sim, u_sim, v_sim, mass_grid = fields_from_grid(sim, (nrows, ncols), csz, density, jac)
            h_ana, u_ana = dam_break_solution(phi, delta, cfg["relTh"], [t], s_grid_cells, 9.81)
            h_ana, u_ana = h_ana[0], u_ana[0]

            # Error domain: like AvaFrame, only the material downstream of the
            # (analytic) dam back is compared - the rear face of a finite dam
            # develops its own rarefaction that the front-only analytic
            # solution does not describe. The downstream bound follows the
            # material instead of AvaFrame's fixed x window, since with
            # delta = 12 deg the dam leaves that window.
            s_back = cfg["xBack"] + 0.5 * m0 * t * t  # analytic dam back
            domain = ((np.abs(yc_map) <= 50.0)
                      & (s_grid_cells >= s_back + 10.0)
                      & (s_grid_cells <= 640.0))

            h_l2, h_l2r, h_max, h_maxr = norm_l2_scal(h_ana[domain], h_sim[domain], csz, cos_p)
            vh_l2, vh_l2r, vh_max, vh_maxr = _masked_vect_error(
                h_ana * u_ana, h_sim * u_sim, h_sim * v_sim, domain, csz, cos_p)

            front_ana = (2.0 * c_l + m0 * t / 2.0) * t
            pos, _, _ = fetch_layers(sim)
            moving = pos[np.isfinite(pos[:, 0]), 0] + xoff
            front_sim = moving.max() / cos_p if len(moving) else float("nan")
            results.append({"t": t, "hL2rel": h_l2r, "hMaxrel": h_maxr,
                            "vhL2rel": vh_l2r, "vhMaxrel": vh_maxr,
                            "frontSim": front_sim, "frontAna": front_ana,
                            "fields": (h_sim, u_sim, v_sim, h_ana, u_ana)})
            print(f"{t:6.2f} {info.timestep:6d} {h_l2r:10.4f} {h_maxr:10.4f} "
                  f"{vh_l2r:10.4f} {vh_maxr:10.4f} {front_sim:9.1f}/{front_ana:<9.1f}")
            if make_plots:
                _plot_dambreak_snapshot(OUTPUT_DIR / "pics", t, save_times[idx_time],
                                        h_sim, u_sim, v_sim, h_ana, u_ana, cols_c, rows_c)
            idx_time += 1
        if idx_time >= len(save_times):
            break
        info = sim.run_n_steps(20)

    warn_if_no_samples(results, info, save_times, "dambreak")
    if results and make_plots:
        _plot_dambreak(OUTPUT_DIR / "pics", results, cols_c, rows_c, cfg, phi)
    return results


def _masked_vect_error(ana_u, sim_u, sim_v, domain, cell_size, cos_angle):
    """Momentum vector error on a masked domain (fx along slope, fy across)."""
    local = np.where(domain, (ana_u - sim_u) ** 2 + sim_v ** 2, 0.0)
    ref = np.where(domain, ana_u ** 2, 0.0)
    return _error_and_norm(local, ref, cell_size, cos_angle)


def _frame_near(rows, t):
    """Index of the row whose 't' is closest to t."""
    return min(range(len(rows)), key=lambda i: abs(rows[i]["t"] - t))


def _plot_dambreak(out_dir, results, cols, rows, cfg, phi_deg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available - skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    row_mid = int(np.argmin(np.abs(rows - 0.0)))

    # the summary shows 5 s resolution: the sample closest to each 5 s mark
    marker_times = [tm for tm in [0.0, 5.0, 10.0, 15.0, 20.0] if tm <= results[-1]["t"] + 1e-6]
    summary_rows = [results[_frame_near(results, tm)] for tm in marker_times]
    if not summary_rows:
        summary_rows = results  # t_end < 5 s: fall back to every second

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for k, r in enumerate(summary_rows):
        color = f"C{k}"
        last = r is summary_rows[-1]
        h_sim, u_sim, v_sim, h_ana, u_ana = r["fields"]
        axes[0][0].plot(cols, h_ana[row_mid, :], "k--", alpha=0.4 + 0.2 * last)
        axes[0][0].plot(cols, h_sim[row_mid, :], color=color, label=f"t={r['t']:.0f}s")
        axes[1][0].plot(cols, (h_ana * u_ana)[row_mid, :], "k--", alpha=0.5)
        axes[1][0].plot(cols, (h_sim * u_sim)[row_mid, :], color=color)

    axes[0][0].set_title("flow thickness profile (y=0), dashed = analytic")
    axes[0][0].set_xlabel("x [m]")
    axes[0][0].set_ylabel("h [m]")
    axes[0][0].legend()
    axes[1][0].set_title(r"momentum $h\bar{u}$ profile (y=0)")
    axes[1][0].set_xlabel("x [m]")
    axes[1][0].set_ylabel(r"$h u$ [m$^2$/s]")

    ax = axes[0][1]
    times = [r["t"] for r in results]
    ax.plot(times, [r["frontSim"] for r in results], "b-o", label="avalanchers")
    ax.plot(times, [r["frontAna"] for r in results], "k--o", label="analytic")
    ax.set_title("front position (along slope), every 5 s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("s [m]")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(times, [r["hL2rel"] for r in results], "k-o", label="h L2 rel")
    ax.plot(times, [r["vhL2rel"] for r in results], "g-o", label=r"$h\bar{u}$ L2 rel")
    ax.set_yscale("log")
    ax.set_title("relative L2 error vs time, every 5 s")
    ax.set_xlabel("t [s]")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Dam break test - avalanchers vs Faccanoni & Mangeney 2012 "
                 f"(slope {phi_deg:.0f} deg, delta {cfg['delta']:.0f} deg)")
    fig.tight_layout()
    fig.savefig(out_dir / "dambreak_summary.png", dpi=130)
    plt.close(fig)
    print(f"plot saved to {out_dir / 'dambreak_summary.png'}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("test", choices=["simisol", "dambreak", "all"], default="all", nargs="?")
    parser.add_argument("--t-end", type=float, default=None,
                        help="override the simulated end time in seconds")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    env_home = os.environ.get("AVAFRAME_HOME")
    avaframe_home = pathlib.Path(env_home) if env_home else DEFAULT_AVAFRAME_HOME
    if not avaframe_home.exists():
        sys.exit(f"AvaFrame checkout not found at {avaframe_home} "
                 f"(set AVAFRAME_HOME to your AvaFrame/avaframe directory)")

    t_end = args.t_end
    if args.test in ("simisol", "all"):
        run_simisol_test(avaframe_home, t_end=t_end or 20.0, make_plots=not args.no_plots)
    if args.test in ("dambreak", "all"):
        run_dambreak_test(avaframe_home, t_end=t_end or 20.0, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
