"""Similarity solution test analysis and plots for avalanchers, adapted
from AvaFrame.

Port of the AvaFrame similarity solution test chain -
``avaframe/ana1Tests/simiSolTest.py`` (Hutter, Siegel & Savage 1993
similarity solution for a granular avalanche spreading down an inclined
plane), ``avaframe/ana1Tests/analysisTools.py`` (error norms) and the
simiSol plots of ``avaframe/out3Plot/outAna1Plots.py`` (``_plotVariable``,
``saveSimiSolProfile``, ``makeContourSimiPlot``, ``addContour2Plot``,
``addErrorTime``, ``plotSimiSolSummary``) - so it runs on the in-memory
results of an avalanchers simulation (see the companion script
``run_similarity_solution_test.py``). No avaframe imports, no output
files.

The numerical flow fields are derived DIRECTLY from the simulation's
``grid_mass`` / ``grid_momentum`` fields (the same fields the grid physics
pass consumed).

Momentum components follow AvaFrame's Cartesian convention: the analytic
along-slope velocity u is split into u_x = u cos(zeta), u_z = -u sin(zeta),
the cross-slope velocity is u_y. The 3D depth-averaged model carries the
full Cartesian velocity (``grid_momentum`` + ``grid_momentum_z``), so its
profiles show h*u_x, h*u_y AND h*u_z and its momentum error is the 3D vector
norm (as AvaFrame's normL2Vect with fz). The in-plane models
(curvilinear, terrain-following, mpmdac) carry no vertical component: they
show h*u_x, h*u_y only and their momentum error is the in-plane norm, which
equals the 3D norm for a velocity tangent to the plane.

Thesis figures (``plot_thesis_figures``) are written separately with
``thesis_style`` - 15 cm text width, Computer Modern, vector PDF + PNG.

Deviation from the AvaFrame contour plots: the raw quantized grid field is
per-cell noisy (a few particles per 5 m cell), unlike com1DFA's SPH-sampled
fields, so the contour comparison smooths the numerical field with a 3x3
mean first (``blur3x3``). The profiles use the raw field - a row cut
averages along-track naturally.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# plot constants mirroring avaframe/out3Plot/plotUtilsCfg.ini
FIG_W = 6
FIG_H = 6
FS = 12
NAME_FT = "flow thickness"
UNIT_FT = "m"
UNIT_FTV = r"$m^2s^{-1}$"

# "FT" colormap colors of AvaFrame (plotUtils.py colorsT, lajolla)
FT_COLORS = ["#FCFFC9", "#EBCE7B", "#DE9529", "#BE5A32", "#7F2B3F", "#1D0B14"]


# ---------------------------------------------------------------------------
# grid fields, read from the grid mass/momentum buffers
# ---------------------------------------------------------------------------

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


# sim models whose velocity is the surface velocity rotated into the horizontal
# plane: its components already measure surface distance, unlike the horizontal
# dx/dt of the other models
SURFACE_VELOCITY_MODELS = ("terrain-following", "terrain", "t")
# sim models carrying the full 3D Cartesian velocity (vertical momentum buffer)
THREE_D_MODELS = ("3d-depth-averaged", "3d")


def is_three_d(sim_model):
    return str(sim_model).lower() in THREE_D_MODELS


def fields_from_grid(sim, shape, cell_size, density, jac, surface_velocity=False,
                     three_d=False):
    """Decode the sim's grid mass/momentum getters into thickness + velocity.

    These are exactly the fields the grid physics pass consumed this step.
    Returns a dict of (nrows, ncols) arrays, row 0 = south:

    - ``h``: flow thickness normal to the surface [m]
    - ``u``, ``v``: along-slope (surface distance) and cross-slope velocity
    - ``vx``, ``vy``, ``vz``: Cartesian velocity; ``vz`` is the model's own
      vertical velocity for the 3D model and None for the in-plane models
    - ``speed``: |u| (3D norm for the 3D model)
    - ``mass``: grid mass [kg]

    The Cartesian split assumes the test's plane, inclined along x.
    """
    nrows, ncols = shape
    mass = np.asarray(sim.grid_mass, dtype=np.float64)
    mom = np.asarray(sim.grid_momentum, dtype=np.float64)
    mom = mom.reshape((nrows * ncols, 2))
    # mirror the shader's decode: nodes below the mass quantization floor are empty
    has_mass = mass >= 1e-2
    inv_mass = np.where(has_mass, 1.0 / np.maximum(mass, 1e-6), 0.0)
    raw_u = (mom[:, 0] * inv_mass).reshape(shape)
    raw_v = (mom[:, 1] * inv_mass).reshape(shape)
    vel_z = None
    if three_d:
        mom_z = np.asarray(sim.grid_momentum_z, dtype=np.float64)
        vel_z = (mom_z * inv_mass).reshape(shape)
    mass = mass.reshape(shape)
    # The reference uses surface distance along its x-aligned plane, which a
    # horizontal dx/dt only measures with the metric; a surface velocity
    # (terrain-following) is already a surface distance.
    if surface_velocity:
        vel_u, vel_x = raw_u, raw_u / jac
    else:
        vel_u, vel_x = raw_u * jac, raw_u
    if vel_z is not None:
        speed = np.sqrt(vel_x ** 2 + raw_v ** 2 + vel_z ** 2)
    else:
        speed = np.hypot(vel_u, raw_v)
    thickness = mass / (density * cell_size * cell_size * jac)
    return {"h": thickness, "u": vel_u, "v": raw_v, "vx": vel_x, "vy": raw_v,
            "vz": vel_z, "speed": speed, "mass": mass}


def check_grid_mass_consistency(sim, grid_mass, particle_mass):
    """The grid mass must match the particle total mass."""
    grid_total = float(grid_mass.sum())
    particle_total = float(np.nansum(particle_mass))
    if particle_total <= 0:
        return
    rel = abs(grid_total - particle_total) / particle_total
    if rel > 0.01:
        print(f"WARNING: grid mass ({grid_total:.1f} kg) differs from particle mass "
              f"({particle_total:.1f} kg) by {100.0 * rel:.2f}%")


# ---------------------------------------------------------------------------
# error norms - ported from avaframe/ana1Tests/analysisTools.py
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
    """L2/LMax of a vector field given as (fx, fy[, fz]) component arrays."""
    local = sum((a - n) ** 2 for a, n in zip(analytic, numerical))
    ref = sum(a ** 2 for a in analytic)
    return _error_and_norm(local, ref, cell_size, cos_angle)


# ---------------------------------------------------------------------------
# similarity solution - ported from avaframe/ana1Tests/simiSolTest.py
# (Hutter, Siegel & Savage 1993)
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
    """Compute the similarity solution; port of mainSimilaritySol.

    Returns a dict with timeAdim/time, g_sol, g_p_sol, f_sol, f_p_sol.
    """
    zeta = math.radians(zeta_deg)
    delta = math.radians(delta_deg)
    phi = math.radians(phi_deg)

    t_scale = math.sqrt(lx / g)
    eps_x = rel_th / lx
    eps_y = rel_th / ly

    t_end_adim = (t_end + 1.0) / t_scale  # +1 s buffer like AvaFrame
    dt_adim = 0.01 / t_scale

    k = define_earth_press_coeff(phi, delta) if flag_earth else np.ones(6)

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
        print("scipy not available, falling back to fixed step RK4")
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


def similarity_fields(sol, t, x1, y1, lx, ly, rel_th, zeta_deg, delta_deg, g):
    """Analytic h, u (along slope), v (cross slope) and center at time t.

    Port of getSimiSolParameters / computeH / computeU / computeV /
    computeXC. x1 is the along-slope coordinate, y1 the cross-slope
    coordinate, both relative to the initial heap center.
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
    eta2 = (y1 / ly) ** 2
    h = rel_th * (1.0 - (xi / g_sol) ** 2 - eta2 / f_sol ** 2) / (f_sol * g_sol)
    h = np.where(h <= 0, 0.0, h)
    u = u_scale * (a_minus_c * tau + (x1 / lx - a_minus_c / 2.0 * tau ** 2) * g_p / g_sol)
    v = v_scale * y1 / ly * f_p / f_sol
    u = np.where(h <= 0, 0.0, u)
    v = np.where(h <= 0, 0.0, v)
    x_center = lx * a_minus_c / 2.0 * tau ** 2 * math.cos(zeta)
    # Cartesian split on the plane inclined along x (AvaFrame getSimiSolParameters)
    return {"h": h, "u": u, "v": v, "x_center": x_center,
            "vx": u * math.cos(zeta), "vy": v, "vz": -u * math.sin(zeta),
            "speed": np.hypot(u, v), "time": float(sol["time"][idx])}


# ---------------------------------------------------------------------------
# plots - ported from avaframe/out3Plot/outAna1Plots.py (simiSol section)
# ---------------------------------------------------------------------------

def get_plot_limits(field, center_x, center_y, header, scale_coef):
    """Plot limits around the flow field (port of getPlotLimits with
    extentOption=True: widths are in METERS, half the field extent).

    `field` is a fields_from_grid dict; minMz reaches below zero for the
    negative vertical momentum h*u_z of the 3D model (as in AvaFrame)."""
    cs = header["cellsize"]
    h_field = field["h"]
    ind = h_field > 0
    if ind.any():
        cols = np.where(ind.any(axis=0))[0]
        rows = np.where(ind.any(axis=1))[0]
        width_x = scale_coef * ((cols.max() + 1) - cols.min()) * cs / 2.0
        width_y = scale_coef * ((rows.max() + 1) - rows.min()) * cs / 2.0
    else:
        width_x = width_y = 50.0
    max_ft = scale_coef * float(np.nanmax(h_field))
    max_fm = scale_coef * float(np.nanmax(h_field * field["speed"]))
    min_mz = 0.0
    if field.get("vz") is not None:
        min_mz = scale_coef * min(0.0, float(np.nanmin(h_field * field["vz"])))
    return {"widthX": width_x, "widthY": width_y,
            "maxFT": max_ft, "minMz": min_mz, "maxFM": max_fm,
            "centerX": center_x, "centerY": center_y}


# momentum components: key -> (AvaFrame colour, label)
_COMPONENTS = {
    "norm": ("g", r"$\vert h \mathbf{\bar{u}} \vert$"),
    "x": ("m", r"$h \bar{u}_x$"),
    "y": ("b", r"$h \bar{u}_y$"),
    "z": ("c", r"$h \bar{u}_z$"),
}


def momentum_components(field, with_z):
    """Momentum fields per component: |h u|, h u_x, h u_y (and h u_z)."""
    comps = {"norm": field["h"] * field["speed"],
             "x": field["h"] * field["vx"],
             "y": field["h"] * field["vy"]}
    if with_z:
        comps["z"] = field["h"] * field["vz"]
    return comps


def _plot_variable_panel(ax, axis, coordinates, sim_curves, ana_curves,
                         x_center, limits):
    """Twin-axis profile panel (port of _plotVariable): flow thickness on the
    left axis, momentum components on the right axis (|h*u| green, h*ux
    magenta, h*uy blue, h*uz cyan for the 3D model); simulation full line,
    analytic dashed.

    `axis` is "xaxis" (coordinates = x array, cut along the center row) or
    "yaxis" (coordinates = y array, cut at the center column).
    """
    ax2 = ax.twinx()
    if axis == "xaxis":
        ax.axvline(x=x_center, linestyle=":", color="b")
    lns = [ax.plot(coordinates, sim_curves["h"], "k", label="h")[0]]
    ax.plot(coordinates, ana_curves["h"], "--k")
    for key, (color, label) in _COMPONENTS.items():
        if key not in sim_curves:
            continue
        lns.append(ax2.plot(coordinates, sim_curves[key], color, label=label)[0])
        ax2.plot(coordinates, ana_curves[key], "--" + color)

    if axis == "xaxis":
        ax.set_xlabel("x in [m]")
        ax.text(x_center + 5, 0.02 * limits["maxFT"], "x = %.2f m" % x_center, color="b")
        ax.set_xlim([x_center - limits["widthX"], x_center + limits["widthX"]])
    else:
        ax.set_xlabel("y in [m]")
        ax.set_xlim([limits["centerY"] - limits["widthY"],
                     limits["centerY"] + limits["widthY"]])
    # align the zero of both axes when the vertical momentum goes negative
    ratio = limits["minMz"] / limits["maxFM"] if limits["maxFM"] > 0 else 0.0
    ax.set_ylim([min(-0.05, ratio * limits["maxFT"]), limits["maxFT"]])
    ax2.set_ylim([limits["minMz"], limits["maxFM"]])
    ax.set_ylabel(NAME_FT + " [" + UNIT_FT + "]")
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    color = "tab:green"
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.axhline(0.0, color="grey", linewidth=0.5)
    ax2.set_ylabel(r"$h \bar{u}_i$ [" + UNIT_FTV + "]", color=color)
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=FS - 2)
    text = "analytical solution (dashed line) \n numerical solution (full line)"
    ax.text(0.03, 0.95, text, transform=ax.transAxes, verticalalignment="top",
            fontsize=FS - 2)
    return ax, ax2


def _profile_curves(field, h_ana, center_row, center_col):
    """Along-flow (center row) and across-flow (center column) cuts of h and
    the momentum components, for simulation and analytic solution."""
    with_z = field.get("vz") is not None
    sim = dict(momentum_components(field, with_z), h=field["h"])
    ana = dict(momentum_components(h_ana, with_z), h=h_ana["h"])
    along = {k: a[center_row] for k, a in sim.items()}
    across = {k: a[:, center_col] for k, a in sim.items()}
    along_ana = {k: a[center_row] for k, a in ana.items()}
    across_ana = {k: a[:, center_col] for k, a in ana.items()}
    return along, across, along_ana, across_ana


def _cell_centers(header):
    xs = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * header["cellsize"]
    ys = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * header["cellsize"]
    return xs, ys


def _center_indices(header, center_y, x_center):
    xs, ys = _cell_centers(header)
    return int(np.argmin(np.abs(ys - center_y))), int(np.argmin(np.abs(xs - x_center)))


def blur3x3(field, passes=1):
    """NaN-aware 3x3 mean smoothing (same scheme as avalanchers.blur_nan_grid)."""
    smoothed = np.asarray(field, dtype=float).copy()
    for _ in range(max(0, int(passes))):
        valid = np.isfinite(smoothed)
        values = np.where(valid, smoothed, 0.0)
        weights = valid.astype(float)
        values_sum = np.zeros_like(smoothed)
        weights_sum = np.zeros_like(smoothed)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                values_sum += np.roll(np.roll(values, dy, axis=0), dx, axis=1)
                weights_sum += np.roll(np.roll(weights, dy, axis=0), dx, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(weights_sum > 0, values_sum / weights_sum, np.nan)
    return smoothed


def add_contour_2plot(ax, field_ft, h_ana, header, limits, n_levels=9):
    """Contour comparison of numerical (solid) and analytical (dashed) flow
    thickness (port of addContour2Plot). The window and cut lines are
    centered on the ANALYTIC heap center at the plotted time (x_center); the
    numerical field is lightly smoothed first (see module docstring)."""
    field_ft = blur3x3(field_ft)
    xs, ys = _cell_centers(header)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    x_center = h_ana["x_center"]
    y_center = limits["centerY"]
    contour_levels = np.linspace(0, limits["maxFT"], n_levels)[1:-1]
    cmap = LinearSegmentedColormap.from_list("cmapT", FT_COLORS)
    norm = Normalize(vmin=0.0, vmax=limits["maxFT"])
    cs1 = ax.contour(x_mesh, y_mesh, field_ft, levels=contour_levels, cmap=cmap,
                     norm=norm, linewidths=2)
    cs2 = ax.contour(x_mesh, y_mesh, h_ana["h"], levels=contour_levels, cmap=cmap,
                     norm=norm, linewidths=2, linestyles="dashed")
    cb = ax.figure.colorbar(cs1, ax=ax, shrink=0.8, pad=0.02, format="%.2f")
    cb.ax.set_title(NAME_FT + " [" + UNIT_FT + "]", fontsize=FS - 3, pad=8)
    h1, _ = cs1.legend_elements()
    h2, _ = cs2.legend_elements()
    ax.legend([h1[-1], h2[-1]], ["simulation", "analytical"], fontsize=FS - 2,
              loc="upper right")
    ax.set_ylim([y_center - limits["widthY"], y_center + limits["widthY"]])
    ax.set_xlim([x_center - limits["widthX"], x_center + limits["widthX"]])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axvline(x_center, color="b", linestyle=":")
    ax.axhline(y_center, color="r", linestyle=":")
    ax.text(x_center + 5, y_center - limits["widthY"] + 5, "x = %.2f m" % x_center,
            color="b", fontsize=FS - 2)
    ax.text(x_center - limits["widthX"] + 5, y_center + 5, "y = 0 m", color="r",
            fontsize=FS - 2)
    return ax


def add_error_time(ax1, times, errors, relative, t_save):
    """Error-vs-time panel (port of addErrorTime) - single log axis, since
    flow thickness and momentum errors share the same (relative) scale."""
    title = "Relative error" if relative else "Error"
    ax1.plot(times, errors["h_l2"], "k-", label="h L2")
    ax1.plot(times, errors["h_lmax"], "k--", label="h LMax")
    ax1.plot(times, errors["vh_l2"], "g-",
             label=r"$\vert h \mathbf{\bar{u}} \vert$ L2")
    ax1.plot(times, errors["vh_lmax"], "g--",
             label=r"$\vert h \mathbf{\bar{u}} \vert$ LMax")
    if errors.get("vhz_l2") is not None and np.isfinite(errors["vhz_l2"]).any():
        ax1.plot(times, errors["vhz_l2"], "c-", label=r"$h \bar{u}_z$ L2")
        ax1.plot(times, errors["vhz_lmax"], "c--", label=r"$h \bar{u}_z$ LMax")
    ax1.set_xlabel("time in [s]")
    ax1.set_ylabel(title)
    ax1.set_yscale("log")
    ax1.legend(loc="best", fontsize=FS - 3, ncol=2)
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5, which="both")
    ax1.axvline(t_save, color="grey", linestyle="--")
    min_y, _ = ax1.get_ylim()
    ax1.text(t_save, min_y, " %.2f s" % t_save, va="bottom")
    return ax1


def plot_profile_comparison(header, field, h_ana, limits, t, sim_name, out_dir):
    """Standalone profile comparison figure at one time step (port of
    saveSimiSolProfile): flow-direction and across-flow twin-axis profiles."""
    center_row, center_col = _center_indices(header, limits["centerY"], h_ana["x_center"])
    xs, ys = _cell_centers(header)
    along, across, along_ana, across_ana = _profile_curves(field, h_ana,
                                                           center_row, center_col)
    fig = plt.figure(figsize=(2.5 * FIG_W, FIG_H), layout="constrained")
    fig.suptitle("Similarity solution test, t = %.2f s (simulation %s)" % (t, sim_name),
                 fontsize=FS + 2)
    ax1 = fig.add_subplot(1, 2, 1)
    _plot_variable_panel(ax1, "xaxis", xs, along, along_ana,
                         h_ana["x_center"], limits)
    ax1.set_title("Profile in flow direction (y = 0 m)")
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_variable_panel(ax2, "yaxis", ys, across, across_ana,
                         h_ana["x_center"], limits)
    ax2.set_title("Profile across flow direction (x = %.2f m)" % h_ana["x_center"])
    return _save_fig(fig, out_dir, "compare_profile_simi_sol_%s_%07.2f" % (sim_name, t))


def plot_contour_comparison(header, field_ft, h_ana, limits, t, sim_name, out_dir):
    """Standalone contour comparison figure at one time step (port of
    makeContourSimiPlot)."""
    fig, ax1 = plt.subplots(figsize=(1.5 * FIG_W, FIG_H), layout="constrained")
    add_contour_2plot(ax1, field_ft, h_ana, header, limits, n_levels=10)
    ax1.set_title(NAME_FT + " contours at t = %.2f s" % t)
    return _save_fig(fig, out_dir, "compare_contour_simi_sol_%s_%07.2f" % (sim_name, t))


def plot_simi_sol_summary(header, field, h_ana, limits, errors, times, t_save,
                          cfg, sim_name, settings, out_dir):
    """Summary figure (port of plotSimiSolSummary): twin-axis profile panels
    along and across the flow, the FT contour comparison, the error curves
    and the parameter text."""
    relative = cfg["relative_error"]
    center_row, center_col = _center_indices(header, limits["centerY"], h_ana["x_center"])
    xs, ys = _cell_centers(header)
    along, across, along_ana, across_ana = _profile_curves(field, h_ana,
                                                           center_row, center_col)

    fig = plt.figure(figsize=(FIG_W * 3, FIG_H * 1.8), layout="constrained")
    fig.suptitle("Similarity solution test, t = %.2f s (simulation %s)"
                 % (t_save, sim_name), fontsize=FS + 4)
    grid = fig.add_gridspec(2, 6)
    ax1 = fig.add_subplot(grid[0, 0:3])
    _plot_variable_panel(ax1, "xaxis", xs, along, along_ana, h_ana["x_center"], limits)
    ax1.set_title("Profile in flow direction (y = 0 m)")
    ax2 = fig.add_subplot(grid[0, 3:6])
    _plot_variable_panel(ax2, "yaxis", ys, across, across_ana, h_ana["x_center"], limits)
    ax2.set_title("Profile across flow direction (x = %.2f m)" % h_ana["x_center"])

    ax3 = fig.add_subplot(grid[1, 0:2])
    add_contour_2plot(ax3, field["h"], h_ana, header, limits)
    ax3.set_title("Flow thickness contours")

    ax4 = fig.add_subplot(grid[1, 2:4])
    ax4.set_title(("Relative error" if relative else "Error")
                  + " between analytical solution and avalanchers")
    add_error_time(ax4, times, errors, relative, t_save)

    ax7 = fig.add_subplot(grid[1, 4:6])
    ax7.axis("off")
    momentum = "3D (x, y, z)" if field.get("vz") is not None else "in-plane (x, y)"
    text = (f"sim_model = {settings.get('sim_model')}\n"
            f"momentum = {momentum}\n"
            f"zeta = {cfg['zeta']:.2f} deg\ndelta = {cfg['delta']:.2f} deg\n"
            f"phi = {cfg['phi']:.2f} deg\nL_x = {cfg['l_x']:.1f} m\n"
            f"L_y = {cfg['l_y']:.1f} m\nrelTh = {cfg['rel_th']:.1f} m\n"
            f"density = {settings.get('density')} kg/m3\n"
            f"cfl = {settings.get('cfl')}\n"
            f"particles/cell = {settings.get('released_particles_per_cell')}\n"
            f"earth pressure = {cfg['flag_earth']}")
    ax7.text(0.05, 0.5, text, transform=ax7.transAxes, ha="left", va="center",
             fontsize=FS, family="monospace")
    return _save_fig(fig, out_dir, sim_name + "_SimiSolTest")


# ---------------------------------------------------------------------------
# thesis figures - 15 cm text width, Computer Modern, PDF + PNG
# ---------------------------------------------------------------------------

_TEX_COMPONENTS = {
    "norm": r"$h\,\Vert\mathbf{u}\Vert$",
    "x": r"$h\,u_x$",
    "y": r"$h\,u_y$",
    "z": r"$h\,u_z$",
}
_UNIT_H = r"$h$ (m)"
_UNIT_M = r"momentum (m$^2$ s$^{-1}$)"


def _thesis_profiles(ts, header, field, ana, limits, with_z):
    from matplotlib.lines import Line2D
    xs, ys = _cell_centers(header)
    row, col = _center_indices(header, limits["centerY"], ana["x_center"])
    along, across, along_ana, across_ana = _profile_curves(field, ana, row, col)
    fig, axes = plt.subplots(2, 2, figsize=ts.size(1.0, 0.66), sharex="col",
                             sharey="row", layout="constrained")
    cuts = ((xs, along, along_ana, ana["x_center"], limits["widthX"],
             r"$x$ (m)", r"along flow, $y = 0$ m"),
            (ys, across, across_ana, limits["centerY"], limits["widthY"],
             r"$y$ (m)", r"across flow, $x = %.1f$ m" % ana["x_center"]))
    for c, (coord, sim_c, ana_c, center, width, xlabel, title) in enumerate(cuts):
        ax_h, ax_m = axes[0, c], axes[1, c]
        h_color = ts.VARIABLE_COLORS["h"]
        ax_h.plot(coord, sim_c["h"], color=h_color)
        ax_h.plot(coord, ana_c["h"], color=h_color, linestyle="--")
        ax_h.set_title(title)
        for key in ("norm", "x", "y", "z"):
            if key not in sim_c:
                continue
            color = ts.COMPONENT_COLORS[key]
            ax_m.plot(coord, sim_c[key], color=color)
            ax_m.plot(coord, ana_c[key], color=color, linestyle="--")
        ax_m.axhline(0.0, color="grey", linewidth=0.5)
        ax_m.set_xlim(center - width, center + width)
        ax_m.set_xlabel(xlabel)
    axes[0, 0].set_ylabel(_UNIT_H)
    axes[1, 0].set_ylabel(_UNIT_M)
    axes[0, 0].set_ylim(bottom=0.0)
    for label, ax in zip("abcd", axes.ravel()):
        ts.panel_label(ax, label)
    comp = [Line2D([], [], color=ts.VARIABLE_COLORS["h"], label=r"$h$")]
    comp += [Line2D([], [], color=ts.COMPONENT_COLORS[k], label=_TEX_COMPONENTS[k])
             for k in ("norm", "x", "y", "z") if k != "z" or with_z]
    fig.legend(handles=ts.style_handles() + comp, loc="outside lower center",
               ncol=len(comp) + 2)
    return fig


def _thesis_centerline(ts, header, samples, analytic_at, idx, center_y):
    xs, _ = _cell_centers(header)
    row, _ = _center_indices(header, center_y, 0.0)
    fig, axes = plt.subplots(2, 1, figsize=ts.size(1.0, 0.55), sharex=True,
                             layout="constrained")
    x_lo, x_hi = np.inf, -np.inf
    ramps = [ts.time_ramp(ts.VARIABLE_COLORS[k], len(idx)) for k in ("h", "norm")]
    for n, i in enumerate(idx):
        t = samples["t"][i]
        field = sample_field(samples, i)
        ana = analytic_at(t)
        for ax, ramp, sim_v, ana_v in ((axes[0], ramps[0], field["h"], ana["h"]),
                                       (axes[1], ramps[1], field["h"] * field["speed"],
                                        ana["h"] * ana["speed"])):
            ax.plot(xs, sim_v[row], color=ramp[n])
            ax.plot(xs, ana_v[row], color=ramp[n], linestyle="--")
        occupied = np.where((field["h"][row] > 0) | (ana["h"][row] > 0))[0]
        if len(occupied):
            x_lo = min(x_lo, xs[occupied[0]])
            x_hi = max(x_hi, xs[occupied[-1]])
    handles, labels = ts.time_handles(samples["t"][idx], ramps)
    if np.isfinite(x_lo):
        pad = 0.03 * (x_hi - x_lo)
        axes[1].set_xlim(x_lo - pad, x_hi + pad)
    axes[0].set_ylabel(_UNIT_H)
    axes[1].set_ylabel(r"$h\,\Vert\mathbf{u}\Vert$ (m$^2$ s$^{-1}$)")
    axes[1].set_xlabel(r"$x$ (m)")
    for ax in axes:
        ax.set_ylim(bottom=0.0)
    ts.panel_label(axes[0], "a")
    ts.panel_label(axes[1], "b")
    style = ts.style_handles()
    fig.legend(handles=style + handles, labels=[h.get_label() for h in style] + labels,
               loc="outside lower center", ncol=len(handles) + 2,
               **ts.time_legend_kwargs(len(ramps)))
    return fig


def _thesis_contours(ts, header, samples, analytic_at, idx):
    from matplotlib.lines import Line2D
    xs, ys = _cell_centers(header)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=ts.size(1.0, 0.5), layout="constrained")
    handles = []
    occupied = np.zeros(x_mesh.shape, dtype=bool)
    fractions = (0.1, 0.6)
    for color, i in zip(ts.time_ramp(ts.VARIABLE_COLORS["h"], len(idx)), idx):
        t = samples["t"][i]
        h_sim = np.nan_to_num(blur3x3(samples["h"][i]))
        ana = analytic_at(t)
        peak = float(np.nanmax(ana["h"]))
        if peak <= 0:
            continue
        levels = [f * peak for f in fractions]
        ax.contour(x_mesh, y_mesh, h_sim, levels=levels, colors=color,
                   linewidths=(1.2, 0.8))
        ax.contour(x_mesh, y_mesh, ana["h"], levels=levels, colors=color,
                   linewidths=(1.2, 0.8), linestyles="dashed")
        ax.plot(ana["x_center"], 0.0, marker="+", color=color, markersize=6)
        occupied |= (ana["h"] > 0) | (h_sim > fractions[0] * peak)
        handles.append(Line2D([], [], color=color, label=r"$t = %.0f$ s" % t))
    if occupied.any():
        pad = 10.0
        ax.set_xlim(x_mesh[occupied].min() - pad, x_mesh[occupied].max() + pad)
        ax.set_ylim(y_mesh[occupied].min() - pad, y_mesh[occupied].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    fig.legend(handles=ts.style_handles() + handles, loc="outside lower center",
               ncol=len(handles) + 2)
    return fig


def _thesis_momentum_maps(ts, header, field, ana, limits, with_z):
    xs, ys = _cell_centers(header)
    keys = ("x", "y", "z") if with_z else ("x", "y")
    sim_c = momentum_components(field, with_z)
    ana_c = momentum_components(ana, with_z)
    x_c, y_c = ana["x_center"], limits["centerY"]
    wx, wy = limits["widthX"], limits["widthY"]
    in_x = (xs >= x_c - wx) & (xs <= x_c + wx)
    in_y = (ys >= y_c - wy) & (ys <= y_c + wy)
    aspect = wy / wx
    fig, axes = plt.subplots(len(keys), 3, sharex=True, sharey=True,
                             figsize=ts.size(1.0, min(1.2, 0.26 * len(keys) * aspect + 0.1)),
                             layout="constrained", squeeze=False)
    for r, key in enumerate(keys):
        sim_w = sim_c[key][np.ix_(in_y, in_x)]
        ana_w = ana_c[key][np.ix_(in_y, in_x)]
        diff = sim_w - ana_w
        vmax = max(float(np.nanmax(np.abs(sim_w))), float(np.nanmax(np.abs(ana_w))), 1e-9)
        dmax = max(float(np.nanmax(np.abs(diff))), 1e-9)
        for c, (values, lim, cmap) in enumerate(((sim_w, vmax, "RdBu_r"),
                                                  (ana_w, vmax, "RdBu_r"),
                                                  (diff, dmax, "PuOr_r"))):
            ax = axes[r, c]
            im = ax.pcolormesh(xs[in_x], ys[in_y], values, cmap=cmap, vmin=-lim,
                               vmax=lim, shading="nearest", rasterized=True)
            ax.set_aspect("equal")
            ax.grid(False)
            if c == 1:
                fig.colorbar(im, ax=axes[r, :2], shrink=0.9, pad=0.01,
                             label=_TEX_COMPONENTS[key] + r" (m$^2$ s$^{-1}$)")
            elif c == 2:
                fig.colorbar(im, ax=axes[r, 2], shrink=0.9, pad=0.01,
                             label=r"$\Delta$" + _TEX_COMPONENTS[key])
        axes[r, 0].set_ylabel(r"$y$ (m)")
    for c, title in enumerate(("simulation", "analytical",
                               "simulation $-$ analytical")):
        axes[0, c].set_title(title)
        axes[-1, c].set_xlabel(r"$x$ (m)")
    for n, ax in enumerate(axes.ravel()):
        ts.panel_label(ax, "abcdefghi"[n])
    return fig


def sample_field(samples, i):
    """fields_from_grid-like dict of the i-th stored sample."""
    keys = ("h", "u", "v", "vx", "vy", "vz", "speed")
    return {k: (samples[k][i] if samples.get(k) is not None else None) for k in keys}


def write_error_table(out_path, times, errors):
    """CSV of all error norms vs time (for pgfplots / tables in LaTeX)."""
    keys = [k for k in ("h_l2rel", "h_lmaxrel", "vh_l2rel", "vh_lmaxrel",
                        "vhz_l2rel", "vhz_lmaxrel", "h_l2", "h_lmax", "vh_l2",
                        "vh_lmax", "vhz_l2", "vhz_lmax") if k in errors]
    data = np.column_stack([np.asarray(times)] + [np.asarray(errors[k]) for k in keys])
    np.savetxt(out_path, data, delimiter=",", header=",".join(["t"] + keys),
               comments="", fmt="%.6g")
    return out_path


def plot_thesis_figures(header, samples, analytic_at, errors, t_highlight,
                        center_y, out_dir, with_z, scale_coef=1.02):
    """Thesis-ready figures of the similarity solution test (see thesis_style).

    - profiles: h and momentum components along / across the flow at
      t_highlight (h u_z for the 3D model)
    - centerline_evolution: h and h|u| along y = 0 at several times (one
      light-to-dark ramp per variable, see thesis_style.time_ramp)
    - contours_evolution: 10 % / 60 % analytic-peak contours at several times
      (ramp of the h colour)
    - errors: relative L2 / Lmax errors vs time
    - momentum_maps: momentum components (x, y[, z]) simulation / analytical /
      difference at t_highlight
    plus simisol_errors.csv. Returns the PNG paths (PDFs are written next to
    them, in <out_dir>/thesis).
    """
    import pathlib
    import thesis_style as ts

    out_dir = pathlib.Path(out_dir) / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(samples["t"])
    i_save = int(np.argmin(np.abs(times - t_highlight)))
    field = sample_field(samples, i_save)
    ana = analytic_at(times[i_save])
    limits = get_plot_limits(field, 0.0, center_y, header, scale_coef)
    idx = ts.pick_times(times)
    tag = "t%02.0f" % times[i_save]
    paths = []
    with ts.style():
        paths.append(ts.save(_thesis_profiles(ts, header, field, ana, limits, with_z),
                             out_dir, f"simisol_profiles_{tag}"))
        paths.append(ts.save(_thesis_centerline(ts, header, samples, analytic_at, idx,
                                                center_y),
                             out_dir, "simisol_centerline_evolution"))
        paths.append(ts.save(_thesis_contours(ts, header, samples, analytic_at, idx),
                             out_dir, "simisol_contours_evolution"))
        paths.append(ts.save(ts.error_figure(times, errors, with_z,
                                             r"momentum $h\,\mathbf{u}$"
                                             + (" (3D vector)" if with_z else "")),
                             out_dir, "simisol_errors"))
        paths.append(ts.save(_thesis_momentum_maps(ts, header, field, ana, limits, with_z),
                             out_dir, f"simisol_momentum_maps_{tag}"))
    write_error_table(out_dir / "simisol_errors.csv", times, errors)
    return paths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def assemble_gif(png_paths, out_path, duration_ms=500, max_width=1500):
    """Stitch the per-time-step figures (sorted by time) into a looping GIF.

    Used for the contour comparison sequence (AvaFrame's plotSequence
    animation). Frames are downscaled to `max_width` pixels to keep the GIF
    a reasonable size; the individual full-resolution PNGs are kept next to
    it. Returns the GIF path, or None if pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        print("WARNING: pillow not available - keeping the individual PNGs, "
              "no GIF was written")
        return None
    if len(png_paths) == 0:
        return None
    frames = []
    for path in png_paths:
        frame = Image.open(path).convert("RGB")
        if frame.width > max_width:
            frame = frame.resize((max_width, round(frame.height * max_width / frame.width)),
                                 Image.LANCZOS)
        frames.append(frame)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print("animation saved to %s (%d frames)" % (out_path, len(frames)))
    return out_path


def _save_fig(fig, out_dir, name):
    import pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name + ".png")
    fig.savefig(path, dpi=150)
    print("plot saved to %s" % path)
    plt.close(fig)
    return path
