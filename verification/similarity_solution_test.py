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

The numerical flow fields are decoded DIRECTLY from the simulation's
quantized ``grid_mass`` / ``grid_momentum`` buffers (the same fields the
grid physics pass consumed); the mass/momentum quantization factors MUST
match ``crates/compute_core/src/shaders/utils.wgsl``.

Deviation from the AvaFrame momentum profiles: com1DFA fields carry a
vertical velocity component (h*vz curves), avalanchers' depth-averaged
fields do not - the twin-axis momentum panels show |h*u|, h*ux and h*uy
(in-plane) for simulation and analytical solution.

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

# p2g quantization constants - MUST match crates/compute_core/src/shaders/utils.wgsl.
MASS_FACTOR = 1e1       # p2g stores round(mass * MASS_FACTOR) as u32
MOMENTUM_FACTOR = 1e2   # p2g stores round(mass * velocity * MOMENTUM_FACTOR) as i32

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


def fields_from_grid(sim, shape, cell_size, density, jac):
    """Decode the sim's grid mass/momentum getters into thickness + velocity.

    These are exactly the fields the grid physics pass consumed this step.
    Returns flow thickness (normal to the surface, m), along-slope and
    cross-slope velocity, and the dequantized grid mass, all as
    (nrows, ncols) arrays, row 0 = south.
    """
    nrows, ncols = shape
    mass = np.asarray(sim.grid_mass, dtype=np.float64) * (1.0 / MASS_FACTOR)
    mom = np.asarray(sim.grid_momentum, dtype=np.float64) * (1.0 / MOMENTUM_FACTOR)
    mom = mom.reshape((nrows * ncols, 2))
    # mirror the shader's decode: nodes below the mass quantization floor are empty
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
        print(f"WARNING: grid mass ({grid_total:.1f} kg) differs from particle mass "
              f"({particle_total:.1f} kg) by {100.0 * rel:.2f}% - MASS_FACTOR/"
              f"MOMENTUM_FACTOR no longer match crates/compute_core/src/shaders/utils.wgsl")


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
    """L2/LMax of an in-plane vector field given as (fx, fy) arrays."""
    local = (analytic[0] - numerical[0]) ** 2 + (analytic[1] - numerical[1]) ** 2
    ref = analytic[0] ** 2 + analytic[1] ** 2
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
    return {"h": h, "u": u, "v": v, "x_center": x_center}


# ---------------------------------------------------------------------------
# plots - ported from avaframe/out3Plot/outAna1Plots.py (simiSol section)
# ---------------------------------------------------------------------------

def get_plot_limits(h_field, speed, center_x, center_y, header, scale_coef):
    """Plot limits around the flow field (port of getPlotLimits with
    extentOption=True: widths are in METERS, half the field extent)."""
    cs = header["cellsize"]
    ind = h_field > 0
    if ind.any():
        cols = np.where(ind.any(axis=0))[0]
        rows = np.where(ind.any(axis=1))[0]
        width_x = scale_coef * ((cols.max() + 1) - cols.min()) * cs / 2.0
        width_y = scale_coef * ((rows.max() + 1) - rows.min()) * cs / 2.0
    else:
        width_x = width_y = 50.0
    max_ft = scale_coef * float(np.nanmax(h_field))
    max_fm = scale_coef * float(np.nanmax(h_field * speed))
    return {"widthX": width_x, "widthY": width_y,
            "maxFT": max_ft, "minMz": 0.0, "maxFM": max_fm,
            "centerX": center_x, "centerY": center_y}


def _plot_variable_panel(ax, axis, coordinates, sim_curves, ana_curves,
                         x_center, limits):
    """Twin-axis profile panel (port of _plotVariable): flow thickness on the
    left axis (sim full / analytic dashed), momentum components on the right
    axis (|h*u| green, h*ux magenta, h*uy blue; sim full / analytic dashed).

    `axis` is "xaxis" (coordinates = x array, cut along the center row) or
    "yaxis" (coordinates = y array, cut at the center column).
    """
    h_sim, hu_sim, hux_sim, huy_sim = sim_curves
    h_ana, hu_ana, hux_ana, huy_ana = ana_curves
    ax2 = ax.twinx()

    if axis == "xaxis":
        ax.axvline(x=x_center, linestyle=":", color="b")
    l1, = ax.plot(coordinates, h_sim, "k", label="h")
    l2, = ax2.plot(coordinates, hu_sim, "g", label=r"$\vert h \mathbf{\bar{u}} \vert$")
    l3, = ax2.plot(coordinates, hux_sim, "m", label=r"$h \bar{u}_x$")
    l4, = ax2.plot(coordinates, huy_sim, "b", label=r"$h \bar{u}_y$")
    ax.plot(coordinates, h_ana, "--k")
    ax2.plot(coordinates, hu_ana, "--g")
    ax2.plot(coordinates, hux_ana, "--m")
    ax2.plot(coordinates, huy_ana, "--b")

    if axis == "xaxis":
        ax.set_xlabel("x in [m]")
        ax.text(x_center + 5, -0.05, "x = %.2f m" % x_center, color="b")
        ax.set_xlim([x_center - limits["widthX"], x_center + limits["widthX"]])
    else:
        ax.set_xlabel("y in [m]")
        ax.set_xlim([limits["centerY"] - limits["widthY"],
                     limits["centerY"] + limits["widthY"]])
    ax.set_ylim([-0.05, limits["maxFT"]])
    ax2.set_ylim([limits["minMz"], limits["maxFM"]])
    ax.set_ylabel(NAME_FT + " [" + UNIT_FT + "]")
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    color = "tab:green"
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(color="tab:green", linestyle="-", linewidth=0.25, alpha=0.5)
    ax2.set_ylabel(r"$\vert h \mathbf{\bar{u}} \vert$ [" + UNIT_FTV + "]", color=color)
    lns = [l1, l2, l3, l4]
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=FS - 2)
    text = "analytical solution (dashed line) \n numerical solution (full line)"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment="top",
            fontsize=FS - 2)
    return ax, ax2


def _profile_curves(field, h_ana, center_row, center_col):
    """Extract the along-flow and across-flow profile curves."""
    speed = np.hypot(field["u"], field["v"])
    along = (field["h"][center_row], (field["h"] * speed)[center_row],
             (field["h"] * field["u"])[center_row], (field["h"] * field["v"])[center_row])
    across = (field["h"][:, center_col], (field["h"] * speed)[:, center_col],
              (field["h"] * field["u"])[:, center_col], (field["h"] * field["v"])[:, center_col])
    speed_ana = np.hypot(h_ana["u"], h_ana["v"])
    along_ana = (h_ana["h"][center_row], (h_ana["h"] * speed_ana)[center_row],
                 (h_ana["h"] * h_ana["u"])[center_row], (h_ana["h"] * h_ana["v"])[center_row])
    across_ana = (h_ana["h"][:, center_col], (h_ana["h"] * speed_ana)[:, center_col],
                  (h_ana["h"] * h_ana["u"])[:, center_col],
                  (h_ana["h"] * h_ana["v"])[:, center_col])
    return along, across, along_ana, across_ana


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
    xs = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * header["cellsize"]
    ys = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * header["cellsize"]
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    x_center = h_ana["x_center"]
    y_center = limits["centerY"]
    contour_levels = np.linspace(0, limits["maxFT"], n_levels)[:-1]
    cmap = LinearSegmentedColormap.from_list("cmapT", FT_COLORS)
    norm = Normalize(vmin=float(np.nanmin(field_ft)), vmax=float(np.nanmax(field_ft)))
    cs1 = ax.contour(x_mesh, y_mesh, field_ft, levels=contour_levels, cmap=cmap,
                     norm=norm, linewidths=2)
    cs2 = ax.contour(x_mesh, y_mesh, h_ana["h"], levels=contour_levels, cmap=cmap,
                     norm=norm, linewidths=2, linestyles="dashed")
    cb = ax.figure.colorbar(cs1, ax=ax, shrink=0.8, pad=0.05)
    cb.ax.set_title(NAME_FT + " [" + UNIT_FT + "]", fontsize=FS - 3, pad=8)
    h1, _ = cs1.legend_elements()
    h2, _ = cs2.legend_elements()
    ax.legend([h1[-1], h2[-1]], ["simulation", "analytical"], fontsize=FS - 2,
              loc="upper right")
    ax.set_ylim([y_center - limits["widthY"], y_center + limits["widthY"]])
    ax.set_xlim([x_center - limits["widthX"], x_center + limits["widthX"]])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axvline(x_center, color="b", linestyle=":")
    ax.axhline(y_center, color="r", linestyle=":")
    ax.text(x_center + 5, y_center - limits["widthY"] + 5, "x = %.2f m" % x_center,
            color="b", fontsize=FS - 2)
    ax.text(x_center - limits["widthX"] + 5, y_center + 5, "y = 0 m", color="r",
            fontsize=FS - 2)
    return ax


def add_error_time(ax1, times, h_err_l2, h_err_lmax, vh_err_l2, vh_err_lmax,
                   relative, t_save):
    """Error-vs-time panel with twin axes (port of addErrorTime)."""
    title = "Relative error difference" if relative else "Error difference"
    ax2 = ax1.twinx()
    ax1.plot(times, h_err_l2, "k-", label="h L2 error")
    ax1.plot(times, h_err_lmax, "k--", label="h LMax error")
    ax1.set_xlabel("time in [s]")
    ax1.set_ylabel(title + " on h")
    ax1.legend(loc="upper left", fontsize=FS - 2)
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)

    color = "tab:green"
    ax2.plot(times, vh_err_l2, "g-", label=r"$\vert h \mathbf{\bar{u}} \vert$ L2 error")
    ax2.plot(times, vh_err_lmax, "g--", label=r"$\vert h \mathbf{\bar{u}} \vert$ LMax error")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylabel(title + r" on $\vert h \mathbf{\bar{u}} \vert$", color=color)
    ax2.legend(loc="lower right", fontsize=FS - 2)
    ax2.grid(color="tab:green", linestyle="-", linewidth=0.25, alpha=0.5)

    ax1.axvline(t_save, color="grey", linestyle="--")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    min_y, _ = ax1.get_ylim()
    ax1.text(t_save, min_y, "%.2f s" % t_save)
    return ax1, ax2


def plot_error_time(times, h_err_l2, h_err_lmax, vh_err_l2, vh_err_lmax,
                    relative, t_save, sim_name, out_dir):
    """Standalone error-vs-time figure (port of plotErrorTime)."""
    fig1, ax1 = plt.subplots(figsize=(2 * FIG_W, 2 * FIG_H))
    title = " between analytical solution and avalanchers \n(simulation %s)" % sim_name
    ax1.set_title(("Relative error difference" if relative else "Error difference") + title)
    add_error_time(ax1, times, h_err_l2, h_err_lmax, vh_err_l2, vh_err_lmax,
                   relative, t_save)
    _save_fig(fig1, out_dir, "error_time_" + sim_name)


def plot_profile_comparison(header, field, h_ana, limits, t, sim_name, out_dir):
    """Standalone profile comparison figure at one time step (port of
    saveSimiSolProfile): flow-direction and across-flow twin-axis profiles."""
    center_row = int(np.argmin(np.abs(header["ymin"]
                                       + (np.arange(header["nrows"]) + 0.5)
                                       * header["cellsize"] - limits["centerY"])))
    center_col = int(np.argmin(np.abs(header["xmin"]
                                       + (np.arange(header["ncols"]) + 0.5)
                                       * header["cellsize"] - h_ana["x_center"])))
    xs = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * header["cellsize"]
    ys = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * header["cellsize"]
    along, across, along_ana, across_ana = _profile_curves(field, h_ana,
                                                           center_row, center_col)
    fig = plt.figure(figsize=(4 * FIG_W, 2 * FIG_H))
    fig.suptitle("Similarity solution test, t = %.2f s (simulation %s)" % (t, sim_name),
                 fontsize=16)
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    _plot_variable_panel(ax1, "xaxis", xs, along, along_ana,
                         h_ana["x_center"], limits)
    ax1.set_title("Profile in flow direction (y = 0 m)")
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    _plot_variable_panel(ax2, "yaxis", ys, across, across_ana,
                         h_ana["x_center"], limits)
    ax2.set_title("Profile across flow direction (x = %.2f m)" % h_ana["x_center"])
    fig.tight_layout()
    return _save_fig(fig, out_dir, "compare_profile_simi_sol_%s_%07.2f" % (sim_name, t))


def plot_contour_comparison(header, field_ft, h_ana, limits, t, sim_name, out_dir):
    """Standalone contour comparison figure at one time step (port of
    makeContourSimiPlot)."""
    fig, ax1 = plt.subplots(figsize=(4 * FIG_W, 2 * FIG_H))
    add_contour_2plot(ax1, field_ft, h_ana, header, limits, n_levels=16)
    ax1.set_title(NAME_FT + " contours at t = %.2f s" % t)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "compare_contour_simi_sol_%s_%07.2f" % (sim_name, t))


def plot_simi_sol_summary(header, field, h_ana, limits, errors, times, t_save,
                          cfg, sim_name, settings, out_dir):
    """Summary figure (port of plotSimiSolSummary): 2x6 grid with the two
    twin-axis profile panels, the FT contour comparison, the error curves
    and the parameter text."""
    relative = cfg["relative_error"]
    center_row = int(np.argmin(np.abs(header["ymin"]
                                       + (np.arange(header["nrows"]) + 0.5)
                                       * header["cellsize"] - limits["centerY"])))
    center_col = int(np.argmin(np.abs(header["xmin"]
                                       + (np.arange(header["ncols"]) + 0.5)
                                       * header["cellsize"] - h_ana["x_center"])))
    xs = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * header["cellsize"]
    ys = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * header["cellsize"]
    along, across, along_ana, across_ana = _profile_curves(field, h_ana,
                                                           center_row, center_col)
    common = dict(limits=limits)

    fig = plt.figure(figsize=(FIG_W * 4, FIG_H * 2))
    fig.suptitle("Similarity solution test, t = %.2f s (simulation %s)"
                 % (t_save, sim_name), fontsize=20)
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
    _plot_variable_panel(ax1, "xaxis", xs, along, along_ana,
                         h_ana["x_center"], **common)
    ax1.set_title("Profile in flow direction (y = 0 m)")
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
    _plot_variable_panel(ax2, "yaxis", ys, across, across_ana,
                         h_ana["x_center"], **common)
    ax2.set_title("Profile across flow direction (x = %.2f m)" % h_ana["x_center"])

    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    add_contour_2plot(ax3, field["h"], h_ana, header, limits)

    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    title = ("Relative error difference" if relative else "Error difference") \
        + "\nbetween analytical solution and avalanchers"
    ax4.set_title(title)
    add_error_time(ax4, times, errors["h_l2"], errors["h_lmax"],
                   errors["vh_l2"], errors["vh_lmax"], relative, t_save)

    ax7 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax7.axis("off")
    ax7.invert_yaxis()
    text = (f"simulation = {sim_name}\n"
            f"sim_model = {settings.get('sim_model')}\n"
            f"zeta = {cfg['zeta']:.2f} deg\ndelta = {cfg['delta']:.2f} deg\n"
            f"phi = {cfg['phi']:.2f} deg\nL_x = {cfg['l_x']:.1f} m\n"
            f"L_y = {cfg['l_y']:.1f} m\nrelTh = {cfg['rel_th']:.1f} m\n"
            f"density = {settings.get('density')}\n"
            f"cfl = {settings.get('cfl')}\n"
            f"particles/cell = {settings.get('released_particles_per_cell')}\n")
    ax7.text(0.5, 0.5, text, transform=ax7.transAxes, ha="center", va="center",
             fontsize=FS)
    _save_fig(fig, out_dir, sim_name + "_SimiSolTest")


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
