"""Dam break test analysis and plots for avalanchers, adapted from AvaFrame.

Port of the AvaFrame dam break test chain - ``avaframe/ana1Tests/damBreak.py``
(Faccanoni & Mangeney 2012, Test 2, case 1.2: instantaneous release of a
granular column on a dry inclined plane with Coulomb friction),
``avaframe/ana1Tests/analysisTools.py`` (L2/LMax error norms) and the
dam-break plots of ``avaframe/out3Plot/outAna1Plots.py`` - so it runs on the
in-memory results of an avalanchers simulation (see the companion script
``run_dam_break_test.py``). No avaframe imports, no output files.

The numerical flow thickness and velocity fields are derived DIRECTLY from
the simulation's ``grid_mass`` / ``grid_momentum`` fields (the same fields
the grid physics pass consumed this step), not re-scattered from particles.

Conventions: DEM arrays are (nrows, ncols) with row 0 = south; the flow
frame is a plane fit of the DEM (descent direction + slope angle); all
analytic-solution coordinates are along-slope, relative to the dam front.
"""

import math
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Rectangle

# plot constants mirroring avaframe/out3Plot/plotUtilsCfg.ini
FIG_W = 6
FIG_H = 6
FS = 12
NAME_FT = "flow thickness"
NAME_FV = "flow velocity"
NAME_HV = "flow hv"
UNIT_FT = "m"
UNIT_FV = "m/s"
UNIT_HV = r"$m^2s^{-1}$"

# "FT" colormap of AvaFrame: plotUtils.py colorsT (lajolla). Used as a
# CONTINUOUS 0-1.0 m colorscale for the peak flow thickness map (AvaFrame's
# discrete thicknessColorLevels were 0.5|1.0|2.0|3.0|4.0|5.0).
FT_COLORS = ["#FCFFC9", "#EBCE7B", "#DE9529", "#BE5A32", "#7F2B3F", "#1D0B14"]

PLOT_Y_BUFFER = 25.0  # padding (meters) around the flowed cells in the map panel


# ---------------------------------------------------------------------------
# Simulation grid fields, read from the grid mass/momentum buffers
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


def flow_frame(dem, header):
    """Slope angle and horizontal descent direction from a plane DEM fit.

    Returns a dict with phi_deg, cos_phi and e_down (unit vector, horizontal,
    pointing downhill). The avaDamBreak plane descends along +x.
    """
    cs = header["cellsize"]
    x = header["xmin"] + (np.arange(dem.shape[1]) + 0.5) * cs
    y = header["ymin"] + (np.arange(dem.shape[0]) + 0.5) * cs
    x_map, y_map = np.meshgrid(x, y)
    finite = np.isfinite(dem)
    zx = np.polyfit(x_map[finite], dem[finite], 1)[0]
    zy = np.polyfit(y_map[finite], dem[finite], 1)[0]
    slope = math.hypot(zx, zy)
    return {
        "phi_deg": math.degrees(math.atan(slope)),
        "cos_phi": math.cos(math.atan(slope)),
        "e_down": (-zx / slope, -zy / slope),
    }


def flow_coordinates(header, frame):
    """Horizontal flow (f) and cross-flow (p) coordinates of the cell centers."""
    cs = header["cellsize"]
    x = header["xmin"] + (np.arange(header["ncols"]) + 0.5) * cs
    y = header["ymin"] + (np.arange(header["nrows"]) + 0.5) * cs
    x_map, y_map = np.meshgrid(x, y)
    ex, ey = frame["e_down"]
    return x_map * ex + y_map * ey, -x_map * ey + y_map * ex


def fields_from_grid(sim, shape, cell_size, density, jac, frame=None,
                     surface_velocity=False):
    """Decode the sim's grid mass/momentum getters into thickness + velocity.

    These are exactly the fields the grid physics pass consumed this step
    (including p2g quantization and its boundary guards), not a python
    re-scattering of the particles. Returns flow thickness (normal to the
    surface, m), velocity along the flow direction and across it, and the
    grid mass, all as (nrows, ncols) arrays, row 0 = south.
    """
    nrows, ncols = shape
    mass = np.asarray(sim.grid_mass, dtype=np.float64)
    mom = np.asarray(sim.grid_momentum, dtype=np.float64)
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
    if frame is not None:
        # rotate the grid velocity into the flow frame (flow / cross-flow); a
        # horizontal dx/dt needs the metric to measure surface distance, a
        # surface velocity (see SURFACE_VELOCITY_MODELS) already does
        ex, ey = frame["e_down"]
        u_flow = vel_u * ex + vel_v * ey
        if not surface_velocity:
            u_flow = u_flow / frame["cos_phi"]
        v_cross = -vel_u * ey + vel_v * ex
        return thickness, u_flow, v_cross, mass
    return thickness, vel_u, vel_v, mass


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
    """L2 and LMax (absolute + relative) of the difference of two scalars."""
    local = (analytic - numerical) ** 2
    return _error_and_norm(local, analytic * analytic, cell_size, cos_angle)


# ---------------------------------------------------------------------------
# Analytic solution - ported from avaframe/ana1Tests/damBreak.py
# (Faccanoni & Mangeney 2012, Test 2, case 1.2, dry bed)
# ---------------------------------------------------------------------------

def dam_break_solution(phi_deg, delta_deg, h_l, times, s_rel, g=9.81, x_back_rel=None):
    """Analytic dam break over a dry inclined plane.

    Along-slope coordinate ``s_rel`` is relative to the dam front (body
    extends upstream to ``x_back_rel`` < 0). Returns (h, u, x_mid) with h, u
    shaped (len(times),) + np.shape(s_rel) and x_mid the along-slope middle
    of the material per time (start of the AvaFrame error domain).
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    gz = g * math.cos(phi)
    m0 = gz * (math.tan(phi) - math.tan(delta))
    c_l = math.sqrt(gz * h_l)

    s_rel = np.asarray(s_rel, dtype=float)
    h = np.zeros((len(times),) + s_rel.shape)
    u = np.zeros((len(times),) + s_rel.shape)
    x_mid = np.zeros(len(times))
    for mi, t in enumerate(times):
        cond1 = (m0 * t / 2.0 - c_l) * t
        cond2 = (2.0 * c_l + m0 * t / 2.0) * t
        cond2_r = x_back_rel + (m0 * t / 2.0 + c_l) * t
        x_mid[mi] = (2.0 * cond2_r + cond1) / 3.0
        if t <= 0:
            # initial dam of height h_l between x_back_rel and 0
            h[mi] = np.where((s_rel >= x_back_rel) & (s_rel <= 0.0), h_l, 0.0)
            continue
        uu = np.where(cond2 >= s_rel, (2.0 / 3.0) * (c_l + s_rel / t + m0 * t), 0.0)
        hh = np.where(cond2 >= s_rel,
                      (2.0 * c_l - s_rel / t + m0 * t / 2.0) ** 2 / (9.0 * gz), 0.0)
        uu = np.where(cond1 >= s_rel, m0 * t, uu)
        hh = np.where(cond1 >= s_rel, h_l, hh)
        h[mi] = hh
        u[mi] = uu
    return h, u, x_mid


FRONT_THRESHOLDS = (1e-3, 1e-2, 5e-2)


def front_position(s_rel, h, threshold):
    """Most downstream along-slope position where h reaches `threshold`,
    interpolated linearly toward the next thinner sample; nan if none."""
    order = np.argsort(s_rel)
    s_rel = np.asarray(s_rel, dtype=float)[order]
    h = np.asarray(h, dtype=float)[order]
    above = np.nonzero(h >= threshold)[0]
    if len(above) == 0:
        return math.nan
    i = above[-1]
    if i + 1 >= len(h):
        return float(s_rel[i])
    return float(s_rel[i] + (s_rel[i + 1] - s_rel[i]) * (h[i] - threshold) / (h[i] - h[i + 1]))


def front_lags(frame, dam, cfg, h, t, thresholds=FRONT_THRESHOLDS):
    """Numerical minus analytic front position (along-slope m) on the profile
    row for each thickness contour; negative = the numerical front lags."""
    s_row = frame["s_rel_profile"]
    s_fine = np.arange(np.min(s_row), np.max(s_row), 0.05)
    h_ana = dam_break_solution(frame["phi_deg"], cfg["delta"], dam["h_l"], [t], s_fine,
                               cfg["grav_acc"], x_back_rel=dam["x_back_rel"])[0][0]
    h_row = h[frame["row_index"]]
    return {thr: front_position(s_row, h_row, thr) - front_position(s_fine, h_ana, thr)
            for thr in thresholds}


def dam_break_rear_edge(phi_deg, delta_deg, h_l, t, g, x_back_rel):
    """Along-slope position of the physical rear edge of the material.

    Downstream of it the case-1.2 solution is valid; UPSTREAM of it the
    formal uniform extension (h = h_l) covers the already vacated dam area,
    which AvaFrame's figures draw as-is. The comparison plots clip the
    analytic curves at this edge so the animation stays physically readable.
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    gz = g * math.cos(phi)
    m0 = gz * (math.tan(phi) - math.tan(delta))
    c_l = math.sqrt(gz * h_l)
    return x_back_rel + (m0 * t / 2.0 - 2.0 * c_l) * t


# ---------------------------------------------------------------------------
# Plots - ported from avaframe/out3Plot/outAna1Plots.py
# ---------------------------------------------------------------------------

def _initial_dam_profile(f, h_l, f_back, f_front):
    """Grey dashed initial dam shape for the profile plots."""
    y = np.zeros_like(f)
    y[(f >= f_back) & (f <= f_front)] = h_l
    return y


def _plot_dam_profile(ax, f, y_initial, data_initial, data_num, f_ana, y_ana,
                      f_mid, y_max, label, unit, f_end, scale_coef, x_lim):
    """One profile comparison panel (port of _plotDamProfile)."""
    ax.plot(f, y_initial, "grey", linestyle="--")
    ax.plot(f, data_initial, "k--", label="initial")
    ax.plot(f, data_num, "b", label="numerical")
    ax.plot(f_ana, y_ana, "r-", label="analytical")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("%s [%s]" % (label, unit))
    ax.set_xlim(x_lim)
    ax.set_ylim([-0.05, max(y_max, scale_coef * np.nanmax(y_ana))])
    ax.axvspan(f_mid, f_end, color="grey", alpha=0.3, lw=0,
               label="error computation \n domain")
    return ax


def plot_comparison_dam(sample0, sample_t, frame, dam, t, cfg, sim_name, out_dir):
    """1x3 comparison cross-cut figure (FT / FV / hv profiles) at one time
    step (port of plotComparisonDam). Returns the saved figure path."""
    f = frame["f_profile"]
    f_ana_row = frame["f_profile"]
    s_rel_row = frame["s_rel_profile"]
    h_ana, u_ana, x_mid = dam_break_solution(
        frame["phi_deg"], cfg["delta"], dam["h_l"], [t], s_rel_row, cfg["grav_acc"],
        x_back_rel=dam["x_back_rel"])
    h_ana, u_ana = h_ana[0], u_ana[0]
    # clip the analytic curves at the physical rear edge: upstream of it the
    # solution formally extends h = h_l over the vacated dam area, where no
    # material exists (AvaFrame draws this artifact; we clip it so the
    # late-time animation frames stay readable)
    rear = dam_break_rear_edge(frame["phi_deg"], cfg["delta"], dam["h_l"], t,
                               cfg["grav_acc"], dam["x_back_rel"])
    behind = s_rel_row < rear
    h_ana = np.where(behind, np.nan, h_ana)
    u_ana = np.where(behind, np.nan, u_ana)
    f_mid = dam["f_front"] + x_mid[0] * frame["cos_phi"]
    f_end = dam["f_front"] + cfg["x_end"]
    row = frame["row_index"]
    speed0 = np.hypot(sample0["u"][row], sample0["v"][row])
    speed_t = np.hypot(sample_t["u"][row], sample_t["v"][row])
    h0_row = sample0["h"][row]
    h_t_row = sample_t["h"][row]
    y_initial = _initial_dam_profile(f, dam["h_l"], dam["f_back"], dam["f_front"])
    common = dict(f_end=f_end, scale_coef=cfg["scale_coef"], x_lim=tuple(cfg["x_lim"]))
    ylim = frame.get("ylim", {})  # fixed limits across all frames if provided

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True,
                             figsize=(FIG_W * 4, FIG_H * 2))
    _plot_dam_profile(axes[0], f, y_initial, h0_row, h_t_row,
                      f_ana_row, h_ana, f_mid,
                      ylim.get("ft", frame["limits"]["max_ft"]),
                      NAME_FT, UNIT_FT, **common)
    axes[0].set_title(NAME_FT + " profile")
    _plot_dam_profile(axes[1], f, y_initial * 0, speed0, speed_t,
                      f_ana_row, u_ana, f_mid,
                      ylim.get("fv", frame["limits"]["max_fv"]),
                      NAME_FV, UNIT_FV, **common)
    axes[1].set_title(NAME_FV + " profile")
    axes[1].legend(loc="upper left")
    _plot_dam_profile(axes[2], f, y_initial * 0, h0_row * speed0,
                      h_t_row * speed_t, f_ana_row, h_ana * u_ana,
                      f_mid, ylim.get("fm", frame["limits"]["max_fm"]),
                      NAME_HV, UNIT_HV, **common)
    axes[2].set_title(r"$h \mathbf{\bar{u}}$ profile")
    fig.suptitle("Simulation %s, t = %.2f s" % (sim_name, t))
    return _save_fig(fig, out_dir, "compare_dam_break_%s_%07.2f" % (sim_name, t))


def add_error_time(ax1, times, h_err_l2, h_err_lmax, vh_err_l2, vh_err_lmax,
                   relative, t_save):
    """Error-vs-time panel with twin axes (port of addErrorTime)."""
    title = "Relative error difference" if relative else "Error difference"
    ax2 = ax1.twinx()
    ax1.plot(times, h_err_l2, "k-", label="h L2 error")
    ax1.plot(times, h_err_lmax, "k--", label="h LMax error")
    ax1.set_xlabel("time in [s]")
    ax1.set_ylabel(title + " on h")
    ax1.legend(loc="upper left")
    ax1.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)

    color = "tab:green"
    ax2.plot(times, vh_err_l2, "g-", label=r"$\vert h \mathbf{\bar{u}} \vert$ L2 error")
    ax2.plot(times, vh_err_lmax, "g--", label=r"$\vert h \mathbf{\bar{u}} \vert$ LMax error")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylabel(title + r" on $\vert h \mathbf{\bar{u}} \vert$", color=color)
    ax2.legend(loc="lower right")
    ax2.grid(color="tab:green", linestyle="-", linewidth=0.25, alpha=0.5)

    ax1.axvline(t_save, color="grey", linestyle="--")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    # timestamp just above the bottom spine (x in data coords, y in axes coords)
    trans = blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.text(t_save, 0.03, "%.2f s" % t_save, transform=trans, va="bottom",
             ha="center", color="grey", fontsize=FS - 2)
    return ax1, ax2


def plot_dam_break_summary(dem, header, frame, dam, samples, errors, t_save,
                           cfg, sim_name, settings, out_dir):
    """Summary figure (port of plotDamBreakSummary): 2x6 grid with the
    FT / FV / hv profiles on top and (bird's-eye map with the error domain
    rectangle, error vs time, parameter text) below.

    The bird's-eye map (bottom left) shows the PEAK flow thickness up to the
    highlighted time (``samples["peak_h"]``: running per-cell maximum of the
    decoded grid fields over the saved time steps) on a CONTINUOUS colorscale
    from 0 to 1.0 m; all panels use the fixed x window cfg["x_lim"].
    """
    ind_t = int(min(np.searchsorted(samples["t"], t_save), len(samples["t"]) - 1))
    t_plot = samples["t"][ind_t]
    sample0 = {k: samples[k][0] for k in ("h", "u", "v")}
    sample_t = {k: samples[k][ind_t] for k in ("h", "u", "v")}
    h_ana_row, u_ana_row, x_mid = dam_break_solution(
        frame["phi_deg"], cfg["delta"], dam["h_l"], [t_plot],
        frame["s_rel_profile"], cfg["grav_acc"], x_back_rel=dam["x_back_rel"])
    h_ana_row, u_ana_row = h_ana_row[0], u_ana_row[0]
    # clip the analytic curves at the physical rear edge (see
    # plot_comparison_dam): upstream of it no material exists
    rear = dam_break_rear_edge(frame["phi_deg"], cfg["delta"], dam["h_l"], t_plot,
                               cfg["grav_acc"], dam["x_back_rel"])
    behind = frame["s_rel_profile"] < rear
    h_ana_row = np.where(behind, np.nan, h_ana_row)
    u_ana_row = np.where(behind, np.nan, u_ana_row)
    f_mid = dam["f_front"] + x_mid[0] * frame["cos_phi"]
    f_end = dam["f_front"] + cfg["x_end"]
    row = frame["row_index"]
    speed0 = np.hypot(sample0["u"][row], sample0["v"][row])
    speed_t = np.hypot(sample_t["u"][row], sample_t["v"][row])
    h0_row = sample0["h"][row]
    h_t_row = sample_t["h"][row]
    f = frame["f_profile"]
    y_initial = _initial_dam_profile(f, dam["h_l"], dam["f_back"], dam["f_front"])
    common = dict(f_end=f_end, scale_coef=cfg["scale_coef"], x_lim=tuple(cfg["x_lim"]))

    fig = plt.figure(figsize=(FIG_W * 4, FIG_H * 2))
    fig.suptitle("DamBreak test, t = %.2f s (simulation %s)" % (t_plot, sim_name),
                 fontsize=20)
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    _plot_dam_profile(ax1, f, y_initial, h0_row, h_t_row, f, h_ana_row,
                      f_mid, frame["limits"]["max_ft"], NAME_FT, UNIT_FT, **common)
    ax1.set_title(NAME_FT + " profile")
    ax3 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    _plot_dam_profile(ax3, f, y_initial * 0, speed0, speed_t, f, u_ana_row,
                      f_mid, frame["limits"]["max_fv"], NAME_FV, UNIT_FV, **common)
    ax3.set_title(NAME_FV + " profile")
    plt.legend(loc="upper left")
    ax2 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    _plot_dam_profile(ax2, f, y_initial * 0, h0_row * speed0,
                      h_t_row * speed_t, f, h_ana_row * u_ana_row,
                      f_mid, frame["limits"]["max_fm"], NAME_HV, UNIT_HV, **common)
    ax2.set_title(r"$h \mathbf{\bar{u}}$ profile")

    # bird's-eye view: PEAK flow thickness on the DEM slope + error domain
    ax6 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    extent = [header["xmin"], header["xmax"], header["ymin"], header["ymax"]]
    nz = 1.0 / jacobian(dem, header["cellsize"])  # z component of the slope normal
    # Greys_r: dark = steep, light = flat (matplotlib "Greys" runs white->black)
    cmap_grey = plt.get_cmap("Greys_r").copy()
    cmap_grey.set_bad(color="w")
    ax6.imshow(np.ma.masked_invalid(nz), cmap=cmap_grey, vmin=0.4, vmax=1.0,
               extent=extent, origin="lower", zorder=0, aspect="equal")
    h_map = samples["peak_h"][ind_t] if "peak_h" in samples else sample_t["h"]
    h_plot = np.ma.masked_where(h_map == 0, h_map)
    # continuous colorscale from 0 to 1.0 m for the peak field (thin tails
    # stay visible; values above 1 m clamp to the top color)
    ft_cmap = LinearSegmentedColormap.from_list("peak_ft", FT_COLORS)
    ft_cmap.set_bad(alpha=0)
    ft_norm = Normalize(vmin=0.0, vmax=1.0)
    im = ax6.imshow(h_plot, origin="lower", extent=extent, cmap=ft_cmap,
                    norm=ft_norm, zorder=9, aspect="equal")
    cbar = fig.colorbar(im, ax=ax6, shrink=0.8, pad=0.05)
    cbar.ax.set_title("peak " + NAME_FT + " [" + UNIT_FT + "]", fontsize=FS - 3, pad=8)
    half_width = (cfg["y_end"] - cfg["y_start"]) / 2.0
    # error domain rectangle, drawn in the rotated flow frame
    ex, ey = frame["e_down"]
    angle = math.degrees(math.atan2(ey, ex))
    p_lo = dam["p_center"] - half_width
    corner_x = f_mid * ex - p_lo * ey
    corner_y = f_mid * ey + p_lo * ex
    rect = Rectangle((corner_x, corner_y), f_end - f_mid, 2.0 * half_width,
                     linewidth=3, linestyle="dashed", edgecolor="None",
                     facecolor="gray", alpha=0.2, zorder=200, angle=angle,
                     label="error computation domain")
    ax6.add_patch(rect)
    ax6.set_xlabel("x [m]")
    ax6.set_ylabel("y [m]")
    # fixed x window; fit the y window around the flowed cells so the equal
    # aspect is satisfied by the axes box instead of by moving the x limits
    ax6.set_xlim(cfg["x_lim"])
    flowed = np.where(h_map > 0)
    if len(flowed[0]) > 0:
        cs = header["cellsize"]
        y_lo = header["ymin"] + max(flowed[0].min() * cs - PLOT_Y_BUFFER, 0.0)
        y_hi = header["ymin"] + min((flowed[0].max() + 1) * cs + PLOT_Y_BUFFER,
                                    header["ymax"] - header["ymin"])
        ax6.set_ylim(y_lo, y_hi)
    ax6.set_aspect("equal", adjustable="box")
    leg = ax6.legend(loc="upper right")
    leg.set(zorder=200)

    # error vs time
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
    title = ("Relative error difference" if cfg["relative_error"]
             else "Error difference") + "\nbetween dam break solution and avalanchers"
    ax4.set_title(title)
    add_error_time(ax4, samples["t"], errors["h_l2"], errors["h_lmax"],
                   errors["vh_l2"], errors["vh_lmax"], cfg["relative_error"],
                   t_plot)

    # parameter text
    ax7 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax7.axis("off")
    ax7.invert_yaxis()
    text = (f"simulation = {sim_name}\n"
            f"sim_model = {settings.get('sim_model')}\n"
            f"phi = {frame['phi_deg']:.2f}\n"
            f"delta = {cfg['delta']:.2f}\n"
            f"mu = {math.tan(math.radians(cfg['delta'])):.4f}\n"
            f"h0 = {dam['h_l']:.2f}\n"
            f"density = {settings.get('density')}\n"
            f"cfl = {settings.get('cfl')}\n"
            f"particles/cell = {settings.get('released_particles_per_cell')}\n"
            f"cell size = {header['cellsize']:g} m\n")
    ax7.text(0.5, 0.5, text, transform=ax7.transAxes, ha="center", va="center",
             fontsize=FS)
    # extra horizontal room so the rotated error-axis labels do not collide
    # with the colorbar of the map panel
    fig.subplots_adjust(wspace=0.55, left=0.06, right=0.98, top=0.86, bottom=0.09)
    _save_fig(fig, out_dir, sim_name + "_DamBreakTest")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_dir(out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_fig(fig, out_dir, name):
    out_dir = _ensure_dir(out_dir)
    path = out_dir / (name + ".png")
    fig.savefig(path, dpi=150)
    print("plot saved to %s" % path)
    plt.close(fig)
    return path


def assemble_gif(png_paths, out_path, duration_ms=500, max_width=1500,
                 cleanup=False):
    """Stitch per-time-step figures (sorted by time) into a looping GIF.

    The frames are the comparison cross-cut figures of all saved time steps
    (AvaFrame's animated plotSequence figure). Frames are downscaled to
    `max_width` pixels to keep the GIF a reasonable size; with `cleanup` the
    individual frame PNGs are deleted once the GIF is written. Returns the
    GIF path, or None if pillow is unavailable.
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
    if cleanup:
        for path in png_paths:
            pathlib.Path(path).unlink(missing_ok=True)
    print("animation saved to %s (%d frames)" % (out_path, len(frames)))
    return out_path


# ---------------------------------------------------------------------------
# thesis figures - 15 cm text width, Computer Modern, PDF + PNG
# ---------------------------------------------------------------------------

def _thesis_dam_profiles(ts, frame, dam, samples, cfg, idx):
    """h, |u| and h|u| along the profile row at several times against the
    analytic solution; along-slope coordinate relative to the dam front.

    The analytic curves are drawn only downstream of the material mid-point
    x_mid (start of the error domain): upstream, the semi-infinite-dam
    solution keeps a uniform h_l reservoir the finite dam does not have."""
    from matplotlib.lines import Line2D

    row = frame["row_index"]
    s_row = frame["s_rel_profile"]
    s_fine = np.linspace(s_row.min(), s_row.max(), 2000)
    fig, axes = plt.subplots(3, 1, figsize=ts.size(1.0, 0.85), sharex=True,
                             layout="constrained")
    for ax in axes[:1]:
        ax.fill_between([dam["x_back_rel"], 0.0], 0.0, dam["h_l"], color="0.85",
                        linewidth=0, label="initial dam")
    handles = []
    occupied_hi = 0.0
    for color, i in zip(ts.SERIES_COLORS, idx):
        t = samples["t"][i]
        h_ana, u_ana, x_mid = dam_break_solution(frame["phi_deg"], cfg["delta"],
                                                 dam["h_l"], [t], s_fine, cfg["grav_acc"],
                                                 x_back_rel=dam["x_back_rel"])
        behind = s_fine < x_mid[0]
        axes[0].plot(x_mid[0], 0.0, marker="|", markersize=8, color=color)
        h_ana = np.where(behind, np.nan, h_ana[0])
        u_ana = np.where(behind | (h_ana <= 0), np.nan, u_ana[0])
        h = samples["h"][i][row]
        speed = np.hypot(samples["u"][i][row], samples["v"][i][row])
        speed = np.where(h > 0, speed, np.nan)
        for ax, sim_v, ana_v in ((axes[0], h, h_ana), (axes[1], speed, u_ana),
                                 (axes[2], h * np.nan_to_num(speed), h_ana * u_ana)):
            ax.plot(s_row, sim_v, color=color)
            ax.plot(s_fine, ana_v, color=color, linestyle="--")
        front = np.where(h > 1e-3)[0]
        if len(front):
            occupied_hi = max(occupied_hi, s_row[front[-1]])
        occupied_hi = max(occupied_hi, float(np.nanmax(np.where(h_ana > 0, s_fine, np.nan))))
        handles.append(Line2D([], [], color=color, label=r"$t = %s$ s" % _time_label(t)))
    pad = 0.04 * (occupied_hi - dam["x_back_rel"])
    axes[2].set_xlim(dam["x_back_rel"] - pad, occupied_hi + pad)
    axes[0].set_ylabel(r"$h$ (m)")
    axes[1].set_ylabel(r"$\Vert\mathbf{u}\Vert$ (m s$^{-1}$)")
    axes[2].set_ylabel(r"$h\,\Vert\mathbf{u}\Vert$ (m$^2$ s$^{-1}$)")
    axes[2].set_xlabel(r"along-slope distance from the dam front $s$ (m)")
    for label, ax in zip("abc", axes):
        ax.set_ylim(bottom=0.0)
        ts.panel_label(ax, label, loc=(0.015, 0.93))
    initial = plt.Rectangle((0, 0), 1, 1, color="0.85", label="initial dam")
    fig.legend(handles=ts.style_handles() + [initial] + handles,
               loc="outside lower center", ncol=4)
    return fig


def _time_label(t):
    """Simulated time for labels, to 0.1 s without a trailing .0 (15.5, 20)."""
    return "%g" % round(float(t), 1)


def _time_tag(t):
    """Time for file names: t20, t15p5 (no dot, for LaTeX includegraphics)."""
    return "t" + _time_label(t).replace(".", "p")


def _thesis_dam_map(ts, dem, header, frame, dam, samples, cfg, i_t):
    """Flow thickness at one time in the slope frame of the profile figure:
    along-slope distance s from the dam front and cross-slope distance from
    the dam axis, with the error domain, the profile line and the analytical
    front."""
    t = samples["t"][i_t]
    h = samples["h"][i_t]
    cos_phi = frame["cos_phi"]
    s = (frame["f"] - dam["f_front"]) / cos_phi
    q = frame["p"] - dam["p_center"]
    flowed = h > 1e-3
    _, _, x_mid = dam_break_solution(frame["phi_deg"], cfg["delta"], dam["h_l"], [t],
                                     np.zeros(1), cfg["grav_acc"],
                                     x_back_rel=dam["x_back_rel"])
    s_mid = x_mid[0]
    s_end = cfg["x_end"] / cos_phi
    # analytical front of the dry-bed solution (dam_break_solution: cond2)
    gz = cfg["grav_acc"] * cos_phi
    m0 = gz * (math.tan(math.radians(frame["phi_deg"])) - math.tan(math.radians(cfg["delta"])))
    s_front = (2.0 * math.sqrt(gz * dam["h_l"]) + m0 * t / 2.0) * t
    half_width = (cfg["y_end"] - cfg["y_start"]) / 2.0

    pad = 15.0
    s_lo = min(s[flowed].min(), dam["x_back_rel"]) - pad
    s_hi = max(s[flowed].max(), s_end, s_front) + pad
    q_lo = min(q[flowed].min(), -half_width) - pad
    q_hi = max(q[flowed].max(), half_width) + pad
    aspect = (q_hi - q_lo) / (s_hi - s_lo)
    fig, ax = plt.subplots(figsize=ts.size(1.0, min(0.9, aspect + 0.14)),
                           layout="constrained")
    # plane bed: a hillshade would only show the DEM quantization steps
    ax.set_facecolor("0.93")
    im = ax.pcolormesh(s, q, np.ma.masked_where(~flowed, h), shading="nearest",
                       cmap="viridis", zorder=2, rasterized=True)
    ax.add_patch(plt.Rectangle((s_mid, -half_width), s_end - s_mid, 2.0 * half_width,
                               fill=False, edgecolor=ts.VERMILION, linestyle="--",
                               linewidth=1.0, zorder=3, label="error domain"))
    ax.axhline(q[frame["row_index"], 0], color=ts.BLACK, linestyle=":", linewidth=0.9,
               zorder=3, label="profile line")
    ax.axvline(s_front, color=ts.BLACK, linestyle="-.", linewidth=0.9, zorder=3,
               label="analytical front")
    ax.set_xlim(s_lo, s_hi)
    ax.set_ylim(q_lo, q_hi)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel(r"along-slope distance from the dam front $s$ (m)")
    ax.set_ylabel(r"cross-slope distance (m)")
    ax.set_title(r"$t = %s$ s" % _time_label(t))
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01, label=r"$h$ (m)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="none")
    return fig


def plot_thesis_figures(dem, header, frame, dam, samples, errors, cfg, out_dir):
    """Thesis figures of the dam break test (see thesis_style): profiles at
    several times, flow thickness map at cfg["t_save"], relative errors over
    time; PDF + PNG in <out_dir>/thesis plus dambreak_errors.csv. Returns the
    PNG paths."""
    import thesis_style as ts

    out_dir = pathlib.Path(out_dir) / "thesis"
    times = np.asarray(samples["t"])
    i_t = int(np.argmin(np.abs(times - cfg["t_save"])))
    idx = [i for i in ts.pick_times(times) if times[i] > 0]
    with ts.style():
        paths = [
            ts.save(_thesis_dam_profiles(ts, frame, dam, samples, cfg, idx),
                    out_dir, "dambreak_profiles"),
            ts.save(_thesis_dam_map(ts, dem, header, frame, dam, samples, cfg, i_t),
                    out_dir, "dambreak_map_" + _time_tag(times[i_t])),
            ts.save(ts.error_figure(times[1:], {k: np.asarray(v)[1:] for k, v in errors.items()},
                                    momentum_title=r"momentum $h\,\Vert\mathbf{u}\Vert$",
                                    momentum_symbol=r"h\,\Vert\mathbf{u}\Vert"),
                    out_dir, "dambreak_errors"),
        ]
    keys = [k for k in ("h_l2rel", "h_lmaxrel", "vh_l2rel", "vh_lmaxrel",
                        "h_l2", "h_lmax", "vh_l2", "vh_lmax") if k in errors]
    data = np.column_stack([times] + [np.asarray(errors[k]) for k in keys])
    np.savetxt(out_dir / "dambreak_errors.csv", data, delimiter=",",
               header=",".join(["t"] + keys), comments="", fmt="%.6g")
    return paths
