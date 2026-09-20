"""Energy line test analysis and plots for avalanchers, adapted from AvaFrame.

Port of ``avaframe/ana1Tests/energyLineTest.py`` (runout-angle, alpha-line
intersection and energy-height analysis plus the three-panel result figure).
The AvaFrame version reads com1DFA output files; this port consumes the
in-memory results of an avalanchers simulation directly (see the companion
script ``runEnergyLineTest.py``):

- ``ava_profile_mass``: mass-averaged center-of-mass path sampled live from
  the particle buffers while the simulation advances, with keys
  ``x``, ``y``, ``z``, ``u2`` (mass-averaged squared velocity) and ``s``
  (cumulative path length), one value per sample
- ``dem``: dict with ``rasterData`` (2D array, row 0 = south) and ``header``
  with ``cellsize``, ``ncols``, ``nrows``, ``xmin``, ``ymin``, ``xmax``, ``ymax``
- ``fields``: dict with the ``pfv`` peak flow velocity field

Only numpy + matplotlib are required - no avaframe imports. The analysis
functions are faithful ports; the plot reproduces the AvaFrame layout:
bird's-eye view (peak flow velocity on DEM hillshade + center-of-mass path),
profile with the simulated energy line vs the analytic alpha line, and a
zoomed profile with the error measures.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
import matplotlib.patheffects as pe

# plot constants mirroring avaframe/out3Plot/plotUtilsCfg.ini
FIG_W = 6
FIG_H = 6
MS = 5              # marker size
FS = 12             # font size
UNIT_FT = "m"       # unit used on the velocity-altitude colorbar
UNIT_FV = "m/s"     # unit of the peak flow velocity
PLOT_BUFFER = 25.0  # padding (meters) around the data in the bird's-eye view

# "pfv" colormap: AvaFrame plots peak velocities with the (reversed) batlow
# scientific colormap, discretized with the speedColorLevels
# (avaframe/out3Plot/plotUtils.py: colorsS + speedColorLevels)
PFV_COLORS = ["#FFCEF4", "#FFA7A8", "#C19A1B", "#578B21", "#007054", "#004960", "#201158"]
PFV_LEVELS = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

HILLSHADE_AZIMUTH = 315.0  # light azimuth (degrees), plotUtilsCfg.ini
HILLSHADE_ALTITUDE = 15.0  # light altitude (degrees), plotUtilsCfg.ini
HILLSHADE_VERT_EXAG = 10.0
HILLSHADE_CONTOUR_LEVELS = 15


def make_color_map(vmin, vmax, colors=PFV_COLORS, levels=PFV_LEVELS):
    """Discrete colormap and norm between vmin and vmax.

    Port of the discrete branch of ``pU.makeColorMap`` with the AvaFrame
    speed colormap: the static color/level lists are truncated at vmax, the
    values below the first level are drawn white.
    """
    levels = np.asarray(levels, dtype=float)
    inside = np.where(levels >= vmax)[0]
    ind_end = int(inside[0]) if inside.size else len(levels)
    ind_end = max(ind_end, 1)
    levels_new = list(levels[:ind_end]) + [float(vmax)]
    colors_new = list(colors[:ind_end]) + [colors[ind_end - 1]]
    cmap = ListedColormap(colors_new)
    cmap.set_bad(color="w")
    cmap.set_under(color="w")
    if len(levels_new) == 1:
        levels_new = [0.0] + levels_new
    norm = BoundaryNorm(levels_new, cmap.N)
    return cmap, norm


def constrain_plots_to_data(data, header, buffer_m=PLOT_BUFFER):
    """Bounding box (map coordinates) of the cells where data > 0, padded.

    Port of ``pU.constrainPlotsToData`` (extentOption=True), but the limits
    are returned in the DEM's real-world coordinates instead of indices.
    """
    cs = header["cellsize"]
    plot_buffer = int(buffer_m / cs)
    ind = np.where(data > 0)
    if len(ind[0]) > 0:
        row_min = max(int(np.amin(ind[0])) - plot_buffer, 0)
        row_max = min(int(np.amax(ind[0])) + plot_buffer, data.shape[0] - 1)
        col_min = max(int(np.amin(ind[1])) - plot_buffer, 0)
        col_max = min(int(np.amax(ind[1])) + plot_buffer, data.shape[1] - 1)
    else:
        row_min, row_max = 0, data.shape[0] - 1
        col_min, col_max = 0, data.shape[1] - 1
    return (header["xmin"] + col_min * cs, header["xmin"] + col_max * cs,
            header["ymin"] + row_min * cs, header["ymin"] + row_max * cs)


def get_runout_angle(ava_profile_mass, ind_start=0, ind_end=-1):
    """Compute the center of mass runout angle (port of getRunOutAngle).

    Returns
    --------
    runout_angle_rad, runout_angle_deg
    """
    delta_s = ava_profile_mass["s"][ind_end] - ava_profile_mass["s"][ind_start]
    delta_z = ava_profile_mass["z"][ind_end] - ava_profile_mass["z"][ind_start]
    runout_angle_rad = np.arctan(np.abs(delta_z / delta_s))
    return runout_angle_rad, np.rad2deg(runout_angle_rad)


def get_alpha_profile_intersection(cfg, ava_profile_mass, mu, csz):
    """Extend the profile and intersect it with the alpha line of slope mu.

    Port of getAlphaProfileIntersection: the profile is extended with a line
    whose slope is a regression on the last ``nCellsExtrapolation`` * csz
    meters of the path, until the line of slope mu starting at z[0] crosses it.

    Returns
    --------
    slope_ext, s_intersection, z_intersection, coef_ext
    """
    n_cells_extrapolation = cfg["nCellsExtrapolation"]
    idx_extra = np.nanmin(np.argwhere(
        ava_profile_mass["s"][-1] - ava_profile_mass["s"] < n_cells_extrapolation * csz))
    p1 = np.polyfit(ava_profile_mass["s"][idx_extra:], ava_profile_mass["z"][idx_extra:], 1)
    slope_ext = p1[0]
    # first check if the intersection is on the not extended profile
    s = ava_profile_mass["s"]
    z = ava_profile_mass["z"]
    alpha_line = z[0] - s * mu
    idx = np.argwhere(np.diff(np.sign(z - alpha_line))).flatten()
    if idx.size == 0:
        raise RuntimeError("the alpha line does not intersect the path profile - "
                           "the avalanche did not reach the altitude z0 - mu * s?")
    idx = idx[-1]
    coef_ext = 0
    # intersection not found on the profile, extend it step by step
    while s[idx] == 0 and coef_ext < 4:
        s = np.append(ava_profile_mass["s"], (1 + coef_ext) * ava_profile_mass["s"][-1])
        z = np.append(ava_profile_mass["z"],
                      ava_profile_mass["z"][-1] + coef_ext * slope_ext * ava_profile_mass["s"][-1])
        alpha_line = z[0] - s * mu
        idx = np.argwhere(np.diff(np.sign(z - alpha_line))).flatten()[-1]
        coef_ext = coef_ext + 1
    # exact intersection point within the segment
    s0, s1 = s[idx], s[idx + 1]
    z_p0, z_p1 = z[idx], z[idx + 1]
    z_a0, z_a1 = alpha_line[idx], alpha_line[idx + 1]
    s_intersection = s0 + (s1 - s0) * (z_a0 - z_p0) / ((z_p1 - z_p0) - (z_a1 - z_a0))
    z_intersection = z_p0 + (s_intersection - s0) * (z_p1 - z_p0) / (s1 - s0)
    if coef_ext > 0:
        s[-1] = s_intersection
        z[-1] = z_intersection
    coef_ext = max(s[-1] / ava_profile_mass["s"][-1] - 1, 0)
    return slope_ext, s_intersection, z_intersection, coef_ext


def get_energy_info(ava_profile_mass, g, mu, s_intersection, z_intersection,
                    runout_angle_deg, alpha_deg):
    """Compute energy heights and error measures (port of getEnergyInfo).

    Returns
    --------
    z_ene, u2_path, s_geom_line, z_geom_line, result_energy_test
    """
    u2_path = ava_profile_mass["u2"]
    # energy altitude of the center of mass at every sample
    z_ene = ava_profile_mass["z"] + u2_path / (2 * g)

    # analytic alpha line from the start point through the intersection
    gk = s_intersection * mu
    z_end = ava_profile_mass["z"][0] - gk
    z_ene_target = ava_profile_mass["z"][0] - ava_profile_mass["s"] * mu
    s_geom_line = [ava_profile_mass["s"][0], s_intersection]
    z_geom_line = [ava_profile_mass["z"][0], z_end]
    # errors between the simulated energy line and the alpha line
    rmse_velocity_elevation = np.sqrt(((z_ene - z_ene_target) ** 2).mean())
    runout_s_error = ava_profile_mass["s"][-1] - s_intersection
    runout_z_error = ava_profile_mass["z"][-1] - z_intersection
    runout_angle_error = runout_angle_deg - alpha_deg
    result_energy_test = {
        "zEnd": ava_profile_mass["z"][-1], "sEnd": ava_profile_mass["s"][-1],
        "runoutAngle": runout_angle_deg, "rmseVelocityElevation": rmse_velocity_elevation,
        "runOutSError": runout_s_error, "runOutZError": runout_z_error,
        "runOutAngleError": runout_angle_error}
    return z_ene, u2_path, s_geom_line, z_geom_line, result_energy_test


def _add_hillshade(ax, dem, extent):
    """DEM hillshade with contour lines under the result field (port of
    ``outCom1DFA.addDem2Plot(what='hillshade')``)."""
    raster = dem["rasterData"]
    ls = LightSource(azdeg=HILLSHADE_AZIMUTH, altdeg=HILLSHADE_ALTITUDE)
    shaded = ls.hillshade(raster, vert_exag=HILLSHADE_VERT_EXAG,
                          dx=dem["header"]["cellsize"], dy=dem["header"]["cellsize"])
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad(color="w")
    ax.imshow(shaded, cmap=cmap, vmin=0, vmax=1, extent=extent,
              origin="lower", zorder=0)
    ax.contour(raster, levels=HILLSHADE_CONTOUR_LEVELS, colors="k",
               linewidths=0.5, extent=extent, origin="lower", zorder=1)


def generate_energy_plot(ava_profile_mass, dem, fields, cfg, g, mu, sim_name,
                         out_dir, show=False):
    """Make the energy line test analysis and plot the results.

    Port of ``generateCom1DFAEnergyPlot``. Same three panels: peak flow
    velocity bird's-eye view with the center-of-mass path (bottom left),
    profile with energy line and alpha line (top), zoomed profile with the
    error measures (bottom right).

    Parameters
    -----------
    ava_profile_mass: dict
        mass averaged center of mass path (x, y, z, u2, s arrays)
    dem: dict
        dem dict (rasterData + header with cellsize/xmin/ymin/xmax/ymax)
    fields: dict
        result fields, needs the ``pfv`` peak flow velocity array
    cfg: dict
        energy line test configuration: nCellsExtrapolation, shiftedAlphaLine,
        plotScor (only used if ava_profile_mass has an 'sCor' key)
    g: float
        gravity
    mu: float
        friction coefficient (tan of the bed friction angle)
    sim_name: str
        simulation name used in the plot and the output file name
    out_dir: pathlib.Path
        directory where the figure is saved
    show: bool
        also open an interactive plot window

    Returns
    --------
    result_energy_test: dict
        runout and velocity-height error measures
    save_path: pathlib.Path
        path of the saved figure
    """
    plot_scor = cfg.get("plotScor", False) and "sCor" in ava_profile_mass
    alpha_deg = np.rad2deg(np.arctan(mu))
    csz = dem["header"]["cellsize"]
    # compute simulation run out angle
    runout_angle_rad, runout_angle_deg = get_runout_angle(ava_profile_mass)
    # extend path profile and find intersection between the alpha line and the profile
    slope_ext, s_intersection, z_intersection, coef_ext = get_alpha_profile_intersection(
        cfg, ava_profile_mass, mu, csz)
    # compute errors on runout and velocity altitude
    z_ene, u2_path, s_geom_line, z_geom_line, result_energy_test = get_energy_info(
        ava_profile_mass, g, mu, s_intersection, z_intersection, runout_angle_deg, alpha_deg)
    z0 = ava_profile_mass["z"][0]

    fig = plt.figure(figsize=(FIG_W * 3, FIG_H * 2))
    cmap, norm = make_color_map(np.amin(u2_path / (2 * g)), np.amax(u2_path / (2 * g)))
    path_style = dict(lw=1, path_effects=[pe.Stroke(linewidth=2, foreground="k"), pe.Normal()])

    # ---------------- bird's-eye view: peak velocity + dem + com path
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    pfv = fields["pfv"]
    x_min, x_max, y_min, y_max = constrain_plots_to_data(pfv, dem["header"])
    extent = [dem["header"]["xmin"], dem["header"]["xmax"],
              dem["header"]["ymin"], dem["header"]["ymax"]]
    _add_hillshade(ax1, dem, extent)
    pfv_cmap, pfv_norm = make_color_map(np.amin(pfv[pfv > 0]) if np.any(pfv > 0) else 1.0,
                                        max(np.amax(pfv), 1.0))
    # masked (no flow) cells must stay transparent so the hillshade below stays visible
    pfv_cmap.set_bad(alpha=0)
    im1 = ax1.imshow(np.ma.masked_where(pfv == 0, pfv), origin="lower", extent=extent,
                     cmap=pfv_cmap, norm=pfv_norm, zorder=9, aspect="equal")
    cbar0 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05)
    cbar0.ax.set_ylabel("peak flow velocity [%s]" % UNIT_FV)
    ax1.plot(ava_profile_mass["x"], ava_profile_mass["y"], "-y.", zorder=20,
             label="Center of mass path", **path_style)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    l1 = ax1.legend(loc="upper right")
    l1.set_zorder(40)
    ax1.text(0.02, 0.02, sim_name, transform=ax1.transAxes, fontsize=FS, zorder=40,
             bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2))

    # ---------------- profile plot, zoomed out
    ax2 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2.plot(ava_profile_mass["s"], ava_profile_mass["z"], "-y.",
             label="Center of mass altitude", **path_style)
    if plot_scor:
        ax2.plot(ava_profile_mass["sCor"], ava_profile_mass["z"], "--k.",
                 label="Center of mass altitude (corrected s)")
    # extend this curve towards the bottom with the regression slope
    ax2.plot(ava_profile_mass["s"][-1] * np.array([1, 1 + coef_ext]),
             ava_profile_mass["z"][-1] + slope_ext * ava_profile_mass["s"][-1] * np.array([0, coef_ext]),
             ":k", label="Center of mass altitude extrapolation")

    # center of mass energy line and velocity altitude points
    ax2.plot(ava_profile_mass["s"][[0, -1]], z_ene[[0, -1]], "-r",
             label="avalanchers energy line (%.2f°)" % runout_angle_deg)
    scat = ax2.scatter(ava_profile_mass["s"], z_ene, marker="s", cmap=cmap, norm=norm,
                       s=8 * MS, c=u2_path / (2 * g), label="Center of mass velocity altitude",
                       zorder=20)
    cbar2 = ax2.figure.colorbar(scat, ax=ax2, use_gridspec=True)
    cbar2.ax.set_title("[" + UNIT_FT + "]", pad=10)
    cbar2.ax.set_ylabel("Center of mass velocity altitude")

    # alpha line
    ax2.plot(s_geom_line, z_geom_line, "b-", label=r"$\alpha$ line (%.2f°)" % alpha_deg)
    if cfg.get("shiftedAlphaLine", False):
        ax2.plot(ava_profile_mass["s"],
                 ava_profile_mass["z"][-1] - (ava_profile_mass["s"] - ava_profile_mass["s"][-1]) * mu,
                 "b-.", label=r"shifted $\alpha$ line (%.2f°)" % alpha_deg)
    z_lim = ax2.get_ylim()
    s_lim = ax2.get_xlim()
    z_min_ax = z_lim[0]
    ax2.vlines(x=ava_profile_mass["s"][-1], ymin=z_min_ax, ymax=ava_profile_mass["z"][-1],
               color="r", linestyle="--")
    ax2.vlines(x=s_intersection, color="b", ymin=z_min_ax, ymax=z_intersection, linestyle="--")
    ax2.hlines(y=ava_profile_mass["z"][-1], xmin=0, xmax=ava_profile_mass["s"][-1],
               color="r", linestyle="--")
    ax2.hlines(y=z_intersection, color="b", xmin=0, xmax=s_intersection, linestyle="--")
    ax2.set_xlabel("s [m]")
    ax2.set_ylabel("z [m]")
    ax2.set_xlim(s_lim)
    ax2.set_ylim(z_lim)
    ax2.legend(loc="upper right")
    ax2.set_title("Energy line test")

    # ---------------- profile plot, zoomed in
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.plot(ava_profile_mass["s"], ava_profile_mass["z"] - z0, "-y.",
             label="Center of mass altitude", **path_style)
    if plot_scor:
        ax3.plot(ava_profile_mass["sCor"], ava_profile_mass["z"] - z0, "--k.",
                 label="Center of mass altitude (corrected s)")
    ax3.plot(ava_profile_mass["s"][-1] * np.array([1, 1 + coef_ext]),
             ava_profile_mass["z"][-1] + slope_ext * ava_profile_mass["s"][-1] * np.array([0, coef_ext]) - z0,
             ":k", label="Center of mass altitude extrapolation")

    ax3.plot(ava_profile_mass["s"][[0, -1]], z_ene[[0, -1]] - z0, "-r",
             label="avalanchers energy line (%.2f°)" % runout_angle_deg)
    scat = ax3.scatter(ava_profile_mass["s"], z_ene - z0, marker="s", cmap=cmap, norm=norm,
                       s=8 * MS, c=u2_path / (2 * g), zorder=20,
                       label="Center of mass velocity altitude extrapolation")
    cbar3 = ax3.figure.colorbar(scat, ax=ax3, use_gridspec=True)
    cbar3.ax.set_title("[" + UNIT_FT + "]", pad=10)
    cbar3.ax.set_ylabel("Center of mass velocity altitude")

    ax3.plot(s_geom_line, np.asarray(z_geom_line) - z0, "b-", label=r"$\alpha$ line (%.2f°)" % alpha_deg)
    if cfg.get("shiftedAlphaLine", False):
        ax3.plot(ava_profile_mass["s"],
                 ava_profile_mass["z"][-1] - z0 - (ava_profile_mass["s"] - ava_profile_mass["s"][-1]) * mu,
                 "b-.", label=r"shifted $\alpha$ line (%.2f°)" % alpha_deg)

    # plot limits around the runout points
    runout_s_error = result_energy_test["runOutSError"]
    runout_z_error = result_energy_test["runOutZError"]
    error_s = abs(runout_s_error)
    error_z = abs(runout_z_error)
    s_min = min(ava_profile_mass["s"][-1], s_intersection) - max(error_s, 0)
    s_max = max(ava_profile_mass["s"][-1], s_intersection) + max(error_s, 0)
    z_min = ava_profile_mass["z"][-1] + min(slope_ext * (s_max - ava_profile_mass["s"][-1]),
                                            0 - 2 * error_z) - z0
    z_max = ava_profile_mass["z"][0] - s_min * np.tan(min(runout_angle_rad, np.arctan(mu))) - z0
    if ava_profile_mass["z"][-1] == z_intersection:
        z_min = z_min - (z_max - z_min) * 0.1
    if error_z < 1e-3:
        z_min = z_min - (z_max - z_min) * 0.1
    ax3.vlines(x=ava_profile_mass["s"][-1], ymin=z_min, ymax=ava_profile_mass["z"][-1] - z0,
               color="r", linestyle="--")
    ax3.vlines(x=s_intersection, color="b", ymin=z_min, ymax=z_intersection - z0, linestyle="--")
    ax3.hlines(y=ava_profile_mass["z"][-1] - z0, xmin=s_min, xmax=ava_profile_mass["s"][-1],
               color="r", linestyle="--")
    ax3.hlines(y=z_intersection - z0, color="b", xmin=s_min, xmax=s_intersection, linestyle="--")

    ax3.set_xlabel("s [m]")
    ax3.set_ylabel("$\\Delta z$ [m]")
    ax3.yaxis.set_label_coords(0, 0.9)
    ax3.set_xlim([s_min, s_max])
    ax3.set_ylim([z_min, z_max])
    ax3.tick_params(axis="x", which="major", rotation=45)
    ax3.tick_params(axis="y", which="major", rotation=45)
    ax3.set_xticks([ava_profile_mass["s"][-1], s_intersection])
    ax3.set_yticks([ava_profile_mass["z"][-1] - z0, z_intersection - z0])
    ax3.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    text = ("Runout s diff : %.4e m \nRunout z diff : %.4e m \n"
            "Runout angle diff : %.4e° \nvelocity height rmse : %.4e m \n(energy line - alpha line)"
            % (runout_s_error, runout_z_error, result_energy_test["runOutAngleError"],
               result_energy_test["rmseVelocityElevation"]))
    text_box = AnchoredText(text, frameon=False, loc=1, pad=0.5, prop=dict(fontsize=FS))
    ax3.add_artist(text_box)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_path = out_dir / (sim_name + "_EnergyTest.png")
    fig.savefig(save_path, dpi=150)
    print("plot saved to %s" % save_path)
    if show:
        plt.show()
    plt.close(fig)
    return result_energy_test, save_path
