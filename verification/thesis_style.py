"""Shared matplotlib style for the thesis figures of the verification tests.

Matches the thesis layout (12pt book class, Computer Modern, 15 cm text
width, small captions): figures are sized in centimeters of the text width,
use Computer Modern for text and math, 9 pt labels and a colour-blind safe
palette (Okabe & Ito 2008) with one hue per variable (curves of a variable at
several times: light-to-dark ramp of that hue, see time_ramp), and are
written as vector PDF (for LaTeX) plus a 300 dpi PNG (for the markdown
report). Figures carry no overall title: the LaTeX caption names them.

Usage:
    import thesis_style as ts
    with ts.style():
        fig, ax = plt.subplots(figsize=ts.size(1.0, 0.45))
        ...
        ts.save(fig, out_dir, "name")
"""

import pathlib

import matplotlib.pyplot as plt

CM = 1.0 / 2.54
TEXT_WIDTH_CM = 15.0  # \setlength{\textwidth}{15cm} in _thesis.tex

# Okabe & Ito colour-blind safe palette
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"

# one hue per variable, shared by every figure: a single curve uses the base
# colour, curves of one variable at several times a light-to-dark ramp of it
# (time_ramp), so a variable keeps its hue across all figures
VARIABLE_COLORS = {
    "h": BLUE,          # flow thickness
    "speed": ORANGE,    # flow velocity / velocity height
    "norm": GREEN,      # momentum h|u|
    "x": VERMILION,     # momentum components
    "y": PURPLE,
    "z": BLACK,
}
COMPONENT_COLORS = {k: VARIABLE_COLORS[k] for k in ("norm", "x", "y", "z")}
# neutral colour of the solid/dashed legend entries and of time markers
# that belong to no variable
NEUTRAL = "0.45"
# one colour per case (not per time) in multi-line figures, e.g. rotation angles
SERIES_COLORS = (BLUE, GREEN, ORANGE, VERMILION, PURPLE, SKY, YELLOW)

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def style():
    """Context manager applying the thesis rcParams."""
    return plt.rc_context(RC)


def size(width_fraction=1.0, aspect=0.6):
    """Figure size (inches) for a fraction of the thesis text width."""
    width = TEXT_WIDTH_CM * CM * width_fraction
    return (width, width * aspect)


def panel_label(ax, label, loc=(0.015, 0.97)):
    """Bold (a), (b), ... panel label in the upper left corner."""
    ax.text(*loc, f"({label})", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold")


def style_handles(sim_label="simulation", ref_label="analytical"):
    """Legend entries for the solid (simulation) / dashed (reference) convention."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=NEUTRAL, linestyle="-", label=sim_label),
            Line2D([], [], color=NEUTRAL, linestyle="--", label=ref_label)]


def _mix(color, other, amount):
    import numpy as np
    from matplotlib.colors import to_rgb
    return tuple((1.0 - amount) * np.asarray(to_rgb(color)) + amount * np.asarray(to_rgb(other)))


def time_ramp(color, n, tint=0.55, shade=0.45):
    """n colours of one hue on a continuous light-to-dark scale for curves of
    one variable at n increasing times: from a light tint of `color` (earliest
    time) through `color` to a dark shade of it (latest time)."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ramp", [_mix(color, "white", tint), color, _mix(color, "black", shade)])
    return [cmap(v) for v in (np.linspace(0.0, 1.0, n) if n > 1 else [0.5])]


def time_handles(times, ramps, linestyle="-"):
    """Legend handles of the times of a time-ramp figure: one entry per time
    showing a short segment of each variable's ramp colour (pass the
    handler_map of time_legend_kwargs to the legend)."""
    from matplotlib.lines import Line2D
    handles = [tuple(Line2D([], [], color=ramp[k], linestyle=linestyle) for ramp in ramps)
               for k in range(len(times))]
    labels = [r"$t = %g$ s" % round(float(t), 1) for t in times]
    return handles, labels


def time_legend_kwargs(n_variables):
    """Legend keyword arguments drawing the tuple handles of time_handles side
    by side; wider handles for several variables."""
    from matplotlib.legend_handler import HandlerTuple
    return {"handler_map": {tuple: HandlerTuple(ndivide=None, pad=0.0)},
            "handlelength": max(2.2, 1.3 * n_variables)}


def pick_times(times, wanted=(5.0, 10.0, 15.0, 20.0)):
    """Indices of the samples nearest to the wanted times (fallback: 4 evenly
    spaced), sorted and unique."""
    import numpy as np
    times = np.asarray(times)
    picks = [t for t in wanted if t <= times[-1] + 0.5]
    if len(picks) < 2:
        picks = np.linspace(times[-1] / 4, times[-1], 4)
    return sorted({int(np.argmin(np.abs(times - t))) for t in picks})


def error_figure(times, errors, with_z=False, momentum_title=r"momentum $h\,\mathbf{u}$",
                 momentum_symbol=r"h\,\mathbf{u}"):
    """Relative L2 / Lmax errors of flow thickness (a) and momentum (b) over
    time; `errors` holds h_l2rel, h_lmaxrel, vh_l2rel, vh_lmaxrel (and
    vhz_l2rel, vhz_lmaxrel drawn as markers when with_z)."""
    fig, axes = plt.subplots(1, 2, figsize=size(1.0, 0.36), sharey=True,
                             layout="constrained")
    blue = VARIABLE_COLORS["h"]
    axes[0].plot(times, errors["h_l2rel"], color=blue, label=r"$h$, $L_2$")
    axes[0].plot(times, errors["h_lmaxrel"], color=blue, linestyle="--",
                 label=r"$h$, $L_{\max}$")
    axes[0].set_title(r"flow thickness $h$")
    green = COMPONENT_COLORS["norm"]
    axes[1].plot(times, errors["vh_l2rel"], color=green,
                 label=r"$%s$, $L_2$" % momentum_symbol)
    axes[1].plot(times, errors["vh_lmaxrel"], color=green, linestyle="--",
                 label=r"$%s$, $L_{\max}$" % momentum_symbol)
    if with_z:
        # h u_z nearly coincides with the 3D norm: markers keep both visible
        z_color = COMPONENT_COLORS["z"]
        axes[1].plot(times, errors["vhz_l2rel"], color=z_color, linestyle="none",
                     marker="o", markersize=3, markerfacecolor="none",
                     label=r"$h\,u_z$, $L_2$")
        axes[1].plot(times, errors["vhz_lmaxrel"], color=z_color, linestyle="none",
                     marker="s", markersize=3, markerfacecolor="none",
                     label=r"$h\,u_z$, $L_{\max}$")
    axes[1].set_title(momentum_title)
    axes[0].set_ylabel("relative error")
    for label, ax in zip("ab", axes):
        ax.set_yscale("log")
        ax.set_xlabel(r"$t$ (s)")
        ax.grid(which="minor", alpha=0.15)
        panel_label(ax, label)
    handles = axes[0].get_legend_handles_labels()[0] + axes[1].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    return fig


def hillshade(ax, raster, extent, cell_size, zorder=0):
    """Light grey DEM hillshade with thin contour lines as a map background."""
    import numpy as np
    from matplotlib.colors import LightSource
    shaded = LightSource(azdeg=315.0, altdeg=35.0).hillshade(
        np.nan_to_num(raster, nan=float(np.nanmin(raster))), vert_exag=3.0,
        dx=cell_size, dy=cell_size)
    # light grey range (0.65-0.97) so the result field on top stays dominant
    ax.imshow(shaded, cmap="Greys_r", vmin=-2.0, vmax=1.1, extent=extent,
              origin="lower", zorder=zorder, interpolation="bilinear")
    ax.contour(raster, levels=12, colors="0.45", linewidths=0.3, extent=extent,
               origin="lower", zorder=zorder + 0.5)


def save(fig, out_dir, name, formats=("pdf", "png")):
    """Save the figure as vector PDF + 300 dpi PNG; returns the PNG path."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path)
        paths[fmt] = path
    plt.close(fig)
    print("thesis figure saved to %s.{%s}" % (out_dir / name, ",".join(formats)))
    return paths.get("png", next(iter(paths.values())))
