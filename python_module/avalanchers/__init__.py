import importlib.util
import os
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

from ._avalanchers import *
from .output import AvalanchersOutput, AvalanchersScenario  # noqa: F401

__version__ = version("avalanchers")


# Calibration-relevant simulation parameters and the engine defaults used
# when a key is missing from the settings dictionary. Mirrors
# `SimSettings::new()` in crates/compute_core/src/settings.rs.
CALIBRATION_PARAMETER_DEFAULTS = {
    "sim_model": "terrain-following",
    "friction_model": "voellmy",
    "constitutive_model": "drucker-prager",
    "density": 200.0,
    "slab_thickness_factor": 1.0,
    "friction_coefficient": 0.2,
    "drag_coefficient": 2000.0,
    "n0": 70.0,
    "i0": 0.29,
    "mu0": 0.38,
    "mu2": 0.65,
    "grain_diameter": 0.002,
    "internal_friction_angle": 40.0,
    "basal_friction_angle": 25.0,
    "shear_modulus": 5.0e4,
    "hardening_modulus": 0.0,
    "bulk_modulus": 2.0e4,
    "compaction_pressure": 2.0e3,
    "released_particles_per_cell": 8,
    "cfl": 0.9,
    "cell_size": 1.0,
    "min_slope_angle": 28.0,
    "max_slope_angle": 60.0,
    "release_min_elevation": 1500.0,
    "release_max_elevation": 8848.0,
    "release_area_fraction": None,
    "outlines_padding": None,
    "velocity_threshold": 0.1,
    "roughness_threshold": 0.01,
    "peak_flow_thickness_threshold": 0.1,
    "enable_curvature": True,
    "enable_particle_interaction": True,
    "enable_earth_pressure_coefficient": False,
    "enable_entrainment": True,
    "enable_weak_pressure": False,
}

# Metric names of PyEvaluation (see crates/python_bindings/src/lib.rs),
# exported alongside the parameters. chamfer_distance is omitted because it
# duplicates sim_to_roi/roi_to_sim/chamfer.
EVALUATION_METRICS = (
    "intersection",
    "undershoot",
    "overshoot",
    "iou",
    "horizontal_distance",
    "vertical_drop",
    "beeline_3d",
    "horizontal_distance_ref",
    "vertical_drop_ref",
    "beeline_3d_ref",
    "peak_velocity",
    "sim_to_roi",
    "roi_to_sim",
    "chamfer",
)

# Aliases accepted by the FromStr implementations of the Rust settings
# enums, normalized to the first canonical name of each group.
_ENUM_ALIASES = {
    "sim_model": {
        "terrain-following": "terrain-following",
        "terrain": "terrain-following",
        "t": "terrain-following",
        "curvilinear": "curvilinear",
        "curvi": "curvilinear",
        "c": "curvilinear",
        "mpmdac": "mpmdac",
        "mpm-dac": "mpmdac",
        "mpm_depth_averaged_curvilinear": "mpmdac",
        "sliding-block": "sliding-block",
        "block": "sliding-block",
        "b": "sliding-block",
    },
    "friction_model": {
        "coulomb": "coulomb",
        "voellmy": "voellmy",
        "voellmy-min-shear": "voellmy-min-shear",
        "voellmyminshear": "voellmy-min-shear",
        "samosat": "samosat",
        "samos-at": "samosat",
    },
    "constitutive_model": {
        "drucker-prager": "drucker-prager",
        "dp": "drucker-prager",
        "mohr-coulomb": "drucker-prager",
        "mu-i": "mu-i",
        "mui": "mu-i",
        "mu(i)": "mu-i",
        "mu-i-rheology": "mu-i",
    },
}


def _normalize_enum(name, value):
    aliases = _ENUM_ALIASES[name]
    if isinstance(value, (int, np.integer)):
        choices = list(dict.fromkeys(aliases.values()))
        if 0 <= int(value) < len(choices):
            return choices[int(value)]
        raise ValueError(f"Invalid {name} index: {value}")
    key = str(value).lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown {name}: '{value}'")


def calibration_parameters(settings):
    """Resolve the calibration-relevant parameters from a settings dictionary.

    ``settings`` uses the same keys as :meth:`PySimulation.create`. Missing
    keys fall back to the engine defaults (``SimSettings::new`` in
    ``crates/compute_core/src/settings.rs``).

    Returns:
        dict: Parameter name -> value, with the model enums normalized to
        their canonical names (e.g. ``"voellmy"`` instead of ``1``).
    """
    params = {}
    for key, default in CALIBRATION_PARAMETER_DEFAULTS.items():
        value = settings.get(key, default)
        if key in _ENUM_ALIASES:
            value = _normalize_enum(key, value)
        params[key] = value
    return params


def evaluation_metrics(evaluation):
    """Extract the results of a :class:`PyEvaluation` as a flat dict.

    ``evaluation`` may be a ``PyEvaluation`` (as returned by
    :meth:`PySimulation.evaluate`) or a dict of metrics, which may contain a
    subset of the names.

    Returns:
        dict: Metric name -> value for all :data:`EVALUATION_METRICS`;
        metrics missing from a dict input are ``None``.
    """
    return {
        name: (evaluation.get(name) if isinstance(evaluation, dict) else evaluation[name])
        for name in EVALUATION_METRICS
    }


def export_calibration_parameters(settings, path, run_name=None, append=True, evaluation=None):
    """Export the parameters and results of a calibration run to a parquet file.

    Each settings dictionary (the same schema as :meth:`PySimulation.create`)
    becomes one row. Parameters that are not set explicitly are recorded
    with their engine defaults, so the file fully describes what was
    simulated. When a :class:`PyEvaluation` is given, its metrics (IoU,
    chamfer distance, runout, peak velocity, ...) are stored in the same
    row, pairing a run outcome with its parameters. By default new rows are
    appended to an existing file, letting a calibration loop grow the table
    run by run.

    Args:
        settings: Settings dict, or a list of dicts for a parameter sweep.
        path: Destination ``.parquet`` file path.
        run_name: Name (or list of names, one per settings dict) stored in
            the ``run_name`` column. Defaults to ``run_1``, ``run_2``, ...
            continuing the numbering of an existing file.
        append: When True (default) and ``path`` exists, new rows are
            appended; otherwise the file is overwritten.
        evaluation: :class:`PyEvaluation` as returned by
            :meth:`PySimulation.evaluate` (or a flat dict of metrics), or a
            list of these matching ``settings``. Without an evaluation the
            metric columns are written as null.

    Returns:
        pandas.DataFrame: The full parameter table written to ``path``.
    """
    try:
        import pandas as pd
        import pyarrow  # noqa: F401  (parquet engine)
    except ImportError:
        raise ImportError(
            "The 'pandas' and 'pyarrow' packages are required to export "
            "calibration parameters. Install them using: pip install pandas pyarrow"
        )

    configs = [settings] if isinstance(settings, dict) else list(settings)
    if run_name is None:
        start = len(pd.read_parquet(path, columns=["run_name"])) if append and os.path.exists(path) else 0
        run_names = [f"run_{start + i + 1}" for i in range(len(configs))]
    elif isinstance(run_name, str):
        run_names = [run_name]
    else:
        run_names = list(run_name)
    if len(run_names) != len(configs):
        raise ValueError(
            f"Got {len(run_names)} run names for {len(configs)} settings dicts"
        )

    if evaluation is None:
        evaluations = [None] * len(configs)
    elif isinstance(evaluation, (list, tuple)):
        evaluations = list(evaluation)
    else:
        evaluations = [evaluation]
    if len(evaluations) != len(configs):
        raise ValueError(
            f"Got {len(evaluations)} evaluations for {len(configs)} settings dicts"
        )

    rows = []
    for config, result in zip(configs, evaluations):
        row = calibration_parameters(config)
        if result is not None:
            row.update(evaluation_metrics(result))
        rows.append(row)

    columns = list(CALIBRATION_PARAMETER_DEFAULTS) + list(EVALUATION_METRICS)
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(0, "run_name", run_names)
    frame.insert(1, "timestamp", datetime.now(timezone.utc).isoformat())
    frame.insert(2, "avalanchers_version", __version__)

    if append and os.path.exists(path):
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame.to_parquet(path, index=False)
    return frame


def create_mesh(sim):
    dem = sim.dem
    dem_mask = dem < sim.elevation_threshold
    dem[dem_mask] = np.nan
    ny, nx = dem.shape
    x = np.arange(nx).astype(np.float32) * sim.cell_size
    y = np.arange(ny).astype(np.float32) * sim.cell_size
    x, y = np.meshgrid(x, y)
    return x, y, dem, dem_mask


def blur_nan_grid(data, passes=1):
    blurred = data.copy()
    for _ in range(max(0, int(passes))):
        valid = np.isfinite(blurred)
        values = np.where(valid, blurred, 0.0)
        weights = valid.astype(np.float32)

        values_sum = np.zeros_like(blurred, dtype=np.float32)
        weights_sum = np.zeros_like(blurred, dtype=np.float32)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                values_sum += np.roll(np.roll(values, dy, axis=0), dx, axis=1)
                weights_sum += np.roll(np.roll(weights, dy, axis=0), dx, axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            blurred = np.where(weights_sum > 0, values_sum / weights_sum, np.nan)
    return blurred


def plot3d(
    sim,
    parameter,
    particles=False,
    threshold_value=1e-3,
    blur_passes=0,
):
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "The 'pyvista' package is required for 3d visualization. "
            "Install it using: pip install 'avalanchers[viz]'"
        )
    data = getattr(sim, parameter).astype(np.float32)
    x, y, dem, dem_mask = create_mesh(sim)

    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan


    if blur_passes > 0:
        data = blur_nan_grid(data, passes=blur_passes)

    # 2. Create the StructuredGrid
    # We pass x, y, and the elevation (dem) directly as coordinates
    grid = pv.StructuredGrid(x, y, dem)
    # 3. Add the aspect data as point scalars
    # Use .flatten() to match the point count of the mesh
    grid.point_data["Elevation"] = dem.flatten(order="F")
    grid.point_data[parameter] = data.flatten(order="F")

    clipped = grid.clip_scalar(value=threshold_value, scalars="Elevation", invert=False)

    # Vertical exaggeration
    final_mesh = clipped.scale([1, 1, 1.3], inplace=False)

    # 4. Visualization
    plotter = pv.Plotter()
    plotter.add_mesh(
        final_mesh,
        scalars=parameter,
        cmap="rainbow",
        # clim=[0, 1],
        scalar_bar_args={"title": parameter.replace("_", " ").title()},
        edge_color="black",
        show_edges=False,
    )
    if particles:
        # add particles
        
        positions = sim.particles_position.copy()
        positions[:, 2] *= 1.3
        poly = pv.PolyData(positions)
        plotter.add_mesh(
            poly,
            color="red",
            point_size=3,
            render_points_as_spheres=True,
            # style='points_gaussian',
        )
    plotter.enable_eye_dome_lighting()
    plotter.show(jupyter_backend="trame") if is_jupyter() else plotter.show()
    return plotter


def plot_dem(sim, ax, dark=True):
    import_plt()
    xx, yy, dem, dem_mask = create_mesh(sim)    
    ls = LightSource(azdeg=315, altdeg=45)

    ax.imshow(
        ls.hillshade(dem),
        cmap="gray",
        alpha=0.5,
        vmin=0,
        vmax=1,
        extent=[xx.min(), xx.max(), yy.max(), yy.min()],
        origin="upper",
    )
    ax.contour(
        xx, yy, 
        dem,
        levels=np.arange(0, 5000, 100),
        colors="grey",
        linewidths=0.8,
        alpha=0.65,
    )
    # ax.clabel(cs100, inline=True, fontsize=14, fmt="%d m", use_clabeltext=True)

    # Major contours
    cs = ax.contour(
        xx, yy, 
        dem,
        levels=np.arange(0, 5000, 500),
        colors="black",
        linewidths=0.8,
        alpha=0.65,
    )
    ax.clabel(cs, inline=True, fontsize=14, fmt="%d m", use_clabeltext=True)
    ax.invert_yaxis()
    return xx, yy, dem, dem_mask


def import_plt():
    global plt, make_axes_locatable, ListedColormap, mpltPath, LightSource
    try:
        import matplotlib.path as _mpltPath
        import matplotlib.pyplot as _plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable

        # from matplotlib.colors
        plt = _plt
        make_axes_locatable = _make_axes_locatable
        ListedColormap = importlib.import_module("matplotlib.colors").ListedColormap
        LightSource = importlib.import_module("matplotlib.colors").LightSource
        mpltPath = _mpltPath
    except ImportError:
        raise ImportError(
            "The 'matplotlib' package is required for 2d plots. "
            "Install it using: pip install 'avalanchers[viz]'"
        )

def plot2d(sim, parameter, title="Avalanche Simulation", threshold_value=1e-3, particles=False, roi=False): 
    import_plt()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax, surf, x, y = ax2d(ax, sim, parameter, title, threshold_value)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(surf, ax=ax, cax=cax)
    cbar.set_label(parameter.replace("_", " ").title())
    if particles:
        positions = sim.particles_position.copy()
        velocities = sim.particles_velocity.copy()
        speed = np.linalg.norm(velocities, axis=1)
        mask = speed > 0
        ax.scatter(positions[mask, 0], positions[mask, 1], c=speed[mask], s=2, alpha=0.7, cmap='Blues')
    if roi:
        ax.contour(x, y, sim.roi, levels=[0.99], colors='red')
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="cyan", lw=2, label="Release Areas"),
                plt.Line2D([0], [0], color="red", lw=2, label="Mapped Outline")
            ]
        )
    if not is_jupyter():    
        plt.show()
    return fig, ax, x, y

def ax2d(ax, sim, parameter, title="Avalanche Simulation", threshold_value=1e-3):
    import_plt()
    data = getattr(sim, parameter).astype(np.float32)
    ax.set_aspect("equal")
    x, y, _, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan
    surf = ax.contourf(x, y, data, cmap='magma')
    ax.contour(x, y, sim.release_areas.astype(np.float32), colors='cyan', linewidths=1, alpha=0.3)
    ax.legend(
        handles=[plt.Line2D([0], [0], color="cyan", lw=1, alpha=0.8, label="Release Areas")],
    )
    ax.set(title=title)
    return ax, surf, x, y

def plot_overview(sim, threshold_value=1e-3):
    # Setup parameters, titles, and distinct colormaps
    params = ['peak_velocity', 'peak_flow_thickness']
    colormaps = ['magma', 'viridis']
    
    import_plt()

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    for i, (param, cmap) in enumerate(zip(params, colormaps)):
        ax = axes[i]
        ax.set_aspect("equal")

        # 1. Plot DEM in the background
        x, y, _dem, dem_mask = plot_dem(sim, ax, dark=False)

        # 2. Prepare Data
        data = getattr(sim, param).astype(np.float32)
        data[dem_mask] = np.nan

        # Apply thresholds
        data[data < threshold_value] = np.nan
            
        if param == "cell_count":
            data = np.log10(data)
        
        # 3. Plot Simulation Results
        surf = ax.contourf(x, y, data, cmap=cmap)

        # 4. Plot Release Areas (Cyan Outline)
        ax.contour(
            x,
            y,
            sim.release_areas.astype(np.float32),
            colors="cyan",
            linewidths=1,
            alpha=0.3,
        )

        # 5. Configure Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(surf, cax=cax)
        cbar.set_label(param.replace("_", " ").title())

        ax.set_title(param.replace("_", " ").title(), fontsize=14, fontweight="bold")
        ax.set_xlabel("X-Coordinate")
        if i == 0:
            ax.set_ylabel("Y-Coordinate")

    plt.tight_layout()
    plt.show()
    return fig, axes


def calculate_dice(model_a, model_b):
    """
    Calculates the Sørensen-Dice coefficient for two binary numpy arrays.

    Args:
        model_a (np.ndarray): Binary mask of the Reference Model.
        model_b (np.ndarray): Binary mask of the Proposed Model.

    Returns:
        float: Dice coefficient (0.0 to 1.0)
    """
    # Ensure the arrays are boolean for logical operations
    mask_a = model_a > 0
    mask_b = model_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    total_area = mask_a.sum() + mask_b.sum()

    if total_area == 0:
        return 1.0  # Both models agree on "no runout"

    return (2.0 * intersection) / total_area


def plot_comparison_binary(sim, parameter, reference_array, threshold_value=1, title="Avalanche Simulation Comparison"):
    import_plt()
    data = getattr(sim, parameter).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y, _dem, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan
    only_reference = ~(data > 0) & (reference_array > 0)
    only_sim = (data > 0) & ~(reference_array > 0)
    both = (data > 0) & (reference_array > 0)
    comparison = np.zeros_like(data, dtype=int)
    comparison[only_reference] = 1
    comparison[both] = 2
    comparison[only_sim] = 3
    cmap = ListedColormap(["red", "magenta", "blue"])
    cont = ax.contourf(
        x,
        y,
        comparison,
        cmap=cmap,
        levels=[0.5, 1.5, 2.5, 3.5],
        alpha=0.7,
        antialiased=False,
    )
    cbar = fig.colorbar(cont, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8, aspect=10)
    cbar.ax.set_yticklabels(["No avalanche", "reference only", "both", "sim only"])
    dice = calculate_dice(reference_array, data)
    ax.set_title(title + f"\nDice coefficient: {dice:.4f}")
    print(f"Dice coefficient: {dice:.4f}")
    return fig, ax

def plot_comparison(sim, parameter, reference_array, title="Avalanche Simulation Comparison"):
    import_plt()
    data = getattr(sim, parameter).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y, _dem, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    diff = reference_array - data
    diff[diff == 0] = np.nan
    # diff[(data == 0) | (reference_array == 0)] = np.nan
    max_abs = np.nanmax(np.abs(diff))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    levels = np.linspace(-max_abs, max_abs, 21)
    cont = ax.contourf(x, y, diff, cmap="bwr", levels=levels)
    cbar = fig.colorbar(cont, ax=ax, shrink=0.8, aspect=10)
    cbar.ax.set_yticklabels(["No avalanche", "reference only", "both", "sim only"])
    dice = calculate_dice(reference_array, data)
    ax.set_title(title + f"\nDice coefficient: {dice:.4f}")
    print(f"Dice coefficient: {dice:.4f}")
    return fig, ax


def is_jupyter():
    try:
        from IPython import get_ipython

        # ZMQInteractiveShell is the standard Jupyter kernel
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, NameError):
        return False


async def setup_jupyter_3d():
    if importlib.util.find_spec("pyvista") is None:
        raise ImportError(
            "The 'pyvista' package is required for this feature. "
            "Install it using: pip install 'avalanchers[viz]'"
        )
    if is_jupyter():
        try:
            from pyvista.trame.jupyter import launch_server

            print("Jupyter detected: Launching PyVista Trame server...")
            await launch_server().ready
        except ImportError:
            print(
                "Warning: pyvista or trame not installed. Visualization may be limited."
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to start Trame server: {e}")
    else:
        print("Standard environment detected: Skipping Jupyter server launch.")
