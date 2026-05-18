import numpy as np
import importlib.util

from ._avalanchers import * # noqa: F403

def create_mesh(sim):
    dem = sim.dem
    dem_mask = dem < sim.elevation_threshold 
    dem[dem_mask] = np.nan
    ny, nx = dem.shape
    x = np.arange(nx).astype(np.float32) * sim.cell_size
    y = np.arange(ny).astype(np.float32) * sim.cell_size
    x, y = np.meshgrid(x, y)
    return x, y, dem, dem_mask

def plot3d(sim, parameter, particles=False, threshold_value=1e-3, particle_threshold=0):
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
    if particle_threshold > 0:
        data[sim.cell_count < particle_threshold * sim.released_particles_per_cell] = np.nan
    # data[sim.peak_flow_thickness < 1.5 / sim.released_particles_per_cell/2] = np.nan

    if parameter == "cell_count":
        data = np.log10(data)
    if parameter == "peak_velocity":
        data[sim.peak_velocity < 1] = np.nan
    if parameter == "peak_flow_thickness":
        data[sim.peak_flow_thickness < 0.5] = np.nan

    
    # 2. Create the StructuredGrid
    # We pass x, y, and the elevation (dem) directly as coordinates
    grid = pv.StructuredGrid(x, y, dem)
    # 3. Add the aspect data as point scalars
    # Use .flatten() to match the point count of the mesh
    grid.point_data["Elevation"] = dem.flatten(order='F')
    grid.point_data[parameter] = data.flatten(order='F')

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
        scalar_bar_args={'title': parameter.replace("_", " ").title()},
        edge_color="black",
        show_edges=False
    )
    if particles:
        # add particles
        positions = sim.positions.copy()
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
    plotter.show(jupyter_backend='trame') if is_jupyter() else plotter.show()
    return plotter

def plot_dem(sim, ax, dark=True):
    import_plt()
    xx, yy, dem, dem_mask = create_mesh(sim)
    cmap_contours = "Greys_r" if dark else "Greys"
    color_lines = "white" if dark else "black"
    levels_dem = np.arange(0, 4000, 200)
    ax.set_aspect('equal')
    ax.contourf(xx, yy, dem, levels=levels_dem, cmap=cmap_contours)
    CS = ax.contour(xx, yy, dem, levels=levels_dem, linewidths=.5, colors=color_lines)
    ax.clabel(CS, fontsize=10)
    return xx, yy, dem, dem_mask

def import_plt():
    global plt, make_axes_locatable, ListedColormap, mpltPath
    try:
        import matplotlib.pyplot as _plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
        import matplotlib.path as _mpltPath
        # from matplotlib.colors
        plt = _plt
        make_axes_locatable = _make_axes_locatable
        ListedColormap = importlib.import_module("matplotlib.colors").ListedColormap
        mpltPath = _mpltPath
    except ImportError:
        raise ImportError(
            "The 'matplotlib' package is required for 2d plots. "
            "Install it using: pip install 'avalanchers[viz]'"
        )

def plot2d(sim, parameter, title="Avalanche Simulation", threshold_value=1e-3, particle_threshold=0): 
    import_plt()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax, surf = ax2d(ax, sim, parameter, title, threshold_value, particle_threshold)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(surf, ax=ax, cax=cax)
    cbar.set_label(parameter.replace("_", " ").title())
    if not is_jupyter():    
        plt.show()
    return fig, ax

def ax2d(ax, sim, parameter, title="Avalanche Simulation", threshold_value=1e-3, particle_threshold=0):
    import_plt()
    data = getattr(sim, parameter).astype(np.float32)
    ax.set_aspect('equal')
    x, y, _, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan
    if sim.state == "Finished":
        # Mask based on particle count threshold
        data[sim.cell_count < particle_threshold * sim.released_particles_per_cell] = np.nan
    surf = ax.contourf(x, y, data, cmap='magma')
    ax.contour(x, y, sim.release_areas.astype(np.float32), colors='cyan', linewidths=1, alpha=0.3)
    ax.set(title=title)
    return ax, surf

def plot_overview(sim, threshold_value=1e-3, particle_threshold=0):
    # Setup parameters, titles, and distinct colormaps
    params = ['peak_velocity', 'peak_flow_thickness', 'cell_count']
    colormaps = ['magma', 'viridis', 'plasma']
    
    import_plt()
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    for i, (param, cmap) in enumerate(zip(params, colormaps)):
        ax = axes[i]
        ax.set_aspect('equal')
        
        # 1. Plot DEM in the background
        x, y, dem, dem_mask = plot_dem(sim, ax, dark=False)
        
        # 2. Prepare Data
        data = getattr(sim, param).astype(np.float32)
        data[dem_mask] = np.nan
        
        # Apply thresholds
        data[data < threshold_value] = np.nan
        if sim.state == "Finished":
            # Mask based on particle count threshold
            data[sim.cell_count < particle_threshold * sim.released_particles_per_cell] = np.nan
            
        if param == "cell_count":
            data = np.log10(data)
        
        # 3. Plot Simulation Results
        surf = ax.contourf(x, y, data, cmap=cmap)
        
        # 4. Plot Release Areas (Cyan Outline)
        ax.contour(x, y, sim.release_areas.astype(np.float32), 
                   colors='cyan', linewidths=1, alpha=0.3)
        
        # 5. Configure Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(surf, cax=cax)
        cbar.set_label(param.replace("_", " ").title())
        
        ax.set_title(param.replace("_", " ").title(), fontsize=14, fontweight='bold')
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
        
    return (2. * intersection) / total_area

def plot_comparison_binary(sim, parameter, reference_array, threshold_value=1, particle_threshold=0, title="Avalanche Simulation Comparison"):
    import_plt()
    data = getattr(sim, parameter).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y, dem, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan
    data[sim.cell_count < particle_threshold * sim.released_particles_per_cell] = np.nan
    only_reference = ~(data > 0) & (reference_array > 0)
    only_sim = (data > 0) & ~(reference_array > 0)
    both = (data > 0) & (reference_array > 0)
    comparison = np.zeros_like(data, dtype=int)
    comparison[only_reference] = 1
    comparison[both] = 2
    comparison[only_sim] = 3
    cmap = ListedColormap(['red', 'magenta', 'blue'])
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

def plot_comparison(sim, parameter, reference_array, particle_threshold=0, title="Avalanche Simulation Comparison"):
    import_plt()
    data = getattr(sim, parameter).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y, dem, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    data[sim.cell_count < particle_threshold * sim.released_particles_per_cell] = np.nan
    diff = reference_array - data
    diff[diff == 0] = np.nan
    # diff[(data == 0) | (reference_array == 0)] = np.nan
    max_abs = np.nanmax(np.abs(diff))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    levels = np.linspace(-max_abs, max_abs, 21)
    cont = ax.contourf(x, y, diff, cmap='bwr', levels=levels)
    cbar = fig.colorbar(cont, ax=ax,  shrink=0.8, aspect=10)
    cbar.ax.set_yticklabels(["No avalanche", "reference only", "both", "sim only"])
    dice = calculate_dice(reference_array, data)
    ax.set_title(title  + f"\nDice coefficient: {dice:.4f}")
    print(f"Dice coefficient: {dice:.4f}")
    return fig, ax

def is_jupyter():
    try:
        from IPython import get_ipython
        # ZMQInteractiveShell is the standard Jupyter kernel
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
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
            print("Warning: pyvista or trame not installed. Visualization may be limited.")
        except Exception as e:
            print(f"Failed to start Trame server: {e}")
    else:
        print("Standard environment detected: Skipping Jupyter server launch.")