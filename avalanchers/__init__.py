import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._avalanchers import *

# Define native Python logic
def python_helper():
    print("This is pure Python!")

def create_mesh(sim):
    dem = sim.dem
    dem_mask = dem < sim.elevation_threshold + 1
    dem[dem_mask] = np.nan
    ny, nx = dem.shape
    x = np.arange(nx).astype(np.float32)
    y = np.arange(ny).astype(np.float32)
    x, y = np.meshgrid(x, y)
    return x, y, dem, dem_mask


def plot3d(sim, parameter, threshold_value=1):
    data = getattr(sim, parameter).astype(np.float32)
    x, y, dem, dem_mask = create_mesh(sim)

    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan

    # 2. Create the StructuredGrid
    # We pass x, y, and the elevation (dem) directly as coordinates
    grid = pv.StructuredGrid(x, y, dem)
    # 3. Add the aspect data as point scalars
    # Use .flatten() to match the point count of the mesh
    grid.point_data["Elevation"] = dem.flatten(order='F')
    grid.point_data[parameter] = data.flatten(order='F')

    clipped = grid.clip_scalar(value=threshold_value, scalars="Elevation", invert=False)

    # Vertical exaggeration
    final_mesh = clipped.scale([sim.cell_size, sim.cell_size, 1.3], inplace=False)


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
    plotter.enable_eye_dome_lighting()
    plotter.show(jupyter_backend='trame') if is_jupyter() else plotter.show()
        


def plot_dem(sim, ax, dark=True):
    xx, yy, dem, dem_mask = create_mesh(sim)
    cmap_contours = "Greys_r" if dark else "Greys"
    color_lines = "white" if dark else "black"
    levels_dem = np.arange(0, 4000, 200)
    ax.contourf(xx, yy, dem, levels=levels_dem, cmap=cmap_contours)
    CS = ax.contour(xx, yy, dem, levels=levels_dem, linewidths=.5, colors=color_lines)
    ax.clabel(CS, fontsize=10)
    return xx, yy, dem, dem_mask

def plot2d(sim, parameter, title="Avalanche Simulation", threshold_value=1, step=10, max_velocity=100, dark=True): 
    data = getattr(sim, parameter).astype(np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    x, y, dem, dem_mask = plot_dem(sim, ax, dark=False)
    data[dem_mask] = np.nan
    data[data < threshold_value] = np.nan
    # surf = ax.contourf(xx, yy, flow_velocity, cmap='viridis', levels=get_levels(flow_velocity, step), vmin=0.01)
    surf = ax.contourf(x, y, data, cmap='magma')#, levels=get_levels(np.array(max_velocity - 1), step), vmin=0.00001)
    ax.contour(x, y, sim.release_areas.astype(np.float32), colors='cyan', linewidths=1, alpha=0.3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
    # cbar = fig.colorbar(surf, ax=ax, ticks=[round(tick) for tick in get_levels(flow_velocity, step)], cax=cax)
    cbar = fig.colorbar(surf, ax=ax, cax=cax)
    cbar.set_label(parameter.replace("_", " ").title())
    # rounded_ticks = [round(tick) for tick in cbar.get_ticks()]
    # cbar.set_ticks(rounded_ticks)
    # cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set(title=title)
    plt.show()
    return fig, ax

def is_jupyter():
    try:
        from IPython import get_ipython
        # ZMQInteractiveShell is the standard Jupyter kernel
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except (ImportError, NameError):
        return False
    
async def setup_jupyter_3d():
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