import avalanchers
import matplotlib.pyplot as plt
import numpy as np

sim = avalanchers.PySimulation.create_default("data/avaframe/avaGar.png")
sim.run()
normals_x = sim.normals_x
release_areas = sim.release_areas
max_velocity = sim.max_velocity
dem = sim.dem


import pyvista as pv

# 1. Create the coordinate grid (X and Y) based on your DEM dimensions
ny, nx = dem.shape
x = np.arange(nx).astype(np.float32)
y = np.arange(ny).astype(np.float32)
x, y = np.meshgrid(x, y)

threshold_value = 50
cell_size = 5.0
dem[dem < threshold_value] = np.nan
normals_x[dem < threshold_value] = np.nan
release_areas[dem < threshold_value] = np.nan
max_velocity[dem < threshold_value] = np.nan

# 2. Create the StructuredGrid
# We pass x, y, and the elevation (dem) directly as coordinates
grid = pv.StructuredGrid(x, y, dem)
# 3. Add the aspect data as point scalars
# Use .flatten() to match the point count of the mesh
grid.point_data["Elevation"] = dem.flatten(order='F')
grid.point_data["Aspect"] = normals_x.flatten(order='F')
grid.point_data["ReleaseAreas"] = release_areas.flatten(order='F')
grid.point_data["MaxVelocity"] = max_velocity.flatten(order='F')

clipped = grid.clip_scalar(value=threshold_value, scalars="Elevation", invert=False)

# 3. SCALE SECOND: Apply the vertical exaggeration to the clipped result
# This preserves the scalars attached to the points
final_mesh = clipped.scale([sim.cell_size, sim.cell_size, 1.3], inplace=False)
# 4. Visualization
plotter = pv.Plotter()

# We use the 'twilight' colormap for cyclic data (0-360)
# 'clim' ensures the colormap covers the full circle regardless of data range
plotter.add_mesh(
    final_mesh, 
    scalars="MaxVelocity", 
    cmap="rainbow", 
    # clim=[0, 1],
    scalar_bar_args={'title': "Maximum Velocity"},
    edge_color="black",
    show_edges=False
)
# Optional: Add a light source or enable Eye Dome Lighting for better depth
plotter.enable_eye_dome_lighting()
plotter.show()