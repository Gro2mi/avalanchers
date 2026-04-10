import avalanchers
import matplotlib.pyplot as plt
import numpy as np

sim = avalanchers.PySimulation.create_default("data/avaframe/avaMal.png")
sim.run()
normals_x = sim.get_normals_numpy()[0]

z_data = normals_x
rows, cols = z_data.shape # 1001, 401
x = np.linspace(0, cols - 1, cols)
y = np.linspace(0, rows - 1, rows)
X, Y = np.meshgrid(x, y)

# 3. Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# rstride and cstride help performance by skipping lines on large meshes
surf = ax.plot_surface(X, Y, z_data, cmap='terrain', 
                       linewidth=0, antialiased=True,
                       rstride=10, cstride=10)

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title("Simulation Mesh Visualization")
plt.show()
