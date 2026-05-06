import time
import avalanchers

start = time.time()


example_case = "avaMal"

settings = {
    # mandatory: path to the DEM, currently only custom encoded png files are supported
    "dem_path": f"frontend/data/avaframe/{example_case}.png",
    # optional, if not provided, a simple algorithm will be used to determine release areas based on the DEM
    "release_areas_path": f"frontend/data/avaframe/{example_case}releaseTexture.png",   
    # "dem_path": f"data/vals/PAR6_Vals_Gries_dtm_10_utm32n_bil_.tif",
    # "release_areas_path": f"data/vals/release.tif",   
    "max_steps": 3000,
    "sim_model": 0,
    "friction_model": 1,
    "density": 200.0,
    "slab_thickness": .50,
    "friction_coefficient": 0.155,
    "drag_coefficient": 4000.0,
    "cfl": 0.5,
    "min_slope_angle": 28.0,
    "max_slope_angle": 60.0,
    "release_min_elevation": 1500.0,
    "velocity_threshold": 1e-6,
    "roughness_threshold": 0.01,
}
sim = avalanchers.PySimulation.new()
sim.create(settings)

# or easier for examples
# sim.create_example("frontend/data/avaframe/avaMal.png")

sim.run()

end = time.time()

print(f"Execution time without plotting: {end - start:.2f} seconds")

avalanchers.plot3d(sim, "max_velocity")
avalanchers.plot2d(sim, "max_velocity")
