import time
import avalanchers

start = time.time()


example_case = "avaWog"
# example_case = "avaMal"
# example_case = "avaKot"
# example_case = "avaParabola"
# example_case = "avaInclinedPlane"
# example_case = "avaFlatPlane"
# example_case = "avaHockeyChannel"
# example_case = "avaHelixChannel"
# example_case = "avaPyramid"

settings = {
    # mandatory: path to the DEM, currently only custom encoded png files are supported
    # optional, if not provided, a simple algorithm will be used to determine release areas based on the DEM
    "dem_path": f"data/avaframe/{example_case}.png",
    "release_areas_path": f"data/avaframe/{example_case}releaseTexture.png",   
    # "dem_path": f"data/stubai/10DTM_pilotStubai.tif", 
    # "release_areas_path": f"data/stubai/stubai_release_areas.png",
    # "dem_path": f"data/vals/PAR6_Vals_Gries_dtm_10_utm32n_bil_.tif",
    # "release_areas_path": f"data/vals/release.tif",   
    
    "max_steps": 5000,
    "batch_compute_steps": 300,
    "sim_model": 0,
    "released_particles_per_cell": 4,
    "friction_model": 2,
    "density": 200.0,
    "slab_thickness": 1.5,
    "friction_coefficient": 0.4,
    "drag_coefficient": 2000.0,
    "cfl": 0.5,
    "min_slope_angle": 28.0,
    "max_slope_angle": 50.0,
    "release_min_elevation": 1500.0,
    "velocity_threshold": 0.1,
    "roughness_threshold": 0.01,
    "enable_curvature": True,
    "enable_particle_interaction": True,
    "enable_entrainment": True,
}
sim = avalanchers.PySimulation.new()
sim.create(settings)

# or easier for examples
# sim.create_example("frontend/data/avaframe/avaMal.png")
sim.run()
positions = sim.particles_position

end = time.time()

print(f"Execution time without plotting: {end - start:.2f} seconds")


# avalanchers.plot3d(sim, "dem")


avalanchers.plot3d(sim, "peak_flow_thickness", False)
avalanchers.plot3d(sim, "peak_velocity", False)
# avalanchers.plot_overview(sim)

# avalanchers.plot2d(sim, "peak_flow_thickness")
# avalanchers.plot2d(sim, "peak_velocity", particles=True)
# avalanchers.plot2d(sim, "normals_x")
