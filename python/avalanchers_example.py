import time
import avalanchers

start = time.time()


example_case = "avaMal"

settings = {
    # mandatory: path to the DEM, currently only custom encoded png files are supported
    "dem_path": f"frontend/data/avaframe/{example_case}.png",
    # optional, if not provided, a simple algorithm will be used to determine release areas based on the DEM
    "release_areas_path": f"frontend/data/avaframe/{example_case}releaseTexture.png",   
}
sim = avalanchers.PySimulation.new()
sim.create(settings)

# or easier for examples
# sim.create_example("frontend/data/avaframe/avaMal.png")

sim.run()

end = time.time()

print(f"Execution time without plotting: {end - start:.2f} seconds")

avalanchers.plot3d(sim, "cell_count")
avalanchers.plot2d(sim, "max_velocity")
