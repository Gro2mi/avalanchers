import avalanchers

example_case = "avaMal"

settings = {
    "dem_path": "frontend/data/avaframe/avaMal.png",
    "release_areas_path": "frontend/data/avaframe/avaMalreleaseTexture.png",
}
sim = avalanchers.PySimulation.new()
sim.create(settings)

# or easier for examples
# sim.create_example("frontend/data/avaframe/avaMal.png")

sim.run()
avalanchers.plot3d(sim, "max_velocity")
avalanchers.plot2d(sim, "max_velocity")
