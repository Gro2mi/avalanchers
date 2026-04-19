import avalanchers

settings = {
    "dem_path": "data/avaframe/avaMal.png",
}
sim = avalanchers.PySimulation.create(settings)

# or easier for default settings
# sim = avalanchers.PySimulation.create_default("data/avaframe/avaMal.png")

sim.run()
avalanchers.plot3d(sim, "max_velocity")
avalanchers.plot2d(sim, "max_velocity")