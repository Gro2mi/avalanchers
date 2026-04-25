import avalanchers

settings = {
    "dem_path": "frontend/data/avaframe/avaKot.png",
}
sim = avalanchers.PySimulation.new()
sim.create(settings)

# or easier for default settings
# sim = avalanchers.PySimulation.create_default("frontend/data/avaframe/avaMal.png")

sim.run()
avalanchers.plot3d(sim, "max_velocity")
avalanchers.plot2d(sim, "max_velocity")