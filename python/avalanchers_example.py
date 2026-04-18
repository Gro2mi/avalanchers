import avalanchers

sim = avalanchers.PySimulation.create_default("data/avaframe/avaMal.png")
sim.run()
max_velocity = sim.max_velocity
avalanchers.plot3d(sim, "max_velocity")
avalanchers.plot2d(sim, "max_velocity")


