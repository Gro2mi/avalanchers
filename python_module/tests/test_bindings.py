import avalanchers
import numpy as np


def test_list_available_gpus_returns_strings():
    gpus = avalanchers.list_available_gpus()

    assert isinstance(gpus, list)
    assert all(isinstance(gpu, str) for gpu in gpus)


def test_pysettings_round_trip(tmp_path):
    settings = avalanchers.PySettings()
    settings.dem_path = "example/dem.tif"

    path = tmp_path / "settings.json"
    settings.to_json(str(path))

    loaded = avalanchers.PySettings.from_json(str(path))
    assert isinstance(loaded, avalanchers.PySettings)
    assert loaded.dem_path == "example/dem.tif"


def test_simulation_binding_smoke():
    sim = avalanchers.PySimulation.new()
    sim.create_example("data/avaframe/avaParabola.png")
    sim.set_max_timesteps(3)
    sim.run()
    sim.post_process()

    assert isinstance(sim.state, str)
    assert sim.cell_size > 0.0
    assert isinstance(sim.released_particles_per_cell, int)
    assert sim.released_particles_per_cell >= 0

    dem = sim.dem
    assert isinstance(dem, np.ndarray)
    assert dem.ndim == 2
    assert dem.shape[0] > 0 and dem.shape[1] > 0

    bounds = sim.dem_bounds
    assert isinstance(bounds, np.ndarray)
    assert bounds.shape == (4,)

    peak_velocity = sim.peak_velocity
    assert peak_velocity.shape == dem.shape
    assert peak_velocity.dtype == np.float32

    terrain_x = sim.terrain_geometry_x
    terrain_y = sim.terrain_geometry_y
    terrain_z = sim.terrain_geometry_z
    assert terrain_x.shape == dem.shape
    assert terrain_y.shape == dem.shape
    assert terrain_z.shape == dem.shape

    gravity_x = sim.gravity_x
    gravity_y = sim.gravity_y
    assert gravity_x.shape == dem.shape
    assert gravity_y.shape == dem.shape

    release_areas = sim.release_areas
    peak_flow_thickness = sim.peak_flow_thickness
    assert release_areas.shape == dem.shape
    assert peak_flow_thickness.shape == dem.shape

    timestep = sim.timestep_data
    assert hasattr(timestep, "velocity")
    assert hasattr(timestep, "position")
    assert hasattr(timestep, "dt")
    np_velocity = np.asarray(timestep.velocity)
    np_position = np.asarray(timestep.position)
    np_dt = np.asarray(timestep.dt)
    assert np_velocity.ndim == 2 and np_velocity.shape[1] == 3
    assert np_position.ndim == 2 and np_position.shape[1] == 3
    assert np_dt.ndim == 1

    assert isinstance(sim.elevation_threshold, (float, np.floating))
    assert sim.elevation_threshold >= 0.0

    positions = sim.particles_position
    velocities = sim.particles_velocity
    elevations = sim.particles_elevation
    stopped = sim.stopped

    assert positions.ndim == 2 and positions.shape[1] == 3
    assert velocities.ndim == 2 and velocities.shape[1] == 3
    assert elevations.ndim == 1
    assert stopped.ndim == 1

    
    positions_xy = sim.particles_position_xy
    velocities_xy = sim.particles_velocity_xy

    assert positions_xy.ndim == 2 and positions_xy.shape[1] == 2
    assert velocities_xy.ndim == 2 and velocities_xy.shape[1] == 2

    # Ensure the underlying arrays are still numeric and finite when populated.
    assert np.all(np.isfinite(dem))
    assert np.all(np.isfinite(peak_velocity))
    assert np.all(np.isfinite(terrain_x))
    assert np.all(np.isfinite(terrain_y))
    assert np.all(np.isfinite(terrain_z))
    assert np.all(np.isfinite(gravity_x))
    assert np.all(np.isfinite(gravity_y))
    assert np.all(np.isfinite(release_areas))
    assert np.all(np.isfinite(peak_flow_thickness))
    assert np.all(np.isfinite(np_velocity))
    assert np.all(np.isfinite(np_position))
    assert np.all(np.isfinite(np_dt))


def test_simulation_run_n_steps():
    sim = avalanchers.PySimulation.new()
    sim.create_example("data/avaframe/avaParabola.png")

    initial = sim.run_n_steps(0)
    assert initial["timestep"] == 1
    assert initial["number_particles"] > 0
    assert sim.state == "Running"

    advanced = sim.run_n_steps(1)
    assert advanced["timestep"] >= initial["timestep"]
    assert isinstance(advanced["dt"], float)
    assert isinstance(advanced["elapsed_time"], float)
    assert isinstance(advanced["flags"], int)
