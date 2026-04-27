import avalanchers
import numpy as np

def test_simulation_run_with_settings():
    settings = {
        "dem_path": "frontend/data/avaframe/avaMal.png",
        "release_areas_path": "frontend/data/avaframe/avaMalreleaseTexture.png",
    }
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.run()

def test_simulation_run_with_example():
    sim = avalanchers.PySimulation.new()
    sim.create_example("frontend/data/avaframe/avaMal.png")
    sim.run()

def test_np_array_as_dem_roundtrip():
    sim = avalanchers.PySimulation.new()
    dem_2d = np.random.uniform(0, 50, (20, 20)).astype(np.float32)
    sim.set_dem(
        dem_data=dem_2d, 
        cell_size=1.0,
        bounds_xmin=0.0,
        bounds_xmax=20.0,
        bounds_ymin=0.0,
        bounds_ymax=20.0,
        map_factor=1.0
    )
    np.testing.assert_array_equal(dem_2d, sim.dem)

def test_np_array_as_release_areas_roundtrip():
    sim = avalanchers.PySimulation.new()
    dem_2d = np.random.uniform(0, 50, (20, 20)).astype(np.float32)
    sim.set_dem(
        dem_data=dem_2d, 
        cell_size=1.0,
        bounds_xmin=0.0,
        bounds_xmax=20.0,
        bounds_ymin=0.0,
        bounds_ymax=20.0,
        map_factor=1.0
    )
    release_areas_2d = np.random.choice([0.0, 1.0], (20, 20)).astype(np.float32)
    sim.set_release_areas(release_areas_2d)
    # release areas only get updated on run, so we need to run again to see the changes
    sim.run()
    np.testing.assert_array_equal(release_areas_2d, sim.release_areas)