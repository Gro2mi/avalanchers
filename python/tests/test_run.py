import avalanchers
import numpy as np

def test_simulation_run_with_settings():
    settings = {
        "dem_path": "frontend/data/avaframe/avaMal.png",
    }
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.run()

def test_simulation_run_with_default_settings():
    sim = avalanchers.PySimulation.new()
    sim.create_default("frontend/data/avaframe/avaMal.png")
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