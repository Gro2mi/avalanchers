import avalanchers
import pytest

def test_simulation_run_with_settings():
    settings = {
        "dem_path": "frontend/data/avaframe/avaMal.png",
    }
    sim = avalanchers.PySimulation.create(settings)
    sim.run()

def test_simulation_run_with_default_settings():
    sim = avalanchers.PySimulation.create_default("frontend/data/avaframe/avaMal.png")
    sim.run()