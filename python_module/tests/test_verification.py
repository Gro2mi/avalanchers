import math

import avalanchers
import numpy as np
import pytest


def plane_simulation(angle, friction=0.0):
    n = 64
    cell_size = 5.0
    y, x = np.mgrid[:n, :n]
    x = (x + 0.5) * cell_size
    y = (y + 0.5) * cell_size
    direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    slope = -math.tan(math.radians(35.0)) * direction
    dem = np.ascontiguousarray((1000.0 + slope[0] * x + slope[1] * y).astype(np.float32))
    release = np.zeros((n, n), dtype=np.float32)
    release[12:52, 12:52] = 1.0
    sim = avalanchers.PySimulation.new()
    sim.create({
        "sim_model": "curvilinear",
        "friction_model": "coulomb",
        "friction_coefficient": friction,
        "density": 200.0,
        "released_particles_per_cell": 16,
        "max_steps": 200,
        "cfl": 0.05,
        "velocity_threshold": 0.0,
        "enable_curvature": False,
        "enable_particle_interaction": True,
        "enable_particle_relaxation": False,
    })
    sim.set_dem_with_bounds(dem, cell_size, 0.0, n * cell_size, 0.0, n * cell_size, 1.0)
    sim.set_release_areas(release)
    sim.prepare()
    return sim, slope, direction


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 90.0, 135.0, 225.0])
@pytest.mark.parametrize("friction", [0.0, 0.3])
def test_curvilinear_plane_acceleration_is_rotation_invariant(angle, friction):
    sim, slope, direction = plane_simulation(angle, friction)
    before = np.asarray(sim.particles_position_xy, dtype=np.float64)
    central = ((before > 100.0) & (before < 220.0)).all(axis=1)
    sim.run_n_steps(1)
    velocity = np.asarray(sim.particles_velocity, dtype=np.float64)
    after = np.asarray(sim.particles_position_xy, dtype=np.float64)
    dt = math.sqrt(2.0 * 0.05 * 5.0 / (9.81 * 0.866))
    np.testing.assert_allclose(after - before, velocity[:, :2] * dt, atol=2e-5)
    np.testing.assert_allclose(velocity[central, 2], velocity[central, :2] @ slope, rtol=0.002, atol=3e-5)
    acceleration = 9.81 * (math.sin(math.radians(35.0)) - friction * math.cos(math.radians(35.0)))
    expected = dt * acceleration * math.cos(math.radians(35.0)) * direction
    np.testing.assert_allclose(velocity[central, :2].mean(axis=0), expected, atol=0.003)
    physical_speed = np.linalg.norm(velocity[central], axis=1).mean()
    assert physical_speed == pytest.approx(dt * acceleration, abs=0.004)