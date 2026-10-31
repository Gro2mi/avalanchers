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


@pytest.mark.parametrize("angle", [0.0, 45.0, 135.0])
def test_curvilinear_pressure_uses_normal_thickness_and_full_metric(angle):
    sim, slope, _ = plane_simulation(angle)
    before = np.asarray(sim.particles_position_xy, dtype=np.float64)
    sim.run_n_steps(1)
    mass = np.asarray(sim.grid_mass, dtype=np.float64).reshape((64, 64)) / 10.0
    jac = np.asarray(sim.terrain_geometry_z, dtype=np.float64)
    # The terrain kernel leaves the excluded outermost cells uninitialized.
    jac = np.pad(jac[1:-1, 1:-1], 1, mode="edge")
    h = mass / (200.0 * 5.0**2 * jac)
    grad = np.zeros((64, 64, 2))
    grad[:, 1:-1, 0] = (h[:, 2:] - h[:, :-2]) / 10.0
    grad[1:-1, :, 1] = (h[2:, :] - h[:-2, :]) / 10.0
    inverse_grad = grad - slope * np.sum(grad * slope, axis=2, keepdims=True) / (1.0 + slope @ slope)
    acceleration = -9.81 * slope / (1.0 + slope @ slope) - 9.81 / jac[:, :, None] * inverse_grad
    q = before / 5.0 - 0.5
    base = np.floor(q - 0.5).astype(int)
    expected = np.zeros_like(before)
    for i in range(3):
        for j in range(3):
            node = base + [i, j]
            distance = np.abs(q - node)
            weights = np.where(distance < 0.5, 0.75 - distance**2, 0.5 * (1.5 - distance)**2)
            weight = np.prod(weights, axis=1)
            expected += weight[:, None] * acceleration[node[:, 1], node[:, 0]]
    expected *= math.sqrt(2.0 * 0.05 * 5.0 / (9.81 * 0.866))
    central = ((before > 100.0) & (before < 220.0)).all(axis=1)
    actual = np.asarray(sim.particles_velocity_xy, dtype=np.float64)
    np.testing.assert_allclose(actual[central], expected[central], atol=3e-5)