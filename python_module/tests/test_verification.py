import math
import pathlib
import sys
from types import SimpleNamespace

import avalanchers
import numpy as np
import pytest

VERIFICATION = pathlib.Path(__file__).resolve().parents[2] / "verification"
sys.path.insert(0, str(VERIFICATION))


def gather_grid(values, positions):
    q = positions / 5.0 - 0.5
    base = np.floor(q - 0.5).astype(int)
    gathered = np.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            node = base + [i, j]
            distance = np.abs(q - node)
            weights = np.where(distance < 0.5, 0.75 - distance**2, 0.5 * (1.5 - distance)**2)
            gathered += np.prod(weights, axis=1)[:, None] * values[node[:, 1], node[:, 0]]
    return gathered


def plane_simulation(angle, friction=0.0, **overrides):
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
        **overrides,
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
    expected = gather_grid(acceleration, before)
    expected *= math.sqrt(2.0 * 0.05 * 5.0 / (9.81 * 0.866))
    central = ((before > 100.0) & (before < 220.0)).all(axis=1)
    actual = np.asarray(sim.particles_velocity_xy, dtype=np.float64)
    np.testing.assert_allclose(actual[central], expected[central], atol=3e-5)


@pytest.mark.parametrize("normal_load_correction", [False, True])
def test_geometric_transport_is_independent_of_curvature_load_flag(normal_load_correction):
    sim, _, _ = plane_simulation(
        0.0, enable_particle_interaction=False, enable_curvature=normal_load_correction
    )
    _, x = np.mgrid[:64, :64]
    x = (x + 0.5) * 5.0
    dem = np.ascontiguousarray(
        (1000.0 - math.tan(math.radians(35.0)) * x + 0.005 * (x - 160.0)**2).astype(np.float32)
    )
    release = np.zeros((64, 64), dtype=np.float32)
    release[12:52, 12:52] = 1.0
    sim.set_dem_with_bounds(dem, 5.0, 0.0, 320.0, 0.0, 320.0, 1.0)
    sim.set_release_areas(release)
    sim.prepare()
    info = sim.run_n_steps(1)
    used_dt = info.dt
    before = np.asarray(sim.particles_position_xy, dtype=np.float64)
    sim.run_n_steps(1)
    mass = np.asarray(sim.grid_mass, dtype=np.float64).reshape((64, 64)) / 10.0
    momentum = np.asarray(sim.grid_momentum, dtype=np.float64).reshape((64, 64, 2)) / 100.0
    node_velocity = momentum / (mass[:, :, None] + 1e-6)
    node_velocity[mass < 0.01] = 0.0
    dem = dem.astype(np.float64)
    slope = np.stack(np.gradient(dem, 5.0)[::-1], axis=2)
    hessian_xx = np.zeros((64, 64))
    hessian_xx[:, 1:-1] = (dem[:, 2:] - 2.0 * dem[:, 1:-1] + dem[:, :-2]) / 25.0
    jac_squared = 1.0 + np.sum(slope**2, axis=2)
    normal_acceleration = hessian_xx * node_velocity[:, :, 0]**2
    acceleration = -slope * ((9.81 + normal_acceleration) / jac_squared)[:, :, None]
    expected = gather_grid(node_velocity + used_dt * acceleration, before)
    central = ((before > 100.0) & (before < 220.0)).all(axis=1)
    actual = np.asarray(sim.particles_velocity_xy, dtype=np.float64)
    np.testing.assert_allclose(actual[central], expected[central], atol=3e-5)


def test_curvilinear_elapsed_time_uses_executed_timestep():
    sim, _, _ = plane_simulation(45.0)
    info = sim.run_n_steps(0)
    for _ in range(3):
        before = np.asarray(sim.particles_position_xy, dtype=np.float64)
        start_time, used_dt = info.elapsed_time, info.dt
        info = sim.run_n_steps(1)
        assert info.grid_elapsed_time == pytest.approx(start_time)
        assert info["grid_elapsed_time"] == info.grid_elapsed_time
        assert info.elapsed_time - start_time == pytest.approx(used_dt, abs=1e-7)
        after = np.asarray(sim.particles_position_xy, dtype=np.float64)
        velocity = np.asarray(sim.particles_velocity_xy, dtype=np.float64)
        np.testing.assert_allclose(after - before, velocity * used_dt, atol=2e-5)


def test_stopped_batch_preserves_last_grid_snapshot():
    sim, _, _ = plane_simulation(0.0, friction=10.0)
    info = sim.run_n_steps(100)
    assert info.flags & (1 << 31)
    mass = np.asarray(sim.grid_mass, dtype=np.float64).sum() / 10.0
    assert mass == pytest.approx(np.asarray(sim.particles_mass).sum(), rel=0.01)
    final_time = info.elapsed_time
    assert sim.run_n_steps(1).elapsed_time == final_time


@pytest.mark.parametrize("threshold", [0.0, 0.1])
def test_flat_heap_spreading_and_retained_freezing_policy(threshold):
    sim, _, _ = plane_simulation(0.0, velocity_threshold=threshold)
    y, x = np.mgrid[:64, :64]
    x, y = (x + 0.5) * 5.0, (y + 0.5) * 5.0
    release = np.ascontiguousarray(
        np.maximum(1.0 - ((x - 160.0) / 50.0)**2 - ((y - 160.0) / 50.0)**2, 0.0).astype(np.float32)
    )
    sim.set_dem_with_bounds(np.zeros((64, 64), dtype=np.float32), 5.0, 0.0, 320.0, 0.0, 320.0, 1.0)
    sim.set_release_areas(release)
    sim.prepare()
    before = np.asarray(sim.particles_position_xy, dtype=np.float64)
    initial_radius = np.sqrt(np.mean(np.sum((before - 160.0)**2, axis=1)))
    info = sim.run_n_steps(100)
    after = np.asarray(sim.particles_position_xy, dtype=np.float64)
    final_radius = np.sqrt(np.mean(np.sum((after - 160.0)**2, axis=1)))
    if threshold == 0.0:
        assert not info.flags & (1 << 31)
        assert final_radius > initial_radius + 1.0
    else:
        assert info.flags & (1 << 31)
        assert final_radius < initial_radius + 0.2


@pytest.mark.parametrize("error, final_time, expected", [
    (0.15, 20.0, "passed"),
    (0.1501, 20.0, "failed"),
    (float("nan"), 20.0, "failed"),
    (0.10, 19.9, "failed"),
])
def test_verification_guardrails_check_errors_and_end_time(error, final_time, expected):
    import run_all_tests

    args = SimpleNamespace(sim_model="curvilinear", friction_model=None, quick=False)
    ret = ({"t": np.array([5.0, final_time])},
           {"h_l2rel": np.array([0.08, error]), "vh_l2rel": np.array([0.1, 0.2])})
    assert run_all_tests._assessment("similarity_solution", ret, args)[0] == expected