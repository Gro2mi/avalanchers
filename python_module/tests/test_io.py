import os
import shutil

import avalanchers
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from avalanchers import AvalanchersOutput


def test_export_calibration_parameters(tmp_path):
    path = str(tmp_path / "calibration.parquet")

    frame = avalanchers.export_calibration_parameters(
        {
            "friction_model": "samosat",
            "mu0": 0.42,
            "enable_entrainment": False,
            "release_area_fraction": 0.25,
        },
        path,
        run_name="base",
        evaluation={"iou": 0.75, "peak_velocity": 12.5},
    )

    assert frame.loc[0, "run_name"] == "base"
    assert frame.loc[0, "friction_model"] == "samosat"
    assert frame.loc[0, "mu0"] == 0.42
    assert not frame.loc[0, "enable_entrainment"]
    assert frame.loc[0, "release_area_fraction"] == 0.25
    # Unset parameters are recorded with the engine defaults
    assert frame.loc[0, "friction_coefficient"] == 0.2
    assert frame.loc[0, "drag_coefficient"] == 2000.0
    assert frame.loc[0, "sim_model"] == "terrain-following"
    assert frame.loc[0, "avalanchers_version"] == avalanchers.__version__
    # Evaluation results are stored in the same row, metrics without a
    # value stay null
    assert frame.loc[0, "iou"] == 0.75
    assert frame.loc[0, "peak_velocity"] == 12.5
    assert pd.isna(frame.loc[0, "chamfer"])
    assert all(column in frame.columns for column in avalanchers.EVALUATION_METRICS)

    # Appending accumulates one row per run and continues the numbering
    avalanchers.export_calibration_parameters({"mu0": 0.5}, path)
    reread = pd.read_parquet(path)
    assert list(reread["run_name"]) == ["base", "run_2"]
    assert reread.loc[1, "mu0"] == 0.5
    assert reread.loc[1, "friction_model"] == "voellmy"
    assert pd.isna(reread.loc[1, "iou"])

    # Batch export of a sweep with explicit names
    sweep = avalanchers.export_calibration_parameters(
        [{"mu0": 0.4}, {"mu0": 0.6}], path, run_name=["low", "high"]
    )
    assert list(sweep["run_name"]) == ["base", "run_2", "low", "high"]


def test_calibration_parameters_enum_normalization():
    params = avalanchers.calibration_parameters(
        {
            "sim_model": 2,
            "friction_model": "VoellmyMinShear",
            "constitutive_model": "mui",
        }
    )
    assert params["sim_model"] == "mpmdac"
    assert params["friction_model"] == "voellmy-min-shear"
    assert params["constitutive_model"] == "mu-i"
    assert avalanchers.calibration_parameters({"sim_model": 3})["sim_model"] == "sliding-block"
    assert avalanchers.calibration_parameters({"sim_model": "block"})["sim_model"] == "sliding-block"
    assert params["release_fill_method"] == "geodesic"
    assert (
        avalanchers.calibration_parameters({"release_fill_method": "Slope_Weighted"})[
            "release_fill_method"
        ]
        == "slope-weighted"
    )


def test_export_calibration_parameters_with_evaluation(tmp_path):
    sim = avalanchers.PySimulation.new()
    sim.create_example("data/avaframe/avaParabola.png")
    sim.set_max_timesteps(3)
    sim.run()
    sim.post_process()
    evaluation = sim.evaluate()

    path = str(tmp_path / "calibration.parquet")
    frame = avalanchers.export_calibration_parameters(
        {"mu0": 0.44}, path, run_name="smoke", evaluation=evaluation
    )

    assert frame.loc[0, "iou"] == evaluation.iou
    assert frame.loc[0, "beeline_3d"] == evaluation.beeline_3d
    assert frame.loc[0, "chamfer"] == evaluation.chamfer
    assert frame.loc[0, "peak_velocity"] == evaluation.peak_velocity
    assert pd.notna(frame.loc[0, "horizontal_distance_ref"])


@pytest.mark.slow
def test_simulation_save():
    path = "avalanchers_test.zarr"
    site_name = "avaPfa"
    if os.path.exists(path):
        shutil.rmtree(path)
    sim = avalanchers.PySimulation.new()
    sim.create_example(f"data/avaframe/{site_name}.png")
    sim.run()
    sim.save(path)
    sim.save()

    data = AvalanchersOutput(path)
    print(data)
    assert data.avalanchers_version == avalanchers.__version__
    assert data.avalanchers_format == "0.2.0"
    assert data.conventions == "CF-1.8"
    assert data.avalanchers_repo == "https://github.com/Gro2mi/avalanchers"

    site = data.get_site(site_name)

    expected_site_hash = "5435f4bd7c2c9a31"
    assert site.hash == expected_site_hash, (
        f"Site hash does not match, expected '{expected_site_hash}', got {site.hash}"
    )
    assert site.name_no_hash == site_name, (
        f"Site name does not match, expected '{site_name}', got {site.name_no_hash}"
    )
    assert site.name == site_name + "_" + expected_site_hash, (
        f"Site name does not match, expected '{site_name}_{expected_site_hash}', "
        f"got {site.name}"
    )

    dem = site.dem

    assert isinstance(dem, xr.DataArray)
    assert dem.name == "dem"
    # Data type
    assert dem.dtype == np.float32

    # Dimensions / shape
    assert dem.dims == ("y", "x")
    assert dem.shape == (70, 87)
    assert dem.size == 70 * 87

    # Coordinates
    assert "x" in dem.coords
    assert "y" in dem.coords
    assert dem.x.dims == ("x",)
    assert dem.y.dims == ("y",)
    assert dem.x.size == 87
    assert dem.y.size == 70
    assert dem.y[0] < dem.y[-1], "Row 0 is the south edge, so y has to ascend"

    # Metadata
    assert dem.attrs["standard_name"] == "surface_altitude"
    assert dem.attrs["long_name"] == "Digital Elevation Model"
    assert dem.attrs["units"] == "m"

    scenario = site.get_scenario(site.scenarios[0])
    print(scenario)

    expected_scenario_hash = "1a2ebf85e357a9c8"
    assert scenario.hash == expected_scenario_hash, (
        f"Scenario hash does not match, expected '{expected_scenario_hash}', "
        f"got {scenario.hash}"
    )
    assert scenario.name == site_name + "releaseTexture_" + expected_scenario_hash, (
        f"Scenario name does not match, expected '{site_name}releaseTexture_{expected_scenario_hash}', "
        f"got {scenario.name}"
    )
    assert scenario.name_no_hash == site_name + "releaseTexture"
    # assert np.isclose(scenario.aspect_release_degrees, None)
    assert scenario.aspect_release_degrees is None
    assert scenario.number_of_runs == 2

    ds = scenario.dataset

    assert_dataarray_shape(ds, "peak_flow_velocity", 87, 70)
    assert_dataarray_shape(ds, "peak_flow_thickness", 87, 70)
    assert_dataarray_shape(ds, "release_area", 87, 70)
    assert ds.sizes["run"] == 2
    assert np.isclose(ds.release_volume_m3.isel(run=1).item(), 7283.78076171875, atol=0.4)
    assert np.isclose(ds.friction_coefficient.isel(run=1).item(), 0.2)
    assert np.isclose(ds.drag_coefficient.isel(run=1).item(), 2000.0)
    assert ds.released_particles_per_cell.isel(run=1).item() > 0
    assert np.isfinite(ds.travel_length.isel(run=1).item())
    assert ds.travel_length.isel(run=1).item() > 0
    assert ds.final_timestep.isel(run=1).item() > 0
    time = ds.time.isel(run=1).values
    time = time[np.isfinite(time)]
    assert time.size == ds.final_timestep.isel(run=1).item()
    # The step that stops a run advances the clock without adding a record.
    remaining = ds.simulated_time.isel(run=1).item() - time[-1]
    assert 0 <= remaining <= 1.01 * (time[-1] - time[-2])
    assert scenario.software_versions[ds.software_version.isel(run=1).item()].startswith(
        avalanchers.__version__
    )
    assert np.issubdtype(ds.created_at.dtype, np.datetime64)
    # x and y are cell centers, so they are spaced by exactly one cell
    assert np.allclose(np.diff(ds.x.values), ds.cell_size.isel(run=1).item())
    # The store provides the DEM and release of a further run of the same scenario.
    imported = avalanchers.PySimulation.new()
    imported.create({"dem_path": f"{path}/{site_name}", "output_path": path})
    imported.run()
    imported.save()
    assert scenario.number_of_runs == 3
    assert len(data.list_all_scenarios()) == 1

    if os.path.exists(path):
        shutil.rmtree(path)


def assert_dataarray_shape(
    ds: xr.Dataset,
    variable: str,
    x_len: int,
    y_len: int,
    *,
    run: int = 1,
) -> None:
    da = ds[variable]
    if "run" in da.dims:
        da = da.isel(run=run)

    assert da.dims == ("y", "x"), f"Unexpected dimensions for {variable}: {da.dims}"
    assert da.sizes == {"y": y_len, "x": x_len}, (
        f"Unexpected shape for {variable}: {da.sizes}"
    )
