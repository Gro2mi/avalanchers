import os
import shutil

import avalanchers
import numpy as np
import xarray as xr
from avalanchers import AvalanchersOutput


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
    assert data.avalanchers_format == "0.1.0"
    assert data.avalanchers_repo == "https://github.com/Gro2mi/avalanchers"

    site = data.get_site(site_name)

    expected_site_hash = "30b379fd01bbec1a"
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

    # Metadata
    assert dem.attrs["standard_name"] == "elevation"
    assert dem.attrs["long_name"] == "Digital Elevation Model"
    assert dem.attrs["units"] == "m"

    scenario = site.get_scenario(site.scenarios[0])
    print(scenario)

    expected_scenario_hash = "d1fba762150c532c"
    assert scenario.hash == expected_scenario_hash, (
        f"Scenario hash does not match, expected '{expected_scenario_hash}', "
        f"got {scenario.hash}"
    )
    assert scenario.name == site_name + "releaseTexture_" + expected_scenario_hash, (
        f"Scenario name does not match, expected '{site_name}releaseTexture_{expected_scenario_hash}', "
        f"got {scenario.name}"
    )
    assert scenario.name_no_hash == site_name + "releaseTexture"
    assert np.isclose(scenario.release_volume_m3, 10000.0)
    assert np.isclose(scenario.aspect_release_degrees, 20.0)
    assert scenario.number_of_runs == 2

    ds = scenario.dataset

    assert_dataarray_shape(ds, "peak_flow_velocity", 87, 70)
    assert_dataarray_shape(ds, "peak_flow_thickness", 87, 70)
    assert_dataarray_shape(ds, "release_area", 87, 70)
    assert np.isclose(ds.mu.isel(run=1).item(), 0.155)
    assert np.isclose(ds.xsi.isel(run=1).item(), 4000.0)
    assert np.isclose(ds.travel_angle.isel(run=1).item(), 25.0)
    assert np.isclose(ds.travel_length.isel(run=1).item(), 1000.0)
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
