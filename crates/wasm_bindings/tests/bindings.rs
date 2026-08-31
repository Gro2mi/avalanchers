use wasm_bindgen_test::*;

use avalanchers::{WasmSettings, WasmSimulation};
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn settings_defaults_and_getters_work() {
    let mut settings = WasmSettings::new();
    assert_eq!(settings.dem_path(), "");

    settings.set_dem_path("demo/dem.png".into());
    assert_eq!(settings.dem_path(), "demo/dem.png");
}

#[wasm_bindgen_test]
async fn simulation_can_be_constructed_and_basic_geometry_is_available() {
    let sim = WasmSimulation::new()
        .await
        .expect("simulation should construct");

    let dem = sim.dem();
    assert_eq!(dem.length(), 0);

    assert_eq!(sim.width(), 0);
    assert_eq!(sim.height(), 0);
    assert_eq!(sim.cell_size(), 1.0);

    let bounds = sim.bounds();
    assert_eq!(bounds.length(), 4);

    let x = sim.x();
    let y = sim.y();
    assert_eq!(x.length(), 0);
    assert_eq!(y.length(), 0);
}

#[wasm_bindgen_test]
async fn simulation_getters_and_timestep_data_are_available_after_run() {
    let mut sim = WasmSimulation::new()
        .await
        .expect("simulation should construct");
    let width = 4;
    let height = 4;
    let dem = [
        100.0, 99.0, 98.0, 97.0, 99.0, 98.0, 97.0, 96.0, 98.0, 97.0, 96.0, 95.0, 97.0, 96.0, 95.0,
        94.0,
    ];
    let release_areas = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    sim.set_dem_default(&dem, width, height, 1.0)
        .await
        .expect("DEM should be set");
    sim.set_release_areas(&release_areas)
        .await
        .expect("release areas should be set");
    sim.run().await.expect("simulation should run");
    sim.fetch_results()
        .await
        .expect("simulation results should be fetched");
    sim.fetch_roughness()
        .await
        .expect("roughness should be fetched");
    sim.fetch_release_areas()
        .await
        .expect("release areas should be fetched");
    sim.fetch_peak_velocity()
        .await
        .expect("peak velocity should be fetched");
    sim.fetch_peak_flow_thickness()
        .await
        .expect("peak flow thickness should be fetched");

    let grid_size = (width * height) as u32;
    assert_eq!(sim.dem().length(), grid_size);
    assert_eq!(sim.width(), width);
    assert_eq!(sim.height(), height);
    assert_eq!(sim.x().length(), width);
    assert_eq!(sim.y().length(), height);
    assert_eq!(sim.bounds().length(), 4);
    assert_eq!(sim.dem_trajectory_info().length(), 3);
    assert_eq!(sim.peak_velocity().length(), grid_size);
    assert_eq!(sim.slope_aspect().length(), grid_size);
    assert_eq!(sim.slope_angle().length(), grid_size);
    assert_eq!(sim.roughness().length(), grid_size);
    assert_eq!(sim.release_areas().length(), grid_size);
    assert_eq!(sim.peak_flow_thickness().length(), grid_size);
    #[cfg(target_arch = "wasm32")]
    assert!(sim.result_store_name().ends_with(".zarr"));

    let timestep = sim
        .get_timestep_data()
        .await
        .expect("timestep data should be fetched");
    assert_eq!(timestep.position().length() % 3, 0);
    assert!(timestep.dt().length() <= timestep.time().length());
    assert!(timestep.velocity_magnitude().length() <= timestep.time().length());
    assert!(timestep.step_distance().length() <= timestep.time().length());
    assert!(timestep.travel_distance().length() <= timestep.time().length());
    assert!(timestep.cfl().length() <= timestep.time().length());
}

#[wasm_bindgen_test]
fn settings_from_json_parses_dem_path() {
    let settings = WasmSettings::from_json(r#"{"dem_path":"data/avaframe/avaMal.png"}"#)
        .expect("settings JSON should parse");

    assert_eq!(settings.dem_path(), "data/avaframe/avaMal.png");
}
