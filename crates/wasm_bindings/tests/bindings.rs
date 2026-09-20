use wasm_bindgen_test::*;

use avalanchers::WasmSettings;

#[wasm_bindgen_test]
fn settings_defaults_and_getters_work() {
    let mut settings = WasmSettings::new();
    assert_eq!(settings.dem_path(), "");

    settings.set_dem_path("demo/dem.png".into());
    assert_eq!(settings.dem_path(), "demo/dem.png");
}

#[wasm_bindgen_test]
fn settings_from_json_parses_dem_path() {
    let settings = WasmSettings::from_json(r#"{"dem_path":"data/avaframe/avaMal.png"}"#)
        .expect("settings JSON should parse");

    assert_eq!(settings.dem_path(), "data/avaframe/avaMal.png");
}
