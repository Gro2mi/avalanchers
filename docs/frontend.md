# Frontend

## Workflow

The frontend is organised as a step-by-step workflow:

* **Shortcut** — pick a bundled AvaFrame example and run it directly.
* **Step 1: Select DEM** — upload or drag & drop a `.gpx`, `.tif`/`.tiff`, `.asc` file, or a Zarr store directory.
  A `.gpx` track downloads matching elevation tiles; the zoom level slider controls their resolution.
  When a Zarr store is selected, a dropdown lists the sites found in the store.
* **Step 2: Select release areas** — either derive them from the terrain with *Calculate release areas*, or upload
  a `.tif`/`.tiff` or `.asc` raster matching the DEM grid. For a Zarr store, a scenario dropdown lists the scenarios
  of the site selected in step 1.
* **Step 3: Simulation settings** — model and solver parameters.
* **Step 4: Run simulation** — starts the run and shows progress and errors.

Uploaded `.asc` and GeoTIFF files are decoded by the WASM build so that cell size, bounds and row order match the
native and Python paths. After changing the Rust API, regenerate the WASM bindings (see the README build section).

See [data inputs](data_inputs.md) for Zarr store handling in the browser.

## Chrome on Windows with multiple GPUs

Chromium on Windows currently ignores the `high-performance` option in `powerPreference` if you have multiple
GPUs [[Issue](https://crbug.com/369219127)]. Options are:

1. Run slow on the integrated GPU.
2. Start Chrome with `"C:\Program Files\Google\Chrome\Application\chrome.exe" --force_high_performance_gpu`.
3. Activate `chrome://flags/#force-high-performance-gpu` (was unreliable in the past).
4. Assign the dedicated GPU to Chrome in the system settings.
