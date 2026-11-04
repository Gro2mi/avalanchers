# Avalanche Simulation with Rust and WebGPU

[Try it yourself!](https://gro2mi.github.io/avalanchers/ "Avalanche Simulation")

Most modern browsers now support WebGPU ([check the current status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)). The application has been tested on Windows, Linux, and Android, though support on mobile devices may still be limited.

You can explore one of the predefined AvaFrame examples using the dropdown menu, or upload your own Austrian GPX file to run a custom simulation.

This project was initiated to streamline the development of avalanche simulations using WebGPU, building on concepts from [weBIGeo](https://github.com/weBIGeo/webigeo/tree/main). It provides an easy way to visualize results directly in the browser.

The core has since been rewritten in Rust, with Python and WebAssembly (WASM) bindings available. The underlying model is a block-based approach without particle interactions, enabling fast initial estimates of runout distance and flow routing.

## Get started with Python

`pip install avalanchers[viz]`

```
import avalanchers

sim = avalanchers.PySimulation.new()
sim.create_example("frontend/data/avaframe/avaMal.png")
sim.run()

# needs the viz option
avalanchers.plot2d(sim, "max_velocity")
avalanchers.plot3d(sim, "max_velocity")
```

## Known Issues with Chrome

* Chromium on Windows currently ignores the `high-performance` option in `powerPreference` WebGPU flag if you have multiple GPUs [[Issue](https://crbug.com/369219127)]. Options are:
  1. Run slow on integrated GPU
  2. Start Chrome with high performance gpu flag `"C:\Program Files\Google\Chrome\Application\chrome.exe" --force_high_performance_gpu`
  3. Activate flag in Chrome `chrome://flags/#force-high-performance-gpu`. I got slow runs about 50% of the time even though it actually runs on the fast GPU. Might work by now.
  4. Activate the dedicated GPU for Chrome in the system settings.

## Development Setup

### Windows

Install [Rust](https://rust-lang.org/tools/install/)
Install [Python](https://www.python.org/)

```
# wasm bindings
cargo install wasm-bindgen-cli
cargo install wasm-pack

# coverage
cargo install cargo-tarpaulin

# python bindings
python -m venv .venv
.venv\Scripts\Activate.ps1  
python -m pip install maturin
```

### Linux

#### Install dependencies

Install [Rust](https://rust-lang.org/tools/install/) (WSL (2026.04.04): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
Install [Python](https://www.python.org/)

```
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential clang lld pkg-config libssl-dev

# wasm bindings
cargo install wasm-bindgen-cli
cargo install wasm-pack

# python bindings
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install maturin
```

## Build

```

# cli tool
cargo build -p cli --release

# python bindings
maturin develop 
maturin build --release

# wasm bindings
wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --no-opt
wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --release
```

## Run

### Curvilinear velocity convention

The curvilinear solver stores horizontal velocities `(dx/dt, dy/dt)`. Gravity,
pressure and friction use the full surface metric, including its cross term.
APIC transfers preserve affine velocity variations between particles and grid.
Geometric transport along curved terrain is always included.
`enable_curvature` controls the additional centripetal basal normal load;
the alpha-line energy benchmark disables that load correction to match its
reference assumption.
`enable_weak_pressure` (default off) takes the hydrostatic pressure gradient
from the MPM weak form, `m_i grad h = -sum_p m_p h_p / 2 grad N_i(x_p)`, in an
extra particle pass after P2G instead of the centred difference of the nodal
thickness. The small mass of a thin front node then receives the full push of
the particles behind it; on the dam break benchmark the front moves about 5 m
closer to the analytic solution at 5 m cells.
Particle xyz velocities reconstruct the vertical component from the terrain
slope; xy velocities and grid momentum are horizontal components. Analytical
comparisons convert these to surface-distance velocities where needed.

See [the curvilinear model review](docs/curvilinear_model_review.md) for the
diagnosis and staged correction results.

`SimInfo.elapsed_time` is particle time after the executed step; `SimInfo.dt`
is the next timestep. `SimInfo.grid_elapsed_time` timestamps the deposited
grid mass/momentum snapshot before that step's advection. Verification uses
the snapshot time and applies numerical regression guardrails instead of
treating successful execution as numerical acceptance.

The curvature benchmark reuses the energy-line parabola and release, compares
curvature-on/off runout against an analytical integrating-factor solution, and
plots the curved energy line. Run it with
`.venv\Scripts\python.exe verification\run_curvature_test.py`, or use
`verification\run_all_tests.py --tests curvature`. It is included in the full
curvilinear/Coulomb verification suite; other comparison configurations explicitly
skip this benchmark. The derivation and finite-release assumptions are documented
in [the model review](docs/curvilinear_model_review.md#curvature-verification-analytical-reference).

### Python

`python avalanchers_example.py`

### Frontend

1. Install OpenSSL
2. Go to the project directory
3. Start the development server: `python .\frontend\dev_server.py` This launches the app with caching disabled and HTTPS enabled using self-signed certificates, which are required for WebGPU support in most browsers.
4. Open your browser and go to: [https://localhost/](https://localhost/) If you are accessing the server from another device on the same network, replace `localhost` with the host machine’s IP address.

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
native and Python paths. After changing the Rust API, regenerate the bindings with the `wasm-pack` command above.

#### Zarr stores

Stores follow the layout in `data_processor::store_schema`, written by `data_processor::output` natively and by
`data_processor::zarr_writer` in the browser. Stores of another `avalanchers_format` are rejected:

```
<store>.zarr/
  <site>/            # group, one per site
    dem, x, y        # arrays: the terrain of the site
    <scenario>/      # group, one per scenario
      release_area   # array: the release areas of the scenario
      peak_flow_velocity, peak_flow_thickness  # (run, y, x)
      center_of_mass_x, center_of_mass_y, time # (run, timestep), indexed by step, NaN without enable_center_of_mass
      friction_coefficient, cfl, travel_length, created_at, ...  # (run): results, provenance, every setting
```

The `run` dimension grows in blocks, so only the first `number_of_runs` entries (a scenario attribute) are runs;
`AvalanchersScenario.dataset` already cuts it to those. `x` and `y` are cell centers. Each save holds a lock in the file
`<store>.zarr.lock` next to the store, so several processes can write to one store.

Grids keep row 0 at the south edge, so `y` ascends with the rows, in the store as well as in arrays such as
`sim.peak_velocity`. Formats that list the north row first are flipped when they are read and written, see
`sim.write_asc`. Use `.sortby("y", ascending=False)` on an xarray dataset before exporting it as a north-up raster.

Stores also work as inputs. In the settings, `dem_path` accepts `<store>.zarr`, `<store>.zarr/<site>` or
`<store>.zarr/<site>/<scenario>`; a site or scenario can be named in full, without its hash, or by its hash, and may be
left out if it is the only one. The release areas of the selected scenario are used too, unless `release_areas_path` is
set, which can point into a store as well, or `release_area_fraction` is set and no scenario was named. A site with
several scenarios is an error unless one is named or another release is set; a site with none gets calculated release
areas and a warning. Saving into the same store continues the scenario.

```json
{ "dem_path": "avalanchers.zarr/avaWog/avaWogreleaseTexture" }
```

`release_areas_path` also accepts a polygon shapefile (`.shp`) in the coordinate system of the DEM, native and Python
only. All its features are rasterized onto the DEM grid: cells whose center lies inside a polygon release, holes stay
free. As in AvaFrame, the thickness of each feature comes from its `thickness` or `d0` attribute, either all features
or none must have one, and features must not overlap. Without these attributes `slab_thickness_factor` is used.

Directory uploads use the browser's `webkitdirectory` support, so the whole store folder has to be selected
(dropping the folder onto the drop zone works as well).

Chunks are decoded by the WASM build. The blosc codec (zstd compressor, with the byte-shuffle or bitshuffle
filter) is implemented in pure Rust in `data_processor::blosc`, using [`ruzstd`](https://crates.io/crates/ruzstd)
rather than the C `zstd` bindings, because `zstd-sys` needs a clang cross-compiler to target
`wasm32-unknown-unknown`. Half precision (`float16`) arrays are widened to `f32` when read.

### Benchmarks

``cargo bench -p simulation``

## Code Guidelines

Before pushing use:

Install `pip install ruff`

```
cargo fmt
cargo clippy -- -D warnings
cargo test -p compute_core -p data_processor -p simulation -p cli -p render_core
ruff check ./python_module
pytest python_module
cd crates/wasm_bindings
wasm-pack test --headless --chrome
```

## Data Sources

Test examples are from [AvaFrame](https://docs.avaframe.org/en/latest/testing.html#tests-for-model-validation) under [EUPL-1.2 license
](https://github.com/OpenNHM/AvaFrame#EUPL-1.2-1-ov-file)

Vals data: D'Amboise Christopher J. L., Neuhauser Michael, Teich Michaela, & Fischer Jan-Thomas. (2021). Maverick-bfw/Flow_py_inputs_results: First releases (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5154787

Swiss maps and geodata provided by the Federal Office of Topography swisstopo. © [swisstopo](https://www.swisstopo.admin.ch/en)

Austrian maps provided by  © [basemap.at](https://basemap.at)
