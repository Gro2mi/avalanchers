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

### Python

`python avalanchers_example.py`

### Frontend

1. Install OpenSSL
2. Go to the project directory
3. Start the development server: `python .\frontend\dev_server.py` This launches the app with caching disabled and HTTPS enabled using self-signed certificates, which are required for WebGPU support in most browsers.
4. Open your browser and go to: [https://localhost/](https://localhost/) If you are accessing the server from another device on the same network, replace `localhost` with the host machine’s IP address.

### Benchmarks

```cargo bench -p simulation```

## Code Guidelines

Before pushing use:

Install `pip install ruff`

```
cargo fmt
cargo clippy -- -D warnings
cargo test -p compute_core -p data_processor -p simulation
ruff check ./python_module
pytest python_module
```

## Data Sources

Test examples are from [AvaFrame](https://docs.avaframe.org/en/latest/testing.html#tests-for-model-validation) under [EUPL-1.2 license
](https://github.com/OpenNHM/AvaFrame#EUPL-1.2-1-ov-file)

Vals data: D'Amboise Christopher J. L., Neuhauser Michael, Teich Michaela, & Fischer Jan-Thomas. (2021). Maverick-bfw/Flow_py_inputs_results: First releases (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5154787

### Avalanche outlines (`data/pools/`)

The per-case avalanche outlines under `data/pools/<year>/cases/` are extracted
subsets of three published datasets from the WSL Institute for Snow and
Avalanche Research SLF, redistributed here under their respective licences.
Full provenance — authors, DOI, licence, distribution filename, exact byte
count, sha256, and the filter funnel applied — is recorded in
`data/pools/<year>/SOURCE.md`.

**Two different licences apply. They are not interchangeable.**

*2018 and 2019 — ODbL 1.0* ([opendatacommons.org/licenses/odbl/1-0/](https://opendatacommons.org/licenses/odbl/1-0/)):

> Hafner, E. & Bühler, Y. (2019). *SPOT6 Avalanche outlines 24 January 2018.*
> EnviDat. [doi:10.16904/envidat.77](https://doi.org/10.16904/envidat.77)

> Hafner, E. & Bühler, Y. *SPOT6 Avalanche outlines 16 January 2019.*
> EnviDat. [doi:10.16904/envidat.235](https://doi.org/10.16904/envidat.235)

Attribution is required, and the ODbL's share-alike condition applies to
derived *databases*.

*1999 — CC-BY-SA 4.0* ([creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)):

> Hafner, E. D. & Dal, J. F. *Avalanche outlines February and March 1999 from
> aerial imagery.* EnviDat.
> [doi:10.16904/envidat.579](https://doi.org/10.16904/envidat.579)

Attribution is required, and the share-alike condition applies to derivative
*works*. Do not carry the ODbL wording over to this dataset, or vice versa.

Only the filtered candidate pools are in this repository (1,502 cases). The
master distributions are roughly 350 MB and are not redistributed here; each
sidecar carries the DOI and checksum needed to fetch and verify the original.

Terrain throughout is swissALTI3D © swisstopo (open government data).
