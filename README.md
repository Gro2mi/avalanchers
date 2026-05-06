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

Install Rust, follow guide: [https://rust-lang.org/tools/install/]()
Install Python

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

Install Rust, follow guide: [https://rust-lang.org/tools/install/]() (WSL (2026.04.04): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
Install Python

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

1. Install module
2. See `python avalanchers_example.py`

### Frontend

1. Go to this directory
2. Start server with `python .\frontend\dev_server.py` for disabled cache and a secure connection with self signed certs which are needed to use WebGPU (except for localhost where `python -m http.server 8000` works as well)
3. Open Browser on [https://localhost/index.html](https://localhost/index.html) or [https://localhost/index.html?debug=vscode](https://localhost/index.html?debug=vscode) for debugging mode or replace localhost with IP if accessing from another device.

## Code Guidelines

Before pushing use:

Install `pip install ruff`

```
cargo fmt
cargo clippy -- -D warnings
cargo test -p compute_core -p data_processor
ruff check ./python
pytest python
```


## Data Sources

Test examples are from [AvaFrame](https://docs.avaframe.org/en/latest/testing.html#tests-for-model-validation) under [EUPL-1.2 license
](https://github.com/OpenNHM/AvaFrame#EUPL-1.2-1-ov-file)


Vals data: D'Amboise Christopher J. L., Neuhauser Michael, Teich Michaela, & Fischer Jan-Thomas. (2021). Maverick-bfw/Flow_py_inputs_results: First releases (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5154787
