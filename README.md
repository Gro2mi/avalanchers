# Avalanchers

GPU-accelerated avalanche simulation written in Rust on WebGPU, with a browser frontend (WASM) including a live renderer, Python bindings and a CLI. Building on ideas from [weBIGeo](https://github.com/weBIGeo/webigeo/tree/main), it gives fast estimates of runout distance and flow routing for AvaFrame examples, your own DEMs or Austrian GPX tracks. It uses a depth-averaged material point method (DAMPM), with the goal of calibrating friction parameters on a large dataset. This is a work in progress and many features are not verified yet.



https://github.com/user-attachments/assets/9d240487-5569-427c-8a1f-00f1af1d59f5

<sub>Live simulation of the AvaFrame example <i>avaHit</i>: particles colored by velocity, overlay shows peak flow thickness. Orthophoto: [basemap.at](https://basemap.at) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), draped on the terrain model.</sub>



**[▶ Try the demo](https://gro2mi.github.io/avalanchers/)**

## Requirements

- **Browser:** WebGPU support ([status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), [test your device and browser](https://webgpureport.org/)). Tested on Windows, Linux, Mac and Android. On Windows with multiple GPUs, Chrome may pick the integrated one, see [docs/frontend.md](docs/frontend.md#chrome-on-windows-with-multiple-gpus).
- **Native / Python:** a GPU with Vulkan, DirectX 12 or Metal support. **On Linux, Vulkan is required** (GPU driver + Vulkan loader); check with `vulkaninfo --summary` (from `vulkan-tools`). Browsers on Linux may also need WebGPU enabled, e.g. Chrome with `--enable-unsafe-webgpu --enable-features=Vulkan`.
- **Python** ≥ 3.10

## Quick start (Python)

```
pip install avalanchers[viz]
```

```python
import avalanchers

sim = avalanchers.PySimulation.new()
sim.create_example("frontend/data/avaframe/avaMal.png")
sim.run()

avalanchers.plot2d(sim, "max_velocity")  # needs [viz]
avalanchers.plot3d(sim, "max_velocity")
```

## Development setup

Install [Rust](https://rust-lang.org/tools/install/) and [Python](https://www.python.org/).

**Linux** (additionally):

```
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential clang lld pkg-config libssl-dev

# Vulkan loader, Mesa drivers (Intel/AMD) and vulkaninfo for checking
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary   # should list your GPU
```

NVIDIA GPUs need the proprietary driver (e.g. `sudo ubuntu-drivers install`), which ships its own Vulkan driver. Under WSL2, Vulkan support is limited and may fall back to software rendering.

**All platforms:**

```
cargo install wasm-bindgen-cli wasm-pack   # wasm bindings
cargo install cargo-tarpaulin              # coverage (optional)

python -m venv .venv
.venv\Scripts\Activate.ps1                  # Linux: source .venv/bin/activate
python -m pip install maturin ruff
```

## Build

```
# CLI
cargo build -p cli --release

# Python bindings
maturin develop                 # into the active venv
maturin build --release         # wheel

# WASM bindings (dev / release)
wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --no-opt
wasm-pack build crates/wasm_bindings --target web --out-dir ../../frontend/js/pkg --release
```

## Run

- **Python:** `python avalanchers_example.py`
- **Frontend:** requires OpenSSL. From the project root run `python frontend/dev_server.py` and open [https://localhost/](https://localhost/) (HTTPS with self-signed certs, needed for WebGPU; use the host IP from other devices).
- **Benchmarks:** `cargo bench -p simulation`

## Before pushing

```
cargo fmt
cargo clippy -- -D warnings
cargo test -p compute_core -p data_processor -p simulation -p cli -p render_core
ruff check ./python_module
pytest python_module
cd crates/wasm_bindings && wasm-pack test --headless --chrome
```

## Documentation

- [Frontend workflow](docs/frontend.md)
- [DAMPM global model](docs/dampm_global.md)
- [Data inputs, release areas and Zarr stores](docs/data_inputs.md)
- [Model conventions, travel metrics and benchmarks](docs/model_conventions.md)
## Data sources

- Test examples from [AvaFrame](https://docs.avaframe.org/en/latest/testing.html#tests-for-model-validation) ([EUPL-1.2](https://github.com/OpenNHM/AvaFrame#EUPL-1.2-1-ov-file))
- Vals data: D'Amboise C. J. L., Neuhauser M., Teich M., Fischer J.-T. (2021). Maverick-bfw/Flow_py_inputs_results (v1). Zenodo. https://doi.org/10.5281/zenodo.5154787
- Swiss maps and geodata © [swisstopo](https://www.swisstopo.admin.ch/en)
- Austrian maps © [basemap.at](https://basemap.at)
