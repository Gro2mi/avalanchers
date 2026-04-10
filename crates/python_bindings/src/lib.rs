// use numpy::{PyArrayDyn, PyReadonlyArrayDyn, IntoPyArray};
use compute_core::Simulation;
use compute_core::settings::Settings;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use numpy::ToPyArray;
use pollster::FutureExt;
use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

#[pyclass]
pub struct PySimulation {
    // Wrap the core struct
    inner: Simulation,
}

#[pyclass]
#[derive(Default)]
pub struct PySettings {
    inner: Settings,
}

#[pymethods]
impl PySettings {
    #[new]
    pub fn new() -> Self {
        PySettings {
            inner: Settings::new(),
        }
    }

    #[staticmethod]
    pub fn from_json(path: String) -> PyResult<Self> {
        let settings = Settings::from_json(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PySettings { inner: settings })
    }

    pub fn to_json(&self, path: String) -> PyResult<()> {
        self.inner
            .to_json(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    // Expose dem_path so Python can tweak it manually if needed
    #[getter]
    pub fn get_dem_path(&self) -> String {
        self.inner.dem_path.clone()
    }

    #[setter]
    pub fn set_dem_path(&mut self, path: String) {
        self.inner.dem_path = path;
    }
}

#[pymethods]
impl PySimulation {
    // We use a static method for 'create' because Python '__init__' can't be async
    #[allow(clippy::new_ret_no_self)]
    #[staticmethod]
    pub fn create_default(dem_path: String) -> PyResult<Self> {
        // block_on is used here to bridge async Rust to sync Python
        let inner = Simulation::create_default(dem_path)
            .block_on()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(PySimulation { inner })
    }

    pub fn get_state(&self) -> String {
        format!("{:?}", self.inner.get_state())
    }

    pub fn run(&mut self) -> PyResult<()> {
        self.inner
            .run()
            .block_on()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }

    // Expose fields as properties
    #[getter]
    pub fn dem_path(&self) -> String {
        self.inner.dem_path.clone()
    }

    pub fn get_normals_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyTexture<'py>> {
        // 1. Get the data from the core (async -> sync)
        let (nx, ny, nz, nw) = self
            .inner
            .get_normals_texture()
            .block_on()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // 2. Convert each Vec<f32> into a NumPy array
        // .to_pyarray(py) copies the data into memory managed by NumPy
        let dims = [self.inner.dem.height, self.inner.dem.width];
        let rx = nx.to_pyarray(py).reshape(dims).map_err(to_val_err)?;
        let ry = ny.to_pyarray(py).reshape(dims).map_err(to_val_err)?;
        let rz = nz.to_pyarray(py).reshape(dims).map_err(to_val_err)?;
        let rw = nw.to_pyarray(py).reshape(dims).map_err(to_val_err)?;

        Ok((rx, ry, rz, rw))
    }

    pub fn get_dem_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let dims = [self.inner.dem.height, self.inner.dem.width];
        self.inner
            .dem
            .data1d
            .to_pyarray(py)
            .reshape(dims)
            .map_err(to_val_err)
    }
}

type PyTexture<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
);

fn to_val_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "Dimension mismatch: Could not reshape data into 1001x401. Error: {}",
        e
    ))
}

#[pymodule]
fn avalanchers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1. Initialize the bridge between Rust and Python logging
    pyo3_log::init();
    compute_core::init_logging();

    m.add_class::<PySimulation>()?;
    m.add_class::<PySettings>()?;
    Ok(())
}
