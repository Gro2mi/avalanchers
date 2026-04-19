use compute_core::{Simulation, TimestepData, settings::Settings};
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pollster::FutureExt;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pythonize::depythonize;

// A helper trait to make error conversion less verbose
trait IntoPyResult<T> {
    fn map_runtime_err(self) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> IntoPyResult<T> for Result<T, E> {
    fn map_runtime_err(self) -> PyResult<T> {
        self.map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))
    }
}

#[pyclass]
pub struct PyTimestepData {
    // We store the inner core struct
    inner: TimestepData,
}

#[pymethods]
impl PyTimestepData {
    #[getter]
    fn get_velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.velocity)
    }

    #[getter]
    fn get_position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.position)
    }

    #[getter]
    fn get_acceleration_normal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.acceleration_normal)
    }
    #[getter]
    fn get_acceleration_tangential<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.acceleration_tangential)
    }
    #[getter]
    fn get_normal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.normal)
    }

    #[getter]
    fn get_dt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.dt.to_pyarray(py)
    }
    #[getter]
    fn get_acceleration_friction_magnitude<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f32>> {
        self.inner.acceleration_friction_magnitude.to_pyarray(py)
    }
    #[getter]
    fn get_elevation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.elevation.to_pyarray(py)
    }
}

pub fn to_2d_numpy<'py, const N: usize>(
    py: Python<'py>,
    data: &[[f32; N]],
) -> Bound<'py, PyArray2<f32>> {
    let flattened = data.as_flattened();
    let rows = data.len();

    flattened
        .to_pyarray(py)
        .reshape([rows, N])
        .map_err(|_| PyErr::new::<PyValueError, _>("Dimension mismatch during data conversion"))
        .expect("Failed to convert data to numpy array")
}

#[pyclass]
pub struct PySettings {
    pub inner: Settings,
}

#[allow(clippy::new_without_default)]
#[pymethods]
impl PySettings {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Settings::default(),
        }
    }

    #[staticmethod]
    pub fn from_json(path: String) -> PyResult<Self> {
        let settings =
            Settings::from_json(&path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(PySettings { inner: settings })
    }

    pub fn to_json(&self, path: String) -> PyResult<()> {
        self.inner
            .to_json(&path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))
    }

    #[getter]
    pub fn get_dem_path(&self) -> String {
        self.inner.dem_path.clone()
    }

    #[setter]
    pub fn set_dem_path(&mut self, path: String) {
        self.inner.dem_path = path;
    }
}

#[pyclass]
pub struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    #[staticmethod]
    pub fn create(dict: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json_value: serde_json::Value = depythonize(dict)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

        // 2. Turn that Value into a JSON String
        let json_str = serde_json::to_string(&json_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let settings = Settings::loads(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let inner = Simulation::create(settings.clone())
            .block_on()
            .map_runtime_err()?;
        Ok(PySimulation { inner })
    }

    #[staticmethod]
    pub fn create_default(dem_path: String) -> PyResult<Self> {
        // block_on is used here to bridge async Rust to sync Python
        let inner = Simulation::create_default(dem_path)
            .block_on()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(PySimulation { inner })
    }

    pub fn run(&mut self) -> PyResult<()> {
        self.inner.run().block_on().map_runtime_err()
    }

    #[getter]
    pub fn state(&self) -> String {
        format!("{:?}", self.inner.get_state())
    }

    #[getter]
    pub fn cell_size(&self) -> f32 {
        self.inner.dem.cell_size
    }

    #[getter]
    pub fn dem<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let dims = [self.inner.dem.height, self.inner.dem.width];
        self.inner
            .dem
            .data1d
            .to_pyarray(py)
            .reshape(dims)
            .map_err(|_| {
                PyErr::new::<PyValueError, _>("Dimension mismatch during texture conversion")
            })
    }
    /// Generic helper to get a 2D array from a GPU-backed buffer
    fn get_layer_u32<'py>(
        &self,
        py: Python<'py>,
        data: Vec<u32>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let h = self.inner.dem.height;
        let w = self.inner.dem.width;

        data.to_pyarray(py).reshape([h, w]).map_err(|_| {
            PyErr::new::<PyValueError, _>(format!(
                "Data size {} does not match DEM dimensions {}x{}",
                data.len(),
                h,
                w
            ))
        })
    }
    fn get_layer_f32<'py>(
        &self,
        py: Python<'py>,
        data: Vec<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let h = self.inner.dem.height;
        let w = self.inner.dem.width;

        data.to_pyarray(py).reshape([h, w]).map_err(|_| {
            PyErr::new::<PyValueError, _>(format!(
                "Data size {} does not match DEM dimensions {}x{}",
                data.len(),
                h,
                w
            ))
        })
    }

    #[getter]
    pub fn get_max_velocity<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = self
            .inner
            .get_max_velocity()
            .block_on()
            .map_runtime_err()?
            .to_vec();
        self.get_layer_f32(py, data)
    }

    #[getter]
    pub fn get_cell_count<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let data = self
            .inner
            .get_cell_count()
            .block_on()
            .map_runtime_err()?
            .to_vec();
        self.get_layer_u32(py, data)
    }

    #[getter]
    pub fn get_normals_x<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let normals = self.inner.get_normals_x().block_on().map_runtime_err()?;
        self.get_layer_f32(py, normals.to_vec())
    }

    #[getter]
    pub fn get_release_areas<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = self
            .inner
            .get_release_areas()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, data.to_vec())
    }

    #[getter]
    pub fn get_timestep_data(&mut self) -> PyResult<PyTimestepData> {
        let data = self
            .inner
            .get_timestep_data()
            .block_on()
            .map_runtime_err()?;
        Ok(PyTimestepData {
            inner: data.clone(),
        })
    }

    #[getter]
    pub fn get_elevation_threshold(&self) -> f32 {
        self.inner.get_sim_info().elevation_threshold
    }

    fn convert_rgba_texture<'py>(
        &self,
        py: Python<'py>,
        r: Vec<f32>,
        g: Vec<f32>,
        b: Vec<f32>,
        a: Vec<f32>,
    ) -> PyResult<PyTexture<'py>> {
        let dims = [self.inner.dem.height, self.inner.dem.width];

        let to_arr = |data: Vec<f32>| -> PyResult<Bound<'py, PyArray2<f32>>> {
            data.to_pyarray(py).reshape(dims).map_err(|_| {
                PyErr::new::<PyValueError, _>("Dimension mismatch during texture conversion")
            })
        };

        Ok((to_arr(r)?, to_arr(g)?, to_arr(b)?, to_arr(a)?))
    }
}

type PyTexture<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
);

#[pymodule]
fn _avalanchers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    compute_core::init_logging();

    m.add_class::<PySimulation>()?;
    m.add_class::<PySettings>()?;
    Ok(())
}
