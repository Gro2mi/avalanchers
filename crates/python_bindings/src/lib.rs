use compute_core::{TimestepData, settings::Settings};
use data_processor::{settings_from_json_file, settings_to_json_file};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pollster::FutureExt;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pythonize::depythonize;
use simulation::{Simulation, init_logging};

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
    fn get_dt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.dt.to_pyarray(py)
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
        let settings = settings_from_json_file(&path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(PySettings { inner: settings })
    }

    pub fn to_json(&self, path: String) -> PyResult<()> {
        settings_to_json_file(&self.inner, &path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))
    }

    #[getter]
    pub fn get_dem_path(&self) -> String {
        self.inner.dem_path.clone().unwrap_or_else(|| "".into())
    }

    #[setter]
    pub fn set_dem_path(&mut self, path: String) {
        self.inner.dem_path = Some(path);
    }
}

#[pyclass]
pub struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    #[staticmethod]
    pub fn new() -> PyResult<Self> {
        let inner = Simulation::new().block_on().map_runtime_err()?;
        Ok(PySimulation { inner })
    }

    pub fn create(&mut self, dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let json_value: serde_json::Value = depythonize(dict)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

        // 2. Turn that Value into a JSON String
        let json_str = serde_json::to_string(&json_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let settings = Settings::loads(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        self.inner
            .create(settings.clone())
            .block_on()
            .map_runtime_err()?;
        Ok(())
    }

    pub fn create_example(&mut self, dem_path: String) -> PyResult<()> {
        // block_on is used here to bridge async Rust to sync Python
        self.inner
            .create_example(&dem_path)
            .block_on()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(())
    }

    pub fn set_dem(&mut self, dem_data: PyReadonlyArray2<f32>, cell_size: f32) -> PyResult<()> {
        let view = dem_data.as_array();
        let height = view.shape()[0];
        let width = view.shape()[1];

        // Ensure the data is contiguous in memory so we can treat it as a slice
        let slice = dem_data.as_slice()?;

        self.inner
            .set_dem(slice, width, height, cell_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_dem_with_bounds(
        &mut self,
        dem_data: PyReadonlyArray2<f32>, // Accepts (height, width) array
        cell_size: f32,
        bounds_xmin: f32,
        bounds_xmax: f32,
        bounds_ymin: f32,
        bounds_ymax: f32,
        map_factor: f32,
    ) -> PyResult<()> {
        // NumPy shape is usually (height, width)
        let view = dem_data.as_array();
        let height = view.shape()[0];
        let width = view.shape()[1];

        // Ensure the data is contiguous in memory so we can treat it as a slice
        let slice = dem_data.as_slice()?;

        self.inner
            .set_dem_with_bounds(
                slice,
                width,
                height,
                cell_size,
                bounds_xmin,
                bounds_xmax,
                bounds_ymin,
                bounds_ymax,
                map_factor,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    pub fn set_dem_default(
        &mut self,
        dem_data: PyReadonlyArray2<f32>,
        cell_size: f32,
    ) -> PyResult<()> {
        let view = dem_data.as_array();
        let height = view.shape()[0];
        let width = view.shape()[1];

        let slice = dem_data.as_slice()?;

        self.inner
            .set_dem_default(slice, width, height, cell_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    pub fn set_release_areas(&mut self, release_areas: PyReadonlyArray2<f32>) -> PyResult<()> {
        self.inner
            .set_release_areas(release_areas.as_array().as_slice().expect("Failed to convert release areas to slice. In case you manipulated the numpy array, try passing it with .copy() to ensure it's contiguous in memory."))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    pub fn run(&mut self) -> PyResult<()> {
        self.inner.run().block_on().map_runtime_err()
    }

    pub fn prepare(&mut self) -> PyResult<()> {
        self.inner.prepare().block_on().map_runtime_err()
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
    pub fn released_particles_per_cell(&self) -> u32 {
        self.inner.settings.released_particles_per_cell
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

    #[getter]
    pub fn dem_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let bounds = [
            self.inner.dem.bounds.xmin,
            self.inner.dem.bounds.xmax,
            self.inner.dem.bounds.ymin,
            self.inner.dem.bounds.ymax,
        ];
        Ok(bounds.to_pyarray(py))
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
    pub fn get_peak_velocity<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = self
            .inner
            .fetch_peak_velocity()
            .block_on()
            .map_runtime_err()?
            .to_vec();
        self.get_layer_f32(py, data)
    }

    #[getter]
    pub fn get_terrain_geometry_x<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let terrain = self
            .inner
            .get_terrain_geometry_x()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, terrain.to_vec())
    }

    #[getter]
    pub fn get_terrain_geometry_y<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let terrain = self
            .inner
            .get_terrain_geometry_y()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, terrain.to_vec())
    }

    #[getter]
    pub fn get_terrain_geometry_z<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let terrain = self
            .inner
            .get_terrain_geometry_z()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, terrain.to_vec())
    }

    #[getter]
    pub fn get_gravity_x<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (gravity_x, _) = self
            .inner
            .get_slope_gravity()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, gravity_x.to_vec())
    }

    #[getter]
    pub fn get_gravity_y<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (_, gravity_y) = self
            .inner
            .get_slope_gravity()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, gravity_y.to_vec())
    }

    #[getter]
    pub fn get_release_areas<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = self
            .inner
            .fetch_release_areas()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, data.to_vec())
    }

    #[getter]
    pub fn get_peak_flow_thickness<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let data = self
            .inner
            .fetch_peak_flow_thickness()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, data.to_vec())
    }

    #[getter]
    pub fn get_timestep_data(&mut self) -> PyResult<PyTimestepData> {
        let data = self
            .inner
            .fetch_timestep_data()
            .block_on()
            .map_runtime_err()?;
        Ok(PyTimestepData {
            inner: data.clone(),
        })
    }

    #[getter]
    pub fn get_elevation_threshold(&mut self) -> PyResult<f32> {
        Ok(self
            .inner
            .fetch_sim_info()
            .block_on()
            .map_runtime_err()?
            .elevation_threshold)
    }

    #[getter]
    fn get_particles_position<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let positions = self
            .inner
            .fetch_particles_position()
            .block_on()
            .map_runtime_err()?;
        let mut flat_positions: Vec<f32> = Vec::with_capacity(positions.len() * 2);
        for [x, y] in positions {
            flat_positions.push(*x);
            flat_positions.push(*y);
        }

        // Convert the flat Vec into an Nx2 NumPy Array
        flat_positions.to_pyarray(py).reshape([positions.len(), 2])
    }

    #[getter]
    fn get_particles_velocity<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let velocities = self
            .inner
            .fetch_particles_velocity()
            .block_on()
            .map_runtime_err()?;
        let mut flat_velocities: Vec<f32> = Vec::with_capacity(velocities.len() * 2);
        for [x, y] in velocities {
            flat_velocities.push(*x);
            flat_velocities.push(*y);
        }

        // Convert the flat Vec into an Nx2 NumPy Array
        flat_velocities
            .to_pyarray(py)
            .reshape([velocities.len(), 2])
    }

    #[getter]
    fn get_particles_elevation<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let stopped = self
            .inner
            .fetch_particles_elevation()
            .block_on()
            .map_runtime_err()?;
        Ok(stopped.to_pyarray(py))
    }

    #[getter]
    fn get_stopped<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let stopped = self
            .inner
            .fetch_particles_stopped()
            .block_on()
            .map_runtime_err()?;
        Ok(stopped.to_pyarray(py))
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
    init_logging();

    m.add_class::<PySimulation>()?;
    m.add_class::<PySettings>()?;
    Ok(())
}
