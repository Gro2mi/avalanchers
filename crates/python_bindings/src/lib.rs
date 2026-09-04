//! Python bindings for the avalanchers simulation engine.

use compute_core::{TimestepData, list_devices, settings::Settings};
use data_processor::{settings_from_json_file, settings_to_json_file};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pollster::FutureExt;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

/// Return the names of GPUs available to the simulation backend.
#[pyfunction]
pub fn list_available_gpus() -> PyResult<Vec<String>> {
    let devices = pollster::block_on(list_devices())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    Ok(devices)
}

/// Snapshot of a single simulation timestep of a random particle
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

/// Simulation settings container used by the Python API.
#[pyclass]
pub struct PySettings {
    pub inner: Settings,
}

#[allow(clippy::new_without_default)]
#[pymethods]
impl PySettings {
    /// Create a new settings object with default values.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Settings::default(),
        }
    }

    /// Load settings from a JSON file.
    #[staticmethod]
    pub fn from_json(path: String) -> PyResult<Self> {
        let settings = settings_from_json_file(&path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(PySettings { inner: settings })
    }

    /// Write the current settings to a JSON file.
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

/// Main avalanche simulation object
#[pyclass]
pub struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    /// Create a simulation instance on a specific GPU, if provided.
    #[staticmethod]
    #[pyo3(signature = (gpu=None))]
    pub fn new(gpu: Option<String>) -> PyResult<Self> {
        let inner = Simulation::new_with_gpu(gpu).block_on().map_runtime_err()?;
        Ok(PySimulation { inner })
    }

    /// Configure the simulation from a Python dictionary of settings.
    pub fn create(&mut self, dict: &Bound<'_, PyAny>) -> PyResult<()> {
        let json_value: serde_json::Value = depythonize(dict)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

        // 2. Turn that Value into a JSON String
        let json_str = serde_json::to_string(&json_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let settings = Settings::loads(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_runtime_err()?;
        runtime
            .block_on(self.inner.create(settings.clone()))
            .map_runtime_err()?;
        Ok(())
    }

    pub fn set_max_timesteps(&mut self, max_timesteps: u32) -> PyResult<()> {
        self.inner.settings.max_steps = max_timesteps;
        Ok(())
    }

    /// Initialize the simulation from the bundled example digital elevation model.
    pub fn create_example(&mut self, dem_path: String) -> PyResult<()> {
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

    /// Run the simulation to completion.
    pub fn run(&mut self) -> PyResult<()> {
        self.inner.run().block_on().map_runtime_err()
    }

    /// Advance the simulation by a fixed number of steps.
    pub fn run_n_steps<'py>(
        &mut self,
        py: Python<'py>,
        steps: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sim_info = self.inner.run_n_steps(steps).block_on().map_runtime_err()?;

        let dict = PyDict::new(py);
        dict.set_item("timestep", sim_info.timestep)?;
        dict.set_item("dt", sim_info.dt)?;
        dict.set_item("elapsed_time", sim_info.elapsed_time)?;
        dict.set_item("number_particles", sim_info.number_particles)?;
        dict.set_item("elevation_threshold", sim_info.elevation_threshold)?;
        dict.set_item("max_velocity", sim_info.max_velocity)?;
        dict.set_item("max_flow_thickness", sim_info.max_flow_thickness)?;
        dict.set_item("flags", sim_info.flags)?;
        Ok(dict)
    }

    /// Run post-processing steps after the simulation finishes.
    pub fn post_process(&mut self) -> PyResult<()> {
        self.inner.post_process().block_on().map_runtime_err()
    }

    /// Save the current results to disk. Optionally takes an output path. Default is avalanchers.zarr
    #[pyo3(signature = (path=None))]
    pub fn save(&mut self, path: Option<String>) -> PyResult<()> {
        match path {
            Some(path) => self
                .inner
                .save_with_path(&path)
                .block_on()
                .map_runtime_err(),
            None => self.inner.save().block_on().map_runtime_err(),
        }
    }

    /// Evaluate the final simulation against the configured metrics.
    pub fn evaluate<'a>(&mut self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let (
            iou,
            horizontal_distance,
            vertical_drop,
            horizontal_distance_ref,
            vertical_drop_ref,
            beeline_3d,
            beeline_3d_ref,
            peak_velocity,
        ) = self.inner.evaluate().block_on().map_runtime_err()?;

        let dict = PyDict::new(py);
        dict.set_item("iou", iou)?;
        dict.set_item("horizontal_distance", horizontal_distance)?;
        dict.set_item("vertical_drop", vertical_drop)?;
        dict.set_item("horizontal_distance_ref", horizontal_distance_ref)?;
        dict.set_item("vertical_drop_ref", vertical_drop_ref)?;
        dict.set_item("peak_velocity", peak_velocity)?;
        dict.set_item("beeline_3d", beeline_3d)?;
        dict.set_item("beeline_3d_ref", beeline_3d_ref)?;

        Ok(dict)
    }

    /// Prepare the simulation resources before running it.
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
    pub fn roi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let dims = [self.inner.dem.height, self.inner.dem.width];
        // let roi: Vec<bool> = self
        //     .inner
        //     .roi
        //     .iter()
        //     .flat_map(|&word| {
        //         // Unpack each u32 into 32 booleans (Least Significant Bit first)
        //         (0..32).map(move |bit_idx| (word & (1 << bit_idx)) != 0)
        //     })
        //     .collect();

        self.inner.roi.to_pyarray(py).reshape(dims).map_err(|_| {
            PyErr::new::<PyValueError, _>(format!(
                "Dimension mismatch during texture conversion. Expected: {}x{}, got: {}",
                dims[0],
                dims[1],
                self.inner.roi.len()
            ))
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
            .map_runtime_err()?
            .to_vec();
        self.get_layer_f32(py, data)
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
            .map_runtime_err()?
            .to_vec();
        self.get_layer_f32(py, data)
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
            .map_runtime_err()?
            .to_vec();
        let elevation = self
            .inner
            .fetch_particles_elevation()
            .block_on()
            .map_runtime_err()?
            .to_vec();
        let mut flat_positions: Vec<f32> = Vec::with_capacity(positions.len() * 3);
        for ([x, y], z) in positions.iter().zip(elevation.iter()) {
            flat_positions.push(*x);
            flat_positions.push(*y);
            flat_positions.push(*z);
        }

        // Convert the flat Vec into an Nx3 NumPy Array
        flat_positions.to_pyarray(py).reshape([positions.len(), 3])
    }

    #[getter]
    fn get_particles_position_xy<'py>(
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

        // Convert the flat Vec into an Nx3 NumPy Array
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
            .map_runtime_err()?
            .to_vec();
        let vel_z = self
            .inner
            .fetch_particles_velocity_z()
            .block_on()
            .map_runtime_err()?
            .to_vec();
        let mut flat_velocities: Vec<f32> = Vec::with_capacity(velocities.len() * 3);
        for ([x, y], z) in velocities.iter().zip(vel_z.iter()) {
            flat_velocities.push(*x);
            flat_velocities.push(*y);
            flat_velocities.push(*z);
        }

        // Convert the flat Vec into an Nx2 NumPy Array
        flat_velocities
            .to_pyarray(py)
            .reshape([velocities.len(), 3])
    }

    #[getter]
    fn get_particles_velocity_xy<'py>(
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
        let elevation = self
            .inner
            .fetch_particles_elevation()
            .block_on()
            .map_runtime_err()?;
        Ok(elevation.to_pyarray(py))
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

/// Python module entry point for the avalanche simulator.
#[pymodule]
fn _avalanchers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    init_logging();

    m.add_class::<PySimulation>()?;
    m.add_class::<PySettings>()?;

    m.add_function(wrap_pyfunction!(list_available_gpus, m)?)?;
    Ok(())
}
