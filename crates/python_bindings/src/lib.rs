//! Python bindings for the avalanchers GPU avalanche simulation engine.
//!
//! The module (``avalanchers._avalanchers``) exposes the [`PySimulation`]
//! entry point, [`PySettings`] for configuration, and read-only result
//! objects ([`PyEvaluation`], [`PySimInfo`], [`PyTimestepData`],
//! [`PyChamferDistance`]) returned by the simulation.
//!
//! Distances are in world units (meters), velocities in m/s, and all grid
//! results are returned as ``(height, width)`` numpy arrays matching the
//! DEM shape.

use compute_core::{
    SimInfo, TimestepData,
    evaluation::{ChamferDistance, MassMovementEvaluation},
    list_devices,
    settings::Settings,
};
use data_processor::{settings_from_json_file, settings_to_json_file};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pollster::FutureExt;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
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

/// Return the names of the GPUs available to the simulation backend.
///
/// The names can be passed to :meth:`PySimulation.new` to pin a simulation
/// to a specific device.
///
/// Returns:
///     list[str]: Names of all adapters visible to wgpu.
#[pyfunction]
pub fn list_available_gpus() -> PyResult<Vec<String>> {
    let devices = pollster::block_on(list_devices()).map_runtime_err()?;
    Ok(devices)
}

/// Diagnostic state recorded after a simulation timestep.
///
/// Returned by :meth:`PySimulation.run_n_steps` after each call. Also
/// supports mapping access, e.g. ``info["timestep"]``.
#[pyclass]
pub struct PySimInfo {
    inner: SimInfo,
}

#[pymethods]
impl PySimInfo {
    /// Current timestep number (1-based).
    #[getter]
    fn timestep(&self) -> u32 {
        self.inner.timestep
    }

    /// Timestep length in seconds used for this step.
    #[getter]
    fn dt(&self) -> f32 {
        self.inner.dt
    }

    /// Simulated time in seconds since the simulation started.
    #[getter]
    fn elapsed_time(&self) -> f32 {
        self.inner.elapsed_time
    }

    /// Number of particles currently alive in the simulation.
    #[getter]
    fn number_particles(&self) -> u32 {
        self.inner.number_particles
    }

    /// DEM elevation below which cells cannot be reached by the avalanche.
    #[getter]
    fn elevation_threshold(&self) -> f32 {
        self.inner.elevation_threshold
    }

    /// Maximum particle velocity of the last step in m/s.
    #[getter]
    fn max_velocity(&self) -> f32 {
        self.inner.max_velocity
    }

    /// Maximum flow thickness of the last step in meters.
    #[getter]
    fn max_flow_thickness(&self) -> f32 {
        self.inner.max_flow_thickness
    }

    /// Bitmask of simulation status flags.
    #[getter]
    fn flags(&self) -> u32 {
        self.inner.flags
    }

    /// Mapping-style access, e.g. ``info["timestep"]``.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "timestep" => self.inner.timestep.into_py_any(py),
            "dt" => self.inner.dt.into_py_any(py),
            "elapsed_time" => self.inner.elapsed_time.into_py_any(py),
            "number_particles" => self.inner.number_particles.into_py_any(py),
            "elevation_threshold" => self.inner.elevation_threshold.into_py_any(py),
            "max_velocity" => self.inner.max_velocity.into_py_any(py),
            "max_flow_thickness" => self.inner.max_flow_thickness.into_py_any(py),
            "flags" => self.inner.flags.into_py_any(py),
            other => Err(PyKeyError::new_err(format!(
                "Unknown sim info field: '{other}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SimInfo(timestep={}, dt={:.4}, elapsed_time={:.3}, number_particles={}, \
             max_velocity={:.3}, max_flow_thickness={:.4}, flags={:#x})",
            self.inner.timestep,
            self.inner.dt,
            self.inner.elapsed_time,
            self.inner.number_particles,
            self.inner.max_velocity,
            self.inner.max_flow_thickness,
            self.inner.flags,
        )
    }
}

/// Diagonal-normalized chamfer distance between two cell sets.
///
/// All values are distances measured in world units and then divided by the
/// length of the grid diagonal, so they are dimensionless and comparable
/// across grid resolutions. A value of 0 means the two cell sets coincide.
#[pyclass]
pub struct PyChamferDistance {
    inner: ChamferDistance,
}

#[pymethods]
impl PyChamferDistance {
    /// Mean distance from every simulated cell to the nearest reference
    /// (region of interest) cell, normalized by the grid diagonal.
    ///
    /// Infinite if the simulation produced cells but the reference is empty.
    #[getter]
    fn sim_to_roi(&self) -> f64 {
        self.inner.sim_to_roi
    }

    /// Mean distance from every reference cell to the nearest simulated
    /// cell, normalized by the grid diagonal.
    ///
    /// Infinite if the reference has cells but the simulation is empty.
    #[getter]
    fn roi_to_sim(&self) -> f64 {
        self.inner.roi_to_sim
    }

    /// Symmetric chamfer distance, ``sim_to_roi + roi_to_sim``.
    ///
    /// 0.0 if both sets are empty, infinite if exactly one is empty.
    #[getter]
    fn chamfer(&self) -> f64 {
        self.inner.chamfer
    }

    fn __repr__(&self) -> String {
        format!(
            "ChamferDistance(sim_to_roi={:.6}, roi_to_sim={:.6}, chamfer={:.6})",
            self.inner.sim_to_roi, self.inner.roi_to_sim, self.inner.chamfer
        )
    }
}

/// Complete evaluation of a finished simulation.
///
/// Combines the GPU-computed metrics (overlap components, chamfer distance
/// and the beeline distance between the highest and lowest avalanche point)
/// with the reference-based runout and velocity diagnostics. All distances
/// are in meters. Also supports mapping access, e.g. ``result["iou"]``.
#[pyclass]
pub struct PyEvaluation {
    metrics: MassMovementEvaluation,
}

#[pymethods]
impl PyEvaluation {
    /// Overlap normalized component (intersection / union) in [0, 1].
    #[getter]
    fn intersection(&self) -> f64 {
        self.metrics.intersection
    }

    /// Underestimation normalized component: reference cells the simulation
    /// missed, normalized by the union.
    #[getter]
    fn undershoot(&self) -> f64 {
        self.metrics.undershoot
    }

    /// Overestimation normalized component: cells the simulation reached
    /// without a reference, normalized by the union.
    #[getter]
    fn overshoot(&self) -> f64 {
        self.metrics.overshoot
    }

    /// Intersection over Union (IoU), also known as Jaccard index, in [0, 1]. 1.0 is a perfect match.
    #[getter]
    fn iou(&self) -> f64 {
        self.metrics.iou
    }

    /// 3D beeline distance in meters between the highest and the lowest
    /// point of the simulated avalanche (extreme DEM elevations among the
    /// avalanche cells).
    #[getter]
    fn sim_to_roi(&self) -> f64 {
        self.metrics.sim_to_roi
    }

    /// Diagonal-normalized chamfer distance between the simulated cells and
    /// the reference region of interest (see :class:`PyChamferDistance`).
    #[getter]
    fn roi_to_sim(&self) -> f64 {
        self.metrics.roi_to_sim
    }

    #[getter]
    fn chamfer(&self) -> f64 {
        self.metrics.chamfer
    }

    #[getter]
    fn chamfer_distance(&self) -> PyChamferDistance {
        PyChamferDistance {
            inner: ChamferDistance {
                sim_to_roi: self.metrics.sim_to_roi,
                roi_to_sim: self.metrics.roi_to_sim,
                chamfer: self.metrics.chamfer,
            },
        }
    }

    /// Horizontal (plan view) distance in meters between the highest and
    /// lowest point of the simulated avalanche mask.
    #[getter]
    fn horizontal_distance(&self) -> f32 {
        self.metrics.horizontal_distance as f32
    }

    /// Vertical drop in meters between the highest and lowest point of the
    /// simulated avalanche mask.
    #[getter]
    fn vertical_drop(&self) -> f32 {
        self.metrics.vertical_drop as f32
    }

    /// 3D beeline of the simulated mask,
    /// ``sqrt(horizontal_distance^2 + vertical_drop^2)``, in meters.
    #[getter]
    fn beeline_3d(&self) -> f32 {
        self.metrics.beeline_3d as f32
    }

    /// Horizontal distance in meters covered by the reference region of
    /// interest.
    #[getter]
    fn horizontal_distance_ref(&self) -> f32 {
        self.metrics.horizontal_distance_ref as f32
    }

    /// Vertical drop in meters of the reference region of interest.
    #[getter]
    fn vertical_drop_ref(&self) -> f32 {
        self.metrics.vertical_drop_ref as f32
    }

    /// 3D beeline of the reference region of interest, in meters.
    #[getter]
    fn beeline_3d_ref(&self) -> f32 {
        self.metrics.beeline_3d_ref as f32
    }

    /// Maximum peak flow velocity of the simulation in m/s.
    #[getter]
    fn peak_velocity(&self) -> f32 {
        self.metrics.peak_velocity as f32
    }

    /// Mapping-style access, e.g. ``result["iou"]``.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "intersection" => self.metrics.intersection.into_py_any(py),
            "undershoot" => self.metrics.undershoot.into_py_any(py),
            "overshoot" => self.metrics.overshoot.into_py_any(py),
            "iou" => self.metrics.iou.into_py_any(py),
            "horizontal_distance" => self.metrics.horizontal_distance.into_py_any(py),
            "vertical_drop" => self.metrics.vertical_drop.into_py_any(py),
            "horizontal_distance_ref" => self.metrics.horizontal_distance_ref.into_py_any(py),
            "vertical_drop_ref" => self.metrics.vertical_drop_ref.into_py_any(py),
            "beeline_3d" => self.metrics.beeline_3d.into_py_any(py),
            "beeline_3d_ref" => self.metrics.beeline_3d_ref.into_py_any(py),
            "peak_velocity" => self.metrics.peak_velocity.into_py_any(py),
            "sim_to_roi" => self.metrics.sim_to_roi.into_py_any(py),
            "roi_to_sim" => self.metrics.roi_to_sim.into_py_any(py),
            "chamfer" => self.metrics.chamfer.into_py_any(py),
            "chamfer_distance" => Ok(Py::new(
                py,
                PyChamferDistance {
                    inner: ChamferDistance {
                        sim_to_roi: self.metrics.sim_to_roi,
                        roi_to_sim: self.metrics.roi_to_sim,
                        chamfer: self.metrics.chamfer,
                    },
                },
            )?
            .into_any()),
            other => Err(PyKeyError::new_err(format!(
                "Unknown evaluation metric: '{other}'"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Evaluation(intersection={:.4}, undershoot={:.4}, overshoot={:.4}, \
             iou={:.4}, beeline_3d={:.2}, beeline_3d_ref={:.2}, peak_velocity={:.2}, \
             chamfer={:.6})",
            self.metrics.intersection,
            self.metrics.undershoot,
            self.metrics.overshoot,
            self.metrics.iou,
            self.metrics.beeline_3d,
            self.metrics.beeline_3d_ref,
            self.metrics.peak_velocity,
            self.metrics.chamfer,
        )
    }
}

/// Snapshot of a single simulation timestep of one tracked particle.
///
/// Returned by the :attr:`PySimulation.timestep_data` property. All arrays
/// are per-timestep traces of the particle with index 0.
#[pyclass]
pub struct PyTimestepData {
    inner: TimestepData,
}

#[pymethods]
impl PyTimestepData {
    /// ``(timesteps, 3)`` float32 array of the particle x/y/z velocity.
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.velocity)
    }

    /// ``(timesteps, 3)`` float32 array of the particle x/y/z position.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_2d_numpy(py, &self.inner.position)
    }

    /// float32 array of the timestep lengths in seconds.
    #[getter]
    fn dt<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
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
///
/// Wraps the JSON settings schema. Create with default values, or load from
/// / save to JSON files. The full set of physical parameters is configured
/// through :meth:`PySimulation.create` with a dictionary.
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
    ///
    /// Args:
    ///     path: Path to a JSON settings file.
    #[staticmethod]
    pub fn from_json(path: String) -> PyResult<Self> {
        let settings = settings_from_json_file(&path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;
        Ok(PySettings { inner: settings })
    }

    /// Write the current settings to a JSON file.
    ///
    /// Args:
    ///     path: Destination path for the JSON settings file.
    pub fn to_json(&self, path: String) -> PyResult<()> {
        settings_to_json_file(&self.inner, &path)
            .map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))
    }

    /// Path to the digital elevation model, or an empty string if unset.
    #[getter]
    pub fn get_dem_path(&self) -> String {
        self.inner.dem_path.clone().unwrap_or_else(|| "".into())
    }

    /// Set the path to the digital elevation model.
    #[setter]
    pub fn set_dem_path(&mut self, path: String) {
        self.inner.dem_path = Some(path);
    }
}

/// Main avalanche simulation object.
///
/// Lifecycle: create with :meth:`new`, configure with :meth:`create` (or one
/// of the ``set_*`` methods / :meth:`create_example`), then :meth:`prepare`,
/// :meth:`run` (or :meth:`run_n_steps`), :meth:`post_process`, and finally
/// :meth:`evaluate` and :meth:`save`.
#[pyclass]
pub struct PySimulation {
    inner: Simulation,
}

#[pymethods]
impl PySimulation {
    /// Create a simulation instance, optionally pinned to a GPU.
    ///
    /// Args:
    ///     gpu: Name of the GPU to use (see :func:`list_available_gpus`).
    ///         Defaults to the primary adapter.
    #[staticmethod]
    #[pyo3(signature = (gpu=None))]
    pub fn new(gpu: Option<String>) -> PyResult<Self> {
        let inner = Simulation::new_with_gpu(gpu).block_on().map_runtime_err()?;
        Ok(PySimulation { inner })
    }

    /// Configure the simulation from a settings dictionary.
    ///
    /// The dictionary follows the avalanchers JSON settings schema (the same
    /// keys as :class:`PySettings`, all optional).
    ///
    /// Args:
    ///     settings: Dictionary of settings, e.g.
    ///         ``{"peak_flow_thickness_threshold": 0.1}``.
    pub fn create(&mut self, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let json_value: serde_json::Value =
            depythonize(settings).map_err(|e| PyErr::new::<PyTypeError, _>(e.to_string()))?;

        // 2. Turn that Value into a JSON String
        let json_str = serde_json::to_string(&json_value).map_runtime_err()?;
        let parsed = Settings::loads(&json_str).map_runtime_err()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_runtime_err()?;
        runtime
            .block_on(self.inner.create(parsed.clone()))
            .map_runtime_err()?;
        Ok(())
    }

    /// Set the maximum number of timesteps the simulation may take.
    ///
    /// Args:
    ///     max_timesteps: Upper bound on simulation steps.
    pub fn set_max_timesteps(&mut self, max_timesteps: u32) -> PyResult<()> {
        self.inner.settings.max_steps = max_timesteps;
        Ok(())
    }

    /// Initialize the simulation from a DEM image bundled with the repository.
    ///
    /// Args:
    ///     dem_path: Path to the example DEM, relative to the repository root
    ///         (e.g. ``"data/avaframe/avaParabola.png"``).
    pub fn create_example(&mut self, dem_path: String) -> PyResult<()> {
        self.inner
            .create_example(&dem_path)
            .block_on()
            .map_runtime_err()
    }

    /// Set the digital elevation model from a 2D float32 array.
    ///
    /// Args:
    ///     dem_data: ``(height, width)`` array of elevations in meters. Must
    ///         be C-contiguous (call ``.copy()`` on non-contiguous input).
    ///     cell_size: Cell edge length in meters.
    pub fn set_dem(&mut self, dem_data: PyReadonlyArray2<f32>, cell_size: f32) -> PyResult<()> {
        let view = dem_data.as_array();
        let height = view.shape()[0];
        let width = view.shape()[1];

        // Ensure the data is contiguous in memory so we can treat it as a slice
        let slice = dem_data.as_slice()?;

        self.inner
            .set_dem(slice, width, height, cell_size)
            .map_runtime_err()
    }

    /// Set the DEM together with its real-world geographic bounds.
    ///
    /// Args:
    ///     dem_data: ``(height, width)`` array of elevations in meters.
    ///     cell_size: Cell edge length in meters.
    ///     bounds_xmin: Smallest x coordinate of the DEM in meters.
    ///     bounds_xmax: Largest x coordinate of the DEM in meters.
    ///     bounds_ymin: Smallest y coordinate of the DEM in meters.
    ///     bounds_ymax: Largest y coordinate of the DEM in meters.
    ///     map_factor: Scale factor between map units and world units.
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
            .map_runtime_err()
    }

    /// Set the DEM using defaults derived from the data itself.
    ///
    /// Args:
    ///     dem_data: ``(height, width)`` array of elevations in meters.
    ///     cell_size: Cell edge length in meters.
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
            .map_runtime_err()
    }

    /// Set the release areas from a 2D float32 array.
    ///
    /// Non-zero cells act as release zones for the avalanche.
    ///
    /// Args:
    ///     release_areas: ``(height, width)`` array matching the DEM shape.
    ///         Must be C-contiguous (call ``.copy()`` if needed).
    pub fn set_release_areas(&mut self, release_areas: PyReadonlyArray2<f32>) -> PyResult<()> {
        self.inner
            .set_release_areas(release_areas.as_array().as_slice().expect("Failed to convert release areas to slice. In case you manipulated the numpy array, try passing it with .copy() to ensure it's contiguous in memory."))
            .map_runtime_err()
    }

    /// Run the simulation to completion (up to ``max_timesteps``).
    pub fn run(&mut self) -> PyResult<()> {
        self.inner.run().block_on().map_runtime_err()
    }

    /// Advance the simulation by a fixed number of steps.
    ///
    /// Args:
    ///     steps: Number of timesteps to simulate. ``0`` initializes and
    ///         returns the state after preparation.
    ///
    /// Returns:
    ///     PySimInfo: Diagnostic state after the executed steps.
    pub fn run_n_steps(&mut self, steps: u32) -> PyResult<PySimInfo> {
        let sim_info = self.inner.run_n_steps(steps).block_on().map_runtime_err()?;
        Ok(PySimInfo { inner: sim_info })
    }

    /// Run the post-processing steps after the simulation finishes.
    ///
    /// Downloads the peak fields from the GPU and derives the simulated
    /// avalanche mask used by :meth:`evaluate`.
    pub fn post_process(&mut self) -> PyResult<()> {
        self.inner.post_process().block_on().map_runtime_err()
    }

    /// Evaluate the finished simulation and return all metrics.
    ///
    /// Runs both the GPU metrics (overlap components, diagonal-normalized
    /// chamfer distance against the region of interest and the 3D beeline
    /// distance between the highest and lowest avalanche point) and the
    /// reference-based runout diagnostics.
    ///
    /// Returns:
    ///     PyEvaluation: All evaluation metrics; also indexable by name,
    ///     e.g. ``result["iou"]``.
    pub fn evaluate(&mut self) -> PyResult<PyEvaluation> {
        let mut metrics = self.inner.evaluate().block_on().map_runtime_err()?;
        let gpu_metrics = self.inner.evaluate_gpu().block_on().map_runtime_err()?;
        metrics.intersection = gpu_metrics.intersection;
        metrics.undershoot = gpu_metrics.undershoot;
        metrics.overshoot = gpu_metrics.overshoot;
        metrics.iou = gpu_metrics.iou;
        metrics.beeline_3d = gpu_metrics.beeline_3d;
        metrics.sim_to_roi = gpu_metrics.sim_to_roi;
        metrics.roi_to_sim = gpu_metrics.roi_to_sim;
        metrics.chamfer = gpu_metrics.chamfer;

        Ok(PyEvaluation { metrics })
    }

    /// Prepare the simulation resources before running it.
    ///
    /// Called automatically by :meth:`run`; only needed when stepping
    /// manually via :meth:`run_n_steps`.
    pub fn prepare(&mut self) -> PyResult<()> {
        self.inner.prepare().block_on().map_runtime_err()
    }

    /// Save the current results to disk.
    ///
    /// Args:
    ///     path: Output zarr path. Defaults to ``avalanchers.zarr``.
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

    /// Current lifecycle state of the simulation (read-only).
    #[getter]
    pub fn state(&self) -> String {
        format!("{:?}", self.inner.get_state())
    }

    /// Cell edge length of the DEM grid in meters.
    #[getter]
    pub fn cell_size(&self) -> f32 {
        self.inner.dem.cell_size
    }

    /// Number of particles released per active grid cell.
    #[getter]
    pub fn released_particles_per_cell(&self) -> u32 {
        self.inner.settings.released_particles_per_cell
    }

    /// ``(height, width)`` float32 array of the DEM elevations in meters.
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

    /// ``(height, width)`` bool array of the reference (mapped) region of
    /// interest.
    #[getter]
    pub fn roi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let dims = [self.inner.dem.height, self.inner.dem.width];
        self.inner.roi.to_pyarray(py).reshape(dims).map_err(|_| {
            PyErr::new::<PyValueError, _>(format!(
                "Dimension mismatch during texture conversion. Expected: {}x{}, got: {}",
                dims[0],
                dims[1],
                self.inner.roi.len()
            ))
        })
    }

    /// ``(height, width)`` bool array marking the detected crown line cells of
    /// the outline. Only available after :meth:`prepare` when the settings
    /// contained ``release_area_fraction``; useful to visually verify the
    /// release area estimation.
    #[getter]
    pub fn crown_line<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        if self.inner.crown_line.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "crown line not available: set 'release_area_fraction' in the settings and call prepare()",
            ));
        }
        let dims = [self.inner.dem.height, self.inner.dem.width];
        self.inner
            .crown_line
            .to_pyarray(py)
            .reshape(dims)
            .map_err(|_| {
                PyErr::new::<PyValueError, _>(format!(
                    "Dimension mismatch during texture conversion. Expected: {}x{}, got: {}",
                    dims[0],
                    dims[1],
                    self.inner.crown_line.len()
                ))
            })
    }

    /// ``[xmin, xmax, ymin, ymax]`` real-world bounds of the DEM in meters.
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

    /// ``(height, width)`` float32 array of the peak velocities in m/s.
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

    /// ``(height, width)`` float32 array of the x terrain components.
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

    /// ``(height, width)`` float32 array of the y terrain components.
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

    /// ``(height, width)`` float32 array of the z terrain components.
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

    /// ``(height, width)`` float32 array of the x slope gravity components.
    #[getter]
    pub fn get_gravity_x<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (gravity_x, _) = self
            .inner
            .get_slope_gravity()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, gravity_x.to_vec())
    }

    /// ``(height, width)`` float32 array of the y slope gravity components.
    #[getter]
    pub fn get_gravity_y<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (_, gravity_y) = self
            .inner
            .get_slope_gravity()
            .block_on()
            .map_runtime_err()?;
        self.get_layer_f32(py, gravity_y.to_vec())
    }

    /// ``(height, width)`` float32 array of the configured release areas.
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

    /// ``(height, width)`` float32 array of the peak flow thickness in meters.
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

    /// :class:`PyTimestepData` trace of the first particle.
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

    /// DEM elevation threshold below which no avalanche is simulated.
    #[getter]
    pub fn get_elevation_threshold(&mut self) -> PyResult<f32> {
        Ok(self
            .inner
            .fetch_sim_info()
            .block_on()
            .map_runtime_err()?
            .elevation_threshold)
    }

    /// ``(n, 3)`` float32 array of the particle x/y/z positions (z is the
    /// elevation above the DEM).
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

    /// ``(n, 2)`` float32 array of the particle x/y positions.
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

    /// ``(n, 3)`` float32 array of the particle x/y/z velocities.
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

    /// ``(n, 2)`` float32 array of the particle x/y velocities.
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

    /// float32 array of the particle elevations above the DEM.
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

    /// float32 array of the particle masses in kg.
    #[getter]
    fn get_particles_mass<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mass = self
            .inner
            .fetch_particles_mass()
            .block_on()
            .map_runtime_err()?;
        Ok(mass.to_pyarray(py))
    }

    /// u32 array of the per-cell deposited mass as quantized by p2g
    /// (scaled by ``MASS_FACTOR`` in the shader utils), ``height * width``.
    #[getter]
    fn get_grid_mass<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let data = self.inner.fetch_grid_mass().block_on().map_runtime_err()?;
        Ok(data.to_pyarray(py))
    }

    /// i32 array of the per-cell deposited momentum as quantized by p2g
    /// (``u, v`` interleaved, scaled by ``MOMENTUM_FACTOR`` in the shader
    /// utils), ``height * width * 2``.
    #[getter]
    fn get_grid_momentum<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let data = self
            .inner
            .fetch_grid_momentum()
            .block_on()
            .map_runtime_err()?;
        Ok(data.to_pyarray(py))
    }

    /// u32 array of the particle stop timesteps (0 = still moving).
    #[getter]
    fn get_stopped<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let stopped = self
            .inner
            .fetch_particles_state()
            .block_on()
            .map_runtime_err()?;
        let timesteps: Vec<u32> = stopped.iter().map(|state| state.timestep).collect();
        Ok(timesteps.to_pyarray(py))
    }

    /// float32 array of the biggest-blob center of mass x coordinate per
    /// timestep (world coordinates).
    #[getter]
    fn get_center_of_mass_x<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (x, _, _) = self
            .inner
            .get_center_of_mass()
            .block_on()
            .map_runtime_err()?;
        Ok(x.to_pyarray(py))
    }

    /// float32 array of the biggest-blob center of mass y coordinate per
    /// timestep (world coordinates).
    #[getter]
    fn get_center_of_mass_y<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (_, y, _) = self
            .inner
            .get_center_of_mass()
            .block_on()
            .map_runtime_err()?;
        Ok(y.to_pyarray(py))
    }

    /// float32 array of the biggest-blob center of mass z coordinate (DEM
    /// elevation) per timestep.
    #[getter]
    fn get_center_of_mass_z<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (_, _, z) = self
            .inner
            .get_center_of_mass()
            .block_on()
            .map_runtime_err()?;
        Ok(z.to_pyarray(py))
    }

    /// Convert four channel arrays into a ``(height, width)`` RGBA texture
    /// tuple for rendering.
    ///
    /// Args:
    ///     r: Red channel, ``height * width`` floats.
    ///     g: Green channel, ``height * width`` floats.
    ///     b: Blue channel, ``height * width`` floats.
    ///     a: Alpha channel, ``height * width`` floats.
    ///
    /// Returns:
    ///     tuple: Four ``(height, width)`` float32 numpy arrays ``(r, g, b, a)``.
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
    m.add_class::<PyEvaluation>()?;
    m.add_class::<PyChamferDistance>()?;
    m.add_class::<PySimInfo>()?;
    m.add_class::<PyTimestepData>()?;

    m.add_function(wrap_pyfunction!(list_available_gpus, m)?)?;
    Ok(())
}
