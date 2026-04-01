
// use numpy::{PyArrayDyn, PyReadonlyArrayDyn, IntoPyArray};

use pyo3::prelude::*;
use tracing::{debug, info, warn, error};

#[pymodule]
fn avalanchers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1. Initialize the bridge between Rust and Python logging
    // This allows Python's `logging.basicConfig(level=logging.INFO)` 
    // to catch your Rust `info!()` macros.
    pyo3_log::init();

    // 2. (Optional) Call your internal library init if you have 
    // specific non-python logging logic (like file logging).
    compute_core::init_logging();

    // Register your functions
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    // This will now show up in Python's console!
    info!("Adding {} and {}", a, b); 
    Ok((a + b).to_string())
}

// pub fn read_lz4(path: &Path) -> PyResult<Vec<u8>> {
//     read_bin(&path.with_extension("lz4")).and_then(|buffer| {
//         decompress_size_prepended(&buffer)
//             .map_err(|e| PyValueError::new_err(format!("Failed to decompress data: {}", e)))
//     })
// }
