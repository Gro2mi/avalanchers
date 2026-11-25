// Simulated-avalanche mask shared by the evaluation shaders. A cell belongs to
// the simulated avalanche when its peak flow thickness reaches
// evaluation_params.min_peak_flow_thickness and its peak velocity reaches
// evaluation_params.min_peak_velocity. The peak fields themselves are never
// modified, so a stored run can be re-evaluated with other thresholds.
//
// The importing shader declares the evaluation_params uniform
// (EvaluationParams) and the grid_peak_flow_thickness / grid_peak_velocity
// storage buffers at its own binding slots.

struct EvaluationParams {
    min_peak_flow_thickness: f32,
    min_peak_velocity: f32,
    _padding: vec2<f32>,
}

fn is_simulated_cell(idx: u32) -> bool {
    return grid_peak_flow_thickness[idx] >= evaluation_params.min_peak_flow_thickness
        && grid_peak_velocity[idx] >= evaluation_params.min_peak_velocity;
}
