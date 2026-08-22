/// Structure to hold the evaluated normalized components of the metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassMovementEvaluation {
    /// Overlap normalized component (alpha_T)
    pub alpha: f64,
    /// Underestimation normalized component (beta_T)
    pub beta: f64,
    /// Overestimation normalized component (gamma_T)
    pub gamma: f64,
    /// The final composite index (Omega_T) which ranges between -1 and 1
    pub jaccard: f64,
}

/// Errors that could happen during the evaluation phase.
#[derive(Debug, PartialEq, Eq)]
pub enum MassMovementEvaluationError {
    DimensionMismatch,
    EmptyGrid,
    StartPointOutOfBounds,
    InvalidLambda,
}

/// Computes the proposed evaluation index Omega_T comparing an observed
/// (reference) binary grid with a simulated binary grid.
///
/// `true` indicates a cell is affected by the mass movement (above threshold),
/// and `false` indicates it is unaffected.
pub fn evaluate_mass_movement_area(
    reference_grid: &[Vec<bool>],
    simulated_grid: &[Vec<bool>],
) -> Result<MassMovementEvaluation, MassMovementEvaluationError> {
    let rows = reference_grid.len();
    if rows == 0 {
        return Err(MassMovementEvaluationError::EmptyGrid);
    }

    let cols = reference_grid[0].len();
    if cols == 0 {
        return Err(MassMovementEvaluationError::EmptyGrid);
    }

    // Ensure the simulated grid has the exact same dimensions
    if simulated_grid.len() != rows {
        return Err(MassMovementEvaluationError::DimensionMismatch);
    }

    let mut count_intersection = 0;
    let mut count_undershoot = 0;
    let mut count_overshoot = 0;

    for i in 0..rows {
        if reference_grid[i].len() != cols || simulated_grid[i].len() != cols {
            return Err(MassMovementEvaluationError::DimensionMismatch);
        }

        for j in 0..cols {
            let obs = reference_grid[i][j];
            let sim = simulated_grid[i][j];

            match (obs, sim) {
                (true, true) => count_intersection += 1, // Intersection (X)
                (true, false) => count_undershoot += 1,  // Underestimated (U)
                (false, true) => count_overshoot += 1,   // Overestimated (O)
                (false, false) => {}                     // True Negatives (Ignored in T)
            }
        }
    }

    // Total Area T is the union of affected areas: X + U + O
    let total_union = count_intersection + count_undershoot + count_overshoot;

    // Handle edge case where neither simulation nor observation has any affected pixels
    if total_union == 0 {
        return Ok(MassMovementEvaluation {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            jaccard: 1.0, // Perfect fit if nothing was supposed to happen and nothing did
        });
    }

    let t_f64 = total_union as f64;

    // Calculate components normalized by total area T
    let alpha_t = (count_intersection as f64) / t_f64;
    let beta_t = (count_undershoot as f64) / t_f64;
    let gamma_t = (count_overshoot as f64) / t_f64;

    // Omega_T = alpha_T - beta_T - gamma_T
    let omega_t = alpha_t - beta_t - gamma_t;

    Ok(MassMovementEvaluation {
        alpha: alpha_t,
        beta: beta_t,
        gamma: gamma_t,
        jaccard: (omega_t + 1.0) / 2.0,
    })
}

/// Computes the Hazard-Weighted Runout Index (omega).
///
/// * `reference_grid` - 2D Matrix of the actual observed events.
/// * `simulated_grid` - 2D Matrix of the simulation outputs.
/// * `start_point` - `(row, col)` index mapping the initiation apex or source point.
/// * `lambda` - Overshoot penalty modifier between 0.0 (overshoot not penalized) and 1.0 (symmetric penalty).
pub fn evaluate_distance_weighted_mass_movement_runout(
    reference_grid: &[Vec<bool>],
    simulated_grid: &[Vec<bool>],
    start_point: (usize, usize),
    lambda: f64,
) -> Result<MassMovementEvaluation, MassMovementEvaluationError> {
    // Basic structural validation
    let rows = reference_grid.len();
    if rows == 0 {
        return Err(MassMovementEvaluationError::EmptyGrid);
    }
    let cols = reference_grid[0].len();
    if cols == 0 {
        return Err(MassMovementEvaluationError::EmptyGrid);
    }
    if simulated_grid.len() != rows {
        return Err(MassMovementEvaluationError::DimensionMismatch);
    }

    // Bounds check for start/apex coordinates
    if start_point.0 >= rows || start_point.1 >= cols {
        return Err(MassMovementEvaluationError::StartPointOutOfBounds);
    }

    // Constraints check for lambda parameter
    if !(0.0..=1.0).contains(&lambda) {
        return Err(MassMovementEvaluationError::InvalidLambda);
    }

    let mut w_x = 0.0; // Weighted Intersection
    let mut w_u = 0.0; // Weighted Underestimation (Critical Failure)
    let mut w_o = 0.0; // Weighted Overestimation (Overshoot)

    let (r_s, c_s) = (start_point.0 as f64, start_point.1 as f64);

    for i in 0..rows {
        if reference_grid[i].len() != cols || simulated_grid[i].len() != cols {
            return Err(MassMovementEvaluationError::DimensionMismatch);
        }

        for j in 0..cols {
            let obs = reference_grid[i][j];
            let sim = simulated_grid[i][j];

            if obs || sim {
                // Calculate Euclidean Distance from the initiation point
                let di = (i as f64) - r_s;
                let dj = (j as f64) - c_s;
                let distance = (di * di + dj * dj).sqrt();

                // Add a baseline (+1.0) so the apex cell doesn't get multiplying factor 0
                let weight = distance + 1.0;

                match (obs, sim) {
                    (true, true) => w_x += weight,
                    (true, false) => w_u += weight,
                    (false, true) => w_o += weight,
                    (false, false) => unreachable!(),
                }
            }
        }
    }

    let w_t = w_x + w_u + w_o;

    // Edge case where no cells are active anywhere
    if w_t == 0.0 {
        return Ok(MassMovementEvaluation {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            jaccard: 1.0,
        });
    }

    let alpha = w_x / w_t;
    let beta = w_u / w_t;
    let gamma = w_o / w_t;

    // Asymmetric penalty: overshoots (gamma) are less penalized via lambda scaling
    let omega = alpha - beta - (lambda * gamma);

    Ok(MassMovementEvaluation {
        alpha,
        beta,
        gamma,
        jaccard: (omega + 1.0) / 2.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_match() {
        // A perfect simulation totally fits the observed deposition area (A = \hat{A})
        // Expected: alpha_t = 1.0, beta_t = 0.0, gamma_t = 0.0, omega_t = 1.0
        let reference = vec![vec![true, false, true], vec![false, true, false]];
        let simulated = vec![vec![true, false, true], vec![false, true, false]];

        let result = evaluate_mass_movement_area(&reference, &simulated).unwrap();
        assert_eq!(result.alpha, 1.0);
        assert_eq!(result.beta, 0.0);
        assert_eq!(result.gamma, 0.0);
        assert_eq!(result.jaccard, 1.0);
    }

    #[test]
    fn test_complete_mismatch() {
        // Simulation completely misses the reference area (X = 0)
        // Expected: alpha_t = 0.0, jaccard = 0.0
        let reference = vec![vec![true, true, false], vec![false, false, false]];
        let simulated = vec![vec![false, false, false], vec![false, true, true]];

        let result = evaluate_mass_movement_area(&reference, &simulated).unwrap();
        assert_eq!(result.alpha, 0.0);
        assert_eq!(result.jaccard, 0.0);

        // Total area T is the union of 2 reference cells + 2 simulated cells = 4 cells
        assert_eq!(result.beta, 0.5); // 2/4
        assert_eq!(result.gamma, 0.5); // 2/4
    }

    #[test]
    fn test_partial_overlap() {
        // Grid setup:
        // Ref: [1, 1, 0]  |  Sim: [0, 1, 1]
        //      [0, 1, 0]  |       [0, 1, 0]
        // Total Union (T) cells affected = 4 cells (indices (0,0), (0,1), (0,2), (1,1))
        // Intersection (X) = 2 cells (indices (0,1), (1,1)) -> alpha = 2/4 = 0.5
        // Underestimated (U) = 1 cell (index (0,0))       -> beta  = 1/4 = 0.25
        // Overestimated (O) = 1 cell (index (0,2))        -> gamma = 1/4 = 0.25
        // Omega_T = 0.5 - 0.25 - 0.25 = 0.0
        let reference = vec![vec![true, true, false], vec![false, true, false]];
        let simulated = vec![vec![false, true, true], vec![false, true, false]];

        let result = evaluate_mass_movement_area(&reference, &simulated).unwrap();
        assert_eq!(result.alpha, 0.5);
        assert_eq!(result.beta, 0.25);
        assert_eq!(result.gamma, 0.25);
        assert_eq!(result.jaccard, 0.5);
    }

    #[test]
    fn test_empty_union_edge_case() {
        // Neither reference nor simulation contains any mass movement activity.
        // Expected: System handles division cleanly, returns perfect baseline score.
        let reference = vec![vec![false, false], vec![false, false]];
        let simulated = vec![vec![false, false], vec![false, false]];

        let result = evaluate_mass_movement_area(&reference, &simulated).unwrap();
        assert_eq!(result.alpha, 1.0);
        assert_eq!(result.jaccard, 1.0);
    }

    #[test]
    fn test_grid_dimension_mismatches() {
        let reference = vec![vec![true, true]];
        let simulated_wrong_rows = vec![vec![true, true], vec![true, true]];
        let simulated_wrong_cols = vec![vec![true, true, true]];

        assert_eq!(
            evaluate_mass_movement_area(&reference, &simulated_wrong_rows),
            Err(MassMovementEvaluationError::DimensionMismatch)
        );
        assert_eq!(
            evaluate_mass_movement_area(&reference, &simulated_wrong_cols),
            Err(MassMovementEvaluationError::DimensionMismatch)
        );
    }

    #[test]
    fn test_empty_grids() {
        let empty_ref: Vec<Vec<bool>> = vec![];
        let valid_sim = vec![vec![true]];

        assert_eq!(
            evaluate_mass_movement_area(&empty_ref, &valid_sim),
            Err(MassMovementEvaluationError::EmptyGrid)
        );
    }

    fn assert_near(actual: f64, expected: f64) {
        let tolerance = 1e-7;
        assert!(
            (actual - expected).abs() < tolerance,
            "Expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_perfect_match_keeps_unity() {
        let grid = vec![vec![false, false], vec![false, true]];
        let result =
            evaluate_distance_weighted_mass_movement_runout(&grid, &grid, (0, 0), 0.5).unwrap();
        assert_near(result.jaccard, 1.0);
    }

    #[test]
    fn test_asymmetry_overshoot_vs_undershoot() {
        // Source point is at (0, 0)
        // Let's configure a scenario with lambda = 0.5 (overshoots are penalized half as much)

        // Scenario A: Undershoot (Simulation stopped short, missing a cell at index (0,2))
        let ref_a = vec![vec![true, true, true]];
        let sim_a = vec![vec![true, true, false]];

        // Scenario B: Overshoot (Simulation went too far, predicting an extra cell at index (0,2))
        let ref_b = vec![vec![true, true, false]];
        let sim_b = vec![vec![true, true, true]];

        let start = (0, 0);
        let lambda = 0.5;

        let eval_undershoot =
            evaluate_distance_weighted_mass_movement_runout(&ref_a, &sim_a, start, lambda).unwrap();
        let eval_overshoot =
            evaluate_distance_weighted_mass_movement_runout(&ref_b, &sim_b, start, lambda).unwrap();

        // The spatial configurations are mirror mismatches, but because overshoot is safer,
        // the overshoot scenario MUST yield a higher hazard evaluation score than the undershoot.
        assert!(
            eval_overshoot.jaccard > eval_undershoot.jaccard,
            "Overshoot score ({}) should be preferred over Undershoot score ({})",
            eval_overshoot.jaccard,
            eval_undershoot.jaccard
        );
    }

    #[test]
    fn test_distance_sensitivity_on_runout() {
        // Source point is at (0,0). We test two identical 1-cell mismatches (undershoots).
        // Mismatch close to apex vs Mismatch far away from apex.

        // Close error: missed cell at (0, 1)
        let ref_close = vec![vec![true, true, false, false, false]];
        let sim_close = vec![vec![true, false, false, false, false]];

        // Distant error: missed cell at (0, 4) -- representing a failed runout calculation
        let ref_far = vec![vec![true, false, false, false, true]];
        let sim_far = vec![vec![true, false, false, false, false]];

        let start = (0, 0);
        let lambda = 1.0; // Keep it symmetric to isolate only distance effects

        let eval_close =
            evaluate_distance_weighted_mass_movement_runout(&ref_close, &sim_close, start, lambda)
                .unwrap();
        let eval_far =
            evaluate_distance_weighted_mass_movement_runout(&ref_far, &sim_far, start, lambda)
                .unwrap();

        // The runout-failing simulation (eval_far) must get penalized more heavily (lower score)
        // because the error happened further down the path from the apex point.
        assert!(
            eval_close.jaccard > eval_far.jaccard,
            "Distal runout errors must yield a worse metric score than proximal errors. Close: {}, Far: {}",
            eval_close.jaccard,
            eval_far.jaccard
        );
    }

    #[test]
    fn test_parameter_errors() {
        let grid = vec![vec![true]];

        // Start point out of bounds
        assert_eq!(
            evaluate_distance_weighted_mass_movement_runout(&grid, &grid, (5, 5), 0.5),
            Err(MassMovementEvaluationError::StartPointOutOfBounds)
        );

        // Invalid lambda boundaries
        assert_eq!(
            evaluate_distance_weighted_mass_movement_runout(&grid, &grid, (0, 0), -0.1),
            Err(MassMovementEvaluationError::InvalidLambda)
        );
        assert_eq!(
            evaluate_distance_weighted_mass_movement_runout(&grid, &grid, (0, 0), 1.5),
            Err(MassMovementEvaluationError::InvalidLambda)
        );
    }
    #[test]
    fn test_metrics_always_less_than_or_equal_to_one() {
        // Helper function to decode a 9-bit integer into a 3x3 boolean grid matrix
        fn integer_to_3x3_grid(bits: u16) -> Vec<Vec<bool>> {
            let mut grid = vec![vec![false; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let bit_index = i * 3 + j;
                    grid[i][j] = (bits & (1 << bit_index)) != 0;
                }
            }
            grid
        }

        // Define test parameters to vary across permutations
        let lambda_variants = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let start_point_variants = vec![(0, 0), (1, 1), (2, 2)];

        // Iterate through all 512 possibilities for the reference grid
        for ref_bits in 0..512 {
            let reference = integer_to_3x3_grid(ref_bits);

            // Iterate through all 512 possibilities for the simulated grid
            for sim_bits in 0..512 {
                let simulated = integer_to_3x3_grid(sim_bits);

                // Verify across different hazard configurations
                for &lambda in &lambda_variants {
                    for &start_point in &start_point_variants {
                        let result = evaluate_distance_weighted_mass_movement_runout(
                            &reference,
                            &simulated,
                            start_point,
                            lambda,
                        )
                        .unwrap();

                        // Absolute upper bound invariant assertions (<= 1.0)
                        assert!(
                            result.alpha <= 1.0,
                            "alpha ({}) exceeded 1.0 at ref: {}, sim: {}",
                            result.alpha,
                            ref_bits,
                            sim_bits
                        );
                        assert!(
                            result.beta <= 1.0,
                            "beta ({}) exceeded 1.0 at ref: {}, sim: {}",
                            result.beta,
                            ref_bits,
                            sim_bits
                        );
                        assert!(
                            result.gamma <= 1.0,
                            "gamma ({}) exceeded 1.0 at ref: {}, sim: {}",
                            result.gamma,
                            ref_bits,
                            sim_bits
                        );
                        assert!(
                            result.jaccard <= 1.0,
                            "omega ({}) exceeded 1.0 at ref: {}, sim: {}",
                            result.jaccard,
                            ref_bits,
                            sim_bits
                        );

                        // Optional safety check: ensure the normalized subsets are never negative
                        assert!(result.alpha >= 0.0);
                        assert!(result.beta >= 0.0);
                        assert!(result.gamma >= 0.0);

                        // Omega can go down to -1.0, but never past +1.0
                        assert!(
                            result.jaccard >= -1.00001,
                            "omega ({}) below -1.0 at ref: {}, sim: {}",
                            result.jaccard,
                            ref_bits,
                            sim_bits
                        );
                    }
                }
            }
        }
    }
}
