use std::collections::VecDeque;

/// Represents a 2D grid flattened into a 1D vector.
#[derive(Debug, Clone)]
pub struct FlattenedGrid {
    pub data: Vec<bool>,
    pub rows: usize,
    pub cols: usize,
}

/// Represents a bounding box or metadata for a detected blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blob {
    pub id: usize,
    pub pixels: Vec<(usize, usize)>, // 2D coordinates (row, col)
    pub min_x: usize,                // Column min
    pub max_x: usize,                // Column max
    pub min_y: usize,                // Row min
    pub max_y: usize,                // Row max
}

impl Blob {
    pub fn to_mask_1d(&self, rows: usize, cols: usize) -> Vec<bool> {
        let mut mask = vec![false; rows * cols];
        for &(r, c) in &self.pixels {
            if r < rows && c < cols {
                mask[r * cols + c] = true;
            }
        }
        mask
    }
}

/// Detects blobs in a flattened 2D boolean grid using 4-way or 8-way connectivity.
pub fn detect_blobs_1d(grid: &FlattenedGrid, diagonal_connectivity: bool) -> Vec<Blob> {
    if grid.rows == 0 || grid.cols == 0 || grid.data.is_empty() {
        return Vec::new();
    }

    let rows = grid.rows;
    let cols = grid.cols;
    let mut visited = vec![false; rows * cols];
    let mut blobs = Vec::new();
    let mut blob_id = 0;

    // Define neighbor offsets as (row_offset, col_offset)
    let directions: Vec<(isize, isize)> = if diagonal_connectivity {
        vec![
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1), // 4-way
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1), // Diagonals
        ]
    } else {
        vec![(-1, 0), (1, 0), (0, -1), (0, 1)] // 4-way only
    };

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;

            // If we find an unvisited foreground pixel, start a new blob
            if grid.data[idx] && !visited[idx] {
                blob_id += 1;
                let mut pixels = Vec::new();
                let mut queue = VecDeque::new();

                visited[idx] = true;
                queue.push_back((r, c));

                let mut min_r = r;
                let mut max_r = r;
                let mut min_c = c;
                let mut max_c = c;

                while let Some((curr_r, curr_c)) = queue.pop_front() {
                    pixels.push((curr_r, curr_c));

                    // Update bounding box metrics
                    min_r = min_r.min(curr_r);
                    max_r = max_r.max(curr_r);
                    min_c = min_c.min(curr_c);
                    max_c = max_c.max(curr_c);

                    // Check neighbors
                    for &(dr, dc) in &directions {
                        let next_r = curr_r as isize + dr;
                        let next_c = curr_c as isize + dc;

                        if next_r >= 0
                            && next_r < rows as isize
                            && next_c >= 0
                            && next_c < cols as isize
                        {
                            let nr = next_r as usize;
                            let nc = next_c as usize;
                            let n_idx = nr * cols + nc;

                            if grid.data[n_idx] && !visited[n_idx] {
                                visited[n_idx] = true;
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }

                blobs.push(Blob {
                    id: blob_id,
                    pixels,
                    min_x: min_c,
                    max_x: max_c,
                    min_y: min_r,
                    max_y: max_r,
                });
            }
        }
    }

    blobs
}

/// Returns a reference to the largest blob by pixel count, if any exist.
pub fn get_biggest_blob(blobs: &[Blob]) -> Option<&Blob> {
    blobs.iter().max_by_key(|blob| blob.pixels.len())
}

/// Converts a vector of floats into a boolean vector using a threshold.
/// Values strictly greater than (or equal to) the threshold become `true`.
pub fn threshold_to_bool(data: &[f32], threshold: f32) -> Vec<bool> {
    data.iter().map(|&val| val >= threshold).collect()
}

/// Masks the original float vector with a given blob.
/// Returns a new `Vec<f32>` where pixels outside the blob are set to `0.0`.
pub fn mask_with_blob(data: &[f32], cols: usize, blob: &Blob) -> Vec<f32> {
    let mut masked = vec![0.0; data.len()];

    for &(r, c) in &blob.pixels {
        let idx = r * cols + c;
        if idx < data.len() {
            masked[idx] = data[idx];
        }
    }

    masked
}

pub fn mask_threshold_and_biggest_blob(
    data: &[f32],
    cols: usize,
    threshold: f32,
) -> (Vec<f32>, Vec<bool>) {
    let rows = data.len() / cols;
    let grid = FlattenedGrid {
        data: threshold_to_bool(data, threshold),
        rows,
        cols,
    };

    let blobs = detect_blobs_1d(&grid, false); // Using 4-way connectivity
    if let Some(biggest_blob) = get_biggest_blob(&blobs) {
        (
            mask_with_blob(data, cols, biggest_blob),
            biggest_blob.to_mask_1d(rows, cols),
        )
    } else {
        (vec![0.0; data.len()], vec![false; data.len()]) // No blobs found, return all zeros
    }
}

pub fn mask_in_place(values: &mut [f32], mask: &[bool]) {
    assert_eq!(values.len(), mask.len());

    for (value, &valid) in values.iter_mut().zip(mask) {
        if !valid {
            *value = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_to_mask_1d() {
        let blob = Blob {
            id: 1,
            pixels: vec![(0, 0), (0, 1), (1, 0)],
            min_x: 0,
            max_x: 1,
            min_y: 0,
            max_y: 1,
        };
        let mask = blob.to_mask_1d(3, 3);
        let expected = vec![true, true, false, true, false, false, false, false, false];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_in_place() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        mask_in_place(&mut values, &mask);
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_threshold_to_bool() {
        let data = vec![0.1, 0.5, 0.9, 0.3];
        let threshold = 0.5;
        let expected = vec![false, true, true, false];
        assert_eq!(threshold_to_bool(&data, threshold), expected);
    }

    #[test]
    fn test_mask_with_blob() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let cols = 4;
        let blob = Blob {
            id: 1,
            pixels: vec![(0, 1), (1, 2), (2, 3)],
            min_x: 1,
            max_x: 3,
            min_y: 0,
            max_y: 2,
        };
        let expected = vec![0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 1.2];
        assert_eq!(mask_with_blob(&data, cols, &blob), expected);
    }

    #[test]
    fn test_biggest_blob_selection() {
        // create size 2 and size 4
        let grid = FlattenedGrid {
            rows: 4,
            cols: 4,
            data: vec![
                true, true, false, false, // Blob 1 (size 2)
                false, false, false, false, false, false, true, true, // Blob 2 (size 4)
                false, false, true, true,
            ],
        };

        let blobs = detect_blobs_1d(&grid, false);
        assert_eq!(blobs.len(), 2);

        let biggest = get_biggest_blob(&blobs).expect("Should find a biggest blob");
        assert_eq!(biggest.pixels.len(), 4);
    }

    /// Empty grid / zero dimensions
    #[test]
    fn test_empty_grid_and_zero_dimensions() {
        // Zero dimensions with empty vector
        let empty_grid = FlattenedGrid {
            rows: 0,
            cols: 0,
            data: vec![],
        };
        assert!(detect_blobs_1d(&empty_grid, false).is_empty());
        assert!(detect_blobs_1d(&empty_grid, true).is_empty());

        // Zero rows with non-empty column dimension specification
        let zero_rows = FlattenedGrid {
            rows: 0,
            cols: 5,
            data: vec![],
        };
        assert!(detect_blobs_1d(&zero_rows, false).is_empty());

        // Zero columns with non-empty row dimension specification
        let zero_cols = FlattenedGrid {
            rows: 5,
            cols: 0,
            data: vec![],
        };
        assert!(detect_blobs_1d(&zero_cols, false).is_empty());
    }

    /// Test Case 2: Grid with no foreground pixels
    #[test]
    fn test_grid_with_no_foreground_pixels() {
        let grid = FlattenedGrid {
            rows: 3,
            cols: 3,
            data: vec![
                false, false, false, false, false, false, false, false, false,
            ],
        };

        let blobs_4way = detect_blobs_1d(&grid, false);
        let blobs_8way = detect_blobs_1d(&grid, true);

        assert_eq!(blobs_4way.len(), 0);
        assert_eq!(blobs_8way.len(), 0);
    }

    /// Test Case 3: Grid with all foreground pixels
    #[test]
    fn test_grid_with_all_foreground_pixels() {
        let grid = FlattenedGrid {
            rows: 3,
            cols: 4,
            data: vec![true; 12],
        };

        let blobs = detect_blobs_1d(&grid, false);

        assert_eq!(blobs.len(), 1);
        let blob = &blobs[0];

        assert_eq!(blob.id, 1);
        assert_eq!(blob.pixels.len(), 12);
        assert_eq!(blob.min_y, 0); // min row
        assert_eq!(blob.max_y, 2); // max row
        assert_eq!(blob.min_x, 0); // min col
        assert_eq!(blob.max_x, 3); // max col
    }

    /// Test Case 4: Multiple distinct blobs (4-way connectivity)
    #[test]
    fn test_multiple_distinct_blobs_4way() {
        // Grid with 3 isolated blobs
        // [ T, T, F, F ]
        // [ F, F, F, T ]
        // [ F, F, F, T ]
        // [ T, F, F, F ]
        let grid = FlattenedGrid {
            rows: 4,
            cols: 4,
            data: vec![
                true, true, false, false, false, false, false, true, false, false, false, true,
                true, false, false, false,
            ],
        };

        let blobs = detect_blobs_1d(&grid, false);

        assert_eq!(blobs.len(), 3);

        // Blob 1: (0,0), (0,1) -> length 2
        assert_eq!(blobs[0].pixels.len(), 2);
        assert_eq!(blobs[0].min_y, 0);
        assert_eq!(blobs[0].max_y, 0);

        // Blob 2: (1,3), (2,3) -> length 2
        assert_eq!(blobs[1].pixels.len(), 2);
        assert_eq!(blobs[1].min_y, 1);
        assert_eq!(blobs[1].max_y, 2);

        // Blob 3: (3,0) -> length 1
        assert_eq!(blobs[2].pixels.len(), 1);
        assert_eq!(blobs[2].min_y, 3);
        assert_eq!(blobs[2].max_y, 3);
    }

    /// Test Case 5: Comparison between 4-way and 8-way connectivity with diagonal pixels
    #[test]
    fn test_connectivity_comparison_diagonal() {
        // Checkerboard / diagonal pattern
        // [ T, F, T ]
        // [ F, T, F ]
        // [ T, F, T ]
        let grid = FlattenedGrid {
            rows: 3,
            cols: 3,
            data: vec![true, false, true, false, true, false, true, false, true],
        };

        // In 4-way connectivity, none of the 5 pixels connect orthogonally -> 5 separate blobs
        let blobs_4way = detect_blobs_1d(&grid, false);
        assert_eq!(blobs_4way.len(), 5);

        // In 8-way connectivity, all 5 pixels connect diagonally through the center -> 1 single blob
        let blobs_8way = detect_blobs_1d(&grid, true);
        assert_eq!(blobs_8way.len(), 1);
        assert_eq!(blobs_8way[0].pixels.len(), 5);
        assert_eq!(blobs_8way[0].min_x, 0);
        assert_eq!(blobs_8way[0].max_x, 2);
        assert_eq!(blobs_8way[0].min_y, 0);
        assert_eq!(blobs_8way[0].max_y, 2);
    }

    /// Test Case 6: A single complex or irregular blob
    #[test]
    fn test_complex_irregular_blob() {
        // Spiral / U-shaped single continuous path:
        // [ T, T, T, T ]
        // [ F, F, F, T ]
        // [ T, T, F, T ]
        // [ T, T, T, T ]
        let grid = FlattenedGrid {
            rows: 4,
            cols: 4,
            data: vec![
                true, true, true, true, false, false, false, true, true, true, false, true, true,
                true, true, true,
            ],
        };

        let blobs = detect_blobs_1d(&grid, false);

        // Should form exactly 1 continuous blob
        assert_eq!(blobs.len(), 1);

        let blob = &blobs[0];
        assert_eq!(blob.pixels.len(), 12);
        assert_eq!(blob.min_x, 0);
        assert_eq!(blob.max_x, 3);
        assert_eq!(blob.min_y, 0);
        assert_eq!(blob.max_y, 3);
    }
}
