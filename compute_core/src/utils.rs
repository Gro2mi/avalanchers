use rand::Rng;

pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: Option<f32>,
}

pub fn create_random_rgba_data(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 4);
    let mut rng = rand::rng();
    for _ in 0..(width * height) {
        let r = rng.random_range(0..=255);
        let g = rng.random_range(0..=255);
        let b = rng.random_range(0..=255);
        data.extend_from_slice(&[r, g, b, 255]); // RGBA format
    }
    data
}

// #[wasm_bindgen]
// pub fn get_image_bytes(data: Vec<f32>, width: u32, height: u32) -> Vec<u8> {
//     let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
//         width as u32,
//         height as u32,
//         f32_to_rgba_bytes(&data)
//     ).unwrap();
//     let mut buf = std::io::Cursor::new(Vec::new());
//     img.write_to(&mut buf, image::ImageOutputFormat::Png).unwrap();
//     buf.into_inner()
// }





pub fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    if num == 1 {
        return vec![start];
    }
    let step = (end - start) / (num - 1) as f32;
    (0..num).map(|i| start + i as f32 * step).collect()
}

pub fn to_2d(data: &[f32], width: usize, height: usize) -> Vec<Vec<f32>> {
    (0..height)
        .map(|row| {
            let start = row * width;
            let end = start + width;
            data[start..end].to_vec()
        })
        .collect()
}

pub fn highest_power_of_two(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        1 << (31 - n.leading_zeros())
    }
}

pub fn bilinear_interpolate(x: f32, y: f32, grid: &Vec<Vec<f32>>) -> Option<f32> {
    let x0 = x.floor() as isize;
    let x1 = x.ceil() as isize;
    let y0 = y.floor() as isize;
    let y1 = y.ceil() as isize;

    let height = grid.len() as isize;
    let width = if height > 0 {
        grid[0].len() as isize
    } else {
        0
    };

    if x0 < 0 || x1 >= width || y0 < 0 || y1 >= height {
        return None;
    }

    let q11 = grid[y0 as usize][x0 as usize];
    let q21 = grid[y0 as usize][x1 as usize];
    let q12 = grid[y1 as usize][x0 as usize];
    let q22 = grid[y1 as usize][x1 as usize];

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let r1 = q11 * (1.0 - fx) + q21 * fx;
    let r2 = q12 * (1.0 - fx) + q22 * fx;

    Some(r1 * (1.0 - fy) + r2 * fy)
}

pub fn subtract(vec: &mut Vec<f32>, value: f32) {
    for v in vec.iter_mut() {
        *v -= value;
    }
}

pub fn add(vec: &mut Vec<f32>, value: f32) {
    for v in vec.iter_mut() {
        *v += value;
    }
}

pub fn multiply(vec: &mut Vec<f32>, value: f32) {
    for v in vec.iter_mut() {
        *v *= value;
    }
}

pub fn divide(vec: &mut Vec<f32>, value: f32) {
    if value != 0.0 {
        for v in vec.iter_mut() {
            *v /= value;
        }
    } else {
        panic!("Division by zero in vector division");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_interpolate_center() {
        let grid = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        // At (0.5, 0.5), should be average of all four corners
        let result = bilinear_interpolate(0.5, 0.5, &grid);
        assert_eq!(result, Some(2.5));
    }

    #[test]
    fn test_bilinear_interpolate_exact_point() {
        let grid = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        // At (0, 0), should be 10.0
        assert_eq!(bilinear_interpolate(0.0, 0.0, &grid), Some(10.0));
        // At (1, 0), should be 20.0
        assert_eq!(bilinear_interpolate(1.0, 0.0, &grid), Some(20.0));
        // At (0, 1), should be 30.0
        assert_eq!(bilinear_interpolate(0.0, 1.0, &grid), Some(30.0));
        // At (1, 1), should be 40.0
        assert_eq!(bilinear_interpolate(1.0, 1.0, &grid), Some(40.0));
    }

    #[test]
    fn test_bilinear_interpolate_out_of_bounds() {
        let grid = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        // Negative coordinates
        assert_eq!(bilinear_interpolate(-1.0, 0.0, &grid), None);
        assert_eq!(bilinear_interpolate(0.0, -1.0, &grid), None);
        // Coordinates outside grid
        assert_eq!(bilinear_interpolate(2.0, 0.0, &grid), None);
        assert_eq!(bilinear_interpolate(0.0, 2.0, &grid), None);
    }

    #[test]
    fn test_bilinear_interpolate_non_square_grid() {
        let grid = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        // Interpolate at (1, 0.5)
        let result = bilinear_interpolate(1.0, 0.5, &grid);
        // Should interpolate between (1,0)=2, (2,0)=3, (1,1)=5, (2,1)=6
        // r1 = 2, r2 = 5, so halfway between 2 and 5 is 3.5
        assert_eq!(result, Some(3.5));
    }
    #[test]
    fn test_linspace_basic() {
        let result = linspace(0.0, 1.0, 5);
        assert_eq!(result, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_linspace_single_element() {
        let result = linspace(2.0, 5.0, 1);
        assert_eq!(result, vec![2.0]);
    }

    #[test]
    fn test_linspace_two_elements() {
        let result = linspace(3.0, 7.0, 2);
        assert_eq!(result, vec![3.0, 7.0]);
    }

    #[test]
    fn test_linspace_negative_range() {
        let result = linspace(1.0, -1.0, 3);
        assert_eq!(result, vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_to_2d_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = to_2d(&data, 2, 3);
        assert_eq!(
            result,
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],]
        );
    }

    #[test]
    fn test_to_2d_single_row() {
        let data = vec![1.0, 2.0, 3.0];
        let result = to_2d(&data, 3, 1);
        assert_eq!(result, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_to_2d_single_column() {
        let data = vec![1.0, 2.0, 3.0];
        let result = to_2d(&data, 1, 3);
        assert_eq!(result, vec![vec![1.0], vec![2.0], vec![3.0]]);
    }

    #[test]
    fn test_vector_arithmetic_add() {
        let mut v = vec![1.0, 2.0, 3.0];
        add(&mut v, 2.0);
        assert_eq!(v, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vector_arithmetic_subtract() {
        let mut v = vec![5.0, 7.0, 9.0];
        subtract(&mut v, 2.0);
        assert_eq!(v, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_vector_arithmetic_multiply() {
        let mut v = vec![2.0, 3.0, 4.0];
        multiply(&mut v, 3.0);
        assert_eq!(v, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "Division by zero in vector division")]
    fn test_vector_arithmetic_divide_by_zero() {
        let mut v = vec![1.0, 2.0, 3.0];
        divide(&mut v, 0.0);
    }

    #[test]
    fn test_vector_arithmetic_divide() {
        let mut v = vec![10.0, 20.0, 30.0];
        divide(&mut v, 10.0);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }
}
