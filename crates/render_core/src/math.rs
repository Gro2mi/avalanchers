//! Minimal column-major 4x4 math, matching the memory layout WGSL expects for `mat4x4<f32>`.
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len <= f32::EPSILON {
            Self::ZERO
        } else {
            self * (1.0 / len)
        }
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

/// Column-major 4x4 matrix: `m[col][row]`.
pub type Mat4 = [[f32; 4]; 4];

pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Right-handed perspective projection mapping depth to the wgpu range `0..1`.
pub fn perspective_rh(fov_y_radians: f32, aspect: f32, znear: f32, zfar: f32) -> Mat4 {
    let f = 1.0 / (fov_y_radians * 0.5).tan();
    let aspect = if aspect.abs() < f32::EPSILON {
        1.0
    } else {
        aspect
    };
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, zfar / (znear - zfar), -1.0],
        [0.0, 0.0, znear * zfar / (znear - zfar), 0.0],
    ]
}

/// Right-handed look-at view matrix.
pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    [
        [s.x, u.x, -f.x, 0.0],
        [s.y, u.y, -f.y, 0.0],
        [s.z, u.z, -f.z, 0.0],
        [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
    ]
}

pub fn mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for (col, out_col) in out.iter_mut().enumerate() {
        for (row, value) in out_col.iter_mut().enumerate() {
            *value = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

/// Transforms a point into clip space, keeping `w`.
pub fn transform_point4(m: Mat4, p: Vec3) -> [f32; 4] {
    [
        m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
        m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
        m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2],
        m[0][3] * p.x + m[1][3] * p.y + m[2][3] * p.z + m[3][3],
    ]
}

/// Transforms a point, dividing by w.
pub fn transform_point(m: Mat4, p: Vec3) -> Vec3 {
    let [x, y, z, w] = transform_point4(m, p);
    if w.abs() < f32::EPSILON {
        Vec3::new(x, y, z)
    } else {
        Vec3::new(x / w, y / w, z / w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_at_places_eye_at_origin_of_view_space() {
        let eye = Vec3::new(10.0, 5.0, 10.0);
        let view = look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let eye_in_view = transform_point(view, eye);
        assert!(eye_in_view.length() < 1e-4, "{eye_in_view:?}");
    }

    #[test]
    fn perspective_maps_near_and_far_to_zero_and_one() {
        let proj = perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 1.0, 100.0);
        let near = transform_point(proj, Vec3::new(0.0, 0.0, -1.0));
        let far = transform_point(proj, Vec3::new(0.0, 0.0, -100.0));
        assert!((near.z - 0.0).abs() < 1e-4, "{near:?}");
        assert!((far.z - 1.0).abs() < 1e-4, "{far:?}");
    }

    #[test]
    fn view_projection_keeps_target_centered() {
        let view = look_at_rh(Vec3::new(0.0, 20.0, 20.0), Vec3::ZERO, Vec3::Y);
        let proj = perspective_rh(std::f32::consts::FRAC_PI_4, 1.6, 0.1, 500.0);
        let ndc = transform_point(mul(proj, view), Vec3::ZERO);
        assert!(ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4, "{ndc:?}");
    }
}
