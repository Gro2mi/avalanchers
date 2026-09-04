use crate::math::{Mat4, Vec3, look_at_rh, mul, perspective_rh, transform_point4};
use crate::terrain::TerrainData;

const MIN_PITCH: f32 = 0.02;
const MAX_PITCH: f32 = 1.5;
const MIN_DISTANCE: f32 = 0.01;
/// Rotation added to the computed downhill yaw so the default view meets the
/// slope at an angle instead of head-on: a camera looking straight up the fall
/// line flattens the relief and hides the release areas behind the ridge.
const VIEW_YAW_OFFSET: f32 = 0.3;

/// Orbit camera: looks at `target` from a point on a sphere defined by `yaw`, `pitch` and `distance`.
///
/// World space is y-up: `x` runs along DEM columns, `z` along DEM rows, `y` is elevation.
#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation around the vertical axis, in radians.
    pub yaw: f32,
    /// Elevation angle above the horizon, in radians.
    pub pitch: f32,
    pub fov_y: f32,
    pub znear: f32,
    pub zfar: f32,
    pub aspect: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 100.0,
            yaw: -0.6,
            pitch: 0.55,
            fov_y: std::f32::consts::FRAC_PI_4,
            znear: 0.1,
            zfar: 10_000.0,
            aspect: 1.0,
        }
    }
}

impl OrbitCamera {
    /// Default viewpoint for a DEM: centered on the real data, standing in the
    /// valley on the low side of the terrain so the mountain face and its release
    /// area face the camera — offset in yaw so the slope is seen at an angle —
    /// and pulled back until the whole terrain fits.
    pub fn framing(terrain: &TerrainData, aspect: f32) -> Self {
        let samples = terrain.fit_samples();
        let (min_p, max_p) = samples.iter().fold(
            (
                Vec3::new(f32::MAX, f32::MAX, f32::MAX),
                Vec3::new(f32::MIN, f32::MIN, f32::MIN),
            ),
            |(min_p, max_p), p| {
                (
                    Vec3::new(min_p.x.min(p.x), min_p.y.min(p.y), min_p.z.min(p.z)),
                    Vec3::new(max_p.x.max(p.x), max_p.y.max(p.y), max_p.z.max(p.z)),
                )
            },
        );

        // No-data cells are excluded, so the camera centres on the part of the grid that is drawn.
        let target = (min_p + max_p) * 0.5;
        let radius = ((max_p - min_p) * 0.5).length().max(MIN_DISTANCE);

        let fov_y = std::f32::consts::FRAC_PI_4;
        let aspect = if aspect > 0.0 { aspect } else { 1.0 };

        // Fit against whichever field of view is narrower, so tall windows do not clip the terrain.
        let half_fov_y = fov_y * 0.5;
        let half_fov = half_fov_y.min((aspect * half_fov_y.tan()).atan());

        let horizontal_extent = (max_p.x - min_p.x).max(max_p.z - min_p.z).max(f32::EPSILON);
        let mut camera = Self {
            target,
            distance: (radius / half_fov.tan()).max(MIN_DISTANCE),
            yaw: downhill_yaw(&samples, min_p.y, max_p.y, horizontal_extent) + VIEW_YAW_OFFSET,
            pitch: 0.55,
            fov_y,
            znear: 0.1,
            zfar: 1.0,
            aspect,
        };
        camera.refresh_clip_planes(radius);
        camera.fit_to(&samples);

        camera
    }

    /// Finds the closest orbit distance at which every point still projects inside the
    /// viewport, with a small margin.
    ///
    /// This bisects rather than scaling by the projected extent: points close to the eye
    /// project non-linearly, so a ratio-based update oscillates for large oblique terrain.
    fn fit_to(&mut self, points: &[Vec3]) {
        const TARGET_FILL: f32 = 0.9;
        const MAX_STEPS: usize = 48;

        let mut far = self.distance.max(MIN_DISTANCE);
        for _ in 0..MAX_STEPS {
            if self.fits(points, far, TARGET_FILL) {
                break;
            }
            far *= 1.5;
        }

        let mut near = 0.0;
        for _ in 0..MAX_STEPS {
            let mid = 0.5 * (near + far);
            if mid <= MIN_DISTANCE {
                break;
            }
            if self.fits(points, mid, TARGET_FILL) {
                far = mid;
            } else {
                near = mid;
            }
        }

        self.distance = far.max(MIN_DISTANCE);
        self.refresh_clip_planes(self.distance);
    }

    fn fits(&mut self, points: &[Vec3], distance: f32, target: f32) -> bool {
        self.distance = distance;
        self.refresh_clip_planes(distance);
        let view_proj = self.view_projection();

        points.iter().all(|p| {
            let clip = transform_point4(view_proj, *p);
            // A non-positive w means the point sits behind the eye.
            clip[3] > 0.0
                && (clip[0] / clip[3]).abs() <= target
                && (clip[1] / clip[3]).abs() <= target
        })
    }

    fn refresh_clip_planes(&mut self, scene_radius: f32) {
        self.znear = (self.distance * 0.001).max(0.1);
        self.zfar = self.distance * 10.0 + scene_radius;
    }

    pub fn eye(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        self.target
            + Vec3::new(
                self.distance * cos_pitch * sin_yaw,
                self.distance * sin_pitch,
                self.distance * cos_pitch * cos_yaw,
            )
    }

    pub fn view_projection(&self) -> Mat4 {
        let view = look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = perspective_rh(self.fov_y, self.aspect, self.znear, self.zfar);
        mul(proj, view)
    }

    /// Right and up vectors of the view plane, for orienting camera-facing quads.
    pub fn billboard_axes(&self) -> (Vec3, Vec3) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        (right, right.cross(forward))
    }

    /// Rotates the camera around the target. Inputs are in radians.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw -= delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(MIN_PITCH, MAX_PITCH);
    }

    /// Moves the target in the camera's screen plane. Inputs are fractions of the viewport.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        // Scale by the world-space height of the view frustum at the target distance.
        let scale = 2.0 * self.distance * (self.fov_y * 0.5).tan();
        self.target = self.target + right * (-delta_x * scale) + up * (delta_y * scale);
    }

    /// Moves towards or away from the target. Positive `delta` zooms in.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (-delta).exp()).max(MIN_DISTANCE);
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }
}

/// Yaw that places the camera on the low side of the terrain, looking up at the
/// high ground where release areas sit. Flat or symmetric terrain (a bowl, a
/// pyramid) has no favoured side and falls back to the classic fixed angle.
fn downhill_yaw(samples: &[Vec3], min_y: f32, max_y: f32, horizontal_extent: f32) -> f32 {
    const FALLBACK_YAW: f32 = -0.6;

    let relief = max_y - min_y;
    if relief <= f32::EPSILON {
        return FALLBACK_YAW;
    }

    // Centroids of the top and bottom elevation quarters; averaging keeps a
    // single noisy pixel from steering the camera.
    let (mut high, mut high_count) = (Vec3::ZERO, 0u32);
    let (mut low, mut low_count) = (Vec3::ZERO, 0u32);
    for point in samples {
        if point.y > min_y + 0.75 * relief {
            high = high + *point;
            high_count += 1;
        } else if point.y < min_y + 0.25 * relief {
            low = low + *point;
            low_count += 1;
        }
    }
    if high_count == 0 || low_count == 0 {
        return FALLBACK_YAW;
    }

    let downhill = (low * (1.0 / low_count as f32)) - (high * (1.0 / high_count as f32));
    let horizontal = (downhill.x * downhill.x + downhill.z * downhill.z).sqrt();
    if horizontal < 0.05 * horizontal_extent {
        return FALLBACK_YAW;
    }
    downhill.x.atan2(downhill.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::transform_point;

    fn flat_terrain() -> TerrainData {
        TerrainData::new(64, 32, 10.0, vec![1000.0; 64 * 32]).unwrap()
    }

    /// Projected half-extent of the terrain: 1.0 means it exactly touches the viewport edge.
    fn projected_fill(camera: &OrbitCamera, terrain: &TerrainData) -> f32 {
        let view_proj = camera.view_projection();
        terrain
            .fit_samples()
            .iter()
            .map(|p| {
                let ndc = transform_point(view_proj, *p);
                ndc.x.abs().max(ndc.y.abs())
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn default_framing_looks_down_at_terrain_center() {
        let terrain = flat_terrain();
        let camera = OrbitCamera::framing(&terrain, 16.0 / 9.0);

        assert_eq!(camera.target, terrain.center());
        assert!(
            camera.eye().y > camera.target.y,
            "camera should be above the terrain"
        );
        assert!(camera.pitch > 0.0 && camera.pitch < MAX_PITCH);
    }

    #[test]
    fn default_framing_keeps_whole_terrain_in_view() {
        let terrain = flat_terrain();
        let (size_x, size_z) = terrain.extent();

        for aspect in [16.0 / 9.0, 1.0, 0.5] {
            let view_proj = OrbitCamera::framing(&terrain, aspect).view_projection();
            for corner in [
                Vec3::new(0.0, 1000.0, 0.0),
                Vec3::new(size_x, 1000.0, 0.0),
                Vec3::new(0.0, 1000.0, size_z),
                Vec3::new(size_x, 1000.0, size_z),
            ] {
                let ndc = transform_point(view_proj, corner);
                assert!(
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                    "corner {corner:?} outside view at aspect {aspect}: {ndc:?}"
                );
            }
        }
    }

    #[test]
    fn default_framing_fills_the_viewport() {
        // A long, steep valley: the shape that made a ratio-based fit oscillate.
        let (width, height) = (417u32, 915u32);
        let heights = (0..height)
            .flat_map(|y| (0..width).map(move |x| 600.0 + y as f32 * 2.0 + x as f32 * 0.5))
            .collect();
        let terrain = TerrainData::new(width, height, 5.0, heights).unwrap();

        for aspect in [16.0 / 9.0, 1.0, 0.6] {
            let camera = OrbitCamera::framing(&terrain, aspect);
            let fill = projected_fill(&camera, &terrain);
            assert!(
                (0.75..=1.0).contains(&fill),
                "terrain should fill the viewport at aspect {aspect}, got {fill}"
            );
        }
    }

    #[test]
    fn default_framing_faces_the_mountain_from_the_valley() {
        // Terrain rising towards +z: the camera must stand on the low (-z) side
        // looking uphill at the mountain face, not behind the summit.
        let (width, height) = (64u32, 64u32);
        let heights = (0..height)
            .flat_map(|y| (0..width).map(move |x| 100.0 + y as f32 * 5.0))
            .collect();
        let terrain = TerrainData::new(width, height, 10.0, heights).unwrap();

        let camera = OrbitCamera::framing(&terrain, 16.0 / 9.0);
        let eye = camera.eye();
        assert!(
            eye.z < camera.target.z,
            "camera should stand on the low side: eye.z {} vs target.z {}",
            eye.z,
            camera.target.z
        );
        // The high corner must stay in view, so the framing still fits the terrain.
        let view_proj = camera.view_projection();
        let ndc = transform_point(view_proj, Vec3::new(0.0, 400.0, 630.0));
        assert!(
            ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
            "summit corner out of view: {ndc:?}"
        );
    }

    #[test]
    fn zoom_and_pitch_stay_within_limits() {
        let mut camera = OrbitCamera::framing(&flat_terrain(), 1.0);
        for _ in 0..200 {
            camera.zoom(1.0);
            camera.orbit(0.0, 1.0);
        }
        assert!(camera.distance >= MIN_DISTANCE);
        assert!(camera.pitch <= MAX_PITCH);

        for _ in 0..200 {
            camera.orbit(0.0, -1.0);
        }
        assert!(camera.pitch >= MIN_PITCH);
    }
}
