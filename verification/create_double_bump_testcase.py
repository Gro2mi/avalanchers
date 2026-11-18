"""Double-bump test case for flying particles, viewable in the live viewer.

Avalanche-scale take on the double-bump granular-flow example of Matter
(github.com/larsblatny/matter, whose bump is the sech profile of Viroulet et
al. 2017, JFM, doi:10.1017/jfm.2017.41): snow released on an inclined plane
accelerates, takes off at the crest of the first bump and lands on the
upslope face of the second, then runs out on a flat valley floor.

The terrain is uniform across the slope (a quasi-2D ridge profile along x):

    z(x) = plane(x) + sum_i H / cosh((x - x_i) / w)

where the plane falls at SLOPE_DEG and bends smoothly (tanh in the slope)
onto flat ground around RUNOUT_X. Writes, following the avaframe convention
the live viewer's testcase dropdown scans,

    data/avaframe/avaDoubleBump.png               float32 elevations in the RGBA bytes
    data/avaframe/avaDoubleBump.aabb              xmin, ymin, xmax, ymax of the cell centres
    data/avaframe/avaDoubleBumpreleaseTexture.png release thickness * 100 in alpha
    verification/double_bump_settings.json        3D model with curvature (detachment)

View with
    cargo run --release -p render_core --example live_sim -- verification/double_bump_settings.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CASE = "avaDoubleBump"

CELL_SIZE = 5.0  # m
LENGTH = 1500.0  # m along the fall line (x)
WIDTH = 300.0  # m across the slope (y)
BASE_ELEVATION = 2000.0  # m at the foot; 0 is the PNG no-data value

SLOPE_DEG = 32.0
RUNOUT_X = 800.0  # m, centre of the bend onto flat ground
RUNOUT_BEND = 60.0  # m, width of that bend

BUMP_HEIGHT = 20.0  # m
BUMP_WIDTH = 20.0  # m, decay length w of H / cosh((x - x_i) / w)
BUMP_CENTRES = (380.0, 460.0)  # m; with Coulomb mu = 0.4 the jump lands on the second

RELEASE_X = (60.0, 180.0)  # m
RELEASE_Y = (90.0, 210.0)  # m
RELEASE_THICKNESS = 2.0  # m, at most 2.55 (8-bit alpha / 100)


def drop(x: np.ndarray) -> np.ndarray:
    """Height lost from x = 0: the integral of the incline's slope
    tan(theta) * (1 - tanh((x - RUNOUT_X) / L)) / 2, in closed form
    (log cosh written overflow-safe with logaddexp)."""
    tan_slope = np.tan(np.radians(SLOPE_DEG))
    s = (x - RUNOUT_X) / RUNOUT_BEND
    log_cosh = np.logaddexp(s, -s) - np.log(2.0)
    return 0.5 * tan_slope * (x - RUNOUT_BEND * log_cosh)


def plane(x: np.ndarray) -> np.ndarray:
    """Incline of SLOPE_DEG falling along +x that bends onto flat ground at
    BASE_ELEVATION."""
    return BASE_ELEVATION + drop(np.asarray(LENGTH)) - drop(x)


def profile(x: np.ndarray) -> np.ndarray:
    z = plane(x)
    for centre in BUMP_CENTRES:
        z += BUMP_HEIGHT / np.cosh((x - centre) / BUMP_WIDTH)
    return z


def bump(x: np.ndarray) -> np.ndarray:
    """Height of the bumps above the plane."""
    return profile(x) - plane(x)


def profile_slope(x: np.ndarray) -> np.ndarray:
    """dz/dx of the profile."""
    tan_slope = np.tan(np.radians(SLOPE_DEG))
    slope = -0.5 * tan_slope * (1.0 - np.tanh((x - RUNOUT_X) / RUNOUT_BEND))
    for centre in BUMP_CENTRES:
        u = (x - centre) / BUMP_WIDTH
        slope -= BUMP_HEIGHT / BUMP_WIDTH * np.tanh(u) / np.cosh(u)
    return slope


def profile_curvature(x: np.ndarray) -> np.ndarray:
    """d2z/dx2 of the profile."""
    tan_slope = np.tan(np.radians(SLOPE_DEG))
    curvature = 0.5 * tan_slope / RUNOUT_BEND / np.cosh((x - RUNOUT_X) / RUNOUT_BEND) ** 2
    for centre in BUMP_CENTRES:
        u = (x - centre) / BUMP_WIDTH
        sech = 1.0 / np.cosh(u)
        curvature += BUMP_HEIGHT / BUMP_WIDTH**2 * sech * (np.tanh(u) ** 2 - sech**2)
    return curvature


# model settings of the test case, shared with run_double_bump_test.py:
# Coulomb friction, so that a point slider gives the reference
MODEL_SETTINGS = {
    "sim_model": "3d",
    "friction_model": "coulomb",
    "friction_coefficient": 0.4,
    "enable_curvature": True,
    "enable_detachment": True,
    "flight_terminal_velocity": 0.0,  # pure ballistic jump onto the second bump
    "enable_particle_interaction": True,
    "released_particles_per_cell": 8,
    "cfl": 0.5,
    "max_steps": 6000,
    "velocity_threshold": 0.1,
}


def raster_paths(out_dir: Path) -> tuple[Path, Path]:
    """DEM and release texture of the test case in out_dir."""
    return out_dir / f"{CASE}.png", out_dir / f"{CASE}releaseTexture.png"


def viewer_settings(out_dir: Path) -> dict:
    """Live viewer settings, paths relative to the repository root."""
    dem, release = raster_paths(out_dir)
    return {
        "dem_path": dem.relative_to(ROOT).as_posix(),
        "release_areas_path": release.relative_to(ROOT).as_posix(),
        **MODEL_SETTINGS,
    }


def write_testcase(out_dir: Path, settings_path: Path | None = None) -> None:
    """Writes DEM, bounds and release texture to out_dir and, if given, the
    live viewer settings to settings_path."""
    nx = int(round(LENGTH / CELL_SIZE)) + 1
    ny = int(round(WIDTH / CELL_SIZE)) + 1
    x = np.arange(nx) * CELL_SIZE
    y = np.arange(ny) * CELL_SIZE

    dem = np.tile(profile(x), (ny, 1)).astype("<f4")
    assert np.all(dem > 0.0), "0 is the PNG no-data value"
    rgba = dem.view(np.uint8).reshape(ny, nx, 4)
    Image.fromarray(rgba, mode="RGBA").save(out_dir / f"{CASE}.png")

    # bounds of the cell centres (load_png_as_dem pads half a cell)
    (out_dir / f"{CASE}.aabb").write_text(
        f"{x[0]:.1f}\n{y[0]:.1f}\n{x[-1]:.1f}\n{y[-1]:.1f}\n"
    )

    xx, yy = np.meshgrid(x, y)
    inside = (
        (xx >= RELEASE_X[0]) & (xx < RELEASE_X[1]) & (yy >= RELEASE_Y[0]) & (yy < RELEASE_Y[1])
    )
    release = np.zeros((ny, nx, 4), dtype=np.uint8)
    release[inside] = (255, 0, 0, int(round(RELEASE_THICKNESS * 100)))
    Image.fromarray(release, mode="RGBA").save(out_dir / f"{CASE}releaseTexture.png")

    print(f"{CASE}: {nx} x {ny} cells of {CELL_SIZE} m, elevation {dem.min():.1f}..{dem.max():.1f} m")
    print(f"wrote {out_dir / CASE}.{{png,aabb}} and {CASE}releaseTexture.png")
    if settings_path is not None:
        settings_path.write_text(json.dumps(viewer_settings(out_dir), indent=2) + "\n")
        print(f"wrote {settings_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "avaframe")
    parser.add_argument("--settings", type=Path, default=ROOT / "verification" / "double_bump_settings.json")
    args = parser.parse_args()
    write_testcase(args.out_dir.resolve(), args.settings.resolve())


if __name__ == "__main__":
    main()
