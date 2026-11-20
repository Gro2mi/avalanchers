"""Wall-impact test case, viewable in the live viewer.

Snow released on a parabolic slope runs into a vertical wall. The terrain is
uniform across the slope (a quasi-2D profile along x):

    z(x) = BOTTOM + c (VERTEX_X - x)^2         for x <  WALL_X  (parabola)
    z(x) = z(WALL_X) + WALL_HEIGHT             for x >= WALL_X  (plateau)

The parabola falls along +x and ends at WALL_X, short of its vertex, so the
flow still moves downhill when it hits the wall. A DEM cannot hold a vertical
face: the wall is the elevation step between two neighboring cell centers,
and the central differences of the bed normal spread it over two cells
(atan(WALL_HEIGHT / (2 CELL_SIZE)) = 79 degrees for 50 m on 5 m cells).

Writes, following the avaframe convention the live viewer's testcase
dropdown scans,

    data/avaframe/avaParabolaWall.png               float32 elevations in the RGBA bytes
    data/avaframe/avaParabolaWall.aabb              xmin, ymin, xmax, ymax of the cell centres
    data/avaframe/avaParabolaWallreleaseTexture.png release thickness * 100 in alpha
    verification/wall_settings.json                 3D model, Coulomb friction, curvature on

View with
    cargo run --release -p render_core --example live_sim -- verification/wall_settings.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CASE = "avaParabolaWall"

CELL_SIZE = 5.0  # m
LENGTH = 1100.0  # m along the fall line (x)
WIDTH = 400.0  # m across the slope (y)
BOTTOM = 2000.0  # m, elevation of the parabola vertex; 0 is the PNG no-data value

VERTEX_X = 1000.0  # m
DROP = 500.0  # m, height of the parabola at x = 0 above its vertex
WALL_X = 900.0  # m, first cell center of the wall plateau
WALL_HEIGHT = 50.0  # m

RELEASE_X = (100.0, 220.0)  # m
RELEASE_Y = (140.0, 260.0)  # m
RELEASE_THICKNESS = 2.0  # m, at most 2.55 (8-bit alpha / 100)

# model settings of the test case: as in the double-bump case
MODEL_SETTINGS = {
    "sim_model": "dampm-global-cartesian",
    "friction_model": "coulomb",
    "friction_coefficient": 0.4,
    "enable_curvature": True,
    "enable_detachment": True,
    "enable_particle_interaction": True,
    "released_particles_per_cell": 8,
    "cfl": 0.5,
    "max_steps": 6000,
    "velocity_threshold": 0.1,
}


def parabola(x: np.ndarray) -> np.ndarray:
    return BOTTOM + DROP / VERTEX_X**2 * (VERTEX_X - np.asarray(x, dtype=float)) ** 2


def profile(x: np.ndarray) -> np.ndarray:
    """Bed elevation: the parabola up to the wall, the plateau behind it."""
    x = np.asarray(x, dtype=float)
    return np.where(x < WALL_X, parabola(x), parabola(WALL_X) + WALL_HEIGHT)


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
    dem_path, release_path = raster_paths(out_dir)
    Image.fromarray(rgba, mode="RGBA").save(dem_path)

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
    Image.fromarray(release, mode="RGBA").save(release_path)

    foot = float(parabola(WALL_X - CELL_SIZE))
    print(f"{CASE}: {nx} x {ny} cells of {CELL_SIZE} m, elevation {dem.min():.1f}..{dem.max():.1f} m, "
          f"wall step {foot:.1f} -> {float(profile(WALL_X)):.1f} m at x = {WALL_X} m "
          f"({np.degrees(np.arctan(WALL_HEIGHT / (2 * CELL_SIZE))):.0f} deg in the bed normal)")
    print(f"wrote {dem_path.with_suffix('')}.{{png,aabb}} and {release_path.name}")
    if settings_path is not None:
        settings_path.write_text(json.dumps(viewer_settings(out_dir), indent=2) + "\n")
        print(f"wrote {settings_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "avaframe")
    parser.add_argument("--settings", type=Path, default=ROOT / "verification" / "wall_settings.json")
    args = parser.parse_args()
    write_testcase(args.out_dir.resolve(), args.settings.resolve())


if __name__ == "__main__":
    main()
