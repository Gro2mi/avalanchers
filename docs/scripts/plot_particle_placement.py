"""Thesis figure of the particle placement with relaxation.

Seeds the particles of a square release on a flat plane twice with the
avalanchers engine: once without and once with relaxation by the soft sphere
repulsion. With ``--placement uniform-relaxed`` (default) the seeding is
stratified (``uniform-random``: one particle per equal-area stratum of the
cell); with ``--placement relaxed`` the particles start at uniformly random
positions in the cell (``random``). The seeding is deterministic per cell
(its random numbers are hashed from the cell index) and each cell's particles
occupy a contiguous block of the particle buffer in seeding order, which the
relaxation keeps, so particle k of a cell in the relaxed run is particle k of
the same cell in the unrelaxed run.

The figure shows a window of 2 x 1 release cells: the cell outlines bold,
the equal-area strata of the stratified sampling thin (stratified seeding
only), the seeded (initial) positions as small open circles, the relaxed
(final) positions as filled circles, and an arrow from each initial to its
final position. Particles of the neighbouring cells are drawn too, so the
window is not cut out of the packing.

Usage:
    python plot_particle_placement.py                  # 13 particles per cell
    python plot_particle_placement.py --particles-per-cell 7
    python plot_particle_placement.py --placement relaxed  # random seeding
    python plot_particle_placement.py --location interior  # release corner
    python plot_particle_placement.py --show
"""

import argparse
import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# shared helpers (thesis_style, cli_args) live in verification/
sys.path.insert(0, str(REPO_ROOT / "verification"))
OUTPUT_DIR = REPO_ROOT / "docs" / "figures"

CELL_SIZE = 1.0          # m; the figure is in cell units, so any size works
GRID_CELLS = 16          # flat plane of GRID_CELLS x GRID_CELLS cells
RELEASE_CELLS = (4, 12)  # release block [start, end) in x and y
RELEASE_THICKNESS = 1.0  # m
# window size in cells and its lower-left cell relative to the release block
WINDOW_CELLS = (2, 1)
WINDOW = {"interior": (3, 3), "corner": (0, 0)}
# an odd, non-square count shows rows with different numbers of strata
DEFAULT_PARTICLES_PER_CELL = 13
# relaxed placement -> (the same seeding without relaxation, its label)
PLACEMENTS = {
    "uniform-relaxed": ("uniform-random", "stratified"),
    "relaxed": ("random", "random"),
}
# margin of the plotted window around its cells, in cells
MARGIN = 0.25
# accepted nearest-neighbour distance of the relaxed particles in target
# spacings h = sqrt(2 / (sqrt(3) N)) cells of the hexagonal packing; the
# relaxation reaches 0.82-1.25 h for 4 to 64 particles per cell from stratified
# and 0.77-1.12 h from random seeding
RELAXED_SPACING_RANGE = (0.7, 1.35)
# displacement arrows: small filled tip (lengths in points); a displacement
# too short to show the tip between the two markers is drawn as a plain line
ARROW_HEAD_LENGTH = 3.2
ARROW_STYLE = f"-|>,head_length={ARROW_HEAD_LENGTH},head_width=1.4"

SETTINGS = {
    "sim_model": "terrain-following",
    "density": 200.0,
    "release_thickness": RELEASE_THICKNESS,
}


def strata_rows(n):
    """Row bounds of the equal-area strata of a cell with n particles, as in
    initialize_particles.wgsl: R = round(sqrt(n)) rows, row j holds the
    particles s_j <= k < s_{j+1} with s_j = floor(j n / R) side by side.
    Returns a list of (y0, y1, columns) in cell units."""
    rows = max(1, round(math.sqrt(n)))
    starts = [j * n // rows for j in range(rows + 1)]
    return [(starts[j] / n, starts[j + 1] / n, starts[j + 1] - starts[j])
            for j in range(rows)]


def seed(placement, particles_per_cell):
    """Particle positions (n, 2) in m after seeding with `placement`."""
    import avalanchers

    dem = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)
    release = np.zeros_like(dem)
    lo, hi = RELEASE_CELLS
    release[lo:hi, lo:hi] = RELEASE_THICKNESS
    sim = avalanchers.PySimulation.new()
    settings = dict(SETTINGS, particle_placement=placement,
                    released_particles_per_cell=particles_per_cell)
    sim.create(settings)
    sim.set_dem(np.ascontiguousarray(dem), CELL_SIZE)
    sim.set_release_areas(np.ascontiguousarray(release))
    sim.prepare()
    return np.asarray(sim.particles_position_xy, dtype=float)


def blocks_by_cell(positions, particles_per_cell, use_mean):
    """Map (cx, cy) -> (particles_per_cell, 2) positions of the cell's block.

    Each release cell seeds one contiguous block of particles. The cell of a
    block is the cell of its first particle (unrelaxed positions, which all
    lie in their cell) or of its mean position (relaxed positions, which may
    have left the cell by a fraction of the particle spacing)."""
    blocks = positions.reshape(-1, particles_per_cell, 2)
    cells = {}
    for block in blocks:
        anchor = block.mean(axis=0) if use_mean else block[0]
        cell = tuple(int(v) for v in np.floor(anchor / CELL_SIZE))
        if cell in cells:
            raise RuntimeError(f"two particle blocks map to cell {cell}")
        cells[cell] = block
    return cells


def paired_positions(placement, particles_per_cell):
    """{cell: (initial, final)} positions in cell units for every release
    cell, seeded with the relaxed `placement` and without its relaxation."""
    seeding, _ = PLACEMENTS[placement]
    initial = blocks_by_cell(seed(seeding, particles_per_cell),
                             particles_per_cell, use_mean=False)
    final = blocks_by_cell(seed(placement, particles_per_cell),
                           particles_per_cell, use_mean=True)
    if initial.keys() != final.keys():
        raise RuntimeError("the two seedings cover different cells")
    pairs = {cell: (initial[cell] / CELL_SIZE, final[cell] / CELL_SIZE)
             for cell in initial}
    worst = max(np.max(np.linalg.norm(b - a, axis=1)) for a, b in pairs.values())
    spacing = math.sqrt(2.0 / (math.sqrt(3.0) * particles_per_cell))
    print(f"{len(pairs)} release cells, largest relaxation displacement "
          f"{worst:.3f} cells ({worst / spacing:.2f} target spacings)")
    if worst > 1.0:
        raise RuntimeError("particles moved by more than a cell: pairing is unreliable")
    return pairs, spacing


def nearest_neighbour_distances(points):
    """Distance of every point to its nearest other point."""
    nearest = np.empty(len(points))
    for start in range(0, len(points), 512):
        chunk = points[start:start + 512]
        distance = np.linalg.norm(chunk[:, None, :] - points[None, :, :], axis=2)
        distance[np.arange(len(chunk)), np.arange(start, start + len(chunk))] = np.inf
        nearest[start:start + len(chunk)] = distance.min(axis=1)
    return nearest


def check_spacing(pairs, spacing, initial_label):
    """Print the smallest and largest nearest-neighbour distance of the
    seeded and the relaxed particles, in cells and in target spacings h, and
    assert that the relaxed ones lie in RELAXED_SPACING_RANGE.

    Seeded particles can almost touch (random seeding anywhere in the cell,
    stratified seeding in neighbouring strata); the relaxation pushes them
    apart to about the hexagonal packing spacing h. Particles at the release
    edge are included: they slide along it and stay at about h too."""
    low, high = RELAXED_SPACING_RANGE
    print(f"nearest-neighbour distance (target spacing h = {spacing:.3f} cells):")
    for k, name in ((0, initial_label), (1, "relaxed")):
        nearest = nearest_neighbour_distances(
            np.concatenate([positions[k] for positions in pairs.values()]))
        print(f"  {name:<10}  min {nearest.min():.3f} cells ({nearest.min() / spacing:.2f} h), "
              f"max {nearest.max():.3f} cells ({nearest.max() / spacing:.2f} h), "
              f"mean {nearest.mean() / spacing:.2f} h")
    relaxed = nearest / spacing
    assert relaxed.min() >= low, (
        f"relaxed particles closer than {low} h: {relaxed.min():.3f} h")
    assert relaxed.max() <= high, (
        f"relaxed particle farther than {high} h from its nearest neighbour: "
        f"{relaxed.max():.3f} h")
    print(f"  relaxed distances lie within [{low}, {high}] h")


def plot(pairs, particles_per_cell, placement, window, out_dir, show):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    class HandlerArrow(HandlerPatch):
        """Legend entry drawing the displacement arrow itself."""

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                           height, fontsize, trans):
            y = 0.5 * height - ydescent
            arrow = FancyArrowPatch((-xdescent, y), (width - xdescent, y),
                                    arrowstyle=ARROW_STYLE, mutation_scale=1.0,
                                    color=orig_handle.get_edgecolor(),
                                    linewidth=orig_handle.get_linewidth(),
                                    transform=trans)
            return [arrow]

    import thesis_style as ts

    x0, y0 = RELEASE_CELLS[0] + window[0], RELEASE_CELLS[0] + window[1]
    width, height = WINDOW_CELLS
    view = (x0 - MARGIN, x0 + width + MARGIN, y0 - MARGIN, y0 + height + MARGIN)
    release_lo, release_hi = RELEASE_CELLS
    stratified = placement == "uniform-relaxed"
    initial_label = PLACEMENTS[placement][1]
    # the relaxed positions are the result and dominate; the seeded start
    # stays in the background as small open grey circles
    initial_color, final_color, arrow_color = ts.NEUTRAL, ts.VERMILION, ts.BLUE
    initial_size, final_size = 7.0, 18.0  # marker areas in pt^2

    with ts.style():
        aspect = (height + 2 * MARGIN) / (width + 2 * MARGIN)
        fig, ax = plt.subplots(figsize=ts.size(0.85, 0.85 * aspect))
        ax.grid(False)
        ax.set_xlim(view[0] - x0, view[1] - x0)
        ax.set_ylim(view[2] - y0, view[3] - y0)
        ax.set_aspect("equal")

        visible = [(cx, cy) for cx in range(math.floor(view[0]), math.ceil(view[1]))
                   for cy in range(math.floor(view[2]), math.ceil(view[3]))]
        # light fill of the release cells, so the release edge of the corner
        # window is visible
        for cx, cy in visible:
            if release_lo <= cx < release_hi and release_lo <= cy < release_hi:
                ax.add_patch(plt.Rectangle((cx - x0, cy - y0), 1, 1, facecolor="0.94",
                                           edgecolor="none", zorder=0))
        # thin strata outlines of the stratified sampling
        for cx, cy in visible if stratified else ():
            if not (release_lo <= cx < release_hi and release_lo <= cy < release_hi):
                continue
            for row_y0, row_y1, columns in strata_rows(particles_per_cell):
                ax.plot([cx - x0, cx - x0 + 1], [cy - y0 + row_y1] * 2,
                        color="0.55", linewidth=0.35, zorder=1)
                for c in range(1, columns):
                    ax.plot([cx - x0 + c / columns] * 2,
                            [cy - y0 + row_y0, cy - y0 + row_y1],
                            color="0.55", linewidth=0.35, zorder=1)
        # bold cell outlines
        for gx in range(math.floor(view[0]), math.ceil(view[1]) + 1):
            ax.axvline(gx - x0, color=ts.BLACK, linewidth=1.3, zorder=2)
        for gy in range(math.floor(view[2]), math.ceil(view[3]) + 1):
            ax.axhline(gy - y0, color=ts.BLACK, linewidth=1.3, zorder=2)

        # points per cell, to find the displacements too short for a tip
        fig.tight_layout(pad=0.3)
        points_per_cell = (ax.get_position().width * fig.get_figwidth() * 72.0
                           / (view[1] - view[0]))
        shrink_a = math.sqrt(initial_size) / 2
        shrink_b = math.sqrt(final_size) / 2 + 0.2
        min_arrow_points = shrink_a + shrink_b + ARROW_HEAD_LENGTH + 1.0
        for cell, (initial, final) in pairs.items():
            if cell not in visible:
                continue
            offset = np.array([x0, y0], dtype=float)
            a, b = initial - offset, final - offset
            for start, end in zip(a, b):
                if np.linalg.norm(end - start) * points_per_cell < min_arrow_points:
                    # plain line from the edge of the open initial marker
                    # (not through it) to under the relaxed one
                    ax.add_patch(FancyArrowPatch(
                        start, end, arrowstyle="-", color=arrow_color, linewidth=0.6,
                        zorder=3, clip_on=True, shrinkA=shrink_a, shrinkB=0.0))
                    continue
                # the arrow starts at the edge of the initial marker and its
                # tip touches the relaxed one (shrinks in points)
                ax.add_patch(FancyArrowPatch(
                    start, end, arrowstyle=ARROW_STYLE, mutation_scale=1.0,
                    color=arrow_color, linewidth=0.6, zorder=3, clip_on=True,
                    shrinkA=shrink_a, shrinkB=shrink_b))
            ax.scatter(a[:, 0], a[:, 1], s=initial_size, facecolors="none",
                       edgecolors=initial_color, linewidths=0.6, zorder=4)
            ax.scatter(b[:, 0], b[:, 1], s=final_size, color=final_color,
                       edgecolors="white", linewidths=0.4, zorder=5)

        ax.set_xlabel(r"$x / \Delta x$")
        ax.set_ylabel(r"$y / \Delta x$")
        ax.set_xticks(range(width + 1))
        ax.set_yticks(range(height + 1))
        handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=math.sqrt(initial_size),
                   markerfacecolor="none", markeredgecolor=initial_color,
                   markeredgewidth=0.6, label=f"{initial_label} (initial)"),
            Line2D([], [], linestyle="none", marker="o", markersize=math.sqrt(final_size),
                   markerfacecolor=final_color, markeredgecolor="white",
                   markeredgewidth=0.4, label="relaxed (final)"),
            FancyArrowPatch((0, 0), (1, 0), arrowstyle=ARROW_STYLE, mutation_scale=1.0,
                            color=arrow_color, linewidth=0.6, label="displacement"),
        ]
        if stratified:
            handles.append(Line2D([], [], color="0.55", linewidth=0.35,
                                  label="equal-area strata"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  handletextpad=0.4, borderaxespad=0.6,
                  handler_map={FancyArrowPatch: HandlerArrow()})
        fig.tight_layout(pad=0.3)
        name = f"particle_placement_n{particles_per_cell}"
        if not stratified:
            name += "_random"
        if window != WINDOW["interior"]:
            name += "_corner"
        if show:
            plt.show(block=False)
        ts.save(fig, out_dir, name)


def main():
    from cli_args import positive_int

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--particles-per-cell", type=positive_int,
                        default=DEFAULT_PARTICLES_PER_CELL, metavar="N",
                        help="released particles per release cell "
                             f"(default {DEFAULT_PARTICLES_PER_CELL})")
    parser.add_argument("--placement", choices=sorted(PLACEMENTS), default="uniform-relaxed",
                        help="relaxed placement to show: uniform-relaxed (stratified "
                             "seeding, default) or relaxed (random seeding)")
    parser.add_argument("--location", choices=sorted(WINDOW), default="corner",
                        help="2 x 1 window inside the release or at its corner")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="also open the figure")
    args = parser.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    pairs, spacing = paired_positions(args.placement, args.particles_per_cell)
    check_spacing(pairs, spacing, PLACEMENTS[args.placement][1])
    plot(pairs, args.particles_per_cell, args.placement, WINDOW[args.location],
         args.out_dir, args.show)


if __name__ == "__main__":
    main()
