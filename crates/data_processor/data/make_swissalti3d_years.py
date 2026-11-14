"""Compress swisstopo's swissALTI3D link export into swissalti3d_years.bin.xz.

Get the export from https://www.swisstopo.admin.ch/en/height-model-swissalti3d
(selection: whole Switzerland, resolution 2 m, format GeoTIFF, "export all links"),
then run:

    python make_swissalti3d_years.py <export.csv> swissalti3d_years.bin.xz

The only information per tile is its acquisition year; tile_manager.rs rebuilds the
URL from it. Layout before xz compression (all integers little-endian u16):

    b"SA3D", min_easting_km, min_northing_km, width, height,
    then width * height bytes, row-major by northing: year - 2000, 0 = no tile.
"""

import lzma
import re
import struct
import sys

PATTERN = re.compile(r"swissalti3d_(\d{4})_(\d+)-(\d+)_2_2056_5728\.tif$")


def main(src: str, dst: str) -> None:
    years: dict[tuple[int, int], int] = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = PATTERN.search(line)
            if m is None:
                sys.exit(f"unrecognised link: {line}")
            year, e, n = map(int, m.groups())
            if not 2001 <= year <= 2255:
                sys.exit(f"year {year} does not fit the format: {line}")
            years[(e, n)] = max(year, years.get((e, n), 0))

    min_e = min(e for e, _ in years)
    min_n = min(n for _, n in years)
    width = max(e for e, _ in years) - min_e + 1
    height = max(n for _, n in years) - min_n + 1

    grid = bytearray(width * height)
    for (e, n), year in years.items():
        grid[(n - min_n) * width + (e - min_e)] = year - 2000

    raw = b"SA3D" + struct.pack("<4H", min_e, min_n, width, height) + bytes(grid)
    packed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    with open(dst, "wb") as f:
        f.write(packed)
    print(f"{len(years)} tiles, {width}x{height} grid, {len(packed)} bytes -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
