#!/usr/bin/env python3
# @atlas: Blocker 0 -- master avalanche-outline shapefile (2018/2019/1999) -> per-case shapefiles + JSON panel, reusing campaign/analysis/census.py's filter funnel.
"""extract_cases.py -- turn a master avalanche-outline shapefile into the
per-case files the calibration harness consumes (`crates/cli/src/bin/calibrate.rs`,
`CaseSpec` / `prepare_case`).

Reuses the filter funnel and thresholds documented in `campaign/analysis/census.py`
verbatim (MIN_AREA/MAX_AREA/MAX_BBOX/MIN_DROP/MIN_START/MIN_NBR, the same
500 m neighbour-distance grid method). Does not invent new thresholds.

Output schema matches `data/panel/cases100.json` field-for-field:
    name, idx, objectid, shp, area, sze, start_zone, dpo_alt, frac_wdh,
    aspect, typ, aval_shape, nbr_dist, split, npts, xmin, xmax, ymin, ymax

Key convention, reverse-engineered from the existing 105-case panel (not
stated anywhere in the repo): `idx` is the 0-based feature index (FID) in the
master shapefile, `objectid` is `idx + 1`, and `name` / the output filename
are both `aval_<idx>` -- e.g. FID 52 (OBJECTID 53 in the 2018 file, which
carries a native, sequential OBJECTID field) is `aval_52`. This holds
regardless of whether the source dataset has a native id field: 2019 has no
id field at all, and 1999's `Id` field is not unique (100 distinct values
across 11,120 features), so `objectid = FID + 1` is used as a synthetic,
dataset-agnostic id for all three datasets.

Geometry engine: pyshp (pure Python) is used for the bulk read/write path,
because it is what `census.py` itself uses and because it reproduces the
existing panel's npts/area/bbox to float precision (verified: FID 52 of
outlines2018.shp gives npts=1570, bbox matching cases100.json's aval_52
entry to the last printed digit). GDAL/OGR (via subprocess, `ogrinfo`/
`ogr2ogr -dialect SQLite`) is used only for the exception path -- validity
checking and `ST_MakeValid` repair, and (opt-in) correct hole-vs-disjoint-part
resolution for genuine MultiPolygon features -- because that requires real
computational geometry (GEOS) that pyshp cannot provide, and hand-rolling a
ring-winding heuristic was tried and shown to disagree with GDAL's own
containment-based classification (see the multipart handling notes below).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter

try:
    import shapefile  # pyshp
except ImportError:
    sys.exit(
        "pyshp is required (pip install pyshp). It is not part of the "
        "project's default environment; see the extraction report for the "
        "venv this was developed against."
    )

# --------------------------------------------------------------------------
# Thresholds reused verbatim from campaign/analysis/census.py. Do not retune here.
MIN_AREA, MAX_AREA = 20000.0, 600000.0   # m^2 (2-60 ha)
MAX_BBOX = 2000.0                        # m
MIN_DROP = 150.0                         # m (start_zone - dpo_alt)
MIN_START = 1550.0                       # m
MIN_NBR = 25.0                           # m, isolation filter
CELLW = 500.0                            # neighbour-distance grid cell, m

# A field is treated as "dead" (guarded off, not filtered on) once this
# fraction of features carry a null/zero placeholder rather than real data.
# 1999's aspect/start_zone/dpo_alt/trg_typ are ~100%/~100%/~100%/99.97% dead;
# 2018/2019 are 0% dead on all four. The threshold just needs to sit between
# those two regimes; it is not a tuned parameter.
DEAD_FIELD_FRACTION = 0.95

TRG_TYP_NORMALIZE = {"unbekannt": "UNKNOWN"}  # 2019: 40 rows spelled in German


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ------------------------------------------------------------------ geometry
def shoelace_signed(ring):
    a = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def polygon_stats(rings):
    """(area, npts, (xmin, xmax, ymin, ymax)) for a polygon given as a list of
    rings (each a list of (x, y), first == last). Area is hole-aware: it sums
    signed ring areas, which is correct as long as holes are wound opposite to
    their exterior -- the ESRI shapefile convention, and what both pyshp and
    GDAL preserve. `npts` counts every stored vertex across every ring
    (closing point included), matching the historical panel exactly (verified
    against cases100.json's aval_52: 1570)."""
    area = abs(sum(shoelace_signed(r) for r in rings if len(r) >= 2))
    npts = sum(len(r) for r in rings)
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return area, npts, (min(xs), max(xs), min(ys), max(ys))


def rings_from_pyshp_shape(shp):
    parts = list(shp.parts) + [len(shp.points)]
    return [shp.points[parts[i] : parts[i + 1]] for i in range(len(parts) - 1)]


def largest_part(polygon_rings_groups):
    """Given a MultiPolygon's parts (each already correctly grouped into its
    own exterior + holes by GDAL), return the rings of the largest by area and
    a log line describing what was dropped."""
    areas = [abs(sum(shoelace_signed(r) for r in part)) for part in polygon_rings_groups]
    best = max(range(len(areas)), key=lambda i: areas[i])
    dropped = [round(a, 1) for i, a in enumerate(areas) if i != best]
    note = (
        f"multipart: kept largest of {len(areas)} parts "
        f"(area {areas[best]:.1f} m^2; dropped {dropped} m^2)"
    )
    return polygon_rings_groups[best], note


# --------------------------------------------------------------------- GDAL
def _layer_name(shp_path: str) -> str:
    return os.path.splitext(os.path.basename(shp_path))[0]


def _ogr_geojson(shp_path: str, sql: str) -> dict:
    proc = subprocess.run(
        ["ogr2ogr", "-f", "GeoJSON", "/vsistdout/", shp_path, "-dialect", "SQLite", "-sql", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def bulk_geometry_flags(shp_path: str) -> dict:
    """fid -> (valid: bool, geomtype: str) for every feature, in one pass, with
    no geometry payload in the response (fast: ~1s for 18,737 features)."""
    layer = _layer_name(shp_path)
    sql = f"SELECT ROWID as fid, ST_IsValid(geometry) as valid, GeometryType(geometry) as gt FROM {layer}"
    d = _ogr_geojson(shp_path, sql)
    out = {}
    for f in d["features"]:
        p = f["properties"]
        out[p["fid"]] = (bool(p["valid"]), p["gt"])
    return out


def fetch_gdal_geoms(shp_path: str, fids, make_valid: bool) -> dict:
    """fid -> parsed GeoJSON geometry dict, fetched straight from GDAL (used
    only for the exception path: invalid geometries needing ST_MakeValid, and
    genuine MultiPolygon features whose hole/disjoint-part grouping pyshp
    cannot resolve)."""
    if not fids:
        return {}
    layer = _layer_name(shp_path)
    expr = "ST_MakeValid(geometry)" if make_valid else "geometry"
    fid_list = ",".join(str(f) for f in sorted(fids))
    sql = f"SELECT ROWID as fid, AsGeoJSON({expr}) as g FROM {layer} WHERE ROWID IN ({fid_list})"
    d = _ogr_geojson(shp_path, sql)
    out = {}
    for f in d["features"]:
        p = f["properties"]
        g = p["g"]
        # AsGeoJSON() marks the column OFSTJSON; the GeoJSON driver embeds it
        # as a parsed object rather than an escaped string, so accept either.
        if isinstance(g, str):
            g = json.loads(g) if g else None
        out[p["fid"]] = g
    return out


# ------------------------------------------------------------------ reading
def read_master(shp_path: str):
    """Read every feature via pyshp: raw geometry + every DBF attribute,
    keyed by 0-based FID. No filtering yet."""
    r = shapefile.Reader(shp_path)
    field_names = [f[0] for f in r.fields[1:]]  # skip DeletionFlag
    feats = []
    for fid in range(len(r)):
        shp = r.shape(fid)
        rec = r.record(fid)
        rings = rings_from_pyshp_shape(shp)
        area, npts, (xmin, xmax, ymin, ymax) = polygon_stats(rings)
        feats.append(
            {
                "fid": fid,
                "objectid": fid + 1,
                "rings": rings,
                "nparts": len(shp.parts),
                "npts": npts,
                "area": area,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "attrs": dict(zip(field_names, rec)),
                "geom_ok": True,
                "geom_note": None,
            }
        )
    return feats


def detect_dead_fields(feats) -> dict:
    n = len(feats) or 1
    dead = {}
    for key in ("aspect", "start_zone", "dpo_alt", "trg_typ"):
        n_dead = sum(1 for f in feats if not f["attrs"].get(key))
        dead[key] = n_dead / n
    return dead


def normalize_trg_typ(feats) -> int:
    n = 0
    for f in feats:
        v = f["attrs"].get("trg_typ")
        if v in TRG_TYP_NORMALIZE:
            f["attrs"]["trg_typ"] = TRG_TYP_NORMALIZE[v]
            n += 1
    return n


# ------------------------------------------------------- geometry hygiene
def handle_invalid_geometries(feats, shp_path: str, force_repair: bool = False) -> None:
    """Detect GEOS-invalid geometries (ST_IsValid) and decide what to do with
    each.

    Default (force_repair=False): an invalid-but-non-degenerate geometry is
    KEPT AS RAW pyshp read it, not run through ST_MakeValid. This was not the
    obvious choice -- it was reverse-engineered from a disagreement with the
    reference panel. `aval_8912` (objectid 8913, one of 2018's 26 invalid
    geometries and the only one of the 26 that overlaps the existing 105-case
    panel) is a self-intersecting ring. `ST_MakeValid` resolves it into a
    Polygon with TWO rings: a 1508-point body and a spurious 4-point sliver
    (the self-intersection artifact) -- which then fails this pipeline's own
    "single ring" filter as a false multipart. The reference panel's
    `data/panel/cases/aval_8912.shp` instead has exactly ONE ring of 1511
    points and area 136469 m^2 -- matching the RAW, unrepaired pyshp read to
    the last point (verified). The harness's own rasteriser
    (`RasterGrid`/`read_shapefile_nth_polygon`) has no validity requirement --
    it just fills whatever rings it is given -- so there is no correctness
    reason to prefer the topologically-clean repair over the raw ring, and a
    reason not to: it silently changes which cases survive the "single ring"
    filter. Only a geometry that is genuinely unusable raw (degenerate,
    non-finite) gets repaired regardless.

    `--repair-invalid` forces ST_MakeValid on every invalid geometry, for
    users who want strict topological cleanliness and accept that this
    diverges from the historical panel on this one case.
    """
    flags = bulk_geometry_flags(shp_path)
    invalid_fids = [f["fid"] for f in feats if not flags.get(f["fid"], (True, ""))[0]]
    if not invalid_fids:
        log("  geometry: 0 invalid geometries found")
        return
    log(f"  geometry: {len(invalid_fids)} invalid geometries found (GEOS ST_IsValid = false)")
    by_fid = {f["fid"]: f for f in feats}
    to_repair = [fid for fid in invalid_fids if force_repair or by_fid[fid]["area"] <= 0.0]
    kept_raw = [fid for fid in invalid_fids if fid not in to_repair]
    for fid in kept_raw:
        by_fid[fid]["geom_note"] = (
            "flagged invalid by GEOS (self-intersecting ring); raw geometry "
            "kept as-is -- matches the reference panel's aval_8912 precedent "
            "(pass --repair-invalid to force ST_MakeValid instead)"
        )

    n_fixed = n_failed = 0
    if to_repair:
        log(f"  geometry: repairing {len(to_repair)} via ST_MakeValid (forced, or degenerate raw geometry)")
        fixed = fetch_gdal_geoms(shp_path, to_repair, make_valid=True)
        for fid in to_repair:
            feat = by_fid[fid]
            geom = fixed.get(fid)
            if geom is None or geom.get("type") not in ("Polygon", "MultiPolygon"):
                feat["geom_ok"] = False
                feat["geom_note"] = "invalid geometry; ST_MakeValid did not yield a usable polygon"
                n_failed += 1
                continue
            if geom["type"] == "Polygon":
                rings = geom["coordinates"]
                note = "invalid geometry repaired via ST_MakeValid"
            else:  # MakeValid produced a MultiPolygon out of a self-intersection
                rings, mp_note = largest_part(geom["coordinates"])
                note = f"invalid geometry repaired via ST_MakeValid, then {mp_note}"
            area, npts, (xmin, xmax, ymin, ymax) = polygon_stats(rings)
            if area <= 0.0:
                feat["geom_ok"] = False
                feat["geom_note"] = "repaired geometry is degenerate (zero area)"
                n_failed += 1
                continue
            feat.update(
                rings=rings, nparts=len(rings), npts=npts, area=area,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, geom_note=note,
            )
            n_fixed += 1
    log(f"  geometry: {len(kept_raw)} invalid kept raw, {n_fixed} repaired, {n_failed} unrepairable (dropped)")


def mark_degenerate(feats) -> int:
    """Zero/near-zero-area polygons, independent of the area *filter* stage:
    basic hygiene applied unconditionally, not gated behind MIN_AREA (which a
    relaxed/exploratory run might not apply)."""
    n = 0
    for f in feats:
        if f["geom_ok"] and f["area"] <= 0.0:
            f["geom_ok"] = False
            f["geom_note"] = "degenerate zero-area polygon"
            n += 1
    return n


def apply_multipart_largest(feats, shp_path: str, fids) -> None:
    """Opt-in: for genuinely disjoint (GDAL-classified MultiPolygon) features
    among `fids`, replace the pyshp-derived rings (which cannot distinguish a
    hole from a disjoint part -- see module docstring) with the largest part,
    fetched from GDAL so the hole/part grouping is geometrically correct."""
    if not fids:
        return
    flags = bulk_geometry_flags(shp_path)
    multi_fids = [fid for fid in fids if flags.get(fid, (True, ""))[1] == "MULTIPOLYGON"]
    if not multi_fids:
        return
    geoms = fetch_gdal_geoms(shp_path, multi_fids, make_valid=False)
    by_fid = {f["fid"]: f for f in feats}
    for fid in multi_fids:
        feat = by_fid.get(fid)
        geom = geoms.get(fid)
        if feat is None or geom is None or geom.get("type") != "MultiPolygon":
            continue
        rings, note = largest_part(geom["coordinates"])
        area, npts, (xmin, xmax, ymin, ymax) = polygon_stats(rings)
        feat.update(
            rings=rings, nparts=len(rings), npts=npts, area=area,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            geom_note=(f"{feat['geom_note']}; {note}" if feat["geom_note"] else note),
        )
        log(f"  aval_{fid}: {note}")


# --------------------------------------------------------- neighbour distance
def compute_neighbour_distance(feats) -> None:
    """Same 500 m grid method as census.py / select_sample.py: gap between
    bounding boxes (not true polygon-to-polygon distance -- an approximation
    baked into the original method, reproduced as-is), computed against every
    other feature in the SAME dataset (a different storm's polygons are not
    "neighbours" in the sense this isolation filter means)."""
    grid: dict = {}
    for p in feats:
        gx0, gx1 = int(p["xmin"] // CELLW) - 1, int(p["xmax"] // CELLW) + 2
        gy0, gy1 = int(p["ymin"] // CELLW) - 1, int(p["ymax"] // CELLW) + 2
        for gx in range(gx0, gx1):
            for gy in range(gy0, gy1):
                grid.setdefault((gx, gy), []).append(p["fid"])
    by = {p["fid"]: p for p in feats}

    def gap(a, b):
        dx = max(b["xmin"] - a["xmax"], a["xmin"] - b["xmax"], 0.0)
        dy = max(b["ymin"] - a["ymax"], a["ymin"] - b["ymax"], 0.0)
        return math.hypot(dx, dy)

    for p in feats:
        gx0, gx1 = int(p["xmin"] // CELLW) - 1, int(p["xmax"] // CELLW) + 2
        gy0, gy1 = int(p["ymin"] // CELLW) - 1, int(p["ymax"] // CELLW) + 2
        best, seen = 1e9, set()
        for gx in range(gx0, gx1):
            for gy in range(gy0, gy1):
                for j in grid.get((gx, gy), ()):
                    if j == p["fid"] or j in seen:
                        continue
                    seen.add(j)
                    best = min(best, gap(p, by[j]))
        p["nbr_dist"] = round(min(best, 1e9), 1)


# ------------------------------------------------------------------- funnel
def build_stages(dead: dict, min_area, max_area, max_bbox, min_drop, min_start, min_nbr):
    """Returns [(label, predicate)]. Predicates assume `nbr_dist` and
    `geom_ok` are already populated on every feature."""
    stages = [
        ("all mapped polygons", lambda p: True),
        ("geometry valid/repairable & non-degenerate", lambda p: p["geom_ok"]),
        ("typ = SLAB", lambda p: p["attrs"].get("typ") == "SLAB"),
    ]
    if dead["trg_typ"] < DEAD_FIELD_FRACTION:
        stages.append(("trg_typ = NATURAL", lambda p: p["attrs"].get("trg_typ") == "NATURAL"))
    else:
        log(
            f"  !! trg_typ is {dead['trg_typ']*100:.1f}% null/dead in this dataset -- "
            "GUARDED: skipping the 'trg_typ = NATURAL' filter stage entirely "
            "(applying it would eliminate ~all candidates on a dead field, not a real trigger signal)"
        )
    stages.append(("single ring (nparts = 1)", lambda p: p["nparts"] == 1))
    stages.append(("size class 2-5", lambda p: 2 <= (p["attrs"].get("sze") or -1) <= 5))
    stages.append(("area 2-60 ha", lambda p: min_area <= p["area"] <= max_area))
    stages.append(
        ("bbox <= 2000 m", lambda p: max(p["xmax"] - p["xmin"], p["ymax"] - p["ymin"]) <= max_bbox)
    )
    if dead["start_zone"] < DEAD_FIELD_FRACTION and dead["dpo_alt"] < DEAD_FIELD_FRACTION:
        stages.append(
            (
                f"drop >= {min_drop:.0f} m",
                lambda p: (p["attrs"].get("start_zone") or 0.0) - (p["attrs"].get("dpo_alt") or 0.0)
                >= min_drop,
            )
        )
        stages.append(("start zone >= %.0f m" % min_start, lambda p: (p["attrs"].get("start_zone") or 0.0) >= min_start))
    else:
        log(
            f"  !! start_zone/dpo_alt are dead in this dataset "
            f"(start_zone {dead['start_zone']*100:.1f}% zero/null, dpo_alt {dead['dpo_alt']*100:.1f}%) -- "
            "GUARDED: skipping the 'drop >= 150 m' and 'start zone >= 1550 m' filters. "
            "Elevation-based filtering is UNAVAILABLE for this dataset; do not trust any "
            "elevation-derived field emitted for it."
        )
    stages.append((f"no neighbour within {min_nbr:.0f} m", lambda p: p["nbr_dist"] >= min_nbr))
    return stages


def run_funnel(feats, stages):
    funnel = []
    cur = list(feats)
    for label, fn in stages:
        before = len(cur)
        cur = [p for p in cur if fn(p)]
        alone = sum(1 for p in feats if not fn(p))
        funnel.append(
            {
                "stage": label,
                "remaining": len(cur),
                "removed": before - len(cur),
                "removed_pct_of_prev": 100 * (before - len(cur)) / before if before else 0.0,
                "would_remove_alone": alone,
            }
        )
    return funnel, cur


# --------------------------------------------------------------------- split
def assign_split(dataset: str, objectid: int, seed: str) -> str:
    """Deterministic hash of (seed, dataset, objectid) -> 'val' 1/3 of the
    time, 'cal' 2/3 -- the same ~2:1 ratio as the existing 105-case panel
    (66 cal / 39 val = 1.69:1). This is independent of, and will generally
    disagree with, the historical panel's recorded split for the 105 cases
    that overlap it (that split is preserved as-is in cases100.json; nothing
    here overwrites it)."""
    h = hashlib.sha256(f"{seed}:{dataset}:{objectid}".encode()).hexdigest()
    return "val" if int(h, 16) % 3 == 0 else "cal"


# -------------------------------------------------------------------- write
def write_case_shapefile(out_dir: str, name: str, objectid: int, rings, prj_text: str):
    os.makedirs(out_dir, exist_ok=True)
    # pyshp's Writer strips a trailing .shp itself before appending
    # .shp/.shx/.dbf, but our own .prj write needs the same treatment
    # explicitly -- otherwise it lands as "aval_52.shp.prj" (silently ignored
    # by GDAL/QGIS/every consumer that looks for "aval_52.prj") instead of
    # matching the sibling .shp/.shx/.dbf basename.
    base = os.path.join(out_dir, os.path.splitext(name)[0])
    w = shapefile.Writer(base, shapeType=shapefile.POLYGON)
    w.field("OBJECTID", "N", 10, 0)
    w.poly(rings)
    w.record(OBJECTID=objectid)
    w.close()
    with open(base + ".prj", "w") as fh:
        fh.write(prj_text)


def case_json(dataset, feat, split):
    a = feat["attrs"]
    return {
        "name": f"aval_{feat['fid']}",
        "idx": feat["fid"],
        "objectid": feat["objectid"],
        "shp": f"cases/aval_{feat['fid']}.shp",
        "area": round(feat["area"], 2),
        "sze": a.get("sze"),
        "start_zone": a.get("start_zone") if a.get("start_zone") else None,
        "dpo_alt": a.get("dpo_alt") if a.get("dpo_alt") else None,
        "frac_wdh": a.get("frac_wdh"),
        "aspect": a.get("aspect") or None,
        "typ": a.get("typ"),
        "aval_shape": a.get("aval_shape"),
        "nbr_dist": feat.get("nbr_dist"),
        "split": split,
        "npts": feat["npts"],
        "xmin": feat["xmin"],
        "xmax": feat["xmax"],
        "ymin": feat["ymin"],
        "ymax": feat["ymax"],
    }


DEAD_ELEV_FIELDS = ("start_zone", "dpo_alt", "aspect", "trg_typ")


def guard_dead_fields_in_output(dataset, entry, dead):
    """Write explicit nulls for fields the dataset cannot support, rather than
    passing through a placeholder zero/empty-string that looks like real
    data. Only applied to fields actually detected as dead at load time."""
    if dead.get("aspect", 0) >= DEAD_FIELD_FRACTION:
        entry["aspect"] = None
    if dead.get("start_zone", 0) >= DEAD_FIELD_FRACTION:
        entry["start_zone"] = None
    if dead.get("dpo_alt", 0) >= DEAD_FIELD_FRACTION:
        entry["dpo_alt"] = None
    return entry


# ---------------------------------------------------------------------- main
def parse_objectid_list(spec: str):
    if os.path.isfile(spec):
        with open(spec) as fh:
            text = fh.read()
    else:
        text = spec
    return {int(tok) for tok in text.replace(",", " ").split()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="path to the master .shp")
    ap.add_argument("--dataset", required=True, choices=["2018", "2019", "1999"])
    ap.add_argument("--out", required=True, help="output directory (gets a cases/ subdir + JSON + funnel report)")
    ap.add_argument("--objectids", default=None, help="comma-separated ids or path to a file, to bypass the funnel and extract exactly these (synthetic objectid = FID + 1)")
    ap.add_argument("--keep-multipart-largest", action="store_true", help="instead of excluding genuinely multi-part (disjoint) features via the single-ring filter, reduce them to their largest part and keep them")
    ap.add_argument("--repair-invalid", action="store_true", help="force ST_MakeValid on every GEOS-invalid geometry rather than keeping raw geometry as-is (default matches the reference panel's aval_8912 precedent; see handle_invalid_geometries docstring)")
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--max-area", type=float, default=MAX_AREA)
    ap.add_argument("--max-bbox", type=float, default=MAX_BBOX)
    ap.add_argument("--min-drop", type=float, default=MIN_DROP)
    ap.add_argument("--min-start", type=float, default=MIN_START)
    ap.add_argument("--min-nbr", type=float, default=MIN_NBR)
    ap.add_argument("--split-seed", default="avalanchers-panel-v1")
    ap.add_argument("--dry-run", action="store_true", help="print the funnel, write no files")
    args = ap.parse_args()

    log(f"=== extract_cases: dataset {args.dataset}, master {args.master} ===")
    feats = read_master(args.master)
    log(f"  read {len(feats)} features via pyshp")

    dead = detect_dead_fields(feats)
    for k, v in dead.items():
        log(f"  field '{k}': {v*100:.2f}% null/zero")

    if args.dataset == "2019":
        n_norm = normalize_trg_typ(feats)
        log(f"  normalized {n_norm} 'unbekannt' -> 'UNKNOWN' in trg_typ")

    handle_invalid_geometries(feats, args.master, force_repair=args.repair_invalid)
    n_degen = mark_degenerate(feats)
    log(f"  degenerate zero-area polygons found post-repair: {n_degen}")

    compute_neighbour_distance(feats)

    prj_path = os.path.splitext(args.master)[0] + ".prj"
    prj_text = open(prj_path).read() if os.path.exists(prj_path) else ""

    os.makedirs(args.out, exist_ok=True)
    cases_dir = os.path.join(args.out, "cases")

    if args.objectids:
        wanted = parse_objectid_list(args.objectids)
        by_oid = {f["objectid"]: f for f in feats}
        missing = wanted - set(by_oid)
        if missing:
            log(f"  !! {len(missing)} requested objectids not found in this dataset: {sorted(missing)[:20]}")
        pool = [by_oid[o] for o in sorted(wanted) if o in by_oid]
        unusable = [f for f in pool if not f["geom_ok"]]
        for f in unusable:
            log(f"  aval_{f['fid']}: SKIPPED ({f['geom_note']})")
        pool = [f for f in pool if f["geom_ok"]]
        if args.keep_multipart_largest:
            apply_multipart_largest(feats, args.master, [f["fid"] for f in pool])
        else:
            multi = [f for f in pool if f["nparts"] != 1]
            for f in multi:
                log(f"  aval_{f['fid']}: SKIPPED (nparts={f['nparts']}, pass --keep-multipart-largest to reduce instead)")
            pool = [f for f in pool if f["nparts"] == 1]
        funnel = None
    else:
        stages = build_stages(
            dead, args.min_area, args.max_area, args.max_bbox, args.min_drop, args.min_start, args.min_nbr
        )
        funnel, pool = run_funnel(feats, stages)
        if args.keep_multipart_largest:
            # re-admit genuinely-multipart features the "single ring" stage
            # dropped, reducing each to its largest part.
            excluded_multi = [f for f in feats if f["geom_ok"] and f["nparts"] != 1]
            apply_multipart_largest(feats, args.master, [f["fid"] for f in excluded_multi])
            readmitted = []
            passed_before_nparts = feats  # recompute pool with nparts stage skipped for these
            for f in excluded_multi:
                if f["nparts"] == 1:  # now reduced to a single part
                    readmitted.append(f)
            pool = pool + readmitted
            if readmitted:
                log(f"  --keep-multipart-largest: readmitted {len(readmitted)} reduced features")

    pool.sort(key=lambda f: f["objectid"])
    log(f"  candidate pool: {len(pool)}")

    n_written = 0
    entries = []
    for feat in pool:
        split = assign_split(args.dataset, feat["objectid"], args.split_seed)
        entry = case_json(args.dataset, feat, split)
        entry = guard_dead_fields_in_output(args.dataset, entry, dead)
        entries.append(entry)
        if not args.dry_run:
            write_case_shapefile(cases_dir, f"aval_{feat['fid']}.shp", feat["objectid"], feat["rings"], prj_text)
            n_written += 1

    if not args.dry_run:
        with open(os.path.join(args.out, f"cases_{args.dataset}.json"), "w") as fh:
            json.dump(entries, fh, indent=1)
        log(f"  wrote {n_written} case shapefiles + cases_{args.dataset}.json to {args.out}")

    report = {
        "dataset": args.dataset,
        "n_total": len(feats),
        "dead_fields": dead,
        "funnel": funnel,
        "n_pool": len(pool),
        "pool_by_size": [{"value": str(s), "n": sum(1 for p in pool if p["attrs"].get("sze") == s)} for s in sorted({p["attrs"].get("sze") for p in pool})],
    }
    if not args.dry_run:
        with open(os.path.join(args.out, f"funnel_{args.dataset}.json"), "w") as fh:
            json.dump(report, fh, indent=1)

    if funnel:
        log("\nfunnel:")
        for f in funnel:
            log(
                f"  {f['stage']:44s} -> {f['remaining']:6d}  (-{f['removed']:5d}, "
                f"{f['removed_pct_of_prev']:5.1f}% of previous; alone would cut {f['would_remove_alone']})"
            )
    log(f"\npool by size: {report['pool_by_size']}")
    return report


if __name__ == "__main__":
    main()
