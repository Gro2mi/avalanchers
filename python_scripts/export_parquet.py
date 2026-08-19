#!/usr/bin/env python3
# @atlas: Exports the calibration SQLite database to per-stage Parquet, plus the joined candidate metadata. Columnar copies for the final report and for anything that wants to read the campaign without SQLite.
"""Export the calibration database to Parquet, one file per stage.

SQLite is the write path -- it takes concurrent appends while a stage runs and
survives a hard kill. Parquet is the read path: columnar, compressed, and
readable by pandas/polars/duckdb without a driver. Both are kept; this does not
replace the database.

Writes, into --out-dir:
    evals_<stage>.parquet    one row per simulation, candidate columns joined on
    candidates.parquet       every global candidate with its full params JSON
    sources.parquet          job-level provenance (git hash, host, timestamps)
    case_best_<stage>.parquet  the per-(candidate, case) best inner-loop score,
                             which is the table every comparison in the plan
                             actually operates on

    python3 python_scripts/export_parquet.py \\
        --db ~/data/calibration.sqlite --out-dir ~/data/parquet
"""
import argparse
import sqlite3
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow is required: pip install pyarrow")


def fetch(con, sql, params=()):
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    # Column-major without pandas: pyarrow wants columns anyway.
    return pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)}) if rows else None


def write(table, path: Path, label: str):
    if table is None:
        print(f"  (skipping {label}: no rows)")
        return 0
    pq.write_table(table, path, compression="zstd")
    mb = path.stat().st_size / 1e6
    print(f"  {label:>28}  {table.num_rows:>9,} rows  {mb:>7.2f} MB  -> {path.name}")
    return table.num_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stage", default=None, help="export only this stage")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)

    stages = ([args.stage] if args.stage else
              [r[0] for r in con.execute("SELECT DISTINCT stage FROM evals ORDER BY 1")])
    if not stages:
        sys.exit(f"no stages found in {args.db}")

    total = 0
    write(fetch(con, "SELECT * FROM candidates"), out / "candidates.parquet", "candidates")
    write(fetch(con, "SELECT * FROM sources"), out / "sources.parquet", "sources")

    for st in stages:
        # Candidate knobs joined on, so a stage file stands alone in the report.
        t = fetch(con, """
            SELECT e.*, c.friction_model, c.flags, c.drag_coefficient,
                   c.internal_friction_angle, c.cfl AS cand_cfl,
                   c.released_particles_per_cell AS cand_ppc, c.max_steps AS cand_max_steps
            FROM evals e LEFT JOIN candidates c
              ON c.candidate_id = e.candidate_id AND c.stage = e.stage
            WHERE e.stage = ?""", (st,))
        total += write(t, out / f"evals_{st}.parquet", f"evals[{st}]")

        # The decision-relevant view: best inner-loop score per (candidate, case).
        b = fetch(con, """
            SELECT candidate_id, case_name,
                   MAX(omega) AS omega_best, MAX(omega_cells) AS omega_cells_best,
                   MAX(hwri_l1) AS hwri_l1_best, COUNT(*) AS n_evals,
                   SUM(seconds) AS seconds_total,
                   MIN(release_clipped_frac) AS release_clipped_frac,
                   MAX(release_clip_severe) AS release_clip_severe,
                   MAX(connect_fallback) AS connect_fallback
            FROM evals WHERE stage = ? AND ok = 1
            GROUP BY candidate_id, case_name""", (st,))
        write(b, out / f"case_best_{st}.parquet", f"case_best[{st}]")

    print(f"\nexported {total:,} evaluation rows across {len(stages)} stage(s) to {out}")


if __name__ == "__main__":
    main()
