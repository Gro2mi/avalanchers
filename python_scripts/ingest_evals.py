#!/usr/bin/env python3
# @atlas: Loads calibrate's per-evaluation JSONL into one SQLite database with stage/candidate/case/iteration provenance. Idempotent: re-ingesting a file replaces its rows. Rasters are never stored -- footprints only, via `calibrate dump`, for final winning vectors.
"""Ingest `calibrate` evaluation logs into one SQLite database.

Every simulation of every stage lands in `evals`, keyed to the global
candidate that produced it and the case it ran on. The full parameter vector
is stored once per candidate in `candidates` rather than repeated on every
row, which is what keeps the database an order of magnitude smaller than the
JSONL it came from.

Only a `<job>.evals.jsonl` with a completed sibling `<job>.json` is read: an
unmatched log is a job that died partway, and its rows are not data. Ingest is
idempotent -- re-running deletes and reloads each source file's rows -- so it
is safe to run repeatedly while a stage is still going.

    python3 python_scripts/ingest_evals.py \\
        --results-dir ~/data/results --db ~/data/calibration.sqlite

    python3 python_scripts/ingest_evals.py --db ... --summary
"""
import argparse
import json
import sqlite3
import sys
import subprocess
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source_id    INTEGER PRIMARY KEY,
  path         TEXT UNIQUE NOT NULL,   -- job log, relative to the results root
  stage        TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  n_rows       INTEGER NOT NULL,
  git_hash     TEXT,
  host         TEXT,
  mtime_utc    TEXT,
  ingested_utc TEXT NOT NULL
);

-- One row per distinct global candidate: the knobs the experiment is choosing
-- between. The per-case nuisance parameters (mu, slab) live on the eval rows,
-- because they vary within a candidate by construction.
CREATE TABLE IF NOT EXISTS candidates (
  candidate_id            TEXT NOT NULL,
  stage                   TEXT NOT NULL,
  friction_model          INTEGER,
  flags                   INTEGER,
  drag_coefficient        REAL,
  internal_friction_angle REAL,
  roughness_threshold     REAL,
  cfl                     REAL,
  released_particles_per_cell INTEGER,
  max_steps               INTEGER,
  release_band_frac       REAL,
  flow_threshold          REAL,
  slab_mode               INTEGER,
  density                 REAL,
  min_residence           REAL,
  require_release_connected INTEGER,
  clip_release_to_drainage  INTEGER,
  params_json             TEXT NOT NULL,
  PRIMARY KEY (stage, candidate_id)
);

CREATE TABLE IF NOT EXISTS evals (
  eval_id      INTEGER PRIMARY KEY,
  source_id    INTEGER NOT NULL REFERENCES sources(source_id),
  stage        TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  case_name    TEXT NOT NULL,
  iter         INTEGER NOT NULL,      -- inner-loop index within (job, case)
  ok           INTEGER NOT NULL,
  err          TEXT,
  -- the nuisance parameters the inner loop moves
  mu           REAL,
  slab         REAL,
  -- scores
  omega        REAL,
  omega_cells  REAL,
  hwri_l1      REAL,
  hwri_l05     REAL,
  release_only_omega REAL,
  alpha        REAL,
  beta         REAL,
  gamma        REAL,
  -- diagnostics that decide whether a score is trustworthy
  reach_err_m  REAL,
  sim_cells    INTEGER,
  ref_cells    INTEGER,
  release_cells INTEGER,
  steps        INTEGER,
  max_velocity REAL,
  clipped_at_edge INTEGER,
  sim_flags    INTEGER,
  seconds      REAL,
  -- what the fixed scoring conventions removed; the before/after evidence for
  -- the tail and ridge-leak fixes, recorded per evaluation
  sim_reach_raw_m REAL,
  sim_cells_raw   INTEGER,
  omega_raw       REAL,
  release_clipped_frac REAL,
  release_clip_severe INTEGER,
  connect_fallback INTEGER
);

CREATE INDEX IF NOT EXISTS idx_evals_stage_cand ON evals(stage, candidate_id);
CREATE INDEX IF NOT EXISTS idx_evals_case ON evals(stage, case_name);
CREATE INDEX IF NOT EXISTS idx_evals_source ON evals(source_id);
"""

# Columns lifted out of params_json onto the candidates row for querying.
CAND_COLS = [
    "friction_model", "flags", "drag_coefficient", "internal_friction_angle",
    "roughness_threshold", "cfl", "released_particles_per_cell", "max_steps",
    "release_band_frac", "flow_threshold", "slab_mode", "density",
    # scoring conventions -- fixed across a campaign, but recorded so that a
    # database spanning a convention change can still be sliced correctly
    "min_residence", "require_release_connected", "clip_release_to_drainage",
]

EVAL_METRIC_COLS = [
    "omega", "omega_cells", "hwri_l1", "hwri_l05", "release_only_omega",
    "alpha", "beta", "gamma", "reach_err_m", "sim_cells", "ref_cells",
    "release_cells", "steps", "max_velocity", "clipped_at_edge", "sim_flags",
    "seconds", "sim_reach_raw_m", "sim_cells_raw", "omega_raw", "release_clipped_frac",
    "release_clip_severe", "connect_fallback",
]


def git_hash(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    # WAL keeps the database readable while a stage is still writing into it.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def find_logs(results_dir: Path) -> list[Path]:
    """Job logs whose sibling result JSON exists. The result JSON is the
    driver's done-signal; a log without one belongs to a job that died.

    Two naming conventions are in the wild and both must be accepted:
    `run_stage.py` renames to `<tag>.evals.jsonl` beside `<tag>.json`, while a
    bare `calibrate --out X.json` defaults its log to `X.json.evals.jsonl`
    beside `X.json`. Matching only the first silently dropped every one-shot
    job's evaluations -- with no error, because an unmatched log is
    indistinguishable from a crashed one. Unmatched logs are now reported.
    """
    out, orphans = [], []
    for log in sorted(results_dir.rglob("*.evals.jsonl")):
        if log.name.startswith("."):
            continue  # in-flight temp file
        stem = log.name[: -len(".evals.jsonl")]
        # `<tag>.evals.jsonl` -> `<tag>.json`; `X.json.evals.jsonl` -> `X.json`
        if log.with_name(stem + ".json").exists() or log.with_name(stem).exists():
            out.append(log)
        else:
            orphans.append(log)
    if orphans:
        print(f"!! {len(orphans)} evaluation log(s) have no completed result file and "
              f"were SKIPPED -- a died job, or a naming convention this does not know:",
              file=sys.stderr)
        for o in orphans[:5]:
            print(f"     {o}", file=sys.stderr)
        if len(orphans) > 5:
            print(f"     ... and {len(orphans) - 5} more", file=sys.stderr)
    return out


def ingest_file(con, log: Path, root: Path, ghash: str | None, host: str) -> tuple[int, int]:
    rel = str(log.relative_to(root))
    cur = con.cursor()

    rows, cands = [], {}
    skipped = 0
    with open(log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # The last line of a hard-killed job can be a partial write.
                skipped += 1
                continue
            p = r.get("params", {})
            cid, stage = r.get("candidate", ""), r.get("stage", "")
            if (stage, cid) not in cands:
                cands[(stage, cid)] = p
            m = r.get("m") or {}
            rows.append((
                stage, cid, r.get("case", ""), r.get("iter", 0),
                1 if r.get("ok") else 0, r.get("err"),
                p.get("friction_coefficient"), p.get("slab_thickness"),
                *[m.get(c) for c in EVAL_METRIC_COLS],
            ))

    # Idempotent: a re-ingest of the same job replaces its rows rather than
    # doubling them, so this can run on a live results directory.
    old = cur.execute("SELECT source_id FROM sources WHERE path = ?", (rel,)).fetchone()
    if old:
        cur.execute("DELETE FROM evals WHERE source_id = ?", (old[0],))
        cur.execute("DELETE FROM sources WHERE source_id = ?", (old[0],))

    stage = rows[0][0] if rows else ""
    cid = rows[0][1] if rows else ""
    cur.execute(
        "INSERT INTO sources (path, stage, candidate_id, n_rows, git_hash, host, "
        "mtime_utc, ingested_utc) VALUES (?,?,?,?,?,?,?,?)",
        (rel, stage, cid, len(rows), ghash, host,
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(log.stat().st_mtime)),
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    sid = cur.lastrowid

    for (st, ci), p in cands.items():
        cur.execute(
            f"INSERT OR REPLACE INTO candidates (candidate_id, stage, "
            f"{', '.join(CAND_COLS)}, params_json) "
            f"VALUES (?,?,{','.join('?' * len(CAND_COLS))},?)",
            (ci, st, *[p.get(c) for c in CAND_COLS], json.dumps(p, sort_keys=True)))

    cur.executemany(
        f"INSERT INTO evals (source_id, stage, candidate_id, case_name, iter, ok, err, "
        f"mu, slab, {', '.join(EVAL_METRIC_COLS)}) "
        f"VALUES (?,?,?,?,?,?,?,?,?,{','.join('?' * len(EVAL_METRIC_COLS))})",
        [(sid, *r) for r in rows])
    con.commit()
    return len(rows), skipped


def summarise(con):
    q = con.execute("""
        SELECT stage, COUNT(DISTINCT candidate_id), COUNT(DISTINCT case_name),
               COUNT(*), SUM(ok = 0), AVG(seconds), SUM(seconds)
        FROM evals GROUP BY stage ORDER BY stage""").fetchall()
    print(f"{'stage':>16} {'cands':>6} {'cases':>6} {'evals':>10} {'failed':>7} "
          f"{'s/sim':>7} {'gpu-h':>7}")
    for st, nc, ncase, n, nf, avg, tot in q:
        print(f"{st:>16} {nc:>6} {ncase:>6} {n:>10,} {nf or 0:>7} "
              f"{avg or 0:>7.3f} {(tot or 0) / 3600:>7.2f}")
    tot = con.execute("SELECT COUNT(*) FROM evals").fetchone()[0]
    print(f"\ntotal {tot:,} evaluations")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", help="root written by run_stage.py")
    p.add_argument("--db", required=True)
    p.add_argument("--repo", default=".", help="repo to read the git hash from")
    p.add_argument("--summary", action="store_true", help="print a summary and exit")
    args = p.parse_args()

    con = connect(args.db)
    if args.summary and not args.results_dir:
        summarise(con)
        return

    root = Path(args.results_dir)
    logs = find_logs(root)
    ghash = git_hash(Path(args.repo))
    import socket
    host = socket.gethostname()
    print(f"{len(logs)} completed job logs under {root}")

    total = bad = 0
    for log in logs:
        n, skipped = ingest_file(con, log, root, ghash, host)
        total += n
        bad += skipped
        if skipped:
            print(f"  {log.name}: {n} rows (+{skipped} unparseable, truncated write)")
    print(f"ingested {total:,} evaluations" + (f", skipped {bad} malformed lines" if bad else ""))
    size_mb = Path(args.db).stat().st_size / 1e6
    print(f"database {size_mb:.1f} MB ({size_mb * 1e6 / max(total, 1):.0f} bytes/eval)")
    summarise(con)


if __name__ == "__main__":
    main()
