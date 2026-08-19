#!/usr/bin/env python3
# @atlas: Resumable batch runner: splits a case list across (GPUs × per-GPU) workers, one `calibrate run` subprocess per case, atomic per-case writes so an interrupt loses at most one case. `--per-gpu 8` is the measured saturation point.
"""Resumable, GPU-sharded batch runner for `calibrate run`.

Splits a case list across (gpus x per_gpu) worker slots and runs one
`calibrate --only <case> --gpu-index <i> run` subprocess per case. Each
case's result is written atomically (temp file + rename), so a crashed
worker or an interrupted harness invocation loses at most the one case
that was in flight -- re-running the same command skips every case whose
result file already exists.

This is pure orchestration around the existing `calibrate run` CLI; it
does not touch the solver.

Example (single GPU, 8-way concurrency -- the measured saturation point
on an RTX 5090, see notes_for_markus.md):

    python3 python_scripts/shard_calibrate.py \\
        --calibrate-bin ~/avalanchers/target/release/calibrate \\
        --cases ~/data/cases100.json --cache ~/data/dtm_cache.zarr \\
        --cwd ~/data --out-dir ~/data/results \\
        --gpus 1 --per-gpu 8

Re-run the identical command to resume after a crash/interrupt/OOM; only
cases still missing a `<out-dir>/<name>.json` get (re)submitted.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_case_names(cases_path: str, only: str | None, split: str | None) -> list[str]:
    cases = json.loads(Path(cases_path).read_text())
    if split:
        cases = [c for c in cases if c.get("split") == split]
    names = [c["name"] for c in cases]
    if only:
        want = set(only.split(","))
        names = [n for n in names if n in want]
    return names


def run_one(case: str, args, gpu_index: int, out_dir: Path, log_dir: Path):
    final = out_dir / f"{case}.json"
    if final.exists():
        return case, "skipped", 0.0

    # unique per-attempt tmp name: two workers should never collide, but a
    # stale tmp file from a killed worker must never be mistaken for a
    # finished result (we only ever read from `final`, so this is just
    # hygiene).
    tmp = out_dir / f".{case}.json.tmp.{os.getpid()}"
    log_path = log_dir / f"{case}.log"
    cmd = [
        args.calibrate_bin,
        "--cases", args.cases,
        "--cache", args.cache,
        "--only", case,
        "--padding", str(args.padding),
        "--gpu-index", str(gpu_index),
        "--out", str(tmp),
        "run",
        *args.extra_run_args,
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=args.cwd)
    dt = time.time() - t0

    if proc.returncode != 0 or not tmp.exists():
        if tmp.exists():
            tmp.unlink()
        return case, "failed", dt

    tmp.rename(final)  # atomic on the same filesystem -> the only "done" signal
    return case, "ok", dt


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibrate-bin", required=True)
    p.add_argument("--cases", required=True, help="path to the case-list JSON")
    p.add_argument("--cache", required=True, help="swissALTI3D zarr cache dir")
    p.add_argument("--out-dir", required=True, help="one <case>.json per finished case")
    p.add_argument(
        "--cwd", default=".",
        help="working directory to run calibrate from (case `shp` paths are relative to it)",
    )
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument(
        "--per-gpu", type=int, default=8,
        help="concurrent processes per GPU; 8 is the measured saturation point on a single "
        "RTX 5090 (23 cases/s aggregate -- 16/GPU bought +0.8%% over 8, not worth it)",
    )
    p.add_argument("--padding", type=float, default=300.0)
    p.add_argument("--only", default=None, help="comma-separated case names")
    p.add_argument("--split", default=None, help="restrict to 'cal' or 'val'")
    p.add_argument(
        "--extra-run-args", nargs=argparse.REMAINDER, default=[],
        help="everything after this flag is appended to `calibrate ... run`, e.g. "
        "--extra-run-args --mu 0.2 --xi 3500",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    fail_log = out_dir / "failures.jsonl"

    names = load_case_names(args.cases, args.only, args.split)
    todo = [n for n in names if not (out_dir / f"{n}.json").exists()]
    print(
        f"{len(names)} cases total, {len(names) - len(todo)} already done, "
        f"{len(todo)} to run",
        file=sys.stderr,
    )
    if not todo:
        return

    workers = max(1, args.gpus * args.per_gpu)
    t_start = time.time()
    done = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, case, args, i % args.gpus, out_dir, log_dir): case
            for i, case in enumerate(todo)
        }
        for fut in as_completed(futures):
            case, status, dt = fut.result()
            if status == "ok":
                done += 1
                print(f"[{done + fail}/{len(todo)}] {case}: ok ({dt:.1f}s)", file=sys.stderr)
            else:
                fail += 1
                with open(fail_log, "a") as f:
                    f.write(json.dumps({"case": case, "time": time.time()}) + "\n")
                print(
                    f"[{done + fail}/{len(todo)}] {case}: FAILED, see {log_dir}/{case}.log",
                    file=sys.stderr,
                )

    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 0.0
    print(f"done: {done} ok, {fail} failed, {elapsed:.1f}s ({rate:.2f} cases/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
