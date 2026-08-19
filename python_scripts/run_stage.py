#!/usr/bin/env python3
# @atlas: Resumable driver for one stage of the global-constants calibration. Expands a manifest from make_stages.py into (candidate x case-block) jobs, runs them across (GPUs x per-GPU) workers, atomic per-job writes so an interrupt loses at most one block.
"""Run one stage of the global-constants calibration experiment.

Reads a manifest written by `make_stages.py`, expands it into one job per
(global candidate x block of panel cases), and runs those jobs across
(gpus x per_gpu) worker slots. Each job is a single `calibrate` invocation
carrying `--stage`/`--candidate` provenance, writing its result JSON and its
per-evaluation JSONL to temporary paths that are renamed on success -- so a
crashed worker, an OOM or a hard interrupt loses at most the one block that
was in flight, and re-running the identical command resumes.

The done-signal is the final `<job>.json` existing. `ingest_evals.py` only
reads a `<job>.evals.jsonl` that has such a sibling, so a partial log from a
killed job is never mistaken for data.

    python3 python_scripts/run_stage.py \\
        --manifest ~/data/stages/s1_structure.json \\
        --calibrate-bin ~/avalanchers/target/release/calibrate \\
        --cases ~/data/cases100.json --cache ~/data/dtm_cache.zarr \\
        --cwd ~/data --out-dir ~/data/results --gpus 8 --per-gpu 8

Add `--dry-run` to print the job list, the simulation count and the wall-clock
estimate without running anything.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Throughput used only for the ETA and the dry-run estimate, never for
# scheduling. Measured aggregate across the whole box, in simulations/second.
# The 2026-07-27 RTX 5090 figures were 23.4 cases/s/GPU on the `run` path and
# ~34.6 evals/s/GPU on the `per-event` path (process startup and case
# preparation amortise over the inner loop there). A 4070 Super is slower;
# override with --throughput once the box has been benchmarked.
DEFAULT_THROUGHPUT = 90.0


def load_manifest(path: str) -> dict:
    m = json.loads(Path(path).read_text())
    for k in ("stage", "panel", "candidates", "subcommand"):
        if k not in m:
            raise SystemExit(f"manifest {path} is missing required key '{k}'")
    return m


def blocks(panel: list[str], size: int) -> list[list[str]]:
    return [panel[i:i + size] for i in range(0, len(panel), size)]


def sims_per_job(m: dict, block: list[str]) -> int:
    """Simulations one job will run. `per-event` spends the inner budget on
    every case; `run`/`apply` evaluate each case once. An over-estimate for
    per-event, since the median search terminates by simplex collapse well
    before the budget."""
    budget = 1
    if m["subcommand"] == "per-event":
        sub = m.get("sub_args", [])
        budget = int(sub[sub.index("--budget") + 1]) if "--budget" in sub else 40
    return len(block) * budget


def apply_inner_start(sub_args: list[str], inner_start: str) -> list[str]:
    """Override the inner loop's start point without editing the manifest, so
    a multi-start confirmation run is three invocations of one manifest rather
    than three manifests that could drift apart."""
    if not inner_start:
        return sub_args
    try:
        mu, slab = (float(v) for v in inner_start.split(","))
    except ValueError:
        raise SystemExit(f"--inner-start wants 'mu,slab', got {inner_start!r}")
    out = list(sub_args)
    for flag, val in (("--mu", mu), ("--slab", slab)):
        if flag in out:
            out[out.index(flag) + 1] = str(val)
        else:
            out += [flag, str(val)]
    return out


def build_cmd(m, cand, block, bi, args, tmp_out, tmp_log) -> list[str]:
    cmd = [
        args.calibrate_bin,
        "--cases", args.cases,
        "--cache", args.cache,
        "--only", ",".join(block),
        "--padding", str(args.padding),
        "--stage", m["stage"],
        "--candidate", cand["id"],
        "--out", str(tmp_out),
        "--eval-log", str(tmp_log),
        *m.get("global_args", []),
    ]
    if args.gpus > 0:
        cmd += ["--gpu-index", str(bi % args.gpus)]
    sub = apply_inner_start(m.get("sub_args", []), args.inner_start)
    cmd += [m["subcommand"], *sub, *cand.get("args", [])]
    return cmd


def job_tag(cand_id: str, block_index: int, run_tag: str) -> str:
    """Job filenames carry the run tag, candidate provenance does not. Running
    a stage again under a second tag with a different inner-loop start point
    therefore produces new job files whose rows still belong to the same
    candidate -- and since the analysis takes the best inner-loop score per
    (candidate, case), that is exactly a multi-start search."""
    suffix = f"__{run_tag}" if run_tag else ""
    return f"{cand_id}{suffix}__b{block_index:03d}"


def run_one(job, args, m, out_dir, log_dir):
    cand, block, bi = job["cand"], job["block"], job["block_index"]
    tag = job_tag(cand["id"], bi, args.run_tag)
    final = out_dir / f"{tag}.json"
    if final.exists():
        return tag, "skipped", 0.0, 0

    tmp_out = out_dir / f".{tag}.json.tmp.{os.getpid()}"
    tmp_log = out_dir / f".{tag}.evals.jsonl.tmp.{os.getpid()}"
    cmd = build_cmd(m, cand, block, bi, args, tmp_out, tmp_log)

    t0 = time.time()
    with open(log_dir / f"{tag}.log", "w") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=args.cwd)
    dt = time.time() - t0

    if proc.returncode != 0 or not tmp_out.exists():
        for p in (tmp_out, tmp_log):
            if p.exists():
                p.unlink()
        return tag, "failed", dt, 0

    n_rows = 0
    if tmp_log.exists():
        with open(tmp_log) as f:
            n_rows = sum(1 for _ in f)
        tmp_log.rename(out_dir / f"{tag}.evals.jsonl")
    # Renamed last: the result JSON is the done-signal, so it must not appear
    # before the log it vouches for.
    tmp_out.rename(final)
    return tag, "ok", dt, n_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--calibrate-bin", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out-dir", required=True,
                   help="results root; a subdirectory per stage is created inside it")
    p.add_argument("--cwd", default=".",
                   help="working directory for calibrate (case `shp` paths are relative to it)")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--per-gpu", type=int, default=8,
                   help="concurrent processes per GPU; 8 was the saturation point on an "
                        "RTX 5090, re-measure on other hardware")
    p.add_argument("--padding", type=float, default=300.0)
    p.add_argument("--throughput", type=float, default=DEFAULT_THROUGHPUT,
                   help="aggregate simulations/second across the box, for ETA only")
    p.add_argument("--only-candidates", default=None,
                   help="comma-separated candidate ids, for reruns and spot checks")
    p.add_argument("--run-tag", default="",
                   help="suffix for job filenames, leaving candidate ids untouched; "
                        "re-run a stage under a second tag with a different --mu/--slab "
                        "in the manifest's sub_args to get a multi-start inner loop")
    p.add_argument("--inner-start", default="",
                   help="'mu,slab' overriding the manifest's inner-loop start point; "
                        "pair with --run-tag for a multi-start confirmation run")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the first command, run nothing")
    args = p.parse_args()

    m = load_manifest(args.manifest)
    out_dir = Path(args.out_dir) / m["stage"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    cands = m["candidates"]
    if args.only_candidates:
        want = set(args.only_candidates.split(","))
        cands = [c for c in cands if c["id"] in want]
    bl = blocks(m["panel"], m.get("block_size", 8))

    jobs = [{"cand": c, "block": b, "block_index": i}
            for c in cands for i, b in enumerate(bl)]
    todo = [j for j in jobs
            if not (out_dir / f"{job_tag(j['cand']['id'], j['block_index'], args.run_tag)}.json"
                    ).exists()]
    total_sims = sum(sims_per_job(m, j["block"]) for j in todo)
    eta = total_sims / args.throughput if args.throughput > 0 else float("nan")

    print(f"stage {m['stage']}: {len(cands)} candidates x {len(bl)} blocks "
          f"({m['n_panel'] if 'n_panel' in m else len(m['panel'])} cases) = {len(jobs)} jobs",
          file=sys.stderr)
    print(f"  {len(jobs) - len(todo)} already done, {len(todo)} to run", file=sys.stderr)
    print(f"  ~{total_sims:,} simulations; at {args.throughput:.0f} sims/s that is "
          f"{eta / 60:.1f} min", file=sys.stderr)
    if m.get("notes"):
        print(f"  note: {m['notes']}", file=sys.stderr)

    if args.dry_run:
        if todo:
            j = todo[0]
            tag = job_tag(j["cand"]["id"], j["block_index"], args.run_tag)
            cmd = build_cmd(m, j["cand"], j["block"], j["block_index"], args,
                            out_dir / f".{tag}.json.tmp", out_dir / f".{tag}.evals.jsonl.tmp")
            print("\nfirst command would be:\n  " + " ".join(cmd), file=sys.stderr)
        return
    if not todo:
        return

    workers = max(1, args.gpus * args.per_gpu)
    t_start = time.time()
    done = fail = rows = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, j, args, m, out_dir, log_dir): j for j in todo}
        for fut in as_completed(futures):
            tag, status, dt, n = fut.result()
            if status == "ok":
                done += 1
                rows += n
                print(f"[{done + fail}/{len(todo)}] {tag}: ok ({dt:.1f}s, {n} evals)",
                      file=sys.stderr)
            elif status == "failed":
                fail += 1
                with open(out_dir / "failures.jsonl", "a") as f:
                    f.write(json.dumps({"job": tag, "time": time.time()}) + "\n")
                print(f"[{done + fail}/{len(todo)}] {tag}: FAILED, see {log_dir}/{tag}.log",
                      file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"stage {m['stage']}: {done} ok, {fail} failed, {rows:,} evaluations logged, "
          f"{elapsed:.0f}s ({rows / elapsed if elapsed else 0:.1f} sims/s measured)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
