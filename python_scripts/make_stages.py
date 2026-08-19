#!/usr/bin/env python3
# @atlas: Generates the stage manifests for the global-constants calibration (campaign/GLOBAL_CALIBRATION_PLAN.md). One JSON per stage: panel, candidate list, inner-loop args. Consumed by run_stage.py.
"""Build the stage manifests for the global-constants calibration experiment.

Each manifest is a self-contained description of one stage: which cases form
the panel, which global candidates to evaluate, and what inner-loop (per-case
mu/slab) calibration to run at each candidate. `run_stage.py` executes one;
`ingest_evals.py` loads the results; `analyze_stage.py` decides the winner.

The design and the reasoning behind each stage are in
`campaign/GLOBAL_CALIBRATION_PLAN.md`. Read that before changing numbers here.

Usage:

    python3 python_scripts/make_stages.py \\
        --cases ~/data/cases100.json --out-dir ~/data/stages

Stages 2-4 depend on the winner of the stage before, so they are written with
placeholder candidate arguments and regenerated once that winner is known:

    python3 python_scripts/make_stages.py --cases ... --out-dir ... \\
        --stage s2_xi --base-args "--model 1 --flags 7"
"""
import argparse
import json
import time
from pathlib import Path

# Cases whose DEM is fabricated (swissALTI3D edge-replication or -9999
# sentinels) and overlaps a footprint by >=1% -- see the campaign's working record, "Domain
# and data quality". Both are valid floats so the harness NaN check misses
# them; they have to be excluded by name.
CONTAMINATED = ["aval_4117", "aval_7743", "aval_847"]

# Inner-loop start point: the panel medians of the fixed-xi per-event optima
# (campaign/analysis/perevent_fixedxi.json). A single shared start for every
# candidate keeps the comparison paired -- every candidate gets the identical
# initial simplex -- without warm-starting from any one candidate's own fit.
INNER_START_MU = 0.36
INNER_START_SLAB = 0.56

# 40 evaluations is where the fixed-xi (mu, slab) search was measured to sit:
# mean 34.2 evals, 81/105 terminating by simplex collapse rather than budget.
INNER_BUDGET = 40

# Stage 4 start points, spread across the (mu, slab) box rather than clustered
# near the median, so that a candidate whose basin is awkward from one start is
# not quietly penalised for it.
MULTISTARTS = [(0.36, 0.56), (0.20, 0.30), (0.50, 1.00)]

# The incumbent to beat: Voellmy, curvature + particle interaction + earth
# pressure, no entrainment, at the xi the +0.2864 per-event run used.
INCUMBENT_MODEL = 1
INCUMBENT_FLAGS = 0b0111
INCUMBENT_XI = 754.0

FRICTION_MODELS = {0: "coulomb", 1: "voellmy", 2: "voellmyminshear", 3: "samosat"}


def load_panel(cases_path: str, exclude: list[str]) -> list[str]:
    cases = json.loads(Path(cases_path).read_text())
    names = [c["name"] for c in cases]
    drop = set(exclude)
    panel = [n for n in names if n not in drop]
    missing = drop - set(names)
    if missing:
        print(f"  note: exclusion list mentions {sorted(missing)}, absent from {cases_path}")
    return panel


def merge_args(base: list[str], override: list[str]) -> list[str]:
    """Combine two `--flag value` lists so the override wins, rather than
    concatenating them. clap rejects a repeated single-value flag outright
    ("the argument '--xi <XI>' cannot be used multiple times"), so appending a
    frozen value on top of a screening one produces a command that dies before
    it simulates anything."""
    out = list(base)
    for i in range(0, len(override) - 1, 2):
        flag, val = override[i], override[i + 1]
        if not flag.startswith("--"):
            raise SystemExit(f"expected a --flag at position {i} of {override}, got {flag!r}")
        if flag in out:
            out[out.index(flag) + 1] = val
        else:
            out += [flag, val]
    return out


def inner_args() -> list[str]:
    """Per-case nuisance calibration: mu and slab free, everything else pinned
    by the candidate. This is the 'fix the global knob first, then calibrate
    mu/slab at it' ordering that the campaign's working record measures as costing 3% of the
    achievable per-event gain, against 52% done the other way round."""
    return [
        "--free", "mu,slab",
        "--budget", str(INNER_BUDGET),
        "--mu", str(INNER_START_MU),
        "--slab", str(INNER_START_SLAB),
    ]


def structure_candidates(base_args: list[str] | None = None) -> list[dict]:
    """Tier 1: friction model x optional components.

    `base_args` carries whatever earlier stages froze -- in particular the
    tier-0 numerics. Without it every structure candidate would silently run
    at `Params::default()` numerics rather than the frozen ones, so stage 1
    would compare structures at a configuration the campaign has already
    decided not to use.

    Two exact redundancies are pruned rather than simulated:
      - the earth-pressure flag is read only inside `grid_physics`, which
        early-returns when particle interaction is off, so EP on/off with PI
        off are the same simulation;
      - `internal_friction_angle` likewise only bites with PI and EP both on,
        which is why the Tier-2 ifa stage is conditional on the winner.
    """
    out = []
    for model in sorted(FRICTION_MODELS):
        for curvature in (0, 1):
            for pi in (0, 1):
                for ep in (0, 1):
                    if not pi and ep:
                        continue  # identical simulation to (pi=0, ep=0)
                    for entrain in (0, 1):
                        flags = curvature | (pi << 1) | (ep << 2) | (entrain << 3)
                        cid = (f"m{model}_{FRICTION_MODELS[model]}"
                               f"_c{curvature}p{pi}e{ep}n{entrain}")
                        args = ["--model", str(model), "--flags", str(flags)]
                        # xi is read only by Voellmy (1) and VoellmyMinShear
                        # (2); Coulomb returns zero drag and samosAT uses a
                        # fixed log-profile, so passing it there would imply a
                        # dependence that does not exist.
                        if model in (1, 2):
                            args += ["--xi", str(INCUMBENT_XI)]
                        out.append({
                            "id": cid,
                            "args": merge_args(args, base_args or []),
                            "meta": {"friction_model": model, "flags": flags,
                                     "curvature": curvature, "particle_interaction": pi,
                                     "earth_pressure": ep, "entrainment": entrain},
                        })
    return out


def numerics_candidates() -> list[dict]:
    """Tier 0: cost vs accuracy. cfl and max_steps interact -- halving cfl
    halves dt and so roughly doubles the steps needed to reach the same
    physical time -- so they are varied jointly rather than one at a time.
    `released_particles_per_cell` is a pure convergence knob: particle mass is
    divided by it in `initialize_particles.wgsl` and multiplied back in
    `mass_per_area`, so it changes sampling density and cost, not the physics.
    """
    out = []
    for cfl in (0.9, 0.7, 0.5, 0.35, 0.25):
        for ppc in (2, 4, 8, 16, 32):
            for max_steps in (1500, 3000, 6000):
                cid = f"cfl{cfl}_ppc{ppc}_ms{max_steps}"
                out.append({
                    "id": cid,
                    "args": ["--cfl", str(cfl), "--ppc", str(ppc),
                             "--max-steps", str(max_steps)],
                    "meta": {"cfl": cfl, "ppc": ppc, "max_steps": max_steps,
                             "relative_cost": ppc / 8.0},
                })
    return out


def xi_candidates(base_args: list[str]) -> list[dict]:
    """Tier 2: the xi profile. Each value gets a full per-case (mu, slab)
    recalibration, which is the profile-likelihood form of the ordering result:
    the optimal mu depends on xi, so mu must be refitted at every xi rather
    than carried across."""
    values = [250, 400, 650, 1000, 1600, 2600, 4000, 6500, 10000]
    return [{"id": f"xi{v}", "args": merge_args(base_args, ["--xi", str(v)]),
             "meta": {"drag_coefficient": v}} for v in values]


def ifa_candidates(base_args: list[str]) -> list[dict]:
    """Tier 2: internal friction angle, over the search bounds already in
    calibrate.rs. Only meaningful if the structure winner has particle
    interaction and earth pressure both on."""
    values = [15, 20, 25, 30, 35, 40, 45]
    return [{"id": f"ifa{v}", "args": merge_args(base_args, ["--ifa", str(v)]),
             "meta": {"internal_friction_angle": v}} for v in values]


def roughness_null_candidates(base_args: list[str]) -> list[dict]:
    """A null check, not a refinement. `roughness_threshold` is read only by
    `compute_roughness.wgsl` and `compute_release_areas.wgsl`, and
    `Simulation::get_release_areas` runs neither when release areas arrive as
    an array -- which is what this harness always does. The expectation is
    therefore bit-identical scores across the whole range; anything else means
    the code has been read wrong and the plan needs revisiting."""
    return [{"id": f"rough{v}", "args": merge_args(base_args, ["--roughness", str(v)]),
             "meta": {"roughness_threshold": v}} for v in (0.002, 0.01, 0.05)]


def write_params_from(perevent_json: str, dest: Path) -> int:
    """Stage 0 evaluates numerics at each case's own fitted operating point
    rather than at one shared vector, because a numerics setting that is
    harmless at a well-fitted mu can still be wrong at a badly-fitted one, and
    the well-fitted case is the one the harvest will actually run at. Builds
    the {case: {mu, slab}} map `calibrate apply --params-from` expects."""
    recs = json.loads(Path(perevent_json).read_text())
    table = {r["name"]: {"mu": r["best"]["friction_coefficient"],
                         "slab": r["best"]["slab_thickness"]} for r in recs}
    dest.write_text(json.dumps(table, indent=1))
    return len(table)


def confirm_candidates(finalists: list[str], base_args: list[str]) -> list[dict]:
    """Stage 4: the stage-1 finalists re-run at the frozen xi and ifa. Their
    model/flag arguments are looked up from the structure candidate list rather
    than parsed back out of the id, so a renamed id fails loudly instead of
    silently producing the wrong flags."""
    by_id = {c["id"]: c for c in structure_candidates()}  # ids only; base merged below
    unknown = [f for f in finalists if f not in by_id]
    if unknown:
        raise SystemExit(f"unknown finalist id(s): {unknown}\n"
                         f"expected ids like {sorted(by_id)[0]}")
    out = []
    for f in finalists:
        c = dict(by_id[f])
        # Frozen values come last so they override the screening xi carried in
        # the structure candidate's own args.
        c["args"] = merge_args(c["args"], base_args)
        out.append(c)
    return out


def manifest(stage, panel, candidates, subcommand, sub_args, block_size, notes=""):
    return {
        "stage": stage,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": notes,
        "panel": panel,
        "n_panel": len(panel),
        "block_size": block_size,
        "subcommand": subcommand,
        "global_args": [],
        "sub_args": sub_args,
        "candidates": candidates,
        "n_candidates": len(candidates),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", required=True, help="path to the case-list JSON")
    p.add_argument("--out-dir", required=True, help="directory to write manifests into")
    p.add_argument("--stage", default="all",
                   help="which stage to (re)generate: all, s0_numerics, s1_structure, "
                        "s2_xi, s3_ifa, s3_roughness, s4_confirm")
    p.add_argument("--base-args", default="",
                   help="candidate args fixed by earlier stages, e.g. '--model 1 --flags 7'")
    p.add_argument("--params-from", default="perevent_fixedxi_params.json",
                   help="stage 0 only: per-case {mu, slab} to evaluate numerics at; "
                        "written into --out-dir from --perevent-json")
    p.add_argument("--perevent-json", default="campaign/analysis/perevent_fixedxi.json",
                   help="the fixed-xi per-event run the stage 0 operating points come from")
    p.add_argument("--finalists", default="",
                   help="s4_confirm only: comma-separated stage-1 candidate ids to confirm")
    p.add_argument("--block-size", type=int, default=8,
                   help="cases per worker process; larger amortises GPU init, "
                        "smaller balances load better")
    p.add_argument("--exclude", default=",".join(CONTAMINATED),
                   help="comma-separated case names to drop from the panel")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = [s for s in args.exclude.split(",") if s]
    panel = load_panel(args.cases, exclude)
    print(f"panel: {len(panel)} cases (excluded {len(exclude)}: {', '.join(exclude)})")

    base = args.base_args.split() if args.base_args else []
    want = args.stage
    built = []

    if want in ("all", "s0_numerics"):
        pf = out_dir / Path(args.params_from).name
        try:
            n = write_params_from(args.perevent_json, pf)
            print(f"  wrote {n} per-case operating points -> {pf}")
        except FileNotFoundError:
            print(f"  !! {args.perevent_json} not found; stage 0 needs it for --params-from")
        m = manifest(
            "s0_numerics", panel, numerics_candidates(), "apply",
            ["--params-from", str(pf), "--xi", str(INCUMBENT_XI),
             "--model", str(INCUMBENT_MODEL), "--flags", str(INCUMBENT_FLAGS)],
            args.block_size,
            notes="Scored at each case's own fixed-xi optimum, so no inner loop: "
                  "numerics is being asked whether it changes the score at the "
                  "operating point we will actually use, not whether it moves the "
                  "optimum. One simulation per (case, candidate).")
        built.append(m)

    if want in ("all", "s1_structure"):
        m = manifest(
            "s1_structure", panel, structure_candidates(base), "per-event",
            inner_args(), args.block_size,
            notes="Compare on omega_cells as well as omega: the particle-interaction "
                  "flag changes which footprint rule evaluate_case applies, so the "
                  "primary metric is not comparable across it.")
        built.append(m)

    if want in ("all", "s2_xi"):
        m = manifest("s2_xi", panel, xi_candidates(base or ["--model", "1", "--flags", "7"]),
                     "per-event", inner_args(), args.block_size,
                     notes="Only meaningful if the structure winner is Voellmy (1) or "
                           "VoellmyMinShear (2). Regenerate with --base-args once "
                           "stage 1 has decided.")
        built.append(m)

    if want in ("all", "s3_ifa"):
        m = manifest("s3_ifa", panel, ifa_candidates(base or ["--model", "1", "--flags", "7"]),
                     "per-event", inner_args(), args.block_size,
                     notes="Only meaningful if the structure winner has both particle "
                           "interaction and earth pressure on.")
        built.append(m)

    if want in ("all", "s3_roughness"):
        m = manifest("s3_roughness", panel,
                     roughness_null_candidates(base or ["--model", "1", "--flags", "7"]),
                     "run", [], args.block_size,
                     notes="Null check: expect bit-identical omega across all three "
                           "values, because this harness never runs the shaders that "
                           "read roughness_threshold.")
        built.append(m)

    if want in ("all", "s4_confirm"):
        finalists = [s for s in args.finalists.split(",") if s]
        if not finalists:
            if want == "s4_confirm":
                raise SystemExit("s4_confirm needs --finalists, e.g. "
                                 "--finalists m1_voellmy_c1p1e1n0,m1_voellmy_c0p1e1n0")
            print("  (skipping s4_confirm: needs --finalists from stage 1)")
        else:
            m = manifest("s4_confirm", panel, confirm_candidates(finalists, base),
                         "per-event", inner_args(), args.block_size,
                         notes="Run three times with --run-tag start1/2/3 and a different "
                               "--inner-start each time; the analysis pools starts because "
                               "it takes the best inner-loop score per (candidate, case). "
                               f"Suggested starts: {MULTISTARTS}")
            built.append(m)

    for m in built:
        path = out_dir / f"{m['stage']}.json"
        path.write_text(json.dumps(m, indent=1))
        sims = m["n_candidates"] * m["n_panel"]
        if m["subcommand"] == "per-event":
            sims *= INNER_BUDGET
        print(f"  {m['stage']:>14}: {m['n_candidates']:>3} candidates x {m['n_panel']} cases"
              f"  ~{sims:,} sims  -> {path}")


if __name__ == "__main__":
    main()
