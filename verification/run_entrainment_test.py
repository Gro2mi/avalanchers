"""Entrainment comparison with AvaFrame com1DFA on the avaHockeyChannel test.

AvaFrame's standard entrainment test (benchmarks/avaHockeyChannelEntTest):
a 1 m release (release1HS) on the hockey-stick channel runs over an erodible
snow cover (entrainment1HS, about 2.5 ha across the channel). com1DFA erodes it
with the erosion-energy law q = tau_b |u| / e_b (entEroEnergy = 5000 J/kg,
rhoEnt = 100 kg/m^3) under samosAT friction (mu 0.155 for a release above
60 000 m^3).

Two scenarios:
  - cover-0.3m: the benchmark cover, 0.3 m from the shapefile. The flow strips
    it almost completely where it passes, so the entrained mass is mostly set
    by the supply.
  - cover-3m: a 3 m cover the flow cannot strip, so the erosion-energy rate
    itself sets the entrained mass.
Both codes run on the same inputs: avalanchers reads the DEM and the release
shapefile directly (com1DFA seeds its particles from the polygon; the
cell-centre release raster com1DFA exports holds 5 % more area) and the
entrainment shapefile rasterized on the DEM grid. Compared are
  - the entrained mass and its growth over time (mass_<sim>.txt of com1DFA,
    the particle mass of avalanchers),
  - the effect of entrainment on runout and on the peak fields (ent - null),
    which separates the entrainment law from the differences of the two flow
    models (SPH vs depth-averaged MPM) that the null runs already show,
  - the entrained depth map of avalanchers against the erodible area.
avalanchers additionally runs its excess-shear-stress and velocity-
proportional laws for reference (no com1DFA counterpart).

com1DFA needs an AvaFrame environment (numpy < 2, compiled Cython); give its
interpreter with --avaframe-python and the repository with --avaframe. The
com1DFA results are cached in verification/outputs/entrainment/com1DFA* and
only recomputed with --rerun-com1dfa. Without an AvaFrame interpreter and
cached runs, the cover-0.3m scenario falls back to the reference results of
the AvaFrame benchmark (ent run only).

    python verification/run_entrainment_test.py \
        --avaframe C:/git/AvaFrame --avaframe-python <venv>/Scripts/python.exe

Writes report.md, results.json and figures to verification/outputs/entrainment.
"""

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "verification" / "outputs" / "entrainment"
CASE = "avaHockeyChannel"
RELEASE = "release1HS"
BENCHMARK = "avaHockeyChannelEntTest"
SIM_INFO_STOPPED = 1 << 31
BATCH_STEPS = 50
# peak flow thickness above which a cell counts as reached (AvaFrame AIMEC
# uses 1 m for pft runout; 0.1 m keeps the thin entrainment tail visible)
RUNOUT_THRESHOLD = 0.1

# com1DFA defaults of the comparison (com1DFACfg.ini)
SETTINGS = {
    "sim_model": "dampm-global-cartesian",
    "friction_model": "samos-at",
    "friction_coefficient": 0.155,
    "density": 200.0,
    "enable_entrainment": False,
    "entrainment_model": "erosion-energy",
    "erosion_energy": 5000.0,
    "entrainment_density": 100.0,
    "max_steps": 200000,
}

# tolerance of the entrained mass against com1DFA: the two codes resolve the
# flow differently (SPH particles vs MPM grid), so the comparison checks the
# order of magnitude and the trend, not a match to the percent
ENTRAINED_MASS_TOLERANCE = 0.5

SCENARIOS = {
    "cover-0.3m": {"thickness": None, "sim_types": "null|ent", "dir": "com1DFA"},
    "cover-3m": {"thickness": 3.0, "sim_types": "ent", "dir": "com1DFA_thick3m"},
}


# ---------------------------------------------------------------- ESRI grids
def read_asc(path):
    """ESRI ASCII grid as (data north-up, header dict)."""
    header = {}
    with open(path, encoding="utf8") as f:
        for _ in range(6):
            key, value = f.readline().split()
            header[key.lower()] = float(value)
        data = np.loadtxt(f, dtype=np.float64)
    nodata = header.get("nodata_value")
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, header


def write_asc(path, data, header):
    with open(path, "w", encoding="utf8") as f:
        for key in ("ncols", "nrows"):
            f.write(f"{key} {int(header[key])}\n")
        for key in ("xllcenter", "yllcenter", "xllcorner", "yllcorner"):
            if key in header:
                f.write(f"{key} {header[key]}\n")
        f.write(f"cellsize {header['cellsize']}\nnodata_value -9999\n")
        np.savetxt(f, data, fmt="%.4f")


def x_coordinates(header):
    x0 = header.get("xllcenter", header.get("xllcorner", 0.0) + header["cellsize"] / 2)
    return x0 + header["cellsize"] * np.arange(int(header["ncols"]))


def runout_x(peak, header, threshold=RUNOUT_THRESHOLD):
    """Largest x of a cell whose peak value exceeds the threshold."""
    columns = np.where((np.nan_to_num(peak) > threshold).any(axis=0))[0]
    return float(x_coordinates(header)[columns.max()]) if columns.size else float("nan")


# ------------------------------------------------------------------- com1DFA
def com1dfa_dir(scenario):
    return OUT_DIR / SCENARIOS[scenario]["dir"] / CASE / "Outputs"


def run_com1dfa(avaframe, python, scenario):
    config = SCENARIOS[scenario]
    runner = REPO_ROOT / "verification" / "entrainment_com1dfa_runner.py"
    command = [python, str(runner), "--avaframe", str(avaframe), "--case", CASE,
               "--release", RELEASE, "--out", str(OUT_DIR / config["dir"]),
               "--sim-types", config["sim_types"]]
    if config["thickness"] is not None:
        command += ["--entrainment-thickness", str(config["thickness"])]
    print("running com1DFA:", " ".join(command))
    log = OUT_DIR / (config["dir"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf8") as f:
        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)


def read_mass(path):
    """com1DFA mass file: time, total mass, entrained mass per step."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {"time": data[:, 0], "mass": data[:, 1]}


def load_com1dfa(avaframe, scenario):
    """com1DFA results of a scenario per simulation type ('null', 'ent')."""
    results = {}
    out = com1dfa_dir(scenario)
    if out.exists():
        for sim_type in ("null", "ent"):
            masses = sorted((out / "com1DFA").glob(f"mass_{RELEASE}_*_{sim_type}_*.txt"))
            if not masses:
                continue
            name = masses[-1].stem[len("mass_"):]
            peaks = out / "com1DFA" / "peakFiles"
            results[sim_type] = {
                "name": name,
                **read_mass(masses[-1]),
                "pft": read_asc(peaks / f"{name}_pft.asc"),
                "pfv": read_asc(peaks / f"{name}_pfv.asc"),
                "source": "com1DFA run",
            }
    if "ent" not in results and SCENARIOS[scenario]["thickness"] is None:
        bench = pathlib.Path(avaframe) / "benchmarks" / BENCHMARK
        name = json.loads((bench / f"{BENCHMARK}_desDict.json").read_text())["simNameRef"]
        print(f"no com1DFA run found, using the benchmark {BENCHMARK} ({name})")
        results["ent"] = {
            "name": name,
            **read_mass(bench / f"mass_{name}.txt"),
            "pft": read_asc(bench / f"{name}_pft.asc"),
            "pfv": read_asc(bench / f"{name}_pfv.asc"),
            "source": f"AvaFrame benchmark {BENCHMARK}",
        }
    return results


def input_files(avaframe):
    """DEM and release shapefile of the comparison."""
    inputs = pathlib.Path(avaframe) / "avaframe" / "data" / CASE / "Inputs"
    dem = next(inputs.glob("*.asc"))
    # the release polygon itself: com1DFA seeds its particles from it, while
    # its exported cell-centre releaseRaster.asc covers 5 % more area
    return dem, inputs / "REL" / f"{RELEASE}.shp"


def rasterize_entrainment(avaframe, dem_path, thickness=None):
    """Entrainment shapefile on the DEM grid (cell centres inside the polygon)
    with the shapefile thickness, or `thickness` everywhere. com1DFA's own
    exported entrainmentRaster.asc is overwritten by its null run (all zero)."""
    import geopandas
    from rasterio import features, transform

    _, header = read_asc(dem_path)
    shp = pathlib.Path(avaframe) / "avaframe" / "data" / CASE / "Inputs" / "ENT"
    frame = geopandas.read_file(next(shp.glob("*.shp")))
    cell = header["cellsize"]
    x0 = header.get("xllcenter", header.get("xllcorner", 0.0) + cell / 2) - cell / 2
    y0 = header.get("yllcenter", header.get("yllcorner", 0.0) + cell / 2) - cell / 2
    nrows, ncols = int(header["nrows"]), int(header["ncols"])
    affine = transform.from_origin(x0, y0 + nrows * cell, cell, cell)
    if thickness is not None:
        values = [thickness] * len(frame)
    elif "thickness" in frame:
        values = frame["thickness"].fillna(0.3)
    else:
        values = [0.3] * len(frame)
    depth = features.rasterize(zip(frame.geometry, values), out_shape=(nrows, ncols),
                               transform=affine, fill=0.0, dtype="float32")
    suffix = "shp" if thickness is None else f"{thickness:g}m"
    path = OUT_DIR / f"entrainment1HS_{suffix}.asc"
    write_asc(path, depth, header)
    return path


def available_snow_mass(entrainment, dem):
    """Erodible snow mass: bed-normal depths on the inclined bed area."""
    erodible, header = read_asc(entrainment)
    cell = header["cellsize"]
    dz_dy, dz_dx = np.gradient(read_asc(dem)[0], cell)
    bed_area = cell * cell * np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
    return float(np.nansum(erodible * bed_area) * SETTINGS["entrainment_density"])


# --------------------------------------------------------------- avalanchers
def run_avalanchers(name, settings, dem, release, entrainment, out):
    import avalanchers

    settings = dict(settings, dem_path=str(dem), release_areas_path=str(release))
    if settings.get("enable_entrainment"):
        settings["entrainment_area_path"] = str(entrainment)
    sim = avalanchers.PySimulation.new()
    sim.create(settings)
    sim.prepare()
    times, masses = [0.0], [sim.total_mass]
    info = sim.run_n_steps(1)
    last_step = -1
    while not (info.flags & SIM_INFO_STOPPED) and sim.state != "Finished":
        if info.timestep == last_step:
            break
        last_step = info.timestep
        info = sim.run_n_steps(BATCH_STEPS)
        times.append(float(info.elapsed_time))
        masses.append(sim.total_mass)
    out = out / name
    out.mkdir(parents=True, exist_ok=True)
    sim.write_asc("peak_flow_thickness", str(out / "pft.asc"))
    sim.write_asc("peak_velocity", str(out / "pfv.asc"))
    result = {
        "time": np.array(times),
        "mass": np.array(masses, dtype=np.float64),
        "pft": read_asc(out / "pft.asc"),
        "pfv": read_asc(out / "pfv.asc"),
        "steps": int(info.timestep),
    }
    if settings.get("enable_entrainment"):
        sim.write_asc("entrained_depth", str(out / "entrained_depth.asc"))
        result["entrained_depth"] = read_asc(out / "entrained_depth.asc")
    print(f"avalanchers {name:<22} {info.timestep:6d} steps, t = {info.elapsed_time:6.1f} s, "
          f"mass {masses[0]:.4e} -> {masses[-1]:.4e} kg")
    return result


# ---------------------------------------------------------------- evaluation
def half_time(result):
    """Time at which half of the finally entrained mass is taken up."""
    gained = result["mass"] - result["mass"][0]
    if gained[-1] <= 0.0:
        return float("nan")
    return float(np.interp(0.5 * gained[-1], gained, result["time"]))


def summarize(result):
    pft, header = result["pft"]
    return {
        "initial_mass": float(result["mass"][0]),
        "final_mass": float(result["mass"][-1]),
        "entrained_mass": float(result["mass"][-1] - result["mass"][0]),
        "half_entrainment_time": half_time(result),
        "runout_x": runout_x(pft, header),
        "max_pft": float(np.nanmax(pft)),
        "max_pfv": float(np.nanmax(result["pfv"][0])),
        "end_time": float(result["time"][-1]),
    }


def plot(scenario, com, ava, entrainment_path, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mass over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(com["ent"]["time"], (com["ent"]["mass"] - com["ent"]["mass"][0]) / 1e3,
            label="com1DFA erosion energy", color="k")
    for name, result in ava.items():
        if name == "null":
            continue
        ax.plot(result["time"], (result["mass"] - result["mass"][0]) / 1e3,
                label=f"avalanchers {name}", ls="-" if name == "erosion-energy" else "--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("entrained mass (t)")
    ax.set_title(f"{CASE}, {scenario}: entrained mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "entrained_mass.png", dpi=150)
    plt.close(fig)

    # peak flow thickness, ent and ent - null, both codes
    panels = [("com1DFA ent", com["ent"]["pft"][0]),
              ("avalanchers ent", ava["erosion-energy"]["pft"][0])]
    if "null" in com:
        panels += [("com1DFA ent - null", com["ent"]["pft"][0] - com["null"]["pft"][0]),
                   ("avalanchers ent - null",
                    ava["erosion-energy"]["pft"][0] - ava["null"]["pft"][0])]
    header = ava["erosion-energy"]["pft"][1]
    x = x_coordinates(header)
    nrows = int(header["nrows"])
    y0 = header.get("yllcenter", header.get("yllcorner", 0.0))
    extent = [x[0], x[-1], y0, y0 + header["cellsize"] * (nrows - 1)]
    erodible = read_asc(entrainment_path)[0]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels)), sharex=True)
    for ax, (title, data) in zip(np.atleast_1d(axes), panels):
        diff = "null" in title
        limit = np.nanmax(np.abs(data)) if diff else np.nanmax(data)
        image = ax.imshow(np.where(np.abs(np.nan_to_num(data)) > 1e-3, data, np.nan),
                          extent=extent, origin="upper", aspect="equal",
                          cmap="RdBu_r" if diff else "viridis",
                          vmin=-limit if diff else 0.0, vmax=limit)
        ax.contour(np.flipud(erodible > 0), levels=[0.5], extent=extent, colors="w",
                   linewidths=0.8)
        ax.set_title(f"{title}: peak flow thickness (m)")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "peak_flow_thickness.png", dpi=150)
    plt.close(fig)

    # entrained depth of avalanchers on the erodible area
    depth = ava["erosion-energy"]["entrained_depth"][0]
    rows, cols = np.where(erodible > 0)
    window = (slice(max(rows.min() - 5, 0), rows.max() + 6),
              slice(max(cols.min() - 5, 0), cols.max() + 6))
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(depth[window], cmap="magma", vmin=0.0, vmax=float(np.nanmax(erodible)))
    ax.contour(erodible[window] > 0, levels=[0.5], colors="c", linewidths=0.8)
    ax.set_title(f"avalanchers erosion energy, {scenario}:\nentrained depth (m), "
                 "erodible area outlined")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "entrained_depth.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--avaframe", default="C:/git/AvaFrame", help="AvaFrame repository root")
    parser.add_argument("--avaframe-python", help="interpreter of an AvaFrame environment")
    parser.add_argument("--rerun-com1dfa", action="store_true")
    parser.add_argument("--no-other-laws", action="store_true",
                        help="skip the excess-shear-stress and velocity-proportional runs")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dem, release = input_files(args.avaframe)
    laws = ["erosion-energy"] if args.no_other_laws else [
        "erosion-energy", "excess-shear-stress", "velocity-proportional"]
    null = run_avalanchers("null", SETTINGS, dem, release, None, OUT_DIR / "avalanchers")
    summary = {"scenarios": {}, "checks": {}}
    com_null = None
    for scenario, config in SCENARIOS.items():
        print(f"\n--- {scenario}")
        if args.avaframe_python and (args.rerun_com1dfa or not com1dfa_dir(scenario).exists()):
            run_com1dfa(args.avaframe, args.avaframe_python, scenario)
        com = load_com1dfa(args.avaframe, scenario)
        com_null = com.get("null", com_null)
        if com_null is not None:
            com["null"] = com_null
        if "ent" not in com:
            print(f"no com1DFA results for {scenario}; pass --avaframe-python")
            continue
        entrainment = rasterize_entrainment(args.avaframe, dem, config["thickness"])
        out = OUT_DIR / scenario
        out.mkdir(parents=True, exist_ok=True)
        ava = {"null": null}
        for law in laws:
            ava[law] = run_avalanchers(law, dict(SETTINGS, enable_entrainment=True,
                                                 entrainment_model=law),
                                       dem, release, entrainment, out / "avalanchers")

        result = {
            "com1DFA_source": com["ent"]["source"],
            "available_snow_mass": available_snow_mass(entrainment, dem),
            "com1DFA": {k: summarize(v) for k, v in com.items()},
            "avalanchers": {k: summarize(v) for k, v in ava.items()},
        }
        com_ent = result["com1DFA"]["ent"]["entrained_mass"]
        ava_ent = result["avalanchers"]["erosion-energy"]["entrained_mass"]
        result["entrained_mass_ratio"] = ava_ent / com_ent if com_ent > 0 else float("nan")
        summary["scenarios"][scenario] = result
        checks = summary["checks"]
        checks[f"{scenario}: avalanchers entrains snow"] = ava_ent > 0.0
        checks[f"{scenario}: entrained mass at most the available snow"] = (
            ava_ent <= result["available_snow_mass"] * 1.01)
        checks[f"{scenario}: entrained mass within {ENTRAINED_MASS_TOLERANCE:.0%} of com1DFA"] = (
            abs(result["entrained_mass_ratio"] - 1.0) <= ENTRAINED_MASS_TOLERANCE)
        plot(scenario, com, ava, entrainment, out)
        print(f"entrained mass: com1DFA {com_ent / 1e3:.1f} t, avalanchers {ava_ent / 1e3:.1f} t "
              f"(ratio {result['entrained_mass_ratio']:.2f}); "
              f"available {result['available_snow_mass'] / 1e3:.1f} t")

    null_summary = summarize(null)
    summary["checks"]["null run keeps its mass"] = (
        abs(null_summary["entrained_mass"]) <= 1e-4 * null_summary["initial_mass"])
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print()
    for name, ok in summary["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"report: {OUT_DIR / 'report.md'}")
    return 0 if all(summary["checks"].values()) else 1


def write_report(summary):
    sections = []
    for scenario, result in summary["scenarios"].items():
        rows = []
        for code in ("com1DFA", "avalanchers"):
            for name, s in result[code].items():
                rows.append(f"| {code} | {name} | {s['initial_mass'] / 1e3:.1f} | "
                            f"{s['entrained_mass'] / 1e3:.1f} | "
                            f"{s['half_entrainment_time']:.1f} | {s['runout_x']:.0f} | "
                            f"{s['max_pft']:.2f} | {s['max_pfv']:.1f} | {s['end_time']:.0f} |")
        sections.append(f"""## {scenario}

com1DFA results: {result['com1DFA_source']}. Available erodible snow:
{result['available_snow_mass'] / 1e3:.1f} t. Entrained mass ratio avalanchers / com1DFA
(erosion energy): {result['entrained_mass_ratio']:.2f}.

| code | run | initial mass (t) | entrained (t) | t half entrained (s) | runout x (m, pft > {RUNOUT_THRESHOLD} m) | max pft (m) | max pfv (m/s) | end time (s) |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

![entrained mass]({scenario}/entrained_mass.png)

![peak flow thickness]({scenario}/peak_flow_thickness.png)

![entrained depth]({scenario}/entrained_depth.png)
""")
    checks = "\n".join(f"- [{'x' if ok else ' '}] {name}" for name, ok in summary["checks"].items())
    text = f"""# Entrainment comparison: {CASE}

## Checks

{checks}

{chr(10).join(sections)}"""
    (OUT_DIR / "report.md").write_text(text, encoding="utf8")


if __name__ == "__main__":
    sys.exit(main())
