"""Run AvaFrame com1DFA for the entrainment comparison (run_entrainment_test.py).

Runs in an AvaFrame environment (numpy < 2, compiled Cython extensions), not
the avalanchers one, so run_entrainment_test.py starts it as a subprocess:

    <avaframe-python> verification/entrainment_com1dfa_runner.py \
        --avaframe C:/git/AvaFrame --case avaHockeyChannel \
        --release release1HS --out verification/outputs/entrainment/com1DFA

Copies Inputs/ of the AvaFrame test case into ``--out/<case>`` and runs the
null and the entrainment simulation with the default com1DFA configuration
(samosAT, entEroEnergy 5000 J/kg, rhoEnt 100 kg/m^3, entrainment thickness
from the shapefile, or --entrainment-thickness everywhere in the polygon). The release and entrainment rasters com1DFA derives from
the shapefiles are exported to Outputs/internalRasters; avalanchers uses the
release raster (the exported entrainment raster is overwritten by the null
run, so run_entrainment_test.py rasterizes the entrainment shapefile itself).
"""

import argparse
import pathlib
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--avaframe", required=True, help="AvaFrame repository root")
    parser.add_argument("--case", default="avaHockeyChannel")
    parser.add_argument("--release", default="release1HS")
    parser.add_argument("--out", required=True, help="directory receiving the avalanche directory")
    parser.add_argument("--friction-model", default="samosAT")
    parser.add_argument("--sim-types", default="null|ent")
    parser.add_argument("--entrainment-thickness", type=float,
                        help="uniform erodible depth (m) instead of the shapefile thickness")
    args = parser.parse_args()

    avaframe_root = pathlib.Path(args.avaframe).resolve()
    sys.path.insert(0, str(avaframe_root))
    from avaframe.com1DFA import com1DFA
    from avaframe.in3Utils import cfgUtils, logUtils

    source = avaframe_root / "avaframe" / "data" / args.case / "Inputs"
    avalanche_dir = pathlib.Path(args.out).resolve() / args.case
    if avalanche_dir.exists():
        shutil.rmtree(avalanche_dir)
    shutil.copytree(source, avalanche_dir / "Inputs")

    cfg_main = cfgUtils.getGeneralConfig()
    cfg_main["MAIN"]["avalancheDir"] = str(avalanche_dir)
    # one simulation at a time: parallel runs rewrite the same exported
    # internal rasters, which fails on Windows (file in use)
    cfg_main["MAIN"]["nCPU"] = "1"
    logUtils.initiateLogger(str(avalanche_dir), "entrainmentComparison")

    cfg = cfgUtils.getModuleConfig(com1DFA, str(avalanche_dir), toPrint=False)
    cfg["GENERAL"]["simTypeList"] = args.sim_types
    cfg["GENERAL"]["frictModel"] = args.friction_model
    cfg["INPUT"]["releaseScenario"] = args.release
    if args.entrainment_thickness is not None:
        cfg["GENERAL"]["entThFromFile"] = "False"
        cfg["GENERAL"]["entTh"] = str(args.entrainment_thickness)
    cfg["EXPORTS"]["exportRasters"] = "True"
    # plain ASCII grids, readable without rasterio on the avalanchers side
    cfg["EXPORTS"]["useCompression"] = "False"
    com1DFA.com1DFAMain(cfg_main, cfgInfo=cfg)
    print(f"com1DFA results in {avalanche_dir / 'Outputs'}")


if __name__ == "__main__":
    main()
