# 2019 avalanche outlines — provenance

**Source dataset**
: SPOT6 Avalanche outlines 16 January 2019

**Authors**
: Hafner, E. & Bühler, Y., WSL Institute for Snow and Avalanche Research SLF

**Published by**
: EnviDat — <https://www.envidat.ch/dataset/spot6-avalanche-outlines-16-january-2019>

**DOI**
: [10.16904/envidat.235](https://doi.org/10.16904/envidat.235)

**Licence**
: ODbL — attribution required; share-alike applies to derived databases.

**Distribution file**
: `aval_outlines16012019.shp.zip` (54 MiB).

**Checksum of the distribution file**
: `sha256:af4099d949fb567c0bc07b3e46cbca40e6b8f7c4340a6fbe7311e2104e93251f`
  (56,256,600 bytes). Computed from the retrieved copy on 2026-08-19.
  Verify a fresh download against this before trusting it is the same release.

**CRS**
: EPSG:2056 (CH1903+ / LV95)

**Identity check**
: The published dataset describes 6,041 mapped avalanches; the master file here
  carries exactly 6,041 features, so this is that release.

## What is in this directory

`cases/` holds one shapefile per candidate avalanche, extracted from the master
outline file by `python_scripts/extract_cases.py`. `cases_2019.json` is the pool
manifest; `funnel_2019.json` records the filter cascade that produced it.

Note: this dataset has no native id field, so `objectid = FID + 1` is used as a
synthetic id, consistently with the other two years.

## Filter funnel

| stage | remaining |
|---|---|
| all mapped polygons | 6,041 |
| geometry valid/repairable & non-degenerate | 6,041 |
| `typ = SLAB` | 3,040 |
| `trg_typ = NATURAL` | 2,818 |
| single ring (`nparts = 1`) | 2,791 |
| size class 2–5 | 2,789 |
| area 2–60 ha | 2,205 |
| bbox ≤ 2000 m | 2,184 |
| drop ≥ 150 m | 2,059 |
| start zone ≥ 1550 m | 2,044 |
| no neighbour within 25 m | **443** |

## Role in the calibration campaign

A transfer test: the vector frozen on 2018 was applied here unchanged, to
measure whether the fit survives a different acquisition and a different
avalanche period.
