# 2018 avalanche outlines — provenance

**Source dataset**
: SPOT6 Avalanche outlines 24 January 2018

**Authors**
: Hafner, E. & Bühler, Y. (2019), WSL Institute for Snow and Avalanche Research SLF

**Published by**
: EnviDat — <https://www.envidat.ch/dataset/spot6-avalanche-outlines-24-january-2018>

**DOI**
: [10.16904/envidat.77](https://doi.org/10.16904/envidat.77)

**Licence**
: ODbL — attribution required; share-alike applies to derived databases.

**Distribution file**
: `aval_outlines2018.zip` (150 MiB).

**Checksum of the distribution file**
: `sha256:087c036f1a3e4213c2332fad4497fd292c6af6f0df1629c5aaa887a45387c2f5`
  (157,117,977 bytes). Computed from the retrieved copy on 2026-08-19.
  Verify a fresh download against this before trusting it is the same release.

**CRS**
: EPSG:2056 (CH1903+ / LV95)

## What is in this directory

`cases/` holds one shapefile per candidate avalanche, extracted from the master
outline file by `python_scripts/extract_cases.py`. `cases_2018.json` is the pool
manifest; `funnel_2018.json` records the filter cascade that produced it.

Only the polygons are redistributed here — the extraction is a subset and a
reformat, not a modification of the geometry.

## Filter funnel

| stage | remaining |
|---|---|
| all mapped polygons | 18,737 |
| geometry valid/repairable & non-degenerate | 18,737 |
| `typ = SLAB` | 13,492 |
| `trg_typ = NATURAL` | 12,616 |
| single ring (`nparts = 1`) | 12,500 |
| size class 2–5 | 12,497 |
| area 2–60 ha | 7,265 |
| bbox ≤ 2000 m | 7,211 |
| drop ≥ 150 m | 6,762 |
| start zone ≥ 1550 m | 6,741 |
| no neighbour within 25 m | **602** |

Thresholds are not invented here — they follow the funnel documented in
`campaign/analysis/census.py`.

## Role in the calibration campaign

The primary dataset. The 105-case panel in `data/panel/` is drawn from this
pool and is what the parameter vector was fitted against; a held-out split
("2018hold") was scored separately as a same-dataset noise floor.
