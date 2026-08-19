# 1999 avalanche outlines — provenance

**Source dataset**
: Avalanche outlines February and March 1999 from aerial imagery

**Authors**
: Hafner, E. D. & Dal, J. F., WSL Institute for Snow and Avalanche Research SLF

**Published by**
: EnviDat — <https://www.envidat.ch/dataset/avalanche-outlines>

**DOI**
: [10.16904/envidat.579](https://doi.org/10.16904/envidat.579)

**Licence**
: CC-BY-SA 4.0 — <https://creativecommons.org/licenses/by-sa/4.0/> — attribution
  required; share-alike applies to derivatives. (2018/2019 are ODbL, not
  CC-BY-SA; do not carry the ODbL wording over to this dataset.)

**Distribution file**
: `avalanche_data_1999_all.zip` (published size 150,084,348 bytes ≈ 143 MiB).
  The published resource bundles three shapefiles — `avalanches1999_endversion1.shp` (the mapped
  avalanches, used here), `area_images_1999.shp` (image coverage area), and
  `clouds_1999.shp` (cloud outlines).

**Checksum of the distribution file**
: `sha256:7a456616f8dfd01c39c8a7b945abf9ebe46436f084b4c13a7f09f2651fd64427`
  (150,084,348 bytes). Computed from the retrieved copy on 2026-08-19.
  Verify a fresh download against this before trusting it is the same release.

**CRS**
: EPSG:2056 (CH1903+ / LV95)

**Identity check**
: The published record describes 11,120 mapped avalanche outlines from
  panchromatic aerial imagery taken 25 February – 1 March 1999; the master
  file here has exactly 11,120 features, and the distribution filename
  matches the published resource filename exactly
  (`avalanche_data_1999_all.zip`). This is a positive identification, not a
  guess — feature count, date range, and filename all agree.

**Source imagery**
: Panchromatic aerial photographs, rectified versions available through
  Swisstopo (not redistributed here).

## Field quirks

The `Id` field is **not unique** — 100 distinct values across 11,120 features —
so `objectid = FID + 1` is used as a synthetic id, as for 2019. The `aspect`,
`start_zone` and `dpo_alt` fields are entirely dead (100% null), and `trg_typ`
is 99.97% null, which is why the funnel below skips the `trg_typ = NATURAL`,
drop and start-zone stages that the 2018 and 2019 funnels apply. Field names
(`sze`, `typ`, `trg_typ`, `frac_wdh`, `aval_shape`, `dpo_alt`, `start_zone`)
match the same SLF mapping schema used in the 2018 and 2019 releases, which is
further corroborating evidence for the SLF origin.

## What is in this directory

`cases/` holds one shapefile per candidate avalanche, extracted by
`python_scripts/extract_cases.py`. `cases_1999.json` is the pool manifest;
`funnel_1999.json` records the filter cascade.

## Filter funnel

| stage | remaining |
|---|---|
| all mapped polygons | 11,120 |
| geometry valid/repairable & non-degenerate | 11,120 |
| `typ = SLAB` | 4,306 |
| single ring (`nparts = 1`) | 4,062 |
| size class 2–5 | 4,055 |
| area 2–60 ha | 2,993 |
| bbox ≤ 2000 m | 2,880 |
| no neighbour within 25 m | **457** |

Three stages present in the other two years are absent here because the fields
they test are dead — see "Field quirks". The 1999 pool is therefore filtered
less strictly than 2018 and 2019, which matters when comparing scores across
datasets.

## Role in the calibration campaign

A transfer test, and the weakest of the three: with no deployable terrain
features it cannot test the standing-wall hypothesis, and the observed flip
relative to the other panels sits inside same-dataset noise.
