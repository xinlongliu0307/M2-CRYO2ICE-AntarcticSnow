# Data directory

This directory ships with the small derived data products needed to reproduce most figures in the accompanying manuscript directly from the notebooks. Larger files are archived separately on Zenodo.

## Files included in this repository

### Monthly collocated tracks
`collocated_YYYYMM.csv` — 34 files covering August 2022 to September 2025.

Each row is one CS2 radar-freeboard footprint with the inverse-distance-weighted ATL10 V7 total freeboard from all six beams. Columns:

| Column | Description |
|---|---|
| `lat`, `lon` | CS2 footprint centre (degrees) |
| `cs2_rfb` | CryoSat-2 Ku-band radar freeboard (m) |
| `is2_fb_wm` | IDW-weighted ATL10 total freeboard at the CS2 footprint (m) |
| `is2_fb_mean`, `is2_fb_std` | Unweighted mean and standard deviation of contributing IS2 segments (m) |
| `is2_fb_unc_mean` | Mean of `beam_fb_unc` across contributing IS2 segments (m) |
| `n_is2` | Total number of IS2 segments contributing to the IDW weight |
| `n_strong`, `n_weak` | Segment counts by beam type |
| `date` | Observation date (YYYYMMDD) |
| `year_month` | Observation month (YYYYMM) |
| `snow_thickness` | Retrieved snow thickness at δ* = 0.55, η_s = 1.263 (m) |
| `snow_thickness_unc` | Propagated 1-σ uncertainty (m) |

### Summary tables
- `penetration_sensitivity_results.csv` — δ-sensitivity sweep (Section 4.2 of the manuscript).
- `effective_sample_size_summary.csv` — Along-track autocorrelation-based effective sample size.
- `sensitivity_R_summary.csv` — Summary statistics of the collocation-radius sensitivity experiment (the full per-row output is on Zenodo).
- `autocorrelation_results.npz` — Raw autocorrelation curves used in notebook 10.

## Files archived on Zenodo (download separately)

The following larger derived products are not tracked in Git. Download them from the Zenodo record (DOI: **TBD — insert after first Zenodo deposit**) and place them in this directory before running the notebooks that need them.

| File | Size | Needed by |
|---|---|---|
| `gridded_snow_thickness_6250m.nc` | 74 MB | notebook 08 |
| `snow_thickness_with_amsr2_all.csv` | 20 MB | notebooks 05, 12 |
| `sensitivity_R3.5km.csv` | 13 MB | notebook covering radius sensitivity |
| `sensitivity_R5.0km.csv` | 15 MB | same |
| `sensitivity_R7.0km.csv` | 17 MB | same |

## Raw input data

Raw ICESat-2 ATL10 V7 granules, CryoSat-2 Baseline-E L2E files, and AMSR2 swath files are not redistributed in this repository or on Zenodo. They are freely available from NSIDC (https://nsidc.org) and ESA (https://science-pds.cryosat.esa.int). The exact product identifiers, temporal coverage, and access procedures are given in Section 2 of the manuscript.
