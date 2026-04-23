# Antarctic CRYO2ICE Snow Thickness Retrieval (Manuscript 2)

Code and derived data for the manuscript:

> Liu, X., Fraser, A. D., Corney, S., Heil, P., and Tilling, R. L. (2026). *Antarctic snow thickness from CRYO2ICE dual altimetry: retrieval, sensitivity, and validation against AMSR2* (Submitted).

This repository contains the complete processing pipeline used to retrieve Antarctic snow thickness from coincident ICESat-2 ATL10 V7 and CryoSat-2 Baseline-E (L2E) observations over the CRYO2ICE period (August 2022 – September 2025), together with the scripts that generate the manuscript figures and tables.

## Repository structure

```
M2-CRYO2ICE-AntarcticSnow/
├── src/              Core Python library and batch driver scripts
│   ├── utils.py                        ATL10 V7 reader, CS2 L2E reader,
│   │                                   collocation, snow-thickness retrieval
│   ├── batch_process.py                Monthly batch driver (NCI Gadi)
│   ├── sic_sensitivity_test.py         SIC-threshold sensitivity experiment
│   └── collocation_radius_sensitivity.py  Collocation-radius sensitivity experiment
├── notebooks/        Analysis and figure-making Jupyter notebooks (run in numerical order)
│   ├── 01_read_ATL10_V7.ipynb
│   ├── 02_read_CS2_L2E.ipynb
│   ├── 03_collocation_binning.ipynb
│   ├── 04_snow_thickness_retrieval.ipynb
│   ├── 05_amsr2_comparison.ipynb
│   ├── 06_manuscript_figures.ipynb
│   ├── 07_aug2022_correction_check.ipynb
│   ├── 08_gridded_map.ipynb
│   ├── 09_penetration_sensitivity.ipynb
│   ├── 10_autocorrelation.ipynb
│   ├── 11_sic_sensitivity.ipynb
│   └── 12_amsr2_spatial_temporal.ipynb
├── data/             Monthly collocated tracks and small summary files
│   └── README.md                      Download link for the full archive (Zenodo)
├── logs/             Run logs from key pipeline steps
├── requirements.txt  Pinned Python dependencies
├── environment.yml   Conda environment specification
├── LICENSE           MIT Licence (applies to all code in this repository)
├── CITATION.cff      Machine-readable citation metadata
└── README.md         This file
```

## Scientific overview

Antarctic snow thickness on sea ice is retrieved using the dual-frequency altimetry identity

&nbsp;&nbsp;&nbsp;&nbsp;*h*<sub>s</sub> = (*h*<sub>f</sub><sup>IS2</sup> − *h*<sub>f</sub><sup>CS2</sup>) / (δ · η<sub>s</sub>)

where *h*<sub>f</sub><sup>IS2</sup> is ATL10 V7 total freeboard, *h*<sub>f</sub><sup>CS2</sup> is the CryoSat-2 L2E Ku-band radar freeboard, δ is the Ku-band snow-penetration factor, and η<sub>s</sub> = (1 + 0.51 ρ<sub>s</sub>)<sup>1.5</sup> = 1.263 at ρ<sub>s</sub> = 330 kg m⁻³ corrects for the slower radar propagation in snow. The reference central value δ* = 0.55 (median) / 0.58 (mean) is determined empirically by minimising the bias against AMSR2 snow depths.

The pipeline implements: all-six-beam ATL10 V7 ingestion, inverse-distance-weighted collocation of IS2 freeboard onto CS2 footprints, monthly binning, uncertainty propagation, δ sensitivity (0.30–0.90), SIC-threshold sensitivity (50–90 %), collocation-radius sensitivity (3.5, 5.0, 7.0 km), along-track autocorrelation diagnostics, gridding onto the 6.25 km NSIDC polar stereographic South grid, and direct comparison with AMSR2 swath snow depths.

## Data availability

The repository ships with the per-month collocated CSVs (`data/collocated_YYYYMM.csv`, 34 months) and small summary tables needed to reproduce most figures directly from the notebooks. The larger derived products are archived separately on Zenodo with a persistent DOI (see `data/README.md`):

| Product | Approx. size | Purpose |
|---|---|---|
| `gridded_snow_thickness_6250m.nc` | 74 MB | 6.25 km gridded snow-thickness fields, used in notebook 08 |
| `snow_thickness_with_amsr2_all.csv` | 20 MB | Paired C2I–AMSR2 retrievals, used in notebooks 05 and 12 |
| `sensitivity_R{3.5,5.0,7.0}km.csv` | 13–17 MB each | Collocation-radius sensitivity output, used in notebook 11 |

**Raw input data** (ICESat-2 ATL10 V7, CryoSat-2 L2E, AMSR2) are not redistributed here; they are openly available from NSIDC (ATL10, AMSR2) and ESA (CS2 L2E). Exact product identifiers and access procedures are documented in Section 2 of the manuscript.

## Reproducing the analysis

1. **Environment.** Create a Python 3.9 environment matching `environment.yml` (conda) or `requirements.txt` (pip). The pipeline was developed and tested on the NCI Gadi system (project gv90) using Python 3.9.2.
2. **Small-figure reproduction.** Notebooks 05, 06, 09, 10, 11, and 12 run end-to-end from the `data/` files shipped with the repository. No external downloads are required.
3. **Full pipeline reproduction.** Download the large derived files from Zenodo into `data/`; download the ATL10 V7 and CS2 L2E raw inputs from NSIDC and ESA to a local directory; update the paths at the top of `src/batch_process.py`; run `python src/batch_process.py` to regenerate the monthly collocations; then work through notebooks 01 → 12 in order.

## Citation

If you use this code or data, please cite both the manuscript (above) and the archived code release (see `CITATION.cff`).

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`). The archived data products on Zenodo are released under CC-BY-4.0.

## Contact

Xinlong Liu, Institute for Marine and Antarctic Studies, University of Tasmania, Australia.
