# Antarctic CRYO2ICE Snow Thickness Retrieval

Code and derived data for:

> Liu, X., Fraser, A. D., Tilling, R. L., Heil, P., and Corney, S. (2026).
> Robust Regional Contrast but Penetration-Sensitive Snow Thickness
> over Sea Ice from Multi-Winter Antarctic CRYO2ICE Observations.
> *Earth and Space Science (AGU)*, 13(8), e2026EA005247.
> https://doi.org/10.1029/2026EA005247

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19220192.svg)](https://doi.org/10.5281/zenodo.19220192)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

This repository contains the complete processing pipeline used to retrieve
Antarctic snow thickness from coincident ICESat-2 ATL10 V7 and CryoSat-2
Baseline-E (L2E) observations over the CRYO2ICE period (August 2022 –
September 2025), together with the scripts and notebooks that generate the
manuscript figures, tables, and sensitivity-test outputs.

---

## Repository structure

```
cryo2ice-antarctic-snow-depth/
├── src/                 Core Python library and batch driver scripts
│   ├── utils.py                              ATL10 V7 / CS2 L2E readers,
│   │                                         collocation, retrieval routines
│   ├── batch_process.py                      Monthly batch driver (NCI Gadi)
│   ├── sic_sensitivity_test.py               SIC-threshold sensitivity
│   └── collocation_radius_sensitivity.py     Collocation-radius sensitivity
├── notebooks/           Analysis and figure-making Jupyter notebooks
│                        (run in numerical order)
├── data/                Monthly collocated tracks and small summary tables
│   └── README.md                             Download link for the full
│                                             archive (Zenodo)
├── logs/                Run logs from key pipeline steps
├── .gitignore
├── CITATION.cff         Machine-readable citation metadata
├── LICENSE              MIT Licence (applies to all code in this repository)
├── README.md            This file
├── environment.yml      Conda environment specification
└── requirements.txt     Pinned Python dependencies
```

---

## Scientific overview

Antarctic snow thickness on sea ice is retrieved using the dual-frequency
altimetry identity

> *h*ₛ = (*h*<sub>f,IS2</sub> − *h*<sub>f,CS2</sub>) / (δ · η<sub>s</sub>)

where *h*<sub>f,IS2</sub> is the ATL10 V7 total freeboard,
*h*<sub>f,CS2</sub> is the CryoSat-2 L2E Ku-band radar freeboard,
δ is the Ku-band snow-penetration factor, and
η<sub>s</sub> = (1 + 0.51 ρ<sub>s</sub>)^1.5 = 1.263 at
ρ<sub>s</sub> = 330 kg m⁻³ corrects for the slower radar propagation in
snow. The reference central value δ\* = 0.55 (median) / 0.58 (mean) is
determined empirically by minimising the bias against AMSR2 snow depths.

The pipeline implements:

- all-six-beam ATL10 V7 ingestion;
- inverse-distance-weighted collocation of IS2 freeboard onto CS2 footprints;
- monthly binning with uncertainty propagation;
- δ sensitivity (0.30 – 0.90);
- SIC-threshold sensitivity (50 – 90 %);
- collocation-radius sensitivity (3.5, 5.0, 7.0 km);
- along-track autocorrelation and effective-sample-size diagnostics;
- gridding onto the 6.25 km NSIDC polar stereographic South grid;
- direct comparison with AMSR2 swath snow depths over the same period.

---

## Workflow

The Python pipeline and notebooks are run in numerical order. The `src/`
modules provide the heavy data-processing functions; the `notebooks/` folder
contains the analysis and figure-generation steps that consume their output.

| Notebook | Purpose |
|---|---|
| `01_read_ATL10_V7.ipynb` | Read and validate ATL10 V7 total-freeboard inputs |
| `02_read_CS2_L2E.ipynb` | Read and validate CryoSat-2 L2E Ku-band radar-freeboard inputs |
| `03_collocation_binning.ipynb` | Inverse-distance-weighted collocation of IS2 freeboard onto CS2 footprints and monthly binning |
| `04_snow_thickness_retrieval.ipynb` | Apply the dual-frequency snow-thickness identity with uncertainty propagation |
| `05_amsr2_comparison.ipynb` | Compare retrieved snow thickness with AMSR2 swath snow depths |
| `06_manuscript_figures.ipynb` | Generate the main manuscript figures |
| `07_aug2022_correction_check.ipynb` | August-2022 correction-check diagnostic |
| `08_gridded_map.ipynb` | Grid retrievals onto the 6.25 km NSIDC polar stereographic South grid |
| `09_penetration_sensitivity.ipynb` | δ-sensitivity experiment across 0.30 – 0.90 |
| `10_autocorrelation.ipynb` | Along-track autocorrelation and effective-sample-size diagnostics |
| `11_sic_sensitivity.ipynb` | Sea-ice concentration threshold sensitivity (50 – 90 %) |
| `12_amsr2_spatial_temporal.ipynb` | Spatial and temporal comparison against AMSR2 |

---

## Data availability

The repository ships with the per-month collocated CSVs
(`data/collocated_YYYYMM.csv`, 34 months, August 2022 – September 2025) and
small summary tables sufficient to reproduce most figures directly from the
notebooks. The larger derived products are archived separately on Zenodo
with a persistent DOI:

> Liu, X., Fraser, A. D., Tilling, R. L., Heil, P., and Corney, S. (2026).
> Python Scripts, Jupyter Notebooks, and Results for "Robust Regional Contrast
> but Penetration-Sensitive Snow Thickness over Sea Ice from Multi-Winter
> Antarctic CRYO2ICE Observations." *Zenodo*.
> https://doi.org/10.5281/zenodo.19220192

| Product | Approx. size | Used by |
|---|---|---|
| `gridded_snow_thickness_6250m.nc` | 74 MB | Notebook 08 (gridded snow-thickness fields) |
| `snow_thickness_with_amsr2_all.csv` | 20 MB | Notebooks 05 and 12 (paired C2I–AMSR2 retrievals) |
| `sensitivity_R3.5km.csv` | 13–17 MB | Notebook 11 (collocation-radius sensitivity, R = 3.5 km) |
| `sensitivity_R5.0km.csv` | 13–17 MB | Notebook 11 (collocation-radius sensitivity, R = 5.0 km) |
| `sensitivity_R7.0km.csv` | 13–17 MB | Notebook 11 (collocation-radius sensitivity, R = 7.0 km) |

**Raw input data** (ICESat-2 ATL10 V7, CryoSat-2 L2E, AMSR2) are not
redistributed here. They are openly available from NSIDC (ATL10, AMSR2)
and ESA (CryoSat-2 L2E). Exact product identifiers and access procedures
are documented in Section 2 of the manuscript.

---

## Reproducing the analysis

### 1. Environment

The pipeline was developed and tested on the NCI Gadi system (project gv90)
using Python 3.9.2. Create a matching environment with either conda or pip:

```bash
# Conda
conda env create -f environment.yml
conda activate cryo2ice-snow

# Pip alternative
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Small-figure reproduction (no external downloads)

Notebooks 05, 06, 09, 10, 11, and 12 run end-to-end from the `data/` files
shipped with this repository. After environment setup, simply launch
JupyterLab and execute the relevant notebook:

```bash
jupyter lab notebooks/06_manuscript_figures.ipynb
```

### 3. Full pipeline reproduction

```bash
# (a) Download the large derived files from Zenodo into data/
#     https://doi.org/10.5281/zenodo.19220192

# (b) Download ATL10 V7 (NSIDC), CS2 L2E (ESA), and AMSR2 (NSIDC)
#     raw inputs to a local directory of your choice

# (c) Update the input/output paths at the top of src/batch_process.py

# (d) Regenerate monthly collocations
python src/batch_process.py

# (e) Run the notebooks in numerical order, 01 → 12
```

---

## Software environment

- Python 3.9 (3.9.2 used in production runs on NCI Gadi)
- Core scientific stack: `numpy`, `pandas`, `xarray`, `netCDF4`, `scipy`,
  `matplotlib`, `cartopy`, `pyproj`, `cmocean`
- Geospatial: `shapely`, `geopandas`, `pyresample`
- Notebook tooling: `jupyterlab`, `ipykernel`

Exact pins used to produce the manuscript outputs are recorded in
`environment.yml` and `requirements.txt`.

---

## How to cite

If you use this code or data, please cite both the paper (or the
archived release) **and** the data deposit.
A machine-readable `CITATION.cff` is provided in the repository
root and is automatically rendered by GitHub's "Cite this repository"
button.

**Paper**

> Liu, X., Fraser, A. D., Tilling, R. L., Heil, P., and Corney, S. (2026).
> Robust Regional Contrast but Penetration-Sensitive Snow Thickness
> over Sea Ice from Multi-Winter Antarctic CRYO2ICE Observations.
> *Earth and Space Science*, 13(8), e2026EA005247.
> https://doi.org/10.1029/2026EA005247

**Software and derived data (Zenodo)**

> Liu, X., Fraser, A. D., Tilling, R. L., Heil, P., and Corney, S. (2026).
> *Python Scripts, Jupyter Notebooks, and Results for "Robust Regional
> Contrast but Penetration-Sensitive Snow Thickness over Sea Ice from
> Multi-Winter Antarctic CRYO2ICE Observations"*
> [Software and data]. Zenodo.
> https://doi.org/10.5281/zenodo.19220192

---

## Licence

Code in this repository is released under the [MIT Licence](LICENSE).
The archived data products on Zenodo are released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

Xinlong Liu  
Institute for Marine and Antarctic Studies, University of Tasmania,
Hobart, Tasmania, Australia  
xinlong.liu@utas.edu.au
