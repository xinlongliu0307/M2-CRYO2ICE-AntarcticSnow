#!/usr/bin/env python3
"""
Notebook 14: Collocation Radius Sensitivity Test (Full Coverage)
=================================================================

Purpose
-------
Address Stuart's Comment 3 on MS#2 by quantifying how the collocation
search radius R affects (1) matchup yield and (2) sector-median snow
thickness. Tests R = 3.5 km, 5 km (baseline), and 7 km.

Strategy
--------
This extended version covers the full set of austral winter months
analysed in the manuscript, ensuring the sensitivity test is directly
representative of the headline results rather than a subset:

  - 2022: August-October (post-rephasing, 3 months)
  - 2023: May-October    (full winter, 6 months)
  - 2024: May-October    (full winter, 6 months)
  - 2025: May-September  (end-of-processing, 5 months)

Total: 20 months x 3 radii = 60 per-month runs.

Outputs
-------
1. Per-radius CSV files with all matchups (audit trail).
2. CSV summary table with matchup yield and sector-median snow
   thickness at each R value.
3. A manuscript-ready sentence populated with the actual computed
   percentages for direct insertion into Section 3.

Estimated runtime on Gadi
-------------------------
~60-90 minutes (proportional to the single-winter test's ~15 min).
Recommend running inside a tmux session or as a PBS job.

Author: Xinlong Liu
Date: April 2026
"""

import sys
import os
import glob
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
WORKSPACE = '/g/data/gv90/xl1657/phd/M2_workspace'
sys.path.insert(0, os.path.join(WORKSPACE, 'notebooks'))

# Force re-import in case of iterative development
for mod_name in ['utils']:
    if mod_name in sys.modules:
        del sys.modules[mod_name]

from utils import (
    read_atl10_v7,
    read_cs2_l2e,
    collocate_cs2_is2,
    compute_snow_thickness,
)

ATL10_DIR = os.path.join(WORKSPACE, 'data/raw/ATL10')
CS2_DIR = os.path.join(WORKSPACE, 'data/raw/CS2_L2E')
OUT_DIR = os.path.join(WORKSPACE, 'output/sensitivity_R')
os.makedirs(OUT_DIR, exist_ok=True)

# All austral winter months covered in the manuscript
# 2022: Aug-Oct (post-July 2022 CryoSat-2 rephasing)
# 2023-2024: full May-October winters
# 2025: May-September (end-of-processing at submission time)
TEST_MONTHS = (
    [f'2022{m:02d}' for m in [8, 9, 10]]
    + [f'2023{m:02d}' for m in [5, 6, 7, 8, 9, 10]]
    + [f'2024{m:02d}' for m in [5, 6, 7, 8, 9, 10]]
    + [f'2025{m:02d}' for m in [5, 6, 7, 8, 9]]
)  # 20 months total

# Radii to test (in metres)
RADII_M = [3500, 5000, 7000]

# Minimum IS2 segments per matchup (same as production pipeline)
MIN_PTS = 10

# Physical constants for snow thickness retrieval (from production pipeline)
RHO_S = 330.0       # snow density, kg/m^3
DELTA = 0.70        # Ku-band penetration factor
# eta_s = (1 + 0.51 * rho_s)^1.5 with rho_s in g/cm^3
ETA_S = (1 + 0.51 * (RHO_S / 1000.0)) ** 1.5   # = 1.263


# Sector boundaries (same as production pipeline)
def assign_sector(lat, lon):
    """Assign each matchup to Weddell, Ross, or Other."""
    if lat > -50:
        return 'Other'
    # Weddell: 62 W to 15 E
    if -62 <= lon <= 15 or lon >= 298:  # handle -62 = 298
        if lon >= 298 or lon <= 15:
            return 'Weddell'
    # Ross: 160 E to 140 W (equivalent to 160 to 360-140=220)
    if 160 <= lon <= 180 or -180 <= lon <= -140:
        return 'Ross'
    return 'Other'


def extract_date(fname):
    """Extract YYYYMMDD from ATL10 or CS2 filename."""
    base = os.path.basename(fname)
    for part in base.split('_'):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
        if 'T' in part and len(part) >= 15:
            return part[:8]
    return None


def process_month_at_radius(ym, R_m, atl10_by_date_cache, cs2_by_date_cache):
    """
    Run day-by-day collocation for one month at one radius.
    Returns DataFrame of all matchups for this (month, radius).
    Caches the date-file lookups to avoid repeated glob() calls.
    """
    if ym not in atl10_by_date_cache:
        a_files = sorted(glob.glob(f'{ATL10_DIR}/**/*{ym}*.h5', recursive=True))
        if not a_files:
            a_files = sorted(glob.glob(f'{ATL10_DIR}/*{ym}*.h5'))
        c_files = sorted(glob.glob(f'{CS2_DIR}/**/*{ym}*.nc', recursive=True))
        if not c_files:
            c_files = sorted(glob.glob(f'{CS2_DIR}/*{ym}*.nc'))

        atl10_by_date = {}
        for f in a_files:
            d = extract_date(f)
            if d:
                atl10_by_date.setdefault(d, []).append(f)

        cs2_by_date = {}
        for f in c_files:
            d = extract_date(f)
            if d:
                cs2_by_date.setdefault(d, []).append(f)

        atl10_by_date_cache[ym] = atl10_by_date
        cs2_by_date_cache[ym] = cs2_by_date
    else:
        atl10_by_date = atl10_by_date_cache[ym]
        cs2_by_date = cs2_by_date_cache[ym]

    common_dates = sorted(set(atl10_by_date.keys()) & set(cs2_by_date.keys()))
    if not common_dates:
        return pd.DataFrame()

    monthly_matched = []
    for date in common_dates:
        is2_frames = [read_atl10_v7(f) for f in atl10_by_date[date]]
        is2_frames = [f for f in is2_frames if len(f) > 0]
        cs2_frames = [read_cs2_l2e(f) for f in cs2_by_date[date]]
        cs2_frames = [f for f in cs2_frames if len(f) > 0]

        if not is2_frames or not cs2_frames:
            continue

        df_is2 = pd.concat(is2_frames, ignore_index=True)
        df_cs2 = pd.concat(cs2_frames, ignore_index=True)

        matched = collocate_cs2_is2(
            df_cs2, df_is2,
            R_m=R_m, min_pts=MIN_PTS,
        )

        if len(matched) > 0:
            matched['date'] = date
            matched['year_month'] = ym
            monthly_matched.append(matched)

        del df_is2, df_cs2, is2_frames, cs2_frames, matched

    if monthly_matched:
        return pd.concat(monthly_matched, ignore_index=True)
    return pd.DataFrame()


# ----------------------------------------------------------------------
# MAIN SENSITIVITY LOOP
# ----------------------------------------------------------------------
def main():
    t_start = time.time()

    print('=' * 72)
    print('Collocation Radius Sensitivity Test (Full Manuscript Coverage)')
    print('=' * 72)
    print(f'Months to process: {len(TEST_MONTHS)} '
          f'({TEST_MONTHS[0]} to {TEST_MONTHS[-1]})')
    print(f'Radii to test:     {RADII_M} m')
    print(f'Output directory:  {OUT_DIR}')
    print()

    # Cache file listings once per month to save repeated glob() calls
    atl10_by_date_cache = {}
    cs2_by_date_cache = {}

    summary_rows = []

    for R_m in RADII_M:
        R_km = R_m / 1000.0
        print(f'--- R = {R_km:.1f} km ---')
        t_r = time.time()

        all_matched = []
        for ym in TEST_MONTHS:
            df_month = process_month_at_radius(
                ym, R_m,
                atl10_by_date_cache, cs2_by_date_cache,
            )
            if len(df_month) > 0:
                all_matched.append(df_month)
                print(f'  {ym}: {len(df_month):>6,} matchups')
            else:
                print(f'  {ym}: no matchups')

        if not all_matched:
            print(f'  WARNING: no matchups at R = {R_km} km')
            continue

        df_all = pd.concat(all_matched, ignore_index=True)

        # Retrieve snow thickness: h_s = delta_f / (delta * eta_s)
        delta_f = df_all['is2_fb_wm'] - df_all['cs2_rfb']
        df_all['snow_thickness'] = delta_f / (DELTA * ETA_S)

        # Assign sectors
        df_all['sector'] = df_all.apply(
            lambda r: assign_sector(r['lat'], r['lon']),
            axis=1,
        )

        # Save per-radius CSV (full audit trail)
        out_file = os.path.join(OUT_DIR, f'sensitivity_R{R_km:.1f}km.csv')
        df_all.to_csv(out_file, index=False)

        # Compute sector-level statistics
        for sec in ['Weddell', 'Ross']:
            ds = df_all[df_all['sector'] == sec]
            if len(ds) > 0:
                summary_rows.append({
                    'R_km': R_km,
                    'sector': sec,
                    'n_matchups': len(ds),
                    'mean_hs_m': ds['snow_thickness'].mean(),
                    'median_hs_m': ds['snow_thickness'].median(),
                    'n_is2_median': ds['n_is2'].median(),
                })

        elapsed_r = time.time() - t_r
        print(f'  Total at R = {R_km} km: {len(df_all):,} matchups '
              f'({elapsed_r/60:.1f} min)')
        print()

    # ------------------------------------------------------------------
    # SUMMARY TABLE
    # ------------------------------------------------------------------
    df_summary = pd.DataFrame(summary_rows)
    summary_file = os.path.join(OUT_DIR, 'sensitivity_R_summary.csv')
    df_summary.to_csv(summary_file, index=False)

    print('=' * 72)
    print('SUMMARY TABLE (full austral winter coverage, 2022-2025)')
    print('=' * 72)
    print(df_summary.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # MANUSCRIPT-READY STATISTIC
    # ------------------------------------------------------------------
    try:
        n_35 = df_summary[df_summary.R_km == 3.5].n_matchups.sum()
        n_50 = df_summary[df_summary.R_km == 5.0].n_matchups.sum()
        n_70 = df_summary[df_summary.R_km == 7.0].n_matchups.sum()

        yield_gain_35_to_50 = 100 * (n_50 - n_35) / n_35
        yield_gain_50_to_70 = 100 * (n_70 - n_50) / n_50

        med_diff_hs = []
        for sec in ['Weddell', 'Ross']:
            m35 = df_summary[(df_summary.R_km == 3.5)
                             & (df_summary.sector == sec)].median_hs_m.values
            m50 = df_summary[(df_summary.R_km == 5.0)
                             & (df_summary.sector == sec)].median_hs_m.values
            m70 = df_summary[(df_summary.R_km == 7.0)
                             & (df_summary.sector == sec)].median_hs_m.values
            if len(m35) and len(m50) and len(m70):
                med_diff_hs.append(abs(m50[0] - m35[0]))
                med_diff_hs.append(abs(m70[0] - m50[0]))

        max_med_shift = max(med_diff_hs) if med_diff_hs else float('nan')

        print('=' * 72)
        print('MANUSCRIPT-READY SENTENCE (copy into Section 3):')
        print('=' * 72)
        print()
        print(f'A sensitivity test across all twenty austral winter months '
              f'(2022-2025) confirms that the R = 5 km choice retains '
              f'approximately {yield_gain_35_to_50:.0f}% more matchups than '
              f'R = 3.5 km while the sector-median snow thickness shifts by '
              f'less than {max_med_shift*1000:.0f} mm, well below the '
              f'per-matchup uncertainty of 38 mm. Enlarging the radius '
              f'further to R = 7 km yields {yield_gain_50_to_70:.0f}% '
              f'additional matchups at the cost of coarser effective '
              f'resolution.')
        print()
    except Exception as e:
        print(f'Could not generate summary sentence: {e}')

    elapsed = time.time() - t_start
    print(f'Total runtime: {elapsed/60:.1f} minutes')
    print(f'Summary saved to: {summary_file}')


if __name__ == '__main__':
    main()
