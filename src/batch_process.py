"""
Batch collocation: Aug 2022 - Sep 2025
Run as: python batch_process.py
"""
import sys, os, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/g/data/gv90/xl1657/phd/M2_workspace/notebooks')
from utils import read_atl10_v7, read_cs2_l2e, collocate_cs2_is2, compute_snow_thickness

ATL10_DIR = '/g/data/gv90/xl1657/phd/M2_workspace/data/raw/ATL10'
CS2_DIR   = '/g/data/gv90/xl1657/phd/M2_workspace/data/raw/CS2_L2E'
OUT_DIR   = '/g/data/gv90/xl1657/phd/M2_workspace/output/collocated'
HS_DIR    = '/g/data/gv90/xl1657/phd/M2_workspace/output/snow_thickness'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HS_DIR, exist_ok=True)

months = []
for yr in range(2022, 2026):
    for mn in range(1, 13):
        ym = f'{yr}{mn:02d}'
        if 202208 <= int(ym) <= 202509:
            months.append(ym)

def extract_date(fname):
    base = os.path.basename(fname)
    for part in base.split('_'):
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
        if 'T' in part and len(part) >= 15:
            return part[:8]
    return None

print(f'Processing {len(months)} months')
t_start = time.time()

for ym in months:
    out_file = f'{OUT_DIR}/collocated_{ym}.csv'
    if os.path.exists(out_file):
        print(f'{ym}: Already processed, skipping')
        continue

    a_files = sorted(glob.glob(f'{ATL10_DIR}/**/*{ym}*.h5', recursive=True))
    if not a_files:
        a_files = sorted(glob.glob(f'{ATL10_DIR}/*{ym}*.h5'))
    c_files = sorted(glob.glob(f'{CS2_DIR}/**/*{ym}*.nc', recursive=True))
    if not c_files:
        c_files = sorted(glob.glob(f'{CS2_DIR}/*{ym}*.nc'))

    if not a_files or not c_files:
        print(f'{ym}: No files, skipping')
        continue

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

    common_dates = sorted(set(atl10_by_date.keys()) & set(cs2_by_date.keys()))

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
        matched = collocate_cs2_is2(df_cs2, df_is2, R_m=5000, min_pts=10)

        if len(matched) > 0:
            matched['date'] = date
            monthly_matched.append(matched)

        del df_is2, df_cs2, is2_frames, cs2_frames, matched

    if monthly_matched:
        df_month = pd.concat(monthly_matched, ignore_index=True)
        df_month['year_month'] = ym

        # Compute snow thickness
        hs, hs_unc = compute_snow_thickness(
            df_month['is2_fb_wm'].values,
            df_month['cs2_rfb'].values,
            is2_unc=df_month['is2_fb_unc_mean'].values,
            cs2_unc=0.03 * np.ones(len(df_month)),
            rho_s=330.0, rho_s_unc=50.0,
        )
        df_month['snow_thickness'] = hs
        df_month['snow_thickness_unc'] = hs_unc

        df_month.to_csv(out_file, index=False)
        diff = df_month.is2_fb_wm - df_month.cs2_rfb
        v = hs > 0
        print(f'{ym}: matched={len(df_month):>5d}  '
              f'hs_median={np.nanmedian(hs[v]):.3f} m  [saved]')
    else:
        print(f'{ym}: No matchups')

elapsed = time.time() - t_start
print(f'\nDone in {elapsed/60:.1f} minutes')

# Combine all monthly files
all_files = sorted(glob.glob(f'{OUT_DIR}/collocated_2*.csv'))
if all_files:
    df_all = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    df_all.to_csv(f'{HS_DIR}/snow_thickness_all_months.csv', index=False)
    print(f'Combined {len(all_files)} months: {len(df_all)} total matchups')