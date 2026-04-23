"""
utils.py — Core functions for Antarctic CRYO2ICE snow thickness retrieval.

Adapted from Fredensborg Hansen et al. (2024) Arctic pipeline.
Key changes: ATL10 V7 paths, Antarctic parameters, all 6 beams with IDW.

Author: Xinlong Liu, IMAS, University of Tasmania
Date:   March 2026
"""

import h5py
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.neighbors import BallTree
from pyproj import Transformer


# ============================================================
# 4.1  ATL10 V7 Reader — All 6 Beams
# (replaces Renee's read_data, Cell 17)
# ============================================================

def read_atl10_v7(fname, lat_filter=-50.0):
    """
    Read ICESat-2 ATL10 V7 sea ice freeboard — all 6 beams.

    Key differences from Renee's read_data() (Cell 17):
      - V7 path: /gtXx/freeboard_segment/  (not freeboard_beam_segment)
      - sc_orient is a dataset, read with [:], not an attribute
      - Quality flag filtering using beam_fb_quality_flag
      - Antarctic latitude filter (< -50) instead of Arctic (> 60)

    All 6 beams are loaded following Renee's approach to maximise
    spatial coverage for IDW. Each row is tagged with beam_id (1–6),
    beam_name (e.g. gt1r), and beam_type (strong/weak).

    Parameters
    ----------
    fname : str
        Path to ATL10 V7 HDF5 file.
    lat_filter : float
        Latitude threshold. Default -50 for Antarctic.

    Returns
    -------
    pd.DataFrame with columns:
        lat, lon, fb, fb_unc, delta_time, beam_id, beam_name, beam_type
    """
    with h5py.File(fname, 'r') as f:
        # Determine spacecraft orientation
        sc_orient = f['orbit_info/sc_orient'][0]

        # Map orientation to ordered beam list (same logic as Renee)
        if sc_orient == 0:  # Backward
            beam_order = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
            strong_set = {'gt1l', 'gt2l', 'gt3l'}
        elif sc_orient == 1:  # Forward
            beam_order = ['gt3r', 'gt3l', 'gt2r', 'gt2l', 'gt1r', 'gt1l']
            strong_set = {'gt1r', 'gt2r', 'gt3r'}
        else:
            print(f'WARNING: Transitioning orientation in {fname}, skipping')
            return pd.DataFrame()

        frames = []
        for beam_id, beam in enumerate(beam_order, start=1):
            # ATL10 V7: /gtXx/freeboard_segment/
            # (NOT freeboard_beam_segment as in Renee's r005)
            grp = f'{beam}/freeboard_segment'
            try:
                lat = f[f'{grp}/latitude'][:]
                lon = f[f'{grp}/longitude'][:]
                fb  = f[f'{grp}/beam_fb_height'][:]
                unc = f[f'{grp}/beam_fb_unc'][:]
                qf  = f[f'{grp}/beam_fb_quality_flag'][:]
                dt  = f[f'{grp}/delta_time'][:]
            except KeyError as e:
                print(f'  Beam {beam} not found: {e}')
                continue

            # Quality and range filtering
            valid = (qf <= 1) & (fb > -0.5) & (fb < 5.0) & np.isfinite(fb)
            # Antarctic latitude filter (Renee used > 60 for Arctic)
            valid &= (lat < lat_filter)

            beam_type = 'strong' if beam in strong_set else 'weak'

            df_beam = pd.DataFrame({
                'lat': lat[valid],
                'lon': lon[valid],
                'fb': fb[valid],
                'fb_unc': unc[valid],
                'delta_time': dt[valid],
                'beam_id': beam_id,
                'beam_name': beam,
                'beam_type': beam_type,
            })
            frames.append(df_beam)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# 4.2  CryoSat-2 Baseline-E Reader
# (replaces Renee's CS2 loading, Cell 24)
# ============================================================

def read_cs2_l2e(fname, lat_filter=-50.0):
    """
    Read CryoSat-2 Baseline-E L2 Enhanced product.

    Key differences from Renee's CS2 loading (Cell 24):
      - Antarctic latitude filter (< -50) not Arctic (> 60)
      - Only reads ESA Baseline-E; no CCI/LARM retrackers
      - Uses radar_freeboard_20_ku variable name

    Parameters
    ----------
    fname : str
        Path to CS2 L2E NetCDF file.
    lat_filter : float
        Latitude threshold. Default -50 for Antarctic.

    Returns
    -------
    pd.DataFrame with columns: lat, lon, rfb, time_tai
    """
    with Dataset(fname, 'r') as ds:
        lat = np.array(ds.variables['lat_poca_20_ku'][:])
        lon = np.array(ds.variables['lon_poca_20_ku'][:])
        rfb = np.array(ds.variables['radar_freeboard_20_ku'][:], dtype=float)
        time_tai = np.array(ds.variables['time_20_ku'][:])

        # Surface type: 2 = sea ice (same as Renee)
        try:
            surf_type = np.array(ds.variables['surf_type_20_ku'][:])
            sea_ice = (surf_type <= 2)
        except KeyError:
            sea_ice = np.ones(len(lat), dtype=bool)

    # Replace masked/fill values with NaN
    rfb[(rfb > 10) | (rfb < -1)] = np.nan

    # Antarctic latitude filter (Renee used > 60)
    valid = sea_ice & (lat < lat_filter) & np.isfinite(rfb)

    return pd.DataFrame({
        'lat': lat[valid],
        'lon': lon[valid],
        'rfb': rfb[valid],
        'time_tai': time_tai[valid],
    })


# ============================================================
# 4.3  Collocation — All 6 Beams with IDW
# (replaces Renee's BallTree, Cells 24 & 27)
# ============================================================

def collocate_cs2_is2(df_cs2, df_is2, R_m=5000, min_pts=10):
    """
    Bin IS2 observations (all 6 beams) around each CS2 point using BallTree.

    Direct adaptation of Renee's CRYO2ICE_identify_IS2_data2_semivariogram
    (Cell 27). All 6 IS2 beams are pooled for IDW averaging, exactly as
    Renee does.

    Key changes from Renee:
      - R_m default = 5000 m (Renee used 3500 m for Arctic)
      - Returns additional uncertainty columns for propagation
      - Beam composition tracked (n_strong, n_weak) for Section 3.14

    Parameters
    ----------
    df_cs2 : pd.DataFrame
        CS2 data with columns: lat, lon, rfb
    df_is2 : pd.DataFrame
        IS2 data with columns: lat, lon, fb, fb_unc, beam_type
        (all 6 beams from read_atl10_v7)
    R_m : float
        Search radius in metres. Default 5000 (Antarctic).
        Renee used 3500 for Arctic.
    min_pts : int
        Minimum IS2 points per CS2 location. Default 10 (same as Renee).

    Returns
    -------
    pd.DataFrame with collocated CS2-IS2 matchups including:
        lat, lon, cs2_rfb, is2_fb_wm (IDW mean), is2_fb_mean,
        is2_fb_std, is2_fb_unc_mean, n_is2, n_strong, n_weak
    """
    R_earth = 6_371_000  # metres
    R_rad = R_m / R_earth

    # Build BallTree on ALL IS2 positions (all 6 beams, same as Renee)
    tree = BallTree(
        np.deg2rad(df_is2[['lat', 'lon']].values),
        leaf_size=15,
        metric='haversine'
    )

    query_coords = np.deg2rad(df_cs2[['lat', 'lon']].values)

    # Radius query (same as Renee's approach)
    is_within, distances = tree.query_radius(
        query_coords, r=R_rad,
        count_only=False, return_distance=True
    )

    # Compute IDW means across all 6 beams (same logic as Renee Cell 27)
    results = {k: [] for k in [
        'lat', 'lon', 'cs2_rfb', 'is2_fb_wm',
        'is2_fb_mean', 'is2_fb_std', 'is2_fb_unc_mean',
        'n_is2', 'n_strong', 'n_weak',
    ]}

    for k in range(len(df_cs2)):
        idx = is_within[k]
        dist_m = distances[k] * R_earth

        if len(idx) >= min_pts:
            fb_vals = df_is2['fb'].iloc[idx].values
            unc_vals = df_is2['fb_unc'].iloc[idx].values
            beam_types = df_is2['beam_type'].iloc[idx].values
            valid = np.isfinite(fb_vals)

            if valid.sum() >= min_pts:
                # IDW weighting (identical to Renee)
                weights = 1.0 / (dist_m[valid] / R_m)
                w_mean = np.average(fb_vals[valid], weights=weights)

                results['lat'].append(df_cs2['lat'].iloc[k])
                results['lon'].append(df_cs2['lon'].iloc[k])
                results['cs2_rfb'].append(df_cs2['rfb'].iloc[k])
                results['is2_fb_wm'].append(w_mean)
                results['is2_fb_mean'].append(np.nanmean(fb_vals[valid]))
                results['is2_fb_std'].append(np.nanstd(fb_vals[valid]))
                results['is2_fb_unc_mean'].append(np.nanmean(unc_vals[valid]))
                results['n_is2'].append(int(valid.sum()))
                results['n_strong'].append(
                    int((beam_types[valid] == 'strong').sum()))
                results['n_weak'].append(
                    int((beam_types[valid] == 'weak').sum()))
                continue

        # Not enough points — skip (same as Renee's else: np.nan branch)

    return pd.DataFrame(results)


# ============================================================
# 4.4  Snow Thickness Retrieval
# (replaces Renee's Cells 47–48)
# ============================================================

def compute_snow_thickness(is2_fb, cs2_rfb, is2_unc=None, cs2_unc=None,
                           rho_s=330.0, rho_s_unc=70.0):
    """
    LaKu snow thickness from freeboard difference.
    Implements h_s = delta_f / eta_s following the Kacimi/Kwok
    CRYO2ICE freeboard-difference method.

    The Ku-band refractive-index correction eta_s is parameterised
    following Ulaby et al. (1986) as used in Kwok et al. (2020),
    Kacimi & Kwok (2020), and Fredensborg Hansen et al. (2024):

        eta_s = (1 + 0.51 * rho_s)^1.5

    where rho_s is in g/cm3. For rho_s = 0.33 g/cm3 (330 kg/m3):
    eta_s = 1.263, giving h_s = 0.792 * delta_f.

    This is numerically equivalent (within 1-2%) to the Tiuri (1984)
    dry-snow permittivity route (n_s = sqrt(1 + 1.7*rho + 0.7*rho^2)
    = 1.279, giving h_s = 0.782 * delta_f), and to the full
    Ulaby/Hallikainen (1986) formula (eps' = 1 + 1.5995*rho +
    1.861*rho^3, giving n_s = 1.263).

    Parameters
    ----------
    is2_fb : array
        ICESat-2 total freeboard (IDW mean from all 6 beams).
    cs2_rfb : array
        CryoSat-2 radar freeboard (from CS2 L2E).
    is2_unc, cs2_unc : array, optional
        Uncertainties for error propagation.
    rho_s : float
        Snow density in kg/m3. Default 330 (Antarctic).
    rho_s_unc : float
        Snow density uncertainty in kg/m3. Default 70 following
        Kacimi & Kwok (2020).

    Returns
    -------
    hs : array — Snow thickness in metres.
    hs_unc : array or None — Snow thickness uncertainty.
    """
    # Convert density to g/cm3
    rho_gcm3 = rho_s / 1000.0

    # Ku-band refractive index correction (Ulaby et al., 1986;
    # as parameterised in Kwok et al., 2020; Kacimi & Kwok, 2020)
    eta_s = (1.0 + 0.51 * rho_gcm3) ** 1.5
    # eta_s = 1.263 for rho_s = 330 kg/m3

    # Freeboard difference
    delta_f = is2_fb - cs2_rfb

    # Snow thickness: h_s = delta_f / eta_s
    hs = delta_f / eta_s

    # Uncertainty propagation (three-component quadrature)
    hs_unc = None
    if is2_unc is not None and cs2_unc is not None:
        # Derivative of eta_s with respect to rho_s (in g/cm3)
        # d(eta_s)/d(rho) = 1.5 * 0.51 * (1 + 0.51*rho)^0.5
        deta_drho = 1.5 * 0.51 * (1.0 + 0.51 * rho_gcm3) ** 0.5

        # Convert density uncertainty to g/cm3
        sigma_rho_gcm3 = rho_s_unc / 1000.0

        # Propagated uncertainty on eta_s
        sigma_eta = deta_drho * sigma_rho_gcm3

        # Total uncertainty: h_s = delta_f / eta_s
        # Component 1: IS2 freeboard uncertainty
        # Component 2: CS2 freeboard uncertainty
        # Component 3: density/refractive-index uncertainty
        hs_unc = np.sqrt(
            (is2_unc / eta_s) ** 2 +
            (cs2_unc / eta_s) ** 2 +
            (delta_f * sigma_eta / eta_s ** 2) ** 2
        )

    return hs, hs_unc


# ============================================================
# 4.5  AMSR2 Nearest-Neighbour Comparison
# (replaces Renee's Cell 36)
# ============================================================

def read_amsr2_sh(fname):
    """
    Read AMSR2 Southern Hemisphere 12km gridded data.

    Key differences from Renee: uses SpPolarGrid12km (SH)
    instead of NpPolarGrid12km (NH). Lat/lon read directly
    from file rather than computed from projection.
    Snow depth is 5-day composite (SNOWDEPTH_5DAY).
    """
    with h5py.File(fname, 'r') as f:
        grp = 'HDFEOS/GRIDS/SpPolarGrid12km'
        sic = f[f'{grp}/Data Fields/SI_12km_SH_ICECON_DAY'][:]
        sd  = f[f'{grp}/Data Fields/SI_12km_SH_SNOWDEPTH_5DAY'][:]
        lat = f[f'{grp}/lat'][:]
        lon = f[f'{grp}/lon'][:]

    return pd.DataFrame({
        'lat': lat.ravel(),
        'lon': lon.ravel(),
        'sea_ice_concentration': sic.ravel().astype(float),
        'snow_depth': sd.ravel().astype(float) / 100.0,  # cm to m
    })


def match_amsr2_nn(df_c2i, df_amsr2):
    """
    Nearest-neighbour AMSR2 matching at CRYO2ICE locations.
    Same approach as Renee's CRYO2ICE_AMSR2_NN (Cell 36),
    but using SH data.
    """
    tree = BallTree(
        np.deg2rad(df_amsr2[['lat', 'lon']].values),
        leaf_size=15,
        metric='haversine'
    )

    query = np.deg2rad(df_c2i[['lat', 'lon']].values)
    distances, indices = tree.query(query, k=1)

    df_out = df_c2i.copy()
    df_out['AMSR2_SIC'] = df_amsr2['sea_ice_concentration'].iloc[
        indices.ravel()].values
    df_out['AMSR2_snow_depth'] = df_amsr2['snow_depth'].iloc[
        indices.ravel()].values

    return df_out
