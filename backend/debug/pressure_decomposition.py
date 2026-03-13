"""Pressure decomposition: reconstruct SLP from observed T, compare vs observed SLP.

Decomposes the SLP into its component terms:
  - Hadley cell (dp_hadley): ITCZ low, subtropical highs, subpolar lows, polar highs
  - Thermal (dp_thermal): zonal anomalies from column temperature

The wind model computes SLP WITHOUT elevation (elevation_m is not passed to
compute_pressure), so:  SLP = MEAN_P + dp_thermal + dp_hadley.
This script reconstructs SLP the same way using observed temperatures.

Sweeps key parameters (THERMAL_PRESSURE_COEFFICIENT, DP_SUBTROPICS, DP_SUBPOLAR,
Rossby smoothing radius) and diagnoses which components are responsible for
errors at key synoptic features (Icelandic Low, Siberian High, etc.).

Usage:
    DATA_DIR=data PYTHONPATH=backend uv run python backend/debug/pressure_decomposition.py
"""
import numpy as np
import xarray as xr
import pooch
from pathlib import Path
import os

from climate_sim.physics.atmosphere.pressure import (
    _smooth_temperature_field, _get_latitude_centers,
    THERMAL_PRESSURE_COEFFICIENT,
    hadley_pressure_anomaly,
    DP_SUBTROPICS, DP_SUBPOLAR, DP_ITCZ, DP_POLES,
    SIGMA_ITCZ, SIGMA_SUBTROPICS, SIGMA_SUBPOLAR, SIGMA_POLES,
    SUBTROPICS_ITCZ_COUPLING,
    LAT_SUBTROPICS_BASE,
    _rossby_radius_km,
)
from climate_sim.core.math_core import area_weighted_mean, spherical_cell_area
from climate_sim.physics.atmosphere.hadley import (
    compute_itcz_latitude, LAT_POLES, LAT_SUBPOLAR,
)
from climate_sim.data.landmask import compute_land_mask
from climate_sim.data.constants import (
    BOUNDARY_LAYER_HEIGHT_M, ATMOSPHERE_LAYER_HEIGHT_M,
    R_EARTH_METERS,
)

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw" / "noaa"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "air_plev": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/air.mon.mean.nc",
}
BASELINE_START = "1981-01-01"
BASELINE_END = "2010-12-31"

PLOT_DIR = "data/pressure_decomposition"
os.makedirs(PLOT_DIR, exist_ok=True)

MEAN_P = 101325.0


def fetch(url: str, outdir: Path) -> Path:
    return Path(pooch.retrieve(url=url, known_hash=None, path=str(outdir)))


def to_0360(ds: xr.Dataset) -> xr.Dataset:
    if float(ds["lon"].min()) < 0:
        ds = ds.assign_coords(lon=(ds["lon"] % 360)).sortby("lon")
    return ds


# ── Grid setup ─────────────────────────────────────────────────────────
nlat, nlon = 36, 72
lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)
lon = np.linspace(180/nlon, 360 - 180/nlon, nlon)
lon2d, lat2d = np.meshgrid(lon, lat)
land_mask = compute_land_mask(lon2d, lat2d)

lat_deg = _get_latitude_centers(nlat)
lat_rad = np.deg2rad(lat_deg)
cos_lat = np.clip(np.cos(lat_rad), 1e-6, None)
weights = np.broadcast_to(cos_lat[:, None], (nlat, nlon)).astype(float)
lat_rad_2d = np.deg2rad(lat2d)

cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)

# Mass-fraction weighting for column temperature
total_h = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M
bl_w = BOUNDARY_LAYER_HEIGHT_M / total_h
atm_w = ATMOSPHERE_LAYER_HEIGHT_M / total_h


def regrid_to_5deg(field, lat_src, lon_src):
    if field.ndim == 3:
        return np.stack([regrid_to_5deg(field[m], lat_src, lon_src)
                         for m in range(field.shape[0])])
    out = np.full((nlat, nlon), np.nan)
    for j in range(nlat):
        for i in range(nlon):
            j1 = np.argmin(np.abs(lat_src - lat[j]))
            i1 = np.argmin(np.abs(lon_src - lon[i]))
            j_lo, j_hi = max(0, j1 - 2), min(len(lat_src), j1 + 3)
            i_lo, i_hi = max(0, i1 - 2), min(len(lon_src), i1 + 3)
            out[j, i] = np.nanmean(field[j_lo:j_hi, i_lo:i_hi])
    return out


# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
ds_plev = to_0360(xr.open_dataset(fetch(URLS["air_plev"], RAW_DIR)))
air_plev = ds_plev["air"].sel(time=slice(BASELINE_START, BASELINE_END))
t925 = air_plev.sel(level=925).groupby("time.month").mean("time").values
t500 = air_plev.sel(level=500).groupby("time.month").mean("time").values
lat_plev = ds_plev["lat"].values
lon_plev = ds_plev["lon"].values

obs_hp = xr.open_dataset("data/processed/ref_humidity_precip_1deg_1981-2010.nc")
slp_obs_1deg = obs_hp["slp_clim"].values
lat_obs = obs_hp["lat"].values
lon_obs = obs_hp["lon"].values

print("Regridding to 5°...")
t925_5 = regrid_to_5deg(t925, lat_plev, lon_plev)
t500_5 = regrid_to_5deg(t500, lat_plev, lon_plev)
slp_obs_5 = regrid_to_5deg(slp_obs_1deg, lat_obs, lon_obs)


# ── Precompute column T, smoothed T, dp_thermal, ITCZ per month ──────
def compute_slp_components(beta=THERMAL_PRESSURE_COEFFICIENT,
                           dp_sub=DP_SUBTROPICS,
                           dp_subpolar=DP_SUBPOLAR,
                           dp_itcz=DP_ITCZ,
                           dp_poles=DP_POLES,
                           smoothing_km=None):
    """Compute SLP and its component terms for all 12 months.

    Matches the wind model's compute_pressure() with elevation_m=None:
        SLP = MEAN_P + dp_thermal + dp_hadley  (renormalised)

    Returns dict with arrays of shape (12, nlat, nlon):
        slp_total, dp_thermal, dp_hadley, col_T_smooth, col_T_raw
    Also returns itcz_rad (12, nlon).
    """
    dp_thermal = np.zeros((12, nlat, nlon))
    dp_hadley_arr = np.zeros((12, nlat, nlon))
    col_T_smooth_arr = np.zeros((12, nlat, nlon))
    col_T_raw_arr = np.zeros((12, nlat, nlon))
    slp_total = np.zeros((12, nlat, nlon))
    itcz_rad = np.zeros((12, nlon))

    for m in range(12):
        t9 = np.nan_to_num(t925_5[m], nan=np.nanmean(t925_5[m]))
        t5 = np.nan_to_num(t500_5[m], nan=np.nanmean(t500_5[m]))
        col_K = (t9 + 273.15) * bl_w + (t5 + 273.15) * atm_w
        col_T_raw_arr[m] = col_K

        if smoothing_km is not None:
            ts = _smooth_temperature_field(col_K, lat_deg,
                                           smoothing_length_km=smoothing_km)
        else:
            ts = _smooth_temperature_field(col_K, lat_deg)
        ts += area_weighted_mean(col_K, weights) - area_weighted_mean(ts, weights)
        col_T_smooth_arr[m] = ts

        # Thermal pressure: zonal anomaly only (matches compute_pressure)
        zonal_mean = np.mean(ts, axis=1, keepdims=True)
        dT = ts - zonal_mean
        dp = -beta * dT
        dp -= area_weighted_mean(dp, weights)
        dp_thermal[m] = dp

        # ITCZ
        itcz_rad[m] = compute_itcz_latitude(col_K, lat2d, cell_areas)

        # Hadley pressure anomaly (with overridden amplitudes)
        itcz_2d = np.broadcast_to(itcz_rad[m][np.newaxis, :], (nlat, nlon))

        dp_h = _hadley_with_params(lat_rad_2d, itcz_2d,
                                   dp_itcz=dp_itcz, dp_sub=dp_sub,
                                   dp_subpolar=dp_subpolar, dp_poles=dp_poles)
        dp_h -= area_weighted_mean(dp_h, weights)
        dp_hadley_arr[m] = dp_h

        # Total SLP: no elevation (wind model passes elevation_m=None)
        # p_orog = MEAN_P when elevation=0 everywhere
        slp = MEAN_P + dp_thermal[m] + dp_h
        slp *= MEAN_P / area_weighted_mean(slp, weights)
        slp_total[m] = slp

    return {
        "slp_total": slp_total,
        "dp_thermal": dp_thermal,
        "dp_hadley": dp_hadley_arr,
        "col_T_smooth": col_T_smooth_arr,
        "col_T_raw": col_T_raw_arr,
        "itcz_rad": itcz_rad,
    }


def _hadley_with_params(lat_rad, itcz_rad_2d,
                         dp_itcz=DP_ITCZ, dp_sub=DP_SUBTROPICS,
                         dp_subpolar=DP_SUBPOLAR, dp_poles=DP_POLES):
    """Hadley pressure anomaly with overridable amplitudes."""
    lat_subtrop_north = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad_2d
    lat_subtrop_south = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad_2d

    dp_i = dp_itcz * np.exp(-((lat_rad - itcz_rad_2d) / SIGMA_ITCZ) ** 2)
    dp_s = (
        dp_sub * np.exp(-((lat_rad - lat_subtrop_south) / SIGMA_SUBTROPICS) ** 2)
        + dp_sub * np.exp(-((lat_rad - lat_subtrop_north) / SIGMA_SUBTROPICS) ** 2)
    )
    dp_sp = dp_subpolar * (
        np.exp(-((lat_rad + LAT_SUBPOLAR) / SIGMA_SUBPOLAR) ** 2)
        + np.exp(-((lat_rad - LAT_SUBPOLAR) / SIGMA_SUBPOLAR) ** 2)
    )
    dp_p = dp_poles * (
        np.exp(-((lat_rad + LAT_POLES) / SIGMA_POLES) ** 2)
        + np.exp(-((lat_rad - LAT_POLES) / SIGMA_POLES) ** 2)
    )
    return dp_i + dp_s + dp_sp + dp_p


# ── Key synoptic features for regional analysis ──────────────────────
# (name, lat_center, lon_center, lat_range, lon_range, season_month_index)
# Month indices: 0=Jan, 6=Jul
FEATURES = {
    "Icelandic Low (DJF)":     {"lat": 62.5, "lon": 337.5, "dlat": 10, "dlon": 20, "month": 0},
    "Aleutian Low (DJF)":      {"lat": 52.5, "lon": 177.5, "dlat": 10, "dlon": 20, "month": 0},
    "Siberian High (DJF)":     {"lat": 52.5, "lon":  92.5, "dlat": 10, "dlon": 30, "month": 0},
    "Canadian High (DJF)":     {"lat": 62.5, "lon": 257.5, "dlat": 10, "dlon": 20, "month": 0},
    "Azores High (JJA)":       {"lat": 37.5, "lon": 332.5, "dlat": 10, "dlon": 20, "month": 6},
    "N Pacific High (JJA)":    {"lat": 32.5, "lon": 222.5, "dlat": 10, "dlon": 20, "month": 6},
    "S Atlantic High (JJA)":   {"lat":-27.5, "lon": 352.5, "dlat": 10, "dlon": 20, "month": 6},
    "S Indian High (JJA)":     {"lat":-32.5, "lon":  72.5, "dlat": 10, "dlon": 20, "month": 6},
    "Saharan Low (JJA)":       {"lat": 27.5, "lon":   7.5, "dlat": 10, "dlon": 20, "month": 6},
    "Indian Monsoon Low (JJA)":{"lat": 27.5, "lon":  72.5, "dlat": 10, "dlon": 20, "month": 6},
}


def region_mask(lat_c, lon_c, dlat, dlon):
    """Return boolean mask for a lat/lon box on the 5° grid."""
    lat_mask = (lat >= lat_c - dlat) & (lat <= lat_c + dlat)
    lon_c_wrap = lon_c % 360
    lon_lo = (lon_c_wrap - dlon) % 360
    lon_hi = (lon_c_wrap + dlon) % 360
    if lon_lo < lon_hi:
        lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    else:
        lon_mask = (lon >= lon_lo) | (lon <= lon_hi)
    return lat_mask[:, None] & lon_mask[None, :]


def slp_metrics(slp_sim, slp_obs):
    """Return pattern correlation, RMSE (hPa), bias (hPa) for annual means."""
    sim_ann = np.nanmean(slp_sim, axis=0)
    obs_ann = np.nanmean(slp_obs, axis=0)
    valid = np.isfinite(obs_ann) & np.isfinite(sim_ann)
    corr = np.corrcoef(sim_ann[valid], obs_ann[valid])[0, 1]
    rmse = np.sqrt(np.mean((sim_ann[valid] - obs_ann[valid])**2)) / 100.0
    bias = np.mean(sim_ann[valid] - obs_ann[valid]) / 100.0
    return {"corr": corr, "rmse_hpa": rmse, "bias_hpa": bias}


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Baseline decomposition with default parameters
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*120)
print("  PART 1: SLP decomposition — default parameters")
print("="*120)

baseline = compute_slp_components()

m_base = slp_metrics(baseline["slp_total"], slp_obs_5)
print(f"\nBaseline metrics: corr={m_base['corr']:.3f}  RMSE={m_base['rmse_hpa']:.1f} hPa"
      f"  bias={m_base['bias_hpa']:+.1f} hPa")
print(f"  THERMAL_PRESSURE_COEFFICIENT = {THERMAL_PRESSURE_COEFFICIENT}")
print(f"  DP_SUBTROPICS = {DP_SUBTROPICS}  DP_SUBPOLAR = {DP_SUBPOLAR}")
print(f"  DP_ITCZ = {DP_ITCZ}  DP_POLES = {DP_POLES}")

# Component magnitudes (annual mean, area-weighted std)
for comp_name, arr in [("dp_thermal", baseline["dp_thermal"]),
                        ("dp_hadley", baseline["dp_hadley"])]:
    ann = np.mean(arr, axis=0)
    std = np.sqrt(area_weighted_mean(ann**2, weights))
    mn = np.min(ann)
    mx = np.max(ann)
    print(f"  {comp_name:>20}: std={std/100:.1f} hPa  range=[{mn/100:+.1f}, {mx/100:+.1f}] hPa")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Regional feature breakdown
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 2: Regional SLP feature breakdown — obs vs components")
print(f"{'='*120}")

print(f"\n{'Feature':>30} {'Month':>5} | {'Obs':>7} {'Recon':>7} {'Err':>7} | "
      f"{'Hadley':>7} {'Therm':>7}")
print("-" * 85)

for feat_name, feat in FEATURES.items():
    mask = region_mask(feat["lat"], feat["lon"], feat["dlat"], feat["dlon"])
    m = feat["month"]
    if mask.sum() == 0:
        continue

    obs_val = np.nanmean(slp_obs_5[m][mask]) / 100
    rec_val = np.mean(baseline["slp_total"][m][mask]) / 100
    had_val = np.mean(baseline["dp_hadley"][m][mask]) / 100
    thm_val = np.mean(baseline["dp_thermal"][m][mask]) / 100

    print(f"{feat_name:>30} {m+1:>5} | {obs_val:7.1f} {rec_val:7.1f} {rec_val-obs_val:+7.1f} | "
          f"{had_val:+7.1f} {thm_val:+7.1f}")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Parameter sweep — THERMAL_PRESSURE_COEFFICIENT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 3: THERMAL_PRESSURE_COEFFICIENT sweep")
print(f"{'='*120}")

beta_values = [50, 100, 150, 200, 250, 300, 400]
print(f"\n{'beta':>6} | {'corr':>6} {'RMSE':>7} {'bias':>7} | "
      f"{'Icel err':>9} {'Sib err':>9} {'Azor err':>9} {'Sah err':>9}")
print("-" * 90)

best_beta = None
best_beta_corr = -999

for beta in beta_values:
    result = compute_slp_components(beta=beta)
    m = slp_metrics(result["slp_total"], slp_obs_5)

    # Key feature errors (hPa)
    errors = {}
    for feat_key in ["Icelandic Low (DJF)", "Siberian High (DJF)",
                     "Azores High (JJA)", "Saharan Low (JJA)"]:
        feat = FEATURES[feat_key]
        mask = region_mask(feat["lat"], feat["lon"], feat["dlat"], feat["dlon"])
        mi = feat["month"]
        obs_val = np.nanmean(slp_obs_5[mi][mask]) / 100
        rec_val = np.mean(result["slp_total"][mi][mask]) / 100
        errors[feat_key] = rec_val - obs_val

    if m["corr"] > best_beta_corr:
        best_beta_corr = m["corr"]
        best_beta = beta

    print(f"{beta:>6} | {m['corr']:6.3f} {m['rmse_hpa']:7.1f} {m['bias_hpa']:+7.1f} | "
          f"{errors['Icelandic Low (DJF)']:+9.1f} {errors['Siberian High (DJF)']:+9.1f} "
          f"{errors['Azores High (JJA)']:+9.1f} {errors['Saharan Low (JJA)']:+9.1f}")

print(f"\nBest beta: {best_beta} (corr={best_beta_corr:.3f})")


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Parameter sweep — DP_SUBTROPICS and DP_SUBPOLAR
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 4: Hadley amplitude sweep (DP_SUBTROPICS × DP_SUBPOLAR)")
print(f"{'='*120}")

dp_sub_values = [400, 600, 800, 1000, 1200]
dp_subpolar_values = [-600, -900, -1200, -1500, -1800]

print(f"\n{'dp_sub':>7} {'dp_spol':>8} | {'corr':>6} {'RMSE':>7} {'bias':>7} | "
      f"{'Icel err':>9} {'Azor err':>9} {'NPac err':>9}")
print("-" * 95)

best_hadley = None
best_hadley_corr = -999

for dp_sub in dp_sub_values:
    for dp_spol in dp_subpolar_values:
        result = compute_slp_components(dp_sub=dp_sub, dp_subpolar=dp_spol)
        m = slp_metrics(result["slp_total"], slp_obs_5)

        errors = {}
        for feat_key in ["Icelandic Low (DJF)", "Azores High (JJA)", "N Pacific High (JJA)"]:
            feat = FEATURES[feat_key]
            mask = region_mask(feat["lat"], feat["lon"], feat["dlat"], feat["dlon"])
            mi = feat["month"]
            obs_val = np.nanmean(slp_obs_5[mi][mask]) / 100
            rec_val = np.mean(result["slp_total"][mi][mask]) / 100
            errors[feat_key] = rec_val - obs_val

        if m["corr"] > best_hadley_corr:
            best_hadley_corr = m["corr"]
            best_hadley = (dp_sub, dp_spol)

        print(f"{dp_sub:>7} {dp_spol:>8} | {m['corr']:6.3f} {m['rmse_hpa']:7.1f} {m['bias_hpa']:+7.1f} | "
              f"{errors['Icelandic Low (DJF)']:+9.1f} {errors['Azores High (JJA)']:+9.1f} "
              f"{errors['N Pacific High (JJA)']:+9.1f}")

print(f"\nBest Hadley: dp_sub={best_hadley[0]}, dp_spol={best_hadley[1]} (corr={best_hadley_corr:.3f})")


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Rossby smoothing radius sweep
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 5: Rossby smoothing radius sweep (fixed smoothing_length_km)")
print(f"{'='*120}")

# Show the default Rossby radii for context
rossby = _rossby_radius_km(lat_deg)
print("\nDefault Rossby deformation radius (km) by latitude:")
for j in range(0, nlat, 4):
    print(f"  {lat[j]:+6.1f}°: L_R = {rossby[j]:6.0f} km, σ = L_R/3 = {rossby[j]/3:.0f} km")

smooth_values = [200, 400, 600, 800, 1000, 1500, None]  # None = default Rossby-based
print(f"\n{'smooth_km':>10} | {'corr':>6} {'RMSE':>7} {'bias':>7} | "
      f"{'Sib err':>9} {'Sah err':>9} {'Ind err':>9}")
print("-" * 80)

for smooth_km in smooth_values:
    result = compute_slp_components(smoothing_km=smooth_km)
    m = slp_metrics(result["slp_total"], slp_obs_5)

    errors = {}
    for feat_key in ["Siberian High (DJF)", "Saharan Low (JJA)", "Indian Monsoon Low (JJA)"]:
        feat = FEATURES[feat_key]
        mask = region_mask(feat["lat"], feat["lon"], feat["dlat"], feat["dlon"])
        mi = feat["month"]
        obs_val = np.nanmean(slp_obs_5[mi][mask]) / 100
        rec_val = np.mean(result["slp_total"][mi][mask]) / 100
        errors[feat_key] = rec_val - obs_val

    label = f"{smooth_km}" if smooth_km is not None else "Rossby"
    print(f"{label:>10} | {m['corr']:6.3f} {m['rmse_hpa']:7.1f} {m['bias_hpa']:+7.1f} | "
          f"{errors['Siberian High (DJF)']:+9.1f} {errors['Saharan Low (JJA)']:+9.1f} "
          f"{errors['Indian Monsoon Low (JJA)']:+9.1f}")


# ═══════════════════════════════════════════════════════════════════════
# PART 6: Zonal mean SLP profiles — obs vs reconstruction
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 6: Zonal mean SLP (hPa) — obs vs reconstruction, Jan/Jul/Annual")
print(f"{'='*120}")

print(f"{'Lat':>6} | {'ANN obs':>8} {'ANN rec':>8} {'diff':>7} | "
      f"{'JAN obs':>8} {'JAN rec':>8} {'diff':>7} | "
      f"{'JUL obs':>8} {'JUL rec':>8} {'diff':>7}")
print("-" * 105)

obs_ann_zm = np.nanmean(np.nanmean(slp_obs_5, axis=0), axis=1) / 100
rec_ann_zm = np.mean(np.mean(baseline["slp_total"], axis=0), axis=1) / 100

for j, la in enumerate(lat):
    obs_jan = np.nanmean(slp_obs_5[0, j]) / 100
    rec_jan = np.mean(baseline["slp_total"][0, j]) / 100
    obs_jul = np.nanmean(slp_obs_5[6, j]) / 100
    rec_jul = np.mean(baseline["slp_total"][6, j]) / 100

    print(f"{la:6.1f} | {obs_ann_zm[j]:8.1f} {rec_ann_zm[j]:8.1f} {rec_ann_zm[j]-obs_ann_zm[j]:+7.1f} | "
          f"{obs_jan:8.1f} {rec_jan:8.1f} {rec_jan-obs_jan:+7.1f} | "
          f"{obs_jul:8.1f} {rec_jul:8.1f} {rec_jul-obs_jul:+7.1f}")


# ═══════════════════════════════════════════════════════════════════════
# PART 7: Component zonal means — what drives the meridional SLP profile
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 7: Component zonal means (annual, hPa)")
print(f"{'='*120}")

print(f"{'Lat':>6} | {'dp_hadley':>10} {'dp_therm':>10} {'total-MP':>10} | {'obs-MP':>10}")
print("-" * 65)

had_ann_zm = np.mean(np.mean(baseline["dp_hadley"], axis=0), axis=1) / 100
thm_ann_zm = np.mean(np.mean(baseline["dp_thermal"], axis=0), axis=1) / 100

for j, la in enumerate(lat):
    total_anom = rec_ann_zm[j] - MEAN_P / 100
    obs_anom = obs_ann_zm[j] - MEAN_P / 100
    print(f"{la:6.1f} | {had_ann_zm[j]:+10.2f} {thm_ann_zm[j]:+10.2f} "
          f"{total_anom:+10.2f} | {obs_anom:+10.2f}")


# ═══════════════════════════════════════════════════════════════════════
# PART 8: ITCZ position and Hadley sub-components
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 8: ITCZ position and Hadley cell sub-components")
print(f"{'='*120}")

print("\nITCZ latitude (degrees) by month:")
for m in range(12):
    itcz_mean = np.rad2deg(np.mean(baseline["itcz_rad"][m]))
    itcz_min = np.rad2deg(np.min(baseline["itcz_rad"][m]))
    itcz_max = np.rad2deg(np.max(baseline["itcz_rad"][m]))
    print(f"  Month {m+1:2d}: mean={itcz_mean:+6.1f}°  range=[{itcz_min:+6.1f}°, {itcz_max:+6.1f}°]")

# Show individual Hadley sub-components for Jan and Jul
print(f"\nHadley sub-component zonal means (hPa):")
print(f"{'Lat':>6} | {'JAN ITCZ':>9} {'sub_hi':>7} {'sub_lo':>7} {'polar':>7} {'total':>7} | "
      f"{'JUL ITCZ':>9} {'sub_hi':>7} {'sub_lo':>7} {'polar':>7} {'total':>7}")
print("-" * 105)

for midx, mname in [(0, "Jan"), (6, "Jul")]:
    pass  # Compute sub-components below

# Recompute sub-components separately for display
for j, la in enumerate(lat):
    line = f"{la:6.1f} |"
    for midx in [0, 6]:
        itcz_2d = np.broadcast_to(baseline["itcz_rad"][midx][np.newaxis, :], (nlat, nlon))
        lat_subtrop_north = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_2d
        lat_subtrop_south = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_2d

        dp_i = DP_ITCZ * np.exp(-((lat_rad_2d - itcz_2d) / SIGMA_ITCZ) ** 2)
        dp_s = (
            DP_SUBTROPICS * np.exp(-((lat_rad_2d - lat_subtrop_south) / SIGMA_SUBTROPICS) ** 2)
            + DP_SUBTROPICS * np.exp(-((lat_rad_2d - lat_subtrop_north) / SIGMA_SUBTROPICS) ** 2)
        )
        dp_sp = DP_SUBPOLAR * (
            np.exp(-((lat_rad_2d + LAT_SUBPOLAR) / SIGMA_SUBPOLAR) ** 2)
            + np.exp(-((lat_rad_2d - LAT_SUBPOLAR) / SIGMA_SUBPOLAR) ** 2)
        )
        dp_p = DP_POLES * (
            np.exp(-((lat_rad_2d + LAT_POLES) / SIGMA_POLES) ** 2)
            + np.exp(-((lat_rad_2d - LAT_POLES) / SIGMA_POLES) ** 2)
        )

        vals = [np.mean(dp_i[j])/100, np.mean(dp_s[j])/100,
                np.mean(dp_sp[j])/100, np.mean(dp_p[j])/100,
                np.mean(baseline["dp_hadley"][midx, j])/100]
        line += f" {vals[0]:+9.2f} {vals[1]:+7.2f} {vals[2]:+7.2f} {vals[3]:+7.2f} {vals[4]:+7.2f} |"
    print(line)


# ═══════════════════════════════════════════════════════════════════════
# PART 9: Smoothing effect on thermal anomalies at key locations
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print("  PART 9: Column T raw vs smoothed at key locations")
print(f"{'='*120}")

print(f"\n{'Feature':>30} {'Month':>5} | {'T_raw':>7} {'T_smooth':>9} {'dT_zonal':>9} "
      f"{'dp_th':>7} {'dp_th/beta':>10}")
print("-" * 95)

for feat_name, feat in FEATURES.items():
    mask = region_mask(feat["lat"], feat["lon"], feat["dlat"], feat["dlon"])
    m = feat["month"]
    if mask.sum() == 0:
        continue

    t_raw = np.mean(baseline["col_T_raw"][m][mask])
    t_smooth = np.mean(baseline["col_T_smooth"][m][mask])

    # Zonal anomaly of smoothed T at this location
    j_center = np.argmin(np.abs(lat - feat["lat"]))
    zonal_mean_smooth = np.mean(baseline["col_T_smooth"][m, j_center])
    dt_zon = t_smooth - zonal_mean_smooth

    dp_th = np.mean(baseline["dp_thermal"][m][mask])

    print(f"{feat_name:>30} {m+1:>5} | {t_raw:7.1f} {t_smooth:9.1f} {dt_zon:+9.2f} "
          f"{dp_th/100:+7.2f} {dt_zon:+10.2f}")


# ── Plots ────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: Zonal mean SLP — obs vs recon, Jan/Jul/Annual ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Zonal mean SLP: obs vs reconstructed (default params)", fontsize=14)

    panels = [(0, "January"), (6, "July"), (None, "Annual")]
    for ax, (midx, mname) in zip(axes, panels):
        if midx is not None:
            obs_zm = np.nanmean(slp_obs_5[midx], axis=1) / 100
            rec_zm = np.mean(baseline["slp_total"][midx], axis=1) / 100
            had_zm = np.mean(baseline["dp_hadley"][midx], axis=1) / 100
            thm_zm = np.mean(baseline["dp_thermal"][midx], axis=1) / 100
        else:
            obs_zm = obs_ann_zm
            rec_zm = rec_ann_zm
            had_zm = had_ann_zm
            thm_zm = thm_ann_zm

        ax.plot(lat, obs_zm, 'k-', lw=2.5, label="Obs SLP")
        ax.plot(lat, rec_zm, 'r-', lw=1.5, label="Reconstructed")
        ax.axhline(MEAN_P/100, color='gray', lw=0.5, ls='--')
        ax.set_xlabel("Latitude")
        ax.set_ylabel("hPa")
        ax.set_title(mname)
        ax.legend(fontsize=8)
        ax.set_xlim(-90, 90)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/zonal_slp_obs_vs_recon.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {PLOT_DIR}/zonal_slp_obs_vs_recon.png")
    plt.close()

    # ── Figure 2: Zonal mean SLP components ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Zonal mean SLP components (hPa anomaly from global mean)", fontsize=14)

    for ax, (midx, mname) in zip(axes, panels):
        if midx is not None:
            had_zm = np.mean(baseline["dp_hadley"][midx], axis=1) / 100
            thm_zm = np.mean(baseline["dp_thermal"][midx], axis=1) / 100
            obs_anom = np.nanmean(slp_obs_5[midx], axis=1) / 100 - MEAN_P / 100
        else:
            had_zm = had_ann_zm
            thm_zm = thm_ann_zm
            obs_anom = obs_ann_zm - MEAN_P / 100

        ax.plot(lat, obs_anom, 'k-', lw=2.5, label="Obs anomaly")
        ax.plot(lat, had_zm, 'b-', lw=1.5, label="Hadley")
        ax.plot(lat, thm_zm, 'r-', lw=1.5, label="Thermal")
        ax.plot(lat, had_zm + thm_zm, 'm--', lw=1, label="Sum")
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("hPa")
        ax.set_title(mname)
        ax.legend(fontsize=8)
        ax.set_xlim(-90, 90)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/zonal_slp_components.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/zonal_slp_components.png")
    plt.close()

    # ── Figure 3: SLP maps — obs vs recon (Jan, Jul) ─────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("SLP maps: obs vs reconstructed (hPa)", fontsize=14)

    for row, (midx, mname) in enumerate([(0, "January"), (6, "July")]):
        obs_m = np.nan_to_num(slp_obs_5[midx], nan=MEAN_P) / 100
        rec_m = baseline["slp_total"][midx] / 100
        diff_m = rec_m - obs_m

        vmin, vmax = 996, 1030
        im0 = axes[row, 0].pcolormesh(lon, lat, obs_m, cmap="RdBu_r",
                                       vmin=vmin, vmax=vmax, shading="auto")
        axes[row, 0].set_title(f"Obs SLP — {mname}")
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8, label="hPa")

        im1 = axes[row, 1].pcolormesh(lon, lat, rec_m, cmap="RdBu_r",
                                       vmin=vmin, vmax=vmax, shading="auto")
        axes[row, 1].set_title(f"Reconstructed — {mname}")
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8, label="hPa")

        im2 = axes[row, 2].pcolormesh(lon, lat, diff_m, cmap="RdBu_r",
                                       vmin=-15, vmax=15, shading="auto")
        axes[row, 2].set_title(f"Recon − Obs — {mname}")
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8, label="hPa")

    for ax in axes.flat:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slp_maps_obs_vs_recon.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/slp_maps_obs_vs_recon.png")
    plt.close()

    # ── Figure 4: Component maps (July) ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("SLP components — July (hPa)", fontsize=14)

    components = [
        ("dp_hadley", baseline["dp_hadley"][6] / 100, "Hadley cell", "RdBu_r", 15),
        ("dp_thermal", baseline["dp_thermal"][6] / 100, "Thermal (zonal anomaly)", "RdBu_r", 8),
        ("Recon SLP", baseline["slp_total"][6] / 100 - MEAN_P / 100, "Reconstructed SLP anomaly", "RdBu_r", 15),
        ("Recon − Obs", baseline["slp_total"][6] / 100 - np.nan_to_num(slp_obs_5[6], nan=MEAN_P) / 100,
         "Error", "RdBu_r", 15),
    ]

    for ax, (cname, cdata, ctitle, cmap, vlim) in zip(axes.flat, components):
        im = ax.pcolormesh(lon, lat, cdata, cmap=cmap, vmin=-vlim, vmax=vlim, shading="auto")
        ax.set_title(ctitle)
        plt.colorbar(im, ax=ax, shrink=0.8, label="hPa")

    for ax in axes.flat:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/slp_component_maps_july.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/slp_component_maps_july.png")
    plt.close()

    # ── Figure 5: Beta sweep summary ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("THERMAL_PRESSURE_COEFFICIENT sensitivity", fontsize=14)

    corrs = []
    rmses = []
    for beta in beta_values:
        result = compute_slp_components(beta=beta)
        m = slp_metrics(result["slp_total"], slp_obs_5)
        corrs.append(m["corr"])
        rmses.append(m["rmse_hpa"])

    ax1.plot(beta_values, corrs, 'bo-', lw=2)
    ax1.set_xlabel("beta (Pa/K)")
    ax1.set_ylabel("Pattern correlation")
    ax1.set_title("Pattern correlation vs beta")
    ax1.grid(True, alpha=0.3)

    ax2.plot(beta_values, rmses, 'ro-', lw=2)
    ax2.set_xlabel("beta (Pa/K)")
    ax2.set_ylabel("RMSE (hPa)")
    ax2.set_title("RMSE vs beta")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/beta_sweep.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/beta_sweep.png")
    plt.close()

except ImportError:
    print("matplotlib not available, skipping plots")

print("\nDone!")
