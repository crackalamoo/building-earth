"""Wind decomposition: reconstruct surface winds from observed T, q, SLP.

Tests our wind computation (geostrophic + Ekman drag) with:
  - Observed SLP (gold standard)
  - Reconstructed SLP with varying dp_base (subtropical high amplitude)

Also diagnoses turning angles, speed ratios, and drag coefficient sensitivity.

Usage:
    DATA_DIR=data PYTHONPATH=backend uv run python backend/debug/wind_decomposition.py
"""

import numpy as np
import xarray as xr
import pooch
from pathlib import Path
import os

from climate_sim.physics.atmosphere.pressure import (
    _smooth_temperature_field,
    _get_latitude_centers,
    THERMAL_PRESSURE_COEFFICIENT,
    hadley_pressure_anomaly,
)
from climate_sim.core.math_core import area_weighted_mean
from climate_sim.physics.atmosphere.hadley import compute_itcz_latitude
from climate_sim.data.elevation import compute_cell_elevation
from climate_sim.data.landmask import compute_land_mask
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.data.constants import (
    BOUNDARY_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_HEIGHT_M,
    R_EARTH_METERS,
    GAS_CONSTANT_J_KG_K,
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

PLOT_DIR = "data/wind_decomposition"
os.makedirs(PLOT_DIR, exist_ok=True)

OMEGA = 7.2921e-5
GRAVITY = 9.81
MEAN_P = 101325.0


def fetch(url: str, outdir: Path) -> Path:
    return Path(pooch.retrieve(url=url, known_hash=None, path=str(outdir)))


def to_0360(ds: xr.Dataset) -> xr.Dataset:
    if float(ds["lon"].min()) < 0:
        ds = ds.assign_coords(lon=(ds["lon"] % 360)).sortby("lon")
    return ds


# ── Grid setup ─────────────────────────────────────────────────────────
nlat, nlon = 36, 72
lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat)
lon = np.linspace(180 / nlon, 360 - 180 / nlon, nlon)
lon2d, lat2d = np.meshgrid(lon, lat)
land_mask = compute_land_mask(lon2d, lat2d)
ocean_frac_by_lat = 1.0 - land_mask.astype(float).mean(axis=1)

lat_deg = _get_latitude_centers(nlat)
lat_rad = np.deg2rad(lat_deg)
cos_lat = np.clip(np.cos(lat_rad), 1e-6, None)
weights = np.broadcast_to(cos_lat[:, None], (nlat, nlon)).astype(float)
lat_rad_2d = np.deg2rad(lat2d)

cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)

# Coriolis
coriolis = 2.0 * OMEGA * np.sin(np.deg2rad(lat2d))

# Grid spacings
delta_y = R_EARTH_METERS * np.deg2rad(180.0 / nlat)
delta_lon_rad = np.deg2rad(360.0 / nlon)
cos_lat_2d = np.cos(np.deg2rad(lat2d))
delta_x = R_EARTH_METERS * cos_lat_2d * delta_lon_rad
inv_two_delta_x = np.zeros_like(delta_x)
valid_dx = np.abs(delta_x) > 0.0
inv_two_delta_x[valid_dx] = 1.0 / (2.0 * delta_x[valid_dx])
inv_two_delta_y = 0.5 / delta_y


def regrid_to_5deg(field, lat_src, lon_src):
    if field.ndim == 3:
        return np.stack([regrid_to_5deg(field[m], lat_src, lon_src) for m in range(field.shape[0])])
    out = np.full((nlat, nlon), np.nan)
    for j in range(nlat):
        for i in range(nlon):
            j1 = np.argmin(np.abs(lat_src - lat[j]))
            i1 = np.argmin(np.abs(lon_src - lon[i]))
            j_lo, j_hi = max(0, j1 - 2), min(len(lat_src), j1 + 3)
            i_lo, i_hi = max(0, i1 - 2), min(len(lon_src), i1 + 3)
            out[j, i] = np.nanmean(field[j_lo:j_hi, i_lo:i_hi])
    return out


def horizontal_gradient(field):
    grad_y = np.zeros_like(field)
    grad_y[1:-1] = (field[2:] - field[:-2]) * inv_two_delta_y
    grad_y[0] = (field[1] - field[0]) / delta_y
    grad_y[-1] = (field[-1] - field[-2]) / delta_y
    diff_east = np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)
    grad_x = diff_east * inv_two_delta_x
    return grad_x, grad_y


def blended_wind(geopotential, k=2e-6):
    """Geostrophic + friction-balanced blend near equator."""
    grad_x, grad_y = horizontal_gradient(geopotential)
    dp_dx = GRAVITY * grad_x
    dp_dy = GRAVITY * grad_y

    f_abs = np.abs(coriolis)
    f_sign = np.sign(coriolis)
    f_sign[f_sign == 0] = 1.0
    f_safe = np.maximum(f_abs, 1e-10)

    u_geo = -dp_dy / (f_sign * f_safe)
    v_geo = dp_dx / (f_sign * f_safe)

    grad_mag = np.sqrt(dp_dx**2 + dp_dy**2)
    speed_fric = np.sqrt(np.maximum(grad_mag / k, 0))
    grad_mag_safe = np.maximum(grad_mag, 1e-10)
    u_fric = -dp_dx / grad_mag_safe * speed_fric
    v_fric = -dp_dy / grad_mag_safe * speed_fric

    f_crit = 2e-5
    transition_width = 1e-5
    w_geo = 0.5 * (1 + np.tanh((f_abs - f_crit) / transition_width))

    u = w_geo * u_geo + (1 - w_geo) * u_fric
    v = w_geo * v_geo + (1 - w_geo) * v_fric
    return u, v


def apply_ekman_drag(u_geo, v_geo, cd, h_m):
    """Ekman drag + turning."""
    speed_geo = np.hypot(u_geo, v_geo)
    f_abs = np.maximum(np.abs(coriolis), 1e-5)

    k = cd / h_m
    a = (k / f_abs) ** 2

    general = a > 1e-16
    y = np.empty_like(speed_geo)
    y[general] = (-1.0 + np.sqrt(1.0 + 4.0 * a[general] * speed_geo[general] ** 2)) / (
        2.0 * a[general]
    )
    y[~general] = speed_geo[~general] ** 2
    y = np.clip(y, 0, None)
    u_mag = np.sqrt(y)

    zero = speed_geo <= 1e-12
    u_mag[zero] = 0.0

    r = k * u_mag
    r[zero] = 0.0
    alpha = np.arctan(r / f_abs)
    alpha[zero] = 0.0

    rotation = np.where(coriolis >= 0, alpha, -alpha)
    cos_a = np.cos(rotation)
    sin_a = np.sin(rotation)

    Ug_safe = np.maximum(speed_geo, 1e-12)
    ux = u_geo / Ug_safe
    vy = v_geo / Ug_safe

    u_final = u_mag * (ux * cos_a - vy * sin_a)
    v_final = u_mag * (ux * sin_a + vy * cos_a)
    return u_final, v_final


def wind_metrics(u_sim, v_sim, u_obs, v_obs, mask=None):
    if mask is None:
        mask = np.ones_like(u_sim[0], dtype=bool)
    u_s = np.nanmean(u_sim, axis=0)
    v_s = np.nanmean(v_sim, axis=0)
    u_o = np.nanmean(u_obs, axis=0)
    v_o = np.nanmean(v_obs, axis=0)
    valid = np.isfinite(u_o) & np.isfinite(v_o) & mask
    spd_s = np.sqrt(u_s**2 + v_s**2)
    spd_o = np.sqrt(u_o**2 + v_o**2)
    r_u = np.corrcoef(u_s[valid], u_o[valid])[0, 1]
    r_v = np.corrcoef(v_s[valid], v_o[valid])[0, 1]
    r_spd = np.corrcoef(spd_s[valid], spd_o[valid])[0, 1]
    rmse_spd = np.sqrt(np.mean((spd_s[valid] - spd_o[valid]) ** 2))
    bias_spd = np.mean(spd_s[valid]) - np.mean(spd_o[valid])
    return {"r_u": r_u, "r_v": r_v, "r_spd": r_spd, "rmse_spd": rmse_spd, "bias_spd": bias_spd}


def compute_slp_from_T(col_T_smooth_m, itcz_m, dp_thermal_m, dp_base, alpha_fer=4000.0):
    """Reconstruct SLP from smoothed column T for one month.

    dp_base scales the Hadley pressure anomaly amplitude.
    alpha_fer is unused (kept for API compat) — Ferrel/subpolar lows
    are now part of hadley_pressure_anomaly.
    """
    from climate_sim.physics.atmosphere.pressure import DP_SUBTROPICS

    itcz_2d = np.broadcast_to(itcz_m[np.newaxis, :], (nlat, nlon))

    # Scale the Hadley pattern by dp_base / default DP_SUBTROPICS
    dp_hadley = hadley_pressure_anomaly(lat_rad_2d, itcz_2d) * (dp_base / DP_SUBTROPICS)
    dp_hadley -= area_weighted_mean(dp_hadley, weights)

    slp = MEAN_P + dp_thermal_m + dp_hadley
    slp *= MEAN_P / area_weighted_mean(slp, weights)
    return slp


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
uwnd_obs_1deg = obs_hp["uwnd_clim"].values
vwnd_obs_1deg = obs_hp["vwnd_clim"].values
lat_obs = obs_hp["lat"].values
lon_obs = obs_hp["lon"].values

print("Regridding to 5°...")
t925_5 = regrid_to_5deg(t925, lat_plev, lon_plev)
t500_5 = regrid_to_5deg(t500, lat_plev, lon_plev)
slp_obs_5 = regrid_to_5deg(slp_obs_1deg, lat_obs, lon_obs)
uwnd_obs_5 = regrid_to_5deg(uwnd_obs_1deg, lat_obs, lon_obs)
vwnd_obs_5 = regrid_to_5deg(vwnd_obs_1deg, lat_obs, lon_obs)

elevation = compute_cell_elevation(lon2d, lat2d)
H_TROPO = 12000.0
elev_amp = 1.0 + np.clip(elevation, 0.0, H_TROPO) / H_TROPO

total_h = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M
bl_w = BOUNDARY_LAYER_HEIGHT_M / total_h
atm_w = ATMOSPHERE_LAYER_HEIGHT_M / total_h


# ── Precompute column T, dp_thermal, ITCZ per month ─────────────────
dp_thermal = np.zeros((12, nlat, nlon))
col_T_smooth = np.zeros((12, nlat, nlon))
col_T_raw = np.zeros((12, nlat, nlon))
itcz_rad = np.zeros((12, nlon))

for m in range(12):
    t9 = np.nan_to_num(t925_5[m], nan=np.nanmean(t925_5[m]))
    t5 = np.nan_to_num(t500_5[m], nan=np.nanmean(t500_5[m]))
    col_K = (t9 + 273.15) * bl_w + (t5 + 273.15) * atm_w
    col_T_raw[m] = col_K

    ts = _smooth_temperature_field(col_K, lat_deg)
    ts += area_weighted_mean(col_K, weights) - area_weighted_mean(ts, weights)
    col_T_smooth[m] = ts

    zonal_mean = np.mean(ts, axis=1, keepdims=True)
    dT = ts - zonal_mean
    dp = -THERMAL_PRESSURE_COEFFICIENT * dT * elev_amp
    dp -= area_weighted_mean(dp, weights)
    dp_thermal[m] = dp

    itcz_rad[m] = compute_itcz_latitude(col_K, lat2d, cell_areas)

ref_p = 92500.0  # 925 hPa


def geopotential_from_slp(slp, col_T):
    T_safe = np.maximum(col_T, 150.0)
    scale_height = GAS_CONSTANT_J_KG_K * T_safe / GRAVITY
    return scale_height * np.log(np.maximum(slp / ref_p, 1.0))


# ── Drag coefficient configs ────────────────────────────────────────
# Our model currently computes Cd from roughness length via log-law
# but the effective k = Cd/h_m is what matters for turning angle
# Physical values: Cd_ocean ~ 1.2e-3 (Large & Pond 1981), Cd_land ~ 5e-3
drag_configs = {
    "current": {
        "cd": np.where(land_mask, 5e-3, 3e-4),  # our current model
        "h_m": np.where(land_mask, 400.0, 500.0),
    },
    "physical": {
        "cd": np.where(land_mask, 5e-3, 1.2e-3),  # Large & Pond ocean Cd
        "h_m": np.where(land_mask, 400.0, 500.0),
    },
}


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Ekman drag sensitivity with OBSERVED SLP
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("  PART 1: Wind from OBSERVED SLP — drag coefficient sensitivity")
print("=" * 120)

print(
    f"\n{'Config':>20} {'Cd_oc':>8} {'h_m_oc':>7} {'k_oc':>8} | "
    f"{'r_u':>6} {'r_v':>6} {'r_spd':>6} {'RMSE':>6} {'bias':>6} | "
    f"{'r_u_oc':>7} {'r_v_oc':>7}"
)
print("-" * 110)

for dc_name, dc in drag_configs.items():
    u_all = np.zeros((12, nlat, nlon))
    v_all = np.zeros((12, nlat, nlon))
    for m in range(12):
        slp = np.nan_to_num(slp_obs_5[m], nan=MEAN_P)
        Z = geopotential_from_slp(slp, col_T_smooth[m])
        u_g, v_g = blended_wind(Z)
        u_e, v_e = apply_ekman_drag(u_g, v_g, dc["cd"], dc["h_m"])
        u_all[m] = u_e
        v_all[m] = v_e

    m_all = wind_metrics(u_all, v_all, uwnd_obs_5, vwnd_obs_5)
    m_oc = wind_metrics(u_all, v_all, uwnd_obs_5, vwnd_obs_5, mask=~land_mask)

    cd_oc = dc["cd"][~land_mask].mean()
    h_oc = dc["h_m"][~land_mask].mean()
    k_oc = cd_oc / h_oc

    print(
        f"{dc_name:>20} {cd_oc:.1e} {h_oc:7.0f} {k_oc:.1e} | "
        f"{m_all['r_u']:6.3f} {m_all['r_v']:6.3f} {m_all['r_spd']:6.3f} "
        f"{m_all['rmse_spd']:6.2f} {m_all['bias_spd']:+6.2f} | "
        f"{m_oc['r_u']:7.3f} {m_oc['r_v']:7.3f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 2: dp_base sweep — reconstructed SLP with physical drag
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("  PART 2: Wind from RECONSTRUCTED SLP — dp_base sweep (using physical Cd)")
print("=" * 120)

cd_phys = drag_configs["physical"]["cd"]
h_m_phys = drag_configs["physical"]["h_m"]

dp_base_values = [300, 400, 500, 600, 700, 800]
alpha_fer_values = [3000, 4000, 5000]

print(
    f"\n{'dp_base':>8} {'alpha':>6} | {'r_u':>6} {'r_v':>6} {'r_spd':>6} {'RMSE':>6} {'bias':>6} | "
    f"{'SLP_corr':>8} {'SLP_RMSE':>8} | {'r_u_oc':>7} {'r_v_oc':>7} {'r_u_ln':>7} {'r_v_ln':>7}"
)
print("-" * 120)

best_config = None
best_score = -999

for dp_base in dp_base_values:
    for alpha_fer in alpha_fer_values:
        u_all = np.zeros((12, nlat, nlon))
        v_all = np.zeros((12, nlat, nlon))
        slp_all = np.zeros((12, nlat, nlon))

        for m in range(12):
            slp = compute_slp_from_T(
                col_T_smooth[m], itcz_rad[m], dp_thermal[m], dp_base=dp_base, alpha_fer=alpha_fer
            )
            slp_all[m] = slp
            Z = geopotential_from_slp(slp, col_T_smooth[m])
            u_g, v_g = blended_wind(Z)
            u_e, v_e = apply_ekman_drag(u_g, v_g, cd_phys, h_m_phys)
            u_all[m] = u_e
            v_all[m] = v_e

        m_all = wind_metrics(u_all, v_all, uwnd_obs_5, vwnd_obs_5)
        m_oc = wind_metrics(u_all, v_all, uwnd_obs_5, vwnd_obs_5, mask=~land_mask)
        m_ln = wind_metrics(u_all, v_all, uwnd_obs_5, vwnd_obs_5, mask=land_mask)

        # SLP correlation
        slp_ann = np.nanmean(slp_all, axis=0)
        obs_ann = np.nanmean(slp_obs_5, axis=0)
        valid = np.isfinite(obs_ann)
        slp_corr = np.corrcoef(slp_ann[valid], obs_ann[valid])[0, 1]
        slp_rmse = np.sqrt(np.mean((slp_ann[valid] - obs_ann[valid]) ** 2)) / 100

        score = m_all["r_u"] + m_all["r_v"] + 0.5 * slp_corr
        if score > best_score:
            best_score = score
            best_config = (dp_base, alpha_fer)

        print(
            f"{dp_base:>8} {alpha_fer:>6} | {m_all['r_u']:6.3f} {m_all['r_v']:6.3f} {m_all['r_spd']:6.3f} "
            f"{m_all['rmse_spd']:6.2f} {m_all['bias_spd']:+6.2f} | "
            f"{slp_corr:8.3f} {slp_rmse:8.1f} | "
            f"{m_oc['r_u']:7.3f} {m_oc['r_v']:7.3f} {m_ln['r_u']:7.3f} {m_ln['r_v']:7.3f}"
        )

print(f"\nBest config: dp_base={best_config[0]}, alpha_fer={best_config[1]}")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Turning angle comparison with best config
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  PART 3: Turning angles — current vs physical Cd (obs SLP)")
print("=" * 100)
print(f"{'Lat':>6} {'Obs':>8} {'Cur Cd':>8} {'Phys Cd':>8}")
print("-" * 40)

for dc_name in ["current", "physical"]:
    dc = drag_configs[dc_name]
    # Compute winds for this drag config
    u_ekm = np.zeros((12, nlat, nlon))
    v_ekm = np.zeros((12, nlat, nlon))
    u_geo_arr = np.zeros((12, nlat, nlon))
    v_geo_arr = np.zeros((12, nlat, nlon))
    for m in range(12):
        slp = np.nan_to_num(slp_obs_5[m], nan=MEAN_P)
        Z = geopotential_from_slp(slp, col_T_smooth[m])
        u_g, v_g = blended_wind(Z)
        u_geo_arr[m] = u_g
        v_geo_arr[m] = v_g
        u_e, v_e = apply_ekman_drag(u_g, v_g, dc["cd"], dc["h_m"])
        u_ekm[m] = u_e
        v_ekm[m] = v_e

    if dc_name == "current":
        u_geo_obs = u_geo_arr  # same for both
        v_geo_obs = v_geo_arr

    drag_configs[dc_name]["_u_ekm"] = u_ekm
    drag_configs[dc_name]["_v_ekm"] = v_ekm

for j, la in enumerate(lat):
    u_g = np.nanmean(u_geo_obs[:, j], axis=0)
    v_g = np.nanmean(v_geo_obs[:, j], axis=0)
    u_o = np.nanmean(uwnd_obs_5[:, j], axis=0)
    v_o = np.nanmean(vwnd_obs_5[:, j], axis=0)

    spd_g = np.hypot(u_g, v_g)
    spd_o = np.hypot(u_o, v_o)
    valid = (spd_o > 0.5) & (spd_g > 0.5)
    if valid.sum() < 2:
        continue

    cross_obs = u_g[valid] * v_o[valid] - v_g[valid] * u_o[valid]
    dot_obs = u_g[valid] * u_o[valid] + v_g[valid] * v_o[valid]
    angle_obs = np.rad2deg(np.arctan2(np.mean(cross_obs), np.mean(dot_obs)))

    line = f"{la:6.1f} {angle_obs:+8.1f}°"
    for dc_name in ["current", "physical"]:
        u_e = np.nanmean(drag_configs[dc_name]["_u_ekm"][:, j], axis=0)
        v_e = np.nanmean(drag_configs[dc_name]["_v_ekm"][:, j], axis=0)
        cross_e = u_g[valid] * v_e[valid] - v_g[valid] * u_e[valid]
        dot_e = u_g[valid] * u_e[valid] + v_g[valid] * v_e[valid]
        angle_e = np.rad2deg(np.arctan2(np.mean(cross_e), np.mean(dot_e)))
        line += f" {angle_e:+8.1f}°"
    print(line)


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Zonal mean comparison — best recon config
# ═══════════════════════════════════════════════════════════════════════
dp_b, a_f = best_config
print(f"\n{'=' * 120}")
print(f"  PART 4: Zonal mean U-wind — best recon (dp_base={dp_b}, alpha={a_f}) vs obs")
print(f"{'=' * 120}")

# Compute winds for best config
u_best = np.zeros((12, nlat, nlon))
v_best = np.zeros((12, nlat, nlon))
slp_best = np.zeros((12, nlat, nlon))
for m in range(12):
    slp = compute_slp_from_T(
        col_T_smooth[m], itcz_rad[m], dp_thermal[m], dp_base=dp_b, alpha_fer=a_f
    )
    slp_best[m] = slp
    Z = geopotential_from_slp(slp, col_T_smooth[m])
    u_g, v_g = blended_wind(Z)
    u_e, v_e = apply_ekman_drag(u_g, v_g, cd_phys, h_m_phys)
    u_best[m] = u_e
    v_best[m] = v_e

# Also compute with obs SLP + physical Cd for comparison
u_obs_phys = drag_configs["physical"]["_u_ekm"]
v_obs_phys = drag_configs["physical"]["_v_ekm"]

print(
    f"{'Lat':>6} {'Obs u':>7} {'ObsSLP':>7} {'Recon':>7} | {'Obs spd':>8} {'ObsSLP':>8} {'Recon':>8}"
)
print("-" * 65)

for j, la in enumerate(lat):
    u_o = np.nanmean(np.nanmean(uwnd_obs_5, axis=0)[j])
    u_op = np.mean(np.nanmean(u_obs_phys, axis=0)[j])
    u_r = np.mean(np.nanmean(u_best, axis=0)[j])

    spd_o = np.nanmean(
        np.sqrt(np.nanmean(uwnd_obs_5, axis=0)[j] ** 2 + np.nanmean(vwnd_obs_5, axis=0)[j] ** 2)
    )
    spd_op = np.mean(
        np.sqrt(np.nanmean(u_obs_phys, axis=0)[j] ** 2 + np.nanmean(v_obs_phys, axis=0)[j] ** 2)
    )
    spd_r = np.mean(
        np.sqrt(np.nanmean(u_best, axis=0)[j] ** 2 + np.nanmean(v_best, axis=0)[j] ** 2)
    )

    print(
        f"{la:6.1f} {u_o:+7.2f} {u_op:+7.2f} {u_r:+7.2f} | {spd_o:8.2f} {spd_op:8.2f} {spd_r:8.2f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Regional winds for best config
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print("  PART 5: Regional wind comparison (July) — best recon vs obs")
print(f"{'=' * 120}")

regions = {
    "NH Westerlies (40-60N)": (40, 60, 0, 360),
    "SH Westerlies (40-60S)": (-60, -40, 0, 360),
    "NE Trades (10-25N)": (10, 25, 0, 360),
    "SE Trades (25-10S)": (-25, -10, 0, 360),
    "India monsoon (5-25N, 60-100E)": (5, 25, 60, 100),
    "Somali Jet (0-10N, 45-65E)": (0, 10, 45, 65),
}

m = 6
print(
    f"{'Region':>35} | {'Obs u':>6} {'Obs v':>6} {'spd':>5} | {'ObsSLP u':>8} {'v':>5} {'spd':>5} | {'Recon u':>7} {'v':>5} {'spd':>5}"
)
print("-" * 105)

for region_name, (lat_lo, lat_hi, lon_lo, lon_hi) in regions.items():
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    if lon_lo > lon_hi:
        lon_mask = (lon >= lon_lo) | (lon <= lon_hi)
    else:
        lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    mask = lat_mask[:, None] & lon_mask[None, :]
    if mask.sum() == 0:
        continue

    def mean_wind(u, v):
        return np.mean(u[m][mask]), np.mean(v[m][mask])

    uo, vo = np.nanmean(uwnd_obs_5[m][mask]), np.nanmean(vwnd_obs_5[m][mask])
    up, vp = mean_wind(u_obs_phys, v_obs_phys)
    ur, vr = mean_wind(u_best, v_best)

    print(
        f"{region_name:>35} | {uo:+6.2f} {vo:+6.2f} {np.hypot(uo, vo):5.1f} | "
        f"{up:+8.2f} {vp:+5.2f} {np.hypot(up, vp):5.1f} | "
        f"{ur:+7.2f} {vr:+5.2f} {np.hypot(ur, vr):5.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# PART 6: SLP comparison — obs vs best recon
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print(f"  PART 6: Zonal mean SLP — obs vs recon (dp_base={dp_b})")
print(f"{'=' * 100}")
print(f"{'Lat':>6} {'Obs':>7} {'Recon':>7} {'Diff':>7}")
print("-" * 30)

obs_slp_ann_zm = np.nanmean(np.nanmean(slp_obs_5, axis=0), axis=1) / 100
rec_slp_ann_zm = np.mean(np.nanmean(slp_best, axis=0), axis=1) / 100

for j, la in enumerate(lat):
    diff = rec_slp_ann_zm[j] - obs_slp_ann_zm[j]
    print(f"{la:6.1f} {obs_slp_ann_zm[j]:7.1f} {rec_slp_ann_zm[j]:7.1f} {diff:+7.1f}")


# ── Plots ────────────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: Zonal mean U-wind ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Zonal mean U-wind: obs vs models (dp_base={dp_b}, Cd_ocean=1.2e-3)", fontsize=14)

    for ax, (midx, mname) in zip(axes[:2], [(6, "July"), (0, "January")]):
        obs_zm = np.nanmean(uwnd_obs_5[midx], axis=1)
        ax.plot(lat, obs_zm, "k-", lw=2.5, label="Obs 10m")
        ax.plot(
            lat, np.mean(u_obs_phys[midx], axis=1), "b-", lw=1.5, label="Ekman (obs SLP, phys Cd)"
        )
        ax.plot(lat, np.mean(u_best[midx], axis=1), "r-", lw=1.5, label=f"Ekman (recon, dp={dp_b})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("U (m/s)")
        ax.set_title(mname)
        ax.legend(fontsize=8)
        ax.set_xlim(-90, 90)
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    obs_zm = np.nanmean(np.nanmean(uwnd_obs_5, axis=0), axis=1)
    ax.plot(lat, obs_zm, "k-", lw=2.5, label="Obs 10m")
    ax.plot(
        lat, np.mean(np.nanmean(u_obs_phys, axis=0), axis=1), "b-", lw=1.5, label="Ekman (obs SLP)"
    )
    ax.plot(
        lat, np.mean(np.nanmean(u_best, axis=0), axis=1), "r-", lw=1.5, label=f"Recon dp={dp_b}"
    )
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("U (m/s)")
    ax.set_title("Annual mean")
    ax.legend(fontsize=8)
    ax.set_xlim(-90, 90)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/zonal_u_wind.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {PLOT_DIR}/zonal_u_wind.png")
    plt.close()

    # ── Figure 2: SLP comparison ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Zonal mean SLP (dp_base={dp_b}, alpha_fer={a_f})", fontsize=14)

    for ax, (midx, mname) in zip(axes[:2], [(6, "July"), (0, "January")]):
        obs_zm = np.nanmean(slp_obs_5[midx], axis=1) / 100
        rec_zm = np.mean(slp_best[midx], axis=1) / 100
        ax.plot(lat, obs_zm, "k-", lw=2.5, label="Obs SLP")
        ax.plot(lat, rec_zm, "r-", lw=1.5, label="Reconstructed")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("hPa")
        ax.set_title(mname)
        ax.legend(fontsize=8)
        ax.set_xlim(-90, 90)
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(lat, obs_slp_ann_zm, "k-", lw=2.5, label="Obs SLP")
    ax.plot(lat, rec_slp_ann_zm, "r-", lw=1.5, label="Reconstructed")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("hPa")
    ax.set_title("Annual mean")
    ax.legend(fontsize=8)
    ax.set_xlim(-90, 90)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/zonal_slp.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/zonal_slp.png")
    plt.close()

    # ── Figure 3: Wind vectors July ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("July wind vectors", fontsize=14)

    m = 6
    datasets = [
        ("Observed 10m", uwnd_obs_5[m], vwnd_obs_5[m]),
        ("Ekman (obs SLP, phys Cd)", u_obs_phys[m], v_obs_phys[m]),
        (f"Ekman (recon dp={dp_b})", u_best[m], v_best[m]),
        ("Difference: recon - obs", u_best[m] - uwnd_obs_5[m], v_best[m] - vwnd_obs_5[m]),
    ]

    for ax, (title, u_plot, v_plot) in zip(axes.flat, datasets):
        spd = np.sqrt(u_plot**2 + v_plot**2)
        vmax = 12 if "Diff" not in title else 6
        cmap = "YlOrRd" if "Diff" not in title else "RdBu_r"
        vmin = 0 if "Diff" not in title else -6
        im = ax.pcolormesh(
            lon,
            lat,
            spd if "Diff" not in title else u_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.quiver(lon, lat, u_plot, v_plot, scale=100, width=0.003, color="black", alpha=0.7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8, label="m/s")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/wind_vectors_july.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/wind_vectors_july.png")
    plt.close()

    # ── Figure 4: Speed ratio ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Speed: obs 10m vs Ekman(obs SLP) vs Ekman(recon SLP)")

    spd_obs = np.nanmean(
        np.sqrt(np.nanmean(uwnd_obs_5, axis=0) ** 2 + np.nanmean(vwnd_obs_5, axis=0) ** 2), axis=1
    )
    spd_phys = np.mean(
        np.sqrt(np.nanmean(u_obs_phys, axis=0) ** 2 + np.nanmean(v_obs_phys, axis=0) ** 2), axis=1
    )
    spd_rec = np.mean(
        np.sqrt(np.nanmean(u_best, axis=0) ** 2 + np.nanmean(v_best, axis=0) ** 2), axis=1
    )

    ax.plot(lat, spd_obs, "k-", lw=2.5, label="Obs 10m")
    ax.plot(lat, spd_phys, "b-", lw=1.5, label="Ekman (obs SLP)")
    ax.plot(lat, spd_rec, "r-", lw=1.5, label=f"Ekman (recon dp={dp_b})")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    ax.set_xlim(-90, 90)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/speed_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/speed_comparison.png")
    plt.close()

except ImportError:
    print("matplotlib not available, skipping plots")

print("\nDone!")
