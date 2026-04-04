"""Cloud decomposition: run our cloud formula with obs inputs, compare to obs cloud.

Tests whether the cloud physics itself produces correct patterns when given
observed temperature, humidity, and vertical velocity — isolating physics
errors from solver state errors.

Columns:
  sim_state: our cloud formula with sim T, q, w (current model output)
  obs_state: our cloud formula with obs T, q, w (what the formula WOULD give
             if the solver had perfect state)
  obs_cloud: actual observed cloud cover (NCEP total cloud)

If obs_state matches obs_cloud, the cloud formula is fine and the problem
is purely the solver state. If obs_state is still wrong, the formula itself
needs fixing.

Usage:
    PYTHONPATH=backend uv run python backend/debug/cloud_decomposition.py
"""

import numpy as np
import xarray as xr
from pathlib import Path
from scipy.stats import pearsonr

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.data.landmask import compute_land_mask
from climate_sim.physics.clouds import (
    compute_clouds_and_precipitation,
)
from climate_sim.physics.humidity import (
    compute_saturation_specific_humidity,
)
from climate_sim.data.constants import STANDARD_LAPSE_RATE_K_PER_M

P = Path(__file__).resolve().parents[2]  # project root

# ── Load grids and masks ──
lon2d, lat2d = create_lat_lon_grid(5.0)
land = compute_land_mask(lon2d, lat2d)
ocean = ~land
nlat, nlon = lon2d.shape

# ── Load sim data ──
d = np.load(P / "data" / "main.npz")
sim_T_sfc = d["surface"] + 273.15  # (12, nlat, nlon) K
sim_T_bl = d["boundary_layer"] + 273.15
sim_T_atm = d["atmosphere"] + 273.15
sim_q = d["humidity"]
sim_w = d["vertical_velocity"]

# ── Load obs data ──
obs_t_ds = xr.open_dataset(P / "data" / "processed" / "ref_climatology_1deg_1981-2010.nc")
obs_T_2m_1deg = (
    obs_t_ds["t_surface_clim"].values + 273.15
)  # (12, 180, 360) K — T_2m over land, SST over ocean

obs_hq_ds = xr.open_dataset(P / "data" / "processed" / "ref_humidity_precip_1deg_1981-2010.nc")
obs_q_1deg = obs_hq_ds["shum_clim"].values  # (12, 180, 360) kg/kg
obs_slp_1deg = obs_hq_ds["slp_clim"].values  # (12, 180, 360) Pa
obs_u_1deg = obs_hq_ds["uwnd_clim"].values
obs_v_1deg = obs_hq_ds["vwnd_clim"].values

# Load obs cloud cover
obs_cc_ds = xr.open_dataset(
    P / "data" / "raw" / "noaa" / "0af5bd8cca386b916d3cb00ce6f5802e-tcdc.eatm.mon.ltm.1981-2010.nc",
    use_cftime=True,
)
obs_cc_raw = obs_cc_ds["tcdc"].values  # (12, 94, 192) %
obs_cc_lat = obs_cc_ds["lat"].values
obs_cc_lon = obs_cc_ds["lon"].values


# ── Regrid functions ──
def regrid_1deg_to_5deg(data_12, nlat_out=nlat, nlon_out=nlon):
    """Regrid (12, 180, 360) -> (12, nlat, nlon) by block averaging."""
    out = np.full((12, nlat_out, nlon_out), np.nan)
    for m in range(12):
        for i in range(nlat_out):
            for j in range(nlon_out):
                i0, i1 = i * 5, (i + 1) * 5
                j0, j1 = j * 5, (j + 1) * 5
                patch = data_12[m, i0:i1, j0:j1]
                valid = ~np.isnan(patch)
                if valid.any():
                    out[m, i, j] = patch[valid].mean()
    return out


def regrid_gaussian_to_5deg(data_12, obs_lat, obs_lon):
    """Regrid Gaussian grid -> 5deg by nearest-neighbor averaging."""
    out = np.full((12, nlat, nlon), np.nan)
    for m in range(12):
        for i in range(nlat):
            lat_c = lat2d[i, 0]
            lat_mask = np.abs(obs_lat - lat_c) < 2.5
            for j in range(nlon):
                lon_c = lon2d[0, j]
                lon_mask = np.abs(obs_lon - lon_c) < 2.5
                if lon_c < 2.5:
                    lon_mask = lon_mask | (obs_lon > 357.5)
                if lon_c > 357.5:
                    lon_mask = lon_mask | (obs_lon < 2.5)
                patch = data_12[m][np.ix_(lat_mask, lon_mask)]
                valid = ~np.isnan(patch)
                if valid.any():
                    out[m, i, j] = patch[valid].mean()
    return out


# ── Regrid obs to 5deg ──
print("Regridding obs to 5deg...")
obs_T_2m = regrid_1deg_to_5deg(obs_T_2m_1deg)
obs_q = regrid_1deg_to_5deg(obs_q_1deg)
obs_slp = regrid_1deg_to_5deg(obs_slp_1deg)
obs_u = regrid_1deg_to_5deg(obs_u_1deg)
obs_v = regrid_1deg_to_5deg(obs_v_1deg)
obs_cc = regrid_gaussian_to_5deg(obs_cc_raw, obs_cc_lat, obs_cc_lon)

# ── Construct obs BL and atm temperatures ──
# BL is ~500m above surface: T_bl = T_sfc - lapse_rate * 500m
# Atm midpoint is ~5500m: T_atm = T_sfc - lapse_rate * 5500m
obs_T_bl = obs_T_2m - STANDARD_LAPSE_RATE_K_PER_M * 500.0
obs_T_atm = obs_T_2m - STANDARD_LAPSE_RATE_K_PER_M * 5500.0


# ── Construct obs vertical velocity from SLP ──
# Use pressure anomaly as proxy (same as model):
# Low pressure -> rising, high pressure -> sinking
mean_slp = np.nanmean(obs_slp, axis=(1, 2), keepdims=True)
dp_norm = np.clip((obs_slp - mean_slp) / 2000.0, -1.0, 1.0)
obs_w = -dp_norm * 0.01  # Same scale as model


# ── Run cloud formula with sim state and obs state ──
def run_cloud_formula(T_sfc, T_bl, T_atm, q, w, label, T_2m_for_rh=None):
    """Run our cloud formula and return total cloud fraction (0-1).

    If T_2m_for_rh is provided, use it (with q) to compute RH.
    Otherwise compute RH from T_bl and q (what the solver does).
    """
    total = np.zeros((12, nlat, nlon))
    for m in range(12):
        # Handle NaN in obs by filling with sim values
        t_sfc_m = np.where(np.isnan(T_sfc[m]), sim_T_sfc[m], T_sfc[m])
        t_bl_m = np.where(np.isnan(T_bl[m]), sim_T_bl[m], T_bl[m])
        t_atm_m = np.where(np.isnan(T_atm[m]), sim_T_atm[m], T_atm[m])
        q_m = np.where(np.isnan(q[m]), sim_q[m], q[m])
        w_m = np.where(np.isnan(w[m]), sim_w[m], w[m])

        # Compute RH: use T_2m if provided (for obs, where q is 2m humidity),
        # otherwise use T_bl (what the solver does internally)
        if T_2m_for_rh is not None:
            t_rh = np.where(np.isnan(T_2m_for_rh[m]), t_bl_m, T_2m_for_rh[m])
        else:
            t_rh = t_bl_m
        rh = np.clip(q_m / np.maximum(compute_saturation_specific_humidity(t_rh), 1e-10), 0, 1)

        co = compute_clouds_and_precipitation(
            T_bl_K=t_bl_m,
            T_atm_K=t_atm_m,
            q=q_m,
            rh=rh,
            vertical_velocity=w_m,
            T_surface_K=t_sfc_m,
            ocean_mask=ocean,
        )
        total[m] = co.total_frac
    return total * 100.0  # Convert to %


print("Running cloud formula with sim state...")
sim_cc = run_cloud_formula(sim_T_sfc, sim_T_bl, sim_T_atm, sim_q, sim_w, "sim")

print("Running cloud formula with obs state...")
# Use obs T_2m for RH (obs q is 2m humidity, so RH = q_2m / q_sat(T_2m))
obs_state_cc = run_cloud_formula(
    obs_T_2m,
    obs_T_bl,
    obs_T_atm,
    obs_q,
    obs_w,
    "obs",
    T_2m_for_rh=obs_T_2m,
)

# Also try hybrid: obs T + obs q + sim w (test if w matters)
print("Running cloud formula with obs T,q + sim w...")
hybrid_cc = run_cloud_formula(
    obs_T_2m,
    obs_T_bl,
    obs_T_atm,
    obs_q,
    sim_w,
    "hybrid",
    T_2m_for_rh=obs_T_2m,
)

# ── Compare ──
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print("\n" + "=" * 90)
print("CLOUD PATTERN CORRELATION (land) — our formula with different inputs vs obs cloud")
print("=" * 90)
print(
    f"{'Mon':>4}  {'sim_state':>10}  {'obs_state':>10}  {'obs_T,q+sim_w':>14}  {'obs CC mean':>11}  {'sim CC mean':>11}  {'obs_state CC':>12}"
)
print("-" * 90)
for m in range(12):
    v = land & ~np.isnan(obs_cc[m]) & ~np.isnan(obs_state_cc[m])
    if v.sum() < 10:
        print(f"{months[m]:>4}  insufficient data")
        continue
    r_sim, _ = pearsonr(sim_cc[m][v], obs_cc[m][v])
    r_obs, _ = pearsonr(obs_state_cc[m][v], obs_cc[m][v])
    r_hyb, _ = pearsonr(hybrid_cc[m][v], obs_cc[m][v])
    cc_obs_mean = np.nanmean(obs_cc[m][v])
    cc_sim_mean = sim_cc[m][v].mean()
    cc_obsstate_mean = obs_state_cc[m][v].mean()
    print(
        f"{months[m]:>4}  {r_sim:10.3f}  {r_obs:10.3f}  {r_hyb:14.3f}  "
        f"{cc_obs_mean:11.1f}  {cc_sim_mean:11.1f}  {cc_obsstate_mean:12.1f}"
    )

# Annual
print("-" * 90)
ann_sim = sim_cc.mean(axis=0)
ann_obs_state = obs_state_cc.mean(axis=0)
ann_hyb = hybrid_cc.mean(axis=0)
ann_obs = np.nanmean(obs_cc, axis=0)
v = land & ~np.isnan(ann_obs) & ~np.isnan(ann_obs_state)
r_sim, _ = pearsonr(ann_sim[v], ann_obs[v])
r_obs, _ = pearsonr(ann_obs_state[v], ann_obs[v])
r_hyb, _ = pearsonr(ann_hyb[v], ann_obs[v])
print(
    f"{'Ann':>4}  {r_sim:10.3f}  {r_obs:10.3f}  {r_hyb:14.3f}  "
    f"{np.nanmean(ann_obs[v]):11.1f}  {ann_sim[v].mean():11.1f}  {ann_obs_state[v].mean():12.1f}"
)

# ── Key cells ──
print("\n" + "=" * 90)
print("KEY CELLS: obs_cloud vs formula(sim_state) vs formula(obs_state)")
print("=" * 90)
cells = [
    ("New England", 42.5, 287.5),
    ("Korea", 42.5, 132.5),
    ("PNW", 47.5, 237.5),
    ("Kansas", 37.5, 262.5),
    ("Sahara", 27.5, 12.5),
    ("Amazon", -2.5, 297.5),
    ("Germany", 52.5, 12.5),
    ("Siberia", 57.5, 92.5),
]

print(
    f"{'Cell':<14} {'Mon':>3}  {'obs_CC':>6}  {'sim_CC':>6}  {'obsState_CC':>11}  {'obs_RH':>6}  {'sim_RH':>6}"
)
print("-" * 70)
for name, lat_c, lon_c in cells:
    i = np.argmin(np.abs(lat2d[:, 0] - lat_c))
    j = np.argmin(np.abs(lon2d[0, :] - lon_c))
    for m in [0, 6]:
        oc = obs_cc[m, i, j]
        sc = sim_cc[m, i, j]
        osc = obs_state_cc[m, i, j]
        # Compute obs RH at 2m (q_2m / q_sat(T_2m)) and sim RH at BL
        obs_rh_val = (
            obs_q[m, i, j] / max(compute_saturation_specific_humidity(obs_T_2m[m, i, j]), 1e-10)
            if not np.isnan(obs_q[m, i, j])
            else float("nan")
        )
        sim_rh_val = sim_q[m, i, j] / max(
            compute_saturation_specific_humidity(sim_T_bl[m, i, j]), 1e-10
        )
        print(
            f"{name:<14} {months[m]:>3}  {oc:6.1f}  {sc:6.1f}  {osc:11.1f}  "
            f"{obs_rh_val:6.3f}  {sim_rh_val:6.3f}"
        )
    print()
