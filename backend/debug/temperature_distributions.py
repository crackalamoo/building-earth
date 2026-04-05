"""Plot temperature distributions: sim vs obs for land, ocean, global.

Compares the statistical distribution of monthly T2m across all cells,
showing where the model's temperature distribution differs from obs.

Usage:
    PYTHONPATH=backend uv run python backend/debug/temperature_distributions.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.data.landmask import compute_land_mask

P = Path(__file__).resolve().parents[2]  # project root

lon2d, lat2d = create_lat_lon_grid(5.0)
land = compute_land_mask(lon2d, lat2d)
ocean = ~land
nlat, nlon = lon2d.shape

# ── Load sim ──
d = np.load(P / "data" / "main.npz")
sim_T2m = d["temperature_2m"]  # (12, nlat, nlon) °C

# ── Load obs ──
obs_t_ds = xr.open_dataset(P / "data" / "processed" / "ref_climatology_1deg_1981-2010.nc")
obs_T_1deg = obs_t_ds["t_surface_clim"].values  # (12, 180, 360) °C

# Regrid obs to 5deg
obs_T = np.full((12, nlat, nlon), np.nan)
for m in range(12):
    for i in range(nlat):
        for j in range(nlon):
            patch = obs_T_1deg[m, i * 5 : (i + 1) * 5, j * 5 : (j + 1) * 5]
            valid = ~np.isnan(patch)
            if valid.any():
                obs_T[m, i, j] = patch[valid].mean()

# ── Cell area weights ──
cos_lat = np.cos(np.deg2rad(lat2d))
weights = cos_lat / cos_lat.sum()

# ── Plot ──
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle("Temperature Distribution: Sim vs Obs (NH only)", fontsize=15, y=0.97)

bins = np.arange(-55, 45, 2)

nh = lat2d >= 0
masks = {
    "NH All": nh,
    "NH Land": land & nh,
    "NH Ocean": ocean & nh,
}

rows = [
    ("All Months", slice(None)),
    ("January", 0),
    ("July", 6),
]

for row, (row_label, midx) in enumerate(rows):
    for col, (region_name, mask) in enumerate(masks.items()):
        ax = axes[row, col]
        if midx == slice(None):
            # All months: only include cells where obs has data in ALL months
            obs_valid = ~np.isnan(obs_T).any(axis=0) & mask
            sim_vals = sim_T2m[:, obs_valid].ravel()
            obs_vals = obs_T[:, obs_valid].ravel()
        else:
            obs_valid = ~np.isnan(obs_T[midx]) & mask
            sim_vals = sim_T2m[midx][obs_valid].ravel()
            obs_vals = obs_T[midx][obs_valid].ravel()

        ax.hist(obs_vals, bins=bins, alpha=0.5, label="Obs", color="C0", density=True)
        ax.hist(sim_vals, bins=bins, alpha=0.5, label="Sim", color="C1", density=True)
        ax.set_title(f"{region_name} — {row_label}")
        ax.set_xlabel("T2m (°C)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.axvline(np.mean(obs_vals), color="C0", ls="--", lw=1)
        ax.axvline(np.mean(sim_vals), color="C1", ls="--", lw=1)

fig.tight_layout(rect=[0, 0, 1, 0.94])
outpath = P / "data" / "temperature_distributions.png"
fig.savefig(outpath, dpi=150)
print(f"Saved: {outpath}")
plt.close(fig)

# ── Print summary stats (excluding cells without obs) ──
print("\nSummary statistics (°C, obs-coverage cells only):")
print(f"{'':>12} {'Sim mean':>9} {'Obs mean':>9} {'Sim std':>8} {'Obs std':>8}")
for region_name, mask in masks.items():
    for mname, midx in [("Annual", slice(None)), ("Jan", 0), ("Jul", 6)]:
        if midx == slice(None):
            valid = ~np.isnan(obs_T).any(axis=0) & mask
            sv = sim_T2m[:, valid].ravel()
            ov = obs_T[:, valid].ravel()
        else:
            valid = ~np.isnan(obs_T[midx]) & mask
            sv = sim_T2m[midx][valid].ravel()
            ov = obs_T[midx][valid].ravel()
        print(f"{region_name + ' ' + mname:>12} {sv.mean():9.1f} {ov.mean():9.1f} {sv.std():8.1f} {ov.std():8.1f}")
