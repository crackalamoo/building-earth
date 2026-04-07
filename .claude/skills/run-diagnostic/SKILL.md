---
name: run-diagnostic
description: Run ad-hoc diagnostic Python scripts against simulation output. Use this skill whenever you need to write and run a quick analysis script (energy budgets, field comparisons, zonal means, etc.) to avoid common pitfalls with units, imports, and data formats.
allowed-tools: Read, Bash(uv run python:*), Write
context: fork
---

# Run Diagnostic Skill

Write and run diagnostic Python scripts against the climate simulation output.

## Critical rules — read these EVERY TIME before writing a diagnostic script

### 1. Imports and PYTHONPATH

Always run with `PYTHONPATH=backend`:
```bash
PYTHONPATH=backend uv run python utils/scratch.py
```

Or if inlining with `-c`, prefix with `PYTHONPATH=backend`.

### 2. Units in main.npz

| Field | Units | Common mistake |
|-------|-------|----------------|
| `surface`, `boundary_layer`, `atmosphere`, `temperature_2m` | **°C** | Treating as Kelvin |
| `humidity` | **kg/kg** | Treating as g/kg |
| `precipitation` | **kg/m²/s** | Treating as mm/day (multiply by 86400) |
| `wind_*` | **m/s** | — |
| `albedo` | **0-1** | — |
| `soil_moisture` | **0-1 fraction** | — |

To convert temperature to Kelvin: `T_K = data['surface'] + 273.15`

### 3. Grid construction

```python
import numpy as np

data = np.load('data/main.npz')
nmonths, nlat, nlon = data['surface'].shape  # (12, nlat, nlon)

# Cell centers (NOT edges)
lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)   # e.g. -87.5 to 87.5 at 5°
lon = np.linspace(180/nlon, 360 - 180/nlon, nlon)       # e.g. 2.5 to 357.5 at 5°
lon2d, lat2d = np.meshgrid(lon, lat)
```

### 4. Land mask

There is NO land mask in main.npz. Create it:

```python
from climate_sim.data.landmask import compute_land_mask
land_mask = compute_land_mask(lon2d, lat2d)  # boolean array
ocean_mask = ~land_mask
```

### 5. Area weighting

```python
cos_lat = np.cos(np.deg2rad(lat))
weights = cos_lat[:, np.newaxis] * np.ones(nlon)
weights = weights / weights.sum()  # Normalized to sum to 1

# Area-weighted global mean of a 2D field:
global_mean = np.sum(field * weights)

# Land-only mean:
land_weights = weights * land_mask
land_mean = np.sum(field * land_weights) / np.sum(land_weights)
```

### 6. Bulk transfer coefficient

`ch` is a **2D array** computed on `WindModel`, NOT a scalar on `WindConfig`:
- Land: ~1e-3 (depends on roughness)
- Ocean: ~1.2e-3

For quick estimates, use `ch_land = 1.0e-3`, `ch_ocean = 1.2e-3`:
```python
ch = np.where(land_mask, 1.0e-3, 1.2e-3)
```

### 7. Saturation specific humidity

```python
def q_sat(T_K):
    """T in Kelvin, returns kg/kg."""
    T_C = np.clip(T_K - 273.15, -100, 80)
    e_sat = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))  # hPa
    p_hPa = 1013.25
    return 0.622 * e_sat / (p_hPa - 0.378 * e_sat)
```

### 8. Available fields in main.npz

```
surface, boundary_layer, atmosphere, temperature_2m,
humidity, precipitation, soil_moisture, albedo,
vegetation_fraction,
wind_u, wind_v, wind_speed,
wind_u_10m, wind_v_10m, wind_speed_10m,
wind_u_geostrophic, wind_v_geostrophic, wind_speed_geostrophic,
ocean_u, ocean_v,
convective_cloud_frac, stratiform_cloud_frac,
marine_sc_cloud_frac, high_cloud_frac,
vertical_velocity, surface_pressure
```

All have shape `(12, nlat, nlon)`.

## Workflow

1. Write the diagnostic script to `utils/scratch.py` (this file might already exist)
2. Run it: `PYTHONPATH=backend uv run python utils/scratch.py`
3. **Return the FULL raw output** — all tables, numbers, and print statements verbatim. Do NOT summarize, paraphrase, or omit rows/columns. The user needs the actual numbers to reason about physics. If the output is large, still return it in full.

## Common diagnostic patterns

### E-P budget (evaporation vs precipitation balance)

```python
# Evaporation (approximate): E = rho * ch * V * (q_sat - q)
T_K = data['surface'] + 273.15
rho = 101325.0 / (287.05 * T_K)
qs = q_sat(T_K)
q = np.minimum(data['humidity'], qs)
V = np.maximum(data['wind_speed_10m'], 1.0)
ch = np.where(land_mask, 1.0e-3, 1.2e-3)
evap = rho * ch * V * (qs - q)  # kg/m²/s (same units as precipitation)

# Compare
P = data['precipitation']
print(f"E = {np.sum(evap.mean(0) * weights) * 86400:.2f} mm/day")
print(f"P = {np.sum(P.mean(0) * weights) * 86400:.2f} mm/day")
```

### Zonal mean profile

```python
field = data['surface'].mean(axis=0)  # Annual mean
zonal_mean = np.average(field, weights=np.ones(nlon), axis=1)
for i, l in enumerate(lat):
    print(f"{l:6.1f}°: {zonal_mean[i]:.1f}°C")
```

### Energy budget at a point

```python
lat_idx = np.argmin(np.abs(lat - 62.5))  # Siberia latitude
lon_idx = np.argmin(np.abs(lon - 82.5))
month = 0  # January

sigma = 5.67e-8
T_sfc_K = data['surface'][month, lat_idx, lon_idx] + 273.15
T_bl_K = data['boundary_layer'][month, lat_idx, lon_idx] + 273.15
LW_up = sigma * T_sfc_K**4
print(f"Surface emission: {LW_up:.1f} W/m²")
```
