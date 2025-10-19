#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a 1°×1° monthly reference climatology (1981–2010)
from NOAA PSL datasets:
  - Land: GHCN_CAMS 2m air temperature (0.5°, monthly)
  - Ocean: COBE2 SST (1°, monthly)

No OPeNDAP needed; downloads .nc files via HTTPS first.

Outputs:
  - data/processed/ref_climatology_1deg_1981-2010.nc
  - quick summary prints (dims, basic stats)
"""

import os
from pathlib import Path
import numpy as np
import xarray as xr
import pooch

# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "noaa"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    # Land: monthly 2m air temperature (°C), 0.5° grid
    "land": "https://downloads.psl.noaa.gov/Datasets/ghcncams/air.mon.mean.nc",
    # Ocean: monthly SST (°C), 1° grid (COBE2)
    "ocean": "https://downloads.psl.noaa.gov/Datasets/COBE2/sst.mon.mean.nc",
}

BASELINE_START = "1981-01-01"
BASELINE_END   = "2010-12-31"

# 1° target grid (cell centers at .5°)
LAT_T = np.arange(-89.5, 90.5, 1.0)
LON_T = np.arange(0.5, 360.5, 1.0)

OUTFILE = PROC_DIR / "ref_climatology_1deg_1981-2010.nc"

# ----------------------------
# Helpers
# ----------------------------
def fetch(url: str, outdir: Path) -> Path:
    """Download to a local cache and return filepath."""
    path = pooch.retrieve(url=url, known_hash=None, path=str(outdir))
    return Path(path)

def to_0360(ds: xr.Dataset, lon_name="lon") -> xr.Dataset:
    """Normalize longitudes to 0–360 and sort."""
    if float(ds[lon_name].min()) < 0:
        ds = ds.assign_coords({lon_name: (ds[lon_name] % 360)}).sortby(lon_name)
    return ds

def regrid_1deg(da: xr.DataArray, lat="lat", lon="lon") -> xr.DataArray:
    """Interpolate to the 1° target grid."""
    return da.interp({lat: LAT_T, lon: LON_T})

def monthly_climatology(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """Compute monthly climatology over [start, end]."""
    return da.sel(time=slice(start, end)).groupby("time.month").mean("time")

def load_optional_mask_1deg(mask_path: Path) -> xr.DataArray | None:
    """
    If you already have a 1° land mask, load it.
    Expected:
      - NetCDF with dims (lat, lon), boolean or 0/1, 0–360 lon, -89.5..89.5 lat
      - var name guessed from first data_var
    """
    if not mask_path or not mask_path.exists():
        return None
    m = xr.open_dataset(mask_path)
    var = list(m.data_vars)[0]
    mask = m[var]
    # coerce to bool
    if mask.dtype != bool:
        mask = mask.astype(bool)
    # safety: align coords to target grid
    mask = mask.assign_coords(lat=("lat", LAT_T), lon=("lon", LON_T))
    return mask

# ----------------------------
# Main pipeline
# ----------------------------
def main(mask_path: str | None = None):
    # 1) Download files
    land_nc = fetch(URLS["land"], RAW_DIR)
    ocean_nc = fetch(URLS["ocean"], RAW_DIR)

    # Quick sizes
    land_mb = os.path.getsize(land_nc) / (1024**2)
    ocean_mb = os.path.getsize(ocean_nc) / (1024**2)
    print(f"Downloaded: {land_nc.name} ~{land_mb:.1f} MB")
    print(f"Downloaded: {ocean_nc.name} ~{ocean_mb:.1f} MB")

    # 2) Open datasets
    land_ds = xr.open_dataset(land_nc)   # var: 'air' (degC), dims (time, lat, lon)
    sst_ds  = xr.open_dataset(ocean_nc)  # var: 'sst' (degC), dims (time, lat, lon)

    # 3) Normalize longitudes
    land_ds = to_0360(land_ds, "lon")
    sst_ds  = to_0360(sst_ds, "lon")

    # 4) Pick variables
    t_land = land_ds["air"]          # land 2m air temp, °C
    t_sst  = sst_ds["sst"]           # sea surface temp, °C

    # 5) Baseline slice and monthly climatologies
    t_land_baseline = t_land.sel(time=slice(BASELINE_START, BASELINE_END))
    t_sst_baseline  = t_sst.sel(time=slice(BASELINE_START, BASELINE_END))

    # 6) Regrid to 1° (land source is 0.5°; sst is already 1° but we ensure alignment)
    t_land_1deg = regrid_1deg(t_land_baseline)
    t_sst_1deg  = regrid_1deg(t_sst_baseline)

    # 7) Monthly climatologies on the 1° grid
    clim_land = t_land_1deg.groupby("time.month").mean("time")  # [month, lat, lon]
    clim_sst  = t_sst_1deg.groupby("time.month").mean("time")    # [month, lat, lon]

    # 8) Land–sea mask
    # Prefer user's own 1° mask (boolean, True=land). If not provided, auto-derive
    # from where the land climatology has data (not NaN) on the first month.
    user_mask = load_optional_mask_1deg(Path(mask_path)) if mask_path else None
    if user_mask is None:
        mask_land = clim_land.isel(month=0).notnull()
        print("No external land mask provided; auto-derived from land dataset.")
    else:
        mask_land = user_mask
        print("Using external land mask from:", mask_path)

    # 9) Combine into a single surface temperature (°C):
    # use land 2m air temp over land; use SST over ocean.
    # (Broadcast mask to include 'month' dim.)
    mask3 = mask_land.broadcast_like(clim_land)
    clim_surface = xr.where(mask3, clim_land, clim_sst)

    # 10) Write output
    ds_out = xr.Dataset(
        dict(
            t_land_clim=clim_land,       # °C, monthly land 2m air climatology
            t_sst_clim=clim_sst,         # °C, monthly SST climatology
            t_surface_clim=clim_surface, # °C, combined monthly surface field
            land_mask=mask_land.astype("bool"),
        ),
        coords=dict(
            month=("month", np.arange(1, 13, dtype=int)),
            lat=("lat", LAT_T),
            lon=("lon", LON_T),
        ),
        attrs=dict(
            title="Reference 1° monthly climatologies (1981–2010): land (GHCN_CAMS), ocean (COBE2)",
            source_land=str(land_nc),
            source_ocean=str(ocean_nc),
            baseline=f"{BASELINE_START[:4]}–{BASELINE_END[:4]}",
            notes="t_surface_clim uses land air temperature over land, SST over ocean.",
        ),
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(OUTFILE, encoding=encoding)
    print(f"Wrote: {OUTFILE}")

    # 11) Quick exploration
    print("\n--- Summary ---")
    print("Land clim shape:", tuple(ds_out["t_land_clim"].shape))       # (12, 180, 360)
    print("SST  clim shape:", tuple(ds_out["t_sst_clim"].shape))        # (12, 180, 360)
    print("Surf clim shape:", tuple(ds_out["t_surface_clim"].shape))    # (12, 180, 360)
    print("Global mean (Jan) land / sst / surface:",
          float(ds_out["t_land_clim"].isel(month=0).mean()),
          float(ds_out["t_sst_clim"].isel(month=0).mean()),
          float(ds_out["t_surface_clim"].isel(month=0).mean()))

if __name__ == "__main__":
    # If you already have a 1° land mask, pass its path here
    # (NetCDF with boolean or 0/1 values on lat=-89.5..89.5, lon=0.5..359.5)
    main(mask_path=None)
