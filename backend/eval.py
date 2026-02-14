#!/usr/bin/env python3
"""Evaluate the climate model against NOAA reference climatology data.

This script downloads the NOAA PSL reference datasets, constructs a 1°×1°
monthly climatology (1981–2010), runs the climate model with configurable
physics options, and compares the simulated temperatures to the observational
baseline. Results are summarised via area-weighted RMSE statistics over land,
ocean, and the globe and displayed with interactive maps.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

import numpy as np
import pooch
import xarray as xr
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.widgets import RadioButtons

from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.physics.orographic_effects import OrographicConfig
from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.plotting import add_status_readout, plot_monthly_temperature_cycle
from climate_sim.data.calendar import MONTH_NAMES
from climate_sim.data.constants import R_EARTH_METERS
from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.data.landmask import compute_land_mask
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.core.solver import solve_periodic_climate
from climate_sim.core.units import convert_temperature, temperature_unit
from climate_sim.core.interpolation import interpolate_layer_map
from climate_sim.core.timing import time_block, get_profiler
from climate_sim.runtime.config import ModelConfig

load_dotenv()

# ----------------------------
# Configuration
# ----------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "noaa"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "land": "https://downloads.psl.noaa.gov/Datasets/ghcncams/air.mon.mean.nc",
    "ocean": "https://downloads.psl.noaa.gov/Datasets/COBE2/sst.mon.mean.nc",
    "humidity": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface_gauss/shum.2m.mon.mean.nc",
    "precip": "https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
}

BASELINE_START = "1981-01-01"
BASELINE_END = "2010-12-31"

LAT_T = np.arange(-89.5, 90.5, 1.0)
LON_T = np.arange(0.5, 360.5, 1.0)

OUTFILE = PROC_DIR / "ref_climatology_1deg_1981-2010.nc"


# ----------------------------
# CLI
# ----------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA climatology, run the climate model, and evaluate "
            "simulation performance."
        )
    )
    default_atmosphere = RadiationConfig().include_atmosphere
    add_common_model_arguments(
        parser,
        default_atmosphere=default_atmosphere,
        fahrenheit_help="Display plots/statistics in Fahrenheit instead of Celsius",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help=(
            "Optional NetCDF land mask (boolean) already on a 1° grid with "
            "lat=-89.5..89.5 and lon=0.5..359.5"
        ),
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        action="store_true",
        default=False,
        help="load from cache",
    )

    return parser.parse_args()


# ----------------------------
# NOAA climatology helpers
# ----------------------------


def fetch(url: str, outdir: Path) -> Path:
    """Download to a local cache and return the file path."""

    path = pooch.retrieve(url=url, known_hash=None, path=str(outdir))
    return Path(path)


def to_0360(ds: xr.Dataset, lon_name: str = "lon") -> xr.Dataset:
    """Normalise longitudes to the 0–360 range and sort them."""

    if float(ds[lon_name].min()) < 0:
        ds = ds.assign_coords({lon_name: (ds[lon_name] % 360)}).sortby(lon_name)
    return ds


def regrid_1deg(da: xr.DataArray, lat: str = "lat", lon: str = "lon") -> xr.DataArray:
    """Interpolate to the 1° target grid."""

    return da.interp({lat: LAT_T, lon: LON_T})


def load_optional_mask_1deg(mask_path: Path | None) -> xr.DataArray | None:
    """Load a precomputed 1° land mask if available."""

    if mask_path is None or not mask_path.exists():
        return None
    dataset = xr.open_dataset(mask_path)
    var = list(dataset.data_vars)[0]
    mask = dataset[var]
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mask = mask.assign_coords(lat=("lat", LAT_T), lon=("lon", LON_T))
    return mask


def build_reference_climatology(mask_path: Path | None) -> xr.Dataset:
    """Download, harmonise, and save the reference climatology dataset."""

    land_nc = fetch(URLS["land"], RAW_DIR)
    ocean_nc = fetch(URLS["ocean"], RAW_DIR)

    land_mb = os.path.getsize(land_nc) / (1024**2)
    ocean_mb = os.path.getsize(ocean_nc) / (1024**2)
    print(f"Downloaded: {land_nc.name} ~{land_mb:.1f} MB")
    print(f"Downloaded: {ocean_nc.name} ~{ocean_mb:.1f} MB")

    land_ds = xr.open_dataset(land_nc)
    sst_ds = xr.open_dataset(ocean_nc)

    land_ds = to_0360(land_ds, "lon")
    sst_ds = to_0360(sst_ds, "lon")

    t_land = land_ds["air"].sel(time=slice(BASELINE_START, BASELINE_END))
    t_sst = sst_ds["sst"].sel(time=slice(BASELINE_START, BASELINE_END))
    t_land -= 273.15  # K to °C
    # t_sst is already in °C

    t_land_1deg = regrid_1deg(t_land)
    t_sst_1deg = regrid_1deg(t_sst)

    clim_land = t_land_1deg.groupby("time.month").mean("time")
    clim_sst = t_sst_1deg.groupby("time.month").mean("time")

    user_mask = load_optional_mask_1deg(mask_path)
    if user_mask is not None:
        mask_land = user_mask
        print(f"Using external land mask from: {mask_path}")
    else:
        mask_land = clim_land.isel(month=0).notnull()
        print("No external land mask provided; auto-derived from land dataset.")

    mask3 = mask_land.broadcast_like(clim_land)
    clim_surface = xr.where(mask3, clim_land, clim_sst)

    ds_out = xr.Dataset(
        dict(
            t_land_clim=clim_land,
            t_sst_clim=clim_sst,
            t_surface_clim=clim_surface,
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

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
    ds_out.to_netcdf(OUTFILE, encoding=encoding)
    print(f"Wrote: {OUTFILE}")

    return ds_out


HUMIDITY_PRECIP_OUTFILE = PROC_DIR / "ref_humidity_precip_1deg_1981-2010.nc"


def build_humidity_precip_reference() -> xr.Dataset | None:
    """Download NCEP specific humidity and GPCP precipitation climatologies.

    Returns a Dataset with monthly climatology fields on the 1° target grid,
    or None if downloads fail.
    """
    try:
        shum_nc = fetch(URLS["humidity"], RAW_DIR)
        precip_nc = fetch(URLS["precip"], RAW_DIR)
    except Exception as e:
        print(f"Warning: could not download humidity/precip reference data: {e}")
        return None

    # --- Specific humidity (NCEP reanalysis, 2.5° grid) ---
    shum_ds = xr.open_dataset(shum_nc)
    shum_ds = to_0360(shum_ds, "lon")
    # Variable "shum" is near-surface specific humidity in g/kg (NCEP convention)
    shum = shum_ds["shum"]
    if "level" in shum.dims:
        # Take the lowest (highest pressure) level — near-surface
        shum = shum.isel(level=-1)
    shum = shum.sel(time=slice(BASELINE_START, BASELINE_END))
    shum_1deg = regrid_1deg(shum)
    clim_shum = shum_1deg.groupby("time.month").mean("time") / 1000.0  # g/kg → kg/kg

    # --- Precipitation (GPCP, 2.5° grid, mm/day) ---
    precip_ds = xr.open_dataset(precip_nc)
    precip_ds = to_0360(precip_ds, "lon")
    precip = precip_ds["precip"]
    precip = precip.sel(time=slice(BASELINE_START, BASELINE_END))
    precip_1deg = regrid_1deg(precip)
    clim_precip = precip_1deg.groupby("time.month").mean("time")

    ds_out = xr.Dataset(
        dict(
            shum_clim=clim_shum,       # kg/kg
            precip_clim=clim_precip,    # mm/day
        ),
        coords=dict(
            month=("month", np.arange(1, 13, dtype=int)),
            lat=("lat", LAT_T),
            lon=("lon", LON_T),
        ),
    )

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars}
    ds_out.to_netcdf(HUMIDITY_PRECIP_OUTFILE, encoding=encoding)
    print(f"Wrote humidity/precip reference: {HUMIDITY_PRECIP_OUTFILE}")

    return ds_out


def aggregate_humidity_precip_to_sim_grid(
    ds: xr.Dataset,
    lon2d_sim: np.ndarray,
    lat2d_sim: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate 1° humidity/precip reference onto the simulation grid."""
    lat_noaa = np.asarray(ds["lat"].values, dtype=float)
    lon_noaa = np.asarray(ds["lon"].values, dtype=float)

    lat_centers_sim = lat2d_sim[:, 0]
    lon_centers_sim = lon2d_sim[0, :]

    dlat_sim = float(lat_centers_sim[1] - lat_centers_sim[0]) if lat_centers_sim.size > 1 else 180.0
    dlon_sim = float(lon_centers_sim[1] - lon_centers_sim[0]) if lon_centers_sim.size > 1 else 360.0

    lat_edges_min = lat_centers_sim - 0.5 * dlat_sim
    lat_edges_max = lat_centers_sim + 0.5 * dlat_sim

    lon_noaa_wrapped = lon_noaa % 360.0
    lon_centers_sim_wrapped = lon_centers_sim % 360.0
    lon_edges_min_wrapped = (lon_centers_sim - 0.5 * dlon_sim) % 360.0
    lon_edges_max_wrapped = (lon_centers_sim + 0.5 * dlon_sim) % 360.0

    shum_src = np.asarray(ds["shum_clim"].values, dtype=float)
    precip_src = np.asarray(ds["precip_clim"].values, dtype=float)

    nmonth = shum_src.shape[0]
    nlat_sim = lat_centers_sim.size
    nlon_sim = lon_centers_sim.size

    shum_out = np.full((nmonth, nlat_sim, nlon_sim), np.nan, dtype=float)
    precip_out = np.full((nmonth, nlat_sim, nlon_sim), np.nan, dtype=float)

    lat_masks = [
        (lat_noaa >= lat_edges_min[i]) & (lat_noaa < lat_edges_max[i])
        for i in range(nlat_sim)
    ]

    for j in range(nlon_sim):
        lon_min = lon_edges_min_wrapped[j]
        lon_max = lon_edges_max_wrapped[j]
        if lon_min < lon_max:
            lon_mask = (lon_noaa_wrapped >= lon_min) & (lon_noaa_wrapped < lon_max)
        else:
            lon_mask = (lon_noaa_wrapped >= lon_min) | (lon_noaa_wrapped < lon_max)

        if not np.any(lon_mask):
            continue

        for i in range(nlat_sim):
            cell_mask = lat_masks[i][:, None] & lon_mask[None, :]
            if not np.any(cell_mask):
                continue
            with np.errstate(invalid="ignore"):
                shum_out[:, i, j] = np.nanmean(shum_src[:, cell_mask], axis=1)
                precip_out[:, i, j] = np.nanmean(precip_src[:, cell_mask], axis=1)

    return {"shum": shum_out, "precip": precip_out}


def compute_humidity_precip_statistics(
    sim_humidity: np.ndarray,
    sim_precip: np.ndarray,
    obs_shum: np.ndarray,
    obs_precip: np.ndarray,
    land_mask: np.ndarray,
    cell_areas: np.ndarray,
) -> None:
    """Print humidity and precipitation evaluation statistics."""
    weights_land = cell_areas * land_mask
    weights_ocean = cell_areas * (~land_mask)

    # Convert units for display: kg/kg → g/kg, kg/m²/s → mm/day
    sim_q_gkg = sim_humidity * 1000.0
    obs_q_gkg = obs_shum * 1000.0
    sim_p_mmday = sim_precip * 86400.0
    obs_p_mmday = obs_precip  # GPCP already in mm/day

    q_diff = sim_q_gkg - obs_q_gkg
    p_diff = sim_p_mmday - obs_p_mmday

    # Annual stats
    def stats_for(diff: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
        bias = weighted_mean(diff, weights)
        rmse = weighted_rmse(diff, weights)
        return bias, rmse

    print("\n" + "=" * 60)
    print("Humidity Evaluation (g/kg)")
    print("=" * 60)

    header = f"{'Month':<12}{'Land bias':>10}{'Ocean bias':>11}{'Global bias':>12}{'Global RMSE':>12}"
    print(header)
    print("-" * len(header))

    for m in range(12):
        lb, _ = stats_for(q_diff[m], weights_land)
        ob, _ = stats_for(q_diff[m], weights_ocean)
        gb, gr = stats_for(q_diff[m], cell_areas)
        print(f"{MONTH_NAMES[m]:<12}{lb:>10.2f}{ob:>11.2f}{gb:>12.2f}{gr:>12.2f}")

    lb, lr = stats_for(q_diff, weights_land)
    ob, orr = stats_for(q_diff, weights_ocean)
    gb, gr = stats_for(q_diff, cell_areas)
    print("-" * len(header))
    print(f"{'Annual':<12}{lb:>10.2f}{ob:>11.2f}{gb:>12.2f}{gr:>12.2f}")

    # Correlation
    corr_land = weighted_pattern_correlation(sim_q_gkg, obs_q_gkg, weights_land)
    corr_ocean = weighted_pattern_correlation(sim_q_gkg, obs_q_gkg, weights_ocean)
    corr_global = weighted_pattern_correlation(sim_q_gkg, obs_q_gkg, cell_areas)
    print(f"\nPattern correlation:  Land={corr_land:.3f}  Ocean={corr_ocean:.3f}  Global={corr_global:.3f}")

    # Area-weighted means
    sim_q_land = weighted_mean(sim_q_gkg, weights_land)
    obs_q_land = weighted_mean(obs_q_gkg, weights_land)
    sim_q_ocean = weighted_mean(sim_q_gkg, weights_ocean)
    obs_q_ocean = weighted_mean(obs_q_gkg, weights_ocean)
    print(f"Mean q:  Sim land={sim_q_land:.1f}  Obs land={obs_q_land:.1f}  "
          f"Sim ocean={sim_q_ocean:.1f}  Obs ocean={obs_q_ocean:.1f}")

    print("\n" + "=" * 60)
    print("Precipitation Evaluation (mm/day)")
    print("=" * 60)

    header = f"{'Month':<12}{'Land bias':>10}{'Ocean bias':>11}{'Global bias':>12}{'Global RMSE':>12}"
    print(header)
    print("-" * len(header))

    for m in range(12):
        lb, _ = stats_for(p_diff[m], weights_land)
        ob, _ = stats_for(p_diff[m], weights_ocean)
        gb, gr = stats_for(p_diff[m], cell_areas)
        print(f"{MONTH_NAMES[m]:<12}{lb:>10.2f}{ob:>11.2f}{gb:>12.2f}{gr:>12.2f}")

    lb, lr = stats_for(p_diff, weights_land)
    ob, orr = stats_for(p_diff, weights_ocean)
    gb, gr = stats_for(p_diff, cell_areas)
    print("-" * len(header))
    print(f"{'Annual':<12}{lb:>10.2f}{ob:>11.2f}{gb:>12.2f}{gr:>12.2f}")

    corr_land = weighted_pattern_correlation(sim_p_mmday, obs_p_mmday, weights_land)
    corr_ocean = weighted_pattern_correlation(sim_p_mmday, obs_p_mmday, weights_ocean)
    corr_global = weighted_pattern_correlation(sim_p_mmday, obs_p_mmday, cell_areas)
    print(f"\nPattern correlation:  Land={corr_land:.3f}  Ocean={corr_ocean:.3f}  Global={corr_global:.3f}")

    sim_p_land = weighted_mean(sim_p_mmday, weights_land)
    obs_p_land = weighted_mean(obs_p_mmday, weights_land)
    sim_p_ocean = weighted_mean(sim_p_mmday, weights_ocean)
    obs_p_ocean = weighted_mean(obs_p_mmday, weights_ocean)
    print(f"Mean P:  Sim land={sim_p_land:.2f}  Obs land={obs_p_land:.2f}  "
          f"Sim ocean={sim_p_ocean:.2f}  Obs ocean={obs_p_ocean:.2f}")


def plot_humidity_precip_anomaly(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    sim_humidity: np.ndarray,
    sim_precip: np.ndarray,
    obs_shum: np.ndarray,
    obs_precip: np.ndarray,
    headless: bool = False,
) -> None:
    """Plot humidity and precipitation anomaly maps."""
    if headless:
        return

    # Humidity anomaly (g/kg)
    q_anomaly = sim_humidity * 1000.0 - obs_shum * 1000.0
    q_with_annual = np.concatenate(
        [q_anomaly, np.nanmean(q_anomaly, axis=0, keepdims=True)], axis=0
    )

    q_vmax = min(float(np.nanmax(np.abs(q_anomaly))), 15.0)
    if not np.isfinite(q_vmax) or q_vmax <= 0:
        q_vmax = 5.0

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        q_with_annual,
        title="Specific Humidity Anomaly (Sim − Obs)",
        cmap=colormaps["BrBG"],
        norm=Normalize(vmin=-q_vmax, vmax=q_vmax),
        colorbar_label="Humidity anomaly (g/kg)",
    )

    # Precipitation anomaly (mm/day)
    p_anomaly = sim_precip * 86400.0 - obs_precip
    p_with_annual = np.concatenate(
        [p_anomaly, np.nanmean(p_anomaly, axis=0, keepdims=True)], axis=0
    )

    p_vmax = min(float(np.nanmax(np.abs(p_anomaly))), 8.0)
    if not np.isfinite(p_vmax) or p_vmax <= 0:
        p_vmax = 3.0

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        p_with_annual,
        title="Precipitation Anomaly (Sim − Obs)",
        cmap=colormaps["BrBG"],
        norm=Normalize(vmin=-p_vmax, vmax=p_vmax),
        colorbar_label="Precipitation anomaly (mm/day)",
    )


# ----------------------------
# Evaluation helpers
# ----------------------------

def aggregate_reference_to_sim_grid(
    ds: xr.Dataset,
    lon2d_sim: np.ndarray,
    lat2d_sim: np.ndarray,
) -> xr.Dataset:
    """Aggregate the 1° NOAA climatology onto the simulation grid via cell means.

    Each simulation grid cell is classified as land or ocean by the model's own
    land mask (not here), but the observational fields are averaged over all 1°
    NOAA cell centers that fall within the simulation cell's lat/lon bounds.
    """

    lat_noaa = np.asarray(ds["lat"].values, dtype=float)
    lon_noaa = np.asarray(ds["lon"].values, dtype=float)

    # Assume lon2d_sim/lat2d_sim come from a regular grid centered on cell middles.
    lat_centers_sim = lat2d_sim[:, 0]
    lon_centers_sim = lon2d_sim[0, :]

    if lat_centers_sim.size > 1:
        dlat_sim = float(lat_centers_sim[1] - lat_centers_sim[0])
    else:
        dlat_sim = 180.0
    if lon_centers_sim.size > 1:
        dlon_sim = float(lon_centers_sim[1] - lon_centers_sim[0])
    else:
        dlon_sim = 360.0

    lat_edges_min = lat_centers_sim - 0.5 * dlat_sim
    lat_edges_max = lat_centers_sim + 0.5 * dlat_sim
    lon_edges_min = lon_centers_sim - 0.5 * dlon_sim
    lon_edges_max = lon_centers_sim + 0.5 * dlon_sim

    # Ensure longitudes are in 0–360 for comparison, matching LAT_T/LON_T convention.
    lon_noaa_wrapped = lon_noaa % 360.0
    lon_centers_sim_wrapped = lon_centers_sim % 360.0
    lon_edges_min_wrapped = lon_edges_min % 360.0
    lon_edges_max_wrapped = lon_edges_max % 360.0

    t_land_src = np.asarray(ds["t_land_clim"].values, dtype=float)
    t_sst_src = np.asarray(ds["t_sst_clim"].values, dtype=float)

    nmonth = t_land_src.shape[0]
    nlat_sim = lat_centers_sim.size
    nlon_sim = lon_centers_sim.size

    t_land_out = np.full((nmonth, nlat_sim, nlon_sim), np.nan, dtype=float)
    t_sst_out = np.full((nmonth, nlat_sim, nlon_sim), np.nan, dtype=float)

    # Precompute latitude membership masks for efficiency.
    lat_masks = [
        (lat_noaa >= lat_edges_min[i]) & (lat_noaa < lat_edges_max[i])
        for i in range(nlat_sim)
    ]

    for j in range(nlon_sim):
        # Handle potential wrap-around in longitude: assume cell width is not huge.
        lon_min = lon_edges_min_wrapped[j]
        lon_max = lon_edges_max_wrapped[j]

        if lon_min < lon_max:
            lon_mask = (lon_noaa_wrapped >= lon_min) & (lon_noaa_wrapped < lon_max)
        else:
            # Cell crosses the 0° meridian in wrapped coordinates.
            lon_mask = (lon_noaa_wrapped >= lon_min) | (lon_noaa_wrapped < lon_max)

        if not np.any(lon_mask):
            continue

        for i in range(nlat_sim):
            lat_mask = lat_masks[i]
            cell_mask = lat_mask[:, None] & lon_mask[None, :]

            if not np.any(cell_mask):
                continue

            # Flatten spatial dimensions and take the mean over the overlapping NOAA cells.
            land_slice = t_land_src[:, cell_mask]
            sst_slice = t_sst_src[:, cell_mask]

            if land_slice.size > 0:
                with np.errstate(invalid="ignore"):
                    t_land_out[:, i, j] = np.nanmean(land_slice, axis=1)
            if sst_slice.size > 0:
                with np.errstate(invalid="ignore"):
                    t_sst_out[:, i, j] = np.nanmean(sst_slice, axis=1)

    # Build a dataset on the simulation grid; surface field will be constructed later
    # using the model land mask.
    ds_out = xr.Dataset(
        dict(
            t_land_clim=(("month", "lat", "lon"), t_land_out),
            t_sst_clim=(("month", "lat", "lon"), t_sst_out),
        ),
        coords=dict(
            month=("month", np.arange(1, nmonth + 1, dtype=int)),
            lat=("lat", lat_centers_sim),
            lon=("lon", lon_centers_sim_wrapped),
        ),
    )

    return ds_out


def _prepare_weighted_arrays(
    *arrays: np.ndarray, weights: np.ndarray
) -> tuple[tuple[np.ndarray, ...], np.ndarray, float] | None:
    """Prepare arrays and weights for weighted statistics computation.

    Returns:
        Tuple of (valid_arrays, valid_weights, total_weight) or None if no valid data
    """
    arrays_float = tuple(np.asarray(arr, dtype=float) for arr in arrays)
    weights_2d = np.asarray(weights, dtype=float)

    # Broadcast weights to match array shape
    ref_array = arrays_float[0]
    if ref_array.ndim == 2:
        weights_b = weights_2d
    elif ref_array.ndim == 3:
        weights_b = np.broadcast_to(weights_2d, ref_array.shape)
    else:
        raise ValueError("Arrays must be 2-D or 3-D")

    # Find valid data points (finite and positive weight)
    valid = np.isfinite(weights_b) & (weights_b > 0)
    for arr in arrays_float:
        valid &= np.isfinite(arr)

    if not np.any(valid):
        return None

    weights_valid = weights_b[valid]
    total_weight = np.sum(weights_valid)
    if total_weight <= 0:
        return None

    valid_arrays = tuple(arr[valid] for arr in arrays_float)
    return valid_arrays, weights_valid, total_weight


def weighted_rmse(diff: np.ndarray, weights: np.ndarray) -> float:
    """Compute the area-weighted RMSE, ignoring NaNs."""
    result = _prepare_weighted_arrays(diff, weights=weights)
    if result is None:
        return float("nan")

    (diff_valid,), weights_valid, total_weight = result
    rmse = np.sqrt(np.sum(weights_valid * diff_valid**2) / total_weight)
    return float(rmse)


def weighted_mean(diff: np.ndarray, weights: np.ndarray) -> float:
    """Compute the area-weighted mean (bias), ignoring NaNs."""
    result = _prepare_weighted_arrays(diff, weights=weights)
    if result is None:
        return float("nan")

    (diff_valid,), weights_valid, total_weight = result
    mean_val = np.sum(weights_valid * diff_valid) / total_weight
    return float(mean_val)


def weighted_pattern_correlation(
    sim: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the area-weighted pattern correlation coefficient.

    Pattern correlation measures how well the spatial patterns match between
    simulation and observation, independent of the mean bias.
    """
    result = _prepare_weighted_arrays(sim, obs, weights=weights)
    if result is None:
        return float("nan")

    (sim_valid, obs_valid), weights_valid, total_weight = result

    # Compute weighted means
    sim_mean = np.sum(weights_valid * sim_valid) / total_weight
    obs_mean = np.sum(weights_valid * obs_valid) / total_weight

    # Compute anomalies from weighted mean
    sim_anom = sim_valid - sim_mean
    obs_anom = obs_valid - obs_mean

    # Compute weighted covariance and variances
    covariance = np.sum(weights_valid * sim_anom * obs_anom) / total_weight
    sim_variance = np.sum(weights_valid * sim_anom**2) / total_weight
    obs_variance = np.sum(weights_valid * obs_anom**2) / total_weight

    # Compute correlation coefficient
    if sim_variance <= 0 or obs_variance <= 0:
        return float("nan")

    correlation = covariance / np.sqrt(sim_variance * obs_variance)
    return float(correlation)


def weighted_min_max(
    data: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Compute min and max over valid (finite, weighted) cells, ignoring NaNs."""
    result = _prepare_weighted_arrays(data, weights=weights)
    if result is None:
        return float("nan"), float("nan")

    (data_valid,), _, _ = result
    min_val = float(np.min(data_valid))
    max_val = float(np.max(data_valid))
    return min_val, max_val


def compute_temperature_statistics(
    temperature: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Compute area-weighted mean, min, and max temperatures."""

    mean_val = weighted_mean(temperature, weights)
    min_val, max_val = weighted_min_max(temperature, weights)

    return {
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
    }


def log_temperature_statistics(
    sim_t2m: np.ndarray,
    sim_sst: np.ndarray,
    obs_t2m: np.ndarray,
    obs_sst: np.ndarray,
    weights_land: np.ndarray,
    weights_ocean: np.ndarray,
    use_fahrenheit: bool,
) -> None:
    """Log area-weighted mean, min, max temperatures for simulation and observation."""

    unit = temperature_unit(use_fahrenheit)

    def convert(value: float) -> float:
        if not np.isfinite(value):
            return value
        return float(convert_temperature(value, use_fahrenheit, is_delta=False))

    def format_stats(stats: dict[str, float]) -> str:
        return f"mean={convert(stats['mean']):.2f}, min={convert(stats['min']):.2f}, max={convert(stats['max']):.2f}"

    # Compute all statistics
    datasets = [
        ("Sim Land T2m: ", sim_t2m, weights_land),
        ("Sim Ocean SST:", sim_sst, weights_ocean),
        ("Obs Land T2m: ", obs_t2m, weights_land),
        ("Obs Ocean SST:", obs_sst, weights_ocean),
    ]

    print(f"\nTemperature Statistics ({unit}):")
    for label, temperature, weights in datasets:
        stats = compute_temperature_statistics(temperature, weights)
        print(f"  {label} {format_stats(stats)}")


def compute_rmse_statistics(
    sim_surface: np.ndarray,
    sim_t2m: np.ndarray,
    obs: xr.Dataset,
    land_mask: np.ndarray,
    cell_areas: np.ndarray,
) -> tuple[
    list[dict[str, float]],
    dict[str, float],
    list[dict[str, float]],
    dict[str, float],
    list[dict[str, float]],
    dict[str, float],
    np.ndarray,
]:
    """Return monthly/annual RMSE, bias, and pattern correlation values, plus anomaly fields."""

    weights_land = cell_areas * land_mask
    weights_ocean = cell_areas * (~land_mask)

    obs_land = obs["t_land_clim"].values
    obs_sst = obs["t_sst_clim"].values
    obs_surface = obs["t_surface_clim"].values

    sim_combined = np.where(land_mask[None, ...], sim_t2m, sim_surface)

    land_diff = sim_t2m - obs_land
    ocean_diff = sim_surface - obs_sst
    global_diff = sim_combined - obs_surface

    monthly_rmse: list[dict[str, float]] = []
    monthly_bias: list[dict[str, float]] = []
    monthly_pattern_corr: list[dict[str, float]] = []
    for month_idx in range(sim_surface.shape[0]):
        monthly_rmse.append(
            {
                "land": weighted_rmse(land_diff[month_idx], weights_land),
                "ocean": weighted_rmse(ocean_diff[month_idx], weights_ocean),
                "global": weighted_rmse(global_diff[month_idx], cell_areas),
            }
        )
        monthly_bias.append(
            {
                "land": weighted_mean(land_diff[month_idx], weights_land),
                "ocean": weighted_mean(ocean_diff[month_idx], weights_ocean),
                "global": weighted_mean(global_diff[month_idx], cell_areas),
            }
        )
        monthly_pattern_corr.append(
            {
                "land": weighted_pattern_correlation(
                    sim_t2m[month_idx], obs_land[month_idx], weights_land
                ),
                "ocean": weighted_pattern_correlation(
                    sim_surface[month_idx], obs_sst[month_idx], weights_ocean
                ),
                "global": weighted_pattern_correlation(
                    sim_combined[month_idx], obs_surface[month_idx], cell_areas
                ),
            }
        )

    annual_rmse = {
        "land": weighted_rmse(land_diff, weights_land),
        "ocean": weighted_rmse(ocean_diff, weights_ocean),
        "global": weighted_rmse(global_diff, cell_areas),
    }
    annual_bias = {
        "land": weighted_mean(land_diff, weights_land),
        "ocean": weighted_mean(ocean_diff, weights_ocean),
        "global": weighted_mean(global_diff, cell_areas),
    }
    annual_pattern_corr = {
        "land": weighted_pattern_correlation(sim_t2m, obs_land, weights_land),
        "ocean": weighted_pattern_correlation(sim_surface, obs_sst, weights_ocean),
        "global": weighted_pattern_correlation(sim_combined, obs_surface, cell_areas),
    }

    return (
        monthly_rmse,
        annual_rmse,
        monthly_bias,
        annual_bias,
        monthly_pattern_corr,
        annual_pattern_corr,
        global_diff,
    )


def format_statistics_table(
    monthly: Iterable[dict[str, float]],
    annual: dict[str, float],
    title: str,
    unit_suffix: str = "",
    precision: int = 2,
    convert_values: bool = False,
    use_fahrenheit: bool = False,
) -> None:
    """Print a nicely formatted statistics table.

    Args:
        monthly: Monthly statistics with land/ocean/global keys
        annual: Annual statistics with land/ocean/global keys
        title: Table title
        unit_suffix: Unit to display in title (e.g., "°C")
        precision: Number of decimal places
        convert_values: Whether to apply temperature unit conversion
        use_fahrenheit: Whether to convert to Fahrenheit
    """

    def convert(value: float) -> float:
        if not np.isfinite(value):
            return value
        if convert_values:
            return float(convert_temperature(value, use_fahrenheit, is_delta=True))
        return value

    def fmt(value: float) -> str:
        if not np.isfinite(value):
            return "    —"
        return f"{value:8.{precision}f}"

    header = f"{'Month':<12}{'Land':>10}{'Ocean':>10}{'Global':>10}"
    title_with_unit = f"{title} ({unit_suffix})" if unit_suffix else title
    print(f"\n{title_with_unit}")
    print(header)
    print("-" * len(header))

    for name, row in zip(MONTH_NAMES, monthly):
        land = fmt(convert(row["land"]))
        ocean = fmt(convert(row["ocean"]))
        global_val = fmt(convert(row["global"]))
        print(f"{name:<12}{land:>10}{ocean:>10}{global_val:>10}")

    land_avg = fmt(convert(annual["land"]))
    ocean_avg = fmt(convert(annual["ocean"]))
    global_avg = fmt(convert(annual["global"]))
    print("-" * len(header))
    print(f"{'Annual':<12}{land_avg:>10}{ocean_avg:>10}{global_avg:>10}")


def format_rmse_table(
    monthly: Iterable[dict[str, float]],
    annual: dict[str, float],
    use_fahrenheit: bool,
) -> None:
    """Print a nicely formatted RMSE table."""
    unit = temperature_unit(use_fahrenheit)
    format_statistics_table(
        monthly,
        annual,
        title="Area-weighted RMSE",
        unit_suffix=unit,
        precision=2,
        convert_values=True,
        use_fahrenheit=use_fahrenheit,
    )


def format_bias_table(
    monthly: Iterable[dict[str, float]],
    annual: dict[str, float],
    use_fahrenheit: bool,
) -> None:
    """Print a nicely formatted Bias (area-weighted mean error) table."""
    unit = temperature_unit(use_fahrenheit)
    format_statistics_table(
        monthly,
        annual,
        title="Area-weighted Bias",
        unit_suffix=unit,
        precision=2,
        convert_values=True,
        use_fahrenheit=use_fahrenheit,
    )


def format_pattern_correlation_table(
    monthly: Iterable[dict[str, float]],
    annual: dict[str, float],
) -> None:
    """Print a nicely formatted Pattern Correlation table."""
    format_statistics_table(
        monthly,
        annual,
        title="Area-weighted Pattern Correlation",
        precision=3,
        convert_values=False,
    )


def compute_bias_corrected_anomaly(
    sim_surface: np.ndarray,
    sim_t2m: np.ndarray,
    obs_land: np.ndarray,
    obs_sst: np.ndarray,
    obs_surface: np.ndarray,
    land_mask: np.ndarray,
    annual_bias: dict[str, float],
) -> np.ndarray:
    """Compute anomaly with area-specific biases removed.

    Subtracts the annual mean bias over land from land cells and the annual
    mean bias over ocean from ocean cells before computing the final anomaly.
    """

    # Compute raw differences
    sim_combined = np.where(land_mask[None, ...], sim_t2m, sim_surface)
    land_diff = sim_t2m - obs_land
    ocean_diff = sim_surface - obs_sst

    # Subtract area-specific biases
    land_diff_corrected = land_diff - annual_bias["land"]
    ocean_diff_corrected = ocean_diff - annual_bias["ocean"]

    # Combine using land mask
    bias_corrected_anomaly = np.where(
        land_mask[None, ...],
        land_diff_corrected,
        ocean_diff_corrected
    )

    return bias_corrected_anomaly


def _compute_anomaly_display_range(
    anomaly: np.ndarray,
    use_fahrenheit: bool,
    max_display: float = 10.0,
) -> float:
    """Compute symmetric display range for anomaly plots."""
    max_abs = float(np.nanmax(np.abs(anomaly))) if anomaly.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 0.5
    display_max = float(convert_temperature(max_abs, use_fahrenheit, is_delta=True))
    if display_max <= 0:
        display_max = 0.5
    return float(np.minimum(display_max, max_display))


def _plot_anomaly(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    anomaly: np.ndarray,
    title: str,
    use_fahrenheit: bool,
) -> None:
    """Plot an anomaly field with annual mean appended."""
    display_max = _compute_anomaly_display_range(anomaly, use_fahrenheit)
    cmap = colormaps["RdBu_r"]
    norm = Normalize(vmin=-display_max, vmax=display_max)
    unit = temperature_unit(use_fahrenheit)

    anomaly_with_annual = np.concatenate(
        [anomaly, np.mean(anomaly, axis=0, keepdims=True)], axis=0
    )

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        anomaly_with_annual,
        title=title,
        cmap=cmap,
        norm=norm,
        colorbar_label=f"Temperature anomaly ({unit})",
        use_fahrenheit=use_fahrenheit,
        value_is_delta=True,
    )


def plot_baseline_and_anomaly(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    obs_surface: np.ndarray,
    anomaly: np.ndarray,
    use_fahrenheit: bool,
    headless: bool = False,
) -> None:
    """Generate baseline and anomaly plots."""
    if headless:
        return

    obs_with_annual = np.concatenate(
        [obs_surface, np.mean(obs_surface, axis=0, keepdims=True)], axis=0
    )
    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        obs_with_annual,
        title="NOAA 1981–2010 Monthly Climatology (incl. annual mean)",
        use_fahrenheit=use_fahrenheit,
    )

    _plot_anomaly(
        lon2d,
        lat2d,
        anomaly,
        title="Simulation − NOAA Surface Anomaly",
        use_fahrenheit=use_fahrenheit,
    )


def plot_bias_corrected_anomaly(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    bias_corrected_anomaly: np.ndarray,
    use_fahrenheit: bool,
    headless: bool = False,
) -> None:
    """Generate bias-corrected anomaly plot."""
    if headless:
        return

    _plot_anomaly(
        lon2d,
        lat2d,
        bias_corrected_anomaly,
        title="Bias-Corrected Anomaly (land/ocean biases removed)",
        use_fahrenheit=use_fahrenheit,
    )


def compute_cell_rmse(anomaly: np.ndarray) -> np.ndarray:
    """Compute RMSE at each grid cell across all months.

    Args:
        anomaly: 3-D array of shape (months, lat, lon) containing sim - obs differences

    Returns:
        2-D array of shape (lat, lon) with RMSE at each cell
    """
    return np.sqrt(np.mean(anomaly**2, axis=0))


def plot_eval_metrics(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    sim: np.ndarray,
    obs: np.ndarray,
    land_mask: np.ndarray,
    use_fahrenheit: bool,
    headless: bool = False,
) -> None:
    """Plot evaluation metrics with tabs for RMSE map and correlation scatter.

    Tab 1: Per-cell RMSE spatial map
    Tab 2: Obs vs Sim scatter plot showing correlation
    """
    if headless:
        return

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from scipy import stats

    unit = temperature_unit(use_fahrenheit)
    anomaly = sim - obs

    # Compute per-cell RMSE across months
    cell_rmse = compute_cell_rmse(anomaly)
    cell_rmse_display = convert_temperature(cell_rmse, use_fahrenheit, is_delta=True)

    # Compute display range for RMSE
    rmse_vmax = float(np.nanmax(cell_rmse_display))
    if not np.isfinite(rmse_vmax) or rmse_vmax <= 0:
        rmse_vmax = 10.0
    rmse_vmax = min(rmse_vmax, 6.0)

    rmse_cmap = colormaps["Purples"]  # Neutral colormap for error magnitude
    rmse_norm = Normalize(vmin=0, vmax=rmse_vmax)

    # Prepare scatter data - flatten all months and cells
    sim_display = convert_temperature(sim, use_fahrenheit, is_delta=False)
    obs_display = convert_temperature(obs, use_fahrenheit, is_delta=False)

    # Broadcast land_mask to match 3D shape
    land_mask_3d = np.broadcast_to(land_mask[None, ...], sim.shape)

    # Get valid (non-NaN) data points
    valid = np.isfinite(sim_display) & np.isfinite(obs_display)
    land_valid = valid & land_mask_3d
    ocean_valid = valid & ~land_mask_3d

    obs_land = obs_display[land_valid].ravel()
    sim_land = sim_display[land_valid].ravel()
    obs_ocean = obs_display[ocean_valid].ravel()
    sim_ocean = sim_display[ocean_valid].ravel()

    # Compute statistics
    obs_all = obs_display[valid].ravel()
    sim_all = sim_display[valid].ravel()

    r_all, _ = stats.pearsonr(obs_all, sim_all)
    r_land, _ = stats.pearsonr(obs_land, sim_land) if len(obs_land) > 2 else (np.nan, 0)
    r_ocean, _ = stats.pearsonr(obs_ocean, sim_ocean) if len(obs_ocean) > 2 else (np.nan, 0)

    # Linear regression for best-fit lines (overall, land, ocean)
    slope_all, intercept_all, _, _, _ = stats.linregress(obs_all, sim_all)
    slope_land, intercept_land, _, _, _ = stats.linregress(obs_land, sim_land) if len(obs_land) > 2 else (np.nan, np.nan, 0, 0, 0)
    slope_ocean, intercept_ocean, _, _, _ = stats.linregress(obs_ocean, sim_ocean) if len(obs_ocean) > 2 else (np.nan, np.nan, 0, 0, 0)

    # Create figure with two axes (we'll show/hide them with tabs)
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.1, right=0.88, bottom=0.12, top=0.92)

    # Map axis (for RMSE)
    projection = ccrs.PlateCarree()
    ax_map = fig.add_axes([0.1, 0.15, 0.75, 0.75], projection=projection)
    ax_map.set_global()
    ax_map.coastlines(linewidth=0.4)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#444444")
    ax_map.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "lakes", "110m", edgecolor="#000000", facecolor="none"
        ),
        linewidth=0.2,
    )

    mesh = ax_map.pcolormesh(
        lon2d,
        lat2d,
        cell_rmse_display,
        cmap=rmse_cmap,
        norm=rmse_norm,
        shading="auto",
        transform=projection,
    )
    ax_map.set_title(f"Per-Cell RMSE Across Months ({unit})")

    cbar_ax_map = fig.add_axes([0.87, 0.15, 0.02, 0.75])
    cbar_map = fig.colorbar(mesh, cax=cbar_ax_map)
    cbar_map.set_label(f"RMSE ({unit})")

    add_status_readout(fig, ax_map, lon2d, lat2d, cell_rmse_display, unit_label=unit)

    # Scatter axis (for correlation)
    ax_scatter = fig.add_axes([0.1, 0.15, 0.75, 0.75])

    # Use hexbin for density visualization (many points)
    data_min = min(obs_all.min(), sim_all.min())
    data_max = max(obs_all.max(), sim_all.max())
    padding = (data_max - data_min) * 0.05
    plot_min, plot_max = data_min - padding, data_max + padding

    # Hexbin for all data with density coloring
    hb = ax_scatter.hexbin(
        obs_all, sim_all,
        gridsize=50,
        cmap="Blues",
        mincnt=1,
        extent=[plot_min, plot_max, plot_min, plot_max],
    )

    # Overlay land/ocean distinction with transparent scatter
    ax_scatter.scatter(
        obs_land[::10], sim_land[::10],  # Subsample for visibility
        c="#8B4513", alpha=0.3, s=3, label=f"Land (r={r_land:.3f})", rasterized=True
    )
    ax_scatter.scatter(
        obs_ocean[::10], sim_ocean[::10],
        c="#1E90FF", alpha=0.3, s=3, label=f"Ocean (r={r_ocean:.3f})", rasterized=True
    )

    # 1:1 reference line
    ax_scatter.plot(
        [plot_min, plot_max], [plot_min, plot_max],
        "k--", linewidth=1.5, label="1:1 line"
    )

    # Best-fit regression lines for land and ocean
    fit_x = np.array([plot_min, plot_max])

    # Land fit line
    if np.isfinite(slope_land):
        fit_y_land = slope_land * fit_x + intercept_land
        ax_scatter.plot(
            fit_x, fit_y_land,
            color="#8B4513", linewidth=2, linestyle="-",
            label=f"Land fit: y={slope_land:.2f}x{intercept_land:+.1f}"
        )

    # Ocean fit line
    if np.isfinite(slope_ocean):
        fit_y_ocean = slope_ocean * fit_x + intercept_ocean
        ax_scatter.plot(
            fit_x, fit_y_ocean,
            color="#1E90FF", linewidth=2, linestyle="-",
            label=f"Ocean fit: y={slope_ocean:.2f}x{intercept_ocean:+.1f}"
        )

    ax_scatter.set_xlim(plot_min, plot_max)
    ax_scatter.set_ylim(plot_min, plot_max)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel(f"Observed Temperature ({unit})")
    ax_scatter.set_ylabel(f"Simulated Temperature ({unit})")
    ax_scatter.set_title(f"Obs vs Sim Correlation (r={r_all:.3f})")
    ax_scatter.legend(loc="upper left", fontsize=8)
    ax_scatter.grid(True, alpha=0.3)

    # Colorbar for hexbin density
    cbar_ax_scatter = fig.add_axes([0.87, 0.15, 0.02, 0.75])
    cbar_scatter = fig.colorbar(hb, cax=cbar_ax_scatter)
    cbar_scatter.set_label("Point density")

    # Initially hide scatter plot
    ax_scatter.set_visible(False)
    cbar_ax_scatter.set_visible(False)

    # State tracking
    current_state = {"view": "RMSE Map"}

    def update_view() -> None:
        if current_state["view"] == "RMSE Map":
            ax_map.set_visible(True)
            cbar_ax_map.set_visible(True)
            ax_scatter.set_visible(False)
            cbar_ax_scatter.set_visible(False)
        else:
            ax_map.set_visible(False)
            cbar_ax_map.set_visible(False)
            ax_scatter.set_visible(True)
            cbar_ax_scatter.set_visible(True)
        fig.canvas.draw_idle()

    # Radio buttons for view selection
    radio_ax = fig.add_axes([0.01, 0.4, 0.08, 0.15])
    radio_ax.set_title("View", fontsize=9)
    radio = RadioButtons(radio_ax, ["RMSE Map", "Correlation"], active=0)

    def on_view_change(label: str) -> None:
        current_state["view"] = label
        update_view()

    radio.on_clicked(on_view_change)

    plt.show()


# ----------------------------
# Main entry point
# ----------------------------


def main() -> None:
    args = _parse_args()

    reference = build_reference_climatology(args.mask_path)

    radiation_config = RadiationConfig(include_atmosphere=args.atmosphere)
    diffusion_config = DiffusionConfig(enabled=args.diffusion)
    snow_config = SnowAlbedoConfig(
        enabled=args.snow,
        latent_heat_enabled=args.latent_heat,
    )
    sensible_heat_config = SensibleHeatExchangeConfig(
        enabled=args.bulk_exchange,
        include_lapse_rate_elevation=args.lapse_rate_elevation,
    )
    latent_heat_config = LatentHeatExchangeConfig(
        enabled=args.latent_heat_exchange,
    )
    advection_config = AdvectionConfig(enabled=args.advection)
    orographic_config = OrographicConfig(enabled=args.orographic)
    model_config = ModelConfig(
        radiation=radiation_config,
        diffusion=diffusion_config,
        advection=advection_config,
        snow=snow_config,
        sensible_heat=sensible_heat_config,
        latent_heat=latent_heat_config,
        orographic=orographic_config,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.elliptical_orbit,
    )

    data_dir = os.getenv("DATA_DIR")
    assert data_dir is not None, "Please set the DATA_DIR environment variable to enable caching."
    data_dir = Path(data_dir)
    cache_path = data_dir / "main.npz"
    if args.cache:
        lon2d, lat2d = create_lat_lon_grid(args.resolution)
        with np.load(cache_path) as cached:
            layers = {k: cached[k] for k in cached}
    else:
        lon2d, lat2d, layers = solve_periodic_climate(
            resolution_deg=args.resolution,
            model_config=model_config,
            return_layer_map=True,
        )
        np.savez_compressed(cache_path, **layers)

    surface_cycle = layers["surface"]
    temperature_2m = layers.get("temperature_2m")

    if temperature_2m is None:
        print(
            "Warning: atmosphere layer disabled; using surface temperatures as a "
            "proxy for 2 m land temperatures."
        )
        sim_t2m = surface_cycle.copy()
    else:
        sim_t2m = temperature_2m

    # Apply interpolation if requested - upscale simulation to 1° for comparison
    if args.interpolate:
        print("Interpolating simulation output to 1° for evaluation...")
        with time_block("interpolation"):
            interp_layers = {"surface": surface_cycle}
            if temperature_2m is not None:
                interp_layers["temperature_2m"] = temperature_2m

            eval_lon2d, eval_lat2d, interpolated = interpolate_layer_map(
                interp_layers,
                lon2d,
                lat2d,
                output_resolution_deg=1.0,
                apply_lapse_rate_to_2m=True,
            )

            surface_cycle = interpolated["surface"]
            if "temperature_2m" in interpolated:
                sim_t2m = interpolated["temperature_2m"]
            else:
                sim_t2m = surface_cycle.copy()

        get_profiler().print_summary()
        print(f"Interpolation complete. Eval grid: {eval_lat2d.shape}")

        # Use interpolated grid for evaluation
        lon2d, lat2d = eval_lon2d, eval_lat2d

    land_mask = compute_land_mask(lon2d, lat2d)
    cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)

    # Clamp simulated SST to seawater freezing point (-1.8°C) over ocean cells,
    # matching how real SST observations handle sea ice.
    SEAWATER_FREEZING_POINT_C = -1.8
    ocean_mask_3d = ~land_mask[None, ...]
    surface_cycle = np.where(
        ocean_mask_3d,
        np.maximum(surface_cycle, SEAWATER_FREEZING_POINT_C),
        surface_cycle,
    )

    # Get observation data at the evaluation resolution
    if args.interpolate:
        # Use 1° NOAA reference directly (no aggregation needed)
        obs_land = reference["t_land_clim"].values
        obs_sst = reference["t_sst_clim"].values
        obs_surface = reference["t_surface_clim"].values
    else:
        # Aggregate NOAA reference data onto the simulation grid using cell-mean values.
        aggregated_obs = aggregate_reference_to_sim_grid(reference, lon2d, lat2d)
        obs_land = aggregated_obs["t_land_clim"].values
        obs_sst = aggregated_obs["t_sst_clim"].values
        obs_surface = np.where(land_mask[None, ...], obs_land, obs_sst)

    # Wrap fields into a lightweight Dataset so the existing
    # compute_rmse_statistics interface can be reused without modification.
    obs_for_stats = xr.Dataset(
        dict(
            t_land_clim=(("month", "lat", "lon"), obs_land),
            t_sst_clim=(("month", "lat", "lon"), obs_sst),
            t_surface_clim=(("month", "lat", "lon"), obs_surface),
        ),
        coords=dict(
            month=("month", np.arange(1, 13, dtype=int)),
            lat=("lat", lat2d[:, 0]),
            lon=("lon", lon2d[0, :]),
        ),
    )

    weights_land = cell_areas * land_mask
    weights_ocean = cell_areas * (~land_mask)

    (
        monthly_rmse,
        annual_rmse,
        monthly_bias,
        annual_bias,
        monthly_pattern_corr,
        annual_pattern_corr,
        anomaly,
    ) = compute_rmse_statistics(
        surface_cycle,
        sim_t2m,
        obs_for_stats,
        land_mask,
        cell_areas,
    )

    # Log temperature statistics
    log_temperature_statistics(
        sim_t2m,
        surface_cycle,
        obs_land,
        obs_sst,
        weights_land,
        weights_ocean,
        args.fahrenheit,
    )

    format_rmse_table(monthly_rmse, annual_rmse, args.fahrenheit)
    format_bias_table(monthly_bias, annual_bias, args.fahrenheit)
    format_pattern_correlation_table(monthly_pattern_corr, annual_pattern_corr)

    # --- Humidity and precipitation evaluation ---
    sim_humidity = layers.get("humidity")
    sim_precip = layers.get("precipitation")
    obs_shum: np.ndarray | None = None
    obs_precip: np.ndarray | None = None

    if HUMIDITY_PRECIP_OUTFILE.exists():
        hp_ref: xr.Dataset | None = xr.open_dataset(HUMIDITY_PRECIP_OUTFILE)
    else:
        hp_ref = build_humidity_precip_reference()

    if hp_ref is not None and sim_humidity is not None and sim_precip is not None:
        if args.interpolate:
            obs_shum = hp_ref["shum_clim"].values
            obs_precip = hp_ref["precip_clim"].values
        else:
            hp_agg = aggregate_humidity_precip_to_sim_grid(hp_ref, lon2d, lat2d)
            obs_shum = hp_agg["shum"]
            obs_precip = hp_agg["precip"]

        compute_humidity_precip_statistics(
            sim_humidity,
            sim_precip,
            obs_shum,
            obs_precip,
            land_mask,
            cell_areas,
        )
    elif hp_ref is not None:
        print("\nWarning: humidity/precipitation not in simulation output, skipping eval.")

    plot_baseline_and_anomaly(
        lon2d,
        lat2d,
        obs_surface,
        anomaly,
        args.fahrenheit,
        args.headless,
    )

    # Compute and plot bias-corrected anomaly
    bias_corrected_anomaly = compute_bias_corrected_anomaly(
        surface_cycle,
        sim_t2m,
        obs_land,
        obs_sst,
        obs_surface,
        land_mask,
        annual_bias,
    )

    plot_bias_corrected_anomaly(
        lon2d,
        lat2d,
        bias_corrected_anomaly,
        args.fahrenheit,
        args.headless,
    )

    # --- Humidity and precipitation anomaly plots ---
    if obs_shum is not None and obs_precip is not None:
        plot_humidity_precip_anomaly(
            lon2d,
            lat2d,
            sim_humidity,
            sim_precip,
            obs_shum,
            obs_precip,
            args.headless,
        )

    # Compute combined simulation field (T2m over land, surface over ocean)
    sim_combined = np.where(land_mask[None, ...], sim_t2m, surface_cycle)

    plot_eval_metrics(
        lon2d,
        lat2d,
        sim_combined,
        obs_surface,
        land_mask,
        args.fahrenheit,
        args.headless,
    )


if __name__ == "__main__":
    main()
