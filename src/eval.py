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

from climate_sim.physics.diffusion import DiffusionConfig
from climate_sim.physics.radiation import RadiationConfig
from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeConfig
from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeConfig
from climate_sim.physics.snow_albedo import SnowAlbedoConfig
from climate_sim.physics.atmosphere.advection import AdvectionConfig
from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.plotting import plot_monthly_temperature_cycle
from climate_sim.data.calendar import MONTH_NAMES
from climate_sim.data.constants import R_EARTH_METERS
from climate_sim.runtime.cli import add_common_model_arguments
from climate_sim.data.landmask import compute_land_mask
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.core.solver import compute_periodic_cycle_results
from climate_sim.core.units import convert_temperature, temperature_unit

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
}

BASELINE_START = "1981-01-01"
BASELINE_END = "2010-12-31"

LAT_T = np.arange(-89.5, 90.5, 1.0)
LON_T = np.arange(0.5, 360.5, 1.0)

OUTFILE = PROC_DIR / "ref_climatology_1deg_1981-2010.nc"
ATMOSPHERE_REFERENCE_HEIGHT_M = 5000.0


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
    NOAA cell centres that fall within the simulation cell's lat/lon bounds.
    """

    lat_noaa = np.asarray(ds["lat"].values, dtype=float)
    lon_noaa = np.asarray(ds["lon"].values, dtype=float)

    # Assume lon2d_sim/lat2d_sim come from a regular grid centred on cell middles.
    lat_centres_sim = lat2d_sim[:, 0]
    lon_centres_sim = lon2d_sim[0, :]

    if lat_centres_sim.size > 1:
        dlat_sim = float(lat_centres_sim[1] - lat_centres_sim[0])
    else:
        dlat_sim = 180.0
    if lon_centres_sim.size > 1:
        dlon_sim = float(lon_centres_sim[1] - lon_centres_sim[0])
    else:
        dlon_sim = 360.0

    lat_edges_min = lat_centres_sim - 0.5 * dlat_sim
    lat_edges_max = lat_centres_sim + 0.5 * dlat_sim
    lon_edges_min = lon_centres_sim - 0.5 * dlon_sim
    lon_edges_max = lon_centres_sim + 0.5 * dlon_sim

    # Ensure longitudes are in 0–360 for comparison, matching LAT_T/LON_T convention.
    lon_noaa_wrapped = lon_noaa % 360.0
    lon_centres_sim_wrapped = lon_centres_sim % 360.0
    lon_edges_min_wrapped = lon_edges_min % 360.0
    lon_edges_max_wrapped = lon_edges_max % 360.0

    t_land_src = np.asarray(ds["t_land_clim"].values, dtype=float)
    t_sst_src = np.asarray(ds["t_sst_clim"].values, dtype=float)

    nmonth = t_land_src.shape[0]
    nlat_sim = lat_centres_sim.size
    nlon_sim = lon_centres_sim.size

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
            lat=("lat", lat_centres_sim),
            lon=("lon", lon_centres_sim_wrapped),
        ),
    )

    return ds_out


def weighted_rmse(diff: np.ndarray, weights: np.ndarray) -> float:
    """Compute the area-weighted RMSE, ignoring NaNs."""

    diff_array = np.asarray(diff, dtype=float)
    weights_2d = np.asarray(weights, dtype=float)

    if diff_array.ndim == 2:
        weights_b = weights_2d
    elif diff_array.ndim == 3:
        weights_b = np.broadcast_to(weights_2d, diff_array.shape)
    else:
        raise ValueError("diff must be 2-D or 3-D")

    valid = np.isfinite(diff_array) & np.isfinite(weights_b) & (weights_b > 0)
    if not np.any(valid):
        return float("nan")

    diff_sq = diff_array[valid] ** 2
    weights_valid = weights_b[valid]

    total_weight = np.sum(weights_valid)
    if total_weight <= 0:
        return float("nan")

    rmse = np.sqrt(np.sum(weights_valid * diff_sq) / total_weight)
    return float(rmse)


def weighted_mean(diff: np.ndarray, weights: np.ndarray) -> float:
    """Compute the area-weighted mean (bias), ignoring NaNs."""

    diff_array = np.asarray(diff, dtype=float)
    weights_2d = np.asarray(weights, dtype=float)

    if diff_array.ndim == 2:
        weights_b = weights_2d
    elif diff_array.ndim == 3:
        weights_b = np.broadcast_to(weights_2d, diff_array.shape)
    else:
        raise ValueError("diff must be 2-D or 3-D")

    valid = np.isfinite(diff_array) & np.isfinite(weights_b) & (weights_b > 0)
    if not np.any(valid):
        return float("nan")

    weights_valid = weights_b[valid]
    total_weight = np.sum(weights_valid)
    if total_weight <= 0:
        return float("nan")

    mean_val = np.sum(weights_valid * diff_array[valid]) / total_weight
    return float(mean_val)


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
    np.ndarray,
]:
    """Return monthly/annual RMSE and bias values, and anomaly fields."""

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

    return monthly_rmse, annual_rmse, monthly_bias, annual_bias, global_diff


def format_rmse_table(
    monthly: Iterable[dict[str, float]],
    annual: dict[str, float],
    use_fahrenheit: bool,
) -> None:
    """Print a nicely formatted RMSE table."""

    unit = temperature_unit(use_fahrenheit)

    def convert(value: float) -> float:
        if not np.isfinite(value):
            return value
        return float(convert_temperature(value, use_fahrenheit, is_delta=True))

    def fmt(value: float) -> str:
        if not np.isfinite(value):
            return "    —"
        return f"{value:8.2f}"

    header = f"{'Month':<12}{'Land':>10}{'Ocean':>10}{'Global':>10}"
    print("\nArea-weighted RMSE (" + unit + ")")
    print(header)
    print("-" * len(header))

    for name, row in zip(MONTH_NAMES, monthly):
        land = fmt(convert(row["land"]))
        ocean = fmt(convert(row["ocean"]))
        global_rmse = fmt(convert(row["global"]))
        print(f"{name:<12}{land:>10}{ocean:>10}{global_rmse:>10}")

    land_avg = fmt(convert(annual["land"]))
    ocean_avg = fmt(convert(annual["ocean"]))
    global_avg = fmt(convert(annual["global"]))
    print("-" * len(header))
    print(f"{'Annual':<12}{land_avg:>10}{ocean_avg:>10}{global_avg:>10}")


def format_bias_table(
    monthly: Iterable[dict[str, float]],
    annual: dict[str, float],
    use_fahrenheit: bool,
) -> None:
    """Print a nicely formatted Bias (area-weighted mean error) table."""

    unit = temperature_unit(use_fahrenheit)

    def convert(value: float) -> float:
        if not np.isfinite(value):
            return value
        return float(convert_temperature(value, use_fahrenheit, is_delta=True))

    def fmt(value: float) -> str:
        if not np.isfinite(value):
            return "    —"
        return f"{value:8.2f}"

    header = f"{'Month':<12}{'Land':>10}{'Ocean':>10}{'Global':>10}"
    print("\nArea-weighted Bias (" + unit + ")")
    print(header)
    print("-" * len(header))

    for name, row in zip(MONTH_NAMES, monthly):
        land = fmt(convert(row["land"]))
        ocean = fmt(convert(row["ocean"]))
        global_bias = fmt(convert(row["global"]))
        print(f"{name:<12}{land:>10}{ocean:>10}{global_bias:>10}")

    land_avg = fmt(convert(annual["land"]))
    ocean_avg = fmt(convert(annual["ocean"]))
    global_avg = fmt(convert(annual["global"]))
    print("-" * len(header))
    print(f"{'Annual':<12}{land_avg:>10}{ocean_avg:>10}{global_avg:>10}")


def plot_baseline_and_anomaly(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    obs_surface: np.ndarray,
    anomaly: np.ndarray,
    use_fahrenheit: bool,
) -> None:
    """Generate baseline and anomaly plots."""

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

    max_abs = float(np.nanmax(np.abs(anomaly))) if anomaly.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 0.5
    display_max = float(convert_temperature(max_abs, use_fahrenheit, is_delta=True))
    if display_max <= 0:
        display_max = 0.5
    display_max = np.minimum(display_max, 10.0)

    cmap = colormaps["RdBu_r"]
    norm = Normalize(vmin=-display_max, vmax=display_max)
    unit = temperature_unit(use_fahrenheit)
    anomaly = np.concatenate([anomaly, np.mean(anomaly, axis=0, keepdims=True)], axis=0)

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        anomaly,
        title="Simulation − NOAA Surface Anomaly",
        cmap=cmap,
        norm=norm,
        colorbar_label=f"Temperature anomaly ({unit})",
        use_fahrenheit=use_fahrenheit,
        value_is_delta=True,
    )


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

    data_dir = os.getenv("DATA_DIR")
    assert data_dir is not None, "Please set the DATA_DIR environment variable to enable caching."
    data_dir = Path(data_dir)
    cache_path = data_dir / "main.npz"
    if args.cache:
        lon2d, lat2d = create_lat_lon_grid(args.resolution)
        with np.load(cache_path) as cached:
            layers = {k: cached[k] for k in cached}
    else:
        lon2d, lat2d, layers = compute_periodic_cycle_results(
            resolution_deg=args.resolution,
            solar_constant=args.solar_constant,
            use_elliptical_orbit=args.elliptical_orbit,
            radiation_config=radiation_config,
            diffusion_config=diffusion_config,
            advection_config=advection_config,
            snow_config=snow_config,
            sensible_heat_config=sensible_heat_config,
            latent_heat_config=latent_heat_config,
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

    land_mask = compute_land_mask(lon2d, lat2d)
    cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)

    # Aggregate NOAA reference data onto the simulation grid using cell-mean values.
    aggregated_obs = aggregate_reference_to_sim_grid(reference, lon2d, lat2d)
    obs_land = aggregated_obs["t_land_clim"].values
    obs_sst = aggregated_obs["t_sst_clim"].values
    obs_surface = np.where(land_mask[None, ...], obs_land, obs_sst)

    # Wrap aggregated fields back into a lightweight Dataset so the existing
    # compute_rmse_statistics interface can be reused without modification.
    obs_for_stats = xr.Dataset(
        dict(
            t_land_clim=(("month", "lat", "lon"), obs_land),
            t_sst_clim=(("month", "lat", "lon"), obs_sst),
            t_surface_clim=(("month", "lat", "lon"), obs_surface),
        ),
        coords=dict(
            month=("month", aggregated_obs["month"].values),
            lat=("lat", aggregated_obs["lat"].values),
            lon=("lon", aggregated_obs["lon"].values),
        ),
    )

    monthly_rmse, annual_rmse, monthly_bias, annual_bias, anomaly = compute_rmse_statistics(
        surface_cycle,
        sim_t2m,
        obs_for_stats,
        land_mask,
        cell_areas,
    )

    format_rmse_table(monthly_rmse, annual_rmse, args.fahrenheit)
    format_bias_table(monthly_bias, annual_bias, args.fahrenheit)

    plot_baseline_and_anomaly(
        lon2d,
        lat2d,
        obs_surface,
        anomaly,
        args.fahrenheit,
    )


if __name__ == "__main__":
    main()
