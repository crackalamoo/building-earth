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

import numpy as np
import pooch
import xarray as xr
from matplotlib import cm
from matplotlib.colors import Normalize

from climate_sim.modeling.diffusion import DiffusionConfig
from climate_sim.modeling.radiation import RadiationConfig
from climate_sim.modeling.sensible_heat_exchange import (
    SensibleHeatExchangeConfig,
)
from climate_sim.modeling.snow_albedo import SnowAlbedoConfig
from climate_sim.plotting import plot_monthly_temperature_cycle
from climate_sim.utils.atmosphere import adjust_temperature_by_elevation
from climate_sim.utils.landmask import compute_land_mask
from climate_sim.utils.math_core import spherical_cell_area
from climate_sim.utils.solver import compute_periodic_cycle_results
from climate_sim.utils.temperature import convert_temperature, temperature_unit

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
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=1.0,
        help="Grid resolution in degrees",
    )
    parser.add_argument(
        "--solar-constant",
        type=float,
        default=None,
        help="Override the solar constant (W m^-2)",
    )
    parser.add_argument(
        "--diffusion",
        dest="diffusion",
        action="store_true",
        default=True,
        help="Enable lateral diffusion (default)",
    )
    parser.add_argument(
        "--no-diffusion",
        dest="diffusion",
        action="store_false",
        help="Disable lateral diffusion",
    )

    default_atmosphere = RadiationConfig().include_atmosphere
    parser.add_argument(
        "--atmosphere",
        dest="atmosphere",
        action="store_true",
        default=default_atmosphere,
        help="Include an explicit atmospheric layer",
    )
    parser.add_argument(
        "--no-atmosphere",
        dest="atmosphere",
        action="store_false",
        help="Exclude the atmospheric layer",
    )

    parser.add_argument(
        "--snow",
        dest="snow",
        action="store_true",
        default=True,
        help="Enable diagnostic snow-albedo adjustments (default)",
    )
    parser.add_argument(
        "--no-snow",
        dest="snow",
        action="store_false",
        help="Disable snow-albedo adjustments",
    )
    parser.add_argument(
        "--latent-heat",
        dest="latent_heat",
        action="store_true",
        default=True,
        help="Include latent heat of fusion in the surface heat capacity",
    )
    parser.add_argument(
        "--no-latent-heat",
        dest="latent_heat",
        action="store_false",
        help="Disable the latent heat of fusion adjustment",
    )
    parser.add_argument(
        "--bulk-exchange",
        dest="bulk_exchange",
        action="store_true",
        default=True,
        help="Enable the neutral bulk sensible heat exchange model",
    )
    parser.add_argument(
        "--no-bulk-exchange",
        dest="bulk_exchange",
        action="store_false",
        help="Disable the neutral bulk sensible heat exchange model",
    )
    parser.add_argument(
        "--elliptical-orbit",
        dest="elliptical_orbit",
        action="store_true",
        default=True,
        help="Apply Earth's orbital eccentricity correction to insolation",
    )
    parser.add_argument(
        "--circular-orbit",
        dest="elliptical_orbit",
        action="store_false",
        help="Assume a circular orbit and disable the eccentricity correction",
    )
    parser.add_argument(
        "--fahrenheit",
        "-f",
        dest="fahrenheit",
        action="store_true",
        help="Display plots/statistics in Fahrenheit instead of Celsius",
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


def interpolate_to_sim_grid(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Interpolate the NOAA fields to the simulation grid."""

    coords = {
        "lat": xr.DataArray(lat, dims="lat"),
        "lon": xr.DataArray(lon, dims="lon"),
    }

    interpolated = ds.drop_vars("land_mask").interp(coords, method="linear")

    # xarray cannot interpolate boolean arrays directly, so cast to float prior to
    # interpolation and convert back after applying the nearest-neighbor scheme.
    mask_interp = (
        ds["land_mask"].astype("float32").interp(coords, method="nearest") >= 0.5
    )
    interpolated["land_mask"] = mask_interp.astype(bool)
    return interpolated


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


def compute_rmse_statistics(
    sim_surface: np.ndarray,
    sim_t2m: np.ndarray,
    obs: xr.Dataset,
    land_mask: np.ndarray,
    cell_areas: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float], np.ndarray]:
    """Return monthly RMSE values, annual RMSE values, and anomaly fields."""

    weights_land = cell_areas * land_mask
    weights_ocean = cell_areas * (~land_mask)

    obs_land = obs["t_land_clim"].values
    obs_sst = obs["t_sst_clim"].values
    obs_surface = obs["t_surface_clim"].values

    sim_combined = np.where(land_mask[None, ...], sim_t2m, sim_surface)

    land_diff = sim_t2m - obs_land
    ocean_diff = sim_surface - obs_sst
    global_diff = sim_combined - obs_surface

    monthly_results: list[dict[str, float]] = []
    for month_idx in range(sim_surface.shape[0]):
        monthly_results.append(
            {
                "land": weighted_rmse(land_diff[month_idx], weights_land),
                "ocean": weighted_rmse(ocean_diff[month_idx], weights_ocean),
                "global": weighted_rmse(global_diff[month_idx], cell_areas),
            }
        )

    annual_results = {
        "land": weighted_rmse(land_diff, weights_land),
        "ocean": weighted_rmse(ocean_diff, weights_ocean),
        "global": weighted_rmse(global_diff, cell_areas),
    }

    return monthly_results, annual_results, global_diff


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

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    header = f"{'Month':<12}{'Land':>10}{'Ocean':>10}{'Global':>10}"
    print("\nArea-weighted RMSE (" + unit + ")")
    print(header)
    print("-" * len(header))

    for name, row in zip(month_names, monthly):
        land = fmt(convert(row["land"]))
        ocean = fmt(convert(row["ocean"]))
        global_rmse = fmt(convert(row["global"]))
        print(f"{name:<12}{land:>10}{ocean:>10}{global_rmse:>10}")

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

    plot_monthly_temperature_cycle(
        lon2d,
        lat2d,
        obs_surface,
        title="NOAA 1981–2010 Monthly Climatology",
        use_fahrenheit=use_fahrenheit,
    )

    max_abs = float(np.nanmax(np.abs(anomaly))) if anomaly.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 0.5
    display_max = float(convert_temperature(max_abs, use_fahrenheit, is_delta=True))
    if display_max <= 0:
        display_max = 0.5

    cmap = cm.get_cmap("RdBu_r")
    norm = Normalize(vmin=-display_max, vmax=display_max)
    unit = temperature_unit(use_fahrenheit)

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
    sensible_heat_config = SensibleHeatExchangeConfig(enabled=args.bulk_exchange)

    lon2d, lat2d, layers = compute_periodic_cycle_results(
        resolution_deg=args.resolution,
        solar_constant=args.solar_constant,
        use_elliptical_orbit=args.elliptical_orbit,
        radiation_config=radiation_config,
        diffusion_config=diffusion_config,
        snow_config=snow_config,
        sensible_heat_config=sensible_heat_config,
        return_layer_map=True,
    )

    surface_cycle = layers["surface"]
    atmosphere_cycle = layers.get("atmosphere")

    if atmosphere_cycle is None:
        print(
            "Warning: atmosphere layer disabled; using surface temperatures as a "
            "proxy for 2 m land temperatures."
        )
        sim_t2m = surface_cycle.copy()
    else:
        delta_to_two_m = 2.0 - ATMOSPHERE_REFERENCE_HEIGHT_M
        sim_t2m = adjust_temperature_by_elevation(atmosphere_cycle, delta_to_two_m)

    lat_sim = lat2d[:, 0]
    lon_sim = lon2d[0, :]

    interpolated_obs = interpolate_to_sim_grid(reference, lat_sim, lon_sim)

    land_mask = compute_land_mask(lon2d, lat2d)
    cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=diffusion_config.earth_radius_m)

    monthly, annual, anomaly = compute_rmse_statistics(
        surface_cycle,
        sim_t2m,
        interpolated_obs,
        land_mask,
        cell_areas,
    )

    format_rmse_table(monthly, annual, args.fahrenheit)

    plot_baseline_and_anomaly(
        lon2d,
        lat2d,
        interpolated_obs["t_surface_clim"].values,
        anomaly,
        args.fahrenheit,
    )


if __name__ == "__main__":
    main()

