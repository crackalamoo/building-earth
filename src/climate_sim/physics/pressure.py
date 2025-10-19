"""Surface pressure estimation utilities driven by climate model fields."""

import numpy as np
from functools import lru_cache
from climate_sim.physics.diffusion import DiffusionConfig, DiffusionOperator, create_diffusion_operator
from climate_sim.core.math_core import area_weighted_mean
from climate_sim.data.constants import ATMOSPHERE_MASS, EARTH_SURFACE_AREA_M2, GAS_CONSTANT_J_KG_K

@lru_cache(maxsize=8)
def _grid_latitude_and_diffusion(shape: tuple[int, int]) -> tuple[np.ndarray, DiffusionOperator]:
    """Return latitude centres (deg) and a cached diffusion operator for the grid shape."""

    if len(shape) != 2:
        raise ValueError("Grid shape for diffusion support must have exactly two dimensions")

    nlat, nlon = shape
    if nlat <= 0 or nlon <= 0:
        raise ValueError("Grid shape must be strictly positive in both dimensions")

    lat_spacing = 180.0 / float(nlat)
    lon_spacing = 360.0 / float(nlon)

    lat_centres = -90.0 + (np.arange(nlat, dtype=float) + 0.5) * lat_spacing
    lon_centres = (np.arange(nlon, dtype=float) + 0.5) * lon_spacing
    lon2d, lat2d = np.meshgrid(lon_centres, lat_centres)

    heat_capacity = np.ones(shape, dtype=float)
    land_mask = np.zeros(shape, dtype=bool)
    diffusion_operator = create_diffusion_operator(
        lon2d,
        lat2d,
        heat_capacity,
        land_mask=land_mask,
        config=DiffusionConfig(),
    ).atmosphere

    return lat_centres, diffusion_operator


def _smooth_temperature_field(
    field: np.ndarray,
    diffusion_operator: "DiffusionOperator",
    *,
    passes: int = 10,
) -> np.ndarray:
    """Apply iterative metric-aware diffusion to construct a smoothed background field."""

    if not diffusion_operator.enabled or passes <= 0:
        return np.asarray(field, dtype=float)

    smoothed = np.asarray(field, dtype=float)
    max_rate = float(np.max(np.abs(diffusion_operator.diagonal)))
    if max_rate <= 0.0:
        return smoothed

    dt = 0.45 / max_rate
    for _ in range(passes):
        smoothed = smoothed + dt * diffusion_operator.tendency(smoothed)

    return smoothed


def pressure_from_temperature_elevation(
    temperature_K: np.ndarray,
    elevation_m: np.ndarray | None = None,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Compute surface pressure (Pa) from temperature and elevation using hydrostatic balance."""

    temperature = np.asarray(temperature_K, dtype=float)
    if temperature.ndim != 2:
        raise ValueError("temperature_K must be a 2-D latitude/longitude field")

    shape = temperature.shape
    if elevation_m is None:
        elevation = np.zeros(shape, dtype=float)
    else:
        elevation = np.asarray(elevation_m, dtype=float)
        if elevation.shape != shape:
            raise ValueError("Temperature and elevation fields must share the same shape")

    mean_p = ATMOSPHERE_MASS * gravity_m_s2 / EARTH_SURFACE_AREA_M2

    lat_deg, diffusion_operator = _grid_latitude_and_diffusion(shape)
    cos_lat = np.clip(np.cos(np.deg2rad(lat_deg)), 1.0e-6, None)
    weights = np.asarray(np.broadcast_to(cos_lat[:, None], shape), dtype=float)

    temp_smooth = _smooth_temperature_field(temperature, diffusion_operator, passes=10)

    target_mean = area_weighted_mean(temperature, weights)
    smooth_mean = area_weighted_mean(temp_smooth, weights)
    temp_smooth = temp_smooth + (target_mean - smooth_mean)

    dT = temp_smooth - area_weighted_mean(temp_smooth, weights)

    beta = 200.0
    dp_th = -beta * dT
    dp_th = dp_th - area_weighted_mean(dp_th, weights)

    t_ref_lat = area_weighted_mean(temp_smooth, weights, axis=1)
    t_ref_lat_2d = np.broadcast_to(t_ref_lat[:, None], shape)
    t_ref_safe = np.maximum(t_ref_lat_2d, 1.0)

    p_orog = mean_p * np.exp(-gravity_m_s2 * elevation / (GAS_CONSTANT_J_KG_K * t_ref_safe))

    p_surface = p_orog + dp_th
    p_surface = p_surface * (mean_p / area_weighted_mean(p_surface, weights))

    return p_surface