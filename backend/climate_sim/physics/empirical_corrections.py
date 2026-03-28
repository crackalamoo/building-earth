"""Empirical corrections for processes the model cannot resolve from first principles."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmpiricalCorrectionsConfig:
    """Configuration for empirical correction terms."""

    enabled: bool = True

    # AMOC thermohaline component: the inter-hemispheric overturning
    # that transports warm water northward through the Atlantic.
    #
    # The wind-driven Sverdrup-Stommel gyre delivers warm water to ~35N.
    # The thermohaline component is the ~4 Sv that doesn't recirculate
    # with the gyre but continues NE as the North Atlantic Current /
    # Drift toward the Nordic Seas where deep water forms (35-65N).
    #
    # We omit the tropical cross-equatorial component because our
    # surface-only model lacks the compensating deep return flow —
    # tropical northward advection would evacuate surface water
    # without replacement, causing spurious cooling.
    amoc_enabled: bool = True
    amoc_peak_velocity_m_s: float = 0.07  # ~4 Sv through 100m mixed layer

    # Geographic ice sheet mask parameters
    # Antarctica: all land south of this latitude
    ice_sheet_antarctic_lat: float = -60.0
    # Greenland: lat/lon bounding box
    ice_sheet_greenland_lat_min: float = 62.0
    ice_sheet_greenland_lat_max: float = 85.0
    ice_sheet_greenland_lon_min: float = 300.0  # 60°W
    ice_sheet_greenland_lon_max: float = 340.0  # 20°W
    # Heat capacity multiplier for ice sheet cells near/above freezing
    # Represents massive latent heat of kilometers-thick ice
    ice_sheet_heat_capacity_multiplier: float = 100.0


def compute_ice_sheet_mask(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    land_mask: np.ndarray,
    config: EmpiricalCorrectionsConfig,
) -> np.ndarray:
    """Compute a boolean mask for permanent ice sheet grid cells.

    Identifies Antarctica (all land south of -60°) and Greenland
    (land within a lat/lon bounding box).
    """
    lon_norm = lon2d % 360

    antarctica = (lat2d < config.ice_sheet_antarctic_lat) & land_mask

    # Greenland narrows toward the south and east coast curves westward.
    # Lat-dependent eastern bound avoids capturing Iceland:
    #   >= 70°N: east edge at lon_max (340°E / 20°W)
    #   62-70°N: east edge tapers to lon_max - 15° (325°E / 35°W)
    gl_east_bound = np.where(
        lat2d >= 70.0,
        config.ice_sheet_greenland_lon_max,
        config.ice_sheet_greenland_lon_max - 15.0 * (70.0 - np.clip(lat2d, 60.0, 70.0)) / 10.0,
    )
    greenland = (
        (lat2d >= config.ice_sheet_greenland_lat_min)
        & (lat2d <= config.ice_sheet_greenland_lat_max)
        & (lon_norm >= config.ice_sheet_greenland_lon_min)
        & (lon_norm <= gl_east_bound)
        & land_mask
    )

    return antarctica | greenland


def compute_amoc_velocity(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    land_mask: np.ndarray,
    config: EmpiricalCorrectionsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AMOC North Atlantic Current / Drift velocity field.

    The thermohaline overturning diverges from the wind-driven gyre
    at ~35-45N and continues NE toward the Nordic Seas. The flow
    path crosses from ~315E (45W) at 35N to ~355E (5W) at 65N.

    The taper is asymmetric: narrower on the western side, wider on
    the eastern side where the current fans out toward Europe. This
    matches the observed NAC branching pattern (Rockall Trough branch,
    shelf-edge current)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u_amoc, v_amoc) velocity components in m/s. NaN outside region.
    """
    u = np.full_like(lat2d, np.nan, dtype=float)
    v = np.full_like(lat2d, np.nan, dtype=float)

    if not config.enabled or not config.amoc_enabled:
        return u, v

    lon_norm = lon2d % 360
    ocean_mask = ~land_mask
    nlat, nlon = lat2d.shape
    lat_1d = lat2d[:, 0]
    v_peak = config.amoc_peak_velocity_m_s

    # Velocity envelope: ramp up from 35N, flat 40-57N, taper to 65N
    envelope = np.where(
        lat_1d < 35,
        0.0,
        np.where(
            lat_1d < 40,
            (lat_1d - 35.0) / 5.0,
            np.where(lat_1d < 57, 1.0, np.maximum(1.0 - (lat_1d - 57.0) / 8.0, 0.0)),
        ),
    )

    for j in range(nlat):
        lat_here = lat_1d[j]
        if lat_here < 35 or lat_here > 65 or envelope[j] == 0:
            continue

        t = (lat_here - 35.0) / 30.0  # 0 at 35N, 1 at 65N

        # Flow direction: NE at 45° from north
        angle = np.pi / 4

        # Path center: 315E (45W) at 35N -> 355E (5W) at 65N
        center_lon = 315.0 + t * 40.0
        # Asymmetric width: narrower west (10°), wider east (20°)
        # The NAC fans out eastward toward Europe
        width_west = 10.0
        width_east = 20.0

        for i in range(nlon):
            if not ocean_mask[j, i]:
                continue
            dlon = lon_norm[j, i] - center_lon
            if dlon > 180:
                dlon -= 360
            if dlon < -180:
                dlon += 360

            # Use asymmetric widths
            if dlon < 0:
                if abs(dlon) > width_west:
                    continue
                weight = 0.5 * (1.0 + np.cos(np.pi * dlon / width_west))
            else:
                if dlon > width_east:
                    continue
                weight = 0.5 * (1.0 + np.cos(np.pi * dlon / width_east))

            speed = v_peak * envelope[j] * weight

            v[j, i] = speed * np.cos(angle)
            u[j, i] = speed * np.sin(angle)

    return u, v
