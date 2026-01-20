"""Humidity utilities."""

from __future__ import annotations

import numpy as np

from climate_sim.physics.atmosphere.pressure import compute_pressure
from climate_sim.physics.atmosphere.hadley import LAT_POLES, LAT_SUBPOLAR, compute_itcz_latitude
from climate_sim.physics.atmosphere.pressure import LAT_SUBTROPICS_BASE, SUBTROPICS_ITCZ_COUPLING
from climate_sim.core.math_core import spherical_cell_area, compute_divergence
from climate_sim.core.timing import time_block
from climate_sim.data.constants import R_EARTH_METERS
from climate_sim.physics.vertical_motion import compute_subsidence_drying
from climate_sim.physics.precipitation import compute_precipitation_rate

# Mean RH and anomalies at key latitude bands (like pressure anomalies)
RH_MEAN = 0.65
DRH_ITCZ = +0.15           # Humid at ITCZ (convergence, rising air)
DRH_SUBTROPICS = -0.30     # Dry at subtropics (descending air, deserts)
DRH_SUBPOLAR = +0.10       # Moderately humid (storm tracks)
DRH_POLES = +0.20          # Humid at poles

# Ocean has higher RH overall
RH_MEAN_WATER = 0.75
DRH_ITCZ_WATER = +0.13
DRH_SUBTROPICS_WATER = -0.12
DRH_SUBPOLAR_WATER = +0.05
DRH_POLES_WATER = +0.15

# Width of humidity features (radians)
SIGMA_RH_ITCZ = np.deg2rad(10.0)       # ITCZ humid zone width
SIGMA_RH_SUBTROPICS = np.deg2rad(15.0)  # Subtropical dry zone width
SIGMA_RH_SUBPOLAR = np.deg2rad(12.0)   # Subpolar storm track width
SIGMA_RH_POLES = np.deg2rad(10.0)      # Polar humid zone width

# Moisture advection parameters
# Penetration length is longer for q advection (conserved quantity) vs RH
# ~2500 km allows monsoon moisture to reach interior continents
MOISTURE_PENETRATION_LENGTH = 2500e3  # meters, e-folding distance for moisture decay over land
RH_OCEAN_SOURCE = 0.80  # Relative humidity over ocean (moisture source)
MOISTURE_ADVECTION_ITERATIONS = 15  # Number of iterations for convergence


def advect_moisture_from_ocean(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    land_mask: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    temperature_field: np.ndarray | None = None,
    itcz_rad: np.ndarray | None = None,
    penetration_length: float = MOISTURE_PENETRATION_LENGTH,
    n_iterations: int = MOISTURE_ADVECTION_ITERATIONS,
) -> np.ndarray:
    """Compute specific humidity by advecting moisture from ocean sources.

    Ocean cells are moisture sources with q computed from RH pattern and temperature.
    Land cells receive q from upwind with exponential decay (representing
    precipitation loss, mixing, etc).

    We advect q (not RH) because specific humidity is the conserved quantity
    during adiabatic transport. RH changes with temperature even without
    moisture exchange.

    The decay follows: dq/ds = -q/L, giving q(s) = q_0 * exp(-s/L)
    where s is distance and L is the penetration length scale.
    """
    nlat, nlon = land_mask.shape
    ocean_mask = ~land_mask

    # Compute q_sat from temperature (using boundary layer temperature)
    if temperature_field is None:
        raise ValueError("temperature_field is required for q advection")

    temperature_C = temperature_field - 273.15
    e_sat = 6.112 * np.exp(17.67 * temperature_C / (temperature_C + 243.5))
    p_hPa = 1013.25  # Approximate surface pressure
    q_sat = (0.622 * e_sat) / (p_hPa - (1 - 0.622) * e_sat)

    # Ocean boundary layer humidity: set based on AIR temperature q_sat, not SST
    # The marine BL maintains ~78% RH due to rapid mixing with the surface.
    # Using a FIXED RH (not latitude-dependent) lets the evaporation physics
    # determine the actual moisture flux based on the q_sat(SST) - q deficit.
    #
    # Key physics insight: Over ocean, the relevant q_sat is at the AIR temperature
    # (boundary layer), not SST. The ocean is a moisture source, but the equilibrium
    # humidity in the marine BL is set by air temperature. This allows evaporative
    # cooling to work: hot SST → large q_sat(SST) - q_air deficit → strong evaporation
    # → cooling. If we set q = f(SST), this feedback breaks.
    rh_ocean = 0.78  # Fixed marine BL relative humidity
    q_ocean = rh_ocean * q_sat

    # Initialize: ocean = source q, land = 0
    q = np.where(ocean_mask, q_ocean, 0.0)

    # Compute grid spacings in meters
    lon_rad = np.deg2rad(lon2d)
    lat_rad = np.deg2rad(lat2d)

    # Grid spacing
    dlat_rad = np.abs(lat_rad[1, 0] - lat_rad[0, 0]) if nlat > 1 else np.deg2rad(1.0)
    dlon_rad = np.abs(lon_rad[0, 1] - lon_rad[0, 0]) if nlon > 1 else np.deg2rad(1.0)

    # Physical grid spacing (meters)
    dy = R_EARTH_METERS * dlat_rad  # Constant
    dx = R_EARTH_METERS * np.cos(lat_rad) * dlon_rad  # Varies with latitude

    # Iterate: propagate moisture from ocean following wind
    for _ in range(n_iterations):
        # Sample q from upwind location
        q_upwind = _sample_upwind(q, wind_u, wind_v, dx, dy)

        # Compute decay factor based on physical distance traveled
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.where(
                wind_speed > 0.1,
                np.sqrt((wind_u * dx / np.maximum(wind_speed, 0.1))**2 +
                        (wind_v * dy / np.maximum(wind_speed, 0.1))**2),
                0.5 * (dx + dy)  # Fallback for calm conditions
            )

        # Physical decay: exp(-distance / penetration_length)
        decay = np.exp(-dist / penetration_length)

        # Update: land gets upwind q with decay, ocean stays at source
        q = np.where(ocean_mask, q_ocean, q_upwind * decay)

    # Ensure valid range (small positive floor, capped at saturation)
    q = np.maximum(q, 1e-6)
    q = np.minimum(q, q_sat)

    return q


def _sample_upwind(
    field: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    dx: np.ndarray,
    dy: float,
) -> np.ndarray:
    """Sample field values from upwind locations using first-order upwind scheme.
    """
    nlat, nlon = field.shape

    # Upwind indices for zonal direction (periodic)
    # If u > 0 (eastward wind), upwind is to the west (j-1)
    # If u < 0 (westward wind), upwind is to the east (j+1)
    j_upwind_east = np.roll(np.arange(nlon), -1)  # j+1
    j_upwind_west = np.roll(np.arange(nlon), 1)   # j-1

    # Sample from west or east based on wind direction
    field_west = field[:, j_upwind_west]
    field_east = field[:, j_upwind_east]
    zonal_upwind = np.where(wind_u >= 0, field_west, field_east)

    # Upwind indices for meridional direction (bounded at poles)
    # If v > 0 (northward wind), upwind is to the south (i-1)
    # If v < 0 (southward wind), upwind is to the north (i+1)
    field_south = np.roll(field, 1, axis=0)
    field_north = np.roll(field, -1, axis=0)

    # Handle pole boundaries
    field_south[0, :] = field[0, :]  # South pole: no upwind from further south
    field_north[-1, :] = field[-1, :]  # North pole: no upwind from further north

    merid_upwind = np.where(wind_v >= 0, field_south, field_north)

    # Blend zonal and meridional based on wind magnitude
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_u = np.abs(wind_u) / np.maximum(wind_speed, 0.01)
        weight_v = np.abs(wind_v) / np.maximum(wind_speed, 0.01)

    # Weighted average of upwind samples
    upwind = weight_u * zonal_upwind + weight_v * merid_upwind

    # For very weak winds, use local value
    upwind = np.where(wind_speed < 0.1, field, upwind)

    return upwind


def compute_saturation_specific_humidity(
    temperature_K: np.ndarray,
    pressure_Pa: float = 101325.0,
) -> np.ndarray:
    """Compute saturation specific humidity from temperature.

    Uses the Magnus formula for saturation vapor pressure.
    """
    temperature_C = temperature_K - 273.15
    # Magnus formula for saturation vapor pressure (hPa)
    e_sat_hPa = 6.112 * np.exp(17.67 * temperature_C / (temperature_C + 243.5))
    p_hPa = pressure_Pa / 100.0
    # Convert to specific humidity
    q_sat = (0.622 * e_sat_hPa) / (p_hPa - (1 - 0.622) * e_sat_hPa)
    return q_sat


def specific_humidity_to_relative_humidity(
    q: np.ndarray,
    temperature_K: np.ndarray,
    itcz_rad: np.ndarray | None = None,
    lat2d: np.ndarray | None = None,
    lon2d: np.ndarray | None = None,
) -> np.ndarray:
    """Convert specific humidity to relative humidity.

    Parameters
    ----------
    q : np.ndarray
        Specific humidity (kg/kg).
    temperature_K : np.ndarray
        Temperature field in Kelvin.
    itcz_rad : np.ndarray | None
        ITCZ latitude in radians (optional).
    lat2d : np.ndarray | None
        Latitude field in degrees.
    lon2d : np.ndarray | None
        Longitude field in degrees.

    Returns
    -------
    np.ndarray
        Relative humidity as a fraction (0-1).
    """
    # Compute saturation vapor pressure (Magnus formula)
    # Magnus formula requires temperature in Celsius
    temperature_C = temperature_K - 273.15

    p_Pa = compute_pressure(temperature_K, itcz_rad=itcz_rad, lat2d=lat2d, lon2d=lon2d)
    p_hPa = p_Pa / 100.0  # Convert Pa to hPa

    # Magnus formula: e_sat in hPa
    e_sat = 6.112 * np.exp(17.67 * temperature_C / (temperature_C + 243.5))
    q_sat = (0.622 * e_sat) / (p_hPa - (1 - 0.622) * e_sat)

    # Compute RH = q / q_sat, clamped to valid range
    rh = q / np.maximum(q_sat, 1e-10)
    return np.clip(rh, 0.0, 1.0)


def relative_humidity_pattern(
    lat_rad: np.ndarray,
    itcz_rad: np.ndarray,
    rh_mean: float = RH_MEAN,
    drh_itcz: float = DRH_ITCZ,
    drh_subtropics: float = DRH_SUBTROPICS,
    drh_subpolar: float = DRH_SUBPOLAR,
    drh_poles: float = DRH_POLES,
) -> np.ndarray:
    """Compute relative humidity pattern using anomalies from mean.

    Like pressure, RH is computed as mean + sum of Gaussian anomalies:
    - ITCZ: positive anomaly (humid, rising air)
    - Subtropics: negative anomaly (dry, descending air)
    - Subpolar: positive anomaly (storm tracks)
    - Poles: positive anomaly (humid)

    Parameters
    ----------
    lat_rad : np.ndarray
        Latitude field in radians, shape (nlat, nlon).
    itcz_rad : np.ndarray
        ITCZ latitude in radians, shape (nlon,) or broadcast-compatible.
    rh_mean : float
        Mean relative humidity.
    drh_itcz, drh_subtropics, drh_subpolar, drh_poles : float
        RH anomalies at key latitude bands.

    Returns
    -------
    np.ndarray
        Relative humidity (0-1), same shape as lat_rad.
    """
    # Subtropical dry zones: base at ~29° with small ITCZ-tracking component
    # (matches pressure.py pattern exactly)
    lat_subtrop_north = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad
    lat_subtrop_south = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad

    # ITCZ humid anomaly
    rh_itcz = drh_itcz * np.exp(-((lat_rad - itcz_rad) / SIGMA_RH_ITCZ) ** 2)

    # Subtropical dry anomaly (both hemispheres)
    rh_subtrop = drh_subtropics * (
        np.exp(-((lat_rad - lat_subtrop_south) / SIGMA_RH_SUBTROPICS) ** 2)
        + np.exp(-((lat_rad - lat_subtrop_north) / SIGMA_RH_SUBTROPICS) ** 2)
    )

    # Subpolar humid anomaly (fixed latitudes, both hemispheres)
    rh_subpolar = drh_subpolar * (
        np.exp(-((lat_rad + LAT_SUBPOLAR) / SIGMA_RH_SUBPOLAR) ** 2)
        + np.exp(-((lat_rad - LAT_SUBPOLAR) / SIGMA_RH_SUBPOLAR) ** 2)
    )

    # Polar humid anomaly (fixed latitudes, both hemispheres)
    rh_polar = drh_poles * (
        np.exp(-((lat_rad + LAT_POLES) / SIGMA_RH_POLES) ** 2)
        + np.exp(-((lat_rad - LAT_POLES) / SIGMA_RH_POLES) ** 2)
    )

    rh = rh_mean + rh_itcz + rh_subtrop + rh_subpolar + rh_polar

    return np.clip(rh, 0.1, 0.98)

def compute_humidity_q(
    lat_2d: np.ndarray,
    temperature: np.ndarray,
    *,
    return_rh: bool = False,
    land_mask: np.ndarray | None = None,
    lon_2d: np.ndarray | None = None,
    itcz_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Compute humidity field (specific humidity or relative humidity).

    Parameters
    ----------
    lat_2d : np.ndarray
        Latitude field in degrees.
    temperature : np.ndarray
        Temperature field in Kelvin.
    return_rh : bool, optional
        If True, return relative humidity (0-1). If False, return specific humidity (kg/kg).
        Default is False.
    land_mask : np.ndarray | None, optional
        Boolean array indicating land cells (True) vs ocean cells (False).
    lon_2d : np.ndarray | None
        Longitude field in degrees. Used to compute ITCZ if itcz_rad not provided.
    itcz_rad : np.ndarray | None
        Pre-computed ITCZ latitude in radians, shape (nlon,).

    Returns
    -------
    np.ndarray
        Humidity field. If return_rh is False, returns specific humidity in kg/kg.
        If return_rh is True, returns relative humidity as a fraction (0-1).
    """
    # Convert latitude to radians for internal computation
    lat_2d_rad = np.deg2rad(lat_2d)

    # Compute ITCZ from temperature or use pre-computed
    if itcz_rad is None:
        with time_block("compute_itcz_in_humidity"):
            cell_areas = spherical_cell_area(lon_2d, lat_2d, earth_radius_m=R_EARTH_METERS)
            itcz = compute_itcz_latitude(temperature, lat_2d, cell_areas)
    else:
        itcz = itcz_rad

    nlat, nlon = temperature.shape

    # Broadcast ITCZ to 2D grid
    itcz_2d = np.broadcast_to(itcz[np.newaxis, :], (nlat, nlon))

    # Compute RH using analytical formula for land
    rh_land = relative_humidity_pattern(lat_2d_rad, itcz_2d)

    # Blend with ocean values at cell level if land mask is provided
    if land_mask is not None:
        ocean_mask = ~land_mask  # True for ocean cells

        if ocean_mask.any():
            # Ocean ITCZ is scaled by 0.75 (less land-ocean contrast)
            itcz_ocean_2d = np.broadcast_to((itcz * 0.75)[np.newaxis, :], (nlat, nlon))

            # Compute ocean RH with ocean-specific parameters
            rh_ocean = relative_humidity_pattern(
                lat_2d_rad, itcz_ocean_2d,
                RH_MEAN_WATER, DRH_ITCZ_WATER, DRH_SUBTROPICS_WATER,
                DRH_SUBPOLAR_WATER, DRH_POLES_WATER
            )

            # Blend: use ocean values for ocean cells, land values for land cells
            rh = np.where(ocean_mask, rh_ocean, rh_land)
        else:
            rh = rh_land
    else:
        rh = rh_land

    if return_rh:
        return rh

    # Magnus formula requires temperature in Celsius
    temperature_C = temperature - 273.15

    p_Pa = compute_pressure(temperature, lat2d=lat_2d, lon2d=lon_2d, itcz_rad=itcz)
    p_hPa = p_Pa / 100.0  # Convert Pa to hPa

    e_sat = 6.112 * np.exp(17.67 * temperature_C / (temperature_C + 243.5))
    q_sat = (0.622 * e_sat) / (p_hPa - (1 - 0.622) * e_sat)

    return q_sat * rh

def compute_cloud_cover(
    relative_humidity: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cloud cover fraction from relative humidity.
    
    Uses a physically-based parameterization where cloud cover depends on
    relative humidity with different thresholds for land and ocean:
        C = max(0, (RH - RH_threshold) / (1 - RH_threshold)) ^ power
    
    Parameters
    ----------
    relative_humidity : np.ndarray | None, optional
        Relative humidity field (0-1). If None, falls back to latitude-based
        prescription for backward compatibility.
    land_mask : np.ndarray | None, optional
        Boolean array indicating land cells (True) vs ocean cells (False).
        If provided, different values are used for land vs ocean in the fallback.
    temperature : np.ndarray | None, optional
        Temperature field (K).
    Returns
    -------
    np.ndarray
        Cloud cover fraction (0-1) at each grid point.
    """
    # Fallback to latitude-based prescription if RH not provided
    if relative_humidity is None:
        if temperature is None:
            raise ValueError("Either relative_humidity or temperature must be provided")
        return _compute_cloud_cover_latitude_fallback(temperature, land_mask)
    
    rh_crit = 0.35
    rh_max = 0.85
    cloud_param = np.clip((relative_humidity - rh_crit)/(rh_max - rh_crit), 0, 1)

    cloud_cover = 3*cloud_param**2 - 2*cloud_param**3
    cloud_cover = 0.07 + (0.8 - 0.07) * cloud_cover
    
    return cloud_cover


def _compute_cloud_cover_latitude_fallback(
    temperature: np.ndarray, land_mask: np.ndarray | None = None
) -> np.ndarray:
    """Fallback latitude-dependent cloud cover (for backward compatibility).

    Uses a simple parameterization based on latitude only:
        C = 0.65 - 2.59 * sin²(lat) + 3.55 * sin⁴(lat)
    """
    nlat, nlon = temperature.shape[1], temperature.shape[2]

    # Reconstruct the latitude grid (cell centers)
    lat_spacing = 180.0 / float(nlat)
    lat_centers = -90.0 + (np.arange(nlat, dtype=float) + 0.5) * lat_spacing

    # Create 2D latitude field matching the grid
    lat2d = lat_centers[:, np.newaxis]  # Shape: (nlat, 1)
    lat2d = np.broadcast_to(lat2d, (nlat, nlon))  # Shape: (nlat, nlon)

    # Convert to radians for trigonometric functions
    lat_rad = np.deg2rad(lat2d)

    # Compute the cloud cover formula
    sin_lat = np.sin(lat_rad)
    sin2_lat = sin_lat * sin_lat
    sin4_lat = sin2_lat * sin2_lat

    cloud_cover = 0.7 - 2.59 * sin2_lat + 3.55 * sin4_lat

    # Clamp to maximum of 0.9
    cloud_cover = np.minimum(cloud_cover, 0.9)

    return cloud_cover


def compute_humidity_and_precipitation(
    wind_u: np.ndarray | None,
    wind_v: np.ndarray | None,
    land_mask: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    temperature_field: np.ndarray,
    itcz_rad: np.ndarray | None = None,
    atmosphere_temperature: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    vertical_velocity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute humidity field and precipitation.

    If wind is available, uses wind-advected moisture from ocean with
    subsidence drying applied over land. If wind is not available,
    falls back to pattern-based humidity computation.

    Parameters
    ----------
    wind_u, wind_v : np.ndarray | None
        Wind components (m/s). If None, falls back to pattern-based humidity.
    land_mask : np.ndarray
        Boolean land mask.
    lat2d, lon2d : np.ndarray
        Coordinate grids (degrees).
    temperature_field : np.ndarray
        Boundary layer temperature (K).
    itcz_rad : np.ndarray | None
        ITCZ latitude per longitude (radians).
    atmosphere_temperature : np.ndarray | None
        Free atmosphere temperature (K). Used for MSE-based convective precipitation.
    elevation : np.ndarray | None
        Terrain elevation (m). Used for orographic precipitation.
    vertical_velocity : np.ndarray | None
        Large-scale vertical velocity (m/s). Positive = rising. Used to suppress
        convection in subsidence zones.
    """
    if wind_u is not None and wind_v is not None:
        # Advect specific humidity (conserved) from ocean to land
        humidity_field = advect_moisture_from_ocean(
            wind_u, wind_v,
            land_mask,
            lat2d,
            lon2d,
            temperature_field=temperature_field,
            itcz_rad=itcz_rad,
        )

        # Apply subsidence drying over LAND only: descending air brings dry air down
        # Over ocean, evaporation replenishes moisture so subsidence doesn't dry it out
        divergence = compute_divergence(wind_u, wind_v, lat2d, lon2d)
        # Subsidence drying is a tendency (kg/kg/s), apply for ~1 month
        # Use a scaling factor to convert instantaneous tendency to monthly effect
        seconds_per_month = 30.0 * 24.0 * 3600.0
        drying_tendency = compute_subsidence_drying(divergence, humidity_field)
        # Apply as multiplicative factor to avoid negative humidity
        # dq/q = (dq/dt * dt) / q = drying_tendency * dt / humidity_field
        drying_factor = 1.0 + (drying_tendency * seconds_per_month) / np.maximum(humidity_field, 1e-10)
        # Cap the drying: minimum factor of 0.8 means at most 20% reduction per month.
        # This allows subtropical deserts to maintain realistic humidity levels (~5-10 g/kg)
        # through a balance of advected moisture vs subsidence drying.
        drying_factor = np.clip(drying_factor, 0.8, 1.0)
        # Only apply over land - ocean evaporation maintains humidity despite subsidence
        drying_factor = np.where(land_mask, drying_factor, 1.0)
        humidity_field = humidity_field * drying_factor

        # Compute precipitation from wind and humidity
        # Pass temperature fields for MSE-based convective precipitation
        # Pass vertical velocity to suppress convection in subsidence zones
        precipitation_field = compute_precipitation_rate(
            humidity_field, wind_u, wind_v, lat2d, lon2d,
            T_surface_K=temperature_field,
            T_atm_K=atmosphere_temperature,
            elevation=elevation,
            vertical_velocity=vertical_velocity,
        )

        return humidity_field, precipitation_field
    else:
        # Fallback to latitude-pattern humidity when no wind available
        humidity_field = compute_humidity_q(
            lat2d,
            temperature_field,
            land_mask=land_mask,
            lon_2d=lon2d,
            itcz_rad=itcz_rad,
        )
        return humidity_field, None


# Column mass for moisture budget (kg/m²)
# Water vapor is concentrated in lower troposphere with scale height ~2 km
COLUMN_MASS_KG_M2 = 5000.0
