"""Ocean current computation using Sverdrup-Stommel balance.

Computes wind-driven ocean circulation from wind stress curl:
- Sverdrup: Interior transport V = curl(τ) / (ρβ)
- Stommel: Western boundary closure with friction

The streamfunction ψ is integrated from the eastern boundary westward,
with a western boundary layer that closes the gyre via friction.
"""

from dataclasses import dataclass

import numpy as np

from climate_sim.data.constants import R_EARTH_METERS

@dataclass(frozen=True)
class OceanAdvectionConfig:
    """Configuration for ocean heat advection by Sverdrup-Stommel currents."""
    enabled: bool = True
    include_stommel: bool = True  # Include western boundary closure

# Ocean properties
RHO_WATER = 1025.0  # kg/m³, seawater density
RHO_AIR = 1.225  # kg/m³, air density at sea level
C_D = 1.5e-3  # dimensionless, drag coefficient

# Earth rotation
OMEGA = 7.2921e-5  # rad/s, Earth rotation rate

# Latitude bounds for valid Sverdrup computation
MIN_LATITUDE_DEG = 5.0  # Exclude equatorial region (Sverdrup breaks down)
MAX_LATITUDE_DEG = 65.0  # Exclude polar regions

# Stommel friction parameter
# r has units of 1/s, typical values ~1e-6 to 1e-5
# This gives boundary layer width L = r/β ~ 50-100 km
STOMMEL_FRICTION = 5.0e-6  # 1/s, Rayleigh friction coefficient


def compute_beta(lat_deg: np.ndarray) -> np.ndarray:
    """Compute the meridional gradient of the Coriolis parameter.

    β = df/dy = (2Ω/R) cos(lat)

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude in degrees.

    Returns
    -------
    np.ndarray
        Beta in 1/(m·s), same shape as input.
    """
    lat_rad = np.deg2rad(lat_deg)
    return (2.0 * OMEGA / R_EARTH_METERS) * np.cos(lat_rad)


def compute_wind_stress(
    u: np.ndarray,
    v: np.ndarray,
    rho_air: float = RHO_AIR,
    c_d: float = C_D,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wind stress from 10m wind components.

    Uses bulk formula: τ = ρ_air × C_D × |U| × U

    Parameters
    ----------
    u : np.ndarray
        Zonal wind component (m/s), shape (nlat, nlon).
    v : np.ndarray
        Meridional wind component (m/s), shape (nlat, nlon).
    rho_air : float
        Air density in kg/m³.
    c_d : float
        Drag coefficient (dimensionless).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (tau_x, tau_y) wind stress components in N/m² (Pa).
    """
    wind_speed = np.sqrt(u**2 + v**2)
    tau_x = rho_air * c_d * u * wind_speed
    tau_y = rho_air * c_d * v * wind_speed
    return tau_x, tau_y


def compute_wind_stress_curl(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    lat_deg: np.ndarray,
    resolution_deg: float,
) -> np.ndarray:
    """Compute wind stress curl on a spherical grid.

    curl(τ) = ∂τ_y/∂x - ∂τ_x/∂y + τ_y tan(lat)/R

    Parameters
    ----------
    tau_x : np.ndarray
        Zonal wind stress (N/m²), shape (nlat, nlon).
    tau_y : np.ndarray
        Meridional wind stress (N/m²), shape (nlat, nlon).
    lat_deg : np.ndarray
        Latitude centers (degrees), shape (nlat,).
    resolution_deg : float
        Grid resolution in degrees.

    Returns
    -------
    np.ndarray
        Wind stress curl (N/m³), shape (nlat, nlon).
    """
    # Grid spacing in meters
    dy = R_EARTH_METERS * np.deg2rad(resolution_deg)
    dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(np.deg2rad(lat_deg))[:, np.newaxis]

    # Spatial derivatives
    dtau_y_dx = np.gradient(tau_y, axis=1) / dx
    dtau_x_dy = np.gradient(tau_x, axis=0) / dy

    # Spherical metric correction
    lat_rad = np.deg2rad(lat_deg)[:, np.newaxis]
    metric_term = tau_y * np.tan(lat_rad) / R_EARTH_METERS

    return dtau_y_dx - dtau_x_dy + metric_term


def compute_sverdrup_transport(
    curl_tau: np.ndarray,
    lat_deg: np.ndarray,
    rho_water: float = RHO_WATER,
) -> np.ndarray:
    """Compute Sverdrup meridional transport from wind stress curl.

    Sverdrup balance: β V = curl(τ) / ρ
    Therefore: V = curl(τ) / (ρ β)

    Parameters
    ----------
    curl_tau : np.ndarray
        Wind stress curl (N/m³), shape (nlat, nlon).
    lat_deg : np.ndarray
        Latitude centers (degrees), shape (nlat,).
    rho_water : float
        Seawater density in kg/m³.

    Returns
    -------
    np.ndarray
        Meridional volume transport per unit width (m²/s), shape (nlat, nlon).
        Positive = northward. NaN outside valid latitude range.
    """
    beta = compute_beta(lat_deg)[:, np.newaxis]

    # Mask invalid latitudes
    lat_2d = np.broadcast_to(lat_deg[:, np.newaxis], curl_tau.shape)
    valid = (np.abs(lat_2d) >= MIN_LATITUDE_DEG) & (np.abs(lat_2d) <= MAX_LATITUDE_DEG)

    # Avoid division by zero (beta is small near poles, but we exclude those)
    beta_safe = np.where(valid, beta, np.nan)

    V = curl_tau / (rho_water * beta_safe)
    return np.where(valid, V, np.nan)


def sverdrup_transport_to_velocity(
    V_transport: np.ndarray,
    ocean_mask: np.ndarray,
    mixed_layer_depth: float = 100.0,
) -> np.ndarray:
    """Convert Sverdrup transport to meridional velocity.

    Parameters
    ----------
    V_transport : np.ndarray
        Meridional transport per unit width (m²/s), shape (nlat, nlon).
    ocean_mask : np.ndarray
        Boolean mask, True for ocean cells.
    mixed_layer_depth : float
        Depth over which transport is distributed (meters).

    Returns
    -------
    np.ndarray
        Meridional velocity (m/s), positive northward.
    """
    v = V_transport / mixed_layer_depth
    return np.where(ocean_mask, v, np.nan)


def find_basin_boundaries(
    ocean_mask: np.ndarray,
    lat_idx: int,
) -> list[tuple[int, int]]:
    """Find ocean basin boundaries at a given latitude.

    Returns list of (west_idx, east_idx) pairs for each contiguous ocean segment.
    Handles periodic longitude wrapping.

    Parameters
    ----------
    ocean_mask : np.ndarray
        Boolean mask, True for ocean cells, shape (nlat, nlon).
    lat_idx : int
        Latitude index to analyze.

    Returns
    -------
    list[tuple[int, int]]
        List of (west_boundary_idx, east_boundary_idx) for each basin.
        Indices are in the range [0, nlon).
    """
    row = ocean_mask[lat_idx, :]
    nlon = len(row)

    if not np.any(row):
        return []

    # Find transitions from land to ocean and ocean to land
    # Extend row to handle wraparound
    row_ext = np.concatenate([row[-1:], row, row[:1]])

    basins = []
    in_ocean = False
    west_idx = 0

    for i in range(1, nlon + 1):
        if row_ext[i] and not row_ext[i - 1]:
            # Transition from land to ocean: western boundary
            west_idx = i - 1  # Convert back to original index
            in_ocean = True
        elif not row_ext[i] and row_ext[i - 1] and in_ocean:
            # Transition from ocean to land: eastern boundary
            east_idx = i - 2  # Last ocean cell
            basins.append((west_idx, east_idx))
            in_ocean = False

    # Handle case where ocean wraps around
    if in_ocean:
        # Check if we started in ocean
        if row_ext[1]:
            # Find where the ocean ends in the wrapped portion
            for i in range(1, nlon + 1):
                if not row_ext[i]:
                    east_idx = i - 2
                    basins.append((west_idx, east_idx))
                    break
        else:
            # Ocean extends to the end but not wrapped
            east_idx = nlon - 1
            basins.append((west_idx, east_idx))

    return basins


def compute_streamfunction(
    sverdrup_transport: np.ndarray,
    ocean_mask: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    resolution_deg: float,
) -> np.ndarray:
    """Integrate Sverdrup transport to get streamfunction.

    Integrates from eastern boundary westward: ψ(x) = ∫[x to x_east] V dx
    Sets ψ = 0 at eastern boundary of each basin.

    Parameters
    ----------
    sverdrup_transport : np.ndarray
        Meridional transport per unit width (m²/s), shape (nlat, nlon).
    ocean_mask : np.ndarray
        Boolean mask, True for ocean cells.
    lon_deg : np.ndarray
        Longitude centers (degrees), shape (nlon,).
    lat_deg : np.ndarray
        Latitude centers (degrees), shape (nlat,).
    resolution_deg : float
        Grid resolution in degrees.

    Returns
    -------
    np.ndarray
        Streamfunction in Sverdrups (10^6 m³/s), shape (nlat, nlon).
        Positive = clockwise gyre (NH subtropical), negative = counterclockwise.
    """
    nlat, nlon = sverdrup_transport.shape
    psi = np.full((nlat, nlon), np.nan)

    for j in range(nlat):
        lat = lat_deg[j]

        # Skip invalid latitudes
        if abs(lat) < MIN_LATITUDE_DEG or abs(lat) > MAX_LATITUDE_DEG:
            continue

        # Grid spacing at this latitude (in meters)
        dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(np.deg2rad(lat))

        # Find basin boundaries
        basins = find_basin_boundaries(ocean_mask, j)

        for west_idx, east_idx in basins:
            # Set ψ = 0 at eastern boundary
            psi[j, east_idx] = 0.0

            # Integrate westward: ψ(i) = ψ(i+1) - V(i) × dx
            # Handle wraparound if basin crosses the dateline
            if west_idx <= east_idx:
                # Normal case: basin doesn't wrap
                for i in range(east_idx - 1, west_idx - 1, -1):
                    V = sverdrup_transport[j, i]
                    if np.isnan(V):
                        V = 0.0
                    psi[j, i] = psi[j, i + 1] - V * dx
            else:
                # Basin wraps around (west_idx > east_idx)
                # First integrate from east_idx down to 0
                for i in range(east_idx - 1, -1, -1):
                    V = sverdrup_transport[j, i]
                    if np.isnan(V):
                        V = 0.0
                    psi[j, i] = psi[j, i + 1] - V * dx
                # Then from nlon-1 to west_idx, continuing from index 0
                psi[j, nlon - 1] = psi[j, 0] - sverdrup_transport[j, nlon - 1] * dx
                for i in range(nlon - 2, west_idx - 1, -1):
                    V = sverdrup_transport[j, i]
                    if np.isnan(V):
                        V = 0.0
                    psi[j, i] = psi[j, i + 1] - V * dx

    # Convert from m³/s to Sverdrups (10^6 m³/s)
    psi_sv = psi / 1.0e6

    return psi_sv


def compute_stommel_boundary(
    psi_interior: np.ndarray,
    ocean_mask: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    resolution_deg: float,
    friction: float = STOMMEL_FRICTION,
) -> np.ndarray:
    """Add Stommel western boundary layer to close the gyre.

    The boundary layer width is L = r/β, and the streamfunction decays
    exponentially from the western boundary inward.

    The total streamfunction is: ψ_total = ψ_interior + ψ_boundary
    where ψ_boundary ensures mass conservation (no flow through coasts).

    Parameters
    ----------
    psi_interior : np.ndarray
        Interior (Sverdrup) streamfunction in Sv, shape (nlat, nlon).
    ocean_mask : np.ndarray
        Boolean mask, True for ocean cells.
    lon_deg : np.ndarray
        Longitude centers (degrees), shape (nlon,).
    lat_deg : np.ndarray
        Latitude centers (degrees), shape (nlat,).
    resolution_deg : float
        Grid resolution in degrees.
    friction : float
        Stommel friction coefficient r (1/s).

    Returns
    -------
    np.ndarray
        Total streamfunction with boundary layer closure (Sv).
    """
    nlat, nlon = psi_interior.shape
    psi_total = psi_interior.copy()

    for j in range(nlat):
        lat = lat_deg[j]

        # Skip invalid latitudes
        if abs(lat) < MIN_LATITUDE_DEG or abs(lat) > MAX_LATITUDE_DEG:
            continue

        # Compute boundary layer width L = r/β
        beta = compute_beta(np.array([lat]))[0]
        L = friction / beta  # meters

        # Grid spacing at this latitude
        dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(np.deg2rad(lat))

        # Find basin boundaries
        basins = find_basin_boundaries(ocean_mask, j)

        for west_idx, east_idx in basins:
            # Value at western boundary (what needs to be closed)
            psi_west = psi_interior[j, west_idx]
            if np.isnan(psi_west):
                continue

            # The boundary correction decays exponentially from the west
            # We need ψ_total = 0 at the western coast (no flow through)
            # So ψ_boundary = -ψ_west × exp(-distance/L)

            if west_idx <= east_idx:
                # Normal basin
                for i in range(west_idx, east_idx + 1):
                    # Distance from western boundary (in meters)
                    dist = (i - west_idx) * dx
                    decay = np.exp(-dist / L)
                    psi_total[j, i] = psi_interior[j, i] - psi_west * decay
            else:
                # Wrapped basin
                for i in range(west_idx, nlon):
                    dist = (i - west_idx) * dx
                    decay = np.exp(-dist / L)
                    if not np.isnan(psi_interior[j, i]):
                        psi_total[j, i] = psi_interior[j, i] - psi_west * decay
                for i in range(0, east_idx + 1):
                    dist = (nlon - west_idx + i) * dx
                    decay = np.exp(-dist / L)
                    if not np.isnan(psi_interior[j, i]):
                        psi_total[j, i] = psi_interior[j, i] - psi_west * decay

    return psi_total


def streamfunction_to_velocity(
    psi: np.ndarray,
    ocean_mask: np.ndarray,
    lat_deg: np.ndarray,
    resolution_deg: float,
    mixed_layer_depth: float = 100.0,
    friction: float = STOMMEL_FRICTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive velocity components from streamfunction.

    u = -(1/R) ∂ψ/∂lat  (scaled by depth)
    v = (1/(R cos lat)) ∂ψ/∂lon  (scaled by depth)

    For western boundary currents, the velocity is computed using the true
    Stommel boundary layer width (L = r/β) rather than the grid spacing,
    to correctly represent the narrow, fast WBC at coarse resolution.

    Parameters
    ----------
    psi : np.ndarray
        Streamfunction in Sverdrups, shape (nlat, nlon).
    ocean_mask : np.ndarray
        Boolean mask, True for ocean cells.
    lat_deg : np.ndarray
        Latitude centers (degrees), shape (nlat,).
    resolution_deg : float
        Grid resolution in degrees.
    mixed_layer_depth : float
        Depth over which transport is distributed (meters).
    friction : float
        Stommel friction coefficient r (1/s), used for boundary layer width.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u, v) velocity components in m/s.
    """
    nlat, nlon = psi.shape

    # Convert from Sv to m³/s
    psi_m3s = psi * 1.0e6

    # Grid spacing
    dy = R_EARTH_METERS * np.deg2rad(resolution_deg)
    cos_lat = np.cos(np.deg2rad(lat_deg))[:, np.newaxis]
    dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * cos_lat

    # Compute derivatives (handle NaN by filling temporarily)
    psi_filled = np.where(np.isnan(psi_m3s), 0.0, psi_m3s)

    # ∂ψ/∂lat (axis 0) - for zonal velocity
    dpsi_dy = np.gradient(psi_filled, dy, axis=0)

    # ∂ψ/∂lon (axis 1) - for meridional velocity
    # Use central differences for interior
    dpsi_dx = np.gradient(psi_filled, axis=1) / dx

    # Velocities from interior gradients
    u = -dpsi_dy / mixed_layer_depth
    v = dpsi_dx / mixed_layer_depth

    # Western boundary current correction:
    # At coarse resolution, the WBC is spread over one grid cell (~500km)
    # but should be concentrated in the Stommel boundary layer (L = r/β ~ 50-100km).
    # This makes the WBC velocity ~10x too slow.
    #
    # The grid-based velocity is: v_grid = Δψ / (dx * H)
    # The true WBC velocity should be: v_true = Δψ / (L_stommel * H)
    # So: v_true = v_grid * (dx / L_stommel)
    for j in range(nlat):
        lat = lat_deg[j]
        if abs(lat) < MIN_LATITUDE_DEG or abs(lat) > MAX_LATITUDE_DEG:
            continue

        # Stommel boundary layer width at this latitude
        beta = compute_beta(np.array([lat]))[0]
        L_stommel = friction / beta  # meters

        # Grid spacing at this latitude
        dx_local = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(np.deg2rad(lat))

        # Scale factor: how much faster the true WBC is than the grid-resolved one
        wbc_scale = dx_local / L_stommel

        # Find western boundary cells (first ocean cell from west in each basin)
        basins = find_basin_boundaries(ocean_mask, j)
        for west_idx, east_idx in basins:
            if not ocean_mask[j, west_idx]:
                continue

            # Scale up the WBC velocity at the western boundary cell
            if not np.isnan(v[j, west_idx]):
                v[j, west_idx] = v[j, west_idx] * wbc_scale

    # Apply ocean mask
    u = np.where(ocean_mask, u, np.nan)
    v = np.where(ocean_mask, v, np.nan)

    # Apply latitude exclusion mask (Sverdrup theory invalid near equator/poles)
    lat_2d = np.broadcast_to(lat_deg[:, np.newaxis], u.shape)
    valid_lat = (np.abs(lat_2d) >= MIN_LATITUDE_DEG) & (np.abs(lat_2d) <= MAX_LATITUDE_DEG)
    u = np.where(valid_lat, u, np.nan)
    v = np.where(valid_lat, v, np.nan)

    return u, v


def compute_ocean_currents(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    land_mask: np.ndarray,
    include_stommel: bool = True,
) -> dict[str, np.ndarray]:
    """Compute ocean currents from wind field using Sverdrup-Stommel balance.

    Computes:
    1. Wind stress and curl from 10m winds
    2. Sverdrup interior transport: V = curl(τ) / (ρβ)
    3. Interior streamfunction by integrating from east coast
    4. Stommel western boundary layer closure (optional)
    5. Velocity field derived from streamfunction

    Parameters
    ----------
    u_wind : np.ndarray
        Zonal 10m wind (m/s), shape (nlat, nlon).
    v_wind : np.ndarray
        Meridional 10m wind (m/s), shape (nlat, nlon).
    lon2d : np.ndarray
        Longitude grid (degrees), shape (nlat, nlon).
    lat2d : np.ndarray
        Latitude grid (degrees), shape (nlat, nlon).
    land_mask : np.ndarray
        Boolean mask, True for land cells, shape (nlat, nlon).
    include_stommel : bool
        If True, include Stommel western boundary closure.
        If False, return interior Sverdrup solution only.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - 'tau_x': Zonal wind stress (N/m²)
        - 'tau_y': Meridional wind stress (N/m²)
        - 'curl_tau': Wind stress curl (N/m³)
        - 'sverdrup_transport': Sverdrup meridional transport (m²/s)
        - 'psi_interior': Interior streamfunction (Sv)
        - 'psi': Total streamfunction with boundary closure (Sv)
        - 'u_velocity': Zonal velocity (m/s), positive eastward
        - 'v_velocity': Meridional velocity (m/s), positive northward
    """
    nlat, nlon = u_wind.shape
    resolution_deg = 180.0 / nlat

    # Extract 1D coordinate arrays
    lat_deg = lat2d[:, 0]
    lon_deg = lon2d[0, :]

    # Ocean mask (invert land mask)
    ocean_mask = ~land_mask

    # Compute wind stress
    tau_x, tau_y = compute_wind_stress(u_wind, v_wind)

    # Compute wind stress curl
    curl_tau = compute_wind_stress_curl(tau_x, tau_y, lat_deg, resolution_deg)

    # Compute Sverdrup transport (m²/s)
    sverdrup_transport = compute_sverdrup_transport(curl_tau, lat_deg)

    # Integrate to get interior streamfunction
    psi_interior = compute_streamfunction(
        sverdrup_transport, ocean_mask, lon_deg, lat_deg, resolution_deg
    )

    # Add Stommel boundary closure
    if include_stommel:
        psi = compute_stommel_boundary(
            psi_interior, ocean_mask, lon_deg, lat_deg, resolution_deg
        )
    else:
        psi = psi_interior

    # Derive velocity from streamfunction
    u_velocity, v_velocity = streamfunction_to_velocity(
        psi, ocean_mask, lat_deg, resolution_deg
    )

    return {
        'tau_x': tau_x,
        'tau_y': tau_y,
        'curl_tau': curl_tau,
        'sverdrup_transport': sverdrup_transport,
        'psi_interior': psi_interior,
        'psi': psi,
        'u_velocity': u_velocity,
        'v_velocity': v_velocity,
    }


