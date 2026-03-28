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

# Ekman layer parameters
EKMAN_DEPTH = 50.0  # meters, typical Ekman layer depth
MIN_EKMAN_LATITUDE_DEG = 3.0  # Exclude very near-equator (f→0 singularity)

# Antarctic Circumpolar Current (ACC) parameters
# At circumpolar latitudes, there are no land barriers so Sverdrup-Stommel breaks down.
# Instead, use wind-friction balance: τ_x = ρ * r * H * u → u = τ_x / (ρ * r * H)
ACC_FRICTION_COEFF = 5.0e-6  # 1/s, gives realistic ACC speeds (~0.2 m/s)

# Equatorial current parameters
# At equatorial latitudes (|lat| < 5°), Coriolis is negligible so Sverdrup breaks down.
# Use wind-friction balance like ACC: u = τ_x / (ρ * r * H)
# Trade winds are easterly (τ_x < 0), giving westward South/North Equatorial Currents.
EQUATORIAL_FRICTION_COEFF = 1.0e-5  # 1/s, higher friction due to strong vertical mixing
EQUATORIAL_MIXED_LAYER_DEPTH = 50.0  # meters, shallower than mid-latitude


def compute_beta(lat_deg: np.ndarray) -> np.ndarray:
    """Compute β = df/dy = (2Ω/R) cos(lat), the meridional Coriolis gradient."""
    return (2.0 * OMEGA / R_EARTH_METERS) * np.cos(np.deg2rad(lat_deg))


def compute_coriolis_parameter(lat_deg: np.ndarray) -> np.ndarray:
    """Compute Coriolis parameter f = 2Ω sin(lat)."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


def compute_wind_friction_velocity(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    ocean_mask: np.ndarray,
    friction_coeff: float,
    layer_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity from wind-friction balance: τ = ρ * r * H * u.

    Used for circumpolar (ACC) and equatorial currents where Sverdrup breaks down.
    """
    u = tau_x / (RHO_WATER * friction_coeff * layer_depth)
    v = tau_y / (RHO_WATER * friction_coeff * layer_depth)
    return (
        np.where(ocean_mask, u, np.nan),
        np.where(ocean_mask, v, np.nan),
    )


def compute_wind_stress(
    u: np.ndarray,
    v: np.ndarray,
    rho_air: float = RHO_AIR,
    c_d: float = C_D,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wind stress using bulk formula: τ = ρ_air × C_D × |U| × U."""
    wind_speed = np.sqrt(u**2 + v**2)
    return rho_air * c_d * u * wind_speed, rho_air * c_d * v * wind_speed


def compute_wind_stress_curl(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    lat_deg: np.ndarray,
    resolution_deg: float,
) -> np.ndarray:
    """Compute wind stress curl: curl(τ) = ∂τ_y/∂x - ∂τ_x/∂y + τ_y tan(lat)/R."""
    lat_rad = np.deg2rad(lat_deg)
    dy = R_EARTH_METERS * np.deg2rad(resolution_deg)
    dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(lat_rad)[:, np.newaxis]

    dtau_y_dx = np.gradient(tau_y, axis=1) / dx
    dtau_x_dy = np.gradient(tau_x, axis=0) / dy
    metric_term = tau_y * np.tan(lat_rad)[:, np.newaxis] / R_EARTH_METERS

    return dtau_y_dx - dtau_x_dy + metric_term


def compute_ekman_transport(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    lat_deg: np.ndarray,
    rho_water: float = RHO_WATER,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Ekman mass transport from wind stress.

    Ekman transport is perpendicular to wind stress (to the right in NH):
        M_x = τ_y / (ρf)
        M_y = -τ_x / (ρf)

    """
    # Compute Coriolis parameter
    f = compute_coriolis_parameter(lat_deg)[:, np.newaxis]
    lat_2d = np.broadcast_to(lat_deg[:, np.newaxis], tau_x.shape)
    valid = np.abs(lat_2d) >= MIN_EKMAN_LATITUDE_DEG
    f_safe = np.where(valid, f, np.nan)

    M_x = tau_y / (rho_water * f_safe)
    M_y = -tau_x / (rho_water * f_safe)
    return np.where(valid, M_x, np.nan), np.where(valid, M_y, np.nan)


def ekman_transport_to_velocity(
    M_x: np.ndarray,
    M_y: np.ndarray,
    ocean_mask: np.ndarray,
    ekman_depth: float = EKMAN_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Ekman transport to velocity by distributing over Ekman depth."""
    return (
        np.where(ocean_mask, M_x / ekman_depth, np.nan),
        np.where(ocean_mask, M_y / ekman_depth, np.nan),
    )


def compute_sverdrup_transport(
    curl_tau: np.ndarray,
    lat_deg: np.ndarray,
    rho_water: float = RHO_WATER,
) -> np.ndarray:
    """Compute Sverdrup transport: V = curl(τ) / (ρβ). NaN outside valid latitude range."""
    beta = compute_beta(lat_deg)[:, np.newaxis]
    lat_2d = np.broadcast_to(lat_deg[:, np.newaxis], curl_tau.shape)
    valid = (np.abs(lat_2d) >= MIN_LATITUDE_DEG) & (np.abs(lat_2d) <= MAX_LATITUDE_DEG)
    beta_safe = np.where(valid, beta, np.nan)

    return np.where(valid, curl_tau / (rho_water * beta_safe), np.nan)


def sverdrup_transport_to_velocity(
    V_transport: np.ndarray,
    ocean_mask: np.ndarray,
    mixed_layer_depth: float = 100.0,
) -> np.ndarray:
    """Convert Sverdrup transport to meridional velocity."""
    return np.where(ocean_mask, V_transport / mixed_layer_depth, np.nan)


def find_basin_boundaries(ocean_mask: np.ndarray, lat_idx: int) -> list[tuple[int, int]]:
    """Find (west_idx, east_idx) pairs for each contiguous ocean segment at a latitude."""
    row = ocean_mask[lat_idx, :]
    nlon = len(row)

    if not np.any(row):
        return []

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
    """Add Stommel western boundary layer closure: ψ_boundary = -ψ_west * exp(-dist/L)."""
    nlat, nlon = psi_interior.shape
    psi_total = psi_interior.copy()

    for j in range(nlat):
        lat = lat_deg[j]
        if abs(lat) < MIN_LATITUDE_DEG or abs(lat) > MAX_LATITUDE_DEG:
            continue

        L = friction / compute_beta(np.array([lat]))[0]
        dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(np.deg2rad(lat))
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
                for i in range(west_idx, east_idx + 1):
                    decay = np.exp(-(i - west_idx) * dx / L)
                    psi_total[j, i] = psi_interior[j, i] - psi_west * decay
            else:
                for i in range(west_idx, nlon):
                    decay = np.exp(-(i - west_idx) * dx / L)
                    if not np.isnan(psi_interior[j, i]):
                        psi_total[j, i] = psi_interior[j, i] - psi_west * decay
                for i in range(0, east_idx + 1):
                    decay = np.exp(-(nlon - west_idx + i) * dx / L)
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
    """Derive velocity from streamfunction: u = -∂ψ/∂y, v = ∂ψ/∂x (scaled by depth).

    Western boundary currents are scaled by dx/L_stommel to correct for coarse resolution.
    """
    nlat = psi.shape[0]
    psi_m3s = psi * 1.0e6
    dy = R_EARTH_METERS * np.deg2rad(resolution_deg)
    cos_lat = np.cos(np.deg2rad(lat_deg))[:, np.newaxis]
    dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * cos_lat

    psi_filled = np.where(np.isnan(psi_m3s), 0.0, psi_m3s)

    # ∂ψ/∂lat (axis 0) - for zonal velocity
    dpsi_dy = np.gradient(psi_filled, dy, axis=0)

    # ∂ψ/∂lon (axis 1) - for meridional velocity
    # Use central differences for interior
    dpsi_dx = np.gradient(psi_filled, axis=1) / dx

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

        L_stommel = friction / compute_beta(np.array([lat]))[0]
        dx_local = R_EARTH_METERS * np.deg2rad(resolution_deg) * np.cos(np.deg2rad(lat))
        wbc_scale = dx_local / L_stommel

        for west_idx, _ in find_basin_boundaries(ocean_mask, j):
            if ocean_mask[j, west_idx] and not np.isnan(v[j, west_idx]):
                v[j, west_idx] *= wbc_scale

    lat_2d = np.broadcast_to(lat_deg[:, np.newaxis], u.shape)
    valid = ocean_mask & (np.abs(lat_2d) >= MIN_LATITUDE_DEG) & (np.abs(lat_2d) <= MAX_LATITUDE_DEG)
    return np.where(valid, u, np.nan), np.where(valid, v, np.nan)


def compute_ekman_pumping(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    lat_deg: np.ndarray,
    resolution_deg: float,
    ocean_mask: np.ndarray,
    land_mask: np.ndarray,
    rho_water: float = RHO_WATER,
) -> np.ndarray:
    """Compute coastal upwelling from offshore Ekman transport.

    At each coastal ocean cell, wind-driven Ekman transport has a component
    directed away from land (offshore). This water cannot be replaced from
    land, so it must upwell from below. The upwelling velocity is:

        w_E = M_offshore / dx

    where M_offshore is the Ekman transport component perpendicular to
    and away from the coast. This isolates the coastal upwelling mechanism
    from the large-scale gyre convergence (which is already captured by
    the Sverdrup-Stommel ocean advection).

    Only upwelling (offshore transport) is modeled. Onshore transport
    causes coastal downwelling/piling which doesn't entrain deep water.

    Positive values indicate upwelling.
    """
    nlat, nlon = tau_x.shape

    # Compute Ekman transport: M_x = τ_y/(ρf), M_y = -τ_x/(ρf)
    M_x, M_y = compute_ekman_transport(tau_x, tau_y, lat_deg, rho_water)
    M_x = np.nan_to_num(M_x, nan=0.0)
    M_y = np.nan_to_num(M_y, nan=0.0)

    # Grid spacing
    lat_rad = np.deg2rad(lat_deg)
    dy = R_EARTH_METERS * np.deg2rad(resolution_deg)
    cos_lat = np.cos(lat_rad)[:, np.newaxis]
    dx = R_EARTH_METERS * np.deg2rad(resolution_deg) * cos_lat

    w_E = np.zeros((nlat, nlon))

    # For each direction, find ocean cells with a land neighbor in that direction.
    # Upwelling = water moving AWAY from land (offshore), creating a deficit
    # that must be filled from below.
    # East neighbor is land → offshore = M_x < 0 (westward, away from land)
    # West neighbor is land → offshore = M_x > 0 (eastward, away from land)
    # North neighbor is land → offshore = M_y < 0 (southward, away from land)
    # South neighbor is land → offshore = M_y > 0 (northward, away from land)
    for di, dj, get_transport, get_dx in [
        (0, 1, lambda: -M_x, lambda: dx),  # east land → M_x < 0 is offshore
        (0, -1, lambda: M_x, lambda: dx),  # west land → M_x > 0 is offshore
        (-1, 0, lambda: -M_y, lambda: dy),  # north land → M_y < 0 is offshore
        (1, 0, lambda: M_y, lambda: dy),  # south land → M_y > 0 is offshore
    ]:
        # Find land neighbors in this direction
        neighbor_land = np.roll(np.roll(land_mask, -di, axis=0), -dj, axis=1)
        coastal = ocean_mask & neighbor_land

        # Transport component toward land (positive = offshore = upwelling)
        M_offshore = get_transport()
        cell_dx = get_dx()

        # Only offshore transport contributes (water leaving coast)
        upwelling = np.maximum(M_offshore, 0.0) / cell_dx
        w_E += np.where(coastal, upwelling, 0.0)

    return w_E


def compute_deep_ocean_temperature(lat_deg: np.ndarray) -> np.ndarray:
    """Compute latitude-dependent deep ocean temperature (~200m thermocline).

    Ranges from ~18°C at equator to ~2°C at poles, converted to Kelvin.
    """
    T_deep_C = 18.0 - 16.0 * np.minimum(np.abs(lat_deg) / 70.0, 1.0)
    return T_deep_C + 273.15


def compute_ocean_currents(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    land_mask: np.ndarray,
    include_stommel: bool = True,
    include_ekman: bool = True,
) -> dict[str, np.ndarray]:
    """Compute ocean currents from wind using Sverdrup-Stommel balance + Ekman.

    Returns dict with tau_x/y, curl_tau, sverdrup_transport, psi_interior, psi,
    u/v_gyre, u/v_ekman, and u/v_velocity (total).
    """
    nlat = u_wind.shape[0]
    resolution_deg = 180.0 / nlat
    lat_deg, lon_deg = lat2d[:, 0], lon2d[0, :]
    ocean_mask = ~land_mask

    tau_x, tau_y = compute_wind_stress(u_wind, v_wind)
    curl_tau = compute_wind_stress_curl(tau_x, tau_y, lat_deg, resolution_deg)
    sverdrup_transport = compute_sverdrup_transport(curl_tau, lat_deg)
    psi_interior = compute_streamfunction(
        sverdrup_transport, ocean_mask, lon_deg, lat_deg, resolution_deg
    )

    if include_stommel:
        psi = compute_stommel_boundary(psi_interior, ocean_mask, lon_deg, lat_deg, resolution_deg)
    else:
        psi = psi_interior

    u_gyre, v_gyre = streamfunction_to_velocity(psi, ocean_mask, lat_deg, resolution_deg)

    # Handle regions where Sverdrup breaks down using wind-friction balance
    mixed_layer_depth = 100.0
    for j in range(nlat):
        row_mask = ocean_mask[j : j + 1, :]
        if np.all(ocean_mask[j, :]):  # Circumpolar (ACC)
            u_gyre[j, :], _ = compute_wind_friction_velocity(
                tau_x[j : j + 1, :],
                tau_y[j : j + 1, :],
                row_mask,
                ACC_FRICTION_COEFF,
                mixed_layer_depth,
            )
            v_gyre[j, :] = 0.0
        elif np.abs(lat_deg[j]) < MIN_LATITUDE_DEG:  # Equatorial
            u_gyre[j, :], v_gyre[j, :] = compute_wind_friction_velocity(
                tau_x[j : j + 1, :],
                tau_y[j : j + 1, :],
                row_mask,
                EQUATORIAL_FRICTION_COEFF,
                EQUATORIAL_MIXED_LAYER_DEPTH,
            )

    if include_ekman:
        M_x, M_y = compute_ekman_transport(tau_x, tau_y, lat_deg)
        u_ekman, v_ekman = ekman_transport_to_velocity(M_x, M_y, ocean_mask)
    else:
        u_ekman = np.full_like(u_gyre, np.nan)
        v_ekman = np.full_like(v_gyre, np.nan)

    u_velocity = np.nan_to_num(u_gyre, nan=0.0) + np.nan_to_num(u_ekman, nan=0.0)
    v_velocity = np.nan_to_num(v_gyre, nan=0.0) + np.nan_to_num(v_ekman, nan=0.0)
    u_velocity = np.where(ocean_mask, u_velocity, np.nan)
    v_velocity = np.where(ocean_mask, v_velocity, np.nan)

    # Compute coastal upwelling from offshore Ekman transport
    w_ekman_pumping = compute_ekman_pumping(
        tau_x,
        tau_y,
        lat_deg,
        resolution_deg,
        ocean_mask,
        ~ocean_mask,
    )
    w_ekman_pumping = np.where(ocean_mask, w_ekman_pumping, np.nan)

    return {
        "tau_x": tau_x,
        "tau_y": tau_y,
        "curl_tau": curl_tau,
        "sverdrup_transport": sverdrup_transport,
        "psi_interior": psi_interior,
        "psi": psi,
        "u_gyre": u_gyre,
        "v_gyre": v_gyre,
        "u_ekman": u_ekman,
        "v_ekman": v_ekman,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "w_ekman_pumping": w_ekman_pumping,
    }
