"""Precipitation physics: moisture flux convergence and precipitation estimation.

Physics:
- Convergence precipitation: P_conv = max(-∇·(qV), 0) - where wind converges, moisture rains out
- Convective precipitation: P_mse = f(MSE instability, q) - where moist static energy is unstable
- Supersaturation precipitation: when q > q_sat, excess condenses immediately
- Total: P = P_conv + P_mse + P_supersat
- Latent heat release Q = P × L_v (warms atmosphere where condensation occurs)
"""

from __future__ import annotations

import numpy as np

from climate_sim.data.constants import (
    R_EARTH_METERS,
    HEAT_CAPACITY_AIR_J_KG_K,
    BOUNDARY_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_HEIGHT_M,
    LATENT_HEAT_VAPORIZATION_J_KG,
    GAS_CONSTANT_WATER_VAPOR_J_KG_K,
)
from climate_sim.physics.atmosphere.pressure import _smooth_temperature_field, _get_latitude_centers

# Physical constants for MSE-based convection
GRAVITY = 9.81  # m/s²
SPECIFIC_HEAT_AIR = HEAT_CAPACITY_AIR_J_KG_K  # J/kg/K

# Convection parameters
MOIST_LAYER_DEPTH_M = 3000.0  # Characteristic depth of moist layer
RHO_AIR_SURFACE = 1.2  # kg/m³ surface air density
UPPER_TROPOSPHERE_Q_FRACTION = 0.20  # q_upper / q_surface (dry air aloft)
# Height of atmosphere layer midpoint (where T_atm represents)
ATMOSPHERE_LAYER_MIDPOINT_M = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M / 2.0
MSE_INSTABILITY_THRESHOLD = 5000.0  # J/kg - minimum instability for convection
MSE_SATURATION_SCALE = 20000.0  # J/kg - instability scale for saturation

# Maximum precipitation rate (enforces physical limit)
# ~15 mm/day is a reasonable upper bound for monthly-mean convective precip
# Note: 1 mm of water = 1 kg/m², so mm/day ÷ 86400 = kg/m²/s
MAX_CONVECTIVE_PRECIP_RATE = 15.0 / 86400.0  # kg/m²/s (15 mm/day)

# Orographic precipitation parameters
# Reduced efficiency (0.2) allows more moisture to penetrate inland
# Real world: ~20-40% of lifted moisture precipitates, rest continues inland
ORO_PRECIP_EFFICIENCY = 0.2  # Fraction of lifted moisture that precipitates
ORO_SCALE_HEIGHT = 2500.0  # m - exponential decay of moisture with altitude

# Frontal precipitation parameters
FRONTAL_STORM_TRACK_LAT = 45.0  # degrees - latitude of peak storm activity
FRONTAL_STORM_TRACK_WIDTH = 20.0  # degrees - width of storm track belt
# Maximum frontal precip rate ~5 mm/day in active storm tracks
MAX_FRONTAL_PRECIP_RATE = 5.0 / 86400.0  # kg/m²/s


def compute_moist_adiabatic_lapse_rate(T_K: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute moist adiabatic lapse rate (K/m).

    The moist adiabatic lapse rate is less than the dry rate (9.8 K/km)
    because latent heat is released during condensation:

    Γ_m = Γ_d × (1 + L_v × q / (R_d × T)) / (1 + L_v² × q / (c_p × R_v × T²))

    Typical values:
    - Dry: 9.8 K/km
    - Moist tropical (T=300K, q=20g/kg): ~4-5 K/km
    - Moist midlatitude (T=280K, q=10g/kg): ~6 K/km
    """
    # Constants
    g = GRAVITY  # 9.81 m/s²
    cp = SPECIFIC_HEAT_AIR  # 1004 J/kg/K
    Lv = LATENT_HEAT_VAPORIZATION_J_KG  # 2.5e6 J/kg
    Rd = 287.0  # J/kg/K - gas constant for dry air
    Rv = GAS_CONSTANT_WATER_VAPOR_J_KG_K  # 461.0 J/(kg·K)

    # Dry adiabatic lapse rate
    gamma_d = g / cp  # ~0.00976 K/m = 9.76 K/km

    # Moist adiabatic lapse rate (simplified form)
    # Using mixing ratio r ≈ q for small q
    numerator = 1 + (Lv * q) / (Rd * T_K)
    denominator = 1 + (Lv**2 * q) / (cp * Rv * T_K**2)

    gamma_m = gamma_d * numerator / denominator

    return gamma_m


# Cloud base height for large-scale condensation (LCL approximation)
CLOUD_BASE_HEIGHT_M = 1500.0  # ~1.5 km typical LCL


def compute_supersaturation_precipitation(
    q: np.ndarray,
    T_bl_K: np.ndarray,
    tau_seconds: float = 7 * 86400,  # 1 week relaxation timescale
) -> np.ndarray:
    """Compute precipitation from large-scale condensation at cloud level.

    Large-scale (stratiform) precipitation occurs when air rises and cools
    to saturation. We compute the temperature at cloud base using the
    moist adiabatic lapse rate, then check if humidity exceeds saturation
    at that level.

    Physics:
    - Air at BL temperature rises to cloud base (~1.5 km)
    - It cools following the moist adiabatic lapse rate
    - At cloud base, if q > q_sat(T_cloud), condensation occurs
    - Excess moisture precipitates with a relaxation timescale

    Parameters
    ----------
    q : np.ndarray
        Specific humidity (kg/kg)
    T_bl_K : np.ndarray
        Boundary layer temperature (K)
    tau_seconds : float
        Relaxation timescale (s). Default 7 days.

    Returns
    -------
    np.ndarray
        Large-scale condensation precipitation rate (kg/m²/s)
    """
    # Lazy import to avoid circular dependency (humidity imports precipitation)
    from climate_sim.physics.humidity import compute_saturation_specific_humidity

    # Column mass for converting kg/kg to kg/m²
    # Water vapor is concentrated in lower troposphere (scale height ~2km)
    # Effective column mass for moisture ~5000 kg/m² (not full 10000 kg/m²)
    COLUMN_MASS = 5000.0  # kg/m²

    # Compute moist adiabatic lapse rate based on local conditions
    gamma_m = compute_moist_adiabatic_lapse_rate(T_bl_K, q)

    # Temperature at cloud base (air has cooled while rising)
    T_cloud = T_bl_K - gamma_m * CLOUD_BASE_HEIGHT_M

    # Saturation specific humidity at cloud level (colder → lower q_sat)
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    # Excess moisture above saturation at cloud level
    excess_q = np.maximum(q - q_sat_cloud, 0)

    # Relaxation: P = excess * COLUMN_MASS / tau
    # This removes excess moisture with e-folding time tau
    P_supersat = excess_q * COLUMN_MASS / tau_seconds

    return P_supersat


def compute_moisture_flux_convergence(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> np.ndarray:
    """Compute moisture flux convergence: -∇·(qV).
    """
    R = R_EARTH_METERS
    lat_rad = np.deg2rad(lat2d)

    cos_lat = np.cos(lat_rad)
    cos_lat = np.maximum(cos_lat, 0.01)  # Avoid division by zero at poles

    # Grid spacings (assumes uniform grid)
    dlat = np.deg2rad(lat2d[1, 0] - lat2d[0, 0])
    dlon = np.deg2rad(lon2d[0, 1] - lon2d[0, 0])

    # Moisture fluxes
    qu = q * u
    qv = q * v

    # ∂(qu)/∂λ
    dqu_dlon = np.gradient(qu, axis=1) / dlon

    # ∂(qv cos φ)/∂φ
    qv_cos_lat = qv * cos_lat
    d_qvcos_dlat = np.gradient(qv_cos_lat, axis=0) / dlat

    # Divergence of moisture flux
    div_qV = (1 / (R * cos_lat)) * dqu_dlon + (1 / (R * cos_lat)) * d_qvcos_dlat

    # Convergence is negative divergence
    convergence = -div_qV

    return convergence


def compute_mse(
    temperature_K: np.ndarray,
    specific_humidity: np.ndarray,
    height_m: float = 0.0,
) -> np.ndarray:
    """Compute moist static energy.

    MSE = c_p * T + L_v * q + g * z

    MSE is approximately conserved during moist adiabatic processes and
    determines whether a parcel can rise convectively.
    """
    return (
        SPECIFIC_HEAT_AIR * temperature_K
        + LATENT_HEAT_VAPORIZATION_J_KG * specific_humidity
        + GRAVITY * height_m
    )


def compute_convective_precipitation(
    q_surface: np.ndarray,
    T_surface_K: np.ndarray,
    T_atm_K: np.ndarray,
    rho_air: float = RHO_AIR_SURFACE,
    h_moist: float = MOIST_LAYER_DEPTH_M,
    vertical_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Compute convective precipitation rate from MSE instability.

    Physics:
    - Convection occurs when surface MSE exceeds upper-troposphere MSE
    - The instability drives overturning that removes moisture via precipitation
    - Rate is proportional to instability × moisture × (depth / timescale)
    - **Subsidence suppresses convection**: Even with high MSE, sinking air
      prevents parcels from rising. This is why deserts (under subtropical
      high pressure) stay dry despite having warm, moist boundary layers.

    Parameters
    ----------
    q_surface : np.ndarray
        Surface specific humidity (kg/kg)
    T_surface_K : np.ndarray
        Boundary layer temperature (K)
    T_atm_K : np.ndarray
        Free atmosphere temperature (K)
    rho_air : float
        Air density (kg/m³)
    h_moist : float
        Characteristic depth of moist layer (m)
    vertical_velocity : np.ndarray | None
        Large-scale vertical velocity (m/s). Positive = rising (enhances convection),
        negative = sinking (suppresses convection). If None, no suppression applied.

    Returns
    -------
    np.ndarray
        Convective precipitation rate (kg/m²/s)
    """
    # MSE at surface (boundary layer)
    MSE_surface = compute_mse(T_surface_K, q_surface, height_m=0.0)

    # MSE at atmosphere layer (midpoint of free troposphere)
    # Upper troposphere is much drier (moisture precipitates out during ascent)
    q_upper = q_surface * UPPER_TROPOSPHERE_Q_FRACTION
    MSE_upper = compute_mse(T_atm_K, q_upper, height_m=ATMOSPHERE_LAYER_MIDPOINT_M)

    # Instability: positive when surface MSE exceeds upper MSE
    # The geopotential term (g*z) means surface air must have enough
    # thermal + latent energy to overcome the potential energy barrier
    instability = MSE_surface - MSE_upper  # J/kg

    # Only precipitate where unstable (and above threshold)
    instability_excess = np.maximum(instability - MSE_INSTABILITY_THRESHOLD, 0)

    # Use a saturating function for instability - once strongly unstable,
    # precipitation is limited by moisture supply, not instability
    # f(x) = x / (x + scale) approaches 1 as x → ∞
    instability_factor = instability_excess / (instability_excess + MSE_SATURATION_SCALE)

    # Convective precipitation scales with instability factor and max rate
    # This approach is more physically meaningful for a monthly-mean model:
    # - Max rate ~15 mm/day represents sustained tropical convection
    # - Instability factor (0-1) modulates how much of this max is realized
    P_convective = instability_factor * MAX_CONVECTIVE_PRECIP_RATE

    # Subsidence suppression: sinking air prevents convection (creates desert belts at ~30 deg)
    if vertical_velocity is not None:
        w_scale = 0.01  # m/s - characteristic convective velocity
        suppression_factor = np.where(
            vertical_velocity >= 0,
            1.0,
            np.exp(vertical_velocity / w_scale)
        )
        suppression_factor = np.maximum(suppression_factor, 0.05)
        P_convective = P_convective * suppression_factor

    return P_convective


def compute_orographic_precipitation(
    q: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    elevation: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    rho_air: float = RHO_AIR_SURFACE,
) -> np.ndarray:
    """Compute orographic precipitation from terrain-forced ascent.

    Physics:
    - Wind blowing against terrain slopes forces air to rise
    - Rising air cools adiabatically and moisture condenses
    - Precipitation rate proportional to: ascent rate × moisture × efficiency

    The vertical velocity from terrain is: w = V · ∇z
    where ∇z is the terrain gradient.

    Parameters
    ----------
    q : np.ndarray
        Specific humidity (kg/kg)
    wind_u, wind_v : np.ndarray
        Wind components (m/s)
    elevation : np.ndarray
        Terrain elevation (m)
    lat2d, lon2d : np.ndarray
        Coordinate grids (degrees)
    rho_air : float
        Air density (kg/m³)

    Returns
    -------
    np.ndarray
        Orographic precipitation rate (kg/m²/s)
    """
    R = R_EARTH_METERS
    lat_rad = np.deg2rad(lat2d)
    cos_lat = np.cos(lat_rad)
    cos_lat = np.maximum(cos_lat, 0.01)

    # Grid spacings
    dlat = np.deg2rad(lat2d[1, 0] - lat2d[0, 0])
    dlon = np.deg2rad(lon2d[0, 1] - lon2d[0, 0])

    # Physical distances
    dx = R * cos_lat * dlon  # m
    dy = R * dlat  # m

    # Terrain gradient (∂z/∂x, ∂z/∂y) in m/m
    dz_dlon = np.gradient(elevation, axis=1)  # Change per grid cell
    dz_dlat = np.gradient(elevation, axis=0)
    dz_dx = dz_dlon / dx  # Convert to m/m
    dz_dy = dz_dlat / dy

    # Vertical velocity from terrain: w = u * ∂z/∂x + v * ∂z/∂y
    # Positive w means upslope flow (forced ascent)
    w_terrain = wind_u * dz_dx + wind_v * dz_dy

    # Only precipitate where air is rising (upslope)
    w_up = np.maximum(w_terrain, 0)

    # Moisture available decreases with elevation (exponential profile)
    # This accounts for air getting drier as it ascends
    moisture_factor = np.exp(-elevation / ORO_SCALE_HEIGHT)
    moisture_factor = np.clip(moisture_factor, 0.1, 1.0)

    # Orographic precipitation: P = efficiency × w × q × ρ × moisture_factor
    # Units: [1] × [m/s] × [kg/kg] × [kg/m³] × [1] = kg/m²/s
    P_orographic = ORO_PRECIP_EFFICIENCY * w_up * q * rho_air * moisture_factor

    return P_orographic


def compute_static_stability(
    T_bl_K: np.ndarray,
    T_atm_K: np.ndarray,
    delta_z: float = 5000.0,
) -> np.ndarray:
    """Compute static stability (Brunt-Väisälä frequency N).

    N² = (g/θ) × (∂θ/∂z) ≈ (g/T) × (∂T/∂z + g/cp)

    The term g/cp is the dry adiabatic lapse rate (~9.8 K/km).
    If the actual lapse rate is less steep than dry adiabatic (stable),
    N² > 0 and the atmosphere resists vertical motion.

    Parameters
    ----------
    T_bl_K : np.ndarray
        Boundary layer temperature (K)
    T_atm_K : np.ndarray
        Free atmosphere temperature (K)
    delta_z : float
        Height difference between BL and atmosphere layers (m)

    Returns
    -------
    np.ndarray
        Brunt-Väisälä frequency N (1/s), clipped to positive values
    """
    g = GRAVITY
    cp = SPECIFIC_HEAT_AIR

    # Mean temperature for the layer
    T_mean = 0.5 * (T_bl_K + T_atm_K)

    # Actual lapse rate (K/m) - positive if T decreases with height
    dT_dz = (T_atm_K - T_bl_K) / delta_z

    # N² = (g/T) × (dT/dz + g/cp)
    # The g/cp term (~0.0098 K/m) is the dry adiabatic lapse rate
    # If dT/dz > -g/cp (less steep than dry adiabatic), N² > 0 (stable)
    # If dT/dz < -g/cp (steeper than dry adiabatic), N² < 0 (unstable)
    N_squared = (g / T_mean) * (dT_dz + g / cp)

    # Clip to small positive value when unstable (convection dominates there anyway)
    N_squared = np.maximum(N_squared, 1e-6)
    N = np.sqrt(N_squared)

    return N


def compute_frontal_precipitation(
    q: np.ndarray,
    T_bl_K: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    T_atm_K: np.ndarray | None = None,
    vertical_velocity: np.ndarray | None = None,
    rho_air: float = RHO_AIR_SURFACE,
) -> np.ndarray:
    """Compute frontal precipitation from baroclinic instability.

    Physics:
    - Baroclinic instability drives midlatitude storm systems
    - Growth rate scales with f × |∂T/∂y| / N (Eady model)
    - f = Coriolis parameter: weak at low latitudes → suppresses storms
    - N = static stability: high in subsidence zones → suppresses storms
    - Subsidence (w < 0): large-scale sinking prevents storm development
    - This naturally confines frontal precipitation to midlatitudes with
      active weather (low stability, no subsidence)

    The baroclinic growth rate (Eady model):
        σ ∝ f × |∂T/∂y| / N

    Temperature is smoothed (~1000 km) before computing gradients to filter
    out local features (land-ocean contrast, topography) and retain only
    synoptic-scale gradients that drive baroclinic storms.

    Parameters
    ----------
    q : np.ndarray
        Specific humidity (kg/kg)
    T_bl_K : np.ndarray
        Boundary layer temperature (K) - used for horizontal gradient
    lat2d, lon2d : np.ndarray
        Coordinate grids (degrees)
    T_atm_K : np.ndarray, optional
        Atmosphere temperature (K) - used for static stability calculation.
        If None, stability term is omitted.
    vertical_velocity : np.ndarray, optional
        Large-scale vertical velocity (m/s). Positive = rising, negative = sinking.
        Subsidence suppresses storm development.
    rho_air : float
        Air density (kg/m³)

    Returns
    -------
    np.ndarray
        Frontal precipitation rate (kg/m²/s)
    """
    R = R_EARTH_METERS
    OMEGA = 7.2921e-5  # Earth's rotation rate (rad/s)

    nlat, nlon = T_bl_K.shape
    lat_rad = np.deg2rad(lat2d)
    cos_lat = np.cos(lat_rad)
    cos_lat = np.maximum(cos_lat, 0.01)

    # Coriolis parameter: f = 2Ω sin(φ)
    # Use absolute value since we care about magnitude of baroclinic instability
    f = 2.0 * OMEGA * np.abs(np.sin(lat_rad))

    # Smooth temperature to get synoptic-scale gradients only
    # This filters out local land-ocean contrast and topographic effects
    lat_centers = _get_latitude_centers(nlat)
    T_smooth = _smooth_temperature_field(T_bl_K, lat_centers, smoothing_length_km=1000.0)

    # Grid spacings
    dlat = np.deg2rad(lat2d[1, 0] - lat2d[0, 0])
    dlon = np.deg2rad(lon2d[0, 1] - lon2d[0, 0])

    # Physical distances
    dx = R * cos_lat * dlon
    dy = R * dlat

    # Temperature gradient magnitude |∇T| from smoothed field
    dT_dlon = np.gradient(T_smooth, axis=1)
    dT_dlat = np.gradient(T_smooth, axis=0)
    dT_dx = dT_dlon / dx  # K/m
    dT_dy = dT_dlat / dy
    grad_T_mag = np.sqrt(dT_dx**2 + dT_dy**2)

    # Static stability (Brunt-Väisälä frequency)
    # High N in subsidence zones (deserts) suppresses baroclinic growth
    if T_atm_K is not None:
        N = compute_static_stability(T_bl_K, T_atm_K)
        # Reference N for midlatitudes: ~0.01 s⁻¹
        N_ref = 0.01
        stability_factor = N_ref / N  # <1 where stable (suppressed), >1 where unstable
        stability_factor = np.clip(stability_factor, 0.1, 2.0)  # Limit range
    else:
        stability_factor = 1.0

    # Baroclinic instability measure: f × |∇T| / N
    # This naturally suppresses frontal activity where:
    # - f→0 (low latitudes)
    # - N is high (stable subsidence zones)
    # Units: [1/s] × [K/m] = [K/(m·s)]
    # Typical midlatitude value: f~1e-4 × grad_T~1e-5 = 1e-9 K/(m·s)
    baroclinic_measure = f * grad_T_mag * stability_factor

    # Normalize to dimensionless factor
    # Reference: f=1e-4 (45°N), grad_T=1e-5 K/m → product = 1e-9
    baroclinic_factor = baroclinic_measure / 1e-9
    baroclinic_factor = np.clip(baroclinic_factor, 0, 3)  # Cap at 3x typical

    # Frontal precipitation: baroclinic instability × moisture
    # Scale so that max rate ~5 mm/day in strongest storm tracks
    P_frontal = MAX_FRONTAL_PRECIP_RATE * baroclinic_factor * (q / 0.01)

    # Subsidence suppression: large-scale sinking prevents storm development
    # Even if baroclinic conditions exist, subsidence stabilizes the atmosphere
    # and prevents the vertical motion needed for frontal precipitation
    if vertical_velocity is not None:
        w_scale = 0.005  # m/s - smaller scale than convection (fronts more sensitive)
        suppression_factor = np.where(
            vertical_velocity >= 0,
            1.0,
            np.exp(vertical_velocity / w_scale)
        )
        suppression_factor = np.maximum(suppression_factor, 0.05)
        P_frontal = P_frontal * suppression_factor

    return P_frontal


def compute_convergence_precipitation(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    rho_air: float = RHO_AIR_SURFACE,
    h_moist: float = MOIST_LAYER_DEPTH_M,
) -> np.ndarray:
    """Compute precipitation from moisture flux convergence.

    Where winds converge, moisture accumulates and precipitates.
    This represents large-scale (frontal/ITCZ) precipitation.

    Units:
    - mfc has units of [1/s] (divergence of qV / q ≈ divergence of V for slowly-varying q)
    - P = mfc * q * rho * h gives [1/s] * [kg/kg] * [kg/m³] * [m] = [kg/m²/s]
    """
    mfc = compute_moisture_flux_convergence(q, u, v, lat2d, lon2d)
    # Precipitate where there's convergence
    # The formula P = mfc * q * rho * h represents the column moisture
    # that accumulates due to convergence
    P_convergence = np.maximum(mfc, 0) * q * rho_air * h_moist
    return P_convergence


def compute_precipitation_rate(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    T_surface_K: np.ndarray | None = None,
    T_atm_K: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    rho_air: float = RHO_AIR_SURFACE,
    h_moist: float = MOIST_LAYER_DEPTH_M,
    vertical_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Compute total precipitation rate from all mechanisms.

    Combines three precipitation types:
    1. Convective: MSE-based instability (tropics, summer continents)
    2. Orographic: Terrain-forced ascent (mountains, windward slopes)
    3. Frontal: Midlatitude storm track activity

    Parameters
    ----------
    q : np.ndarray
        Specific humidity (kg/kg)
    u, v : np.ndarray
        Wind components (m/s)
    lat2d, lon2d : np.ndarray
        Coordinate grids (degrees)
    T_surface_K : np.ndarray, optional
        Boundary layer temperature for convective precipitation
    T_atm_K : np.ndarray, optional
        Atmosphere temperature for convective/frontal precipitation
    elevation : np.ndarray, optional
        Terrain elevation (m) for orographic precipitation
    rho_air : float
        Air density (kg/m³)
    h_moist : float
        Moist layer depth (m)
    vertical_velocity : np.ndarray, optional
        Large-scale vertical velocity (m/s). Positive = rising, negative = sinking.
        Used to suppress convective precipitation in subsidence zones.

    Returns
    -------
    np.ndarray
        Total precipitation rate (kg/m²/s)
    """
    P_total = np.zeros_like(q)
    P_convective = np.zeros_like(q)

    # 1. Convective precipitation (MSE-based)
    # Subsidence (negative w) suppresses convection even with high MSE
    if T_surface_K is not None and T_atm_K is not None:
        P_convective = compute_convective_precipitation(
            q, T_surface_K, T_atm_K, rho_air, h_moist,
            vertical_velocity=vertical_velocity,
        )
        P_total += P_convective

    # 2. Orographic precipitation (terrain-forced)
    # TODO: Disabled - needs Jacobian implementation
    # if elevation is not None:
    #     P_orographic = compute_orographic_precipitation(
    #         q, u, v, elevation, lat2d, lon2d, rho_air
    #     )
    #     P_total += P_orographic

    # 3. Frontal precipitation (storm tracks)
    # TODO: Disabled - needs Jacobian implementation (non-local due to T gradient)
    # if T_surface_K is not None:
    #     P_frontal = compute_frontal_precipitation(
    #         q, T_surface_K, lat2d, lon2d,
    #         T_atm_K=T_atm_K,
    #         vertical_velocity=vertical_velocity,
    #         rho_air=rho_air,
    #     )
    #     P_total += P_frontal

    # 4. Supersaturation precipitation (large-scale condensation)
    # When air rises to cloud level and q > q_sat(T_cloud), excess condenses
    # Only apply where there's NO convective precipitation (mutually exclusive)
    # Convection already handles moisture removal in unstable conditions
    if T_surface_K is not None:
        P_supersat = compute_supersaturation_precipitation(q, T_surface_K)
        # Only add supersaturation where convection is weak (<0.5 mm/day)
        # This avoids double-counting in convectively active regions
        convection_threshold = 0.5 / 86400.0  # 0.5 mm/day in kg/m²/s
        P_supersat = np.where(P_convective < convection_threshold, P_supersat, 0.0)
        P_total += P_supersat

    return P_total


def compute_precipitation_jacobian(
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian of precipitation rate w.r.t. temperatures and humidity.

    Returns derivatives for the atmosphere heating tendency from precipitation:
        dT_atm/dt = P × L_v / C_atm

    Parameters
    ----------
    T_bl : np.ndarray
        Boundary layer temperature (K)
    T_atm : np.ndarray
        Free atmosphere temperature (K)
    q : np.ndarray
        Specific humidity (kg/kg)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (dP_dT_bl, dP_dT_atm, dP_dq) - derivatives of precipitation rate (kg/m²/s)
        w.r.t. T_bl (K), T_atm (K), and q (kg/kg)
    """

    cp = SPECIFIC_HEAT_AIR
    Lv = LATENT_HEAT_VAPORIZATION_J_KG

    # Initialize Jacobians
    dP_dT_bl = np.zeros_like(T_bl)
    dP_dT_atm = np.zeros_like(T_atm)
    dP_dq = np.zeros_like(q)

    # =========================================================================
    # 1. Convective precipitation Jacobian
    # P_conv = MAX_RATE × f(instability)  where f(x) = x / (x + scale)
    # instability = MSE_bl - MSE_atm = cp×T_bl + Lv×q - (cp×T_atm + Lv×q_upper + g×z)
    # =========================================================================
    MSE_surface = compute_mse(T_bl, q, height_m=0.0)
    q_upper = q * UPPER_TROPOSPHERE_Q_FRACTION
    MSE_upper = compute_mse(T_atm, q_upper, height_m=ATMOSPHERE_LAYER_MIDPOINT_M)
    instability = MSE_surface - MSE_upper

    instability_excess = np.maximum(instability - MSE_INSTABILITY_THRESHOLD, 0)
    is_unstable = instability > MSE_INSTABILITY_THRESHOLD

    # Derivative of f(x) = x / (x + scale): f'(x) = scale / (x + scale)²
    denom = instability_excess + MSE_SATURATION_SCALE
    df_dx = MSE_SATURATION_SCALE / (denom * denom)

    # ∂instability/∂T_bl = cp, ∂instability/∂T_atm = -cp
    # ∂instability/∂q = Lv × (1 - UPPER_TROP_Q_FRACTION)
    dP_conv_dT_bl = np.where(is_unstable, MAX_CONVECTIVE_PRECIP_RATE * df_dx * cp, 0.0)
    dP_conv_dT_atm = np.where(is_unstable, -MAX_CONVECTIVE_PRECIP_RATE * df_dx * cp, 0.0)
    dP_conv_dq = np.where(
        is_unstable,
        MAX_CONVECTIVE_PRECIP_RATE * df_dx * Lv * (1 - UPPER_TROPOSPHERE_Q_FRACTION),
        0.0
    )

    dP_dT_bl += dP_conv_dT_bl
    dP_dT_atm += dP_conv_dT_atm
    dP_dq += dP_conv_dq

    # =========================================================================
    # 2. Supersaturation precipitation Jacobian
    # P_ss = max(q - q_sat(T_cloud), 0) × COLUMN_MASS / tau
    # T_cloud = T_bl - γ_m × h_cloud  (γ_m depends on T_bl and q, but weakly)
    # Approximate: ∂T_cloud/∂T_bl ≈ 1, ∂γ_m/∂T_bl ≈ 0
    # =========================================================================
    # Lazy import to avoid circular dependency (humidity imports precipitation)
    from climate_sim.physics.humidity import compute_saturation_specific_humidity

    COLUMN_MASS = 5000.0  # kg/m² (lower troposphere)
    tau_seconds = 7 * 86400  # 1 week

    gamma_m = compute_moist_adiabatic_lapse_rate(T_bl, q)
    T_cloud = T_bl - gamma_m * CLOUD_BASE_HEIGHT_M
    q_sat_cloud = compute_saturation_specific_humidity(T_cloud)

    is_supersaturated = q > q_sat_cloud

    # ∂q_sat/∂T using Clausius-Clapeyron (Magnus formula derivative)
    T_cloud_C = T_cloud - 273.15
    e_sat = 6.112 * np.exp(17.67 * T_cloud_C / (T_cloud_C + 243.5))
    de_sat_dT = e_sat * 17.67 * 243.5 / np.power(T_cloud_C + 243.5, 2)
    p_hPa = 1013.25
    denom_q = p_hPa - 0.378 * e_sat
    dq_sat_dT = 0.622 * p_hPa / (denom_q * denom_q) * de_sat_dT

    # ∂P_ss/∂T_bl = -∂q_sat/∂T_cloud × ∂T_cloud/∂T_bl × COLUMN_MASS/tau
    #             ≈ -dq_sat_dT × COLUMN_MASS/tau  (since ∂T_cloud/∂T_bl ≈ 1)
    dP_ss_dT_bl = np.where(is_supersaturated, -dq_sat_dT * COLUMN_MASS / tau_seconds, 0.0)

    # ∂P_ss/∂q = COLUMN_MASS / tau (direct q dependence)
    dP_ss_dq = np.where(is_supersaturated, COLUMN_MASS / tau_seconds, 0.0)

    dP_dT_bl += dP_ss_dT_bl
    dP_dq += dP_ss_dq

    return dP_dT_bl, dP_dT_atm, dP_dq
