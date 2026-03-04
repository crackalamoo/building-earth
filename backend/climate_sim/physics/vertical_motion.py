"""Vertical motion physics: adiabatic heating/cooling and humidity transport.

Physics:
- Subsidence warming: Air descending in the subtropics warms adiabatically
  at the dry lapse rate (Γ_dry = 9.8 K/km) since descending air is unsaturated.

- Ascent cooling: Rising air at the ITCZ cools at the moist lapse rate
  (Γ_moist ≈ 6.5 K/km) because condensation releases latent heat, partially
  offsetting the adiabatic cooling. This implicitly accounts for latent heat
  release - no separate precipitation heating term needed.

- Humidity transport:
  - Subsidence: Brings dry upper-tropospheric air down, reducing surface humidity.
    Upper troposphere has ~20% of boundary layer humidity due to precipitation
    removing moisture as air ascends.
  - Ascent: Moisture is removed via precipitation (handled separately).

Vertical velocity is inferred from horizontal divergence via mass continuity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climate_sim.data.constants import (
    ATMOSPHERE_LAYER_HEIGHT_M,
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    BOUNDARY_LAYER_HEIGHT_M,
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    STANDARD_LAPSE_RATE_K_PER_M,
    HEAT_CAPACITY_AIR_J_KG_K,
    R_EARTH_METERS,
    STEFAN_BOLTZMANN_W_M2_K4,
)
from climate_sim.physics.atmosphere.pressure import (
    LAT_SUBTROPICS_BASE,
    SUBTROPICS_ITCZ_COUPLING,
    SIGMA_SUBTROPICS,
    SIGMA_ITCZ,
)

# Physical constants
GAMMA_DRY = 0.0098  # K/m, dry adiabatic lapse rate (for subsidence)
GAMMA_MOIST = STANDARD_LAPSE_RATE_K_PER_M  # ~0.0065 K/m, moist lapse rate (for ascent)

# Upper troposphere humidity fraction relative to boundary layer
# Air rising through the troposphere loses most moisture via precipitation
# By the time it reaches the upper troposphere (~300 hPa), q is ~20% of surface value
UPPER_TROPOSPHERE_Q_FRACTION = 0.20


@dataclass(frozen=True)
class VerticalMotionConfig:
    """Configuration for vertical motion physics."""

    enabled: bool = True

    # Hadley subsidence at BL top (~850 hPa).
    # Mid-tropospheric (500 hPa) omega is 3-5 mm/s, but at BL top it's
    # ~1/3 of that due to mass continuity (w scales with height above ground).
    hadley_descent_velocity_m_s: float = 0.001

    # Humidity of air entrained into BL top from subsidence.
    # Descending air has lost most moisture via precipitation during ascent;
    # upper-tropospheric q is ~15-20% of BL value.
    upper_troposphere_q_fraction: float = 0.20

    # Background BL-atmosphere mixing timescale (seconds).
    # Represents subsidence, entrainment, and turbulent exchange that
    # returns latent heat from the free atmosphere back to the BL.
    tau_bl_atm_mixing_s: float = 3.0 * 86400.0  # 3.0 days


def compute_vertical_motion_heating(
    divergence: np.ndarray,
    h_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute atmosphere heating/cooling from vertical motion.

    When surface divergence > 0, air descends and warms at the dry lapse rate.
    When surface convergence < 0, air ascends and cools at the moist lapse rate
    (latent heat release partially offsets the cooling).
    """
    # Surface divergence implies vertical motion through mass continuity.
    # We scale divergence to a realistic vertical velocity using an
    # effective depth scale.
    # Empirically: div ~ 1e-5 /s in subtropics
    # Target: ~1 K/day = 1e-5 K/s warming
    effective_depth = 100.0  # m

    # Vertical velocity: positive = descent, negative = ascent
    w = divergence * effective_depth  # m/s

    # Use dry lapse rate for descent (unsaturated), moist for ascent (saturated)
    # dT/dt = w * Γ (positive w and positive Γ = warming)
    heating = np.where(
        w > 0,
        w * GAMMA_DRY,   # Descent: warm at dry rate
        w * GAMMA_MOIST  # Ascent: cool at moist rate (less cooling due to latent heat)
    )

    return heating


def compute_vertical_motion_tendency(
    divergence: np.ndarray,
    h_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute atmosphere heating from vertical motion.

    Uses dry lapse rate for descent (subsidence warming in subtropics)
    and moist lapse rate for ascent (reduced cooling at ITCZ due to
    latent heat release from condensation).
    """
    return compute_vertical_motion_heating(divergence, h_atm)


def compute_vertical_motion_tendencies(
    divergence: np.ndarray,
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    C_bl: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    C_atm: float = ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BL and atmosphere heating from vertical motion.

    Uses potential temperature conservation for adiabatic vertical motion.

    T_bl is sea-level equivalent temperature (≈ potential temperature at P₀).
    T_atm is actual temperature at ~500 hPa.

    Physics:
    - Subsidence (div > 0): Air descends from above into the BL. Descending air
      conserves θ; if θ_exchange > θ_bl, subsidence warms the BL. The atmosphere
      cools by the same amount (energy conserving). This is a true inter-layer
      heat exchange.

    - Ascent (div < 0): Air rises from BL into atmosphere. The BL is replenished
      by horizontal convergence (advection) at similar θ, so the BL temperature
      is unchanged. However, the atmosphere DOES receive air at θ_bl, which
      affects its temperature. The latent heat from condensation is handled
      separately by the precipitation module.

    The exchange happens at the BL top (~850 hPa), not at 500 hPa where T_atm
    is defined. We interpolate θ in log-pressure space between θ_bl (surface)
    and θ_atm (500 hPa) to get θ at the exchange level.
    """
    h_bl = BOUNDARY_LAYER_HEIGHT_M
    w = divergence * h_bl  # m/s, positive = downward (subsidence)

    rho = 1.0  # kg/m³
    cp = HEAT_CAPACITY_AIR_J_KG_K  # J/kg/K

    # Pressure levels
    P0 = 1013.25  # hPa, sea-level reference
    P_ATM = 500.0  # hPa, atmosphere layer
    P_EXCHANGE = 850.0  # hPa, exchange level just above BL
    KAPPA = 0.286  # R/cp

    # Potential temperatures at known levels
    theta_bl = T_bl  # T_bl is sea-level equivalent ≈ θ
    theta_atm = T_atm * (P0 / P_ATM) ** KAPPA

    # Interpolate θ to exchange level (850 hPa) in log-pressure space
    # f = fraction of the way from surface to 500 hPa
    ln_P0 = np.log(P0)
    ln_P_atm = np.log(P_ATM)
    ln_P_exchange = np.log(P_EXCHANGE)
    f = (ln_P0 - ln_P_exchange) / (ln_P0 - ln_P_atm)  # ≈ 0.25

    theta_exchange = theta_bl + f * (theta_atm - theta_bl)

    # Vertical heat exchange between BL and atmosphere.
    # Q > 0 when w > 0 (subsidence warms BL), Q < 0 when w < 0 (ascent cools BL).
    # Energy conserving: BL gains Q/C_bl, atmosphere loses Q/C_atm.
    Q_vertical = rho * cp * w * (theta_exchange - theta_bl)

    dT_bl = Q_vertical / C_bl
    dT_atm = -Q_vertical / C_atm

    return dT_bl, dT_atm


def compute_subsidence_drying(
    divergence: np.ndarray,
    humidity_field: np.ndarray,
) -> np.ndarray:
    """Compute humidity reduction from subsidence bringing dry air down.

    In regions of divergence (subtropics), air descends from the upper
    troposphere where it is much drier (moisture precipitated out during
    ascent at ITCZ). This mixes dry air into the boundary layer, reducing
    humidity.

    The tendency is proportional to:
    - Divergence (descent rate)
    - Humidity difference between BL and upper troposphere

    Parameters
    ----------
    divergence : np.ndarray
        Horizontal divergence (1/s). Positive = subsidence.
    humidity_field : np.ndarray
        Current specific humidity (kg/kg) in the boundary layer.

    Returns
    -------
    np.ndarray
        Humidity tendency (kg/kg/s). Negative where subsidence occurs.
    """
    # Upper troposphere humidity (dry air coming down)
    q_upper = humidity_field * UPPER_TROPOSPHERE_Q_FRACTION

    # Humidity difference that gets mixed in during subsidence
    delta_q = q_upper - humidity_field  # Always negative (upper is drier)

    # Same effective depth scaling as temperature
    effective_depth = 100.0  # m

    # Descent rate (m/s), only where divergence > 0
    w = np.maximum(divergence, 0) * effective_depth

    # Mixing timescale: how fast subsidence replaces BL air
    # w / h_BL gives the fraction of BL replaced per second
    h_bl = 1000.0  # boundary layer height (m)
    mixing_rate = w / h_bl  # 1/s

    # Humidity tendency: rate of mixing * humidity difference
    dq_dt = mixing_rate * delta_q

    return dq_dt


def compute_hadley_subsidence_velocity(
    lat_rad: np.ndarray,
    itcz_rad: np.ndarray,
    peak_velocity_m_s: float = 0.001,
) -> np.ndarray:
    """Compute vertical velocity from Hadley cell overturning at BL top.

    Returns vertical velocity (m/s), positive = descent, negative = ascent.

    Peak is BL-top subsidence (~1 mm/s), not mid-tropospheric omega (3-5 mm/s).
    Uses a narrow Gaussian (σ=7°) for descent, with compensating ITCZ ascent.
    """
    lat_subtrop_north = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad
    lat_subtrop_south = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_rad

    # Hadley-Ferrel boundary: descent must go to zero here
    sigma_descent = np.deg2rad(7.0)  # Narrower than pressure σ=12°

    # Subtropical descent (positive = downward)
    w_descent = peak_velocity_m_s * (
        np.exp(-((lat_rad - lat_subtrop_south) / sigma_descent) ** 2)
        + np.exp(-((lat_rad - lat_subtrop_north) / sigma_descent) ** 2)
    )

    # ITCZ ascent (negative = upward)
    w_ascent = peak_velocity_m_s * np.exp(-((lat_rad - itcz_rad) / SIGMA_ITCZ) ** 2)

    return w_descent - w_ascent


def compute_hadley_subsidence_drying(
    w_descent: np.ndarray,
    humidity_field: np.ndarray,
    upper_troposphere_q_fraction: float = UPPER_TROPOSPHERE_Q_FRACTION,
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute humidity tendency from Hadley subsidence mixing dry air into BL.

    dq/dt = (w / h_BL) * (q_upper - q_BL)

    Only applies where w > 0 (descent). Ascent regions (w < 0) are handled
    by convergence via advection, not by this term.
    """
    q_upper = humidity_field * upper_troposphere_q_fraction
    delta_q = q_upper - humidity_field  # Always negative
    mixing_rate = np.maximum(w_descent, 0.0) / boundary_layer_height_m
    return mixing_rate * delta_q


def hadley_subsidence_drying_jacobian(
    w_descent: np.ndarray,
    upper_troposphere_q_fraction: float = UPPER_TROPOSPHERE_Q_FRACTION,
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Diagonal of the humidity Jacobian from Hadley subsidence drying.

    d(dq/dt)/dq = (w / h_BL) * (f_upper - 1) = -(1 - f_upper) * w / h_BL
    Only where w > 0 (descent). Always negative (stabilizing).
    """
    return np.maximum(w_descent, 0.0) / boundary_layer_height_m * (upper_troposphere_q_fraction - 1.0)


def compute_hadley_convergence_moistening(
    w_hadley: np.ndarray,
    humidity_field: np.ndarray,
    lat_rad: np.ndarray,
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Compute humidity tendency from Hadley cell surface convergence near the ITCZ.

    Where the Hadley cell ascends (w < 0), mass continuity requires surface
    convergence.  Trade winds bring moist subtropical BL air toward the ITCZ.
    This is the moisture source that the 2-layer model's advection scheme
    cannot resolve because it lacks mean-meridional overturning.

    The moistening rate is:
        dq/dt = |w| / h_BL × (q_source - q_local)

    where q_source is the zonal-mean humidity in the surrounding subtropical
    belt (15-30° from equator in both hemispheres).  This represents the
    moisture carried equatorward by the trade winds.

    Only applies where w < 0 (ascent).  Descent regions are handled by
    ``compute_hadley_subsidence_drying``.
    """
    # Ascent rate (positive magnitude where w < 0)
    ascent_rate = np.maximum(-w_hadley, 0.0) / boundary_layer_height_m  # 1/s

    # Compute zonal-mean subtropical q as the moisture source.
    # Subtropics = 15-30° latitude in both hemispheres.
    lat_deg = np.rad2deg(np.abs(lat_rad))
    # Use a smooth weight to select subtropical belt
    # Peaks at 22.5°, tapers at 15° and 30°
    subtrop_weight = np.exp(-((lat_deg - 22.5) / 7.0) ** 2)
    # Zonal mean weighted by subtropical belt
    weighted_q = humidity_field * subtrop_weight
    # Average over longitude (axis=1) and latitude (weighted)
    q_source_zonal = (np.sum(weighted_q, axis=1) /
                      np.maximum(np.sum(subtrop_weight, axis=1), 1e-10))
    # Broadcast back to 2D (same q_source at all longitudes)
    q_source = q_source_zonal[:, np.newaxis] * np.ones(humidity_field.shape[1])

    # Moistening tendency: convergence brings subtropical air into ITCZ
    dq_dt = ascent_rate * (q_source - humidity_field)

    # Only moisten (don't dry) — convergence adds moisture
    return np.maximum(dq_dt, 0.0)


def hadley_convergence_moistening_jacobian(
    w_hadley: np.ndarray,
    boundary_layer_height_m: float = BOUNDARY_LAYER_HEIGHT_M,
) -> np.ndarray:
    """Diagonal of the humidity Jacobian from Hadley convergence moistening.

    d(dq/dt)/dq ≈ -|w| / h_BL  (where w < 0)

    The q_source term also depends on q but across multiple cells,
    so we only include the local (diagonal) part: -ascent_rate.
    This is negative (stabilizing).
    """
    ascent_rate = np.maximum(-w_hadley, 0.0) / boundary_layer_height_m
    return -ascent_rate

# Potential temperature factor: θ_atm = T_atm × (P0/P_ATM)^κ
_P0 = 1013.25  # hPa
_P_ATM = 500.0  # hPa
_KAPPA = 0.286  # R/cp
_ALPHA = (_P0 / _P_ATM) ** _KAPPA  # ≈ 1.219


def compute_hadley_upper_velocity(
    w_hadley: np.ndarray,
    lat_rad: np.ndarray,
    itcz_rad: np.ndarray,
    h_atm: float = ATMOSPHERE_LAYER_HEIGHT_M,
    rho: float = 0.5,  # kg/m³, upper-troposphere density (~300 hPa)
) -> np.ndarray:
    """Compute upper-branch meridional velocity from Hadley cell mass continuity.

    Integrates w outward from the ITCZ in both directions, confined to
    the Hadley cell (between ITCZ and subtropical descent latitudes).
    """
    # Zonal-mean w and cos(lat) — work on 1D latitude profile
    lat_1d = lat_rad[:, 0]  # radians
    cos_lat = np.cos(lat_1d)
    w_zonal = np.mean(w_hadley, axis=1)  # (nlat,)
    nlat = lat_rad.shape[0]
    dlat = np.pi / nlat  # radians

    # ITCZ index from passed itcz_rad
    itcz_lat_rad = float(np.ravel(itcz_rad)[0])
    itcz_idx = int(np.argmin(np.abs(lat_1d - itcz_lat_rad)))
    lat_sub_n = LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_lat_rad
    lat_sub_s = -LAT_SUBTROPICS_BASE + SUBTROPICS_ITCZ_COUPLING * itcz_lat_rad

    integrand = w_zonal * cos_lat * R_EARTH_METERS * dlat  # m²/s per band
    v_upper_1d = np.zeros(nlat)
    denom = rho * h_atm * np.maximum(cos_lat, 0.05)

    # Integrate northward from ITCZ (northern Hadley cell)
    cum = 0.0
    for j in range(itcz_idx, nlat):
        cum += integrand[j]
        if cum > 0:
            cum = 0.0
        v_upper_1d[j] = -cum / denom[j]

    # Integrate southward from ITCZ (southern Hadley cell)
    cum = 0.0
    for j in range(itcz_idx - 1, -1, -1):
        cum += integrand[j]
        if cum > 0:
            cum = 0.0
        v_upper_1d[j] = cum / denom[j]

    # Zero outside Hadley cells (poleward of descent zones + margin)
    margin = np.deg2rad(10.0)
    hadley_mask = np.ones(nlat)
    for j in range(nlat):
        if lat_1d[j] > lat_sub_n + margin or lat_1d[j] < lat_sub_s - margin:
            hadley_mask[j] = 0.0
        elif lat_1d[j] > lat_sub_n:
            # Taper to zero
            hadley_mask[j] = 1.0 - (lat_1d[j] - lat_sub_n) / margin
        elif lat_1d[j] < lat_sub_s:
            hadley_mask[j] = 1.0 - (lat_sub_s - lat_1d[j]) / margin

    v_upper_1d *= hadley_mask

    # Broadcast to 2D
    nlon = lat_rad.shape[1]
    v_upper = v_upper_1d[:, np.newaxis] * np.ones(nlon)

    return v_upper


def compute_bl_atm_mixing_tendencies(
    T_bl: np.ndarray,
    T_atm: np.ndarray,
    C_bl: float = BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    C_atm: float = ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    tau_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Background BL-atmosphere heat exchange via subsidence and mixing.

    Relaxes the BL toward the potential temperature of the free atmosphere,
    closing the latent heat return loop: surface evaporation -> condensation
    aloft -> radiative cooling -> subsidence warming back to BL.

    Energy-conserving: C_bl * dT_bl + C_atm * dT_atm = 0.
    """
    theta_atm = T_atm * _ALPHA  # Potential temperature of free atm at surface

    tau = tau_s if tau_s is not None else 7.0 * 86400.0

    heat_flux = C_bl * (theta_atm - T_bl) / tau  # W/m²

    dT_bl = heat_flux / C_bl      # = (theta_atm - T_bl) / tau_rad
    dT_atm = -heat_flux / C_atm   # Energy conservation

    return dT_bl, dT_atm
