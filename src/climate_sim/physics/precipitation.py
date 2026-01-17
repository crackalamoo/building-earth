"""Precipitation physics: moisture flux convergence and precipitation estimation.

Physics:
- Precipitation ≈ moisture convergence (what comes in must rain out)
- P = -∇·(q*V) where q is specific humidity, V is wind
- Latent heat release Q = P × L_v
"""

from __future__ import annotations

import numpy as np

from climate_sim.data.constants import R_EARTH_METERS


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


def compute_precipitation_rate(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    rho_air: float = 1.0,
    h_moist: float = 3000.0,
) -> np.ndarray:
    """Estimate precipitation rate from moisture flux convergence.

    Computes moisture flux convergence internally and precipitates
    where there is convergence (mfc > 0).
    """
    mfc = compute_moisture_flux_convergence(q, u, v, lat2d, lon2d)
    P = np.maximum(mfc, 0) * q * rho_air * h_moist
    return P
