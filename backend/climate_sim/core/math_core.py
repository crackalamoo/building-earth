"""Lightweight numerical helpers shared across model components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse import linalg as splinalg

from climate_sim.data.constants import R_EARTH_METERS


def sigmoid(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Smooth sigmoid transition: 0 for x << 0, 1 for x >> 0.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    scale : float
        Controls transition width. Larger scale = smoother transition.

    Returns
    -------
    np.ndarray
        Values in (0, 1).
    """
    z = np.clip(x / scale, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the element-wise harmonic mean for positive arrays."""

    denom = np.zeros_like(a)
    valid = (a > 0.0) & (b > 0.0)
    denom[valid] = (1.0 / a[valid]) + (1.0 / b[valid])

    result = np.zeros_like(a)
    valid_denom = valid & (denom > 0.0)
    result[valid_denom] = 2.0 / denom[valid_denom]
    return result


def _ensure_strictly_increasing(values: np.ndarray, name: str) -> None:
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing for a regular grid")


def regular_latitude_edges(lat_centers_deg: np.ndarray) -> np.ndarray:
    """Infer latitude edges for a regularly spaced grid of cell centers."""

    if lat_centers_deg.ndim != 1:
        raise ValueError("Latitude centers must be a one-dimensional array")

    nlat = lat_centers_deg.size
    if nlat == 0:
        raise ValueError("Latitude centers array must be non-empty")

    if nlat > 1:
        _ensure_strictly_increasing(lat_centers_deg, "Latitude centers")
        spacing = np.diff(lat_centers_deg)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("Latitude grid must have constant spacing")
        delta = float(spacing[0])
    else:
        delta = 180.0

    edges = np.empty(nlat + 1, dtype=float)
    edges[1:-1] = 0.5 * (lat_centers_deg[:-1] + lat_centers_deg[1:])
    edges[0] = lat_centers_deg[0] - 0.5 * delta
    edges[-1] = lat_centers_deg[-1] + 0.5 * delta

    edges[0] = max(edges[0], -90.0)
    edges[-1] = min(edges[-1], 90.0)
    return edges


def regular_longitude_edges(lon_centers_deg: np.ndarray) -> np.ndarray:
    """Infer longitude edges for a regularly spaced, wrapped grid."""

    if lon_centers_deg.ndim != 1:
        raise ValueError("Longitude centers must be a one-dimensional array")

    nlon = lon_centers_deg.size
    if nlon == 0:
        raise ValueError("Longitude centers array must be non-empty")

    if nlon > 1:
        spacing = np.diff(lon_centers_deg)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError("Longitude grid must have constant spacing")
        delta = float(spacing[0])
    else:
        delta = 360.0

    edges = np.empty(nlon + 1, dtype=float)
    edges[1:-1] = 0.5 * (lon_centers_deg[:-1] + lon_centers_deg[1:])
    edges[0] = lon_centers_deg[0] - 0.5 * delta
    edges[-1] = lon_centers_deg[-1] + 0.5 * delta
    return edges


def spherical_cell_area(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    earth_radius_m: float,
) -> np.ndarray:
    """Return the physical surface area of each lon/lat grid cell."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    nlat, nlon = lon2d.shape
    if nlat < 1 or nlon < 1:
        raise ValueError("Longitude/latitude grids must be non-empty")

    lat_centers = lat2d[:, 0]
    lon_centers = lon2d[0, :]

    lat_edges = regular_latitude_edges(lat_centers)
    lon_edges = regular_longitude_edges(lon_centers)

    lat_edges_rad = np.deg2rad(lat_edges)
    lon_edges_rad = np.deg2rad(lon_edges)

    delta_sin = np.sin(lat_edges_rad[1:]) - np.sin(lat_edges_rad[:-1])
    delta_lon = lon_edges_rad[1:] - lon_edges_rad[:-1]

    area = (earth_radius_m**2) * delta_sin[:, np.newaxis] * delta_lon[np.newaxis, :]
    return area


def area_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Compute an area-weighted mean with explicit weight validation."""

    values_arr, weights_arr = np.broadcast_arrays(
        np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    )

    if axis is None:
        total_weight = np.sum(weights_arr)
        if total_weight <= 0.0:
            raise ValueError("Total weight must be positive for a weighted mean")
        return np.sum(values_arr * weights_arr) / total_weight

    numerator = np.sum(values_arr * weights_arr, axis=axis)
    weight_sum = np.sum(weights_arr, axis=axis)

    if np.any(weight_sum <= 0.0):
        raise ValueError("Weight sums must be positive along the specified axis")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / weight_sum
    return result


@dataclass
class LinearSolveCache:
    """Cache for factorized linear solvers and identity matrices."""
    identity_matrices: Dict[int, sparse.csc_matrix] = field(default_factory=dict)
    factorized_solvers: Dict[str, Callable[[np.ndarray], np.ndarray]] = field(default_factory=dict)

DEFAULT_LINEAR_SOLVE_CACHE = LinearSolveCache()


def get_identity_matrix(size: int, *, cache: LinearSolveCache) -> sparse.csc_matrix:
    """Get or create a cached identity matrix of the specified size."""
    identity = cache.identity_matrices.get(size)
    if identity is None:
        identity = sparse.eye(size, format="csc")
        cache.identity_matrices[size] = identity
    return identity


def fingerprint_csc_matrix(matrix: sparse.csc_matrix) -> str:
    """Compute a SHA-1 hash fingerprint of a sparse CSC matrix for caching purposes."""
    if not sparse.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()
    hasher = hashlib.sha1()
    hasher.update(matrix.shape[0].to_bytes(4, byteorder="little", signed=False))
    hasher.update(matrix.shape[1].to_bytes(4, byteorder="little", signed=False))
    hasher.update(matrix.indptr.tobytes())
    hasher.update(matrix.indices.tobytes())
    hasher.update(matrix.data.tobytes())
    return hasher.hexdigest()


def get_factorized_solver(
    matrix: sparse.csc_matrix, *, cache: LinearSolveCache, fingerprint: str | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Get or create a cached factorized linear solver for the given matrix."""
    key = fingerprint or fingerprint_csc_matrix(matrix)
    solver = cache.factorized_solvers.get(key)
    if solver is None:
        solver = splinalg.factorized(matrix)
        cache.factorized_solvers[key] = solver
    return solver


def compute_scalar_gradient_magnitude(
    field: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> np.ndarray:
    """Compute |∇field| on a lat-lon grid.

    |∇f| = sqrt((∂f/∂y)² + (∂f/∂x)²)
    where ∂f/∂x = (1/(R cos φ)) ∂f/∂λ and ∂f/∂y = (1/R) ∂f/∂φ.
    """
    R = R_EARTH_METERS
    lat_rad = np.deg2rad(lat2d)
    cos_lat = np.maximum(np.cos(lat_rad), 0.01)

    dlat = np.deg2rad(lat2d[1, 0] - lat2d[0, 0])
    dlon = np.deg2rad(lon2d[0, 1] - lon2d[0, 0])

    df_dlat = np.gradient(field, axis=0) / dlat
    df_dlon = np.gradient(field, axis=1) / dlon

    df_dy = df_dlat / R
    df_dx = df_dlon / (R * cos_lat)

    return np.sqrt(df_dy**2 + df_dx**2)


def compute_divergence(
    u: np.ndarray,
    v: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    smoothing_length_km: float = 0.0,
) -> np.ndarray:
    """Compute horizontal divergence on a lat-lon grid.

    div(V) = (1/R cos φ) * ∂u/∂λ + (1/R cos φ) * ∂(v cos φ)/∂φ

    The wind field is already derived from a Rossby-radius-smoothed pressure
    field, so divergence inherits that physical filtering.  No additional
    smoothing is applied by default.
    """
    R = R_EARTH_METERS
    lat_rad = np.deg2rad(lat2d)

    cos_lat = np.cos(lat_rad)
    cos_lat = np.maximum(cos_lat, 0.01)  # Avoid division by zero at poles

    # Grid spacings (assumes uniform grid)
    dlat = np.deg2rad(lat2d[1, 0] - lat2d[0, 0])
    dlon = np.deg2rad(lon2d[0, 1] - lon2d[0, 0])

    # ∂u/∂λ
    du_dlon = np.gradient(u, axis=1) / dlon

    # ∂(v cos φ)/∂φ
    v_cos_lat = v * cos_lat
    d_vcos_dlat = np.gradient(v_cos_lat, axis=0) / dlat

    # Divergence
    div = (1 / (R * cos_lat)) * du_dlon + (1 / (R * cos_lat)) * d_vcos_dlat

    # Smooth divergence to remove grid-scale noise
    # Convert smoothing length to grid cells
    grid_spacing_km = np.abs(lat2d[1, 0] - lat2d[0, 0]) * 111.0  # km per degree
    sigma_cells = smoothing_length_km / grid_spacing_km

    if sigma_cells > 0.5:  # Only smooth if smoothing scale > half a grid cell
        # Smooth in latitude (axis 0)
        div = gaussian_filter(div, sigma=(sigma_cells, 0), mode='nearest')
        # Smooth in longitude with wrapping
        div = gaussian_filter(div, sigma=(0, sigma_cells), mode='wrap')

    return div
