"""Atmospheric advection operator for temperature transport by wind."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from climate_sim.data.constants import R_EARTH_METERS


@dataclass(frozen=True)
class AdvectionConfig:
    """Configuration for atmospheric advection."""
    enabled: bool = True


class AdvectionOperator:
    """Compute advection tendency -u·∇T for atmospheric temperature transport."""

    def __init__(
        self,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        *,
        config: AdvectionConfig | None = None,
    ) -> None:
        """Initialize advection operator on a regular lat-lon grid.

        Parameters
        ----------
        lon2d : np.ndarray
            2D longitude grid in degrees, shape (nlat, nlon)
        lat2d : np.ndarray
            2D latitude grid in degrees, shape (nlat, nlon)
        config : AdvectionConfig | None
            Configuration options for advection scheme
        """
        if lon2d.shape != lat2d.shape:
            raise ValueError("Longitude and latitude grids must share the same shape")
        if lon2d.ndim != 2:
            raise ValueError("Longitude and latitude grids must be two-dimensional")

        self._lon2d = np.asarray(lon2d, dtype=float)
        self._lat2d = np.asarray(lat2d, dtype=float)
        self._config = config or AdvectionConfig()

        # Cache for linearized advection matrix
        self._cached_winds: tuple[np.ndarray, np.ndarray] | None = None
        self._cached_matrix: sparse.csr_matrix | None = None

        nlat, nlon = self._lon2d.shape
        if nlat < 3 or nlon < 3:
            raise ValueError("Grid must have at least 3 points in each dimension")

        # Extract 1D coordinate arrays
        lat_centers = self._lat2d[:, 0]
        lon_centers = self._lon2d[0, :]

        # Verify regular spacing
        lat_spacing = np.diff(lat_centers)
        if not np.allclose(lat_spacing, lat_spacing[0]):
            raise ValueError("Latitude grid must have constant spacing")

        lon_spacing = np.diff(lon_centers)
        if not np.allclose(lon_spacing, lon_spacing[0]):
            raise ValueError("Longitude grid must have constant spacing")

        # Compute metric coefficients
        # dy is constant for regular latitude grid
        self._delta_y = R_EARTH_METERS * np.deg2rad(float(lat_spacing[0]))

        # dx varies with latitude: dx = R * cos(lat) * dlon
        delta_lon_rad = np.deg2rad(float(lon_spacing[0]))
        cos_lat = np.cos(np.deg2rad(lat_centers))[:, np.newaxis]
        self._delta_x = R_EARTH_METERS * cos_lat * delta_lon_rad

        # Precompute inverse spacings for gradient calculations
        self._inv_two_delta_y = 1.0 / (2.0 * self._delta_y)

        # Handle potential division by zero near poles
        with np.errstate(divide='ignore', invalid='ignore'):
            self._inv_two_delta_x = np.zeros_like(self._delta_x)
            valid = np.abs(self._delta_x) > 0.0
            self._inv_two_delta_x[valid] = 1.0 / (2.0 * self._delta_x[valid])

    @property
    def enabled(self) -> bool:
        """Return whether advection is enabled."""
        return self._config.enabled

    def tendency(
        self,
        temperature: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
    ) -> np.ndarray:
        """Compute advection tendency -u·∇T.

        Parameters
        ----------
        temperature : np.ndarray
            Temperature field in K, shape (nlat, nlon)
        wind_u : np.ndarray
            Zonal wind component (eastward) in m/s, shape (nlat, nlon)
        wind_v : np.ndarray
            Meridional wind component (northward) in m/s, shape (nlat, nlon)

        Returns
        -------
        np.ndarray
            Temperature tendency in K/s from advection, shape (nlat, nlon)
        """
        if not self.enabled:
            return np.zeros_like(temperature)

        if temperature.shape != self._lon2d.shape:
            raise ValueError("Temperature field must match grid shape")
        if wind_u.shape != self._lon2d.shape:
            raise ValueError("Zonal wind field must match grid shape")
        if wind_v.shape != self._lon2d.shape:
            raise ValueError("Meridional wind field must match grid shape")

        # Compute temperature gradients using upwind differencing
        dT_dx, dT_dy = self._upwind_gradient(temperature, wind_u, wind_v)

        # Advection tendency: -u·∇T = -(u ∂T/∂x + v ∂T/∂y)
        # Already in K/s - no heat capacity division needed (unlike radiation which converts energy flux)
        tendency = -(wind_u * dT_dx + wind_v * dT_dy)

        return tendency

    def _upwind_gradient(
        self,
        field: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradients using upwind finite differences.

        Uses first-order upwind differencing based on wind direction.
        More diffusive but numerically stable.

        Parameters
        ----------
        field : np.ndarray
            Scalar field to differentiate
        wind_u : np.ndarray
            Zonal wind component in m/s
        wind_v : np.ndarray
            Meridional wind component in m/s

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (∂field/∂x, ∂field/∂y) using upwind scheme
        """
        nlat, nlon = field.shape

        # Zonal upwind gradient
        field_east = np.roll(field, -1, axis=1)
        field_west = np.roll(field, 1, axis=1)

        # If u > 0 (eastward), use backward difference (upwind from west)
        # If u < 0 (westward), use forward difference (upwind from east)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_delta_x = np.zeros_like(self._delta_x)
            valid = np.abs(self._delta_x) > 0.0
            inv_delta_x[valid] = 1.0 / self._delta_x[valid]

        grad_x_backward = (field - field_west) * inv_delta_x
        grad_x_forward = (field_east - field) * inv_delta_x
        grad_x = np.where(wind_u >= 0, grad_x_backward, grad_x_forward)

        # Meridional upwind gradient
        inv_delta_y = 1.0 / self._delta_y
        grad_y = np.zeros_like(field)

        # Interior points
        if nlat > 1:
            field_north = np.roll(field, -1, axis=0)
            field_south = np.roll(field, 1, axis=0)

            grad_y_backward = (field - field_south) * inv_delta_y
            grad_y_forward = (field_north - field) * inv_delta_y

            # If v > 0 (northward), use backward (from south)
            # If v < 0 (southward), use forward (from north)
            grad_y = np.where(wind_v >= 0, grad_y_backward, grad_y_forward)

            # Handle pole boundaries with one-sided differences
            grad_y[0, :] = (field[1, :] - field[0, :]) * inv_delta_y
            grad_y[-1, :] = (field[-1, :] - field[-2, :]) * inv_delta_y

        return grad_x, grad_y

    def linearised_tendency(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
    ) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        """Return diagonal and off-diagonal pieces of the advection Jacobian.

        The Jacobian of -u·∇T with respect to T is the linear operator -u·∇,
        which has zero diagonal and off-diagonal entries representing the gradient stencil.

        Parameters
        ----------
        wind_u : np.ndarray
            Zonal wind component in m/s
        wind_v : np.ndarray
            Meridional wind component in m/s

        Returns
        -------
        tuple[np.ndarray, sparse.csr_matrix | None]
            (diagonal, off_diagonal_matrix) where diagonal is zero and matrix is the -u·∇ operator
        """
        if not self.enabled:
            return np.zeros_like(self._lon2d), None

        nlat, nlon = self._lon2d.shape
        size = nlat * nlon

        # Check if we can reuse cached matrix (winds haven't changed)
        if self._cached_winds is not None:
            cached_u, cached_v = self._cached_winds
            if np.array_equal(cached_u, wind_u) and np.array_equal(cached_v, wind_v):
                return np.zeros((nlat, nlon)), self._cached_matrix

        # Build the sparse matrix for -u·∇ operator using upwind differencing
        row_indices = []
        col_indices = []
        values = []

        # Precompute inverse spacings
        inv_delta_y = 1.0 / self._delta_y
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_delta_x = np.zeros_like(self._delta_x)
            valid = np.abs(self._delta_x) > 0.0
            inv_delta_x[valid] = 1.0 / self._delta_x[valid]

        for i in range(nlat):
            for j in range(nlon):
                idx = i * nlon + j
                u_val = wind_u[i, j]
                v_val = wind_v[i, j]

                # Zonal advection contribution: -u * ∂T/∂x (upwind)
                # If u >= 0 (eastward): ∂T/∂x ≈ (T[i,j] - T[i,j-1])/dx  (backward difference)
                # If u < 0 (westward): ∂T/∂x ≈ (T[i,j+1] - T[i,j])/dx  (forward difference)
                j_east = (j + 1) % nlon
                j_west = (j - 1) % nlon
                inv_dx = inv_delta_x[i, 0] if inv_delta_x.ndim == 2 else inv_delta_x[i]

                if u_val >= 0:
                    # Backward: -u * (T[j] - T[j-1])/dx
                    row_indices.extend([idx, idx])
                    col_indices.extend([idx, i * nlon + j_west])
                    values.extend([-u_val * inv_dx, u_val * inv_dx])
                else:
                    # Forward: -u * (T[j+1] - T[j])/dx
                    row_indices.extend([idx, idx])
                    col_indices.extend([i * nlon + j_east, idx])
                    values.extend([-u_val * inv_dx, u_val * inv_dx])

                # Meridional advection contribution: -v * ∂T/∂y (upwind)
                if nlat > 1:
                    if 0 < i < nlat - 1:
                        # Interior points
                        if v_val >= 0:
                            # Northward: backward difference from south
                            row_indices.extend([idx, idx])
                            col_indices.extend([idx, (i - 1) * nlon + j])
                            values.extend([-v_val * inv_delta_y, v_val * inv_delta_y])
                        else:
                            # Southward: forward difference from north
                            row_indices.extend([idx, idx])
                            col_indices.extend([(i + 1) * nlon + j, idx])
                            values.extend([-v_val * inv_delta_y, v_val * inv_delta_y])
                    else:
                        # Pole boundaries: one-sided differences
                        if i == 0:
                            # South pole: forward difference
                            row_indices.extend([idx, idx])
                            col_indices.extend([(i + 1) * nlon + j, idx])
                            values.extend([-v_val * inv_delta_y, v_val * inv_delta_y])
                        else:  # i == nlat - 1
                            # North pole: backward difference
                            row_indices.extend([idx, idx])
                            col_indices.extend([idx, (i - 1) * nlon + j])
                            values.extend([-v_val * inv_delta_y, v_val * inv_delta_y])

        # Convert to numpy arrays with explicit dtypes (matching diffusion.py pattern)
        rows_arr = np.asarray(row_indices, dtype=np.int64)
        cols_arr = np.asarray(col_indices, dtype=np.int64)
        data_arr = np.asarray(values, dtype=np.float64)

        # Build COO matrix first, then convert to CSR
        coo = sparse.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(size, size))
        matrix = coo.tocsr()

        # Cache for reuse
        self._cached_winds = (wind_u.copy(), wind_v.copy())
        self._cached_matrix = matrix

        return np.zeros((nlat, nlon)), matrix
