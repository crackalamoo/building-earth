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

    def flux_tendency(
        self,
        field: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        dt: float | None = None,
    ) -> np.ndarray:
        """Compute flux-form advection tendency -∇·(u*field) using finite volumes.

        This is exactly conservative: the global area-weighted integral of the
        tendency is exactly zero (to machine precision) because each face flux
        appears twice with opposite signs.

        Uses first-order upwind interpolation for field values at cell faces.
        When dt is provided, applies Zalesak-style flux limiting to prevent
        negative values while maintaining conservation.

        Parameters
        ----------
        field : np.ndarray
            Scalar field to advect (e.g., humidity), shape (nlat, nlon)
        wind_u : np.ndarray
            Zonal wind component (eastward) in m/s, shape (nlat, nlon)
        wind_v : np.ndarray
            Meridional wind component (northward) in m/s, shape (nlat, nlon)
        dt : float | None
            Timestep in seconds. If provided, fluxes are limited to prevent
            negative values. If None, no limiting is applied.
        """
        if not self.enabled:
            return np.zeros_like(field)

        nlat, nlon = field.shape
        R = R_EARTH_METERS
        lat_rad = np.deg2rad(self._lat2d)
        cos_lat = np.cos(lat_rad)

        # Grid spacings in radians
        dlat = np.deg2rad(self._lat2d[1, 0] - self._lat2d[0, 0]) if nlat > 1 else 0.0
        dlon = np.deg2rad(self._lon2d[0, 1] - self._lon2d[0, 0]) if nlon > 1 else 0.0

        # Cell area: A = R² * cos(lat) * dlat * dlon
        cell_area = R**2 * cos_lat * dlat * dlon
        # Avoid division by zero at poles
        cell_area = np.maximum(cell_area, 1e-10)

        # === ZONAL FLUXES (through east/west faces) ===
        field_east = np.roll(field, -1, axis=1)
        u_east_face = 0.5 * (wind_u + np.roll(wind_u, -1, axis=1))
        q_east_face = np.where(u_east_face >= 0, field, field_east)
        face_height = R * dlat
        flux_east = u_east_face * q_east_face * face_height * cos_lat
        flux_west = np.roll(flux_east, 1, axis=1)

        # === MERIDIONAL FLUXES (through north/south faces) ===
        lat_north_face = lat_rad + 0.5 * dlat
        cos_lat_north = np.cos(lat_north_face)
        field_north = np.roll(field, -1, axis=0)
        v_north_face = 0.5 * (wind_v + np.roll(wind_v, -1, axis=0))
        q_north_face = np.where(v_north_face >= 0, field, field_north)
        face_width = R * dlon
        flux_north = v_north_face * q_north_face * face_width * cos_lat_north
        flux_south = np.roll(flux_north, 1, axis=0)

        # Handle polar boundaries
        flux_south[0, :] = 0.0
        flux_north[-1, :] = 0.0

        # === FLUX LIMITING ===
        # Limit outgoing fluxes so they can't remove more than available mass
        # This maintains conservation because each face flux is limited by the
        # donating cell's capacity, and the receiving cell uses the same limited flux.
        if dt is not None and dt > 0:
            # Total mass in each cell
            mass = field * cell_area  # [kg/kg * m²]

            # For each cell, compute total outgoing flux (before limiting)
            # East face: flux_east > 0 means flux leaving this cell eastward
            # West face: we need flux through west face when going westward
            #   flux_west is the flux INTO this cell from the west
            #   flux going OUT to west = -flux_west when flux_west < 0...
            #   Actually, let's be more careful.

            # flux_east[i,j] = flux through face between (i,j) and (i,j+1)
            #   positive = eastward = leaving cell (i,j), entering (i,j+1)
            #   negative = westward = leaving cell (i,j+1), entering (i,j)

            # For cell (i,j), outgoing fluxes are:
            #   East: max(flux_east[i,j], 0) - positive flux_east leaves eastward
            #   West: max(-flux_east[i,j-1], 0) - negative flux at west face leaves westward
            #   North: max(flux_north[i,j], 0) - positive flux_north leaves northward
            #   South: max(-flux_north[i-1,j], 0) - negative flux at south face leaves southward

            # Outgoing through east face of this cell (flux_east > 0)
            out_east = np.maximum(flux_east, 0)
            # Outgoing through west face = negative flux_east of west neighbor's east face
            out_west = np.maximum(-np.roll(flux_east, 1, axis=1), 0)
            # Outgoing through north face (flux_north > 0)
            out_north = np.maximum(flux_north, 0)
            # Outgoing through south face = negative flux_north of south neighbor
            out_south = np.maximum(-np.roll(flux_north, 1, axis=0), 0)
            out_south[0, :] = 0  # No south face at south pole

            total_outflux = (out_east + out_west + out_north + out_south) * dt

            # Limit factor: can't remove more than 90% of mass
            max_removable = 0.9 * np.maximum(mass, 0)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                limit_factor = np.where(
                    total_outflux > 1e-30,
                    np.minimum(1.0, max_removable / (total_outflux + 1e-30)),
                    1.0
                )
            limit_factor = np.nan_to_num(limit_factor, nan=1.0, posinf=1.0, neginf=1.0)

            # Apply limits to fluxes based on donating cell
            # East face: if flux_east > 0, limit by this cell's factor
            #           if flux_east < 0, limit by east neighbor's factor
            limit_east_neighbor = np.roll(limit_factor, -1, axis=1)
            flux_east = np.where(
                flux_east >= 0,
                flux_east * limit_factor,  # Leaving this cell
                flux_east * limit_east_neighbor  # Leaving east neighbor
            )

            # North face: if flux_north > 0, limit by this cell's factor
            #            if flux_north < 0, limit by north neighbor's factor
            limit_north_neighbor = np.roll(limit_factor, -1, axis=0)
            flux_north = np.where(
                flux_north >= 0,
                flux_north * limit_factor,
                flux_north * limit_north_neighbor
            )
            # Re-enforce polar boundaries
            flux_north[-1, :] = 0.0

            # Recompute flux_west and flux_south from the limited face fluxes
            flux_west = np.roll(flux_east, 1, axis=1)
            flux_south = np.roll(flux_north, 1, axis=0)
            flux_south[0, :] = 0.0

        # Compute tendencies
        zonal_tendency = -(flux_east - flux_west) / cell_area
        merid_tendency = -(flux_north - flux_south) / cell_area

        return zonal_tendency + merid_tendency

    def subcycled_flux_tendency(
        self,
        field: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        dt: float,
        max_cfl: float = 0.9,
        field_max: np.ndarray | None = None,
    ) -> np.ndarray:
        """Flux-form advection with subcycling to keep CFL ≤ max_cfl.

        Splits the full timestep into substeps short enough that the flux
        limiter rarely activates, giving physically accurate transport even
        when the outer timestep is much longer than the CFL limit.

        If *field_max* is provided, each substep clamps the field to this
        ceiling.  For humidity advection, this should be q_sat(T_bl) to
        prevent unphysical supersaturation at wind convergence zones.

        Returns the time-averaged tendency (same units as flux_tendency).
        """
        if not self.enabled:
            return np.zeros_like(field)

        # Estimate maximum CFL number
        nlat, nlon = field.shape
        R = R_EARTH_METERS
        lat_rad = np.deg2rad(self._lat2d)
        cos_lat = np.cos(lat_rad)
        dlat = np.deg2rad(self._lat2d[1, 0] - self._lat2d[0, 0]) if nlat > 1 else 1.0
        dlon = np.deg2rad(self._lon2d[0, 1] - self._lon2d[0, 0]) if nlon > 1 else 1.0

        dx = R * np.maximum(cos_lat, 0.05) * dlon
        dy = R * dlat

        max_u = np.max(np.abs(wind_u))
        max_v = np.max(np.abs(wind_v))
        cfl = dt * (max_u / np.min(dx) + max_v / dy)

        n_sub = min(8, max(1, int(np.ceil(cfl / max_cfl))))
        dt_sub = dt / n_sub

        q = field.copy()
        for _ in range(n_sub):
            dq = self.flux_tendency(q, wind_u, wind_v, dt=dt_sub)
            q += dq * dt_sub
            np.maximum(q, 0.0, out=q)
            if field_max is not None:
                np.minimum(q, field_max, out=q)

        # Return average tendency over the full period
        return (q - field) / dt

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

    def linearised_flux_tendency(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
    ) -> tuple[np.ndarray, sparse.csr_matrix | None]:
        """Return diagonal and off-diagonal pieces of the flux-form advection Jacobian.

        The Jacobian of -∇·(u*q) with respect to q is the linear operator -∇·(u·),
        which differs from the advective form -u·∇ by including the divergence term.

        The linearization gives a sparse matrix where each row sums to zero
        (conservation property).
        """
        if not self.enabled:
            return np.zeros_like(self._lon2d), None

        nlat, nlon = self._lon2d.shape
        size = nlat * nlon
        R = R_EARTH_METERS

        lat_rad = np.deg2rad(self._lat2d)
        cos_lat = np.cos(lat_rad)

        # Grid spacings
        dlat = np.deg2rad(self._lat2d[1, 0] - self._lat2d[0, 0]) if nlat > 1 else 0.0
        dlon = np.deg2rad(self._lon2d[0, 1] - self._lon2d[0, 0]) if nlon > 1 else 0.0

        # Cell areas
        cell_area = R**2 * cos_lat * dlat * dlon
        cell_area = np.maximum(cell_area, 1e-10)

        # Face dimensions
        face_height = R * dlat  # For zonal faces
        face_width = R * dlon   # For meridional faces

        # Wind at faces
        u_east_face = 0.5 * (wind_u + np.roll(wind_u, -1, axis=1))
        v_north_face = 0.5 * (wind_v + np.roll(wind_v, -1, axis=0))

        # Latitude at north face
        lat_north_face = lat_rad + 0.5 * dlat
        cos_lat_north = np.cos(lat_north_face)

        # Build sparse matrix
        row_indices = []
        col_indices = []
        values = []

        for i in range(nlat):
            for j in range(nlon):
                idx = i * nlon + j
                area = cell_area[i, j]

                # === ZONAL FLUXES ===
                # East face of cell (i,j)
                j_east = (j + 1) % nlon
                u_e = u_east_face[i, j]
                coeff_east = u_e * face_height * cos_lat[i, j] / area

                if u_e >= 0:
                    # Flux uses q from this cell -> contributes to diagonal
                    # tendency[idx] -= coeff_east * q[idx]
                    row_indices.append(idx)
                    col_indices.append(idx)
                    values.append(-coeff_east)
                else:
                    # Flux uses q from east neighbor
                    # tendency[idx] -= coeff_east * q[idx_east]
                    row_indices.append(idx)
                    col_indices.append(i * nlon + j_east)
                    values.append(-coeff_east)

                # West face of cell (i,j) = East face of cell (i, j-1)
                j_west = (j - 1) % nlon
                u_w = u_east_face[i, j_west]
                coeff_west = u_w * face_height * cos_lat[i, j_west] / area

                if u_w >= 0:
                    # Flux into this cell uses q from west neighbor
                    # tendency[idx] += coeff_west * q[idx_west]
                    row_indices.append(idx)
                    col_indices.append(i * nlon + j_west)
                    values.append(coeff_west)
                else:
                    # Flux into this cell uses q from this cell
                    # tendency[idx] += coeff_west * q[idx]
                    row_indices.append(idx)
                    col_indices.append(idx)
                    values.append(coeff_west)

                # === MERIDIONAL FLUXES ===
                if nlat > 1:
                    # North face of cell (i,j)
                    if i < nlat - 1:
                        i_north = i + 1
                        v_n = v_north_face[i, j]
                        coeff_north = v_n * face_width * cos_lat_north[i, j] / area

                        if v_n >= 0:
                            # Flux uses q from this cell
                            row_indices.append(idx)
                            col_indices.append(idx)
                            values.append(-coeff_north)
                        else:
                            # Flux uses q from north neighbor
                            row_indices.append(idx)
                            col_indices.append(i_north * nlon + j)
                            values.append(-coeff_north)

                    # South face of cell (i,j) = North face of cell (i-1, j)
                    if i > 0:
                        i_south = i - 1
                        v_s = v_north_face[i_south, j]
                        coeff_south = v_s * face_width * cos_lat_north[i_south, j] / area

                        if v_s >= 0:
                            # Flux into this cell uses q from south neighbor
                            row_indices.append(idx)
                            col_indices.append(i_south * nlon + j)
                            values.append(coeff_south)
                        else:
                            # Flux into this cell uses q from this cell
                            row_indices.append(idx)
                            col_indices.append(idx)
                            values.append(coeff_south)

        # Build sparse matrix
        rows_arr = np.asarray(row_indices, dtype=np.int64)
        cols_arr = np.asarray(col_indices, dtype=np.int64)
        data_arr = np.asarray(values, dtype=np.float64)

        coo = sparse.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(size, size))
        matrix = coo.tocsr()

        return np.zeros((nlat, nlon)), matrix
