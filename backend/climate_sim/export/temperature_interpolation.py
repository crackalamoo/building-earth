"""Spherical interpolation utilities for upscaling coarse solver output."""

from __future__ import annotations

import numpy as np

from climate_sim.data.constants import STANDARD_LAPSE_RATE_K_PER_M
from climate_sim.data.landmask import compute_land_mask
from climate_sim.data.elevation import compute_cell_elevation, load_elevation_data
from climate_sim.core.timing import time_block


def sample_elevation_at_points(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> np.ndarray:
    """Sample elevation directly from ETOPO at each point using fast numpy indexing.

    This gives true high-resolution elevation for lapse rate corrections.
    Uses direct array indexing instead of slow xarray interpolation.

    Parameters
    ----------
    lat2d, lon2d : np.ndarray
        Grid coordinates (any resolution)

    Returns
    -------
    np.ndarray
        Elevation in meters at each point, same shape as input
    """
    dataset = load_elevation_data()
    if dataset is None:
        return np.zeros_like(lat2d, dtype=float)

    # Get the raw numpy array and coordinate info
    elev_data = dataset.values
    x_coords = dataset.coords["x"].values  # longitudes, typically -180 to 180
    y_coords = dataset.coords["y"].values  # latitudes, typically 90 to -90 (descending)

    # ETOPO is 60 arc-seconds = 1/60 degree resolution
    # x_coords goes from -180 to ~180, y_coords from 90 to -90
    x_min, x_max = float(x_coords[0]), float(x_coords[-1])
    y_min, y_max = float(y_coords[-1]), float(y_coords[0])  # y is descending

    nx = len(x_coords)
    ny = len(y_coords)

    # Compute pixel spacing
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    # Wrap longitudes to [-180, 180) for ETOPO
    lon_wrapped = ((lon2d + 180.0) % 360.0) - 180.0

    # Convert coordinates to array indices (nearest neighbor)
    # x index: (lon - x_min) / dx
    # y index: (y_max - lat) / dy  (because y is descending from 90 to -90)
    x_idx = np.round((lon_wrapped - x_min) / dx).astype(int)
    y_idx = np.round((y_max - lat2d) / dy).astype(int)

    # Clip to valid range
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)

    # Direct indexing - very fast!
    elevation = elev_data[y_idx, x_idx]

    return np.asarray(elevation, dtype=float)


def _apply_nearest_neighbor_fallback(
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    weights: np.ndarray,
    needs_fallback: np.ndarray,
    fine_lat2d: np.ndarray,
    fine_lon2d: np.ndarray,
    coarse_lats: np.ndarray,
    coarse_lons: np.ndarray,
    coarse_land_mask: np.ndarray,
    fine_land_mask: np.ndarray,
    *,
    max_search_distance_deg: float = 15.0,
) -> np.ndarray:
    """Replace indices/weights for fallback points with nearest same-type neighbor.

    Modifies lat_indices, lon_indices, and weights in-place.
    For fallback points, sets weight=1 for index 0 pointing to nearest same-type cell.

    If the nearest same-type cell is farther than max_search_distance_deg, falls back
    to nearest cell of any type (e.g., isolated islands use surrounding ocean).

    Uses vectorized computation for efficiency.

    Returns
    -------
    uses_wrong_type : np.ndarray
        Boolean mask (nlat_fine, nlon_fine) indicating fine points that ended up
        using a coarse cell of the wrong type (e.g., land point using ocean data).
        These points should NOT have lapse rate correction applied.
    """
    # Track which points use wrong-type source data
    uses_wrong_type = np.zeros(fine_lat2d.shape, dtype=bool)

    # Get coordinates of all coarse land and ocean cells
    land_indices_arr = np.argwhere(coarse_land_mask)
    ocean_indices_arr = np.argwhere(~coarse_land_mask)

    land_lats = coarse_lats[land_indices_arr[:, 0]]
    land_lons = coarse_lons[land_indices_arr[:, 1]]
    ocean_lats = coarse_lats[ocean_indices_arr[:, 0]]
    ocean_lons = coarse_lons[ocean_indices_arr[:, 1]]

    max_dist_sq = max_search_distance_deg**2

    # Process land fallback points
    land_fallback = needs_fallback & fine_land_mask
    if np.any(land_fallback):
        fb_lats = fine_lat2d[land_fallback]
        fb_lons = fine_lon2d[land_fallback]

        # Compute distances to all land coarse cells (vectorized)
        lat_diff = fb_lats[:, np.newaxis] - land_lats[np.newaxis, :]
        lon_diff = fb_lons[:, np.newaxis] - land_lons[np.newaxis, :]
        lon_diff = np.where(lon_diff > 180, lon_diff - 360, lon_diff)
        lon_diff = np.where(lon_diff < -180, lon_diff + 360, lon_diff)
        cos_lat = np.cos(np.radians(fb_lats))[:, np.newaxis]
        land_dist_sq = lat_diff**2 + (lon_diff * cos_lat) ** 2

        # Find nearest land cell for each fallback point
        nearest_land_idx = np.argmin(land_dist_sq, axis=1)
        nearest_land_dist_sq = land_dist_sq[np.arange(len(fb_lats)), nearest_land_idx]

        # For points too far from land, find nearest ocean instead
        too_far = nearest_land_dist_sq > max_dist_sq

        # Default: use nearest land
        nearest_lat_idx = land_indices_arr[nearest_land_idx, 0]
        nearest_lon_idx = land_indices_arr[nearest_land_idx, 1]

        # Override for points too far from land: use nearest ocean
        if np.any(too_far):
            far_fb_lats = fb_lats[too_far]
            far_fb_lons = fb_lons[too_far]

            lat_diff_o = far_fb_lats[:, np.newaxis] - ocean_lats[np.newaxis, :]
            lon_diff_o = far_fb_lons[:, np.newaxis] - ocean_lons[np.newaxis, :]
            lon_diff_o = np.where(lon_diff_o > 180, lon_diff_o - 360, lon_diff_o)
            lon_diff_o = np.where(lon_diff_o < -180, lon_diff_o + 360, lon_diff_o)
            cos_lat_o = np.cos(np.radians(far_fb_lats))[:, np.newaxis]
            ocean_dist_sq = lat_diff_o**2 + (lon_diff_o * cos_lat_o) ** 2

            nearest_ocean_idx = np.argmin(ocean_dist_sq, axis=1)
            nearest_lat_idx[too_far] = ocean_indices_arr[nearest_ocean_idx, 0]
            nearest_lon_idx[too_far] = ocean_indices_arr[nearest_ocean_idx, 1]

            # Mark these land points as using ocean data
            land_fallback_indices = np.argwhere(land_fallback)
            for idx, is_too_far in enumerate(too_far):
                if is_too_far:
                    i, j = land_fallback_indices[idx]
                    uses_wrong_type[i, j] = True

        # Apply to arrays
        lat_indices[land_fallback, 0] = nearest_lat_idx
        lon_indices[land_fallback, 0] = nearest_lon_idx
        weights[land_fallback, 0] = 1.0
        weights[land_fallback, 1:] = 0.0

    # Process ocean fallback points (ocean is everywhere, so no distance limit needed)
    ocean_fallback = needs_fallback & ~fine_land_mask
    if np.any(ocean_fallback):
        fb_lats = fine_lat2d[ocean_fallback]
        fb_lons = fine_lon2d[ocean_fallback]

        lat_diff = fb_lats[:, np.newaxis] - ocean_lats[np.newaxis, :]
        lon_diff = fb_lons[:, np.newaxis] - ocean_lons[np.newaxis, :]
        lon_diff = np.where(lon_diff > 180, lon_diff - 360, lon_diff)
        lon_diff = np.where(lon_diff < -180, lon_diff + 360, lon_diff)
        cos_lat = np.cos(np.radians(fb_lats))[:, np.newaxis]
        dist_sq = lat_diff**2 + (lon_diff * cos_lat) ** 2

        nearest_idx = np.argmin(dist_sq, axis=1)
        nearest_lat_idx = ocean_indices_arr[nearest_idx, 0]
        nearest_lon_idx = ocean_indices_arr[nearest_idx, 1]

        lat_indices[ocean_fallback, 0] = nearest_lat_idx
        lon_indices[ocean_fallback, 0] = nearest_lon_idx
        weights[ocean_fallback, 0] = 1.0
        weights[ocean_fallback, 1:] = 0.0

    return uses_wrong_type


def _apply_interpolation_overrides(
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    weights: np.ndarray,
    fine_lat2d: np.ndarray,
    fine_lon2d: np.ndarray,
    fine_land_mask: np.ndarray,
    coarse_lats: np.ndarray,
    coarse_lons: np.ndarray,
) -> None:
    """Override interpolation sources for regions with bad automatic matches.

    Modifies lat_indices, lon_indices, weights in-place.
    """
    # Each entry: dict with keys:
    #   fine_lat: (min, max) — fine cell latitude range
    #   fine_lon: (min, max) — fine cell longitude range
    #   fine_type: "land" or "ocean"
    #   sources: list of (coarse_lat, coarse_lon) — cells to blend between
    #   blend_axis: "lat" or "lon" — which axis to weight by proximity
    OVERRIDES = [
        {  # Sicily → Italy (not Tunisia)
            "fine_lat": (36.0, 39.0),
            "fine_lon": (12.0, 16.0),
            "fine_type": "land",
            "sources": [(42.5, 12.5)],
            "blend_axis": "lat",
        },
        {  # Libya coast → Libya interior (not Greece)
            "fine_lat": (30.0, 35.0),
            "fine_lon": (15.0, 25.0),
            "fine_type": "land",
            "sources": [(27.5, 17.5), (27.5, 22.5)],
            "blend_axis": "lon",
        },
        {  # Red Sea → E Mediterranean + Gulf of Aden
            "fine_lat": (12.0, 30.0),
            "fine_lon": (32.0, 44.0),
            "fine_type": "ocean",
            "sources": [(32.5, 27.5), (7.5, 52.5)],
            "blend_axis": "lat",
        },
        {  # Aegean Sea → E Mediterranean
            "fine_lat": (35.0, 41.0),
            "fine_lon": (23.0, 28.0),
            "fine_type": "ocean",
            "sources": [(32.5, 22.5), (32.5, 27.5)],
            "blend_axis": "lon",
        },
    ]

    for override in OVERRIDES:
        lat_min, lat_max = override["fine_lat"]
        lon_min, lon_max = override["fine_lon"]
        is_land = override["fine_type"] == "land"
        sources = override["sources"]
        blend_axis = override["blend_axis"]

        type_mask = fine_land_mask if is_land else ~fine_land_mask
        region = (
            (fine_lat2d >= lat_min)
            & (fine_lat2d <= lat_max)
            & (fine_lon2d >= lon_min)
            & (fine_lon2d <= lon_max)
            & type_mask
        )
        if not np.any(region):
            continue

        # Look up coarse grid indices for each source cell
        source_indices = []
        for s_lat, s_lon in sources:
            i_lat = int(np.argmin(np.abs(coarse_lats - s_lat)))
            i_lon = int(np.argmin(np.abs(coarse_lons - s_lon)))
            source_indices.append((i_lat, i_lon))

        # Compute weights by proximity along the blend axis
        src_coords = np.array(
            [
                coarse_lats[si[0]] if blend_axis == "lat" else coarse_lons[si[1]]
                for si in source_indices
            ]
        )

        fine_coord = fine_lat2d if blend_axis == "lat" else fine_lon2d

        # Inverse-distance weights between the two sources
        n_sources = len(source_indices)
        for idx_fine in zip(*np.where(region)):
            fc = fine_coord[idx_fine]
            dists = np.array([abs(fc - sc) for sc in src_coords])
            dists = np.maximum(dists, 0.01)  # avoid div-by-zero
            inv_dists = 1.0 / dists
            w = inv_dists / inv_dists.sum()

            # Set the first n_sources slots to our override sources
            for k in range(min(n_sources, 4)):
                lat_indices[idx_fine][k] = source_indices[k][0]
                lon_indices[idx_fine][k] = source_indices[k][1]
                weights[idx_fine][k] = w[k]
            # Zero out remaining slots
            for k in range(n_sources, 4):
                weights[idx_fine][k] = 0.0


def _build_bilinear_weights(
    coarse_lats: np.ndarray,
    coarse_lons: np.ndarray,
    fine_lat2d: np.ndarray,
    fine_lon2d: np.ndarray,
    coarse_land_mask: np.ndarray,
    fine_land_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build bilinear interpolation indices and weights.

    For each fine grid point, finds the 4 surrounding coarse cell centers
    and computes bilinear weights. Filters by land/ocean type.

    Returns
    -------
    lat_indices : np.ndarray
        Shape (nlat_fine, nlon_fine, 4) - latitude indices of 4 neighbors [SW, SE, NW, NE]
    lon_indices : np.ndarray
        Shape (nlat_fine, nlon_fine, 4) - longitude indices of 4 neighbors
    weights : np.ndarray
        Shape (nlat_fine, nlon_fine, 4) - normalized bilinear weights
    uses_wrong_type : np.ndarray
        Boolean mask (nlat_fine, nlon_fine) for points using wrong surface type
    """
    nlat_coarse = len(coarse_lats)
    nlon_coarse = len(coarse_lons)

    fine_shape = fine_lat2d.shape
    fine_lats_flat = fine_lat2d.ravel()
    fine_lons_flat = fine_lon2d.ravel()

    # Find the coarse cell indices that bracket each fine point
    # For latitude: find i such that coarse_lats[i] <= fine_lat < coarse_lats[i+1]
    # searchsorted gives us the index where fine_lat would be inserted
    lat_idx_upper = np.searchsorted(coarse_lats, fine_lats_flat, side="right")
    lat_idx_lower = lat_idx_upper - 1

    # Clamp to valid range
    lat_idx_lower = np.clip(lat_idx_lower, 0, nlat_coarse - 1)
    lat_idx_upper = np.clip(lat_idx_upper, 0, nlat_coarse - 1)

    # For longitude: handle wraparound
    lon_idx_upper = np.searchsorted(coarse_lons, fine_lons_flat, side="right")
    lon_idx_lower = lon_idx_upper - 1

    # Wrap longitude indices
    lon_idx_lower = lon_idx_lower % nlon_coarse
    lon_idx_upper = lon_idx_upper % nlon_coarse

    # Get the actual coordinate values of the bracketing coarse cells
    lat_lo = coarse_lats[lat_idx_lower]
    lat_hi = coarse_lats[lat_idx_upper]
    lon_lo = coarse_lons[lon_idx_lower]
    lon_hi = coarse_lons[lon_idx_upper]

    # Compute fractional position within the cell [0, 1]
    # Handle edge case where lat_lo == lat_hi
    lat_range = lat_hi - lat_lo
    lat_range = np.where(lat_range == 0, 1.0, lat_range)
    t_lat = (fine_lats_flat - lat_lo) / lat_range
    t_lat = np.clip(t_lat, 0.0, 1.0)

    # For longitude, handle wraparound (when lon_hi < lon_lo due to wrapping)
    lon_range = lon_hi - lon_lo
    # If lon_hi < lon_lo, we wrapped around, so add 360
    lon_range = np.where(lon_range < 0, lon_range + 360.0, lon_range)
    lon_range = np.where(lon_range == 0, 1.0, lon_range)

    # Compute fine_lon position relative to lon_lo, handling wrap
    fine_lon_rel = fine_lons_flat - lon_lo
    fine_lon_rel = np.where(fine_lon_rel < 0, fine_lon_rel + 360.0, fine_lon_rel)
    t_lon = fine_lon_rel / lon_range
    t_lon = np.clip(t_lon, 0.0, 1.0)

    # Bilinear weights for [SW, SE, NW, NE]
    # SW = (1-t_lat) * (1-t_lon)
    # SE = (1-t_lat) * t_lon
    # NW = t_lat * (1-t_lon)
    # NE = t_lat * t_lon
    w_sw = (1.0 - t_lat) * (1.0 - t_lon)
    w_se = (1.0 - t_lat) * t_lon
    w_nw = t_lat * (1.0 - t_lon)
    w_ne = t_lat * t_lon

    # Stack indices: [SW, SE, NW, NE]
    lat_indices = np.stack([lat_idx_lower, lat_idx_lower, lat_idx_upper, lat_idx_upper], axis=-1)
    lon_indices = np.stack([lon_idx_lower, lon_idx_upper, lon_idx_lower, lon_idx_upper], axis=-1)
    weights = np.stack([w_sw, w_se, w_nw, w_ne], axis=-1)

    # Reshape to (nlat_fine, nlon_fine, 4)
    lat_indices = lat_indices.reshape(fine_shape + (4,))
    lon_indices = lon_indices.reshape(fine_shape + (4,))
    weights = weights.reshape(fine_shape + (4,))

    # Apply land/ocean masking
    # Get land mask for each neighbor cell
    neighbor_land_mask = coarse_land_mask[lat_indices, lon_indices]  # (nlat_fine, nlon_fine, 4)
    fine_is_land = fine_land_mask[..., np.newaxis]  # (nlat_fine, nlon_fine, 1)

    # Mask: 1 where surface type matches, 0 otherwise
    type_match = (neighbor_land_mask == fine_is_land).astype(float)

    # Apply mask to weights
    masked_weights = weights * type_match

    # Check if we have any valid weights after masking
    weight_sum = masked_weights.sum(axis=-1, keepdims=True)
    has_valid_neighbors = weight_sum >= 1e-10

    # Normalize weights where we have valid same-type neighbors
    normalized_weights = np.where(
        has_valid_neighbors,
        masked_weights / np.where(weight_sum > 0, weight_sum, 1.0),
        weights,  # placeholder, will be replaced by nearest-neighbor below
    )

    # For points with no same-type neighbors, find nearest same-type coarse cell
    needs_fallback = ~has_valid_neighbors[..., 0]  # (nlat_fine, nlon_fine)
    uses_wrong_type = np.zeros(fine_lat2d.shape, dtype=bool)
    if np.any(needs_fallback):
        uses_wrong_type = _apply_nearest_neighbor_fallback(
            lat_indices,
            lon_indices,
            normalized_weights,
            needs_fallback,
            fine_lat2d,
            fine_lon2d,
            coarse_lats,
            coarse_lons,
            coarse_land_mask,
            fine_land_mask,
        )

    # Apply manual overrides for regions where type-aware interpolation
    # produces geographic teleconnections (e.g., Sicily matching Tunisia).
    _apply_interpolation_overrides(
        lat_indices,
        lon_indices,
        normalized_weights,
        fine_lat2d,
        fine_lon2d,
        fine_land_mask,
        coarse_lats,
        coarse_lons,
    )

    return lat_indices, lon_indices, normalized_weights, uses_wrong_type


def interpolate_field_bilinear(
    coarse_field: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Interpolate a 2D field using precomputed bilinear weights.

    Parameters
    ----------
    coarse_field : np.ndarray
        Field on coarse grid, shape (nlat_coarse, nlon_coarse)
    lat_indices, lon_indices : np.ndarray
        Shape (nlat_fine, nlon_fine, 4) - indices of 4 neighbors
    weights : np.ndarray
        Shape (nlat_fine, nlon_fine, 4) - normalized weights

    Returns
    -------
    np.ndarray
        Interpolated field, shape (nlat_fine, nlon_fine)
    """
    # Gather neighbor values
    neighbor_values = coarse_field[lat_indices, lon_indices]  # (nlat_fine, nlon_fine, 4)

    # Weighted sum
    return (neighbor_values * weights).sum(axis=-1)


def interpolate_monthly_temperature(
    monthly_field: np.ndarray,
    coarse_lon2d: np.ndarray,
    coarse_lat2d: np.ndarray,
    fine_lon2d: np.ndarray,
    fine_lat2d: np.ndarray,
    *,
    coarse_elevation: np.ndarray | None = None,
    fine_elevation: np.ndarray | None = None,
    apply_lapse_rate: bool = False,
) -> np.ndarray:
    """Interpolate a monthly temperature cycle from coarse to fine resolution.

    Uses bilinear interpolation with land/ocean separation.

    Parameters
    ----------
    monthly_field : np.ndarray
        Temperature cycle on coarse grid, shape (12, nlat_coarse, nlon_coarse).
        Expected in Celsius.
    coarse_lon2d, coarse_lat2d : np.ndarray
        Coarse grid coordinates.
    fine_lon2d, fine_lat2d : np.ndarray
        Fine grid coordinates.
    coarse_elevation, fine_elevation : np.ndarray | None
        Elevation grids. Required if apply_lapse_rate=True.
    apply_lapse_rate : bool
        If True, apply lapse rate correction for 2m temperature.

    Returns
    -------
    np.ndarray
        Interpolated monthly cycle, shape (12, nlat_fine, nlon_fine).
    """
    coarse_lats = coarse_lat2d[:, 0]
    coarse_lons = coarse_lon2d[0, :]

    coarse_land_mask = compute_land_mask(coarse_lon2d, coarse_lat2d)
    fine_land_mask = compute_land_mask(fine_lon2d, fine_lat2d)

    # Build interpolation weights once (reused for all months)
    lat_indices, lon_indices, weights, uses_wrong_type = _build_bilinear_weights(
        coarse_lats,
        coarse_lons,
        fine_lat2d,
        fine_lon2d,
        coarse_land_mask,
        fine_land_mask,
    )

    n_months = monthly_field.shape[0]
    fine_shape = fine_lat2d.shape
    result = np.zeros((n_months, fine_shape[0], fine_shape[1]), dtype=float)

    for month_idx in range(n_months):
        result[month_idx] = interpolate_field_bilinear(
            monthly_field[month_idx],
            lat_indices,
            lon_indices,
            weights,
        )

    # Apply lapse rate correction if requested (land only - ocean is at sea level)
    # Skip lapse correction for points using wrong-type data (e.g., isolated islands using ocean)
    if apply_lapse_rate and coarse_elevation is not None and fine_elevation is not None:
        # Interpolate elevation to get the "expected" elevation at fine points
        interp_elevation = interpolate_field_bilinear(
            coarse_elevation,
            lat_indices,
            lon_indices,
            weights,
        )
        elevation_delta = fine_elevation - interp_elevation
        lapse_correction = -STANDARD_LAPSE_RATE_K_PER_M * elevation_delta
        # Only apply lapse correction to land points that are NOT using ocean data
        valid_for_lapse = fine_land_mask & ~uses_wrong_type
        valid_mask_3d = valid_for_lapse[np.newaxis, :, :]
        result = np.where(valid_mask_3d, result + lapse_correction[np.newaxis, :, :], result)

    return result


def interpolate_layer_map(
    layers: dict[str, np.ndarray],
    coarse_lon2d: np.ndarray,
    coarse_lat2d: np.ndarray,
    output_resolution_deg: float = 1.0,
    *,
    apply_lapse_rate_to_2m: bool = False,
    compute_snow_temperature: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Interpolate temperature layers from solver output to a finer grid.

    Only interpolates temperature fields (surface, atmosphere, temperature_2m, boundary_layer).
    Other fields (wind, humidity, etc.) are not interpolated.

    Parameters
    ----------
    layers : dict[str, np.ndarray]
        Layer map from solver, containing temperature fields with shape (12, nlat, nlon).
    coarse_lon2d, coarse_lat2d : np.ndarray
        Coarse grid coordinates.
    output_resolution_deg : float
        Target resolution in degrees.
    apply_lapse_rate_to_2m : bool
        If True, apply lapse rate correction to temperature_2m based on fine elevation.
        Default False to avoid double-counting since solver already applies coarse lapse.

    Returns
    -------
    fine_lon2d, fine_lat2d : np.ndarray
        Fine grid coordinates.
    interpolated_layers : dict[str, np.ndarray]
        Dictionary with interpolated temperature fields.
    """
    from climate_sim.core.grid import create_lat_lon_grid

    with time_block("create_fine_grid"):
        fine_lon2d, fine_lat2d = create_lat_lon_grid(output_resolution_deg)

    with time_block("compute_elevation"):
        # Coarse: cell-averaged elevation (matches what solver used)
        coarse_elevation = compute_cell_elevation(coarse_lon2d, coarse_lat2d)
        # Fine: point-sampled for true high-resolution lapse rate correction
        fine_elevation = sample_elevation_at_points(fine_lat2d, fine_lon2d)

    # Only interpolate surface and 2m temperature (not free atmosphere layers)
    temperature_keys = ["surface", "temperature_2m"]

    interpolated = {}
    for key in temperature_keys:
        if key not in layers:
            continue

        field = layers[key]
        if field.ndim != 3 or field.shape[0] != 12:
            continue

        should_apply_lapse = apply_lapse_rate_to_2m and key == "temperature_2m"

        with time_block(f"interpolate_{key}"):
            interpolated[key] = interpolate_monthly_temperature(
                field,
                coarse_lon2d,
                coarse_lat2d,
                fine_lon2d,
                fine_lat2d,
                coarse_elevation=coarse_elevation,
                fine_elevation=fine_elevation,
                apply_lapse_rate=should_apply_lapse,
            )

    # Compute snow_temperature: lapse-corrected surface temp at point elevation.
    # Uses point-sampled elevation (what the vertex is displaced to) so snow
    # only appears where the rendered terrain is actually high enough to be cold.
    if compute_snow_temperature and "surface" in interpolated:
        interp_surface = interpolated["surface"]  # (12, fine_nlat, fine_nlon)
        # Interpolate coarse elevation to fine grid to get smooth baseline
        coarse_lats = coarse_lat2d[:, 0]
        coarse_lons = coarse_lon2d[0, :]
        coarse_land_mask = compute_land_mask(coarse_lon2d, coarse_lat2d)
        fine_land_mask = compute_land_mask(fine_lon2d, fine_lat2d)
        lat_idx, lon_idx, wts, _ = _build_bilinear_weights(
            coarse_lats,
            coarse_lons,
            fine_lat2d,
            fine_lon2d,
            coarse_land_mask,
            fine_land_mask,
        )
        interp_elev = interpolate_field_bilinear(coarse_elevation, lat_idx, lon_idx, wts)
        elev_delta = np.maximum(0, fine_elevation - interp_elev)
        SNOW_LAPSE_K_PER_M = 5.0e-3
        snow_temp = interp_surface - SNOW_LAPSE_K_PER_M * elev_delta[np.newaxis, :, :]
        interpolated["snow_temperature"] = snow_temp

    return fine_lon2d, fine_lat2d, interpolated
