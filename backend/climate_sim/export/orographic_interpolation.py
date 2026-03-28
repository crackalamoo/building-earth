"""Recompute precipitation at 1° using orographic physics, conserving 5° totals.

This is a visualization refinement: the solver's 5° precipitation is ground truth,
but within each 5° cell we redistribute rain based on 1° terrain, so mountains
get more and valleys get less.
"""

from __future__ import annotations

from climate_sim.physics.humidity import (
    compute_itcz_latitude,
    specific_humidity_to_relative_humidity,
)
import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.data.elevation import (
    compute_cell_elevation,
    compute_cell_elevation_statistics,
    compute_face_elevation_statistics,
)
from climate_sim.data.landmask import compute_land_mask
from climate_sim.export.temperature_interpolation import (
    _build_bilinear_weights,
    interpolate_field_bilinear,
)
from climate_sim.physics.orographic_effects import OrographicConfig, OrographicModel
from climate_sim.core.math_core import R_EARTH_METERS, spherical_cell_area


def recompute_fields_at_1deg(
    layers: dict[str, np.ndarray],
    coarse_lon2d: np.ndarray,
    coarse_lat2d: np.ndarray,
) -> dict[str, np.ndarray]:
    """Redistribute precipitation from 5° to 1° using orographic physics.

    The solver's 5° precipitation totals are conserved: within each 5° block
    of 25 sub-cells, the 1° precipitation is rescaled so the block mean
    matches the original 5° value.

    Parameters
    ----------
    layers : dict[str, np.ndarray]
        Solver output fields (12, nlat_coarse, nlon_coarse).
    coarse_lon2d, coarse_lat2d : np.ndarray
        Coarse grid coordinates.

    Returns
    -------
    dict with keys: precipitation, humidity, soil_moisture — all (12, 180, 360).
    """
    # --- 1. Build 1° grid ---
    fine_lon2d, fine_lat2d = create_lat_lon_grid(1.0)
    nlat_fine, nlon_fine = fine_lat2d.shape  # 180, 360

    coarse_lats = coarse_lat2d[:, 0]
    coarse_lons = coarse_lon2d[0, :]

    # --- 2. Build bilinear weights (no land/ocean separation for physics fields) ---
    coarse_land_mask = compute_land_mask(coarse_lon2d, coarse_lat2d)
    fine_land_mask = compute_land_mask(fine_lon2d, fine_lat2d)

    lat_indices, lon_indices, weights, _ = _build_bilinear_weights(
        coarse_lats,
        coarse_lons,
        fine_lat2d,
        fine_lon2d,
        coarse_land_mask,
        fine_land_mask,
    )

    # --- 3. Interpolate input fields to 1° ---
    def _interp_monthly(field_name: str) -> np.ndarray:
        coarse = layers[field_name]  # (12, nlat_c, nlon_c)
        result = np.zeros((12, nlat_fine, nlon_fine))
        for m in range(12):
            result[m] = interpolate_field_bilinear(
                coarse[m],
                lat_indices,
                lon_indices,
                weights,
            )
        return result

    T_bl = _interp_monthly("boundary_layer")  # °C
    q = _interp_monthly("humidity")  # kg/kg
    wind_u = _interp_monthly("wind_u_10m")  # m/s
    wind_v = _interp_monthly("wind_v_10m")  # m/s

    # Convert to Kelvin for physics
    T_bl_K = T_bl + 273.15

    # --- 4. Build 1° OrographicModel ---
    print("  Building 1° and 5° orographic models...")

    elevation_1deg = compute_cell_elevation(fine_lon2d, fine_lat2d, cache=False)
    elevation_std_1deg, _ = compute_cell_elevation_statistics(
        fine_lon2d,
        fine_lat2d,
        cache=True,
        cache_name="elevation_statistics_1deg_cache.npz",
    )
    face_stats_1deg = compute_face_elevation_statistics(
        fine_lon2d,
        fine_lat2d,
        cache=True,
        cache_name="face_elevation_1deg_cache.npz",
    )

    oro_model_1deg = OrographicModel(
        lon2d=fine_lon2d,
        lat2d=fine_lat2d,
        elevation=elevation_1deg,
        elevation_std=elevation_std_1deg,
        face_stats=face_stats_1deg,
        config=OrographicConfig(),
        land_mask=fine_land_mask,
    )

    elevation_5deg = compute_cell_elevation(coarse_lon2d, coarse_lat2d, cache=True)
    elevation_std_5deg, _ = compute_cell_elevation_statistics(
        coarse_lon2d,
        coarse_lat2d,
        cache=True,
        cache_name="elevation_statistics_5deg_cache.npz",
    )
    face_stats_5deg = compute_face_elevation_statistics(
        coarse_lon2d,
        coarse_lat2d,
        cache=True,
        cache_name="face_elevation_5deg_cache.npz",
    )

    oro_model_5deg = OrographicModel(
        lon2d=coarse_lon2d,
        lat2d=coarse_lat2d,
        elevation=elevation_5deg,
        elevation_std=elevation_std_5deg,
        face_stats=face_stats_5deg,
        config=OrographicConfig(),
        land_mask=coarse_land_mask,
    )

    # --- 5. Compute precipitation at 1° for each month ---
    #
    # Strategy:
    # 1. Compute coarse orographic precipitation
    # 2. Compute coarse large scale precipitation as the remaining coarse precipitation excluding orographic effects
    # 3. Bilinearly interpolate coarse large scale precipitation
    # 4. Compute and rescale fine grained orographic precipitation
    # 5. Compute final precipitation as rescaled sum of bilinearly interpolated large scale precipitation and fine grained orographic precipitation

    print("  Computing 1° precipitation...")
    precip_1deg = np.zeros((12, nlat_fine, nlon_fine))
    original_precip = layers["precipitation"]  # (12, nlat_c, nlon_c) in kg/m²/s

    p_oro_list = []
    p_ls_bilinear_list = []

    for m in range(12):
        # --- Step 2: orographic modulation at 1° ---
        itcz_rad = compute_itcz_latitude(
            layers["boundary_layer"][m],
            coarse_lat2d,
            spherical_cell_area(coarse_lon2d, coarse_lat2d, earth_radius_m=R_EARTH_METERS),
        )
        assert np.all(itcz_rad == layers["itcz_rad"][m])
        itcz_rad_fine = compute_itcz_latitude(
            T_bl_K[m],
            fine_lat2d,
            spherical_cell_area(fine_lon2d, fine_lat2d, earth_radius_m=R_EARTH_METERS),
        )
        rh = specific_humidity_to_relative_humidity(
            q[m], T_bl_K[m], itcz_rad=itcz_rad_fine, lat2d=fine_lat2d, lon2d=fine_lon2d
        )

        T_bl_K_coarse = layers["boundary_layer"][m] + 273.15  # convert to Kelvin
        rh_coarse = specific_humidity_to_relative_humidity(
            layers["humidity"][m],
            T_bl_K_coarse,
            itcz_rad=itcz_rad,
            lat2d=coarse_lat2d,
            lon2d=coarse_lon2d,
        )

        w_oro_coarse = oro_model_5deg.compute_orographic_vertical_velocity(
            wind_u=layers["wind_u"][m], wind_v=layers["wind_v"][m]
        )
        p_oro_coarse = oro_model_5deg.compute_orographic_precipitation(
            w_oro_coarse, layers["humidity"][m], T_bl_K_coarse, rh_coarse
        )
        p_oro_coarse = np.where(coarse_land_mask, np.maximum(p_oro_coarse, 0.0), 0.0)

        w_oro = oro_model_1deg.compute_orographic_vertical_velocity(wind_u[m], wind_v[m])
        p_oro = oro_model_1deg.compute_orographic_precipitation(
            w_oro,
            q[m],
            T_bl_K[m],
            rh,
        )
        p_oro = np.where(fine_land_mask, np.maximum(p_oro, 0.0), 0.0)

        p_ls = original_precip[m] - p_oro_coarse
        p_ls = np.maximum(p_ls, 0.0)
        p_ls_bilinear = interpolate_field_bilinear(
            p_ls,
            lat_indices,
            lon_indices,
            weights,
        )

        p_ls_bilinear_list.append(p_ls_bilinear)
        p_oro_list.append(p_oro)

    p_oro_list = np.array(p_oro_list)
    p_ls_bilinear_list = np.array(p_ls_bilinear_list)

    for m in range(12):
        precip_1deg[m] = np.maximum(p_oro_list[m] + p_ls_bilinear_list[m], 0.0)

    precip_1deg = np.array(precip_1deg)
    precip_1deg = precip_1deg * np.mean(original_precip) / np.mean(precip_1deg)

    # --- 6. Interpolate humidity (simple bilinear) ---
    humidity_1deg = np.maximum(q, 0.0)

    # --- 7. Derive soil moisture at 1° ---
    if "soil_moisture" in layers:
        sm_interp = _interp_monthly("soil_moisture")
        # Scale by sqrt(P_ratio) for nonlinear response
        coarse_precip_interp = np.zeros((12, nlat_fine, nlon_fine))
        for m in range(12):
            coarse_precip_interp[m] = interpolate_field_bilinear(
                original_precip[m],
                lat_indices,
                lon_indices,
                weights,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            p_ratio = np.where(
                coarse_precip_interp > 1e-15,
                precip_1deg / coarse_precip_interp,
                1.0,
            )
        sm_1deg = np.clip(sm_interp * np.sqrt(p_ratio), 0.0, 0.35)
    else:
        sm_1deg = np.zeros((12, nlat_fine, nlon_fine))

    print("  1° orographic interpolation complete.")
    return {
        "precipitation": precip_1deg,
        "humidity": humidity_1deg,
        "soil_moisture": sm_1deg,
    }
