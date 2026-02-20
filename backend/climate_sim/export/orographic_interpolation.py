"""Recompute precipitation at 1° using orographic physics, conserving 5° totals.

This is a visualization refinement: the solver's 5° precipitation is ground truth,
but within each 5° cell we redistribute rain based on 1° terrain, so mountains
get more and valleys get less.
"""

from __future__ import annotations

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
from climate_sim.physics.clouds import compute_clouds_and_precipitation
from climate_sim.physics.orographic_effects import OrographicConfig, OrographicModel


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
    nlat_coarse = len(coarse_lats)
    nlon_coarse = len(coarse_lons)

    # Resolution ratio (should be 5 for 5°→1°)
    lat_ratio = nlat_fine // nlat_coarse
    lon_ratio = nlon_fine // nlon_coarse

    # --- 2. Build bilinear weights (no land/ocean separation for physics fields) ---
    coarse_land_mask = compute_land_mask(coarse_lon2d, coarse_lat2d)
    fine_land_mask = compute_land_mask(fine_lon2d, fine_lat2d)

    lat_indices, lon_indices, weights, _ = _build_bilinear_weights(
        coarse_lats, coarse_lons,
        fine_lat2d, fine_lon2d,
        coarse_land_mask, fine_land_mask,
    )

    # --- 3. Interpolate input fields to 1° ---
    def _interp_monthly(field_name: str) -> np.ndarray:
        coarse = layers[field_name]  # (12, nlat_c, nlon_c)
        result = np.zeros((12, nlat_fine, nlon_fine))
        for m in range(12):
            result[m] = interpolate_field_bilinear(
                coarse[m], lat_indices, lon_indices, weights,
            )
        return result

    T_bl = _interp_monthly("boundary_layer")  # °C
    T_atm = _interp_monthly("atmosphere")     # °C
    q = _interp_monthly("humidity")           # kg/kg
    wind_u = _interp_monthly("wind_u_10m")    # m/s
    wind_v = _interp_monthly("wind_v_10m")    # m/s

    # Vertical velocity (large-scale)
    if "vertical_velocity" in layers:
        w_ls = _interp_monthly("vertical_velocity")
    else:
        w_ls = np.zeros((12, nlat_fine, nlon_fine))

    # Surface temperature for cloud computation
    T_sfc = _interp_monthly("surface")  # °C

    # Convert to Kelvin for physics
    T_bl_K = T_bl + 273.15
    T_atm_K = T_atm + 273.15
    T_sfc_K = T_sfc + 273.15

    # --- 4. Build 1° OrographicModel ---
    print("  Building 1° orographic model...")
    elevation_1deg = compute_cell_elevation(fine_lon2d, fine_lat2d, cache=False)
    elevation_std_1deg, _ = compute_cell_elevation_statistics(
        fine_lon2d, fine_lat2d,
        cache=True,
        cache_name="elevation_statistics_1deg_cache.npz",
    )
    face_stats_1deg = compute_face_elevation_statistics(
        fine_lon2d, fine_lat2d,
        cache=True,
        cache_name="face_elevation_1deg_cache.npz",
    )

    oro_model = OrographicModel(
        lon2d=fine_lon2d,
        lat2d=fine_lat2d,
        elevation=elevation_1deg,
        elevation_std=elevation_std_1deg,
        face_stats=face_stats_1deg,
        config=OrographicConfig(),
        land_mask=fine_land_mask,
    )

    # --- 5. Compute precipitation at 1° for each month ---
    print("  Computing 1° precipitation...")
    ocean_mask = ~fine_land_mask
    precip_1deg = np.zeros((12, nlat_fine, nlon_fine))
    original_precip = layers["precipitation"]  # (12, nlat_c, nlon_c) in kg/m²/s

    for m in range(12):
        # Orographic vertical velocity and precipitation (land only)
        w_oro = oro_model.compute_orographic_vertical_velocity(wind_u[m], wind_v[m])
        p_oro = oro_model.compute_orographic_precipitation(
            w_oro, q[m], T_bl_K[m],
        )
        p_oro = np.where(fine_land_mask, p_oro, 0.0)

        # Large-scale cloud precipitation
        # Compute RH = q / q_sat using Magnus formula at 1013.25 hPa
        T_C = T_bl_K[m] - 273.15
        e_sat = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
        q_sat = 0.622 * e_sat / (1013.25 - 0.378 * e_sat)
        rh = np.clip(q[m] / np.maximum(q_sat, 1e-10), 0.0, 1.0)
        cloud_out = compute_clouds_and_precipitation(
            T_bl_K[m], T_atm_K[m], q[m], rh, w_ls[m],
            T_surface_K=T_sfc_K[m], ocean_mask=ocean_mask,
        )
        p_ls = (cloud_out.convective_precip
                + cloud_out.stratiform_precip
                + cloud_out.marine_sc_precip)

        p_total = p_ls + p_oro
        p_total = np.maximum(p_total, 0.0)

        # --- Conservation rescale: match original 5° cell means ---
        for i in range(nlat_coarse):
            fi0 = i * lat_ratio
            fi1 = fi0 + lat_ratio
            for j in range(nlon_coarse):
                fj0 = j * lon_ratio
                fj1 = fj0 + lon_ratio

                block = p_total[fi0:fi1, fj0:fj1]
                block_mean = block.mean()
                target = original_precip[m, i, j]

                if block_mean > 1e-15 and target > 0:
                    ratio = target / block_mean
                    precip_1deg[m, fi0:fi1, fj0:fj1] = block * ratio
                else:
                    # Uniform fallback (desert or zero precip)
                    precip_1deg[m, fi0:fi1, fj0:fj1] = target

    # --- 6. Interpolate humidity (simple bilinear) ---
    humidity_1deg = np.maximum(q, 0.0)

    # --- 7. Derive soil moisture at 1° ---
    if "soil_moisture" in layers:
        sm_interp = _interp_monthly("soil_moisture")
        # Scale by sqrt(P_ratio) for nonlinear response
        coarse_precip_interp = np.zeros((12, nlat_fine, nlon_fine))
        for m in range(12):
            coarse_precip_interp[m] = interpolate_field_bilinear(
                original_precip[m], lat_indices, lon_indices, weights,
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
