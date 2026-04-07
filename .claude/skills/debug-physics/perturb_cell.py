"""Perturbation analysis: decompose restoring tendency at a single cell or latitude band.

Usage:
    DATA_DIR=data uv run python .claude/skills/debug-physics/perturb_cell.py --lat 67.5 --lon 137.5 --target-bl -46 --month 0
    DATA_DIR=data uv run python .claude/skills/debug-physics/perturb_cell.py --lat-band 50 70 --target-bl obs --month 0
"""

import argparse
import sys
import numpy as np

sys.path.insert(0, "backend")

from climate_sim.core.operators import build_model_operators
from climate_sim.core.rhs_builder import create_rhs_functions, RhsBuildInputs
from climate_sim.core.solver import ModelState
from climate_sim.runtime.config import ModelConfig
from climate_sim.data.constants import (
    BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
    ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K,
    R_EARTH_METERS,
)
from climate_sim.core.math_core import spherical_cell_area
from climate_sim.physics.atmosphere.hadley import compute_itcz_latitude
from climate_sim.physics.vertical_motion import compute_bl_atm_mixing_tendencies
from climate_sim.data.landmask import compute_land_mask
import xarray as xr


def build_state(d, month):
    T = np.array([
        d["surface"][month] + 273.15,
        d["boundary_layer"][month] + 273.15,
        d["atmosphere"][month] + 273.15,
    ])
    state = ModelState(
        temperature=T,
        albedo_field=d["albedo"][month],
        wind_field=(
            d["wind_u_geostrophic"][month],
            d["wind_v_geostrophic"][month],
            d["wind_speed_geostrophic"][month],
        ),
        humidity_field=d["humidity"][month],
        precipitation_field=d["precipitation"][month],
        ocean_current_field=(d["ocean_u"][month], d["ocean_v"][month]),
        ocean_ekman_pumping=d["w_ekman_pumping"][month],
        soil_moisture=d["soil_moisture"][month],
    )
    state.boundary_layer_wind_field = (
        d["wind_u"][month],
        d["wind_v"][month],
        d["wind_speed"][month],
    )
    return state


def main():
    parser = argparse.ArgumentParser(description="Perturbation analysis for physics debugging")
    parser.add_argument("--lat", type=float, help="Latitude of cell")
    parser.add_argument("--lon", type=float, help="Longitude of cell (degrees E, use 360-x for W)")
    parser.add_argument("--lat-band", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Perturb all land cells in latitude band")
    parser.add_argument("--target-t2m", type=str, required=True,
                        help='Target T_2m in °C, or "obs". Inverted to T_bl via lapse rate.')
    parser.add_argument("--month", type=int, default=0, help="Month index (0=Jan)")
    sfc_group = parser.add_mutually_exclusive_group()
    sfc_group.add_argument("--also-perturb-sfc", action="store_true",
                           help="Also perturb T_sfc by the same delta as T_bl")
    sfc_group.add_argument("--target-sfc", type=str, default=None,
                           help='Separate T_sfc target in °C, or "obs". Independent of T_bl perturbation.')
    parser.add_argument("--resolution", type=float, default=5.0)
    args = parser.parse_args()

    config = ModelConfig()
    ops = build_model_operators(args.resolution, config)
    lon2d, lat2d = ops.lon2d, ops.lat2d
    lats = lat2d[:, 0]
    lons = lon2d[0, :]
    land_mask = compute_land_mask(lon2d, lat2d)

    rhs_inputs = RhsBuildInputs(
        radiation_config=ops.radiation_config,
        diffusion_operator=ops.diffusion_operator,
        heat_capacity_field=ops.heat_capacity_field,
        land_mask=ops.land_mask,
        lat2d=lat2d,
        lon2d=lon2d,
        wind_model=ops.wind_model,
        sensible_heat_cfg=ops.sensible_heat_cfg,
        latent_heat_cfg=ops.latent_heat_cfg,
        advection_operator=ops.advection_operator,
        vertical_motion_cfg=ops.vertical_motion_cfg,
        topographic_elevation=ops.topographic_elevation,
        orographic_model=ops.orographic_model,
        ocean_advection_cfg=ops.ocean_advection_cfg,
        amoc_velocity=ops.amoc_velocity,
        roughness_length=ops.roughness_length,
    )
    rhs_fn, _ = create_rhs_functions(rhs_inputs)

    d = np.load("data/main.npz")
    m = args.month
    state = build_state(d, m)
    T = state.temperature.copy()
    insolation = ops.monthly_insolation[m]
    cell_areas = spherical_cell_area(lon2d, lat2d, earth_radius_m=R_EARTH_METERS)
    itcz_rad = compute_itcz_latitude(T[1], lat2d, cell_areas)

    # Determine which cells to perturb
    if args.lat_band:
        mask = (lats >= args.lat_band[0]) & (lats <= args.lat_band[1])
        cells = [
            (i, j)
            for i in np.where(mask)[0]
            for j in range(len(lons))
            if land_mask[i, j]
        ]
        label = f"land cells {args.lat_band[0]:.0f}-{args.lat_band[1]:.0f}N"
    elif args.lat is not None and args.lon is not None:
        li = np.argmin(np.abs(lats - args.lat))
        lj = np.argmin(np.abs(lons - args.lon))
        cells = [(li, lj)]
        label = f"({lats[li]:.1f}N, {lons[lj]:.1f}E)"
    else:
        parser.error("Provide either --lat/--lon or --lat-band")

    # Get target T_2m values
    if args.target_t2m == "obs":
        obs = xr.open_dataset("data/processed/ref_climatology_1deg_1981-2010.nc")
        t_obs = obs["t_surface_clim"].values
        lats_obs, lons_obs = obs["lat"].values, obs["lon"].values
    target_t2m = None if args.target_t2m == "obs" else float(args.target_t2m)

    # Invert T_2m → T_bl using the model's own formula:
    # T_2m = T_bl + lapse_rate × (BL_midpoint - 2m) - lapse_rate × (elevation - 2m)
    #      = T_bl + lapse_rate × (BL_midpoint - elevation)
    # So: T_bl = T_2m - lapse_rate × (BL_midpoint - elevation)
    from climate_sim.data.constants import STANDARD_LAPSE_RATE_K_PER_M, BOUNDARY_LAYER_HEIGHT_M
    bl_midpoint = BOUNDARY_LAYER_HEIGHT_M / 2.0  # 375m
    elev = ops.topographic_elevation

    def t2m_to_tbl(t2m_K: float, cell_elev: float) -> float:
        """Invert model's T_2m formula to get T_bl."""
        correction = STANDARD_LAPSE_RATE_K_PER_M * (bl_midpoint - cell_elev)
        return t2m_K - correction

    # Parse surface target if provided
    target_sfc_val = None
    if args.target_sfc is not None and args.target_sfc != "obs":
        target_sfc_val = float(args.target_sfc)

    # Build perturbed T field
    T_pert = T.copy()
    n_perturbed = 0
    for i, j in cells:
        cell_elev = elev[i, j] if elev is not None else 0.0
        if target_t2m is not None:
            T_bl_target = t2m_to_tbl(target_t2m + 273.15, cell_elev)
        else:
            oi = np.argmin(np.abs(lats_obs - lats[i]))
            oj = np.argmin(np.abs(lons_obs - lons[j]))
            obs_v = t_obs[m, oi, oj]
            if np.isnan(obs_v):
                continue
            T_bl_target = t2m_to_tbl(obs_v + 273.15, cell_elev)

        delta = T_bl_target - T[1, i, j]
        T_pert[1, i, j] = T_bl_target

        # Surface perturbation
        if args.also_perturb_sfc:
            T_pert[0, i, j] = T[0, i, j] + delta
        elif args.target_sfc is not None:
            if args.target_sfc == "obs":
                oi = np.argmin(np.abs(lats_obs - lats[i]))
                oj = np.argmin(np.abs(lons_obs - lons[j]))
                obs_sfc = t_obs[m, oi, oj]
                if not np.isnan(obs_sfc):
                    # Use obs T_2m as proxy for skin T (rough but reasonable for land)
                    T_pert[0, i, j] = obs_sfc + 273.15
            else:
                T_pert[0, i, j] = target_sfc_val + 273.15

        n_perturbed += 1

    # Full RHS at equilibrium and perturbed
    rhs_eq = rhs_fn(state, insolation, itcz_rad)

    state_pert = build_state(d, m)
    state_pert.temperature = T_pert
    rhs_pert = rhs_fn(state_pert, insolation, itcz_rad)

    # Individual terms
    tau_mix = ops.vertical_motion_cfg.tau_bl_atm_mixing_s
    T_sfc_eq = T[0]
    T_sfc_pt = T_pert[0]  # same as T_sfc_eq unless --also-perturb-sfc
    T_bl_eq = T[1]
    T_bl_pt = T_pert[1]
    T_atm_eq = T[2]

    wind_u_bl, wind_v_bl = d["wind_u"][m], d["wind_v"][m]
    wind_speed_bl = d["wind_speed"][m]
    q_field = d["humidity"][m]
    precip_field = d["precipitation"][m]
    sm_field = d["soil_moisture"][m]

    zeros = np.zeros_like(T_bl_eq)

    # --- Diffusion ---
    diff_op = ops.diffusion_operator.boundary_layer
    diff_eq = diff_op.tendency(T_bl_eq) if diff_op else zeros
    diff_pt = diff_op.tendency(T_bl_pt) if diff_op else zeros

    # --- Advection ---
    adv_op = ops.advection_operator
    adv_eq = adv_op.tendency(T_bl_eq, wind_u_bl, wind_v_bl) if adv_op else zeros
    adv_pt = adv_op.tendency(T_bl_pt, wind_u_bl, wind_v_bl) if adv_op else zeros

    # --- BL-atm mixing ---
    mix_eq, _ = compute_bl_atm_mixing_tendencies(
        T_bl_eq, T_atm_eq,
        C_bl=BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
        C_atm=ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K, tau_s=tau_mix,
    )
    mix_pt, _ = compute_bl_atm_mixing_tendencies(
        T_bl_pt, T_atm_eq,
        C_bl=BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K,
        C_atm=ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K, tau_s=tau_mix,
    )

    # --- Sensible heat ---
    from climate_sim.physics.sensible_heat_exchange import SensibleHeatExchangeModel

    sh_model = SensibleHeatExchangeModel(
        land_mask=ops.land_mask,
        surface_heat_capacity_J_m2_K=ops.heat_capacity_field,
        atmosphere_heat_capacity_J_m2_K=ops.radiation_config.atmosphere_heat_capacity,
        wind_model=ops.wind_model,
        config=ops.sensible_heat_cfg,
        boundary_layer_heat_capacity_J_m2_K=ops.radiation_config.boundary_layer_heat_capacity,
        topographic_elevation=ops.topographic_elevation,
    )
    _, sh_bl_eq, _ = sh_model.compute_tendencies(
        surface_temperature_K=T_sfc_eq, atmosphere_temperature_K=T_atm_eq,
        wind_speed_reference_m_s=wind_speed_bl, itcz_rad=itcz_rad,
        boundary_layer_temperature_K=T_bl_eq,
    )
    _, sh_bl_pt, _ = sh_model.compute_tendencies(
        surface_temperature_K=T_sfc_pt, atmosphere_temperature_K=T_atm_eq,
        wind_speed_reference_m_s=wind_speed_bl, itcz_rad=itcz_rad,
        boundary_layer_temperature_K=T_bl_pt,
    )

    # --- Latent heat ---
    from climate_sim.physics.latent_heat_exchange import LatentHeatExchangeModel

    lh_model = LatentHeatExchangeModel(
        land_mask=ops.land_mask,
        surface_heat_capacity_J_m2_K=ops.heat_capacity_field,
        atmosphere_heat_capacity_J_m2_K=ops.radiation_config.atmosphere_heat_capacity,
        wind_model=ops.wind_model,
        config=ops.latent_heat_cfg,
        boundary_layer_heat_capacity_J_m2_K=ops.radiation_config.boundary_layer_heat_capacity,
    )
    _, lh_bl_eq, _, _ = lh_model.compute_tendencies(
        surface_temperature_K=T_sfc_eq, atmosphere_temperature_K=T_atm_eq,
        humidity_q=q_field, wind_speed_reference_m_s=wind_speed_bl,
        itcz_rad=itcz_rad, boundary_layer_temperature_K=T_bl_eq,
        precipitation_rate=precip_field, soil_moisture=sm_field,
    )
    _, lh_bl_pt, _, _ = lh_model.compute_tendencies(
        surface_temperature_K=T_sfc_eq, atmosphere_temperature_K=T_atm_eq,
        humidity_q=q_field, wind_speed_reference_m_s=wind_speed_bl,
        itcz_rad=itcz_rad, boundary_layer_temperature_K=T_bl_pt,
        precipitation_rate=precip_field, soil_moisture=sm_field,
    )

    # --- Vertical motion (divergence-driven) ---
    from climate_sim.core.math_core import compute_divergence
    from climate_sim.physics.vertical_motion import compute_vertical_motion_tendencies

    divergence = compute_divergence(wind_u_bl, wind_v_bl, lat2d, lon2d)
    vm_bl_eq, _ = compute_vertical_motion_tendencies(divergence, T_bl_eq, T_atm_eq)
    vm_bl_pt, _ = compute_vertical_motion_tendencies(divergence, T_bl_pt, T_atm_eq)

    # --- Radiation (residual) ---
    known_eq = diff_eq + adv_eq + mix_eq + sh_bl_eq + lh_bl_eq + vm_bl_eq
    known_pt = diff_pt + adv_pt + mix_pt + sh_bl_pt + lh_bl_pt + vm_bl_pt
    rad_eq = rhs_eq[1] - known_eq
    rad_pt = rhs_pert[1] - known_pt

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    print(f"\n=== Perturbation analysis at {label}, {month_names[m]} ===")
    print(f"  Perturbed {n_perturbed} cells")

    terms = [
        ("Total RHS", rhs_eq[1], rhs_pert[1]),
        ("Diffusion*", diff_eq, diff_pt),
        ("Advection*", adv_eq, adv_pt),
        ("BL-atm mixing", mix_eq, mix_pt),
        ("Sensible heat", sh_bl_eq, sh_bl_pt),
        ("Latent heat", lh_bl_eq, lh_bl_pt),
        ("Vertical motion", vm_bl_eq, vm_bl_pt),
        ("Radiation (resid)", rad_eq, rad_pt),
    ]

    if len(cells) == 1:
        i, j = cells[0]
        bias = T[1, i, j] - T_pert[1, i, j]
        print(
            f"  T_bl: sim={T[1,i,j]-273.15:.1f}°C → target={T_pert[1,i,j]-273.15:.1f}°C"
            f" (Δ={-bias:.1f}°C)"
        )
        print(f"\n  {'Term':<20s} {'Equil':>10s} {'Perturbed':>10s} {'Change':>10s}  (K/day)")
        for name, eq_f, pt_f in terms:
            e = eq_f[i, j] * 86400
            p = pt_f[i, j] * 86400
            print(f"  {name:<20s} {e:+10.2f} {p:+10.2f} {p - e:+10.2f}")
    else:
        avg_bias = np.mean([T[1, i, j] - T_pert[1, i, j] for i, j in cells])
        print(f"  Mean T_bl perturbation: {-avg_bias:.1f}°C")
        print(f"\n  {'Term':<20s} {'Equil':>10s} {'Perturbed':>10s} {'Change':>10s}  (K/day)")
        for name, eq_f, pt_f in terms:
            e_vals = [eq_f[i, j] * 86400 for i, j in cells]
            p_vals = [pt_f[i, j] * 86400 for i, j in cells]
            e = np.mean(e_vals)
            p = np.mean(p_vals)
            print(f"  {name:<20s} {e:+10.2f} {p:+10.2f} {p - e:+10.2f}")


if __name__ == "__main__":
    main()
