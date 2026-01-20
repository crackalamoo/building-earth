"""Shared physical constants used throughout the climate simulator."""

R_EARTH_METERS = 6.371e6
ATMOSPHERE_MASS = 5.1480e18  # kg
EARTH_SURFACE_AREA_M2 = 4.0 * 3.141592653589793 * R_EARTH_METERS**2
GAS_CONSTANT_J_KG_K = 287.0  # J kg-1 K-1
HEAT_CAPACITY_AIR_J_KG_K = 1004.0  # J kg-1 K-1 (dry air, ~constant cp)
HEAT_CAPACITY_AIR_J_M2_K = 1.02e7  # J m-2 K-1
SHORTWAVE_ABSORPTANCE_ATMOSPHERE = 0.2 - 0.025
STANDARD_LAPSE_RATE_K_PER_M = 6.5 / 1000.0

# Boundary layer constants
BOUNDARY_LAYER_EMISSIVITY = 0.24
BOUNDARY_LAYER_HEIGHT_M = 1000.0
BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K = 1.2 * HEAT_CAPACITY_AIR_J_KG_K * BOUNDARY_LAYER_HEIGHT_M  # density * cp * height ≈ 9.05e5 J m-2 K-1

# Free atmosphere layer constants (when boundary layer enabled)
ATMOSPHERE_LAYER_HEIGHT_M = 7500.0
ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K = 1.2 * HEAT_CAPACITY_AIR_J_KG_K * ATMOSPHERE_LAYER_HEIGHT_M  # density * cp * height ≈ 7.24e6 J m-2 K-1

BOUNDARY_ATMOSPHERE_EQUILIBRATION_DAYS = 7

# Thermodynamic constants for water vapor
LATENT_HEAT_VAPORIZATION_J_KG = 2.5e6  # J/kg - latent heat of vaporization
GAS_CONSTANT_WATER_VAPOR_J_KG_K = 461.0  # J/(kg·K) - gas constant for water vapor
