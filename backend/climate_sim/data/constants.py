"""Shared physical constants used throughout the climate simulator."""

# Fundamental constants
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8  # W m-2 K-4
GRAVITY_M_S2 = 9.81  # m/s²

R_EARTH_METERS = 6.371e6
GAS_CONSTANT_J_KG_K = 287.0  # J kg-1 K-1
HEAT_CAPACITY_AIR_J_KG_K = 1004.0  # J kg-1 K-1 (dry air, ~constant cp)
HEAT_CAPACITY_AIR_J_M2_K = 1.02e7  # J m-2 K-1
SHORTWAVE_ABSORPTANCE_ATMOSPHERE = 0.2 - 0.025
STANDARD_LAPSE_RATE_K_PER_M = 6.5 / 1000.0

# Environmental lapse rate for the T2m elevation correction.
# The observed temperature decrease between surface stations at different
# elevations (~3 °C/km) is much gentler than the free-air rate (6.5 °C/km)
# because mountain surfaces are warmed by solar heating, reducing the
# effective lapse. The free-air rate is correct for atmospheric layer
# physics (radiation, clouds, vertical profiles), but the T2m diagnostic
# should use the environmental rate when adjusting for surface elevation.
ENVIRONMENTAL_LAPSE_RATE_K_PER_M = 3.5 / 1000.0

# Boundary layer constants
# BL emissivity is now humidity-dependent in radiation.py
# This constant is used as fallback when humidity is not available
BOUNDARY_LAYER_EMISSIVITY = 0.50  # Fallback: moderate humidity
BOUNDARY_LAYER_HEIGHT_M = 1000.0
BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K = (
    1.2 * HEAT_CAPACITY_AIR_J_KG_K * BOUNDARY_LAYER_HEIGHT_M
)  # density * cp * height ≈ 9.05e5 J m-2 K-1

# Free atmosphere layer constants (when boundary layer enabled)
ATMOSPHERE_LAYER_HEIGHT_M = 7500.0
ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K = (
    1.2 * HEAT_CAPACITY_AIR_J_KG_K * ATMOSPHERE_LAYER_HEIGHT_M
)  # density * cp * height ≈ 7.24e6 J m-2 K-1
# Midpoint of the atmosphere layer (where T_atm represents)
ATMOSPHERE_LAYER_MIDPOINT_M = BOUNDARY_LAYER_HEIGHT_M + ATMOSPHERE_LAYER_HEIGHT_M / 2.0  # ~4750 m

# Thermodynamic constants for water vapor
LATENT_HEAT_VAPORIZATION_J_KG = 2.5e6  # J/kg - latent heat of vaporization
GAS_CONSTANT_WATER_VAPOR_J_KG_K = 461.0  # J/(kg·K) - gas constant for water vapor
