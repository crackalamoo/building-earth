"""Shared physical constants used throughout the climate simulator."""

R_EARTH_METERS = 6.371e6
ATMOSPHERE_MASS = 5.1480e18  # kg
EARTH_SURFACE_AREA_M2 = 4.0 * 3.141592653589793 * R_EARTH_METERS**2
GAS_CONSTANT_J_KG_K = 287.0  # J kg-1 K-1
HEAT_CAPACITY_AIR_J_KG_K = 1005.0  # J kg-1 K-1 (dry air, ~constant cp)
HEAT_CAPACITY_AIR_J_M2_K = 1.0e7  # J m-2 K-1, ~2-3 km troposphere column (legacy, used when boundary layer disabled)

# Boundary layer constants
BOUNDARY_LAYER_EMISSIVITY = 0.24
BOUNDARY_LAYER_HEIGHT_M = 750.0
BOUNDARY_LAYER_HEAT_CAPACITY_J_M2_K = 1.2 * 1005.0 * 750.0  # density * cp * height ≈ 9.05e5 J m-2 K-1

# Free atmosphere layer constants (when boundary layer enabled)
ATMOSPHERE_LAYER_HEIGHT_M = 6000.0
ATMOSPHERE_LAYER_HEAT_CAPACITY_J_M2_K = 1.2 * 1005.0 * 6000.0  # density * cp * height ≈ 7.24e6 J m-2 K-1

BOUNDARY_ATMOSPHERE_EQUILIBRATION_DAYS = 7