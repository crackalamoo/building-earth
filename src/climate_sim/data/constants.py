"""Shared physical constants used throughout the climate simulator."""

R_EARTH_METERS = 6.371e6
ATMOSPHERE_MASS = 5.1480e18  # kg
EARTH_SURFACE_AREA_M2 = 4.0 * 3.141592653589793 * R_EARTH_METERS**2
GAS_CONSTANT_J_KG_K = 287.0  # J kg-1 K-1
HEAT_CAPACITY_AIR_J_KG_K = 1005.0  # J kg-1 K-1 (dry air, ~constant cp)
OCEAN_CLOUD_COVER_BOOST = 1.3
