"""OpenAI function-calling tool definitions."""

SAMPLE_CLIMATE_TOOL = {
    "type": "function",
    "function": {
        "name": "sample_climate",
        "description": (
            "Look up one or more simulated climate variables at a specific latitude, "
            "longitude, and month. Request multiple fields at once to reduce round-trips."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "temperature_2m", "precipitation", "humidity",
                            "wind_speed_10m", "cloud_fraction", "surface_pressure",
                            "elevation",
                            "surface", "boundary_layer", "atmosphere",
                            "wind_u_10m", "wind_v_10m",
                            "wind_u", "wind_v", "wind_speed",
                            "wind_u_geostrophic", "wind_v_geostrophic",
                            "wind_speed_geostrophic",
                            "cloud_high", "cloud_low", "cloud_convective",
                            "stratiform_cloud_frac", "marine_sc_cloud_frac",
                            "ocean_u", "ocean_v", "w_ekman_pumping",
                            "vertical_velocity",
                            "albedo", "soil_moisture", "vegetation_fraction",
                            "relative_humidity", "saturation_humidity",
                            "wind_direction_10m", "dew_point", "lapse_rate",
                        ],
                    },
                    "description": "Climate variables to sample.",
                },
                "lat": {
                    "type": "number",
                    "description": "Latitude in degrees (-90 to 90).",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude in degrees (-180 to 180).",
                },
                "month": {
                    "type": "integer",
                    "description": "Month index (0 = January, 11 = December).",
                },
            },
            "required": ["fields", "lat", "lon", "month"],
        },
    },
}

SAMPLE_OBSERVATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "sample_observations",
        "description": (
            "Look up one or more real-world observations from NOAA 1981-2010 climatology "
            "at a specific latitude, longitude, and month. Use this to ground explanations "
            "in observed data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "land_temperature",
                            "sst",
                            "humidity",
                            "precipitation",
                            "pressure",
                            "wind_u",
                            "wind_v",
                        ],
                    },
                    "description": (
                        "Observations to look up. "
                        "'land_temperature' = station data (land only, null over ocean). "
                        "'sst' = satellite SST (ocean only, null over land/ice)."
                    ),
                },
                "lat": {
                    "type": "number",
                    "description": "Latitude in degrees (-90 to 90).",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude in degrees (-180 to 180).",
                },
                "month": {
                    "type": "integer",
                    "description": "Month index (0 = January, 11 = December).",
                },
            },
            "required": ["fields", "lat", "lon", "month"],
        },
    },
}

CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a math expression over climate data. "
            "Use these aliases as variable names — they resolve automatically at the given location. "
            "Aliases: T (2m air temp °C), T_s (surface °C), T_bl (boundary layer °C), "
            "T_atm (free atmosphere °C), q (specific humidity kg/kg), q_sat (saturation humidity kg/kg), "
            "RH (relative humidity %), Td (dew point °C), P (surface pressure Pa), "
            "u (10m E/W wind m/s), v (10m N/S wind m/s), ws (10m wind speed m/s), "
            "wind_dir (wind direction °), u_bl, v_bl, ws_bl (BL winds), "
            "u_g, v_g, ws_g (geostrophic winds), u_ocean, v_ocean (ocean currents), "
            "z (elevation m), precip (precipitation kg/m²/s), clouds (cloud fraction 0-1), "
            "lapse (lapse rate °C/km). "
            "IMPORTANT: Values are in raw SI units which differ from sample_climate display units: "
            "q in kg/kg (not g/kg), P in Pa (not mb), wind in m/s (not km/h). "
            "Supports: +, -, *, /, **, sqrt, exp, log, sin, cos, atan2, abs, min, max, pi."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression using field names/aliases and math functions.",
                },
                "lat": {
                    "type": "number",
                    "description": "Latitude in degrees (-90 to 90).",
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude in degrees (-180 to 180).",
                },
                "month": {
                    "type": "integer",
                    "description": "Month index (0 = January, 11 = December).",
                },
                "unit": {
                    "type": "string",
                    "description": "Optional unit label for the result (e.g. '°C', '%', 'm/s').",
                },
            },
            "required": ["expression", "lat", "lon", "month"],
        },
    },
}

TOOLS = [SAMPLE_CLIMATE_TOOL, SAMPLE_OBSERVATIONS_TOOL, CALCULATE_TOOL]
