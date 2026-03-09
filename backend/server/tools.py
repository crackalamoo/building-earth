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

TOOLS = [SAMPLE_CLIMATE_TOOL, SAMPLE_OBSERVATIONS_TOOL]
