"""OpenAI function-calling tool definitions."""

SAMPLE_CLIMATE_TOOL = {
    "type": "function",
    "function": {
        "name": "sample_climate",
        "description": (
            "Look up a single climate variable at a specific latitude, longitude, "
            "and month. Use this to investigate climate conditions before explaining them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "The climate variable to sample (e.g. 'temperature_2m', 'precipitation', 'wind_speed_10m').",
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
            "required": ["field", "lat", "lon", "month"],
        },
    },
}

SAMPLE_OBSERVATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "sample_observations",
        "description": (
            "Look up a real-world observation from NOAA 1981-2010 climatology at a "
            "specific latitude, longitude, and month. Use this to ground explanations "
            "in observed data and compare with simulation output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
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
                    "description": (
                        "The observation to look up. "
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
            "required": ["field", "lat", "lon", "month"],
        },
    },
}

TOOLS = [SAMPLE_CLIMATE_TOOL, SAMPLE_OBSERVATIONS_TOOL]
