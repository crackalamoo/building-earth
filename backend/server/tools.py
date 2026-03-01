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

TOOLS = [SAMPLE_CLIMATE_TOOL]
