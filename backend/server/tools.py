"""OpenAI function-calling tool definitions."""

SAMPLE_CLIMATE_TOOL = {
    "type": "function",
    "name": "sample_climate",
    "description": (
        "Look up one or more simulated climate variables at a specific latitude, "
        "longitude, and month. Request multiple fields at once to reduce round-trips."
    ),
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "temperature_2m",
                        "precipitation",
                        "humidity",
                        "wind_speed_10m",
                        "cloud_fraction",
                        "surface_pressure",
                        "elevation",
                        "surface",
                        "boundary_layer",
                        "atmosphere",
                        "wind_u_10m",
                        "wind_v_10m",
                        "wind_u",
                        "wind_v",
                        "wind_speed",
                        "wind_u_geostrophic",
                        "wind_v_geostrophic",
                        "wind_speed_geostrophic",
                        "cloud_high",
                        "cloud_low",
                        "cloud_convective",
                        "stratiform_cloud_frac",
                        "marine_sc_cloud_frac",
                        "ocean_u",
                        "ocean_v",
                        "w_ekman_pumping",
                        "vertical_velocity",
                        "albedo",
                        "soil_moisture",
                        "vegetation_fraction",
                        "relative_humidity",
                        "saturation_humidity",
                        "wind_direction_10m",
                        "dew_point",
                        "lapse_rate",
                        "sst",
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
                "description": "Month index (0=Jan, 11=Dec). Omit for annual mean.",
            },
        },
        "required": ["fields", "lat", "lon"],
    },
}

SAMPLE_OBSERVATIONS_TOOL = {
    "type": "function",
    "name": "sample_observations",
    "description": (
        "Look up one or more real-world observations from NOAA 1981-2010 climatology "
        "at a specific latitude, longitude, and month. Use this to ground explanations "
        "in observed data."
    ),
    "strict": False,
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
                "description": "Month index (0=Jan, 11=Dec). Omit for annual mean.",
            },
        },
        "required": ["fields", "lat", "lon"],
    },
}

CALCULATE_TOOL = {
    "type": "function",
    "name": "calculate",
    "description": (
        "Evaluate a math expression over climate data at a single cell, or "
        "with spatial / temporal reductions to compare a location to broader regions.\n\n"
        "Field aliases (raw SI units): T (2m air temp °C), T_s (surface °C), "
        "T_bl, T_atm, q (kg/kg), q_sat, RH (%), Td (°C), P (Pa), "
        "u, v, ws (10m winds m/s), u_bl, v_bl, ws_bl (BL winds), u_g, v_g, ws_g (geostrophic winds), "
        "u_ocean, v_ocean, z (m), precip (kg/m²/s), clouds (0-1), lapse (°C/km), "
        "wind_dir (°). Math: +-*/, **, sqrt, exp, log, sin/cos/tan, abs, pi, e.\n\n"
        "Reductions:\n"
        "  global_mean(expr) — cos-lat weighted global mean\n"
        "  zonal_mean(expr) — mean over all lons at the requested lat\n"
        "  lat_band_mean(expr, lat_min, lat_max) — area-weighted band mean\n"
        "  box_mean(expr, lat_min, lon_min, lat_max, lon_max) — area-weighted box mean. "
        "lon_min > lon_max wraps the antimeridian.\n\n"
        "Reductions compose: `T - zonal_mean(T)` is the local zonal anomaly.\n\n"
        "Parameters: bare field references need lat+lon. zonal_mean needs lat. "
        "Other reductions supply their own geography. month is always optional — "
        "omit for an annual mean over all 12 months."
    ),
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression using field names/aliases, math functions, and reductions.",
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
                "description": "Month index (0=Jan, 11=Dec). Omit for annual mean.",
            },
            "unit": {
                "type": "string",
                "description": "Optional unit label for the result.",
            },
        },
        "required": ["expression"],
    },
}

TOOLS = [SAMPLE_CLIMATE_TOOL, SAMPLE_OBSERVATIONS_TOOL, CALCULATE_TOOL]


_PLACEHOLDER_LABELS: dict[str, str] = {
    SAMPLE_CLIMATE_TOOL["name"]: "Looking up climate…",
    SAMPLE_OBSERVATIONS_TOOL["name"]: "Looking up observations…",
    CALCULATE_TOOL["name"]: "Calculating…",
}


def tool_placeholder_label(tool_name: str) -> str:
    """Friendly placeholder shown the moment a tool call's name appears in
    the OpenAI stream, before its arguments have finished streaming."""
    return _PLACEHOLDER_LABELS.get(tool_name, "Looking up…")
