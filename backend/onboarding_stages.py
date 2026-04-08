"""Onboarding stage metadata for the chat server.

Provides stage-aware context so the LLM can explain what physics
is and isn't active at each stage of the progressive reveal.
"""

from __future__ import annotations

STAGE_NAMES = {
    1: "Radiation Only",
    2: "Atmosphere & Greenhouse",
    3: "Wind & Diffusion",
    4: "Hadley Circulation",
    5: "Full Model",
}

_STAGE_CHAT_CONTEXT: dict[int, dict] = {
    1: {
        "physics": ["solar radiation", "surface albedo", "Stefan-Boltzmann emission"],
        "missing": [
            "atmosphere",
            "greenhouse effect",
            "wind",
            "humidity",
            "clouds",
            "ocean currents",
        ],
    },
    2: {
        "physics": [
            "solar radiation",
            "surface albedo",
            "atmospheric absorption",
            "greenhouse effect",
            "sensible heat exchange",
        ],
        "missing": [
            "lateral heat transport",
            "wind",
            "humidity",
            "clouds",
            "ocean currents",
        ],
    },
    3: {
        "physics": [
            "solar radiation",
            "surface albedo",
            "atmospheric absorption",
            "greenhouse effect",
            "sensible heat exchange",
            "wind (thermal pressure gradients)",
            "atmospheric advection",
            "lateral diffusion (eddy transport)",
            "snow/ice albedo feedback",
            "latent heat exchange",
            "humidity",
            "clouds",
            "orographic effects",
        ],
        "missing": [
            "Hadley circulation",
            "ITCZ",
            "subtropical drying",
            "vertical motion",
            "trade winds",
            "ocean currents",
        ],
    },
    4: {
        "physics": [
            "solar radiation",
            "surface albedo",
            "atmospheric absorption",
            "greenhouse effect",
            "sensible heat exchange",
            "wind",
            "atmospheric advection",
            "lateral diffusion",
            "snow/ice albedo feedback",
            "latent heat exchange",
            "humidity",
            "clouds",
            "orographic effects",
            "Hadley circulation",
            "ITCZ",
            "trade winds",
            "westerlies",
            "vertical motion (subtropical drying, adiabatic heating/cooling)",
        ],
        "missing": ["ocean currents"],
    },
    5: {
        "physics": [
            "solar radiation",
            "surface albedo",
            "atmospheric absorption",
            "greenhouse effect",
            "sensible heat exchange",
            "wind from thermal pressure gradients",
            "Coriolis deflection",
            "atmospheric advection",
            "lateral diffusion (eddy heat transport)",
            "humidity, latent heat exchange, evaporation",
            "Sundqvist clouds and precipitation",
            "snow/ice albedo feedback",
            "orographic effects",
            "Hadley circulation, ITCZ, trade winds, subtropical drying",
            "ocean currents (Stommel-style gyres, Ekman pumping)",
            "vegetation–climate feedback",
        ],
        "missing": [
            "ENSO and other interannual ocean variability",
            "aerosols",
            "dynamic ice sheets and mountain glaciers",
            "river hydrology and detailed soil moisture",
        ],
    },
}


def get_stage_chat_context(stage: int) -> str:
    """Return a string to inject into the LLM system prompt for stage-aware explanations."""
    info = _STAGE_CHAT_CONTEXT.get(stage)
    if not info:
        return ""

    name = STAGE_NAMES.get(stage, f"Stage {stage}")
    lines = [
        f'\n\nIMPORTANT CONTEXT — The user is viewing the model at Stage {stage}: "{name}".',
        f"Physics active: {', '.join(info['physics'])}.",
    ]
    if stage < 5:
        if info["missing"]:
            lines.append(
                f"Physics NOT yet active (will be added in later stages): {', '.join(info['missing'])}."
            )
        lines.append(
            "When explaining observations, frame your answer in terms of what physics IS and IS NOT "
            "active at this stage. For example, if wind is missing, explain that there's no lateral "
            "heat transport yet. Help the user understand what each physics component contributes."
        )
    else:
        if info["missing"]:
            lines.append(f"Not modeled: {', '.join(info['missing'])}.")
        lines.append(
            "If the user asks about a phenomenon that depends on something this model doesn't "
            "include, say so plainly and explain the closest related mechanism that the model does "
            "capture."
        )
    return "\n".join(lines)
