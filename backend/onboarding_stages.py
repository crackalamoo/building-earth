"""Onboarding stage metadata for the chat server.

Provides stage-aware context so the LLM can explain what physics
is and isn't active at each stage of the progressive reveal.
"""

from __future__ import annotations

STAGE_NAMES = {
    1: "Radiation Only",
    2: "Atmosphere & Greenhouse",
    3: "Diffusion & Wind",
    4: "Full Model",
}

_STAGE_CHAT_CONTEXT: dict[int, dict] = {
    1: {
        "physics": ["solar radiation", "surface albedo", "Stefan-Boltzmann emission"],
        "missing": [
            "atmosphere", "greenhouse effect", "wind", "humidity",
            "clouds", "ocean currents",
        ],
        "description": (
            "Bare rock equilibrium. Energy in = energy out at each grid cell. "
            "No atmospheric absorption or lateral transport."
        ),
    },
    2: {
        "physics": [
            "solar radiation", "surface albedo", "atmospheric absorption",
            "greenhouse effect", "sensible heat exchange", "latent heat exchange",
        ],
        "missing": [
            "wind", "lateral diffusion", "humidity transport",
            "clouds", "ocean currents",
        ],
        "description": (
            "Atmosphere traps outgoing LW radiation. Surface warms to push energy "
            "through the blanket. No lateral mixing."
        ),
    },
    3: {
        "physics": [
            "solar radiation", "surface albedo", "atmospheric absorption",
            "greenhouse effect", "sensible heat exchange", "latent heat exchange",
            "wind", "lateral diffusion", "atmospheric advection",
            "snow/ice albedo", "orographic effects", "vertical motion",
        ],
        "missing": ["ocean currents"],
        "description": (
            "Full atmospheric dynamics with wind-driven circulation and eddy diffusion. "
            "No ocean heat transport — without the Gulf Stream and other currents, "
            "the temperature field is more zonally symmetric."
        ),
    },
    4: {
        "physics": ["all"],
        "missing": [],
        "description": (
            "Complete energy-balance model with moisture cycle, clouds, snow/ice, "
            "ocean currents, and vegetation feedback."
        ),
    },
}


def get_stage_chat_context(stage: int) -> str:
    """Return a string to inject into the LLM system prompt for stage-aware explanations."""
    info = _STAGE_CHAT_CONTEXT.get(stage)
    if not info:
        return ""

    name = STAGE_NAMES.get(stage, f"Stage {stage}")
    lines = [
        f"\n\nIMPORTANT CONTEXT — The user is viewing the model at Stage {stage}: \"{name}\".",
        f"Physics active: {', '.join(info['physics'])}.",
    ]
    if info["missing"]:
        lines.append(f"Physics NOT yet active (will be added in later stages): {', '.join(info['missing'])}.")
    lines.append(f"Description: {info['description']}")
    lines.append(
        "When explaining observations, frame your answer in terms of what physics IS and IS NOT "
        "active at this stage. For example, if wind is missing, explain that there's no lateral "
        "heat transport yet. Help the user understand what each physics component contributes."
    )
    return "\n".join(lines)
