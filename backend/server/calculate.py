"""Lightweight expression evaluator for derived climate quantities.

Safely evaluates math expressions over climate field values using AST validation.
"""

from __future__ import annotations

import ast
import math
from typing import Any

# ── Field aliases: short names → canonical field names ────────────────────────

FIELD_ALIASES: dict[str, str] = {
    "T": "temperature_2m",
    "T_2m": "temperature_2m",
    "T_s": "surface",
    "T_bl": "boundary_layer",
    "T_atm": "atmosphere",
    "q": "humidity",
    "q_sat": "saturation_humidity",
    "RH": "relative_humidity",
    "P": "surface_pressure",
    "u": "wind_u_10m",
    "v": "wind_v_10m",
    "ws": "wind_speed_10m",
    "u_bl": "wind_u",
    "v_bl": "wind_v",
    "ws_bl": "wind_speed",
    "u_g": "wind_u_geostrophic",
    "v_g": "wind_v_geostrophic",
    "ws_g": "wind_speed_geostrophic",
    "u_ocean": "ocean_u",
    "v_ocean": "ocean_v",
    "z": "elevation",
    "precip": "precipitation",
    "clouds": "cloud_fraction",
    "Td": "dew_point",
    "wind_dir": "wind_direction_10m",
    "lapse": "lapse_rate",
}

# ── Allowed math functions ────────────────────────────────────────────────────

ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "abs": abs,
    "min": min,
    "max": max,
}

# ── Math constants ────────────────────────────────────────────────────────────

MATH_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

# ── AST node whitelist ────────────────────────────────────────────────────────

_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Compare,
    ast.IfExp,
    ast.BoolOp,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    # Comparisons
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
)

MAX_EXPRESSION_LENGTH = 500


class ExpressionError(Exception):
    """Raised when an expression is invalid or unsafe."""


def _validate_ast(tree: ast.AST) -> None:
    """Walk the AST and reject any disallowed node types."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"Disallowed expression element: {type(node).__name__}")
        # Only allow calls to known math functions
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("Only direct function calls allowed (e.g. sqrt(x))")
            if node.func.id not in ALLOWED_FUNCTIONS:
                raise ExpressionError(f"Unknown function: {node.func.id}")


def _collect_names(tree: ast.AST) -> set[str]:
    """Collect all Name nodes from the AST."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in ALLOWED_FUNCTIONS
    }


def safe_eval(
    expression: str,
    field_values: dict[str, float],
) -> float:
    """Safely evaluate a math expression with resolved field values.

    Args:
        expression: Math expression string.
        field_values: Mapping of all names in the expression to float values.

    Returns:
        The computed result as a float.

    Raises:
        ExpressionError: If the expression is invalid, unsafe, or references
            unknown names.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"Expression too long ({len(expression)} chars, max {MAX_EXPRESSION_LENGTH})"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid syntax: {exc}") from exc

    _validate_ast(tree)

    # Build namespace: field values + math functions + constants
    namespace: dict[str, Any] = {"__builtins__": {}}
    namespace.update(ALLOWED_FUNCTIONS)
    namespace.update(MATH_CONSTANTS)
    namespace.update(field_values)

    try:
        result = eval(compile(tree, "<calculate>", "eval"), namespace)  # noqa: S307
    except Exception as exc:
        raise ExpressionError(f"Evaluation error: {exc}") from exc

    return float(result)


def resolve_fields(
    expression: str,
    sample_fn,
    lat: float,
    lon: float,
    month: int,
) -> dict[str, Any]:
    """Parse expression, resolve field names, evaluate, and return result dict.

    Args:
        expression: Math expression string with field names/aliases.
        sample_fn: Callable(field, lat, lon, month) → float (raw value).
        lat: Latitude.
        lon: Longitude.
        month: Month index (0-11).

    Returns:
        Dict with expression, result, unit, and inputs.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"Expression too long ({len(expression)} chars, max {MAX_EXPRESSION_LENGTH})"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid syntax: {exc}") from exc

    _validate_ast(tree)

    # Collect names and resolve to field values
    names = _collect_names(tree)
    field_values: dict[str, float] = {}
    inputs: dict[str, float] = {}

    for name in names:
        if name in MATH_CONSTANTS:
            field_values[name] = MATH_CONSTANTS[name]
            continue

        # Resolve alias → canonical field name
        canonical = FIELD_ALIASES.get(name, name)
        value = sample_fn(canonical, lat, lon, month)
        if value is None:
            raise ExpressionError(f"Unknown field: '{name}'")
        field_values[name] = value
        inputs[name] = value

    result = safe_eval(expression, field_values)
    return {
        "expression": expression,
        "result": round(result, 4),
        "inputs": inputs,
    }
