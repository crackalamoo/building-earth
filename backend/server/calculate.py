"""Lightweight expression evaluator for derived climate quantities.

Safely evaluates math expressions over climate field values using AST validation.
"""

import ast
import math
from typing import Any

import numpy as np

from .climate_data import ClimateDataStore

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

# ── Reduction functions ───────────────────────────────────────────────────────
# Reductions evaluate their inner expression on the underlying numpy arrays
# and collapse one or more axes, returning a scalar that the rest of the
# expression then uses like any other value.

REDUCTION_FUNCTIONS: frozenset[str] = frozenset(
    {
        "global_mean",
        "zonal_mean",
        "lat_band_mean",
        "box_mean",
    }
)

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
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("Only direct function calls allowed (e.g. sqrt(x))")
            if node.func.id not in ALLOWED_FUNCTIONS and node.func.id not in REDUCTION_FUNCTIONS:
                raise ExpressionError(f"Unknown function: {node.func.id}")


def _collect_names(tree: ast.AST) -> set[str]:
    """Collect Name nodes that need scalar resolution at the call site.

    Names that appear *inside* a reduction call are skipped — they get
    resolved as numpy arrays during vectorized evaluation, not as scalars.
    """
    names: set[str] = set()

    class _Collector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in REDUCTION_FUNCTIONS:
                # Skip the entire subtree under a reduction.
                return
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id not in ALLOWED_FUNCTIONS and node.id not in REDUCTION_FUNCTIONS:
                names.add(node.id)

    _Collector().visit(tree)
    return names


def _resolve_field_array(
    name: str,
    store: ClimateDataStore,
    month: int | None,
) -> np.ndarray:
    """Return the (nlat, nlon) array for a field, slicing or averaging
    along the month axis. If month is None, average over all 12 months."""
    canonical = FIELD_ALIASES.get(name, name)
    arr = store.get_array(canonical)
    if arr is None:
        raise ExpressionError(f"Unknown field: '{name}'")
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if month is None:
            return arr.mean(axis=0)
        return arr[month % 12]
    raise ExpressionError(f"Unexpected shape for '{name}': {arr.shape}")


def _collect_inner_names(tree: ast.AST) -> set[str]:
    """Collect Name nodes from a sub-tree, ignoring functions and constants."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in ALLOWED_FUNCTIONS
        and node.id not in REDUCTION_FUNCTIONS
        and node.id not in MATH_CONSTANTS
    }


_VECTORIZED_FUNCTIONS: dict[str, Any] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
}


def _vectorized_eval(
    inner_tree: ast.AST,
    store: ClimateDataStore,
    month: int | None,
) -> np.ndarray:
    """Evaluate an expression sub-tree against full numpy arrays so a
    reduction function can collapse it. The inner expression must reference
    only fields, math functions, and constants — no nested reductions.

    Math functions resolve to their numpy equivalents so they broadcast over
    arrays instead of erroring on the scalar `math.sqrt`."""
    namespace: dict[str, Any] = {"__builtins__": {}}
    namespace.update(_VECTORIZED_FUNCTIONS)
    namespace.update(MATH_CONSTANTS)
    for name in _collect_inner_names(inner_tree):
        namespace[name] = _resolve_field_array(name, store, month)
    expr = ast.Expression(body=inner_tree)  # type: ignore[arg-type]
    ast.fix_missing_locations(expr)
    try:
        return eval(compile(expr, "<reduction>", "eval"), namespace)  # noqa: S307
    except Exception as exc:
        raise ExpressionError(f"Reduction evaluation error: {exc}") from exc


def _const_arg(node: ast.AST, what: str) -> float:
    """Extract a numeric literal from a reduction argument. Accepts plain
    numbers and unary +/- literals (e.g. -90, +0)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return sign * float(node.operand.value)
    raise ExpressionError(f"{what} must be a numeric literal")


def _grid_axes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive cell-center latitudes and longitudes from a 2-D field's shape.

    Lats and lons are computed from the array shape directly because the
    store may hold fields at multiple resolutions (e.g. 36x72 and 720x1440)
    simultaneously, so a cached store-level grid is unsafe.
    """
    nlat, nlon = values.shape
    lats = (np.arange(nlat) + 0.5) / nlat * 180 - 90
    lons = (np.arange(nlon) + 0.5) / nlon * 360
    return lats, lons


def _is_scalar(values: Any) -> bool:
    """True if the inner expression already collapsed to a scalar — e.g. a
    nested reduction or a literal numeric arg. Reducing a scalar is a no-op."""
    return not isinstance(values, np.ndarray) or values.ndim < 2


def _global_mean(values: np.ndarray) -> float:
    """Area-weighted (cos-lat) mean over the full (nlat, nlon) field."""
    if _is_scalar(values):
        return float(values)
    lats, _ = _grid_axes(values)
    weights = np.cos(np.deg2rad(lats))
    weighted_rows = (values * weights[:, None]).sum(axis=1)
    return float(weighted_rows.sum() / (weights.sum() * values.shape[1]))


def _zonal_mean(values: np.ndarray, lat: float) -> float:
    """Mean over all longitudes at the lat-row nearest the requested lat."""
    if _is_scalar(values):
        return float(values)
    lats, _ = _grid_axes(values)
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    return float(values[lat_idx].mean())


def _lat_band_mean(values: np.ndarray, lat_min: float, lat_max: float) -> float:
    """Area-weighted (cos-lat) mean over rows whose lat falls in [lat_min, lat_max]."""
    if lat_min > lat_max:
        raise ExpressionError(f"lat_band_mean: lat_min ({lat_min}) must be <= lat_max ({lat_max})")
    if _is_scalar(values):
        return float(values)
    lats, _ = _grid_axes(values)
    mask = (lats >= lat_min) & (lats <= lat_max)
    if not mask.any():
        raise ExpressionError(f"lat_band_mean: no grid rows fall in [{lat_min}, {lat_max}]")
    band_values = values[mask]
    band_weights = np.cos(np.deg2rad(lats[mask]))
    weighted_rows = (band_values * band_weights[:, None]).sum(axis=1)
    return float(weighted_rows.sum() / (band_weights.sum() * values.shape[1]))


def _box_mean(
    values: np.ndarray,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
) -> float:
    """Area-weighted (cos-lat) mean over a lat/lon box. Longitude bounds may
    wrap across the antimeridian (lon_min > lon_max means wraparound)."""
    if lat_min > lat_max:
        raise ExpressionError(f"box_mean: lat_min ({lat_min}) must be <= lat_max ({lat_max})")
    if _is_scalar(values):
        return float(values)
    lats, lons = _grid_axes(values)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    # Normalize to [0, 360) for the longitude comparison.
    lons_mod = lons % 360
    lo, hi = lon_min % 360, lon_max % 360
    if lo <= hi:
        lon_mask = (lons_mod >= lo) & (lons_mod <= hi)
    else:
        # Wraparound box: [lo, 360) ∪ [0, hi].
        lon_mask = (lons_mod >= lo) | (lons_mod <= hi)
    if not lat_mask.any() or not lon_mask.any():
        raise ExpressionError(
            f"box_mean: no grid cells fall in lat [{lat_min}, {lat_max}], "
            f"lon [{lon_min}, {lon_max}]"
        )
    box_values = values[np.ix_(lat_mask, lon_mask)]
    weights = np.cos(np.deg2rad(lats[lat_mask]))
    weighted_rows = (box_values * weights[:, None]).sum(axis=1)
    return float(weighted_rows.sum() / (weights.sum() * box_values.shape[1]))


def _apply_reductions(
    tree: ast.AST,
    store: ClimateDataStore,
    lat: float | None,
    month: int | None,
) -> ast.AST:
    """Replace each reduction call in the tree with an ast.Constant holding
    its computed scalar result. Returns the rewritten tree."""

    class _Rewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if not (isinstance(node.func, ast.Name) and node.func.id in REDUCTION_FUNCTIONS):
                return node
            fn = node.func.id
            if fn == "global_mean":
                if len(node.args) != 1:
                    raise ExpressionError("global_mean takes exactly one argument")
                values = _vectorized_eval(node.args[0], store, month)
                scalar = _global_mean(values)
                return ast.copy_location(ast.Constant(value=scalar), node)
            if fn == "zonal_mean":
                if len(node.args) != 1:
                    raise ExpressionError("zonal_mean takes exactly one argument")
                if lat is None:
                    raise ExpressionError("zonal_mean requires lat to be provided")
                values = _vectorized_eval(node.args[0], store, month)
                scalar = _zonal_mean(values, lat)
                return ast.copy_location(ast.Constant(value=scalar), node)
            if fn == "lat_band_mean":
                if len(node.args) != 3:
                    raise ExpressionError(
                        "lat_band_mean(field, lat_min, lat_max) takes 3 arguments"
                    )
                values = _vectorized_eval(node.args[0], store, month)
                lat_min = _const_arg(node.args[1], "lat_band_mean lat_min")
                lat_max = _const_arg(node.args[2], "lat_band_mean lat_max")
                scalar = _lat_band_mean(values, lat_min, lat_max)
                return ast.copy_location(ast.Constant(value=scalar), node)
            if fn == "box_mean":
                if len(node.args) != 5:
                    raise ExpressionError(
                        "box_mean(field, lat_min, lon_min, lat_max, lon_max) takes 5 arguments"
                    )
                values = _vectorized_eval(node.args[0], store, month)
                lat_min = _const_arg(node.args[1], "box_mean lat_min")
                lon_min = _const_arg(node.args[2], "box_mean lon_min")
                lat_max = _const_arg(node.args[3], "box_mean lat_max")
                lon_max = _const_arg(node.args[4], "box_mean lon_max")
                scalar = _box_mean(values, lat_min, lon_min, lat_max, lon_max)
                return ast.copy_location(ast.Constant(value=scalar), node)
            raise ExpressionError(f"Unknown reduction: {fn}")

    return _Rewriter().visit(tree)


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
    store: ClimateDataStore,
    lat: float | None,
    lon: float | None,
    month: int | None,
) -> dict[str, Any]:
    """Parse expression, resolve field names, evaluate, and return result dict.

    `lat`, `lon`, and `month` are each optional. Reduction functions in the
    expression remove the corresponding axis from the requirement; e.g.
    `global_mean(T)` doesn't need lat/lon, and an omitted month means the
    field is averaged over all 12 months at the requested cell.
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

    # Reductions are evaluated first and substituted into the tree as
    # constants, leaving plain scalar arithmetic for the rest of the eval.
    rewritten = _apply_reductions(tree, store, lat, month)

    names = _collect_names(rewritten)
    field_values: dict[str, float] = {}
    inputs: dict[str, float] = {}

    for name in names:
        if name in MATH_CONSTANTS:
            field_values[name] = MATH_CONSTANTS[name]
            continue
        if lat is None or lon is None:
            raise ExpressionError(
                f"'{name}' is a single-cell reference but lat/lon were not provided"
            )
        canonical = FIELD_ALIASES.get(name, name)
        if month is None:
            arr = store.get_array(canonical)
            if arr is None:
                raise ExpressionError(f"Unknown field: '{name}'")
            cell = _sample_cell_from_array(arr, lat, lon)
            value = float(np.mean(cell)) if cell.ndim == 1 else float(cell)
        else:
            value = store.sample_raw(canonical, lat, lon, month)
            if value is None:
                raise ExpressionError(f"Unknown field: '{name}'")
        field_values[name] = value
        inputs[name] = value

    result = _eval_rewritten(rewritten, field_values)
    return {
        "expression": expression,
        "result": round(result, 4),
        "inputs": inputs,
    }


def _sample_cell_from_array(arr: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """Pull the time series (or scalar) at a single grid cell from a full
    field array. Returns a 1-D (12,) slice for (12, nlat, nlon) fields or a
    0-D scalar for static (nlat, nlon) fields like elevation."""
    if arr.ndim == 2:
        nlat, nlon = arr.shape
    else:
        _, nlat, nlon = arr.shape
    lat_idx = int(np.clip(round((lat + 90) / 180 * nlat - 0.5), 0, nlat - 1))
    lon_idx = int(np.floor(((lon % 360) + 360) % 360 / 360 * nlon)) % nlon
    if arr.ndim == 2:
        return arr[lat_idx, lon_idx]
    return arr[:, lat_idx, lon_idx]


def _eval_rewritten(tree: ast.AST, field_values: dict[str, float]) -> float:
    """Evaluate a post-reduction-rewrite tree against scalar field values."""
    namespace: dict[str, Any] = {"__builtins__": {}}
    namespace.update(ALLOWED_FUNCTIONS)
    namespace.update(MATH_CONSTANTS)
    namespace.update(field_values)
    expr = tree if isinstance(tree, ast.Expression) else ast.Expression(body=tree)  # type: ignore[arg-type]
    ast.fix_missing_locations(expr)
    try:
        result = eval(compile(expr, "<calculate>", "eval"), namespace)  # noqa: S307
    except Exception as exc:
        raise ExpressionError(f"Evaluation error: {exc}") from exc
    return float(result)
