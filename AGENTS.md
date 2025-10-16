# Climate Simulator

I’m developing a real-time, visually beautiful, educational climate simulator that explains the “why” behind local climate patterns and future warming.

The concept:
•	A 2-D monthly energy-balance model (1.0° global grid).
•	Light physics: radiative balance, meridional diffusion, water-vapor and ice feedbacks, simple orographic/lapse effects. Everything based on first principles.
•	Eventually, wind and precipitation diagnostics (thermal-wind scaling, Clausius-Clapeyron moisture).
•	Backend-authoritative (Python + SciPy/NumPy) for precomputed 1° tiles.
•	Frontend: Three.js for animation; LLM layer for “natural-language explanation” of local results.

The goal is an app that’s physically honest, first-principles, and emotionally engaging — a small masterpiece of climate storytelling.

Assume I’m comfortable with Python, Rust, and frontend development.

## Repository tour
- `src/climate_sim/modeling/`: physical parameterizations (radiation balance, diffusion, snow albedo, etc.). These modules are pure numerics with extensive `numpy` usage and rely on dataclasses for configuration objects.
- `src/climate_sim/utils/`: numerical helpers such as grid generation, heat-capacity bookkeeping, and the solver (`compute_periodic_cycle_celsius`). Expect well-factored, unit-testable functions.
- `src/climate_sim/plotting.py`: convenience wrappers around Matplotlib/Cartopy to render global temperature fields.
- `src/main.py`: single-run driver that prints summary statistics and produces plots for the surface (and optionally atmosphere) temperature cycles.
- `src/scenario_compare.py`: CLI that toggles specific physics options and shows anomaly maps between two model configurations.
- `tests/`: `pytest` suite focused on smoke-testing the plotting pipeline.

## Models Implemented
The following models have already been implemented:

- `src/climate_sim/modeling/advection.py`: Coriolis force.
- `src/climate_sim/modeling/diffusion.py`: Atmospheric and oceanic diffusion.
- `src/climate_sim/modeling/radiation.py`: Radiation with a solar constant and a single-layer atmosphere.
- `src/climate_sim/modeling/snow_albedo.py`: Increase albedo of land for freezing temperatures.

All models are tied together in the main solver: `src/climate_sim/utils/solver.py`.

Also of note: `src/climate_sim/utils/landmask.py` computes which cells are land vs water, including heat capacities in each case.

The main goal at the current stage of development is to extend this with new models while maintaining good performance.

## Local development workflow
1. **Install dependencies**
Use [uv](https://docs.astral.sh/uv/) (lockfile already checked in)
```bash
uv sync
```
2. **Run commands via uv** (keeps dependencies isolated):
- Launch the primary CLI with defaults:
```bash
uv run python -m main
```
- Compare scenarios (example toggling diffusion):
```bash
uv run python -m scenario_compare --no-base-diffusion
```
3. **Execute tests**:
```bash
uv run pytest
```

In general, when iterating, use a larger resolution (like 5 or 10 degrees) for local tests to keep runtime fast.

## Coding guidelines
- **Type hints everywhere**: all public functions and methods include precise type annotations. Prefer `numpy.typing.NDArray` when returning arrays.
- **Dataclass configs**: keep model configuration inputs grouped into frozen dataclasses (`RadiationConfig`, `DiffusionConfig`, etc.) and thread them through the solver. Extend these objects instead of proliferating kwargs.
- **Vectorized math**: the solver assumes array-oriented operations. Avoid Python loops over grid points; leverage broadcasting and `numpy` utilities.

## Testing philosophy
- Fast unit tests belong under `tests/` and must run headless (`matplotlib.use("Agg")`).
- Expensive experiment scripts (`main.py`, `scenario_compare.py`) are exercised manually; keep them thin wrappers around tested utilities.
- For now, focus on improving the core functionality rather than writing tests unless specifically instructed. The code is in an early stage, and still iterating too quickly for tests to be very useful.