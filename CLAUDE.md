# Climate Simulator

I’m developing a real-time, visually beautiful, educational climate simulator that explains the “why” behind local climate patterns and future warming.

The concept:
- A 2-D monthly energy-balance model (1.0° global grid).
- Light physics: radiative balance, meridional diffusion, water-vapor and ice feedbacks, simple orographic/lapse effects. Everything based on first principles.
- Eventually, wind and precipitation diagnostics (thermal-wind scaling, Clausius-Clapeyron moisture).
- Backend-authoritative (Python + SciPy/NumPy) for 1° tiles.
- Frontend: Three.js for animation; LLM layer for “natural-language explanation” of local results.

The goal is an app that’s physically honest, first-principles, and emotionally engaging — a small masterpiece of climate storytelling.

## Repository tour
- `src/climate_sim/core/`: grid generation, solver infrastructure, and shared numerical utilities. Includes:
  - `operators.py`: builds spatial operators (diffusion, advection, wind) and grids from configuration
  - `rhs_builder.py`: assembles physics tendencies into right-hand-side functions
  - `solver.py`: core numerical algorithms (Newton solver, Anderson acceleration, timestep evolution)
- `src/climate_sim/data/`: static datasets, constants, and loaders for elevation and calendar information.
- `src/climate_sim/physics/`: parameterization modules for radiation, diffusion, advection, heat exchange, and related processes.
- `src/climate_sim/runtime/`: command-line plumbing shared across entry points.
- `src/climate_sim/plotting.py`: rendering helpers for global temperature and diagnostic plots.
- `src/main.py`: interactive visualization driver for a single model configuration.
- `src/scenario_compare.py`: CLI for contrasting physics toggles and producing anomaly outputs.

### Core architecture

The solver pipeline has three clear stages:

1. **Operator construction** (`operators.py`):
   - Takes `ModelConfig` as input
   - Builds grids, masks, and all spatial operators (diffusion, advection, wind)
   - Returns `ModelOperators` dataclass containing everything needed for physics

2. **Physics assembly** (`rhs_builder.py`):
   - Takes `ModelOperators` as input
   - Combines individual physics operators into complete RHS tendency functions
   - Returns functions: `rhs(state) -> tendencies` and `rhs_derivative(state) -> Jacobian`

3. **Numerical solver** (`solver.py`):
   - Takes RHS functions as input
   - Executes Newton iterations for monthly timesteps
   - Uses Anderson acceleration to find the periodic annual cycle
   - Main entry point: `solve_periodic_climate(resolution, config)`

Also of note: `src/climate_sim/data/landmask.py` computes which cells are land vs water, including heat capacities in each case.

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

In general, when iterating, use a larger resolution (like 5 degrees) for local tests to keep runtime fast.

## Coding guidelines
- **Type hints everywhere**: all public functions and methods include precise type annotations.
- **Dataclass configs**: keep model configuration inputs grouped into frozen dataclasses (`RadiationConfig`, `DiffusionConfig`, etc.) and thread them through the solver. Extend these objects instead of proliferating kwargs.
- **Vectorized math**: the solver assumes array-oriented operations. Avoid Python loops over grid points; leverage broadcasting and `numpy` utilities.

## Adding new physics operators

New physics processes follow a standard pipeline: create the physics module in `src/climate_sim/physics/` with a config dataclass and operator class implementing `tendency()` and optionally `linearised_tendency()` for the Jacobian. Add the config to `ModelConfig`, then wire it through: construct the operator in `operators.py` (adding to `ModelOperators`), pass it via `RhsBuildInputs` to `rhs_builder.py`, and integrate the tendencies into the `rhs()` and `rhs_derivative()` functions. The flow is: config → `build_model_operators()` → `ModelOperators` → `RhsBuildInputs` → `create_rhs_functions()` → solver.

## Testing philosophy
- Expensive experiment scripts (`main.py`, `scenario_compare.py`) are exercised manually; keep them thin wrappers around tested utilities.
- For now, focus on improving the core functionality rather than writing tests unless specifically instructed. The code is in an early stage, and still iterating too quickly for tests to be very useful.
