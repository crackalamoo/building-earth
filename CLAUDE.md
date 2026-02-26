# Climate Simulator

A real-time, visually beautiful, educational climate simulator that explains the "why" behind local climate patterns.

**The emotional core**: You live inside a machine made of sunlight, water, and wind. It took 4 billion years to tune itself. This app shows you how it works — and what you're part of.

## Quick reference

| Task | Command |
|------|---------|
| Run simulation | `uv run python backend/main.py --resolution 5 --headless` |
| Evaluate vs NOAA | `uv run python backend/eval.py --cache --headless --resolution 5` |
| Export for frontend | `uv run python backend/export_frontend_data.py --cache --resolution 5 --interpolate` |

To get a compact summary of eval metrics, use:
```bash
uv run python backend/eval.py --cache --headless --resolution 5 2>&1 | grep -E "^(Area-weighted (RMSE|Bias|Pattern)|Annual|Pattern corr|U comp|Mean )"
```
This shows labeled T RMSE/bias/corr, then humidity, RH, precipitation, SLP, and wind sections. The "Annual" rows are: `Land Ocean Global [RMSE]`.

## Vision

The goal is wonder combined with understanding: make people *feel* why climate works the way it does, not just see data on a globe.

**What we're building**:
- A 2-D monthly energy-balance model (1°-5° global grid)
- First-principles physics: radiation, diffusion, humidity, clouds, wind, ocean currents
- Backend-authoritative (Python + NumPy/SciPy)
- Frontend: Three.js globe with layered visualization
- LLM layer for natural-language "why?" explanations of any location

**What we're NOT building**:
- A research-grade GCM (we sacrifice some accuracy for clarity)
- A data visualization tool (we tell stories, not just show numbers)
- A doom-and-gloom climate warning (we inspire curiosity first)

## Repository tour

### Backend (`backend/`)
- `climate_sim/core/`: solver infrastructure
  - `operators.py`: builds spatial operators (diffusion, advection, wind) from config
  - `rhs_builder.py`: assembles physics tendencies into RHS functions
  - `solver.py`: Newton solver with Anderson acceleration for periodic annual cycle
  - `postprocess.py`: extracts output layers from solver state
- `climate_sim/data/`: static datasets, constants, elevation, land mask
- `climate_sim/physics/`: parameterization modules (see Physics Status below)
- `climate_sim/runtime/`: CLI plumbing and config dataclasses
- **Entry points** (run from project root):
  - `backend/main.py`: run simulation, save to `data/main.npz`
  - `backend/eval.py`: compare against NOAA climatology
  - `backend/export_frontend_data.py`: export JSON for frontend

### Frontend (`frontend/`)
- Svelte + Three.js globe visualization
- `src/App.svelte`: main app with month animation controls
- `src/lib/Globe.svelte`: Three.js globe with temperature coloring
- `src/lib/colormap.ts`: temperature-to-color mapping

When testing, run from project root with `--headless --resolution 5` to avoid plotting delays.

### Solver architecture

The solver finds a periodic annual cycle (12 monthly states that repeat year-to-year):

1. **Operator construction** (`operators.py`): config → grids, masks, spatial operators
2. **Physics assembly** (`rhs_builder.py`): operators → RHS tendency functions + Jacobian
3. **Newton solver** (`solver.py`): iterates each month to steady state, uses Anderson acceleration across months

Entry point: `solve_periodic_climate(resolution_deg, model_config, return_layer_map=True)`

### Physics modules

Located in `climate_sim/physics/`. Each module has a config dataclass and implements `tendency()` for the solver. Key modules: `radiation.py`, `diffusion.py`, `humidity.py`, `clouds.py`, `snow_albedo.py`, `sensible_heat_exchange.py`, `latent_heat_exchange.py`, `atmosphere/wind.py`, `atmosphere/advection.py`, `ocean_currents.py`.

Run `backend/eval.py` to check current model performance against NOAA climatology.

## Coding guidelines

- **Type hints everywhere**: precise annotations on all public functions
- **Dataclass configs**: group parameters in frozen dataclasses, extend rather than add kwargs
- **Vectorized math**: no Python loops over grid points; use broadcasting and NumPy

### Adding new physics

Pipeline: `physics/` module with config dataclass + operator class (`tendency()`, optionally `linearised_tendency()`) → add to `ModelConfig` → wire through `operators.py` → `rhs_builder.py` → solver picks it up.

### Debugging physics

- Run simulations via scripts, analyze `data/main.npz`
- Physics parameters should match observations from first principles — don't tune to fit output
- Use `backend/eval.py` to check RMSE/bias/correlation against NOAA
- Don't jump to conclusions like "this is a fundamental resolution limitation" or "this is a fundamental 2 layer limitation" without physically justifying why this is the case. For example, 5 degree resolution can't resolve the Sierra Nevada, but it can resolve the Rockies just fine.

## Visualization design philosophy

The story we're telling is not "here's climate data" but rather: "Here's the machine you live inside."

Every design choice should serve **wonder at the system** + **understanding of the mechanism**.
