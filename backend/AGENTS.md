# Backend

## Structure

- `climate_sim/core/`: solver infrastructure
  - `operators.py`: builds spatial operators (diffusion, advection, wind) from config
  - `rhs_builder.py`: assembles physics tendencies into RHS functions
  - `solver.py`: Newton solver with Anderson acceleration for periodic annual cycle
  - `postprocess.py`: extracts output layers from solver state
- `climate_sim/data/`: static datasets, constants, elevation, land mask
- `climate_sim/physics/`: parameterization modules
- `climate_sim/runtime/`: CLI plumbing and config dataclasses
- **Entry points** (run from project root):
  - `backend/main.py`: run simulation, save to `data/main.npz`
  - `backend/eval.py`: compare against NOAA climatology
  - `backend/export_frontend_data.py`: export JSON for frontend

## Solver architecture

The solver finds a periodic annual cycle (12 monthly states that repeat year-to-year):

1. **Operator construction** (`operators.py`): config → grids, masks, spatial operators
2. **Physics assembly** (`rhs_builder.py`): operators → RHS tendency functions + Jacobian
3. **Newton solver** (`solver.py`): iterates each month to steady state, uses Anderson acceleration across months

Entry point: `solve_periodic_climate(resolution_deg, model_config, return_layer_map=True)`

## Physics modules

Located in `climate_sim/physics/`. Each module has a config dataclass and implements `tendency()` for the solver. Key modules: `radiation.py`, `diffusion.py`, `humidity.py`, `clouds.py`, `surface_albedo.py`, `sensible_heat_exchange.py`, `latent_heat_exchange.py`, `atmosphere/wind.py`, `atmosphere/advection.py`, `ocean_currents.py`.

Variable classification:
- **Prognostic** (in state vector, solved by Newton): temperature (surface, BL, atmosphere), humidity
- **Semi-diagnostic** (recomputed between Newton iterations, frozen during): wind, albedo, clouds, ocean currents, Ekman pumping
- **Annual-cycle prognostic** (updated after each month, Anderson-accelerated across years): soil moisture
- **Diagnostic** (computed after convergence): precipitation, vertical velocity, surface pressure, relative humidity, vegetation fraction


## Coding guidelines

- **Type hints everywhere**: precise annotations on all public functions
- **Dataclass configs**: group parameters in frozen dataclasses, extend rather than add kwargs
- **Vectorized math**: no Python loops over grid points; use broadcasting and NumPy

### Adding new physics

Pipeline: `physics/` module with config dataclass + operator class (`tendency()`, optionally `linearised_tendency()`) → add to `ModelConfig` → wire through `operators.py` → `rhs_builder.py` → solver picks it up.

### Debugging physics

- After a successful run, analyze `data/main.npz` for simulation output
- Physics parameters should match observations from first principles — don't tune to fit output
- Don't jump to conclusions like "this is a fundamental resolution limitation" or "this is a fundamental 2 layer limitation" without physically justifying why this is the case. For example, 5 degree resolution can't resolve the Sierra Nevada, but it can resolve the Rockies just fine.
