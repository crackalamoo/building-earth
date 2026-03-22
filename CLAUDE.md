# Climate Simulator

A real-time, visually beautiful, educational climate simulator that explains the "why" behind local climate patterns.

**The emotional core**: You live inside a machine made of sunlight, water, and wind. It took 4 billion years to tune itself. This app shows you how it works — and what you're part of.

## Quick reference

| Task | Command |
|------|---------|
| Run simulation | `uv run python backend/main.py --resolution 5 --headless` |
| Evaluate vs NOAA | `uv run python backend/eval.py --cache --headless --resolution 5` |
| Export for frontend | `PYTHONPATH=backend uv run python -m export_frontend_data --cache --resolution 5 --interpolate` |
| Export onboarding stages | `PYTHONPATH=backend uv run python -m export_frontend_data --onboarding --cache --resolution 5 --interpolate` |
| Compare with previous cached run | `uv run python backend/scenario_compare.py --base-cache --headless --resolution 5` |

`--cache` means eval reads `data/main.npz` from the LAST simulation run. If you changed physics and want to compare before/after, you MUST re-run `main.py` for each version — `--cache` does NOT re-simulate.

To get a compact summary of eval metrics, use:
```bash
uv run python backend/eval.py --cache --headless --resolution 5 2>&1 | grep -E "^(Area-weighted (RMSE|Bias|Pattern)|Annual|Pattern corr|U comp|Mean )"
```
This shows labeled T RMSE/bias/corr, then humidity, RH, precipitation, SLP, and wind sections. The "Annual" rows are: `Land Ocean Global [RMSE]`.

## Vision

The goal is wonder combined with understanding: make people *feel* why climate works the way it does, not just see data on a globe.

**What we're building**:
- A 3-layer monthly energy-balance model (1°-5° global grid)
- First-principles physics: radiation, diffusion, humidity, clouds, wind, ocean currents
- Backend-authoritative (Python + NumPy/SciPy)
- Frontend: Three.js globe with layered visualization
- LLM layer for natural-language "why?" explanations of any location

**What we're NOT building**:
- A research-grade GCM (we sacrifice some accuracy for clarity)
- A data visualization tool (we tell stories, not just show numbers)
- A doom-and-gloom climate warning (we inspire curiosity first)

## Repository tour

- **`backend/`** — Python climate simulation: 3-layer energy-balance model on a global grid with first-principles physics (radiation, diffusion, humidity, clouds, wind, ocean currents), Newton solver for periodic annual cycle, and eval against NOAA climatology.
- **`frontend/`** — Svelte + Three.js globe visualization that renders simulation output as an interactive, animated globe.

## Design philosophy

The story we're telling is not "here's climate data" but rather: "Here's the machine you live inside."

Every design choice should serve **wonder at the system** + **understanding of the mechanism**.
