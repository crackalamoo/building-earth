# Physics Debugging Skill

Practices for diagnosing physics biases in the climate simulator.

## Principles

### 1. The system is at equilibrium — budget decomposition is useless
Every term balances to zero at the current state. Don't compute individual budget terms and try to find which one is "too big." Instead, ask: **if the state were correct (e.g., obs values), what tendency would the model produce that pushes it away from correct?**

### 2. Find the spatial signature before hypothesizing
Map the bias first. The spatial pattern constrains what physics can be responsible — a zonally uniform process can't cause a zonally varying bias. Ask what's *specifically* unique about the biased region, not vague things like "it's midlatitude land."

### 3. Respect nonlinearity and feedbacks
Process A can cause a bias at location Y even if process A is zero at Y in the current state. The system has feedback loops: A changes gradients → changes B → feeds back on A. Don't dismiss a process as irrelevant just because it appears small at the current (wrong) equilibrium.

### 4. Don't confuse framing errors with physics errors
Before computing anything, ask: "does this calculation discriminate between hypotheses?" Bad framings that waste time:
- Computing budget terms at equilibrium (they sum to zero by definition)
- Thinking in terms of Lagrangian trajectories when the model solves for Eulerian steady states on monthly timesteps
- Comparing a tendency "if obs values were plugged in" and drawing the wrong conclusion about sign (e.g., if obs T gives *more* diffusive warming than sim T, and sim is already too warm, that doesn't mean diffusion is the problem — it means the real world has an even stronger compensating cooling)

### 5. Use perturbation experiments
Swap ONE input field (T, wind, q) between sim and obs and recompute a tendency. The 4 combinations (simT×simW, simT×obsW, obsT×simW, obsT×obsW) reveal which field matters and how they interact.

### 6. Trace the causal chain — don't stop at symptoms
If field X is wrong, ask what sets X. Follow upstream: wrong wind → check pressure → check what drives pressure → check temperature pattern or orography. The root cause is often several steps removed from the symptom.

## Perturbation analysis tool

Use `perturb_cell.py` in this skill's directory to decompose restoring tendencies:

```bash
# Single cell: what pushes T_bl back from obs at Yakutia?
DATA_DIR=data uv run python .claude/skills/debug-physics/perturb_cell.py --lat 67.5 --lon 137.5 --target-t2m obs --month 0

# Latitude band with surface co-perturbed (most realistic)
DATA_DIR=data uv run python .claude/skills/debug-physics/perturb_cell.py --lat-band 50 70 --target-t2m obs --month 0 --also-perturb-sfc

# Explicit T_2m target (inverted to T_bl via lapse rate + elevation)
DATA_DIR=data uv run python .claude/skills/debug-physics/perturb_cell.py --lat 67.5 --lon 137.5 --target-t2m -46 --month 0
```

The `--target-t2m` argument specifies T_2m (not T_bl). The script inverts to T_bl using the model's own lapse rate + elevation formula. Use `--also-perturb-sfc` to shift T_sfc by the same delta, preserving the surface-BL gap — this eliminates the artificial SH signal from only perturbing one layer.

Reports: diffusion, advection, BL-atm mixing, sensible heat, latent heat, vertical motion, radiation (residual).

**Caveat**: perturbations create artificial gradients at the boundary of the perturbed region. This makes **diffusion and advection changes unreliable** — they reflect the artificial edge, not the real physics. Only trust terms that depend on **local** state. Use `--also-perturb-sfc` to avoid artificial SH signals. Diffusion and advection can still be useful signals, but they should not be trusted in isolation without further checks and logical inference.

## Common codebase pitfalls

- Grid starts at -87.5 (South Pole). Use `create_lat_lon_grid(5.0)`, never `np.linspace(87.5, -87.5, ...)`.
- Month indexing: output[0] = January, output[6] = July. No shift.
- Units: temperatures in main.npz are °C; precipitation is kg/m²/s; humidity is kg/kg.
- T2m is NOT T_bl — T2m is interpolated with a lapse rate correction. Compare T2m to obs.
