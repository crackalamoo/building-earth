---
name: debug-convergence
description: Debug solver convergence issues after physics changes. Use when the simulation fails to converge or oscillates after modifying physics parameterizations.
allowed-tools: Read, Bash, Grep, Glob
---

# Debug Convergence Skill

Use this when `main.py` fails to converge or convergence is slow/oscillating.

## 1. Capture convergence trajectory

The solver prints per-iteration residuals at the **Anderson acceleration level** (year-to-year periodic cycle). Grep for them:

```bash
uv run python backend/main.py --resolution 5 --headless 2>&1 | grep -Ei "^iter |Converged|Failed to converge"
```

Output format:
```
iter  0: RMS=11.830K  95p=27.280K  max=55.889K
iter  1: RMS=2.405K  95p=4.618K  max=18.185K
```

- **RMS**: root-mean-square temperature difference between consecutive annual cycles
- **95p**: 95th percentile
- **max**: maximum absolute difference at any grid cell/month
- These are NOT Newton residuals — Newton converges within each monthly step

## 2. Diagnose the pattern

| Pattern | Likely cause |
|---------|-------------|
| RMS decreasing but max stays high | One grid cell or small region oscillating between years |
| RMS oscillating, never settling | Physics don't admit a stable periodic solution — year-to-year mapping has no fixed point |
| Diverging (RMS increasing) | Positive feedback loop with no restoring force |

## 3. Common causes of non-convergence

- **Positive feedback without restoring force**: e.g. more moisture → more convection → more moisture. The year-to-year map amplifies perturbations.
- **ITCZ-coupled forcing over wide area**: if a forcing depends on ITCZ position and covers a large region, small ITCZ shifts cause large forcing changes, which feed back into temperature and move the ITCZ further.
- **Historically failed**: THERMAL_PRESSURE_COEFFICIENT > 200, reduced Rossby smoothing, Sundqvist cloud formula alone, LTS-based clouds, θ_e convective trigger.

## 4. What to investigate

1. **Where is the max residual?** Temporarily add location info to the print in `solver.py` (~line 1253):
```python
max_idx = np.unravel_index(np.argmax(np.abs(monthly_temp_diff)), monthly_temp_diff.shape)
max_month, max_lat_idx, max_lon_idx = max_idx
nlat_grid, nlon_grid = monthly_temp_diff.shape[1], monthly_temp_diff.shape[2]
max_lat = -90 + 90/nlat_grid + max_lat_idx * 180/nlat_grid
max_lon = 180/nlon_grid + max_lon_idx * 360/nlon_grid
print(f"iter {iter_idx:2d}: ... @({max_lat:+.1f},{max_lon:.1f},mon={max_month})")
```
Revert after debugging.
2. **Is there a feedback loop?** Trace the physics chain at the oscillating location. Does the parameterization create a self-amplifying cycle?
3. **Is the forcing area too large relative to ITCZ sensitivity?** Broad forcings coupled to ITCZ position are inherently destabilizing — the wider the forced area, the stronger the feedback on the ITCZ itself.
