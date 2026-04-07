---
name: evaluate-branch
description: Run eval on current branch and main, compare metrics side-by-side, and suggest whether to merge or wait.
allowed-tools: Bash(git branch:*), Bash(git checkout:*), Bash(git stash:*), Bash(git status:*), Bash(cp data/*), Bash(rm data/*), Read, Grep
user-invocable: true
---

# Evaluate Branch Skill

Compare the current feature branch against main by running simulation + eval on both, then present a side-by-side comparison with a merge recommendation. **Focus on land metrics** — land is where the model struggles most and where improvements matter.

## Steps

### 1. Record the current branch name

```bash
git branch --show-current
```

If already on `main`, tell the user this skill is for comparing feature branches against main, and stop.

### 2. Run simulation + eval on the feature branch

**Check if simulation is cached** (no source files newer than `data/main.npz`):
```bash
find backend -name "*.py" -newer data/main.npz 2>/dev/null | head -5
```

If not cached, run:
```bash
uv run python backend/main.py --resolution 5 --headless
```

Then run eval (full output, no truncation):
```bash
uv run python backend/eval.py --resolution 5 --headless --cache 2>&1
```

Save the **complete** eval output for later comparison. Parse out metrics, focusing on **land** columns:
- Temperature: bias, RMSE, pattern correlation (land, ocean, global × annual/Jan/Jul)
- Precipitation: bias, RMSE, pattern correlation (land, ocean, global)
- Humidity: correlation (land)
- SLP: correlation (land, ocean)
- Wind: correlation

### 3. Stash any uncommitted changes and save branch simulation

```bash
cp data/main.npz data/main_branch.npz
git stash --include-untracked -m "evaluate-branch auto-stash"
```

Note whether stash actually saved anything (check output for "No local changes").

### 4. Switch to main and run simulation + eval

```bash
git checkout main
```

Always re-run simulation after switching branches since the code changed:
```bash
uv run python backend/main.py --resolution 5 --headless
uv run python backend/eval.py --resolution 5 --headless --cache 2>&1
```

Save the complete eval output.

### 5. Switch back to the feature branch and restore state

```bash
git checkout <branch-name>
cp data/main_branch.npz data/main.npz
rm data/main_branch.npz
```

If changes were stashed in step 3, pop the stash:
```bash
git stash pop
```

### 6. Present side-by-side comparison

Display a markdown table comparing branch vs main. **Land rows first**, then ocean/global. Use format:

```
## Branch `<name>` vs `main`

### Land (primary)

| Metric              | Main   | Branch | Delta  | Better? |
|----------------------|--------|--------|--------|---------|
| T bias (land)        | -X.XX  | -X.XX  | +X.XX  | ✓ / ✗   |
| T RMSE (land)        | X.XX   | X.XX   | -X.XX  | ✓ / ✗   |
| T corr (land)        | 0.XXX  | 0.XXX  | +0.XXX | ✓ / ✗   |
| P bias (land)        | ...    | ...    | ...    | ...     |
| P RMSE (land)        | ...    | ...    | ...    | ...     |
| P corr (land)        | ...    | ...    | ...    | ...     |
| RH corr (land)       | ...    | ...    | ...    | ...     |
| SLP corr (land)      | ...    | ...    | ...    | ...     |

### Ocean & Global

| Metric              | Main   | Branch | Delta  | Better? |
|----------------------|--------|--------|--------|---------|
| T bias (ocean)       | ...    | ...    | ...    | ...     |
| T bias (global)      | ...    | ...    | ...    | ...     |
| T RMSE (global)      | ...    | ...    | ...    | ...     |
| T corr (global)      | ...    | ...    | ...    | ...     |
| P corr (global)      | ...    | ...    | ...    | ...     |
| SLP corr (ocean)     | ...    | ...    | ...    | ...     |
| Wind corr            | ...    | ...    | ...    | ...     |
```

"Better?" column: ✓ if the branch improves the metric, ✗ if it degrades it, — if unchanged (within 0.01).

### 7. Merge recommendation

Based on the comparison, give one of three recommendations. **Land metrics carry more weight** since that's where we need improvement.

- **Merge** — if land metrics improve (especially P corr land, T bias land, RH corr land) without significantly degrading ocean. State which land metrics improved.
- **Wait** — if land results are mixed or land gains come at the cost of large ocean regressions. Identify what needs fixing.
- **Revert** — if the branch degrades land metrics. Suggest what went wrong.

Be specific about which metrics drove the recommendation and by how much. Do not recommend purely based on metrics; also consider the overall pattern and the physical plausibility of the changes.

## Output

The side-by-side table and merge recommendation are the primary output. Keep commentary concise and actionable.
