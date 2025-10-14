# Diffusion anomaly brainstorming

*The following notes capture hypotheses about the runaway warming that appears when diffusion is enabled at 1° resolution.*

## 1. Interface conductance might double-count the cosine metric

The spherical rewrite now multiplies meridional conductance by the east-west interface length (∝ cos φ) and divides by the physical grid spacing (also ∝ cos φ). At low latitudes this is benign, but toward the poles the ratio scales as 1 / cos²φ. With κ fixed, that makes the effective coupling ~15× stronger at 75° than at the equator. The scheme still conserves energy, yet it can flatten gradients so aggressively that the global steady state is governed by the warm, high-insolation tropics. The uniform solution that balances the global radiation budget lands close to 300 K, matching the observed 27 °C mean when diffusion is on.

## 2. Ocean/atmosphere fluxes ignore the land-mask heat capacity suppression

We zero the surface diffusion mask over land by replacing the total heat capacity with 1 J K⁻¹ (see `safe_capacity` in `_build_single_layer_operator`). That prevents division by zero, but it also means coast-adjacent ocean cells see a larger conductance than their neighbors because the harmonic mean still samples κ·C from the land cell before being truncated. The resulting flux imbalance pumps energy into the ocean column until the radiative loss compensates, again raising the global mean.

## 3. Total heat capacity scaling differs between radiation and diffusion terms

Radiation tendencies divide by the per-area heat capacity `C` (J m⁻² K⁻¹), whereas diffusion divides fluxes by `C × area`. That mismatch showed up only after we introduced explicit cell areas. Large-area tropical cells therefore react more sluggishly to diffusion than to radiation, so the Newton solve can overshoot toward a warm bias before the global balance reins it in.

## 4. The harmonic-mean diffusivity drops land neighbours entirely

When one side of an interface is masked (κ·C = 0), the harmonic mean collapses to zero. That removes the land cell from the coupled system while the ocean side still keeps its diagonal subtraction term, so total conductance in the ocean row exceeds the sum of off-diagonal entries. The sparse Jacobian then fails to conserve energy exactly, letting the solver settle at a hotter equilibrium.

## 5. Under-relaxed fixed-point iteration interacts poorly with the stiff matrix

Even though the annual map conserves energy, the extremely stiff diffusion matrix (row sums near ±500 s⁻¹ in the test problem) forces the under-relaxed fixed-point step to take many iterations. The adaptive damping may boost the iterate in warm regions first, leading to a quasi-steady 300 K plateau before the cooler poles catch up. Diffusion-free runs avoid the stiffness and therefore converge to the expected ~15 °C mean.

These suspects all trace back to the geometry-aware diffusion rewrite; validating them will require instrumenting the flux assembly and comparing the implied global energy budget against the radiation-only baseline.
