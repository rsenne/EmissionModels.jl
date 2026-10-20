# ACDC Model Selection

The **Accumulated Cutoff Discrepancy Criterion** (ACDC, Li et al. 2026) is a
robust model-selection method for hidden Markov models. Instead of penalizing
likelihood by parameter count (AIC/BIC), ACDC scores each state by how far its
*stochastic drivers* deviate from uniformity.

## Idea

If a fitted model is correctly specified, the probability integral transform
(PIT) of each observation under its generating emission is uniform on ``[0,1]``.
ACDC recovers these drivers per state, measures their discrepancy from
``U([0,1]^D)``, and selects the number of states minimizing the accumulated loss
``\sum_k \max(0, \hat{D}_k - \rho)`` at cutoff ``\rho``, breaking ties toward
fewer states.

This works with any `AbstractHMM` from
[HiddenMarkovModels.jl](https://github.com/JuliaStats/HiddenMarkovModels.jl) whose
emissions are standard `Distributions` (continuous via the CDF, discrete via a
randomized PIT, `MvNormal` via the Cholesky-Rosenblatt transform) or emission
types from this package, except `CalciumEmission`.

## Example

```julia
using EmissionModels, HiddenMarkovModels, Distributions

hmm = HMM([0.5, 0.5], [0.95 0.05; 0.05 0.95],
          [Normal(-4.0, 1.0), Normal(4.0, 1.0)])
_, obs_seq = rand(hmm, 3000)

# Fit K = 1, 2, 3 and score each state's drivers; small means well specified.
results = map(1:3) do K
    trans = K == 1 ? ones(1, 1) : [i == j ? 0.9 : 0.1 / (K - 1) for i in 1:K, j in 1:K]
    dists = [Normal(quantile(obs_seq, (2k - 1) / (2K)), 1.0) for k in 1:K]
    hmm_est, _ = baum_welch(HMM(fill(1 / K, K), trans, dists), obs_seq)
    component_discrepancies(hmm_est, obs_seq, KSDiscrepancy())
end

# Pick the number of states with smallest ACDC loss at cutoff ρ.
K = acdc_select(results, 0.05)
```

To score a candidate set of fitted models, build one [`ACDCResult`](@ref) per
model (ordered by state count) and pass the vector to [`acdc_select`](@ref).
The internal helper `EmissionModels.get_critical_rho_values` returns the cutoffs
at which the selection changes.

## Custom emissions

To use ACDC with an emission type that is not a standard `Distributions` object,
add a method `EmissionModels._emission_to_driver(rng, dist, obs)` returning the
driver vector in ``[0,1]``; for a controlled emission, add the 4-argument form
`(rng, dist, obs, control)`. This hook is internal and unstable (see API.md).
Draw any randomness (e.g. for a randomized PIT) from `rng` so driver recovery
stays reproducible.

## API Reference

### Model selection

```@docs
component_discrepancies
acdc_select
```

### Discrepancy measures

```@docs
KLDiscrepancy
KSDiscrepancy
WassersteinDiscrepancy
SquaredErrorDiscrepancy
MMDDiscrepancy
```

### Result type

```@docs
ACDCResult
```

### Internal helpers

These are not exported; reach them through `EmissionModels.` if needed. They
back [`component_discrepancies`](@ref) and [`acdc_select`](@ref) and are
documented for reference rather than as part of the stable public API.

```@docs
EmissionModels.stochastic_drivers
EmissionModels.compute_discrepancy
EmissionModels.acdc_loss
EmissionModels.get_critical_rho_values
EmissionModels.StochasticDriverResult
EmissionModels.ComponentDiscrepancy
```
