# EmissionModels

## What is EmissionModels?

EmissionModels.jl provides ready-to-use emission models for
[HiddenMarkovModels.jl](https://github.com/JuliaStats/HiddenMarkovModels.jl). An
*emission model* (or *emission distribution*) describes how observations are
generated conditioned on the latent state of a hidden Markov model.

The package covers count data, multivariate continuous data, and generalized
linear models (GLMs), so you can build HMMs without writing custom emission
code from scratch.

## Quick start

```julia
using EmissionModels
using HiddenMarkovModels

# A zero-inflated Poisson emission
emission = PoissonZeroInflated(5.0, 0.3)

# Sample and evaluate the log density
x = rand(emission)
lp = logdensityof(emission, x)

# Use in a two-state HMM
init = [0.5, 0.5]
trans = [0.8 0.2; 0.3 0.7]
emissions = [PoissonZeroInflated(5.0, 0.3), PoissonZeroInflated(20.0, 0.1)]
hmm = HMM(init, trans, emissions)

# Fit with Baum-Welch
hmm_est, loglik_trace = baum_welch(hmm, obs_seq)
```

## Where to go from here

- [Tutorials](examples/basics.md): worked examples, from built-in emissions
  to GLMs and model selection.
- [Distributions](distributions.md): all available count and multivariate
  emission models.
- [GLM Emissions](glm.md): regression-based emissions that take a control
  (design) vector.
- [Priors](priors.md): regularization for GLM emissions.
- [ACDC Model Selection](acdc.md): choose the number of hidden states.
- [Custom Emission Models](custom.md): write your own emission types.
