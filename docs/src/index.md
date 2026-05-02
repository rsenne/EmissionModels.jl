# EmissionModels

## What is EmissionModels?

EmissionModels.jl is a Julia package that provides ready-to-use **emission models** for [HiddenMarkovModels.jl](https://github.com/baggepinn/HiddenMarkovModels.jl). An *emission model* (or *emission distribution*) describes the probabilistic relationship between the latent state of a hidden Markov model and its observable output.

This package supplies a collection of well-tested distributions — from count and multivariate continuous models to generalized linear models (GLMs) — so you can get started with HMM modeling without writing custom emission code from scratch.

## Quick start

```julia
using EmissionModels
using HiddenMarkovModels

# --- A simple count emission model ---
emission = PoissonZeroInflated(5.0, 0.3)

# Sample
x = rand(emission)

# Evaluate log-density
lp = logdensityof(emission, x)

# --- Use in an HMM (example) ---
# Transition matrix (2 states → 2 states)
T = [0.8 0.2; 0.3 0.7]

# Two different emission models for the two states
emissions = [PoissonZeroInflated(5.0, 0.3), PoissonZeroInflated(20.0, 0.1)]

hmm = HiddenMarkModel(T, emissions)

# Fit to observed sequence (uses Viterbi / forward-backward under the hood)
fit!(hmm, observations)
```

## Where to go from here

- **[Distributions](distributions.md)** — Browse all available count, multivariate, and GLM emission models.
- **[GLM Emissions](glm.md)** — Learn about regression-based emissions that take a control (design) vector.
- **[Priors](priors.md)** — Add regularization to GLM emissions.
- **[Custom Emission Models](custom.md)** — Build your own emission types and plug them into the HMM framework.
