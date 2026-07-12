# EmissionModels.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/EmissionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/EmissionModels.jl/dev/)
[![Build Status](https://github.com/rsenne/EmissionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rsenne/EmissionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/EmissionModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/EmissionModels.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A Julia package providing emission models for [HiddenMarkovModels.jl](https://github.com/JuliaStats/HiddenMarkovModels.jl). It supplies ready-to-use distributions that describe how observations are generated conditioned on the HMM's latent states.

## Quick start

```julia
using Pkg
Pkg.add("HiddenMarkovModels")
Pkg.add(url="https://github.com/rsenne/EmissionModels.jl")

using EmissionModels
using HiddenMarkovModels

# Create an emission model
dist = PoissonZeroInflated(5.0, 0.3)

# Sample, evaluate densities, or fit to data
x = rand(dist)
logp = logdensityof(dist, x)
fit!(dist, observations, weights)
```

## Distribution models

All types implement the `HiddenMarkovModels` emission interface (`rand`, `logdensityof`, `fit!`).

### Count data

| Type | Description |
|------|-------------|
| `PoissonZeroInflated(λ, π)` | Zero-inflated Poisson for excess zeros in count data. |

### Multivariate continuous

| Type | Description |
|------|-------------|
| `MultivariateT(μ, Σ, ν)` | Full-covariance multivariate Student's t. |
| `MultivariateTDiag(μ, σ², ν)` | Diagonal-covariance multivariate Student's t. |

### GLM emissions (observation depends on a control vector)

| Type | Description |
|------|-------------|
| `GaussianGLM(β, σ²)` | Linear regression with Gaussian noise. |
| `BernoulliGLM(β)` | Logistic regression for binary data. |
| `PoissonGLM(β)` | Log-linear regression for count data. |
| `MvGaussianGLM(B, Σ)` | Multivariate linear regression with shared full covariance. |
| `MvBernoulliGLM(B)` | Independent logistic regressions, one per output dimension. |
| `MvPoissonGLM(B)` | Independent log-linear regressions, one per output dimension. |

The univariate types carry a coefficient vector `β` and emit scalars; the
multivariate types carry a `p × k` coefficient matrix `B` and emit length-`k`
vectors. All of them support regularization via priors:

```julia
β = zeros(3)
glm = GaussianGLM(β, 1.0, RidgePrior(0.5))  # L2 regularization
```

Each GLM is fit via `fit!(glm, y, w; control_seq=X)`, where `control_seq` (design matrix `X`) maps latent states to the regression covariates. Since the GLMs subtype `ControlledEmission`, a vector of them also works directly as the emissions of a `ControlledEmissionHMM`.

## ACDC model selection

The Accumulated Cutoff Discrepancy Criterion (ACDC) picks the number of hidden states without penalizing likelihood by parameter count. It inverts each fitted emission through the probability integral transform to recover per-state "stochastic drivers", which are uniform when the model is well specified, and selects the smallest state count whose per-state discrepancies from uniform all fall below a cutoff.

```julia
using EmissionModels, HiddenMarkovModels, Distributions

hmm = HMM([0.5, 0.5], [0.95 0.05; 0.05 0.95],
          [Normal(-4.0, 1.0), Normal(4.0, 1.0)])
_, obs_seq = rand(hmm, 3000)

result = component_discrepancies(hmm, obs_seq, KSDiscrepancy())
K = acdc_select([result], 0.05)
```

It works with any `AbstractHMM` whose emissions are standard `Distributions` objects or the types in this package (including the GLMs, via their covariates). Several discrepancy measures are available: `KSDiscrepancy`, `KLDiscrepancy`, `WassersteinDiscrepancy`, `MMDDiscrepancy`, and `SquaredErrorDiscrepancy`.

## Creating custom emission models

`HiddenMarkovModels.jl` accepts any type that implements the following interface:

```julia
Random.rand(rng::AbstractRNG, dist::MyEmission)
DensityInterface.DensityKind(::MyEmission)      # return HasDensity()
DensityInterface.logdensityof(dist::MyEmission, obs)
StatsAPI.fit!(dist::MyEmission, obs_seq, weight_seq)
```

See the [documentation](https://rsenne.github.io/EmissionModels.jl/dev/) for details.

## Installation

EmissionModels.jl is not yet registered. Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/rsenne/EmissionModels.jl")
```

## Contributing

Contributions are welcome. Please follow the [Julia Blue Style](https://github.com/JuliaDiff/BlueStyle) and add tests for new behavior. Pull requests and issues are appreciated.

## License

EmissionModels.jl is licensed under the terms of the `LICENSE` file.
