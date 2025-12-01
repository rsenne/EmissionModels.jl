# EmissionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://rsenne.github.io/EmissionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://rsenne.github.io/EmissionModels.jl/dev/)
[![Build Status](https://github.com/rsenne/EmissionModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rsenne/EmissionModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/rsenne/EmissionModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/rsenne/EmissionModels.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## What purpose does this package serve?

HiddenMarkovModels.jl is completely *generic*. This means users can have models that emit generic julia objects. This level of expressiveness is allowed through the ability to create custom emission models (i.e., the models that describe how observvations are generated conditioned on the current latent state of an HMM). HiddenMarkovModels.jl expects emission types to implement a small set of methods.

```julia
Random.rand(rng::AbstractRNG, dist::EmissionModel)
DensityInterface.DensityKind(::EmissionModel) = HasDensity()
DensityInterface.logdensityof(dist::EmissionModel, obs)
StatsAPI.fit!(dist::EmissionModel, obs_seq, weight_seq)
```

This package supplies many models that already implement those methods and are thus ready-to-use with HiddenMarkovModels.jl.

## Installation

This package is not yet registered on the Julia REPL. To add directly from GitHub (latest main):

```julia
Pkg.add(url="https://github.com/rsenne/EmissionModels.jl")
```

## Contributing

Contributions are very welcome. Suggested ways to help:

- Open an issue for bugs or feature requests.
- Submit a pull request with tests when adding features or fixing bugs.
- Improve examples and documentation under `docs/`.

When contributing, please follow the repository coding style and add tests for new behavior.

## License

This project is licensed under the terms in the `LICENSE` file in the repository root.

---
*Thank you for using EmissionModels.jl — feedback and contributions appreciated.*
