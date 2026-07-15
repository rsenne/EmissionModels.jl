# Priors

## Available priors

### `RidgePrior(λ)`

L2 (ridge) penalization ``\frac{λ}{2} \|\beta\|_2^2``, i.e. a Gaussian prior
``β \sim \mathcal{N}(0, λ^{-1} I)``.

```julia
using EmissionModels: RidgePrior

β = zeros(3)
glm = GaussianGLM(β, 1.0, RidgePrior(0.5))  # ridge with λ = 0.5
```

## Implementing a custom prior

A prior must implement three methods:

```julia
# Negative log-prior value (up to an additive constant).
neglogprior(prior::MyPrior, β)

# Accumulate the gradient of the negative log-prior into `g` (use +=).
neglogprior_grad!(prior::MyPrior, g, β)

# Accumulate the Hessian of the negative log-prior into `H` (use +=).
neglogprior_hess!(prior::MyPrior, H, β)
```

The gradient and Hessian methods accumulate into their output arguments rather
than overwriting them, so a prior that wraps several penalties can simply call
each one in turn.

### Example: Lasso prior (L1 penalty)

The L1 penalty is not differentiable at 0, so a smooth Newton solver needs a
stand-in for the missing curvature:

```julia
struct LassoPrior{T}
    λ::T
end

neglogprior(p::LassoPrior, β) = p.λ * sum(abs, β)

function neglogprior_grad!(p::LassoPrior, g, β)
    for j in eachindex(β)
        g[j] += p.λ * sign(β[j])
    end
end

function neglogprior_hess!(p::LassoPrior, H, β)
    # No curvature away from 0; add a large diagonal entry at exact zeros
    # to keep the Newton step well-defined.
    for j in eachindex(β)
        if β[j] == 0
            H[j, j] += 1e6
        end
    end
end

# Usage:
glm = GaussianGLM(zeros(3), 1.0, LassoPrior(0.5))
```

## API Reference

```@docs
AbstractPrior
NoPrior
RidgePrior
neglogprior
neglogprior_grad!
neglogprior_hess!
```
