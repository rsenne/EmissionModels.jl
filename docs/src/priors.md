# Priors

## Available priors

### `RidgePrior(λ)`

Implements L2 (ridge) penalization ``\frac{λ}{2} \|\beta\|_2^2``.

```julia
using EmissionModels: RidgePrior

β = zeros(3)
glm = GaussianGLM(β, 1.0, RidgePrior(0.5))  # ridge with λ = 0.5
```

## Implementing a custom prior

Every prior must conform to the following interface:

```julia
# Required methods

"""Return +neglog prior value (no penalty => 0)."""
neglogprior(prior::MyPrior, β)

"""Return + neglog prior gradient (in-place update to `g`)."""
neglogprior_grad!(prior::MyPrior, g, β)

"""Return + neglog prior hessian (in-place update to `H`)."""
neglogprior_hess!(prior::MyPrior, H, β)
```

### Example: Lasso prior (L1 penalty)

L1 penalties are non-differentiable at 0, so they only participate in the gradient.

```julia
struct LassoPrior{T}
    λ::T
end

neglogprior(p::LassoPrior, β) = p.λ * sum(abs, β)

function neglogprior_grad!(p::LassoPrior, g, β)
    @inbounds for j in eachindex(β)
        g[j] += p.λ * sign(β[j])
    end
end

function neglogprior_hess!(p::LassoPrior, H, β)
    # Lasso is not twice differentiable at β == 0.
    # Use the subdifferential (zero everywhere except β == 0).
    @inbounds for j in eachindex(β)
        if β[j] == 0
            H[j, j] += 1e6  # effectively ∞ at 0
        end
    end
end

# Usage:
glm = GaussianGLM(zeros(3), 1.0, LassoPrior(0.5))
```

## Composing priors

Priors can be composed with `ComposedPrior`:

```julia
prior = ComposedPrior(RidgePrior(0.5), LassoPrior(0.1))
glm = GaussianGLM(zeros(5), 1.0, prior)
```

The composed prior simply sums the individual penalties, gradients, and Hessians.
