"""
    AbstractGLM

Abstract type for Generalized Linear Model emission distributions.

GLM subtypes should implement the HiddenMarkovModels.jl interface:
- `DensityInterface.DensityKind(::YourGLM)` - Declare HasDensity()
- `DensityInterface.logdensityof(glm, obs)` - Compute log density
- `Random.rand(rng, glm, X)` - Generate samples given covariates
- `StatsAPI.fit!(glm, obs_seq, weight_seq)` - Update parameters
"""
abstract type AbstractGLM end

mutable struct GaussianGLM{T<:Real} <: AbstractGLM
    β::Vector{T}       # Coefficients
    σ2::T               # Variance
end

DensityInterface.DensityKind(::GaussianGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(reg::GaussianGLM, y::Real; control_seq::AbstractVector{<:Real})
    μ = dot(reg.β, control_seq)
    return -0.5 * log(2π * reg.σ2) - 0.5 * ((y-μ)^2 / reg.σ2)
end

function Random.rand(rng::AbstractRNG, reg::GaussianGLM; control_seq::AbstractVector{<:Real})
    return rand(rng, Normal(dot(reg.β, control_seq), sqrt(reg.σ2)))
end

function StatsAPI.fit!(reg::GaussianGLM, obs_seq::AbstractVector{<:Real}, weights::AbstractVector{<:Real};
                       control_seq::AbstractMatrix{<:Real})
    # Fit coefficients using weighted least squares
    W = Diagonal(weights)
    XWX = control_seq' * W * control_seq
    XWy = control_seq' * W * obs_seq
    reg.β = XWX \ XWy

    # Update variance
    residuals = obs_seq .- control_seq * reg.β
    reg.σ2 = sum(weights .* (residuals .^ 2)) / sum(weights)

    return reg
end