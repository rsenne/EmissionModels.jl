"""
	PoissonZeroInflated{T<:Real}

Zero-Inflated Poisson distribution for count data with excess zeros.

# Fields
- `λ::T`: Rate parameter of the Poisson component (λ > 0)
- `π::T`: Probability of extra zeros (0 ≤ π ≤ 1)

# Model
The distribution generates zeros with probability π + (1-π)exp(-λ), and
count k > 0 with probability (1-π) * Poisson(λ)(k).

# Example
```julia
using Random
dist = PoissonZeroInflated(3.0, 0.2)
samples = rand(dist, 100)
logp = logdensityof(dist, 0)
```
"""
mutable struct PoissonZeroInflated{T<:Real}
    λ::T
    π::T

    function PoissonZeroInflated{T}(λ::T, π::T) where {T<:Real}
        λ > 0 || throw(ArgumentError("λ must be positive, got $λ"))
        0 ≤ π ≤ 1 || throw(ArgumentError("π must be in [0,1], got $π"))
        return new{T}(λ, π)
    end
end

PoissonZeroInflated(λ::T, π::T) where {T<:Real} = PoissonZeroInflated{T}(λ, π)
PoissonZeroInflated(λ::Real, π::Real) = PoissonZeroInflated(promote(λ, π)...)

# DensityInterface implementation
DensityInterface.DensityKind(::PoissonZeroInflated) = DensityInterface.HasDensity()

"""
	logdensityof(dist::PoissonZeroInflated, x::Integer)

Compute the log probability mass of observing `x` under the zero-inflated Poisson distribution.

For x = 0: log(π + (1-π)exp(-λ))
For x > 0: log(1-π) + x*log(λ) - λ - loggamma(x+1)
"""
function DensityInterface.logdensityof(dist::PoissonZeroInflated, x::Integer)
    x >= 0 || return oftype(dist.λ, -Inf)

    if x == 0
        # P(X=0) = π + (1-π)exp(-λ)
        log_pi = log(dist.π)
        log_1minus_pi_poisson_0 = log(1 - dist.π) - dist.λ
        return logaddexp(log_pi, log_1minus_pi_poisson_0)
    else
        # log P(X=k) = log(1-π) + k*log(λ) - λ - log(k!)
        return log(1 - dist.π) + x * log(dist.λ) - dist.λ - logfactorial(x)
    end
end

# Random number generation
"""
	rand(rng::AbstractRNG, dist::PoissonZeroInflated)

Generate a random sample from the zero-inflated Poisson distribution: a
structural zero with probability π, otherwise a draw from Poisson(λ).
"""
function Random.rand(rng::Random.AbstractRNG, dist::PoissonZeroInflated)
    if rand(rng) < dist.π
        return 0
    else
        return rand(rng, Poisson(dist.λ))
    end
end

#= Array forms (`rand(dist, n)` etc.) go through Random's sampler machinery,
   which needs the sample eltype and a method on the default trivial sampler. =#
Base.eltype(::Type{<:PoissonZeroInflated}) = Int
function Random.rand(rng::AbstractRNG, sp::Random.SamplerTrivial{<:PoissonZeroInflated})
    return rand(rng, sp[])
end

# Parameter estimation
"""
	fit!(dist::PoissonZeroInflated, obs_seq, weight_seq)

Fit the zero-inflated Poisson parameters to weighted observations by EM, in
place. The E-step computes the posterior probability that each zero is
structural; the M-step updates π and λ from the weighted responsibilities.

# Arguments
- `dist`: PoissonZeroInflated instance to update in place
- `obs_seq`: sequence of integer observations
- `weight_seq`: per-observation weights (e.g. HMM posterior state probabilities)
"""
function StatsAPI.fit!(
    dist::PoissonZeroInflated,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector;
    max_iter=100,
    tol=1e-6,
)
    length(obs_seq) == length(weight_seq) ||
        throw(DimensionMismatch("obs_seq and weight_seq must have the same length"))

    total_weight = sum(weight_seq)
    if total_weight == 0 || isempty(obs_seq)
        return dist
    end

    epsilon = eps(typeof(dist.λ))
    if all(iszero, obs_seq)
        dist.π = 1.0 - epsilon
        dist.λ = epsilon
        return dist
    end

    for iter in 1:max_iter
        old_λ = dist.λ
        old_π = dist.π

        weighted_structural_zeros = zero(total_weight)
        weighted_sum_x = zero(total_weight)
        weight_non_structural = zero(total_weight)

        prob_sampling_zero = exp(-dist.λ)

        # E-step and M-step accumulation fused into one pass (no allocation).
        for i in eachindex(obs_seq)
            w = weight_seq[i]
            x = obs_seq[i]

            if x == 0
                # P(structural | X=0)
                prob_structural = dist.π
                prob_sampling = (1 - dist.π) * prob_sampling_zero
                total = prob_structural + prob_sampling
                if total < epsilon
                    p_structural_zero = 1.0 # treat as structural when both probs vanish
                else
                    p_structural_zero = prob_structural / total
                end

                weighted_structural_zeros += w * p_structural_zero
                weight_non_structural += w * (1 - p_structural_zero)
            else
                # A positive count cannot be structural.
                weight_non_structural += w
                weighted_sum_x += w * x
            end
        end

        dist.π = weighted_structural_zeros / total_weight
        dist.π = clamp(dist.π, epsilon, 1 - epsilon)

        if weight_non_structural > 0
            dist.λ = weighted_sum_x / weight_non_structural
        else
            dist.λ = epsilon # no non-structural mass left
        end
        dist.λ = max(dist.λ, epsilon)

        if abs(dist.λ - old_λ) < tol && abs(dist.π - old_π) < tol
            break
        end
    end

    return dist
end
