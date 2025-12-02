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
        # Use logsumexp trick for numerical stability
        log_π = log(dist.π)
        log_1minus_π_poisson_0 = log(1 - dist.π) - dist.λ

        # log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) + exp(b-max))
        m = max(log_π, log_1minus_π_poisson_0)
        return m + log(exp(log_π - m) + exp(log_1minus_π_poisson_0 - m))
    else
        # P(X=k) = (1-π) * Poisson(λ)(k)
        # log P(X=k) = log(1-π) + k*log(λ) - λ - log(k!)
        return log(1 - dist.π) + x * log(dist.λ) - dist.λ - loggamma(x + 1)
    end
end

# Random number generation
"""
	rand(rng::AbstractRNG, dist::PoissonZeroInflated)

Generate a random sample from the zero-inflated Poisson distribution.

Uses a two-stage process: first decide if observation comes from zero-inflation,
then sample from Poisson if not.
"""
function Random.rand(rng::Random.AbstractRNG, dist::PoissonZeroInflated)
    # First, decide if we get a structural zero
    if rand(rng) < dist.π
        return 0
    else
        # Sample from Poisson(λ)
        return _rand_poisson(rng, dist.λ)
    end
end

# Knuth's algorithm for Poisson sampling
function _rand_poisson(rng::Random.AbstractRNG, λ::Real)
    if λ < 30
        # Knuth's algorithm for small λ
        L = exp(-λ)
        k = 0
        p = 1.0
        while p > L
            k += 1
            p *= rand(rng)
        end
        return k - 1
    else
        # Use normal approximation for large λ
        return max(0, round(Int, randn(rng) * sqrt(λ) + λ))
    end
end

# Parameter estimation
"""
	fit!(dist::PoissonZeroInflated, obs_seq, weight_seq)

Fit the zero-inflated Poisson parameters using weighted EM algorithm.

# Arguments
- `dist`: PoissonZeroInflated instance to update in-place
- `obs_seq`: Sequence of integer observations
- `weight_seq`: Sequence of weights (typically posterior state probabilities from HMM)

# Algorithm
Uses EM to estimate π and λ:
- E-step: Compute posterior probability that each zero is structural vs sampling zero
- M-step: Update π and λ based on weighted responsibilities
"""
function StatsAPI.fit!(
    dist::PoissonZeroInflated, obs_seq::AbstractVector, weight_seq::AbstractVector
)
    length(obs_seq) == length(weight_seq) ||
        throw(DimensionMismatch("obs_seq and weight_seq must have the same length"))

    # Handle edge cases
    total_weight = sum(weight_seq)
    if total_weight == 0 || isempty(obs_seq)
        return dist
    end

    # Separate zeros and non-zeros
    zero_mask = obs_seq .== 0
    n_zeros = sum(zero_mask)

    if n_zeros == length(obs_seq)
        # All zeros: set high π, arbitrary small λ
        dist.π = 0.9
        dist.λ = 0.1
        return dist
    end

    # EM algorithm for parameter estimation
    max_iter = 100
    tol = 1e-6

    for iter in 1:max_iter
        old_λ = dist.λ
        old_π = dist.π

        # E-step: Compute responsibility that each zero is structural
        # P(structural zero | X=0) = π / (π + (1-π)exp(-λ))
        posterior_structural_zero = zeros(eltype(dist.λ), length(obs_seq))
        for i in eachindex(obs_seq)
            if obs_seq[i] == 0
                prob_structural = dist.π
                prob_sampling = (1 - dist.π) * exp(-dist.λ)
                total = prob_structural + prob_sampling
                posterior_structural_zero[i] = prob_structural / total
            else
                posterior_structural_zero[i] = 0.0
            end
        end

        # M-step: Update parameters
        # Update π: weighted average of structural zero probabilities
        weighted_structural_zeros = sum(
            weight_seq[i] * posterior_structural_zero[i] for i in eachindex(obs_seq)
        )
        dist.π = weighted_structural_zeros / total_weight

        # Clamp π to valid range
        dist.π = clamp(dist.π, 1e-10, 1 - 1e-10)

        # Update λ: weighted mean of non-structural observations
        # Only non-structural zeros and all positive counts contribute
        weight_non_structural = sum(
            weight_seq[i] * (1 - posterior_structural_zero[i]) for i in eachindex(obs_seq)
        )

        if weight_non_structural > 0
            weighted_sum = sum(
                weight_seq[i] * obs_seq[i] * (1 - posterior_structural_zero[i]) for
                i in eachindex(obs_seq)
            )
            dist.λ = weighted_sum / weight_non_structural
        else
            dist.λ = 1.0  # Fallback if no non-structural observations
        end

        # Ensure λ stays positive
        dist.λ = max(dist.λ, 1e-10)

        # Check convergence
        if abs(dist.λ - old_λ) < tol && abs(dist.π - old_π) < tol
            break
        end
    end

    return dist
end

