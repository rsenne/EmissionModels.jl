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

        # log(exp(a) + exp(b))
        return logaddexp(log_pi, log_1minus_pi_poisson_0)
    else
        # log P(X=k) = log(1-π) + k*log(λ) - λ - log(k!)
        return log(1 - dist.π) + x * log(dist.λ) - dist.λ - logfactorial(x)
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
        return rand(rng, Poisson(dist.λ))
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
    dist::PoissonZeroInflated,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector;
    max_iter=100,
    tol=1e-6,
)
    length(obs_seq) == length(weight_seq) ||
        throw(DimensionMismatch("obs_seq and weight_seq must have the same length"))

    # Handle edge cases
    total_weight = sum(weight_seq)
    if total_weight == 0 || isempty(obs_seq)
        return dist
    end

    epsilon = eps(typeof(dist.λ))
    zero_mask = obs_seq .== 0
    if sum(zero_mask) == length(obs_seq)
        dist.π = 1.0 - epsilon
        dist.λ = epsilon
        return dist
    end

    for iter in 1:max_iter
        old_λ = dist.λ
        old_π = dist.π

        # Initialize M-step accumulators
        weighted_structural_zeros = zero(total_weight)
        weighted_sum_x = zero(total_weight)
        weight_non_structural = zero(total_weight)

        prob_sampling_zero = exp(-dist.λ)

        # Integrated E-step and M-step (avoids allocation)
        for i in eachindex(obs_seq)
            w = weight_seq[i]
            x = obs_seq[i]

            if x == 0
                # E-step: Compute P(structural | X=0)
                prob_structural = dist.π
                prob_sampling = (1 - dist.π) * prob_sampling_zero
                total = prob_structural + prob_sampling

                # Check for numerical stability issues (total ~= 0)
                if total < epsilon
                    p_structural_zero = 1.0 # Assume structural if all probs near zero
                else
                    p_structural_zero = prob_structural / total
                end

                # M-step update for π
                weighted_structural_zeros += w * p_structural_zero

                # M-step update for λ denominator: (1 - P(structural | X=0)) * w
                weight_non_structural += w * (1 - p_structural_zero)

            else # x > 0
                # E-step: P(structural | X=x>0) = 0

                # M-step update for λ denominator and numerator
                weight_non_structural += w
                weighted_sum_x += w * x
            end
        end

        # M-step: Final parameter update
        dist.π = weighted_structural_zeros / total_weight
        dist.π = clamp(dist.π, epsilon, 1 - epsilon)

        if weight_non_structural > 0
            dist.λ = weighted_sum_x / weight_non_structural
        else
            dist.λ = epsilon # Fallback if no non-structural observations
        end
        dist.λ = max(dist.λ, epsilon)

        # Check convergence
        if abs(dist.λ - old_λ) < tol && abs(dist.π - old_π) < tol
            break
        end
    end

    return dist
end
