#= Stochastic-driver recovery for individual emission distributions.

   Given an emission distribution and an observation, `_emission_to_driver`
   returns the driver vector ε ∈ [0,1]^D via the probability integral transform
   (PIT):
   - continuous univariate: ε = F(x);
   - discrete univariate: randomized PIT, ε ~ U(F(x⁻), F(x)), exactly uniform
     under the true model;
   - MvNormal: the Rosenblatt transform, which for a Gaussian reduces to
     whitening the residual with the Cholesky factor and pushing each coordinate
     through Φ.

   These methods dispatch on `Distributions` types, so ACDC works with any
   HiddenMarkovModels.jl HMM whose emissions are standard distributions. To
   support a custom emission type, add a `_emission_to_driver(dist, obs)` method
   returning a `Vector` of drivers in [0,1]. =#

# Clamp a PIT value strictly inside (0,1); the discrepancy measures map drivers
# through the probit transform, which is ±Inf at the boundary.
_clamp01(u::T) where {T<:Real} = clamp(u, eps(T), one(T) - eps(T))

# Draw a category index from probability vector `p` (assumed to sum to ≈1).
function _sample_categorical(p::AbstractVector{T}) where {T<:Real}
    u = rand(T)
    cumsum_p = zero(T)
    for i in eachindex(p)
        cumsum_p += p[i]
        u <= cumsum_p && return i
    end
    return lastindex(p)
end

# Recover the stochastic drivers for a single observation under emission `dist`.
# Returns a length-`D` vector of values in (0,1).
function _emission_to_driver end

# Continuous univariate: standard PIT through the CDF.
function _emission_to_driver(d::ContinuousUnivariateDistribution, obs::Real)
    return [_clamp01(float(cdf(d, obs)))]
end

# Discrete univariate: randomized PIT, ε ~ U(F(x⁻), F(x)).
function _emission_to_driver(d::DiscreteUnivariateDistribution, obs::Real)
    upper = float(cdf(d, obs))
    lower = float(cdf(d, obs - 1))
    return [_clamp01(lower + rand() * (upper - lower))]
end

# Multivariate normal: Rosenblatt transform = whiten residual, then push each
# whitened coordinate through Φ. Whitening with the lower-Cholesky factor yields
# independent N(0,1) coordinates, so the result is uniform under the true model.
function _emission_to_driver(d::AbstractMvNormal, obs::AbstractVector)
    μ = mean(d)
    L = cholesky(Symmetric(Matrix(cov(d)))).L
    z = L \ (collect(float.(obs)) .- μ)
    return [_clamp01(float(cdf(Normal(), zi))) for zi in z]
end

# Clear error for emission types without a PIT recipe, pointing the user at the
# extension hook.
function _emission_to_driver(d, obs)
    throw(
        ArgumentError(
            "ACDC has no `_emission_to_driver` method for emission of type " *
            "$(typeof(d)). Supported out of the box: continuous/discrete " *
            "univariate `Distributions` and `MvNormal`. Define " *
            "`EmissionModels._emission_to_driver(dist, obs)` returning a vector " *
            "of drivers in [0,1] to add support.",
        ),
    )
end
