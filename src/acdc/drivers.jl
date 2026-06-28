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
   - PoissonZeroInflated: randomized PIT against the ZIP CDF
     F(k) = π + (1-π)·F_Poisson(k).
   - MultivariateT / MultivariateTDiag: the conditional-t Rosenblatt — whiten
     the residual to a standardized spherical t, then push each coordinate
     through its conditional Student-t CDF. (A Gaussian-style whiten-then-Φ is
     wrong here: whitened t-coordinates are uncorrelated but not independent.)
   - GLMs (`AbstractGLM`): conditional emissions f(y | x). Recovered through a
     3-arg `_emission_to_driver(dist, obs, x)` that reduces the GLM at covariate
     `x` to the standard emission it is (Normal / Bernoulli / Poisson / MvNormal)
     and reuses the recipes above.

   These methods dispatch on `Distributions` types and the package's own emission
   types, so ACDC works with any HiddenMarkovModels.jl HMM whose emissions are
   standard distributions or those types. To support a further custom emission
   type, add a `_emission_to_driver(dist, obs)` method (or the 3-arg form for a
   covariate-dependent emission) returning a `Vector` of drivers in [0,1]. =#

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

# Zero-inflated Poisson: randomized PIT against the ZIP CDF. The mixture CDF is
# F(k) = π + (1-π)·F_Poisson(k) for k ≥ 0; the structural-zero mass collapses
# into F(0), so a true zero still maps uniformly into [0, F(0)].
function _emission_to_driver(d::PoissonZeroInflated, obs::Real)
    pois = Poisson(d.λ)
    upper = d.π + (1 - d.π) * cdf(pois, obs)
    lower = obs > 0 ? d.π + (1 - d.π) * cdf(pois, obs - 1) : zero(upper)
    return [_clamp01(float(lower + rand() * (upper - lower)))]
end

# Multivariate Student-t (full and diagonal scale): the conditional-t Rosenblatt
# transform. Whitening the residual yields a standardized spherical t whose
# coordinates are uncorrelated but share the common χ² scale, so they are NOT
# independent — pushing each through Φ (as for a Gaussian) would be wrong. The
# correct map sends each whitened coordinate through its conditional Student-t
# CDF, which has ν+(j-1) degrees of freedom and scale √((ν+Σ_{i<j} zᵢ²)/(ν+j-1)).
function _emission_to_driver(d::MultivariateT, obs::AbstractVector)
    z = d.Σ_chol.L \ (collect(float.(obs)) .- d.μ)
    return _conditional_t_drivers(z, d.ν)
end

function _emission_to_driver(d::MultivariateTDiag, obs::AbstractVector)
    z = (collect(float.(obs)) .- d.μ) ./ sqrt.(d.σ²)
    return _conditional_t_drivers(z, d.ν)
end

# Sequential conditional Student-t PIT for a whitened spherical-t residual `z`.
function _conditional_t_drivers(z::AbstractVector, ν::Real)
    T = float(promote_type(eltype(z), typeof(ν)))
    ε = Vector{T}(undef, length(z))
    sumsq = zero(T)  # Σ_{i<j} zᵢ²
    for (j, zj) in enumerate(z)
        dof = ν + (j - 1)
        scale = sqrt((ν + sumsq) / dof)
        ε[j] = _clamp01(T(cdf(TDist(dof), zj / scale)))
        sumsq += T(zj)^2
    end
    return ε
end

#=
GLMs are conditional emissions f(y | x); the driver is the conditional PIT
F(y | x). Rather than re-deriving a randomized PIT per family, we reduce each
GLM at covariate `x` to the standard `Distributions` emission it is, then route
through the recipes above. 

The covariate reaches drivers via a 3-arg `_emission_to_driver(dist, obs, x)`.
The generic method ignores `x`, so every non-GLM emission (Normal, ZIP, MvT, …)
is unaffected; only GLMs override it.
=#

# Conditional `Distributions` emission of a GLM at covariate `x`. `β`/`B` already
# absorb any intercept (it must be a column of `x`), matching the GLM densities.
_conditional(g::GaussianGLM, x) = Normal(dot(g.β, x), sqrt(g.σ2))
_conditional(g::BernoulliGLM, x) = Bernoulli(logistic(dot(g.β, x)))
_conditional(g::PoissonGLM, x) = Poisson(exp(dot(g.β, x)))
_conditional(g::MvGaussianGLM, x) = MvNormal(g.B' * x, g.Σ)

# Generic 3-arg form: covariate ignored (all non-GLM emissions defer here).
_emission_to_driver(dist, obs, control) = _emission_to_driver(dist, obs)

# Univariate and correlated-multivariate (Gaussian) GLMs: reduce, then reuse the
# conditional PIT. MvGaussianGLM has correlated outputs ⇒ the MvNormal Rosenblatt.
_emission_to_driver(g::AbstractGLM, obs, x) = _emission_to_driver(_conditional(g, x), obs)

# Independent-by-column multivariate GLMs: stack per-coordinate randomized PITs.
# The columns are independent given `x`, so the stacked drivers are genuinely
# independent uniforms — no joint Rosenblatt needed.
function _emission_to_driver(g::MvBernoulliGLM, obs::AbstractVector, x)
    η = g.B' * x
    return [_emission_to_driver(Bernoulli(logistic(η[j])), obs[j])[1] for j in eachindex(η)]
end

function _emission_to_driver(g::MvPoissonGLM, obs::AbstractVector, x)
    η = g.B' * x
    return [_emission_to_driver(Poisson(exp(η[j])), obs[j])[1] for j in eachindex(η)]
end

# 2-arg form on a GLM: no covariate to condition on — point at the 3-arg hook.
function _emission_to_driver(g::AbstractGLM, obs)
    throw(
        ArgumentError(
            "GLM emission $(typeof(g)) is conditional on a covariate — call " *
            "`EmissionModels._emission_to_driver(glm, obs, control)` with the " *
            "covariate vector for that observation.",
        ),
    )
end

# Clear error for emission types without a PIT recipe, pointing the user at the
# extension hook.
function _emission_to_driver(d, obs)
    throw(
        ArgumentError(
            "ACDC has no `_emission_to_driver` method for emission of type " *
            "$(typeof(d)). Supported out of the box: continuous/discrete " *
            "univariate `Distributions`, `MvNormal`, `PoissonZeroInflated`, and " *
            "`MultivariateT`/`MultivariateTDiag`. Define " *
            "`EmissionModels._emission_to_driver(dist, obs)` returning a vector " *
            "of drivers in [0,1] to add support.",
        ),
    )
end
