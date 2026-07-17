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
   - MvT / MvTDiag: the conditional-t Rosenblatt; whiten
     the residual to a standardized spherical t, then push each coordinate
     through its conditional Student-t CDF. (A Gaussian-style whiten-then-Φ is
     wrong here: whitened t-coordinates are uncorrelated but not independent.)
   - GLMs (`AbstractGLM`): conditional emissions f(y | x). Recovered through a
     4-arg `_emission_to_driver(rng, dist, obs, x)` that reduces the GLM at
     covariate `x` to the standard emission it is (Normal / Bernoulli / Poisson /
     MvNormal) and reuses the recipes above.

   These methods dispatch on `Distributions` types and the package's own emission
   types, so ACDC works with any HiddenMarkovModels.jl HMM whose emissions are
   standard distributions or those types. To support a further custom emission
   type, add a `_emission_to_driver(rng, dist, obs)` method (or the 4-arg form
   for a covariate-dependent emission) returning a `Vector` of drivers in [0,1].
   The `rng` feeds the randomized PITs so driver recovery is reproducible. =#

#= Clamp a PIT value strictly inside (0,1); the discrepancy measures map drivers
   through the probit transform, which is ±Inf at the boundary. =#
_clamp01(u::T) where {T<:Real} = clamp(u, eps(T), one(T) - eps(T))

#= Randomized-PIT draw ε ~ U(lower, upper), typed on the bounds so a Float32 /
   Dual pool stays in its element type instead of silently widening to Float64. =#
function _randomized_pit(rng::AbstractRNG, lower::Real, upper::Real)
    T = float(promote_type(typeof(lower), typeof(upper)))
    return lower + rand(rng, T) * (upper - lower)
end

# Draw a category index from probability vector `p` (assumed to sum to ≈1).
function _sample_categorical(rng::AbstractRNG, p::AbstractVector{T}) where {T<:Real}
    u = rand(rng, T)
    cumsum_p = zero(T)
    for i in eachindex(p)
        cumsum_p += p[i]
        u <= cumsum_p && return i
    end
    return lastindex(p)
end

#= Recover the stochastic drivers for a single observation under emission `dist`.
   Returns a length-`D` vector of values in (0,1). =#
function _emission_to_driver end

# Continuous univariate: standard PIT through the CDF (no randomness needed).
function _emission_to_driver(::AbstractRNG, d::ContinuousUnivariateDistribution, obs::Real)
    return [_clamp01(float(cdf(d, obs)))]
end

# Discrete univariate: randomized PIT, ε ~ U(F(x⁻), F(x)). `F(x⁻)` uses `cdf(d,
# obs - 1)`, which assumes unit-spaced integer support (Poisson/Binomial/
# Bernoulli); a distribution with a coarser support step would need its own
# predecessor.
function _emission_to_driver(rng::AbstractRNG, d::DiscreteUnivariateDistribution, obs::Real)
    upper = float(cdf(d, obs))
    lower = float(cdf(d, obs - 1))
    return [_clamp01(_randomized_pit(rng, lower, upper))]
end

#= Multivariate normal: Rosenblatt transform = whiten residual, then push each
   whitened coordinate through Φ. Whitening with the lower-Cholesky factor yields
   independent N(0,1) coordinates, so the result is uniform under the true model. =#
function _emission_to_driver(::AbstractRNG, d::AbstractMvNormal, obs::AbstractVector)
    μ = mean(d)
    L = cholesky(Symmetric(Matrix(cov(d)))).L
    z = L \ (collect(float.(obs)) .- μ)
    return [_clamp01(float(cdf(Normal(), zi))) for zi in z]
end

#= Zero-inflated Poisson: randomized PIT against the ZIP CDF. The mixture CDF is
   F(k) = π + (1-π)·F_Poisson(k) for k ≥ 0; the structural-zero mass collapses
   into F(0), so a true zero still maps uniformly into [0, F(0)]. =#
function _emission_to_driver(rng::AbstractRNG, d::PoissonZeroInflated, obs::Real)
    pois = Poisson(d.λ)
    upper = d.π + (1 - d.π) * cdf(pois, obs)
    lower = obs > 0 ? d.π + (1 - d.π) * cdf(pois, obs - 1) : zero(upper)
    return [_clamp01(_randomized_pit(rng, lower, upper))]
end

#= Multivariate Student-t (full and diagonal scale): the conditional-t Rosenblatt
   transform. Whitening the residual yields a standardized spherical t whose
   coordinates are uncorrelated but share the common χ² scale, so they are not
   independent and pushing each through Φ (as for a Gaussian) would be wrong. The
   correct map sends each whitened coordinate through its conditional Student-t
   CDF, which has ν+(j-1) degrees of freedom and scale √((ν+Σ_{i<j} zᵢ²)/(ν+j-1)). =#
function _emission_to_driver(::AbstractRNG, d::MvT, obs::AbstractVector)
    z = d.Σ_chol.L \ (collect(float.(obs)) .- d.μ)
    return _conditional_t_drivers(z, d.ν)
end

function _emission_to_driver(::AbstractRNG, d::MvTDiag, obs::AbstractVector)
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

The covariate reaches drivers via a 4-arg `_emission_to_driver(rng, dist, obs, x)`.
The generic method ignores `x`, so every non-GLM emission (Normal, ZIP, MvT, ...)
is unaffected; only GLMs override it.
=#

#= Conditional `Distributions` emission of a GLM at covariate `x`. `β`/`B` already
   absorb any intercept (it must be a column of `x`), matching the GLM densities. =#
_conditional(g::GaussianGLM, x) = Normal(dot(g.β, x), sqrt(g.σ2))
_conditional(g::BernoulliGLM, x) = Bernoulli(logistic(dot(g.β, x)))
_conditional(g::PoissonGLM, x) = Poisson(exp(dot(g.β, x)))
_conditional(g::MvGaussianGLM, x) = MvNormal(g.B' * x, g.Σ)

# Generic 4-arg form: covariate ignored (all non-GLM emissions defer here).
function _emission_to_driver(rng::AbstractRNG, dist, obs, control)
    return _emission_to_driver(rng, dist, obs)
end

#= Univariate and correlated-multivariate (Gaussian) GLMs: reduce, then reuse the
   conditional PIT. MvGaussianGLM has correlated outputs, so it gets the MvNormal
   Rosenblatt. =#
function _emission_to_driver(rng::AbstractRNG, g::AbstractGLM, obs, x)
    return _emission_to_driver(rng, _conditional(g, x), obs)
end

#= Independent-by-column multivariate GLMs: stack per-coordinate randomized PITs.
   The columns are independent given `x`, so the stacked drivers are genuinely
   independent uniforms and no joint Rosenblatt is needed. =#
function _emission_to_driver(rng::AbstractRNG, g::MvBernoulliGLM, obs::AbstractVector, x)
    η = g.B' * x
    return [
        _emission_to_driver(rng, Bernoulli(logistic(η[j])), obs[j])[1] for j in eachindex(η)
    ]
end

function _emission_to_driver(rng::AbstractRNG, g::MvPoissonGLM, obs::AbstractVector, x)
    η = g.B' * x
    return [_emission_to_driver(rng, Poisson(exp(η[j])), obs[j])[1] for j in eachindex(η)]
end

#= MultinomialGLM: the count coordinates are dependent through the shared total,
   so stacked per-coordinate PITs are wrong. Use the discrete Rosenblatt
   transform through the sequential conditional binomials
       y_j | y_1..y_{j-1} ~ Binomial(n − Σ_{i<j} yᵢ, p_j / (1 − Σ_{i<j} pᵢ)),
   applying the randomized PIT to each conditional (mirrors the sampler in
   `rand!`). The K-th coordinate is determined by the total, so its conditional
   is degenerate and carries no information: the driver has K−1 dimensions. =#
function _emission_to_driver(rng::AbstractRNG, g::MultinomialGLM, obs::AbstractVector, x)
    K = g.out_dim
    T = float(promote_type(eltype(g.B), eltype(x)))

    # Cache the K−1 logits once; both the log-sum-exp and the per-category
    # binomial probabilities below read them.
    ηs = Vector{T}(undef, K - 1)
    lse = zero(T)   # log Σₗ exp(ηₗ), reference category contributes exp(0)
    for j in 1:(K - 1)
        η = zero(T)
        for r in 1:(g.in_dim)
            η += g.B[r, j] * x[r]
        end
        ηs[j] = η
        lse = logaddexp(lse, η)
    end

    n_rem = 0
    for yj in obs
        n_rem += Int(yj)
    end

    ε = Vector{T}(undef, K - 1)
    p_rem = one(T)
    for j in 1:(K - 1)
        pj = exp(ηs[j] - lse)
        # p_rem can undershoot 0 by rounding once most mass is spent.
        pc = p_rem > 0 ? clamp(pj / p_rem, zero(T), one(T)) : one(T)
        yj = Int(obs[j])
        # `float(pc)` keeps a Float32/Dual pool in its own type (vs Float64(pc)).
        ε[j] = _emission_to_driver(rng, Binomial(n_rem, float(pc)), yj)[1]
        n_rem -= yj
        p_rem -= pj
    end
    return ε
end

# Label form: a single-trial choice y ∈ 1:K is its one-hot count vector.
function _emission_to_driver(rng::AbstractRNG, g::MultinomialGLM, obs::Real, x)
    return _emission_to_driver(rng, g, _OneHot(Int(obs), g.out_dim), x)
end

#= DDM emission (control -> drift): Rosenblatt transform of the (choice, rt) pair
   into a randomized PIT on the boundary choice and the conditional RT PIT, both
   uniform on (0,1) under the true model. Uses the extension's defective CDF. =#
function _emission_to_driver(rng::AbstractRNG, d::AbstractDDMEmission, obs, control)
    ν = _drift(d, control)
    choice = obs[1]
    rt = obs[2]
    T = float(promote_type(typeof(ν), typeof(d.α), typeof(rt)))
    # P(choice = 1), the t → ∞ limit of the defective CDF, in closed form.
    p1 = _clamp01(T(_ddm_prob_upper(ν, d.α, d.z)))
    Fc = T(_ddm_cdf(ν, d.α, d.z, d.τ, choice, rt))   # P(choice, RT ≤ rt)

    if choice == 1
        lower, upper, p_choice = zero(T), p1, p1
    else
        lower, upper, p_choice = p1, one(T), one(T) - p1
    end
    ε_choice = _randomized_pit(rng, lower, upper)
    # A never-hit boundary has an uninformative conditional RT: draw uniformly.
    ε_rt = p_choice > 0 ? _clamp01(Fc / p_choice) : rand(rng, T)
    return [_clamp01(ε_choice), ε_rt]
end

#= `obs_distributions` on a ControlledEmissionHMM yields ControlBoundEmissions;
   unwrap to the emission and its control for the conditional PIT methods. =#
function _emission_to_driver(rng::AbstractRNG, ce::ControlBoundEmission, obs, control)
    return _emission_to_driver(rng, ce.dist, obs, control)
end

# 3-arg form on a GLM: no covariate to condition on, so point at the 4-arg hook.
function _emission_to_driver(::AbstractRNG, g::AbstractGLM, obs)
    throw(
        ArgumentError(
            "GLM emission $(typeof(g)) is conditional on a covariate; call " *
            "`EmissionModels._emission_to_driver(rng, glm, obs, control)` with the " *
            "covariate vector for that observation.",
        ),
    )
end

#= Clear error for emission types without a PIT recipe, pointing the user at the
   extension hook. =#
function _emission_to_driver(::AbstractRNG, d, obs)
    throw(
        ArgumentError(
            "ACDC has no `_emission_to_driver` method for emission of type " *
            "$(typeof(d)). Supported out of the box: continuous/discrete " *
            "univariate `Distributions`, `MvNormal`, `PoissonZeroInflated`, and " *
            "`MvT`/`MvTDiag`. Define " *
            "`EmissionModels._emission_to_driver(rng, dist, obs)` returning a " *
            "vector of drivers in [0,1] to add support.",
        ),
    )
end
