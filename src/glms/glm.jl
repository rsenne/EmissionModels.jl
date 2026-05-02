"""
    AbstractGLM

Abstract type for Generalized Linear Model emission distributions.

GLM subtypes should implement the HiddenMarkovModels.jl interface:
- `DensityInterface.DensityKind(::YourGLM)` → `HasDensity()`
- `DensityInterface.logdensityof(glm, obs; control_seq)` — log density
- `Random.rand(rng, glm; control_seq)` — conditional sample
- `StatsAPI.fit!(glm, obs_seq, weight_seq; control_seq)` — weighted in-place update
"""
abstract type AbstractGLM end

"""
    AbstractPrior

Supertype for priors on GLM coefficients β.

Subtypes must implement:
- `neglogprior(prior, β)` → scalar negative log-prior (up to constant)
- `neglogprior_grad!(prior, g, β)` → accumulate ∂(-log p(β))/∂β into `g`
- `neglogprior_hess!(prior, H, β)` → accumulate ∂²(-log p(β))/∂β² into `H`

The gradient and Hessian methods accumulate (+=) rather than overwrite so
multiple priors can compose additively.
"""
abstract type AbstractPrior end

"""
    NoPrior <: AbstractPrior

Flat (improper uniform) prior — no regularization.
"""
struct NoPrior <: AbstractPrior end

neglogprior(::NoPrior, β) = zero(eltype(β))
neglogprior_grad!(::NoPrior, g, β) = nothing
neglogprior_hess!(::NoPrior, H, β) = nothing

"""
    RidgePrior{T<:Real} <: AbstractPrior

Isotropic Gaussian prior β ~ N(0, (1/λ)I), equivalent to L2 regularization.
Contributes 0.5 λ ‖β‖² to the negative log-posterior.
"""
struct RidgePrior{T<:Real} <: AbstractPrior
    λ::T
end

neglogprior(p::RidgePrior, β) = oftype(p.λ, 0.5) * p.λ * dot(β, β)

function neglogprior_grad!(p::RidgePrior, g, β)
    for i in eachindex(g, β)
        g[i] += p.λ * β[i]
    end
end

function neglogprior_hess!(p::RidgePrior, H, β)
    for i in eachindex(β)
        H[i, i] += p.λ
    end
end

"""
    GaussianGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Linear regression emission with Gaussian noise.

E[Y|x] = xᵀβ, Var[Y|x] = σ². `fit!` uses weighted least squares; a
`RidgePrior(λ)` augments the normal equations with `λI`, which is the
exact MAP solution for a Gaussian prior β ~ N(0, (1/λ)I).

# Fields
- `β`: coefficient vector (length p)
- `σ2`: noise variance
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct GaussianGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    σ2::T
    prior::P
end

GaussianGLM(β::AbstractVector{T}, σ2::T) where {T<:Real} =
    GaussianGLM{T, NoPrior}(Vector{T}(β), σ2, NoPrior())

GaussianGLM(β::AbstractVector{T}, σ2::T, prior::P) where {T<:Real, P<:AbstractPrior} =
    GaussianGLM{T, P}(Vector{T}(β), σ2, prior)

# Convenience: promote β eltype and σ2 together
GaussianGLM(β::AbstractVector, σ2::Real) =
    GaussianGLM(float.(β), float(σ2))

GaussianGLM(β::AbstractVector, σ2::Real, prior::AbstractPrior) =
    GaussianGLM(float.(β), float(σ2), prior)

DensityInterface.DensityKind(::GaussianGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(reg::GaussianGLM, y::Real; control_seq::AbstractVector{<:Real})
    μ = dot(reg.β, control_seq)
    return -0.5 * log(2π * reg.σ2) - 0.5 * ((y - μ)^2 / reg.σ2)
end

function Random.rand(rng::AbstractRNG, reg::GaussianGLM; control_seq::AbstractVector{<:Real})
    return rand(rng, Normal(dot(reg.β, control_seq), sqrt(reg.σ2)))
end

function StatsAPI.fit!(
    reg::GaussianGLM,
    obs_seq::AbstractVector{<:Real},
    weights::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
)
    W = Diagonal(weights)
    XWX = control_seq' * W * control_seq
    XWy = control_seq' * W * obs_seq

    #= For RidgePrior(λ), the MAP normal equations are (X'WX + λI)β = X'Wy.
       neglogprior_hess! accumulates λI into XWX in-place, so other prior
       types compose here automatically. =#
    neglogprior_hess!(reg.prior, XWX, reg.β)

    reg.β = XWX \ XWy

    residuals = obs_seq .- control_seq * reg.β
    reg.σ2 = sum(weights .* (residuals .^ 2)) / sum(weights)

    return reg
end

"""
    BernoulliGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Logistic-regression emission for binary (0/1) observations.

P(Y=1|x) = σ(xᵀβ) where σ is the logistic function. `fit!` minimizes the
weighted negative log-posterior via Optim Newton, so any `AbstractPrior`
composes without changes to the solver.

# Fields
- `β`: coefficient vector (length p)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct BernoulliGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    prior::P
end

BernoulliGLM(β::AbstractVector{T}) where {T<:Real} =
    BernoulliGLM{T, NoPrior}(Vector{T}(β), NoPrior())

BernoulliGLM(β::AbstractVector{T}, prior::P) where {T<:Real, P<:AbstractPrior} =
    BernoulliGLM{T, P}(Vector{T}(β), prior)

DensityInterface.DensityKind(::BernoulliGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(
    glm::BernoulliGLM,
    y::Integer;
    control_seq::AbstractVector{<:Real},
)
    (y == 0 || y == 1) || return oftype(dot(glm.β, control_seq), -Inf)
    η = dot(glm.β, control_seq)
    return y == 1 ? -log1pexp(-η) : -log1pexp(η)
end

function Random.rand(rng::AbstractRNG, glm::BernoulliGLM; control_seq::AbstractVector{<:Real})
    return rand(rng, Bernoulli(logistic(dot(glm.β, control_seq))))
end

"""
    fit!(glm::BernoulliGLM, obs_seq, weight_seq; control_seq, optim_opts)

Minimize the weighted negative log-posterior via Optim Newton.

The objective is Σᵢ wᵢ · ℓ(β; yᵢ, xᵢ) + neglogprior(prior, β) where ℓ is the
Bernoulli log-likelihood. Gradient and Hessian are computed analytically; the
Hessian is the standard logistic regression Fisher information weighted by wᵢ,
plus the prior's Hessian contribution.
"""
function StatsAPI.fit!(
    glm::BernoulliGLM{T},
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    optim_opts::Optim.Options=Optim.Options(),
) where {T}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n ||
        throw(DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"))
    length(glm.β) == p ||
        throw(DimensionMismatch("β length $(length(glm.β)) ≠ control_seq columns $p"))

    prior = glm.prior  # avoid closure over glm to keep captures minimal

    function neglogpost(β)
        nll = zero(T)
        for i in 1:n
            x_i = view(control_seq, i, :)
            η_i = dot(β, x_i)
            #= log1pexp is numerically safe for both signs of η =#
            nll += weight_seq[i] * (obs_seq[i] == 1 ? log1pexp(-η_i) : log1pexp(η_i))
        end
        return nll + neglogprior(prior, β)
    end

    function neglogpost_grad!(g, β)
        fill!(g, zero(T))
        for i in 1:n
            x_i = view(control_seq, i, :)
            r_i = weight_seq[i] * (logistic(dot(β, x_i)) - obs_seq[i])
            for j in 1:p
                g[j] += r_i * x_i[j]
            end
        end
        neglogprior_grad!(prior, g, β)
    end

    function neglogpost_hess!(H, β)
        fill!(H, zero(T))
        for i in 1:n
            x_i = view(control_seq, i, :)
            μ_i = logistic(dot(β, x_i))
            W_i = weight_seq[i] * μ_i * (one(T) - μ_i)
            for j in 1:p
                for k in 1:p
                    H[j, k] += W_i * x_i[j] * x_i[k]
                end
            end
        end
        neglogprior_hess!(prior, H, β)
    end

    td = TwiceDifferentiable(neglogpost, neglogpost_grad!, neglogpost_hess!, glm.β)
    result = optimize(td, glm.β, Newton(), optim_opts)
    copyto!(glm.β, Optim.minimizer(result))

    return glm
end

"""
    PoissonGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Log-linear (Poisson) regression emission for count observations.

E[Y|x] = exp(xᵀβ). `fit!` minimizes the weighted negative log-posterior via
Optim Newton. Any `AbstractPrior` composes without changes to the solver.

# Fields
- `β`: coefficient vector (length p)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct PoissonGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    prior::P
end

PoissonGLM(β::AbstractVector{T}) where {T<:Real} =
    PoissonGLM{T, NoPrior}(Vector{T}(β), NoPrior())

PoissonGLM(β::AbstractVector{T}, prior::P) where {T<:Real, P<:AbstractPrior} =
    PoissonGLM{T, P}(Vector{T}(β), prior)

DensityInterface.DensityKind(::PoissonGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(
    glm::PoissonGLM,
    y::Integer;
    control_seq::AbstractVector{<:Real},
)
    y >= 0 || return oftype(dot(glm.β, control_seq), -Inf)
    η = dot(glm.β, control_seq)
    return y * η - exp(η) - logfactorial(y)
end

function Random.rand(rng::AbstractRNG, glm::PoissonGLM; control_seq::AbstractVector{<:Real})
    return rand(rng, Poisson(exp(dot(glm.β, control_seq))))
end

"""
    fit!(glm::PoissonGLM, obs_seq, weight_seq; control_seq, optim_opts)

Minimize the weighted negative log-posterior via Optim Newton.

The objective is Σᵢ wᵢ · ℓ(β; yᵢ, xᵢ) + neglogprior(prior, β) where ℓ is
the Poisson log-likelihood. The Hessian is the Fisher information weighted by
wᵢ (Hessian of Poisson negloglik = Σ wᵢμᵢ xᵢxᵢᵀ), plus the prior contribution.
"""
function StatsAPI.fit!(
    glm::PoissonGLM{T},
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    optim_opts::Optim.Options=Optim.Options(),
) where {T}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n ||
        throw(DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"))
    length(glm.β) == p ||
        throw(DimensionMismatch("β length $(length(glm.β)) ≠ control_seq columns $p"))

    prior = glm.prior

    function neglogpost(β)
        nll = zero(T)
        for i in 1:n
            x_i = view(control_seq, i, :)
            η_i  = dot(β, x_i)
            μ_i  = exp(η_i)
            nll += weight_seq[i] * (μ_i - obs_seq[i] * η_i)
        end
        return nll + neglogprior(prior, β)
    end

    function neglogpost_grad!(g, β)
        fill!(g, zero(T))
        for i in 1:n
            x_i = view(control_seq, i, :)
            η_i = clamp(dot(β, x_i), -500, 500)
            r_i = weight_seq[i] * (exp(η_i) - obs_seq[i])
            for j in 1:p
                g[j] += r_i * x_i[j]
            end
        end
        neglogprior_grad!(prior, g, β)
    end

    function neglogpost_hess!(H, β)
        fill!(H, zero(T))
        for i in 1:n
            x_i = view(control_seq, i, :)
            η_i = clamp(dot(β, x_i), -500, 500)
            W_i = weight_seq[i] * exp(η_i)
            for j in 1:p
                for k in 1:p
                    H[j, k] += W_i * x_i[j] * x_i[k]
                end
            end
        end
        neglogprior_hess!(prior, H, β)
    end

    td = TwiceDifferentiable(neglogpost, neglogpost_grad!, neglogpost_hess!, glm.β)
    result = optimize(td, glm.β, Newton(), optim_opts)
    copyto!(glm.β, Optim.minimizer(result))

    return glm
end
