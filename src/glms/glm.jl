"""
    AbstractGLM <: HiddenMarkovModels.ControlledEmission

Abstract type for Generalized Linear Model emission distributions.

Subtyping `ControlledEmission` lets a `Vector` of GLMs serve as the `dists` of a
`HiddenMarkovModels.ControlledEmissionHMM`. Each concrete GLM implements the
keyword (`control_seq`) interface internally:
- `DensityInterface.logdensityof(glm, obs; control_seq)`: log density
- `Random.rand(rng, glm; control_seq)`: conditional sample
- `StatsAPI.fit!(glm, obs_seq, weight_seq; control_seq)`: weighted in-place update

The `ControlledEmission` positional signatures HMM expects (`logdensityof(glm,
obs, control)`, `rand(rng, glm, control)`, `fit!(glm, obs_seq, control_seq, weights)`)
are provided as thin adapters at the bottom of this file. `DensityKind` is
inherited from `ControlledEmission`.

Univariate types (`GaussianGLM`, `BernoulliGLM`, `PoissonGLM`) carry a coefficient
vector `β` and emit scalar observations. Multivariate variants (`MvGaussianGLM`,
`MvBernoulliGLM`, `MvPoissonGLM`) carry a coefficient matrix `B` of size `p × k`
and emit length-`k` observation vectors.
"""
abstract type AbstractGLM <: ControlledEmission end

"""
    AbstractPrior

Supertype for priors on GLM coefficients β.

Subtypes must implement:
- `neglogprior(prior, β)`: scalar negative log-prior (up to a constant)
- `neglogprior_grad!(prior, g, β)`: accumulate ∂(-log p(β))/∂β into `g`
- `neglogprior_hess!(prior, H, β)`: accumulate ∂²(-log p(β))/∂β² into `H`

The gradient and Hessian methods accumulate (+=) rather than overwrite so
multiple priors can compose additively.
"""
abstract type AbstractPrior end

"""
    NoPrior <: AbstractPrior

Flat (improper uniform) prior, i.e. no regularization.
"""
struct NoPrior <: AbstractPrior end

neglogprior(::NoPrior, β) = zero(eltype(β))
neglogprior_grad!(::NoPrior, g, β) = nothing
neglogprior_hess!(::NoPrior, H, β) = nothing

"""
    RidgePrior{T<:Real} <: AbstractPrior

Isotropic Gaussian prior β ~ N(0, (1/λ)I), equivalent to L2 regularization.
Contributes 0.5 λ ‖β‖² to the negative log-posterior. For multivariate GLMs
the same prior is applied independently to each column of the coefficient
matrix `B`, giving a Frobenius-norm penalty 0.5 λ ‖B‖_F².
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
- `σ2`: noise variance (must be positive)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct GaussianGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    σ2::T
    prior::P
    function GaussianGLM{T,P}(
        β::Vector{T}, σ2::T, prior::P
    ) where {T<:Real,P<:AbstractPrior}
        σ2 > 0 || throw(ArgumentError("σ2 must be positive, got $σ2"))
        return new{T,P}(β, σ2, prior)
    end
end

function GaussianGLM(
    β::AbstractVector{T}, σ2::T, prior::P
) where {T<:AbstractFloat,P<:AbstractPrior}
    return GaussianGLM{T,P}(Vector{T}(β), σ2, prior)
end
function GaussianGLM(β::AbstractVector{T}, σ2::T) where {T<:AbstractFloat}
    return GaussianGLM(β, σ2, NoPrior())
end

function GaussianGLM(β::AbstractVector, σ2::Real, prior::AbstractPrior)
    T = float(promote_type(eltype(β), typeof(σ2)))
    return GaussianGLM(convert(Vector{T}, β), convert(T, σ2), prior)
end
GaussianGLM(β::AbstractVector, σ2::Real) = GaussianGLM(β, σ2, NoPrior())

function DensityInterface.logdensityof(
    reg::GaussianGLM, y::Real; control_seq::AbstractVector{<:Real}
)
    T = float(promote_type(eltype(reg.β), typeof(reg.σ2), eltype(control_seq), typeof(y)))
    μ = zero(T)
    for i in eachindex(reg.β, control_seq)
        μ += reg.β[i] * control_seq[i]
    end
    σ2 = T(reg.σ2)
    diff = T(y) - μ
    return -log(T(2π) * σ2) / 2 - diff * diff / (2 * σ2)
end

function Random.rand(
    rng::AbstractRNG, reg::GaussianGLM; control_seq::AbstractVector{<:Real}
)
    return rand(rng, Normal(dot(reg.β, control_seq), sqrt(reg.σ2)))
end

function StatsAPI.fit!(
    reg::GaussianGLM{T},
    obs_seq::AbstractVector{<:Real},
    weights::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weights) == n ||
        throw(DimensionMismatch("weights length $(length(weights)) ≠ control_seq rows $n"))
    length(reg.β) == p ||
        throw(DimensionMismatch("β length $(length(reg.β)) ≠ control_seq columns $p"))

    XWX = zeros(T, p, p)
    XWy = zeros(T, p)
    wsum = zero(T)

    #= XᵀWX is symmetric: accumulate only the lower triangle (stride-1 in the
       first index), halving the inner-loop work; the Cholesky below reads
       uplo = :L. =#
    for i in 1:n
        w = T(weights[i])
        wsum += w
        x_i = view(control_seq, i, :)
        y_i = T(obs_seq[i])
        for a in 1:p
            xa = x_i[a]
            wxa = w * xa
            for b in a:p
                XWX[b, a] += wxa * x_i[b]
            end
            XWy[a] += wxa * y_i
        end
    end
    wsum > 0 || throw(ArgumentError("weights must have positive sum, got $wsum"))

    # RidgePrior(λ) accumulates λI into XᵀWX, giving (XᵀWX + λI)β = XᵀWy.
    neglogprior_hess!(reg.prior, XWX, reg.β)

    F = cholesky!(Symmetric(XWX, :L); check=false)
    issuccess(F) || throw(
        ArgumentError(
            "XᵀWX is not positive definite: `control_seq` is rank-deficient " *
            "or weights are degenerate. Add a RidgePrior(λ) to regularize, or " *
            "drop collinear columns from `control_seq`.",
        ),
    )
    copyto!(reg.β, XWy)
    ldiv!(F, reg.β)

    sw_r2 = zero(T)
    for i in 1:n
        x_i = view(control_seq, i, :)
        r_i = T(obs_seq[i]) - dot(reg.β, x_i)
        sw_r2 += T(weights[i]) * r_i * r_i
    end
    #= Type-aware variance floor (same floor as the diagonal-t M-step): a
       perfect fit (e.g. n ≤ p) would otherwise set σ2 = 0, breaking the
       σ2 > 0 invariant and turning every later logdensityof into NaN. =#
    reg.σ2 = max(sw_r2 / wsum, sqrt(eps(T)))

    return reg
end

#= Optim-based Newton solver shared by the Bernoulli and Poisson GLM fits. Each
   family provides a fused `fgh!(F, G, H, β)` functor that computes the objective,
   gradient and Hessian in a single pass over the data, mirroring the ν-update
   pattern in `multivariate/t.jl`. =#

#= Lazy column accessor: indexing into a Vector{Vector{T}} along output dim j
   without copying. Used by the multivariate GLM fits to avoid an n-sized
   y-buffer per column. =#
struct _ColumnElementView{T,V<:AbstractVector{<:AbstractVector}} <: AbstractVector{T}
    seq::V
    j::Int
end
function _ColumnElementView(seq::V, j::Int) where {V<:AbstractVector{<:AbstractVector}}
    return _ColumnElementView{eltype(eltype(V)),V}(seq, j)
end
Base.size(c::_ColumnElementView) = (length(c.seq),)
Base.IndexStyle(::Type{<:_ColumnElementView}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(c::_ColumnElementView, i::Integer) = c.seq[i][c.j]

#= Fused Bernoulli negative log-posterior in the `fgh!(F, G, H, β)` form that
   `Optim.only_fgh!` expects. `F`, `G`, `H` are `nothing` when Optim doesn't
   need that piece on a given call; each nothing/array combination gets its own
   JIT specialization, so skipped work is compiled away. =#
struct _BernoulliFGH{
    Y<:AbstractVector,W<:AbstractVector{<:Real},M<:AbstractMatrix{<:Real},P<:AbstractPrior
}
    y::Y
    w::W
    X::M
    prior::P
end

function (o::_BernoulliFGH)(F, G, H, β::AbstractVector{T}) where {T<:Real}
    n, p = size(o.X)
    G === nothing || fill!(G, zero(T))
    H === nothing || fill!(H, zero(T))
    nll = zero(T)
    for i in 1:n
        x_i = view(o.X, i, :)
        wi = T(o.w[i])
        η_i = dot(β, x_i)
        y_i = o.y[i]
        if F !== nothing
            nll += wi * (y_i == 1 ? log1pexp(-η_i) : log1pexp(η_i))
        end
        if G !== nothing || H !== nothing
            μ_i = logistic(η_i)
            r_i = wi * (μ_i - T(y_i))
            W_i = wi * μ_i * (one(T) - μ_i)
            for a in 1:p
                xa = x_i[a]
                G === nothing || (G[a] += r_i * xa)
                if H !== nothing
                    wxa = W_i * xa
                    # Symmetric H: accumulate the lower triangle only,
                    # mirrored after the data pass.
                    for b in a:p
                        H[b, a] += wxa * x_i[b]
                    end
                end
            end
        end
    end
    if H !== nothing
        # Optim's Newton reads the full matrix — mirror the lower triangle.
        for a in 1:p, b in (a + 1):p
            H[a, b] = H[b, a]
        end
    end
    G === nothing || neglogprior_grad!(o.prior, G, β)
    H === nothing || neglogprior_hess!(o.prior, H, β)
    F === nothing || return nll + neglogprior(o.prior, β)
    return nothing
end

#= Bound η so that exp(η) stays well below floatmax(T), with a few nats of
   headroom for the w·exp(η) products that accumulate downstream. Type-aware:
   about 707 for Float64, 86 for Float32. =#
@inline _η_bound(::Type{T}) where {T<:AbstractFloat} = log(floatmax(T)) - T(2)

#= Fused Poisson negative log-posterior, same `fgh!(F, G, H, β)` contract as
   `_BernoulliFGH`. =#
struct _PoissonFGH{
    Y<:AbstractVector,W<:AbstractVector{<:Real},M<:AbstractMatrix{<:Real},P<:AbstractPrior
}
    y::Y
    w::W
    X::M
    prior::P
end

function (o::_PoissonFGH)(F, G, H, β::AbstractVector{T}) where {T<:Real}
    n, p = size(o.X)
    G === nothing || fill!(G, zero(T))
    H === nothing || fill!(H, zero(T))
    nll = zero(T)
    η_max = _η_bound(T)
    for i in 1:n
        x_i = view(o.X, i, :)
        wi = T(o.w[i])
        η_i = clamp(dot(β, x_i), -η_max, η_max)
        eη = exp(η_i)
        y_i = T(o.y[i])
        if F !== nothing
            nll += wi * (eη - y_i * η_i)
        end
        if G !== nothing || H !== nothing
            r_i = wi * (eη - y_i)
            W_i = wi * eη
            for a in 1:p
                xa = x_i[a]
                G === nothing || (G[a] += r_i * xa)
                if H !== nothing
                    wxa = W_i * xa
                    # Symmetric H: accumulate the lower triangle only,
                    # mirrored after the data pass.
                    for b in a:p
                        H[b, a] += wxa * x_i[b]
                    end
                end
            end
        end
    end
    if H !== nothing
        # Optim's Newton reads the full matrix — mirror the lower triangle.
        for a in 1:p, b in (a + 1):p
            H[a, b] = H[b, a]
        end
    end
    G === nothing || neglogprior_grad!(o.prior, G, β)
    H === nothing || neglogprior_hess!(o.prior, H, β)
    F === nothing || return nll + neglogprior(o.prior, β)
    return nothing
end

function _newton_fit!(β::Vector{T}, fgh::F; max_iter::Int, gtol::Real) where {T<:Real,F}
    td = TwiceDifferentiable(only_fgh!(fgh), β)
    result = optimize(td, β, Newton(), Optim.Options(; iterations=max_iter, g_abstol=gtol))
    copyto!(β, Optim.minimizer(result))
    return β
end

#= Shared validation + Newton driver for the univariate `BernoulliGLM.fit!` /
   `PoissonGLM.fit!`. `FGH` is the family's fused functor type (`_BernoulliFGH`
   or `_PoissonFGH`); the two fits are otherwise identical. =#
function _fit_glm_newton!(
    FGH::Type,
    β::Vector{T},
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real},
    control_seq::AbstractMatrix{<:Real},
    prior::AbstractPrior;
    max_iter::Int=50,
    gtol::Real=1e-8,
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"),
    )
    length(β) == p ||
        throw(DimensionMismatch("β length $(length(β)) ≠ control_seq columns $p"))

    fgh = FGH(obs_seq, weight_seq, control_seq, prior)
    return _newton_fit!(β, fgh; max_iter=max_iter, gtol=gtol)
end

"""
    BernoulliGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Logistic-regression emission for binary (0/1) observations.

P(Y=1|x) = σ(xᵀβ) where σ is the logistic function. `fit!` minimizes the
weighted negative log-posterior via Optim's Newton method with analytic
gradient and Hessian, so any `AbstractPrior` composes without changes to
the solver.

!!! note
    If the data are linearly separable, the unpenalized MLE does not exist —
    ‖β‖ diverges and the Hessian becomes singular. Use a `RidgePrior(λ)` to
    keep the fit finite in that regime.

# Fields
- `β`: coefficient vector (length p)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct BernoulliGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    prior::P

    function BernoulliGLM{T,P}(β::Vector{T}, prior::P) where {T<:Real,P<:AbstractPrior}
        return new{T,P}(β, prior)
    end
end

function BernoulliGLM(
    β::AbstractVector{T}, prior::P
) where {T<:AbstractFloat,P<:AbstractPrior}
    return BernoulliGLM{T,P}(Vector{T}(β), prior)
end
function BernoulliGLM(β::AbstractVector{T}) where {T<:AbstractFloat}
    return BernoulliGLM(β, NoPrior())
end

#= Promoting fallback for integer / mixed-eltype β (e.g., Vector{Int}). The
   typed constructor above is restricted to AbstractFloat because the Newton
   solver allocates `Vector{T}` workspace and cannot use Int. =#
function BernoulliGLM(β::AbstractVector, prior::AbstractPrior)
    T = float(eltype(β))
    return BernoulliGLM(convert(Vector{T}, β), prior)
end
BernoulliGLM(β::AbstractVector) = BernoulliGLM(β, NoPrior())

#= Observations are accepted as any Real (a 0/1 count stored as Float64 is
   common in HMM obs sequences); anything outside {0, 1} has zero mass. =#
function DensityInterface.logdensityof(
    glm::BernoulliGLM, y::Real; control_seq::AbstractVector{<:Real}
)
    T = float(promote_type(eltype(glm.β), eltype(control_seq)))
    (y == 0 || y == 1) || return T(-Inf)
    η = zero(T)
    for i in eachindex(glm.β, control_seq)
        η += glm.β[i] * control_seq[i]
    end
    return y == 1 ? -log1pexp(-η) : -log1pexp(η)
end

function Random.rand(
    rng::AbstractRNG, glm::BernoulliGLM; control_seq::AbstractVector{<:Real}
)
    return rand(rng, Bernoulli(logistic(dot(glm.β, control_seq))))
end

"""
    fit!(glm::BernoulliGLM, obs_seq, weight_seq;
         control_seq, max_iter=50, gtol=1e-8)

Minimize the weighted negative log-posterior via Optim's Newton method. The
objective is `Σᵢ wᵢ · ℓ(β; yᵢ, xᵢ) + neglogprior(prior, β)` where `ℓ` is the
Bernoulli log-likelihood; gradient and Hessian are analytic, supplied through
a fused `fgh!`.
"""
function StatsAPI.fit!(
    glm::BernoulliGLM,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
)
    _fit_glm_newton!(
        _BernoulliFGH,
        glm.β,
        obs_seq,
        weight_seq,
        control_seq,
        glm.prior;
        max_iter=max_iter,
        gtol=gtol,
    )
    return glm
end

"""
    PoissonGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Log-linear (Poisson) regression emission for count observations.

E[Y|x] = exp(xᵀβ). `fit!` minimizes the weighted negative log-posterior via
Optim's Newton method with analytic gradient and Hessian. Any `AbstractPrior`
composes without changes to the solver.

# Fields
- `β`: coefficient vector (length p)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct PoissonGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    prior::P

    function PoissonGLM{T,P}(β::Vector{T}, prior::P) where {T<:Real,P<:AbstractPrior}
        return new{T,P}(β, prior)
    end
end

function PoissonGLM(
    β::AbstractVector{T}, prior::P
) where {T<:AbstractFloat,P<:AbstractPrior}
    return PoissonGLM{T,P}(Vector{T}(β), prior)
end
function PoissonGLM(β::AbstractVector{T}) where {T<:AbstractFloat}
    return PoissonGLM(β, NoPrior())
end

function PoissonGLM(β::AbstractVector, prior::AbstractPrior)
    T = float(eltype(β))
    return PoissonGLM(convert(Vector{T}, β), prior)
end
PoissonGLM(β::AbstractVector) = PoissonGLM(β, NoPrior())

#= Observations are accepted as any Real (counts stored as Float64 are common
   in HMM obs sequences); non-integer or negative values have zero mass.
   `loggamma(y + 1)` is `logfactorial(y)` extended to float arguments. =#
function DensityInterface.logdensityof(
    glm::PoissonGLM, y::Real; control_seq::AbstractVector{<:Real}
)
    T = float(promote_type(eltype(glm.β), eltype(control_seq)))
    (y >= 0 && isinteger(y)) || return T(-Inf)
    η = zero(T)
    for i in eachindex(glm.β, control_seq)
        η += glm.β[i] * control_seq[i]
    end
    return T(y) * η - exp(η) - T(loggamma(T(y) + one(T)))
end

function Random.rand(rng::AbstractRNG, glm::PoissonGLM; control_seq::AbstractVector{<:Real})
    return rand(rng, Poisson(exp(dot(glm.β, control_seq))))
end

"""
    fit!(glm::PoissonGLM, obs_seq, weight_seq;
         control_seq, max_iter=50, gtol=1e-8)

Minimize the weighted negative log-posterior via Optim's Newton method. The
objective is `Σᵢ wᵢ · ℓ(β; yᵢ, xᵢ) + neglogprior(prior, β)` where `ℓ` is the
Poisson log-likelihood; gradient and Hessian are analytic, supplied through
a fused `fgh!`.
"""
function StatsAPI.fit!(
    glm::PoissonGLM,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
)
    _fit_glm_newton!(
        _PoissonFGH,
        glm.β,
        obs_seq,
        weight_seq,
        control_seq,
        glm.prior;
        max_iter=max_iter,
        gtol=gtol,
    )
    return glm
end

"""
    MvGaussianGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Multivariate linear regression emission with shared full covariance.

For input x ∈ ℝᵖ the conditional distribution of y ∈ ℝᵏ is
`y | x ~ MvNormal(Bᵀ x, Σ)`, where `B` is `p × k` and `Σ` is `k × k`.

`fit!` is the closed-form weighted multivariate least squares update:
`B = (XᵀWX + λI) \\ XᵀWY` and `Σ = (Y − XB)ᵀ W (Y − XB) / Σwᵢ`. A
`RidgePrior(λ)` adds `λI` to `XᵀWX`, equivalent to a per-column Gaussian
prior on `B` (Frobenius-norm penalty).

# Fields
- `B`: coefficient matrix of size `p × k`
- `Σ`: residual covariance of size `k × k`
- `prior`: regularization prior on `B` (default `NoPrior()`)
"""
mutable struct MvGaussianGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    B::Matrix{T}
    Σ::Matrix{T}
    prior::P

    Σ_chol::Cholesky{T,Matrix{T}}
    logdetΣ::T
    in_dim::Int
    out_dim::Int

    function MvGaussianGLM{T,P}(
        B::Matrix{T},
        Σ::Matrix{T},
        prior::P,
        Σ_chol::Cholesky{T,Matrix{T}},
        logdetΣ::T,
        in_dim::Int,
        out_dim::Int,
    ) where {T<:Real,P<:AbstractPrior}
        return new{T,P}(B, Σ, prior, Σ_chol, logdetΣ, in_dim, out_dim)
    end
end

function MvGaussianGLM(
    B::AbstractMatrix{T}, Σ::AbstractMatrix{T}, prior::P
) where {T<:AbstractFloat,P<:AbstractPrior}
    p, k = size(B)
    p > 0 || throw(ArgumentError("B must have at least one row"))
    k > 0 || throw(ArgumentError("B must have at least one column"))
    size(Σ) == (k, k) || throw(
        DimensionMismatch("Σ must be $(k)×$(k) for B with $(k) columns, got $(size(Σ))")
    )

    Σ_chol = cholesky(Symmetric(Σ, :L); check=false)
    issuccess(Σ_chol) || throw(ArgumentError("Σ must be positive definite"))

    return MvGaussianGLM{T,P}(
        Matrix{T}(B), Matrix{T}(Σ), prior, Σ_chol, logdet(Σ_chol), p, k
    )
end

function MvGaussianGLM(B::AbstractMatrix{T}, Σ::AbstractMatrix{T}) where {T<:AbstractFloat}
    return MvGaussianGLM(B, Σ, NoPrior())
end

#= Promoting fallback: B and Σ are promoted to a common float eltype. Handles
   mixed (Float32/Float64) and integer eltypes so user code can write
   `MvGaussianGLM([1 0; 0 1], [1.0 0; 0 1.0])` without first converting. =#
function MvGaussianGLM(B::AbstractMatrix, Σ::AbstractMatrix, prior::AbstractPrior)
    T = float(promote_type(eltype(B), eltype(Σ)))
    return MvGaussianGLM(convert(Matrix{T}, B), convert(Matrix{T}, Σ), prior)
end
MvGaussianGLM(B::AbstractMatrix, Σ::AbstractMatrix) = MvGaussianGLM(B, Σ, NoPrior())

"""
    logdensityof(glm::MvGaussianGLM, y::AbstractVector; control_seq)

Log density of `y ∈ ℝᵏ` under the conditional MvNormal model. Allocates one
length-`k` residual vector per call and is thread-safe (matches
`Distributions.MvNormal`).
"""
function DensityInterface.logdensityof(
    glm::MvGaussianGLM, y::AbstractVector; control_seq::AbstractVector{<:Real}
)
    length(y) == glm.out_dim ||
        throw(DimensionMismatch("y length $(length(y)) ≠ out_dim $(glm.out_dim)"))
    length(control_seq) == glm.in_dim || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ in_dim $(glm.in_dim)"
        ),
    )

    k = glm.out_dim
    p = glm.in_dim
    T = eltype(glm.B)
    diff = Vector{T}(undef, k)

    #= μ = Bᵀ x and diff = y − μ, fused as a single loop to avoid the
       Adjoint*Vector temporary that BLAS would otherwise allocate. The output
       type is `eltype(B)`: inputs in higher precision are downcast on entry
       and the residual buffer stays at the struct's float type. =#
    for j in 1:k
        sj = zero(T)
        for r in 1:p
            sj += glm.B[r, j] * T(control_seq[r])
        end
        diff[j] = T(y[j]) - sj
    end
    ldiv!(glm.Σ_chol.L, diff)
    mahal² = zero(T)
    for j in 1:k
        mahal² += diff[j] * diff[j]
    end

    return -T(k) * log(T(2π)) / 2 - glm.logdetΣ / 2 - mahal² / 2
end

function Random.rand(
    rng::AbstractRNG, glm::MvGaussianGLM{T}; control_seq::AbstractVector{<:Real}
) where {T<:Real}
    out = Vector{T}(undef, glm.out_dim)
    rand!(rng, glm, out; control_seq=control_seq)
    return out
end

"""
    rand!(rng, glm::MvGaussianGLM, out; control_seq)

In-place sample into `out` (length `out_dim`). Zero allocation, thread-safe:
the trick is to draw `z ~ N(0, I)` directly into `out`, multiply by `L`
in-place (`lmul!`), then add `μ = Bᵀ x` element-wise. Lower-triangular
multiply is well-defined in place because each output `xᵢ = Σⱼ≤ᵢ Lᵢⱼ zⱼ`
only reads `z[1..i]`, so iterating `i = k, k-1, …, 1` never reads an
already-overwritten entry.
"""
function Random.rand!(
    rng::AbstractRNG,
    glm::MvGaussianGLM{T},
    out::AbstractVector;
    control_seq::AbstractVector{<:Real},
) where {T<:Real}
    length(out) == glm.out_dim ||
        throw(DimensionMismatch("out length $(length(out)) ≠ out_dim $(glm.out_dim)"))
    length(control_seq) == glm.in_dim || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ in_dim $(glm.in_dim)"
        ),
    )

    k = glm.out_dim
    p = glm.in_dim

    randn!(rng, out)
    lmul!(glm.Σ_chol.L, out)         # out = L * z, in place

    # out += μ = Bᵀ x (Bᵀx is computed inline; no aliasing with out)
    for j in 1:k
        sj = zero(T)
        for r in 1:p
            sj += glm.B[r, j] * control_seq[r]
        end
        out[j] += sj
    end
    return out
end

"""
    fit!(glm::MvGaussianGLM, obs_seq, weight_seq; control_seq)

Closed-form weighted multivariate WLS update. Each observation `obs_seq[i]`
must be a length-`k` vector. `RidgePrior(λ)` augments the normal equations
with `λI`.
"""
function StatsAPI.fit!(
    glm::MvGaussianGLM{T},
    obs_seq::AbstractVector{<:AbstractVector},
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"),
    )
    p == glm.in_dim ||
        throw(DimensionMismatch("control_seq columns $p ≠ in_dim $(glm.in_dim)"))

    k = glm.out_dim

    XWX = zeros(T, p, p)
    XWY = zeros(T, p, k)
    wsum = zero(T)

    # Build XᵀWX and XᵀWY in a single pass, with no Y matrix or temporaries.
    for i in 1:n
        obs_i = obs_seq[i]
        length(obs_i) == k ||
            throw(DimensionMismatch("obs_seq[$i] length $(length(obs_i)) ≠ out_dim $k"))
        w = T(weight_seq[i])
        wsum += w
        x_i = view(control_seq, i, :)
        for a in 1:p
            xa = x_i[a]
            wxa = w * xa
            for b in a:p
                XWX[b, a] += wxa * x_i[b]
            end
            for j in 1:k
                XWY[a, j] += wxa * T(obs_i[j])
            end
        end
    end

    wsum > 0 || throw(ArgumentError("weights must have positive sum, got $wsum"))

    #= Per-column ridge: λI added to XᵀWX gives B = (XᵀWX + λI) \ XᵀWY,
       the joint MAP for independent N(0, (1/λ)I) priors on each column of B.
       Pass a length-p view so RidgePrior loops over rows of XWX, not p*k. =#
    neglogprior_hess!(glm.prior, XWX, view(glm.B, :, 1))

    #= XᵀWX is PD whenever the (weighted) design matrix has full column rank.
       Failure here means rank-deficient `control_seq`, so surface that as a
       clear error rather than silently shifting eigenvalues. =#
    F = cholesky!(Symmetric(XWX, :L); check=false)
    issuccess(F) || throw(
        ArgumentError(
            "XᵀWX is not positive definite: `control_seq` is rank-deficient " *
            "or weights are degenerate. Add a RidgePrior(λ) to regularize, or " *
            "drop collinear columns from `control_seq`.",
        ),
    )
    copyto!(glm.B, XWY)
    ldiv!(F, glm.B)

    #= Σ M-step: Σ = Σᵢ wᵢ (yᵢ - Bᵀxᵢ)(yᵢ - Bᵀxᵢ)ᵀ / Σwᵢ. Computes residuals
       on the fly into a length-k buffer, then accumulates outer products into
       Σ_new. Avoids the full n×k residual matrix. =#
    r = Vector{T}(undef, k)
    Σ_new = zeros(T, k, k)
    for i in 1:n
        obs_i = obs_seq[i]
        w = T(weight_seq[i])
        x_i = view(control_seq, i, :)
        for j in 1:k
            rj = T(obs_i[j])
            for a in 1:p
                rj -= glm.B[a, j] * x_i[a]
            end
            r[j] = rj
        end
        for j in 1:k
            wrj = w * r[j]
            for l in 1:j
                v = wrj * r[l]
                Σ_new[l, j] += v
                if l != j
                    Σ_new[j, l] += v
                end
            end
        end
    end
    Σ_new ./= wsum

    #= Σ_new is Σwᵢ·rᵢrᵢᵀ / Σwᵢ, which is PSD by construction and PD whenever
       the residuals span ℝᵏ. Failure means a degenerate output dimension
       (zero variance), so error rather than silently regularize. =#
    Σ_chol_new = cholesky(Symmetric(Σ_new, :L); check=false)
    issuccess(Σ_chol_new) || throw(
        ArgumentError(
            "Residual covariance Σ is not positive definite: at least one " *
            "output dimension has zero (or perfectly collinear) residual " *
            "variance. Check `obs_seq` for degenerate output columns.",
        ),
    )

    copyto!(glm.Σ, Σ_new)
    glm.Σ_chol = Σ_chol_new
    glm.logdetΣ = logdet(glm.Σ_chol)

    return glm
end

"""
    MvBernoulliGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Multivariate Bernoulli emission as `k` independent logistic regressions.

For input x ∈ ℝᵖ the conditional distribution of y ∈ {0,1}ᵏ factorizes as
`P(y|x) = ∏ⱼ Bernoulli(σ(B[:,j]ᵀ x))`. Coefficients live in a `p × k` matrix
`B`. `fit!` runs an independent weighted Newton fit per column.

# Fields
- `B`: coefficient matrix of size `p × k`
- `prior`: regularization prior applied per column (default `NoPrior()`)
"""
mutable struct MvBernoulliGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    B::Matrix{T}
    prior::P
    in_dim::Int
    out_dim::Int

    function MvBernoulliGLM{T,P}(
        B::Matrix{T}, prior::P, in_dim::Int, out_dim::Int
    ) where {T<:Real,P<:AbstractPrior}
        return new{T,P}(B, prior, in_dim, out_dim)
    end
end

function MvBernoulliGLM(
    B::AbstractMatrix{T}, prior::P
) where {T<:AbstractFloat,P<:AbstractPrior}
    p, k = size(B)
    p > 0 || throw(ArgumentError("B must have at least one row"))
    k > 0 || throw(ArgumentError("B must have at least one column"))
    return MvBernoulliGLM{T,P}(Matrix{T}(B), prior, p, k)
end
function MvBernoulliGLM(B::AbstractMatrix{T}) where {T<:AbstractFloat}
    return MvBernoulliGLM(B, NoPrior())
end

function MvBernoulliGLM(B::AbstractMatrix, prior::AbstractPrior)
    T = float(eltype(B))
    return MvBernoulliGLM(convert(Matrix{T}, B), prior)
end
MvBernoulliGLM(B::AbstractMatrix) = MvBernoulliGLM(B, NoPrior())

function DensityInterface.logdensityof(
    glm::MvBernoulliGLM, y::AbstractVector; control_seq::AbstractVector{<:Real}
)
    length(y) == glm.out_dim ||
        throw(DimensionMismatch("y length $(length(y)) ≠ out_dim $(glm.out_dim)"))
    length(control_seq) == glm.in_dim || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ in_dim $(glm.in_dim)"
        ),
    )

    Tη = float(promote_type(eltype(glm.B), eltype(control_seq)))
    lp = zero(Tη)
    for j in 1:(glm.out_dim)
        yj = y[j]
        (yj == 0 || yj == 1) || return Tη(-Inf)
        η = zero(Tη)
        for r in 1:(glm.in_dim)
            η += glm.B[r, j] * control_seq[r]
        end
        lp += yj == 1 ? -log1pexp(-η) : -log1pexp(η)
    end
    return lp
end

function Random.rand(
    rng::AbstractRNG, glm::MvBernoulliGLM; control_seq::AbstractVector{<:Real}
)
    out = Vector{Int}(undef, glm.out_dim)
    rand!(rng, glm, out; control_seq=control_seq)
    return out
end

"""
    rand!(rng, glm::MvBernoulliGLM, out; control_seq)

In-place sample. `out` must be a length-`out_dim` integer vector.
Zero allocation.
"""
function Random.rand!(
    rng::AbstractRNG,
    glm::MvBernoulliGLM,
    out::AbstractVector;
    control_seq::AbstractVector{<:Real},
)
    length(out) == glm.out_dim ||
        throw(DimensionMismatch("out length $(length(out)) ≠ out_dim $(glm.out_dim)"))
    length(control_seq) == glm.in_dim || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ in_dim $(glm.in_dim)"
        ),
    )

    T = eltype(glm.B)
    for j in 1:(glm.out_dim)
        η = zero(T)
        for r in 1:(glm.in_dim)
            η += glm.B[r, j] * control_seq[r]
        end
        out[j] = rand(rng) < logistic(η) ? 1 : 0
    end
    return out
end

"""
    fit!(glm::MvBernoulliGLM, obs_seq, weight_seq;
         control_seq, max_iter=50, gtol=1e-8)

Fit each output dimension independently via Optim's Newton method. Each
observation `obs_seq[i]` must be a length-`k` vector of 0/1 values.
`_ColumnElementView` presents column `j` of the observations lazily, so no
n-sized y-buffer is copied per column.
"""
function StatsAPI.fit!(
    glm::MvBernoulliGLM{T},
    obs_seq::AbstractVector{<:AbstractVector},
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"),
    )
    p == glm.in_dim ||
        throw(DimensionMismatch("control_seq columns $p ≠ in_dim $(glm.in_dim)"))

    k = glm.out_dim
    for i in 1:n
        length(obs_seq[i]) == k || throw(
            DimensionMismatch("obs_seq[$i] length $(length(obs_seq[i])) ≠ out_dim $k")
        )
    end

    β_buf = Vector{T}(undef, p)
    for j in 1:k
        fgh = _BernoulliFGH(
            _ColumnElementView(obs_seq, j), weight_seq, control_seq, glm.prior
        )
        copyto!(β_buf, view(glm.B, :, j))
        _newton_fit!(β_buf, fgh; max_iter=max_iter, gtol=gtol)
        copyto!(view(glm.B, :, j), β_buf)
    end
    return glm
end

"""
    MvPoissonGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Multivariate Poisson emission as `k` independent log-linear regressions.

For input x ∈ ℝᵖ the conditional distribution of y ∈ ℤ₊ᵏ factorizes as
`P(y|x) = ∏ⱼ Poisson(exp(B[:,j]ᵀ x))`. Coefficients live in a `p × k` matrix
`B`. `fit!` runs an independent weighted Newton fit per column.

# Fields
- `B`: coefficient matrix of size `p × k`
- `prior`: regularization prior applied per column (default `NoPrior()`)
"""
mutable struct MvPoissonGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    B::Matrix{T}
    prior::P
    in_dim::Int
    out_dim::Int

    function MvPoissonGLM{T,P}(
        B::Matrix{T}, prior::P, in_dim::Int, out_dim::Int
    ) where {T<:Real,P<:AbstractPrior}
        return new{T,P}(B, prior, in_dim, out_dim)
    end
end

function MvPoissonGLM(
    B::AbstractMatrix{T}, prior::P
) where {T<:AbstractFloat,P<:AbstractPrior}
    p, k = size(B)
    p > 0 || throw(ArgumentError("B must have at least one row"))
    k > 0 || throw(ArgumentError("B must have at least one column"))
    return MvPoissonGLM{T,P}(Matrix{T}(B), prior, p, k)
end
function MvPoissonGLM(B::AbstractMatrix{T}) where {T<:AbstractFloat}
    return MvPoissonGLM(B, NoPrior())
end

function MvPoissonGLM(B::AbstractMatrix, prior::AbstractPrior)
    T = float(eltype(B))
    return MvPoissonGLM(convert(Matrix{T}, B), prior)
end
MvPoissonGLM(B::AbstractMatrix) = MvPoissonGLM(B, NoPrior())

function DensityInterface.logdensityof(
    glm::MvPoissonGLM, y::AbstractVector; control_seq::AbstractVector{<:Real}
)
    length(y) == glm.out_dim ||
        throw(DimensionMismatch("y length $(length(y)) ≠ out_dim $(glm.out_dim)"))
    length(control_seq) == glm.in_dim || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ in_dim $(glm.in_dim)"
        ),
    )

    Tη = float(promote_type(eltype(glm.B), eltype(control_seq)))
    lp = zero(Tη)
    for j in 1:(glm.out_dim)
        yj = y[j]
        # Real-valued counts are fine; non-integer or negative ⇒ zero mass.
        (yj >= 0 && isinteger(yj)) || return Tη(-Inf)
        η = zero(Tη)
        for r in 1:(glm.in_dim)
            η += glm.B[r, j] * control_seq[r]
        end
        lp += Tη(yj) * η - exp(η) - Tη(loggamma(Tη(yj) + one(Tη)))
    end
    return lp
end

function Random.rand(
    rng::AbstractRNG, glm::MvPoissonGLM; control_seq::AbstractVector{<:Real}
)
    out = Vector{Int}(undef, glm.out_dim)
    rand!(rng, glm, out; control_seq=control_seq)
    return out
end

"""
    rand!(rng, glm::MvPoissonGLM, out; control_seq)

In-place sample. `out` must be a length-`out_dim` integer vector.
"""
function Random.rand!(
    rng::AbstractRNG,
    glm::MvPoissonGLM,
    out::AbstractVector;
    control_seq::AbstractVector{<:Real},
)
    length(out) == glm.out_dim ||
        throw(DimensionMismatch("out length $(length(out)) ≠ out_dim $(glm.out_dim)"))
    length(control_seq) == glm.in_dim || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ in_dim $(glm.in_dim)"
        ),
    )

    T = eltype(glm.B)
    for j in 1:(glm.out_dim)
        η = zero(T)
        for r in 1:(glm.in_dim)
            η += glm.B[r, j] * control_seq[r]
        end
        out[j] = rand(rng, Poisson(exp(η)))
    end
    return out
end

"""
    fit!(glm::MvPoissonGLM, obs_seq, weight_seq;
         control_seq, max_iter=50, gtol=1e-8)

Fit each output dimension independently via Optim's Newton method. Each
observation `obs_seq[i]` must be a length-`k` vector of non-negative integers.
`_ColumnElementView` presents column `j` of the observations lazily, so no
n-sized y-buffer is copied per column.
"""
function StatsAPI.fit!(
    glm::MvPoissonGLM{T},
    obs_seq::AbstractVector{<:AbstractVector},
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"),
    )
    p == glm.in_dim ||
        throw(DimensionMismatch("control_seq columns $p ≠ in_dim $(glm.in_dim)"))

    k = glm.out_dim
    for i in 1:n
        length(obs_seq[i]) == k || throw(
            DimensionMismatch("obs_seq[$i] length $(length(obs_seq[i])) ≠ out_dim $k")
        )
    end

    β_buf = Vector{T}(undef, p)
    for j in 1:k
        fgh = _PoissonFGH(
            _ColumnElementView(obs_seq, j), weight_seq, control_seq, glm.prior
        )
        copyto!(β_buf, view(glm.B, :, j))
        _newton_fit!(β_buf, fgh; max_iter=max_iter, gtol=gtol)
        copyto!(view(glm.B, :, j), β_buf)
    end
    return glm
end

#= HiddenMarkovModels.ControlledEmission interface.

   `AbstractGLM <: ControlledEmission`, so a `Vector` of GLMs is a valid `dists`
   for a `ControlledEmissionHMM`. That HMM drives each emission through the
   control-aware positional signatures below; each `control` is a single
   timestep's covariate vector (the same `control_seq` argument the keyword
   methods above already consume), and the fit-time `control_seq` is a vector of
   such vectors, one per timestep. The adapters delegate to the keyword
   implementations so the actual math has a single source of truth. =#

# Length of one covariate vector for this GLM (the GLM's input dimension `p`).
_indim(glm::Union{GaussianGLM,BernoulliGLM,PoissonGLM}) = length(glm.β)
_indim(glm::Union{MvGaussianGLM,MvBernoulliGLM,MvPoissonGLM}) = glm.in_dim

function DensityInterface.logdensityof(
    glm::AbstractGLM, obs, control::AbstractVector{<:Real}
)
    return logdensityof(glm, obs; control_seq=control)
end

function Random.rand(rng::AbstractRNG, glm::AbstractGLM, control::AbstractVector{<:Real})
    return rand(rng, glm; control_seq=control)
end

#= Zero-copy `n×p` design matrix over a length-`n` vector of length-`p` covariate
   vectors. `ControlledEmissionHMM` hands `fit!` a `control_seq` shaped as a
   `Vector{<:AbstractVector}` (one covariate vector per timestep), whereas the
   matrix-based keyword `fit!` implementations want an `n×p` matrix. This presents
   the former as the latter without copying: `view(M, i, :)` returns the i-th
   covariate vector directly, so the existing inner loops stay allocation-free. =#
struct _ControlRowsMatrix{T,V<:AbstractVector{<:AbstractVector}} <: AbstractMatrix{T}
    rows::V
    p::Int
end
function _ControlRowsMatrix(rows::V, p::Int) where {V<:AbstractVector{<:AbstractVector}}
    return _ControlRowsMatrix{eltype(eltype(V)),V}(rows, p)
end
Base.size(M::_ControlRowsMatrix) = (length(M.rows), M.p)
Base.@propagate_inbounds Base.getindex(M::_ControlRowsMatrix, i::Int, j::Int) = M.rows[i][j]
Base.@propagate_inbounds Base.view(M::_ControlRowsMatrix, i::Integer, ::Colon) = M.rows[i]

function StatsAPI.fit!(
    glm::AbstractGLM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector{<:AbstractVector},
    weights::AbstractVector{<:Real};
    kwargs...,
)
    X = _ControlRowsMatrix(control_seq, _indim(glm))
    return fit!(glm, obs_seq, weights; control_seq=X, kwargs...)
end
