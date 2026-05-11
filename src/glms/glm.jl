"""
    AbstractGLM

Abstract type for Generalized Linear Model emission distributions.

GLM subtypes should implement the HiddenMarkovModels.jl interface:
- `DensityInterface.DensityKind(::YourGLM)` → `HasDensity()`
- `DensityInterface.logdensityof(glm, obs; control_seq)` — log density
- `Random.rand(rng, glm; control_seq)` — conditional sample
- `StatsAPI.fit!(glm, obs_seq, weight_seq; control_seq)` — weighted in-place update

Univariate types (`GaussianGLM`, `BernoulliGLM`, `PoissonGLM`) carry a coefficient
vector `β` and emit scalar observations. Multivariate variants (`MvGaussianGLM`,
`MvBernoulliGLM`, `MvPoissonGLM`) carry a coefficient matrix `B` of size `p × k`
and emit length-`k` observation vectors.
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
- `σ2`: noise variance
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct GaussianGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    σ2::T
    prior::P
end

function GaussianGLM(β::AbstractVector{T}, σ2::T) where {T<:Real}
    return GaussianGLM{T,NoPrior}(Vector{T}(β), σ2, NoPrior())
end

function GaussianGLM(β::AbstractVector{T}, σ2::T, prior::P) where {T<:Real,P<:AbstractPrior}
    return GaussianGLM{T,P}(Vector{T}(β), σ2, prior)
end

# Convenience: promote β eltype and σ2 together
GaussianGLM(β::AbstractVector, σ2::Real) = GaussianGLM(float.(β), float(σ2))

function GaussianGLM(β::AbstractVector, σ2::Real, prior::AbstractPrior)
    return GaussianGLM(float.(β), float(σ2), prior)
end

DensityInterface.DensityKind(::GaussianGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(
    reg::GaussianGLM, y::Real; control_seq::AbstractVector{<:Real}
)
    μ = dot(reg.β, control_seq)
    return -0.5 * log(2π * reg.σ2) - 0.5 * ((y - μ)^2 / reg.σ2)
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

    for i in 1:n
        w = T(weights[i])
        wsum += w
        x_i = view(control_seq, i, :)
        y_i = T(obs_seq[i])
        for a in 1:p
            xa = x_i[a]
            wxa = w * xa
            for b in 1:p
                XWX[a, b] += wxa * x_i[b]
            end
            XWy[a] += wxa * y_i
        end
    end

    #= RidgePrior(λ) accumulates λI into XᵀWX, giving (XᵀWX + λI)β = XᵀWy. =#
    neglogprior_hess!(reg.prior, XWX, reg.β)

    F = cholesky!(Symmetric(XWX))
    copyto!(reg.β, XWy)
    ldiv!(F, reg.β)

    sw_r2 = zero(T)
    for i in 1:n
        x_i = view(control_seq, i, :)
        r_i = T(obs_seq[i]) - dot(reg.β, x_i)
        sw_r2 += T(weights[i]) * r_i * r_i
    end
    reg.σ2 = sw_r2 / wsum

    return reg
end

#= Hand-rolled Newton solver shared by Bernoulli and Poisson GLM fits.
   Avoids Optim's TwiceDifferentiable + closure-object allocations: workspace
   buffers (g, H, Δ) are passed in by the caller, who allocates them once and
   reuses across columns in the multivariate case. =#

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

function _bernoulli_loss(
    β::AbstractVector{T},
    y::AbstractVector,
    w::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    prior::AbstractPrior,
) where {T<:Real}
    n, _ = size(X)
    nll = zero(T)
    for i in 1:n
        x_i = view(X, i, :)
        η_i = dot(β, x_i)
        nll += T(w[i]) * (y[i] == 1 ? log1pexp(-η_i) : log1pexp(η_i))
    end
    return nll + neglogprior(prior, β)
end

function _bernoulli_gh!(
    g::AbstractVector{T},
    H::AbstractMatrix{T},
    β::AbstractVector{T},
    y::AbstractVector,
    w::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    prior::AbstractPrior,
) where {T<:Real}
    n, p = size(X)
    fill!(g, zero(T))
    fill!(H, zero(T))
    for i in 1:n
        x_i = view(X, i, :)
        wi = T(w[i])
        μ_i = logistic(dot(β, x_i))
        r_i = wi * (μ_i - T(y[i]))
        W_i = wi * μ_i * (one(T) - μ_i)
        for a in 1:p
            xa = x_i[a]
            g[a] += r_i * xa
            wxa = W_i * xa
            for b in 1:p
                H[a, b] += wxa * x_i[b]
            end
        end
    end
    neglogprior_grad!(prior, g, β)
    neglogprior_hess!(prior, H, β)
    return nothing
end

function _poisson_loss(
    β::AbstractVector{T},
    y::AbstractVector,
    w::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    prior::AbstractPrior,
) where {T<:Real}
    n, _ = size(X)
    nll = zero(T)
    for i in 1:n
        x_i = view(X, i, :)
        η_i = clamp(dot(β, x_i), T(-500), T(500))
        nll += T(w[i]) * (exp(η_i) - T(y[i]) * η_i)
    end
    return nll + neglogprior(prior, β)
end

function _poisson_gh!(
    g::AbstractVector{T},
    H::AbstractMatrix{T},
    β::AbstractVector{T},
    y::AbstractVector,
    w::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    prior::AbstractPrior,
) where {T<:Real}
    n, p = size(X)
    fill!(g, zero(T))
    fill!(H, zero(T))
    for i in 1:n
        x_i = view(X, i, :)
        wi = T(w[i])
        η_i = clamp(dot(β, x_i), T(-500), T(500))
        eη = exp(η_i)
        r_i = wi * (eη - T(y[i]))
        W_i = wi * eη
        for a in 1:p
            xa = x_i[a]
            g[a] += r_i * xa
            wxa = W_i * xa
            for b in 1:p
                H[a, b] += wxa * x_i[b]
            end
        end
    end
    neglogprior_grad!(prior, g, β)
    neglogprior_hess!(prior, H, β)
    return nothing
end

#= Newton with backtracking line search. `loss` and `gh!` are concrete callables
   (one per family). With type-stable inputs the JIT specializes and the inner
   loop is allocation-free aside from `cholesky!`'s internal work. =#
function _newton_solve!(
    β::AbstractVector{T},
    g::AbstractVector{T},
    H::AbstractMatrix{T},
    Δ::AbstractVector{T},
    loss::F1,
    gh!::F2,
    y::AbstractVector,
    w::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    prior::AbstractPrior;
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
    ridge::Real=1e-12,
) where {T<:Real,F1,F2}
    p = length(β)
    f_curr = loss(β, y, w, X, prior)

    for _ in 1:max_iter
        gh!(g, H, β, y, w, X, prior)

        gnorm = zero(T)
        for j in 1:p
            ag = abs(g[j])
            ag > gnorm && (gnorm = ag)
        end
        gnorm < gtol && break

        # Tiny ridge for numerical PD (Bernoulli/Poisson Hessians are PSD and
        # PD when X has full column rank; this guards against rare singular cases).
        for j in 1:p
            H[j, j] += T(ridge)
        end

        F = cholesky!(Symmetric(H, :L))
        copyto!(Δ, g)
        ldiv!(F, Δ)

        # Backtracking line search: halve α until the loss decreases.
        α = one(T)
        success = false
        f_new = f_curr
        for _ in 0:max_backtrack
            for j in 1:p
                β[j] -= α * Δ[j]
            end
            f_new = loss(β, y, w, X, prior)
            if isfinite(f_new) && f_new < f_curr
                success = true
                break
            end
            for j in 1:p
                β[j] += α * Δ[j]
            end
            α /= 2
        end
        success || break
        f_curr = f_new
    end
    return β
end

#= Internal helpers used by univariate `BernoulliGLM.fit!` / `PoissonGLM.fit!`.
   Allocate the Newton workspace (g, H, Δ) — three small vectors / one matrix. =#
function _fit_bernoulli_glm!(
    β::Vector{T},
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real},
    control_seq::AbstractMatrix{<:Real},
    prior::AbstractPrior;
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"),
    )
    length(β) == p ||
        throw(DimensionMismatch("β length $(length(β)) ≠ control_seq columns $p"))

    g = Vector{T}(undef, p)
    H = Matrix{T}(undef, p, p)
    Δ = Vector{T}(undef, p)
    return _newton_solve!(
        β,
        g,
        H,
        Δ,
        _bernoulli_loss,
        _bernoulli_gh!,
        obs_seq,
        weight_seq,
        control_seq,
        prior;
        max_iter=max_iter,
        gtol=gtol,
        max_backtrack=max_backtrack,
    )
end

function _fit_poisson_glm!(
    β::Vector{T},
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real},
    control_seq::AbstractMatrix{<:Real},
    prior::AbstractPrior;
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
) where {T<:Real}
    n, p = size(control_seq)
    length(obs_seq) == n ||
        throw(DimensionMismatch("obs_seq length $(length(obs_seq)) ≠ control_seq rows $n"))
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ control_seq rows $n"),
    )
    length(β) == p ||
        throw(DimensionMismatch("β length $(length(β)) ≠ control_seq columns $p"))

    g = Vector{T}(undef, p)
    H = Matrix{T}(undef, p, p)
    Δ = Vector{T}(undef, p)
    return _newton_solve!(
        β,
        g,
        H,
        Δ,
        _poisson_loss,
        _poisson_gh!,
        obs_seq,
        weight_seq,
        control_seq,
        prior;
        max_iter=max_iter,
        gtol=gtol,
        max_backtrack=max_backtrack,
    )
end

"""
    BernoulliGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Logistic-regression emission for binary (0/1) observations.

P(Y=1|x) = σ(xᵀβ) where σ is the logistic function. `fit!` minimizes the
weighted negative log-posterior via a hand-rolled Newton solver with
backtracking line search, so any `AbstractPrior` composes without changes
to the solver.

# Fields
- `β`: coefficient vector (length p)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct BernoulliGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    prior::P
end

function BernoulliGLM(β::AbstractVector{T}) where {T<:Real}
    return BernoulliGLM{T,NoPrior}(Vector{T}(β), NoPrior())
end

function BernoulliGLM(β::AbstractVector{T}, prior::P) where {T<:Real,P<:AbstractPrior}
    return BernoulliGLM{T,P}(Vector{T}(β), prior)
end

DensityInterface.DensityKind(::BernoulliGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(
    glm::BernoulliGLM, y::Integer; control_seq::AbstractVector{<:Real}
)
    (y == 0 || y == 1) || return oftype(dot(glm.β, control_seq), -Inf)
    η = dot(glm.β, control_seq)
    return y == 1 ? -log1pexp(-η) : -log1pexp(η)
end

function Random.rand(
    rng::AbstractRNG, glm::BernoulliGLM; control_seq::AbstractVector{<:Real}
)
    return rand(rng, Bernoulli(logistic(dot(glm.β, control_seq))))
end

"""
    fit!(glm::BernoulliGLM, obs_seq, weight_seq;
         control_seq, max_iter=50, gtol=1e-8, max_backtrack=20)

Minimize the weighted negative log-posterior via a hand-rolled Newton solver
with backtracking line search. The objective is
`Σᵢ wᵢ · ℓ(β; yᵢ, xᵢ) + neglogprior(prior, β)` where `ℓ` is the Bernoulli
log-likelihood; gradient and Hessian are analytic.
"""
function StatsAPI.fit!(
    glm::BernoulliGLM,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
)
    _fit_bernoulli_glm!(
        glm.β,
        obs_seq,
        weight_seq,
        control_seq,
        glm.prior;
        max_iter=max_iter,
        gtol=gtol,
        max_backtrack=max_backtrack,
    )
    return glm
end

"""
    PoissonGLM{T<:Real, P<:AbstractPrior} <: AbstractGLM

Log-linear (Poisson) regression emission for count observations.

E[Y|x] = exp(xᵀβ). `fit!` minimizes the weighted negative log-posterior via
a hand-rolled Newton solver with backtracking line search. Any `AbstractPrior`
composes without changes to the solver.

# Fields
- `β`: coefficient vector (length p)
- `prior`: regularization prior (default `NoPrior()`)
"""
mutable struct PoissonGLM{T<:Real,P<:AbstractPrior} <: AbstractGLM
    β::Vector{T}
    prior::P
end

function PoissonGLM(β::AbstractVector{T}) where {T<:Real}
    return PoissonGLM{T,NoPrior}(Vector{T}(β), NoPrior())
end

function PoissonGLM(β::AbstractVector{T}, prior::P) where {T<:Real,P<:AbstractPrior}
    return PoissonGLM{T,P}(Vector{T}(β), prior)
end

DensityInterface.DensityKind(::PoissonGLM) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(
    glm::PoissonGLM, y::Integer; control_seq::AbstractVector{<:Real}
)
    y >= 0 || return oftype(dot(glm.β, control_seq), -Inf)
    η = dot(glm.β, control_seq)
    return y * η - exp(η) - logfactorial(y)
end

function Random.rand(rng::AbstractRNG, glm::PoissonGLM; control_seq::AbstractVector{<:Real})
    return rand(rng, Poisson(exp(dot(glm.β, control_seq))))
end

"""
    fit!(glm::PoissonGLM, obs_seq, weight_seq;
         control_seq, max_iter=50, gtol=1e-8, max_backtrack=20)

Minimize the weighted negative log-posterior via a hand-rolled Newton solver
with backtracking line search. The objective is
`Σᵢ wᵢ · ℓ(β; yᵢ, xᵢ) + neglogprior(prior, β)` where `ℓ` is the Poisson
log-likelihood; gradient and Hessian are analytic.
"""
function StatsAPI.fit!(
    glm::PoissonGLM,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
)
    _fit_poisson_glm!(
        glm.β,
        obs_seq,
        weight_seq,
        control_seq,
        glm.prior;
        max_iter=max_iter,
        gtol=gtol,
        max_backtrack=max_backtrack,
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
end

function MvGaussianGLM(
    B::AbstractMatrix{T}, Σ::AbstractMatrix{T}, prior::P
) where {T<:Real,P<:AbstractPrior}
    p, k = size(B)
    p > 0 || throw(ArgumentError("B must have at least one row"))
    k > 0 || throw(ArgumentError("B must have at least one column"))
    size(Σ) == (k, k) || throw(
        DimensionMismatch("Σ must be $(k)×$(k) for B with $(k) columns, got $(size(Σ))")
    )

    Σ_chol = try
        cholesky(Symmetric(Σ, :L))
    catch
        throw(ArgumentError("Σ must be positive definite"))
    end

    return MvGaussianGLM{T,P}(
        Matrix{T}(B), Matrix{T}(Σ), prior, Σ_chol, logdet(Σ_chol), p, k
    )
end

function MvGaussianGLM(B::AbstractMatrix{T}, Σ::AbstractMatrix{T}) where {T<:Real}
    return MvGaussianGLM(B, Σ, NoPrior())
end

function MvGaussianGLM(B::AbstractMatrix, Σ::AbstractMatrix)
    T = promote_type(eltype(B), eltype(Σ))
    Tf = float(T)
    return MvGaussianGLM(Matrix{Tf}(B), Matrix{Tf}(Σ), NoPrior())
end

function MvGaussianGLM(B::AbstractMatrix, Σ::AbstractMatrix, prior::AbstractPrior)
    T = promote_type(eltype(B), eltype(Σ))
    Tf = float(T)
    return MvGaussianGLM(Matrix{Tf}(B), Matrix{Tf}(Σ), prior)
end

DensityInterface.DensityKind(::MvGaussianGLM) = DensityInterface.HasDensity()

"""
    logdensityof(glm::MvGaussianGLM, y::AbstractVector; control_seq)

Log density of `y ∈ ℝᵏ` under the conditional MvNormal model. Allocates one
length-`k` residual vector per call — thread-safe (matches `Distributions.MvNormal`).
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
       Adjoint*Vector temporary that BLAS would otherwise allocate. =#
    for j in 1:k
        sj = zero(T)
        for r in 1:p
            sj += glm.B[r, j] * control_seq[r]
        end
        diff[j] = T(y[j]) - sj
    end
    ldiv!(glm.Σ_chol.L, diff)
    mahal² = zero(T)
    for j in 1:k
        mahal² += diff[j] * diff[j]
    end

    return -k / 2 * log(2π) - glm.logdetΣ / 2 - mahal² / 2
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
    wsum = zero(T) #= Build XᵀWX and XᵀWY in a single pass — no Y matrix, no temporaries. =#

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
            for b in 1:p
                XWX[a, b] += wxa * x_i[b]
            end
            for j in 1:k
                XWY[a, j] += wxa * T(obs_i[j])
            end
        end
    end

    #= Per-column ridge: λI added to XᵀWX gives B = (XᵀWX + λI) \ XᵀWY,
       the joint MAP for independent N(0, (1/λ)I) priors on each column of B.
       Pass a length-p view so RidgePrior loops over rows of XWX, not p*k. =#
    neglogprior_hess!(glm.prior, XWX, view(glm.B, :, 1))

    F = try
        cholesky(Symmetric(XWX, :L))
    catch
        min_eig = minimum(eigvals(Hermitian(XWX)))
        XWX .+= (abs(min_eig) + T(1e-6)) * I
        cholesky(Symmetric(XWX, :L))
    end
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

    Σ_chol_new = try
        cholesky(Symmetric(Σ_new, :L))
    catch
        min_eig = minimum(eigvals(Hermitian(Σ_new)))
        Σ_new .+= (abs(min_eig) + T(1e-6)) * I
        cholesky(Symmetric(Σ_new, :L))
    end

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
end

function MvBernoulliGLM(B::AbstractMatrix{T}, prior::P) where {T<:Real,P<:AbstractPrior}
    p, k = size(B)
    p > 0 || throw(ArgumentError("B must have at least one row"))
    k > 0 || throw(ArgumentError("B must have at least one column"))
    return MvBernoulliGLM{T,P}(Matrix{T}(B), prior, p, k)
end

MvBernoulliGLM(B::AbstractMatrix{T}) where {T<:Real} = MvBernoulliGLM(B, NoPrior())

MvBernoulliGLM(B::AbstractMatrix) = MvBernoulliGLM(float.(B), NoPrior())

MvBernoulliGLM(B::AbstractMatrix, prior::AbstractPrior) = MvBernoulliGLM(float.(B), prior)

DensityInterface.DensityKind(::MvBernoulliGLM) = DensityInterface.HasDensity()

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
         control_seq, max_iter=50, gtol=1e-8, max_backtrack=20)

Fit each output dimension independently via weighted Newton. Each observation
`obs_seq[i]` must be a length-`k` vector of 0/1 values. Newton workspace
(`g`, `H`, `Δ`) is allocated once and shared across columns.
"""
function StatsAPI.fit!(
    glm::MvBernoulliGLM{T},
    obs_seq::AbstractVector{<:AbstractVector},
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
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
    g = Vector{T}(undef, p)
    H = Matrix{T}(undef, p, p)
    Δ = Vector{T}(undef, p)

    for j in 1:k
        yview = _ColumnElementView(obs_seq, j)
        copyto!(β_buf, view(glm.B, :, j))
        _newton_solve!(
            β_buf,
            g,
            H,
            Δ,
            _bernoulli_loss,
            _bernoulli_gh!,
            yview,
            weight_seq,
            control_seq,
            glm.prior;
            max_iter=max_iter,
            gtol=gtol,
            max_backtrack=max_backtrack,
        )
        for r in 1:p
            glm.B[r, j] = β_buf[r]
        end
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
end

function MvPoissonGLM(B::AbstractMatrix{T}, prior::P) where {T<:Real,P<:AbstractPrior}
    p, k = size(B)
    p > 0 || throw(ArgumentError("B must have at least one row"))
    k > 0 || throw(ArgumentError("B must have at least one column"))
    return MvPoissonGLM{T,P}(Matrix{T}(B), prior, p, k)
end

MvPoissonGLM(B::AbstractMatrix{T}) where {T<:Real} = MvPoissonGLM(B, NoPrior())

MvPoissonGLM(B::AbstractMatrix) = MvPoissonGLM(float.(B), NoPrior())

MvPoissonGLM(B::AbstractMatrix, prior::AbstractPrior) = MvPoissonGLM(float.(B), prior)

DensityInterface.DensityKind(::MvPoissonGLM) = DensityInterface.HasDensity()

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
        yj >= 0 || return Tη(-Inf)
        η = zero(Tη)
        for r in 1:(glm.in_dim)
            η += glm.B[r, j] * control_seq[r]
        end
        lp += yj * η - exp(η) - logfactorial(yj)
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
         control_seq, max_iter=50, gtol=1e-8, max_backtrack=20)

Fit each output dimension independently via weighted Newton. Each observation
`obs_seq[i]` must be a length-`k` vector of non-negative integers. Newton
workspace (`g`, `H`, `Δ`) is allocated once and shared across columns.
"""
function StatsAPI.fit!(
    glm::MvPoissonGLM{T},
    obs_seq::AbstractVector{<:AbstractVector},
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractMatrix{<:Real},
    max_iter::Int=50,
    gtol::Real=1e-8,
    max_backtrack::Int=20,
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
    g = Vector{T}(undef, p)
    H = Matrix{T}(undef, p, p)
    Δ = Vector{T}(undef, p)

    for j in 1:k
        yview = _ColumnElementView(obs_seq, j)
        copyto!(β_buf, view(glm.B, :, j))
        _newton_solve!(
            β_buf,
            g,
            H,
            Δ,
            _poisson_loss,
            _poisson_gh!,
            yview,
            weight_seq,
            control_seq,
            glm.prior;
            max_iter=max_iter,
            gtol=gtol,
            max_backtrack=max_backtrack,
        )
        for r in 1:p
            glm.B[r, j] = β_buf[r]
        end
    end
    return glm
end
