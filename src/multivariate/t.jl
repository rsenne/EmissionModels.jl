"""
    MvT{T<:Real}

Multivariate Student's t-distribution with full covariance matrix.

# Fields
- `μ::Vector{T}`: Location vector (mean for ν > 1)
- `Σ::Matrix{T}`: Scale matrix (positive definite)
- `ν::T`: Degrees of freedom (ν > 0)

# Constructor
    MvT(μ, Σ, ν)

Create a multivariate t-distribution with location `μ`, scale matrix `Σ`,
and degrees of freedom `ν`.

# Examples
```julia
μ = [0.0, 0.0]
Σ = [1.0 0.5; 0.5 1.0]
ν = 5.0
dist = MvT(μ, Σ, ν)
```
"""
mutable struct MvT{T<:Real}
    μ::Vector{T}
    Σ::Matrix{T}
    ν::T

    # Cached values for efficiency
    Σ_chol::Cholesky{T,Matrix{T}}
    logdetΣ::T
    dim::Int

    function MvT{T}(μ::Vector{T}, Σ::Matrix{T}, ν::T) where {T<:Real}
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))

        dim = length(μ)
        dim > 0 || throw(ArgumentError("μ must be non-empty"))

        size(Σ) == (dim, dim) ||
            throw(DimensionMismatch("Σ must be $(dim)×$(dim), got $(size(Σ))"))

        Σ_chol = cholesky(Symmetric(Σ, :L); check=false)
        issuccess(Σ_chol) || throw(ArgumentError("Σ must be positive definite"))

        logdetΣ = logdet(Σ_chol)

        #= Copy μ and Σ so the struct never aliases caller arrays: an external
           mutation of Σ would silently desync the cached Cholesky/logdetΣ. =#
        return new{T}(copy(μ), copy(Σ), ν, Σ_chol, logdetΣ, dim)
    end
end

MvT(μ::Vector{T}, Σ::Matrix{T}, ν::T) where {T<:Real} = MvT{T}(μ, Σ, ν)

function MvT(μ::Vector, Σ::Matrix, ν::Real)
    T = promote_type(eltype(μ), eltype(Σ), typeof(ν))
    return MvT{T}(convert(Vector{T}, μ), convert(Matrix{T}, Σ), convert(T, ν))
end

"""
    MvTDiag{T<:Real}

Multivariate Student's t-distribution with diagonal covariance matrix.

# Fields
- `μ::Vector{T}`: Location vector (mean for ν > 1)
- `σ²::Vector{T}`: Diagonal variances (all positive)
- `ν::T`: Degrees of freedom (ν > 0)

# Constructor
    MvTDiag(μ, σ², ν)

Create a multivariate t-distribution with location `μ`, diagonal variances `σ²`,
and degrees of freedom `ν`.

# Examples
```julia
μ = [0.0, 0.0]
σ² = [1.0, 2.0]
ν = 5.0
dist = MvTDiag(μ, σ², ν)
```
"""
mutable struct MvTDiag{T<:Real}
    μ::Vector{T}
    σ²::Vector{T}
    ν::T

    # Cached values for efficiency
    logdetΣ::T
    dim::Int

    function MvTDiag{T}(μ::Vector{T}, σ²::Vector{T}, ν::T) where {T<:Real}
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))

        dim = length(μ)
        dim > 0 || throw(ArgumentError("μ must be non-empty"))

        length(σ²) == dim ||
            throw(DimensionMismatch("σ² must have length $(dim), got $(length(σ²))"))

        all(σ²_i > 0 for σ²_i in σ²) ||
            throw(ArgumentError("All elements of σ² must be positive"))

        logdetΣ = sum(log, σ²)

        #= Copy so the struct never aliases caller arrays (mutating σ² externally
           would silently desync the cached logdetΣ). =#
        return new{T}(copy(μ), copy(σ²), ν, logdetΣ, dim)
    end
end

function MvTDiag(μ::Vector{T}, σ²::Vector{T}, ν::T) where {T<:Real}
    return MvTDiag{T}(μ, σ², ν)
end

function MvTDiag(μ::Vector, σ²::Vector, ν::Real)
    T = promote_type(eltype(μ), eltype(σ²), typeof(ν))
    return MvTDiag{T}(convert(Vector{T}, μ), convert(Vector{T}, σ²), convert(T, ν))
end

# DensityInterface implementation
DensityInterface.DensityKind(::MvT) = DensityInterface.HasDensity()
DensityInterface.DensityKind(::MvTDiag) = DensityInterface.HasDensity()

#= Log density of a multivariate Student-t given the precomputed Mahalanobis²
   `mahal² = (x-μ)ᵀ Σ⁻¹ (x-μ)`. The full-covariance and diagonal variants differ
   only in how `mahal²` and `logdetΣ` are formed, so the normalisation constant
   and the `log1p` tail are defined here once. =#
function _t_logpdf(ν, d, logdetΣ, mahal²)
    #= Work in the common float type of the inputs; `d` enters via T(d) so the
       integer never promotes a Float32 computation to Float64. =#
    T = float(promote_type(typeof(ν), typeof(logdetΣ), typeof(mahal²)))
    νT = T(ν)
    half_νd = (νT + T(d)) / 2
    log_norm =
        loggamma(half_νd) - loggamma(νT / 2) - (T(d) / 2) * log(νT * T(π)) - T(logdetΣ) / 2
    return log_norm - half_νd * log1p(T(mahal²) / νT)
end

"""
    logdensityof(dist::MvT, x::AbstractVector)

Compute the log probability density of the multivariate t-distribution at `x`.

Allocates one vector (the residual). The triangular solve is performed in-place
on that vector to avoid a second allocation.
"""
function DensityInterface.logdensityof(dist::MvT, x::AbstractVector)
    length(x) == dist.dim ||
        throw(DimensionMismatch("x must have length $(dist.dim), got $(length(x))"))

    #= Allocate the residual locally (one length-d vector). The struct holds no
       mutable scratch, so concurrent `logdensityof`/`rand` on a shared dist are
       safe; `fit!` likewise allocates its scratch per call. =#
    diff = x - dist.μ
    ldiv!(dist.Σ_chol.L, diff)
    mahal² = zero(eltype(diff))
    for i in eachindex(diff)
        mahal² += diff[i] * diff[i]
    end

    return _t_logpdf(dist.ν, dist.dim, dist.logdetΣ, mahal²)
end

"""
    logdensityof(dist::MvTDiag, x::AbstractVector)

Compute the log probability density of the diagonal multivariate t-distribution at `x`.

Zero allocations: Mahalanobis distance is accumulated element-wise inline.
"""
function DensityInterface.logdensityof(dist::MvTDiag, x::AbstractVector)
    length(x) == dist.dim ||
        throw(DimensionMismatch("x must have length $(dist.dim), got $(length(x))"))

    d = dist.dim
    mahal² = sum((x[i] - dist.μ[i])^2 / dist.σ²[i] for i in 1:d)

    return _t_logpdf(dist.ν, d, dist.logdetΣ, mahal²)
end

# Random sampling
"""
    rand([rng], dist::MvT)

Generate a random sample from the multivariate t-distribution.

Uses the representation: X = μ + √(ν/U) × Z
where Z ~ N(0, Σ) and U ~ χ²(ν) are independent.
"""
function Random.rand(rng::AbstractRNG, dist::MvT{T}) where {T<:Real}
    return rand!(rng, dist, Vector{T}(undef, dist.dim))
end

"""
    rand!(rng, dist::MvT, out)

In-place sample into `out` (length `dim`). Zero allocation, thread-safe: draw
`z ~ N(0, I)` directly into `out`, multiply by the Cholesky factor in place
(`lmul!`), then scale by √(ν/u) and shift by μ element-wise.
"""
function Random.rand!(rng::AbstractRNG, dist::MvT, out::AbstractVector)
    length(out) == dist.dim ||
        throw(DimensionMismatch("out length $(length(out)) ≠ dim $(dist.dim)"))

    u = rand(rng, Chisq(dist.ν))
    s = sqrt(dist.ν / u)
    randn!(rng, out)
    lmul!(dist.Σ_chol.L, out)        # out = L * z, in place
    for i in 1:(dist.dim)
        out[i] = dist.μ[i] + s * out[i]
    end
    return out
end

"""
    rand([rng], dist::MvTDiag)

Generate a random sample from the diagonal multivariate t-distribution.
"""
function Random.rand(rng::AbstractRNG, dist::MvTDiag{T}) where {T<:Real}
    return rand!(rng, dist, Vector{T}(undef, dist.dim))
end

"""
    rand!(rng, dist::MvTDiag, out)

In-place sample into `out` (length `dim`). Zero allocation, thread-safe.
"""
function Random.rand!(rng::AbstractRNG, dist::MvTDiag, out::AbstractVector)
    length(out) == dist.dim ||
        throw(DimensionMismatch("out length $(length(out)) ≠ dim $(dist.dim)"))

    u = rand(rng, Chisq(dist.ν))
    s = sqrt(dist.ν / u)
    randn!(rng, out)
    for i in 1:(dist.dim)
        out[i] = dist.μ[i] + s * sqrt(dist.σ²[i]) * out[i]
    end
    return out
end

#= Array forms (`rand(dist, n)` etc.) go through Random's sampler machinery,
   which needs the sample eltype and a method on the default trivial sampler.
   Samples are length-`dim` vectors, matching the obs_seq convention. =#
Base.eltype(::Type{MvT{T}}) where {T<:Real} = Vector{T}
Base.eltype(::Type{MvTDiag{T}}) where {T<:Real} = Vector{T}
function Random.rand(rng::AbstractRNG, sp::Random.SamplerTrivial{<:Union{MvT,MvTDiag}})
    return rand(rng, sp[])
end

#= ECME degrees-of-freedom update, shared by the `MvT` and
   `MvTDiag` fits. The ν M-step solves the stationarity condition

       f(ν) = -ψ(ν/2) + log(ν/2) + 1 + C + ψ((ν+d)/2) - log((ν+d)/2) = 0,

   where C = Σᵢ w̃ᵢ (log uᵢ - uᵢ) is the supplied `avg_log_u_minus_u` (uᵢ are the
   per-observation posterior weights). f decreases from +∞ (ν → 0) to 1 + C ≤ 0
   (ν → ∞; log u - u ≤ -1 by Jensen), so a bracketed Newton on x = log ν with
   bisection fallback always converges. This runs once per EM iteration, so it
   is solved inline rather than through Optim: zero allocations, and the
   bracket doubles as a hard cap — near-Gaussian data (C → -1) pushes the root
   toward ν = ∞, and the cap returns a large-but-finite ν instead of letting
   exp(x) overflow into the next E-step. =#
function _nu_f(ν, C, d)
    return -digamma(ν / 2) + log(ν / 2) + 1 + C + digamma((ν + d) / 2) - log((ν + d) / 2)
end
# `/2` rather than `0.5 *` keeps the derivative in ν's own type (a Float64 `0.5`
# would make the log-space Newton step type-unstable for Float32 ν).
_nu_df(ν, d) = -trigamma(ν / 2) / 2 + 1 / ν + trigamma((ν + d) / 2) / 2 - 1 / (ν + d)

function _update_nu(ν0::Real, avg_log_u_minus_u, d::Integer)
    C = avg_log_u_minus_u
    T = float(promote_type(typeof(ν0), typeof(C)))
    ν_lo, ν_hi = T(1e-2), T(1e6)
    _nu_f(ν_lo, C, d) <= 0 && return ν_lo
    _nu_f(ν_hi, C, d) >= 0 && return ν_hi

    lo, hi = log(ν_lo), log(ν_hi)
    x = clamp(log(T(ν0)), lo, hi)
    tol = sqrt(eps(T))
    for _ in 1:100
        ν = exp(x)
        fx = _nu_f(ν, C, d)
        abs(fx) < tol && break
        # Maintain the sign-change bracket (f decreasing: f(lo) > 0 > f(hi)).
        if fx > 0
            lo = x
        else
            hi = x
        end
        # Newton step in log-space; bisect whenever it leaves the bracket.
        x_new = x - fx / (_nu_df(ν, d) * ν)
        x = lo < x_new < hi ? x_new : (lo + hi) / 2
        hi - lo < tol && break
    end
    return exp(x)
end

#= Type-specific hooks for the shared multivariate-t EM driver. The two variants
   differ only in how the scale parameter (Σ matrix or σ² vector) is stored and
   updated; these hooks isolate those differences so the EM scaffold below can be
   written once. =#
_scale_current(dist::MvT) = dist.Σ
_scale_current(dist::MvTDiag) = dist.σ²

_scale_snapshot(dist) = copy(_scale_current(dist))
_scale_snapshot!(buf, dist) = copyto!(buf, _scale_current(dist))
function _scale_maxdiff(dist, scale_old)
    s = _scale_current(dist)
    m = zero(eltype(s))
    for i in eachindex(s, scale_old)
        m = max(m, abs(s[i] - scale_old[i]))
    end
    return m
end

#= Per-call EM scratch, allocated once per `fit!` and threaded through the hooks
   so the distribution structs carry no mutable state: `fit!` on distinct dists
   and concurrent `logdensityof`/`rand` on a shared dist are then all thread-safe. =#
function _em_workspace(dist::MvT)
    T = eltype(dist.μ)
    d = dist.dim
    return (scatter=zeros(T, d, d), diff=Vector{T}(undef, d), z=Vector{T}(undef, d))
end
_em_workspace(dist::MvTDiag) = (scatter=zeros(eltype(dist.μ), dist.dim),)

#= E-step Mahalanobis² for one observation. The full-covariance variant uses the
   workspace residual buffers; the diagonal variant needs none (ws is ignored). =#
function _mahalanobis²!(dist::MvT, ws, obs_i)
    ws.diff .= obs_i .- dist.μ
    ldiv!(ws.z, dist.Σ_chol.L, ws.diff)
    return sum(abs2, ws.z)
end
function _mahalanobis²!(dist::MvTDiag, ws, obs_i)
    mahal² = zero(eltype(dist.μ))
    for j in 1:(dist.dim)
        δ = obs_i[j] - dist.μ[j]
        mahal² += δ^2 / dist.σ²[j]
    end
    return mahal²
end

#= Rank-1 update A += α·x·xᵀ. BLAS.ger! covers the fast Float32/Float64 path;
   the generic fallback keeps `fit!` working for any Real eltype (the struct
   accepts any `T<:Real`, e.g. BigFloat, which BLAS would reject). =#
function _rank1_update!(A::Matrix{T}, α::T, x::Vector{T}) where {T<:LinearAlgebra.BlasReal}
    return BLAS.ger!(α, x, x, A)
end
function _rank1_update!(A::AbstractMatrix, α::Real, x::AbstractVector)
    for j in eachindex(x)
        αxj = α * x[j]
        for i in eachindex(x)
            A[i, j] += x[i] * αxj
        end
    end
    return A
end

#= Scale M-step: recompute the scale from residuals against the updated μ,
   weighted by wᵢ·pwᵢ / Σw, then refresh the cached `logdetΣ` (and Cholesky). =#
function _scatter_mstep!(dist::MvT, ws, obs_seq, weight_seq, posterior_weights, weight_sum)
    T = eltype(dist.μ)
    d = dist.dim
    Σ_acc = ws.scatter
    fill!(Σ_acc, zero(T))
    for i in eachindex(obs_seq)
        weight_seq[i] > 0 || continue
        ws.diff .= obs_seq[i] .- dist.μ
        wp = T(weight_seq[i] / weight_sum) * posterior_weights[i]
        _rank1_update!(Σ_acc, wp, ws.diff)
    end
    # Symmetrize numerical noise.
    for j in 1:d, k in 1:(j - 1)
        s = (Σ_acc[j, k] + Σ_acc[k, j]) / 2
        Σ_acc[j, k] = s
        Σ_acc[k, j] = s
    end
    #= Σ_acc is PSD by construction and PD whenever the weighted residuals span
       ℝᵈ. Failure means a degenerate observation set (zero variance along some
       axis), so surface that rather than silently regularizing. =#
    Σ_chol_new = cholesky(Symmetric(Σ_acc, :L); check=false)
    issuccess(Σ_chol_new) || throw(
        ArgumentError(
            "Σ M-step is not positive definite: observations are " *
            "degenerate along at least one dimension. Inspect `obs_seq` " *
            "for collinear or constant components.",
        ),
    )
    dist.Σ .= Σ_acc
    dist.Σ_chol = Σ_chol_new
    dist.logdetΣ = logdet(dist.Σ_chol)
    return dist
end
function _scatter_mstep!(
    dist::MvTDiag, ws, obs_seq, weight_seq, posterior_weights, weight_sum
)
    T = eltype(dist.μ)
    d = dist.dim
    σ²_acc = ws.scatter
    fill!(σ²_acc, zero(T))
    for i in eachindex(obs_seq)
        weight_seq[i] > 0 || continue
        wp = T(weight_seq[i] / weight_sum) * posterior_weights[i]
        for j in 1:d
            diff_j = obs_seq[i][j] - dist.μ[j]
            σ²_acc[j] += wp * diff_j^2
        end
    end
    #= Type-aware variance floor (about 1.5e-8 at Float64, scales with precision)
       to keep σ² strictly positive without a hardcoded Float64 constant. =#
    var_floor = sqrt(eps(T))
    for j in 1:d
        dist.σ²[j] = max(σ²_acc[j], var_floor)
    end
    dist.logdetΣ = sum(log, dist.σ²)
    return dist
end

"""
    fit!(dist::MvT,     obs_seq, weight_seq; kwargs...)
    fit!(dist::MvTDiag, obs_seq, weight_seq; kwargs...)

Fit a multivariate Student-t emission to weighted observations by EM (ECME), in
place. A single driver serves both the full-covariance and diagonal variants;
they differ only in the scale-parameter hooks (`_mahalanobis²!`,
`_scatter_mstep!`, `_scale_current`).

# Arguments
- `dist`: distribution to update (modified in-place)
- `obs_seq`: sequence of observations (vector of vectors)
- `weight_seq`: per-observation weights (e.g. HMM posterior state probabilities)

# Keyword Arguments
- `max_iter::Int=100`: maximum number of EM iterations
- `tol::Real=1e-6`: convergence tolerance on the parameters
- `fix_nu::Bool=false`: if true, keep ν fixed during estimation

# Algorithm
- E-step: posterior weights `pw = (ν+d)/(ν+mahal²)` from the Mahalanobis distance
- M-step: update μ (pw-weighted), then the scale (Σ or σ²), then ν (ECME)

# Reference
Liu, C., & Rubin, D. B. (1995). ML estimation of the t distribution using EM
and its extensions, ECM and ECME. Statistica Sinica, 19-39.
"""
function StatsAPI.fit!(
    dist::Union{MvT,MvTDiag},
    obs_seq,
    weight_seq;
    max_iter::Int=100,
    tol::Real=1e-6,
    fix_nu::Bool=false,
)
    length(obs_seq) == length(weight_seq) ||
        throw(DimensionMismatch("obs_seq and weight_seq must have the same length"))

    isempty(obs_seq) && return dist

    d = dist.dim
    for (i, obs) in enumerate(obs_seq)
        length(obs) == d || throw(
            DimensionMismatch("Observation $i has length $(length(obs)), expected $d")
        )
    end

    # Total weight (skipping zeros avoids inflating the denominator)
    weight_sum = zero(eltype(weight_seq))
    for w in weight_seq
        w > 0 && (weight_sum += w)
    end
    iszero(weight_sum) && return dist

    T = eltype(dist.μ)
    n = length(obs_seq)

    # Working buffers allocated once per call (not per iteration).
    posterior_weights = Vector{T}(undef, n)
    ws = _em_workspace(dist)
    μ_acc = zeros(T, d)
    μ_old = copy(dist.μ)
    scale_old = _scale_snapshot(dist)
    ν_old = dist.ν

    for _ in 1:max_iter
        # E-step + μ M-step accumulation (single pass).
        fill!(μ_acc, zero(T))
        weight_post_sum = zero(T)
        for i in eachindex(obs_seq)
            w = weight_seq[i]
            if w > 0
                mahal² = _mahalanobis²!(dist, ws, obs_seq[i])
                pw = T((dist.ν + d) / (dist.ν + mahal²))
                posterior_weights[i] = pw
                wp = T(w / weight_sum) * pw
                for j in 1:d
                    μ_acc[j] += wp * obs_seq[i][j]
                end
                weight_post_sum += wp
            else
                posterior_weights[i] = zero(T)
            end
        end
        dist.μ .= μ_acc ./ weight_post_sum

        # Scale M-step (Σ or σ²): second residual pass + logdetΣ refresh.
        _scatter_mstep!(dist, ws, obs_seq, weight_seq, posterior_weights, weight_sum)

        # ν M-step (ECME)
        if !fix_nu
            avg_log_u_minus_u = zero(T)
            for i in eachindex(obs_seq)
                weight_seq[i] > 0 || continue
                pw = posterior_weights[i]
                avg_log_u_minus_u += T(weight_seq[i] / weight_sum) * (log(pw) - pw)
            end
            dist.ν = _update_nu(dist.ν, avg_log_u_minus_u, d)
        end

        # Convergence check (allocation-free).
        μ_diff = zero(T)
        for j in 1:d
            μ_diff = max(μ_diff, abs(dist.μ[j] - μ_old[j]))
        end
        scale_diff = _scale_maxdiff(dist, scale_old)
        ν_diff = abs(dist.ν - ν_old)
        if μ_diff < tol && scale_diff < tol && (fix_nu || ν_diff < tol)
            break
        end

        μ_old .= dist.μ
        _scale_snapshot!(scale_old, dist)
        ν_old = dist.ν
    end

    return dist
end
