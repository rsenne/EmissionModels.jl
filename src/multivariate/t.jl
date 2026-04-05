"""
    MultivariateT{T<:Real}

Multivariate Student's t-distribution with full covariance matrix.

# Fields
- `μ::Vector{T}`: Location vector (mean for ν > 1)
- `Σ::Matrix{T}`: Scale matrix (positive definite)
- `ν::T`: Degrees of freedom (ν > 0)

# Constructor
    MultivariateT(μ, Σ, ν)

Create a multivariate t-distribution with location `μ`, scale matrix `Σ`,
and degrees of freedom `ν`.

# Examples
```julia
μ = [0.0, 0.0]
Σ = [1.0 0.5; 0.5 1.0]
ν = 5.0
dist = MultivariateT(μ, Σ, ν)
```
"""
mutable struct MultivariateT{T<:Real}
    μ::Vector{T}
    Σ::Matrix{T}
    ν::T

    # Cached values for efficiency
    Σ_chol::Cholesky{T,Matrix{T}}
    logdetΣ::T
    dim::Int

    # Scratch buffers for fit! (sequential use only — not thread-safe for concurrent fit!)
    _diff::Vector{T}
    _z::Vector{T}

    function MultivariateT{T}(μ::Vector{T}, Σ::Matrix{T}, ν::T) where {T<:Real}
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))

        dim = length(μ)
        dim > 0 || throw(ArgumentError("μ must be non-empty"))

        size(Σ) == (dim, dim) ||
            throw(DimensionMismatch("Σ must be $(dim)×$(dim), got $(size(Σ))"))

        Σ_chol = try
            cholesky(Σ)
        catch
            throw(ArgumentError("Σ must be positive definite"))
        end

        logdetΣ = logdet(Σ_chol)

        return new{T}(μ, Σ, ν, Σ_chol, logdetΣ, dim, zeros(T, dim), zeros(T, dim))
    end
end

MultivariateT(μ::Vector{T}, Σ::Matrix{T}, ν::T) where {T<:Real} = MultivariateT{T}(μ, Σ, ν)

function MultivariateT(μ::Vector, Σ::Matrix, ν::Real)
    T = promote_type(eltype(μ), eltype(Σ), typeof(ν))
    return MultivariateT{T}(convert(Vector{T}, μ), convert(Matrix{T}, Σ), convert(T, ν))
end

"""
    MultivariateTDiag{T<:Real}

Multivariate Student's t-distribution with diagonal covariance matrix.

# Fields
- `μ::Vector{T}`: Location vector (mean for ν > 1)
- `σ²::Vector{T}`: Diagonal variances (all positive)
- `ν::T`: Degrees of freedom (ν > 0)

# Constructor
    MultivariateTDiag(μ, σ², ν)

Create a multivariate t-distribution with location `μ`, diagonal variances `σ²`,
and degrees of freedom `ν`.

# Examples
```julia
μ = [0.0, 0.0]
σ² = [1.0, 2.0]
ν = 5.0
dist = MultivariateTDiag(μ, σ², ν)
```
"""
mutable struct MultivariateTDiag{T<:Real}
    μ::Vector{T}
    σ²::Vector{T}
    ν::T

    # Cached values for efficiency
    logdetΣ::T
    dim::Int

    # Scratch buffer for fit! (sequential use only — not thread-safe for concurrent fit!)
    _diff::Vector{T}

    function MultivariateTDiag{T}(μ::Vector{T}, σ²::Vector{T}, ν::T) where {T<:Real}
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))

        dim = length(μ)
        dim > 0 || throw(ArgumentError("μ must be non-empty"))

        length(σ²) == dim ||
            throw(DimensionMismatch("σ² must have length $(dim), got $(length(σ²))"))

        all(σ²_i > 0 for σ²_i in σ²) ||
            throw(ArgumentError("All elements of σ² must be positive"))

        logdetΣ = sum(log, σ²)

        return new{T}(μ, σ², ν, logdetΣ, dim, zeros(T, dim))
    end
end

function MultivariateTDiag(μ::Vector{T}, σ²::Vector{T}, ν::T) where {T<:Real}
    return MultivariateTDiag{T}(μ, σ², ν)
end

function MultivariateTDiag(μ::Vector, σ²::Vector, ν::Real)
    T = promote_type(eltype(μ), eltype(σ²), typeof(ν))
    return MultivariateTDiag{T}(
        convert(Vector{T}, μ), convert(Vector{T}, σ²), convert(T, ν)
    )
end

# DensityInterface implementation
DensityInterface.DensityKind(::MultivariateT) = DensityInterface.HasDensity()
DensityInterface.DensityKind(::MultivariateTDiag) = DensityInterface.HasDensity()

"""
    logdensityof(dist::MultivariateT, x::AbstractVector)

Compute the log probability density of the multivariate t-distribution at `x`.

Allocates one vector (the residual). The triangular solve is performed in-place
on that vector to avoid a second allocation.
"""
function DensityInterface.logdensityof(dist::MultivariateT, x::AbstractVector)
    length(x) == dist.dim ||
        throw(DimensionMismatch("x must have length $(dist.dim), got $(length(x))"))

    d = dist.dim
    ν = dist.ν

    # 1 allocation. ldiv! solves L\(x-μ) in-place, avoiding a second allocation.
    diff = x - dist.μ
    ldiv!(dist.Σ_chol.L, diff)
    mahal² = sum(abs2, diff)

    log_norm =
        loggamma((ν + d) / 2) - loggamma(ν / 2) - (d / 2) * log(ν * π) - dist.logdetΣ / 2

    return log_norm - ((ν + d) / 2) * log1p(mahal² / ν)
end

"""
    logdensityof(dist::MultivariateTDiag, x::AbstractVector)

Compute the log probability density of the diagonal multivariate t-distribution at `x`.

Zero allocations: Mahalanobis distance is accumulated element-wise inline.
"""
function DensityInterface.logdensityof(dist::MultivariateTDiag, x::AbstractVector)
    length(x) == dist.dim ||
        throw(DimensionMismatch("x must have length $(dist.dim), got $(length(x))"))

    d = dist.dim
    ν = dist.ν

    mahal² = sum((x[i] - dist.μ[i])^2 / dist.σ²[i] for i in 1:d)

    log_norm =
        loggamma((ν + d) / 2) - loggamma(ν / 2) - (d / 2) * log(ν * π) - dist.logdetΣ / 2

    return log_norm - ((ν + d) / 2) * log1p(mahal² / ν)
end

# Random sampling

"""
    rand([rng], dist::MultivariateT)

Generate a random sample from the multivariate t-distribution.

Uses the representation: X = μ + √(ν/U) × Z
where Z ~ N(0, Σ) and U ~ χ²(ν) are independent.
"""
function Random.rand(rng::AbstractRNG, dist::MultivariateT)
    u = rand(rng, Chisq(dist.ν))
    z = randn(rng, dist.dim)
    return dist.μ .+ sqrt(dist.ν / u) .* (dist.Σ_chol.L * z)
end

"""
    rand([rng], dist::MultivariateTDiag)

Generate a random sample from the diagonal multivariate t-distribution.
"""
function Random.rand(rng::AbstractRNG, dist::MultivariateTDiag)
    u = rand(rng, Chisq(dist.ν))
    z = randn(rng, dist.dim)
    return dist.μ .+ sqrt(dist.ν / u) .* sqrt.(dist.σ²) .* z
end

"""
    fit!(dist::MultivariateT, obs_seq, weight_seq; kwargs...)

Fit the multivariate t-distribution parameters to weighted observations using
maximum likelihood estimation via EM algorithm.

# Arguments
- `dist::MultivariateT`: Distribution to update (modified in-place)
- `obs_seq`: Sequence of observations (vector of vectors)
- `weight_seq`: Sequence of weights for each observation

# Keyword Arguments
- `max_iter::Int=100`: Maximum number of EM iterations
- `tol::Float64=1e-6`: Convergence tolerance for parameters
- `fix_nu::Bool=false`: If true, keep ν fixed during estimation

# Returns
- `dist`: The updated distribution (same object as input)

# Algorithm
Uses the EM algorithm for multivariate t-distribution:
- E-step: Compute posterior weights based on Mahalanobis distance
- M-step: Update μ, Σ, and optionally ν

# Reference
Liu, C., & Rubin, D. B. (1995). ML estimation of the t distribution using EM
and its extensions, ECM and ECME. Statistica Sinica, 19-39.
"""
function StatsAPI.fit!(
    dist::MultivariateT,
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

    # Allocate working arrays once per fit! call (not per iteration)
    posterior_weights = Vector{T}(undef, n)
    Σ_acc = zeros(T, d, d)
    μ_acc = zeros(T, d)
    μ_old = copy(dist.μ)
    Σ_old = copy(dist.Σ)
    ν_old = dist.ν

    for _ in 1:max_iter
        # E-step + μ M-step accumulation (single pass; uses dist._diff and dist._z)
        fill!(μ_acc, zero(T))
        weight_post_sum = zero(T)

        for i in eachindex(obs_seq)
            w = weight_seq[i]
            if w > 0
                dist._diff .= obs_seq[i] .- dist.μ
                ldiv!(dist._z, dist.Σ_chol.L, dist._diff)
                pw = T((dist.ν + d) / (dist.ν + sum(abs2, dist._z)))
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

        # Σ M-step: rank-1 outer product accumulation using BLAS.ger!
        fill!(Σ_acc, zero(T))
        for i in eachindex(obs_seq)
            weight_seq[i] > 0 || continue
            dist._diff .= obs_seq[i] .- dist.μ
            wp = T(weight_seq[i] / weight_sum) * posterior_weights[i]
            BLAS.ger!(wp, dist._diff, dist._diff, Σ_acc)
        end

        # Symmetrize numerical noise
        for j in 1:d, k in 1:j-1
            s = (Σ_acc[j, k] + Σ_acc[k, j]) / 2
            Σ_acc[j, k] = s
            Σ_acc[k, j] = s
        end

        # Update Σ and Cholesky, regularizing if needed
        Σ_chol_new = try
            cholesky(Symmetric(Σ_acc))
        catch
            min_eig = minimum(eigvals(Hermitian(Σ_acc)))
            Σ_acc .+= (abs(min_eig) + 1e-6) * I
            cholesky(Symmetric(Σ_acc))
        end
        dist.Σ .= Σ_acc
        dist.Σ_chol = Σ_chol_new
        dist.logdetΣ = logdet(dist.Σ_chol)

        # ν M-step
        if !fix_nu
            avg_log_u_minus_u = zero(T)
            for i in eachindex(obs_seq)
                weight_seq[i] > 0 || continue
                pw = posterior_weights[i]
                avg_log_u_minus_u += T(weight_seq[i] / weight_sum) * (log(pw) - pw)
            end

            function objective(x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                return f^2
            end

            function gradient!(G, x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                return G[1] = 2 * f * df_dν * ν_val
            end

            function hessian!(H, x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                d2f_dν2 =
                    -0.25 * polygamma(2, ν_val / 2) - 1 / ν_val^2 +
                    0.25 * polygamma(2, (ν_val + d) / 2) +
                    1 / (ν_val + d)^2
                return H[1, 1] =
                    (2 * df_dν^2 + 2 * f * d2f_dν2) * ν_val^2 + 2 * f * df_dν * ν_val
            end

            td = TwiceDifferentiable(objective, gradient!, hessian!, [log(dist.ν)])
            result = optimize(td, [log(dist.ν)], Newton())
            dist.ν = exp(Optim.minimizer(result)[1])
        end

        # Convergence check (allocation-free)
        μ_diff = zero(T)
        for j in 1:d
            μ_diff = max(μ_diff, abs(dist.μ[j] - μ_old[j]))
        end
        Σ_diff = zero(T)
        for j in 1:d, k in 1:d
            Σ_diff = max(Σ_diff, abs(dist.Σ[j, k] - Σ_old[j, k]))
        end
        ν_diff = abs(dist.ν - ν_old)

        if μ_diff < tol && Σ_diff < tol && (fix_nu || ν_diff < tol)
            break
        end

        μ_old .= dist.μ
        Σ_old .= dist.Σ
        ν_old = dist.ν
    end

    return dist
end

"""
    fit!(dist::MultivariateTDiag, obs_seq, weight_seq; kwargs...)

Fit the diagonal multivariate t-distribution parameters to weighted observations
using maximum likelihood estimation via EM algorithm.

# Arguments
- `dist::MultivariateTDiag`: Distribution to update (modified in-place)
- `obs_seq`: Sequence of observations (vector of vectors)
- `weight_seq`: Sequence of weights for each observation

# Keyword Arguments
- `max_iter::Int=100`: Maximum number of EM iterations
- `tol::Float64=1e-6`: Convergence tolerance for parameters
- `fix_nu::Bool=false`: If true, keep ν fixed during estimation

# Returns
- `dist`: The updated distribution (same object as input)
"""
function StatsAPI.fit!(
    dist::MultivariateTDiag,
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

    weight_sum = zero(eltype(weight_seq))
    for w in weight_seq
        w > 0 && (weight_sum += w)
    end
    iszero(weight_sum) && return dist

    T = eltype(dist.μ)
    n = length(obs_seq)

    # Allocate working arrays once per fit! call
    posterior_weights = Vector{T}(undef, n)
    σ²_acc = zeros(T, d)
    μ_acc = zeros(T, d)
    μ_old = copy(dist.μ)
    σ²_old = copy(dist.σ²)
    ν_old = dist.ν

    for _ in 1:max_iter
        # E-step + μ M-step accumulation
        fill!(μ_acc, zero(T))
        weight_post_sum = zero(T)

        for i in eachindex(obs_seq)
            w = weight_seq[i]
            if w > 0
                mahal² = zero(T)
                for j in 1:d
                    dist._diff[j] = obs_seq[i][j] - dist.μ[j]
                    mahal² += dist._diff[j]^2 / dist.σ²[j]
                end
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

        # σ² M-step (second pass with updated μ)
        fill!(σ²_acc, zero(T))
        for i in eachindex(obs_seq)
            weight_seq[i] > 0 || continue
            wp = T(weight_seq[i] / weight_sum) * posterior_weights[i]
            for j in 1:d
                diff_j = obs_seq[i][j] - dist.μ[j]
                σ²_acc[j] += wp * diff_j^2
            end
        end
        for j in 1:d
            dist.σ²[j] = max(σ²_acc[j], T(1e-8))
        end
        dist.logdetΣ = sum(log, dist.σ²)

        # ν M-step
        if !fix_nu
            avg_log_u_minus_u = zero(T)
            for i in eachindex(obs_seq)
                weight_seq[i] > 0 || continue
                pw = posterior_weights[i]
                avg_log_u_minus_u += T(weight_seq[i] / weight_sum) * (log(pw) - pw)
            end

            function objective(x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                return f^2
            end

            function gradient!(G, x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                return G[1] = 2 * f * df_dν * ν_val
            end

            function hessian!(H, x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                d2f_dν2 =
                    -0.25 * polygamma(2, ν_val / 2) - 1 / ν_val^2 +
                    0.25 * polygamma(2, (ν_val + d) / 2) +
                    1 / (ν_val + d)^2
                return H[1, 1] =
                    (2 * df_dν^2 + 2 * f * d2f_dν2) * ν_val^2 + 2 * f * df_dν * ν_val
            end

            td = TwiceDifferentiable(objective, gradient!, hessian!, [log(dist.ν)])
            result = optimize(td, [log(dist.ν)], Newton())
            dist.ν = exp(Optim.minimizer(result)[1])
        end

        # Convergence check
        μ_diff = zero(T)
        for j in 1:d
            μ_diff = max(μ_diff, abs(dist.μ[j] - μ_old[j]))
        end
        σ²_diff = zero(T)
        for j in 1:d
            σ²_diff = max(σ²_diff, abs(dist.σ²[j] - σ²_old[j]))
        end
        ν_diff = abs(dist.ν - ν_old)

        if μ_diff < tol && σ²_diff < tol && (fix_nu || ν_diff < tol)
            break
        end

        μ_old .= dist.μ
        σ²_old .= dist.σ²
        ν_old = dist.ν
    end

    return dist
end
