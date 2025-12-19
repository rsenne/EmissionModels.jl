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
    Σ_chol::Cholesky{T,Matrix{T}}  # Cholesky decomposition of Σ
    logdetΣ::T                       # log determinant of Σ
    dim::Int                         # Dimension

    function MultivariateT{T}(μ::Vector{T}, Σ::Matrix{T}, ν::T) where {T<:Real}
        # Validation
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))

        dim = length(μ)
        dim > 0 || throw(ArgumentError("μ must be non-empty"))

        size(Σ) == (dim, dim) ||
            throw(DimensionMismatch("Σ must be $(dim)×$(dim), got $(size(Σ))"))

        # Check positive definiteness via Cholesky
        Σ_chol = try
            cholesky(Σ)
        catch
            throw(ArgumentError("Σ must be positive definite"))
        end

        logdetΣ = logdet(Σ_chol)

        return new{T}(μ, Σ, ν, Σ_chol, logdetΣ, dim)
    end
end

# Outer constructors for convenience
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
    logdetΣ::T      # sum(log.(σ²))
    dim::Int        # Dimension

    function MultivariateTDiag{T}(μ::Vector{T}, σ²::Vector{T}, ν::T) where {T<:Real}
        # Validation
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))

        dim = length(μ)
        dim > 0 || throw(ArgumentError("μ must be non-empty"))

        length(σ²) == dim ||
            throw(DimensionMismatch("σ² must have length $(dim), got $(length(σ²))"))

        all(σ²_i > 0 for σ²_i in σ²) ||
            throw(ArgumentError("All elements of σ² must be positive"))

        logdetΣ = sum(log, σ²)

        return new{T}(μ, σ², ν, logdetΣ, dim)
    end
end

# Outer constructors for convenience
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
    logdensityof(dist::MultivariateT, x::Vector)

Compute the log probability density of the multivariate t-distribution at `x`.

The log density is:
```
log p(x) = log Γ((ν+d)/2) - log Γ(ν/2) - (d/2)log(νπ) - (1/2)log|Σ|
           - ((ν+d)/2) log(1 + (1/ν)(x-μ)' Σ⁻¹ (x-μ))
```
"""
function DensityInterface.logdensityof(dist::MultivariateT, x::AbstractVector)
    length(x) == dist.dim ||
        throw(DimensionMismatch("x must have length $(dist.dim), got $(length(x))"))

    d = dist.dim
    ν = dist.ν

    # Compute (x - μ)' Σ⁻¹ (x - μ) efficiently using Cholesky
    diff = x .- dist.μ
    z = dist.Σ_chol.L \ diff  # Solve L z = diff
    mahal² = sum(abs2, z)      # ||z||² = (x-μ)' Σ⁻¹ (x-μ)

    # Log density computation
    log_norm =
        loggamma((ν + d) / 2) - loggamma(ν / 2) - (d / 2) * log(ν * π) - dist.logdetΣ / 2

    log_kernel = -((ν + d) / 2) * log1p(mahal² / ν)

    return log_norm + log_kernel
end

"""
    logdensityof(dist::MultivariateTDiag, x::Vector)

Compute the log probability density of the diagonal multivariate t-distribution at `x`.
"""
function DensityInterface.logdensityof(dist::MultivariateTDiag, x::AbstractVector)
    length(x) == dist.dim ||
        throw(DimensionMismatch("x must have length $(dist.dim), got $(length(x))"))

    d = dist.dim
    ν = dist.ν

    # Compute (x - μ)' Σ⁻¹ (x - μ) for diagonal Σ
    diff = x .- dist.μ
    mahal² = sum(diff[i]^2 / dist.σ²[i] for i in 1:d)

    # Log density computation
    log_norm =
        loggamma((ν + d) / 2) - loggamma(ν / 2) - (d / 2) * log(ν * π) - dist.logdetΣ / 2

    log_kernel = -((ν + d) / 2) * log1p(mahal² / ν)

    return log_norm + log_kernel
end

# sampling

"""
    rand([rng], dist::MultivariateT)

Generate a random sample from the multivariate t-distribution.

Uses the representation: X = μ + √(ν/U) × Z
where Z ~ N(0, Σ) and U ~ χ²(ν) are independent.
"""
function Random.rand(rng::AbstractRNG, dist::MultivariateT)
    # Sample from chi-squared with ν degrees of freedom
    u = rand(rng, Chisq(dist.ν))

    # Sample from N(0, I) and transform to N(0, Σ)
    z = randn(rng, dist.dim)
    scaled_noise = dist.Σ_chol.L * z  # L z ~ N(0, Σ) where Σ = L L'

    # Apply t-distribution scaling
    return dist.μ .+ sqrt(dist.ν / u) .* scaled_noise
end

"""
    rand([rng], dist::MultivariateTDiag)

Generate a random sample from the diagonal multivariate t-distribution.
"""
function Random.rand(rng::AbstractRNG, dist::MultivariateTDiag)
    # Sample from chi-squared with ν degrees of freedom
    u = rand(rng, Chisq(dist.ν))

    # Sample from N(0, diag(σ²))
    z = randn(rng, dist.dim)
    scaled_noise = sqrt.(dist.σ²) .* z

    # Apply t-distribution scaling
    return dist.μ .+ sqrt(dist.ν / u) .* scaled_noise
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

    # Check dimensions
    d = dist.dim
    for (i, obs) in enumerate(obs_seq)
        length(obs) == d || throw(
            DimensionMismatch("Observation $i has length $(length(obs)), expected $d")
        )
    end

    # Filter out zero-weight observations
    nonzero_idx = findall(w -> w > 0, weight_seq)
    isempty(nonzero_idx) && return dist

    obs_vec = [obs_seq[i] for i in nonzero_idx]
    weights = [weight_seq[i] for i in nonzero_idx]
    n = length(obs_vec)

    # Normalize weights
    weight_sum = sum(weights)
    weights ./= weight_sum

    # EM algorithm
    μ_old = copy(dist.μ)
    Σ_old = copy(dist.Σ)
    ν_old = dist.ν

    for iter in 1:max_iter
        # E-step: Compute posterior weights
        # w_i = (ν + d) / (ν + δ_i²) where δ_i² is Mahalanobis distance
        posterior_weights = zeros(n)

        for i in 1:n
            diff = obs_vec[i] .- dist.μ
            z = dist.Σ_chol.L \ diff
            mahal² = sum(abs2, z)
            posterior_weights[i] = (dist.ν + d) / (dist.ν + mahal²)
        end

        # M-step: Update parameters
        # Update μ
        weighted_sum = sum(weights[i] * posterior_weights[i] * obs_vec[i] for i in 1:n)
        weight_post_sum = sum(weights[i] * posterior_weights[i] for i in 1:n)
        dist.μ .= weighted_sum ./ weight_post_sum

        # Update Σ
        Σ_new = zeros(d, d)
        for i in 1:n
            diff = obs_vec[i] .- dist.μ
            Σ_new .+= weights[i] * posterior_weights[i] * (diff * diff')
        end
        Σ_new ./= sum(weights)

        # Ensure symmetry and positive definiteness
        Σ_new .= (Σ_new .+ Σ_new') ./ 2

        # Add small regularization if needed
        min_eig = minimum(eigvals(Hermitian(Σ_new)))
        if min_eig <= 0
            Σ_new .+= (abs(min_eig) + 1e-6) * I
        end

        dist.Σ .= Σ_new
        dist.Σ_chol = cholesky(dist.Σ)
        dist.logdetΣ = logdet(dist.Σ_chol)

        # Update ν (if not fixed)
        if !fix_nu
            # Use Optim.jl with Newton's method to maximize the Q function w.r.t. ν
            # The equation to solve is:
            # -ψ(ν/2) + log(ν/2) + 1 + (1/n)Σw_i(log(u_i) - u_i) + ψ((ν+d)/2) - log((ν+d)/2) = 0
            # where u_i are the posterior weights

            avg_log_u_minus_u = sum(
                weights[i] * (log(posterior_weights[i]) - posterior_weights[i]) for i in 1:n
            )

            # Optimize over log(ν) to ensure ν > 0
            # Let x = log(ν), so ν = exp(x)
            function objective(x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                return f^2  # Minimize squared residual
            end

            function gradient!(G, x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                # df/dν
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                # d(f²)/dx = d(f²)/dν * dν/dx = 2f * df/dν * ν (since dν/dx = ν)
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
                # df/dν
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                # d²f/dν²
                d2f_dν2 =
                    -0.25 * polygamma(2, ν_val / 2) - 1 / ν_val^2 +
                    0.25 * polygamma(2, (ν_val + d) / 2) +
                    1 / (ν_val + d)^2
                # d²(f²)/dx² using chain rule: d²(f²)/dx² = [2(df/dν)² + 2f*d²f/dν²]*ν² + 2f*df/dν*ν
                return H[1, 1] =
                    (2 * df_dν^2 + 2 * f * d2f_dν2) * ν_val^2 + 2 * f * df_dν * ν_val
            end

            # Use Newton's method with analytical derivatives
            td = TwiceDifferentiable(objective, gradient!, hessian!, [log(dist.ν)])
            result = optimize(td, [log(dist.ν)], Newton())
            dist.ν = exp(Optim.minimizer(result)[1])
        end

        # Check convergence
        μ_diff = maximum(abs.(dist.μ .- μ_old))
        Σ_diff = maximum(abs.(dist.Σ .- Σ_old))
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

    # Check dimensions
    d = dist.dim
    for (i, obs) in enumerate(obs_seq)
        length(obs) == d || throw(
            DimensionMismatch("Observation $i has length $(length(obs)), expected $d")
        )
    end

    # Filter out zero-weight observations
    nonzero_idx = findall(w -> w > 0, weight_seq)
    isempty(nonzero_idx) && return dist

    obs_vec = [obs_seq[i] for i in nonzero_idx]
    weights = [weight_seq[i] for i in nonzero_idx]
    n = length(obs_vec)

    # Normalize weights
    weight_sum = sum(weights)
    weights ./= weight_sum

    # EM algorithm
    μ_old = copy(dist.μ)
    σ²_old = copy(dist.σ²)
    ν_old = dist.ν

    for iter in 1:max_iter
        # E-step: Compute posterior weights
        posterior_weights = zeros(n)

        for i in 1:n
            diff = obs_vec[i] .- dist.μ
            mahal² = sum(diff[j]^2 / dist.σ²[j] for j in 1:d)
            posterior_weights[i] = (dist.ν + d) / (dist.ν + mahal²)
        end

        # M-step: Update parameters
        # Update μ
        weighted_sum = sum(weights[i] * posterior_weights[i] * obs_vec[i] for i in 1:n)
        weight_post_sum = sum(weights[i] * posterior_weights[i] for i in 1:n)
        dist.μ .= weighted_sum ./ weight_post_sum

        # Update σ² (diagonal variances)
        σ²_new = zeros(d)
        for j in 1:d
            for i in 1:n
                diff_j = obs_vec[i][j] - dist.μ[j]
                σ²_new[j] += weights[i] * posterior_weights[i] * diff_j^2
            end
            σ²_new[j] /= sum(weights)
            σ²_new[j] = max(σ²_new[j], 1e-8)  # Ensure positivity
        end

        dist.σ² .= σ²_new
        dist.logdetΣ = sum(log, dist.σ²)

        # Update ν (if not fixed)
        if !fix_nu
            avg_log_u_minus_u = sum(
                weights[i] * (log(posterior_weights[i]) - posterior_weights[i]) for i in 1:n
            )

            # Optimize over log(ν) to ensure ν > 0
            # Let x = log(ν), so ν = exp(x)
            function objective(x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                return f^2  # Minimize squared residual
            end

            function gradient!(G, x::Vector)
                ν_val = exp(x[1])
                f =
                    -digamma(ν_val / 2) +
                    log(ν_val / 2) +
                    1 +
                    avg_log_u_minus_u +
                    digamma((ν_val + d) / 2) - log((ν_val + d) / 2)
                # df/dν
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                # d(f²)/dx = d(f²)/dν * dν/dx = 2f * df/dν * ν (since dν/dx = ν)
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
                # df/dν
                df_dν =
                    -0.5 * trigamma(ν_val / 2) +
                    1 / ν_val +
                    0.5 * trigamma((ν_val + d) / 2) - 1 / (ν_val + d)
                # d²f/dν²
                d2f_dν2 =
                    -0.25 * polygamma(2, ν_val / 2) - 1 / ν_val^2 +
                    0.25 * polygamma(2, (ν_val + d) / 2) +
                    1 / (ν_val + d)^2
                # d²(f²)/dx² using chain rule: d²(f²)/dx² = [2(df/dν)² + 2f*d²f/dν²]*ν² + 2f*df/dν*ν
                return H[1, 1] =
                    (2 * df_dν^2 + 2 * f * d2f_dν2) * ν_val^2 + 2 * f * df_dν * ν_val
            end

            # Use Newton's method with analytical derivatives
            td = TwiceDifferentiable(objective, gradient!, hessian!, [log(dist.ν)])
            result = optimize(td, [log(dist.ν)], Newton())
            dist.ν = exp(Optim.minimizer(result)[1])
        end

        # Check convergence
        μ_diff = maximum(abs.(dist.μ .- μ_old))
        σ²_diff = maximum(abs.(dist.σ² .- σ²_old))
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
