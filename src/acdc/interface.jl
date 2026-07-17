#=
ACDC: Accumulated Cutoff Discrepancy Criterion.

Robust model selection that measures component-level discrepancy via the
"stochastic drivers" framework (Li et al., 2026). The generative process is

```math
\\begin{aligned}
    x_n &= \\sum_k y_{n,k} \\\\
    y_{n,k} &= f(z_{n,k}, \\phi_k, \\varepsilon_{n,k}) \\quad \\text{where } \\varepsilon_{n,k} \\sim U(0,1)
\\end{aligned}
```

If the model is correctly specified, the recovered drivers ``\\varepsilon_{n,k}``
are uniform on ``[0,1]``. ACDC scores each component by how far its drivers
deviate from uniformity, then picks the component count minimizing the hinge
loss ``R^\\rho(K) = \\sum_k \\max(0, \\hat{D}_k - \\rho)`` at cutoff ``\\rho``,
breaking ties toward smaller `K` (see `acdc_loss`/`acdc_select`).

This file holds the model-agnostic core: result types, discrepancy measures,
and the loss/selection machinery. Model-specific recovery of the drivers is
done by `stochastic_drivers`, whose methods live alongside the model types they
support.
=#

"""
    ComponentDiscrepancy

Abstract supertype for discrepancy measures between a sample of stochastic
drivers and the reference ``U([0,1]^D)`` distribution.
"""
abstract type ComponentDiscrepancy end

"""
    StochasticDriverResult{T<:Real}

Stochastic drivers recovered from a fitted model.

# Fields
- `ε_pools::Vector{Matrix{T}}`: per-component driver pools, each a `D × n_k`
  matrix where `n_k` is the number of samples assigned to component `k`.
- `usage::Vector{T}`: per-component contribution magnitudes (length `K`).
"""
struct StochasticDriverResult{T<:Real}
    ε_pools::Vector{Matrix{T}}
    usage::Vector{T}

    function StochasticDriverResult(
        ε_pools::Vector{Matrix{T}}, usage::Vector{T}
    ) where {T<:Real}
        K = length(ε_pools)
        length(usage) == K || throw(ArgumentError("usage must have length K=$(K)"))
        #= All non-empty pools must share the driver dimension D (rows); a
           mismatch would otherwise be silently mis-scored downstream. =#
        nonempty = Iterators.filter(!isempty, ε_pools)
        D = iterate(nonempty)
        if D !== nothing
            d1 = size(D[1], 1)
            all(size(p, 1) == d1 for p in nonempty) ||
                throw(ArgumentError("all non-empty ε_pools must share row dimension D"))
        end
        return new{T}(ε_pools, usage)
    end
end

"""
    ACDCResult{T<:Real}

Per-component ACDC discrepancies for a model with `K` components.

# Fields
- `K::Int`: number of components.
- `component_discrepancies::Vector{T}`: per-component discrepancy values ``\\hat{D}_k``.
- `component_usage::Vector{T}`: per-component contribution magnitudes.
"""
struct ACDCResult{T<:Real}
    K::Int
    component_discrepancies::Vector{T}
    component_usage::Vector{T}

    function ACDCResult(K::Int, discs::Vector{T}, usage::Vector{T}) where {T<:Real}
        length(discs) == K || throw(ArgumentError("must have K discrepancy values"))
        length(usage) == K || throw(ArgumentError("must have K usage values"))
        return new{T}(K, discs, usage)
    end
end

"""
    stochastic_drivers(model, data; n_samples=1, kwargs...) -> StochasticDriverResult

Recover the stochastic drivers ``\\varepsilon_{n,k}`` for a fitted `model` by
inverting its generative process. Returns a [`StochasticDriverResult`](@ref).

This is a generic function; methods are defined per model type. The method for
HiddenMarkovModels.jl `AbstractHMM`s lives in `hmm.jl`. The fallback below errors
for unsupported models.
"""
function stochastic_drivers(model, data; kwargs...)
    throw(
        ArgumentError(
            "ACDC has no `stochastic_drivers` method for a model of type " *
            "$(typeof(model)). Define a `stochastic_drivers` method to add support.",
        ),
    )
end

"""
    component_discrepancies(model, data, discrepancy; n_samples=1, rng, kwargs...) -> ACDCResult
    component_discrepancies(result::StochasticDriverResult, discrepancy; rng) -> ACDCResult

Compute per-component discrepancies from ``U(0,1)`` for a fitted `model`.

Recovers the stochastic drivers via [`stochastic_drivers`](@ref) and scores each
component's driver pool with `discrepancy`. The `rng` drives both the driver
recovery and any Monte Carlo discrepancy estimates; seed it for reproducible
results. Extra keyword arguments are forwarded to `stochastic_drivers`. The
second form scores an already-recovered [`StochasticDriverResult`](@ref).

A component whose driver pool is empty (it was never assigned a sample, which
happens when the model has more components than the data supports) scores
`Inf`: there is no evidence it is well specified, and the discrepancy measures
are undefined on empty pools.
"""
function component_discrepancies(
    model,
    data,
    discrepancy::ComponentDiscrepancy;
    n_samples::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...,
)
    result = stochastic_drivers(model, data; n_samples=n_samples, rng=rng, kwargs...)
    return component_discrepancies(result, discrepancy; rng=rng)
end

function component_discrepancies(
    result::StochasticDriverResult,
    discrepancy::ComponentDiscrepancy;
    rng::AbstractRNG=Random.default_rng(),
)
    usage = result.usage
    K = length(usage)
    T = eltype(usage)

    discs = Vector{T}(undef, K)
    for k in 1:K
        pool = result.ε_pools[k]
        #= Empty pool: the component was never sampled, so score it maximally
           discrepant rather than let the measures divide by zero or throw. =#
        discs[k] = if isempty(pool)
            T(Inf)
        else
            T(compute_discrepancy(discrepancy, pool; rng=rng))
        end
    end

    return ACDCResult(K, discs, usage)
end

"""
    KLDiscrepancy{T<:Real} <: ComponentDiscrepancy

KL divergence ``D_{\\text{KL}}(P \\| U([0,1]^D))`` estimated via k-nearest-neighbor
density estimation. Returns 0 for perfectly uniform drivers, positive otherwise.

To avoid boundary bias on the bounded hypercube, samples are mapped to
``\\mathbb{R}^D`` with the probit transform and compared against ``\\mathcal{N}(0, I)``.

# Fields
- `k_neighbors::Int`: number of neighbors for kNN density estimation (default 5).
"""
struct KLDiscrepancy{T<:Real} <: ComponentDiscrepancy
    k_neighbors::Int

    function KLDiscrepancy{T}(; k_neighbors::Int=5) where {T<:Real}
        k_neighbors > 0 || throw(ArgumentError("k_neighbors must be positive"))
        return new{T}(k_neighbors)
    end
end

KLDiscrepancy(; kwargs...) = KLDiscrepancy{Float64}(; kwargs...)

"""
    KSDiscrepancy{T<:Real} <: ComponentDiscrepancy

Kolmogorov-Smirnov statistic for uniformity on ``[0,1]^D``. For `D=1` it is the
standard KS statistic against ``F(x)=x``; for `D>1` it is the maximum KS
statistic across marginals.

!!! note
    The multivariate version tests marginal uniformity but not independence; use
    [`KLDiscrepancy`](@ref) or [`MMDDiscrepancy`](@ref) for a joint test.
"""
struct KSDiscrepancy{T<:Real} <: ComponentDiscrepancy
    KSDiscrepancy{T}() where {T<:Real} = new{T}()
end

KSDiscrepancy() = KSDiscrepancy{Float64}()

"""
    WassersteinDiscrepancy{T<:Real} <: ComponentDiscrepancy

Wasserstein-`p` distance between the empirical distribution and ``U([0,1]^D)``.
Closed-form (sorted samples vs uniform quantiles) for `D=1`; sliced Wasserstein
(average over `n_projections` random 1D projections) for `D>1`.

# Fields
- `p::Int`: order of the Wasserstein distance (default 2).
- `n_projections::Int`: number of random projections for the sliced estimate
  when `D > 1` (default 50).
"""
struct WassersteinDiscrepancy{T<:Real} <: ComponentDiscrepancy
    p::Int
    n_projections::Int

    function WassersteinDiscrepancy{T}(; p::Int=2, n_projections::Int=50) where {T<:Real}
        p > 0 || throw(ArgumentError("p must be positive"))
        n_projections > 0 || throw(ArgumentError("n_projections must be positive"))
        return new{T}(p, n_projections)
    end
end

WassersteinDiscrepancy(; kwargs...) = WassersteinDiscrepancy{Float64}(; kwargs...)

"""
    SquaredErrorDiscrepancy{T<:Real} <: ComponentDiscrepancy

Moment-based discrepancy from ``U([0,1]^D)``: squared deviation of marginal means
from `0.5`, marginal variances from `1/12`, and cross-covariances from `0`. Fast
but less sensitive than KL or Wasserstein.
"""
struct SquaredErrorDiscrepancy{T<:Real} <: ComponentDiscrepancy
    SquaredErrorDiscrepancy{T}() where {T<:Real} = new{T}()
end

SquaredErrorDiscrepancy() = SquaredErrorDiscrepancy{Float64}()

"""
    MMDDiscrepancy{T<:Real} <: ComponentDiscrepancy

Unbiased Maximum Mean Discrepancy with a Gaussian (RBF) kernel between the
empirical distribution and ``U([0,1]^D)``. Detects cross-dimensional dependence.
For `N > block_size`, averages the unbiased estimate over near-equal blocks of
at most `block_size` samples each (no tail is dropped) to stay ``O(N)``. The
result is clamped at 0, since the unbiased U-statistic can dip slightly
negative.

# Fields
- `sigma::T`: kernel bandwidth (default 0.5).
- `block_size::Int`: max samples per block (default 5000).
"""
struct MMDDiscrepancy{T<:Real} <: ComponentDiscrepancy
    sigma::T
    block_size::Int

    function MMDDiscrepancy{T}(; sigma::T=T(0.5), block_size::Int=5000) where {T<:Real}
        sigma > 0 || throw(ArgumentError("sigma must be positive"))
        block_size > 1 || throw(ArgumentError("block_size must be > 1"))
        return new{T}(sigma, block_size)
    end
end

MMDDiscrepancy(; kwargs...) = MMDDiscrepancy{Float64}(; kwargs...)

"""
    compute_discrepancy(d::ComponentDiscrepancy, samples::AbstractMatrix; rng) -> Real

Divergence between the empirical distribution of `samples` (a `D × N` matrix of
drivers in ``[0,1]``) and ``U([0,1]^D)``. Non-negative. Samples of any `Real`
eltype are accepted and computed at the discrepancy's precision `T`.

The `rng` keyword (default `Random.default_rng()`) feeds the Monte Carlo
estimates in [`WassersteinDiscrepancy`](@ref) (sliced projections for `D > 1`)
and [`MMDDiscrepancy`](@ref) (uniform reference sample); the other measures are
deterministic and ignore it.
"""
function compute_discrepancy end

function compute_discrepancy(
    d::KLDiscrepancy{T},
    samples::AbstractMatrix{<:Real};
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real}
    D, N = size(samples)
    k = d.k_neighbors

    if N < k + 1
        @warn "Too few samples ($N) for k-NN KL estimation (k=$k)"
        return T(Inf)
    end

    # Probit transform to N(0,1) space; clamp to avoid ±Inf at the boundaries.
    ϵ = eps(T)
    samples_clamped = clamp.(T.(samples), ϵ, one(T) - ϵ)
    samples_normal = quantile.(Normal(zero(T), one(T)), samples_clamped)

    # E[log p_data] via k-NN in R^D.
    mean_log_p = _mean_log_pdf_knn_multivariate(samples_normal, k)

    # E[log q_ref] for q_ref = N(0, I): log N(x;0,I) = -D/2 log(2π) - ½‖x‖².
    log_2pi = log(T(2) * T(π))
    mean_sq_norm = mean(sum(samples_normal .^ 2; dims=1))
    mean_log_q = -T(0.5) * (D * log_2pi + mean_sq_norm)

    return max(zero(T), mean_log_p - mean_log_q)
end

function compute_discrepancy(
    d::KSDiscrepancy{T},
    samples::AbstractMatrix{<:Real};
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real}
    D, N = size(samples)

    max_ks = zero(T)
    for dim in 1:D
        x = sort!(T.(view(samples, dim, :)))
        ecdf_vals = collect(T, 1:N) ./ N
        ref_cdf_vals = clamp.(x, zero(T), one(T))
        ks_stat = maximum(
            max.(
                abs.(ecdf_vals .- ref_cdf_vals),
                abs.((ecdf_vals .- one(T) / N) .- ref_cdf_vals),
            ),
        )
        max_ks = max(max_ks, ks_stat)
    end
    return max_ks
end

function compute_discrepancy(
    d::SquaredErrorDiscrepancy{T},
    samples::AbstractMatrix{<:Real};
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real}
    D, N = size(samples)
    total_err = zero(T)

    # Marginal moments: mean should be 0.5 and variance 1/12.
    for dim in 1:D
        x = view(samples, dim, :)
        total_err += (mean(x) - T(0.5))^2
        total_err += (var(x) - T(1 / 12))^2
    end

    # Cross-covariances should be 0.
    for i in 1:D
        for j in (i + 1):D
            total_err += cov(view(samples, i, :), view(samples, j, :))^2
        end
    end

    return T(total_err / D)
end

function compute_discrepancy(
    d::WassersteinDiscrepancy{T},
    samples::AbstractMatrix{<:Real};
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real}
    D, N = size(samples)

    if D == 1
        x_sorted = sort!(T.(vec(samples)))
        quantiles = [(i - T(0.5)) / N for i in 1:N]
        w_dist = zero(T)
        for i in 1:N
            w_dist += abs(x_sorted[i] - quantiles[i])^d.p
        end
        return (w_dist / N)^(one(T) / d.p)
    else
        # Sliced Wasserstein: average over random 1D projections.
        S = T.(samples)
        n_projections = d.n_projections
        total_w = zero(T)
        for _ in 1:n_projections
            direction = randn(rng, T, D)
            direction ./= norm(direction)

            x_sorted = sort!(vec(direction' * S))
            ref_samples = sort!(vec(direction' * rand(rng, T, D, N)))

            w_dist = zero(T)
            for i in 1:N
                w_dist += abs(x_sorted[i] - ref_samples[i])^d.p
            end
            total_w += (w_dist / N)^(one(T) / d.p)
        end
        return total_w / n_projections
    end
end

function compute_discrepancy(
    d::MMDDiscrepancy{T},
    samples::AbstractMatrix{<:Real};
    rng::AbstractRNG=Random.default_rng(),
) where {T<:Real}
    D, N = size(samples)
    S = T.(samples)

    if N <= d.block_size
        reference_uniform = rand(rng, T, D, N)
        return max(zero(T), _compute_mmd_quadratic_unbiased(S, reference_uniform, d.sigma))
    end

    #= Block strategy for large N: partition ALL samples into near-equal blocks
       of at most `block_size` (no tail remainder is dropped) and average the
       per-block unbiased estimates. The `div(N, 2)` cap keeps every block at
       ≥ 2 samples, which the U-statistic needs. =#
    n_blocks = min(cld(N, d.block_size), div(N, 2))
    total_mmd = zero(T)
    for b in 1:n_blocks
        start_idx = div((b - 1) * N, n_blocks) + 1
        end_idx = div(b * N, n_blocks)
        block_samples = view(S, :, start_idx:end_idx)
        reference = rand(rng, T, D, end_idx - start_idx + 1)
        total_mmd += _compute_mmd_quadratic_unbiased(block_samples, reference, d.sigma)
    end
    return max(zero(T), total_mmd / n_blocks)
end

"""
    acdc_loss(result::ACDCResult, ρ::Real) -> Real

ACDC loss for cutoff ``\\rho``: ``R^\\rho(K) = \\sum_k \\max(0, \\hat{D}_k - \\rho)``.
"""
function acdc_loss(result::ACDCResult{T}, ρ::Real) where {T<:Real}
    return sum(max(zero(T), d - T(ρ)) for d in result.component_discrepancies)
end

"""
    acdc_select(results::AbstractVector{<:ACDCResult}, ρ::Real) -> Int

Select the component count `K` with minimum ACDC loss at cutoff ``\\rho``,
breaking ties toward smaller `K`. `results` must be ordered by `K`.
"""
function acdc_select(results::AbstractVector{<:ACDCResult}, ρ::Real)
    isempty(results) && throw(ArgumentError("results must be non-empty"))
    losses = [acdc_loss(r, ρ) for r in results]
    #= `results` is ordered by K, so return the earliest entry whose loss ties
       the minimum (`≈` absorbs float noise in the hinge sums). =#
    best = argmin(losses)
    for i in 1:(best - 1)
        losses[i] ≈ losses[best] && return results[i].K
    end
    return results[best].K
end

"""
    get_critical_rho_values(results::AbstractVector{<:ACDCResult}) -> Vector

Sorted unique ``\\rho`` values at which the ACDC loss changes slope, i.e. the
component discrepancy values across all `K`.
"""
function get_critical_rho_values(results::AbstractVector{<:ACDCResult})
    isempty(results) && return Float64[]
    T = eltype(first(results).component_discrepancies)
    all_discs = T[]
    for r in results
        append!(all_discs, r.component_discrepancies)
    end
    return unique(sort(all_discs))
end

#= Estimate E_P[log p(x)] via the Kozachenko-Leonenko k-NN estimator with KDTree
   acceleration. =#
function _mean_log_pdf_knn_multivariate(samples::AbstractMatrix{T}, k::Int) where {T<:Real}
    D, N = size(samples)

    tree = KDTree(samples)
    # k+1 neighbors because the nearest is the point itself.
    _, dists = knn(tree, samples, k + 1, true)

    log_c_D = (T(D) / 2) * log(T(π)) - loggamma(T(D) / 2 + 1)
    bias_correction = digamma(T(k)) - digamma(T(N))

    sum_log_p = zero(T)
    for i in 1:N
        rho_k = max(dists[i][k + 1], eps(T))
        sum_log_p += bias_correction - log_c_D - D * log(rho_k)
    end
    return sum_log_p / N
end

#= Unbiased U-statistic MMD² between D × N matrices X and Y with a Gaussian RBF
   kernel k(x,y) = exp(-‖x-y‖² / (2σ²)). O(N²). =#
function _compute_mmd_quadratic_unbiased(
    X::AbstractMatrix{T}, Y::AbstractMatrix{T}, sigma::T
) where {T<:Real}
    D, N = size(X)
    gamma = one(T) / (T(2) * sigma^2)

    sum_xx = zero(T)
    sum_yy = zero(T)
    sum_xy = zero(T)

    for i in 1:N
        # Intra-group sums, excluding the diagonal (unbiased estimator).
        for j in (i + 1):N
            dist_sq_xx = zero(T)
            dist_sq_yy = zero(T)
            for d in 1:D
                diff_x = X[d, i] - X[d, j]
                diff_y = Y[d, i] - Y[d, j]
                dist_sq_xx += diff_x^2
                dist_sq_yy += diff_y^2
            end
            sum_xx += exp(-gamma * dist_sq_xx)
            sum_yy += exp(-gamma * dist_sq_yy)
        end
        # Cross-group sum over all pairs.
        for j in 1:N
            dist_sq_xy = zero(T)
            for d in 1:D
                diff_xy = X[d, i] - Y[d, j]
                dist_sq_xy += diff_xy^2
            end
            sum_xy += exp(-gamma * dist_sq_xy)
        end
    end

    # 2/(N(N-1)) restores the symmetric lower triangle skipped above.
    norm_intra = T(2) / (N * (N - 1))
    norm_inter = T(2) / (N * N)
    return (norm_intra * sum_xx) + (norm_intra * sum_yy) - (norm_inter * sum_xy)
end
