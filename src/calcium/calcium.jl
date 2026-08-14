#= Calcium-imaging emission model of Keeley, Zoltowski, Charles & Pillow (2026),
   "Improved inference of latent neural states from calcium imaging data"
   (eLife 109405, reviewed preprint).

   Fluorescence is modelled directly, with the spike count marginalized out, so
   an HMM can be fit to raw traces without a separate deconvolution step. For
   neuron `n` at time `t`,

       s[n,t] ~ Poisson(λ[n, state])
       y[n,t] = Σ_l α[l,n] y[n,t-l] + c[n] s[n,t] + ε,   ε ~ N(0, σ²[n])

   and the emission density marginalizes the unobserved count numerically,

       p(y_t | y_{t-1..t-L}, k) = Σ_{s=0}^{R} N(y_t; μ + c s, σ²) Poisson(s; λ_k).

   The AR history enters through the HiddenMarkovModels control interface: each
   timestep's control is the vector of lagged fluorescence values, so a
   `ControlledEmissionHMM` over these emissions is an AR-HMM. `lagged_controls`
   builds that control sequence from an observation sequence.

   The AR order is free (`size(α, 1)`). AR(2) resolves the finite rise time of
   the transient that AR(1) collapses to a single step, and the paper finds it
   recovers latent structure better on biophysically simulated data. =#

"""
    CalciumParams{T<:Real}

Per-neuron calcium observation parameters: AR coefficients, fluorescence influx
per spike, and noise variance. These are the parameters the paper ties *across*
HMM states — only the firing rates λ are state-dependent — so a set of
[`CalciumEmission`](@ref)s normally shares one `CalciumParams` object.

# Fields
- `α`: AR coefficient matrix, `order × nneurons` (`α[l, n]` multiplies `y[n, t-l]`)
- `c`: fluorescence influx per spike, one per neuron (positive)
- `σ2`: observation noise variance, one per neuron (positive)
- `b`: additive baseline, one per neuron (zero by default; see below)
- `fit_c`: whether `fit!` updates `c` (see the identifiability note below)
- `fit_b`: whether `fit!` updates `b`
- `gaussian`: use the large-rate Gaussian marginal instead of summing `0:R`
- `nshare`: number of emissions sharing this object; the pooled M-step fires
  once every `nshare` calls to `fit!`

# Baseline
The paper's model has no intercept: with `s ≥ 0` and `c > 0` the fluorescence
can only be driven upwards, and the resting level is pinned at zero. That is
right for single-cell ΔF/F but wrong for anything with a standing baseline, so
`b` adds one. It is zero and held fixed by default, which reproduces the paper.

`b` is tied across states while `λ` is not, which is what identifies it: the
part of the offset common to every state is `b`, the state-specific part is
`cλ`. With a single state the two are separated only by the count noise, so
fitting `b` with `nstates = 1` is ill-posed.

# Large rates
Summing `s = 0:R` costs `O(R)` per neuron per timestep and silently truncates
once `λ` approaches `R`. For aggregate signals (fiber photometry, bulk ROIs)
where `λ` is tens to hundreds, `gaussian=true` replaces the Poisson by
`N(λ, λ)`, which makes the marginal analytic and `O(1)`:

    y | state ~ N(b + Σ α y_lag + cλ, c²λ + σ²).

Sampling always uses the exact Poisson generative model regardless.

# Tied fitting
`HiddenMarkovModels` fits emissions one state at a time, so a tied M-step cannot
be expressed per state. Instead each `fit!` call pushes its weighted sufficient
statistics into this object and the `nshare`-th call solves the pooled weighted
least-squares problem for `α`, `c`, `σ2` and resets the accumulators. This is
correct under the serial per-state fitting loop `ControlledEmissionHMM` uses; a
tied set must always be fit as a complete set (as Baum-Welch does).

# Identifiability
`c` and `λ` are identifiable only through the count noise: `c·s` has mean `cλ`
and variance `c²λ`, so the two are separated by the mean/variance ratio alone.
That direction is poorly conditioned when `σ2` is large. `fit_c=false` pins `c`
to its supplied value (e.g. from a deconvolution pass or a known indicator
amplitude) and fits the rest, which is markedly more stable.
"""
mutable struct CalciumParams{T<:Real}
    α::Matrix{T}
    c::Vector{T}
    σ2::Vector{T}
    b::Vector{T}
    fit_c::Bool
    fit_b::Bool
    gaussian::Bool
    nshare::Int

    #= Pooled M-step state. `XtX`/`Xty`/`yy`/`wsum` accumulate the weighted
       normal equations for the regressors x = [y_{t-1} … y_{t-L}, s_t, 1].
       The spike entry is latent and so enters through its posterior moments
       E[s], E[s²]. `ncalls` counts fit! calls since the last solve; `_free`
       holds the regressor indices the current flags leave free. =#
    ncalls::Int
    XtX::Array{T,3}
    Xty::Matrix{T}
    yy::Vector{T}
    wsum::Vector{T}
    _A::Matrix{T}
    _θ::Vector{T}
    _rhs::Vector{T}
    _free::Vector{Int}

    function CalciumParams{T}(
        α::Matrix{T},
        c::Vector{T},
        σ2::Vector{T},
        b::Vector{T},
        fit_c::Bool,
        fit_b::Bool,
        gaussian::Bool,
        nshare::Int,
    ) where {T<:Real}
        ord, nn = size(α)
        ord > 0 || throw(ArgumentError("α must have at least one row (AR order ≥ 1)"))
        nn > 0 || throw(ArgumentError("α must have at least one column (one per neuron)"))
        length(c) == nn || throw(DimensionMismatch("c length $(length(c)) ≠ neurons $(nn)"))
        length(σ2) == nn ||
            throw(DimensionMismatch("σ2 length $(length(σ2)) ≠ neurons $(nn)"))
        length(b) == nn || throw(DimensionMismatch("b length $(length(b)) ≠ neurons $(nn)"))
        all(>(0), c) || throw(ArgumentError("c must be positive, got $c"))
        all(>(0), σ2) || throw(ArgumentError("σ2 must be positive, got $σ2"))
        nshare > 0 || throw(ArgumentError("nshare must be positive, got $nshare"))
        d = ord + 2
        return new{T}(
            α,
            c,
            σ2,
            b,
            fit_c,
            fit_b,
            gaussian,
            nshare,
            0,
            zeros(T, d, d, nn),
            zeros(T, d, nn),
            zeros(T, nn),
            zeros(T, nn),
            zeros(T, d, d),
            zeros(T, d),
            zeros(T, d),
            zeros(Int, d),
        )
    end
end

function CalciumParams(
    α::AbstractMatrix,
    c::AbstractVector,
    σ2::AbstractVector;
    b=nothing,
    fit_c::Bool=false,
    fit_b::Bool=false,
    gaussian::Bool=false,
    nshare=1,
)
    T = float(promote_type(eltype(α), eltype(c), eltype(σ2)))
    bvec = b === nothing ? zeros(T, (length(c),)) : Vector{T}(b)
    return CalciumParams{T}(
        Matrix{T}(α), Vector{T}(c), Vector{T}(σ2), bvec, fit_c, fit_b, gaussian, Int(nshare)
    )
end

"""
    ar_order(p::CalciumParams)

AR order of the calcium dynamics (1 for `y_t = α y_{t-1} + …`, 2 for AR(2)).
"""
ar_order(p::CalciumParams) = size(p.α, 1)

"""
    nneurons(p::CalciumParams)

Number of simultaneously recorded neurons (the observation dimension).
"""
nneurons(p::CalciumParams) = length(p.c)

"""
    CalciumEmission{T<:Real}

Calcium fluorescence emission for one HMM state: a vector of per-neuron firing
rates plus the (usually shared) [`CalciumParams`](@ref) describing the calcium
dynamics. Neurons are conditionally independent given the state.

Observations are length-`nneurons` fluorescence vectors; controls are the
lagged fluorescence values laid out as `control[(l-1)*N + n] = y[n, t-l]`, which
[`lagged_controls`](@ref) produces.

# Fields
- `λ`: per-neuron Poisson firing rate in this state (positive, length `N`)
- `params`: calcium dynamics parameters, shared across states when tied
- `R`: spike-count truncation for the marginalization, summing `s = 0:R`

# Example
```julia
params = CalciumParams([0.9 0.85], [1.0, 1.0], [0.05, 0.05]; nshare=2)
dists = [CalciumEmission([0.2, 1.5], params), CalciumEmission([1.8, 0.3], params)]
hmm = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.05 0.95], dists)
```
"""
mutable struct CalciumEmission{T<:Real} <: ControlledEmission
    λ::Vector{T}
    params::CalciumParams{T}
    R::Int
    _logfact::Vector{T}   # loggamma(s+1) for s = 0:R, cached since R is fixed at construction

    function CalciumEmission{T}(
        λ::Vector{T}, params::CalciumParams{T}, R::Int
    ) where {T<:Real}
        length(λ) == nneurons(params) ||
            throw(DimensionMismatch("λ length $(length(λ)) ≠ neurons $(nneurons(params))"))
        all(>(0), λ) || throw(ArgumentError("λ must be positive, got $λ"))
        R > 0 || throw(ArgumentError("R must be positive, got $R"))
        return new{T}(λ, params, R, [loggamma(T(s + 1)) for s in 0:R])
    end
end

function CalciumEmission(
    λ::AbstractVector, params::CalciumParams{T}; R::Integer=10
) where {T}
    return CalciumEmission{T}(Vector{T}(λ), params, Int(R))
end

ar_order(em::CalciumEmission) = ar_order(em.params)
nneurons(em::CalciumEmission) = nneurons(em.params)

"""
    calcium_emissions(λ0, α, c, σ2; b=nothing, R=10, tied=true, fit_c=false,
                      fit_b=false, gaussian=false)

Build one [`CalciumEmission`](@ref) per HMM state from an `nneurons × nstates`
matrix of initial firing rates `λ0`.

With `tied=true` (the paper's model) every state shares a single
[`CalciumParams`](@ref), so `α`, `c`, `σ2` and `b` are pooled across states
during fitting and only `λ` is state-specific. With `tied=false` each state gets
its own independent copy of the calcium dynamics — which also unties `b`, and
so gives up the cross-state contrast that identifies it.
"""
function calcium_emissions(
    λ0::AbstractMatrix,
    α::AbstractMatrix,
    c::AbstractVector,
    σ2::AbstractVector;
    b=nothing,
    R::Integer=10,
    tied::Bool=true,
    fit_c::Bool=false,
    fit_b::Bool=false,
    gaussian::Bool=false,
)
    K = size(λ0, 2)
    K > 0 || throw(ArgumentError("λ0 must have at least one column (one per state)"))
    size(λ0, 1) == length(c) || throw(
        DimensionMismatch("λ0 has $(size(λ0, 1)) rows but c has $(length(c)) neurons")
    )
    if tied
        params = CalciumParams(
            α, c, σ2; b=b, fit_c=fit_c, fit_b=fit_b, gaussian=gaussian, nshare=K
        )
        return [CalciumEmission(view(λ0, :, k), params; R=R) for k in 1:K]
    end
    return [
        CalciumEmission(
            view(λ0, :, k),
            CalciumParams(
                copy(α),
                copy(c),
                copy(σ2);
                b=(b === nothing ? nothing : copy(b)),
                fit_c=fit_c,
                fit_b=fit_b,
                gaussian=gaussian,
                nshare=1,
            );
            R=R,
        ) for k in 1:K
    ]
end

"""
    lagged_controls(obs_seq, order; seq_ends=nothing, pad=:zero)

Build the control sequence an AR calcium HMM needs: one vector per timestep
holding the `order` most recent fluorescence values, laid out as
`control[(l-1)*N + n] = y[n, t-l]`.

Pass `seq_ends` (the same vector `baum_welch` takes) when `obs_seq` concatenates
several recordings, so lags do not bleed across the boundaries.

# Padding the pre-recording history
The first `order` timesteps have no history, and `pad` decides what to assume
in its place.

- `:zero` (default) asserts the process was resting at zero. That matches the
  paper's baseline-free model, where the resting level *is* zero.
- `:first` holds it at the first observed sample.

The choice is not cosmetic once `b` is in play. With a baseline the resting
level is far from zero, so `:zero` presents the first sample as an enormous
jump the emission cannot produce — `s ≥ 0` and `c > 0` mean fluorescence can
only be driven upwards, so a first sample below the resting level has
essentially zero density. On a trace centred to mean zero that single timestep
can cost tens of thousands of nats and drag `σ2` up by orders of magnitude as
EM strains to accommodate it. Use `:first` whenever `fit_b` is on, or whenever
the trace does not start near zero.
"""
function lagged_controls(
    obs_seq::AbstractVector{<:AbstractVector},
    order::Integer;
    seq_ends=nothing,
    pad::Symbol=:zero,
)
    order > 0 || throw(ArgumentError("order must be positive, got $order"))
    pad === :zero ||
        pad === :first ||
        throw(ArgumentError("pad must be :zero or :first, got :$pad"))
    n_t = length(obs_seq)
    n_t > 0 || throw(ArgumentError("obs_seq must be non-empty"))
    N = length(first(obs_seq))
    T = float(eltype(first(obs_seq)))

    starts = Set{Int}(1)
    if seq_ends !== nothing
        for e in seq_ends
            e < n_t && push!(starts, e + 1)
        end
    end

    controls = [zeros(T, N * order) for _ in 1:n_t]
    start = 1
    for t in 1:n_t
        t in starts && (start = t)
        u = controls[t]
        for l in 1:order
            src = t - l
            #= Lags reaching before the recording get the padding value. `:zero`
               asserts the process was at rest at zero, which is only true
               without a baseline; `:first` holds it at the first sample. =#
            src < start && (src = pad === :zero ? 0 : start)
            src == 0 && break
            copyto!(view(u, ((l - 1) * N + 1):(l * N)), obs_seq[src])
        end
    end
    return controls
end

function DensityInterface.logdensityof(
    em::CalciumEmission, y::AbstractVector; control_seq::AbstractVector{<:Real}
)
    p = em.params
    N = nneurons(p)
    ord = ar_order(p)
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) ≠ neurons $N"))
    length(control_seq) == N * ord || throw(
        DimensionMismatch(
            "control_seq length $(length(control_seq)) ≠ neurons × order $(N * ord)"
        ),
    )

    T = float(
        promote_type(
            eltype(p.α),
            eltype(p.c),
            eltype(p.σ2),
            eltype(em.λ),
            eltype(control_seq),
            eltype(y),
        ),
    )
    total = zero(T)
    for n in 1:N
        μ = T(p.b[n])
        for l in 1:ord
            μ += T(p.α[l, n]) * T(control_seq[(l - 1) * N + n])
        end
        σ2 = T(p.σ2[n])
        c = T(p.c[n])
        λ = T(em.λ[n])
        resid = T(y[n]) - μ

        if p.gaussian
            #= Poisson → N(λ, λ), so the convolution with the observation noise
               is another Gaussian and the sum collapses to a closed form. =#
            v = c * c * λ + σ2
            d = resid - c * λ
            total += -log(2 * T(π) * v) / 2 - d * d / (2 * v)
        else
            #= Marginalize the spike count numerically, folding the running
               log-sum-exp so no length-R buffer is needed. The Gaussian
               normalizer and the Poisson e^{-λ} are constant in s and pulled
               out below. =#
            logλ = log(λ)
            acc = T(-Inf)
            for s in 0:(em.R)
                d = resid - c * s
                acc = logaddexp(acc, -d * d / (2 * σ2) + s * logλ - em._logfact[s + 1])
            end
            total += acc - log(2 * T(π) * σ2) / 2 - λ
        end
    end
    return total
end

"""
    rand!(rng, em::CalciumEmission, out; control_seq)

Sample one fluorescence vector into `out`, drawing an untruncated Poisson spike
count per neuron. Both the truncation `R` and `gaussian` are approximations the
likelihood makes, never the sampler, so choose `R` well above the plausible
per-bin count.
"""
function Random.rand!(
    rng::AbstractRNG,
    em::CalciumEmission{T},
    out::AbstractVector;
    control_seq::AbstractVector{<:Real},
) where {T<:Real}
    p = em.params
    N = nneurons(p)
    ord = ar_order(p)
    length(out) == N || throw(DimensionMismatch("out length $(length(out)) ≠ neurons $N"))
    for n in 1:N
        μ = p.b[n]
        for l in 1:ord
            μ += p.α[l, n] * T(control_seq[(l - 1) * N + n])
        end
        s = rand(rng, Poisson(em.λ[n]))
        out[n] = μ + p.c[n] * s + sqrt(p.σ2[n]) * randn(rng, T)
    end
    return out
end

function Random.rand(
    rng::AbstractRNG, em::CalciumEmission{T}; control_seq::AbstractVector{<:Real}
) where {T<:Real}
    out = Vector{T}(undef, nneurons(em))
    return rand!(rng, em, out; control_seq=control_seq)
end

#= ControlledEmission positional interface: `ControlledEmissionHMM` drives each
   emission through these, binding one timestep's control. They delegate to the
   keyword methods so the math has a single source of truth. =#
function DensityInterface.logdensityof(
    em::CalciumEmission, obs, control::AbstractVector{<:Real}
)
    return logdensityof(em, obs; control_seq=control)
end

function Random.rand(rng::AbstractRNG, em::CalciumEmission, control::AbstractVector{<:Real})
    return rand(rng, em; control_seq=control)
end

#= Row accessors so `fit!` takes both the vector-of-vectors layout that
   `ControlledEmissionHMM` hands it and a plain `n×p` matrix. =#
_seq_at(v::AbstractVector{<:AbstractVector}, t::Integer) = v[t]
_seq_at(m::AbstractMatrix, t::Integer) = view(m, t, :)
_nsteps(v::AbstractVector{<:AbstractVector}) = length(v)
_nsteps(m::AbstractMatrix) = size(m, 1)

"""
    fit!(em::CalciumEmission, obs_seq, weight_seq; control_seq)

One M-step for a calcium emission. The state-specific rates `λ` are updated
from the posterior expected spike counts; the calcium dynamics `α`, `c`, `σ2`
and `b` are accumulated into the shared [`CalciumParams`](@ref) and solved once
the whole tied set has been visited (see the tied-fitting note there).

The posterior over the latent count is evaluated the same way the likelihood
marginalizes it — on the `0:R` grid, or in closed form under `gaussian` — which
makes both the `λ` update and the pooled weighted least-squares solve closed
form. The whole M-step is a single pass, no inner optimizer.
"""
function StatsAPI.fit!(
    em::CalciumEmission{T},
    obs_seq::Union{AbstractVector,AbstractMatrix},
    weight_seq::AbstractVector{<:Real};
    control_seq,
    kwargs..., # closed-form M-step: solver kwargs accepted, ignored
) where {T<:Real}
    p = em.params
    N = nneurons(p)
    ord = ar_order(p)
    R = em.R
    n_t = _nsteps(obs_seq)
    length(weight_seq) == n_t || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ obs_seq length $n_t"),
    )

    #= Accumulates Σ w E[s] in the Poisson case and Σ w E[s²] in the Gaussian
       case, which is what each λ update needs. =#
    λ_num = zeros(T, (N,))
    wtot = zero(T)
    si = ord + 1   # regressor index of the latent spike count
    bi = ord + 2   # regressor index of the constant

    for t in 1:n_t
        w = T(weight_seq[t])
        iszero(w) && continue
        y = _seq_at(obs_seq, t)
        u = _seq_at(control_seq, t)
        wtot += w

        for n in 1:N
            μ = p.b[n]
            for l in 1:ord
                μ += p.α[l, n] * T(u[(l - 1) * N + n])
            end
            σ2 = p.σ2[n]
            c = p.c[n]
            λ = em.λ[n]
            resid = T(y[n]) - μ

            Es = zero(T)
            Es2 = zero(T)
            if p.gaussian
                #= Gaussian prior N(λ, λ) conjugate to the Gaussian
                   observation: the posterior over s is Gaussian in closed
                   form. Clamp the mean at zero — the approximation has no
                   nonnegativity constraint, but a negative rate does. =#
                v = λ * σ2 / (σ2 + c * c * λ)
                Es = max(v * (one(T) + c * resid / σ2), zero(T))
                Es2 = Es * Es + v
                λ_num[n] += w * Es2
            else
                #= Posterior over the latent count, p(s | y_t, y_{t-1..}, state)
                   on the 0:R grid. Two passes (max, then normalize) keep the
                   exponentials in range without a buffer. =#
                logλ = log(λ)
                m = T(-Inf)
                for s in 0:R
                    d = resid - c * s
                    m = max(m, -d * d / (2 * σ2) + s * logλ - em._logfact[s + 1])
                end
                Z = zero(T)
                for s in 0:R
                    d = resid - c * s
                    q = exp(-d * d / (2 * σ2) + s * logλ - em._logfact[s + 1] - m)
                    Z += q
                    Es += q * s
                    Es2 += q * s * s
                end
                Es /= Z
                Es2 /= Z
                λ_num[n] += w * Es
            end

            #= Pooled normal equations for x = [y_{t-1} … y_{t-L}, s, 1]. The
               latent spike entry contributes its posterior moments, so E[xxᵀ]
               is exact rather than a plug-in at E[s]. =#
            yn = T(y[n])
            for a in 1:ord
                ua = T(u[(a - 1) * N + n])
                wua = w * ua
                for bb in 1:ord
                    p.XtX[a, bb, n] += wua * T(u[(bb - 1) * N + n])
                end
                p.XtX[a, si, n] += wua * Es
                p.XtX[si, a, n] += wua * Es
                p.XtX[a, bi, n] += wua
                p.XtX[bi, a, n] += wua
                p.Xty[a, n] += wua * yn
            end
            p.XtX[si, si, n] += w * Es2
            p.XtX[si, bi, n] += w * Es
            p.XtX[bi, si, n] += w * Es
            p.XtX[bi, bi, n] += w
            p.Xty[si, n] += w * yn * Es
            p.Xty[bi, n] += w * yn
            p.yy[n] += w * yn * yn
            p.wsum[n] += w
        end
    end

    if wtot > 0
        for n in 1:N
            #= Poisson: λ = E[s]. Gaussian: maximizing E[log N(s; λ, λ)] gives
               λ² + λ = E[s²], the same relation a Poisson's second moment
               satisfies exactly. =#
            λ̂ = if p.gaussian
                (sqrt(one(T) + 4 * λ_num[n] / wtot) - one(T)) / 2
            else
                λ_num[n] / wtot
            end
            em.λ[n] = max(λ̂, _variance_floor(T))
        end
    end

    #= Tied M-step: solve once the whole sharing set has contributed. A state
       with zero responsibility still counts as a visit, so the schedule stays
       aligned with the HMM's per-state fitting loop. =#
    p.ncalls += 1
    if p.ncalls >= p.nshare
        _solve_calcium!(p)
        p.ncalls = 0
    end
    return em
end

# ControlledEmission positional fit signature; delegates to the keyword method.
function StatsAPI.fit!(
    em::CalciumEmission,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    kwargs...,
)
    return fit!(em, obs_seq, weight_seq; control_seq=control_seq, kwargs...)
end

#= Pooled weighted least squares for the calcium dynamics, one neuron at a
   time, followed by the residual variance implied by the same accumulators:

       σ² = (Σ w y² − 2 θᵀ Σ w y E[x] + θᵀ Σ w E[xxᵀ] θ) / Σ w.

   The AR coefficients are always free; the influx and the baseline are free
   only under `fit_c`/`fit_b`. A held-fixed column contributes its known value
   to the right-hand side instead of being solved for, so the four flag
   combinations share one solve over the free index set. =#
function _solve_calcium!(p::CalciumParams{T}) where {T<:Real}
    ord = ar_order(p)
    d = ord + 2
    si, bi = ord + 1, ord + 2
    θ = p._θ
    rhs = p._rhs
    free = p._free

    nfree = ord
    for l in 1:ord
        free[l] = l
    end
    p.fit_c && (free[nfree += 1] = si)
    p.fit_b && (free[nfree += 1] = bi)

    for n in 1:nneurons(p)
        w = p.wsum[n]
        if w <= 0
            continue
        end

        # Current values, so the fixed columns can be folded into the residual.
        for l in 1:ord
            θ[l] = p.α[l, n]
        end
        θ[si] = p.c[n]
        θ[bi] = p.b[n]

        A = view(p._A, 1:nfree, 1:nfree)
        for j in 1:nfree, i in 1:nfree
            A[i, j] = p.XtX[free[i], free[j], n]
        end
        for i in 1:nfree
            fi = free[i]
            acc = p.Xty[fi, n]
            p.fit_c || (acc -= θ[si] * p.XtX[fi, si, n])
            p.fit_b || (acc -= θ[bi] * p.XtX[fi, bi, n])
            rhs[i] = acc
        end

        F = cholesky!(Symmetric(A, :L); check=false)
        # A singular normal-equation block means this neuron's history is
        # rank-deficient under the current weights: leave it untouched.
        issuccess(F) || continue
        ldiv!(F, view(rhs, 1:nfree))
        for i in 1:nfree
            θ[free[i]] = rhs[i]
        end
        # A non-positive influx would invert the sign of the spike response.
        θ[si] = max(θ[si], _variance_floor(T))

        quad = zero(T)
        cross = zero(T)
        for a in 1:d
            cross += θ[a] * p.Xty[a, n]
            for bb in 1:d
                quad += θ[a] * p.XtX[a, bb, n] * θ[bb]
            end
        end

        for l in 1:ord
            p.α[l, n] = θ[l]
        end
        p.fit_c && (p.c[n] = θ[si])
        p.fit_b && (p.b[n] = θ[bi])
        p.σ2[n] = max((p.yy[n] - 2 * cross + quad) / w, _variance_floor(T))
    end

    fill!(p.XtX, zero(T))
    fill!(p.Xty, zero(T))
    fill!(p.yy, zero(T))
    fill!(p.wsum, zero(T))
    return p
end

#= Normalize a user-supplied AR initialization to an `ord × N` matrix. A scalar
   or a length-`ord` vector applies the same dynamics to every neuron. =#
function _ar_matrix(α, ord::Int, N::Int, ::Type{T}) where {T}
    α === nothing && return nothing
    M = zeros(T, (ord, N))
    if α isa Real
        ord == 1 || throw(
            ArgumentError(
                "scalar α needs order 1, got order $ord; pass a length-$ord vector"
            ),
        )
        fill!(M, T(α))
    elseif α isa AbstractMatrix
        size(α) == (ord, N) ||
            throw(DimensionMismatch("α must be $(ord)×$(N), got $(size(α))"))
        copyto!(M, α)
    else
        length(α) == ord || throw(DimensionMismatch("α length $(length(α)) ≠ order $(ord)"))
        for n in 1:N, l in 1:ord
            M[l, n] = T(α[l])
        end
    end
    return M
end

# One AR residual y[n,t] − Σ α[l,n] y[n,t−l], recomputed rather than buffered.
function _residual(obs_seq, α, n::Int, t::Int, ord::Int, ::Type{T}) where {T}
    r = T(obs_seq[t][n])
    for l in 1:ord
        r -= α[l, n] * T(obs_seq[t - l][n])
    end
    return r
end

#= Segment boundaries as (start, stop) pairs, so the initializer never forms a
   lag pair that straddles two recordings. =#
function _segments(n_t::Integer, seq_ends)
    seq_ends === nothing && return [(1, Int(n_t))]
    segs = Tuple{Int,Int}[]
    start = 1
    for e in seq_ends
        push!(segs, (start, Int(e)))
        start = Int(e) + 1
    end
    start <= n_t && push!(segs, (start, Int(n_t)))
    return segs
end

"""
    init_calcium(obs_seq, nstates, order; α=nothing, c=nothing, R=10, tied=true,
                 fit_c=false, fit_b=false, gaussian=false,
                 rng=Random.default_rng(), spread=0.5, seq_ends=nothing)

Initialize a set of [`CalciumEmission`](@ref)s from raw fluorescence, returning
one emission per state.

The AR coefficients come from Yule-Walker equations on the centered trace: the
spike input is white around its mean, so the trace autocovariance identifies
`α` directly. The residual `r = y - Σ α y_lag` then has mean `cλ` and
within-state variance `c²λ + σ²`, which pins down the rate and the noise.
Per-state rates are that global rate jittered by `exp(spread * randn(rng))`,
since identical states give Baum-Welch no asymmetry to break.

This matters more than it looks: the marginal likelihood has a long shallow
ridge where the AR coefficient and the firing rate trade off against each other,
and EM started from a flat guess crawls along it for hundreds of iterations.
Started from these moments it converges in tens.

# When to pass `α` yourself
Yule-Walker assumes the drive is white, which fails when the states are both
strongly separated and long-dwelling — the trace autocorrelation is then
dominated by the state process rather than by the indicator decay, and `α` comes
back near 1. That start is a trap: an almost-integrated AR can mimic the slow
state drift, and EM will sit there. Aggregate signals are the usual offender,
since their rate differences dwarf the count noise.

`α` overrides the estimate with a scalar, a length-`order` vector, or an
`order × nneurons` matrix. For an AR(1) indicator with decay time `τ` sampled
every `Δt`, use `exp(-Δt/τ)` — which is normally known.

The baseline starts at zero unless the residual mean is too small to carry the
rate (a mean-centred trace), because the moments cannot otherwise say how much
of that mean is `b` and how much is `cλ`. In that case the within-state variance
sets `λ` and the leftover mean becomes `b`.
"""
function init_calcium(
    obs_seq::AbstractVector{<:AbstractVector{S}},
    nstates::Integer,
    order::Integer;
    α=nothing,
    c=nothing,
    R::Integer=10,
    tied::Bool=true,
    fit_c::Bool=false,
    fit_b::Bool=false,
    gaussian::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
    spread::Real=0.5,
    seq_ends=nothing,
) where {S<:Real}
    nstates > 0 || throw(ArgumentError("nstates must be positive, got $nstates"))
    order > 0 || throw(ArgumentError("order must be positive, got $order"))
    #= `S` fixes the working precision at compile time, and the dimensions are
       narrowed to `Int` up front. Array shapes are passed as tuples rather
       than as varargs (`zeros(T, (ord, N))`, not `zeros(T, ord, N)`) so the
       result's dimensionality stays inferable when `T` is not a compile-time
       constant — the vararg method leaves it unknown. =#
    T = float(S)
    ord = Int(order)
    K = Int(nstates)
    n_t = length(obs_seq)
    N = length(first(obs_seq))
    cvec = c === nothing ? ones(T, (N,)) : Vector{T}(c)
    length(cvec) == N || throw(DimensionMismatch("c length $(length(cvec)) ≠ neurons $N"))

    segs = _segments(n_t, seq_ends)
    α_given = _ar_matrix(α, ord, N, T)
    α_init = α_given === nothing ? zeros(T, (ord, N)) : α_given
    λ0 = zeros(T, (N,))
    σ2 = zeros(T, (N,))
    b0 = zeros(T, (N,))
    Γ = zeros(T, (ord, ord))
    rhs = zeros(T, (ord,))

    for n in 1:N
        #= Per-segment means: centering with one global mean would bias the
           autocovariance toward a spurious near-unit-root whenever segments
           sit at different baseline levels (e.g. separate recordings), since
           the cross-segment mean offset masquerades as slow autocorrelation. =#
        seg_means = zeros(T, length(segs))
        for (si, (s, e)) in enumerate(segs)
            acc = zero(T)
            for t in s:e
                acc += T(obs_seq[t][n])
            end
            seg_means[si] = acc / (e - s + 1)
        end

        # Autocovariance γ(0..ord) of the centered trace, within segments.
        γ = zeros(T, (ord + 1,))
        for k in 0:ord
            acc = zero(T)
            cnt = 0
            for (si, (s, e)) in enumerate(segs)
                ȳ = seg_means[si]
                for t in (s + k):e
                    acc += (T(obs_seq[t][n]) - ȳ) * (T(obs_seq[t - k][n]) - ȳ)
                    cnt += 1
                end
            end
            γ[k + 1] = cnt > 0 ? acc / cnt : zero(T)
        end

        for i in 1:ord, j in 1:ord
            Γ[i, j] = γ[abs(i - j) + 1]
        end
        for i in 1:ord
            rhs[i] = γ[i + 1]
        end
        if α_given === nothing
            F = cholesky!(Symmetric(Γ, :L); check=false)
            if issuccess(F)
                ldiv!(F, rhs)
                for l in 1:ord
                    α_init[l, n] = rhs[l]
                end
            else
                # Degenerate trace (e.g. constant): fall back to a slow decay.
                α_init[1, n] = T(0.9)
            end
        end

        # Residual mean, then its variance and lag-1 autocovariance.
        mr = zero(T)
        cnt = 0
        for (s, e) in segs, t in (s + ord):e
            mr += _residual(obs_seq, α_init, n, t, ord, T)
            cnt += 1
        end
        cnt > 0 && (mr /= cnt)

        vr = zero(T)
        cr = zero(T)
        nlag = 0
        for (s, e) in segs
            rprev = zero(T)
            has_prev = false
            for t in (s + ord):e
                r = _residual(obs_seq, α_init, n, t, ord, T) - mr
                vr += r * r
                if has_prev
                    cr += r * rprev
                    nlag += 1
                end
                rprev = r
                has_prev = true
            end
        end
        vr = cnt > 1 ? vr / (cnt - 1) : one(T)
        cr = nlag > 0 ? cr / nlag : zero(T)

        #= `r = b + cs + ε` is white within a state, so its lag-1
           autocovariance is entirely the slow drift of the rate between
           states. Subtracting it leaves the within-state spread, `c²λ + σ²`,
           which is what the noise estimate needs: on well-separated states the
           raw variance is dominated by the between-state term and would hand
           EM a σ² orders of magnitude too large. =#
        within = max(vr - cr, T(0.05) * vr, _variance_floor(T))

        #= `mean(r) = b + cλ` and `within = c²λ + σ²`. With `b` fixed at zero
           the mean alone gives λ, and that split is the more accurate one.
           But a mean at or below zero is impossible under `b = 0` (both `c` and
           `λ` are positive), which is exactly what a mean-centred trace looks
           like — and it would start EM at λ ≈ 0, a fixed point it cannot climb
           out of. So when the mean carries no usable scale, fall back to
           splitting the within-state spread and let whatever mean is left over
           be the baseline. =#
        if fit_b && mr <= T(0.05) * sqrt(within)
            λ0[n] = max(T(0.9) * within / (cvec[n] * cvec[n]), _variance_floor(T))
            σ2[n] = max(T(0.1) * within, _variance_floor(T))
            b0[n] = mr - cvec[n] * λ0[n]
        else
            λ0[n] = max(mr / cvec[n], _variance_floor(T))
            #= Keep a floor of a few percent of the spread: a noiseless fit here
               would make the E-step posterior a point mass and stall EM. =#
            σ2[n] = max(within - cvec[n] * mr, T(0.05) * within, _variance_floor(T))
        end
    end

    λ = Matrix{T}(undef, N, K)
    for k in 1:K, n in 1:N
        λ[n, k] = max(λ0[n] * exp(T(spread) * randn(rng, T)), _variance_floor(T))
    end
    return calcium_emissions(
        λ,
        α_init,
        cvec,
        σ2;
        b=b0,
        R=R,
        tied=tied,
        fit_c=fit_c,
        fit_b=fit_b,
        gaussian=gaussian,
    )
end

"""
    rand_calcium(rng, init, trans, dists, n_t)

Simulate `n_t` timesteps of a calcium AR-HMM, returning
`(; state_seq, obs_seq, control_seq, spike_seq)`.

A dedicated sampler is needed because the control at time `t` is the *observed*
fluorescence at `t-1`, so it cannot be supplied up front the way
`rand(rng, hmm, control_seq)` expects — the trace has to be rolled out.
"""
function rand_calcium(
    rng::AbstractRNG,
    init::AbstractVector,
    trans::AbstractMatrix,
    dists::AbstractVector{<:CalciumEmission{T}},
    n_t::Integer,
) where {T<:Real}
    K = length(dists)
    length(init) == K || throw(DimensionMismatch("init length $(length(init)) ≠ states $K"))
    size(trans) == (K, K) || throw(DimensionMismatch("trans must be $(K)×$(K)"))
    N = nneurons(first(dists))
    ord = ar_order(first(dists))
    all(d -> nneurons(d) == N, dists) ||
        throw(DimensionMismatch("all dists must share the same neuron count $N"))
    all(d -> ar_order(d) == ord, dists) ||
        throw(DimensionMismatch("all dists must share the same AR order $ord"))

    state_seq = Vector{Int}(undef, n_t)
    obs_seq = [Vector{T}(undef, N) for _ in 1:n_t]
    spike_seq = [Vector{Int}(undef, N) for _ in 1:n_t]
    control_seq = [zeros(T, N * ord) for _ in 1:n_t]

    state = _sample_index(rng, init)
    for t in 1:n_t
        state_seq[t] = state
        em = dists[state]
        p = em.params
        u = control_seq[t]
        for l in 1:ord
            t - l >= 1 && copyto!(view(u, ((l - 1) * N + 1):(l * N)), obs_seq[t - l])
        end
        for n in 1:N
            μ = p.b[n]
            for l in 1:ord
                μ += p.α[l, n] * u[(l - 1) * N + n]
            end
            s = rand(rng, Poisson(em.λ[n]))
            spike_seq[t][n] = s
            obs_seq[t][n] = μ + p.c[n] * s + sqrt(p.σ2[n]) * randn(rng, T)
        end
        t < n_t && (state = _sample_index(rng, view(trans, state, :)))
    end
    return (; state_seq, obs_seq, control_seq, spike_seq)
end

# Categorical draw over a probability vector, without a Distributions detour.
function _sample_index(rng::AbstractRNG, probs::AbstractVector)
    u = rand(rng) * sum(probs)
    acc = zero(eltype(probs))
    for i in eachindex(probs)
        acc += probs[i]
        acc >= u && return Int(i)
    end
    return Int(lastindex(probs))
end
