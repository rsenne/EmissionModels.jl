"""
    AbstractDDMEmission <: HiddenMarkovModels.ControlledEmission

Abstract type for drift-diffusion-model (DDM) emission distributions whose
drift rate depends on a scalar, trial-specific control (a stimulus code or a
signed coherence).

Subtyping `ControlledEmission` lets a `Vector` of DDM emissions serve as the
`dists` of a `HiddenMarkovModels.ControlledEmissionHMM` (the DDM-HMM). Each
concrete type implements the `ControlledEmission` positional interface:

- `DensityInterface.logdensityof(d, obs, control)`: log density of one trial
- `Random.rand(rng, d, control)`: sample one trial as `(; choice, rt)`
- `StatsAPI.fit!(d, obs_seq, control_seq, weights)`: weighted in-place update

Observations are `(choice, rt)` pairs — any positionally indexable pair such
as a `Tuple` or the `(; choice, rt)` `NamedTuple` returned by `rand` — with
`choice ∈ {1, 2}` (1 = upper boundary, 2 = lower boundary, matching
SequentialSamplingModels.jl) and `rt` the reaction time in seconds.
`DensityKind` is inherited from `ControlledEmission`.

The Wiener first-passage-time density and sampler are provided by
SequentialSamplingModels.jl through a package extension: run
`using SequentialSamplingModels` to enable `logdensityof`, `rand`, and `fit!`.
"""
abstract type AbstractDDMEmission <: ControlledEmission end

#= The extension overloads these two hooks with SequentialSamplingModels'
   first-passage density and sampler; the fallbacks fire when it isn't loaded. =#
@noinline function _require_ssm()
    throw(
        ArgumentError(
            "DDM emission models require SequentialSamplingModels.jl: run " *
            "`using SequentialSamplingModels` to enable `logdensityof`, " *
            "`rand`, and `fit!`.",
        ),
    )
end
_ddm_logpdf(ν, α, z, τ, choice, rt) = _require_ssm()
_ddm_rand(rng, ν, α, z, τ) = _require_ssm()

"""
    StimulusCodedDDM{T<:Real} <: AbstractDDMEmission

Stimulus-coded drift diffusion model emission for two-alternative forced
choice data.

Under stimulus coding the two boundaries correspond to stimulus/choice
identity (e.g. right vs. left) rather than correct vs. error, and the drift
on each trial is the drift magnitude multiplied by a trial-specific stimulus
code:

`v_trial = s_trial · ν,  s_trial ∈ {-1, +1}`

The control passed to `logdensityof`/`rand`/`fit!` is the scalar stimulus
code `s` (`+1` when the upper-boundary stimulus is presented, `-1` for the
lower; `0` is valid for no-signal trials). Correctness is determined after
the fact from the trial condition.

# Fields
- `ν::T`: drift magnitude (ν > 0); the stimulus code carries the sign, so
  drift is always toward the boundary matching the stimulus
- `α::T`: boundary separation (α > 0)
- `z::T`: relative starting point, i.e. response bias toward the upper
  boundary (0 < z < 1; 0.5 is unbiased). Because the boundaries are
  stimulus-coded, `z` is a side bias, not an accuracy bias.
- `τ::T`: non-decision time in seconds (τ ≥ 0)

# Example
```julia
using EmissionModels, SequentialSamplingModels

d = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.3)
obs = rand(rng, d, -1)                # (; choice, rt) for a lower stimulus
logdensityof(d, obs, -1)
fit!(d, obs_seq, weights; control_seq=stimulus_codes)
```
"""
mutable struct StimulusCodedDDM{T<:Real} <: AbstractDDMEmission
    ν::T
    α::T
    z::T
    τ::T

    function StimulusCodedDDM{T}(ν::T, α::T, z::T, τ::T) where {T<:Real}
        ν > 0 || throw(ArgumentError("ν must be positive, got $ν"))
        α > 0 || throw(ArgumentError("α must be positive, got $α"))
        0 < z < 1 || throw(ArgumentError("z must be in (0,1), got $z"))
        τ >= 0 || throw(ArgumentError("τ must be non-negative, got $τ"))
        return new{T}(ν, α, z, τ)
    end
end

function StimulusCodedDDM(ν::Real, α::Real, z::Real, τ::Real)
    T = float(promote_type(typeof(ν), typeof(α), typeof(z), typeof(τ)))
    return StimulusCodedDDM{T}(T(ν), T(α), T(z), T(τ))
end
StimulusCodedDDM(; ν=1.0, α=1.0, z=0.5, τ=0.3) = StimulusCodedDDM(ν, α, z, τ)

"""
    CoherenceDDM{T<:Real} <: AbstractDDMEmission

Drift diffusion model emission whose drift is a (possibly nonlinear) function
of signed stimulus coherence:

`v_trial = k · sign(c) · |c|^γ`

where `c` is the trial's signed coherence (sign = stimulus side, magnitude =
stimulus strength), `k` is the drift gain, and `γ` controls the nonlinearity
(`γ = 1` recovers the classic linear drift-coherence relationship). The
control passed to `logdensityof`/`rand`/`fit!` is the scalar signed coherence
`c`; a no-signal trial (`c = 0`) has zero drift. As with
[`StimulusCodedDDM`](@ref), the boundaries are stimulus-coded.

# Fields
- `k::T`: drift gain (k > 0); the signed coherence carries the sign
- `γ::T`: coherence exponent (γ > 0)
- `α::T`: boundary separation (α > 0)
- `z::T`: relative starting point / side bias (0 < z < 1)
- `τ::T`: non-decision time in seconds (τ ≥ 0)

# Example
```julia
using EmissionModels, SequentialSamplingModels

d = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
obs = rand(rng, d, 0.256)             # (; choice, rt) at coherence 0.256
logdensityof(d, obs, 0.256)
fit!(d, obs_seq, weights; control_seq=coherences)
```
"""
mutable struct CoherenceDDM{T<:Real} <: AbstractDDMEmission
    k::T
    γ::T
    α::T
    z::T
    τ::T

    function CoherenceDDM{T}(k::T, γ::T, α::T, z::T, τ::T) where {T<:Real}
        k > 0 || throw(ArgumentError("k must be positive, got $k"))
        γ > 0 || throw(ArgumentError("γ must be positive, got $γ"))
        α > 0 || throw(ArgumentError("α must be positive, got $α"))
        0 < z < 1 || throw(ArgumentError("z must be in (0,1), got $z"))
        τ >= 0 || throw(ArgumentError("τ must be non-negative, got $τ"))
        return new{T}(k, γ, α, z, τ)
    end
end

function CoherenceDDM(k::Real, γ::Real, α::Real, z::Real, τ::Real)
    T = float(promote_type(typeof(k), typeof(γ), typeof(α), typeof(z), typeof(τ)))
    return CoherenceDDM{T}(T(k), T(γ), T(α), T(z), T(τ))
end
CoherenceDDM(; k=1.0, γ=1.0, α=1.0, z=0.5, τ=0.3) = CoherenceDDM(k, γ, α, z, τ)

# odd in c, so the sign carries the stimulus side and 0 stays 0
_signedpow(c::Real, γ::Real) = sign(c) * abs(c)^γ

_drift(d::StimulusCodedDDM, s::Real) = d.ν * s
_drift(d::CoherenceDDM, c::Real) = d.k * _signedpow(c, d.γ)

function DensityInterface.logdensityof(d::AbstractDDMEmission, obs, control::Real)
    return _ddm_logpdf(_drift(d, control), d.α, d.z, d.τ, obs[1], obs[2])
end

function Random.rand(rng::AbstractRNG, d::AbstractDDMEmission, control::Real)
    return _ddm_rand(rng, _drift(d, control), d.α, d.z, d.τ)
end

#= Base owns `rand(rng, S, dims::Integer...)`, which is ambiguous with the
   `Real` method above for integer controls (the common ±1 stimulus codes).
   This more specific method breaks the tie. =#
function Random.rand(rng::AbstractRNG, d::AbstractDDMEmission, control::Integer)
    return _ddm_rand(rng, _drift(d, control), d.α, d.z, d.τ)
end

#= The M-step maximizes the weighted log-likelihood with LBFGS over
   unconstrained transformed parameters, warm-started from the current values:

     ν, k, γ, α = exp(θ)             drift gain, exponent, boundary > 0
     z = logistic(θ_z)               starting point in (0,1)
     τ = rt_min · logistic(θ_τ)      non-decision time in (0, min observed rt)

   Gradients come from a single ForwardDiff pass through the first-passage
   density. `_pack`/`_unpack` map between a model's fields and θ, and
   `_drift_at` evaluates the per-trial drift from the unpacked parameters. =#

function _pack(d::StimulusCodedDDM{T}, rt_min::T) where {T<:Real}
    ϵ = sqrt(eps(T))
    return T[log(d.ν), log(d.α), logit(d.z), logit(clamp(d.τ / rt_min, ϵ, 1 - ϵ))]
end

function _unpack(::StimulusCodedDDM, θ::AbstractVector{<:Real}, rt_min::Real)
    # logistic saturates to exactly 0/1 for large |θ|; keep z interior
    ϵ = eps(float(typeof(rt_min)))
    return (
        ν=exp(θ[1]),
        α=exp(θ[2]),
        z=clamp(logistic(θ[3]), ϵ, 1 - ϵ),
        τ=rt_min * logistic(θ[4]),
    )
end

_drift_at(::StimulusCodedDDM, pars, s::Real) = pars.ν * s

function _setparams!(d::StimulusCodedDDM, pars)
    d.ν = pars.ν
    d.α = pars.α
    d.z = pars.z
    d.τ = pars.τ
    return d
end

function _pack(d::CoherenceDDM{T}, rt_min::T) where {T<:Real}
    ϵ = sqrt(eps(T))
    return T[log(d.k), log(d.γ), log(d.α), logit(d.z), logit(clamp(d.τ / rt_min, ϵ, 1 - ϵ))]
end

function _unpack(::CoherenceDDM, θ::AbstractVector{<:Real}, rt_min::Real)
    ϵ = eps(float(typeof(rt_min)))
    return (
        k=exp(θ[1]),
        γ=exp(θ[2]),
        α=exp(θ[3]),
        z=clamp(logistic(θ[4]), ϵ, 1 - ϵ),
        τ=rt_min * logistic(θ[5]),
    )
end

_drift_at(::CoherenceDDM, pars, c::Real) = pars.k * _signedpow(c, pars.γ)

function _setparams!(d::CoherenceDDM, pars)
    d.k = pars.k
    d.γ = pars.γ
    d.α = pars.α
    d.z = pars.z
    d.τ = pars.τ
    return d
end

#= Weighted negative log-likelihood in θ. Generic in the eltype so ForwardDiff
   can push duals through; a zero-density trial short-circuits to +Inf, which
   the line search treats as any other rejected step. =#
struct _DDMNLL{
    D<:AbstractDDMEmission,
    O<:AbstractVector,
    W<:AbstractVector{<:Real},
    C<:AbstractVector{<:Real},
    T<:Real,
}
    d::D
    obs_seq::O
    weight_seq::W
    control_seq::C
    rt_min::T
end

function (o::_DDMNLL)(θ::AbstractVector{T}) where {T<:Real}
    pars = _unpack(o.d, θ, o.rt_min)
    nll = zero(T)
    for i in eachindex(o.obs_seq, o.weight_seq, o.control_seq)
        w = o.weight_seq[i]
        w > 0 || continue
        obs = o.obs_seq[i]
        ν = _drift_at(o.d, pars, o.control_seq[i])
        lp = _ddm_logpdf(ν, pars.α, pars.z, pars.τ, obs[1], obs[2])
        isfinite(lp) || return T(Inf)
        nll -= T(w) * lp
    end
    return nll
end

#= Fused value+gradient in the `fg!(F, G, θ)` form `Optim.only_fg!` expects:
   one dual pass yields both, mirroring the fgh! pattern in `glms/glm.jl`. =#
struct _DDMFG{N<:_DDMNLL,C<:ForwardDiff.GradientConfig}
    nll::N
    cfg::C
end

function (o::_DDMFG)(F, G, θ::AbstractVector{T}) where {T<:Real}
    if G !== nothing
        dr = DiffResults.MutableDiffResult(zero(T), (G,))
        ForwardDiff.gradient!(dr, o.nll, θ, o.cfg)
        return F === nothing ? nothing : DiffResults.value(dr)
    end
    return F === nothing ? nothing : o.nll(θ)
end

"""
    fit!(d::AbstractDDMEmission, obs_seq, weight_seq;
         control_seq, max_iter=100, gtol=1e-8)

Fit the DDM emission parameters to weighted `(choice, rt)` observations by
maximizing `Σᵢ wᵢ · log p(choiceᵢ, rtᵢ | controlᵢ)` in place, warm-started
from the current parameters (so repeated EM calls refine, not restart).

The optimizer is Optim's LBFGS over unconstrained transformed parameters,
with the gradient supplied by ForwardDiff through the first-passage density.
The non-decision time is constrained to `(0, rt_min)` where `rt_min` is the
smallest reaction time with positive weight. If no positive weight is present
the parameters are left unchanged.

# Arguments
- `d`: [`StimulusCodedDDM`](@ref) or [`CoherenceDDM`](@ref), updated in place
- `obs_seq`: sequence of `(choice, rt)` pairs with `choice ∈ {1, 2}`
- `weight_seq`: per-observation weights (e.g. HMM posterior state
  probabilities)
- `control_seq`: per-observation scalar controls (stimulus codes or signed
  coherences)
- `max_iter`: LBFGS iteration cap per call
- `gtol`: gradient-norm convergence tolerance
"""
function StatsAPI.fit!(
    d::AbstractDDMEmission,
    obs_seq::AbstractVector,
    weight_seq::AbstractVector{<:Real};
    control_seq::AbstractVector{<:Real},
    max_iter::Int=100,
    gtol::Real=1e-8,
)
    n = length(obs_seq)
    length(weight_seq) == n || throw(
        DimensionMismatch("weight_seq length $(length(weight_seq)) ≠ obs_seq length $n")
    )
    length(control_seq) == n || throw(
        DimensionMismatch("control_seq length $(length(control_seq)) ≠ obs_seq length $n"),
    )

    T = typeof(d.α)

    #= Only trials with positive weight constrain the fit, so they alone are
       validated and bound the non-decision time from above. =#
    total_weight = zero(T)
    rt_min = T(Inf)
    for i in 1:n
        w = weight_seq[i]
        w > 0 || continue
        total_weight += T(w)
        obs = obs_seq[i]
        choice = obs[1]
        rt = obs[2]
        (choice == 1 || choice == 2) || throw(
            ArgumentError(
                "observations must be (choice, rt) pairs with choice ∈ {1, 2}, " *
                "got choice = $choice",
            ),
        )
        rt > 0 || throw(ArgumentError("reaction times must be positive, got $rt"))
        rt_min = min(rt_min, T(rt))
    end
    total_weight > 0 || return d

    θ = _pack(d, rt_min)
    nll = _DDMNLL(d, obs_seq, weight_seq, control_seq, rt_min)
    fg = _DDMFG(nll, ForwardDiff.GradientConfig(nll, θ))
    od = OnceDifferentiable(only_fg!(fg), θ)
    result = optimize(od, θ, LBFGS(), Optim.Options(; iterations=max_iter, g_abstol=gtol))
    #= An infinite minimum means no parameter with finite likelihood was found
       (e.g. a degenerate warm start); keep the current parameters instead. =#
    if isfinite(Optim.minimum(result))
        _setparams!(d, _unpack(d, Optim.minimizer(result), rt_min))
    end
    return d
end

#= HiddenMarkovModels.ControlledEmission positional fit signature
   (`fit!(dist, obs_seq, control_seq, weights)`), delegating to the keyword
   method above so the actual M-step has a single source of truth. =#
function StatsAPI.fit!(
    d::AbstractDDMEmission,
    obs_seq::AbstractVector,
    control_seq::AbstractVector{<:Real},
    weights::AbstractVector{<:Real};
    kwargs...,
)
    return fit!(d, obs_seq, weights; control_seq=control_seq, kwargs...)
end
