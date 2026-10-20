# Custom Emission Models

Any type implementing the interface below can be used as an emission in
`HiddenMarkovModels.jl`.

## Static emissions

An emission used in an `HMM` implements:

```julia
Random.rand(rng::AbstractRNG, d::MyEmission)
DensityInterface.DensityKind(::MyEmission) = DensityInterface.HasDensity()
DensityInterface.logdensityof(d::MyEmission, obs)
StatsAPI.fit!(d::MyEmission, obs_seq::AbstractVector, weight_seq::AbstractVector)
```

`fit!` performs the weighted maximum-likelihood update used in the M-step of
`baum_welch`.

```julia
using Random, DensityInterface, StatsAPI
using LinearAlgebra: dot

mutable struct MyBernoulli{T<:Real}
    p::T
end

DensityInterface.DensityKind(::MyBernoulli) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(b::MyBernoulli, obs::Integer)
    obs ∈ (0, 1) || return oftype(b.p, -Inf)
    return obs * log(b.p) + (one(obs) - obs) * log(one(b.p) - b.p)
end

Random.rand(rng::AbstractRNG, b::MyBernoulli) = rand(rng) < b.p ? 1 : 0

function StatsAPI.fit!(b::MyBernoulli, obs_seq, weight_seq)
    b.p = dot(weight_seq, obs_seq) / sum(weight_seq)
    return b
end
```

```julia
using HiddenMarkovModels

hmm = HMM([0.5, 0.5], [0.9 0.1; 0.2 0.8], [MyBernoulli(0.2), MyBernoulli(0.7)])
_, obs_seq = rand(hmm, 500)
hmm_est, loglik = baum_welch(hmm, obs_seq)
```

## Controlled emissions

An emission whose distribution depends on a per-timestep control (covariate)
is used in a `ControlledEmissionHMM`. It must subtype
`HiddenMarkovModels.ControlledEmission`, which supplies `DensityKind`, and
implement the positional methods:

```julia
DensityInterface.logdensityof(d::MyEmission, obs, control)
Random.rand(rng::AbstractRNG, d::MyEmission, control)
StatsAPI.fit!(d::MyEmission, obs_seq, control_seq, weight_seq)
```

`control` is a single timestep's control; `control_seq` is the vector of
controls aligned with `obs_seq`. A Gaussian with mean `β * control`:

```julia
using Random, DensityInterface, StatsAPI
using HiddenMarkovModels: ControlledEmission

mutable struct LinearGaussian{T<:Real} <: ControlledEmission
    β::T
    σ::T
end

function DensityInterface.logdensityof(d::LinearGaussian, obs::Real, control::Real)
    z = (obs - d.β * control) / d.σ
    return -(z^2 + log(2π)) / 2 - log(d.σ)
end

function Random.rand(rng::AbstractRNG, d::LinearGaussian, control::Real)
    return d.β * control + d.σ * randn(rng)
end

function StatsAPI.fit!(d::LinearGaussian, obs_seq, control_seq, weight_seq)
    sxy = sum(w * u * y for (w, u, y) in zip(weight_seq, control_seq, obs_seq))
    sxx = sum(w * u^2 for (w, u) in zip(weight_seq, control_seq))
    d.β = sxy / sxx
    ssr = sum(w * (y - d.β * u)^2 for (w, u, y) in zip(weight_seq, control_seq, obs_seq))
    d.σ = sqrt(ssr / sum(weight_seq))
    return d
end
```

```julia
using HiddenMarkovModels

hmm = ControlledEmissionHMM(
    [0.5, 0.5], [0.95 0.05; 0.05 0.95], [LinearGaussian(1.0, 0.5), LinearGaussian(-2.0, 0.5)]
)
control_seq = randn(500)
_, obs_seq = rand(hmm, control_seq)
hmm_est, loglik = baum_welch(hmm, obs_seq, control_seq)
```

The GLM and DDM emissions in this package follow the same pattern.
