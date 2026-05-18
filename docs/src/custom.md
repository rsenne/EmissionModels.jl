# Custom Emission Models

To create a custom emission model that works with `HiddenMarkovModels.jl`, your type must implement the following methods.

## Required interface

### 1. Sample from the emission

```julia
Random.rand([rng::AbstractRNG,] dist::MyEmission)
Random.rand([rng::AbstractRNG,] dist::MyEmission, n::Integer)  # n samples
```

The first form returns a single sample. The second form (optional) samples `n` i.i.d. draws.

### 2. Declare the density interface

```julia
DensityInterface.DensityKind(::MyEmission) = DensityInterface.HasDensity()
```

Return `HasDensity()` if `logdensityof` is implemented; otherwise return `NotAdjointDensity()`.

### 3. Evaluate log-density / log-probability-mass

```julia
DensityInterface.logdensityof(dist::MyEmission, obs)
```

This is called internally during the forward-backward algorithm.

### 4. Parameter estimation

```julia
StatsAPI.fit!(dist::MyEmission, obs_seq::AbstractVector, weight_seq::AbstractVector)
```

Optionally accept keyword arguments:

```julia
StatsAPI.fit!(dist::MyEmission, obs_seq, weight_seq; max_iter=100, tol=1e-6)
```

## Optional: Conditional emissions (with control vectors)

If your model depends on a design vector, add a `control` keyword to `logdensityof` and `rand`:

```julia
DensityInterface.logdensityof(dist::MyEmission, obs; control=nothing)
Random.rand([rng,] dist::MyEmission; control=nothing)
```

## Example: Bernoulli emission

```julia
using Random, DensityInterface, StatsAPI

mutable struct MyBernoulli{T<:Real}
    p::T  # probability of success
end

DensityInterface.DensityKind(::MyBernoulli) = DensityInterface.HasDensity()

function DensityInterface.logdensityof(b::MyBernoulli, obs::Integer)
    obs ∈ (0, 1) || return oftype(b.p, -Inf)
    return obs * log(b.p) + (one(obs) - obs) * log(one(b.p) - b.p)
end

function Random.rand(rng::AbstractRNG, b::MyBernoulli)
    rand(rng) < b.p ? 1 : 0
end

# Fitting: MLE for Bernoulli is the sample mean
function StatsAPI.fit!(b::MyBernoulli, obs_seq, weight_seq)
    total_w = sum(weight_seq)
    b.p = dot(weight_seq, obs_seq) / total_w
    return b
end

# Use in an HMM:
emission = MyBernoulli(0.5)
```
