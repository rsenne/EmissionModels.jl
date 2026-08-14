# Calcium Imaging Emissions

Calcium fluorescence emissions for the AR-HMM of Keeley, Zoltowski, Charles & Pillow, "Improved inference of latent neural states from calcium imaging data". Fluorescence is modelled directly, with the spike count marginalized out, so an HMM can be fit to raw traces without a separate deconvolution step. For a runnable end-to-end walkthrough, see the [calcium imaging tutorial](examples/calcium.md).

## The model

For neuron ``n`` at time ``t``, a latent spike count drives an autoregressive fluorescence process:

``s_{n,t} \sim \mathrm{Poisson}(\lambda_n), \qquad y_{n,t} = \sum_{l=1}^p \alpha_{l,n} \, y_{n,t-l} + c_n \, s_{n,t} + \varepsilon_{n,t}, \quad \varepsilon_{n,t} \sim \mathcal{N}(0, \sigma_n^2)``

and the emission density marginalizes the unobserved count numerically:

``p(y_t \mid y_{t-1..t-p}, k) = \sum_{s=0}^{R} \mathcal{N}(y_t; \mu + c s, \sigma^2) \, \mathrm{Poisson}(s; \lambda_k)``

[`CalciumEmission`](@ref) holds one HMM state's per-neuron rates ``\lambda``; the AR coefficients ``\alpha``, spike influx ``c`` and noise variance ``\sigma^2`` live in a shared [`CalciumParams`](@ref), since the paper ties the calcium dynamics across states and lets only the firing rate vary.

```julia
params = CalciumParams([0.9 0.85], [1.0, 1.0], [0.05, 0.05]; nshare=2)
dists = [CalciumEmission([0.2, 1.5], params), CalciumEmission([1.8, 0.3], params)]
hmm = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.05 0.95], dists)
```

The AR term looks back at past observations, so the emission needs a control: the ``p`` most recent fluorescence values per neuron, laid out as `control[(l-1)*N + n] = y[n, t-l]`. [`lagged_controls`](@ref) builds this control sequence from an observation sequence.

## Building a set of emissions

[`calcium_emissions`](@ref) builds one [`CalciumEmission`](@ref) per state from a matrix of initial firing rates. With `tied=true` (the default, and the paper's model) every state shares one [`CalciumParams`](@ref); with `tied=false` each state gets an independent copy, which also unties the baseline `b`.

[`init_calcium`](@ref) estimates a starting point directly from a raw trace: the AR coefficients from Yule-Walker on the trace autocovariance, and the rates/noise from moment matching. This matters because the marginal likelihood has a long, shallow ridge between the AR coefficient and the firing rate — EM started from a flat guess can take hundreds of iterations to escape it, while a moment-based start converges in tens.

## Simulating

Because the control at time ``t`` is the *observed* fluorescence at ``t-1``, a calcium AR-HMM cannot be simulated from a pre-built control sequence — the trace has to be rolled out step by step. [`rand_calcium`](@ref) does this, returning the state, observation, control and spike sequences together.

## Fitting

`fit!` runs one closed-form M-step: the per-state rate `λ` is the posterior expected spike count, and the calcium dynamics (`α`, `c`, `σ2`, `b`) are pooled weighted least squares over the whole tied set of states, solved once every `nshare` calls. There is no inner optimizer.

Two flags control what part of the calcium dynamics gets refit:

- `fit_c`: whether the spike influx `c` updates. `c` and `λ` are identifiable only through the count noise (mean `cλ`, variance `c²λ`), which is poorly conditioned when `σ²` is large — leaving `fit_c=false` (the default) and supplying a known indicator amplitude is more stable.
- `fit_b`: whether the baseline `b` updates. `b` is tied across states while `λ` is not, which is what identifies it, so fitting `b` needs at least two states — with `nstates = 1` it is ill-posed.

## Padding and the baseline

The first `order` timesteps of a trace have no history, and `lagged_controls`'s `pad` keyword decides what to assume in their place: `:zero` asserts the process rested at zero (the paper's baseline-free model), `:first` holds it at the first observed sample. This choice stops being cosmetic once `fit_b=true` — with a nonzero resting level, `:zero` presents the first sample as a jump the emission cannot produce (`s ≥ 0` and `c > 0` mean fluorescence only moves upward), which can cost tens of thousands of nats and drag `σ²` up by orders of magnitude. Use `:first` whenever `fit_b` is on, or whenever the trace does not start near zero.

## Aggregate signals

Summing the spike count over `s = 0:R` costs `O(R)` per neuron per timestep and truncates once `λ` approaches `R`. For aggregate signals (fiber photometry, bulk ROIs) where `λ` is tens to hundreds, `gaussian=true` replaces the Poisson by its large-rate Gaussian approximation, which makes the marginal analytic and `O(1)`:

``y \mid \text{state} \sim \mathcal{N}(b + \textstyle\sum \alpha \, y_{\mathrm{lag}} + c\lambda, \ c^2\lambda + \sigma^2)``

Sampling always draws from the exact Poisson generative model regardless of `gaussian`; only the density is approximated.

## API Reference

```@docs
CalciumParams
CalciumEmission
calcium_emissions
init_calcium
lagged_controls
rand_calcium
EmissionModels.ar_order(::CalciumParams)
EmissionModels.nneurons(::CalciumParams)
StatsAPI.fit!(::CalciumEmission{T}, ::Union{AbstractMatrix,AbstractVector}, ::AbstractVector{<:Real}) where {T<:Real}
Random.rand!(::AbstractRNG, ::CalciumEmission{T}, ::AbstractVector) where {T<:Real}
```

## References

- Keeley, S., Zoltowski, D., Charles, A., & Pillow, J. (2026). Improved inference of latent neural states from calcium imaging data. *eLife* [109405](https://doi.org/10.7554/eLife.109405.1) (reviewed preprint).
