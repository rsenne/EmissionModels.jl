# # Calcium imaging emissions

#=
Here we build an AR-HMM over calcium fluorescence traces, following Keeley,
Zoltowski, Charles & Pillow's model for calcium imaging data: the emission
marginalizes an unobserved spike count out of an autoregressive fluorescence
model, so an HMM can be fit directly to raw traces without a separate
deconvolution step.
=#

using EmissionModels
using HiddenMarkovModels
using HiddenMarkovModels: ControlledEmissionHMM
using Random
using Statistics
using Test  #src

#-

rng = MersenneTwister(4);

# ## The emission model

#=
For neuron $n$ at time $t$, a latent spike count drives the fluorescence
through an AR($p$) process:

```math
s_{n,t} \sim \mathrm{Poisson}(\lambda_n), \qquad
y_{n,t} = \sum_{l=1}^p \alpha_{l,n} \, y_{n,t-l} + c_n \, s_{n,t} + \varepsilon_{n,t}
```

with $\varepsilon_{n,t} \sim \mathcal{N}(0, \sigma_n^2)$. [`CalciumEmission`](@ref)
represents one HMM state (a firing rate $\lambda$ per neuron); the AR
coefficients, spike influx and noise variance live in a shared
[`CalciumParams`](@ref), since the paper's model ties the calcium dynamics
across states and lets only the firing rate vary.
=#

params = CalciumParams([0.9 0.85], [1.0, 1.0], [0.05, 0.05])
em = CalciumEmission([0.2, 1.5], params)

#=
Because the AR term looks back at past observations, the emission needs the
lagged fluorescence as a *control*: a length-$p \times N$ vector holding the
$p$ most recent samples for each of the $N$ neurons. Sampling and evaluating
the density both take it as `control_seq`.
=#

control = zeros(2)   # no history yet
y = rand(rng, em; control_seq=control)
logdensityof(em, y; control_seq=control)

# ## Simulating an AR-HMM

#=
A vector of [`CalciumEmission`](@ref)s is a valid `dists` for a
`ControlledEmissionHMM`, so a calcium AR-HMM is built the same way as any
other controlled emission model. [`calcium_emissions`](@ref) builds one
emission per state from a matrix of initial rates, sharing one
[`CalciumParams`](@ref) across all of them (`tied=true`, the default).
=#

λ_true = [0.15 1.60; 1.70 0.20]     # neurons × states
dists = calcium_emissions(λ_true, [0.90 0.85], [1.0, 1.0], [0.02, 0.02])
hmm_true = ControlledEmissionHMM([0.5, 0.5], [0.97 0.03; 0.03 0.97], dists)

#=
The control at time $t$ is the *observed* fluorescence at $t-1$, so the trace
has to be rolled out step by step rather than simulated from a
pre-built control sequence. [`rand_calcium`](@ref) does this, returning the
state, observation, control and spike sequences together.
=#

n_t = 2000
sim = rand_calcium(rng, hmm_true.init, hmm_true.trans, dists, n_t);
sim.obs_seq[1:3]

# ## Fitting with Baum-Welch

#=
Given only the fluorescence trace, [`init_calcium`](@ref) finds a starting
point via Yule-Walker on the trace autocovariance (for the AR coefficients)
and moment matching (for the rates and noise), then
[`lagged_controls`](@ref) turns the trace into the control sequence Baum-Welch
needs.
=#

dists0 = init_calcium(sim.obs_seq, 2, 1; rng=MersenneTwister(1))
control_seq = lagged_controls(sim.obs_seq, 1)
hmm0 = ControlledEmissionHMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], dists0)
hmm_est, lls = baum_welch(hmm0, sim.obs_seq, control_seq; seq_ends=[n_t])
first(lls), last(lls)

#=
The tied AR coefficients and noise variance are recovered directly from the
fitted [`CalciumParams`](@ref) (states are only identifiable up to a
permutation, but the shared dynamics are not).
=#

hmm_est.dists[1].params.α, hmm_est.dists[1].params.σ2

#=
The per-state rates recover the truth up to that same state permutation.
=#

λ_est = hcat((d.λ for d in hmm_est.dists)...)
λ_est, λ_true

# ## Aggregate signals

#=
Summing the spike count over `0:R` costs $O(R)$ per neuron per timestep and
truncates once $\lambda$ approaches $R$. For aggregate signals such as fiber
photometry, where $\lambda$ is tens to hundreds, `gaussian=true` replaces the
Poisson by its large-rate Gaussian approximation $\mathrm{Poisson}(\lambda)
\approx \mathcal{N}(\lambda, \lambda)$, making the marginal analytic and
$O(1)$ instead:
=#

params_big = CalciumParams([0.9;;], [1.0], [4.0]; gaussian=true)
em_big = CalciumEmission([80.0], params_big)
logdensityof(em_big, [75.0]; control_seq=[70.0])

#=
Sampling always draws from the exact Poisson generative model, regardless of
`gaussian`; only the density approximates it.
=#

# ## Tests  #src

@test all(diff(lls) .>= -1e-6)  #src
@test last(lls) > first(lls)  #src
@test hmm_est.dists[1].params.α ≈ [0.90 0.85] rtol = 0.15  #src
@test hmm_est.dists[1].params.σ2 ≈ [0.02, 0.02] rtol = 1.0  #src
err(perm) = maximum(abs, λ_est[:, perm] .- λ_true)  #src
@test min(err([1, 2]), err([2, 1])) < 0.35  #src
@test isfinite(logdensityof(em_big, [75.0]; control_seq=[70.0]))  #src
