# DDM Emissions

Drift diffusion model (DDM) emissions for two-alternative forced choice data. Pairing them with a hidden Markov model gives the DDM-HMM of Senne et al. (2026): the hidden states are decision-making regimes, and each state emits a `(choice, rt)` trial from its own DDM. For a runnable end-to-end walkthrough (simulation, Baum-Welch, ACDC), see the [DDM-HMM tutorial](examples/ddm.md).

The Wiener first-passage-time density and sampler come from [SequentialSamplingModels.jl](https://github.com/itsdfish/SequentialSamplingModels.jl), a weak dependency. The emission types always construct, but `logdensityof`, `rand`, and `fit!` need the extension loaded:

```julia
using EmissionModels
using SequentialSamplingModels   # activates the extension
```

## Observations and controls

An observation is a `(choice, rt)` pair (a plain tuple or the `(; choice, rt)` `NamedTuple` returned by `rand`), with `choice ∈ {1, 2}` for the upper/lower boundary (matching SequentialSamplingModels.jl) and `rt` in seconds. Each trial also carries a scalar control that sets its drift. Single-trial calls take the control as the last positional argument — `logdensityof(d, obs, control)` and `rand(rng, d, control)` — while `fit!` takes the whole per-trial sequence through the `control_seq` keyword:

```julia
logdensityof(d, (1, 0.6), +1)                        # one trial, control = +1
fit!(d, obs_seq, weight_seq; control_seq=controls)   # controls[i] belongs to obs_seq[i]
```

Both models are stimulus-coded: the boundaries mark stimulus/choice identity (e.g. right vs. left), not correct vs. error, so correctness is read off afterward from the trial condition. The starting point ``z`` is therefore a side bias rather than an accuracy bias. The [HSSM stimulus-coding tutorial](https://lnccbrown.github.io/HSSM/tutorials/tutorial_stim_coding/) walks through how this changes parameter interpretation relative to accuracy coding.

## Provided DDM types

### `StimulusCodedDDM(; ν, α, z, τ)`

The drift is the gain ``ν`` multiplied by a trial-specific stimulus code:

``v_{\mathrm{trial}} = s_{\mathrm{trial}} \, ν, \qquad s_{\mathrm{trial}} \in \{-1, +1\}``

The control is the stimulus code ``s`` (`+1` when the upper-boundary stimulus is shown, `-1` for the lower, `0` for no-signal trials). ``ν > 0`` is a magnitude, so the stimulus code carries the sign.

```julia
d = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.3)
```

### `CoherenceDDM(; k, γ, α, z, τ)`

The drift is a (possibly nonlinear) function of signed stimulus coherence:

``v_{\mathrm{trial}} = k \, \operatorname{sign}(c) \, |c|^{γ}``

The control is the signed coherence ``c`` (sign = stimulus side, magnitude = stimulus strength); ``γ = 1`` recovers the classic linear drift/coherence relationship. Both ``k > 0`` and ``γ > 0``, so the coherence carries the sign.

```julia
d = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
```

## Fitting DDM emissions

`fit!` maximizes the weighted log-likelihood ``\sum_i w_i \log p(\mathrm{choice}_i, \mathrm{rt}_i \mid \mathrm{control}_i)`` in place. The density is the Navarro & Fuss (2009) series for the Wiener first passage time; positive parameters are optimized in ``\log`` space and bounded ones (``z``, ``τ``) through a logistic map, so LBFGS runs unconstrained with gradients from ForwardDiff. ``τ`` is capped at the smallest reaction time carrying positive weight. Each call warm-starts from the current parameters, so it refines an EM iterate rather than restarting.

```julia
# obs_seq: vector of (choice, rt) pairs; stimulus_codes: per-trial controls;
# weight_seq: per-trial weights (posterior state probabilities inside EM)
fit!(d, obs_seq, weight_seq; control_seq=stimulus_codes, max_iter=100, gtol=1e-8)
```

## The DDM-HMM

Both types subtype `HiddenMarkovModels.ControlledEmission`, so a vector of them is a valid `dists` for a `ControlledEmissionHMM`. A useful synthetic example is a two-state model pairing an *engaged* state (strong drift, wide boundaries: fast, stimulus-driven, mostly accurate choices) with a *lapsed* state (near-zero drift: slow, near-chance responding), joined by a sticky transition matrix so the subject stays in a regime for many consecutive trials:

```julia
dists = [
    StimulusCodedDDM(; ν=2.5, α=1.2, z=0.5, τ=0.25),  # engaged
    StimulusCodedDDM(; ν=0.3, α=0.7, z=0.5, τ=0.20),  # lapsed
]
hmm = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.1 0.9], dists)
```

`baum_welch` estimates every parameter jointly: the initial state probabilities, the transition matrix, and the DDM parameters of each state. In the M-step each state's DDM is refit by weighted maximum likelihood (`fit!` above), with that state's posterior probabilities as the trial weights. The [DDM-HMM tutorial](examples/ddm.md) walks through simulation and fitting end to end.

## Model selection with ACDC

DDM emissions support [ACDC model selection](acdc.md). Driver recovery inverts each `(choice, rt)` trial through the Rosenblatt transform of its DDM: a randomized PIT on the boundary choice followed by the conditional reaction-time PIT, giving two drivers per trial that are uniform on ``[0,1]`` under a well-specified model.

```julia
acdc = component_discrepancies(hmm, obs_seq, KSDiscrepancy();
                               control_seq=stimulus_codes, seq_ends=[T])
acdc.component_discrepancies    # one score per state; small = well-specified
```

## API Reference

```@docs
EmissionModels.AbstractDDMEmission
StimulusCodedDDM
CoherenceDDM
StatsAPI.fit!(::EmissionModels.AbstractDDMEmission, ::AbstractVector, ::AbstractVector{<:Real})
```

## References

- Senne, R. A., Xia, H., Duebel, H. F., Do, Q., Kane, G., Fourie, J., Ramirez, S., Scott, B., & DePasquale, B. (2026). Diurnal rhythms of choice: a novel state-dependent drift diffusion model uncovers time-dependent changes in rat decision making. *bioRxiv* [2026.05.25.727672](https://doi.org/10.64898/2026.05.25.727672).
- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *Journal of Mathematical Psychology*, 53(4), 222-230.
