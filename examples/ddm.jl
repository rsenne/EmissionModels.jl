# # The DDM-HMM

#=
Here we build the state-dependent drift diffusion model of Senne et al.
(2026): a hidden Markov model whose states are decision-making regimes, each
emitting a `(choice, rt)` trial from its own drift diffusion model (DDM).

The Wiener first-passage density and sampler come from
SequentialSamplingModels.jl, a weak dependency: loading it activates the
extension that backs `logdensityof`, `rand`, and `fit!` for the DDM emissions.
=#

using EmissionModels
using SequentialSamplingModels
using HiddenMarkovModels
using Random
using Statistics
using Test  #src

#-

rng = MersenneTwister(215);

# ## A single stimulus-coded DDM

#=
[`StimulusCodedDDM`](@ref) is a DDM whose boundaries mark stimulus identity
(e.g. right vs. left) rather than correct vs. error. Each trial carries a
stimulus code $s \in \{-1, +1\}$ as its control, and the drift on that trial
is $s \cdot ν$. Let us simulate a session of trials with random stimulus
sides.
=#

truth = StimulusCodedDDM(; ν=2.0, α=1.2, z=0.55, τ=0.3)
stimulus_codes = [rand(rng, (-1.0, 1.0)) for _ in 1:1000]
obs_seq = [rand(rng, truth, s) for s in stimulus_codes]
first(obs_seq, 3)

#=
`fit!` recovers the parameters by weighted maximum likelihood (LBFGS with
ForwardDiff gradients, warm-started from the current values). The weights
are all ones here; inside EM they are posterior state probabilities.
=#

d = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.15)
fit!(d, obs_seq, ones(1000); control_seq=stimulus_codes)
(; d.ν, d.α, d.z, d.τ)

# ## Simulating a DDM-HMM

#=
As a synthetic example, consider a DDM-HMM with two regimes: an *engaged*
state, whose strong drift and wide boundaries produce fast, stimulus-driven,
mostly accurate choices, and a *lapsed* state, whose near-zero drift produces
slow, near-chance responding. The heavy diagonal of the transition matrix makes the regimes
sticky — runs of roughly $1/0.05 = 20$ engaged and $1/0.1 = 10$ lapsed trials
on average — which is the trial-history structure a trial-independent mixture
model cannot capture.

Both DDM types subtype `HiddenMarkovModels.ControlledEmission`, so a vector
of them is a valid `dists` for a `ControlledEmissionHMM`.
=#

dists = [
    StimulusCodedDDM(; ν=2.5, α=1.2, z=0.5, τ=0.25),  ## engaged
    StimulusCodedDDM(; ν=0.3, α=0.7, z=0.5, τ=0.20),  ## lapsed
]
hmm_true = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.1 0.9], dists)

T = 2000
control_seq = [rand(rng, (-1.0, 1.0)) for _ in 1:T]
sim = rand(rng, hmm_true, control_seq);

#=
Engaged trials should be more often "correct" (choice matching the stimulus
side) than lapsed ones.
=#

correct = [(c > 0) == (o.choice == 1) for (c, o) in zip(control_seq, sim.obs_seq)]
mean(correct[sim.state_seq .== 1]), mean(correct[sim.state_seq .== 2])

# ## Fitting with Baum-Welch

#=
`baum_welch` estimates every parameter jointly: the initial state
probabilities, the transition matrix, and the DDM parameters of each state.
In the M-step each state's DDM is refit by weighted maximum likelihood, with
that state's posterior probabilities as the trial weights.
=#

dists0 = [
    StimulusCodedDDM(; ν=1.5, α=1.0, z=0.5, τ=0.15),
    StimulusCodedDDM(; ν=0.8, α=0.9, z=0.5, τ=0.15),
]
hmm0 = ControlledEmissionHMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], dists0)
hmm_est, lls = baum_welch(hmm0, sim.obs_seq, control_seq; seq_ends=[T], max_iterations=10)
[(; d.ν, d.α, d.z, d.τ) for d in hmm_est.dists]

# ## Checking the emissions with ACDC

#=
DDM emissions support [ACDC model selection](../acdc.md): each `(choice, rt)`
trial is inverted through the Rosenblatt transform of its DDM, giving two
stochastic drivers per trial that are uniform on $[0, 1]$ under a
well-specified model. Small per-state discrepancies confirm the fitted
two-state model.
=#

acdc = component_discrepancies(
    hmm_est, sim.obs_seq, KSDiscrepancy(); control_seq=control_seq, seq_ends=[T], rng=rng
)
acdc.component_discrepancies

# ## Tests  #src

@test isapprox(d.ν, 2.0; rtol=0.15)  #src
@test isapprox(d.α, 1.2; rtol=0.15)  #src
@test isapprox(d.z, 0.55; atol=0.05)  #src
@test isapprox(d.τ, 0.3; atol=0.05)  #src
@test mean(correct[sim.state_seq .== 1]) > mean(correct[sim.state_seq .== 2])  #src
@test all(diff(lls) .>= -1e-6)  #src
νs = [d.ν for d in hmm_est.dists]  #src
@test maximum(νs) > 1.5 && minimum(νs) < 1.0  #src
@test all(isfinite, acdc.component_discrepancies)  #src
@test maximum(acdc.component_discrepancies) < 0.1  #src
