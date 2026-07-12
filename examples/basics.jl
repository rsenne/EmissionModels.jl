# # Basics

#=
Here we show how to use the emission models of this package inside a hidden
Markov model from
[HiddenMarkovModels.jl](https://github.com/JuliaStats/HiddenMarkovModels.jl).
=#

using Distributions
using EmissionModels
using HiddenMarkovModels
using Random
using Statistics
using Test  #src

#-

rng = MersenneTwister(63);

# ## Emission models

#=
An *emission model* describes how observations are generated conditioned on the
latent state of an HMM. Every distribution in this package implements the
interface that HiddenMarkovModels.jl expects from an observation distribution:

- `rand(rng, dist)` samples an observation
- `logdensityof(dist, obs)` evaluates the loglikelihood of an observation
- `fit!(dist, obs_seq, weight_seq)` updates the parameters from weighted observations

Take the zero-inflated Poisson ([`PoissonZeroInflated`](@ref)) as an example.
It models count data with excess zeros: with probability $\pi$ the observation
is a structural zero, otherwise it is drawn from a Poisson with rate $\lambda$.
=#

dist = PoissonZeroInflated(5.0, 0.3)

#-

x = rand(rng, dist)

#-

logdensityof(dist, x)

#=
The `fit!` method performs a weighted maximum-likelihood update, which is
exactly the operation Baum-Welch needs during learning. With unit weights it
reduces to ordinary maximum likelihood.
=#

obs = [rand(rng, dist) for _ in 1:2000]
dist_est = PoissonZeroInflated(1.0, 0.5)
fit!(dist_est, obs, ones(2000))
(dist_est.λ, dist_est.π)

# ## Model

#=
Because the interface is the standard one, a vector of emission models is a
valid `dists` argument for the `HMM` type of HiddenMarkovModels.jl. Here is a
two-state model for count data: a quiet state with many structural zeros and
an active state with a high rate.
=#

init = [0.6, 0.4]
trans = [0.9 0.1; 0.2 0.8]
dists = [PoissonZeroInflated(2.0, 0.5), PoissonZeroInflated(15.0, 0.05)]
hmm = HMM(init, trans, dists)

# ## Simulation

#=
You can simulate a pair of state and observation sequences with `rand` by
specifying how long you want them to be.
=#

state_seq, obs_seq = rand(rng, hmm, 1000);

#-

obs_seq[1:5]

# ## Inference

#=
All the inference algorithms of HiddenMarkovModels.jl work out of the box.
Since the two states are well separated, the Viterbi algorithm recovers most
of the true state sequence.
=#

best_state_seq, _ = viterbi(hmm, obs_seq);
mean(best_state_seq .== state_seq)

#=
The loglikelihood of the observation sequence is computed with the forward
algorithm, which `logdensityof` wraps.
=#

logdensityof(hmm, obs_seq)

# ## Learning

#=
The Baum-Welch algorithm is a local optimization procedure, so it requires a
starting point that is close enough to the true model.
=#

dists_guess = [PoissonZeroInflated(1.0, 0.3), PoissonZeroInflated(10.0, 0.2)]
hmm_guess = HMM([0.5, 0.5], [0.8 0.2; 0.3 0.7], dists_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq);
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
The estimated rates and zero-inflation probabilities are close to the truth.
Keep in mind that HMMs are only identifiable up to a permutation of the
states, so in general the estimated states must be matched to the true ones
before comparing parameters.
=#

hcat(
    [(d.λ, d.π) for d in obs_distributions(hmm_est)],
    [(d.λ, d.π) for d in obs_distributions(hmm)],
)

#=
Multivariate emissions like [`MultivariateT`](@ref) work in exactly the same
way, and the [GLM emissions](glm.md) additionally condition on a control
vector.
=#

# ## Tests  #src

@test dist_est.λ ≈ 5.0 atol = 0.3  #src
@test dist_est.π ≈ 0.3 atol = 0.03  #src
@test mean(best_state_seq .== state_seq) > 0.9  #src
@test all(diff(loglikelihood_evolution) .>= -1e-6)  #src
ds_est = obs_distributions(hmm_est)  #src
order = sortperm([d.λ for d in ds_est])  #src
@test [d.λ for d in ds_est[order]] ≈ [2.0, 15.0] atol = 1.0  #src
@test [d.π for d in ds_est[order]] ≈ [0.5, 0.05] atol = 0.1  #src
