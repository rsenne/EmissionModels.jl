# # GLM emissions

#=
Here we build a Markov switching regression: an HMM whose emissions are
generalized linear models, conditioned on a control (design) vector at each
time step.
=#

using Distributions
using EmissionModels
using HiddenMarkovModels
using HiddenMarkovModels: ControlledEmissionHMM
using LinearAlgebra
using Random
using Statistics
using Test  #src

#-

rng = MersenneTwister(63);

# ## Model

#=
A GLM emission maps a control vector $x$ to the parameters of an observation
distribution. For instance [`PoissonGLM`](@ref) models
$y \mid x \sim \mathrm{Poisson}(\exp(\beta^\top x))$. No intercept is added
automatically: if you want one, include a constant entry in $x$.
=#

glm = PoissonGLM([0.5, -0.3])

#=
Outside of any HMM, a GLM can be sampled and evaluated by passing the control
vector as a keyword argument.
=#

x = [1.0, 2.0]
y = rand(rng, glm; control_seq=x)

#-

logdensityof(glm, y; control_seq=x)

#=
All GLMs subtype `ControlledEmission` from HiddenMarkovModels.jl, so a vector
of them is a valid `dists` argument for a `ControlledEmissionHMM`. Here is a
two-state Markov switching Poisson regression.
=#

p = 3
init = [0.6, 0.4]
trans = [0.92 0.08; 0.15 0.85]
dists = [PoissonGLM([0.2, 0.5, -0.3]), PoissonGLM([1.2, -0.4, 0.6])]
hmm = ControlledEmissionHMM(init, trans, dists)

# ## Simulation

#=
A controlled HMM cannot be simulated for a bare length `T`: there is no
sensible default control. Provide one control vector per time step instead,
here with a constant first entry acting as the intercept.
=#

T = 1000
control_seq = [vcat(1.0, randn(rng, p - 1)) for _ in 1:T];
state_seq, obs_seq = rand(rng, hmm, control_seq);

#-

obs_seq[1:5]

# ## Inference

#=
Inference algorithms take the control sequence as an additional positional
argument.
=#

best_state_seq, _ = viterbi(hmm, obs_seq, control_seq);
mean(best_state_seq .== state_seq)

#-

logdensityof(hmm, obs_seq, control_seq)

# ## Learning

#=
During Baum-Welch, each state's `fit!` solves a weighted GLM problem: a
closed-form weighted least squares for [`GaussianGLM`](@ref) and
[`MvGaussianGLM`](@ref), a Newton solve for the others. As always with a
local optimization procedure, start from a guess close enough to the truth.
=#

dists_guess = [PoissonGLM([0.0, 0.3, -0.1]), PoissonGLM([1.0, -0.2, 0.4])]
hmm_guess = ControlledEmissionHMM([0.5, 0.5], [0.8 0.2; 0.3 0.7], dists_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq, control_seq);
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
How did we perform?
=#

hcat(hmm_est.dists[1].β, hmm.dists[1].β)

#-

hcat(hmm_est.dists[2].β, hmm.dists[2].β)

# ## Priors

#=
Every GLM accepts an optional prior on the coefficients, which turns the
M-step into a penalized (MAP) update. [`RidgePrior`](@ref) adds an $\ell_2$
penalty $\frac{\lambda}{2} \lVert \beta \rVert^2$, which is useful when
controls are correlated or data per state is scarce. Fitting the same data
with and without a ridge prior shrinks the coefficients toward zero.
=#

control_mat = permutedims(reduce(hcat, control_seq))
glm_ridge = PoissonGLM(zeros(p), RidgePrior(10.0))
glm_flat = PoissonGLM(zeros(p))
fit!(glm_ridge, obs_seq, ones(T); control_seq=control_mat)
fit!(glm_flat, obs_seq, ones(T); control_seq=control_mat)
norm(glm_ridge.β), norm(glm_flat.β)

#=
The other GLM families ([`GaussianGLM`](@ref), [`BernoulliGLM`](@ref),
[`MultinomialGLM`](@ref) and the multivariate variants) follow the same
interface.
=#

# ## Tests  #src

@test mean(best_state_seq .== state_seq) > 0.7  #src
@test all(diff(loglikelihood_evolution) .>= -1e-6)  #src
@test hmm_est.dists[1].β ≈ hmm.dists[1].β atol = 0.2  #src
@test hmm_est.dists[2].β ≈ hmm.dists[2].β atol = 0.2  #src
@test norm(glm_ridge.β) < norm(glm_flat.β)  #src
