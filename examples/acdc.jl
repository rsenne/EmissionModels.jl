# # Model selection with ACDC

#=
Here we use the Accumulated Cutoff Discrepancy Criterion (ACDC, Li et al.
2026) to choose the number of hidden states, and to detect a misspecified
emission model.
=#

using Distributions
using EmissionModels
using HiddenMarkovModels
using Random
using Statistics
using Test  #src

#-

rng = MersenneTwister(63);

# ## Stochastic drivers

#=
If a fitted model is correctly specified, the probability integral transform
(PIT) of each observation under its generating emission is uniform on
$[0, 1]$. ACDC recovers these *stochastic drivers* state by state, scores each
state by the discrepancy of its drivers from uniformity, and selects the
smallest number of states whose scores all fall below a cutoff $\rho$. Unlike
AIC or BIC, no parameter count is involved.

Let us generate data from a three-state Gaussian HMM.
=#

hmm_true = HMM(
    [0.4, 0.3, 0.3],
    [0.90 0.05 0.05; 0.05 0.90 0.05; 0.05 0.05 0.90],
    [Normal(-5.0, 1.0), Normal(0.0, 1.0), Normal(5.0, 1.0)],
)
_, obs_seq = rand(rng, hmm_true, 3000);

#=
[`stochastic_drivers`](@ref) inverts each emission through the PIT and pools
the drivers by state, assigning each observation according to its posterior
state membership. There is one pool of size $D \times N_k$ per state (the
observations here are scalar, so $D = 1$).
=#

sd = stochastic_drivers(hmm_true, obs_seq; rng=rng)
size.(sd.ε_pools)

#=
Under the true model, every pool should look uniform, with mean $1/2$.
=#

[mean(pool) for pool in sd.ε_pools]

# ## Choosing the number of states

#=
We fit candidate models with $K = 1, \dots, 4$ states by Baum-Welch, starting
each from evenly spread quantiles of the data, and score every fitted model
with [`component_discrepancies`](@ref). Here we measure uniformity with the
Kolmogorov-Smirnov statistic ([`KSDiscrepancy`](@ref)); see the
[ACDC page](../acdc.md) for the other available measures.
=#

function candidate_hmm(K, obs_seq)
    stay = K == 1 ? 1.0 : 0.9
    init = fill(1 / K, K)
    trans = [i == j ? stay : (1 - stay) / (K - 1) for i in 1:K, j in 1:K]
    dists = [Normal(quantile(obs_seq, (2k - 1) / (2K)), 1.0) for k in 1:K]
    return HMM(init, trans, dists)
end

results = map(1:4) do K
    hmm_est, _ = baum_welch(candidate_hmm(K, obs_seq), obs_seq)
    return component_discrepancies(hmm_est, obs_seq, KSDiscrepancy(); rng=rng)
end
[maximum(r.component_discrepancies) for r in results]

#=
Underfitted models leave clearly non-uniform drivers, while the three-state
model passes. [`acdc_select`](@ref) turns these scores into a choice: it
minimizes the accumulated loss $\sum_k \max(0, \hat{D}_k - \rho)$, breaking
ties in favor of fewer states (which is how an overfitted but well-calibrated
four-state model loses to the three-state one).
=#

K_selected = acdc_select(results, 0.1)

#=
[`get_critical_rho_values`](@ref) lists the cutoffs at which the selection
changes, which is useful to check how sensitive the choice is to $\rho$.
=#

get_critical_rho_values(results)

# ## Detecting a misspecified emission

#=
Because ACDC scores each state separately, it also reveals *which* emission is
wrong, even when the number of states is correct. Let us generate zero-inflated
counts and compare a plain Poisson fit against a zero-inflated one.
=#

hmm_zi = HMM(
    [0.5, 0.5],
    [0.95 0.05; 0.05 0.95],
    [PoissonZeroInflated(3.0, 0.4), PoissonZeroInflated(12.0, 0.4)],
)
_, obs_zi = rand(rng, hmm_zi, 2000);

#=
First, a two-state HMM with plain Poisson emissions, which cannot account for
the excess zeros.
=#

guess_pois = HMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], [Poisson(2.0), Poisson(8.0)])
hmm_pois, _ = baum_welch(guess_pois, obs_zi)
res_pois = component_discrepancies(hmm_pois, obs_zi, KSDiscrepancy(); rng=rng)
res_pois.component_discrepancies

#=
Then the correctly specified zero-inflated model.
=#

guess_zi = HMM(
    [0.5, 0.5],
    [0.9 0.1; 0.1 0.9],
    [PoissonZeroInflated(2.0, 0.3), PoissonZeroInflated(8.0, 0.3)],
)
hmm_zi_est, _ = baum_welch(guess_zi, obs_zi)
res_zi = component_discrepancies(hmm_zi_est, obs_zi, KSDiscrepancy(); rng=rng)
res_zi.component_discrepancies

#=
The misspecified Poisson states are flagged with large discrepancies, while
the zero-inflated fit passes.
=#

# ## Tests  #src

@test sum(size.(sd.ε_pools, 2)) == 3000  #src
@test all(pool -> all(0 .<= pool .<= 1), sd.ε_pools)  #src
@test all(pool -> isapprox(mean(pool), 0.5; atol=0.05), sd.ε_pools)  #src
@test K_selected == 3  #src
@test maximum(results[1].component_discrepancies) > 0.1  #src
@test maximum(results[2].component_discrepancies) > 0.1  #src
@test maximum(results[3].component_discrepancies) < 0.1  #src
@test maximum(res_pois.component_discrepancies) > 0.1  #src
@test maximum(res_zi.component_discrepancies) < 0.1  #src
