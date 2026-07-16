"""
    test_hmm_integration(rng, hmm; T=100, max_iterations=5)

Emission-agnostic smoke test of the HiddenMarkovModels.jl pipeline:
structural sanity of `hmm`, simulation of length `T`, `forward`, and a short
`baum_welch` run. Returns the simulated `(; state_seq, obs_seq)` so callers
can add model-specific checks on the observations.
"""
function test_hmm_integration(
    rng::AbstractRNG, hmm::HMM; T::Integer=100, max_iterations::Integer=5
)
    n_states = length(hmm.init)
    @test size(hmm.trans) == (n_states, n_states)
    @test length(hmm.dists) == n_states
    @test all(sum(hmm.trans; dims=2) .≈ 1.0)
    @test sum(hmm.init) ≈ 1.0

    state_seq, obs_seq = rand(rng, hmm, T)
    @test length(state_seq) == T
    @test length(obs_seq) == T

    log_alpha, log_ll = forward(hmm, obs_seq)
    @test size(log_alpha) == (n_states, T)
    @test all(isfinite, log_alpha)
    @test all(isfinite, log_ll)

    _, lls = baum_welch(hmm, obs_seq; max_iterations)
    @test length(lls) <= max_iterations
    @test all(isfinite, lls)

    return (; state_seq, obs_seq)
end
