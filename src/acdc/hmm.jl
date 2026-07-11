#= ACDC stochastic-driver recovery for HiddenMarkovModels.jl HMMs.

   Driver recovery uses the forward-backward state posteriors plus the
   per-emission PIT defined in `_emission_to_driver` (see `drivers.jl`), so it
   works with any HMM whose emissions are standard `Distributions` (or a type
   with a custom driver method). =#

"""
    stochastic_drivers(hmm::AbstractHMM, obs_seq; control_seq, seq_ends, n_samples=1, rng)

Recover the ACDC stochastic drivers for a fitted HiddenMarkovModels.jl `hmm`.

For each time step the hidden state is sampled from its forward-backward
posterior, and that state's emission is inverted to a driver via the probability
integral transform (see `_emission_to_driver`). Component "usage" is the
posterior expected time spent in each state.

# Arguments
- `hmm::AbstractHMM`: a fitted HMM satisfying the HiddenMarkovModels.jl interface.
- `obs_seq::AbstractVector`: observation sequence (scalars or vectors), as passed
  to `forward_backward`.

# Keyword Arguments
- `control_seq::AbstractVector`: controls for control-dependent HMMs; defaults to
  `fill(nothing, length(obs_seq))`.
- `seq_ends`: end indices when `obs_seq` concatenates multiple sequences; defaults
  to `(length(obs_seq),)`.
- `n_samples::Int=1`: number of posterior sampling passes over the data.
- `rng::AbstractRNG=Random.default_rng()`: source of randomness for the posterior
  state sampling and the randomized PITs — seed it for reproducible drivers.

# Returns
- [`StochasticDriverResult`](@ref) with per-state driver pools and usage.
"""
function stochastic_drivers(
    hmm::AbstractHMM,
    obs_seq::AbstractVector;
    control_seq::AbstractVector=fill(nothing, length(obs_seq)),
    seq_ends=(length(obs_seq),),
    n_samples::Int=1,
    rng::AbstractRNG=Random.default_rng(),
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    isempty(obs_seq) && throw(ArgumentError("obs_seq must be non-empty"))
    T_len = length(obs_seq)
    K = length(hmm)

    # State posteriors γ (K × T) from forward-backward.
    γ, _ = forward_backward(hmm, obs_seq, control_seq; seq_ends=seq_ends)

    # Usage: expected fraction of time in each state.
    usage = vec(sum(γ; dims=2)) ./ T_len

    # Driver dimension from a probe on the first observation's sampled-state-able
    # emission (all states share the emission dimension in an HMM).
    probe = _emission_to_driver(
        rng, obs_distributions(hmm, control_seq[1])[1], obs_seq[1], control_seq[1]
    )
    D = length(probe)
    Tε = eltype(probe)

    ε_lists = [Vector{Vector{Tε}}() for _ in 1:K]
    for _ in 1:n_samples
        for t in 1:T_len
            z = _sample_categorical(rng, view(γ, :, t))
            dist = obs_distributions(hmm, control_seq[t])[z]
            push!(ε_lists[z], _emission_to_driver(rng, dist, obs_seq[t], control_seq[t]))
        end
    end

    # Pack each state's drivers into a D × n_k matrix.
    ε_pools = Vector{Matrix{Tε}}(undef, K)
    for k in 1:K
        if isempty(ε_lists[k])
            ε_pools[k] = Matrix{Tε}(undef, D, 0)
        else
            ε_pools[k] = reduce(hcat, ε_lists[k])
        end
    end

    return StochasticDriverResult(ε_pools, collect(Tε, usage))
end
