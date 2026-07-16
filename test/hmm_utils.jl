using EmissionModels
using Distributions
using HiddenMarkovModels
using Random
using LinearAlgebra

function create_hmm(
    emission_model_type; n_states=3, α=10.0, rng=Random.GLOBAL_RNG, kwargs...
)
    # Sticky transitions: α weights the diagonal of each Dirichlet draw.
    trans = Matrix{Float64}(undef, n_states, n_states)
    for i in 1:n_states
        dir_vector = ones(n_states)
        dir_vector[i] = α
        trans[i, :] = rand(rng, Dirichlet(dir_vector))
    end

    init = rand(rng, Dirichlet(ones(n_states)))
    emissions = _create_emissions(emission_model_type, n_states, rng; kwargs...)
    return HMM(init, trans, emissions)
end

function _create_emissions(
    ::Type{PoissonZeroInflated},
    n_states,
    rng;
    λ_range=(1.0, 10.0),
    π_range=(0.1, 0.4),
    kwargs...,
)
    emissions = Vector{PoissonZeroInflated}(undef, n_states)
    for i in 1:n_states
        λ = rand(rng) * (λ_range[2] - λ_range[1]) + λ_range[1]
        π = rand(rng) * (π_range[2] - π_range[1]) + π_range[1]
        emissions[i] = PoissonZeroInflated(λ, π)
    end
    return emissions
end

function _create_emissions(
    ::Type{MvT}, n_states, rng; dim=2, ν_range=(3.0, 10.0), kwargs...
)
    emissions = Vector{MvT}(undef, n_states)
    for i in 1:n_states
        μ = randn(rng, dim) .* 2.0

        A = randn(rng, dim, dim)
        Σ = A' * A + I  # positive definite
        Σ = (Σ + Σ') / 2  # symmetrize

        ν = rand(rng) * (ν_range[2] - ν_range[1]) + ν_range[1]

        emissions[i] = MvT(μ, Σ, ν)
    end
    return emissions
end

function _create_emissions(
    ::Type{MvTDiag},
    n_states,
    rng;
    dim=2,
    ν_range=(3.0, 10.0),
    σ²_range=(0.5, 2.0),
    kwargs...,
)
    emissions = Vector{MvTDiag}(undef, n_states)
    for i in 1:n_states
        μ = randn(rng, dim) .* 2.0
        σ² = rand(rng, dim) .* (σ²_range[2] - σ²_range[1]) .+ σ²_range[1]
        ν = rand(rng) * (ν_range[2] - ν_range[1]) + ν_range[1]

        emissions[i] = MvTDiag(μ, σ², ν)
    end
    return emissions
end
