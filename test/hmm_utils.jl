using EmissionModels
using Distributions
using HiddenMarkovModels
using Random
using LinearAlgebra

function create_hmm(
    emission_model_type; n_states=3, α=10.0, rng=Random.GLOBAL_RNG, kwargs...
)
    # Initialize transition matrix and initial distribution
    trans = Matrix{Float64}(undef, n_states, n_states)

    # Fill transition matrix with sticky transitions
    for i in 1:n_states
        dir_vector = ones(n_states)
        dir_vector[i] = α  # Make diagonal sticky!
        trans[i, :] = rand(rng, Dirichlet(dir_vector))
    end

    # Fill initial distribution (uniform Dirichlet)
    init = rand(rng, Dirichlet(ones(n_states)))

    # Create emission models for each state
    emissions = _create_emissions(emission_model_type, n_states, rng; kwargs...)

    # Create and return the HMM
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
    ::Type{MultivariateT}, n_states, rng; dim=2, ν_range=(3.0, 10.0), kwargs...
)
    emissions = Vector{MultivariateT}(undef, n_states)
    for i in 1:n_states
        # Random mean vector
        μ = randn(rng, dim) .* 2.0

        # Random positive definite covariance matrix
        A = randn(rng, dim, dim)
        Σ = A' * A + I  # Ensure positive definite
        Σ = (Σ + Σ') / 2  # Ensure symmetric

        # Random degrees of freedom
        ν = rand(rng) * (ν_range[2] - ν_range[1]) + ν_range[1]

        emissions[i] = MultivariateT(μ, Σ, ν)
    end
    return emissions
end

function _create_emissions(
    ::Type{MultivariateTDiag},
    n_states,
    rng;
    dim=2,
    ν_range=(3.0, 10.0),
    σ²_range=(0.5, 2.0),
    kwargs...,
)
    emissions = Vector{MultivariateTDiag}(undef, n_states)
    for i in 1:n_states
        # Random mean vector
        μ = randn(rng, dim) .* 2.0

        # Random diagonal variances
        σ² = rand(rng, dim) .* (σ²_range[2] - σ²_range[1]) .+ σ²_range[1]

        # Random degrees of freedom
        ν = rand(rng) * (ν_range[2] - ν_range[1]) + ν_range[1]

        emissions[i] = MultivariateTDiag(μ, σ², ν)
    end
    return emissions
end
