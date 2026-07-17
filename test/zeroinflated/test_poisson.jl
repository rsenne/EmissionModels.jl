using EmissionModels
using EmissionModelsTest
using Test
using Random
using DensityInterface
using StatsAPI
using SpecialFunctions: loggamma
using Statistics: mean
using HiddenMarkovModels

import StatsAPI: fit!

@testset "PoissonZeroInflated" begin
    @testset "Constructor" begin
        # Valid construction
        dist = PoissonZeroInflated(3.0, 0.2)
        @test dist.λ == 3.0
        @test dist.π == 0.2

        # Type promotion
        dist2 = PoissonZeroInflated(3, 0.2)
        @test dist2.λ isa Float64
        @test dist2.π isa Float64

        # Invalid parameters
        @test_throws ArgumentError PoissonZeroInflated(-1.0, 0.2)  # negative λ
        @test_throws ArgumentError PoissonZeroInflated(3.0, -0.1)  # π < 0
        @test_throws ArgumentError PoissonZeroInflated(3.0, 1.1)   # π > 1
    end

    @testset "DensityInterface" begin
        dist = PoissonZeroInflated(3.0, 0.3)

        @test DensityKind(dist) == HasDensity()

        # P(X=0) = π + (1-π)exp(-λ) = 0.3 + 0.7*exp(-3.0)
        expected_p0 = 0.3 + 0.7 * exp(-3.0)
        @test logdensityof(dist, 0) ≈ log(expected_p0) rtol = 1e-10

        # P(X=k) = (1-π) * λ^k * exp(-λ) / k!
        for k in 1:5
            expected = log(1 - 0.3) + k * log(3.0) - 3.0 - loggamma(k + 1)
            @test logdensityof(dist, k) ≈ expected rtol = 1e-10
        end

        @test logdensityof(dist, -1) == -Inf
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        dist = PoissonZeroInflated(5.0, 0.4)

        n_samples = 10000
        samples = [rand(rng, dist) for _ in 1:n_samples]

        # All samples should be non-negative integers
        @test all(s >= 0 for s in samples)
        @test all(s isa Integer for s in samples)

        # Empirical zero fraction should match the mixture.
        zero_proportion = count(==(0), samples) / n_samples
        # Should have more zeros than regular Poisson
        poisson_zero_prob = exp(-5.0)
        zip_zero_prob = 0.4 + 0.6 * poisson_zero_prob
        @test zero_proportion > poisson_zero_prob
        @test zero_proportion ≈ zip_zero_prob atol = 0.05

        # Empirical mean of the nonzero samples.
        nonzero_samples = filter(>(0), samples)
        if !isempty(nonzero_samples)
            # Mean of non-zeros should be close to λ
            @test mean(nonzero_samples) ≈ 5.0 atol = 0.5
        end
    end

    @testset "fit! with uniform weights" begin
        rng = Random.MersenneTwister(123)
        true_λ = 4.0
        true_π = 0.25
        true_dist = PoissonZeroInflated(true_λ, true_π)

        n = 1000
        obs = [rand(rng, true_dist) for _ in 1:n]
        weights = ones(n)

        fitted_dist = PoissonZeroInflated(1.0, 0.1)  # Start with different values
        fit!(fitted_dist, obs, weights)

        # Parameters should be close to true values
        @test fitted_dist.λ ≈ true_λ atol = 0.3
        @test fitted_dist.π ≈ true_π atol = 0.05
    end

    @testset "fit! with weighted observations" begin
        obs = [0, 0, 0, 1, 2, 3, 4, 5]
        weights = [2.0, 1.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]

        dist = PoissonZeroInflated(2.0, 0.5)
        fit!(dist, obs, weights)

        # Should converge to reasonable values
        @test 0 < dist.π < 1
        @test dist.λ > 0

        # Weighted data should fit without issue.
        @test dist.π > 0.1  # Should detect some zero inflation
    end

    @testset "fit! edge cases" begin
        # All zeros
        obs = [0, 0, 0, 0]
        weights = ones(4)
        dist = PoissonZeroInflated(1.0, 0.1)
        fit!(dist, obs, weights)
        @test dist.π > 0.5  # Should have high zero inflation

        # Empty data
        obs = Int[]
        weights = Float64[]
        dist = PoissonZeroInflated(2.0, 0.3)
        old_λ = dist.λ
        old_π = dist.π
        fit!(dist, obs, weights)
        @test dist.λ == old_λ  # Should not change
        @test dist.π == old_π

        # Zero total weight
        obs = [1, 2, 3]
        weights = [0.0, 0.0, 0.0]
        dist = PoissonZeroInflated(2.0, 0.3)
        old_λ = dist.λ
        old_π = dist.π
        fit!(dist, obs, weights)
        @test dist.λ == old_λ
        @test dist.π == old_π

        # Mismatched lengths
        obs = [1, 2, 3]
        weights = [1.0, 1.0]
        dist = PoissonZeroInflated(2.0, 0.3)
        @test_throws DimensionMismatch fit!(dist, obs, weights)
    end

    @testset "Integration test" begin
        # Simulate the full workflow an HMM fit would drive.
        rng = Random.MersenneTwister(456)

        true_dist = PoissonZeroInflated(6.0, 0.15)

        n_obs = 500
        observations = [rand(rng, true_dist) for _ in 1:n_obs]

        # Simulate HMM posterior weights (random weights summing to reasonable values)
        weights = rand(rng, n_obs) .+ 0.5

        fitted = PoissonZeroInflated(1.0, 0.5)
        fit!(fitted, observations, weights)

        # Fitted parameters should be sane.
        @test 0 < fitted.π < 1
        @test fitted.λ > 0

        new_samples = [rand(rng, fitted) for _ in 1:100]
        @test all(s >= 0 for s in new_samples)

        log_probs = [logdensityof(fitted, obs) for obs in observations[1:10]]
        @test all(isfinite, log_probs)
    end

    @testset "HMM Integration" begin
        rng = Random.MersenneTwister(777)

        hmm = create_hmm(PoissonZeroInflated; n_states=3, α=15.0, rng=rng)
        @test length(hmm.init) == 3
        @test all(dist -> dist isa PoissonZeroInflated, hmm.dists)

        _, obs_seq = test_hmm_integration(rng, hmm; T=100)
        @test all(obs -> obs >= 0, obs_seq)

        # Again with custom parameters.
        hmm_custom = create_hmm(
            PoissonZeroInflated;
            n_states=4,
            α=5.0,
            λ_range=(2.0, 8.0),
            π_range=(0.2, 0.5),
            rng=rng,
        )
        @test length(hmm_custom.init) == 4
        @test all(2.0 <= dist.λ <= 8.0 for dist in hmm_custom.dists)
        @test all(0.2 <= dist.π <= 0.5 for dist in hmm_custom.dists)
    end
end

@testset "Array sampling API" begin
    rng = Random.MersenneTwister(42)
    dist = PoissonZeroInflated(3.0, 0.2)
    samples = rand(rng, dist, 100)
    @test samples isa Vector{Int}
    @test length(samples) == 100
    @test all(s >= 0 for s in samples)
    # The docstring form without an explicit rng.
    @test rand(dist, 5) isa Vector{Int}
end

@testset "logdensityof accepts Real-stored counts" begin
    dist = PoissonZeroInflated(3.0, 0.3)
    # Count data often arrives as Float64; integer-valued floats must score
    # identically, and zero-mass values must give -Inf rather than throw.
    @test logdensityof(dist, 4.0) == logdensityof(dist, 4)
    @test logdensityof(dist, 0.0) == logdensityof(dist, 0)
    @test logdensityof(dist, 2.5) == -Inf
    @test logdensityof(dist, -1.0) == -Inf
end

@testset "Float32 type stability" begin
    dist = PoissonZeroInflated(3.0f0, 0.2f0)
    @test dist.λ isa Float32
    # Every logdensityof branch must return the field type T (no Float64 leak
    # from loggamma or hardcoded literals).
    @test @inferred(logdensityof(dist, 2)) isa Float32   # x > 0 branch
    @test @inferred(logdensityof(dist, 0)) isa Float32   # x == 0 branch
    @test @inferred(logdensityof(dist, -1)) isa Float32  # zero-mass branch

    # fit! keeps the parameters in Float32.
    rng = Random.MersenneTwister(2024)
    obs = [rand(rng, dist) for _ in 1:200]
    w = ones(Float32, 200)
    fit!(dist, obs, w)
    @test dist.λ isa Float32
    @test dist.π isa Float32

    # All-zeros branch also stays in T.
    dz = PoissonZeroInflated(3.0f0, 0.2f0)
    fit!(dz, zeros(Int, 5), ones(Float32, 5))
    @test dz.π isa Float32
    @test dz.λ isa Float32
end
