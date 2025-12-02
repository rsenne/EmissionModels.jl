using EmissionModels
using Test
using Random
using DensityInterface
using StatsAPI
using SpecialFunctions: loggamma
using Statistics: mean

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

        # Verify HasDensity trait
        @test DensityKind(dist) == HasDensity()

        # Test log density for zero
        # P(X=0) = π + (1-π)exp(-λ) = 0.3 + 0.7*exp(-3.0)
        expected_p0 = 0.3 + 0.7 * exp(-3.0)
        @test logdensityof(dist, 0) ≈ log(expected_p0) rtol = 1e-10

        # Test log density for positive values
        # P(X=k) = (1-π) * λ^k * exp(-λ) / k!
        for k in 1:5
            expected = log(1 - 0.3) + k * log(3.0) - 3.0 - loggamma(k + 1)
            @test logdensityof(dist, k) ≈ expected rtol = 1e-10
        end

        # Test negative values return -Inf
        @test logdensityof(dist, -1) == -Inf
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        dist = PoissonZeroInflated(5.0, 0.4)

        # Generate samples
        n_samples = 10000
        samples = [rand(rng, dist) for _ in 1:n_samples]

        # All samples should be non-negative integers
        @test all(s >= 0 for s in samples)
        @test all(s isa Integer for s in samples)

        # Check empirical zero inflation
        zero_proportion = count(==(0), samples) / n_samples
        # Should have more zeros than regular Poisson
        poisson_zero_prob = exp(-5.0)
        zip_zero_prob = 0.4 + 0.6 * poisson_zero_prob
        @test zero_proportion > poisson_zero_prob
        @test zero_proportion ≈ zip_zero_prob atol = 0.05

        # Check empirical mean of non-zero samples
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

        # Generate synthetic data
        n = 1000
        obs = [rand(rng, true_dist) for _ in 1:n]
        weights = ones(n)

        # Fit distribution
        fitted_dist = PoissonZeroInflated(1.0, 0.1)  # Start with different values
        fit!(fitted_dist, obs, weights)

        # Parameters should be close to true values
        @test fitted_dist.λ ≈ true_λ atol = 0.3
        @test fitted_dist.π ≈ true_π atol = 0.05
    end

    @testset "fit! with weighted observations" begin
        # Create observations with varying weights
        obs = [0, 0, 0, 1, 2, 3, 4, 5]
        weights = [2.0, 1.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]

        dist = PoissonZeroInflated(2.0, 0.5)
        fit!(dist, obs, weights)

        # Should converge to reasonable values
        @test 0 < dist.π < 1
        @test dist.λ > 0

        # Verify it handles the weighted data
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
        # Simulate a complete workflow like HMM would use
        rng = Random.MersenneTwister(456)

        # Create true distribution
        true_dist = PoissonZeroInflated(6.0, 0.15)

        # Generate observations
        n_obs = 500
        observations = [rand(rng, true_dist) for _ in 1:n_obs]

        # Simulate HMM posterior weights (random weights summing to reasonable values)
        weights = rand(rng, n_obs) .+ 0.5

        # Fit new distribution
        fitted = PoissonZeroInflated(1.0, 0.5)
        fit!(fitted, observations, weights)

        # Check that fitted parameters are reasonable
        @test 0 < fitted.π < 1
        @test fitted.λ > 0

        # Generate new samples from fitted distribution
        new_samples = [rand(rng, fitted) for _ in 1:100]
        @test all(s >= 0 for s in new_samples)

        # Compute log densities
        log_probs = [logdensityof(fitted, obs) for obs in observations[1:10]]
        @test all(isfinite, log_probs)
    end
end
