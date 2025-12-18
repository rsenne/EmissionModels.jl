using EmissionModels
using Test
using Random
using DensityInterface
using StatsAPI
using SpecialFunctions: loggamma
using Statistics: mean, cov, var
using LinearAlgebra
using HiddenMarkovModels

import StatsAPI: fit!

@testset "MultivariateT" begin
    @testset "Constructor" begin
        # Valid construction
        μ = [0.0, 0.0]
        Σ = [1.0 0.5; 0.5 1.0]
        ν = 5.0
        dist = MultivariateT(μ, Σ, ν)
        @test dist.μ == μ
        @test dist.Σ == Σ
        @test dist.ν == ν
        @test dist.dim == 2

        # Type promotion
        dist2 = MultivariateT([0, 0], [1.0 0.0; 0.0 1.0], 5)
        @test eltype(dist2.μ) == Float64
        @test eltype(dist2.Σ) == Float64
        @test dist2.ν isa Float64

        # Invalid parameters
        @test_throws ArgumentError MultivariateT([0.0, 0.0], [1.0 0.5; 0.5 1.0], -1.0)  # negative ν
        @test_throws ArgumentError MultivariateT(Float64[], [1.0;;], 5.0)  # empty μ
        @test_throws DimensionMismatch MultivariateT([0.0, 0.0], [1.0;;], 5.0)  # wrong Σ size
        @test_throws ArgumentError MultivariateT([0.0, 0.0], [1.0 0.5; 0.5 -1.0], 5.0)  # non-PD Σ
    end

    @testset "DensityInterface" begin
        μ = [1.0, 2.0]
        Σ = [2.0 0.5; 0.5 1.0]
        ν = 5.0
        dist = MultivariateT(μ, Σ, ν)

        # Verify HasDensity trait
        @test DensityKind(dist) == HasDensity()

        # Test log density at mean
        logp_mean = logdensityof(dist, μ)
        @test isfinite(logp_mean)

        # Test log density at various points
        x1 = [1.0, 2.0]  # at mean
        x2 = [2.0, 3.0]  # shifted
        x3 = [0.0, 1.0]  # shifted other direction

        logp1 = logdensityof(dist, x1)
        logp2 = logdensityof(dist, x2)
        logp3 = logdensityof(dist, x3)

        @test isfinite(logp1)
        @test isfinite(logp2)
        @test isfinite(logp3)

        # Density should be highest at the mean
        @test logp1 > logp2
        @test logp1 > logp3

        # Test symmetry (for symmetric covariance)
        μ_sym = [0.0, 0.0]
        Σ_sym = [1.0 0.0; 0.0 1.0]
        dist_sym = MultivariateT(μ_sym, Σ_sym, ν)
        @test logdensityof(dist_sym, [1.0, 0.0]) ≈ logdensityof(dist_sym, [-1.0, 0.0])

        # Test dimension mismatch
        @test_throws DimensionMismatch logdensityof(dist, [1.0])
        @test_throws DimensionMismatch logdensityof(dist, [1.0, 2.0, 3.0])
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        μ = [2.0, -1.0]
        Σ = [2.0 0.8; 0.8 1.5]
        ν = 10.0
        dist = MultivariateT(μ, Σ, ν)

        # Generate samples
        n_samples = 5000
        samples = [rand(rng, dist) for _ in 1:n_samples]

        # All samples should have correct dimension
        @test all(length(s) == 2 for s in samples)

        # Convert to matrix for statistics
        sample_matrix = hcat(samples...)'

        # Check empirical mean (should be close to μ for ν > 1)
        empirical_mean = vec(mean(sample_matrix, dims=1))
        @test empirical_mean ≈ μ atol = 0.2

        # Check empirical covariance (should be close to Σ * ν/(ν-2) for ν > 2)
        expected_cov = Σ * (ν / (ν - 2))
        empirical_cov = cov(sample_matrix)
        @test isapprox(empirical_cov, expected_cov, atol=0.5)

        # Test with low degrees of freedom (heavier tails)
        dist_heavy = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 3.0)
        samples_heavy = [rand(rng, dist_heavy) for _ in 1:1000]
        @test all(length(s) == 2 for s in samples_heavy)
    end

    @testset "fit! with uniform weights" begin
        rng = Random.MersenneTwister(123)
        true_μ = [1.0, -0.5]
        true_Σ = [2.0 0.3; 0.3 1.0]
        true_ν = 8.0
        true_dist = MultivariateT(true_μ, true_Σ, true_ν)

        # Generate synthetic data
        n = 2000
        obs = [rand(rng, true_dist) for _ in 1:n]
        weights = ones(n)

        # Fit distribution
        fitted_dist = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        fit!(fitted_dist, obs, weights; max_iter=50)

        # Parameters should be close to true values
        @test fitted_dist.μ ≈ true_μ atol = 0.2
        @test fitted_dist.Σ ≈ true_Σ atol = 0.3
        @test fitted_dist.ν ≈ true_ν atol = 2.0
    end

    @testset "fit! with weighted observations" begin
        rng = Random.MersenneTwister(456)

        # Create observations with varying weights
        n = 500
        obs = [randn(rng, 2) for _ in 1:n]
        weights = rand(rng, n) .+ 0.5

        dist = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        fit!(dist, obs, weights; max_iter=30)

        # Should converge to reasonable values
        @test dist.ν > 0
        @test all(isfinite, dist.μ)
        @test isposdef(dist.Σ)
    end

    @testset "fit! with fix_nu option" begin
        rng = Random.MersenneTwister(789)
        n = 500
        obs = [randn(rng, 2) for _ in 1:n]
        weights = ones(n)

        fixed_ν = 7.0
        dist = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], fixed_ν)
        fit!(dist, obs, weights; max_iter=30, fix_nu=true)

        # ν should remain unchanged
        @test dist.ν == fixed_ν

        # But μ and Σ should be updated
        @test dist.μ != [0.0, 0.0]
    end

    @testset "fit! edge cases" begin
        # Empty data
        obs = Vector{Float64}[]
        weights = Float64[]
        dist = MultivariateT([1.0, 2.0], [1.0 0.0; 0.0 1.0], 5.0)
        old_μ = copy(dist.μ)
        old_Σ = copy(dist.Σ)
        old_ν = dist.ν
        fit!(dist, obs, weights)
        @test dist.μ == old_μ
        @test dist.Σ == old_Σ
        @test dist.ν == old_ν

        # Zero total weight
        obs = [[1.0, 2.0], [2.0, 3.0], [0.0, 1.0]]
        weights = [0.0, 0.0, 0.0]
        dist = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        old_μ = copy(dist.μ)
        old_Σ = copy(dist.Σ)
        old_ν = dist.ν
        fit!(dist, obs, weights)
        @test dist.μ == old_μ
        @test dist.Σ == old_Σ
        @test dist.ν == old_ν

        # Mismatched lengths
        obs = [[1.0, 2.0], [2.0, 3.0], [0.0, 1.0]]
        weights = [1.0, 1.0]
        dist = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)

        # Wrong observation dimension
        obs = [[1.0], [2.0], [3.0]]
        weights = [1.0, 1.0, 1.0]
        dist = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)
    end

    @testset "Integration test" begin
        # Simulate a complete workflow like HMM would use
        rng = Random.MersenneTwister(999)

        # Create true distribution
        true_dist = MultivariateT([2.0, -1.0], [1.5 0.4; 0.4 1.0], 6.0)

        # Generate observations
        n_obs = 800
        observations = [rand(rng, true_dist) for _ in 1:n_obs]

        # Simulate HMM posterior weights
        weights = rand(rng, n_obs) .+ 0.5

        # Fit new distribution
        fitted = MultivariateT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        fit!(fitted, observations, weights; max_iter=40)

        # Check that fitted parameters are reasonable
        @test fitted.ν > 0
        @test all(isfinite, fitted.μ)
        @test isposdef(fitted.Σ)

        # Generate new samples from fitted distribution
        new_samples = [rand(rng, fitted) for _ in 1:100]
        @test all(length(s) == 2 for s in new_samples)

        # Compute log densities
        log_probs = [logdensityof(fitted, obs) for obs in observations[1:10]]
        @test all(isfinite, log_probs)
    end
end

@testset "MultivariateTDiag" begin
    @testset "Constructor" begin
        # Valid construction
        μ = [0.0, 0.0]
        σ² = [1.0, 2.0]
        ν = 5.0
        dist = MultivariateTDiag(μ, σ², ν)
        @test dist.μ == μ
        @test dist.σ² == σ²
        @test dist.ν == ν
        @test dist.dim == 2

        # Type promotion
        dist2 = MultivariateTDiag([0, 0], [1.0, 2.0], 5)
        @test eltype(dist2.μ) == Float64
        @test eltype(dist2.σ²) == Float64
        @test dist2.ν isa Float64

        # Invalid parameters
        @test_throws ArgumentError MultivariateTDiag([0.0, 0.0], [1.0, 2.0], -1.0)  # negative ν
        @test_throws ArgumentError MultivariateTDiag(Float64[], [1.0], 5.0)  # empty μ
        @test_throws DimensionMismatch MultivariateTDiag([0.0, 0.0], [1.0], 5.0)  # wrong σ² length
        @test_throws ArgumentError MultivariateTDiag([0.0], [-1.0], 5.0)  # negative variance
        @test_throws ArgumentError MultivariateTDiag([0.0], [0.0], 5.0)  # zero variance
    end

    @testset "DensityInterface" begin
        μ = [1.0, 2.0]
        σ² = [2.0, 1.0]
        ν = 5.0
        dist = MultivariateTDiag(μ, σ², ν)

        # Verify HasDensity trait
        @test DensityKind(dist) == HasDensity()

        # Test log density at mean
        logp_mean = logdensityof(dist, μ)
        @test isfinite(logp_mean)

        # Test log density at various points
        x1 = [1.0, 2.0]  # at mean
        x2 = [2.0, 3.0]  # shifted
        x3 = [0.0, 1.0]  # shifted other direction

        logp1 = logdensityof(dist, x1)
        logp2 = logdensityof(dist, x2)
        logp3 = logdensityof(dist, x3)

        @test isfinite(logp1)
        @test isfinite(logp2)
        @test isfinite(logp3)

        # Density should be highest at the mean
        @test logp1 > logp2
        @test logp1 > logp3

        # Test dimension mismatch
        @test_throws DimensionMismatch logdensityof(dist, [1.0])
        @test_throws DimensionMismatch logdensityof(dist, [1.0, 2.0, 3.0])
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        μ = [2.0, -1.0]
        σ² = [2.0, 1.5]
        ν = 10.0
        dist = MultivariateTDiag(μ, σ², ν)

        # Generate samples
        n_samples = 5000
        samples = [rand(rng, dist) for _ in 1:n_samples]

        # All samples should have correct dimension
        @test all(length(s) == 2 for s in samples)

        # Convert to matrix for statistics
        sample_matrix = hcat(samples...)'

        # Check empirical mean
        empirical_mean = vec(mean(sample_matrix, dims=1))
        @test empirical_mean ≈ μ atol = 0.2

        # Check empirical variances (diagonal only)
        expected_var = σ² * (ν / (ν - 2))
        empirical_var = vec(var(sample_matrix, dims=1))
        @test isapprox(empirical_var, expected_var, atol=0.5)
    end

    @testset "fit! with uniform weights" begin
        rng = Random.MersenneTwister(123)
        true_μ = [1.0, -0.5]
        true_σ² = [2.0, 1.0]
        true_ν = 8.0
        true_dist = MultivariateTDiag(true_μ, true_σ², true_ν)

        # Generate synthetic data
        n = 2000
        obs = [rand(rng, true_dist) for _ in 1:n]
        weights = ones(n)

        # Fit distribution
        fitted_dist = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(fitted_dist, obs, weights; max_iter=50)

        # Parameters should be close to true values
        @test fitted_dist.μ ≈ true_μ atol = 0.2
        @test fitted_dist.σ² ≈ true_σ² atol = 0.3
        @test fitted_dist.ν ≈ true_ν atol = 2.0
    end

    @testset "fit! with weighted observations" begin
        rng = Random.MersenneTwister(456)

        # Create observations with varying weights
        n = 500
        obs = [randn(rng, 2) for _ in 1:n]
        weights = rand(rng, n) .+ 0.5

        dist = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(dist, obs, weights; max_iter=30)

        # Should converge to reasonable values
        @test dist.ν > 0
        @test all(isfinite, dist.μ)
        @test all(>(0), dist.σ²)
    end

    @testset "fit! with fix_nu option" begin
        rng = Random.MersenneTwister(789)
        n = 500
        obs = [randn(rng, 2) for _ in 1:n]
        weights = ones(n)

        fixed_ν = 7.0
        dist = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], fixed_ν)
        fit!(dist, obs, weights; max_iter=30, fix_nu=true)

        # ν should remain unchanged
        @test dist.ν == fixed_ν

        # But μ and σ² should be updated
        @test dist.μ != [0.0, 0.0]
    end

    @testset "fit! edge cases" begin
        # Empty data
        obs = Vector{Float64}[]
        weights = Float64[]
        dist = MultivariateTDiag([1.0, 2.0], [1.0, 1.0], 5.0)
        old_μ = copy(dist.μ)
        old_σ² = copy(dist.σ²)
        old_ν = dist.ν
        fit!(dist, obs, weights)
        @test dist.μ == old_μ
        @test dist.σ² == old_σ²
        @test dist.ν == old_ν

        # Zero total weight
        obs = [[1.0, 2.0], [2.0, 3.0], [0.0, 1.0]]
        weights = [0.0, 0.0, 0.0]
        dist = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        old_μ = copy(dist.μ)
        old_σ² = copy(dist.σ²)
        old_ν = dist.ν
        fit!(dist, obs, weights)
        @test dist.μ == old_μ
        @test dist.σ² == old_σ²
        @test dist.ν == old_ν

        # Mismatched lengths
        obs = [[1.0, 2.0], [2.0, 3.0], [0.0, 1.0]]
        weights = [1.0, 1.0]
        dist = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)

        # Wrong observation dimension
        obs = [[1.0], [2.0], [3.0]]
        weights = [1.0, 1.0, 1.0]
        dist = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)
    end

    @testset "Integration test" begin
        # Simulate a complete workflow like HMM would use
        rng = Random.MersenneTwister(999)

        # Create true distribution
        true_dist = MultivariateTDiag([2.0, -1.0], [1.5, 1.0], 6.0)

        # Generate observations
        n_obs = 800
        observations = [rand(rng, true_dist) for _ in 1:n_obs]

        # Simulate HMM posterior weights
        weights = rand(rng, n_obs) .+ 0.5

        # Fit new distribution
        fitted = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(fitted, observations, weights; max_iter=40)

        # Check that fitted parameters are reasonable
        @test fitted.ν > 0
        @test all(isfinite, fitted.μ)
        @test all(>(0), fitted.σ²)

        # Generate new samples from fitted distribution
        new_samples = [rand(rng, fitted) for _ in 1:100]
        @test all(length(s) == 2 for s in new_samples)

        # Compute log densities
        log_probs = [logdensityof(fitted, obs) for obs in observations[1:10]]
        @test all(isfinite, log_probs)
    end
end
