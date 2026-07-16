using EmissionModels
using EmissionModelsTest
using Test
using Random
using DensityInterface
using StatsAPI
using SpecialFunctions: loggamma
using Statistics: mean, cov, var
using LinearAlgebra
using HiddenMarkovModels

import StatsAPI: fit!

@testset "MvT" begin
    @testset "Constructor" begin
        # Valid construction
        μ = [0.0, 0.0]
        Σ = [1.0 0.5; 0.5 1.0]
        ν = 5.0
        dist = MvT(μ, Σ, ν)
        @test dist.μ == μ
        @test dist.Σ == Σ
        @test dist.ν == ν
        @test dist.dim == 2

        # Type promotion
        dist2 = MvT([0, 0], [1.0 0.0; 0.0 1.0], 5)
        @test eltype(dist2.μ) == Float64
        @test eltype(dist2.Σ) == Float64
        @test dist2.ν isa Float64

        # Invalid parameters
        @test_throws ArgumentError MvT([0.0, 0.0], [1.0 0.5; 0.5 1.0], -1.0)  # negative ν
        @test_throws ArgumentError MvT(Float64[], [1.0;;], 5.0)  # empty μ
        @test_throws DimensionMismatch MvT([0.0, 0.0], [1.0;;], 5.0)  # wrong Σ size
        @test_throws ArgumentError MvT([0.0, 0.0], [1.0 0.5; 0.5 -1.0], 5.0)  # non-PD Σ
    end

    @testset "DensityInterface" begin
        μ = [1.0, 2.0]
        Σ = [2.0 0.5; 0.5 1.0]
        ν = 5.0
        dist = MvT(μ, Σ, ν)

        @test DensityKind(dist) == HasDensity()

        logp_mean = logdensityof(dist, μ)
        @test isfinite(logp_mean)

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

        # The density is symmetric when the scale is.
        μ_sym = [0.0, 0.0]
        Σ_sym = [1.0 0.0; 0.0 1.0]
        dist_sym = MvT(μ_sym, Σ_sym, ν)
        @test logdensityof(dist_sym, [1.0, 0.0]) ≈ logdensityof(dist_sym, [-1.0, 0.0])

        @test_throws DimensionMismatch logdensityof(dist, [1.0])
        @test_throws DimensionMismatch logdensityof(dist, [1.0, 2.0, 3.0])
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        μ = [2.0, -1.0]
        Σ = [2.0 0.8; 0.8 1.5]
        ν = 10.0
        dist = MvT(μ, Σ, ν)

        n_samples = 5000
        samples = [rand(rng, dist) for _ in 1:n_samples]
        @test all(length(s) == 2 for s in samples)

        sample_matrix = hcat(samples...)'

        # The empirical mean should be close to μ (defined for ν > 1).
        empirical_mean = vec(mean(sample_matrix; dims=1))
        @test empirical_mean ≈ μ atol = 0.2

        # The empirical covariance should be close to Σ * ν/(ν-2) (ν > 2).
        expected_cov = Σ * (ν / (ν - 2))
        empirical_cov = cov(sample_matrix)
        @test isapprox(empirical_cov, expected_cov, atol=0.5)

        # Low degrees of freedom (heavier tails) should still sample cleanly.
        dist_heavy = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 3.0)
        samples_heavy = [rand(rng, dist_heavy) for _ in 1:1000]
        @test all(length(s) == 2 for s in samples_heavy)
    end

    @testset "fit! with uniform weights" begin
        rng = Random.MersenneTwister(123)
        true_μ = [1.0, -0.5]
        true_Σ = [2.0 0.3; 0.3 1.0]
        true_ν = 8.0
        true_dist = MvT(true_μ, true_Σ, true_ν)

        n = 2000
        obs = [rand(rng, true_dist) for _ in 1:n]
        weights = ones(n)

        fitted_dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        fit!(fitted_dist, obs, weights; max_iter=50)

        # Parameters should be close to true values
        @test fitted_dist.μ ≈ true_μ atol = 0.2
        @test fitted_dist.Σ ≈ true_Σ atol = 0.3
        @test fitted_dist.ν ≈ true_ν atol = 2.0
    end

    @testset "fit! with weighted observations" begin
        rng = Random.MersenneTwister(456)

        n = 500
        obs = [randn(rng, 2) for _ in 1:n]
        weights = rand(rng, n) .+ 0.5

        dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
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
        dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], fixed_ν)
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
        dist = MvT([1.0, 2.0], [1.0 0.0; 0.0 1.0], 5.0)
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
        dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
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
        dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)

        # Wrong observation dimension
        obs = [[1.0], [2.0], [3.0]]
        weights = [1.0, 1.0, 1.0]
        dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)
    end

    @testset "Integration test" begin
        # Simulate the full workflow an HMM fit would drive.
        rng = Random.MersenneTwister(999)

        true_dist = MvT([2.0, -1.0], [1.5 0.4; 0.4 1.0], 6.0)

        n_obs = 800
        observations = [rand(rng, true_dist) for _ in 1:n_obs]

        # Simulate HMM posterior weights
        weights = rand(rng, n_obs) .+ 0.5

        fitted = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
        fit!(fitted, observations, weights; max_iter=40)

        # Fitted parameters should be sane.
        @test fitted.ν > 0
        @test all(isfinite, fitted.μ)
        @test isposdef(fitted.Σ)

        new_samples = [rand(rng, fitted) for _ in 1:100]
        @test all(length(s) == 2 for s in new_samples)

        log_probs = [logdensityof(fitted, obs) for obs in observations[1:10]]
        @test all(isfinite, log_probs)
    end

    @testset "HMM Integration" begin
        rng = Random.MersenneTwister(888)

        hmm = create_hmm(MvT; n_states=3, α=12.0, dim=2, rng=rng)
        @test length(hmm.init) == 3
        @test all(dist -> dist isa MvT, hmm.dists)
        @test all(dist -> dist.dim == 2, hmm.dists)

        _, obs_seq = test_hmm_integration(rng, hmm; T=100)
        @test all(obs -> length(obs) == 2, obs_seq)

        # Again with custom parameters.
        hmm_custom = create_hmm(MvT; n_states=4, dim=3, α=8.0, ν_range=(4.0, 12.0), rng=rng)
        @test length(hmm_custom.init) == 4
        @test all(dist -> dist.dim == 3, hmm_custom.dists)
        @test all(4.0 <= dist.ν <= 12.0 for dist in hmm_custom.dists)
    end
end

@testset "MvTDiag" begin
    @testset "Constructor" begin
        # Valid construction
        μ = [0.0, 0.0]
        σ² = [1.0, 2.0]
        ν = 5.0
        dist = MvTDiag(μ, σ², ν)
        @test dist.μ == μ
        @test dist.σ² == σ²
        @test dist.ν == ν
        @test dist.dim == 2

        # Type promotion
        dist2 = MvTDiag([0, 0], [1.0, 2.0], 5)
        @test eltype(dist2.μ) == Float64
        @test eltype(dist2.σ²) == Float64
        @test dist2.ν isa Float64

        # Invalid parameters
        @test_throws ArgumentError MvTDiag([0.0, 0.0], [1.0, 2.0], -1.0)  # negative ν
        @test_throws ArgumentError MvTDiag(Float64[], [1.0], 5.0)  # empty μ
        @test_throws DimensionMismatch MvTDiag([0.0, 0.0], [1.0], 5.0)  # wrong σ² length
        @test_throws ArgumentError MvTDiag([0.0], [-1.0], 5.0)  # negative variance
        @test_throws ArgumentError MvTDiag([0.0], [0.0], 5.0)  # zero variance
    end

    @testset "DensityInterface" begin
        μ = [1.0, 2.0]
        σ² = [2.0, 1.0]
        ν = 5.0
        dist = MvTDiag(μ, σ², ν)

        @test DensityKind(dist) == HasDensity()

        logp_mean = logdensityof(dist, μ)
        @test isfinite(logp_mean)

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

        @test_throws DimensionMismatch logdensityof(dist, [1.0])
        @test_throws DimensionMismatch logdensityof(dist, [1.0, 2.0, 3.0])
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        μ = [2.0, -1.0]
        σ² = [2.0, 1.5]
        ν = 10.0
        dist = MvTDiag(μ, σ², ν)

        n_samples = 5000
        samples = [rand(rng, dist) for _ in 1:n_samples]
        @test all(length(s) == 2 for s in samples)

        sample_matrix = hcat(samples...)'

        # Check empirical mean
        empirical_mean = vec(mean(sample_matrix; dims=1))
        @test empirical_mean ≈ μ atol = 0.2

        # Check empirical variances (diagonal only)
        expected_var = σ² * (ν / (ν - 2))
        empirical_var = vec(var(sample_matrix; dims=1))
        @test isapprox(empirical_var, expected_var, atol=0.5)
    end

    @testset "fit! with uniform weights" begin
        rng = Random.MersenneTwister(123)
        true_μ = [1.0, -0.5]
        true_σ² = [2.0, 1.0]
        true_ν = 8.0
        true_dist = MvTDiag(true_μ, true_σ², true_ν)

        n = 2000
        obs = [rand(rng, true_dist) for _ in 1:n]
        weights = ones(n)

        fitted_dist = MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(fitted_dist, obs, weights; max_iter=50)

        # Parameters should be close to true values
        @test fitted_dist.μ ≈ true_μ atol = 0.2
        @test fitted_dist.σ² ≈ true_σ² atol = 0.3
        @test fitted_dist.ν ≈ true_ν atol = 2.0
    end

    @testset "fit! with weighted observations" begin
        rng = Random.MersenneTwister(456)

        n = 500
        obs = [randn(rng, 2) for _ in 1:n]
        weights = rand(rng, n) .+ 0.5

        dist = MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
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
        dist = MvTDiag([0.0, 0.0], [1.0, 1.0], fixed_ν)
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
        dist = MvTDiag([1.0, 2.0], [1.0, 1.0], 5.0)
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
        dist = MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
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
        dist = MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)

        # Wrong observation dimension
        obs = [[1.0], [2.0], [3.0]]
        weights = [1.0, 1.0, 1.0]
        dist = MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        @test_throws DimensionMismatch fit!(dist, obs, weights)
    end

    @testset "Integration test" begin
        # Simulate the full workflow an HMM fit would drive.
        rng = Random.MersenneTwister(999)

        true_dist = MvTDiag([2.0, -1.0], [1.5, 1.0], 6.0)

        n_obs = 800
        observations = [rand(rng, true_dist) for _ in 1:n_obs]

        # Simulate HMM posterior weights
        weights = rand(rng, n_obs) .+ 0.5

        fitted = MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(fitted, observations, weights; max_iter=40)

        # Fitted parameters should be sane.
        @test fitted.ν > 0
        @test all(isfinite, fitted.μ)
        @test all(>(0), fitted.σ²)

        new_samples = [rand(rng, fitted) for _ in 1:100]
        @test all(length(s) == 2 for s in new_samples)

        log_probs = [logdensityof(fitted, obs) for obs in observations[1:10]]
        @test all(isfinite, log_probs)
    end

    @testset "HMM Integration" begin
        rng = Random.MersenneTwister(999)

        hmm = create_hmm(MvTDiag; n_states=3, α=10.0, dim=2, rng=rng)
        @test length(hmm.init) == 3
        @test all(dist -> dist isa MvTDiag, hmm.dists)
        @test all(dist -> dist.dim == 2, hmm.dists)

        _, obs_seq = test_hmm_integration(rng, hmm; T=100)
        @test all(obs -> length(obs) == 2, obs_seq)

        # Again with custom parameters.
        hmm_custom = create_hmm(
            MvTDiag;
            n_states=4,
            dim=3,
            α=8.0,
            ν_range=(5.0, 15.0),
            σ²_range=(1.0, 3.0),
            rng=rng,
        )
        @test length(hmm_custom.init) == 4
        @test all(dist -> dist.dim == 3, hmm_custom.dists)
        @test all(5.0 <= dist.ν <= 15.0 for dist in hmm_custom.dists)
        @test all(all(1.0 .<= dist.σ² .<= 3.0) for dist in hmm_custom.dists)
    end
end

@testset "Constructor does not alias caller arrays" begin
    μ = [0.0, 0.0]
    Σ = [1.0 0.3; 0.3 1.0]
    dist = MvT(μ, Σ, 5.0)
    lp_before = logdensityof(dist, [0.1, 0.2])
    # Mutating the caller's arrays must not affect the (cached) distribution.
    Σ[1, 1] = 100.0
    μ[1] = 50.0
    @test dist.Σ[1, 1] == 1.0
    @test dist.μ[1] == 0.0
    @test logdensityof(dist, [0.1, 0.2]) == lp_before

    σ² = [1.0, 2.0]
    μd = [0.0, 0.0]
    distd = MvTDiag(μd, σ², 5.0)
    lp_before_d = logdensityof(distd, [0.1, 0.2])
    σ²[1] = 100.0
    μd[1] = 50.0
    @test distd.σ²[1] == 1.0
    @test logdensityof(distd, [0.1, 0.2]) == lp_before_d
end

@testset "Array sampling API" begin
    rng = Random.MersenneTwister(42)
    dist = MvT([0.0, 0.0], [1.0 0.3; 0.3 1.0], 5.0)
    samples = rand(rng, dist, 10)
    @test samples isa Vector{Vector{Float64}}
    @test length(samples) == 10
    @test all(length(s) == 2 for s in samples)

    distd = MvTDiag([0.0, 0.0], [1.0, 2.0], 5.0)
    samples_d = rand(rng, distd, 10)
    @test samples_d isa Vector{Vector{Float64}}
    @test length(samples_d) == 10
end

@testset "Float32 type stability" begin
    dist = MvT(Float32[0, 0], Float32[1 0; 0 1], 5.0f0)
    @test logdensityof(dist, Float32[0.1, 0.2]) isa Float32

    distd = MvTDiag(Float32[0, 0], Float32[1, 1], 5.0f0)
    @test logdensityof(distd, Float32[0.1, 0.2]) isa Float32
end

@testset "ν M-step stays finite on near-Gaussian data" begin
    # Gaussian observations push the ECME ν update toward ∞; the bracketed
    # solver must return a finite (capped) ν instead of overflowing.
    rng = Random.MersenneTwister(202)
    obs = [randn(rng, 2) for _ in 1:2000]
    w = ones(2000)
    dist = MvT([0.0, 0.0], [1.0 0.0; 0.0 1.0], 5.0)
    fit!(dist, obs, w; max_iter=100)
    @test isfinite(dist.ν)
    @test dist.ν <= 1.0e6
    @test all(isfinite, dist.μ)
    @test all(isfinite, dist.Σ)
end

@testset "rand! in-place sampling" begin
    rng = Random.MersenneTwister(303)
    dist = MvT([2.0, -1.0], [2.0 0.8; 0.8 1.5], 10.0)
    out = zeros(2)
    @test rand!(rng, dist, out) === out
    @test_throws DimensionMismatch rand!(rng, dist, zeros(3))
    S = reduce(hcat, (copy(rand!(rng, dist, out)) for _ in 1:20_000))
    @test vec(mean(S; dims=2)) ≈ dist.μ atol = 0.1

    distd = MvTDiag([1.0, -2.0], [1.0, 2.0], 8.0)
    @test rand!(rng, distd, out) === out
    @test_throws DimensionMismatch rand!(rng, distd, zeros(3))
    Sd = reduce(hcat, (copy(rand!(rng, distd, out)) for _ in 1:20_000))
    @test vec(mean(Sd; dims=2)) ≈ distd.μ atol = 0.1
end

@testset "fit! works for non-BLAS eltypes (BigFloat)" begin
    #= The struct accepts any T<:Real, but the scatter M-step used BLAS.ger!,
       which only exists for Float32/Float64, so fit! crashed for e.g.
       BigFloat. The generic rank-1 fallback must fit end to end. ν stays
       fixed: the ν root-find needs digamma/trigamma, whose BigFloat support
       belongs to SpecialFunctions, not this code path. =#
    rng = Random.MersenneTwister(404)
    obs = [BigFloat.(randn(rng, 2) .+ [1.0, -1.0]) for _ in 1:200]
    w = ones(200)
    dist = MvT(BigFloat[0.0, 0.0], BigFloat[1.0 0.0; 0.0 1.0], BigFloat(5.0))
    fit!(dist, obs, w; max_iter=5, fix_nu=true)
    @test all(isfinite, dist.μ)
    @test isapprox(Float64.(dist.μ), [1.0, -1.0]; atol=0.3)
    @test all(isfinite, dist.Σ)
end
