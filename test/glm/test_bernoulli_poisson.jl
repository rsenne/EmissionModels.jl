using EmissionModels
using Test
using Random
using DensityInterface
using StatsAPI
using LinearAlgebra
using Distributions: Bernoulli, Poisson, logpdf


function _sigmoid(η)
    return one(η) / (one(η) + exp(-η))
end

function _synthetic_bernoulli(rng, n, β_true; weights=:uniform)
    p = length(β_true)
    X = hcat(ones(n), randn(rng, n, p - 1))
    η = X * β_true
    y = Int[rand(rng) < _sigmoid(η[i]) ? 1 : 0 for i in 1:n]
    w = weights === :uniform ? ones(n) : rand(rng, n) .+ 0.5
    return X, y, w
end

function _synthetic_poisson(rng, n, β_true; weights=:uniform)
    p = length(β_true)
    X = hcat(ones(n), randn(rng, n, p - 1))
    η = X * β_true
    y = Int[rand(rng, Poisson(exp(η[i]))) for i in 1:n]
    w = weights === :uniform ? ones(n) : rand(rng, n) .+ 0.5
    return X, y, w
end

@testset "AbstractPrior" begin
    @testset "NoPrior" begin
        prior = NoPrior()
        β = [1.0, -2.0, 0.5]
        g = zeros(3)
        H = zeros(3, 3)

        @test neglogprior(prior, β) == 0.0
        neglogprior_grad!(prior, g, β)
        @test g == zeros(3)
        neglogprior_hess!(prior, H, β)
        @test H == zeros(3, 3)
    end

    @testset "RidgePrior" begin
        prior = RidgePrior(2.0)
        β = [1.0, -1.0]
        g = zeros(2)
        H = zeros(2, 2)

        @test neglogprior(prior, β) ≈ 0.5 * 2.0 * dot(β, β)

        neglogprior_grad!(prior, g, β)
        @test g ≈ 2.0 .* β

        neglogprior_hess!(prior, H, β)
        @test H ≈ 2.0 * I(2)
    end

    @testset "RidgePrior accumulates into existing g and H" begin
        prior = RidgePrior(1.0)
        β = [2.0, 3.0]
        g = [10.0, 20.0]
        H = [1.0 0.0; 0.0 1.0]

        neglogprior_grad!(prior, g, β)
        @test g ≈ [12.0, 23.0]

        neglogprior_hess!(prior, H, β)
        @test H ≈ [2.0 0.0; 0.0 2.0]
    end
end

@testset "BernoulliGLM" begin
    rng = Random.MersenneTwister(42)

    @testset "Constructor" begin
        glm = BernoulliGLM([0.5, -1.0])
        @test glm.β == [0.5, -1.0]
        @test glm.prior isa NoPrior

        glm2 = BernoulliGLM([0.5, -1.0], RidgePrior(1.0))
        @test glm2.prior isa RidgePrior
    end

    @testset "DensityInterface" begin
        glm = BernoulliGLM([0.0, 1.0])
        @test DensityKind(glm) == HasDensity()

        x = [1.0, 0.5]
        η = dot(glm.β, x)
        μ = _sigmoid(η)

        @test logdensityof(glm, 1; control_seq=x) ≈ log(μ) rtol=1e-10
        @test logdensityof(glm, 0; control_seq=x) ≈ log(1 - μ) rtol=1e-10
        @test logdensityof(glm, 2; control_seq=x) == -Inf
        @test logdensityof(glm, -1; control_seq=x) == -Inf
    end

    @testset "logdensityof large η (numerical stability)" begin
        glm = BernoulliGLM([50.0])
        x = [1.0]
        @test isfinite(logdensityof(glm, 1; control_seq=x))
        @test isfinite(logdensityof(glm, 0; control_seq=x))

        glm2 = BernoulliGLM([-50.0])
        @test isfinite(logdensityof(glm2, 1; control_seq=x))
        @test isfinite(logdensityof(glm2, 0; control_seq=x))
    end

    @testset "Random sampling" begin
        glm = BernoulliGLM([0.0, 2.0])
        x = [1.0, 0.0]
        n = 5_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]

        @test all(s -> s == 0 || s == 1, samples)
        @test mean(samples) ≈ _sigmoid(dot(glm.β, x)) atol=0.05
    end

    @testset "fit! uniform weights" begin
        β_true = [0.3, -1.2]
        X, y, w = _synthetic_bernoulli(rng, 2000, β_true)

        glm = BernoulliGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol=0.15
        @test all(isfinite, glm.β)
    end

    @testset "fit! random weights" begin
        β_true = [1.0, 0.5]
        X, y, w = _synthetic_bernoulli(rng, 2000, β_true; weights=:random)

        glm = BernoulliGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol=0.20
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        β_true = [2.0, -2.0]
        X, y, w = _synthetic_bernoulli(rng, 500, β_true)

        glm_noprior = BernoulliGLM(zeros(2))
        fit!(glm_noprior, y, w; control_seq=X)

        glm_ridge = BernoulliGLM(zeros(2), RidgePrior(10.0))
        fit!(glm_ridge, y, w; control_seq=X)

        # Ridge should shrink coefficients toward zero
        @test norm(glm_ridge.β) < norm(glm_noprior.β)
        @test all(isfinite, glm_ridge.β)
    end

    @testset "fit! DimensionMismatch" begin
        glm = BernoulliGLM(zeros(2))
        X = ones(5, 2)
        @test_throws DimensionMismatch fit!(glm, ones(Int, 4), ones(5); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, ones(Int, 5), ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, ones(Int, 5), ones(5); control_seq=ones(5, 3))
    end

    @testset "fit! all-zero observations" begin
        X = hcat(ones(50), randn(rng, 50))
        y = zeros(Int, 50)
        w = ones(50)

        glm = BernoulliGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        # All y=0 ⟹ intercept should be strongly negative
        @test glm.β[1] < -2.0
        @test all(isfinite, glm.β)
    end
end


@testset "PoissonGLM" begin
    rng = Random.MersenneTwister(99)

    @testset "Constructor" begin
        glm = PoissonGLM([1.0, 0.5])
        @test glm.β == [1.0, 0.5]
        @test glm.prior isa NoPrior

        glm2 = PoissonGLM([1.0, 0.5], RidgePrior(0.5))
        @test glm2.prior isa RidgePrior
    end

    @testset "DensityInterface" begin
        glm = PoissonGLM([0.5, 0.2])
        @test DensityKind(glm) == HasDensity()

        x = [1.0, 1.0]
        η = dot(glm.β, x)
        μ = exp(η)

        for k in 0:5
            @test logdensityof(glm, k; control_seq=x) ≈ logpdf(Poisson(μ), k) rtol=1e-10
        end

        @test logdensityof(glm, -1; control_seq=x) == -Inf
    end

    @testset "Random sampling" begin
        glm = PoissonGLM([1.5])
        x = [1.0]
        n = 5_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]

        @test all(s -> s >= 0, samples)
        @test mean(samples) ≈ exp(glm.β[1]) atol=0.2
    end

    @testset "fit! uniform weights" begin
        β_true = [1.0, 0.4]
        X, y, w = _synthetic_poisson(rng, 2000, β_true)

        glm = PoissonGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol=0.15
        @test all(isfinite, glm.β)
    end

    @testset "fit! random weights" begin
        β_true = [0.5, -0.3]
        X, y, w = _synthetic_poisson(rng, 2000, β_true; weights=:random)

        glm = PoissonGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol=0.15
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        β_true = [2.0, -1.5]
        X, y, w = _synthetic_poisson(rng, 500, β_true)

        glm_noprior = PoissonGLM(zeros(2))
        fit!(glm_noprior, y, w; control_seq=X)

        glm_ridge = PoissonGLM(zeros(2), RidgePrior(10.0))
        fit!(glm_ridge, y, w; control_seq=X)

        @test norm(glm_ridge.β) < norm(glm_noprior.β)
        @test all(isfinite, glm_ridge.β)
    end

    @testset "fit! DimensionMismatch" begin
        glm = PoissonGLM(zeros(2))
        X = ones(5, 2)
        @test_throws DimensionMismatch fit!(glm, ones(Int, 4), ones(5); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, ones(Int, 5), ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, ones(Int, 5), ones(5); control_seq=ones(5, 3))
    end

    @testset "fit! all-zero counts" begin
        X = hcat(ones(50), randn(rng, 50))
        y = zeros(Int, 50)
        w = ones(50)

        glm = PoissonGLM([0.0, 0.0])
        fit!(glm, y, w; control_seq=X)

        # All y=0 ⟹ intercept (log mean) should be strongly negative
        @test glm.β[1] < -1.0
        @test all(isfinite, glm.β)
    end
end
