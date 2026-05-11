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

        @test logdensityof(glm, 1; control_seq=x) ≈ log(μ) rtol = 1e-10
        @test logdensityof(glm, 0; control_seq=x) ≈ log(1 - μ) rtol = 1e-10
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
        @test mean(samples) ≈ _sigmoid(dot(glm.β, x)) atol = 0.05
    end

    @testset "fit! uniform weights" begin
        β_true = [0.3, -1.2]
        X, y, w = _synthetic_bernoulli(rng, 2000, β_true)

        glm = BernoulliGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol = 0.15
        @test all(isfinite, glm.β)
    end

    @testset "fit! random weights" begin
        β_true = [1.0, 0.5]
        X, y, w = _synthetic_bernoulli(rng, 2000, β_true; weights=:random)

        glm = BernoulliGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol = 0.20
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
        @test_throws DimensionMismatch fit!(
            glm, ones(Int, 5), ones(5); control_seq=ones(5, 3)
        )
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
            @test logdensityof(glm, k; control_seq=x) ≈ logpdf(Poisson(μ), k) rtol = 1e-10
        end

        @test logdensityof(glm, -1; control_seq=x) == -Inf
    end

    @testset "Random sampling" begin
        glm = PoissonGLM([1.5])
        x = [1.0]
        n = 5_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]

        @test all(s -> s >= 0, samples)
        @test mean(samples) ≈ exp(glm.β[1]) atol = 0.2
    end

    @testset "fit! uniform weights" begin
        β_true = [1.0, 0.4]
        X, y, w = _synthetic_poisson(rng, 2000, β_true)

        glm = PoissonGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol = 0.15
        @test all(isfinite, glm.β)
    end

    @testset "fit! random weights" begin
        β_true = [0.5, -0.3]
        X, y, w = _synthetic_poisson(rng, 2000, β_true; weights=:random)

        glm = PoissonGLM(zeros(2))
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol = 0.15
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
        @test_throws DimensionMismatch fit!(
            glm, ones(Int, 5), ones(5); control_seq=ones(5, 3)
        )
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

function _synthetic_mvbernoulli(rng, n, B_true; weights=:uniform)
    p, k = size(B_true)
    X = hcat(ones(n), randn(rng, n, p - 1))
    obs_seq = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        obs_seq[i] = Int[
            rand(rng) < _sigmoid(dot(B_true[:, j], X[i, :])) ? 1 : 0 for j in 1:k
        ]
    end
    w = weights === :uniform ? ones(n) : rand(rng, n) .+ 0.5
    return X, obs_seq, w
end

function _synthetic_mvpoisson(rng, n, B_true; weights=:uniform)
    p, k = size(B_true)
    X = hcat(ones(n), randn(rng, n, p - 1))
    obs_seq = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        obs_seq[i] = Int[rand(rng, Poisson(exp(dot(B_true[:, j], X[i, :])))) for j in 1:k]
    end
    w = weights === :uniform ? ones(n) : rand(rng, n) .+ 0.5
    return X, obs_seq, w
end

@testset "MvBernoulliGLM" begin
    rng = Random.MersenneTwister(123)

    @testset "Constructor" begin
        B = [0.5 -1.0; 1.0 0.5]
        glm = MvBernoulliGLM(B)
        @test glm.B == B
        @test glm.in_dim == 2
        @test glm.out_dim == 2
        @test glm.prior isa NoPrior

        glm_float = MvBernoulliGLM([1.0 0.0; 0.0 1.0])
        @test eltype(glm_float.B) === Float64

        glm_ridge = MvBernoulliGLM(B, RidgePrior(1.0))
        @test glm_ridge.prior isa RidgePrior
    end

    @testset "DensityInterface" begin
        B = [0.0 1.0; 1.0 -0.5]
        glm = MvBernoulliGLM(B)
        @test DensityKind(glm) == HasDensity()

        x = [1.0, 0.5]
        η1 = dot(B[:, 1], x)
        η2 = dot(B[:, 2], x)
        μ1, μ2 = _sigmoid(η1), _sigmoid(η2)

        @test logdensityof(glm, [1, 1]; control_seq=x) ≈ log(μ1) + log(μ2) rtol = 1e-10
        @test logdensityof(glm, [1, 0]; control_seq=x) ≈ log(μ1) + log(1 - μ2) rtol = 1e-10
        @test logdensityof(glm, [0, 1]; control_seq=x) ≈ log(1 - μ1) + log(μ2) rtol = 1e-10
        @test logdensityof(glm, [0, 0]; control_seq=x) ≈ log(1 - μ1) + log(1 - μ2) rtol =
            1e-10
        @test logdensityof(glm, [2, 0]; control_seq=x) == -Inf
    end

    @testset "logdensityof large η (numerical stability)" begin
        glm = MvBernoulliGLM(reshape([50.0, -50.0], 1, 2))
        x = [1.0]
        @test isfinite(logdensityof(glm, [1, 1]; control_seq=x))
        @test isfinite(logdensityof(glm, [0, 0]; control_seq=x))
    end

    @testset "Random sampling" begin
        B = [0.0 0.0; 2.0 -2.0]
        glm = MvBernoulliGLM(B)
        x = [1.0, 0.5]

        n = 5_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]
        Y = reduce(hcat, samples)'   # n × k

        @test all(s -> s == 0 || s == 1, vec(Y))
        @test mean(Y[:, 1]) ≈ _sigmoid(dot(B[:, 1], x)) atol = 0.05
        @test mean(Y[:, 2]) ≈ _sigmoid(dot(B[:, 2], x)) atol = 0.05
    end

    @testset "fit! recovers B" begin
        B_true = [0.3 -1.0; -1.2 0.8]
        X, obs_seq, w = _synthetic_mvbernoulli(rng, 5000, B_true)

        glm = MvBernoulliGLM(zeros(2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol = 0.20
        @test all(isfinite, glm.B)
    end

    @testset "fit! random weights" begin
        B_true = [1.0 -0.5; 0.5 1.0]
        X, obs_seq, w = _synthetic_mvbernoulli(rng, 5000, B_true; weights=:random)

        glm = MvBernoulliGLM(zeros(2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol = 0.25
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        B_true = [2.0 -2.0; -2.0 2.0]
        X, obs_seq, w = _synthetic_mvbernoulli(rng, 500, B_true)

        glm_noprior = MvBernoulliGLM(zeros(2, 2))
        fit!(glm_noprior, obs_seq, w; control_seq=X)

        glm_ridge = MvBernoulliGLM(zeros(2, 2), RidgePrior(10.0))
        fit!(glm_ridge, obs_seq, w; control_seq=X)

        @test norm(glm_ridge.B) < norm(glm_noprior.B)
        @test all(isfinite, glm_ridge.B)
    end

    @testset "matches independent BernoulliGLM fits" begin
        B_true = [0.4 -0.7; -0.9 0.3]
        X, obs_seq, w = _synthetic_mvbernoulli(rng, 1000, B_true)

        glm = MvBernoulliGLM(zeros(2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        for j in 1:2
            y_j = [obs_seq[i][j] for i in eachindex(obs_seq)]
            glm_j = BernoulliGLM(zeros(2))
            fit!(glm_j, y_j, w; control_seq=X)
            @test glm.B[:, j] ≈ glm_j.β rtol = 1e-6
        end
    end

    @testset "fit! DimensionMismatch" begin
        glm = MvBernoulliGLM(zeros(2, 2))
        X = ones(5, 2)
        good_obs = [zeros(Int, 2) for _ in 1:5]
        @test_throws DimensionMismatch fit!(
            glm, [zeros(Int, 2) for _ in 1:4], ones(5); control_seq=X
        )
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(5); control_seq=ones(5, 3))

        wrong_dim_obs = [zeros(Int, 3) for _ in 1:5]
        @test_throws DimensionMismatch fit!(glm, wrong_dim_obs, ones(5); control_seq=X)
    end
end

@testset "MvPoissonGLM" begin
    rng = Random.MersenneTwister(456)

    @testset "Constructor" begin
        B = [1.0 0.5; -0.3 0.2]
        glm = MvPoissonGLM(B)
        @test glm.B == B
        @test glm.in_dim == 2
        @test glm.out_dim == 2
        @test glm.prior isa NoPrior

        glm_float = MvPoissonGLM([1.0 0.0; 0.0 1.0])
        @test eltype(glm_float.B) === Float64

        glm_ridge = MvPoissonGLM(B, RidgePrior(0.5))
        @test glm_ridge.prior isa RidgePrior
    end

    @testset "DensityInterface" begin
        B = [0.5 0.0; 0.2 -0.5]
        glm = MvPoissonGLM(B)
        @test DensityKind(glm) == HasDensity()

        x = [1.0, 1.0]
        μ1 = exp(dot(B[:, 1], x))
        μ2 = exp(dot(B[:, 2], x))

        for k1 in 0:3, k2 in 0:3
            expected = logpdf(Poisson(μ1), k1) + logpdf(Poisson(μ2), k2)
            @test logdensityof(glm, [k1, k2]; control_seq=x) ≈ expected rtol = 1e-10
        end

        @test logdensityof(glm, [-1, 0]; control_seq=x) == -Inf
        @test logdensityof(glm, [0, -1]; control_seq=x) == -Inf
    end

    @testset "Random sampling" begin
        B = reshape([1.5, 0.5], 1, 2)
        glm = MvPoissonGLM(B)
        x = [1.0]

        n = 5_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]
        Y = reduce(hcat, samples)'

        @test all(s -> s >= 0, vec(Y))
        @test mean(Y[:, 1]) ≈ exp(B[1, 1]) atol = 0.2
        @test mean(Y[:, 2]) ≈ exp(B[1, 2]) atol = 0.2
    end

    @testset "fit! recovers B" begin
        B_true = [1.0 0.5; 0.4 -0.3]
        X, obs_seq, w = _synthetic_mvpoisson(rng, 3000, B_true)

        glm = MvPoissonGLM(zeros(2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol = 0.15
        @test all(isfinite, glm.B)
    end

    @testset "fit! random weights" begin
        B_true = [0.5 0.0; -0.3 0.4]
        X, obs_seq, w = _synthetic_mvpoisson(rng, 3000, B_true; weights=:random)

        glm = MvPoissonGLM(zeros(2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol = 0.20
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        B_true = [2.0 -1.5; -1.0 1.5]
        X, obs_seq, w = _synthetic_mvpoisson(rng, 500, B_true)

        glm_noprior = MvPoissonGLM(zeros(2, 2))
        fit!(glm_noprior, obs_seq, w; control_seq=X)

        glm_ridge = MvPoissonGLM(zeros(2, 2), RidgePrior(10.0))
        fit!(glm_ridge, obs_seq, w; control_seq=X)

        @test norm(glm_ridge.B) < norm(glm_noprior.B)
        @test all(isfinite, glm_ridge.B)
    end

    @testset "matches independent PoissonGLM fits" begin
        B_true = [0.5 -0.3; 0.4 0.2]
        X, obs_seq, w = _synthetic_mvpoisson(rng, 1000, B_true)

        glm = MvPoissonGLM(zeros(2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        for j in 1:2
            y_j = [obs_seq[i][j] for i in eachindex(obs_seq)]
            glm_j = PoissonGLM(zeros(2))
            fit!(glm_j, y_j, w; control_seq=X)
            @test glm.B[:, j] ≈ glm_j.β rtol = 1e-6
        end
    end

    @testset "fit! DimensionMismatch" begin
        glm = MvPoissonGLM(zeros(2, 2))
        X = ones(5, 2)
        good_obs = [zeros(Int, 2) for _ in 1:5]
        @test_throws DimensionMismatch fit!(
            glm, [zeros(Int, 2) for _ in 1:4], ones(5); control_seq=X
        )
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(5); control_seq=ones(5, 3))

        wrong_dim_obs = [zeros(Int, 3) for _ in 1:5]
        @test_throws DimensionMismatch fit!(glm, wrong_dim_obs, ones(5); control_seq=X)
    end
end
