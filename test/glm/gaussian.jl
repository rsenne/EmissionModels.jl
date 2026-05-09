
using Test
using Random
using LinearAlgebra
using Distributions
using DensityInterface
using StatsAPI
using EmissionModels

rng = Random.MersenneTwister(1234)

function _synthetic_gaussian_glm(rng, n::Int, p::Int; β::Vector{Float64}, σ2::Float64, weights=:uniform)
    @assert length(β) == p
    X = hcat(ones(n), randn(rng, n, p-1))  # intercept + random features
    μ = X * β
    y = rand.(Ref(rng), Normal.(μ, sqrt(σ2)))
    w = weights === :uniform ? ones(n) :
        weights === :random  ? (rand(rng, n) .+ 0.5) :
        error("weights must be :uniform or :random")
    return X, y, w
end

_expected_logpdf(y, μ, σ2) = -0.5 * log(2π * σ2) - 0.5 * ((y - μ)^2 / σ2)

@testset "GaussianGLM Basic Functionality" begin
        @testset "Constructor" begin
            glm = GaussianGLM([0.5, -1.0], 1.0)
            @test glm.β == [0.5, -1.0]
            @test glm.σ2 == 1.0

            glm2 = GaussianGLM([1, 2], 3)
            @test glm2.β == [1, 2]
            @test glm2.σ2 == 3
        end

        @testset "DensityInterface" begin
            glm = GaussianGLM([0.5, -1.0], 1.0)

            # Trait
            @test DensityKind(glm) == HasDensity()

            X = [1.0 2.0; 1.0 3.0; 1.0 4.0]
            μ = X * glm.β
            y = rand.(Ref(rng), Normal.(μ, sqrt(glm.σ2)))

            # logdensityof matches closed form
            for i in eachindex(y)
                x_i = vec(X[i, :])  # ensure Vector
                logp = logdensityof(glm, y[i]; control_seq=x_i)
                @test isfinite(logp)

                expected = _expected_logpdf(y[i], dot(glm.β, x_i), glm.σ2)
                @test logp ≈ expected rtol=1e-12 atol=0.0
            end

            # Higher density near the mean than far away
            x0 = vec(X[1, :])
            y_at_mean = dot(glm.β, x0)
            y_far = y_at_mean + 10.0
            @test logdensityof(glm, y_at_mean; control_seq=x0) > logdensityof(glm, y_far; control_seq=x0)
        end

        @testset "Random sampling" begin
            glm = GaussianGLM([0.5, -1.0], 2.0) # variance 2.0
            x = [1.0, 3.0]
            n = 10_000
            samples = [rand(rng, glm; control_seq=x) for _ in 1:n]

            @test all(s -> s isa Real, samples)

            # Empirical mean should be close to dot(β,x)
            μ = dot(glm.β, x)
            m = mean(samples)
            @test m ≈ μ atol=0.15

            # Empirical variance should be close to σ2
            v = var(samples)
            @test v ≈ glm.σ2 atol=0.25
        end

        @testset "fit! with uniform weights" begin
            β_true = [0.3, -1.2]          # p=2
            σ2_true = 0.7
            X, y, w = _synthetic_gaussian_glm(rng, 2000, 2; β=β_true, σ2=σ2_true, weights=:uniform)

            glm = GaussianGLM([0.0, 0.0], 1.0)
            fit!(glm, y, w; control_seq=X)

            # Coefficients should be close with enough data
            @test glm.β ≈ β_true atol=0.08
            @test glm.σ2 ≈ σ2_true atol=0.10

            @test all(isfinite, glm.β)
            @test isfinite(glm.σ2)
            @test glm.σ2 > 0
        end

        @testset "fit! with weighted observations" begin
            β_true = [1.0, 0.6]
            σ2_true = 1.5
            X, y, w = _synthetic_gaussian_glm(rng, 2500, 2; β=β_true, σ2=σ2_true, weights=:random)

            glm = GaussianGLM([0.0, 0.0], 1.0)
            fit!(glm, y, w; control_seq=X)

            @test glm.β ≈ β_true atol=0.10
            @test glm.σ2 ≈ σ2_true atol=0.15
        end

        @testset "Constructor with prior" begin
            glm = GaussianGLM([0.0, 0.0], 1.0)
            @test glm.prior isa NoPrior

            glm2 = GaussianGLM([0.0, 0.0], 1.0, RidgePrior(2.0))
            @test glm2.prior isa RidgePrior
            @test glm2.prior.λ == 2.0
        end

        @testset "fit! with RidgePrior shrinks toward zero" begin
            β_true = [3.0, -3.0]
            σ2_true = 1.0
            X, y, w = _synthetic_gaussian_glm(rng, 300, 2; β=β_true, σ2=σ2_true)

            glm_noprior = GaussianGLM([0.0, 0.0], 1.0)
            fit!(glm_noprior, y, w; control_seq=X)

            glm_ridge = GaussianGLM([0.0, 0.0], 1.0, RidgePrior(10.0))
            fit!(glm_ridge, y, w; control_seq=X)

            @test norm(glm_ridge.β) < norm(glm_noprior.β)
            @test all(isfinite, glm_ridge.β)
        end

end

function _synthetic_mvgaussian_glm(rng, n::Int, p::Int, k::Int;
                                   B::Matrix{Float64}, Σ::Matrix{Float64},
                                   weights=:uniform)
    @assert size(B) == (p, k)
    @assert size(Σ) == (k, k)
    X = hcat(ones(n), randn(rng, n, p-1))
    L = cholesky(Σ).L
    obs_seq = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        μ_i = vec(B' * X[i, :])
        obs_seq[i] = μ_i .+ L * randn(rng, k)
    end
    w = weights === :uniform ? ones(n) :
        weights === :random  ? (rand(rng, n) .+ 0.5) :
        error("weights must be :uniform or :random")
    return X, obs_seq, w
end

@testset "MvGaussianGLM" begin
    rng = Random.MersenneTwister(20260507)

    @testset "Constructor" begin
        B = [0.5 -0.3; 1.0 0.2]              # p=2, k=2
        Σ = [1.0 0.2; 0.2 0.8]
        glm = MvGaussianGLM(B, Σ)
        @test glm.B == B
        @test glm.Σ == Σ
        @test glm.in_dim == 2
        @test glm.out_dim == 2
        @test glm.prior isa NoPrior

        glm_float = MvGaussianGLM([1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
        @test glm_float.B == [1.0 0.0; 0.0 1.0]
        @test eltype(glm_float.B) === Float64

        glm_ridge = MvGaussianGLM(B, Σ, RidgePrior(2.0))
        @test glm_ridge.prior isa RidgePrior

        @test_throws DimensionMismatch MvGaussianGLM(B, [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
        @test_throws ArgumentError MvGaussianGLM(B, [1.0 0.0; 0.0 -1.0])
    end

    @testset "DensityInterface" begin
        B = [0.5 -1.0; 1.0 0.5]
        Σ = [1.0 0.3; 0.3 1.5]
        glm = MvGaussianGLM(B, Σ)

        @test DensityKind(glm) == HasDensity()

        x = [1.0, 2.0]
        μ = vec(B' * x)
        y = μ .+ [0.1, -0.2]

        # Compare against MvNormal closed form
        diff = y - μ
        expected = -log(2π) - 0.5 * logdet(Σ) - 0.5 * dot(diff, Σ \ diff)
        @test logdensityof(glm, y; control_seq=x) ≈ expected rtol=1e-10

        # Higher density at the mean than far from it
        @test logdensityof(glm, μ; control_seq=x) >
              logdensityof(glm, μ .+ 10.0; control_seq=x)

        @test_throws DimensionMismatch logdensityof(glm, [1.0]; control_seq=x)
        @test_throws DimensionMismatch logdensityof(glm, y; control_seq=[1.0])
    end

    @testset "Random sampling" begin
        B = [0.5 -1.0; 1.0 0.5]
        Σ = [1.0 0.3; 0.3 1.5]
        glm = MvGaussianGLM(B, Σ)
        x = [1.0, 0.5]

        n = 20_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]
        Y = reduce(hcat, samples)'                # n × k

        μ_true = vec(B' * x)
        m = vec(mean(Y; dims=1))
        @test m ≈ μ_true atol=0.05

        S = (Y .- m')' * (Y .- m') / (n - 1)
        @test S ≈ Σ atol=0.1
    end

    @testset "fit! recovers B and Σ" begin
        B_true = [0.5 -1.0; 1.0 0.5; -0.3 0.8]   # p=3, k=2
        Σ_true = [1.0 0.3; 0.3 1.5]
        X, obs_seq, w = _synthetic_mvgaussian_glm(rng, 4000, 3, 2;
                                                  B=B_true, Σ=Σ_true)

        glm = MvGaussianGLM(zeros(3, 2), Matrix(1.0I, 2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol=0.08
        @test glm.Σ ≈ Σ_true atol=0.10
        @test all(isfinite, glm.B)
        @test all(isfinite, glm.Σ)
        @test isposdef(glm.Σ)
    end

    @testset "fit! random weights" begin
        B_true = [1.0 0.0; -0.5 1.5]
        Σ_true = [0.5 -0.1; -0.1 0.7]
        X, obs_seq, w = _synthetic_mvgaussian_glm(rng, 3000, 2, 2;
                                                  B=B_true, Σ=Σ_true,
                                                  weights=:random)

        glm = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol=0.10
        @test glm.Σ ≈ Σ_true atol=0.12
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        B_true = [3.0 -3.0; 3.0 3.0]
        Σ_true = [1.0 0.0; 0.0 1.0]
        X, obs_seq, w = _synthetic_mvgaussian_glm(rng, 200, 2, 2;
                                                  B=B_true, Σ=Σ_true)

        glm_noprior = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        fit!(glm_noprior, obs_seq, w; control_seq=X)

        glm_ridge = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2), RidgePrior(20.0))
        fit!(glm_ridge, obs_seq, w; control_seq=X)

        @test norm(glm_ridge.B) < norm(glm_noprior.B)
        @test all(isfinite, glm_ridge.B)
    end

    @testset "fit! DimensionMismatch" begin
        glm = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        X = ones(5, 2)
        bad_obs = [zeros(2) for _ in 1:4]
        good_obs = [zeros(2) for _ in 1:5]
        @test_throws DimensionMismatch fit!(glm, bad_obs, ones(5); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(5); control_seq=ones(5, 3))

        wrong_dim_obs = [zeros(3) for _ in 1:5]
        @test_throws DimensionMismatch fit!(glm, wrong_dim_obs, ones(5); control_seq=X)
    end
end