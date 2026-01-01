
using Test
using Random
using LinearAlgebra
using Distributions
using DensityInterface
using StableRNGs
using StatsAPI
using EmissionModels

rng = StableRNG(1234)

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

end