using EmissionModels
using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: ControlledEmission, ControlledEmissionHMM, baum_welch, forward
using DensityInterface
using StatsAPI
using Random
using LinearAlgebra
using Test

#= GLMs must subtype `ControlledEmission` so a `Vector` of them is a valid
   `dists` for a `ControlledEmissionHMM`, and the control-aware positional
   interface (`logdensityof(d, obs, control)`, `rand(rng, d, control)`,
   `fit!(d, obs_seq, control_seq, weights)`) must drive inference and learning. =#

@testset "GLMs as ControlledEmissionHMM emissions" begin
    @testset "subtype relationship" begin
        for G in (
            GaussianGLM,
            BernoulliGLM,
            PoissonGLM,
            MultinomialGLM,
            MvGaussianGLM,
            MvBernoulliGLM,
            MvPoissonGLM,
        )
            @test G <: ControlledEmission
        end
    end

    @testset "positional control-aware interface delegates to keyword methods" begin
        rng = MersenneTwister(0)
        p = 3
        x = vcat(1.0, randn(rng, p - 1))

        pg = PoissonGLM(randn(rng, p) .* 0.2)
        @test logdensityof(pg, 2, x) == logdensityof(pg, 2; control_seq=x)

        gg = GaussianGLM(randn(rng, p), 1.5)
        @test logdensityof(gg, 0.7, x) == logdensityof(gg, 0.7; control_seq=x)

        # rand with a fixed rng must match the keyword path
        @test rand(MersenneTwister(7), pg, x) == rand(MersenneTwister(7), pg; control_seq=x)
    end

    @testset "fit! via vector-of-vectors control_seq matches matrix fit!" begin
        rng = MersenneTwister(1)
        n, p = 200, 3
        X = [vcat(1.0, randn(rng, p - 1)) for _ in 1:n]
        Xmat = permutedims(reduce(hcat, X))           # n×p matrix form
        β_true = [0.5, -0.8, 0.3]
        y = [rand(rng, Distributions.Poisson(exp(dot(β_true, X[i])))) for i in 1:n]
        w = ones(n)

        g_pos = PoissonGLM(zeros(p))
        g_kw = PoissonGLM(zeros(p))
        fit!(g_pos, y, X, w)                          # positional ControlledEmission path
        fit!(g_kw, y, w; control_seq=Xmat)            # keyword matrix path
        @test g_pos.β ≈ g_kw.β rtol = 1e-8
    end

    @testset "Poisson-GLM ControlledEmissionHMM: sample, forward, baum_welch" begin
        rng = MersenneTwister(42)
        p, T = 3, 600
        init = [0.6, 0.4]
        trans = [0.92 0.08; 0.15 0.85]
        dists = [PoissonGLM([0.2, 0.5, -0.3]), PoissonGLM([1.2, -0.4, 0.6])]
        hmm = ControlledEmissionHMM(init, trans, dists)

        control_seq = [vcat(1.0, randn(rng, p - 1)) for _ in 1:T]
        obs_seq = rand(rng, hmm, control_seq).obs_seq
        @test length(obs_seq) == T

        logL = last(forward(hmm, obs_seq, control_seq; seq_ends=[T]))
        @test all(isfinite, logL)

        # fit from a perturbed start; baum_welch must be (weakly) monotone
        init0 = [0.5, 0.5]
        dists0 = [PoissonGLM(zeros(p)), PoissonGLM(zeros(p))]
        hmm0 = ControlledEmissionHMM(init0, copy(trans), dists0)
        _, lls = baum_welch(hmm0, obs_seq, control_seq; seq_ends=[T], max_iterations=30)
        @test all(diff(lls) .>= -1e-6)
        @test last(lls) >= first(lls)
    end

    @testset "MvGaussian-GLM ControlledEmissionHMM end-to-end" begin
        rng = MersenneTwister(123)
        p, k, T = 2, 2, 500
        init = [0.5, 0.5]
        trans = [0.9 0.1; 0.1 0.9]
        B1 = [1.0 0.0; 0.5 -0.5]
        B2 = [-1.0 0.5; 0.0 1.0]
        Σ = Matrix{Float64}(I, k, k)
        dists = [MvGaussianGLM(B1, copy(Σ)), MvGaussianGLM(B2, copy(Σ))]
        hmm = ControlledEmissionHMM(init, trans, dists)

        control_seq = [vcat(1.0, randn(rng)) for _ in 1:T]   # length-p = 2 each
        obs_seq = rand(rng, hmm, control_seq).obs_seq
        @test length(obs_seq) == T
        @test length(first(obs_seq)) == k

        dists0 = [
            MvGaussianGLM(zeros(p, k), Matrix{Float64}(I, k, k)),
            MvGaussianGLM(zeros(p, k), Matrix{Float64}(I, k, k)),
        ]
        hmm0 = ControlledEmissionHMM(init, copy(trans), dists0)
        _, lls = baum_welch(hmm0, obs_seq, control_seq; seq_ends=[T], max_iterations=30)
        @test all(diff(lls) .>= -1e-6)
    end
end
