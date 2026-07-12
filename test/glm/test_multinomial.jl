using EmissionModels
using Test
using Random
using DensityInterface
using StatsAPI
using SpecialFunctions: loggamma
using Statistics: mean
using LinearAlgebra
using HiddenMarkovModels
using HiddenMarkovModels: ControlledEmissionHMM, baum_welch, forward

import StatsAPI: fit!

#= Reference softmax probabilities for a p × (K-1) coefficient matrix and a
   single control vector, with category K as the reference (η_K = 0). =#
function _softmax_probs(B, x)
    η = vcat(vec(x' * B), 0.0)
    e = exp.(η .- maximum(η))
    return e ./ sum(e)
end

#= Explicit multinomial log-pmf at count vector y with probabilities p. =#
function _multinomial_logpmf(y, p)
    n = sum(y)
    return loggamma(n + 1) - sum(loggamma.(y .+ 1)) +
           sum(yk * log(pk) for (yk, pk) in zip(y, p) if yk > 0)
end

@testset "MultinomialGLM" begin
    @testset "Constructor" begin
        B = [0.5 -1.0; 0.2 0.3]  # p = 2, K - 1 = 2
        glm = MultinomialGLM(B, 5)
        @test glm.B == B
        @test glm.n_trials == 5
        @test glm.in_dim == 2
        @test glm.out_dim == 3
        @test glm.prior isa NoPrior

        # Prior-carrying constructor
        glm_r = MultinomialGLM(B, 5, RidgePrior(0.5))
        @test glm_r.prior isa RidgePrior

        # Promoting fallback for integer eltype
        glm_i = MultinomialGLM([1 0; 0 1], 3)
        @test glm_i.B isa Matrix{Float64}

        # Invalid parameters
        @test_throws ArgumentError MultinomialGLM(zeros(0, 2), 5)
        @test_throws ArgumentError MultinomialGLM(zeros(2, 0), 5)
        @test_throws ArgumentError MultinomialGLM(B, 0)
    end

    @testset "logdensityof matches explicit pmf" begin
        rng = Random.MersenneTwister(1)
        B = [0.8 -0.4; -0.5 0.9; 0.2 0.1]  # p = 3, K = 3
        glm = MultinomialGLM(B, 6)

        for _ in 1:20
            x = vcat(1.0, randn(rng, 2))
            probs = _softmax_probs(B, x)
            for y in ([6, 0, 0], [2, 2, 2], [0, 1, 5], [1, 1, 1])
                expected = _multinomial_logpmf(y, probs)
                @test logdensityof(glm, y; control_seq=x) ≈ expected rtol = 1e-10
            end
        end
    end

    @testset "logdensityof edge cases" begin
        glm = MultinomialGLM([0.5 -1.0; 0.2 0.3], 5)
        x = [1.0, 0.5]

        # Counts stored as floats score identically
        @test logdensityof(glm, [2.0, 2.0, 1.0]; control_seq=x) ==
            logdensityof(glm, [2, 2, 1]; control_seq=x)

        #= Totals may differ from n_trials: the pmf conditions on the
           observation's own total. =#
        @test isfinite(logdensityof(glm, [1, 0, 0]; control_seq=x))
        @test isfinite(logdensityof(glm, [4, 4, 4]; control_seq=x))
        # The empty count vector has probability 1.
        @test logdensityof(glm, [0, 0, 0]; control_seq=x) == 0.0

        # Zero-mass observations
        @test logdensityof(glm, [3, -1, 3]; control_seq=x) == -Inf
        @test logdensityof(glm, [2.5, 2.5, 0.0]; control_seq=x) == -Inf

        # Dimension mismatches are programming errors
        @test_throws DimensionMismatch logdensityof(glm, [2, 3]; control_seq=x)
        @test_throws DimensionMismatch logdensityof(glm, [2, 2, 1]; control_seq=[1.0])
    end

    @testset "K = 2 reduces to Bernoulli logistic for one-hot counts" begin
        rng = Random.MersenneTwister(2)
        β = [0.7, -1.2]
        glm = MultinomialGLM(reshape(β, 2, 1), 1)
        blm = BernoulliGLM(β)
        for _ in 1:10
            x = vcat(1.0, randn(rng))
            @test logdensityof(glm, [1, 0]; control_seq=x) ≈
                logdensityof(blm, 1; control_seq=x) rtol = 1e-12
            @test logdensityof(glm, [0, 1]; control_seq=x) ≈
                logdensityof(blm, 0; control_seq=x) rtol = 1e-12
        end
    end

    @testset "Random sampling" begin
        rng = Random.MersenneTwister(42)
        B = [0.5 -1.0; 0.2 0.3]
        n_trials = 8
        glm = MultinomialGLM(B, n_trials)
        x = [1.0, 0.5]
        probs = _softmax_probs(B, x)

        n_samples = 5000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n_samples]

        @test all(sum(s) == n_trials for s in samples)
        @test all(all(≥(0), s) for s in samples)
        @test all(s isa Vector{Int} for s in samples)

        # Marginal means are n_trials * p_k
        for k in 1:3
            @test mean(s[k] for s in samples) ≈ n_trials * probs[k] atol = 0.15
        end

        # In-place sampling
        out = zeros(Int, 3)
        rand!(rng, glm, out; control_seq=x)
        @test sum(out) == n_trials
        @test_throws DimensionMismatch rand!(rng, glm, zeros(Int, 2); control_seq=x)
        @test_throws DimensionMismatch rand!(rng, glm, out; control_seq=[1.0])

        # Positional ControlledEmission adapter matches the keyword path
        @test rand(Random.MersenneTwister(7), glm, x) ==
            rand(Random.MersenneTwister(7), glm; control_seq=x)
    end

    @testset "fit! recovers coefficients" begin
        rng = Random.MersenneTwister(123)
        n, p = 3000, 2
        B_true = [1.0 -0.5; -0.8 0.6]  # K = 3
        n_trials = 5
        gen = MultinomialGLM(B_true, n_trials)

        X = hcat(ones(n), randn(rng, n))
        obs = [rand(rng, gen; control_seq=view(X, i, :)) for i in 1:n]
        w = ones(n)

        fitted = MultinomialGLM(zeros(p, 2), n_trials)
        fit!(fitted, obs, w; control_seq=X)
        @test fitted.B ≈ B_true atol = 0.15

        # Positional vector-of-vectors control path matches the matrix path
        Xvv = [X[i, :] for i in 1:n]
        fitted_pos = MultinomialGLM(zeros(p, 2), n_trials)
        fit!(fitted_pos, obs, Xvv, w)
        @test fitted_pos.B ≈ fitted.B rtol = 1e-8
    end

    @testset "fit! respects weights and priors" begin
        rng = Random.MersenneTwister(456)
        n, p = 500, 2
        B_true = [0.6 -0.4; -0.3 0.8]
        gen = MultinomialGLM(B_true, 4)
        X = hcat(ones(2n), randn(rng, 2n))
        obs = [rand(rng, gen; control_seq=view(X, i, :)) for i in 1:(2n)]

        #= Zero-weight observations must not influence the fit: fitting the
           full data with the second half zero-weighted equals fitting the
           first half alone. =#
        w_half = vcat(ones(n), zeros(n))
        g_masked = MultinomialGLM(zeros(p, 2), 4)
        fit!(g_masked, obs, w_half; control_seq=X)
        g_subset = MultinomialGLM(zeros(p, 2), 4)
        fit!(g_subset, obs[1:n], ones(n); control_seq=X[1:n, :])
        @test g_masked.B ≈ g_subset.B rtol = 1e-6

        # Ridge shrinks the coefficients toward zero
        g_mle = MultinomialGLM(zeros(p, 2), 4)
        fit!(g_mle, obs, ones(2n); control_seq=X)
        g_ridge = MultinomialGLM(zeros(p, 2), 4, RidgePrior(50.0))
        fit!(g_ridge, obs, ones(2n); control_seq=X)
        @test norm(g_ridge.B) < norm(g_mle.B)
    end

    @testset "fit! dimension errors" begin
        glm = MultinomialGLM(zeros(2, 2), 5)
        X = ones(4, 2)
        obs = [[1, 1, 3] for _ in 1:4]
        @test_throws DimensionMismatch fit!(glm, obs[1:3], ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, obs, ones(3); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, obs, ones(4); control_seq=ones(4, 3))
        bad_obs = [[1, 1, 3], [1, 1], [1, 1, 3], [1, 1, 3]]
        @test_throws DimensionMismatch fit!(glm, bad_obs, ones(4); control_seq=X)
    end

    @testset "ControlledEmissionHMM: sample, forward, baum_welch" begin
        rng = Random.MersenneTwister(777)
        p, T = 2, 800
        init = [0.6, 0.4]
        trans = [0.92 0.08; 0.15 0.85]
        dists = [
            MultinomialGLM([1.0 -0.6; -0.5 0.4], 5), MultinomialGLM([-1.0 0.8; 0.6 -0.3], 5)
        ]
        hmm = ControlledEmissionHMM(init, trans, dists)

        control_seq = [vcat(1.0, randn(rng)) for _ in 1:T]
        obs_seq = rand(rng, hmm, control_seq).obs_seq
        @test length(obs_seq) == T
        @test all(sum(obs) == 5 for obs in obs_seq)

        logL = last(forward(hmm, obs_seq, control_seq; seq_ends=[T]))
        @test all(isfinite, logL)

        # fit from a perturbed start; baum_welch must be (weakly) monotone
        dists0 = [MultinomialGLM(zeros(p, 2), 5), MultinomialGLM(zeros(p, 2), 5)]
        hmm0 = ControlledEmissionHMM([0.5, 0.5], copy(trans), dists0)
        _, lls = baum_welch(hmm0, obs_seq, control_seq; seq_ends=[T], max_iterations=20)
        @test all(diff(lls) .>= -1e-6)
        @test last(lls) >= first(lls)
    end
end
