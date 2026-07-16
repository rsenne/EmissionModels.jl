using EmissionModels
using Test
using Random
using LinearAlgebra
using DensityInterface
using StatsAPI
using SequentialSamplingModels: SequentialSamplingModels

#=
  Allocation regression tests. Warm up first, then call the operation many
  times in a type-stable function and divide @allocated by the rep count.

  Steady-state targets (per call):
    - Univariate logdensityof, MvBernoulli/MvPoisson logdensityof,
      MultivariateTDiag logdensityof, PoissonZeroInflated logdensityof: 0 B
    - MvGaussianGLM logdensityof, MultivariateT logdensityof: 1 length-k
      vector per call.
    - All Mv rand!: 0 bytes
    - fit! is bounded by O(p² + k²) workspace plus (for Bernoulli/Poisson)
      Optim's Newton solver state a few KB, independent of n.

  On Julia 1.10 each `@allocated` measurement of these benchmark loops reports
  a constant ~16 B of measurement overhead (independent of REPS; gone on
  1.11+). `ALLOC_SLOP` absorbs it. I'm too lazy to figure out why.
=#
const ALLOC_SLOP = VERSION < v"1.11" ? 32 : 0

bench_logd(d, y, x, n) = (s = 0.0;
for _ in 1:n
    s += logdensityof(d, y; control_seq=x)
end;
s)
bench_logd_unctrl(d, y, n) = (s = 0.0;
for _ in 1:n
    s += logdensityof(d, y)
end;
s)
bench_rand_scalar(rng, d, x, n) = (s = 0.0;
for _ in 1:n
    s += rand(rng, d; control_seq=x)
end;
s)
bench_rand_int(rng, d, x, n) = (s = 0;
for _ in 1:n
    s += rand(rng, d; control_seq=x)
end;
s)
bench_rand!_v(rng, d, out, x, n) = (s = 0.0;
for _ in 1:n
    rand!(rng, d, out; control_seq=x)
    s += out[1]
end;
s)
bench_rand!_i(rng, d, out, x, n) = (s = 0;
for _ in 1:n
    rand!(rng, d, out; control_seq=x)
    s += out[1]
end;
s)
bench_rand_unctrl_scalar(rng, d, n) = (s = 0.0;
for _ in 1:n
    s += rand(rng, d)
end;
s)
bench_rand_unctrl_vec(rng, d, n) = (s = 0.0;
for _ in 1:n
    s += rand(rng, d)[1]
end;
s)
bench_rand!_unctrl(rng, d, out, n) = (s = 0.0;
for _ in 1:n
    rand!(rng, d, out)
    s += out[1]
end;
s)
bench_logd_ddm(d, y, c, n) = (s = 0.0;
for _ in 1:n
    s += logdensityof(d, y, c)
end;
s)
bench_rand_ddm(rng, d, c, n) = (s = 0.0;
for _ in 1:n
    s += rand(rng, d, c).rt
end;
s)

@testset "Allocations (steady state)" begin
    rng = Random.MersenneTwister(0)
    REPS = 1000

    @testset "logdensityof bounded per call" begin
        x = [1.0, 2.0]

        g = GaussianGLM([0.5, -1.0], 1.0)
        b = BernoulliGLM([0.5, -1.0])
        p = PoissonGLM([0.5, -1.0])
        mg = MvGaussianGLM([0.5 -1.0; 1.0 0.5], [1.0 0.3; 0.3 1.5])
        mb = MvBernoulliGLM([0.5 -1.0; 1.0 0.5])
        mp = MvPoissonGLM([0.5 -1.0; 0.2 0.0])
        yv = [0.1, 0.2]
        yi = [0, 1]

        # Warm
        bench_logd(g, 0.5, x, 1)
        bench_logd(b, 1, x, 1)
        bench_logd(p, 2, x, 1)
        bench_logd(mg, yv, x, 1)
        bench_logd(mb, yi, x, 1)
        bench_logd(mp, yi, x, 1)

        # Truly zero-alloc (no scratch needed)
        @test (@allocated bench_logd(g, 0.5, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_logd(b, 1, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_logd(p, 2, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_logd(mb, yi, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_logd(mp, yi, x, REPS)) <= ALLOC_SLOP

        #= MvGaussianGLM: one length-k residual per call (thread-safe).
           Bound to ~256 B/call, so 256 * REPS for the loop. =#
        @test (@allocated bench_logd(mg, yv, x, REPS)) ≤ 256 * REPS
    end

    @testset "rand!/rand zero alloc beyond return" begin
        x = [1.0, 2.0]

        g = GaussianGLM([0.5, -1.0], 1.0)
        b = BernoulliGLM([0.5, -1.0])
        p = PoissonGLM([0.5, -1.0])
        mg = MvGaussianGLM([0.5 -1.0; 1.0 0.5], [1.0 0.3; 0.3 1.5])
        mb = MvBernoulliGLM([0.5 -1.0; 1.0 0.5])
        mp = MvPoissonGLM([0.5 -1.0; 0.2 0.0])

        # Univariate rand returns a scalar, so zero alloc.
        bench_rand_scalar(rng, g, x, 1)
        bench_rand_int(rng, b, x, 1)
        bench_rand_int(rng, p, x, 1)
        @test (@allocated bench_rand_scalar(rng, g, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_rand_int(rng, b, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_rand_int(rng, p, x, REPS)) <= ALLOC_SLOP

        # Multivariate rand! into a pre-allocated buffer, so zero alloc.
        out_f = zeros(2)
        out_i = zeros(Int, 2)
        bench_rand!_v(rng, mg, out_f, x, 1)
        bench_rand!_i(rng, mb, out_i, x, 1)
        bench_rand!_i(rng, mp, out_i, x, 1)
        @test (@allocated bench_rand!_v(rng, mg, out_f, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_rand!_i(rng, mb, out_i, x, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_rand!_i(rng, mp, out_i, x, REPS)) <= ALLOC_SLOP
    end

    @testset "GLM fit! bounded, independent of n" begin
        n = 500
        X = hcat(ones(n), randn(rng, n))
        w = ones(n)

        # GaussianGLM: closed-form WLS. Workspace is XWX (p²) + XWy (p).
        yg = randn(rng, n)
        gg = GaussianGLM([0.0, 0.0], 1.0)
        fit!(gg, yg, w; control_seq=X)
        gg = GaussianGLM([0.0, 0.0], 1.0)
        @test (@allocated fit!(gg, yg, w; control_seq=X)) ≤ 1_000

        #= MvGaussianGLM: closed-form weighted MvN regression.
           Workspace is XWX (p²) + XWY (p·k) + Σ_new (k²) + r (k). =#
        ymv = [randn(rng, 2) for _ in 1:n]
        gmv = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        fit!(gmv, ymv, w; control_seq=X)
        gmv = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        @test (@allocated fit!(gmv, ymv, w; control_seq=X)) ≤ 2_000

        #= BernoulliGLM/PoissonGLM: Optim Newton via only_fgh!. Solver state is
           O(p²) per fit, measured ~4-6 KB for p=2, independent of n. =#
        yb = Int[rand(rng) < 0.5 ? 1 : 0 for _ in 1:n]
        gb = BernoulliGLM(zeros(2))
        fit!(gb, yb, w; control_seq=X)
        gb = BernoulliGLM(zeros(2))
        @test (@allocated fit!(gb, yb, w; control_seq=X)) ≤ 20_000

        yp = Int[rand(rng, 0:5) for _ in 1:n]
        gp = PoissonGLM(zeros(2))
        fit!(gp, yp, w; control_seq=X)
        gp = PoissonGLM(zeros(2))
        @test (@allocated fit!(gp, yp, w; control_seq=X)) ≤ 20_000

        #= Multivariate Newton: one Optim solve per column (k × O(p²) state).
           _ColumnElementView avoids the n-sized per-column copy that
           Vector-of-Vectors would force. =#
        ymb = [Int[rand(rng) < 0.5 ? 1 : 0 for _ in 1:2] for _ in 1:n]
        gmb = MvBernoulliGLM(zeros(2, 2))
        fit!(gmb, ymb, w; control_seq=X)
        gmb = MvBernoulliGLM(zeros(2, 2))
        @test (@allocated fit!(gmb, ymb, w; control_seq=X)) ≤ 40_000

        ymp = [Int[rand(rng, 0:3) for _ in 1:2] for _ in 1:n]
        gmp = MvPoissonGLM(zeros(2, 2))
        fit!(gmp, ymp, w; control_seq=X)
        gmp = MvPoissonGLM(zeros(2, 2))
        @test (@allocated fit!(gmp, ymp, w; control_seq=X)) ≤ 40_000
    end

    @testset "PoissonZeroInflated" begin
        zip = PoissonZeroInflated(3.0, 0.2)

        bench_logd_unctrl(zip, 0, 1)
        bench_logd_unctrl(zip, 5, 1)
        @test (@allocated bench_logd_unctrl(zip, 0, REPS)) <= ALLOC_SLOP
        @test (@allocated bench_logd_unctrl(zip, 5, REPS)) <= ALLOC_SLOP

        bench_rand_unctrl_scalar(rng, zip, 1)
        @test (@allocated bench_rand_unctrl_scalar(rng, zip, REPS)) <= ALLOC_SLOP

        n = 500
        y = [rand(rng, zip) for _ in 1:n]
        w = ones(n)
        zip2 = PoissonZeroInflated(1.0, 0.1)
        fit!(zip2, y, w)
        zip2 = PoissonZeroInflated(1.0, 0.1)
        # No n-length mask anymore (all(iszero, ...)) — accumulators only.
        @test (@allocated fit!(zip2, y, w)) ≤ 1_000
    end

    @testset "MultinomialGLM" begin
        x = [1.0, 2.0]
        # p = 2 inputs, K = 3 categories (B is p × K-1), 6 trials per draw.
        mn = MultinomialGLM([0.5 -1.0; 0.2 0.3], 6)
        yv = [3, 2, 1]

        bench_logd(mn, yv, x, 1)
        @test (@allocated bench_logd(mn, yv, x, REPS)) <= ALLOC_SLOP

        # rand! into a pre-allocated buffer — zero alloc.
        out = zeros(Int, 3)
        bench_rand!_i(rng, mn, out, x, 1)
        @test (@allocated bench_rand!_i(rng, mn, out, x, REPS)) <= ALLOC_SLOP

        #= fit!: Newton over the flattened (K-1)·p coefficients — Optim solver
           state is O(((K-1)p)²), independent of n. =#
        n = 500
        X = hcat(ones(n), randn(rng, n))
        w = ones(n)
        ymn = [rand(rng, mn; control_seq=view(X, i, :)) for i in 1:n]
        gmn = MultinomialGLM(zeros(2, 2), 6)
        fit!(gmn, ymn, w; control_seq=X)
        gmn = MultinomialGLM(zeros(2, 2), 6)
        @test (@allocated fit!(gmn, ymn, w; control_seq=X)) ≤ 40_000
    end

    @testset "MultivariateT (full Σ)" begin
        d = 2
        mvt = MultivariateT([0.0, 0.0], [1.0 0.3; 0.3 1.0], 5.0)
        xv = [0.1, 0.2]

        #= Cholesky stored as :L (no .L copy). Residual is allocated locally
           per call so concurrent calls on the same dist are race-free. =#
        bench_logd_unctrl(mvt, xv, 1)
        @test (@allocated bench_logd_unctrl(mvt, xv, REPS)) ≤ 256 * REPS

        # rand! into a pre-allocated buffer — zero alloc.
        out = zeros(d)
        bench_rand!_unctrl(rng, mvt, out, 1)
        @test (@allocated bench_rand!_unctrl(rng, mvt, out, REPS)) <= ALLOC_SLOP

        # rand allocates only the returned vector.
        bench_rand_unctrl_vec(rng, mvt, 1)
        @test (@allocated bench_rand_unctrl_vec(rng, mvt, REPS)) ≤ 256 * REPS

        n = 500
        obs = [randn(rng, d) for _ in 1:n]
        w = ones(n)
        mvt2 = MultivariateT([0.0, 0.0], Matrix(1.0I, d, d), 5.0)
        fit!(mvt2, obs, w; max_iter=5)
        mvt2 = MultivariateT([0.0, 0.0], Matrix(1.0I, d, d), 5.0)
        # The ν step still goes through Optim, hence the loose bound.
        @test (@allocated fit!(mvt2, obs, w; max_iter=5)) ≤ 1_000_000
    end

    @testset "DDM emissions" begin
        d = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.2)

        #= Each density evaluation constructs SSM's mutable `DDM` (twice for
           choice 1, which flips the boundary) that is SSM's public API, so
           the bound is a small per-call constant, not zero. =#
        bench_logd_ddm(d, (1, 0.6), 1.0, 1)
        bench_logd_ddm(d, (2, 0.6), -1.0, 1)
        @test (@allocated bench_logd_ddm(d, (1, 0.6), 1.0, REPS)) ≤ 256 * REPS
        @test (@allocated bench_logd_ddm(d, (2, 0.6), -1.0, REPS)) ≤ 256 * REPS

        # rand: one DDM construction; SSM's rejection sampler is scalar math.
        bench_rand_ddm(rng, d, 1.0, 1)
        @test (@allocated bench_rand_ddm(rng, d, 1.0, REPS)) ≤ 256 * REPS

        #= fit!: every NLL/gradient evaluation rebuilds a Dual-typed DDM per
           observation, so allocations scale with n × line-search evaluations
           for as long as the density goes through SSM's mutable struct.
           Regression guard against something worse, not a zero-alloc claim. =#
        rngf = Random.MersenneTwister(1)
        n = 200
        controls = [rand(rngf, (-1.0, 1.0)) for _ in 1:n]
        obs = [rand(rngf, d, controls[i]) for i in 1:n]
        w = ones(n)
        d2 = StimulusCodedDDM(; ν=1.5, α=0.9, z=0.5, τ=0.15)
        fit!(d2, obs, w; control_seq=controls, max_iter=5)
        d2 = StimulusCodedDDM(; ν=1.5, α=0.9, z=0.5, τ=0.15)
        @test (@allocated fit!(d2, obs, w; control_seq=controls, max_iter=5)) ≤ 10_000_000
    end

    @testset "MultivariateTDiag" begin
        d = 2
        mvtd = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        xv = [0.1, 0.2]

        bench_logd_unctrl(mvtd, xv, 1)
        @test (@allocated bench_logd_unctrl(mvtd, xv, REPS)) <= ALLOC_SLOP

        out = zeros(d)
        bench_rand!_unctrl(rng, mvtd, out, 1)
        @test (@allocated bench_rand!_unctrl(rng, mvtd, out, REPS)) <= ALLOC_SLOP

        bench_rand_unctrl_vec(rng, mvtd, 1)
        @test (@allocated bench_rand_unctrl_vec(rng, mvtd, REPS)) ≤ 256 * REPS

        n = 500
        obs = [randn(rng, d) for _ in 1:n]
        w = ones(n)
        mvtd2 = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(mvtd2, obs, w; max_iter=5)
        mvtd2 = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        @test (@allocated fit!(mvtd2, obs, w; max_iter=5)) ≤ 20_000
    end
end
