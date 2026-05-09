using EmissionModels
using Test
using Random
using LinearAlgebra
using DensityInterface
using StatsAPI

#=
  Allocation regression tests. The pattern
  is: warm up by running the operation once, then call it many times in a
  type-stable function and divide @allocated by the rep count to get
  per-call bytes.

  Steady-state targets (per call):
    - All logdensityof: 0 bytes
    - All rand!: 0 bytes (rand allocates only the return vector)
    - fit! is bounded by O(p² + k²), independent of n

  See `bench_*` helpers below for why a top-level `@allocated foo()` would
  report misleading kwarg-lowering noise that does NOT exist in real loops.
=#

bench_logd(d, y, x, n) = (s = 0.0; for _ in 1:n; s += logdensityof(d, y; control_seq=x); end; s)
bench_logd_unctrl(d, y, n) = (s = 0.0; for _ in 1:n; s += logdensityof(d, y); end; s)
bench_rand_scalar(rng, d, x, n) = (s = 0.0; for _ in 1:n; s += rand(rng, d; control_seq=x); end; s)
bench_rand_int(rng, d, x, n) = (s = 0; for _ in 1:n; s += rand(rng, d; control_seq=x); end; s)
bench_rand!_v(rng, d, out, x, n) = (s = 0.0; for _ in 1:n; rand!(rng, d, out; control_seq=x); s += out[1]; end; s)
bench_rand!_i(rng, d, out, x, n) = (s = 0; for _ in 1:n; rand!(rng, d, out; control_seq=x); s += out[1]; end; s)
bench_rand_unctrl_scalar(rng, d, n) = (s = 0.0; for _ in 1:n; s += rand(rng, d); end; s)
bench_rand_unctrl_vec(rng, d, n) = (s = 0.0; for _ in 1:n; s += rand(rng, d)[1]; end; s)

@testset "Allocations (steady state)" begin
    rng = Random.MersenneTwister(0)
    REPS = 1000

    @testset "logdensityof — zero alloc per call" begin
        x = [1.0, 2.0]

        g  = GaussianGLM([0.5, -1.0], 1.0)
        b  = BernoulliGLM([0.5, -1.0])
        p  = PoissonGLM([0.5, -1.0])
        mg = MvGaussianGLM([0.5 -1.0; 1.0 0.5], [1.0 0.3; 0.3 1.5])
        mb = MvBernoulliGLM([0.5 -1.0; 1.0 0.5])
        mp = MvPoissonGLM([0.5 -1.0; 0.2 0.0])
        yv = [0.1, 0.2]; yi = [0, 1]

        # Warm
        bench_logd(g, 0.5, x, 1); bench_logd(b, 1, x, 1); bench_logd(p, 2, x, 1)
        bench_logd(mg, yv, x, 1); bench_logd(mb, yi, x, 1); bench_logd(mp, yi, x, 1)

        @test (@allocated bench_logd(g,  0.5, x, REPS)) == 0
        @test (@allocated bench_logd(b,  1,   x, REPS)) == 0
        @test (@allocated bench_logd(p,  2,   x, REPS)) == 0
        @test (@allocated bench_logd(mg, yv,  x, REPS)) == 0
        @test (@allocated bench_logd(mb, yi,  x, REPS)) == 0
        @test (@allocated bench_logd(mp, yi,  x, REPS)) == 0
    end

    @testset "rand!/rand — zero alloc beyond return" begin
        x = [1.0, 2.0]

        g  = GaussianGLM([0.5, -1.0], 1.0)
        b  = BernoulliGLM([0.5, -1.0])
        p  = PoissonGLM([0.5, -1.0])
        mg = MvGaussianGLM([0.5 -1.0; 1.0 0.5], [1.0 0.3; 0.3 1.5])
        mb = MvBernoulliGLM([0.5 -1.0; 1.0 0.5])
        mp = MvPoissonGLM([0.5 -1.0; 0.2 0.0])

        # Univariate rand returns a scalar — zero alloc
        bench_rand_scalar(rng, g, x, 1)
        bench_rand_int(rng, b, x, 1)
        bench_rand_int(rng, p, x, 1)
        @test (@allocated bench_rand_scalar(rng, g, x, REPS)) == 0
        @test (@allocated bench_rand_int(rng, b, x, REPS)) == 0
        @test (@allocated bench_rand_int(rng, p, x, REPS)) == 0

        # Multivariate rand! into pre-allocated buffer — zero alloc
        out_f = zeros(2)
        out_i = zeros(Int, 2)
        bench_rand!_v(rng, mg, out_f, x, 1)
        bench_rand!_i(rng, mb, out_i, x, 1)
        bench_rand!_i(rng, mp, out_i, x, 1)
        @test (@allocated bench_rand!_v(rng, mg, out_f, x, REPS)) == 0
        @test (@allocated bench_rand!_i(rng, mb, out_i, x, REPS)) == 0
        @test (@allocated bench_rand!_i(rng, mp, out_i, x, REPS)) == 0
    end

    @testset "GLM fit! — bounded, independent of n" begin
        n = 500
        X = hcat(ones(n), randn(rng, n))
        w = ones(n)

        # GaussianGLM: closed-form WLS. Workspace: XWX (p²) + XWy (p).
        yg = randn(rng, n)
        gg = GaussianGLM([0.0, 0.0], 1.0); fit!(gg, yg, w; control_seq=X)
        gg = GaussianGLM([0.0, 0.0], 1.0)
        @test (@allocated fit!(gg, yg, w; control_seq=X)) ≤ 1_000

        # MvGaussianGLM: closed-form weighted MvN regression.
        # Workspace: XWX (p²) + XWY (p·k) + Σ_new (k²) + r (k).
        ymv = [randn(rng, 2) for _ in 1:n]
        gmv = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        fit!(gmv, ymv, w; control_seq=X)
        gmv = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        @test (@allocated fit!(gmv, ymv, w; control_seq=X)) ≤ 2_000

        # BernoulliGLM/PoissonGLM: hand-rolled Newton.
        # Workspace: g, H, Δ — three small alloc, total ~300 bytes for p=2.
        yb = Int[rand(rng) < 0.5 ? 1 : 0 for _ in 1:n]
        gb = BernoulliGLM(zeros(2)); fit!(gb, yb, w; control_seq=X)
        gb = BernoulliGLM(zeros(2))
        @test (@allocated fit!(gb, yb, w; control_seq=X)) ≤ 1_000

        yp = Int[rand(rng, 0:5) for _ in 1:n]
        gp = PoissonGLM(zeros(2)); fit!(gp, yp, w; control_seq=X)
        gp = PoissonGLM(zeros(2))
        @test (@allocated fit!(gp, yp, w; control_seq=X)) ≤ 1_000

        # Multivariate Newton: workspace shared across columns. _ColumnElementView
        # avoids the n-sized per-column copy that Vector-of-Vectors would force.
        ymb = [Int[rand(rng) < 0.5 ? 1 : 0 for _ in 1:2] for _ in 1:n]
        gmb = MvBernoulliGLM(zeros(2, 2)); fit!(gmb, ymb, w; control_seq=X)
        gmb = MvBernoulliGLM(zeros(2, 2))
        @test (@allocated fit!(gmb, ymb, w; control_seq=X)) ≤ 1_000

        ymp = [Int[rand(rng, 0:3) for _ in 1:2] for _ in 1:n]
        gmp = MvPoissonGLM(zeros(2, 2)); fit!(gmp, ymp, w; control_seq=X)
        gmp = MvPoissonGLM(zeros(2, 2))
        @test (@allocated fit!(gmp, ymp, w; control_seq=X)) ≤ 1_000
    end

    @testset "PoissonZeroInflated" begin
        zip = PoissonZeroInflated(3.0, 0.2)

        bench_logd_unctrl(zip, 0, 1); bench_logd_unctrl(zip, 5, 1)
        @test (@allocated bench_logd_unctrl(zip, 0, REPS)) == 0
        @test (@allocated bench_logd_unctrl(zip, 5, REPS)) == 0

        bench_rand_unctrl_scalar(rng, zip, 1)
        @test (@allocated bench_rand_unctrl_scalar(rng, zip, REPS)) == 0

        n = 500
        y = [rand(rng, zip) for _ in 1:n]
        w = ones(n)
        zip2 = PoissonZeroInflated(1.0, 0.1); fit!(zip2, y, w)
        zip2 = PoissonZeroInflated(1.0, 0.1)
        @test (@allocated fit!(zip2, y, w)) ≤ 1_000
    end

    @testset "MultivariateT (full Σ)" begin
        d = 2
        mvt = MultivariateT([0.0, 0.0], [1.0 0.3; 0.3 1.0], 5.0)
        xv = [0.1, 0.2]

        # Cholesky now stored as :L and residual reuses struct scratch.
        bench_logd_unctrl(mvt, xv, 1)
        @test (@allocated bench_logd_unctrl(mvt, xv, REPS)) == 0

        bench_rand_unctrl_vec(rng, mvt, 1)
        # rand still allocates the return vector; bound it loosely.
        @test (@allocated bench_rand_unctrl_vec(rng, mvt, REPS)) ≤ 600 * REPS

        n = 500
        obs = [randn(rng, d) for _ in 1:n]
        w = ones(n)
        mvt2 = MultivariateT([0.0, 0.0], Matrix(1.0I, d, d), 5.0)
        fit!(mvt2, obs, w; max_iter=5)
        mvt2 = MultivariateT([0.0, 0.0], Matrix(1.0I, d, d), 5.0)
        # ν step still goes through Optim — loose bound.
        @test (@allocated fit!(mvt2, obs, w; max_iter=5)) ≤ 1_000_000
    end

    @testset "MultivariateTDiag" begin
        d = 2
        mvtd = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        xv = [0.1, 0.2]

        bench_logd_unctrl(mvtd, xv, 1)
        @test (@allocated bench_logd_unctrl(mvtd, xv, REPS)) == 0

        bench_rand_unctrl_vec(rng, mvtd, 1)
        @test (@allocated bench_rand_unctrl_vec(rng, mvtd, REPS)) ≤ 400 * REPS

        n = 500
        obs = [randn(rng, d) for _ in 1:n]
        w = ones(n)
        mvtd2 = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        fit!(mvtd2, obs, w; max_iter=5)
        mvtd2 = MultivariateTDiag([0.0, 0.0], [1.0, 1.0], 5.0)
        @test (@allocated fit!(mvtd2, obs, w; max_iter=5)) ≤ 500_000
    end
end
