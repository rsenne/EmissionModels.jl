using EmissionModels
using Distributions
using HiddenMarkovModels
using HiddenMarkovModels: ControlledEmission, ControlledEmissionHMM, baum_welch, forward
using DensityInterface
using StatsAPI
using Random
using LinearAlgebra
using Statistics: mean
using Test

#= Reference marginal likelihood, written directly from the generative model:
   y_n = Σ_l α[l,n] y_{n,t-l} + c_n s + ε, with s ~ Poisson(λ_n) summed out. =#
function _ref_logdensity(em, y, u)
    p = em.params
    N = length(y)
    ord = size(p.α, 1)
    total = 0.0
    for n in 1:N
        μ = p.b[n] + sum(p.α[l, n] * u[(l - 1) * N + n] for l in 1:ord)
        terms = [
            pdf(Normal(μ + p.c[n] * s, sqrt(p.σ2[n])), y[n]) * pdf(Poisson(em.λ[n]), s) for
            s in 0:(em.R)
        ]
        total += log(sum(terms))
    end
    return total
end

@testset "CalciumEmission" begin
    @testset "construction and validation" begin
        @test CalciumEmission <: ControlledEmission

        params = CalciumParams([0.9 0.8], [1.0, 1.2], [0.05, 0.07])
        @test EmissionModels.ar_order(params) == 1
        @test EmissionModels.nneurons(params) == 2

        em = CalciumEmission([0.5, 1.0], params)
        @test EmissionModels.nneurons(em) == 2
        @test DensityInterface.DensityKind(em) isa DensityInterface.HasDensity

        @test_throws ArgumentError CalciumParams([0.9 0.8], [1.0, -1.0], [0.05, 0.07])
        @test_throws ArgumentError CalciumParams([0.9 0.8], [1.0, 1.0], [0.05, 0.0])
        @test_throws DimensionMismatch CalciumParams([0.9 0.8], [1.0], [0.05])
        @test_throws DimensionMismatch CalciumEmission([0.5], params)
        @test_throws ArgumentError CalciumEmission([0.5, 0.0], params)
        @test_throws ArgumentError CalciumEmission([0.5, 1.0], params; R=0)

        # Promotion: mixed integer/float inputs land on a common float eltype.
        pmix = CalciumParams([1 0], [1, 2], [1.0, 2.0])
        @test pmix isa CalciumParams{Float64}
    end

    @testset "logdensityof matches the explicit spike sum" begin
        params = CalciumParams([0.92 0.85], [1.0, 0.7], [0.04, 0.09])
        em = CalciumEmission([0.6, 1.4], params; R=12)
        y = [0.8, 1.3]
        u = [0.5, 0.2]
        @test logdensityof(em, y; control_seq=u) ≈ _ref_logdensity(em, y, u) rtol = 1e-10
        # positional ControlledEmission path must agree with the keyword path
        @test logdensityof(em, y, u) == logdensityof(em, y; control_seq=u)

        # AR(2)
        params2 = CalciumParams([1.3 1.1; -0.4 -0.25], [1.0, 0.9], [0.03, 0.05])
        em2 = CalciumEmission([0.9, 0.4], params2; R=12)
        u2 = [0.7, 0.3, 0.4, 0.1]
        @test logdensityof(em2, y; control_seq=u2) ≈ _ref_logdensity(em2, y, u2) rtol =
            1e-10

        @test_throws DimensionMismatch logdensityof(em, [1.0]; control_seq=u)
        @test_throws DimensionMismatch logdensityof(em, y; control_seq=[1.0, 2.0, 3.0])
    end

    @testset "lagged_controls layout" begin
        obs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        u1 = lagged_controls(obs, 1)
        @test u1[1] == [0.0, 0.0]
        @test u1[2] == [1.0, 2.0]
        @test u1[3] == [3.0, 4.0]

        u2 = lagged_controls(obs, 2)
        @test u2[3] == [3.0, 4.0, 1.0, 2.0]   # [lag1; lag2]
        @test u2[2] == [1.0, 2.0, 0.0, 0.0]

        # sequence boundaries must not leak history across recordings
        ub = lagged_controls(obs, 1; seq_ends=[2, 3])
        @test ub[3] == [0.0, 0.0]

        # pad = :first holds the pre-recording history at the first sample
        f1 = lagged_controls(obs, 1; pad=:first)
        @test f1[1] == [1.0, 2.0]
        @test f1[2] == [1.0, 2.0]
        @test f1[3] == [3.0, 4.0]
        f2 = lagged_controls(obs, 2; pad=:first)
        @test f2[1] == [1.0, 2.0, 1.0, 2.0]
        @test f2[2] == [1.0, 2.0, 1.0, 2.0]
        @test f2[3] == [3.0, 4.0, 1.0, 2.0]
        fb = lagged_controls(obs, 1; seq_ends=[2, 3], pad=:first)
        @test fb[3] == [5.0, 6.0]      # own value, not the previous recording's

        @test_throws ArgumentError lagged_controls(obs, 1; pad=:mean)
    end

    @testset "rand_calcium rolls out a consistent trace" begin
        rng = MersenneTwister(0)
        params = CalciumParams([0.9 0.9], [1.0, 1.0], [0.02, 0.02]; nshare=2)
        dists = [CalciumEmission([0.2, 1.5], params), CalciumEmission([1.5, 0.2], params)]
        sim = rand_calcium(rng, [1.0, 0.0], [0.9 0.1; 0.1 0.9], dists, 50)

        @test length(sim.obs_seq) == 50
        @test all(length.(sim.obs_seq) .== 2)
        @test sim.state_seq[1] == 1
        # the control at t is exactly the observation at t-1
        @test sim.control_seq[1] == zeros(2)
        @test sim.control_seq[10] == sim.obs_seq[9]
        # generated controls agree with rebuilding them from the trace
        @test lagged_controls(sim.obs_seq, 1) == sim.control_seq
    end

    @testset "tied M-step pools across states" begin
        rng = MersenneTwister(7)
        α_true = [0.90 0.80]
        params = CalciumParams(α_true, [1.0, 1.0], [0.02, 0.02]; nshare=2)
        dists = [CalciumEmission([0.2, 1.2], params), CalciumEmission([1.2, 0.2], params)]
        sim = rand_calcium(rng, [0.5, 0.5], [0.9 0.1; 0.1 0.9], dists, 400)

        p_fit = CalciumParams([0.5 0.5], [1.0, 1.0], [0.5, 0.5]; nshare=2)
        e1 = CalciumEmission([0.5, 0.5], p_fit)
        e2 = CalciumEmission([1.0, 1.0], p_fit)
        w1 = [s == 1 ? 1.0 : 0.0 for s in sim.state_seq]
        w2 = 1 .- w1

        # the shared parameters must stay put until the whole set has been visited
        fit!(e1, sim.obs_seq, w1; control_seq=sim.control_seq)
        @test p_fit.α == [0.5 0.5]
        @test p_fit.ncalls == 1
        fit!(e2, sim.obs_seq, w2; control_seq=sim.control_seq)
        @test p_fit.ncalls == 0
        @test p_fit.α != [0.5 0.5]
        # accumulators reset after the pooled solve
        @test all(iszero, p_fit.wsum)

        # iterating the pair (known states, so this is plain EM on the calcium
        # parameters) recovers the tied dynamics
        for _ in 1:60
            fit!(e1, sim.obs_seq, w1; control_seq=sim.control_seq)
            fit!(e2, sim.obs_seq, w2; control_seq=sim.control_seq)
        end
        @test p_fit.α ≈ α_true rtol = 0.15
        @test p_fit.σ2 ≈ [0.02, 0.02] rtol = 1.0

        # untied emissions carry independent parameter sets
        untied = calcium_emissions(
            [0.5 1.0; 1.0 0.5], [0.5 0.5], [1.0, 1.0], [0.1, 0.1]; tied=false
        )
        @test untied[1].params !== untied[2].params
        tied = calcium_emissions([0.5 1.0; 1.0 0.5], [0.5 0.5], [1.0, 1.0], [0.1, 0.1])
        @test tied[1].params === tied[2].params
        @test tied[1].params.nshare == 2
    end

    @testset "positional fit! matches the keyword path" begin
        rng = MersenneTwister(3)
        params = CalciumParams([0.9 0.9], [1.0, 1.0], [0.02, 0.02])
        dists = [CalciumEmission([0.4, 1.0], params)]
        sim = rand_calcium(rng, [1.0], reshape([1.0], 1, 1), dists, 200)
        w = rand(rng, 200)

        pa = CalciumParams([0.5 0.5], [1.0, 1.0], [0.3, 0.3])
        pb = CalciumParams([0.5 0.5], [1.0, 1.0], [0.3, 0.3])
        ea = CalciumEmission([0.5, 0.5], pa)
        eb = CalciumEmission([0.5, 0.5], pb)
        fit!(ea, sim.obs_seq, sim.control_seq, w)          # positional
        fit!(eb, sim.obs_seq, w; control_seq=sim.control_seq)  # keyword
        @test ea.λ == eb.λ
        @test pa.α == pb.α
        @test pa.σ2 == pb.σ2
    end

    @testset "init_calcium moment initialization" begin
        rng = MersenneTwister(19)
        params = CalciumParams([0.90 0.80], [1.0, 1.0], [0.02, 0.02])
        dists = [CalciumEmission([1.0, 1.0], params)]
        sim = rand_calcium(rng, [1.0], reshape([1.0], 1, 1), dists, 3000)

        ems = init_calcium(sim.obs_seq, 3, 1; rng=MersenneTwister(1))
        @test length(ems) == 3
        @test ems[1].params === ems[3].params        # tied by default
        @test ems[1].params.α ≈ [0.90 0.80] rtol = 0.1
        @test ems[1].params.σ2 ≈ [0.02, 0.02] rtol = 3.0   # moments only, EM refines
        @test all(>(0), ems[1].params.σ2)
        # states must differ, or Baum-Welch has no asymmetry to break
        @test ems[1].λ != ems[2].λ
        @test all(e -> all(>(0), e.λ), ems)

        # spread = 0 gives identical states (useful for single-state fits)
        flat = init_calcium(sim.obs_seq, 2, 1; spread=0.0)
        @test flat[1].λ == flat[2].λ

        # AR(2) initialization on AR(2) data
        p2 = CalciumParams([1.20 1.10; -0.35 -0.30], [1.0, 1.0], [0.02, 0.02])
        sim2 = rand_calcium(
            MersenneTwister(21),
            [1.0],
            reshape([1.0], 1, 1),
            [CalciumEmission([1.0, 1.0], p2)],
            3000,
        )
        e2 = init_calcium(sim2.obs_seq, 2, 2; rng=MersenneTwister(1))
        @test e2[1].params.α ≈ [1.20 1.10; -0.35 -0.30] rtol = 0.2

        @test_throws ArgumentError init_calcium(sim.obs_seq, 0, 1)
        @test_throws ArgumentError init_calcium(sim.obs_seq, 2, 0)
        @test_throws DimensionMismatch init_calcium(sim.obs_seq, 2, 1; c=[1.0])
    end

    @testset "Baum-Welch on an AR(1) calcium HMM" begin
        rng = MersenneTwister(11)
        n_t = 2000
        init = [0.5, 0.5]
        trans = [0.97 0.03; 0.03 0.97]
        λ_true = [0.15 1.60; 1.70 0.20]        # neurons × states
        dists = calcium_emissions(λ_true, [0.90 0.85], [1.0, 1.0], [0.02, 0.02]; R=8)
        sim = rand_calcium(rng, init, trans, dists, n_t)

        dists0 = init_calcium(sim.obs_seq, 2, 1; R=8, rng=MersenneTwister(2))
        hmm0 = ControlledEmissionHMM(copy(init), copy(trans), dists0)
        hmm_fit, lls = baum_welch(
            hmm0, sim.obs_seq, sim.control_seq; seq_ends=[n_t], max_iterations=40
        )
        @test all(diff(lls) .>= -1e-6)          # monotone EM
        @test last(lls) > first(lls)

        p_fit = hmm_fit.dists[1].params
        @test p_fit.α ≈ [0.90 0.85] rtol = 0.1
        @test p_fit.σ2 ≈ [0.02, 0.02] rtol = 0.6

        # rates recovered up to a state permutation
        λ_fit = hcat((d.λ for d in hmm_fit.dists)...)
        err(perm) = maximum(abs, λ_fit[:, perm] .- λ_true)
        @test min(err([1, 2]), err([2, 1])) < 0.35

        # state sequence recovered well above chance
        q = [
            argmax(
                view(
                    forward(hmm_fit, sim.obs_seq, sim.control_seq; seq_ends=[n_t])[1], :, t
                ),
            ) for t in 1:n_t
        ]
        acc = max(mean(q .== sim.state_seq), mean(q .!= sim.state_seq))
        @test acc > 0.9
    end

    @testset "Baum-Welch on an AR(2) calcium HMM" begin
        rng = MersenneTwister(13)
        n_t = 1500
        α_true = [1.20 1.10; -0.35 -0.30]
        dists = calcium_emissions(
            [0.15 1.50; 1.60 0.20], α_true, [1.0, 1.0], [0.02, 0.02]; R=8
        )
        sim = rand_calcium(rng, [0.5, 0.5], [0.97 0.03; 0.03 0.97], dists, n_t)
        @test length(first(sim.control_seq)) == 4

        dists0 = init_calcium(sim.obs_seq, 2, 2; R=8, rng=MersenneTwister(4))
        hmm0 = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.05 0.95], dists0)
        hmm_fit, lls = baum_welch(
            hmm0, sim.obs_seq, sim.control_seq; seq_ends=[n_t], max_iterations=40
        )
        @test all(diff(lls) .>= -1e-6)
        @test hmm_fit.dists[1].params.α ≈ α_true rtol = 0.2
    end

    @testset "fitting the influx c" begin
        rng = MersenneTwister(17)
        params = CalciumParams([0.90 0.90], [1.5, 0.6], [0.01, 0.01])
        dists = [CalciumEmission([0.8, 0.8], params)]
        sim = rand_calcium(rng, [1.0], reshape([1.0], 1, 1), dists, 3000)

        em = only(init_calcium(sim.obs_seq, 1, 1; fit_c=true, R=8, spread=0.0))
        p_fit = em.params
        for _ in 1:60
            fit!(em, sim.obs_seq, ones(3000); control_seq=sim.control_seq)
        end
        @test p_fit.c ≈ [1.5, 0.6] rtol = 0.3
        @test p_fit.α ≈ [0.90 0.90] rtol = 0.1
        @test em.λ ≈ [0.8, 0.8] rtol = 0.35

        # with fit_c = false the influx is held exactly at its supplied value
        p_fixed = CalciumParams([0.7 0.7], [1.0, 1.0], [0.1, 0.1])
        em_fixed = CalciumEmission([0.5, 0.5], p_fixed; R=8)
        fit!(em_fixed, sim.obs_seq, ones(3000); control_seq=sim.control_seq)
        @test p_fixed.c == [1.0, 1.0]
    end

    @testset "baseline term" begin
        # b shifts the conditional mean and nothing else
        p0 = CalciumParams([0.9 0.9], [1.0, 1.0], [0.05, 0.05])
        pb = CalciumParams([0.9 0.9], [1.0, 1.0], [0.05, 0.05]; b=[2.0, -1.0])
        @test p0.b == [0.0, 0.0]
        e0 = CalciumEmission([0.7, 0.7], p0; R=12)
        eb = CalciumEmission([0.7, 0.7], pb; R=12)
        y = [1.4, 0.3]
        u = [0.5, 0.2]
        @test logdensityof(eb, y; control_seq=u) ≈ _ref_logdensity(eb, y, u) rtol = 1e-10
        @test logdensityof(eb, y .+ [2.0, -1.0]; control_seq=u) ≈
            logdensityof(e0, y; control_seq=u)

        # a standing baseline the fixed-b model cannot represent
        rng = MersenneTwister(23)
        n_t = 4000
        b_true = 0.5
        ptrue = CalciumParams([0.95;;], [1.0], [0.0225]; b=[b_true], nshare=2)
        dtrue = [CalciumEmission([0.5], ptrue), CalciumEmission([4.0], ptrue)]
        sim = rand_calcium(rng, [0.5, 0.5], [0.98 0.02; 0.02 0.98], dtrue, n_t)
        @test all(v -> only(v) > 0, sim.obs_seq)

        function run(; kw...)
            d0 = init_calcium(sim.obs_seq, 2, 1; R=25, rng=MersenneTwister(1), kw...)
            hmm = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.05 0.95], d0)
            hf, lls = baum_welch(
                hmm, sim.obs_seq, sim.control_seq; seq_ends=[n_t], max_iterations=80
            )
            @test all(diff(lls) .>= -1e-6)
            return hf, last(lls)
        end

        fixed, ll_fixed = run()
        fitted, ll_fitted = run(; fit_b=true)
        @test only(fixed.dists[1].params.b) == 0.0          # held at its value
        @test only(fitted.dists[1].params.b) ≈ b_true rtol = 0.15
        @test only(fitted.dists[1].params.α) ≈ 0.95 rtol = 0.05
        # the baseline was absorbed by the noise term when it had nowhere else
        # to go, so fitting it is a large likelihood gain
        @test only(fitted.dists[1].params.σ2) ≈ 0.0225 rtol = 0.5
        @test only(fixed.dists[1].params.σ2) > 4 * only(fitted.dists[1].params.σ2)
        @test ll_fitted > ll_fixed + 100
    end

    @testset "shift invariance with a fitted baseline" begin
        #= Shifting a trace by a constant `m` is exactly absorbed by the
           baseline: y_t = b + αy_{t-1} + cs + ε implies
           (y-m)_t = [b - m(1-α)] + α(y-m)_{t-1} + cs + ε. So the likelihood at
           corresponding parameters must be identical — but only under
           `pad=:first`, since `pad=:zero` claims a pre-recording rest level of
           zero, which is a different assumption for the two traces. =#
        rng = MersenneTwister(41)
        n_t = 1500
        α, m = 0.95, 30.0
        ptrue = CalciumParams([α;;], [1.0], [0.05]; b=[0.5], fit_b=true, nshare=2)
        dtrue = [CalciumEmission([0.5], ptrue; R=20), CalciumEmission([4.0], ptrue; R=20)]
        sim = rand_calcium(rng, [0.5, 0.5], [0.98 0.02; 0.02 0.98], dtrue, n_t)

        shifted = [v .- m for v in sim.obs_seq]
        pshift = CalciumParams(
            [α;;], [1.0], [0.05]; b=[0.5 - m * (1 - α)], fit_b=true, nshare=2
        )
        dshift = [
            CalciumEmission([0.5], pshift; R=20), CalciumEmission([4.0], pshift; R=20)
        ]

        trans = [0.98 0.02; 0.02 0.98]
        hmm0 = ControlledEmissionHMM([0.5, 0.5], copy(trans), dtrue)
        hmm1 = ControlledEmissionHMM([0.5, 0.5], copy(trans), dshift)
        ll(h, o, u) = last(forward(h, o, u; seq_ends=[length(o)])[2])

        c0 = lagged_controls(sim.obs_seq, 1; pad=:first)
        c1 = lagged_controls(shifted, 1; pad=:first)
        @test ll(hmm0, sim.obs_seq, c0) ≈ ll(hmm1, shifted, c1) rtol = 1e-10

        #= With `pad=:zero` the shifted trace's first sample sits far below the
           asserted rest level of zero, which `s ≥ 0` cannot reach — one
           timestep is then wildly improbable and the equality breaks. =#
        z0 = lagged_controls(sim.obs_seq, 1)
        z1 = lagged_controls(shifted, 1)
        @test ll(hmm0, sim.obs_seq, z0) - ll(hmm1, shifted, z1) > 100
    end

    @testset "Gaussian large-rate marginal" begin
        #= Error against the exact Poisson sum, measured over ±2 marginal SDs
           and expected to fall like 1/√λ. Absolute bounds are loose; the
           monotone decrease is the real assertion. =#
        errs = map((5.0, 20.0, 100.0, 400.0)) do λ
            pp = CalciumParams([0.9;;], [1.0], [0.5])
            pg = CalciumParams([0.9;;], [1.0], [0.5]; gaussian=true)
            exact = CalciumEmission([λ], pp; R=ceil(Int, λ + 10 * sqrt(λ)))
            approx = CalciumEmission([λ], pg)
            sd = sqrt(λ + 0.5)
            maximum(
                abs(
                    logdensityof(exact, [0.9 + λ + k * sd]; control_seq=[1.0]) -
                    logdensityof(approx, [0.9 + λ + k * sd]; control_seq=[1.0]),
                ) for k in (-2, -1, 0, 1, 2)
            )
        end
        @test all(diff(collect(errs)) .< 0)
        @test errs[2] < 0.15      # λ = 20
        @test errs[3] < 0.05      # λ = 100
        @test errs[4] < 0.025     # λ = 400

        # closed form: N(y; b + Σαy_lag + cλ, c²λ + σ²)
        pg = CalciumParams([0.9 0.8], [1.0, 0.5], [0.5, 0.2]; b=[1.0, 0.0], gaussian=true)
        eg = CalciumEmission([30.0, 50.0], pg)
        y = [29.0, 26.0]
        u = [2.0, 1.0]
        ref = sum(
            logpdf(
                Normal(
                    pg.b[n] + pg.α[1, n] * u[n] + pg.c[n] * eg.λ[n],
                    sqrt(pg.c[n]^2 * eg.λ[n] + pg.σ2[n]),
                ),
                y[n],
            ) for n in 1:2
        )
        @test logdensityof(eg, y; control_seq=u) ≈ ref rtol = 1e-12

        # fitting at aggregate rates, where the 0:R sum would be impractical
        rng = MersenneTwister(31)
        n_t = 3000
        dtrue = calcium_emissions([40.0 120.0], [0.9;;], [1.0], [4.0]; gaussian=true)
        sim = rand_calcium(rng, [0.5, 0.5], [0.98 0.02; 0.02 0.98], dtrue, n_t)
        d0 = init_calcium(sim.obs_seq, 2, 1; α=0.9, gaussian=true, rng=MersenneTwister(9))
        @test d0[1].params.gaussian
        hmm = ControlledEmissionHMM([0.5, 0.5], [0.95 0.05; 0.05 0.95], d0)
        hf, lls = baum_welch(
            hmm, sim.obs_seq, sim.control_seq; seq_ends=[n_t], max_iterations=60
        )
        @test all(diff(lls) .>= -1e-6)
        @test only(hf.dists[1].params.α) ≈ 0.9 rtol = 0.1
        λ_fit = sort(vcat((d.λ for d in hf.dists)...))
        @test λ_fit ≈ [40.0, 120.0] rtol = 0.2
        q = [
            argmax(
                view(forward(hf, sim.obs_seq, sim.control_seq; seq_ends=[n_t])[1], :, t)
            ) for t in 1:n_t
        ]
        @test max(mean(q .== sim.state_seq), mean(q .!= sim.state_seq)) > 0.95
    end

    @testset "init_calcium α override" begin
        rng = MersenneTwister(37)
        params = CalciumParams([0.9 0.9], [1.0, 1.0], [0.02, 0.02])
        sim = rand_calcium(
            rng, [1.0], reshape([1.0], 1, 1), [CalciumEmission([1.0, 1.0], params)], 500
        )
        # scalar, vector and matrix forms all bypass the Yule-Walker estimate
        @test init_calcium(sim.obs_seq, 2, 1; α=0.75)[1].params.α == [0.75 0.75]
        @test init_calcium(sim.obs_seq, 2, 1; α=[0.75])[1].params.α == [0.75 0.75]
        @test init_calcium(sim.obs_seq, 2, 1; α=[0.75 0.6])[1].params.α == [0.75 0.6]
        @test init_calcium(sim.obs_seq, 2, 2; α=[1.2, -0.3])[1].params.α ==
            [1.2 1.2; -0.3 -0.3]

        @test_throws ArgumentError init_calcium(sim.obs_seq, 2, 2; α=0.9)
        @test_throws DimensionMismatch init_calcium(sim.obs_seq, 2, 1; α=[0.9, 0.5])
        @test_throws DimensionMismatch init_calcium(sim.obs_seq, 2, 1; α=[0.9 0.5 0.2])
    end

    @testset "type genericity" begin
        params = CalciumParams(Float32[0.9 0.85], Float32[1.0, 1.0], Float32[0.05, 0.05])
        @test params isa CalciumParams{Float32}
        em = CalciumEmission(Float32[0.5, 1.0], params)
        @test em isa CalciumEmission{Float32}
        ld = logdensityof(em, Float32[0.4, 0.9]; control_seq=Float32[0.1, 0.2])
        @test ld isa Float32
        @test isfinite(ld)

        rng = MersenneTwister(5)
        sim = rand_calcium(rng, Float32[1.0], reshape(Float32[1.0], 1, 1), [em], 20)
        @test eltype(first(sim.obs_seq)) == Float32
        fit!(em, sim.obs_seq, ones(Float32, 20); control_seq=sim.control_seq)
        @test all(isfinite, params.α)
        @test all(>(0), params.σ2)
    end
end
