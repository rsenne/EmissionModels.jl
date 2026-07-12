using EmissionModels
using HiddenMarkovModels
using Distributions
using LinearAlgebra
using Random
using Statistics
using Test

@testset "Discrepancy measures on (non)uniform samples" begin
    rng = Random.MersenneTwister(0)

    # Uniform drivers should score small on every discrepancy.
    U = rand(rng, 1, 4000)
    @test compute_discrepancy(KSDiscrepancy(), U) < 0.05
    @test compute_discrepancy(SquaredErrorDiscrepancy(), U) < 1e-2
    @test compute_discrepancy(WassersteinDiscrepancy(), U) < 0.05
    @test abs(compute_discrepancy(KLDiscrepancy(), U)) < 0.1
    @test abs(compute_discrepancy(MMDDiscrepancy(; block_size=4000), U)) < 1e-2

    # Strongly non-uniform drivers (concentrated near 0.5) should score larger.
    B = clamp.(0.5 .+ 0.05 .* randn(rng, 1, 4000), 1e-6, 1 - 1e-6)
    @test compute_discrepancy(KSDiscrepancy(), B) > compute_discrepancy(KSDiscrepancy(), U)
    @test compute_discrepancy(KLDiscrepancy(), B) > compute_discrepancy(KLDiscrepancy(), U)
    @test compute_discrepancy(SquaredErrorDiscrepancy(), B) >
        compute_discrepancy(SquaredErrorDiscrepancy(), U)

    # Multivariate uniform keeps KS and MMD small.
    U2 = rand(rng, 2, 3000)
    @test compute_discrepancy(KSDiscrepancy(), U2) < 0.05
    @test compute_discrepancy(MMDDiscrepancy(; block_size=3000), U2) < 1e-2
end

@testset "ACDC loss and selection" begin
    # K=2 fits well (both components below cutoff), K=3 has one bad component.
    r2 = ACDCResult(2, [0.01, 0.02], [0.5, 0.5])
    r3 = ACDCResult(3, [0.01, 0.02, 0.40], [0.4, 0.4, 0.2])

    @test acdc_loss(r2, 0.1) == 0.0
    @test acdc_loss(r3, 0.1) ≈ 0.30
    @test acdc_select([r2, r3], 0.1) == 2          # prefer smaller K at min loss
    @test acdc_select([r2, r3], 0.5) == 2          # both zero, ties go to smaller K

    crit = get_critical_rho_values([r2, r3])
    @test crit == sort(unique([0.01, 0.02, 0.01, 0.02, 0.40]))
end

@testset "HMM stochastic drivers (Normal emissions)" begin
    rng = Random.MersenneTwister(42)
    init = [0.34, 0.33, 0.33]
    trans = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]
    dists = [Normal(-6.0, 1.0), Normal(0.0, 1.0), Normal(6.0, 1.0)]
    hmm = HMM(init, trans, dists)
    _, obs_seq = rand(rng, hmm, 3000)

    sd = stochastic_drivers(hmm, obs_seq)
    @test length(sd.ε_pools) == 3
    @test all(p -> size(p, 1) == 1, sd.ε_pools)          # univariate, so D=1
    @test sum(size.(sd.ε_pools, 2)) == 3000              # every step assigned once
    @test isapprox(sum(sd.usage), 1.0; atol=1e-8)
    @test all(p -> all(0 .<= p .<= 1), sd.ε_pools)       # drivers in [0,1]

    # A well-specified model gives near-uniform drivers and small discrepancies.
    res = component_discrepancies(hmm, obs_seq, KSDiscrepancy())
    @test res.K == 3
    @test all(<(0.1), res.component_discrepancies)
end

@testset "HMM well-specified vs misspecified" begin
    rng = Random.MersenneTwister(7)
    init = [0.5, 0.5]
    trans = [0.95 0.05; 0.05 0.95]
    true_hmm = HMM(init, trans, [Normal(-4.0, 1.0), Normal(4.0, 1.0)])
    _, obs_seq = rand(rng, true_hmm, 3000)

    good = component_discrepancies(true_hmm, obs_seq, KLDiscrepancy())
    # Misspecified emission variances push the drivers away from uniform.
    bad_hmm = HMM(init, trans, [Normal(-4.0, 3.0), Normal(4.0, 3.0)])
    bad = component_discrepancies(bad_hmm, obs_seq, KLDiscrepancy())

    @test maximum(bad.component_discrepancies) > maximum(good.component_discrepancies)
end

@testset "HMM discrete (Poisson) and multivariate (MvNormal) emissions" begin
    rng = Random.MersenneTwister(11)
    init = [0.5, 0.5]
    trans = [0.9 0.1; 0.1 0.9]

    pois = HMM(init, trans, [Poisson(2.0), Poisson(20.0)])
    _, pobs = rand(rng, pois, 2500)
    pres = component_discrepancies(pois, pobs, KSDiscrepancy())
    @test pres.K == 2
    @test all(<(0.1), pres.component_discrepancies)

    mv = HMM(
        init,
        trans,
        [MvNormal([-3.0, -3.0], I(2)), MvNormal([3.0, 3.0], [1.0 0.5; 0.5 1.0])],
    )
    _, mobs = rand(rng, mv, 2500)
    sd = stochastic_drivers(mv, mobs)
    @test all(p -> size(p, 1) == 2, sd.ε_pools)          # bivariate, so D=2
    mres = component_discrepancies(mv, mobs, KSDiscrepancy())
    @test all(<(0.1), mres.component_discrepancies)
end

@testset "Driver recovery for PoissonZeroInflated and MultivariateT" begin
    rng = Random.MersenneTwister(2024)
    EM = EmissionModels

    # ZIP: randomized PIT drivers are ~U(0,1) under the true model.
    zip = PoissonZeroInflated(3.0, 0.25)
    zeps = [EM._emission_to_driver(rng, zip, rand(rng, zip))[1] for _ in 1:40_000]
    @test all(0 .<= zeps .<= 1)
    @test isapprox(Statistics.mean(zeps), 0.5; atol=0.02)
    @test isapprox(Statistics.var(zeps), 1 / 12; atol=0.01)
    @test compute_discrepancy(KSDiscrepancy(), reshape(zeps, 1, :)) < 0.05

    Σ = [1.0 0.5 0.2; 0.5 2.0 0.3; 0.2 0.3 1.5]
    mvt = MultivariateT([1.0, -2.0, 0.5], Σ, 6.0)
    E = reduce(hcat, (EM._emission_to_driver(rng, mvt, rand(rng, mvt)) for _ in 1:40_000))
    @test size(E, 1) == 3
    @test all(0 .<= E .<= 1)
    @test all(isapprox.(vec(Statistics.mean(E; dims=2)), 0.5; atol=0.02))
    @test compute_discrepancy(KSDiscrepancy(), E) < 0.05
    @test abs(Statistics.cov(E[1, :], E[2, :])) < 0.01   # independence, not just uniform marginals
    @test abs(Statistics.cov(E[1, :], E[3, :])) < 0.01

    # Diagonal scale variant.
    mvtd = MultivariateTDiag([0.0, 1.0], [1.0, 3.0], 4.0)
    Ed = reduce(
        hcat, (EM._emission_to_driver(rng, mvtd, rand(rng, mvtd)) for _ in 1:40_000)
    )
    @test all(0 .<= Ed .<= 1)
    @test compute_discrepancy(KSDiscrepancy(), Ed) < 0.05
    @test abs(Statistics.cov(Ed[1, :], Ed[2, :])) < 0.01
end

@testset "Driver recovery for GLM emissions (conditional PIT)" begin
    rng = Random.MersenneTwister(99)
    EM = EmissionModels
    N = 40_000
    mkx() = [1.0, randn(rng), randn(rng)]   # intercept + two covariates

    # Univariate GLMs: reduced to Normal / Bernoulli / Poisson, then PIT.
    for g in (
        GaussianGLM([0.5, 1.0, -0.7], 2.0),
        BernoulliGLM([0.2, 1.5, -1.0]),
        PoissonGLM([0.3, 0.8, -0.4]),
    )
        e = Float64[]
        for _ in 1:N
            x = mkx()
            push!(e, EM._emission_to_driver(rng, g, rand(rng, g; control_seq=x), x)[1])
        end
        @test all(0 .<= e .<= 1)
        @test isapprox(Statistics.mean(e), 0.5; atol=0.02)
        @test compute_discrepancy(KSDiscrepancy(), reshape(e, 1, :)) < 0.05
    end

    # MvGaussianGLM
    mg = MvGaussianGLM([0.5 -0.3; 1.0 0.2; -0.6 0.7], [1.0 0.6; 0.6 1.5])
    Eg = Matrix{Float64}(undef, 2, N)
    for i in 1:N
        x = mkx()
        Eg[:, i] = EM._emission_to_driver(rng, mg, rand(rng, mg; control_seq=x), x)
    end
    @test compute_discrepancy(KSDiscrepancy(), Eg) < 0.05
    @test abs(Statistics.cov(Eg[1, :], Eg[2, :])) < 0.01

    # Independent-by-column multivariate GLMs: stacked per-column PITs.
    mb = MvBernoulliGLM([0.1 -0.2; 1.0 0.5; -0.8 0.3])
    mp = MvPoissonGLM([0.2 0.1; 0.6 -0.3; -0.4 0.5])
    for g in (mb, mp)
        E = Matrix{Float64}(undef, 2, N)
        for i in 1:N
            x = mkx()
            E[:, i] = EM._emission_to_driver(rng, g, rand(rng, g; control_seq=x), x)
        end
        @test all(0 .<= E .<= 1)
        @test compute_discrepancy(KSDiscrepancy(), E) < 0.05
        @test abs(Statistics.cov(E[1, :], E[2, :])) < 0.01
    end

    # The covariate-free form on a GLM errors; a covariate is required.
    @test_throws ArgumentError EM._emission_to_driver(rng, GaussianGLM([0.5], 1.0), 1.0)
end

@testset "Discrepancy edge cases" begin
    rng = Random.MersenneTwister(123)

    # KL with too few samples (N < k+1) returns Inf.
    @test compute_discrepancy(KLDiscrepancy(; k_neighbors=5), rand(rng, 2, 3)) == Inf

    # Multivariate paths: SquaredError cross-covariance loop, sliced Wasserstein.
    U3 = rand(rng, 3, 1500)
    @test compute_discrepancy(SquaredErrorDiscrepancy(), U3) < 1e-2
    @test compute_discrepancy(WassersteinDiscrepancy(), U3) < 0.1

    # MMD block-averaging path (N > block_size).
    @test compute_discrepancy(MMDDiscrepancy(; block_size=200), rand(rng, 2, 600)) < 0.05

    # Type stability through a parametric discrepancy.
    @test compute_discrepancy(KSDiscrepancy{Float32}(), rand(rng, Float32, 1, 200)) isa
        Float32

    # Unsupported emission gives a clear error.
    @test_throws ArgumentError EmissionModels._emission_to_driver(rng, missing, 1.0)

    #= Unsupported model hits the stochastic_drivers fallback (also via the
       component_discrepancies path, which forwards to stochastic_drivers). =#
    @test_throws ArgumentError stochastic_drivers(missing, [1.0, 2.0])
    @test_throws ArgumentError component_discrepancies(missing, [1.0, 2.0], KSDiscrepancy())
end

@testset "HMM adapter argument validation" begin
    rng = Random.MersenneTwister(5)
    hmm = HMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], [Normal(0.0, 1.0), Normal(5.0, 1.0)])
    _, obs_seq = rand(rng, hmm, 100)

    @test_throws ArgumentError stochastic_drivers(hmm, obs_seq; n_samples=0)
    @test_throws ArgumentError stochastic_drivers(hmm, eltype(obs_seq)[])

    # n_samples > 1 multiplies the pool size (one assignment per step per pass).
    sd = stochastic_drivers(hmm, obs_seq; n_samples=2)
    @test sum(size.(sd.ε_pools, 2)) == 200
end

@testset "Reproducibility with a seeded rng" begin
    rng = Random.MersenneTwister(31)
    hmm = HMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], [Poisson(2.0), Poisson(15.0)])
    _, obs_seq = rand(rng, hmm, 500)

    # The same seed reproduces identical drivers and discrepancies.
    sd1 = stochastic_drivers(hmm, obs_seq; rng=Random.MersenneTwister(7))
    sd2 = stochastic_drivers(hmm, obs_seq; rng=Random.MersenneTwister(7))
    @test sd1.ε_pools == sd2.ε_pools
    @test sd1.usage == sd2.usage

    r1 = component_discrepancies(
        hmm, obs_seq, KSDiscrepancy(); rng=Random.MersenneTwister(7)
    )
    r2 = component_discrepancies(
        hmm, obs_seq, KSDiscrepancy(); rng=Random.MersenneTwister(7)
    )
    @test r1.component_discrepancies == r2.component_discrepancies

    # Monte Carlo discrepancies are reproducible under a seeded rng too.
    U = rand(Random.MersenneTwister(1), 2, 500)
    w1 = compute_discrepancy(WassersteinDiscrepancy(), U; rng=Random.MersenneTwister(3))
    w2 = compute_discrepancy(WassersteinDiscrepancy(), U; rng=Random.MersenneTwister(3))
    @test w1 == w2
    m1 = compute_discrepancy(MMDDiscrepancy(), U; rng=Random.MersenneTwister(3))
    m2 = compute_discrepancy(MMDDiscrepancy(), U; rng=Random.MersenneTwister(3))
    @test m1 == m2
end

@testset "Discrepancies accept any Real eltype" begin
    rng = Random.MersenneTwister(6)
    U32 = rand(rng, Float32, 2, 500)
    # Default (Float64) discrepancies on Float32 pools compute at Float64.
    @test compute_discrepancy(KSDiscrepancy(), U32) isa Float64
    @test compute_discrepancy(KLDiscrepancy(), U32) isa Float64
    @test compute_discrepancy(SquaredErrorDiscrepancy(), U32) isa Float64
    @test compute_discrepancy(WassersteinDiscrepancy(), U32) isa Float64
    @test compute_discrepancy(MMDDiscrepancy(), U32) isa Float64
end

@testset "Empty driver pools score Inf" begin
    rng = Random.MersenneTwister(17)

    #= Direct scoring path: a never-sampled component has a D × 0 pool, on
       which the raw measures throw or return NaN; component_discrepancies
       must map it to Inf for every measure. =#
    pools = [rand(rng, 2, 300), Matrix{Float64}(undef, 2, 0)]
    sd = StochasticDriverResult(pools, [1.0, 0.0])
    for disc in (
        KSDiscrepancy(),
        KLDiscrepancy(),
        SquaredErrorDiscrepancy(),
        WassersteinDiscrepancy(),
        MMDDiscrepancy(),
    )
        res = component_discrepancies(sd, disc; rng=Random.MersenneTwister(2))
        @test res.component_discrepancies[2] == Inf
        @test isfinite(res.component_discrepancies[1])
    end

    #= End-to-end: an unreachable state (the K-too-large scenario ACDC scans
       for) never gets sampled, scores Inf, and pushes selection to the
       smaller K instead of poisoning it with NaN. =#
    hmm2 = HMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], [Normal(-4.0, 1.0), Normal(4.0, 1.0)])
    hmm3 = HMM(
        [0.5, 0.5, 0.0],
        [0.9 0.1 0.0; 0.1 0.9 0.0; 0.0 0.0 1.0],
        [Normal(-4.0, 1.0), Normal(4.0, 1.0), Normal(0.0, 1.0)],
    )
    _, obs_seq = rand(Random.MersenneTwister(4), hmm2, 2000)

    r2 = component_discrepancies(hmm2, obs_seq, KSDiscrepancy(); rng=rng)
    r3 = component_discrepancies(hmm3, obs_seq, KSDiscrepancy(); rng=rng)
    @test r3.component_discrepancies[3] == Inf
    @test acdc_loss(r3, 0.1) == Inf
    @test acdc_select([r2, r3], 0.1) == 2
end

@testset "MMD blocks keep the tail; sliced Wasserstein projections" begin
    rng = Random.MersenneTwister(23)

    #= N = 401 with block_size = 200 previously dropped the last sample block
       remainder; near-equal partitioning must consume all samples and stay
       finite and small on uniform input. =#
    m = compute_discrepancy(
        MMDDiscrepancy(; block_size=200), rand(rng, 2, 401); rng=Random.MersenneTwister(2)
    )
    @test isfinite(m)
    @test 0 <= m < 0.05

    # The clamp keeps the estimate non-negative on both code paths.
    @test compute_discrepancy(
        MMDDiscrepancy(; block_size=500), rand(rng, 1, 400); rng=Random.MersenneTwister(2)
    ) >= 0

    # n_projections is honored (and validated).
    @test compute_discrepancy(
        WassersteinDiscrepancy(; n_projections=5),
        rand(rng, 3, 500);
        rng=Random.MersenneTwister(2),
    ) < 0.1
    @test_throws ArgumentError WassersteinDiscrepancy(; n_projections=0)
end
