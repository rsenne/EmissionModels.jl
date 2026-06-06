using EmissionModels
using HiddenMarkovModels
using Distributions
using LinearAlgebra
using Random
using Test

@testset "Discrepancy measures on (non)uniform samples" begin
    rng = Random.MersenneTwister(0)

    # Uniform drivers ⇒ every discrepancy is small.
    U = rand(rng, 1, 4000)
    @test compute_discrepancy(KSDiscrepancy(), U) < 0.05
    @test compute_discrepancy(SquaredErrorDiscrepancy(), U) < 1e-2
    @test compute_discrepancy(WassersteinDiscrepancy(), U) < 0.05
    @test abs(compute_discrepancy(KLDiscrepancy(), U)) < 0.1
    @test abs(compute_discrepancy(MMDDiscrepancy(; block_size=4000), U)) < 1e-2

    # Strongly non-uniform drivers (concentrated near 0.5) ⇒ larger scores.
    B = clamp.(0.5 .+ 0.05 .* randn(rng, 1, 4000), 1e-6, 1 - 1e-6)
    @test compute_discrepancy(KSDiscrepancy(), B) > compute_discrepancy(KSDiscrepancy(), U)
    @test compute_discrepancy(KLDiscrepancy(), B) > compute_discrepancy(KLDiscrepancy(), U)
    @test compute_discrepancy(SquaredErrorDiscrepancy(), B) >
        compute_discrepancy(SquaredErrorDiscrepancy(), U)

    # Multivariate uniform ⇒ small KS / MMD.
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
    @test acdc_select([r2, r3], 0.5) == 2          # both zero ⇒ smaller K

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
    @test all(p -> size(p, 1) == 1, sd.ε_pools)          # univariate ⇒ D=1
    @test sum(size.(sd.ε_pools, 2)) == 3000              # every step assigned once
    @test isapprox(sum(sd.usage), 1.0; atol=1e-8)
    @test all(p -> all(0 .<= p .<= 1), sd.ε_pools)       # drivers in [0,1]

    # Well-specified model ⇒ drivers ≈ uniform ⇒ small per-state discrepancy.
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
    # Misspecified emission variances ⇒ drivers deviate from uniform.
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
    @test all(p -> size(p, 1) == 2, sd.ε_pools)          # bivariate ⇒ D=2
    mres = component_discrepancies(mv, mobs, KSDiscrepancy())
    @test all(<(0.1), mres.component_discrepancies)
end

@testset "Discrepancy edge cases" begin
    rng = Random.MersenneTwister(123)

    # KL with too few samples (N < k+1) ⇒ Inf.
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

    # Unsupported emission ⇒ clear error.
    @test_throws ArgumentError EmissionModels._emission_to_driver(missing, 1.0)

    # Unsupported model ⇒ stochastic_drivers fallback errors (also via the
    # component_discrepancies path, which forwards to stochastic_drivers).
    @test_throws ArgumentError stochastic_drivers(missing, [1.0, 2.0])
    @test_throws ArgumentError component_discrepancies(missing, [1.0, 2.0], KSDiscrepancy())
end

@testset "HMM adapter argument validation" begin
    rng = Random.MersenneTwister(5)
    hmm = HMM([0.5, 0.5], [0.9 0.1; 0.1 0.9], [Normal(0.0, 1.0), Normal(5.0, 1.0)])
    _, obs_seq = rand(rng, hmm, 100)

    @test_throws ArgumentError stochastic_drivers(hmm, obs_seq; n_samples=0)

    # n_samples > 1 multiplies the pool size (one assignment per step per pass).
    sd = stochastic_drivers(hmm, obs_seq; n_samples=2)
    @test sum(size.(sd.ε_pools, 2)) == 200
end
