using EmissionModels
using EmissionModels: stochastic_drivers, component_discrepancies, KSDiscrepancy
using SequentialSamplingModels
using SequentialSamplingModels: DDM
using HiddenMarkovModels: ControlledEmission, ControlledEmissionHMM, baum_welch, forward
using DensityInterface
using StatsAPI
using Statistics: mean, var
using Random
using Test

const EM = EmissionModels

@testset "DDM emissions (SequentialSamplingModels extension)" begin
    @testset "construction and validation" begin
        d = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.4, τ=0.3)
        @test d isa StimulusCodedDDM{Float64}
        @test (d.ν, d.α, d.z, d.τ) == (2.0, 1.0, 0.4, 0.3)

        # promoting positional constructor
        @test StimulusCodedDDM(2, 1, 0.5, 0.3) isa StimulusCodedDDM{Float64}

        c = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
        @test c isa CoherenceDDM{Float64}
        @test (c.k, c.γ, c.α, c.z, c.τ) == (8.0, 0.7, 1.2, 0.5, 0.25)

        # gains are magnitudes: the control carries the sign
        @test_throws ArgumentError StimulusCodedDDM(; ν=-1.0)
        @test_throws ArgumentError StimulusCodedDDM(; ν=0.0)
        @test_throws ArgumentError CoherenceDDM(; k=-1.0)
        @test_throws ArgumentError CoherenceDDM(; k=0.0)
        @test_throws ArgumentError StimulusCodedDDM(; α=-1.0)
        @test_throws ArgumentError StimulusCodedDDM(; z=0.0)
        @test_throws ArgumentError StimulusCodedDDM(; z=1.0)
        @test_throws ArgumentError StimulusCodedDDM(; τ=-0.1)
        @test_throws ArgumentError CoherenceDDM(; γ=0.0)

        for D in (StimulusCodedDDM, CoherenceDDM)
            @test D <: ControlledEmission
        end
    end

    @testset "logdensityof matches SequentialSamplingModels" begin
        d = StimulusCodedDDM(; ν=1.5, α=1.0, z=0.4, τ=0.2)
        # stimulus code multiplies the drift gain
        @test logdensityof(d, (1, 0.6), 1.0) ≈ logpdf(DDM(1.5, 1.0, 0.4, 0.2), 1, 0.6)
        @test logdensityof(d, (2, 0.6), -1.0) ≈ logpdf(DDM(-1.5, 1.0, 0.4, 0.2), 2, 0.6)
        # zero-drift no-signal trial
        @test logdensityof(d, (1, 0.6), 0) ≈ logpdf(DDM(0.0, 1.0, 0.4, 0.2), 1, 0.6)

        # NamedTuple observations (as returned by rand) index positionally
        @test logdensityof(d, (; choice=1, rt=0.6), 1.0) == logdensityof(d, (1, 0.6), 1.0)

        # coherence model: v = k * sign(c) * |c|^γ
        c = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
        v = 8.0 * 0.256^0.7
        @test logdensityof(c, (1, 0.7), 0.256) ≈ logpdf(DDM(v, 1.2, 0.5, 0.25), 1, 0.7)
        @test logdensityof(c, (2, 0.7), -0.256) ≈ logpdf(DDM(-v, 1.2, 0.5, 0.25), 2, 0.7)

        #= stimulus-coding mirror symmetry: with no side bias (z = 0.5),
           hitting the upper boundary under s = +1 is as likely as hitting
           the lower boundary under s = -1 =#
        du = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.2)
        for rt in (0.3, 0.5, 1.0)
            @test logdensityof(du, (1, rt), 1.0) ≈ logdensityof(du, (2, rt), -1.0)
        end

        # zero-density observations return -Inf instead of throwing
        @test logdensityof(d, (1, 0.1), 1.0) == -Inf   # rt < τ
        @test logdensityof(d, (1, 0.2), 1.0) == -Inf   # rt == τ
        @test logdensityof(d, (3, 0.6), 1.0) == -Inf   # invalid choice
    end

    @testset "_ddm_prob_upper closed form" begin
        # matches the t → ∞ limit of the defective CDF computed by SSM
        for (ν, α, z) in
            ((1.5, 1.0, 0.4), (-2.0, 0.8, 0.6), (0.3, 2.0, 0.5), (-0.1, 1.2, 0.3))
            @test EM._ddm_prob_upper(ν, α, z) ≈
                SequentialSamplingModels.cdf(DDM(ν, α, z, 0.0), 1, 1e5) atol = 1e-8
        end
        # zero drift: P(upper) is the relative start point
        @test EM._ddm_prob_upper(0.0, 1.0, 0.3) == 0.3
        # extreme drifts saturate without overflow (a naive expm1 ratio NaNs)
        @test EM._ddm_prob_upper(500.0, 2.0, 0.5) ≈ 1.0
        @test EM._ddm_prob_upper(-500.0, 2.0, 0.5) ≈ 0.0 atol = 1e-12
        @test EM._ddm_prob_upper(1.5f0, 1.0f0, 0.4f0) isa Float32
    end

    @testset "_ddm_cdf guards" begin
        ν, α, z, τ = 1.5, 1.0, 0.4, 0.2
        # valid calls defer to SSM, whatever Real type carries the choice
        ref = SequentialSamplingModels.cdf(DDM(ν, α, z, τ), 1, 0.6)
        @test EM._ddm_cdf(ν, α, z, τ, 1, 0.6) == clamp(ref, 0.0, 1.0)
        @test EM._ddm_cdf(ν, α, z, τ, 1.0, 0.6) == EM._ddm_cdf(ν, α, z, τ, 1, 0.6)
        # cases SSM's cdf throws on carry zero mass instead
        @test EM._ddm_cdf(ν, α, z, τ, 3, 0.6) == 0.0      # invalid choice
        @test EM._ddm_cdf(ν, α, z, τ, 1.5, 0.6) == 0.0    # non-integral choice
        @test EM._ddm_cdf(ν, α, z, τ, 1, 0.1) == 0.0      # rt < τ
        @test EM._ddm_cdf(ν, α, z, τ, 1, τ) == 0.0        # rt == τ
        # defective CDF: monotone in rt, saturating at the boundary-hit mass
        Fs = [EM._ddm_cdf(ν, α, z, τ, 1, rt) for rt in (0.25, 0.4, 0.8, 2.0, 10.0)]
        @test issorted(Fs)
        @test all(0 .≤ Fs .≤ 1)
        @test last(Fs) ≈ EM._ddm_prob_upper(ν, α, z) atol = 1e-6
    end

    @testset "type stability" begin
        rng = MersenneTwister(0)
        d = StimulusCodedDDM(; ν=1.5, α=1.0, z=0.4, τ=0.2)
        c = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
        @test @inferred(logdensityof(d, (1, 0.6), 1.0)) isa Float64
        @test @inferred(logdensityof(c, (2, 0.7), -0.256)) isa Float64
        @test @inferred(EM._ddm_cdf(1.5, 1.0, 0.4, 0.2, 1, 0.6)) isa Float64
        @test @inferred(EM._ddm_prob_upper(1.5, 1.0, 0.4)) isa Float64
        @test @inferred(EM._emission_to_driver(rng, d, (1, 0.6), 1.0)) isa Vector{Float64}
    end

    @testset "rand" begin
        rng = MersenneTwister(0)
        d = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.3)
        for control in (1.0, -1.0, 1, -1)   # Real and Integer control paths
            obs = rand(rng, d, control)
            @test obs.choice in (1, 2)
            @test obs.rt > d.τ
            @test isfinite(logdensityof(d, obs, control))
        end

        # a strong positive drift should mostly hit the upper boundary
        n_upper = count(_ -> rand(rng, d, 1.0).choice == 1, 1:200)
        @test n_upper > 150
    end

    @testset "fit! recovers StimulusCodedDDM parameters" begin
        rng = MersenneTwister(1)
        ν, α, z, τ = 2.0, 1.2, 0.55, 0.30
        truth = StimulusCodedDDM(; ν, α, z, τ)
        n = 1500
        controls = [rand(rng, (-1.0, 1.0)) for _ in 1:n]
        obs = [rand(rng, truth, controls[i]) for i in 1:n]
        w = ones(n)

        d = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.15)
        fit!(d, obs, w; control_seq=controls)

        @test isapprox(d.ν, ν; rtol=0.15)
        @test isapprox(d.α, α; rtol=0.15)
        @test isapprox(d.z, z; atol=0.05)
        @test isapprox(d.τ, τ; atol=0.05)

        # the fit must improve the weighted log-likelihood over the start
        ll(m) = sum(logdensityof(m, obs[i], controls[i]) for i in 1:n)
        @test ll(d) > ll(StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.15))
    end

    @testset "fit! recovers CoherenceDDM drift function" begin
        rng = MersenneTwister(2)
        truth = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
        levels = (0.032, 0.064, 0.128, 0.256, 0.512)
        n = 2500
        controls = [rand(rng, (-1, 1)) * rand(rng, levels) for _ in 1:n]
        obs = [rand(rng, truth, controls[i]) for i in 1:n]
        w = ones(n)

        d = CoherenceDDM(; k=4.0, γ=1.0, α=1.0, z=0.5, τ=0.15)
        fit!(d, obs, w; control_seq=controls)

        #= k and γ trade off against each other, so compare the drift
           function they parameterize at the sampled coherence levels
           rather than the raw parameters =#
        for c in levels
            v_true = 8.0 * c^0.7
            v_fit = d.k * c^d.γ
            @test isapprox(v_fit, v_true; rtol=0.2)
        end
        @test isapprox(d.α, 1.2; rtol=0.15)
        @test isapprox(d.τ, 0.25; atol=0.05)
    end

    @testset "fit! weights and edge cases" begin
        rng = MersenneTwister(3)
        truth = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.25)
        n = 400
        controls = [rand(rng, (-1.0, 1.0)) for _ in 1:n]
        obs = [rand(rng, truth, controls[i]) for i in 1:n]

        # zero-weight observations must not affect the fit (even invalid ones)
        obs_dirty = vcat(obs, [(3, -1.0)])
        controls_dirty = vcat(controls, [1.0])
        w_dirty = vcat(ones(n), [0.0])
        d_clean = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.1)
        d_dirty = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.1)
        fit!(d_clean, obs, ones(n); control_seq=controls)
        fit!(d_dirty, obs_dirty, w_dirty; control_seq=controls_dirty)
        @test (d_dirty.ν, d_dirty.α, d_dirty.z, d_dirty.τ) ==
            (d_clean.ν, d_clean.α, d_clean.z, d_clean.τ)

        # all-zero weights leave the parameters unchanged
        d0 = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.1)
        fit!(d0, obs, zeros(n); control_seq=controls)
        @test (d0.ν, d0.α, d0.z, d0.τ) == (1.0, 0.8, 0.5, 0.1)

        # positional ControlledEmission signature delegates to the keyword one
        d_pos = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.1)
        d_kw = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.1)
        fit!(d_pos, obs, controls, ones(n))
        fit!(d_kw, obs, ones(n); control_seq=controls)
        @test (d_pos.ν, d_pos.α, d_pos.z, d_pos.τ) == (d_kw.ν, d_kw.α, d_kw.z, d_kw.τ)

        # invalid weighted observations are rejected up front
        @test_throws ArgumentError fit!(
            StimulusCodedDDM(), [(3, 0.5)], [1.0]; control_seq=[1.0]
        )
        @test_throws ArgumentError fit!(
            StimulusCodedDDM(), [(1, -0.5)], [1.0]; control_seq=[1.0]
        )
        @test_throws DimensionMismatch fit!(
            StimulusCodedDDM(), obs, ones(n - 1); control_seq=controls
        )
        @test_throws DimensionMismatch fit!(
            StimulusCodedDDM(), obs, ones(n); control_seq=controls[1:(n - 1)]
        )

        # solver kwargs are accepted and a capped run stays well-defined
        d_cap = StimulusCodedDDM(; ν=1.0, α=0.8, z=0.5, τ=0.1)
        fit!(d_cap, obs, ones(n); control_seq=controls, max_iter=2, gtol=1e-3)
        @test all(isfinite, (d_cap.ν, d_cap.α, d_cap.z, d_cap.τ))
    end

    @testset "Float32 parameters" begin
        rng = MersenneTwister(5)
        truth = StimulusCodedDDM(; ν=2.0f0, α=1.0f0, z=0.5f0, τ=0.2f0)
        @test truth isa StimulusCodedDDM{Float32}

        n = 300
        controls = [rand(rng, (-1.0f0, 1.0f0)) for _ in 1:n]
        obs = [rand(rng, truth, controls[i]) for i in 1:n]

        d = StimulusCodedDDM(; ν=1.0f0, α=0.8f0, z=0.5f0, τ=0.1f0)
        ll(m) = sum(logdensityof(m, obs[i], controls[i]) for i in 1:n)
        ll0 = ll(d)
        #= SSM's sampler returns Float64 rts, so this exercises the mixed
           Float32-emission / Float64-observation path, which must promote
           rather than silently hit -Inf through SSM's series internals. =#
        @test isfinite(ll0)
        fit!(d, obs, ones(Float32, n); control_seq=controls, gtol=1e-4)
        @test typeof((d.ν, d.α, d.z, d.τ)) == NTuple{4,Float32}
        @test all(isfinite, (d.ν, d.α, d.z, d.τ))
        @test ll(d) > ll0
    end

    @testset "DDM-HMM: sample, forward, baum_welch" begin
        rng = MersenneTwister(42)
        T = 400
        init = [0.6, 0.4]
        trans = [0.95 0.05; 0.1 0.9]
        # an "engaged" state (strong drift, wide bounds) and a "lapse" state
        dists = [
            StimulusCodedDDM(; ν=2.5, α=1.2, z=0.5, τ=0.25),
            StimulusCodedDDM(; ν=0.3, α=0.7, z=0.5, τ=0.2),
        ]
        hmm = ControlledEmissionHMM(init, trans, dists)

        control_seq = [rand(rng, (-1.0, 1.0)) for _ in 1:T]
        obs_seq = rand(rng, hmm, control_seq).obs_seq
        @test length(obs_seq) == T
        @test all(o -> o.choice in (1, 2) && o.rt > 0, obs_seq)

        logL = last(forward(hmm, obs_seq, control_seq; seq_ends=[T]))
        @test all(isfinite, logL)

        # fit from a perturbed start; baum_welch must be (weakly) monotone
        dists0 = [
            StimulusCodedDDM(; ν=1.5, α=1.0, z=0.5, τ=0.15),
            StimulusCodedDDM(; ν=0.8, α=0.9, z=0.5, τ=0.15),
        ]
        hmm0 = ControlledEmissionHMM([0.5, 0.5], copy(trans), dists0)
        _, lls = baum_welch(hmm0, obs_seq, control_seq; seq_ends=[T], max_iterations=8)
        @test all(diff(lls) .>= -1e-6)
        @test last(lls) >= first(lls)
    end

    @testset "ACDC driver recovery" begin
        #= Under the true model the Rosenblatt drivers (choice PIT, RT PIT) are
           independent uniforms on (0,1): mean 1/2, variance 1/12. =#
        rng = MersenneTwister(7)
        for d in (
            StimulusCodedDDM(; ν=1.8, α=1.0, z=0.45, τ=0.2),
            CoherenceDDM(; k=6.0, γ=0.8, α=1.1, z=0.55, τ=0.25),
        )
            controls = d isa StimulusCodedDDM ? (-1.0, 1.0) : (-0.5, -0.2, 0.2, 0.5)
            N = 40_000
            E = Matrix{Float64}(undef, 2, N)
            for i in 1:N
                c = rand(rng, controls)
                obs = rand(rng, d, c)
                E[:, i] = EM._emission_to_driver(rng, d, obs, c)
            end
            @test all(0 .< E .< 1)
            for row in 1:2
                @test isapprox(mean(view(E, row, :)), 0.5; atol=0.02)
                @test isapprox(var(view(E, row, :)), 1 / 12; atol=0.01)
            end
        end
    end

    @testset "ACDC on a DDM-HMM (end-to-end)" begin
        #= Exercises the full stochastic_drivers path on a ControlledEmissionHMM,
           including the ControlBoundEmission unwrap. Drivers recovered under the
           generating model should look uniform, i.e. low discrepancy. =#
        rng = MersenneTwister(11)
        T = 3000
        trans = [0.95 0.05; 0.08 0.92]
        dists = [
            StimulusCodedDDM(; ν=2.5, α=1.2, z=0.5, τ=0.25),
            StimulusCodedDDM(; ν=0.8, α=0.8, z=0.5, τ=0.2),
        ]
        hmm = ControlledEmissionHMM([0.6, 0.4], trans, dists)
        control_seq = [rand(rng, (-1.0, 1.0)) for _ in 1:T]
        obs_seq = rand(rng, hmm, control_seq).obs_seq

        sd = stochastic_drivers(
            hmm, obs_seq; control_seq=control_seq, seq_ends=(T,), rng=MersenneTwister(3)
        )
        @test length(sd.ε_pools) == 2
        @test all(p -> size(p, 1) == 2, sd.ε_pools)          # (choice, rt) drivers
        @test all(p -> all(0 .< p .< 1), sd.ε_pools)
        @test isapprox(sum(sd.usage), 1; atol=1e-8)

        acdc = component_discrepancies(
            hmm,
            obs_seq,
            KSDiscrepancy();
            control_seq=control_seq,
            seq_ends=(T,),
            rng=MersenneTwister(3),
        )
        # well-specified emissions ⇒ near-uniform drivers ⇒ small KS discrepancy
        @test all(isfinite, acdc.component_discrepancies)
        @test maximum(acdc.component_discrepancies) < 0.1
    end

    @testset "extension fallback hooks" begin
        #= The src fallbacks hat a user without SequentialSamplingModels
           loaded hits for any argument types stay reachable  =#
        @test_throws "SequentialSamplingModels" EM._ddm_logpdf(
            nothing, nothing, nothing, nothing, nothing, nothing
        )
        @test_throws "SequentialSamplingModels" EM._ddm_rand(
            nothing, nothing, nothing, nothing, nothing
        )
        @test_throws "SequentialSamplingModels" EM._ddm_cdf(
            nothing, nothing, nothing, nothing, nothing, nothing
        )
    end
end
