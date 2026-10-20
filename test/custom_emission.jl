using HiddenMarkovModels
using HiddenMarkovModels: ControlledEmission, ControlledEmissionHMM, baum_welch
using DensityInterface
using StatsAPI
using Random
using Test

# Guards the controlled-emission recipe in docs/src/custom.md; keep the two in sync.
mutable struct DocsLinearGaussian{T<:Real} <: ControlledEmission
    β::T
    σ::T
end

function DensityInterface.logdensityof(d::DocsLinearGaussian, obs::Real, control::Real)
    z = (obs - d.β * control) / d.σ
    return -(z^2 + log(2π)) / 2 - log(d.σ)
end

function Random.rand(rng::AbstractRNG, d::DocsLinearGaussian, control::Real)
    return d.β * control + d.σ * randn(rng)
end

function StatsAPI.fit!(d::DocsLinearGaussian, obs_seq, control_seq, weight_seq)
    sxy = sum(w * u * y for (w, u, y) in zip(weight_seq, control_seq, obs_seq))
    sxx = sum(w * u^2 for (w, u) in zip(weight_seq, control_seq))
    d.β = sxy / sxx
    ssr = sum(w * (y - d.β * u)^2 for (w, u, y) in zip(weight_seq, control_seq, obs_seq))
    d.σ = sqrt(ssr / sum(weight_seq))
    return d
end

@testset "Custom ControlledEmission (docs recipe)" begin
    rng = Xoshiro(94)
    trans = [0.95 0.05; 0.05 0.95]
    hmm_true = ControlledEmissionHMM(
        [0.5, 0.5], trans, [DocsLinearGaussian(1.0, 0.5), DocsLinearGaussian(-2.0, 0.5)]
    )
    control_seq = randn(rng, 2000)
    _, obs_seq = rand(rng, hmm_true, control_seq)

    hmm_init = ControlledEmissionHMM(
        [0.5, 0.5],
        [0.9 0.1; 0.1 0.9],
        [DocsLinearGaussian(0.5, 1.0), DocsLinearGaussian(-1.0, 1.0)],
    )
    hmm_est, loglik = baum_welch(hmm_init, obs_seq, control_seq; max_iterations=20)

    @test all(diff(loglik) .>= -1e-8)
    @test hmm_est.dists[1].β != 0.5
    @test isapprox(hmm_est.dists[1].β, 1.0; atol=0.1)
    @test isapprox(hmm_est.dists[2].β, -2.0; atol=0.1)
end
