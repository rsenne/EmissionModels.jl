#=
  AirspeedVelocity.jl benchmark suite. Covers the three hot paths hit inside
  an HMM EM loop i.e., logdensityof (every forward/backward step), rand/rand!
  (simulation), and fit! (per state per outer iteration) for each emission
  model. 
=#
include(joinpath(@__DIR__, "utils.jl"))

using LinearAlgebra: I
# Enables the DDM emissions, which live in a package extension.
using SequentialSamplingModels

const SUITE = BenchmarkGroup()

rng = Random.MersenneTwister(0)
N = 500
cfg = (rng=rng, x=[1.0, 2.0], X=hcat(ones(N), randn(rng, N)), w=ones(N))

MODELS = [
    (
        name="GaussianGLM",
        model=GaussianGLM([0.5, -1.0], 1.0),
        fresh=() -> GaussianGLM(zeros(2), 1.0),
        controlled=true,
        buffer=nothing,
    ),
    (
        name="BernoulliGLM",
        model=BernoulliGLM([0.5, -1.0]),
        fresh=() -> BernoulliGLM(zeros(2)),
        controlled=true,
        buffer=nothing,
    ),
    (
        name="PoissonGLM",
        model=PoissonGLM([0.5, -1.0]),
        fresh=() -> PoissonGLM(zeros(2)),
        controlled=true,
        buffer=nothing,
    ),
    (
        # Three categories, so B is 2×2 and observations are length-3 counts.
        name="MultinomialGLM",
        model=MultinomialGLM([0.5 -1.0; 1.0 0.5], 5),
        fresh=() -> MultinomialGLM(zeros(2, 2), 5),
        controlled=true,
        buffer=zeros(Int, 3),
    ),
    (
        name="MvGaussianGLM",
        model=MvGaussianGLM([0.5 -1.0; 1.0 0.5], [1.0 0.3; 0.3 1.5]),
        fresh=() -> MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2)),
        controlled=true,
        buffer=zeros(2),
    ),
    (
        name="MvBernoulliGLM",
        model=MvBernoulliGLM([0.5 -1.0; 1.0 0.5]),
        fresh=() -> MvBernoulliGLM(zeros(2, 2)),
        controlled=true,
        buffer=zeros(Int, 2),
    ),
    (
        name="MvPoissonGLM",
        model=MvPoissonGLM([0.5 -1.0; 0.2 0.0]),
        fresh=() -> MvPoissonGLM(zeros(2, 2)),
        controlled=true,
        buffer=zeros(Int, 2),
    ),
    (
        name="PoissonZeroInflated",
        model=PoissonZeroInflated(3.0, 0.2),
        fresh=() -> PoissonZeroInflated(1.0, 0.1),
        controlled=false,
        buffer=nothing,
    ),
    (
        name="MvT",
        model=MvT([0.0, 0.0], [1.0 0.3; 0.3 1.0], 5.0),
        fresh=() -> MvT([0.0, 0.0], Matrix(1.0I, 2, 2), 5.0),
        controlled=false,
        buffer=zeros(2),
    ),
    (
        name="MvTDiag",
        model=MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0),
        fresh=() -> MvTDiag([0.0, 0.0], [1.0, 1.0], 5.0),
        controlled=false,
        buffer=zeros(2),
    ),
]

for spec in MODELS
    add_model_benchmarks!(SUITE, spec, cfg)
end

cal_c = [1.0, 1.0]
cal_dists = calcium_emissions([0.15 1.60; 1.70 0.20], [0.90 0.85], cal_c, [0.02, 0.02])
cal_sim = rand_calcium(rng, [0.5, 0.5], [0.97 0.03; 0.03 0.97], cal_dists, N)
add_ctrl_seq_benchmarks!(
    SUITE,
    (
        name="CalciumEmission",
        model=first(cal_dists),
        fresh=() -> only(
            calcium_emissions(fill(0.5, 2, 1), [0.50 0.50], cal_c, [0.1, 0.1]; tied=false),
        ),
        buffer=zeros(2),
        obs=cal_sim.obs_seq,
        controls=cal_sim.control_seq,
    ),
    cfg,
)

#= The large-rate path replaces the truncated Poisson sum by an analytic
   Gaussian marginal, so its density cost is O(1) rather than O(R) per neuron.
   Tracked separately because a regression there is invisible in the exact path. =#
cal_g_dists = calcium_emissions(
    [12.0 60.0; 70.0 15.0], [0.90 0.85], cal_c, [0.02, 0.02]; gaussian=true
)
cal_g_sim = rand_calcium(rng, [0.5, 0.5], [0.97 0.03; 0.03 0.97], cal_g_dists, N)
add_ctrl_seq_benchmarks!(
    SUITE,
    (
        name="CalciumEmissionGaussian",
        model=first(cal_g_dists),
        fresh=() -> only(
            calcium_emissions(
                fill(30.0, 2, 1),
                [0.50 0.50],
                cal_c,
                [0.1, 0.1];
                tied=false,
                gaussian=true,
            ),
        ),
        buffer=zeros(2),
        obs=cal_g_sim.obs_seq,
        controls=cal_g_sim.control_seq,
    ),
    cfg,
)

# Stimulus codes ±1; the DDM's observations are (choice, rt) pairs.
ddm_codes = rand(rng, (-1.0, 1.0), N)
ddm_model = StimulusCodedDDM(; ν=2.0, α=1.0, z=0.5, τ=0.3)
add_ctrl_seq_benchmarks!(
    SUITE,
    (
        name="StimulusCodedDDM",
        model=ddm_model,
        fresh=() -> StimulusCodedDDM(; ν=1.0, α=1.0, z=0.5, τ=0.2),
        buffer=nothing,
        obs=[rand(rng, ddm_model, s) for s in ddm_codes],
        controls=ddm_codes,
    ),
    cfg,
)

# Signed coherences, two strengths per side plus a near-threshold pair.
coh = rand(rng, (-0.5, -0.25, -0.06, 0.06, 0.25, 0.5), N)
coh_model = CoherenceDDM(; k=8.0, γ=0.7, α=1.2, z=0.5, τ=0.25)
add_ctrl_seq_benchmarks!(
    SUITE,
    (
        name="CoherenceDDM",
        model=coh_model,
        fresh=() -> CoherenceDDM(; k=4.0, γ=1.0, α=1.0, z=0.5, τ=0.2),
        buffer=nothing,
        obs=[rand(rng, coh_model, c) for c in coh],
        controls=coh,
    ),
    cfg,
)
