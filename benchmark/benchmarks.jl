#=
  AirspeedVelocity.jl benchmark suite. Covers the three hot paths hit inside
  an HMM EM loop i.e., logdensityof (every forward/backward step), rand/rand!
  (simulation), and fit! (per state per outer iteration) for each emission
  model. 
=#
include(joinpath(@__DIR__, "utils.jl"))

using LinearAlgebra: I

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
