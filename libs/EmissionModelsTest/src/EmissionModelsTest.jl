#=
  Shared test recipes for EmissionModels.jl, in the spirit of
  HiddenMarkovModels.jl's libs/HMMTest. Unregistered; the test suite
  `Pkg.develop`s it at runtime (see test/runtests.jl).
=#
module EmissionModelsTest

using DensityInterface: logdensityof
using Distributions: Dirichlet
using EmissionModels
using HiddenMarkovModels: HMM, baum_welch, forward
using LinearAlgebra: I
using Random: Random, AbstractRNG, rand!
using Test: @test

export create_hmm, create_emissions
export test_hmm_integration
export ALLOC_SLOP
export bench_logd, bench_logd_unctrl, bench_logd_ddm
export bench_rand_scalar, bench_rand_int, bench_rand_ddm
export bench_rand_unctrl_scalar, bench_rand_unctrl_vec
export bench_rand!_v, bench_rand!_i, bench_rand!_unctrl

include("hmm.jl")
include("integration.jl")
include("allocations.jl")

end
