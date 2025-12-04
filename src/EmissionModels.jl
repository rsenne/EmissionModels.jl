module EmissionModels

using DensityInterface
using Random
using SpecialFunctions: loggamma
using StatsAPI
using LinearAlgebra

import StatsAPI: fit! # being weird about fit!

include("zeroinflated/poisson.jl")

# exports
export rand, logdensityof, fit!
export PoissonZeroInflated

end
