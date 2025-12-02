module EmissionModels

using DensityInterface
using Random
using SpecialFunctions: loggamma
using StatsAPI

import StatsAPI: fit! # being weird about fit!

include("zeroinflated/poisson.jl")

# exports
export rand, logdensityof
export PoissonZeroInflated

end
