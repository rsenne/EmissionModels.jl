module EmissionModels

using Distributions
using DensityInterface
using LinearAlgebra
using LogExpFunctions: logaddexp, logsumexp
using Optim
using Random
using SpecialFunctions: logfactorial
using StatsAPI
using StatsAPI: fit!


include("zeroinflated/poisson.jl")

# exports
export rand, logdensityof, fit!
export PoissonZeroInflated

end
