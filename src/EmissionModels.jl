module EmissionModels

using Distributions: Poisson, Chisq
using DensityInterface
using LinearAlgebra
using LogExpFunctions: logaddexp, logsumexp
using Optim: optimize, TwiceDifferentiable, Newton, LBFGS, LineSearches
using Optim
using Random
using SpecialFunctions: logfactorial, loggamma, digamma, trigamma, polygamma
using StatsAPI
using StatsAPI: fit!

include("zeroinflated/poisson.jl")
include("multivariate/t.jl")

# exports
export rand, logdensityof, fit!
export PoissonZeroInflated
export MultivariateT, MultivariateTDiag

end
