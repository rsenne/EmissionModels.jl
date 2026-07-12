module EmissionModels

using Distributions: Normal, Bernoulli, Poisson, Chisq, TDist, MvNormal
using Distributions: cdf, quantile
using Distributions:
    ContinuousUnivariateDistribution, DiscreteUnivariateDistribution, AbstractMvNormal
using DensityInterface
using HiddenMarkovModels: ControlledEmission
using HiddenMarkovModels: AbstractHMM, obs_distributions, forward_backward
using LinearAlgebra
using LogExpFunctions: logaddexp, logsumexp, log1pexp, logistic
using NearestNeighbors: KDTree, knn
using Optim: Optim, optimize, TwiceDifferentiable, Newton
using Optim.NLSolversBase: only_fgh!
using Random
using SpecialFunctions: loggamma, digamma, trigamma
using Statistics: mean, var, cov
using StatsAPI
using StatsAPI: fit!

include("zeroinflated/poisson.jl")
include("multivariate/t.jl")
include("glms/glm.jl")
include("acdc/interface.jl")
include("acdc/drivers.jl")
include("acdc/hmm.jl")

# exports
export rand, logdensityof, fit!
export PoissonZeroInflated
export MultivariateT, MultivariateTDiag
export GaussianGLM, BernoulliGLM, PoissonGLM
export MvGaussianGLM, MvBernoulliGLM, MvPoissonGLM
export AbstractPrior, NoPrior, RidgePrior
export neglogprior, neglogprior_grad!, neglogprior_hess!

# ACDC model selection
export ACDCResult, StochasticDriverResult, ComponentDiscrepancy
export stochastic_drivers, component_discrepancies
export acdc_loss, acdc_select, get_critical_rho_values
export compute_discrepancy
export KLDiscrepancy, KSDiscrepancy, WassersteinDiscrepancy
export SquaredErrorDiscrepancy, MMDDiscrepancy

end
