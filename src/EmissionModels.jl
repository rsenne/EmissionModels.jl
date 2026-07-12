module EmissionModels

using Distributions: Normal, Bernoulli, Binomial, Poisson, Chisq, TDist, MvNormal
using Distributions: cdf, quantile
using Distributions:
    ContinuousUnivariateDistribution, DiscreteUnivariateDistribution, AbstractMvNormal
using DensityInterface
using DiffResults: DiffResults
using ForwardDiff: ForwardDiff
using HiddenMarkovModels: ControlledEmission
using HiddenMarkovModels: AbstractHMM, obs_distributions, forward_backward
using LinearAlgebra
using LogExpFunctions: logaddexp, logsumexp, log1pexp, logistic, logit
using NearestNeighbors: KDTree, knn
using Optim: Optim, optimize, OnceDifferentiable, TwiceDifferentiable, Newton, LBFGS
using Optim.NLSolversBase: only_fgh!, only_fg!
using Random
using SpecialFunctions: loggamma, digamma, trigamma
using Statistics: mean, var, cov
using StatsAPI
using StatsAPI: fit!

include("zeroinflated/poisson.jl")
include("multivariate/t.jl")
include("glms/glm.jl")
include("ssm/ddm.jl")
include("acdc/interface.jl")
include("acdc/drivers.jl")
include("acdc/hmm.jl")

# exports
export rand, logdensityof, fit!
export PoissonZeroInflated
export MultivariateT, MultivariateTDiag
export GaussianGLM, BernoulliGLM, PoissonGLM, MultinomialGLM
export MvGaussianGLM, MvBernoulliGLM, MvPoissonGLM
export AbstractPrior, NoPrior, RidgePrior
export StimulusCodedDDM, CoherenceDDM
export neglogprior, neglogprior_grad!, neglogprior_hess!

# ACDC model selection
export ACDCResult, StochasticDriverResult, ComponentDiscrepancy
export stochastic_drivers, component_discrepancies
export acdc_loss, acdc_select, get_critical_rho_values
export compute_discrepancy
export KLDiscrepancy, KSDiscrepancy, WassersteinDiscrepancy
export SquaredErrorDiscrepancy, MMDDiscrepancy

end
