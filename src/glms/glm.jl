"""
    AbstractGLM

Abstract type for Generalized Linear Model emission distributions.

GLM subtypes should implement the HiddenMarkovModels.jl interface:
- `DensityInterface.DensityKind(::YourGLM)` - Declare HasDensity()
- `DensityInterface.logdensityof(glm, obs)` - Compute log density
- `Random.rand(rng, glm, X)` - Generate samples given covariates
- `StatsAPI.fit!(glm, obs_seq, weight_seq)` - Update parameters
"""
abstract type AbstractGLM end

