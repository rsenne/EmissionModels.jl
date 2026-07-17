# Changelog

All notable changes to EmissionModels.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial feature set, targeting the first registered release (0.1.0). Every
emission type implements the same interface — `logdensityof`, `rand`/`rand!`,
and weighted in-place `fit!` — and plugs directly into
[HiddenMarkovModels.jl](https://github.com/JuliaStats/HiddenMarkovModels.jl)
for Baum-Welch EM.

### Added

- **Static emissions**: zero-inflated Poisson (`PoissonZeroInflated`) and
  multivariate Student-t with full (`MvT`) or diagonal (`MvTDiag`) scale,
  including weighted maximum-likelihood fitting of the degrees of freedom.
- **Controlled (GLM) emissions** subtyping HiddenMarkovModels.jl's
  `ControlledEmission`: `GaussianGLM`, `BernoulliGLM`, `PoissonGLM`,
  `MultinomialGLM`, and multivariate `MvGaussianGLM`, `MvBernoulliGLM`,
  `MvPoissonGLM`. Fitting uses Newton/LBFGS via Optim.jl with optional
  regularization through `NoPrior`/`RidgePrior` (`AbstractPrior` is the
  extension point).
- **DDM emissions** via a package extension on
  [SequentialSamplingModels.jl](https://github.com/itsdfish/SequentialSamplingModels.jl)
  (weak dependency): `StimulusCodedDDM` and `CoherenceDDM` drift-diffusion
  emissions for two-alternative forced-choice data, giving the DDM-HMM. The
  types always construct; `logdensityof`, `rand`, and `fit!` require
  `using SequentialSamplingModels`.
- **ACDC model selection**: `component_discrepancies`, `acdc_select`, and
  `ACDCResult` select the number of HMM states by driver recovery
  (probability-integral transform), scored with `KSDiscrepancy`,
  `KLDiscrepancy`, `WassersteinDiscrepancy`, `SquaredErrorDiscrepancy`, or
  `MMDDiscrepancy`.
- Allocation-minimal density and sampling hot paths, with `rand!` for in-place
  sampling and allocation regression tests enforcing the budgets.
- Documentation: Documenter.jl site with per-model pages, Literate.jl
  tutorials (basics, ACDC, DDM-HMM) whose code runs as part of the test
  suite, and a top-level [API.md](API.md) recording the frozen 0.1.0 surface.
- Infrastructure: AirspeedVelocity.jl benchmark suite and the unregistered
  `libs/EmissionModelsTest` package of shared test recipes.
