# Changelog

All notable changes to EmissionModels.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `RidgePrior(λ)` now means `β ~ N(0, λ⁻¹I)` for Gaussian GLMs too; previously
  the Gaussian closed-form path effectively used `N(0, σ²λ⁻¹I)` (#77).
- `MvGaussianGLM` throws `ArgumentError` for priors whose Hessian is not
  isotropic; previously any constant-Hessian prior was accepted (#77).
- DDM emissions floor the first-passage density at `1e-16`, so `rt ≤ τ` gives
  a finite log-density, and `fit!` constrains τ to the fixed box `[1e-3, 5]` s
  instead of `(0, rt_min)`. DDM-HMM states can now separate in τ under
  Baum-Welch, and `fit!` no longer stalls when the starting τ exceeds the
  fastest RT (#116, #69). `logdensityof` for `rt ≤ τ` (and on series
  underflow) now returns `log(1e-16)` instead of `-Inf`; an invalid choice is
  still `-Inf`.
- `CalciumParams` no longer accepts `nshare`; the share count is taken from
  `CalciumEmission` construction (#71).

### Fixed

- Documentation drift after the calcium and ACDC additions: API.md lists the
  calcium exports, ACDC docs describe the hinge-loss selection rule, and
  `MvTDiag` docs call `σ²` the diagonal scale (#95).
- Tied calcium M-step solved on partial statistics when `nshare` was left at its
  default, making Baum-Welch non-monotone (#71).
- `KLDiscrepancy` and `MMDDiscrepancy` no longer blow up on the duplicate drivers
  that `n_samples > 1` pooling creates; exact ties are collapsed into weighted
  atoms (#75).

## [0.1.0] - 2026-08-14

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
  `MvPoissonGLM`. Gaussian GLMs fit in closed form (weighted least squares);
  the rest use Newton via Optim.jl. All take optional regularization through
  `NoPrior`/`RidgePrior` (`AbstractPrior` is the extension point).
- **DDM emissions** via a package extension on
  [SequentialSamplingModels.jl](https://github.com/itsdfish/SequentialSamplingModels.jl)
  (weak dependency): `StimulusCodedDDM` and `CoherenceDDM` drift-diffusion
  emissions for two-alternative forced-choice data, giving the DDM-HMM. The
  types always construct; `logdensityof`, `rand`, and `fit!` require
  `using SequentialSamplingModels`.
- **Calcium imaging emissions** (`CalciumEmission`, `CalciumParams`): the
  autoregressive fluorescence model of Keeley, Zoltowski, Charles, and Pillow
  (eLife 109405) with the latent spike count marginalized out, so an HMM fits
  raw fluorescence traces without a separate deconvolution step. Includes the
  tied M-step across states, simulation (`rand_calcium`), control construction
  (`lagged_controls`), and data-driven initialization (`init_calcium`).
- **ACDC model selection**: `component_discrepancies`, `acdc_select`, and
  `ACDCResult` select the number of HMM states by driver recovery
  (probability-integral transform), scored with `KSDiscrepancy`,
  `KLDiscrepancy`, `WassersteinDiscrepancy`, `SquaredErrorDiscrepancy`, or
  `MMDDiscrepancy`.
- Type genericity (#63): parameters in `Float32` (or AD dual numbers) keep their
  element type through fitting and ACDC driver recovery instead of widening to
  `Float64`.
- Allocation-minimal density and sampling hot paths, with `rand!` for in-place
  sampling and allocation regression tests enforcing the budgets.
- Documentation: Documenter.jl site with per-model pages, Literate.jl
  tutorials (basics, GLM, DDM-HMM, calcium, ACDC) whose code runs as part of
  the test suite, and a top-level [API.md](API.md) recording the frozen 0.1.0
  surface.
- Infrastructure: AirspeedVelocity.jl benchmark suite and the unregistered
  `libs/EmissionModelsTest` package of shared test recipes.
