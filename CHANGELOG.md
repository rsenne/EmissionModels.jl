# Changelog

All notable changes to EmissionModels.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-20

**Breaking.**

- `CalciumParams` no longer accepts the `nshare` keyword (#71).
- `RidgePrior(λ)` now means `β ~ N(0, λ⁻¹I)` for Gaussian GLMs as well, so the
  same `λ` regularizes Gaussian fits differently than in 0.1.0, and
  `MvGaussianGLM` rejects priors whose Hessian is not isotropic (#77).
- DDM `logdensityof` at `rt ≤ τ` returns `log(1e-16)` instead of `-Inf`, and
  `fit!` constrains τ to `[1e-3, 5]` s instead of `(0, rt_min)` (#116).
- `init_calcium` (when `c` is not supplied, Poisson mode) estimates `c` when
  unit influx would put `λ` near the spike cap `R`, and then fits it by
  default (#73).
- `fit!` on `BernoulliGLM`, `PoissonGLM`, `MvBernoulliGLM`, `MvPoissonGLM` and
  `MultinomialGLM` throws `ArgumentError` on observations outside the support,
  which previously fit silently (#84).
- `MvT` and `MvGaussianGLM` require a symmetric `Σ`; a non-symmetric matrix
  now throws (#83).

### Changed

- `CalciumParams` no longer accepts `nshare`; the share count is taken from
  `CalciumEmission` construction, so sharing one `CalciumParams` across states
  always solves the tied M-step on the full statistics (#71). Every emission
  built on a `CalciumParams` must belong to the same HMM; deepcopy the HMM or
  the whole emission vector, not a single emission.
- `init_calcium` estimates `c` from the residual third moment when `c = 1`
  would start `λ` within reach of the `0:R` cap (`λ + 3√λ > R`), and `fit_c`
  now defaults to `nothing` (fit `c` exactly when it was estimated). Calcium
  emissions warn once when a rate puts Poisson mass beyond `R` (#73).
- `RidgePrior(λ)` now means `β ~ N(0, λ⁻¹I)` for Gaussian GLMs too; previously
  the Gaussian closed-form path effectively used `N(0, σ²λ⁻¹I)`. The Gaussian
  step is solved at the current `σ²` (or `Σ`, via a Sylvester equation) (#77).
- `MvGaussianGLM` throws `ArgumentError` for priors whose Hessian is not
  isotropic; previously any constant-Hessian prior was accepted (#77).
- DDM emissions floor the first-passage density at `1e-16`, so `rt ≤ τ` gives
  a finite log-density, and `fit!` constrains τ to the fixed box `[1e-3, 5]` s
  instead of `(0, rt_min)`. DDM-HMM states can now separate in τ under
  Baum-Welch, and `fit!` no longer stalls when the starting τ exceeds the
  fastest RT (#116, #69). `logdensityof` for `rt ≤ τ` (and on series
  underflow) now returns `log(1e-16)` instead of `-Inf`; an invalid choice is
  still `-Inf`.
- Compat widened to allow DensityInterface 1 (#68).
- README installs from the General registry; the benchmark suite now covers
  `MultinomialGLM`, `CalciumEmission` (Poisson and Gaussian) and both DDMs
  (#67).
- CI: allocation budgets are enforced only on release Julia instead of hiding
  all prerelease failures behind `continue-on-error`, and a Windows x64 leg was
  added (#91).
- The benchmark workflow runs on `pull_request` with read-only permissions
  instead of `pull_request_target`, so fork PR code no longer runs with write
  access or a base-branch cache; results are written to the job summary
  instead of a PR comment (#92).
- The test environment allows JET 0.12, which Julia 1.13 requires (#100).

### Fixed

- Calcium Gaussian E-step no longer clamps the spike posterior mean at zero,
  which made it inconsistent with the density and let Baum-Welch decrease the
  log-likelihood (#70).
- Calcium `fit!` validates sequence lengths before accumulating, so a short
  `control_seq` no longer leaves partial sums in the shared accumulators (#72).
- `acdc_select` scores pools with fewer than two samples as `Inf`; a
  one-sample pool previously scored `NaN` and won the selection (#74).
- The DDM CDF hook interpolates around zero drift, where
  SequentialSamplingModels returns a CDF with the wrong sign and ACDC drivers
  piled up at zero (#76).
- `RidgePrior` terms are computed in the coefficient eltype, so `RidgePrior(1)`
  no longer throws `InexactError` (#78).
- `MvGaussianGLM` adds `sqrt(eps(T))` (the same magnitude as the `GaussianGLM`
  `σ²` floor) to the residual covariance diagonal, so a state with fewer than
  `p + k` effective observations no longer throws mid-EM (#79).
- `MvGaussianGLM`, `GaussianGLM` and `MvT` `fit!` commit the new location only
  after the scale update succeeds, instead of leaving a half-updated struct
  when it throws (#80).
- The Student-t normaliser is computed via `logbeta`, removing up to 0.12 nats
  of error in `Float32` at large ν (#81).
- Integer constructor arguments to `MvT`, `MvTDiag` and `PoissonZeroInflated`
  are promoted to float instead of building `{Int}` types that failed (#82).
- `MvT` and `MvGaussianGLM` reject a non-symmetric `Σ` instead of storing it
  while the density used its lower triangle; the cached scale fields are
  documented as read-only (#83).
- `BernoulliGLM`, `PoissonGLM`, `MvBernoulliGLM`, `MvPoissonGLM` and
  `MultinomialGLM` validate observations in `fit!`, rejecting values that
  `logdensityof` scores as `-Inf` (#84).
- `rand` on Poisson GLMs bounds η before sampling, instead of throwing
  `InexactError` from `Poisson(Inf)` at large η (#86).
- DDM hook fallbacks raise `MethodError` for non-`Real` arguments when the
  SequentialSamplingModels extension is loaded, instead of reporting it missing
  (#87).
- `BernoulliGLM`, `PoissonGLM`, `MvBernoulliGLM`, `MvPoissonGLM` and
  `MultinomialGLM` are left unchanged by a zero-total-weight `fit!`, matching
  the Gaussian GLMs, instead of moving to the prior mode (#88).
- The priors.md Lasso example subtypes `AbstractPrior` and extends the
  `EmissionModels` methods, so it runs (#93).
- custom.md documents the positional `ControlledEmission` interface that
  HiddenMarkovModels.jl 0.7 calls, replacing a keyword interface it does not
  use (#94).
- Documentation drift after the calcium and ACDC additions: API.md lists the
  calcium exports, ACDC docs describe the hinge-loss selection rule, and
  `MvTDiag` docs call `σ²` the diagonal scale (#95).
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
