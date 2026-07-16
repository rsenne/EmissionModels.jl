# API Reference: Public vs. Private

This document is the authoritative list of what EmissionModels.jl considers
public API. Anything not listed as public is private, regardless of whether it
is technically reachable via `EmissionModels.foo`.

**Stability policy.** The package follows [SemVer](https://semver.org). Pre-1.0,
breaking changes to the public API bump the minor version (`0.x` → `0.(x+1)`);
private API may change in any release without notice. If you find yourself
needing a private name, open an issue.

## Public API

The public API is exactly the set of exported names, plus the emission
interface they implement.

### Emission interface (re-exported verbs)

These generic functions from Base/Random, DensityInterface, and StatsAPI are
re-exported; every emission type implements the applicable subset:

| Name | Interface |
| --- | --- |
| `logdensityof(d, obs[, control])` | Log density of one observation (DensityInterface). |
| `rand(rng, d[, control])` | Sample one observation (Random). |
| `rand!(rng, d, out)` | In-place sampling for vector-valued emissions; zero-allocation. |
| `fit!(d, obs_seq[, control_seq], weight_seq; kwargs...)` | Weighted in-place M-step update (StatsAPI). |

Static (uncontrolled) emissions take `(obs_seq, weight_seq)`; controlled
emissions (GLMs, DDMs) additionally take `control_seq` per the
HiddenMarkovModels.jl `ControlledEmission` positional interface.

### Static distributions

| Name | Description |
| --- | --- |
| `PoissonZeroInflated` | Zero-inflated Poisson with EM-fittable mixing weight. |
| `MvT` | Multivariate Student-t with full scale matrix. |
| `MvTDiag` | Multivariate Student-t with diagonal scale (zero-allocation density). |

### GLM emissions (controlled)

| Name | Description |
| --- | --- |
| `AbstractGLM` | Abstract supertype and extension point for GLM emissions. |
| `GaussianGLM` | Univariate Gaussian response, identity link. |
| `BernoulliGLM` | Bernoulli response, logit link. |
| `PoissonGLM` | Poisson response, log link. |
| `MultinomialGLM` | Multinomial response, multinomial-logit link. |
| `MvGaussianGLM` | Multivariate Gaussian response. |
| `MvBernoulliGLM` | Vector of independent Bernoulli responses. |
| `MvPoissonGLM` | Vector of independent Poisson responses. |

### Priors

| Name | Description |
| --- | --- |
| `AbstractPrior` | Abstract supertype for coefficient priors; implement the three `neglogprior*` methods to add one. |
| `NoPrior` | No regularization (default). |
| `RidgePrior` | Gaussian (L2) prior on coefficients. |
| `neglogprior(p, β)` | Negative log prior density. |
| `neglogprior_grad!(p, g, β)` | Accumulate gradient in place. |
| `neglogprior_hess!(p, H, β)` | Accumulate Hessian in place. |

### DDM emissions (controlled; require the SequentialSamplingModels extension)

| Name | Description |
| --- | --- |
| `StimulusCodedDDM` | Drift-diffusion emission with stimulus-coded drift sign. |
| `CoherenceDDM` | Drift-diffusion emission with drift linear in signed coherence. |

`logdensityof`, `rand`, and `fit!` for DDM emissions are provided by the
package extension: run `using SequentialSamplingModels` to activate them.

### ACDC model selection

| Name | Description |
| --- | --- |
| `ACDCResult` | Per-component discrepancies and usage for one candidate model. |
| `component_discrepancies(model, data, discrepancy; ...)` | Recover stochastic drivers and score each component. |
| `acdc_select(results, ρ)` | Select the component count from a vector of `ACDCResult`s at cutoff `ρ`. |
| `KLDiscrepancy` | kNN-estimated KL divergence from uniformity. |
| `KSDiscrepancy` | Kolmogorov–Smirnov distance from uniformity. |
| `WassersteinDiscrepancy` | 1-Wasserstein distance from uniformity. |
| `SquaredErrorDiscrepancy` | Squared-error CDF discrepancy. |
| `MMDDiscrepancy` | Maximum mean discrepancy from uniformity. |

## Private API

Everything below is internal. It may be renamed, changed, or removed in any
release.

### Naming convention

Any name prefixed with an underscore (`_newton_fit!`, `_emission_to_driver`,
`_ddm_logpdf`, `_scatter_mstep!`, …) is private by construction and is not
enumerated here. Unexported names without an underscore are also private; the
notable ones are listed below because they appear in tutorials or internal
docstrings.

### ACDC plumbing (unexported, deliberately internal)

These were unexported during the pre-0.1.0 API freeze to keep the selection
surface minimal. Tutorials qualify them as `EmissionModels.stochastic_drivers`
etc.; that spelling carries no stability guarantee.

| Name | Role |
| --- | --- |
| `stochastic_drivers(model, data; ...)` | Recover per-component PIT drivers; method table is the per-model hook. |
| `StochasticDriverResult` | Container for recovered driver pools and usage. |
| `ComponentDiscrepancy` | Abstract supertype of the discrepancy measures. |
| `compute_discrepancy(disc, pool; rng)` | Score one driver pool; implement to add a custom measure. |
| `acdc_loss(result, ρ)` | Hinge loss underlying `acdc_select`. |
| `get_critical_rho_values(results)` | Cutoff values at which the selection changes. |

### DDM internals

| Name | Role |
| --- | --- |
| `AbstractDDMEmission` | Supertype of the DDM emissions; unexported, subtype at your own risk. |
| `_ddm_logpdf`, `_ddm_rand`, `_ddm_cdf` | Extension hooks overloaded by the SequentialSamplingModels extension; the src definitions throw an actionable error when the extension is not loaded. |

### Everything else

Optimizer functors (`_BernoulliFGH`, `_PoissonFGH`, `_MultinomialFGH`,
`_DDMNLL`, `_DDMFG`), lazy array views (`_ColumnElementView`, `_OneHot`,
`_OneHotSeq`, `_ControlRowsMatrix`), EM workspaces, parameter pack/unpack
transforms, and the ACDC driver-recovery methods (`_emission_to_driver` and
friends) are implementation details of the fitting and selection routines.
