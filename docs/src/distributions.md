# Distributions

This page documents the probability distributions provided by EmissionModels.jl. Each type implements the `HiddenMarkovModels` emission interface:

```julia
Random.rand(rng::AbstractRNG, dist::MyEmission)
DensityInterface.DensityKind(::MyEmission) = DensityInterface.HasDensity()
DensityInterface.logdensityof(dist::MyEmission, obs)
StatsAPI.fit!(dist::MyEmission, obs_seq, weight_seq)
```

## Count data

### `PoissonZeroInflated(λ, π)`

A zero-inflated Poisson distribution for count observations with excess zeros.

| Field | Type | Description |
|-------|------|-------------|
| `λ` | `T` | Rate parameter of the Poisson component (``λ > 0``). |
| `π` | `T` | Probability of structural (extra) zeros (``0 ≤ π ≤ 1``). |

**Model:**  ``P(X = k) = π + (1-π)\exp(-λ)``  for ``k = 0``, and  ``P(X = k) = (1-π) \cdot \frac{λ^k e^{-λ}}{k!}``  for ``k > 0``.

**Example:**

```julia
dist = PoissonZeroInflated(5.0, 0.3)
x = rand(dist)          # sample
lp = logdensityof(dist, 0)  # log-density at k = 0
fit!(dist, [1, 0, 3, 0], [1.0, 1.0, 1.0, 1.0])
```

## Multivariate continuous

### `MultivariateT(μ, Σ, ν)`

Full-covariance multivariate Student's t-distribution.

| Field | Description |
|-------|-------------|
| `μ` | Location vector (mean for ``ν > 1``). |
| `Σ` | Positive-definite scale matrix. |
| `ν` | Degrees of freedom (> 0). |

**Example:**

```julia
μ   = [0.0, 1.0]
Σ   = [1.0 0.5; 0.5 2.0]
dist = MultivariateT(μ, Σ, ν=5.0)
x = rand(dist)
lp = logdensityof(dist, x)
```

### `MultivariateTDiag(μ, σ², ν)`

Diagonal-covariance multivariate Student's t-distribution (more efficient for high dimensions).

| Field | Description |
|-------|-------------|
| `μ` | Location vector. |
| `σ²` | Vector of diagonal variances (all positive). |
| `ν` | Degrees of freedom (> 0). |

**Example:**

```julia
dist = MultivariateTDiag([0.0, 0.0], [1.0, 2.0], ν=5.0)
```

## Parameter estimation (`fit!`)

All multivariate t-distributions are fit via a weighted EM algorithm:

```julia
fit!(dist, obs_seq, weight_seq; max_iter=100, tol=1e-6, fix_nu=false)
```

- The E-step posterior weights depend on the current Mahalanobis distance.
- The M-step updates ``μ`` and ``Σ`` (and optionally ``ν``).
- Pass `fix_nu=true` to keep the degrees of freedom fixed during fitting.
- `μ` and ``Σ`` are re-regularized on-the-fly if the Cholesky factorization fails.
