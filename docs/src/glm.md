# GLM Emissions

Generalized linear model (GLM) emission types model observations that depend on an external **control vector** (also called a design or feature vector ``x``). The parameter ``β`` maps the control vector to the latent linear predictor, and a link function transforms this into the emission parameters.

## Provided GLM types

### `GaussianGLM(β, σ²)`

Linear regression with Gaussian noise.

``f(xᵀβ) = xᵀβ``  (identity link)

``Y \mid x \sim \mathcal{N}(xᵀβ, \ σ²)``

```julia
glm = GaussianGLM(zeros(3), 1.0)
# or with a prior:
glm = GaussianGLM(zeros(3), 1.0, RidgePrior(0.5))
```

### `BernoulliGLM(β)`

Logistic regression for binary observations ``y ∈ {0, 1}``.

``f(xᵀβ) = \frac{1}{1 + \exp(-xᵀβ)}``  (logistic / sigmoid)

``Y \mid x \sim \text{Bernoulli}(f(xᵀβ))``

```julia
glm = BernoulliGLM(zeros(3))
```

### `PoissonGLM(β)`

Log-linear regression for count observations ``Y ∈ \{0, 1, \ldots\}``.

``f(xᵀβ) = \exp(xᵀβ)``

``Y \mid x \sim \text{Poisson}(\exp(xᵀβ))``

```julia
glm = PoissonGLM(zeros(3))
```

### `MultinomialGLM(B, n_trials)`

Multinomial logistic regression for count vectors over ``K`` categories e.g., per-time-bin choice counts including a "no choice" category. Category ``K`` is the reference; coefficients are a ``p × (K-1)`` matrix ``B``.

``p_j(x) = \frac{\exp(η_j)}{\sum_{l=1}^{K} \exp(η_l)}, \quad η_j = B[:,j]ᵀx \ (j < K), \quad η_K = 0``  (softmax link)

``Y \mid x \sim \text{Multinomial}(n, \ p(x))``

`logdensityof` and `fit!` condition on each observation's own total count, so totals may vary across time steps; `rand` draws `n_trials` trials. For single-trial choice data they also accept plain integer labels ``y ∈ 1:K`` (treated as one-hot count vectors), so a choice sequence can stay a `Vector{Int}`:

```julia
glm = MultinomialGLM(zeros(3, 2), 1)
choices = [1, 3, 2, 1, 3]              # e.g. left / right / no-choice labels
fit!(glm, choices, ones(5); control_seq=X)
logdensityof(glm, 3; control_seq=x)    # log p₃(x)
```

```julia
glm = MultinomialGLM(zeros(3, 2), 5)   # p = 3 inputs, K = 3 categories, 5 trials
# or with a prior:
glm = MultinomialGLM(zeros(3, 2), 5, RidgePrior(0.5))
```

## Multivariate GLM types

Each univariate GLM has a multivariate counterpart that emits a length-``k`` observation vector for a single input ``x``. Coefficients are stored as a ``p × k`` matrix ``B``; column ``j`` is the coefficient vector for output dimension ``j``.

### `MvGaussianGLM(B, Σ)`

Multivariate linear regression with shared full covariance ``Σ``.

``Y \mid x \sim \mathcal{N}(Bᵀx, \ Σ)``

### `MvBernoulliGLM(B)`

``k`` independent logistic regressions sharing the same input ``x``.

``P(Y \mid x) = \prod_{j=1}^{k} \text{Bernoulli}(σ(B[:,j]ᵀx))``

### `MvPoissonGLM(B)`

``k`` independent Poisson log-linear regressions sharing the same input ``x``.

``P(Y \mid x) = \prod_{j=1}^{k} \text{Poisson}(\exp(B[:,j]ᵀx))``

## Fitting GLM emissions

GLM emissions are fit by minimizing the negative log-posterior (or weighted negative log-likelihood when no prior is used):

``\ell(β) = -\sum_{i=1}^{n} w_i \, \log p(y_i \mid x_i, β) + \text{neglogprior}(β)``

```julia
# Gaussian is closed-form weighted least squares.
fit!(glm, y_seq, w_seq; control_seq=X)

# Bernoulli / Poisson use Optim's Newton with analytic gradient and Hessian
# (supplied via a fused fgh!).
fit!(glm, y_seq, w_seq; control_seq=X, max_iter=50, gtol=1e-8)
```

## API Reference

```@docs
EmissionModels.AbstractGLM
GaussianGLM
BernoulliGLM
PoissonGLM
MultinomialGLM
MvGaussianGLM
MvBernoulliGLM
MvPoissonGLM
DensityInterface.logdensityof(::MvGaussianGLM, ::AbstractVector)
StatsAPI.fit!(::BernoulliGLM, ::AbstractVector, ::AbstractVector{<:Real})
StatsAPI.fit!(::PoissonGLM, ::AbstractVector, ::AbstractVector{<:Real})
StatsAPI.fit!(::MultinomialGLM{T}, ::AbstractVector{<:AbstractVector}, ::AbstractVector{<:Real}) where {T<:Real}
StatsAPI.fit!(::MvGaussianGLM{T}, ::AbstractVector{<:AbstractVector}, ::AbstractVector{<:Real}) where {T<:Real}
StatsAPI.fit!(::MvBernoulliGLM{T}, ::AbstractVector{<:AbstractVector}, ::AbstractVector{<:Real}) where {T<:Real}
StatsAPI.fit!(::MvPoissonGLM{T}, ::AbstractVector{<:AbstractVector}, ::AbstractVector{<:Real}) where {T<:Real}
Random.rand!(::Random.AbstractRNG, ::MultinomialGLM, ::AbstractVector)
Random.rand!(::Random.AbstractRNG, ::MvGaussianGLM{T}, ::AbstractVector) where {T<:Real}
Random.rand!(::Random.AbstractRNG, ::MvBernoulliGLM, ::AbstractVector)
Random.rand!(::Random.AbstractRNG, ::MvPoissonGLM, ::AbstractVector)
```
