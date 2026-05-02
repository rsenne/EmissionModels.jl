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

## Fitting GLM emissions

GLM emissions are fit by minimizing the negative log-posterior (or weighted negative log-likelihood when no prior is used):

``\ell(β) = -\sum_{i=1}^{n} w_i \, \log p(y_i \mid x_i, β) + \text{neglogprior}(β)``

```julia
fit!(glm, y_seq, w_seq; control_seq=X, optim_opts=Optim.Options())
```
