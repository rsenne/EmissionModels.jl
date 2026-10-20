
using Test
using Random
using LinearAlgebra
using Distributions
using DensityInterface
using StatsAPI
using EmissionModels

rng = Random.MersenneTwister(1234)

function _synthetic_gaussian_glm(
    rng, n::Int, p::Int; β::Vector{Float64}, σ2::Float64, weights=:uniform
)
    @assert length(β) == p
    X = hcat(ones(n), randn(rng, n, p - 1))  # intercept + random features
    μ = X * β
    y = rand.(Ref(rng), Normal.(μ, sqrt(σ2)))
    w = if weights === :uniform
        ones(n)
    elseif weights === :random
        (rand(rng, n) .+ 0.5)
    else
        error("weights must be :uniform or :random")
    end
    return X, y, w
end

_expected_logpdf(y, μ, σ2) = -0.5 * log(2π * σ2) - 0.5 * ((y - μ)^2 / σ2)

@testset "GaussianGLM Basic Functionality" begin
    @testset "Constructor" begin
        glm = GaussianGLM([0.5, -1.0], 1.0)
        @test glm.β == [0.5, -1.0]
        @test glm.σ2 == 1.0

        glm2 = GaussianGLM([1, 2], 3)
        @test glm2.β == [1, 2]
        @test glm2.σ2 == 3
    end

    @testset "DensityInterface" begin
        glm = GaussianGLM([0.5, -1.0], 1.0)

        # Trait
        @test DensityKind(glm) == HasDensity()

        X = [1.0 2.0; 1.0 3.0; 1.0 4.0]
        μ = X * glm.β
        y = rand.(Ref(rng), Normal.(μ, sqrt(glm.σ2)))

        # logdensityof matches closed form
        for i in eachindex(y)
            x_i = vec(X[i, :])  # ensure Vector
            logp = logdensityof(glm, y[i]; control_seq=x_i)
            @test isfinite(logp)

            expected = _expected_logpdf(y[i], dot(glm.β, x_i), glm.σ2)
            @test logp ≈ expected rtol = 1e-12 atol = 0.0
        end

        # Higher density near the mean than far away
        x0 = vec(X[1, :])
        y_at_mean = dot(glm.β, x0)
        y_far = y_at_mean + 10.0
        @test logdensityof(glm, y_at_mean; control_seq=x0) >
            logdensityof(glm, y_far; control_seq=x0)
    end

    @testset "Random sampling" begin
        glm = GaussianGLM([0.5, -1.0], 2.0) # variance 2.0
        x = [1.0, 3.0]
        n = 10_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]

        @test all(s -> s isa Real, samples)

        # Empirical mean should be close to dot(β,x)
        μ = dot(glm.β, x)
        m = mean(samples)
        @test m ≈ μ atol = 0.15

        # Empirical variance should be close to σ2
        v = var(samples)
        @test v ≈ glm.σ2 atol = 0.25
    end

    @testset "fit! with uniform weights" begin
        β_true = [0.3, -1.2]          # p=2
        σ2_true = 0.7
        X, y, w = _synthetic_gaussian_glm(
            rng, 2000, 2; β=β_true, σ2=σ2_true, weights=:uniform
        )

        glm = GaussianGLM([0.0, 0.0], 1.0)
        fit!(glm, y, w; control_seq=X)

        # Coefficients should be close with enough data
        @test glm.β ≈ β_true atol = 0.08
        @test glm.σ2 ≈ σ2_true atol = 0.10

        @test all(isfinite, glm.β)
        @test isfinite(glm.σ2)
        @test glm.σ2 > 0
    end

    @testset "fit! with weighted observations" begin
        β_true = [1.0, 0.6]
        σ2_true = 1.5
        X, y, w = _synthetic_gaussian_glm(
            rng, 2500, 2; β=β_true, σ2=σ2_true, weights=:random
        )

        glm = GaussianGLM([0.0, 0.0], 1.0)
        fit!(glm, y, w; control_seq=X)

        @test glm.β ≈ β_true atol = 0.10
        @test glm.σ2 ≈ σ2_true atol = 0.15
    end

    @testset "Constructor with prior" begin
        glm = GaussianGLM([0.0, 0.0], 1.0)
        @test glm.prior isa NoPrior

        glm2 = GaussianGLM([0.0, 0.0], 1.0, RidgePrior(2.0))
        @test glm2.prior isa RidgePrior
        @test glm2.prior.λ == 2.0
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        β_true = [3.0, -3.0]
        σ2_true = 1.0
        X, y, w = _synthetic_gaussian_glm(rng, 300, 2; β=β_true, σ2=σ2_true)

        glm_noprior = GaussianGLM([0.0, 0.0], 1.0)
        fit!(glm_noprior, y, w; control_seq=X)

        glm_ridge = GaussianGLM([0.0, 0.0], 1.0, RidgePrior(10.0))
        fit!(glm_ridge, y, w; control_seq=X)

        @test norm(glm_ridge.β) < norm(glm_noprior.β)
        @test all(isfinite, glm_ridge.β)
    end
end

function _synthetic_mvgaussian_glm(
    rng, n::Int, p::Int, k::Int; B::Matrix{Float64}, Σ::Matrix{Float64}, weights=:uniform
)
    @assert size(B) == (p, k)
    @assert size(Σ) == (k, k)
    X = hcat(ones(n), randn(rng, n, p - 1))
    L = cholesky(Σ).L
    obs_seq = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        μ_i = vec(B' * X[i, :])
        obs_seq[i] = μ_i .+ L * randn(rng, k)
    end
    w = if weights === :uniform
        ones(n)
    elseif weights === :random
        (rand(rng, n) .+ 0.5)
    else
        error("weights must be :uniform or :random")
    end
    return X, obs_seq, w
end

@testset "MvGaussianGLM" begin
    rng = Random.MersenneTwister(20260507)

    @testset "Constructor" begin
        B = [0.5 -0.3; 1.0 0.2]              # p=2, k=2
        Σ = [1.0 0.2; 0.2 0.8]
        glm = MvGaussianGLM(B, Σ)
        @test glm.B == B
        @test glm.Σ == Σ
        @test glm.in_dim == 2
        @test glm.out_dim == 2
        @test glm.prior isa NoPrior

        glm_float = MvGaussianGLM([1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
        @test glm_float.B == [1.0 0.0; 0.0 1.0]
        @test eltype(glm_float.B) === Float64

        glm_ridge = MvGaussianGLM(B, Σ, RidgePrior(2.0))
        @test glm_ridge.prior isa RidgePrior

        @test_throws DimensionMismatch MvGaussianGLM(
            B, [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        )
        @test_throws ArgumentError MvGaussianGLM(B, [1.0 0.0; 0.0 -1.0])
        @test_throws ArgumentError MvGaussianGLM(B, [1.0 0.9; 0.0 1.0])
    end

    @testset "Σ symmetry and cache refresh (#83)" begin
        A = [1.0 0.2; -0.4 0.9]
        S = A * A'
        S[2, 1] = nextfloat(S[2, 1])  # symmetric only up to roundoff
        @test MvGaussianGLM(zeros(2, 2), S).Σ == S
        S32 = Float32.(A * A')
        S32[2, 1] = nextfloat(S32[2, 1])
        @test MvGaussianGLM(zeros(Float32, 2, 2), S32) isa MvGaussianGLM{Float32}

        g = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        EmissionModels._set_scale!(g, Matrix(4.0I, 2, 2))
        @test logdensityof(g, [0.3, -0.2]; control_seq=[1.0, 0.5]) ≈
            logpdf(MvNormal(zeros(2), 4.0I(2)), [0.3, -0.2])
        @test_throws ArgumentError EmissionModels._set_scale!(g, [1.0 0.9; 0.0 1.0])
        @test_throws DimensionMismatch EmissionModels._set_scale!(g, [3.0;;])
    end

    @testset "DensityInterface" begin
        B = [0.5 -1.0; 1.0 0.5]
        Σ = [1.0 0.3; 0.3 1.5]
        glm = MvGaussianGLM(B, Σ)

        @test DensityKind(glm) == HasDensity()

        x = [1.0, 2.0]
        μ = vec(B' * x)
        y = μ .+ [0.1, -0.2]

        # Compare against MvNormal closed form
        diff = y - μ
        expected = -log(2π) - 0.5 * logdet(Σ) - 0.5 * dot(diff, Σ \ diff)
        @test logdensityof(glm, y; control_seq=x) ≈ expected rtol = 1e-10

        # Higher density at the mean than far from it
        @test logdensityof(glm, μ; control_seq=x) >
            logdensityof(glm, μ .+ 10.0; control_seq=x)

        @test_throws DimensionMismatch logdensityof(glm, [1.0]; control_seq=x)
        @test_throws DimensionMismatch logdensityof(glm, y; control_seq=[1.0])
    end

    @testset "Random sampling" begin
        B = [0.5 -1.0; 1.0 0.5]
        Σ = [1.0 0.3; 0.3 1.5]
        glm = MvGaussianGLM(B, Σ)
        x = [1.0, 0.5]

        n = 20_000
        samples = [rand(rng, glm; control_seq=x) for _ in 1:n]
        Y = reduce(hcat, samples)'                # n × k

        μ_true = vec(B' * x)
        m = vec(mean(Y; dims=1))
        @test m ≈ μ_true atol = 0.05

        S = (Y .- m')' * (Y .- m') / (n - 1)
        @test S ≈ Σ atol = 0.1
    end

    @testset "fit! recovers B and Σ" begin
        B_true = [0.5 -1.0; 1.0 0.5; -0.3 0.8]   # p=3, k=2
        Σ_true = [1.0 0.3; 0.3 1.5]
        X, obs_seq, w = _synthetic_mvgaussian_glm(rng, 4000, 3, 2; B=B_true, Σ=Σ_true)

        glm = MvGaussianGLM(zeros(3, 2), Matrix(1.0I, 2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol = 0.08
        @test glm.Σ ≈ Σ_true atol = 0.10
        @test all(isfinite, glm.B)
        @test all(isfinite, glm.Σ)
        @test isposdef(glm.Σ)
    end

    @testset "fit! random weights" begin
        B_true = [1.0 0.0; -0.5 1.5]
        Σ_true = [0.5 -0.1; -0.1 0.7]
        X, obs_seq, w = _synthetic_mvgaussian_glm(
            rng, 3000, 2, 2; B=B_true, Σ=Σ_true, weights=:random
        )

        glm = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        fit!(glm, obs_seq, w; control_seq=X)

        @test glm.B ≈ B_true atol = 0.10
        @test glm.Σ ≈ Σ_true atol = 0.12
    end

    @testset "fit! with RidgePrior shrinks toward zero" begin
        B_true = [3.0 -3.0; 3.0 3.0]
        Σ_true = [1.0 0.0; 0.0 1.0]
        X, obs_seq, w = _synthetic_mvgaussian_glm(rng, 200, 2, 2; B=B_true, Σ=Σ_true)

        glm_noprior = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        fit!(glm_noprior, obs_seq, w; control_seq=X)

        glm_ridge = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2), RidgePrior(20.0))
        fit!(glm_ridge, obs_seq, w; control_seq=X)

        @test norm(glm_ridge.B) < norm(glm_noprior.B)
        @test all(isfinite, glm_ridge.B)
    end

    @testset "fit! DimensionMismatch" begin
        glm = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        X = ones(5, 2)
        bad_obs = [zeros(2) for _ in 1:4]
        good_obs = [zeros(2) for _ in 1:5]
        @test_throws DimensionMismatch fit!(glm, bad_obs, ones(5); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(4); control_seq=X)
        @test_throws DimensionMismatch fit!(glm, good_obs, ones(5); control_seq=ones(5, 3))

        wrong_dim_obs = [zeros(3) for _ in 1:5]
        @test_throws DimensionMismatch fit!(glm, wrong_dim_obs, ones(5); control_seq=X)
    end
end

@testset "GaussianGLM rejects non-positive σ2" begin
    @test_throws ArgumentError GaussianGLM([1.0, 2.0], -2.0)
    @test_throws ArgumentError GaussianGLM([1.0, 2.0], 0.0)
    @test_throws ArgumentError GaussianGLM([1.0, 2.0], -1.0, RidgePrior(0.5))
    # Positive σ2 still constructs fine, including through the promoting path.
    @test GaussianGLM([1, 2], 1) isa GaussianGLM{Float64,NoPrior}
end

@testset "GaussianGLM fit! keeps σ2 positive on degenerate data" begin
    #= Noiseless data (y = 2x exactly, n = p is enough) previously drove
       σ2 to exactly 0, breaking the σ2 > 0 invariant and making every later
       logdensityof NaN. The fit must land on the variance floor instead. =#
    glm = GaussianGLM([0.0], 1.0)
    X = reshape([1.0, 2.0, 3.0], 3, 1)
    y = [2.0, 4.0, 6.0]
    fit!(glm, y, ones(3); control_seq=X)
    @test glm.β[1] ≈ 2.0
    @test glm.σ2 ≈ sqrt(eps(Float64))
    @test isfinite(logdensityof(glm, 2.0; control_seq=[1.0]))

    #= All-zero weights (a state with no responsibility this EM step): fit! is
       a no-op returning the emission unchanged, matching DDM `fit!`, rather
       than erroring. =#
    glm0 = GaussianGLM([0.0], 1.0, RidgePrior(1.0))
    @test fit!(glm0, y, zeros(3); control_seq=X) === glm0
    @test glm0.β == [0.0]
    @test glm0.σ2 == 1.0

    mv0 = MvGaussianGLM(zeros(1, 2), Matrix(1.0I, 2, 2), RidgePrior(1.0))
    mvobs = [randn(Random.MersenneTwister(1), 2) for _ in 1:3]
    B_before = copy(mv0.B)
    Σ_before = copy(mv0.Σ)
    @test fit!(mv0, mvobs, zeros(3); control_seq=X) === mv0
    @test mv0.B == B_before
    @test mv0.Σ == Σ_before
end

@testset "MvGaussianGLM fit! floors Σ on sparse weights (#79)" begin
    #= Weight on 3 of 20 points is fewer than p + k = 4 effective observations,
       so the raw residual covariance is singular. =#
    for T in (Float64, Float32)
        rng79 = Random.MersenneTwister(1)
        X = randn(rng79, T, 20, 2)
        ys = [randn(rng79, T, 2) for _ in 1:20]
        w = zeros(T, 20)
        w[1:3] .= one(T)

        g = MvGaussianGLM(zeros(T, 2, 2), Matrix{T}(I, 2, 2))
        fit!(g, ys, w; control_seq=X)
        @test eltype(g.Σ) === T
        @test isfinite(g.logdetΣ)
        @test all(diag(g.Σ_chol.L) .>= 0.9 * sqrt(EmissionModels._variance_floor(T)))
        @test g.Σ_chol.L * g.Σ_chol.L' ≈ g.Σ
        @test isfinite(logdensityof(g, ys[1]; control_seq=X[1, :]))
    end

    # On well-conditioned data the floor perturbs the closed form by ~sqrt(eps).
    B_true = [0.5 -1.0; 1.0 0.5; -0.3 0.8]
    X, ys, w = _synthetic_mvgaussian_glm(
        Random.MersenneTwister(79), 500, 3, 2; B=B_true, Σ=[1.0 0.3; 0.3 1.5]
    )
    g = MvGaussianGLM(zeros(3, 2), Matrix(1.0I, 2, 2))
    fit!(g, ys, w; control_seq=X)
    Y = reduce(vcat, permutedims.(ys))
    B_cf = (X' * Diagonal(w) * X) \ (X' * Diagonal(w) * Y)
    R = Y - X * B_cf
    Σ_cf = (R' * Diagonal(w) * R) ./ sum(w)
    @test g.B ≈ B_cf atol = 1e-8
    @test g.Σ ≈ Σ_cf atol = 1e-7
    @test diag(g.Σ) ≈ diag(Σ_cf) .+ EmissionModels._variance_floor(Float64) atol = 1e-12
end

#= Issue #77: RidgePrior(λ) is β ~ N(0, λ⁻¹I) added to the weighted negative
   log-likelihood for every GLM. The Gaussian closed form applies it at the
   current σ²/Σ (an ECM step), so iterating fit! reaches the joint MAP. =#

struct _ScaledRidge77 <: AbstractPrior
    λ::Float64
end
EmissionModels.neglogprior(p::_ScaledRidge77, β) = p.λ * dot(β, β) / 2
function EmissionModels.neglogprior_grad!(p::_ScaledRidge77, g, β)
    return g .+= p.λ .* β
end
function EmissionModels.neglogprior_hess!(p::_ScaledRidge77, H, β)
    for i in axes(H, 1)
        H[i, i] += p.λ
    end
end

struct _AnisoPrior77 <: AbstractPrior end
EmissionModels.neglogprior(::_AnisoPrior77, β) = zero(eltype(β))
EmissionModels.neglogprior_grad!(::_AnisoPrior77, g, β) = nothing
function EmissionModels.neglogprior_hess!(::_AnisoPrior77, H, β)
    for i in axes(H, 1)
        H[i, i] += i
    end
end

function _issue77_data(::Type{T}=Float64) where {T}
    rng = MersenneTwister(1)
    X = randn(rng, 200, 2)
    y = X * [1.0, -2.0] .+ 3 .* randn(rng, 200)
    return T.(X), T.(y)
end

#= Independent reference: profile σ² out of
   (1/(2σ²))‖y − Xβ‖² + (n/2) log σ² + (λ/2)‖β‖² (σ̂² = RSS/n) and run Newton
   on the profiled objective in β. =#
function _profiled_ridge_map(X, y, λ)
    n, p = size(X)
    β = (X' * X) \ (X' * y)
    for _ in 1:100
        r = y - X * β
        rss = dot(r, r)
        g = -(n / rss) * (X' * r) + λ * β
        Xr = X' * r
        H = (n / rss) * (X' * X) - (2n / rss^2) * (Xr * Xr') + λ * I
        β -= H \ g
        norm(g) < 1e-13 && break
    end
    r = y - X * β
    return β, dot(r, r) / n
end

# Central-difference gradient of β ↦ -Σ wᵢ log p(yᵢ | xᵢ, β) + neglogprior(β).
function _neglogpost_grad(make_glm, prior, β, y, w, X; h=1e-6)
    function f(b)
        return -sum(
            w[i] * logdensityof(make_glm(b), y[i]; control_seq=X[i, :]) for
            i in eachindex(y)
        ) + neglogprior(prior, b)
    end
    return map(eachindex(β)) do j
        e = zeros(length(β))
        e[j] = h
        (f(β + e) - f(β - e)) / (2h)
    end
end

@testset "RidgePrior has one meaning across GLMs (#77)" begin
    @testset "one step from σ² = 1 matches the old (XᵀWX + λI) closed form" begin
        # Relies on the starting σ² = 1, where σ²λ = λ.
        rng = MersenneTwister(77)
        X, y, w = _synthetic_gaussian_glm(
            rng, 300, 3; β=[0.5, 1.0, -1.0], σ2=0.5, weights=:random
        )
        λ = 4.0
        glm = GaussianGLM(zeros(3), 1.0, RidgePrior(λ))
        fit!(glm, y, w; control_seq=X)
        W = Diagonal(w)
        @test glm.β ≈ (X' * W * X + λ * I) \ (X' * W * y) rtol = 1e-12
    end

    @testset "issue reproducer converges to the joint MAP" begin
        X, y = _issue77_data()
        n = length(y)
        λ = 5.0
        glm = GaussianGLM(zeros(2), 1.0, RidgePrior(λ))
        for _ in 1:20
            fit!(glm, y, ones(n); control_seq=X)
        end
        β̂, σ̂2 = glm.β, glm.σ2
        @test (X' * X + σ̂2 * λ * I) * β̂ ≈ X' * y rtol = 1e-8
        @test σ̂2 ≈ sum(abs2, y - X * β̂) / n rtol = 1e-10

        β_ref, σ2_ref = _profiled_ridge_map(X, y, λ)
        @test β̂ ≈ β_ref rtol = 1e-8
        @test σ̂2 ≈ σ2_ref rtol = 1e-8
        # The old (XᵀX + λI) solution is a different estimate on this data.
        @test !isapprox(β̂, (X' * X + λ * I) \ (X' * y); rtol=1e-2)

        # A custom prior with the same Hessian goes through the generic hook.
        glm_c = GaussianGLM(zeros(2), 1.0, _ScaledRidge77(λ))
        for _ in 1:20
            fit!(glm_c, y, ones(n); control_seq=X)
        end
        @test glm_c.β ≈ β̂ rtol = 1e-12
    end

    @testset "Gaussian and Bernoulli minimise the same penalised objective" begin
        X, y = _issue77_data()
        n = length(y)
        w = ones(n)
        prior = RidgePrior(5.0)

        glm = GaussianGLM(zeros(2), 1.0, prior)
        for _ in 1:20
            fit!(glm, y, w; control_seq=X)
        end
        σ̂2 = glm.σ2
        g_gauss = _neglogpost_grad(b -> GaussianGLM(b, σ̂2), prior, glm.β, y, w, X)
        @test norm(g_gauss) < 1e-5

        yb = Int.(y .> 0)
        bglm = BernoulliGLM(zeros(2), prior)
        fit!(bglm, yb, w; control_seq=X)
        g_bern = _neglogpost_grad(b -> BernoulliGLM(b), prior, bglm.β, yb, w, X)
        @test norm(g_bern) < 1e-5
    end

    @testset "MvGaussianGLM solves the Sylvester stationarity" begin
        rng = MersenneTwister(770)
        B_true = [1.0 -0.5; 2.0 0.5; -1.0 1.5]
        Σ_true = [2.0 0.8; 0.8 1.0]
        X, obs, w = _synthetic_mvgaussian_glm(
            rng, 150, 3, 2; B=B_true, Σ=Σ_true, weights=:random
        )
        Y = reduce(hcat, obs)'
        W = Diagonal(w)
        XWX = X' * W * X
        XWY = X' * W * Y
        λ = 30.0

        # One step: stationarity holds at the Σ the step started from.
        Σ0 = [1.5 -0.3; -0.3 0.7]
        glm1 = MvGaussianGLM(zeros(3, 2), Σ0, RidgePrior(λ))
        fit!(glm1, obs, w; control_seq=X)
        @test XWX * glm1.B + λ * glm1.B * Σ0 ≈ XWY rtol = 1e-10

        glm = MvGaussianGLM(zeros(3, 2), Matrix(1.0I, 2, 2), RidgePrior(λ))
        for _ in 1:200
            fit!(glm, obs, w; control_seq=X)
        end
        @test norm(XWX * glm.B + λ * glm.B * glm.Σ - XWY) < 1e-8 * norm(XWY)

        glm0 = MvGaussianGLM(zeros(3, 2), Σ0, NoPrior())
        fit!(glm0, obs, w; control_seq=X)
        @test glm0.B ≈ XWX \ XWY rtol = 1e-12

        glm_aniso = MvGaussianGLM(zeros(3, 2), Σ0, _AnisoPrior77())
        @test_throws ArgumentError fit!(glm_aniso, obs, w; control_seq=X)

        glm_c = MvGaussianGLM(zeros(3, 2), Σ0, _ScaledRidge77(λ))
        fit!(glm_c, obs, w; control_seq=X)
        @test glm_c.B ≈ glm1.B rtol = 1e-12
    end

    @testset "Float32 keeps its eltype" begin
        X, y = _issue77_data(Float32)
        n = length(y)
        λ = 5.0f0
        glm = GaussianGLM(zeros(Float32, 2), 1.0f0, RidgePrior(λ))
        for _ in 1:20
            fit!(glm, y, ones(Float32, n); control_seq=X)
        end
        @test eltype(glm.β) === Float32
        @test glm.σ2 isa Float32
        @test (X' * X + glm.σ2 * λ * I) * glm.β ≈ X' * y rtol = 1e-4
        β_ref, _ = _profiled_ridge_map(Float64.(X), Float64.(y), Float64(λ))
        @test glm.β ≈ β_ref rtol = 1e-4

        rng32 = MersenneTwister(7)
        obs = [Float32[y[i], X[i, 1] - 2 * X[i, 2] + randn(rng32)] for i in 1:n]
        Y = reduce(hcat, obs)'
        mv = MvGaussianGLM(zeros(Float32, 2, 2), Matrix(1.0f0I, 2, 2), RidgePrior(λ))
        for _ in 1:100
            fit!(mv, obs, ones(Float32, n); control_seq=X)
        end
        @test eltype(mv.B) === Float32
        @test eltype(mv.Σ) === Float32
        @test norm(X' * X * mv.B + λ * mv.B * mv.Σ - X' * Y) < 1.0f-4 * norm(X' * Y)
    end
end

@testset "fit! leaves parameters untouched when it throws (#80)" begin
    #= Identical large-scale output columns make Σ singular, so B must not be
       committed before Σ is checked. The trigger must stay singular after the
       Σ diagonal floor (#79). =#
    rng80 = Random.MersenneTwister(1)
    X = randn(rng80, 20, 2)
    ys = [(v = 1e9 * randn(rng80); [v, v]) for _ in 1:20]
    w = ones(20)
    mv = MvGaussianGLM([0.1 0.2; 0.3 0.4], [2.0 0.5; 0.5 1.0])
    B0, Σ0 = copy(mv.B), copy(mv.Σ)
    L0, ld0 = copy(mv.Σ_chol.factors), mv.logdetΣ
    @test_throws ArgumentError fit!(mv, ys, w; control_seq=X)
    @test mv.B == B0
    @test mv.Σ == Σ0
    @test mv.Σ_chol.factors == L0
    @test mv.logdetΣ == ld0

    #= Guard against future reordering, not a reproduction: the singular-XᵀWX
       throw currently precedes any write to β or σ2. =#
    g = GaussianGLM([0.5, -1.0], 2.0)
    β0, σ20 = copy(g.β), g.σ2
    Xd = hcat(X[:, 1], zeros(20))
    @test_throws ArgumentError fit!(g, randn(rng80, 20), ones(20); control_seq=Xd)
    @test g.β == β0
    @test g.σ2 == σ20
end
