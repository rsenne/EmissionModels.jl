using EmissionModels
using Test
using LinearAlgebra
using DensityInterface
using StatsAPI

@testset "Constructor promotion for integer and mixed eltypes" begin
    @testset "GaussianGLM" begin
        # Integer β/σ2 promote to Float64
        g = GaussianGLM([1, 2], 3)
        @test g isa GaussianGLM{Float64,NoPrior}
        @test eltype(g.β) === Float64
        @test g.σ2 === 3.0

        g_ridge = GaussianGLM([1, 2], 3, RidgePrior(0.5))
        @test g_ridge isa GaussianGLM{Float64,RidgePrior{Float64}}

        # Float32 preserved
        g32 = GaussianGLM(Float32[0.5, -1.0], 1.0f0)
        @test g32 isa GaussianGLM{Float32,NoPrior}
        @test eltype(g32.β) === Float32
        @test g32.σ2 === 1.0f0

        # Mixed Float32 β + Float64 σ2 promotes to Float64
        g_mix = GaussianGLM(Float32[0.5, -1.0], 1.0)
        @test g_mix isa GaussianGLM{Float64,NoPrior}
    end

    @testset "BernoulliGLM" begin
        b = BernoulliGLM([1, 2])
        @test b isa BernoulliGLM{Float64,NoPrior}
        @test eltype(b.β) === Float64

        b_ridge = BernoulliGLM([1, 2], RidgePrior(0.5))
        @test b_ridge isa BernoulliGLM{Float64,RidgePrior{Float64}}

        b32 = BernoulliGLM(Float32[0.5, -1.0])
        @test b32 isa BernoulliGLM{Float32,NoPrior}

        b32_ridge = BernoulliGLM(Float32[0.5, -1.0], RidgePrior(0.5f0))
        @test b32_ridge isa BernoulliGLM{Float32,RidgePrior{Float32}}
    end

    @testset "PoissonGLM" begin
        p = PoissonGLM([1, 2])
        @test p isa PoissonGLM{Float64,NoPrior}

        p_ridge = PoissonGLM([1, 2], RidgePrior(0.5))
        @test p_ridge isa PoissonGLM{Float64,RidgePrior{Float64}}

        p32 = PoissonGLM(Float32[0.5, -1.0])
        @test p32 isa PoissonGLM{Float32,NoPrior}
    end

    @testset "MvGaussianGLM" begin
        # All-integer B and Σ promote to Float64
        g = MvGaussianGLM([1 0; 0 1], [1 0; 0 1])
        @test g isa MvGaussianGLM{Float64,NoPrior}
        @test eltype(g.B) === Float64
        @test eltype(g.Σ) === Float64

        # Mixed: Int B, Float64 Σ
        g_mix = MvGaussianGLM([1 0; 0 1], [1.0 0.0; 0.0 1.0])
        @test g_mix isa MvGaussianGLM{Float64,NoPrior}

        # Float32 preserved across both
        g32 = MvGaussianGLM(Float32[1.0 0.0; 0.0 1.0], Float32[1.0 0.0; 0.0 1.0])
        @test g32 isa MvGaussianGLM{Float32,NoPrior}
        @test eltype(g32.B) === Float32

        # Mixed Float32 / Float64 promotes to Float64
        g_mix32 = MvGaussianGLM(Float32[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])
        @test g_mix32 isa MvGaussianGLM{Float64,NoPrior}

        # Integer + prior
        g_ridge = MvGaussianGLM([1 0; 0 1], [1 0; 0 1], RidgePrior(0.5))
        @test g_ridge isa MvGaussianGLM{Float64,RidgePrior{Float64}}
    end

    @testset "MvBernoulliGLM" begin
        b = MvBernoulliGLM([1 0; 0 1])
        @test b isa MvBernoulliGLM{Float64,NoPrior}

        b_ridge = MvBernoulliGLM([1 0; 0 1], RidgePrior(0.5))
        @test b_ridge isa MvBernoulliGLM{Float64,RidgePrior{Float64}}

        b32 = MvBernoulliGLM(Float32[1.0 0.0; 0.0 1.0])
        @test b32 isa MvBernoulliGLM{Float32,NoPrior}

        b32_ridge = MvBernoulliGLM(Float32[1.0 0.0; 0.0 1.0], RidgePrior(0.5f0))
        @test b32_ridge isa MvBernoulliGLM{Float32,RidgePrior{Float32}}
    end

    @testset "MvPoissonGLM" begin
        p = MvPoissonGLM([1 0; 0 1])
        @test p isa MvPoissonGLM{Float64,NoPrior}

        p_ridge = MvPoissonGLM([1 0; 0 1], RidgePrior(0.5))
        @test p_ridge isa MvPoissonGLM{Float64,RidgePrior{Float64}}

        p32 = MvPoissonGLM(Float32[0.5 -1.0; 1.0 0.5])
        @test p32 isa MvPoissonGLM{Float32,NoPrior}
    end
end

@testset "logdensityof return-type stability" begin
    @testset "Float32 inputs return Float32" begin
        glm_g = GaussianGLM(Float32[0.5, -1.0], 1.0f0)
        glm_b = BernoulliGLM(Float32[0.5, -1.0])
        glm_p = PoissonGLM(Float32[0.5, -1.0])
        glm_mg = MvGaussianGLM(Float32[0.5 -1.0; 1.0 0.5], Float32[1.0 0.3; 0.3 1.5])
        glm_mb = MvBernoulliGLM(Float32[0.5 -1.0; 1.0 0.5])
        glm_mp = MvPoissonGLM(Float32[0.5 -1.0; 0.2 0.0])

        x32 = Float32[1.0, 2.0]
        y32v = Float32[0.1, 0.2]
        yi = [0, 1]

        @test @inferred(logdensityof(glm_g, 0.5f0; control_seq=x32)) isa Float32
        @test @inferred(logdensityof(glm_b, 1; control_seq=x32)) isa Float32
        @test @inferred(logdensityof(glm_b, 0; control_seq=x32)) isa Float32
        @test @inferred(logdensityof(glm_b, 2; control_seq=x32)) isa Float32  # -Inf branch
        @test @inferred(logdensityof(glm_p, 3; control_seq=x32)) isa Float32
        @test @inferred(logdensityof(glm_p, -1; control_seq=x32)) isa Float32  # -Inf branch
        @test @inferred(logdensityof(glm_mg, y32v; control_seq=x32)) isa Float32
        @test @inferred(logdensityof(glm_mb, yi; control_seq=x32)) isa Float32
        @test @inferred(logdensityof(glm_mp, yi; control_seq=x32)) isa Float32
    end

    @testset "Float64 inputs return Float64" begin
        glm_g = GaussianGLM([0.5, -1.0], 1.0)
        glm_b = BernoulliGLM([0.5, -1.0])
        glm_p = PoissonGLM([0.5, -1.0])
        glm_mg = MvGaussianGLM([0.5 -1.0; 1.0 0.5], [1.0 0.3; 0.3 1.5])
        glm_mb = MvBernoulliGLM([0.5 -1.0; 1.0 0.5])
        glm_mp = MvPoissonGLM([0.5 -1.0; 0.2 0.0])

        x = [1.0, 2.0]
        yv = [0.1, 0.2]
        yi = [0, 1]

        @test @inferred(logdensityof(glm_g, 0.5; control_seq=x)) isa Float64
        @test @inferred(logdensityof(glm_b, 1; control_seq=x)) isa Float64
        @test @inferred(logdensityof(glm_p, 3; control_seq=x)) isa Float64
        @test @inferred(logdensityof(glm_mg, yv; control_seq=x)) isa Float64
        @test @inferred(logdensityof(glm_mb, yi; control_seq=x)) isa Float64
        @test @inferred(logdensityof(glm_mp, yi; control_seq=x)) isa Float64
    end
end

@testset "logdensityof accepts Real-stored counts" begin
    x = [1.0, 2.0]
    glm_b = BernoulliGLM([0.5, -1.0])
    glm_p = PoissonGLM([0.5, -1.0])
    glm_mp = MvPoissonGLM([0.5 -1.0; 0.2 0.0])

    #= Obs sequences frequently arrive as Float64 even for count data; an
       integer-valued float must score identically to the Int, and values with
       zero mass (non-integer or negative) must return -Inf, not throw. =#
    @test logdensityof(glm_b, 1.0; control_seq=x) == logdensityof(glm_b, 1; control_seq=x)
    @test logdensityof(glm_b, 0.5; control_seq=x) == -Inf
    @test logdensityof(glm_p, 3.0; control_seq=x) == logdensityof(glm_p, 3; control_seq=x)
    @test logdensityof(glm_p, 2.5; control_seq=x) == -Inf
    @test logdensityof(glm_p, -1.0; control_seq=x) == -Inf
    @test logdensityof(glm_mp, [2.0, 0.0]; control_seq=x) ==
        logdensityof(glm_mp, [2, 0]; control_seq=x)
    @test logdensityof(glm_mp, [2.0, 0.5]; control_seq=x) == -Inf
end

@testset "fit! errors on rank-deficient design" begin
    # X column 2 is constant 0, so XᵀWX is rank-deficient.
    X_sing = ones(5, 2)
    X_sing[:, 2] .= 0.0
    w = ones(5)

    @testset "GaussianGLM" begin
        glm = GaussianGLM([0.0, 0.0], 1.0)
        y = zeros(5)
        @test_throws ArgumentError fit!(glm, y, w; control_seq=X_sing)

        # The same data fits fine with a RidgePrior, as the error message suggests.
        glm_ridge = GaussianGLM([0.0, 0.0], 1.0, RidgePrior(1.0))
        @test fit!(glm_ridge, y, w; control_seq=X_sing) === glm_ridge
        @test all(isfinite, glm_ridge.β)
    end

    @testset "MvGaussianGLM" begin
        #= Varied obs so the Σ M-step is well-conditioned, isolating the failure
           to the (rank-deficient) XᵀWX step. =#
        obs = [[1.0, 2.0], [3.0, -1.0], [-1.0, 0.5], [2.0, 3.0], [0.5, -2.0]]
        glm = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2))
        @test_throws ArgumentError fit!(glm, obs, w; control_seq=X_sing)

        glm_ridge = MvGaussianGLM(zeros(2, 2), Matrix(1.0I, 2, 2), RidgePrior(1.0))
        @test fit!(glm_ridge, obs, w; control_seq=X_sing) === glm_ridge
        @test all(isfinite, glm_ridge.B)
    end
end

@testset "PoissonGLM type-aware η bound (no Float32 overflow)" begin
    #= With the old hardcoded clamp(η, ±500), exp(500) overflows Float32 to Inf.
       The new bound is log(floatmax(T)) − 2, type-aware, so exp(η_max) stays
       within the float range for both Float32 and Float64. =#
    glm32 = PoissonGLM(Float32[1.0, 0.5])
    X = Float32[1.0 1.0; 1.0 -1.0; 1.0 0.5]
    y = [1, 2, 0]
    w = Float32[1.0, 1.0, 1.0]
    fit!(glm32, y, w; control_seq=X)
    @test all(isfinite, glm32.β)
    @test eltype(glm32.β) === Float32

    # Same for Float64
    glm64 = PoissonGLM([1.0, 0.5])
    fit!(glm64, y, w; control_seq=Float64.(X))
    @test all(isfinite, glm64.β)
end
