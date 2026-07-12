using EmissionModels
using Test
using Aqua
using StatsAPI
using JuliaFormatter

@testset "EmissionModels.jl" begin
    @testset "Code Formatting" begin
        @test JuliaFormatter.format(EmissionModels; verbose=false, overwrite=false)
    end
    @testset "Code linting" begin
        # JET lags behind prerelease Julia.
        if isempty(VERSION.prerelease)
            using JET
            JET.test_package(EmissionModels; target_modules=(EmissionModels,))
        end
    end

    @testset "GLM Models" begin
        include("glm/gaussian.jl")
        include("glm/test_bernoulli_poisson.jl")
        include("glm/test_promotion_and_types.jl")
        include("glm/test_multinomial.jl")
        include("glm/test_controlled_hmm.jl")
    end
    @testset "Zero-inflated models" begin
        include("zeroinflated/test_poisson.jl")
    end

    @testset "Multivariate models" begin
        include("multivariate/test_t.jl")
    end

    @testset "ACDC model selection" begin
        include("acdc/test_acdc.jl")
    end

    @testset "Allocations" begin
        include("allocations.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmissionModels)
    end
end
