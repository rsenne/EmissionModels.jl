using EmissionModels
using Test
using Aqua
using JET
using JuliaFormatter
using StatsAPI

@testset "EmissionModels.jl" begin
    @testset "Zero-inflated models" begin
        include("zeroinflated/test_poisson.jl")
    end
    @testset "Multivariate models" begin
        include("multivariate/test_t.jl")
    end
    @testset "Code linting" begin
        if VERSION >= v"1.10"
            JET.test_package(EmissionModels; target_modules=(EmissionModels,))
        end
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmissionModels)
    end
end
