using EmissionModels
using Test
using Aqua
using JET
using JuliaFormatter
using StatsAPI

@testset "EmissionModels.jl" begin
    @testset "Code linting" begin
        if VERSION >= v"1.10" && isempty(VERSION.prerelease)
            JET.test_package(EmissionModels; target_modules=(EmissionModels,))
        else
            @info "Skipping JET on Julia $VERSION (requires >=1.10 and a release build)"
        end
    end
    @testset "Zero-inflated models" begin
        include("zeroinflated/test_poisson.jl")
    end

    @testset "Multivariate models" begin
        include("multivariate/test_t.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmissionModels)
    end
end
