using EmissionModels
using Test
using Aqua
using StatsAPI

@testset "EmissionModels.jl" begin
    @testset "Code linting" begin
        if VERSION >= v"1.10" && isempty(VERSION.prerelease)
            using Pkg
            Pkg.add("JET")
            using JET
            JET.test_package(EmissionModels; target_modules=(EmissionModels,))
        else
            @info "Skipping JET on Julia $VERSION (requires >=1.10 and a release build)"
        end
    end

    @testset "GLM Models" begin
        include("glm/gaussian.jl")
        include("glm/test_bernoulli_poisson.jl")
    end
    @testset "Zero-inflated models" begin
        include("zeroinflated/test_poisson.jl")
    end

    @testset "Multivariate models" begin
        include("multivariate/test_t.jl")
    end

    @testset "Allocations" begin
        include("allocations.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmissionModels)
    end
end
