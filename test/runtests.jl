using EmissionModels
using Test
using Aqua
using StatsAPI

@testset "EmissionModels.jl" begin
    @testset "Code linting" begin
        # JET ≤ 0.10.x crashes inside Base._which on Julia ≥ 1.12 (reflection
        # API change). Until JET ships a 1.12-compatible release we skip the
        # linting test on 1.12+.
        jet_ok = VERSION >= v"1.10" && VERSION < v"1.12" && isempty(VERSION.prerelease)
        if jet_ok
            using Pkg
            Pkg.add("JET")
            using JET
            JET.test_package(EmissionModels; target_modules=(EmissionModels,))
        else
            @info "Skipping JET on Julia $VERSION (requires 1.10 ≤ v < 1.12)"
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
