using EmissionModels
using Test
using Aqua
using JET
using JuliaFormatter

@testset "EmissionModels.jl" begin
    @testset "Formatting" begin
        if VERSION >= v"1.10"
            @test JuliaFormatter.format(EmissionModels; verbose=false, overwrite=false)
        end
    end
    @testset "Code linting" begin
        if VERSION >= v"1.10"
            JET.test_package(EmissionModels; target_defined_modules=true)
        end
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmissionModels)
    end
end
