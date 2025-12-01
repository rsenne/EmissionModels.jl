using EmissionModels
using Test
using Aqua

@testset "EmissionModels.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EmissionModels)
    end
    # Write your tests here.
end
