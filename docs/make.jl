using EmissionModels
using Documenter
using DensityInterface
using Literate
using Random
using StatsAPI

DocMeta.setdocmeta!(EmissionModels, :DocTestSetup, :(using EmissionModels); recursive=true)

# Convert the Literate tutorials into Documenter pages. The `#src` lines
# (embedded tests) are stripped from the generated markdown.
examples_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")
mkpath(examples_md_path)
for file in readdir(examples_md_path)
    if endswith(file, ".md")
        rm(joinpath(examples_md_path, file))
    end
end
for file in readdir(examples_path)
    if endswith(file, ".jl")
        Literate.markdown(joinpath(examples_path, file), examples_md_path)
    end
end

makedocs(;
    modules=[EmissionModels],
    authors="Ryan Senne",
    sitename="EmissionModels.jl",
    format=Documenter.HTML(;
        canonical="https://rsenne.github.io/EmissionModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            joinpath("examples", "basics.md"),
            joinpath("examples", "glm.md"),
            joinpath("examples", "acdc.md"),
        ],
        "Distributions" => "distributions.md",
        "GLM Emissions" => "glm.md",
        "DDM Emissions" => "ddm.md",
        "Priors" => "priors.md",
        "ACDC Model Selection" => "acdc.md",
        "Custom Emission Models" => "custom.md",
    ],
)

deploydocs(; repo="github.com/rsenne/EmissionModels.jl", devbranch="main")
