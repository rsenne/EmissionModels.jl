using EmissionModels
using Documenter

DocMeta.setdocmeta!(EmissionModels, :DocTestSetup, :(using EmissionModels); recursive=true)

makedocs(;
    modules=[EmissionModels],
    authors="Ryan Senne",
    sitename="EmissionModels.jl",
    format=Documenter.HTML(;
        canonical="https://rsenne.github.io/EmissionModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/rsenne/EmissionModels.jl", devbranch="main")
