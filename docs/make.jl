using DistributionVectors
using Documenter

DocMeta.setdocmeta!(DistributionVectors, :DocTestSetup, :(using DistributionVectors); recursive=true)

makedocs(;
    modules=[DistributionVectors],
    authors="Thomas Wutzler",
    repo="https://github.com/bgctw/DistributionVectors.jl/blob/{commit}{path}#{line}",
    sitename="DistributionVectors.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bgctw.github.io/DistributionVectors.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Types" => "distributionvector.md",
        "Aggregation" => "sumdist.md",
    ],
    doctest=false,
)

deploydocs(;
    repo="github.com/bgctw/DistributionVectors.jl",
    devbranch="main",
)
