using DistributionVectors
using Documenter


# allow plots on headless server 
# https://juliadocs.github.io/Documenter.jl/stable/man/syntax/#@example-block
ENV["GKSwstype"] = "100"

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
        "Univariate" => [
            "Overview" => "univariate.md",
            "Normal" => "normal.md",
            "LogNormal" => "lognormal.md",
        ],
    ],
    doctest=false,
)

function tmpf() 
    #https://discourse.julialang.org/t/deploy-documenter-docs-locally/5463/7
    #ENV["TRAVIS_REPO_SLUG"] = "github.com/bgctw/DistributionVectors.jl.git"
    ENV["TRAVIS_REPO_SLUG"] = "."
    ENV["TRAVIS_PULL_REQUEST"] = "false"
    ENV["TRAVIS_OS_NAME"] = "linux"
    ENV["TRAVIS_JULIA_VERSION"] = "1.5"
    ENV["TRAVIS_TAG"] = ""
    ENV["TRAVIS_BRANCH"] = "main"
    # ENV["DOCUMENTER_KEY"] = chomp(open(f->read(f, String), "docs/secrets/DOCUMENTER_KEY"));
    deploydocs(deps = nothing, make = nothing,
        repo = "github.com/bgctw/DistributionVectors.jl.git",
        target = "build",
        branch = "gh-pages",
        devbranch = ENV["TRAVIS_BRANCH"],
    )    
    # does not work, because the build folder is not on github
end


deploydocs(;
    repo="github.com/bgctw/DistributionVectors.jl",
    devbranch="main",
)

function deploydocs_fromlocal()
    # pages directory must be setup to track gh-pages branch
    cd("../DistributionVectors_pages/")
    run(`git checkout gh-pages`)
    run(`git pull`)
    src = "../DistributionVectors/docs/build"
    dst = "dev"
    readdir(src)
    readdir(dst)
    cp("$dst/siteinfo.js","$src/siteinfo.js")
    cp(src, dst, force=true)
    run(`git add .`)
    run(`git commit -m"local doc generation"`)
    run(`git push`)
    cd("../DistributionVectors")
end

