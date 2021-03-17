##### specific distributions #####

const discrete_distributions = [
]

const continuous_distributions = [
    "normal",
    "lognormal",    # LogNormal depends on Normal
#    "logitnormal",    # LogitNormal depends on Normal
]

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end

for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end
