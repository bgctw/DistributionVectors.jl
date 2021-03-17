"""
Support of vectors of random variables, i.e vectors of Distributions
"""
module DistributionVectors

export AbstractDistributionVector, SimpleDistributionVector, ParamDistributionVector,
    vectuptotupvec,
    AutoCorrelationFunction, cormatrix_for_acf

using Distributions, Missings, Statistics
using MissingStrategies
using StatsBase, Random
using FillArrays, RecursiveArrayTools, BandedMatrices, LinearAlgebra, OffsetArrays

import Base: size, length, IndexStyle, similar, getindex, setindex!, sum, ismissing
import Random: rand, rand!
import Statistics: mean
import StatsBase: params, coef

include("distributionvector.jl")
include("autocor.jl")
include("sumdistributionvector.jl")

# specific distributions
include("univariates.jl") 

end # module
