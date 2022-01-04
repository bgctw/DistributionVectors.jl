using DistributionVectors
using Test, Random

@testset "DistributionVector" begin
    include("distributionvector.jl")
end

@testset "autocor" begin
    include("autocor.jl")
end

@testset "sumdistributionvector" begin
    include("sumdistributionvector.jl")
end

const tests = [
    "normal",
    "lognormal",
]

printstyled("Running Distribution tests:\n", color=:blue)

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end

# print method ambiguities
# println("Potentially stale exports: ")
# display(Test.detect_ambiguities(DistributionVectors))
# println()
