using DistributionVectors
using Test, Random

include("distributionvector.jl")
include("autocor.jl")
include("sumdistributionvector.jl")

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
