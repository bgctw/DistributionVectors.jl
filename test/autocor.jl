using LinearAlgebra, Distributions
using Test
using DistributionVectors

@testset "autocor" begin
    acf0 = [1,0.4,0.1]
    nsum = 5
    Sigma = cormatrix_for_acf(nsum, acf0);
    @test diag(Sigma) == fill(1.0, nsum)
    @test diag(Sigma,1) == diag(Sigma,-1) == fill(0.4, nsum - 1)
    @test diag(Sigma,2) == diag(Sigma,-2) == fill(0.1, nsum - 2)
    @test diag(Sigma,3) == diag(Sigma,-3) == fill(0.0, nsum - 3)
    dmn = MvNormal(ones(nsum), Symmetric(Sigma));
end;

