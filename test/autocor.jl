using LinearAlgebra, Distributions
using Test
using DistributionVectors
using StatsBase # coef

@testset "cormatrix_for_acf" begin    
    acf0 = [1,0.4,0.1]
    nsum = 5
    Sigma = cormatrix_for_acf(nsum, acf0);
    @test diag(Sigma) == fill(1.0, nsum)
    @test diag(Sigma,1) == diag(Sigma,-1) == fill(0.4, nsum - 1)
    @test diag(Sigma,2) == diag(Sigma,-2) == fill(0.1, nsum - 2)
    @test diag(Sigma,3) == diag(Sigma,-3) == fill(0.0, nsum - 3)
    #dmn = MvNormal(ones(nsum), Symmetric(Sigma));
end;

@testset "AutoCorrelationFunction" begin  
    acfvec = [1,0.4,0.1]
    acf = @inferred AutoCorrelationFunction(acfvec)
    @test @inferred coef(acf) == acfvec
    @test @inferred coef(acf,1) == acfvec[2] # vector starts with lag 0
end;


