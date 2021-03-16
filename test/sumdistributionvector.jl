using DistributionVectors
using Test
using Distributions
#using MissingStrategies
using LinearAlgebra

if !@isdefined(MyNewDistribution)
  struct MyNewDistribution <: ContinuousUnivariateDistribution; end
end

@testset "sumDistributionVector: not defined for D" begin
  dv = SimpleDistributionVector(MyNewDistribution(), MyNewDistribution());
  acf = AutoCorrelationFunction([1,0.4,0.1]);
  nsum = 5
  corrmat = cormatrix_for_acf(nsum, acf);
  #@test_throws ErrorException sum(dv, acf, PassMissing())
  @test_throws ErrorException sum(dv, Symmetric(corrmat))
  @test_throws ErrorException sum(dv, acf)
  @test_throws ErrorException sum(dv)
  #@test_throws ErrorException mean(dv, acf, PassMissing())
  @test_throws ErrorException mean(dv, Symmetric(corrmat))
  @test_throws ErrorException mean(dv, acf)
  @test_throws ErrorException mean(dv)
end;

