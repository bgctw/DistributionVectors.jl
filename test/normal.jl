using Test, DistributionVectors
using Distributions, MissingStrategies, Missings
using LinearAlgebra

@testset "Normal" begin

@testset "two vars uncorrelated" begin
    # generate nSample values of two lognormal random variables
    d1 = Normal(2, 0.15)
    d2 = Normal(3, 0.25)
    dv = SimpleDistributionVector(d1, d2);
    #dsum = @inferred mean_normals(dv)
    dsum = @inferred sum(dv)
    dmean = @inferred mean(dv)
    @testset "no missings" begin
        @test params(dsum)[1] ≈ 5
        # checked with random numbers 
        #boot_dvsum(dv)
        @test params(dsum)[2] ≈ √(abs2(0.15) + abs2(0.25))
        @test dmean.μ ≈ 2.5
        @test dmean.σ/dmean.μ ≈ dsum.σ/dsum.μ 
        # squared mean uncertainty decreases by √n
        @test dmean.σ ≈ sqrt(mean([abs2(0.15),abs2(0.25)]))/√2
    end;
    @testset "with missings" begin
        dv = SimpleDistributionVector(d1, missing);
        @test isequal(collect(dv), [d1, missing])
        @test_throws Exception dsum2 = sum(dv) # error: forgot SkipMissing()
        dsum2 = @inferred sum(skipmissing(dv))
        @test dsum2 == d1
        dsum3 = @inferred sum(dv, SkipMissing())
        @test dsum3 == d1
        #@btime sum(skipmissing($dv)) # does not allocate
        dmean2 = @inferred mean(dv, SkipMissing())
        @test dmean2 == d1
    end;
    @testset "with gapfilling flag" begin
      dv = SimpleDistributionVector(d1, d2, d1);
      #dv = SimpleDistributionVector(d1, d1, d1);
      isgapfilled = [true, false, false]
      dsum = @inferred sum(dv, isgapfilled=isgapfilled)
      #@code_warntype sum(dv, isgapfilled)
      @test mean(dsum) == mean(sum(dv)) # mean takes all into account
      @test std(dsum) > std(sum(dv))
      # relative error like non-gapfilled
      dsumn = sum(dv[.!isgapfilled])
      @test std(dsum)/mean(dsum) == std(dsumn)/mean(dsumn)   
      # mean function
      dmean = @inferred mean(dv; isgapfilled=isgapfilled)
      @test dmean.μ ≈ dsum.μ / 3
      # relative error like non-gapfilled
      dmeann = mean(dv[.!isgapfilled])
      @test std(dmean)/mean(dmean) ≈ std(dmeann)/mean(dmeann)   
    end;
    @testset "with missings and gapfilling flag" begin
      dv = SimpleDistributionVector(d1, d2, d1, missing);
      isgapfilled = [true, true, false, false]
      dsum5 = @inferred sum(dv[1:3], isgapfilled=isgapfilled[1:3])
      dsum = @inferred sum(dv, SkipMissing(); isgapfilled=isgapfilled)
      #@code_warntype sum(dv, SkipMissing(); isgapfilled)
      @test dsum == dsum5
      # mean function
      dmean = @inferred mean(dv, SkipMissing(); isgapfilled=isgapfilled)
      dmeann = @inferred mean(dv[1:3], isgapfilled=isgapfilled[1:3])
      @test dmean == dmeann
    end;
end;

@testset "few correlated vars" begin
  mu = [110,100,80,120,160.0]
  sigma = [20.0,50,10,30,10]
  acf1 = @inferred AutoCorrelationFunction([1,0.4,0.1])
  n = length(mu)
  corrM = @inferred cormatrix_for_acf(n, acf1)
  dv = SimpleDistributionVector(Normal{eltype(mu)}, mu, sigma);
  mum = allowmissing(mu); mum[1] = missing
  dvm = SimpleDistributionVector(Normal{eltype(mu)}, mum, sigma)
  #
  @testset "matrix without missing" begin
    dsum = @inferred sum(dv, Symmetric(corrM))
    # checked with random numbers
    #boot_dvsums_acf(dv, acf1)
    @test mean(dsum) ≈ sum(mean.(dv))
    dsumuncorr = sum(dv)
    @test params(dsum)[2] > params(dsumuncorr)[2]
    # explicit sum over covariance Sigma
    Sigma = Diagonal(sigma) * corrM * Diagonal(sigma)
    @test params(dsum)[2] == sqrt(sum(Sigma))
    # mean function
    dmean = @inferred mean(dv, Symmetric(corrM))
    @test dmean.μ ≈ dsum.μ / length(dv)
    @test dmean.σ/dmean.μ ≈ dsum.σ/dsum.μ 
  end;
  @testset "matrix with missing" begin
    @test_throws Exception dsumm = sum(dvm, Symmetric(corrM))
    #S = similar(mum);
    dsumm = @inferred sum(dvm, Symmetric(corrM), SkipMissing())
    params(dsumm) == params(sum(dvm[2:end], Symmetric(corrM[2:end,2:end])))
    # mean function
    dmean = @inferred mean(dvm, Symmetric(corrM), SkipMissing())
    @test dmean.μ ≈ dsumm.μ / 4
    @test dmean.σ/dmean.μ ≈ dsumm.σ/dsumm.μ 
  end;
  @testset "with gapfilling flag" begin
    isgapfilled = fill(false, length(dv)); isgapfilled[4:end] .= true
    dsum4 = sum(dv, Symmetric(corrM))
    dsum = @inferred sum(dv, Symmetric(corrM), isgapfilled=isgapfilled)
    #@code_warntype sum(dv, isgapfilled)
    @test mean(dsum) == mean(dsum4)
    @test std(dsum) > std(dsum4)
    # test with explicit sum over Sigma and same relative error
    ifin = .!isgapfilled
    Sigma = Diagonal(sigma[ifin]) * corrM[ifin,ifin] * Diagonal(sigma[ifin])
    @test std(dsum)/mean(dsum) == sqrt(sum(Sigma))/sum(mu[ifin])
    # mean function
    dmean = @inferred mean(dv, Symmetric(corrM), SkipMissing(); isgapfilled=isgapfilled)
    @test dmean.μ ≈ dsum.μ / 5
    @test dmean.σ/dmean.μ ≈ dsum.σ/dsum.μ 
  end;
  @testset "with missings and gapfilling flag" begin
    isgapfilled = fill(false, length(dvm)); isgapfilled[4:end] .= true
    dsum4 = sum(dvm, Symmetric(corrM), SkipMissing())
    dsum4b = @inferred sum(
      dvm, Symmetric(corrM), SkipMissing(); isgapfilled=isgapfilled)
    #@code_warntype sum(dvm, isgapfilled)
    @test mean(dsum4b) == mean(dsum4)
    @test std(dsum4b) > std(dsum4)
    # test with explicit sum over Sigma and same relative error
    ifin = .!(isgapfilled .| ismissing.(dvm))
    Sigma = Diagonal(sigma[ifin]) * corrM[ifin,ifin] * Diagonal(sigma[ifin])
    @test std(dsum4b)/mean(dsum4b) == sqrt(sum(Sigma))/sum(mu[ifin])
    # 
    # acf variant
    dsum4c = @inferred sum(
      dvm, acf1, SkipMissing(); isgapfilled=isgapfilled)
    @test dsum4c == dsum4b
    # mean function
    dmean = @inferred mean(dvm, Symmetric(corrM), SkipMissing(); 
      isgapfilled=isgapfilled)
    @test dmean.μ ≈ dsum4b.μ / 4
    @test dmean.σ/dmean.μ ≈ dsum4b.σ/dsum4b.μ 
    dmean_acf = @inferred mean(dvm, acf1, SkipMissing(); isgapfilled=isgapfilled)
    @test dmean_acf == dmean
  end;
end;  

end;

