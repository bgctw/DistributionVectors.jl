using DistributionVectors
using Test, Distributions, LinearAlgebra, Missings
using BenchmarkTools: @btime
using Plots, StatsPlots

function benchmarkSums()
    nrep = 30
    mu = log.(rand(Normal(120, 10), nrep));
    sigma = log.(rand(Normal(1.2, 0.1), nrep));
    acf1 = [0.8,0.3,0.1];
    dv = SimpleDistributionVector(LogNormal{eltype(mu)}, mu, sigma)
    @btime dsum_v = sum($dv, $acf1, SkipMissing(); method = Val(:vector) )
    @btime dsum_m = sum($dv, $acf1, SkipMissing(); method = Val(:bandedmatrix) )
    # the Bandematrix based slower after optimizing the vector for stability in missings
    #
    # repeat with missings
    mum = allowmissing(copy(mu)); mum[1] = missing
    dv = SimpleDistributionVector(LogNormal{nonmissingtype(eltype(mum))}, mum, sigma)
    @btime dsum_v = sum($dv, $acf1, SkipMissing(); method = Val(:vector) )
    @btime dsum_m = sum($dv, $acf1, SkipMissing(); method = Val(:bandedmatrix) )
    #
    #S = similar(mum);
    @code_warntype sum_lognormals(S, dv, acf1, SkipMissing())
    f() = sum_lognormals(S, dv, acf1, SkipMissing())
    @time f(); @time f()



    # # try allocating instead of view (replace line of view_nonmissing)
    # # allocating is faster
    # function sum_lognormals2!(S, dv, corr::AbstractMatrix, 
    # ms::MissingStrategy=PassMissing())
    #     parms = params(dv)
    #     μ = @view parms[1,:]
    #     σ = @view parms[2,:]
    #     # S = allowmissing(similar(μ))
    #     @. S = exp(μ + abs2(σ)/2)
    #     nmissing = count(ismissing, S)
    #     anymissing = nmissing != 0
    #     !(isa(ms, HandleMissingStrategy)) && anymissing && error(
    #          "Found missing values. Use argument 'SkipMmissing()' to sum over nonmissing.")
    #     Ssum::nonmissingtype(eltype(S)) = sum(skipmissing(S))
    #     @. S = σ * S  # do only after Ssum
    #     # setting S to zero results in summing zero for missing records
    #     # which is the same as filtering both S and corr
    #     anymissing && replace!(S, missing => 0.0)
    #     #s = transpose(disallowmissing(S)) * corr * disallowmissing(S)
    #     #Spure = view_nonmissing(S) # non-allocating
    #     Spure = disallowmissing(S) # allocating
    #     s = transpose(Spure) * corr * Spure
    #     σ2eff = s/abs2(Ssum)
    #     μ_sum = log(Ssum) - σ2eff/2
    #     #@show Ssum, s, length(S) - nmissing
    #     LogNormal(μ_sum, √σ2eff)  
    # end
    # @btime sum_lognormals($storage, $dv, $corMa, SkipMissing())
    # @btime sum_lognormals2!($storage, $dv, $corMa, SkipMissing())
end


function bootstrap_sums_lognormal()
    #using StatsBase
    nObs = 5
    xTrue = fill(10.0, nObs)
    sigmaStar = fill(1.5, nObs) # multiplicative stddev of 1.5
    dv = SimpleDistributionVector(fit.(LogNormal, xTrue, Σstar.(sigmaStar))...)
    dsum = sum(dv)
    nboot = 100_000
    x = rand(dv, nboot);
    sums = vec(sum(x, dims = 2));
    #stds = std.(Ref(x));
    cdf_sums = ecdf(sums);
    @test cdf_sums(median(dsum)) ≈ 0.5 rtol = 0.02
    @test cdf_sums(quantile(dsum, [0.025,0.975])) ≈ [0.025, 0.975] rtol = 0.02
end

function fplot(dsum, sums)
    plot(dsum, lab="computed", xlabel="sum", ylabel="density")
    density!(sums, lab="random sample")
    vline!([mean(dsum)], lab="mean computed")
    vline!([mean(sums)], lab="mean random")
    vline!(quantile(dsum, [0.025, 0.975]), lab="cf computed")
    vline!(quantile(sums, [0.025, 0.975]), lab="cf random")
end

function bootstrap_sums_lognormal_acf()
    #using StatsBase    
    mu = log.([110,100,80,120,160.0])
    sigma = log.([1.2,1.5,1.1,1.3,1.1])
    acf1 = [0.4,0.1]
    dv = SimpleDistributionVector(LogNormal{eltype(mu)}, mu, sigma);
    #ps = 0.025:0.025:0.975; qs = quantile.(dv[1], ps); plot(qs, pdf(dv[1],qs))
    drsum = boot_dvsums_acf(dv, acf1)
    dsum = sum(dv, acf1)
    @test dsum ≈ drsum rtol = 0.02
    # see plotting function in bootstrap_sums_lognormal above
end

  

