## Sum and mean of correlated lognormal random variables

Method sum for a `DistributionVector{<:LogNormal}`
computes approximation of the distribution of the sum of the
corresponding lognormal variables.

See documentation of the [`sum`](@ref) function.

In the following example the computed approximation is compared
to a bootstrap sample of sums over three correlated random variables.

```@example boot
using Distributions,DistributionVectors
mu = log.([110,100,80])
sigma = log.([1.2,1.5,1.1])
acf0 = AutoCorrelationFunction([1,0.4,0.1])
dv = SimpleDistributionVector(LogNormal{eltype(mu)}, mu, sigma);
dsum = sum(dv, acf0)
```

```@setup boot
using StatsPlots,LinearAlgebra, Missings
function boot_dvsums_acf(dv, acf, nboot = 10_000)
    μ, σ = params(dv)
    Sigma = Diagonal(σ) * cormatrix_for_acf(length(dv), acf) * Diagonal(σ);
    dn = MvNormal(disallowmissing(μ), Symmetric(Sigma));
    x = rand(dn, nboot) .|> exp
    sums = vec(sum(x, dims = 1))
end
sums = boot_dvsums_acf(dv, acf0); 
@assert isapprox(dsum, fit(LogNormal, sums), rtol = 0.2) 
p = plot(dsum, lab="computed", xlabel="sum of 3 correlated lognormally distributed random variables", ylabel="density");
density!(p, sums, lab="random sample");
vline!(p, [mean(dsum)], lab="mean computed");
vline!(p, [mean(sums)], lab="mean random");
vline!(p, quantile(dsum, [0.025, 0.975]), lab="cf computed");
vline!(p, quantile(sums, [0.025, 0.975]), lab="cf random");
plot(p)
savefig("sumlognormals.svg"); nothing
```

![plot of sum of lognormals](sumlognormals.svg)


