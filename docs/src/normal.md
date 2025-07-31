## Sum and mean of correlated normal random variables

### Dealing with gapfilled records

Method [`sum`](@ref) for a `DistributionVector{<:Normal}`
takes into account that some of the random variables should contribute to
the overall sum and mean but should not 
contribute to the reduction of relative uncertainty. 
For example, some of them may have been estimated based on observed
data, i.e. gapfilled.

The average (root mean squared) standard deviation of the mean of 
uncorrelated normal random variables decreases with $\sqrt{n}$ 
with number of records.

```math
\begin{aligned}
m &= {S \over n} = {\sum_i x_i \over n} \text{ with } x_i \sim N(\mu_i, \sigma_i) 
\\
Var(m) = \sigma^2_m &= Var\left(\sum_i x_i \over n\right) = 
  {1 \over n^2} Var\left( \sum_i x_i \right) = {1 \over n^2} \sum_i \sigma^2_i = 
  {1 \over n^2} n \overline{ \sigma^2_i}  = {\overline{ \sigma^2_i} \over n} 
\\
\operatorname{se}(x) = \operatorname{std}(m) = \sigma_m &= 
  {\sqrt{\overline{ \sigma^2_i}} \over \sqrt{n}} = 
  \operatorname{rmse}(\sigma_i) / \sqrt{n}
\end{aligned} 
```

The [`sum`](@ref) function therefore support keyword argument `isgapfilled`, 
which is a boolean vector of the length of the sequence of random variables.
Set the positions to `true` for those records that should contribute to relative 
uncertainty reduction.
The distribution parameters are first computed first for 
the records that are not gapfilled.
Next scale parameter for the entire series is adjusted so that the
relative error matches that of the subset of non-gapfilled variables.

In the following example 
- the expected value of the mean of a series containing 
  gapfilled records is the same as the mean for the same series where 
  all records are observed. 
- However, the relative error of the mean is 
  larger than the relative error of the non-gapfilled series and
- matches the relative error of the series excluding the gapfilled records.

```jldoctest sumnormals; output = false, setup = :(using Statistics,StatsBase,Distributions,DistributionVectors)
μ = [110,100,80,120,160.0];
σ = [20.0,50,10,30,10];
dv = SimpleDistributionVector(Normal{eltype(μ)}, μ, σ);

acf = AutoCorrelationFunction([0.4,0.1]);
isgapfilled = fill(false, length(dv)); isgapfilled[4:end] .= true;

dsum = sum(dv, acf; isgapfilled=isgapfilled)

# same mean but larger relative than with ignoring gap-filled flag:
dsum_ig = sum(dv, acf);
mean(dsum) == mean(dsum_ig)
relerr(x) = std(x)/mean(x)
relerr(dsum) > relerr(dsum_ig)
# relative error is equal to the sum across true observations
dsum_obs = sum(dv[.!isgapfilled], acf);
mean(dsum) > mean(dsum_obs)
relerr(dsum) == relerr(dsum_obs)
# output
true
``` 




