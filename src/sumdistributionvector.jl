"""
    sum(dv::AbstractDistributionVector, ms=PassMissing(); <keyword arguments>)
    sum(dv::AbstractDistributionVector, corr, ms=PassMissing(); <keyword arguments>)
    sum(dv::AbstractDistributionVector, acf, ms=PassMissing(); <keyword arguments>)
    
    mean(dv::AbstractDistributionVector, ms=PassMissing(); <keyword arguments>)
    mean(dv::AbstractDistributionVector, corr, ms=PassMissing(); <keyword arguments>)
    mean(dv::AbstractDistributionVector, acf, ms=PassMissing(); <keyword arguments>)

Compute the distribution of the sum or mean of correlated random variables.

# Arguments
- `dv`: The vector of distributions, see [`AbstractDistributionVector`](@ref)
- `ms`: `MissingStrategy`: set to `SkipMissing()` or `ExactMissing()`to 
    consciously care for missing values in `dv`.

An optional second arguments supports correlation between random variables.
- `corr::Symmetric`: correlation matrix, or
- `acf::AutoCorrelationFunction`: coefficients of the 
   [`AutoCorrelationFunction`](@ref)

Keyword arguments:
- `isgapfilled::AbstractVector{Bool}`: set to true for records that should
   contribute to the sum but not to the decrease of relative uncertainty
   with increasing number of records, e.g. for missing records that have
   been estimated (gapfilled). 

The sums of correlated variables require extra allocation and 
support an additional keyword parameter  
- `storage`: a mutable `AbstractVector{eltype(D)}` of length of `dv` 
  that provides storage space to avoid additional allocations.

When implementing sum and mean for another Distribution `MyDist`, 
the `AutoCorrelationFunction`-based method falls back to the `Symmetric` 
correlation based method. Hence one only needs to define methods
```julia
sum(dv::AbstractDistributionVector{<:MyDist}, 
    ms::MissingStrategy=PassMissing())
sum(dv::AbstractDistributionVector{<:MyDist}, corr::Symmetric, 
    ms::MissingStrategy=PassMissing())
```
"""
function sum(dv::AbstractDistributionVector, ms::MissingStrategy=PassMissing(); kwargs...)
    error("sum not defined yet for " * 
    "Distributionvector{$(nonmissingtype(eltype(dv)))}")
end,
function sum(dv::AbstractDistributionVector, acf::AutoCorrelationFunction, ms::MissingStrategy=PassMissing(); kwargs...)
    corrmat = cormatrix_for_acf(length(dv), acf)
    sum(dv, Symmetric(corrmat), ms)
end, 
function sum(dv::AbstractDistributionVector, corr::Symmetric, ms::MissingStrategy=PassMissing(); kwargs...)
    error("sum with correlations not defined yet for " * 
    "Distributionvector{$(nonmissingtype(eltype(dv)))}")
end, 
function mean(dv::AbstractDistributionVector, ms::MissingStrategy=PassMissing(); kwargs...)
    error("mean not defined yet for " * 
    "Distributionvector{$(nonmissingtype(eltype(dv)))}")
end,
function mean(dv::AbstractDistributionVector, acf::AutoCorrelationFunction, ms::MissingStrategy=PassMissing(); kwargs...)
    corrmat = cormatrix_for_acf(length(dv), acf)
    mean(dv, Symmetric(corrmat), ms)
end,
function mean(dv::AbstractDistributionVector, corr::Symmetric, ms::MissingStrategy=PassMissing(); kwargs...)
    error("mean with correlations not defined yet for " * 
    "Distributionvector{$(nonmissingtype(eltype(dv)))}")
end

