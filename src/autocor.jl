"""
    AutoCorrelationFunction{T}

A representation of the autocorrelation function.

It supports accessing the coeficients starting from lag 0 by
- `coef(acf::AutoCorrelationFunction)`: implements StatsBase.coef
- `coef(acf::AutoCorrelationFunction, lag::Integer)`: coefficient for lag

Wrapping the vector of coefficients into its own type helps avoiding
method ambiguities.

# Examples
```jldoctest am; output = false
using StatsBase: coef
acf = AutoCorrelationFunction([1,0.4,0.1])
coef(acf) == [1,0.4,0.1]
coef(acf,1) == 0.4
# output
true
```
"""
struct AutoCorrelationFunction{T}
    coef::T
end
# implements StatsBase coef
AutoCorrelationFunction(coef::AbstractVector{<:Number}) = 
    AutoCorrelationFunction{typeof(coef)}(coef)
coef(acf::AutoCorrelationFunction) = acf.coef
coef(acf::AutoCorrelationFunction, lag::Integer) = acf.coef[lag+1]

cormatrix_for_acf(n::Int,acf::AutoCorrelationFunction) =
    cormatrix_for_acf(n, coef(acf))

function cormatrix_for_acf(n::Int,acf::AbstractVector) 
    nacf::Int = length(acf)
    corrM = BandedMatrix{Float64}(undef, (n,n), (nacf-1,nacf-1))
    corrM[band(0)] .= acf[1]
    for i in 1:(nacf-1)
      corrM[band(i)] .= corrM[band(-i)] .= acf[i+1]
    end
    corrM
end
