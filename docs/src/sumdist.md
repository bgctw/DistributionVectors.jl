# Aggreatation of random variables

## Sum and Mean of several correlated random variables
```@docs
sum(::AbstractDistributionVector)
```

## Helpers
### AutoCorrelationFunction

The autocorrelation function describes the correlation of random variables
of a time series at increasing lags, i.e. ``cor(x_{i+2}, x_{i})`` for lag 2.
It is a Real-valued vector with the first entry the correlation at lag 0.

To support unambiguous dispatch on an autocorrelation function, type
`AutoCorrelationFunction` wraps such a vector.

```@docs
AutoCorrelationFunction
```
