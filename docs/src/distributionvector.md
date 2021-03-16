# Vector of random variables, i.e. distributions

```@docs
AbstractDistributionVector
SimpleDistributionVector
ParamDistributionVector
```

## Helpers

The conversion between a missing-allowed vector of parameter tuples 
to a tuple of vectors for each parameter 
(as used by [`ParamDistributionVector`](@ref))
is provided in a
type-stable manner by function `vectuptotupvec`.

```@docs
vectuptotupvec
```