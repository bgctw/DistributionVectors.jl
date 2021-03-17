# inference on Distribution type D not needed any more
# function paramtypes(::Type{D}) where D<:Distribution 
#     isconcretetype(D) || error("Expected a concrete distibution type," *
#         " Did you specify all type parameters, e.g. $D{Float64}?")
#     D.types
# end
# function tupleofvectype(::Type{D}) where D<:Distribution 
#     # https://discourse.julialang.org/t/allocations-in-comprehensions/21309/3
#     V = Tuple{ntuple(
#         i -> Vector{Union{Missing,paramtypes(D)[i]}}, length(paramtypes(D)))...}
# end

"""
    vectuptotupvec(vectup)

Typesafe convert from Vector of Tuples to Tuple of Vectors.

# Arguments
* `vectup`: A Vector of identical Tuples 

# Examples
```jldoctest; output=false, setup = :(using Distributions)
vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
vectuptotupvec(vectup) == ([1, 2], [1.01, 2.02], ["string 1", "string 2"])
# output
true
```
"""
function vectuptotupvec(vectup::
    AbstractVector{<:Union{Missing,NTuple{N, Any}}}) where N
    Ti = nonmissingtype(eltype(vectup)).parameters
    npar = length(Ti)
    Tim = ntuple(i -> Union{Missing,Ti[i]}, npar)
    #TiVec = ntuple(i -> Vector{Tim[i]}, npar)
    #vectupT = Tuple{TiVec...}
    # need to help collect both with eltype and return type
    ntuple(i -> 
        collect(Tim[i], passmissing(getindex).(vectup, i))::Vector{Tim[i]}, npar)
end

# function vectuptotupvec_old(vectup::AbstractVector{TT}) where 
#     {TT<:Union{Missing,NTuple{N, Any}}} where N
#     v1 = first(skipmissing(vectup))
#     types = typeof.(v1)
#     imiss = findall(ismissing, vectup) # unfortunately allocating
#     if length(imiss) != 0
#         # replace missing by a tuple of correct type (replaced later be missings)
#         # because there is no missing of correct type during getindex
#         vectupc = mappedarray((x -> ismissing(x) ? v1 : x), vectup)
#         function f(i) 
#             v = allowmissing(
#                 getindex.(vectupc,i))::Vector{Union{Missing,types[i]}}
#             v[imiss] .= missing
#             v
#         end
#         ntuple(f, length(v1))
#     else
#         ntuple((i ->
#             allowmissing(getindex.(vectup,i))::Vector{Union{Missing,types[i]}}
#         ), length(v1))
#     end
# end



"""
    AbstractDistributionVector{D <: Distribution}

Is any type able represent a vector of distribution of the same type.
This corresponds to a sequence of random variables, each characterized
by the same type of distribution but with different parameters. 
This allows aggregating functions to work, for
example, computing the distribution of the sum of random variables by 
[`sum(dv::AbstractDistributionVector)`](@ref).

It is parametrized by `D <: Distribution` defining the type of the distribution
used for all the random variables.

Items may be missing. Hence the element type of the iterator is 
`Union{Missing,D}`.

AbstractDistributionVector
- is iterable
- has length and index access, i.e. `dv[i]::D`
- access to entire parameter vectors: `params(dv,Val(i))`
- conversion to Tuple of Vectors: `params(dv)`
- array of random numbers: `rand(n, dv)`: adding one 
  dimension that represents across random variables
- query if entry is missing without needing to construct the 
  distribution entry: `ismissing(dv,i)`: 

Specific implementations,  need
to implement at minimum methods `length` and `getindex`, and `params`.

There are two standard implementations:
- [`SimpleDistributionVector`](@ref): fast indexing but slower `params` method 
- [`ParamDistributionVector`](@ref): possible allocations in indexing but 
  faster `params`

# Examples
```jldoctest; output = false, setup = :(using Distributions)
dmn1 = MvNormal([0,0,0], 1)
dmn2 = MvNormal([1,1,1], 2)
dv = SimpleDistributionVector(dmn1, dmn2, missing, missing);
sample = rand(dv,2);
# 4 distr, each 2 samples of length 3
size(sample) == (3,2,4)
# output
true
```
"""
abstract type AbstractDistributionVector{D <: Distribution} <: AbstractVector{Union{Missing,D}} end

size(dv::AbstractDistributionVector) = (length(dv),)

# the following already definded by AbstractVector{D}
#Base.eltype(::Type{<:AbstractDistributionVector{D}}) where D = Union{Missing,D}
#Base.ndims(::Type{<:AbstractDistributionVector}) = 1

# function Base.iterate(dv::AbstractDistributionVector, state=1) 
#     state > length(dv) ? nothing : (dv[state], state+1)
# end

# function Base.iterate(rds::Iterators.Reverse{AbstractDistributionVector}, 
#     state=length(rds.itr))  
#     state < 1 ? nothing : (rds.itr[state], state-1)
# end
# Base.firstindex(dv::AbstractDistributionVector) = 1
# Base.lastindex(dv::AbstractDistributionVector) = length(dv)


# function Base.getindex(dv::AbstractDistributionVector, i::Number) 
#     @info "getindex(dv::AbstractDistributionVector, i::Number)"
#     dv[convert(Int, i)]
# end

function params(dv::AbstractDistributionVector, ::Val{i}) where i
    # need to help compile to determine the type of tupvec
    T = typeof(params(first(skipmissing(dv)))[i])
    Tm = Union{Missing,T}
    collect(Tm, passmissing(getindex).(passmissing(params).(dv),i))::Vector{Tm}
end

# extends Random
function rand(dv::AbstractDistributionVector, n::Integer) 
    x1 = rand(first(skipmissing(dv)), n)
    xm = Fill(missing,size(x1))
    #xm = fill(missing,size(x1))
    fmiss(x)::Union{typeof(xm),typeof(x1)} = (ismissing(x) ? xm : rand(x,n))
    vecarr = convert(Vector{Union{typeof(xm),typeof(x1)}}, fmiss.(dv))::Vector{Union{typeof(xm),typeof(x1)}}
    VectorOfArray(vecarr)
end
function rand(dv::AbstractDistributionVector)
    x1 = rand(first(skipmissing(dv))) 
    xm = Fill(missing,size(x1))
    #xm = fill(missing,size(x1))
    fmiss(x)::Union{typeof(xm),typeof(x1)} = (ismissing(x) ? xm : rand(x)) 
    vecarr = convert(Vector{Union{typeof(xm),typeof(x1)}}, fmiss.(dv))::Vector{Union{typeof(xm),typeof(x1)}}
    nonmissingtype(eltype(dv)) <: UnivariateDistribution ? vecarr : VectorOfArray(vecarr)
end
function rand!(vecarr::AbstractArray{T}, dv::AbstractDistributionVector) where T
    x1 = rand(first(skipmissing(dv))) 
    xm = Fill(missing,size(x1))
    #xm = fill(missing,size(x1))
    fmiss(x)::Union{typeof(xm),typeof(x1)} = (ismissing(x) ? xm : rand(x)) 
    vecarr .= fmiss.(dv)
end
function ismissing(dv::AbstractDistributionVector, i::Int)
    ismissing(dv[i])
end

## SimpleDistributionVector   
"""
    SimpleDistributionVector{D <: Distribution, V}

Is an Vector-of-Distribution based implementation of 
[`AbstractDistributionVector`](@ref).

Vector of random var can be created by 
- specifying the distributions as arguments.
```jldoctest; output = false, setup = :(using Distributions)
d1 = LogNormal(log(110), 0.25)
d2 = LogNormal(log(100), 0.15)
dv = SimpleDistributionVector(d1, d2, missing);
isequal(params(dv, Val(1)), [log(110), log(100), missing])
# output
true
```

- providing the Type of distribution and vectors of each parameter
```jldoctest; output = false, setup = :(using Distributions)
mu = [1.1,1.2,1.3]
sigma = [1.01, 1.02, missing]
dv = SimpleDistributionVector(LogNormal{eltype(mu)}, mu, sigma);
isequal(params(dv, Val(1)), [1.1,1.2,missing])
# output
true
```
Note that if one of the parameters is missing, then the entire entry of
the distribution is marked missing.

Since Distributions are stored directly, indexing passes a reference.
However, getting parameter vectors, required iterating all distributions, 
and allocating a new vector.
"""
struct SimpleDistributionVector{D <: Distribution, V <: AbstractVector} <: 
    AbstractDistributionVector{D} 
    dvec::V
    # inner constructor checking ?
end

function SimpleDistributionVector(::Type{D}, dvec::V) where 
{D<:Distribution,  V<:AbstractVector} 
    isconcretetype(D) || error("Expected a concrete distibution type," *
        " Did you specify all type parameters, e.g. $D{Float64}?")
    Missing <: eltype(V) || error(
        "Expected type of parameters to allow for missing." *
        " Can you use 'allowmissing' in constructing the " *
        "SimpleDistributionVector?")
    eltype(V) <: Union{Missing, <:D} || error(
        "Expected type of parameters of 'Union{Missing, $(D)}' "*
        " but got $(eltype(V)).")
    SimpleDistributionVector{D, V}(dvec)
end

function SimpleDistributionVector(dv::Vararg{Union{Missing,D},N}) where 
    {D<:Distribution, N} 
    N == 0 && error(
        "Provide at least one argument, i.e. distribtution," *
        "i n SimpleDistributionVector(x...).")
    d1 = first(skipmissing(dv))
    dvec = collect(Union{Missing, typeof(d1)}, dv)::Vector{Union{Missing, typeof(d1)}}
    SimpleDistributionVector(D, allowmissing(dvec))
end

function SimpleDistributionVector(::Type{D}, pvec::Vararg{Any,N}) where 
    {D<:Distribution, N} 
    #Union{Missing,typeof(pvec[1])}
    pvecm = ntuple(i ->
        allowmissing(pvec[i])::Vector{Union{Missing,eltype(pvec[i])}}, N)
    # if one parameter has missing, the entire tuple must be set to missing
    dvec = allowmissing(collect(zip(pvecm...)))  
    for i in axes(dvec,1)
        if any(ismissing.(dvec[i]))
            dvec[i] = missing
        end
    end
    Tm = Union{Missing, D}
    dv = collect(Tm,
        ismissing(x) ? missing : D(x...) for x in dvec)::Vector{Tm}
    SimpleDistributionVector(D, dv)
end


length(dv::SimpleDistributionVector) = length(dv.dvec)
IndexStyle(::Type{<:SimpleDistributionVector{D,V}}) where {D,V} = IndexStyle(V)
similar(dv::SimpleDistributionVector{D,V}) where {D,V} = 
    SimpleDistributionVector{D,V}(similar(dv.dvec))

getindex(dv::SimpleDistributionVector,i::Int) = dv.dvec[i] #::Union{Missing, D}
#function getindex(dv::SimpleDistributionVector, I::Union{AbstractVector{<:Integer},AbstractVector{<:Bool}})
function getindex(dv::SimpleDistributionVector, I) # ambiguous
    # # Catesian indexing?: provided a tuple -> get only a single value in printing
    #length(I) != 1 && return dv[I[1]]
    #dvecsub = dv.dvec[I...]
    dvecsub = dv.dvec[I]
    # #@show I, length(I), typeof(I), typeof(dvecsub)
    # length(I) != 1 && return(dvecsub)
    typeof(dv)(dvecsub)
end
setindex!(dv::SimpleDistributionVector, d, i::Int) = dv.dvec[i] = d
#Base.setindex!(dv::SimpleDistributionVector, d, I::Vararg{Int, N}) where N = dv.dvec[I...] = d

# params(i) already defined as default in AbstractDistributionVector
# function StatsBase.params(dv::SimpleDistributionVector, i::Integer) 
#    # mappedarray(e -> passmissing(getindex)(e,i), dv.dvec)
#    # currentl does not work, see 
# https://github.com/JuliaArrays/MappedArrays.jl/issues/40
#    passmissing(getindex).(passmissing(params).(dv.dvec),i)
# end

function params(dv::SimpleDistributionVector)
    Ti = typeof(params(first(skipmissing(dv.dvec)))).parameters
    npar = length(Ti)
    Tim = ntuple(i -> Union{Missing, Ti[i]}, npar)
    f(x,i) = ismissing(x) ? missing : getindex(params(x),i)
    ntuple(i -> collect(Tim[i], f.(dv.dvec, i))::Vector{Tim[i]}, npar)
end



## ParamDistributionVector
"""
    ParamDistributionVector{D <: Distribution, V}
   
Is an Tuple of Vectors based implementation of 
[`AbstractDistributionVector`](@ref).

Vector of random var can be created by 
- specifying the distributions as arguments with some overhead of converting
  the Distributions to vectors of each parameter
```jldoctest; output = false, setup = :(using Distributions)
d1 = LogNormal(log(110), 0.25)
d2 = LogNormal(log(100), 0.15)
dv = ParamDistributionVector(d1, d2, missing);
isequal(params(dv, Val(1)), [log(110), log(100), missing])
# output
true
```

- providing the Type of distribution and vectors of each parameter
```jldoctest; output = false, setup = :(using Distributions)
mu = [1.1,1.2,1.3]
sigma = [1.01, 1.02, missing]
dv = ParamDistributionVector(LogNormal{eltype(mu)}, mu, sigma);
ismissing(dv[3])
isequal(params(dv, Val(1)), [1.1,1.2,1.3]) # third still not missing here
# output
true
```
Note that if one of the parameters for entry `i` is missing, then `dv[i]`
is missing.

Since distributions are stored by parameter vectors, the acces to these
vectors is just passing a reference.
Indexing, will create Distribution types.
"""
struct ParamDistributionVector{D <: Distribution, V <: Tuple} <: 
    AbstractDistributionVector{D} 
    params::V
    # inner constructor checking ?
end

function ParamDistributionVector(::Type{D}, params::V) where 
{D<:Distribution,  V<:Tuple} 
    isconcretetype(D) || error("Expected a concrete distibution type," *
        " Did you specify all type parameters, e.g. $D{Float64}?")
    all(map((x -> x <: AbstractVector), V.parameters)) || error(
        "Expected all entries in Tuple param to be AbstractVectors.")
    all(map((x -> Missing <: eltype(x)), V.parameters)) || error(
        "Expected type of each vector in params to allow for missing.")
    lenparams = map(length, params)
    all(map(x -> x == first(lenparams), lenparams)) || error(
        "Expected all vectors in params tuple to be of the same length.")
    ParamDistributionVector{D, V}(params)
end

function ParamDistributionVector(dtup::Vararg{Union{Missing,D},N}) where 
    {D<:Distribution, N} 
    N == 0 && error(
        "Provide at least one distribution in ParamDistributionVector(x...).")
    # need to help compile to determine the type of tupvec
    Ti = typeof(params(first(skipmissing(dtup)))).parameters
    npar = length(Ti)
    tupvec = ntuple(i -> begin
        Tim = Union{Missing, Ti[i]}
        collect(Tim, passmissing(getindex).(passmissing(params).(
            dtup),i))::Vector{Tim} 
    end, npar)
    ParamDistributionVector(D, tupvec)
end


function ParamDistributionVector(::Type{D}, pvec::Vararg{Any,N}) where 
    {D<:Distribution, N} 
    pvecm = ntuple(i -> allowmissing(pvec[i]), N)
    ParamDistributionVector(D, pvecm)        
end


length(dv::ParamDistributionVector) = length(first(dv.params))
IndexStyle(::Type{<:ParamDistributionVector{D,V}}) where {D,V} = 
    IndexStyle(first(V.parameters))

similar(dv::ParamDistributionVector{D,V}) where {D,V} = 
    typeof(dv)(ntuple(i->similar(dv.params[i]), length(dv.params)))

function ismissing(dv::ParamDistributionVector, i::Int; 
    params_i = getindex.(dv.params, Ref(i)))
    ismissingt = ntuple(i->ismissing(params_i[i]), length(params_i))
    any(ismissingt)
end
function getindex(dv::ParamDistributionVector, i::Int)
    params_i = getindex.(dv.params, Ref(i))
    ismissing(dv,i; params_i = params_i) && return missing
    nonmissingtype(eltype(dv))(params_i...)
end
function getindex(dv::ParamDistributionVector, I)
    tupvec = map(x -> x[I], dv.params)
    typeof(dv)(tupvec)
    #ParamDistributionVector{D,V}(tupvec)
    #Base.typename(DV).wrapper((dv[i] for i in I)...)
end
function setindex!(dv::ParamDistributionVector, d, i::Int)
    p = ismissing(d) ? ntuple(_->missing,length(dv.params)) : params(d)
    for ti in eachindex(dv.params)
        dv.params[ti][i] = p[ti] 
    end
end



function params(dv::ParamDistributionVector, ::Val{i}) where i
    # if types of parameters differ, then a union type is returned -> need Val
    dv.params[i]
end

params(dv::ParamDistributionVector) = dv.params

