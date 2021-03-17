function sum(dv::AbstractDistributionVector{<:Normal}, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractVector{Bool} = Falses(length(dv)))
    length(dv) == length(isgapfilled) || error(
        "argument gapfilled must have the same length as dv ($(length(dv))" *
        "but was $(length(isgapfilled)).")
    if isa(ms, HandleMissingStrategy)
        # need to allocate anyway with subsetting
        nonmissing = findall(.!ismissing.(Ref(dv), axes(dv,1)))
        if !isempty(nonmissing) 
            return(sum(@inbounds(dv[nonmissing]), 
                isgapfilled = @inbounds(isgapfilled[nonmissing])))
        end
    end
    # uncorrelated, only sum diagonal
    Ssum = s = Ssumnonfilled = zero(eltype(nonmissingtype(eltype(dv))))
    nterm = 0
    for (i,d) in enumerate(dv)
        μ,σ = params(d)
        Ssum += μ
        if !isgapfilled[i]
            Ssumnonfilled += μ
            s += abs2(σ)
            nterm += 1
        end
    end
    nterm > 0 || error("Expected at least one nonmissing term, but mu = $μ")
    relerr = √s/Ssumnonfilled
    Normal(Ssum, Ssum * relerr)
end

function sum(dv::AbstractDistributionVector{D}, 
    acf::AutoCorrelationFunction, ms::MissingStrategy=PassMissing();
    isgapfilled::AbstractVector{Bool}=Falses(length(dv)),
    storage::AbstractVector{Union{Missing,DS}} = 
       Vector{Union{Missing,eltype(D)}}(undef, length(dv))) where 
    {D<:Normal, DS<:eltype(D)}
    # currently use Matrix-method, Maybe implement fast with loop
    corrM = Symmetric(cormatrix_for_acf(length(dv), coef(acf)))
    return(sum_normals(
        dv, corrM, ms;
        isgapfilled=isgapfilled, storage = storage))
end


function sum(dv::AbstractDistributionVector{D}, 
    corr::Symmetric, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractArray{Bool,1}=Falses(length(dv)),
    storage::AbstractVector{Union{Missing,DS}} = 
       Vector{Union{Missing,eltype(D)}}(undef, length(dv))) where 
    {D<:Normal, DS<:eltype(D)}
    sum_normals(dv, corr, ms; isgapfilled=isgapfilled, storage=storage)
end

function sum_normals(dv::AbstractDistributionVector{D}, 
    corr::Symmetric, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractArray{Bool,1} = Falses(length(dv)),
    storage::AbstractVector{Union{Missing,DS}} = 
        Vector{Union{Missing,eltype(D)}}(undef, length(dv))) where 
    {D<:Normal, DS<:eltype(D)}
    μ = params(dv, Val(1))
    σ = params(dv, Val(2))
    # var_sum (s) is the sum across all Sigma, i.e. σT * corr * σ
    # missings and gapfilled values do not count -> set to zero
    if isa(ms, HandleMissingStrategy)
        # check on missings in dv, σ may have more finite values
        storage .= ifelse.(ismissing.(Ref(dv),axes(dv,1)), zero(DS), σ) #coalesce(σ, zero(DS))  
        Spure = disallowmissing(storage) 
    else
        Spure = disallowmissing(σ)
    end 
    Spure[isgapfilled] .= zero(DS)
    s = transpose(Spure) * corr * Spure
    Ssum = Ssumnonfilled = zero(DS)
    nterm = 0
    for (i,d) in enumerate(dv)
        ismissing(d) && continue
        μ,σ = params(d)
        Ssum += μ
        if !isgapfilled[i]
            Ssumnonfilled += μ
            nterm += 1
        end
    end
    nterm > 0 || error("Expected at least one nonmissing term, but mu = $μ")
    relerr = √s/Ssumnonfilled
    Normal(Ssum, Ssum * relerr)
end


mean(dv::AbstractDistributionVector{<:Normal}, 
    ms::MissingStrategy=PassMissing(); kwargs...) =
    mean_normals(dv, ms; kwargs...)
mean(dv::AbstractDistributionVector{<:Normal}, corr::Symmetric, 
    ms::MissingStrategy=PassMissing(); kwargs...) =
    mean_normals(dv, corr, ms; kwargs...)
mean(dv::AbstractDistributionVector{<:Normal}, acf::AutoCorrelationFunction, 
    ms::MissingStrategy=PassMissing(); kwargs...) =
    mean_normals(dv, acf, ms; kwargs...)

function mean_normals(dv::AbstractDistributionVector{<:Normal}, x...; kwargs...) 
    ds = sum(dv, x...; kwargs...)
    n = count(i -> !ismissing(dv,i), axes(dv,1))
    Normal(ds.μ/n, ds.σ/n)
end


