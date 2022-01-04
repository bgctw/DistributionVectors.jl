function sum(dv::AbstractDistributionVector{<:LogNormal}, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractVector{Bool} = Falses(length(dv)))
    length(dv) == length(isgapfilled) || error(
        "argument gapfilled must have the same length as dv ($(length(dv))" *
        "but was $(length(isgapfilled)).")
    if isa(ms, HandleMissingStrategy)
        nonmissing = findall(.!ismissing.(dv))
        #return( dv[nonmissing], isgapfilled[nonmissing])
        if !isempty(nonmissing) 
            return(sum(@inbounds(dv[nonmissing]), 
                isgapfilled = @inbounds(isgapfilled[nonmissing])))
        end
    end
    # uncorrelated, only sum diagonal
    Ssum = s = Ssumnonfilled = zero(eltype(nonmissingtype(eltype(dv))))#zero(T)
    nterm = 0
    for (i,d) in enumerate(dv)
        μ,σ = params(d)
        Si = exp(μ + abs2(σ)/2)
        Ssum += Si
        if !isgapfilled[i]
            Ssumnonfilled += Si
            s += abs2(σ) * abs2(Si)
            nterm += 1
        end
    end
    nterm > 0 || error("Expected at least one nonmissing term, but mu = $μ")
    σ2eff = s/abs2(Ssumnonfilled)
    μ_sum = log(Ssum) - σ2eff/2
    LogNormal(μ_sum, √σ2eff)
end


function sum(dv::AbstractDistributionVector{D}, 
    acf::AutoCorrelationFunction, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractVector{Bool} = Falses(length(dv)), 
    storage::AbstractVector{Union{Missing,ST}} = 
        Vector{Union{Missing,eltype(D)}}(undef, length(dv)),
    method::Val{M} = Val(:vector)) where 
    {D<:LogNormal, ST<:eltype(D), M} 
    #storage = Vector{Union{Missing,eltype(D)}}(undef, length(dv))
    if M == :vector
        return(sum_lognormals(
            dv, acf, ms, isgapfilled=isgapfilled, storage = storage))
    end
    if M == :bandedmatrix
        corrM = Symmetric(cormatrix_for_acf(length(dv), coef(acf)))
        return(sum_lognormals(
            dv, corrM, ms; isgapfilled=isgapfilled, storage = storage))
    end
    error("Unknown method $method. Supported are Val(:vector) and Val(:bandedmatrix)")
end

function sum_lognormals(dv::AbstractDistributionVector{D}, 
    acf::AutoCorrelationFunction, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractVector{Bool} = Falses(length(dv)),
    storage::AbstractVector{Union{Missing,DS}} = 
       Vector{Union{Missing,eltype(D)}}(undef, length(dv))) where 
    #{D<:LogNormal, ST<:, B}
    {D<:LogNormal, DS<:eltype(D)}
    #details<< Implements estimation according to
    # Messica A(2016) A simple low-computation-intensity model for approximating
    # the distribution function of a sum of non-identical lognormals for
    # financial applications. 10.1063/1.4964963
    μ = params(dv, Val(1))
    σ = params(dv, Val(2))
    coef_acf1 = @view coef(acf)[2:end]
    corrlength = length(coef_acf1)
    #acfm = vcat(reverse(coef_acf), 1, coef_acf)
    #use OffsetArrays so that index corresponds to lag: acfm[0] == 1
    acfm = OffsetArray(vcat(reverse(coef_acf1), 1, coef_acf1), -length(coef_acf1)-1)
    n = length(μ)
    @. storage = exp(μ + abs2(σ)/2)
    nmissing = count(ismissing.(storage))
    !(isa(ms, HandleMissingStrategy)) && nmissing != 0 && error(
        "Found missing values. Use argument 'SkipMissing()' " *
        "to sum over nonmissing.")
    # 0 in storage has the effect of not contributing to Ssum nor s 
    # For excluding terms of gapfilled records, must make 
    # to either set storage[gapfilled] to zero or check in the product
    Spure = disallowmissing(replace(storage, missing => 0.0))
    σpure = disallowmissing(replace(σ, missing => 0.0))
    Ssum = sum(Spure)
    # after computing Ssum, can also set gapfilled to zero
    Spure[isgapfilled] .= 0.0
    s = Ssumunfilled = zero(eltype(σpure))
    for i in 1:n
        iszero(Spure[i]) && continue # nothing added
        Ssumunfilled += Spure[i]
        jstart = max(1, i - corrlength)
        jend = min(n, i + corrlength)
        for j in jstart:jend
            #acf_ind = (j-i + corrlength +1)
            # sij will be zero if sigma or storage is missing (replaced by zero)
            # Sj moved to start to help multiplication by early zero
            sij = Spure[j] * acfm[j-i] * σpure[i] * σpure[j] * Spure[i] 
            s += sij
        end
    end
    σ2eff = s/abs2(Ssumunfilled)
    μ_sum = log(Ssum) - σ2eff/2
    tmp = μ_sum, √σ2eff
    #@show Ssum, s, n - nmissing
    LogNormal(μ_sum, √σ2eff)
end

function sum(dv::AbstractDistributionVector{D}, 
    corr::Symmetric, ms::MissingStrategy=PassMissing(); 
    isgapfilled::AbstractArray{Bool,1}=Falses(length(dv)),
    storage::AbstractVector{Union{Missing,DS}} = 
       Vector{Union{Missing,eltype(D)}}(undef, length(dv))) where 
    {D<:LogNormal, DS<:eltype(D)}
    sum_lognormals(
        dv, corr, ms; isgapfilled=isgapfilled, storage = storage)
end

function sum_lognormals(dv::AbstractDistributionVector{D}, 
    corr::Symmetric, ms::MissingStrategy=PassMissing();
    isgapfilled::AbstractVector{Bool} = Falses(length(dv)),
    storage::AbstractVector{Union{Missing,DS}} = 
        Vector{Union{Missing,eltype(D)}}(undef, length(dv))) where 
    {D<:LogNormal, DS<:eltype(D)}
    μ = params(dv, Val(1))
    σ = params(dv, Val(2))
    # storage = allowmissing(similar(μ))
    @. storage = exp(μ + abs2(σ)/2)
    nmissing = count(ismissing, storage)
    anymissing = nmissing != 0
    !(isa(ms, HandleMissingStrategy)) && anymissing && error(
         "Found missing values. Use argument 'SkipMissing()' " *
         "to sum over nonmissing.")
    Ssum::nonmissingtype(eltype(storage)) = sum(skipmissing(storage))
    # gapfilled records only used for Ssum, can set the to 0 now
    # so they do not contribute to s and Ssumfilled for computation of σ2eff
    storage[isgapfilled] .= 0
    Ssumunfilled::nonmissingtype(eltype(storage)) = sum(skipmissing(storage))
    @. storage = σ * storage  # do only after Ssum
    # setting storage to zero results in summing zero for missing records
    # which is the same as filtering both storage and corr
    anymissing && replace!(storage, missing => 0.0)
    #s = transpose(disallowmissing(storage)) * corr * disallowmissing(storage)
    #Spure = view_nonmissing(storage) # non-allocating
    Spure = disallowmissing(storage) # allocating - tested: is faster than the view
    s = transpose(Spure) * corr * Spure
    σ2eff = s/abs2(Ssumunfilled)
    μ_sum = log(Ssum) - σ2eff/2
    #@show Ssum, s, length(storage) - nmissing
    LogNormal(μ_sum, √σ2eff)  
end

mean(dv::AbstractDistributionVector{<:LogNormal}, 
    ms::MissingStrategy=PassMissing(); kwargs...) =
    mean_lognormals(dv, ms; kwargs...)
mean(dv::AbstractDistributionVector{<:LogNormal}, corr::Symmetric, 
    ms::MissingStrategy=PassMissing(); kwargs...) =
    mean_lognormals(dv, corr, ms; kwargs...)
mean(dv::AbstractDistributionVector{<:LogNormal}, acf::AutoCorrelationFunction, 
    ms::MissingStrategy=PassMissing(); 
    kwargs...) =
    mean_lognormals(dv, acf, ms; kwargs...)

function mean_lognormals(dv::AbstractDistributionVector{<:LogNormal}, x...; 
    kwargs...) 
    ds = sum(dv, x...; kwargs...)
    n = count(x -> !ismissing(x), dv)
    # multiplicative uncertainty stays the same
    LogNormal(ds.μ/n, ds.σ)
end



