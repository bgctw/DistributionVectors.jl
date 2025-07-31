using DistributionVectors
using Missings, Distributions, Test
import LinearAlgebra: I
using RecursiveArrayTools, FillArrays

@testset "vectuptotupvec" begin
    vectup = [(1,1.01, "string 1"), (2,2.02, "string 2")] 
    tupvec = @inferred vectuptotupvec(vectup)
    #@code_warntype vectuptotupvec(vectup)
    @test tupvec == ([1, 2], [1.01, 2.02], ["string 1", "string 2"])  
    # empty not allowed
    @test_throws Exception tupvec = vectuptotupvec([])
    # first missing
    vectupm = [missing, (1,1.01, "string 1"), (2,2.02, "string 2")] 
    vectuptotupvec(vectupm)
    tupvecm = @inferred vectuptotupvec(vectupm)
    @test ismissing(vectupm[1]) # did not change underlying vector
    #@code_warntype vectuptotupvec(vectupm)
    @test isequal(tupvecm, ([missing, 1, 2], [missing, 1.01, 2.02], [missing, "string 1", "string 2"]))
    # do not allow tuples of different length
    vectupm = [(1,1.01, "string 1"), (2,2.02, "string 2",:asymbol)] 
    @test_throws Exception tupvecm = vectuptotupvec(vectupm)
    # do not allow tuples of differnt types - note the Float64 in first entry
    vectupm = [(1.00,1.01, "string 1"), (2,2.02, "string 2",:asymbol)] 
    @test_throws Exception tupvecm = vectuptotupvec(vectupm)
end;

@testset "SimpleDistributionVector" begin
    n = [1,2,3]
    p = [.01, .02, .03]
    a = map(x -> Binomial(x...), zip(n,p))
    dv = dv0 = SimpleDistributionVector(Binomial{Float64}, allowmissing(a))
    nm = allowmissing(n); nm[1] = missing
    am = allowmissing(a); am[1] = missing
    dvm = SimpleDistributionVector(Binomial{Float64}, am)
    @testset "checking constructors" begin
        # not using allowmissing
        @test_throws ErrorException SimpleDistributionVector(Binomial, a) 
        # not allowing others than tuples
        @test_throws ErrorException SimpleDistributionVector(Binomial, allowmissing([1,2])) 
        # not allowing for Tuples of different length
        # already coverey by Vector{T}
    end;    
    @testset "iterator nonmissing" begin
        @test @inferred length(dv) == 3
        d = @inferred Missing dv[1]
        @test isa(d, Binomial{Float64})
        @test params(d) == (n[1], p[1])
        darr = [d for d in dv]
        dvec = @inferred collect(dv)
        @test length(dvec) == 3
    end;    
    @testset "iterator missing" begin
        @test @inferred length(dvm) == 3
        @test ismissing(@inferred Binomial dvm[1])
        d = @inferred Missing dvm[2]
        @test isa(d, Binomial{Float64})
        @test params(d) == (n[2], p[2])
        darr = [d for d in dvm]
        ismissing(darr[1])
        dvec = @inferred collect(dvm)
        @test length(dvec) == 3
    end; 
    @testset "getindex" begin
        @test @inferred ismissing(dvm,1)
        @test !@inferred ismissing(dvm,2)
        @test isequal(collect(@inferred dv[1:3]), collect(dv))
        @test isequal(collect(@inferred dvm[1:3]), collect(dvm))
        @test isequal(collect(@inferred dvm[[1,2,3]]), collect(dvm))
    end;
    @testset "copy and setindex" begin
        dvt = @inferred copy(dv)
        @test dvt == dv # same
        @test !(dvt === dv) # but not identical (different object)
        dvt[1:2] .= missing
        @test isequal(dvt[1:2], [missing, missing]) 
        @test !ismissing(dv[1]) # original not overidden
    end;
    @testset "constructor with several Distributions" begin
        d1 = Binomial(1, 0.25)
        d2 = Binomial(2, 0.15)
        dv = @inferred SimpleDistributionVector(d1, d2);
        @test @inferred Missing dv[1] == d1
        # empty not allowed
        @test_throws Exception SimpleDistributionVector()
        # different types not allowed
        @test_throws MethodError SimpleDistributionVector(d1, d2, Normal());
        # with missing 
        dv = @inferred SimpleDistributionVector(d1, d2, missing);
        @test ismissing(dv[3])
        # type no defined if provided missing
        @test_throws Exception dv = SimpleDistributionVector(missing);
    end;
    @testset "constructor with parameter vectors" begin
        dv = @inferred SimpleDistributionVector(Binomial{Float64}, n, p)
        # broadcast slightly slower and more allocations than generator on vectup returned from dv
        #@btime SimpleDistributionVector((ismissing(x) ? missing : Binomial(x...) for x in $dv)...)
        #@btime SimpleDistributionVector((x -> ismissing(x) ? missing : Binomial(x...)).($dv)...)
        # @btime begin
        #     a = collect((x -> ismissing(x) ? missing : Binomial(x...)).($dv))
        #     SimpleDistributionVector(Binomial{Float64}, a)
        # end
        @test params(first(dv)) == (n[1], p[1])
        @test params(dv, Val(1)) == n
        @test params(dv, Val(2)) == p
        # when one parameter has missing, the entire tuple must be set to missing
        dv = @inferred SimpleDistributionVector(Binomial{Float64}, nm, p)
        @test ismissing(dv[1])
        # need concrete type, here test missing {Float64}
        @test_throws ErrorException SimpleDistributionVector(Binomial, n, p)
    end;
    @testset "accessing parameters as array" begin
        @test @inferred(params(dv0,Val(1))) == n
        @test @inferred(params(dv0,Val(2))) == p
        # missing
        @test ismissing(params(dvm, Val(1))[1]) 
        @test params(dvm, Val(1))[2:3] == n[2:3]
        @test_throws BoundsError params(dv0,Val(3))
    end;
    @testset "Multivariate distribution with complex parameter types" begin
        #dmn1 = MvNormal(3, 1*I) # does not work because its of different specific type
        dmn1 = MvNormal([0,0,0], 1*I)
        dmn2 = MvNormal([1,1,1], 2*I)
        #params(dmn1), params(dmn2)
        dv = @inferred SimpleDistributionVector(dmn1, dmn2, missing, missing);
        @test @inferred Missing dv[1] == dmn1
        @test @inferred Missing dv[2] == dmn2
        @test ismissing(dv[3])
        @inferred Distributions.params(dv, Val(1))
        @test nonmissingtype(eltype(@inferred Missing Distributions.params(dv, Val(1)))) <: AbstractVector
        @test nonmissingtype(eltype(@inferred Missing Distributions.params(dv, Val(2)))) <: AbstractMatrix
        T = Union{VectorOfArray{Float64, 2, Vector{Union{FillArrays.Fill{Missing, 1, Tuple{Base.OneTo{Int64}}}, Vector{Float64}}}}, VectorOfArray{Float64, 2, Vector{Vector{Float64}}}}
        r1 = @inferred T rand(dv)
        @test size(r1) == (3, 4)
        T = Union{VectorOfArray{Float64, 3, Vector{Union{FillArrays.Fill{Missing, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, Matrix{Float64}}}}, VectorOfArray{Float64, 3, Vector{Matrix{Float64}}}}
        rvec = rand(dv,1) # in each dist(4): vector of 1 draw - mv(3)
        rvec = @inferred T rand(dv,1) # in each dist(4): vector of 1 draw - mv(3)
        @test size(rvec) == (3, 1, 4)
    end;
    @testset "Tuple of all parameter vectors" begin
        tupvec = @inferred params(dv0)
        @test tupvec == (allowmissing(n), allowmissing(p))
        tupvec = @inferred params(dvm)
        @test isequal(tupvec[1], (allowmissing(nm)))
        @test isequal(tupvec[2][2:end], (allowmissing(p)[2:end]))
    end;
    @testset "rand" begin
        nD = length(dv0)
        x = @inferred rand(dv0)
        @test size(x) == (nD,)
        Distributions.rand!(x, dv0)
        T = Union{RecursiveArrayTools.VectorOfArray{Int64, 2, Vector{Union{FillArrays.Fill{Missing, 1, Tuple{Base.OneTo{Int64}}}, Vector{Int64}}}}, RecursiveArrayTools.VectorOfArray{Int64, 2, Vector{Vector{Int64}}}}
        x = @inferred T rand(dv0,2)
        #x = @inferred rand(dv0,2)   # depending on whether dv0 contained missing
        @test size(x) == (2, nD)
        # not implemented: rand!(x, dvm, 2)
        #@code_warntype rand(dv0)
        # with missings
        nD = length(dv0)
        T = Union{VectorOfArray{Missing, 2, Vector{Union{Fill{Missing, 1, Tuple{Base.OneTo{Int64}}}, Vector{Int64}}}}, VectorOfArray{Int64, 2, Vector{Vector{Int64}}}}
        x = rand(dvm,5)
        x = @inferred T rand(dvm,5)
        @test size(x) == (5,nD)
        @test size(x[1]) == (5,)
        @test all(ismissing.(x[:,1]))
    end;
end; # testset "SimpleDistributionVector"

@testset "ParamDistributionVector" begin
    n = [1,2,3]
    p = [.01, .02, .03]
    a0 = (n,p)
    a = ((allowmissing(n), allowmissing(p)))
    typeof(a).parameters[1]
    @test all(map((x -> x <: AbstractVector), typeof(a).parameters))
    @test all(map((x -> Missing <: eltype(x)), typeof(a).parameters))
    @test !all(map((x -> Missing <: eltype(x)), typeof(a0).parameters))
    first(skipmissing(a))
    lena = map(length, a)
    all(map(x -> x == first(lena), lena))
    dv = dv0 = ParamDistributionVector(Binomial{Float64}, a)
    nm = allowmissing(copy(n)); nm[1] = missing
    am = (nm, allowmissing(p))
    dvm = ParamDistributionVector(Binomial{Float64}, am)
    @testset "checking constructors" begin
        # not using allowmissing
        @test_throws ErrorException ParamDistributionVector(Binomial, a0) 
        # not allowing others than vectors
        @test_throws ErrorException ParamDistributionVector(Binomial{Float64}, (nm, "bla"))
        # not allowing for vectors of different length
        @test_throws ErrorException ParamDistributionVector(Binomial{Float64}, (nm, nm[1:2]))
    end;    
    @testset "iterator nonmissing" begin
        @test @inferred length(dv) == 3
        d = @inferred Missing dv[1]
        @test isa(d, Binomial{Float64})
        @test params(d) == (n[1], p[1])
        darr = [d for d in dv]
    end;    
    @testset "iterator missing" begin
        @test @inferred length(dvm) == 3
        @test ismissing(@inferred Binomial dvm[1])
        d = @inferred Missing dvm[2]
        @test isa(d, Binomial{Float64})
        @test params(d) == (n[2], p[2])
        darr = [d for d in dvm]
        ismissing(darr[1])
    end; 
    @testset "getindex" begin
        @test @inferred ismissing(dvm,1)
        @test !@inferred ismissing(dvm,2)
        @test isequal(collect(@inferred dv[1:3]), collect(dv))
        @test isequal(collect(@inferred dvm[1:3]), collect(dvm))
        @test isequal(collect(@inferred dvm[[1,2,3]]), collect(dvm))
    end;
    @testset "copy and setindex" begin
        dvt = @inferred copy(dv)
        @test dvt == dv # same
        @test !(dvt === dv) # but not identical (different object)
        dvt[2] = Binomial(4, 0.4)
        @test dvt[2] == Binomial(4, 0.4)
        dvt[1:2] .= missing
        @test isequal(dvt[1:2], [missing, missing]) 
        @test !ismissing(dv[1]) # original not overidden
    end;
    @testset "constructor with several Distributions" begin
        d1 = Binomial(1, 0.25)
        d2 = Binomial(2, 0.15)
        td = (d1, d2)
        dv = @inferred ParamDistributionVector(d1, d2);
        @test @inferred Missing dv[1] == d1
        # empty not allowed
        @test_throws Exception ParamDistributionVector()
        # different types not allowed
        @test_throws MethodError ParamDistributionVector(d1, d2, Normal());
        # with missing 
        dv = @inferred ParamDistributionVector(d1, d2, missing);
        @test ismissing(dv[3])
        # type no defined if provided missing
        @test_throws Exception dv = ParamDistributionVector(missing);
    end;
    @testset "constructor with parameter vectors" begin
        try
            # only works from Julia 1.6 with n,p not allowing for missings
            dv = @inferred ParamDistributionVector(Binomial{Float64}, n, p)
            d1 = @inferred Missing dv[1]
            @test params(d1) == (n[1], p[1])
        catch e 
            @test isa(e, ErrorException)
        end
        # when one parameter has missing, the entire tuple must be set to missing
        dv = @inferred ParamDistributionVector(Binomial{Float64}, nm, allowmissing(p))
        @test ismissing(dv[1])
        @test ismissing(dv,1)
        # need concrete type, here test missing {Float64}
        @test_throws ErrorException ParamDistributionVector(Binomial, n, p)
    end;
    @testset "accessing parameters as array" begin
        @test @inferred(params(dv0,Val(1))) == n
        @test params(dv0,Val(2)) == p
        # missing
        @test ismissing(params(dvm, Val(1))[1]) 
        @test params(dvm, Val(1))[2:3] == n[2:3]
        @test_throws BoundsError params(dv0,Val(3))
    end;
    @testset "Multivariate distribution with complex parameter types" begin
        #dmn1 = MvNormal(3, 1*I) # does not work because its of different specific type
        dmn1 = MvNormal([0,0,0], 1*I)
        dmn2 = MvNormal([1,1,1], 2*I)
        #params(dmn1), params(dmn2)
        dv = @inferred ParamDistributionVector(dmn1, dmn2, missing)
        typeof(dv).parameters[1]
        D = typeof(dv).parameters[1]
        #tupleofvectype(D).parameters[1]
        #tupleofvectype(D).parameters[2]
        @test nonmissingtype(eltype(@inferred params(dv, Val(1)))) <: AbstractVector
        #@code_warntype(params(dv, Val(1)))
        @test nonmissingtype(eltype(@inferred params(dv, Val(2)))) <: AbstractMatrix
    end;
    @testset "Tuple of all parameter vectors" begin
        tupvec = @inferred params(dv0)
        @test tupvec == (allowmissing(n), allowmissing(p))
        tupvec = @inferred params(dvm)
        @test isequal(tupvec, (allowmissing(nm), allowmissing(p)))
    end;
    @testset "passing vector by reference" begin
        dv = ParamDistributionVector(Binomial(1, 0.7));
        n = params(dv, Val(1)); # reference
        n[1] = 2
        @test params(dv[1])[1] == 2 # note has changed
        # Binomial is immutable cannot changed
        # dv[1].μ = 3
    end;
    @testset "rand" begin
        nD = length(dv0)
        x = @inferred rand(dv0)
        @test size(x) == (nD,)
        T = Union{
            VectorOfArray{Missing, 2, Vector{Union{Fill{Missing, 1, Tuple{Base.OneTo{Int64}}}, Vector{Int64}}}}, 
            VectorOfArray{Int64, 2, Vector{Union{Fill{Missing, 1, Tuple{Base.OneTo{Int64}}}, Vector{Int64}}}}, 
            VectorOfArray{Int64, 2, Vector{Vector{Int64}}}}
        x = @inferred T rand(dv0,2)
        @test size(x) == (2, nD)
        #@code_warntype rand(dv0)
        # with missings
        nD = length(dv0)
        x = @inferred T rand(dvm,5)
        @test size(x) == (5,nD)
        @test size(x[1]) == (5,)
        @test all(ismissing.(x[:,1]))
    end;
end; #@testset "ParamDistributionVector"

