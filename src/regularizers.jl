# Predefined regularizers
# You may also implement your own regularizer by subtyping
# the abstract type Regularizer.
# Regularizers should implement `evaluate` and `prox`.

import Base: *

export Regularizer, ProductRegularizer, ColumnRegularizer, DependenceMeasure, # abstract types
       # concrete regularizers
       QuadReg, QuadConstraint,
       OneReg, ZeroReg, NonNegConstraint, NonNegOneReg, NonNegQuadReg,
       OneSparseConstraint, UnitOneSparseConstraint, SimplexConstraint,
       KSparseConstraint,
       lastentry1, lastentry_unpenalized,
       fixed_latent_features, FixedLatentFeaturesConstraint,
       fixed_last_latent_features, FixedLastLatentFeaturesConstraint,
       OrdinalReg, MNLOrdinalReg,
       RemQuadReg,
       ZeroColReg,
       OrthogonalReg, SoftOrthogonalReg, # linearly independent regularisers
       HSICReg, SeparationReg, SufficiencyReg, # statistically independent regularisers
       # methods on regularizers
       prox!, prox,
       # utilities
       scale, mul!, *

# numerical tolerance
TOL = 1e-12

# regularizers
# regularizers r should have the method `prox` defined such that
# prox(r)(u,alpha) = argmin_x( alpha r(x) + 1/2 \|x - u\|_2^2)
abstract type Regularizer end
abstract type MatrixRegularizer <: LowRankModels.Regularizer end
abstract type ColumnRegularizer <: Regularizer end

# default inplace prox operator (slower than if inplace prox is implemented)
prox!(r::Regularizer,u::AbstractArray,alpha::Number) = (v = prox(r,u,alpha); @simd for i=1:length(u) @inbounds u[i]=v[i] end; u)

# default scaling
scale(r::Regularizer) = r.scale
mul!(r::Regularizer, newscale::Number) = (r.scale = newscale; r)
mul!(rs::Array{Regularizer}, newscale::Number) = (for r in rs mul!(r, newscale) end; rs)
*(newscale::Number, r::Regularizer) = (newr = typeof(r)(); mul!(newr, scale(r)*newscale); newr)

## utilities

function allnonneg(a::AbstractArray)
  for ai in a
    ai < 0 && return false
  end
  return true
end

choose_bins(i::Int64) = min(i, Int(1 + round(3.322 * log(i))))

function bin(y::AbstractArray)
    if length(size(y)) > 1 # categorical y (one-hot encoded)
        m, n = size(y)
        bins = n
        groups = []
        for i=1:bins push!(groups, []) end
        for i=1:m
            for j=1:n
                if y[i, j] == 1 push!(groups[j], i) end
            end
        end
    else                   # real/ordinal y
        m = length(y)
        bins = choose_bins(m)
        groups = []
        for i=1:bins push!(groups, []) end
        bins = Float64(bins)
        dists = [i / bins for i=1:bins]
        quantiles = [quantile(y, d) for d in dists]
        for (i, e) in enumerate(y)
            # assigned = false
            for (j, q) in enumerate(quantiles)
                if e <= q push!(groups[j], i); break end
            end
        end
    end
    filter(g -> length(g) > 0, groups)
end

## Quadratic regularization
mutable struct QuadReg<:Regularizer
    scale::Float64
end
QuadReg() = QuadReg(1)
prox(r::QuadReg,u::AbstractArray,alpha::Number) = 1/(1+2*alpha*r.scale)*u
prox!(r::QuadReg,u::Array{Float64},alpha::Number) = rmul!(u, 1/(1+2*alpha*r.scale))
evaluate(r::QuadReg,a::AbstractArray) = r.scale*sum(abs2, a)

## constrained quadratic regularization
## the function r such that
## r(x) = inf    if norm(x) > max_2norm
##        0      otherwise
## can be used to implement maxnorm regularization:
##   constraining the maxnorm of XY to be <= mu is achieved
##   by setting glrm.rx = QuadConstraint(sqrt(mu))
##   and the same for every element of glrm.ry
mutable struct QuadConstraint<:Regularizer
    max_2norm::Float64
end
QuadConstraint() = QuadConstraint(1)
prox(r::QuadConstraint,u::AbstractArray,alpha::Number) = (r.max_2norm)/norm(u)*u
prox!(r::QuadConstraint,u::Array{Float64},alpha::Number) = mul!(u, (r.max_2norm)/norm(u))
evaluate(r::QuadConstraint,u::AbstractArray) = norm(u) > r.max_2norm + TOL ? Inf : 0
scale(r::QuadConstraint) = 1
mul!(r::QuadConstraint, newscale::Number) = 1

## one norm regularization
mutable struct OneReg<:Regularizer
    scale::Float64
end
OneReg() = OneReg(1)
function softthreshold(x::Number; alpha=1)
 return max(x-alpha,0) + min(x+alpha,0)
end
prox(r::OneReg,u::AbstractArray,alpha::Number) = (st(x) = softthreshold(x; alpha=r.scale*alpha); st.(u))
prox!(r::OneReg,u::AbstractArray,alpha::Number) = (st(x) = softthreshold(x; alpha=r.scale*alpha); map!(st, u, u))
evaluate(r::OneReg,a::AbstractArray) = r.scale*sum(abs,a)

## no regularization
mutable struct ZeroReg<:Regularizer
end
prox(r::ZeroReg,u::AbstractArray,alpha::Number) = u
prox!(r::ZeroReg,u::Array{Float64},alpha::Number) = u
evaluate(r::ZeroReg,a::AbstractArray) = 0
scale(r::ZeroReg) = 0
mul!(r::ZeroReg, newscale::Number) = 0

## indicator of the nonnegative orthant
## (enforces nonnegativity, eg for nonnegative matrix factorization)
mutable struct NonNegConstraint<:Regularizer
end
prox(r::NonNegConstraint,u::AbstractArray,alpha::Number=1) = broadcast(max,u,0)
prox!(r::NonNegConstraint,u::Array{Float64},alpha::Number=1) = (@simd for i=1:length(u) @inbounds u[i] = max(u[i], 0) end; u)
function evaluate(r::NonNegConstraint,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return 0
end
scale(r::NonNegConstraint) = 1
mul!(r::NonNegConstraint, newscale::Number) = 1

## one norm regularization restricted to nonnegative orthant
## (enforces nonnegativity, in addition to one norm regularization)
mutable struct NonNegOneReg<:Regularizer
    scale::Float64
end
NonNegOneReg() = NonNegOneReg(1)
prox(r::NonNegOneReg,u::AbstractArray,alpha::Number) = max.(u.-alpha,0)

function prox!(r::NonNegOneReg,u::AbstractArray,alpha::Number)
  nonnegsoftthreshold = (x::Number -> max(x-alpha,0))
  map!(nonnegsoftthreshold, u, u)
end

function evaluate(r::NonNegOneReg,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return r.scale*sum(a)
end
scale(r::NonNegOneReg) = 1
mul!(r::NonNegOneReg, newscale::Number) = 1

## Quadratic regularization restricted to nonnegative domain
## (Enforces nonnegativity alongside quadratic regularization)
mutable struct NonNegQuadReg
    scale::Float64
end
NonNegQuadReg() = NonNegQuadReg(1)
prox(r::NonNegQuadReg,u::AbstractArray,alpha::Number) = max.(1/(1+2*alpha*r.scale)*u, 0)
prox!(r::NonNegQuadReg,u::AbstractArray,alpha::Number) = begin
  mul!(u, 1/(1+2*alpha*r.scale))
  maxval = maximum(u)
  clamp!(u, 0, maxval)
end
function evaluate(r::NonNegQuadReg,a::AbstractArray)
    for ai in a
        if ai<0
            return Inf
        end
    end
    return r.scale*sumabs2(a)
end

## indicator of the last entry being equal to 1
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry_unpenalized)
mutable struct lastentry1<:Regularizer
    r::Regularizer
end
lastentry1() = lastentry1(ZeroReg())
prox(r::lastentry1,u::AbstractArray{Float64,1},alpha::Number=1) = [prox(r.r,view(u,1:length(u)-1),alpha); 1]
prox!(r::lastentry1,u::AbstractArray{Float64,1},alpha::Number=1) = (prox!(r.r,view(u,1:length(u)-1),alpha); u[end]=1; u)
prox(r::lastentry1,u::AbstractArray{Float64,2},alpha::Number=1) = [prox(r.r,view(u,1:size(u,1)-1,:),alpha); ones(1, size(u,2))]
prox!(r::lastentry1,u::AbstractArray{Float64,2},alpha::Number=1) = (prox!(r.r,view(u,1:size(u,1)-1,:),alpha); u[end,:]=1; u)
evaluate(r::lastentry1,a::AbstractArray{Float64,1}) = (a[end]==1 ? evaluate(r.r,a[1:end-1]) : Inf)
evaluate(r::lastentry1,a::AbstractArray{Float64,2}) = (all(a[end,:].==1) ? evaluate(r.r,a[1:end-1,:]) : Inf)
scale(r::lastentry1) = scale(r.r)
mul!(r::lastentry1, newscale::Number) = mul!(r.r, newscale)

## makes the last entry unpenalized
## (allows an unpenalized offset term into the glrm when used in conjunction with lastentry1)
mutable struct lastentry_unpenalized<:Regularizer
    r::Regularizer
end
lastentry_unpenalized() = lastentry_unpenalized(ZeroReg())
prox(r::lastentry_unpenalized,u::AbstractArray{Float64,1},alpha::Number=1) = [prox(r.r,u[1:end-1],alpha); u[end]]
prox!(r::lastentry_unpenalized,u::AbstractArray{Float64,1},alpha::Number=1) = (prox!(r.r,view(u,1:size(u,1)-1),alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray{Float64,1}) = evaluate(r.r,a[1:end-1])
prox(r::lastentry_unpenalized,u::AbstractArray{Float64,2},alpha::Number=1) = [prox(r.r,u[1:end-1,:],alpha); u[end,:]]
prox!(r::lastentry_unpenalized,u::AbstractArray{Float64,2},alpha::Number=1) = (prox!(r.r,view(u,1:size(u,1)-1,:),alpha); u)
evaluate(r::lastentry_unpenalized,a::AbstractArray{Float64,2}) = evaluate(r.r,a[1:end-1,:])
scale(r::lastentry_unpenalized) = scale(r.r)
mul!(r::lastentry_unpenalized, newscale::Number) = mul!(r.r, newscale)

## fixes the values of the first n elements of the column to be y
## optionally regularizes the last k-n elements with regularizer r
mutable struct fixed_latent_features<:Regularizer
    r::Regularizer
    y::Array{Float64,1} # the values of the fixed latent features
    n::Int # length of y
end
fixed_latent_features(r::Regularizer, y::Array{Float64,1}) = fixed_latent_features(r,y,length(y))
# standalone use without another regularizer
FixedLatentFeaturesConstraint(y::Array{Float64, 1}) = fixed_latent_features(ZeroReg(),y,length(y))

prox(r::fixed_latent_features,u::AbstractArray,alpha::Number) = [r.y; prox(r.r,u[(r.n+1):end],alpha)]
function prox!(r::fixed_latent_features,u::Array{Float64},alpha::Number)
  	prox!(r.r,u[(r.n+1):end],alpha)
  	u[1:r.n]=y
  	u
end
evaluate(r::fixed_latent_features, a::AbstractArray) = a[1:r.n]==r.y ? evaluate(r.r, a[(r.n+1):end]) : Inf
scale(r::fixed_latent_features) = scale(r.r)
mul!(r::fixed_latent_features, newscale::Number) = mul!(r.r, newscale)

## fixes the values of the last n elements of the column to be y
## optionally regularizes the first k-n elements with regularizer r
mutable struct fixed_last_latent_features<:Regularizer
    r::Regularizer
    y::Array{Float64,1} # the values of the fixed latent features
    n::Int # length of y
end
fixed_last_latent_features(r::Regularizer, y::Array{Float64,1}) = fixed_last_latent_features(r,y,length(y))
# standalone use without another regularizer
FixedLastLatentFeaturesConstraint(y::Array{Float64, 1}) = fixed_last_latent_features(ZeroReg(),y,length(y))

prox(r::fixed_last_latent_features,u::AbstractArray,alpha::Number) = [prox(r.r,u[(r.n+1):end],alpha); r.y]
function prox!(r::fixed_last_latent_features,u::Array{Float64},alpha::Number)
    u[length(u)-r.n+1:end]=y
    prox!(r.r,u[1:length(a)-r.n],alpha)
    u
end
evaluate(r::fixed_last_latent_features, a::AbstractArray) = a[length(a)-r.n+1:end]==r.y ? evaluate(r.r, a[1:length(a)-r.n]) : Inf
scale(r::fixed_last_latent_features) = scale(r.r)
mul!(r::fixed_last_latent_features, newscale::Number) = mul!(r.r, newscale)

## indicator of 1-sparse vectors
## (enforces that exact 1 entry is nonzero, eg for orthogonal NNMF)
mutable struct OneSparseConstraint<:Regularizer
end
prox(r::OneSparseConstraint, u::AbstractArray, alpha::Number=0) = (idx = argmax(u); v=zeros(size(u)); v[idx]=u[idx]; v)
prox!(r::OneSparseConstraint, u::Array, alpha::Number=0) = (idx = argmax(u); ui = u[idx]; mul!(u,0); u[idx]=ui; u)
function evaluate(r::OneSparseConstraint, a::AbstractArray)
    oneflag = false
    for ai in a
        if oneflag
            if ai!=0
                return Inf
            end
        else
            if ai!=0
                oneflag=true
            end
        end
    end
    return 0
end
scale(r::OneSparseConstraint) = 1
mul!(r::OneSparseConstraint, newscale::Number) = 1

## Indicator of k-sparse vectors
mutable struct KSparseConstraint<:Regularizer
  k::Int
end
function evaluate(r::KSparseConstraint, a::AbstractArray)
  k = r.k
  nonzcount = 0
  for ai in a
    if nonzcount == k
      if ai != 0
        return Inf
      end
    else
      if ai != 0
        nonzcount += 1
      end
    end
  end
  return 0
end
function prox(r::KSparseConstraint, u::AbstractArray, alpha::Number)
  k = r.k
  ids = partialsortperm(u, 1:k, by=abs, rev=true)
  uk = zero(u)
  uk[ids] = u[ids]
  uk
end
function prox!(r::KSparseConstraint, u::Array, alpha::Number)
  k = r.k
  ids = partialsortperm(u, 1:k, by=abs, rev=true)
  vals = u[ids]
  mul!(u,0)
  u[ids] = vals
  u
end

## indicator of 1-sparse unit vectors
## (enforces that exact 1 entry is 1 and all others are zero, eg for kmeans)
mutable struct UnitOneSparseConstraint<:Regularizer
end
prox(r::UnitOneSparseConstraint, u::AbstractArray, alpha::Number=0) = (idx = argmax(u); v=zeros(size(u)); v[idx]=1; v)
prox!(r::UnitOneSparseConstraint, u::Array, alpha::Number=0) = (idx = argmax(u); mul!(u,0); u[idx]=1; u)

function evaluate(r::UnitOneSparseConstraint, a::AbstractArray)
    oneflag = false
    for ai in a
        if ai==0
            continue
        elseif ai==1
            if oneflag
                return Inf
            else
                oneflag=true
            end
        else
            return Inf
        end
    end
    return 0
end
scale(r::UnitOneSparseConstraint) = 1
mul!(r::UnitOneSparseConstraint, newscale::Number) = 1

## indicator of vectors in the simplex: nonnegative vectors with unit l1 norm
## (eg for QuadLoss mixtures, ie soft kmeans)
## prox for the simplex is derived by Chen and Ye in [this paper](http://arxiv.org/pdf/1101.6081v2.pdf)
mutable struct SimplexConstraint<:Regularizer
end
function prox(r::SimplexConstraint, u::AbstractArray, alpha::Number=0)
    n = length(u)
    y = sort(u, rev=true)
    ysum = cumsum(y)
    t = (ysum[end]-1)/n
    for i=1:(n-1)
        if (ysum[i]-1)/i >= y[i+1]
            t = (ysum[i]-1)/i
            break
        end
    end
    max.(u .- t, 0)
end
function evaluate(r::SimplexConstraint,a::AbstractArray)
    # check it's a unit vector
    abs(sum(a)-1)>TOL && return Inf
    # check every entry is nonnegative
    for i=1:length(a)
        a[i] < 0 && return Inf
    end
    return 0
end
scale(r::SimplexConstraint) = 1
mul!(r::SimplexConstraint, newscale::Number) = 1

## ordinal regularizer
## a block regularizer which
    # 1) forces the first k-1 entries of each column to be the same
    # 2) forces the last entry of each column to be increasing
    # 3) applies an internal regularizer to the first k-1 entries of each column
## should always be used in conjunction with lastentry1 regularization on x
mutable struct OrdinalReg<:Regularizer
    r::Regularizer
end
OrdinalReg() = OrdinalReg(ZeroReg())
prox(r::OrdinalReg,u::AbstractArray,alpha::Number) = (uc = copy(u); prox!(r,uc,alpha))
function prox!(r::OrdinalReg,u::AbstractArray,alpha::Number)
    um = mean(u[1:end-1, :], dims=2)
    prox!(r.r,um,alpha)
    for i=1:size(u,1)-1
        for j=1:size(u,2)
            u[i,j] = um[i]
        end
    end
    # this enforces rule 2) (increasing last row of u), but isn't exactly the prox function
    # for j=2:size(u,2)
    #     if u[end,j-1] > u[end,j]
    #         m = (u[end,j-1] + u[end,j])/2
    #         u[end,j-1:j] = m
    #     end
    # end
    u
end
evaluate(r::OrdinalReg,a::AbstractArray) = evaluate(r.r,a[1:end-1,1])
scale(r::OrdinalReg) = scale(r.r)
mul!(r::OrdinalReg, newscale::Number) = mul!(r.r, newscale)

# make sure we don't add two offsets cuz that's weird
lastentry_unpenalized(r::OrdinalReg) = r

mutable struct MNLOrdinalReg<:Regularizer
    r::Regularizer
end
MNLOrdinalReg() = MNLOrdinalReg(ZeroReg())
prox(r::MNLOrdinalReg,u::AbstractArray,alpha::Number) = (uc = copy(u); prox!(r,uc,alpha))
function prox!(r::MNLOrdinalReg,u::AbstractArray,alpha::Number; TOL=1e-3)
    um = mean(u[1:end-1, :], dims=2)
    prox!(r.r,um,alpha)
    for i=1:size(u,1)-1
        for j=1:size(u,2)
            u[i,j] = um[i]
        end
    end
    # this enforces rule 2) (decreasing last row of u, all less than 0), but isn't exactly the prox function
    u[end,1] = min(-TOL, u[end,1])
    for j=2:size(u,2)
      u[end,j] = min(u[end,j], u[end,j-1]-TOL)
    end
    u
end
evaluate(r::MNLOrdinalReg,a::AbstractArray) = evaluate(r.r,a[1:end-1,1])
scale(r::MNLOrdinalReg) = scale(r.r)
mul!(r::MNLOrdinalReg, newscale::Number) = mul!(r.r, newscale)
# make sure we don't add two offsets cuz that's weird
lastentry_unpenalized(r::MNLOrdinalReg) = r

## Quadratic regularization with non-zero mean
mutable struct RemQuadReg<:Regularizer
        scale::Float64
        m::Array{Float64, 1}
end
RemQuadReg(m::Array{Float64, 1}) = RemQuadReg(1, m)
prox(r::RemQuadReg, u::AbstractArray, alpha::Number) =
     (u + 2 * alpha * r.scale * r.m) / (1 + 2 * alpha * r.scale)
prox!(r::RemQuadReg, u::Array{Float64}, alpha::Number) = begin
        broadcast!(.+, u, u, 2 * alpha * r.scale * r.m)
        mul!(u, 1 / (1 + 2 * alpha * r.scale))
end
evaluate(r::RemQuadReg, a::AbstractArray) = r.scale * sum(abs2, a - r.m)

mutable struct ZeroColReg<:ColumnRegularizer
end
prox(r::ZeroColReg,u::AbstractArray,alpha::Number) = u
prox!(r::ZeroColReg,u::Array{Float64},alpha::Number) = u
evaluate(r::ZeroColReg,a::AbstractArray) = 0
scale(r::ZeroColReg) = 0
mul!(r::ZeroColReg, newscale::Number) = 0

"""
A regulariser for enforcing that every row of X (assuming the rows of X
are m-dimensional vectors) is orthogonal with the protected characteristic
of the data.

## Parameters
- `scale`: Not used, but needed to be a regulariser.
- `s`:     The protected characteristic which each column of X needs to be
           orthogonal to (after standardising the mean).
"""
mutable struct OrthogonalReg<:ColumnRegularizer
    scale::Float64
    s::AbstractArray
end
OrthogonalReg(s::AbstractArray) = OrthogonalReg(1, normalise(s))

"""
Checks if two vectors are orthogonal to one another.

## Parameters
- `s`: The protected characteristic of the data. This parameter needs to be
       orthogonal to every row in `X`.
- `X`: The data. Each row in this parameter must be orthogonal to `s`.
"""
is_orthogonal(s::AbstractArray, X::AbstractArray) = begin
    # Note there is a small numerical tolerance; feel free to change this if
    # you'd like
    if length(size(X)) == 1 
        dot_product = abs(dot(X, s))
        return dot_product <= TOL 
    end
    for i=1:size(X, 1)
        if abs(dot(X[i, :], s)) > TOL return false end
    end
    return true
end

"""
Calculate the vector projection of each column of `X` onto `s`.

## Parameters
- `s`: The vector onto which each element will be projected.
- `X`: The vector(s) that are being projected onto `s`.
"""
project(s::AbstractArray, X::AbstractArray) = begin
    if size(X, 2) == 1 return dot(X[:, 1], s) / dot(s, s) * s end # 1 dimension   
    return [dot(X[:, i], s) / dot(s, s) * s for i=1:n]            # 2+ dimensions
end

"""
Normalises data. This is required to ensure uncorrelatedness, since the two
vectors need to be mean-centred at 0 and be orthogonal.

## Parameters
- `u`: The array to be normalised.
"""
normalise(u::AbstractArray) = begin
    # 1 dimension
    if length(size(u)) == 1
        μ = mean(u)
        σ = stdm(u, μ)
        return (u .- μ) / σ
    # 2+ dimensions 
    else
        μ = mean(u, dims=2)
        σ = stdm(u, μ, dims=2)
        mean_centred = broadcast(-, μ, u)
        return broadcast(/, σ, mean_centred)
    end
end

"""
Determines how much regularisation penalty to add.

## Parameters
- `r`: The orthogonal regulariser. The type of `r` specifies the procedure for
       `evaluate()`.
- `u`: The data to be evaluated.

## Returns

0 if every column of `u` is orthogonal to `r` when mean-centred (or if `u` is
orthogonal to `r` is `u` is a vector), and ∞ if not.
"""
evaluate(r::OrthogonalReg, u::AbstractArray) = is_orthogonal(r.s, normalise(u)) ? 0 : Inf

"""
Computes a proximal gradient step. This is achieved by finding the vector
projection of the sample data onto the protected characteristic.

## Parameters
- `r`:     The orthogonal regulariser. The type of this regulariser specifies
           the type of proximal gradient step that should be carried out.
- `u`:     The vector(s) on which to perform the proximal gradient step.
- `alpha`: The step size of the proximal gradient step, this isn't used in this
           specific proximal gradient step.

## Returns

The vector projection of `u` onto the protected characteristic stored in `r`.
"""
prox(r::OrthogonalReg, u::AbstractArray, alpha::Number) = begin
    # Save the mean for later, after normalisation
    mean_u = length(size(u)) == 1 ? mean(u) : mean(u, dims=2)
    normalised_u = normalise(u)
    # Get the orthogonal component of the vector projection (the vector
    # rejection?)
    projected_u = project(r.s, normalised_u)
    orthog_u = normalised_u - projected_u
    if length(size(orthog_u)) == 1 return orthog_u .+ mean_u             # 1 dimension
    else                           return broadcast(+, orthog_u, mean_u) # 2+ dimensions
    end
end
prox!(r::OrthogonalReg, u::AbstractArray, alpha::Number) = begin
    u = prox(r, u, alpha)
    u
end

mutable struct SoftOrthogonalReg<:ColumnRegularizer
    scale::Float64
    s::AbstractArray
end
SoftOrthogonalReg(s::AbstractArray) = SoftOrthogonalReg(1, normalise(s))
evaluate(r::SoftOrthogonalReg, u::AbstractArray) = r.scale * sum(dot(normalise(u), r.s).^2)
prox(r::SoftOrthogonalReg, u::AbstractArray, alpha::Number) =
    u - r.scale * alpha * (2 * sum(dot(normalise(u), r.s)) * r.s)
prox!(r::SoftOrthogonalReg, u::AbstractArray, alpha::Number) = begin
    u = prox(r, u, alpha)
    u
end

mutable struct HSICReg<:ColumnRegularizer
    scale::Float64
    s::AbstractArray
    α::Float64
    independence::IndependenceCriterion
end
HSICReg(s::AbstractArray, X::AbstractArray, ic::DataType) = begin
    new_s = CuArray(s)
    new_X = CuArray(X)
    HSICReg(1, new_s, 0.5, get_independence_criterion(new_s, new_X, ic))
end
HSICReg(scale::Float64, s::AbstractArray, X::AbstractArray, ic::DataType) = begin
    new_s = CuArray(s)
    new_X = CuArray(X)
    HSICReg(scale, new_s, 0.5, get_independence_criterion(new_s, new_X, ic))
end
evaluate(r::HSICReg, u::AbstractArray) = begin
    # if length(size(u)) == 1
    #     n = length(u)
    #     hsic = hsic_gam!(r.hsic, CuArray(u))
    # else
    #     hsic = hsic_gam!(r.hsic, CuArray(u))
    # end
    # hsic = hsic_rff(r.hsic, CuArray(u); n_samples=100)
    hsic = evaluate(r.independence, Float32.(CuArray(u[:, :])))
    r.scale * hsic
end
# evaluate(r::HSICReg, u::AbstractArray, e::Int) = begin
#     if length(size(u)) == 1
#         n = length(u)
#         hsic = hsic_gam!(r.hsic, CuArray(reshape(u, (n, 1))), e)
#     else
#         hsic = hsic_gam!(r.hsic, CuArray(u), e)
#     end
#     r.scale * hsic
# end
prox(r::HSICReg, u::AbstractArray, alpha::Number) = begin 
    # n = size(u, 1)
    # if length(size(u)) == 1
    #     grad = hsic_grad!(r.hsic, CuArray(reshape(u, (n, 1))))
    # else
    #     grad = hsic_grad!(r.hsic, CuArray(u))
    # end
    gradient = grad(r.independence, Float32.(CuArray(u[:, :])))
    u .- (r.scale * alpha) * gradient
end
prox!(r::HSICReg, u::AbstractArray, alpha::Number) = begin
    u = prox(r, u, alpha)
    u
end

mutable struct SeparationReg<:ColumnRegularizer
    scale::Float64
    s::AbstractArray
    y::AbstractArray
    groups::AbstractArray
    α::Float64
    r::DataType
end
SeparationReg(s::AbstractArray, y::AbstractArray, r::DataType) =
    SeparationReg(1, s, y, bin(y), 0.5, r)
SeparationReg(scale::Float64, s::AbstractArray, y::AbstractArray, r::DataType) = 
    SeparationReg(scale, s, y, bin(y), 0.5, r)
SeparationReg(s::AbstractArray, y::AbstractArray, groups::AbstractArray, r::DataType) =
    SeparationReg(1, s, y, groups, 0.5, r)
evaluate(r::SeparationReg, u::AbstractArray) = begin
    total_loss = 0.0
    for g in r.groups
        reg = r.r(r.scale, normalise(r.s[g]))
        u_g = normalise(u[g])
        total_loss += evaluate(reg, u_g)
    end
    total_loss
end
prox(r::SeparationReg, u::AbstractArray, alpha::Float64) = begin
    grad = zeros(size(u))
    for g in r.groups
        u_g = u[g]
        s_g = normalise(r.s[g])
        reg = r.r(r.scale, s_g)
        subgrad = u_g .- prox(reg, u_g, alpha)
        i = 1
        for j in g
            grad[j] += subgrad[i]
            i += 1
        end
    end
    u .- grad
end
prox!(r::SeparationReg, u::AbstractArray, alpha::Float64) = begin
    u = prox(r, u, alpha)
    u
end

mutable struct SufficiencyReg<:ColumnRegularizer
    scale::Float64
    s::AbstractArray
    y::AbstractArray
    α::Float64
    r::DataType
end
SufficiencyReg(s::AbstractArray, y::AbstractArray, r::DataType) = SufficiencyReg(1.0, s, y, 0.5, r)
SufficiencyReg(scale::Float64, s::AbstractArray, y::AbstractArray, r::DataType) = SufficiencyReg(scale, s, y, 0.5, r)
evaluate(r::SufficiencyReg, u::AbstractArray) = begin
    total_loss = 0.0
    if length(size(u)) == 1
        groups = bin(u)
        filter!(g -> length(g) > 1, groups)
        for g in groups
            s_g = r.s[g]
            reg = r.r(r.scale, normalise(s_g))
            y_g = r.y[g]
            total_loss += evaluate(reg, y_g)
        end
    else
        for j=1:size(u, 2)
            groups = bin(u[:, j])
            filter!(g -> length(g) > 1, groups)
            for g in groups
                s_g = r.s[g]
                reg = r.r(r.scale, normalise(s_g))
                y_g = r.y[g]
                total_loss += evaluate(reg, y_g)
            end
        end
    end
    r.scale * total_loss
end
prox(r::SufficiencyReg, u::AbstractArray, alpha::Float64) = begin
    u_prime = copy(u)
    if length(size(u)) == 1
        bins = choose_bins(length(u))
        d = (maximum(u) - minimum(u)) / bins
        for i=1:length(u)
            min_independence = evaluate(r, u)
            for d_prime in [d, -d]
                u[i] = u[i] - d_prime
                independence = evaluate(r, u)
                if independence < min_independence
                    min_independence = independence
                    u_prime[i] = u[i] + (1 - alpha) * r.scale * d_prime
                end
                u[i] = u[i] + d_prime
            end
        end
    else
        n_rows, n_cols = size(u)
        bins = choose_bins(n_rows)
        for j=1:n_cols
            u_j = u[:, j]
            d = (max(u_j) - min(u_j)) / bins
            for i=1:n_rows
                min_independence = evaluate(r, u_j)
                for d_prime in [d, -d]
                    u_j[i] = u_j[i] - d_prime
                    independence = evaluate(r, u_j)
                    if independence < min_independence
                        min_independence = independence
                        u_prime[i, j] = u_j[i] + (1 - alpha) * r.scale * d_prime
                    end
                    u_j[i] = u_j[i] + d_prime
                end
            end
        end
    end
    u_prime
end
prox!(r::SufficiencyReg, u::AbstractArray, alpha::Float64) = begin
    u = prox(r, u, alpha)
    u
end

## simpler method for numbers, not arrays
evaluate(r::Regularizer, u::Number) = evaluate(r, [u])
prox(r::Regularizer, u::Number, alpha::Number) = prox(r, [u], alpha)[1]
# if step size not specified, step size = 1
prox(r::Regularizer, u) = prox(r, u, 1)

Label::Type = Union{String, Int64}
TargetDict::Type = Dict{Label, Array{Int64, 1}}
"""
This is a regulariser that will attempt to combine the independence and
separation (and maybe even sufficiency) regularisers and make it possible to
separate the data based on a subset of the possible values of each target
feature.
"""
mutable struct GeneralFairnessRegulariser<:ColumnRegularizer
    scales::Array{Float64, 1}     # The scale for calculating statistical
                                  # dependence wrt each protected characteristic
    protected::Matrix{Float64}    # The protected characteristic(s): layout is
                                  # m × s (where s is the number of protected
                                  # characteristics)
    groups::Vector{Vector{Int64}} # The indices of each separate "group": these
                                  # are separated by the target feature(s)
    reg::ColumnRegularizer        # orthogonality, soft orthogonality or hsic
end
function GeneralFairnessRegulariser(data::DataFrame,
        protected::Matrix{Float64}, regtype::DataType,
        targets::TargetDict=TargetDict();
        scales::Array{Float64, 1}=ones(Float64, size(protected, 1)),
        normalised::Bool=false)
    # Normalise the data if needed (is this even necessary here?)
    if normalised  reg = regtype(1.0, protected)
    else           reg = regtype(1.0, normalise(protected))
    end

    # Set up groups
    groups::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    if isempty(targets)
        groups = [[i for i=1:size(data, 1)]]
    else
        data_copy = copy(data)
        allowmissing!(data_copy)
        for (k, v) in targets
            # https://stackoverflow.com/questions/64957524/how-can-i-obtain-the-complement-of-list-of-indexes-in-julia
            to_drop = unique(data_copy[!, k])
            filter!(d -> !(d in v), to_drop)
            for i=1:size(data_copy, 1)
                if data_copy[i, k] in to_drop
                    data_copy[i, k] = missing
                end
            end
        end
        keyslist = [k for k in keys(targets)]
        gdf = groupby(data_copy, keyslist, skipmissing=true, sort=true)
        indices = groupindices(gdf)
        max_idx = maximum(filter(i -> i !== missing, indices))
        for _=1:max_idx push!(groups, []) end
        for (i, idx) in enumerate(indices)  
            if idx !== missing push!(groups[idx], i) end
        end
    end
    GeneralFairnessRegulariser(scales, protected, groups, reg)
end
function evaluate(r::GeneralFairnessRegulariser, u::AbstractArray)
    total_loss = 0.0
    for g in r.groups
        u_g = normalise(u[g])
        total_loss += evaluate(r.reg, u_g)
    end
    dot(r.scales, total_loss)
end
function prox(r::GeneralFairnessRegulariser, u::AbstractArray, alpha::Number)
    grad = zeros(size(u))
    for g in r.groups
        u_g = u[g]
        subgrad = u_g .- prox(r.reg, u_g, alpha)
        i = 1
        for j in g
            grad[j] += subgrad[i]
            i += 1
        end
    end
    u .- grad
end
function prox!(r::GeneralFairnessRegulariser, u::AbstractArray, alpha::Number)
    u = prox(r, u, alpha)
    u
end